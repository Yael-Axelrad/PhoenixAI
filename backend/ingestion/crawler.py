
import json
import logging
import random
import time

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urldefrag

import httpx
from bs4 import BeautifulSoup
from protego import Protego


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
PAGES_FILE = DATA_DIR / "pages.json"
REPORT_FILE = DATA_DIR / "crawl_report.json"

BASE_URL = "https://www.fnx.co.il"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
ROBOTS_URL = f"{BASE_URL}/robots.txt"

USER_AGENT = "PhoenixAI-ResearchBot/1.0"

REQUEST_TIMEOUT = 25.0
REQUEST_DELAY = 2.5
REQUEST_JITTER = 1.0
MAX_RETRIES = 3
SAVE_EVERY = 10
MIN_CONTENT_LENGTH = 150

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phoenixai.crawler")


# --------------------------------------------------
# File helpers
# --------------------------------------------------

def save_json(path: Path, data) -> None:
    """
    Atomically save JSON to avoid corrupting saved data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temp_path.replace(path)


def load_existing_pages() -> list[dict]:
    """
    Load previously crawled pages to support resume.
    """
    if not PAGES_FILE.exists():
        return []

    with PAGES_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("pages.json must contain a JSON array.")

    return data


# --------------------------------------------------
# URL helpers
# --------------------------------------------------

def normalize_url(url: str) -> str:
    """
    Remove fragments and normalize trailing slashes.
    """
    url, _ = urldefrag(url.strip())

    parsed = urlparse(url)

    if parsed.path and not parsed.path.endswith("/"):
        if "." not in parsed.path.split("/")[-1]:
            url += "/"

    return url


def is_allowed_domain(url: str) -> bool:
    """
    Allow only HTTPS pages on the official Phoenix domain.
    """
    parsed = urlparse(url)

    return (
        parsed.scheme == "https"
        and parsed.hostname == "www.fnx.co.il"
    )


def is_html_candidate(url: str) -> bool:
    """
    Skip files that are not HTML pages.
    """
    path = urlparse(url).path.lower()

    excluded_extensions = (
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
        ".svg",
        ".zip",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".mp4",
        ".mp3",
        ".css",
        ".js",
        ".xml",
    )

    return not path.endswith(excluded_extensions)


# --------------------------------------------------
# HTTP helpers
# --------------------------------------------------

def fetch_with_retries(
    client: httpx.Client,
    url: str,
) -> httpx.Response | None:
    """
    Fetch a URL with bounded retries and polite delays.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.get(url)

            if response.status_code == 200:
                return response

            if response.status_code in (429, 500, 502, 503, 504):
                logger.warning(
                    "Retryable status %s for %s",
                    response.status_code,
                    url,
                )

                if attempt < MAX_RETRIES:
                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    try:
                        wait_seconds = float(retry_after)
                    except (ValueError, TypeError):
                        wait_seconds = min(
                            2 ** attempt * 3,
                            60,
                        )

                    time.sleep(
                        max(0, min(wait_seconds, 120))
                    )
                    continue

            logger.warning(
                "Skipping %s (HTTP %s)",
                url,
                response.status_code,
            )
            return None

        except httpx.RequestError as error:
            logger.warning(
                "Request failed for %s: %s",
                url,
                error,
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    min(2 ** attempt * 3, 60)
                )

    return None


# --------------------------------------------------
# Robots.txt
# --------------------------------------------------

def load_robots(client: httpx.Client) -> Protego:
    """
    Load robots.txt and fail closed if unavailable.
    """
    response = fetch_with_retries(
        client,
        ROBOTS_URL,
    )

    if response is None:
        raise RuntimeError(
            "Could not load robots.txt. "
            "Crawling stopped for safety."
        )

    robots = Protego.parse(response.text)

    logger.info("Loaded robots.txt")

    return robots


# --------------------------------------------------
# Sitemap discovery
# --------------------------------------------------

def discover_urls(
    client: httpx.Client,
    robots: Protego,
) -> list[str]:
    """
    Discover HTML URLs from the sitemap index and
    nested sitemap files.
    """
    discovered = set()
    visited_sitemaps = set()
    pending_sitemaps = [SITEMAP_URL]

    while pending_sitemaps:
        sitemap_url = pending_sitemaps.pop()

        sitemap_url = normalize_url(sitemap_url)

        if sitemap_url in visited_sitemaps:
            continue

        if not is_allowed_domain(sitemap_url):
            continue

        visited_sitemaps.add(sitemap_url)

        logger.info(
            "Reading sitemap: %s",
            sitemap_url,
        )

        response = fetch_with_retries(
            client,
            sitemap_url,
        )

        if response is None:
            continue

        soup = BeautifulSoup(
            response.content,
            "xml",
        )

        sitemap_tags = soup.find_all("sitemap")

        if sitemap_tags:
            for tag in sitemap_tags:
                loc = tag.find("loc")

                if loc and loc.text:
                    pending_sitemaps.append(
                        loc.text.strip()
                    )

        else:
            for tag in soup.find_all("url"):
                loc = tag.find("loc")

                if not loc or not loc.text:
                    continue

                url = normalize_url(
                    loc.text.strip()
                )

                if not is_allowed_domain(url):
                    continue

                if not is_html_candidate(url):
                    continue

                if not robots.can_fetch(
                    url,
                    USER_AGENT,
                ):
                    continue

                discovered.add(url)

        time.sleep(
            REQUEST_DELAY
            + random.uniform(0, REQUEST_JITTER)
        )

    logger.info(
        "Discovered %d allowed HTML URLs",
        len(discovered),
    )

    return sorted(discovered)


# --------------------------------------------------
# HTML extraction
# --------------------------------------------------

def extract_page(
    html: str,
    url: str,
) -> dict | None:
    """
    Extract readable text from a public HTML page.
    """
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
        "footer",
        "nav",
        "header",
        "form",
    ]):
        tag.decompose()

    title_tag = soup.find("title")

    title = (
        title_tag.get_text(" ", strip=True)
        if title_tag
        else ""
    )

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.body
    )

    if main is None:
        return None

    content = main.get_text(
        " ",
        strip=True,
    )

    content = " ".join(
        content.split()
    )

    if len(content) < MIN_CONTENT_LENGTH:
        return None

    return {
        "url": url,
        "title": title,
        "content": content,
        "crawled_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# --------------------------------------------------
# Main crawler
# --------------------------------------------------

def crawl_all() -> None:
    """
    Crawl every discoverable, permitted public HTML page.

    Resume from previously saved pages.
    """
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_pages = load_existing_pages()

    pages_by_url = {
        page["url"]: page
        for page in existing_pages
        if isinstance(page, dict)
        and page.get("url")
    }

    logger.info(
        "Loaded %d existing pages",
        len(pages_by_url),
    )

    stats = {
        "started_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "discovered": 0,
        "already_saved": 0,
        "new_pages": 0,
        "failed": 0,
        "skipped": 0,
        "failed_urls": [],
    }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
    }

    with httpx.Client(
        headers=headers,
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
    ) as client:

        robots = load_robots(client)

        urls = discover_urls(
            client,
            robots,
        )

        stats["discovered"] = len(urls)

        stats["already_saved"] = sum(
            url in pages_by_url
            for url in urls
        )

        pending_urls = [
            url
            for url in urls
            if url not in pages_by_url
        ]

        logger.info(
            "Pending pages: %d",
            len(pending_urls),
        )

        try:
            for index, url in enumerate(
                pending_urls,
                start=1,
            ):
                logger.info(
                    "[%d/%d] Crawling %s",
                    index,
                    len(pending_urls),
                    url,
                )

                if not robots.can_fetch(
                    url,
                    USER_AGENT,
                ):
                    stats["skipped"] += 1
                    continue

                response = fetch_with_retries(
                    client,
                    url,
                )

                if response is None:
                    stats["failed"] += 1
                    stats["failed_urls"].append(url)
                    continue

                final_url = normalize_url(
                    str(response.url)
                )

                if not is_allowed_domain(final_url):
                    stats["skipped"] += 1
                    continue

                if not robots.can_fetch(
                    final_url,
                    USER_AGENT,
                ):
                    stats["skipped"] += 1
                    continue

                content_type = response.headers.get(
                    "Content-Type",
                    "",
                ).lower()

                if (
                    "text/html" not in content_type
                    and "application/xhtml+xml"
                    not in content_type
                ):
                    stats["skipped"] += 1
                    continue

                page = extract_page(
                    response.text,
                    final_url,
                )

                if page is None:
                    stats["skipped"] += 1
                    continue

                pages_by_url[final_url] = page

                stats["new_pages"] += 1

                if stats["new_pages"] % SAVE_EVERY == 0:
                    save_json(
                        PAGES_FILE,
                        list(pages_by_url.values()),
                    )

                    logger.info(
                        "Checkpoint saved: %d pages",
                        len(pages_by_url),
                    )

                time.sleep(
                    REQUEST_DELAY
                    + random.uniform(
                        0,
                        REQUEST_JITTER,
                    )
                )

        except KeyboardInterrupt:
            logger.warning(
                "Crawl interrupted. Saving progress."
            )

        finally:
            save_json(
                PAGES_FILE,
                list(pages_by_url.values()),
            )

            stats["finished_at"] = datetime.now(
                timezone.utc
            ).isoformat()

            stats["total_saved_pages"] = len(
                pages_by_url
            )

            save_json(
                REPORT_FILE,
                stats,
            )

            logger.info(
                "Crawl finished | total=%d | new=%d | "
                "failed=%d | skipped=%d",
                stats["total_saved_pages"],
                stats["new_pages"],
                stats["failed"],
                stats["skipped"],
            )

            logger.info(
                "Pages saved to: %s",
                PAGES_FILE,
            )

            logger.info(
                "Report saved to: %s",
                REPORT_FILE,
            )


if __name__ == "__main__":
    crawl_all()
