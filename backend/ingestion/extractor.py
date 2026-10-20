
import re
import httpx

from bs4 import BeautifulSoup


# Elements that generally do not contain article content
REMOVE_TAGS = [
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "form",
    "button",
    "svg",
    "iframe",
]

# Common interface elements that are usually not useful for RAG
REMOVE_SELECTORS = [
    '[role="navigation"]',
    '[role="dialog"]',
    '[aria-modal="true"]',
    ".cookie-banner",
    ".cookie-consent",
    ".breadcrumbs",
    ".breadcrumb",
]


def clean_text(text: str) -> str:
    """
    Normalize extracted text without removing meaningful content.
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Normalize whitespace inside each line
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        # Remove consecutive duplicate lines only
        if cleaned_lines and cleaned_lines[-1] == line:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_page(url: str) -> dict:
    """
    Download a public HTML page and extract its main textual content.
    """

    response = httpx.get(
        url,
        timeout=20,
        follow_redirects=True,
        headers={
            "User-Agent": "PhoenixAI-ResearchBot"
        },
    )

    response.raise_for_status()

    # Reject non-HTML responses
    content_type = response.headers.get("content-type", "").lower()

    if "text/html" not in content_type:
        return {
            "url": str(response.url),
            "title": "",
            "content": "",
        }

    soup = BeautifulSoup(response.text, "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else ""
    )

    # Remove HTML elements unrelated to the main content
    for element in soup.find_all(REMOVE_TAGS):
        element.decompose()

    # Remove common interface components
    for selector in REMOVE_SELECTORS:
        for element in soup.select(selector):
            element.decompose()

    # Prefer the main content region when available
    main_content = soup.find("main")

    content_element = main_content if main_content else soup.body

    if content_element is None:
        text = ""
    else:
        text = content_element.get_text(
            separator="\n",
            strip=True,
        )

    text = clean_text(text)

    return {
        "url": str(response.url),
        "title": title,
        "content": text,
    }
