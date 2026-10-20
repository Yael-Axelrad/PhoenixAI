
import httpx
import xml.etree.ElementTree as ET

from urllib.parse import urlparse
from protego import Protego


# Official Phoenix website resources
SITEMAP_URL = "https://www.fnx.co.il/sitemap.xml"
ROBOTS_URL = "https://www.fnx.co.il/robots.txt"

# Identifies our crawler when checking robots.txt rules
USER_AGENT = "PhoenixAI-ResearchBot"

# XML namespace used by standard sitemap files
NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9"
}


def discover_urls() -> list[str]:
    """
    Fetch the official sitemap and return unique URLs
    belonging to the Phoenix website.
    """

    # Download the official sitemap
    response = httpx.get(
        SITEMAP_URL,
        timeout=30,
        follow_redirects=True
    )
    response.raise_for_status()

    # Parse the XML document
    root = ET.fromstring(response.content)

    urls = []

    # Extract each page URL from its <loc> element
    for entry in root.findall("sm:url", NS):
        url = entry.findtext("sm:loc", namespaces=NS)

        if not url:
            continue

        parsed = urlparse(url)

        # Only accept HTTPS URLs from the exact Phoenix host
        if parsed.scheme != "https":
            continue

        if parsed.hostname != "www.fnx.co.il":
            continue

        urls.append(url)

    # Remove duplicates while preserving the original order
    return list(dict.fromkeys(urls))


def get_robots_parser() -> Protego:
    """
    Download and parse the website's robots.txt rules.

    Protego supports wildcard patterns such as /*?page=
    that are not handled correctly by RobotFileParser.
    """

    response = httpx.get(
        ROBOTS_URL,
        timeout=20,
        follow_redirects=True
    )
    response.raise_for_status()

    # Parse robots.txt using a wildcard-aware parser
    return Protego.parse(response.text)


def filter_allowed_urls(urls: list[str]) -> list[str]:
    """
    Return only URLs allowed by the website's robots.txt.

    Important: This function only filters URLs.
    It does not download or crawl any content pages.
    """

    parser = get_robots_parser()

    allowed_urls = []

    for url in urls:
        # Protego expects: can_fetch(url, user_agent)
        if parser.can_fetch(url, USER_AGENT):
            allowed_urls.append(url)

    return allowed_urls


if __name__ == "__main__":
    """
    Manual validation of robots.txt parsing.

    This block runs only when discovery.py is
    executed directly, not when it is imported.
    """

    parser = get_robots_parser()

    test_urls = [
        # Expected: ALLOWED
        "https://www.fnx.co.il/car-insurance/",

        # Expected: BLOCKED (disallowed directory)
        "https://www.fnx.co.il/more-information/example/",

        # Expected: BLOCKED (disallowed path)
        "https://www.fnx.co.il/SPF",

        # Expected: BLOCKED (wildcard query rule)
        "https://www.fnx.co.il/articles/?page=2",
    ]

    print("Robots.txt validation:")
    print("----------------------")

    for url in test_urls:
        allowed = parser.can_fetch(url, USER_AGENT)

        status = "ALLOWED" if allowed else "BLOCKED"

        print(status, url)
