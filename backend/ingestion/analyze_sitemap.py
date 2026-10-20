
import httpx
import xml.etree.ElementTree as ET
from collections import Counter
from urllib.parse import urlparse

SITEMAP_URL = "https://www.fnx.co.il/sitemap.xml"

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def analyze_sitemap():
    response = httpx.get(
        SITEMAP_URL,
        timeout=30,
        follow_redirects=True
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)

    urls = []
    sections = Counter()
    with_lastmod = 0

    for entry in root.findall("sm:url", NS):
        location = entry.findtext("sm:loc", namespaces=NS)
        lastmod = entry.findtext("sm:lastmod", namespaces=NS)

        if not location:
            continue

        parsed = urlparse(location)

        if parsed.hostname not in {"fnx.co.il", "www.fnx.co.il"}:
            continue

        urls.append(location)

        if lastmod:
            with_lastmod += 1

        parts = [part for part in parsed.path.split("/") if part]
        section = parts[0] if parts else "homepage"
        sections[section] += 1

    print("Total Phoenix URLs:", len(urls))
    print("URLs with lastmod:", with_lastmod)

    print("\nTop 20 sections:")
    for section, count in sections.most_common(20):
        print(f"{section}: {count}")

    print("\nFirst 10 URLs:")
    for url in urls[:10]:
        print(url)


if __name__ == "__main__":
    analyze_sitemap()
