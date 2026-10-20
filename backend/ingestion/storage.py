
import json
from pathlib import Path
from datetime import datetime, timezone


# Root directory of the PhoenixAI project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Storage directory for extracted pages
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def save_pages(pages: list[dict]) -> Path:
    """
    Save crawled pages with source URLs and timestamps.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    crawled_at = datetime.now(timezone.utc).isoformat()

    records = [
        {
            "url": page["url"],
            "title": page["title"],
            "content": page["content"],
            "crawled_at": crawled_at,
        }
        for page in pages
    ]

    output_path = DATA_DIR / "pages.json"

    # Preserve readable Hebrew characters in JSON
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path
