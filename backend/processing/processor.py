
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "pages.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    """Remove obvious interface noise while preserving insurance information."""

    interface_lines = {
        "עוד",
        "סגור",
        "לקבלת הצעה",
        "לקבלת הצעה אונליין",
    }

    cleaned_lines = []

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()

        if not line or line in interface_lines:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def split_into_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks, preferably at line boundaries."""

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            # Prefer ending at a line break near the chunk boundary
            boundary = text.rfind("\n", start + CHUNK_SIZE // 2, end)

            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Overlap helps preserve context across chunk boundaries
        start = max(start + 1, end - CHUNK_OVERLAP)

    return chunks


def process_pages() -> list[dict]:
    """Load extracted pages and produce chunks with source metadata."""

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        pages = json.load(file)

    all_chunks = []

    for page in pages:
        cleaned_content = clean_text(page["content"])
        page_chunks = split_into_chunks(cleaned_content)

        for index, chunk in enumerate(page_chunks):
            all_chunks.append({
                "id": f"{len(all_chunks) + 1}",
                "url": page["url"],
                "title": page["title"],
                "chunk_index": index,
                "content": chunk,
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(all_chunks, file, ensure_ascii=False, indent=2)

    return all_chunks


if __name__ == "__main__":
    chunks = process_pages()

    print("Processing completed")
    print("Chunks created:", len(chunks))
    print("Saved to:", OUTPUT_FILE)
