from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from scripts.preprocess_utils import normalize_text, read_jsonl, write_jsonl


OPEN_LIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
OPEN_LIBRARY_COVER_URL = "https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"


def save_json(obj: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def openlibrary_search(
    title: str,
    author: str = "",
    timeout: int = 20,
    sleep_seconds: float = 0.1,
) -> Optional[Dict]:
    """Search Open Library and return the top document."""
    title = normalize_text(title)
    author = normalize_text(author)

    if not title:
        return None

    params = {
        "title": title,
        "limit": "1",
    }
    if author:
        params["author"] = author

    url = f"{OPEN_LIBRARY_SEARCH_URL}?{urllib.parse.urlencode(params)}"

    time.sleep(sleep_seconds)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    docs = payload.get("docs", [])
    if not docs:
        return None

    return docs[0]


def extract_openlibrary_metadata(doc: Optional[Dict]) -> Dict[str, Any]:
    if doc is None:
        return {
            "openlibrary_title": "",
            "openlibrary_author": "",
            "openlibrary_publisher": "",
            "openlibrary_first_publish_year": "",
            "openlibrary_cover_id": "",
            "openlibrary_key": "",
            "isbn": "",
        }

    author_names = doc.get("author_name") or []
    publishers = doc.get("publisher") or []
    isbn_values = doc.get("isbn") or []

    return {
        "openlibrary_title": normalize_text(doc.get("title", "")),
        "openlibrary_author": normalize_text(author_names[0] if author_names else ""),
        "openlibrary_publisher": normalize_text(publishers[0] if publishers else ""),
        "openlibrary_first_publish_year": normalize_text(doc.get("first_publish_year", "")),
        "openlibrary_cover_id": normalize_text(doc.get("cover_i", "")),
        "openlibrary_key": normalize_text(doc.get("key", "")),
        "isbn": normalize_text(isbn_values[0] if isbn_values else ""),
    }


def download_cover_by_id(
    cover_id: str,
    output_path: Path,
    timeout: int = 30,
) -> bool:
    cover_id = normalize_text(cover_id)
    if not cover_id:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = OPEN_LIBRARY_COVER_URL.format(cover_id=cover_id)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()

        if not data:
            return False

        output_path.write_bytes(data)
        return True
    except Exception:
        return False


def augment_records_with_covers(
    input_file: str,
    output_file: str,
    image_dir: str,
    stats_file: str,
    sleep_seconds: float,
    timeout: int,
    skip_existing_cover: bool,
    max_records: int | None = None,
) -> None:
    records = read_jsonl(input_file)

    if max_records is not None:
        records = records[:max_records]

    image_dir_path = Path(image_dir)
    image_dir_path.mkdir(parents=True, exist_ok=True)

    matched_count = 0
    cover_count = 0

    for record in tqdm(records, desc="Augmenting BookSum with Open Library covers"):
        if record.get("cover_path") and skip_existing_cover:
            cover_count += 1
            continue

        title = record.get("title", "")
        author = record.get("author", "")

        doc = openlibrary_search(
            title=title,
            author=author,
            timeout=timeout,
            sleep_seconds=sleep_seconds,
        )
        metadata = extract_openlibrary_metadata(doc)

        if metadata["openlibrary_key"]:
            matched_count += 1

        for key, value in metadata.items():
            record[key] = value

        # Fill missing metadata only when original fields are empty.
        if not record.get("title") and metadata["openlibrary_title"]:
            record["title"] = metadata["openlibrary_title"]
        if not record.get("author") and metadata["openlibrary_author"]:
            record["author"] = metadata["openlibrary_author"]
        if not record.get("publisher") and metadata["openlibrary_publisher"]:
            record["publisher"] = metadata["openlibrary_publisher"]

        cover_id = metadata["openlibrary_cover_id"]
        cover_filename = f"{record['book_id']}.jpg"
        cover_output_path = image_dir_path / cover_filename

        saved = download_cover_by_id(
            cover_id=cover_id,
            output_path=cover_output_path,
            timeout=timeout,
        )

        if saved:
            record["cover_path"] = cover_filename
            cover_count += 1
        else:
            record["cover_path"] = ""

    write_jsonl(records, output_file)

    stats = {
        "num_records": len(records),
        "num_openlibrary_matched": matched_count,
        "num_with_cover": cover_count,
        "openlibrary_match_rate": matched_count / max(len(records), 1),
        "cover_rate": cover_count / max(len(records), 1),
    }
    save_json(stats, stats_file)

    print(f"Saved BookSum-Cover records to {output_file}")
    print(f"Saved statistics to {stats_file}")
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/booksum_text_unified.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/booksum_cover_unified.jsonl",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/images/booksum_cover",
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        default="outputs/analysis/booksum_cover_stats.json",
    )
    parser.add_argument("--sleep_seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--skip_existing_cover", action="store_true")
    parser.add_argument("--max_records", type=int, default=None)
    args = parser.parse_args()

    augment_records_with_covers(
        input_file=args.input,
        output_file=args.output,
        image_dir=args.image_dir,
        stats_file=args.stats_file,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
        skip_existing_cover=args.skip_existing_cover,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()