from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from scripts.preprocess_utils import (
    build_readable_book_text,
    list_to_text,
    normalize_id,
    normalize_text,
    read_jsonl,
    safe_get,
    write_jsonl,
)


def build_book_index(book_file: str) -> Dict[str, Dict]:
    """Build mapping from book id to book metadata."""
    book_records = read_jsonl(book_file)
    book_index: Dict[str, Dict] = {}

    for book in book_records:
        book_id = normalize_id(safe_get(book, "id", "book_id"))
        if not book_id:
            continue
        book_index[book_id] = book

    return book_index


def infer_cover_path(book_record: Dict, image_ext: str = ".jpg") -> str:
    """
    Infer local cover image path from book id.

    """
    if book_record.get("cover_path"):
        return normalize_text(book_record["cover_path"])

    book_id = normalize_id(safe_get(book_record, "id", "book_id"))
    if not book_id:
        return ""

    return f"{book_id}{image_ext}"


def normalize_whatsthatbook_record(
    post_record: Dict,
    book_record: Dict,
    split: str,
    image_ext: str,
) -> Dict:
    """Convert one WhatsThatBook post-book pair into the unified format."""
    book_id = normalize_id(post_record.get("book_id", ""))

    title = normalize_text(book_record.get("title", ""))
    author = normalize_text(book_record.get("author", ""))
    publisher = normalize_text(book_record.get("publisher", ""))

    description = normalize_text(book_record.get("description", ""))
    answers = post_record.get("answers", [])

    if not description:
        description = list_to_text(answers)

    genres = book_record.get("genres", {})
    parsed_dates = book_record.get("parsed_dates", [])
    publication_date = list_to_text(parsed_dates, max_items=1) or normalize_text(book_record.get("date", ""))

    extra_fields = {
        "Genres": genres,
        "Publication Date": publication_date,
        "Gold Answers": answers,
    }

    book_text = build_readable_book_text(
        title=title,
        author=author,
        publisher=publisher,
        description=description,
        outline="",
        extra_fields=extra_fields,
    )

    return {
        "book_id": book_id,
        "title": title,
        "author": author,
        "publisher": publisher,
        "description": description,
        "outline": "",
        "book_text": book_text,
        "query": normalize_text(post_record.get("question", "")),
        "question_posted_date": normalize_text(post_record.get("question_posted_date", "")),
        "answers": answers,
        "isbn_13": normalize_text(book_record.get("isbn_13", "")),
        "date": normalize_text(book_record.get("date", "")),
        "parsed_dates": parsed_dates,
        "image_link": normalize_text(book_record.get("image_link", "")),
        "cover_path": infer_cover_path(book_record, image_ext=image_ext),
        "ratings": book_record.get("ratings", None),
        "reviews": book_record.get("reviews", None),
        "genres": genres,
        "split": split,
    }


def process_split(
    post_file: str,
    book_index: Dict[str, Dict],
    split: str,
    image_ext: str,
) -> List[Dict]:
    post_records = read_jsonl(post_file)

    normalized = []
    missing_books = 0

    for post in post_records:
        book_id = normalize_id(post.get("book_id", ""))
        book_record = book_index.get(book_id)

        if book_record is None:
            missing_books += 1
            continue

        normalized.append(
            normalize_whatsthatbook_record(
                post_record=post,
                book_record=book_record,
                split=split,
                image_ext=image_ext,
            )
        )

    print(f"{split}: loaded={len(post_records)}, matched={len(normalized)}, missing_books={missing_books}")
    return normalized


def preprocess_whatsthatbook(
    data_dir: str,
    output_file: str,
    image_ext: str = ".jpg",
) -> None:
    data_dir_path = Path(data_dir)

    book_file = data_dir_path / "whatsthatbook_14441_gold_books.jsonl"
    train_file = data_dir_path / "whatsthatbook_14441_gold_posts_train_11552.split_with_val.jsonl"
    dev_file = data_dir_path / "whatsthatbook_14441_gold_posts_dev_1444.split_with_val.jsonl"
    test_file = data_dir_path / "whatsthatbook_14441_gold_posts_test_1445.split_with_val.jsonl"

    for path in [book_file, train_file, dev_file, test_file]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    book_index = build_book_index(str(book_file))

    all_records = []
    all_records.extend(process_split(str(train_file), book_index, "train", image_ext=image_ext))
    all_records.extend(process_split(str(dev_file), book_index, "valid", image_ext=image_ext))
    all_records.extend(process_split(str(test_file), book_index, "test", image_ext=image_ext))

    write_jsonl(all_records, output_file)

    print(f"Total records: {len(all_records)}")
    print(f"Saved unified WhatsThatBook data to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing WhatsThatBook jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/whatsthatbook_unified.jsonl",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        help="Local cover image extension after extracting the image tar file.",
    )
    args = parser.parse_args()

    preprocess_whatsthatbook(
        data_dir=args.data_dir,
        output_file=args.output,
        image_ext=args.image_ext,
    )


if __name__ == "__main__":
    main()