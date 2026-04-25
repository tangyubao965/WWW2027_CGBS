from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def read_jsonl(path: str) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_id} in {path}") from exc
    return records


def write_jsonl(records: List[Dict], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records(
    records: List[Dict],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if train_ratio <= 0 or valid_ratio < 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Require train_ratio > 0, valid_ratio >= 0, and train_ratio + valid_ratio < 1.")

    rng = random.Random(seed)
    records = records[:]
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = records[:n_train]
    valid = records[n_train:n_train + n_valid]
    test = records[n_train + n_valid:]

    return train, valid, test


def build_book_text(record: Dict) -> str:
    """
    Construct a unified book_text field from available fields.

    This is intentionally conservative: it keeps all available information
    in a readable order and leaves missing fields empty.
    """
    fields = []

    title = record.get("title", "")
    author = record.get("author", "")
    publisher = record.get("publisher", "")
    outline = record.get("outline", "")
    chapter_title = record.get("chapter_title", "")
    section_title = record.get("section_title", "")

    if title:
        fields.append(f"Title: {title}")
    if author:
        fields.append(f"Author: {author}")
    if publisher:
        fields.append(f"Publisher: {publisher}")
    if outline:
        fields.append(f"Outline: {outline}")
    if chapter_title:
        fields.append(f"Chapter: {chapter_title}")
    if section_title:
        fields.append(f"Section: {section_title}")

    text = (
        record.get("book_text")
        or record.get("section_text")
        or record.get("chapter_text")
        or record.get("summary")
        or record.get("description")
        or ""
    )

    if text:
        fields.append(f"Content: {text}")

    return "\n".join(fields)


def normalize_record(record: Dict) -> Dict:
    """
    Convert a raw enriched record into the format consumed by BookSearchDataset.
    """
    if "identifier" not in record or not str(record["identifier"]).strip():
        raise ValueError("Each record must contain a non-empty identifier.")

    query = record.get("query", "")
    if not query:
        query = record.get("description", "") or record.get("summary", "") or record.get("title", "")

    normalized = {
        "book_id": str(record.get("book_id", record.get("id", ""))),
        "query": query,
        "book_text": build_book_text(record),
        "identifier": record["identifier"],
        "cover_path": record.get("cover_path", ""),
        "cover_caption": record.get("cover_caption", ""),
        "visual_tag": record.get("visual_tag", ""),
    }

    if not normalized["book_id"]:
        normalized["book_id"] = normalized["identifier"]

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--already_split",
        action="store_true",
        help="Use this if records already contain a split field with train/valid/test.",
    )

    args = parser.parse_args()

    records = read_jsonl(args.input)
    records = [normalize_record(record) for record in records]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.already_split:
        train = [r for r in records if r.get("split") == "train"]
        valid = [r for r in records if r.get("split") in {"valid", "dev"}]
        test = [r for r in records if r.get("split") == "test"]
    else:
        train, valid, test = split_records(
            records=records,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )

    write_jsonl(train, str(output_dir / "train.jsonl"))
    write_jsonl(valid, str(output_dir / "valid.jsonl"))
    write_jsonl(test, str(output_dir / "test.jsonl"))

    identifiers = sorted({r["identifier"] for r in records})
    with (output_dir / "identifiers.txt").open("w", encoding="utf-8") as f:
        for identifier in identifiers:
            f.write(identifier + "\n")

    print(f"Train: {len(train)}")
    print(f"Valid: {len(valid)}")
    print(f"Test:  {len(test)}")
    print(f"Identifiers: {len(identifiers)}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()