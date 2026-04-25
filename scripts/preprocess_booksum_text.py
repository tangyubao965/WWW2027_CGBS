from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.preprocess_utils import (
    build_readable_book_text,
    normalize_id,
    normalize_text,
    read_json,
    read_jsonl,
    safe_get,
    write_jsonl,
)


def load_records(path: str) -> List[Dict]:
    """Load BookSum records from JSON or JSONL."""
    if path.endswith(".jsonl"):
        return read_jsonl(path)

    data = read_json(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "records", "books", "chapters", "summaries"]:
            if key in data and isinstance(data[key], list):
                return data[key]

        if all(isinstance(v, list) for v in data.values()):
            keys = list(data.keys())
            length = len(data[keys[0]])
            return [{key: data[key][i] for key in keys} for i in range(length)]

    raise ValueError(f"Unsupported BookSum file format: {path}")


def infer_title(record: Dict) -> str:
    return normalize_text(
        safe_get(
            record,
            "title",
            "book_title",
            "book",
            "book_name",
            "document_title",
            default="",
        )
    )


def infer_author(record: Dict) -> str:
    return normalize_text(
        safe_get(
            record,
            "author",
            "authors",
            "book_author",
            default="",
        )
    )


def infer_chapter_title(record: Dict) -> str:
    return normalize_text(
        safe_get(
            record,
            "chapter_title",
            "chapter",
            "chapter_name",
            default="",
        )
    )


def infer_source_text(record: Dict) -> str:
    return normalize_text(
        safe_get(
            record,
            "source",
            "text",
            "document",
            "content",
            "chapter_text",
            "book_text",
            default="",
        )
    )


def infer_summary(record: Dict) -> str:
    return normalize_text(
        safe_get(
            record,
            "summary",
            "target",
            "summaries",
            "chapter_summary",
            "book_summary",
            default="",
        )
    )


def make_booksum_id(record: Dict, index: int) -> str:
    raw_id = safe_get(
        record,
        "book_id",
        "bid",
        "id",
        "source_id",
        "document_id",
        default=f"booksum_{index:06d}",
    )
    return normalize_id(raw_id) or f"booksum_{index:06d}"


def assign_split(index: int, total: int, train_ratio: float, valid_ratio: float) -> str:
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    if index < train_end:
        return "train"
    if index < valid_end:
        return "valid"
    return "test"


def normalize_booksum_text_record(record: Dict, index: int, split: str) -> Dict:
    book_id = make_booksum_id(record, index)

    title = infer_title(record)
    author = infer_author(record)
    chapter_title = infer_chapter_title(record)
    source_text = infer_source_text(record)
    summary = infer_summary(record)

    book_text = build_readable_book_text(
        title=title,
        author=author,
        publisher="",
        description=summary,
        outline=chapter_title,
        extra_fields={
            "BookSum Text": source_text,
        },
    )

    return {
        "book_id": book_id,
        "title": title,
        "author": author,
        "publisher": "",
        "description": summary,
        "outline": chapter_title,
        "book_text": book_text,
        "chapter_title": chapter_title,
        "chapter_text": source_text,
        "query": summary or chapter_title or title,
        "split": split,
    }


def compute_text_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    split_counts: Dict[str, int] = {}
    book_text_lengths = []
    chapter_text_lengths = []
    summary_lengths = []

    for record in records:
        split = record.get("split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1

        book_text_lengths.append(len(str(record.get("book_text", "")).split()))
        chapter_text_lengths.append(len(str(record.get("chapter_text", "")).split()))
        summary_lengths.append(len(str(record.get("description", "")).split()))

    def avg(values: List[int]) -> float:
        return sum(values) / max(len(values), 1)

    return {
        "num_records": len(records),
        "split_counts": split_counts,
        "avg_book_text_length_words": avg(book_text_lengths),
        "avg_chapter_text_length_words": avg(chapter_text_lengths),
        "avg_summary_length_words": avg(summary_lengths),
        "max_book_text_length_words": max(book_text_lengths) if book_text_lengths else 0,
        "max_chapter_text_length_words": max(chapter_text_lengths) if chapter_text_lengths else 0,
    }


def save_json(obj: Dict[str, Any], output_path: str) -> None:
    import json

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def preprocess_booksum_text(
    input_file: str,
    output_file: str,
    stats_file: str,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    max_records: int | None = None,
) -> None:
    records = load_records(input_file)

    if max_records is not None:
        records = records[:max_records]

    rng = random.Random(seed)
    records = records[:]
    rng.shuffle(records)

    total = len(records)
    normalized = []

    for idx, record in enumerate(records):
        split = assign_split(
            index=idx,
            total=total,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )
        normalized.append(
            normalize_booksum_text_record(
                record=record,
                index=idx,
                split=split,
            )
        )

    write_jsonl(normalized, output_file)

    stats = compute_text_statistics(normalized)
    save_json(stats, stats_file)

    print(f"Saved BookSum text records to {output_file}")
    print(f"Saved statistics to {stats_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/booksum_text_unified.jsonl",
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        default="outputs/analysis/booksum_text_stats.json",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_records", type=int, default=None)
    args = parser.parse_args()

    preprocess_booksum_text(
        input_file=args.input,
        output_file=args.output,
        stats_file=args.stats_file,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()