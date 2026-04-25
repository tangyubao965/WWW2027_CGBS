from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from scripts.preprocess_utils import (
    build_readable_book_text,
    normalize_text,
    write_jsonl,
)


def sanitize_filename(value: str) -> str:
    """Create a safe filename."""
    value = normalize_text(value)
    value = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", value)
    value = value.strip("_")
    return value or "unknown"


def parse_bookcover_text(text: str) -> Dict[str, str]:
    """
    Parse the synthetic text field in kayteekay/bookimg_dataset.

    Example:
        the book is titled fantastic beasts and where to find them
        and is written by j. k. rowling and is of genre art-photography
    """
    text = normalize_text(text)
    lower_text = text.lower()

    title = ""
    author = ""
    genre = ""

    title_match = re.search(
        r"the book is titled (.*?) and is written by ",
        lower_text,
    )
    if title_match:
        title = title_match.group(1).strip()

    author_match = re.search(
        r" and is written by (.*?) and is of genre ",
        lower_text,
    )
    if author_match:
        author = author_match.group(1).strip()

    genre_match = re.search(
        r" and is of genre (.*)$",
        lower_text,
    )
    if genre_match:
        genre = genre_match.group(1).strip()

    return {
        "title": title,
        "author": author,
        "genre": genre,
    }


def save_image(image: Any, output_path: Path) -> bool:
    """Save a Hugging Face image object or PIL image to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image is None:
        return False

    try:
        if isinstance(image, Image.Image):
            image.convert("RGB").save(output_path)
            return True

        img = Image.open(image).convert("RGB")
        img.save(output_path)
        return True
    except Exception:
        return False


def split_indices(
    n: int,
    train_size: int,
    valid_size: int,
    test_size: int,
) -> Tuple[range, range, range]:
    """Create deterministic split ranges."""
    required = train_size + valid_size + test_size
    if required > n:
        raise ValueError(
            f"Requested split sizes exceed dataset size: "
            f"{train_size}+{valid_size}+{test_size}={required}, but dataset has {n} examples."
        )

    train_range = range(0, train_size)
    valid_range = range(train_size, train_size + valid_size)
    test_range = range(train_size + valid_size, train_size + valid_size + test_size)

    return train_range, valid_range, test_range


def compute_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute dataset statistics."""
    split_counts: Dict[str, int] = {}
    text_lengths = []
    title_lengths = []
    author_lengths = []

    for record in records:
        split = record.get("split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1

        text_lengths.append(len(record.get("book_text", "").split()))
        title_lengths.append(len(record.get("title", "").split()))
        author_lengths.append(len(record.get("author", "").split()))

    num_records = len(records)

    def avg(values: List[int]) -> float:
        return sum(values) / max(len(values), 1)

    return {
        "num_records": num_records,
        "split_counts": split_counts,
        "avg_book_text_length_words": avg(text_lengths),
        "avg_title_length_words": avg(title_lengths),
        "avg_author_length_words": avg(author_lengths),
        "max_book_text_length_words": max(text_lengths) if text_lengths else 0,
        "min_book_text_length_words": min(text_lengths) if text_lengths else 0,
    }


def save_statistics(stats: Dict[str, Any], output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def normalize_bookcover_record(
    raw_record: Dict[str, Any],
    index: int,
    split: str,
    image_dir: Path,
) -> Dict[str, Any]:
    """Convert one BookCover example into the unified CGBS format."""
    text = normalize_text(raw_record.get("text", ""))
    parsed = parse_bookcover_text(text)

    book_id = f"bookcover_{index:06d}"
    image_filename = f"{book_id}.jpg"
    image_path = image_dir / image_filename

    image_saved = save_image(raw_record.get("image"), image_path)
    cover_path = image_filename if image_saved else ""

    title = parsed["title"]
    author = parsed["author"]
    genre = parsed["genre"]

    book_text = build_readable_book_text(
        title=title,
        author=author,
        publisher="",
        description=text,
        outline="",
        extra_fields={"Genre": genre} if genre else None,
    )

    return {
        "book_id": book_id,
        "title": title,
        "author": author,
        "publisher": "",
        "description": text,
        "outline": "",
        "book_text": book_text,
        "query": text,
        "genre": genre,
        "cover_path": cover_path,
        "split": split,
    }


def preprocess_bookcover(
    dataset_name: str,
    output_file: str,
    image_dir: str,
    stats_file: str,
    train_size: int,
    valid_size: int,
    test_size: int,
    seed: int,
) -> None:
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.shuffle(seed=seed)

    print(f"Loaded dataset: {dataset_name}")
    print(f"Number of examples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    if "image" not in dataset.column_names or "text" not in dataset.column_names:
        raise ValueError(
            f"Expected columns ['image', 'text'], but got {dataset.column_names}."
        )

    train_range, valid_range, test_range = split_indices(
        n=len(dataset),
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
    )

    split_by_index = {}
    for idx in train_range:
        split_by_index[idx] = "train"
    for idx in valid_range:
        split_by_index[idx] = "valid"
    for idx in test_range:
        split_by_index[idx] = "test"

    image_dir_path = Path(image_dir)
    image_dir_path.mkdir(parents=True, exist_ok=True)

    max_index = train_size + valid_size + test_size
    records = []

    for idx in tqdm(range(max_index), desc="Preprocessing BookCover"):
        raw_record = dataset[int(idx)]
        split = split_by_index[idx]

        record = normalize_bookcover_record(
            raw_record=raw_record,
            index=idx,
            split=split,
            image_dir=image_dir_path,
        )
        records.append(record)

    write_jsonl(records, output_file)

    stats = compute_statistics(records)
    save_statistics(stats, stats_file)

    print(f"Saved unified BookCover data to {output_file}")
    print(f"Saved statistics to {stats_file}")
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="kayteekay/bookimg_dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/bookcover_unified.jsonl",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/images/bookcover",
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        default="outputs/analysis/bookcover_stats.json",
    )
    parser.add_argument("--train_size", type=int, default=28000)
    parser.add_argument("--valid_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocess_bookcover(
        dataset_name=args.dataset_name,
        output_file=args.output,
        image_dir=args.image_dir,
        stats_file=args.stats_file,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()