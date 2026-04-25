from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from cgbs.evaluation.metrics import evaluate_retrieval, format_metrics
from cgbs.utils.config import ensure_dir


def read_jsonl(path: str) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

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


def read_prediction_jsonl(path: str) -> List[Dict]:
    """
    Expected prediction format:
    {
        "book_id": "...",
        "query": "...",
        "target_identifier": "...",
        "predictions": [...]
    }
    """
    return read_jsonl(path)


def read_book_metadata(path: str) -> Dict[str, Dict]:
    """
    Read book metadata and return a mapping from book_id to metadata.

    Expected fields can include:
    {
        "book_id": "...",
        "num_chapters": 12
    }

    If num_chapters is absent, this script also checks:
    chapter_count, n_chapters.
    """
    records = read_jsonl(path)
    mapping = {}

    for record in records:
        book_id = str(record.get("book_id", record.get("id", ""))).strip()
        if not book_id:
            continue
        mapping[book_id] = record

    return mapping


def get_chapter_count(metadata: Dict) -> int | None:
    for field in ["num_chapters", "chapter_count", "n_chapters"]:
        if field in metadata and metadata[field] is not None:
            try:
                return int(metadata[field])
            except (TypeError, ValueError):
                return None
    return None


def assign_chapter_count_bin(
    chapter_count: int | None,
    small_threshold: int,
    large_threshold: int,
) -> str:
    """
    Assign a book into chapter-count bins.

    Default interpretation:
    small: <= small_threshold
    medium: between small_threshold and large_threshold
    large: >= large_threshold
    """
    if chapter_count is None:
        return "unknown"

    if chapter_count <= small_threshold:
        return "small"
    if chapter_count >= large_threshold:
        return "large"
    return "medium"


def group_prediction_indices_by_chapter_count(
    prediction_records: Sequence[Dict],
    book_metadata: Dict[str, Dict],
    small_threshold: int,
    large_threshold: int,
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {
        "small": [],
        "medium": [],
        "large": [],
        "unknown": [],
    }

    for idx, record in enumerate(prediction_records):
        book_id = str(record.get("book_id", "")).strip()
        metadata = book_metadata.get(book_id, {})
        chapter_count = get_chapter_count(metadata)

        bin_name = assign_chapter_count_bin(
            chapter_count=chapter_count,
            small_threshold=small_threshold,
            large_threshold=large_threshold,
        )
        groups[bin_name].append(idx)

    return groups


def subset_by_indices(items: Sequence, indices: Sequence[int]) -> List:
    return [items[i] for i in indices]


def analyze_chapter_count_bias(
    prediction_file: str,
    metadata_file: str,
    output_file: str | None = None,
    small_threshold: int = 5,
    large_threshold: int = 20,
    ks: Sequence[int] = (10, 20),
) -> Dict[str, Dict[str, float]]:
    prediction_records = read_prediction_jsonl(prediction_file)
    book_metadata = read_book_metadata(metadata_file)

    groups = group_prediction_indices_by_chapter_count(
        prediction_records=prediction_records,
        book_metadata=book_metadata,
        small_threshold=small_threshold,
        large_threshold=large_threshold,
    )

    predictions = [record.get("predictions", []) for record in prediction_records]
    targets = [record.get("target_identifier", "") for record in prediction_records]

    results: Dict[str, Dict[str, float]] = {}

    for group_name, indices in groups.items():
        metrics: Dict[str, float] = {
            "num_queries": float(len(indices)),
        }

        if indices:
            group_predictions = subset_by_indices(predictions, indices)
            group_targets = subset_by_indices(targets, indices)

            group_metrics = evaluate_retrieval(
                predictions=group_predictions,
                targets=group_targets,
                ks=ks,
            )
            metrics.update(group_metrics)

        results[group_name] = metrics

    if output_file is not None:
        output_path = Path(output_file)
        ensure_dir(str(output_path.parent))
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/analysis/chapter_count_bias.json",
    )
    parser.add_argument("--small_threshold", type=int, default=5)
    parser.add_argument("--large_threshold", type=int, default=20)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    args = parser.parse_args()

    results = analyze_chapter_count_bias(
        prediction_file=args.prediction_file,
        metadata_file=args.metadata_file,
        output_file=args.output,
        small_threshold=args.small_threshold,
        large_threshold=args.large_threshold,
        ks=args.ks,
    )

    for group_name, metrics in results.items():
        print(f"\n[{group_name}]")
        num_queries = int(metrics.get("num_queries", 0))
        print(f"num_queries: {num_queries}")

        metric_values = {
            key: value for key, value in metrics.items()
            if key != "num_queries"
        }
        if metric_values:
            print(format_metrics(metric_values))

    print(f"\nSaved chapter count bias results to {args.output}")


if __name__ == "__main__":
    main()