from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from cgbs.evaluation.metrics import evaluate_retrieval, format_metrics


def read_prediction_jsonl(path: str) -> Dict[str, List]:
    """
    Read prediction JSONL.

    Expected format per line:
    {
        "book_id": "...",
        "query": "...",
        "target_identifier": "...",
        "predictions": [...]
    }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    data = {
        "book_ids": [],
        "queries": [],
        "targets": [],
        "predictions": [],
    }

    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_id} in {path}") from exc

            data["book_ids"].append(str(record.get("book_id", "")))
            data["queries"].append(record.get("query", ""))
            data["targets"].append(record.get("target_identifier", ""))
            data["predictions"].append(record.get("predictions", []))

    return data


def read_invalid_cover_ids(path: str) -> set[str]:
    """
    Read invalid-cover book ids.

    The input can be:
    1. a plain text file with one book_id per line; or
    2. a JSONL file where each line contains a field named book_id.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Invalid cover id file not found: {path}")

    invalid_ids: set[str] = set()

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_id} in {path}") from exc

                book_id = str(record.get("book_id", "")).strip()
                if book_id:
                    invalid_ids.add(book_id)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                book_id = line.strip()
                if book_id:
                    invalid_ids.add(book_id)

    return invalid_ids


def subset_by_indices(items: Sequence, indices: Sequence[int]) -> List:
    return [items[i] for i in indices]


def split_valid_invalid_indices(
    book_ids: Sequence[str],
    invalid_cover_ids: set[str],
) -> Dict[str, List[int]]:
    groups = {
        "valid-cover": [],
        "invalid-cover": [],
    }

    for idx, book_id in enumerate(book_ids):
        if str(book_id) in invalid_cover_ids:
            groups["invalid-cover"].append(idx)
        else:
            groups["valid-cover"].append(idx)

    return groups


def evaluate_valid_invalid(
    prediction_file: str,
    invalid_cover_file: str,
    ks: Sequence[int] = (10, 20),
) -> Dict[str, Dict[str, float]]:
    data = read_prediction_jsonl(prediction_file)
    invalid_cover_ids = read_invalid_cover_ids(invalid_cover_file)

    groups = split_valid_invalid_indices(
        book_ids=data["book_ids"],
        invalid_cover_ids=invalid_cover_ids,
    )

    results: Dict[str, Dict[str, float]] = {}

    for group_name, indices in groups.items():
        if not indices:
            results[group_name] = {"num_queries": 0.0}
            continue

        predictions = subset_by_indices(data["predictions"], indices)
        targets = subset_by_indices(data["targets"], indices)

        metrics = evaluate_retrieval(
            predictions=predictions,
            targets=targets,
            ks=ks,
        )
        metrics["num_queries"] = float(len(indices))
        results[group_name] = metrics

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--invalid_cover_file", type=str, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    args = parser.parse_args()

    results = evaluate_valid_invalid(
        prediction_file=args.prediction_file,
        invalid_cover_file=args.invalid_cover_file,
        ks=args.ks,
    )

    for group_name, metrics in results.items():
        print(f"\n[{group_name}]")
        num_queries = int(metrics.pop("num_queries", 0))
        print(f"num_queries: {num_queries}")

        if num_queries == 0:
            continue

        print(format_metrics(metrics))


if __name__ == "__main__":
    main()