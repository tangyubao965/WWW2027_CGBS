from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from cgbs.evaluation.metrics import evaluate_retrieval, format_metrics
from cgbs.evaluation.query_category import split_indices_by_query_category, subset_by_indices


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

    queries = []
    targets = []
    predictions = []
    book_ids = []

    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_id} in {path}") from exc

            queries.append(record.get("query", ""))
            targets.append(record.get("target_identifier", ""))
            predictions.append(record.get("predictions", []))
            book_ids.append(record.get("book_id", ""))

    return {
        "queries": queries,
        "targets": targets,
        "predictions": predictions,
        "book_ids": book_ids,
    }


def evaluate_by_query_category(
    queries: Sequence[str],
    targets: Sequence[str],
    predictions: Sequence[Sequence[str]],
    ks: Sequence[int] = (10, 20),
) -> Dict[str, Dict[str, float]]:
    groups = split_indices_by_query_category(queries)

    results = {}
    for category, indices in groups.items():
        if not indices:
            results[category] = {}
            continue

        category_predictions = subset_by_indices(list(predictions), indices)
        category_targets = subset_by_indices(list(targets), indices)

        results[category] = evaluate_retrieval(
            predictions=category_predictions,
            targets=category_targets,
            ks=ks,
        )
        results[category]["num_queries"] = float(len(indices))

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    args = parser.parse_args()

    data = read_prediction_jsonl(args.prediction_file)

    results = evaluate_by_query_category(
        queries=data["queries"],
        targets=data["targets"],
        predictions=data["predictions"],
        ks=args.ks,
    )

    for category, metrics in results.items():
        print(f"\n[{category}]")
        if not metrics:
            print("No queries.")
            continue

        num_queries = int(metrics.pop("num_queries"))
        print(f"num_queries: {num_queries}")
        print(format_metrics(metrics))


if __name__ == "__main__":
    main()