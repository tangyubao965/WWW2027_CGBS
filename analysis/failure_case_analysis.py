from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from cgbs.evaluation.query_category import assign_query_category
from cgbs.utils.config import ensure_dir


def read_prediction_jsonl(path: str) -> List[Dict]:
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

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_id} in {path}") from exc

            records.append(record)

    return records


def is_hit_at_k(predictions: Sequence[str], target: str, k: int) -> bool:
    return target in list(predictions)[:k]


def reciprocal_rank(predictions: Sequence[str], target: str, k: int) -> float:
    for rank, pred in enumerate(list(predictions)[:k], start=1):
        if pred == target:
            return 1.0 / rank
    return 0.0


def classify_failure(record: Dict, k: int) -> str:
    """
    Assign a coarse failure type based on query and prediction behavior.

    This is heuristic and intended for analysis, not as ground truth.
    """
    query = str(record.get("query", "")).lower()
    predictions = record.get("predictions", [])
    target = record.get("target_identifier", "")

    if not predictions:
        return "empty_generation"

    if not is_hit_at_k(predictions, target, k):
        if "cover" in query:
            return "missed_cover_oriented_query"

        ambiguity_words = {
            "maybe", "possibly", "probably", "similar", "like", "kind of",
            "not sure", "i think", "remember", "forgot", "vague",
        }
        if any(word in query for word in ambiguity_words):
            return "ambiguous_or_uncertain_query"

        return "wrong_topk_candidates"

    if predictions[0] != target:
        return "low_rank_correct_answer"

    return "success"


def extract_failure_cases(
    prediction_file: str,
    output_file: str,
    k: int = 10,
    max_cases: int = 100,
    category: str | None = None,
) -> List[Dict]:
    records = read_prediction_jsonl(prediction_file)

    failure_cases = []

    for record in records:
        query = record.get("query", "")
        query_category = assign_query_category(query)

        if category is not None and query_category != category:
            continue

        predictions = record.get("predictions", [])
        target = record.get("target_identifier", "")

        if is_hit_at_k(predictions, target, k):
            continue

        failure_type = classify_failure(record, k=k)

        case = {
            "book_id": record.get("book_id", ""),
            "query": query,
            "query_category": query_category,
            "target_identifier": target,
            f"top_{k}_predictions": predictions[:k],
            "failure_type": failure_type,
            f"rr_at_{k}": reciprocal_rank(predictions, target, k),
        }
        failure_cases.append(case)

    failure_cases = failure_cases[:max_cases]

    output_path = Path(output_file)
    ensure_dir(str(output_path.parent))

    with output_path.open("w", encoding="utf-8") as f:
        for case in failure_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Extracted {len(failure_cases)} failure cases to {output_path}")
    return failure_cases


def summarize_failures(prediction_file: str, k: int = 10) -> Dict[str, int]:
    records = read_prediction_jsonl(prediction_file)

    summary: Dict[str, int] = {}

    for record in records:
        failure_type = classify_failure(record, k=k)
        summary[failure_type] = summary.get(failure_type, 0) + 1

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--prediction_file", type=str, required=True)
    extract_parser.add_argument(
        "--output",
        type=str,
        default="outputs/analysis/failure_cases.jsonl",
    )
    extract_parser.add_argument("--k", type=int, default=10)
    extract_parser.add_argument("--max_cases", type=int, default=100)
    extract_parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=[None, "cover-oriented", "other"],
    )

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--prediction_file", type=str, required=True)
    summarize_parser.add_argument("--k", type=int, default=10)

    args = parser.parse_args()

    if args.command == "extract":
        extract_failure_cases(
            prediction_file=args.prediction_file,
            output_file=args.output,
            k=args.k,
            max_cases=args.max_cases,
            category=args.category,
        )
    elif args.command == "summarize":
        summary = summarize_failures(
            prediction_file=args.prediction_file,
            k=args.k,
        )
        for failure_type, count in sorted(summary.items(), key=lambda x: x[0]):
            print(f"{failure_type}: {count}")


if __name__ == "__main__":
    main()