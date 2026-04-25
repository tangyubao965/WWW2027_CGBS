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


def index_by_key(records: Sequence[Dict], key: str = "book_id") -> Dict[str, Dict]:
    indexed = {}
    for record in records:
        value = str(record.get(key, "")).strip()
        if value:
            indexed[value] = record
    return indexed


def is_hit_at_k(predictions: Sequence[str], target: str, k: int) -> bool:
    return target in list(predictions)[:k]


def get_rank(predictions: Sequence[str], target: str, k: int) -> int | None:
    for rank, pred in enumerate(list(predictions)[:k], start=1):
        if pred == target:
            return rank
    return None


def extract_comparative_cases(
    cgbs_prediction_file: str,
    baseline_prediction_file: str,
    output_file: str,
    k: int = 10,
    max_cases: int = 50,
    query_category: str | None = "cover-oriented",
) -> List[Dict]:
    """
    Extract cases where CGBS succeeds but the baseline fails.

    Matching is performed by book_id. If the same book_id appears multiple
    times, the last one is kept, so query-level unique ids are preferred when
    available.
    """
    cgbs_records = read_prediction_jsonl(cgbs_prediction_file)
    baseline_records = read_prediction_jsonl(baseline_prediction_file)

    baseline_by_book_id = index_by_key(baseline_records, key="book_id")

    cases = []

    for cgbs_record in cgbs_records:
        book_id = str(cgbs_record.get("book_id", "")).strip()
        if not book_id or book_id not in baseline_by_book_id:
            continue

        query = cgbs_record.get("query", "")
        category = assign_query_category(query)

        if query_category is not None and category != query_category:
            continue

        target = cgbs_record.get("target_identifier", "")
        cgbs_predictions = cgbs_record.get("predictions", [])

        baseline_record = baseline_by_book_id[book_id]
        baseline_predictions = baseline_record.get("predictions", [])

        cgbs_hit = is_hit_at_k(cgbs_predictions, target, k)
        baseline_hit = is_hit_at_k(baseline_predictions, target, k)

        if not cgbs_hit or baseline_hit:
            continue

        case = {
            "book_id": book_id,
            "query": query,
            "query_category": category,
            "target_identifier": target,
            "cgbs_rank": get_rank(cgbs_predictions, target, k),
            "baseline_rank": get_rank(baseline_predictions, target, k),
            f"cgbs_top_{k}": cgbs_predictions[:k],
            f"baseline_top_{k}": baseline_predictions[:k],
        }

        cases.append(case)

        if len(cases) >= max_cases:
            break

    output_path = Path(output_file)
    ensure_dir(str(output_path.parent))

    with output_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Extracted {len(cases)} comparative cases to {output_path}")
    return cases


def extract_success_cases(
    cgbs_prediction_file: str,
    output_file: str,
    k: int = 10,
    max_cases: int = 50,
    query_category: str | None = "cover-oriented",
) -> List[Dict]:
    """
    Extract successful CGBS cases.
    """
    records = read_prediction_jsonl(cgbs_prediction_file)
    cases = []

    for record in records:
        query = record.get("query", "")
        category = assign_query_category(query)

        if query_category is not None and category != query_category:
            continue

        target = record.get("target_identifier", "")
        predictions = record.get("predictions", [])

        if not is_hit_at_k(predictions, target, k):
            continue

        case = {
            "book_id": record.get("book_id", ""),
            "query": query,
            "query_category": category,
            "target_identifier": target,
            "rank": get_rank(predictions, target, k),
            f"top_{k}_predictions": predictions[:k],
        }
        cases.append(case)

        if len(cases) >= max_cases:
            break

    output_path = Path(output_file)
    ensure_dir(str(output_path.parent))

    with output_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Extracted {len(cases)} success cases to {output_path}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    comp_parser = subparsers.add_parser("compare")
    comp_parser.add_argument("--cgbs_prediction_file", type=str, required=True)
    comp_parser.add_argument("--baseline_prediction_file", type=str, required=True)
    comp_parser.add_argument(
        "--output",
        type=str,
        default="outputs/analysis/comparative_cases.jsonl",
    )
    comp_parser.add_argument("--k", type=int, default=10)
    comp_parser.add_argument("--max_cases", type=int, default=50)
    comp_parser.add_argument(
        "--query_category",
        type=str,
        default="cover-oriented",
        choices=["cover-oriented", "other", "all"],
    )

    success_parser = subparsers.add_parser("success")
    success_parser.add_argument("--cgbs_prediction_file", type=str, required=True)
    success_parser.add_argument(
        "--output",
        type=str,
        default="outputs/analysis/success_cases.jsonl",
    )
    success_parser.add_argument("--k", type=int, default=10)
    success_parser.add_argument("--max_cases", type=int, default=50)
    success_parser.add_argument(
        "--query_category",
        type=str,
        default="cover-oriented",
        choices=["cover-oriented", "other", "all"],
    )

    args = parser.parse_args()

    query_category = None if args.query_category == "all" else args.query_category

    if args.command == "compare":
        extract_comparative_cases(
            cgbs_prediction_file=args.cgbs_prediction_file,
            baseline_prediction_file=args.baseline_prediction_file,
            output_file=args.output,
            k=args.k,
            max_cases=args.max_cases,
            query_category=query_category,
        )
    elif args.command == "success":
        extract_success_cases(
            cgbs_prediction_file=args.cgbs_prediction_file,
            output_file=args.output,
            k=args.k,
            max_cases=args.max_cases,
            query_category=query_category,
        )


if __name__ == "__main__":
    main()