from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from cgbs.identifier.cluster_builder import (
    HierarchicalClusterCodeBuilder,
    HierarchicalClusterConfig,
)
from cgbs.identifier.identifier_builder import BookIdentifierBuilder


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


def build_cluster_codes(
    records: List[Dict],
    text_field: str,
    output_field: str,
    num_cluster_levels: int,
    branching_factor: int,
    random_suffix_size: int,
    random_state: int,
) -> List[Dict]:
    texts = [record.get(text_field, "") for record in records]

    cluster_builder = HierarchicalClusterCodeBuilder(
        HierarchicalClusterConfig(
            num_cluster_levels=num_cluster_levels,
            branching_factor=branching_factor,
            random_suffix_size=random_suffix_size,
            random_state=random_state,
        )
    )
    cluster_codes = cluster_builder.fit_predict(texts)

    for record, code in zip(records, cluster_codes):
        record[output_field] = code

    return records


def build_identifiers(records: List[Dict]) -> List[Dict]:
    builder = BookIdentifierBuilder()

    for record in records:
        level = record.get("level", "book")

        if level not in {"book", "chapter", "section"}:
            raise ValueError(f"Invalid level: {level}")

        record["identifier"] = builder.build_from_record(record)

    return records


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument(
        "--level",
        type=str,
        required=True,
        choices=["book", "chapter", "section"],
        help="Identifier level of the input records.",
    )

    parser.add_argument(
        "--text_field",
        type=str,
        default=None,
        help="Text field used for hierarchical clustering. Required for chapter/section.",
    )

    parser.add_argument("--num_cluster_levels", type=int, default=10)
    parser.add_argument("--branching_factor", type=int, default=100)
    parser.add_argument("--random_suffix_size", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    records = read_jsonl(args.input)

    for record in records:
        record["level"] = args.level

    if args.level == "chapter":
        if args.text_field is None:
            raise ValueError("--text_field is required for chapter-level identifiers.")
        records = build_cluster_codes(
            records=records,
            text_field=args.text_field,
            output_field="chapter_cluster_id",
            num_cluster_levels=args.num_cluster_levels,
            branching_factor=args.branching_factor,
            random_suffix_size=args.random_suffix_size,
            random_state=args.random_state,
        )

    if args.level == "section":
        if args.text_field is None:
            raise ValueError("--text_field is required for section-level identifiers.")
        records = build_cluster_codes(
            records=records,
            text_field=args.text_field,
            output_field="section_cluster_id",
            num_cluster_levels=args.num_cluster_levels,
            branching_factor=args.branching_factor,
            random_suffix_size=args.random_suffix_size,
            random_state=args.random_state,
        )

    records = build_identifiers(records)
    write_jsonl(records, args.output)

    print(f"Saved {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()