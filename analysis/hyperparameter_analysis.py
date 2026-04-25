from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

from cgbs.evaluation.metrics import evaluate_retrieval
from cgbs.utils.config import ensure_dir


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

    predictions = []
    targets = []
    queries = []
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

            predictions.append(record.get("predictions", []))
            targets.append(record.get("target_identifier", ""))
            queries.append(record.get("query", ""))
            book_ids.append(str(record.get("book_id", "")))

    return {
        "predictions": predictions,
        "targets": targets,
        "queries": queries,
        "book_ids": book_ids,
    }


def read_manifest(path: str) -> List[Dict[str, str]]:
    """
    Read a JSONL manifest for hyperparameter analysis.

    Each line should contain:
    {
        "run_name": "distill_1.0_replay_1.0",
        "prediction_file": "outputs/predictions/001.jsonl",
        "distill_weight": "1.0",
        "replay_weight": "1.0",
        "num_adapters": "1"
    }

    Extra fields are preserved and written to the output table.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

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

            if "prediction_file" not in record:
                raise ValueError(f"Missing prediction_file at line {line_id} in {path}")

            if "run_name" not in record:
                record["run_name"] = Path(record["prediction_file"]).stem

            records.append(record)

    return records


def evaluate_run(prediction_file: str, ks: Sequence[int]) -> Dict[str, float]:
    data = read_prediction_jsonl(prediction_file)
    return evaluate_retrieval(
        predictions=data["predictions"],
        targets=data["targets"],
        ks=ks,
    )


def summarize_hyperparameter_runs(
    manifest_file: str,
    output_csv: str,
    output_json: str,
    ks: Sequence[int],
) -> List[Dict]:
    runs = read_manifest(manifest_file)

    summarized = []

    for run in runs:
        prediction_file = run["prediction_file"]
        metrics = evaluate_run(prediction_file, ks=ks)

        row = dict(run)
        for metric_name, value in metrics.items():
            row[metric_name] = value

        summarized.append(row)

    save_csv(summarized, output_csv)
    save_json(summarized, output_json)

    return summarized


def save_csv(rows: List[Dict], output_csv: str) -> None:
    output_path = Path(output_csv)
    ensure_dir(str(output_path.parent))

    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(rows: List[Dict], output_json: str) -> None:
    output_path = Path(output_json)
    ensure_dir(str(output_path.parent))

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def print_table(rows: List[Dict], metric: str = "MRR@20") -> None:
    if not rows:
        return

    rows = sorted(rows, key=lambda x: float(x.get(metric, 0.0)), reverse=True)

    print(f"\nSorted by {metric}:")
    for row in rows:
        run_name = row.get("run_name", "")
        metric_value = float(row.get(metric, 0.0)) * 100
        extra = []

        for key in [
            "distill_weight",
            "replay_weight",
            "temperature",
            "num_adapters",
            "adapter_bottleneck_size",
        ]:
            if key in row:
                extra.append(f"{key}={row[key]}")

        extra_text = ", ".join(extra)
        print(f"{run_name}: {metric}={metric_value:.2f} | {extra_text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/analysis/hyperparameter_results.csv",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/analysis/hyperparameter_results.json",
    )
    parser.add_argument("--sort_metric", type=str, default="MRR@20")
    args = parser.parse_args()

    rows = summarize_hyperparameter_runs(
        manifest_file=args.manifest,
        output_csv=args.output_csv,
        output_json=args.output_json,
        ks=args.ks,
    )

    print_table(rows, metric=args.sort_metric)

    print(f"\nSaved CSV to {args.output_csv}")
    print(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()