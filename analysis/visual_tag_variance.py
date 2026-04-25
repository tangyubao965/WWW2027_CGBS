from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

from cgbs.evaluation.metrics import evaluate_retrieval, format_metrics
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


def evaluate_prediction_file(
    prediction_file: str,
    ks: Sequence[int],
) -> Dict[str, float]:
    data = read_prediction_jsonl(prediction_file)
    return evaluate_retrieval(
        predictions=data["predictions"],
        targets=data["targets"],
        ks=ks,
    )


def summarize_metric_variance(
    prediction_files: Sequence[str],
    ks: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and standard deviation of each retrieval metric across
    multiple prediction files.

    Each prediction file can correspond to a different visual-tag version,
    extraction seed, or visual-tag hyperparameter setting.
    """
    if not prediction_files:
        raise ValueError("prediction_files is empty.")

    all_metrics = []

    for prediction_file in prediction_files:
        metrics = evaluate_prediction_file(prediction_file, ks=ks)
        all_metrics.append(metrics)

    metric_names = sorted(all_metrics[0].keys())
    summary: Dict[str, Dict[str, float]] = {}

    for metric_name in metric_names:
        values = [metrics[metric_name] for metrics in all_metrics]
        summary[metric_name] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    return summary


def read_manifest(manifest_file: str) -> List[str]:
    """
    Read a manifest file containing one prediction file path per line.
    """
    path = Path(manifest_file)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    files = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            file_path = line.strip()
            if file_path:
                files.append(file_path)

    return files


def save_summary(summary: Dict[str, Dict[str, float]], output_file: str) -> None:
    output_path = Path(output_file)
    ensure_dir(str(output_path.parent))

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    for metric_name, stats in summary.items():
        print(
            f"{metric_name}: "
            f"mean={stats['mean'] * 100:.2f}, "
            f"std={stats['std'] * 100:.2f}, "
            f"min={stats['min'] * 100:.2f}, "
            f"max={stats['max'] * 100:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_files",
        type=str,
        nargs="*",
        default=None,
        help="Prediction JSONL files from different visual-tag settings.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional text file with one prediction file path per line.",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/analysis/visual_tag_variance.json",
    )
    args = parser.parse_args()

    prediction_files: List[str] = []

    if args.manifest is not None:
        prediction_files.extend(read_manifest(args.manifest))

    if args.prediction_files is not None:
        prediction_files.extend(args.prediction_files)

    prediction_files = list(dict.fromkeys(prediction_files))

    if not prediction_files:
        raise ValueError("Please provide --prediction_files or --manifest.")

    summary = summarize_metric_variance(
        prediction_files=prediction_files,
        ks=args.ks,
    )

    print_summary(summary)
    save_summary(summary, args.output)

    print(f"Saved visual tag variance summary to {args.output}")


if __name__ == "__main__":
    main()