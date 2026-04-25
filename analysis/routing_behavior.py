from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from cgbs.data.collator import CGBSCollator
from cgbs.data.dataset import BookSearchDataset
from cgbs.evaluation.query_category import assign_query_category
from cgbs.models.retriever import TextOnlyRetriever
from cgbs.utils.config import ensure_dir, load_config
from cgbs.utils.seed import set_seed


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def export_routing_scores(
    config_path: str,
    model_path: str | None,
    split: str,
    output_path: str,
) -> None:
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    model_cfg = config["model"]
    path_cfg = config["paths"]
    train_cfg = config["training"]

    if split == "train":
        data_file = path_cfg["train_file"]
    elif split == "valid":
        data_file = path_cfg["valid_file"]
    elif split == "test":
        data_file = path_cfg["test_file"]
    else:
        raise ValueError(f"Unknown split: {split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    retriever_dir = model_path or str(Path(path_cfg["output_dir"]) / "retriever" / "final")
    tokenizer = AutoTokenizer.from_pretrained(retriever_dir)

    dataset = BookSearchDataset(
        file_path=data_file,
        image_dir=path_cfg.get("image_dir"),
        use_query=True,
        use_cover=False,
        use_caption=False,
    )

    collator = CGBSCollator(
        tokenizer=tokenizer,
        image_processor=None,
        max_input_length=model_cfg.get("retriever_max_input_length", 512),
        max_target_length=model_cfg.get("max_target_length", 128),
        use_cover=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 1))),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        collate_fn=collator,
    )

    model = TextOnlyRetriever.from_pretrained(
        model_dir=retriever_dir,
        text_model_name=model_cfg["text_model_name"],
        use_adapters=model_cfg.get("use_adapters", True),
        num_adapters=model_cfg.get("num_adapters", 1),
        adapter_bottleneck_size=model_cfg.get("adapter_bottleneck_size", 64),
        adapter_dropout=model_cfg.get("adapter_dropout", 0.1),
        map_location=device,
    ).to(device)

    model.eval()

    output_path = Path(output_path)
    ensure_dir(str(output_path.parent))

    cursor = 0
    with output_path.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc=f"Exporting routing scores on {split}"):
            batch = move_batch_to_device(batch, device)

            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_route_score=True,
                )

            if outputs.route_score is None:
                raise RuntimeError(
                    "route_score is None. Please enable use_adapters=True in the config."
                )

            route_scores = outputs.route_score.detach().cpu().tolist()

            for local_idx, score in enumerate(route_scores):
                ex = dataset.examples[cursor + local_idx]
                query = ex.get("query", "")

                record = {
                    "book_id": batch["book_ids"][local_idx],
                    "query": query,
                    "query_category": assign_query_category(query),
                    "target_identifier": batch["identifiers"][local_idx],
                    "route_score": float(score),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            cursor += len(route_scores)

    print(f"Saved routing scores to {output_path}")


def read_routing_scores(path: str) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Routing score file not found: {path}")

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


def analyze_routing_scores(score_file: str) -> Dict[str, float]:
    records = read_routing_scores(score_file)

    grouped: Dict[str, List[float]] = {
        "cover-oriented": [],
        "other": [],
        "all": [],
    }

    for record in records:
        score = float(record["route_score"])
        category = record.get("query_category") or assign_query_category(record.get("query", ""))

        grouped.setdefault(category, []).append(score)
        grouped["all"].append(score)

    results = {
        "num_all": float(len(grouped["all"])),
        "mean_all": mean(grouped["all"]),
        "num_cover_oriented": float(len(grouped.get("cover-oriented", []))),
        "mean_cover_oriented": mean(grouped.get("cover-oriented", [])),
        "num_other": float(len(grouped.get("other", []))),
        "mean_other": mean(grouped.get("other", [])),
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--config", type=str, default="configs/base.yaml")
    export_parser.add_argument("--model_path", type=str, default=None)
    export_parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    export_parser.add_argument(
        "--output",
        type=str,
        default="outputs/analysis/routing_scores.jsonl",
    )

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument(
        "--score_file",
        type=str,
        default="outputs/analysis/routing_scores.jsonl",
    )

    args = parser.parse_args()

    if args.command == "export":
        export_routing_scores(
            config_path=args.config,
            model_path=args.model_path,
            split=args.split,
            output_path=args.output,
        )
    elif args.command == "analyze":
        results = analyze_routing_scores(args.score_file)
        print(f"num_all: {int(results['num_all'])}")
        print(f"mean_all: {results['mean_all']:.4f}")
        print(f"num_cover_oriented: {int(results['num_cover_oriented'])}")
        print(f"mean_cover_oriented: {results['mean_cover_oriented']:.4f}")
        print(f"num_other: {int(results['num_other'])}")
        print(f"mean_other: {results['mean_other']:.4f}")


if __name__ == "__main__":
    main()