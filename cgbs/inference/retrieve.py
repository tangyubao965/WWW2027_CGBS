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
from cgbs.evaluation.metrics import evaluate_retrieval, format_metrics
from cgbs.inference.constrained_decode import constrained_generate
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


def load_valid_identifiers(identifier_file: str | None, dataset: BookSearchDataset) -> List[str]:
    """
    Load valid identifiers.

    If identifier_file is provided, it should be a text file with one identifier
    per line. Otherwise, identifiers are collected from the dataset.
    """
    if identifier_file is not None:
        path = Path(identifier_file)
        if not path.exists():
            raise FileNotFoundError(f"Identifier file not found: {path}")

        identifiers = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    identifiers.append(line)
    else:
        identifiers = []
        for ex in dataset.examples:
            identifier = str(ex.get("identifier", "")).strip()
            if identifier:
                identifiers.append(identifier)

    identifiers = sorted(set(identifiers))
    if not identifiers:
        raise ValueError("No valid identifiers found.")

    return identifiers


def save_predictions(
    output_path: str,
    queries: Sequence[str],
    targets: Sequence[str],
    predictions: Sequence[Sequence[str]],
    book_ids: Sequence[str],
) -> None:
    path = Path(output_path)
    ensure_dir(str(path.parent))

    with path.open("w", encoding="utf-8") as f:
        for query, target, pred_list, book_id in zip(queries, targets, predictions, book_ids):
            record = {
                "book_id": book_id,
                "query": query,
                "target_identifier": target,
                "predictions": list(pred_list),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.no_grad()
def run_retrieval(
    config_path: str,
    model_path: str | None = None,
    split: str = "test",
    identifier_file: str | None = None,
) -> Dict[str, float]:
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    model_cfg = config["model"]
    path_cfg = config["paths"]
    train_cfg = config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if split == "train":
        data_file = path_cfg["train_file"]
    elif split == "valid":
        data_file = path_cfg["valid_file"]
    elif split == "test":
        data_file = path_cfg["test_file"]
    else:
        raise ValueError(f"Unknown split: {split}")

    tokenizer_source = model_path or model_cfg["text_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    dataset = BookSearchDataset(
        file_path=data_file,
        image_dir=path_cfg.get("image_dir"),
        use_query=True,
        use_cover=False,
        use_caption=False,
    )

    valid_identifiers = load_valid_identifiers(identifier_file, dataset)

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

    retriever_dir = model_path or str(Path(path_cfg["output_dir"]) / "retriever" / "final")
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

    all_predictions: List[List[str]] = []
    all_targets: List[str] = []
    all_queries: List[str] = []
    all_book_ids: List[str] = []

    num_beams = int(model_cfg.get("num_beams", 20))
    num_return_sequences = int(model_cfg.get("num_return_sequences", 20))
    max_gen_length = int(model_cfg.get("max_target_length", 128))

    for batch in tqdm(loader, desc=f"Retrieving on {split}"):
        batch = move_batch_to_device(batch, device)

        predictions = constrained_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            valid_identifiers=valid_identifiers,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_length=max_gen_length,
            device=device,
        )

        all_predictions.extend(predictions)
        all_targets.extend(batch["identifiers"])
        all_queries.extend(dataset.examples[len(all_queries) + i].get("query", "") for i in range(len(predictions)))
        all_book_ids.extend(batch["book_ids"])

    metrics = evaluate_retrieval(
        predictions=all_predictions,
        targets=all_targets,
        ks=(1, 5, 10, 20),
    )

    print(format_metrics(metrics))

    output_dir = Path(path_cfg["output_dir"]) / "predictions"
    ensure_dir(str(output_dir))

    save_predictions(
        output_path=str(output_dir / f"{split}_predictions.jsonl"),
        queries=all_queries,
        targets=all_targets,
        predictions=all_predictions,
        book_ids=all_book_ids,
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--identifier_file", type=str, default=None)
    args = parser.parse_args()

    run_retrieval(
        config_path=args.config,
        model_path=args.model_path,
        split=args.split,
        identifier_file=args.identifier_file,
    )


if __name__ == "__main__":
    main()