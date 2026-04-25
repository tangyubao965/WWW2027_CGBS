from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from cgbs.data.collator import CGBSCollator
from cgbs.data.dataset import BookSearchDataset
from cgbs.inference.constrained_decode import constrained_generate
from cgbs.inference.retrieve import load_valid_identifiers
from cgbs.models.retriever import TextOnlyRetriever
from cgbs.utils.config import ensure_dir, load_config
from cgbs.utils.seed import set_seed


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
    }


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_retrieval_efficiency(
    config_path: str,
    model_path: str | None,
    split: str,
    identifier_file: str | None,
    num_warmup_batches: int,
    num_eval_batches: int | None,
    output_path: str,
) -> Dict[str, float]:
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

    param_stats = count_parameters(model)

    num_beams = int(model_cfg.get("num_beams", 20))
    num_return_sequences = int(model_cfg.get("num_return_sequences", 20))
    max_gen_length = int(model_cfg.get("max_target_length", 128))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch_times: List[float] = []
    query_counts: List[int] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Measuring efficiency")):
        batch = move_batch_to_device(batch, device)

        synchronize_if_needed(device)
        start = time.perf_counter()

        _ = constrained_generate(
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

        synchronize_if_needed(device)
        elapsed = time.perf_counter() - start

        if batch_idx >= num_warmup_batches:
            batch_times.append(elapsed)
            query_counts.append(batch["input_ids"].size(0))

        if num_eval_batches is not None and len(batch_times) >= num_eval_batches:
            break

    total_time = sum(batch_times)
    total_queries = sum(query_counts)

    if total_queries == 0:
        raise RuntimeError("No evaluation batches were measured. Reduce num_warmup_batches.")

    latency_per_query = total_time / total_queries
    throughput = total_queries / total_time if total_time > 0 else 0.0

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_memory_mb = 0.0

    results: Dict[str, float] = {
        "num_measured_batches": float(len(batch_times)),
        "num_measured_queries": float(total_queries),
        "total_time_seconds": total_time,
        "latency_per_query_seconds": latency_per_query,
        "throughput_queries_per_second": throughput,
        "peak_memory_mb": peak_memory_mb,
        "total_parameters": float(param_stats["total_parameters"]),
        "trainable_parameters": float(param_stats["trainable_parameters"]),
        "num_beams": float(num_beams),
        "num_return_sequences": float(num_return_sequences),
        "max_gen_length": float(max_gen_length),
    }

    output_path = Path(output_path)
    ensure_dir(str(output_path.parent))
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved efficiency results to {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--identifier_file", type=str, default=None)
    parser.add_argument("--num_warmup_batches", type=int, default=2)
    parser.add_argument("--num_eval_batches", type=int, default=50)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/analysis/efficiency.json",
    )
    args = parser.parse_args()

    measure_retrieval_efficiency(
        config_path=args.config,
        model_path=args.model_path,
        split=args.split,
        identifier_file=args.identifier_file,
        num_warmup_batches=args.num_warmup_batches,
        num_eval_batches=args.num_eval_batches,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()