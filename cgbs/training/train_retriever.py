from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, get_linear_schedule_with_warmup

from cgbs.data.collator import CGBSCollator
from cgbs.data.dataset import BookSearchDataset
from cgbs.models.losses import token_kl_distillation_loss
from cgbs.models.memorizer import CoverAwareBookMemorizer
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


def train_retriever(config_path: str, memorizer_path: str | None = None) -> None:
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    model_cfg = config["model"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    path_cfg = config["paths"]

    output_dir = Path(path_cfg["output_dir"]) / "retriever"
    ensure_dir(str(output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["text_model_name"])
    image_processor = AutoImageProcessor.from_pretrained(model_cfg["vision_model_name"])

    teacher_path = memorizer_path or str(Path(path_cfg["output_dir"]) / "memorizer" / "final")

    teacher = CoverAwareBookMemorizer.from_pretrained(
        model_dir=teacher_path,
        text_model_name=model_cfg["text_model_name"],
        vision_model_name=model_cfg["vision_model_name"],
        dropout=model_cfg.get("dropout", 0.1),
        freeze_vision_encoder=True,
        map_location=device,
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    retriever = TextOnlyRetriever(
        text_model_name=model_cfg["text_model_name"],
        use_adapters=model_cfg.get("use_adapters", True),
        num_adapters=model_cfg.get("num_adapters", 1),
        adapter_bottleneck_size=model_cfg.get("adapter_bottleneck_size", 64),
        adapter_dropout=model_cfg.get("adapter_dropout", 0.1),
    ).to(device)

    if model_cfg.get("initialize_retriever_from_memorizer", True):
        retriever.initialize_from_memorizer(teacher)

    train_dataset = BookSearchDataset(
        file_path=path_cfg["train_file"],
        image_dir=path_cfg.get("image_dir"),
        use_query=True,
        use_cover=True,
        use_caption=False,
    )

    collator = CGBSCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_input_length=model_cfg.get("retriever_max_input_length", 512),
        max_target_length=model_cfg.get("max_target_length", 128),
        use_cover=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=collator,
    )

    optimizer = AdamW(
        retriever.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    num_epochs = int(train_cfg.get("num_epochs", 5))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    total_steps = (len(train_loader) * num_epochs) // grad_accum

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        num_training_steps=max(total_steps, 1),
    )

    distill_weight = float(loss_cfg.get("distill_weight", 1.0))
    replay_weight = float(loss_cfg.get("replay_weight", 1.0))
    temperature = float(loss_cfg.get("temperature", 2.0))

    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    log_steps = int(train_cfg.get("log_steps", 50))
    save_steps = int(train_cfg.get("save_steps", 1000))

    retriever.train()
    global_step = 0
    running_loss = 0.0

    for epoch in range(num_epochs):
        progress = tqdm(train_loader, desc=f"Retriever epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress):
            batch = move_batch_to_device(batch, device)

            student_outputs = retriever(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                decoder_attention_mask=batch.get("decoder_attention_mask"),
                return_route_score=True,
            )

            mle_loss = student_outputs.loss

            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    decoder_attention_mask=batch.get("decoder_attention_mask"),
                    pixel_values=batch.get("pixel_values"),
                )

            kd_loss = token_kl_distillation_loss(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_outputs.logits,
                labels=batch["labels"],
                temperature=temperature,
            )

            loss = replay_weight * mle_loss + distill_weight * kd_loss
            loss = loss / grad_accum
            loss.backward()

            running_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(retriever.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % log_steps == 0:
                    avg_loss = running_loss / log_steps
                    progress.set_postfix({"loss": f"{avg_loss:.4f}"})
                    running_loss = 0.0

                if global_step % save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ensure_dir(str(ckpt_dir))
                    retriever.save_pretrained(str(ckpt_dir))
                    tokenizer.save_pretrained(str(ckpt_dir))

        epoch_dir = output_dir / f"epoch-{epoch + 1}"
        ensure_dir(str(epoch_dir))
        retriever.save_pretrained(str(epoch_dir))
        tokenizer.save_pretrained(str(epoch_dir))

    final_dir = output_dir / "final"
    ensure_dir(str(final_dir))
    retriever.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"Saved retriever to {final_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--memorizer_path", type=str, default=None)
    args = parser.parse_args()

    train_retriever(args.config, args.memorizer_path)


if __name__ == "__main__":
    main()