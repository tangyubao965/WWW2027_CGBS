from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


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


def resolve_image_path(cover_path: str, image_dir: Optional[str]) -> Path:
    path = Path(cover_path)
    if image_dir is not None and not path.is_absolute():
        path = Path(image_dir) / path
    return path


def load_florence2(model_name: str, device: torch.device):
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def generate_caption(
    processor,
    model,
    image: Image.Image,
    device: torch.device,
    task_prompt: str = "<DETAILED_CAPTION>",
    max_new_tokens: int = 128,
    num_beams: int = 3,
) -> str:
    inputs = processor(
        text=task_prompt,
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
    )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0]

    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=image.size,
    )

    caption = parsed.get(task_prompt, generated_text)
    return str(caption).strip()


def build_cover_captions(
    input_path: str,
    output_path: str,
    image_dir: Optional[str],
    model_name: str,
    device_name: str,
    task_prompt: str,
    max_new_tokens: int,
    num_beams: int,
    overwrite: bool = False,
) -> None:
    records = read_jsonl(input_path)

    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    processor, model = load_florence2(model_name, device)

    for record in tqdm(records, desc="Generating cover captions with Florence-2"):
        if record.get("cover_caption") and not overwrite:
            continue

        cover_path = record.get("cover_path", "")
        if not cover_path:
            record["cover_caption"] = ""
            continue

        image_path = resolve_image_path(cover_path, image_dir)
        if not image_path.exists():
            record["cover_caption"] = ""
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            record["cover_caption"] = generate_caption(
                processor=processor,
                model=model,
                image=image,
                device=device,
                task_prompt=task_prompt,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        except Exception as exc:
            record["cover_caption"] = ""
            record["caption_error"] = str(exc)

    write_jsonl(records, output_path)
    print(f"Saved records with cover captions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Florence-2-base",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task_prompt", type=str, default="<DETAILED_CAPTION>")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_cover_captions(
        input_path=args.input,
        output_path=args.output,
        image_dir=args.image_dir,
        model_name=args.model_name,
        device_name=args.device,
        task_prompt=args.task_prompt,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()