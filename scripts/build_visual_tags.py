from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yake
from tqdm import tqdm


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


def build_yake_extractor(
    language: str = "en",
    max_ngram_size: int = 3,
    dedup_threshold: float = 0.9,
    num_keywords: int = 6,
):
    return yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=dedup_threshold,
        top=num_keywords,
        features=None,
    )


def extract_visual_tag(
    caption: str,
    extractor,
    num_keywords: int = 6,
) -> str:
    caption = str(caption).strip()
    if not caption:
        return ""

    keywords = extractor.extract_keywords(caption)

    selected = []
    seen = set()

    for keyword, score in keywords:
        keyword = " ".join(str(keyword).lower().strip().split())
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)
        selected.append(keyword)

        if len(selected) >= num_keywords:
            break

    return " ".join(selected)


def build_visual_tags(
    input_path: str,
    output_path: str,
    caption_field: str,
    output_field: str,
    language: str,
    max_ngram_size: int,
    dedup_threshold: float,
    num_keywords: int,
    overwrite: bool = False,
) -> None:
    records = read_jsonl(input_path)

    extractor = build_yake_extractor(
        language=language,
        max_ngram_size=max_ngram_size,
        dedup_threshold=dedup_threshold,
        num_keywords=num_keywords,
    )

    for record in tqdm(records, desc="Extracting visual tags with YAKE"):
        if record.get(output_field) and not overwrite:
            continue

        caption = record.get(caption_field, "")
        record[output_field] = extract_visual_tag(
            caption=caption,
            extractor=extractor,
            num_keywords=num_keywords,
        )

    write_jsonl(records, output_path)
    print(f"Saved records with visual tags to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--caption_field", type=str, default="cover_caption")
    parser.add_argument("--output_field", type=str, default="visual_tag")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--max_ngram_size", type=int, default=3)
    parser.add_argument("--dedup_threshold", type=float, default=0.9)
    parser.add_argument("--num_keywords", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_visual_tags(
        input_path=args.input,
        output_path=args.output,
        caption_field=args.caption_field,
        output_field=args.output_field,
        language=args.language,
        max_ngram_size=args.max_ngram_size,
        dedup_threshold=args.dedup_threshold,
        num_keywords=args.num_keywords,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()