from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_json(path: str) -> Any:
    """Read a JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict]:
    """Read a JSONL file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_id} in {file_path}") from exc
    return records


def write_jsonl(records: Iterable[Dict], path: str) -> None:
    """Write records to a JSONL file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    """Normalize a text field."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_id(value: Any) -> str:
    """Normalize an id field."""
    if value is None:
        return ""
    return str(value).strip()


def safe_get(record: Dict, *keys: str, default: Any = "") -> Any:
    """Return the first non-empty field from a record."""
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return default


def list_to_text(values: Any, max_items: Optional[int] = None) -> str:
    """Convert a list-like field into readable text."""
    if values is None:
        return ""

    if isinstance(values, str):
        return normalize_text(values)

    if isinstance(values, dict):
        items = []
        for key, value in values.items():
            items.append(f"{key}: {value}")
        values = items

    if not isinstance(values, list):
        return normalize_text(values)

    if max_items is not None:
        values = values[:max_items]

    return " ".join(normalize_text(v) for v in values if normalize_text(v))


def build_readable_book_text(
    title: str = "",
    author: str = "",
    publisher: str = "",
    description: str = "",
    outline: str = "",
    extra_fields: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a readable book text string from available fields."""
    parts = []

    title = normalize_text(title)
    author = normalize_text(author)
    publisher = normalize_text(publisher)
    description = normalize_text(description)
    outline = normalize_text(outline)

    if title:
        parts.append(f"Title: {title}")
    if author:
        parts.append(f"Author: {author}")
    if publisher:
        parts.append(f"Publisher: {publisher}")
    if outline:
        parts.append(f"Outline: {outline}")
    if description:
        parts.append(f"Description: {description}")

    if extra_fields:
        for name, value in extra_fields.items():
            value_text = list_to_text(value)
            if value_text:
                parts.append(f"{name}: {value_text}")

    return "\n".join(parts)


def infer_split(
    index: int,
    train_ratio: float,
    valid_ratio: float,
    total: int,
) -> str:
    """Assign a split by index after shuffling or fixed ordering."""
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    if index < train_end:
        return "train"
    if index < valid_end:
        return "valid"
    return "test"