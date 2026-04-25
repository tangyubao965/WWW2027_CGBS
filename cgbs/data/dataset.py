import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class BookSearchDataset(Dataset):
    """
    Dataset for CGBS.

    Each JSONL example can contain:
    {
        "book_id": "...",
        "query": "...",
        "book_text": "...",
        "identifier": "...",
        "cover_path": "...",
        "cover_caption": "...",
        "visual_tag": "..."
    }

    use_query=False: source_text = book_text, used by the memorizer.
    use_query=True: source_text = query, used by the retriever.
    use_caption=True: target_text = cover_caption.
    use_caption=False: target_text = identifier.
    """

    def __init__(
        self,
        file_path: str,
        image_dir: Optional[str] = None,
        use_query: bool = False,
        use_cover: bool = True,
        use_caption: bool = False,
    ) -> None:
        self.file_path = Path(file_path)
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.use_query = use_query
        self.use_cover = use_cover
        self.use_caption = use_caption

        self.examples = self._load_jsonl(self.file_path)

    @staticmethod
    def _load_jsonl(file_path: Path) -> List[Dict]:
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        examples = []
        with file_path.open("r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_id} in {file_path}"
                    ) from exc
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _load_image(self, cover_path: str) -> Image.Image:
        path = Path(cover_path)
        if self.image_dir is not None and not path.is_absolute():
            path = self.image_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Cover image not found: {path}")

        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]

        if self.use_query:
            source_text = ex.get("query", "")
        else:
            source_text = ex.get("book_text", "")

        if self.use_caption:
            target_text = ex.get("cover_caption", "")
        else:
            target_text = ex.get("identifier", "")

        item = {
            "book_id": ex.get("book_id", str(idx)),
            "source_text": source_text,
            "target_text": target_text,
            "identifier": ex.get("identifier", ""),
            "query": ex.get("query", ""),
            "book_text": ex.get("book_text", ""),
            "visual_tag": ex.get("visual_tag", ""),
        }

        if self.use_cover and ex.get("cover_path"):
            item["cover_image"] = self._load_image(ex["cover_path"])
        else:
            item["cover_image"] = None

        return item