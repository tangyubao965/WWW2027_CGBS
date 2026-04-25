from typing import Dict, List, Optional

import torch


class CGBSCollator:
    """Collator for text-only and text-cover CGBS batches."""

    def __init__(
        self,
        tokenizer,
        image_processor: Optional[object] = None,
        max_input_length: int = 512,
        max_target_length: int = 128,
        use_cover: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.use_cover = use_cover

    def __call__(self, batch: List[Dict]) -> Dict:
        source_texts = [x["source_text"] for x in batch]
        target_texts = [x["target_text"] for x in batch]

        source = self.tokenizer(
            source_texts,
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )

        target = self.tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        labels = target["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        output = {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "labels": labels,
            "decoder_attention_mask": target["attention_mask"],
            "book_ids": [x["book_id"] for x in batch],
            "identifiers": [x["identifier"] for x in batch],
        }

        if self.use_cover and self.image_processor is not None:
            images = [x.get("cover_image") for x in batch]
            if all(img is not None for img in images):
                image_inputs = self.image_processor(
                    images=images,
                    return_tensors="pt",
                )
                output["pixel_values"] = image_inputs["pixel_values"]

        return output