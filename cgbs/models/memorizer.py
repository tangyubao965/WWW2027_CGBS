from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, ViTModel

from cgbs.models.alignment import HierarchicalCoverTextAlignment


class CoverAwareBookMemorizer(nn.Module):
    """
    Cover-aware book memorizer.

    The memorizer takes book-side textual input and the corresponding cover
    image as input. It fuses text and cover representations with hierarchical
    cover-text alignment, and then uses a shared seq2seq decoder to generate
    either the hierarchical book identifier or the cover caption.

    This module is used as the teacher for the later text-only retriever.
    """

    def __init__(
        self,
        text_model_name: str = "google/long-t5-tglobal-base",
        vision_model_name: str = "google/vit-base-patch16-224",
        dropout: float = 0.1,
        freeze_vision_encoder: bool = False,
    ) -> None:
        super().__init__()

        self.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(text_model_name)
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)

        self.text_hidden_size = self.seq2seq.config.d_model
        self.vision_hidden_size = self.vision_encoder.config.hidden_size

        self.cover_projection = nn.Linear(
            self.vision_hidden_size,
            self.text_hidden_size,
        )

        self.cover_text_alignment = HierarchicalCoverTextAlignment(
            hidden_size=self.text_hidden_size,
            dropout=dropout,
        )

        if freeze_vision_encoder:
            self.freeze_vision_encoder()

    def freeze_vision_encoder(self) -> None:
        """Freeze the cover image encoder."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def unfreeze_vision_encoder(self) -> None:
        """Unfreeze the cover image encoder."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode book-side text with the seq2seq encoder."""
        encoder = self.seq2seq.get_encoder()
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def encode_cover(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Encode cover images and project them to the text hidden space."""
        outputs = self.vision_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        cover_hidden = self.cover_projection(outputs.last_hidden_state)
        return cover_hidden

    def build_cover_aware_memory(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build cover-aware book memory representation.

        If pixel_values is None, this falls back to text-only memory.
        """
        text_hidden = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if pixel_values is None:
            return text_hidden

        cover_hidden = self.encode_cover(pixel_values)
        fused_hidden = self.cover_text_alignment(
            text_hidden=text_hidden,
            cover_hidden=cover_hidden,
            text_attention_mask=attention_mask,
        )
        return fused_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        """
        Forward pass.

        Args:
            input_ids:
                Token ids of book-side text.
            attention_mask:
                Attention mask of book-side text.
            labels:
                Target ids for identifier or caption generation.
            pixel_values:
                Cover images processed by a vision processor.
            decoder_attention_mask:
                Optional decoder attention mask.
            output_hidden_states:
                Whether to return decoder hidden states.
            output_attentions:
                Whether to return attentions.

        Returns:
            Hugging Face Seq2SeqLMOutput.
        """
        memory_hidden = self.build_cover_aware_memory(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        encoder_outputs = self.seq2seq.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_outputs.last_hidden_state = memory_hidden

        outputs = self.seq2seq(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generate identifier or caption from book-side inputs.
        """
        memory_hidden = self.build_cover_aware_memory(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        encoder_outputs = self.seq2seq.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_outputs.last_hidden_state = memory_hidden

        return self.seq2seq.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    def save_pretrained(self, output_dir: str) -> None:
        """
        Save the text seq2seq model and additional memorizer parameters.

        The vision encoder and alignment parameters are saved in a PyTorch file.
        """
        self.seq2seq.save_pretrained(output_dir)

        extra_state = {
            "vision_encoder": self.vision_encoder.state_dict(),
            "cover_projection": self.cover_projection.state_dict(),
            "cover_text_alignment": self.cover_text_alignment.state_dict(),
            "text_hidden_size": self.text_hidden_size,
            "vision_hidden_size": self.vision_hidden_size,
        }

        torch.save(extra_state, f"{output_dir}/memorizer_extra.pt")

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        text_model_name: str = "t5-base",
        vision_model_name: str = "google/vit-base-patch16-224",
        dropout: float = 0.1,
        freeze_vision_encoder: bool = False,
        map_location: str | torch.device = "cpu",
    ) -> "CoverAwareBookMemorizer":
        """
        Load a saved memorizer.

        model_dir should contain:
            - Hugging Face seq2seq files
            - memorizer_extra.pt
        """
        model = cls(
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            dropout=dropout,
            freeze_vision_encoder=freeze_vision_encoder,
        )

        model.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        extra_state = torch.load(
            f"{model_dir}/memorizer_extra.pt",
            map_location=map_location,
        )

        model.vision_encoder.load_state_dict(extra_state["vision_encoder"])
        model.cover_projection.load_state_dict(extra_state["cover_projection"])
        model.cover_text_alignment.load_state_dict(extra_state["cover_text_alignment"])

        return model