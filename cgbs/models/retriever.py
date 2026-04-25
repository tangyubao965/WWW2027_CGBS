from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from cgbs.models.adapters import AdapterStack


class TextOnlyRetriever(nn.Module):
    """
    Text-only generative retriever.

    The retriever takes a textual user query as input and generates a book
    identifier in the same identifier space as the memorizer. It optionally
    uses routed adapters to modulate query-side cover-aware behavior.
    """

    def __init__(
        self,
        text_model_name: str = "google/long-t5-tglobal-base",
        use_adapters: bool = True,
        num_adapters: int = 1,
        adapter_bottleneck_size: int = 64,
        adapter_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(text_model_name)
        self.hidden_size = self.seq2seq.config.d_model
        self.use_adapters = use_adapters

        if use_adapters:
            self.adapter_stack = AdapterStack(
                hidden_size=self.hidden_size,
                num_adapters=num_adapters,
                bottleneck_size=adapter_bottleneck_size,
                dropout=adapter_dropout,
            )
        else:
            self.adapter_stack = None

    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode textual query and optionally apply routed adapters.

        Returns:
            query_hidden:
                Tensor with shape [batch_size, seq_len, hidden_size].
            route_score:
                Tensor with shape [batch_size], or None if adapters are disabled.
        """
        encoder = self.seq2seq.get_encoder()
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        query_hidden = outputs.last_hidden_state
        route_score = None

        if self.adapter_stack is not None:
            query_hidden, route_score = self.adapter_stack(
                hidden_states=query_hidden,
                attention_mask=attention_mask,
            )

        return query_hidden, route_score

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_route_score: bool = False,
    ):
        """
        Forward pass.

        Args:
            input_ids:
                Token ids of textual user queries.
            attention_mask:
                Query attention mask.
            labels:
                Target identifier token ids.
            decoder_attention_mask:
                Optional decoder attention mask.
            output_hidden_states:
                Whether to return decoder hidden states.
            output_attentions:
                Whether to return attentions.
            return_route_score:
                Whether to attach route_score to the output object.

        Returns:
            Hugging Face Seq2SeqLMOutput. If return_route_score=True,
            the output object has an additional attribute route_score.
        """
        query_hidden, route_score = self.encode_query(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        encoder_outputs = BaseModelOutput(last_hidden_state=query_hidden)

        outputs = self.seq2seq(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        if return_route_score:
            outputs.route_score = route_score

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_route_score: bool = False,
        **generate_kwargs,
    ):
        """
        Generate book identifiers from textual user queries.

        If return_route_score=True, returns a tuple:
            generated_ids, route_score
        Otherwise, returns generated_ids.
        """
        query_hidden, route_score = self.encode_query(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        encoder_outputs = BaseModelOutput(last_hidden_state=query_hidden)

        generated_ids = self.seq2seq.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        if return_route_score:
            return generated_ids, route_score

        return generated_ids

    def initialize_from_memorizer(self, memorizer) -> None:
        """
        Initialize retriever seq2seq parameters from a trained memorizer.

        This copies the shared encoder-decoder backbone from the memorizer.
        The retriever-specific adapters are kept unchanged.
        """
        self.seq2seq.load_state_dict(memorizer.seq2seq.state_dict())

    def freeze_backbone(self) -> None:
        """Freeze the seq2seq backbone and train only adapters."""
        for param in self.seq2seq.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the seq2seq backbone."""
        for param in self.seq2seq.parameters():
            param.requires_grad = True

    def save_pretrained(self, output_dir: str) -> None:
        """
        Save retriever backbone and adapter parameters.
        """
        self.seq2seq.save_pretrained(output_dir)

        extra_state = {
            "use_adapters": self.use_adapters,
            "adapter_stack": self.adapter_stack.state_dict()
            if self.adapter_stack is not None
            else None,
        }

        torch.save(extra_state, f"{output_dir}/retriever_extra.pt")

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        text_model_name: str = "t5-base",
        use_adapters: bool = True,
        num_adapters: int = 1,
        adapter_bottleneck_size: int = 64,
        adapter_dropout: float = 0.1,
        map_location: str | torch.device = "cpu",
    ) -> "TextOnlyRetriever":
        """
        Load a saved retriever.
        """
        model = cls(
            text_model_name=text_model_name,
            use_adapters=use_adapters,
            num_adapters=num_adapters,
            adapter_bottleneck_size=adapter_bottleneck_size,
            adapter_dropout=adapter_dropout,
        )

        model.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        extra_path = f"{model_dir}/retriever_extra.pt"
        extra_state = torch.load(extra_path, map_location=map_location)

        if model.adapter_stack is not None and extra_state["adapter_stack"] is not None:
            model.adapter_stack.load_state_dict(extra_state["adapter_stack"])

        return model