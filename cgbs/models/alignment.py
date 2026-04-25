from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalCoverTextAlignment(nn.Module):
    """
    Hierarchical cover-text alignment module.

    It aligns cover representations with book text representations at two levels:
    1. global alignment: uses the global cover representation to modulate all text tokens;
    2. local alignment: computes token-level attention between text tokens and cover patches.

    Args:
        hidden_size: hidden size of the text encoder.
        dropout: dropout probability.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.text_global_proj = nn.Linear(hidden_size, hidden_size)
        self.cover_global_proj = nn.Linear(hidden_size, hidden_size)

        self.text_local_proj = nn.Linear(hidden_size, hidden_size)
        self.cover_local_proj = nn.Linear(hidden_size, hidden_size)

        self.global_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.local_gate = nn.Linear(hidden_size * 2, hidden_size)

        self.output_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        text_hidden: torch.Tensor,
        cover_hidden: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            text_hidden:
                Tensor with shape [batch_size, text_len, hidden_size].
            cover_hidden:
                Tensor with shape [batch_size, cover_len, hidden_size].
                For ViT, this usually includes CLS and patch tokens.
            text_attention_mask:
                Optional tensor with shape [batch_size, text_len].

        Returns:
            cover-aware text representation with shape [batch_size, text_len, hidden_size].
        """
        if text_hidden.dim() != 3:
            raise ValueError(f"text_hidden must be 3D, got shape {tuple(text_hidden.shape)}")

        if cover_hidden.dim() != 3:
            raise ValueError(f"cover_hidden must be 3D, got shape {tuple(cover_hidden.shape)}")

        if text_hidden.size(0) != cover_hidden.size(0):
            raise ValueError("text_hidden and cover_hidden must have the same batch size.")

        # Global alignment.
        # For ViT, the first token is usually CLS. We use it as the global cover representation.
        cover_global = cover_hidden[:, :1, :]
        cover_global = self.cover_global_proj(cover_global)
        cover_global = cover_global.expand(-1, text_hidden.size(1), -1)

        text_global = self.text_global_proj(text_hidden)
        global_gate = torch.sigmoid(self.global_gate(torch.cat([text_global, cover_global], dim=-1)))
        global_aligned = global_gate * text_global + (1.0 - global_gate) * cover_global

        # Local alignment.
        text_local = self.text_local_proj(text_hidden)
        cover_local = self.cover_local_proj(cover_hidden)

        attn_scores = torch.matmul(text_local, cover_local.transpose(1, 2))
        attn_scores = attn_scores / (self.hidden_size ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        local_context = torch.matmul(attn_weights, cover_local)

        local_gate = torch.sigmoid(self.local_gate(torch.cat([text_local, local_context], dim=-1)))
        local_aligned = local_gate * text_local + (1.0 - local_gate) * local_context

        # Combine global and local aligned representations.
        fused = self.output_proj(torch.cat([global_aligned, local_aligned], dim=-1))
        fused = self.dropout(fused)
        fused = self.layer_norm(text_hidden + fused)

        if text_attention_mask is not None:
            fused = fused * text_attention_mask.unsqueeze(-1).to(fused.dtype)

        return fused