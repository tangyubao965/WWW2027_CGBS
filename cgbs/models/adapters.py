from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """
    Standard bottleneck adapter.

    Args:
        hidden_size: input and output hidden size.
        bottleneck_size: adapter bottleneck size.
        dropout: dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.down(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.up(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(residual + hidden_states)


class RoutedAdapter(nn.Module):
    """
    Query-conditioned routed adapter.

    It contains two adapters:
    1. a general adapter;
    2. a cover-aware adapter.

    A routing gate predicts how much cover-aware behavior should be used
    for the current input.
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.general_adapter = BottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            dropout=dropout,
        )
        self.cover_adapter = BottleneckAdapter(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            dropout=dropout,
        )

        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:
                Tensor with shape [batch_size, seq_len, hidden_size].
            attention_mask:
                Optional tensor with shape [batch_size, seq_len].

        Returns:
            routed_hidden:
                Tensor with shape [batch_size, seq_len, hidden_size].
            route_score:
                Tensor with shape [batch_size], indicating cover-aware routing strength.
        """
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        route_score = torch.sigmoid(self.router(pooled)).squeeze(-1)

        general_hidden = self.general_adapter(hidden_states)
        cover_hidden = self.cover_adapter(hidden_states)

        gate = route_score.view(-1, 1, 1)
        routed_hidden = (1.0 - gate) * general_hidden + gate * cover_hidden

        return routed_hidden, route_score


class AdapterStack(nn.Module):
    """
    A stack of routed adapters applied to hidden states.

    This module is intentionally independent from a specific Transformer backbone.
    It can be applied to encoder hidden states after the encoder, or inserted into
    selected layers if the backbone is later customized.
    """

    def __init__(
        self,
        hidden_size: int,
        num_adapters: int = 1,
        bottleneck_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_adapters <= 0:
            raise ValueError("num_adapters must be positive.")

        self.adapters = nn.ModuleList(
            [
                RoutedAdapter(
                    hidden_size=hidden_size,
                    bottleneck_size=bottleneck_size,
                    dropout=dropout,
                )
                for _ in range(num_adapters)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        route_scores = []

        for adapter in self.adapters:
            hidden_states, route_score = adapter(hidden_states, attention_mask)
            route_scores.append(route_score)

        stacked_scores = torch.stack(route_scores, dim=0)
        mean_route_score = stacked_scores.mean(dim=0)

        return hidden_states, mean_route_score