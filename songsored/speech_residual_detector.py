"""Score a sequence of frozen speech features with a small temporal head."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Combine nearby frames while keeping the input through a skip connection."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.conv(x).relu()).relu()


class ResidualSpeechDetector(nn.Module):
    """Return frame logits, or one pooled logit for the comparison model."""

    def __init__(self, embedding_dim: int = 1024, channels: int = 64) -> None:
        super().__init__()
        if any(type(size) is not int or size <= 0 for size in (embedding_dim, channels)):
            raise ValueError("Feature and channel dimensions must be positive integers.")
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(embedding_dim, channels)
        self.blocks = nn.Sequential(
            *(ResidualBlock(channels, dilation) for dilation in (1, 2, 4))
        )
        self.classifier = nn.Linear(channels, 1)

    def forward(
        self,
        features: torch.Tensor,
        scored_mask: torch.Tensor,
        pooled: bool = False,
    ) -> torch.Tensor:
        """Accept features shaped (batch, frames, embedding_dim).

        The boolean mask has shape (batch, frames). In pooled mode it selects
        frames to average. In temporal mode all frame logits are returned;
        the caller selects which ones to score. Context outside the mask
        can influence selected frames through the convolutions.
        """
        if features.ndim != 3 or features.shape[-1] != self.embedding_dim:
            raise ValueError("Expected features shaped (batch, frames, embedding_dim).")
        if (
            scored_mask.shape != features.shape[:2]
            or scored_mask.dtype != torch.bool
            or not scored_mask.any(1).all()
        ):
            raise ValueError("Each sequence needs a boolean mask with at least one selected frame.")

        x = self.projection(features).relu()
        x = x.transpose(1, 2)  # Conv1d expects (batch, channels, frames).
        x = self.blocks(x).transpose(1, 2)

        if pooled:
            x = (x * scored_mask.unsqueeze(-1)).sum(1)
            x = x / scored_mask.sum(1, keepdim=True)
        return self.classifier(x).squeeze(-1)
