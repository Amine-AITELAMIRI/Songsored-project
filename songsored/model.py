"""CNN classifier for detecting target words in audio."""

from __future__ import annotations

import torch
import torch.nn as nn


class WordClassifier(nn.Module):
    """Classify mel-spectrogram patches with a small 2-D CNN.

    Args:
        n_mels: Number of mel bands in the input spectrogram.
        num_classes: Number of output logits. Defaults to one for binary detection.
        pool_size: Output size of adaptive pooling as (frequency_bins, time_frames).
    """

    def __init__(
        self,
        n_mels: int = 128,
        num_classes: int = 1,
        pool_size: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.num_classes = num_classes
        self.pool_size = pool_size

        self.features = nn.Sequential(
            # Block 1: 1 → 16 channels
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 16 → 32 channels
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(self.pool_size)
        pooled_features = 32 * self.pool_size[0] * self.pool_size[1]
        self.classifier = nn.Linear(pooled_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of spectrogram patches.

        Args:
            x: Tensor shaped (batch, 1, n_mels, time_frames).

        Returns:
            Logits shaped (batch, num_classes).
        """
        x = self.features(x)          # (batch, 32, H', W')
        x = self.global_pool(x)        # (batch, 32, pool_h, pool_w)
        x = torch.flatten(x, start_dim=1)  # (batch, 32 * pool_h * pool_w)
        return self.classifier(x)      # (batch, num_classes)
