"""Linear classifier for embeddings from a frozen speech encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpeechFeatureClassifier(nn.Module):
    """Map each pooled speech embedding to a target-presence logit.

    Args:
        embedding_dim: Number of values in each input embedding. Defaults to 1024.
    """

    def __init__(self, embedding_dim: int = 1024) -> None:
        super().__init__()
        if (
            not isinstance(embedding_dim, int)
            or isinstance(embedding_dim, bool)
            or embedding_dim <= 0
        ):
            raise ValueError(
                f"embedding_dim must be a positive integer, got {embedding_dim!r}."
            )
        self.embedding_dim: int = embedding_dim
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one raw classification logit for each input embedding.

        Args:
            features: Floating-point tensor shaped (batch_size, embedding_dim).

        Returns:
            Logits shaped (batch_size, 1).

        Raises:
            TypeError: If features is not a torch.Tensor.
            ValueError: If features is not 2D or its embedding size is incorrect.
        """
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"Expected features to be a torch.Tensor, got {type(features).__name__}."
            )

        if features.ndim != 2 or features.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected features with shape [batch_size, {self.embedding_dim}], "
                f"got {tuple(features.shape)}."
            )

        return self.classifier(features)
