"""Fit and apply normalization for audio features."""

from __future__ import annotations

import math
from typing import Any

import torch


def fit_feature_stats(train_dataset: Any) -> dict[str, Any]:
    """Compute global mean and standard deviation from raw training features.

    Accumulate the values in float64 so the statistics are more precise than
    the input tensors. The returned standard deviation is the population
    standard deviation across every value in the dataset.

    Args:
        train_dataset: Non-empty training dataset with ``split == "train"``
            and no transform applied.

    Returns:
        Dictionary containing:
            - ``method``: ``"global_log_mel"``.
            - ``mean``: Mean across all feature values.
            - ``std``: Population standard deviation.
            - ``count``: Number of values included in the calculation.
            - ``eps``: Small positive value used as a minimum scale.

    Raises:
        ValueError: If the dataset is not the raw training split, is empty, or
            contains empty or non-finite features.
        TypeError: If a dataset item is not a tensor.
    """
    # Only fit on the unmodified training split to avoid data leakage.
    if not hasattr(train_dataset, "split") or train_dataset.split != "train":
        raise ValueError(
            f"train_dataset must have split == 'train', got {getattr(train_dataset, 'split', None)!r}."
        )
    if not hasattr(train_dataset, "transform") or train_dataset.transform is not None:
        raise ValueError(
            "train_dataset.transform must be None to avoid fitting statistics on "
            "already-normalized or augmented features."
        )
    if len(train_dataset) == 0:
        raise ValueError("train_dataset must be non-empty.")

    total_sum = 0.0
    total_squared_sum = 0.0
    count: int = 0

    # Accumulate every raw feature value in double precision.
    for sample_index in range(len(train_dataset)):
        features, _ = train_dataset[sample_index]
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor at dataset index {sample_index}, got {type(features).__name__}."
            )
        if features.numel() == 0:
            raise ValueError(f"Features at dataset index {sample_index} are empty.")
        if not torch.isfinite(features).all():
            raise ValueError(
                f"Features at dataset index {sample_index} contain non-finite values (NaN or inf)."
            )

        features64 = features.to(torch.float64)
        total_sum += float(features64.sum().item())
        total_squared_sum += float((features64**2).sum().item())
        count += features.numel()

    if count == 0:
        raise ValueError("No feature values were accumulated from train_dataset.")

    # Use E[x²] - E[x]² for population variance and guard rounding below zero.
    mean = total_sum / count
    variance = max(total_squared_sum / count - mean**2, 0.0)
    std = variance**0.5

    return {
        "method": "global_log_mel",
        "mean": float(mean),
        "std": float(std),
        "count": int(count),
        "eps": 1e-6,
    }


def normalize_features(
    features: torch.Tensor,
    stats: dict[str, Any],
) -> torch.Tensor:
    """Standardize features with previously fitted statistics.

    The scale is ``max(std, eps)`` to avoid division by a value near zero.

    Args:
        features: Floating-point tensor to normalize.
        stats: Dictionary containing finite ``mean``, ``std``, and ``eps`` values.

    Returns:
        A new tensor with the same shape, device, and dtype as ``features``.

    Raises:
        TypeError: If features is not a torch.Tensor or stats is not a dict.
        ValueError: If features or statistics contain non-finite values, if features
            is not floating-point, if stats['std'] is negative, or if stats['eps'] is non-positive.
    """
    if not isinstance(features, torch.Tensor):
        raise TypeError(
            f"features must be a torch.Tensor, got {type(features).__name__}."
        )
    if not torch.is_floating_point(features):
        raise ValueError(
            f"features must be a floating-point tensor, got {features.dtype}."
        )
    if not torch.isfinite(features).all():
        raise ValueError("features tensor contains non-finite values (NaN or inf).")

    if not isinstance(stats, dict):
        raise TypeError(f"stats must be a dictionary, got {type(stats).__name__}.")

    # Check the supplied statistics before using them in the calculation.
    for name in ("mean", "std", "eps"):
        if name not in stats:
            raise ValueError(f"stats dictionary missing required key: {name!r}")
        value = stats[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"stats[{name!r}] must be a finite number, got {value!r}")

    std = float(stats["std"])
    if std < 0:
        raise ValueError(f"stats['std'] must be non-negative, got {std}.")

    eps = float(stats["eps"])
    if eps <= 0:
        raise ValueError(f"stats['eps'] must be positive, got {eps}.")

    mean = float(stats["mean"])
    scale = max(std, eps)

    mean_t = torch.tensor(mean, dtype=features.dtype, device=features.device)
    scale_t = torch.tensor(scale, dtype=features.dtype, device=features.device)

    return (features - mean_t) / scale_t
