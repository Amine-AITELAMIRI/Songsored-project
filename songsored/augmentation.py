"""Augmentation transforms for spectrogram features."""

from __future__ import annotations

import torch


def random_frequency_mask(
    features: torch.Tensor,
    max_width: int = 8,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mask a randomly chosen band of frequency channels.

    The same frequency band is set to zero across every time frame. In
    normalized feature space, zero represents the mean feature value. The input
    tensor is left unchanged.

    Args:
        features: Floating-point spectrogram shaped (1, F, T), where F is the
            number of frequency channels and T is the number of frames.
        max_width: Largest band that may be masked. Must be in [0, F). A value
            of zero returns a clone without drawing random numbers.
        generator: Optional CPU random generator for choosing the mask width
            and start channel.

    Returns:
        A copy of ``features`` with the mask applied, preserving its shape,
        dtype, and device.

    Raises:
        TypeError: If ``features`` is not a tensor or ``generator`` is not a
            ``torch.Generator``.
        ValueError: If features is not floating-point, has an invalid shape,
            contains non-finite values, or ``max_width`` is not an integer in
            [0, F).
    """
    if not isinstance(features, torch.Tensor):
        raise TypeError(
            f"features must be a torch.Tensor, got {type(features).__name__}."
        )
    if not torch.is_floating_point(features):
        raise ValueError(
            f"features must be a floating-point tensor, got {features.dtype}."
        )
    if (
        features.ndim != 3
        or features.shape[0] != 1
        or features.shape[1] <= 0
        or features.shape[2] <= 0
    ):
        raise ValueError(
            f"features must have shape (1, F, T) with F, T > 0, got {tuple(features.shape)}."
        )
    if not torch.isfinite(features).all():
        raise ValueError("features contains non-finite values (NaN or inf).")

    if isinstance(max_width, bool) or not isinstance(max_width, int):
        raise ValueError(
            f"max_width must be an integer (excluding bool), got {max_width!r}."
        )

    frequency_count = features.shape[1]
    if not (0 <= max_width < frequency_count):
        raise ValueError(
            f"max_width must satisfy 0 <= max_width < {frequency_count}, got {max_width}."
        )

    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError(
            f"generator must be a torch.Generator or None, got {type(generator).__name__}."
        )

    masked_features = features.clone()
    if max_width == 0:
        return masked_features

    # Allow a zero-width draw, which leaves the cloned input unchanged.
    width = int(
        torch.randint(
            0,
            max_width + 1,
            (1,),
            generator=generator,
        ).item()
    )

    if width == 0:
        return masked_features

    start = int(
        torch.randint(
            0,
            frequency_count - width + 1,
            (1,),
            generator=generator,
        ).item()
    )

    masked_features[:, start : start + width, :] = 0.0
    return masked_features
