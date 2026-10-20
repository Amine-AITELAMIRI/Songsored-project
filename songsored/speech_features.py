"""Utilities for loading one-second speech clips at the XLS-R sample rate."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16000
TARGET_SAMPLES = 16000


def load_speech_waveform(path: str | Path) -> np.ndarray:
    """Load a single-channel audio clip as a 1-second float32 waveform.

    The waveform is resampled to 16 kHz and padded to exactly 16,000 samples.
    Clips longer than one second are rejected.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Audio clip file not found: {path_obj}")

    try:
        audio, _ = librosa.load(
            path_obj,
            sr=SAMPLE_RATE,
            mono=True,
            dtype=np.float32,
        )
    except Exception as exc:
        raise ValueError(f"Failed to decode audio file: {path_obj}") from exc

    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path_obj}")

    if not np.isfinite(audio).all():
        raise ValueError(f"Audio contains NaN or infinite values: {path_obj}")

    sample_count = len(audio)
    if sample_count < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - sample_count))
    elif sample_count > TARGET_SAMPLES:
        raise ValueError(
            f"Clip is longer than 1.0 second ({sample_count} samples > {TARGET_SAMPLES} "
            f"at {SAMPLE_RATE} Hz): {path_obj}. Must not exceed 16,000 samples."
        )

    return audio.astype(np.float32, copy=False)
