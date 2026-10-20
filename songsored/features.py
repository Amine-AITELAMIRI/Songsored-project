"""Helpers for turning audio into mel-spectrogram features."""

from __future__ import annotations

import librosa
import numpy as np


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Return a log-scaled mel spectrogram for a mono waveform."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    log_mel = librosa.power_to_db(mel)
    return log_mel.astype(np.float32)
