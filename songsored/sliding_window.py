"""Score complete one-second windows of an audio file with the CNN."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import librosa
import numpy as np
import torch

from .cnn_inference import CNNPredictor, FEATURE_CONFIG
from .features import extract_mel_spectrogram
from .normalization import normalize_features


class WindowScore(TypedDict):
    start_s: float
    end_s: float
    score: float


def iter_audio_windows(
    audio: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield the start, exclusive end, and samples of each complete window."""
    if not isinstance(audio, np.ndarray) or audio.ndim != 1:
        raise ValueError("Audio must be a one-dimensional NumPy array.")
    for name, size in (("window_samples", window_samples), ("hop_samples", hop_samples)):
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    last_start = len(audio) - window_samples
    for start in range(0, last_start + 1, hop_samples):
        end = start + window_samples
        yield start, end, audio[start:end]


@torch.inference_mode()
def score_cnn_windows(
    audio_path: str | Path,
    checkpoint_path: str | Path,
    *,
    hop_samples: int = 2205,
) -> list[WindowScore]:
    """Return scores on CPU, advancing 100 ms at the default hop.

    Short files and incomplete final windows are skipped. The timestamps
    describe the input window; they do not locate a word inside it.
    """
    predictor = CNNPredictor.from_checkpoint(checkpoint_path)
    sample_rate = FEATURE_CONFIG["sample_rate"]
    window_samples = round(sample_rate * FEATURE_CONFIG["clip_duration"])
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty and finite.")

    scores: list[WindowScore] = []
    for start, end, window in iter_audio_windows(audio, window_samples, hop_samples):
        mel = extract_mel_spectrogram(window)
        # The CNN expects (batch, channel, mel bands, time frames).
        features = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
        if predictor.stats is not None:
            features = normalize_features(features, predictor.stats)
        logit = predictor.model(features)
        if not torch.isfinite(logit).all():
            raise ValueError(f"The CNN produced a non-finite logit at sample {start}.")
        scores.append({
            "start_s": start / sample_rate,
            "end_s": end / sample_rate,
            "score": logit.sigmoid().item(),
        })
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--hop-samples", default=2205, type=int, help="Hop at 22,050 Hz (default: 100 ms)")
    args = parser.parse_args()
    scores = score_cnn_windows(args.audio, args.checkpoint, hop_samples=args.hop_samples)
    print(json.dumps(scores, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
