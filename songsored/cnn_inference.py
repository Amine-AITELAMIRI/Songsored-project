"""Run CPU inference on one clip with a trained CNN checkpoint.

Uses the project's existing feature extraction, normalization, and
``WordClassifier``. The repository does not include trained weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch

from .features import extract_mel_spectrogram
from .model import WordClassifier
from .normalization import normalize_features


FEATURE_CONFIG = {
    "sample_rate": 22050,
    "clip_duration": 1.0,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
}


def load_cnn_features(path: str | Path) -> torch.Tensor:
    """Load a clip and prepare its log-mel features for the CNN."""
    audio, _ = librosa.load(path, sr=22050, mono=True, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty and finite.")
    if len(audio) > 22050:
        raise ValueError("Clip is longer than one second; supply a prepared clip.")

    # Match the dataset preprocessing by padding shorter clips on the right.
    audio = np.pad(audio, (0, 22050 - len(audio)))
    mel = extract_mel_spectrogram(audio)
    return torch.from_numpy(mel).unsqueeze(0)  # (channel=1, mel=128, frames=44)


class CNNPredictor:
    """Load a CNN checkpoint and use its saved preprocessing settings."""

    def __init__(self, checkpoint: dict) -> None:
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected a checkpoint dictionary.")
        if checkpoint.get("feature_config") != FEATURE_CONFIG:
            raise ValueError("Checkpoint must use the supported one-second CNN features.")

        config = checkpoint.get("model_config")
        expected_config_keys = {"n_mels", "num_classes", "pool_size"}
        if not isinstance(config, dict) or set(config) != expected_config_keys:
            raise ValueError("Checkpoint is missing the CNN model configuration.")
        if config["n_mels"] != 128 or config["num_classes"] != 1:
            raise ValueError("Expected 128 mel bands and one binary output.")
        if config["pool_size"] not in ((1, 1), (1, 4)):
            raise ValueError("Supported CNN pooling sizes are (1, 1) and (1, 4).")

        # Missing statistics are ambiguous; None explicitly means no normalization.
        if "normalization" not in checkpoint:
            raise ValueError("Checkpoint must explicitly record normalization.")
        stats = checkpoint["normalization"]
        if stats is not None:
            if not isinstance(stats, dict) or stats.get("method") != "global_log_mel":
                raise ValueError("Expected global_log_mel normalization.")
            normalize_features(torch.zeros(1), stats)  # Validate the saved values.
            stats = dict(stats)
        self.stats = stats

        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict) or not state:
            raise ValueError("Checkpoint is missing model weights.")
        if any(
            not isinstance(weight, torch.Tensor) or not torch.isfinite(weight).all()
            for weight in state.values()
        ):
            raise ValueError("Model weights must be finite tensors.")

        self.model = WordClassifier(**config)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> CNNPredictor:
        """Load model weights from a trusted checkpoint file."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls(checkpoint)

    def predict_file(self, path: str | Path) -> dict:
        """Return the target probability and binary decision for one clip."""
        features = load_cnn_features(path)
        if self.stats is not None:
            features = normalize_features(features, self.stats)

        with torch.no_grad():
            batch = features.unsqueeze(0)
            logit = self.model(batch)
            probability = logit.sigmoid().item()

        if not np.isfinite(probability):
            raise ValueError("Inference produced a non-finite probability.")
        return {
            "target_probability": probability,
            "target_present": probability >= 0.5,
            "threshold": 0.5,
            "scope": "one-second clip, selected target families only",
        }


def main() -> None:
    """Parse command-line arguments and print the prediction as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Generated tone and random CNN; no accuracy claim")
    args = parser.parse_args()
    if args.smoke:
        if args.audio or args.checkpoint:
            parser.error("--smoke cannot be combined with audio or checkpoint arguments")
        print(json.dumps(smoke(), indent=2))
        return
    if args.audio is None or args.checkpoint is None:
        parser.error("Supply both --audio and --checkpoint, or use --smoke")
    predictor = CNNPredictor.from_checkpoint(args.checkpoint)
    print(json.dumps(predictor.predict_file(args.audio), indent=2))


def smoke() -> dict:
    """Exercise the actual CNN preprocessing/inference path with random weights."""
    import tempfile
    import soundfile as sf

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = WordClassifier()
        predictor = CNNPredictor({
            "feature_config": FEATURE_CONFIG,
            "model_config": {"n_mels": 128, "num_classes": 1, "pool_size": (1, 1)},
            "normalization": None,
            "model_state_dict": model.state_dict(),
        })
    samples = np.arange(22050) / 22050
    tone = (0.1 * np.sin(2 * np.pi * 440 * samples)).astype(np.float32)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "generated.wav"
        sf.write(path, tone, 22050)
        result = predictor.predict_file(path)
    return {
        "mode": "smoke",
        "trained": False,
        "note": "Generated tone and random CNN weights; not a trained word prediction.",
        "forward_output": result,
    }


if __name__ == "__main__":
    main()
