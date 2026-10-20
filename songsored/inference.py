"""Run single-clip inference with the original global-mean XLS-R probe.

The predictor applies training-set normalization before passing each embedding
to the classifier. No trained checkpoint is included in the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import torch

from .speech_classifier import SpeechFeatureClassifier
from .speech_features import load_speech_waveform

MODEL_ID = "facebook/wav2vec2-xls-r-300m"
REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
DIM = 1024


class SpeechPredictor:
    """Load a trained classifier and its training-only normalization stats."""

    def __init__(self, checkpoint: dict) -> None:
        metadata = checkpoint.get("cache_metadata", {})
        expected = {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "sample_rate": 16000,
            "target_samples": 16000,
            "pooling": "last_hidden_state_mean",
            "embedding_dim": DIM,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Checkpoint must use the pinned original global-mean XLS-R representation."
            )
        if checkpoint.get("embedding_dim") != DIM:
            raise ValueError("Checkpoint embedding dimension must be 1024.")

        stats = checkpoint.get("stats", {})
        for key in ("mean", "scale"):
            value = stats.get(key)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (DIM,)
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid checkpoint normalization {key}.")
        if not (stats["scale"] > 0).all():
            raise ValueError("Normalization scale must be positive.")
        self.mean = stats["mean"].detach().cpu().float().clone()
        self.scale = stats["scale"].detach().cpu().float().clone()

        state = checkpoint.get("model_state_dict", {})
        if set(state) != {"classifier.weight", "classifier.bias"}:
            raise ValueError(
                "Checkpoint must contain exactly the linear head parameters."
            )
        if any(
            not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
            for value in state.values()
        ):
            raise ValueError("Head parameters must be finite tensors.")

        self.model = SpeechFeatureClassifier(DIM)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "SpeechPredictor":
        """Load a checkpoint from a trusted source."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected a checkpoint dictionary.")
        return cls(checkpoint)

    def probability(self, embedding: torch.Tensor) -> float:
        if (
            not isinstance(embedding, torch.Tensor)
            or embedding.shape != (DIM,)
            or not embedding.is_floating_point()
            or not torch.isfinite(embedding).all()
        ):
            raise ValueError("Expected one finite floating-point 1024-dimensional embedding.")

        normalized_embedding = (
            embedding.detach().cpu().float() - self.mean
        ) / self.scale
        with torch.no_grad():
            probability = (
                self.model(normalized_embedding.unsqueeze(0)).sigmoid().item()
            )
        if not np.isfinite(probability):
            raise ValueError("Inference produced a non-finite probability.")
        return probability

    def predict_file(self, path: str | Path, encoder) -> dict:
        """Encode a clip and return its target probability and decision."""
        if (
            encoder.model_id != MODEL_ID
            or encoder.revision != REVISION
            or encoder.embedding_dim != DIM
        ):
            raise ValueError("Encoder does not match this checkpoint.")

        probability = self.probability(encoder.encode(load_speech_waveform(path)))
        return {
            "target_probability": probability,
            "target_present": probability >= 0.5,
            "threshold": 0.5,
            "scope": "one-second clip, selected target families only",
        }


def make_encoder(download: bool = False):
    """Create the pinned speech encoder, optionally allowing a model download."""
    from .speech_encoder import FrozenSpeechEncoder

    return FrozenSpeechEncoder(MODEL_ID, revision=REVISION, local_files_only=not download)


def smoke(use_encoder: bool = False, download: bool = False) -> dict:
    """Check the inference path with a generated tone and a random classifier."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as folder:
        audio_path = Path(folder) / "generated.wav"
        sample_times = np.arange(16000) / 16000
        waveform = (
            0.1 * np.sin(2 * np.pi * 440 * sample_times)
        ).astype(np.float32)
        sf.write(audio_path, waveform, 16000, subtype="FLOAT")
        loaded_waveform = load_speech_waveform(audio_path)

    if use_encoder:
        embedding = make_encoder(download).encode(loaded_waveform)
    else:
        embedding = torch.zeros(DIM)

    with torch.no_grad():
        model = SpeechFeatureClassifier(DIM).eval()
        logits = model(embedding.unsqueeze(0))
    if not torch.isfinite(logits).all():
        raise ValueError("Smoke forward produced non-finite output.")
    return {
        "waveform_samples": len(loaded_waveform),
        "embedding_shape": list(embedding.shape),
        "logit_shape": list(logits.shape),
        "finite": True,
        "encoder": (
            "pretrained XLS-R" if use_encoder else "not run; synthetic embedding"
        ),
        "head": "random initialization; no trained predictions",
    }


def main() -> None:
    """Parse CLI options and print an inference result as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--audio",
        type=Path,
        help="Clip no longer than one second; requires a trained checkpoint.",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Offline waveform-loader and random-head check; no encoder.",
    )
    mode.add_argument(
        "--smoke-encoder",
        action="store_true",
        help="Run the pretrained encoder and random head on a generated tone.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--download-encoder",
        action="store_true",
        help="Allow download of the pinned external model (~1.2 GB).",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.audio:
        if args.checkpoint is None:
            parser.error(
                "--audio requires --checkpoint; no trained task checkpoint is distributed."
            )
        predictor = SpeechPredictor.from_checkpoint(args.checkpoint)
        result = predictor.predict_file(args.audio, make_encoder(args.download_encoder))
    else:
        if args.checkpoint or (args.smoke and args.download_encoder):
            parser.error(
                "Checkpoint is only for --audio; encoder downloads only for "
                "--audio or --smoke-encoder."
            )
        result = smoke(args.smoke_encoder, args.download_encoder)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
