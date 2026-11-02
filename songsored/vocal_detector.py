"""Run the four-second vocal detector with its learned frame-score correction."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import expit, logit
import torch

from .speech_residual_detector import ResidualSpeechDetector


ENCODER_ID = "facebook/wav2vec2-xls-r-300m"
ENCODER_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
KERNELS = [10, 3, 3, 3, 3, 2, 2]
STRIDES = [5, 2, 2, 2, 2, 2, 2]


def four_second_window(audio, start):
    """Zero-pad the edges and mark frames whose 400-sample support is real."""
    left = round(start * 16000)
    lo, hi = max(left, 0), min(left + 64000, len(audio))
    if hi <= lo:
        raise ValueError("Window does not intersect the recording.")
    wave = np.zeros(64000, dtype=np.float32)
    wave[lo - left:hi - left] = audio[lo:hi]
    frame_starts = np.arange(199) * 320
    valid = (frame_starts >= lo - left) & (frame_starts + 400 <= hi - left)
    return wave, valid


def correct_scores(probabilities, features, valid, weight):
    """Apply the saved correction only to valid frames scoring at least 0.5."""
    scores = probabilities.copy()
    selected = (scores >= .5) & valid
    # The private replay stores these normalized features in float16 first.
    x = features[selected].astype(np.float16).astype(np.float32)
    delta = np.einsum("ij,j->i", x, weight, optimize=False)
    if not np.isfinite(delta).all():
        raise ValueError("Score correction is nonfinite.")
    p = np.clip(scores[selected], 1e-7, 1 - 1e-7)
    scores[selected] = expit(logit(p) + 4 * np.tanh(delta))
    return scores


def persistent_events(cells, threshold, persistence):
    """Keep runs of consecutive above-threshold cells; gaps break a run.

    Columns are start, end, score and context availability, all times in seconds.
    Availability is when the first required cells' context ends, not runtime.
    """
    if type(persistence) is not int or persistence < 1:
        raise ValueError("Persistence must be a positive integer.")
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Supply a calibrated threshold in (0, 1].")
    if not len(cells):
        return []
    above = cells[:, 2] >= threshold
    adjacent = np.r_[False, cells[1:, 0] <= cells[:-1, 1] + 1e-6]
    starts = np.flatnonzero(above & ~(np.r_[False, above[:-1]] & adjacent))
    ends = np.flatnonzero(above & ~(np.r_[above[1:], False] & np.r_[adjacent[1:], False])) + 1
    return [
        {
            "start": float(cells[a, 0]),
            "end": float(cells[b - 1, 1]),
            "score": float(cells[a:b, 2].max()),
            "availability": float(cells[a:a + persistence, 3].max()),
        }
        for a, b in zip(starts, ends) if b - a >= persistence
    ]


class VocalDetector:
    """Scan mono 16 kHz vocals with a frozen encoder, head and correction.

    Supply matching trained artifacts and a cutoff selected on calibration.
    This is offline contextual inference, not a causal or real-time model.
    """

    def __init__(self, encoder, head, stats, correction, *, threshold, persistence=2, batch_size=4):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("Supply a calibrated threshold in (0, 1].")
        if any(type(value) is not int or value < 1 for value in (persistence, batch_size)):
            raise ValueError("Persistence and batch size must be positive integers.")
        config = encoder.model.config
        if (
            encoder.model_id != ENCODER_ID
            or encoder.revision != ENCODER_REVISION
            or list(config.conv_kernel) != KERNELS
            or list(config.conv_stride) != STRIDES
            or config.hidden_size != 1024
        ):
            raise ValueError("Expected the pinned XLS-R encoder and its frame geometry.")
        self.device = torch.device(encoder.device)
        self.stats = {}
        for name in ("mean", "scale"):
            value = stats[name]
            if not isinstance(value, torch.Tensor) or value.shape != (1024,) or not torch.isfinite(value).all():
                raise ValueError("Expected finite 1024-dimensional normalization statistics.")
            self.stats[name] = value.detach().to(self.device, dtype=torch.float32).clone()
        if not (self.stats["scale"] > 0).all():
            raise ValueError("Normalization scale must be positive.")
        self.correction = np.array(correction, dtype=np.float32, copy=True)
        if self.correction.shape != (1024,) or not np.isfinite(self.correction).all():
            raise ValueError("Expected 1024 finite correction weights.")
        if head.embedding_dim != 1024 or not all(torch.isfinite(p).all() for p in head.parameters()):
            raise ValueError("Expected a finite 1024-input temporal head.")
        self.encoder = encoder
        self.encoder.model.eval().requires_grad_(False)
        self.head = head.to(self.device).eval().requires_grad_(False)
        self.threshold = threshold
        self.persistence = persistence
        self.batch_size = batch_size

    @classmethod
    def from_files(cls, head_path, correction_path, *, threshold, device="cpu", persistence=2, batch_size=4):
        """Load trusted local artifacts. No weights are bundled or downloaded.

        The head checkpoint contains head_state and stats; the correction NPZ
        contains weight. The caller must pair them with the right calibration
        setting. Their file formats alone do not establish training provenance.
        """
        from .speech_encoder import FrozenSpeechEncoder

        checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
        head = ResidualSpeechDetector()
        head.load_state_dict(checkpoint["head_state"], strict=True)
        with np.load(correction_path, allow_pickle=False) as saved:
            correction = saved["weight"].copy()
        encoder = FrozenSpeechEncoder(
            ENCODER_ID, device=device, revision=ENCODER_REVISION, local_files_only=True,
        )
        return cls(encoder, head, checkpoint["stats"], correction, threshold=threshold,
                   persistence=persistence, batch_size=batch_size)

    @torch.inference_mode()
    def score_cells(self, audio, duration):
        """Return valid 20 ms cells after normalization, correction and silence rejection."""
        if (
            not isinstance(audio, np.ndarray) or audio.ndim != 1 or not audio.size
            or not np.issubdtype(audio.dtype, np.floating) or not np.isfinite(audio).all()
        ):
            raise ValueError("Expected nonempty, finite mono floating-point audio at 16 kHz.")
        if not math.isfinite(duration) or duration <= 0 or abs(len(audio) / 16000 - duration) > 1 / 16000 + 1e-9:
            raise ValueError("Duration must match the 16 kHz waveform.")

        starts = np.arange(-1, duration - 1, 2)
        cells = []
        for offset in range(0, len(starts), self.batch_size):
            times = starts[offset:offset + self.batch_size]
            windows, valid = zip(*(four_second_window(audio, float(t)) for t in times))
            valid = np.stack(valid)
            scored = np.zeros_like(valid)
            scored[:, 50:150] = valid[:, 50:150]
            keep = np.flatnonzero(scored.any(1))
            probabilities = np.zeros((len(windows), 100), dtype=np.float32)

            if len(keep):
                inputs = self.encoder.feature_extractor(
                    [windows[i] for i in keep], sampling_rate=16000,
                    return_tensors="pt", padding=False,
                )
                self.encoder.model.eval()
                features = self.encoder.model(
                    **{key: value.to(self.device) for key, value in inputs.items()}
                ).last_hidden_state.cpu().float()
                if features.shape != (len(keep), 199, 1024) or not torch.isfinite(features).all():
                    raise ValueError("Unexpected encoder features.")
                x = (features.to(self.device) - self.stats["mean"]) / self.stats["scale"]
                if not torch.isfinite(x).all():
                    raise ValueError("Normalized features are nonfinite.")
                self.head.eval()
                logits = self.head(x, torch.from_numpy(scored[keep]).to(self.device))
                if logits.shape != (len(keep), 199) or not torch.isfinite(logits).all():
                    raise ValueError("Unexpected temporal-head logits.")
                p = logits.sigmoid()[:, 50:150].cpu().numpy()
                probabilities[keep] = correct_scores(
                    p, x[:, 50:150].cpu().numpy(), valid[keep, 50:150], self.correction,
                )

            for i, (start, window) in enumerate(zip(times, windows)):
                # Exact silence is rejected; quiet audio is not automatically silence.
                if not np.any(window[16000:48000]):
                    probabilities[i] = 0
                left = start + 1 + np.arange(100) * .02
                right = np.minimum(left + .02, duration)
                mask = valid[i, 50:150] & (right > left)
                available = np.full(100, min(duration, start + 4))
                cells.append(np.column_stack([left, right, probabilities[i], available])[mask])
        return np.concatenate(cells) if cells else np.empty((0, 4))

    def __call__(self, audio, duration):
        cells = self.score_cells(audio, duration)
        return persistent_events(cells, self.threshold, self.persistence)
