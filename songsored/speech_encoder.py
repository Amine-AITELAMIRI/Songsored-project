"""Load a frozen pretrained speech encoder and create audio embeddings."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model


class WaveformValidationError(ValueError, TypeError):
    """Raised when a waveform has an unsupported type, shape, or value."""


class FrozenSpeechEncoder:
    """Create fixed-size embeddings with a frozen pretrained speech model.

    The feature extractor prepares the waveform for the model. The encoder's
    frame-level outputs are averaged over time to produce one embedding.
    """

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-xls-r-300m",
        device: str | torch.device = "cpu",
        revision: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        self.revision = revision

        # Load the feature extractor and its matching model.
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.model = Wav2Vec2Model.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )

        sample_rate = getattr(self.feature_extractor, "sampling_rate", None)
        if sample_rate != 16000:
            raise ValueError(
                f"Feature extractor expects sampling rate {sample_rate} Hz, but 16000 Hz is required."
            )

        self.embedding_dim: int = int(self.model.config.hidden_size)

        # Keep the model fixed and place it on the requested device.
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.model.eval()

    def _validate_waveform(self, waveform: np.ndarray) -> None:
        """Require a finite, mono, one-second waveform sampled at 16 kHz."""
        if not isinstance(waveform, np.ndarray):
            raise WaveformValidationError(
                f"Waveform must be a numpy.ndarray, got {type(waveform).__name__}."
            )

        if waveform.ndim != 1:
            raise WaveformValidationError(
                f"Waveform must be 1D (mono), got shape {waveform.shape} with {waveform.ndim} dimensions."
            )

        if not np.issubdtype(waveform.dtype, np.floating):
            raise WaveformValidationError(
                f"Waveform samples must be floating-point (e.g. float32), got dtype {waveform.dtype}."
            )

        if waveform.shape[0] != 16000:
            raise WaveformValidationError(
                f"Waveform must have exactly 16000 samples (1 second at 16 kHz), got {waveform.shape[0]} samples."
            )

        if not np.all(np.isfinite(waveform)):
            raise WaveformValidationError(
                "Waveform contains non-finite values (NaN or Inf)."
            )

    def encode(self, waveform: np.ndarray) -> torch.Tensor:
        """Turn a one-second, 16 kHz mono waveform into one embedding.

        Args:
            waveform: 1D NumPy array with 16,000 floating-point samples.

        Returns:
            Detached CPU float32 tensor of shape [embedding_dim].
        """
        # Validate the input before passing it to the feature extractor.
        self._validate_waveform(waveform)
        waveform = waveform.astype(np.float32, copy=False)

        # The feature extractor returns tensors ready for the encoder.
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            # Average frame embeddings to get one vector for the clip.
            embedding = last_hidden_state.mean(dim=1).squeeze(0)
            return embedding.detach().cpu().to(torch.float32)
