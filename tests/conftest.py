"""Offline stand-ins for external weights; never used by the shipped package."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def stub_speech_backbone(monkeypatch):
    from songsored import speech_encoder

    class FeatureExtractor:
        sampling_rate = 16000

        def to_dict(self):
            return {"sampling_rate": 16000, "test_double": True}

        def __call__(self, waveform, **kwargs):
            assert waveform.dtype == np.float32
            return {"input_values": torch.from_numpy(waveform)[None]}

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=1024)
            self.scale = torch.nn.Parameter(torch.ones(1024))

        def forward(self, input_values):
            frames = input_values.reshape(1, 4, 4000).mean(-1, keepdim=True)
            return SimpleNamespace(last_hidden_state=frames * self.scale[None, None])

    monkeypatch.setattr(speech_encoder.AutoFeatureExtractor, "from_pretrained",
                        lambda *args, **kwargs: FeatureExtractor())
    monkeypatch.setattr(speech_encoder.Wav2Vec2Model, "from_pretrained",
                        lambda *args, **kwargs: Backbone())
