"""Generated audio and controlled tensors only; no song data or downloads."""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from songsored.inference import DIM, MODEL_ID, REVISION, SpeechPredictor, smoke
from songsored.speech_classifier import SpeechFeatureClassifier
from songsored.speech_features import load_speech_waveform
from songsored.speech_encoder import FrozenSpeechEncoder


def checkpoint():
    model = SpeechFeatureClassifier(DIM)
    with torch.no_grad():
        model.classifier.weight.fill_(0.01)
        model.classifier.bias.fill_(-0.2)

    return {
        "embedding_dim": DIM,
        "model_state_dict": model.state_dict(),
        "stats": {
            "mean": torch.ones(DIM),
            "scale": torch.full((DIM,), 2.0),
        },
        "cache_metadata": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "sample_rate": 16000,
            "target_samples": 16000,
            "pooling": "last_hidden_state_mean",
            "embedding_dim": DIM,
        },
    }


def test_loader_resamples_mixes_and_pads(tmp_path):
    sample_rate = 32000
    sample_times = np.arange(sample_rate // 2) / sample_rate
    tone = np.sin(2 * np.pi * 440 * sample_times).astype(np.float32)
    stereo_audio = np.stack([tone, np.zeros_like(tone)], axis=1)
    audio_path = tmp_path / "generated.wav"
    sf.write(audio_path, stereo_audio, sample_rate, subtype="FLOAT")

    waveform = load_speech_waveform(audio_path)
    expected_tone = 0.5 * np.sin(
        2 * np.pi * 440 * np.arange(8000) / 16000
    )
    assert waveform.shape == (16000,)
    assert waveform.dtype == np.float32
    np.testing.assert_allclose(waveform[100:7900], expected_tone[100:7900], atol=1e-4)
    assert np.count_nonzero(waveform[8000:]) == 0


@pytest.mark.parametrize(
    "sample_count,value",
    [(16001, 0.0), (0, 0.0), (16000, float("nan"))],
)
def test_loader_rejects_invalid_audio(tmp_path, sample_count, value):
    audio_path = tmp_path / "invalid.wav"
    invalid_audio = np.full(sample_count, value, dtype=np.float32)
    sf.write(audio_path, invalid_audio, 16000, subtype="FLOAT")
    with pytest.raises(ValueError):
        load_speech_waveform(audio_path)


def test_checkpoint_roundtrip_preserves_normalized_prediction(tmp_path):
    saved_checkpoint = checkpoint()
    checkpoint_path = tmp_path / "controlled.pt"
    torch.save(saved_checkpoint, checkpoint_path)
    predictor = SpeechPredictor.from_checkpoint(checkpoint_path)
    embedding = torch.full((DIM,), 3.0)
    # Normalization maps each value to 1; 1024 weights of .01 give a logit of 10.04.
    expected_logit = 1024 * 0.01 - 0.2
    expected_probability = torch.sigmoid(torch.tensor(expected_logit)).item()
    assert predictor.probability(embedding) == pytest.approx(
        expected_probability,
        abs=1e-7,
    )
    # An embedding equal to the saved mean should normalize to zero.
    mean_embedding = torch.ones(DIM)
    expected_at_mean = torch.sigmoid(torch.tensor(-0.2)).item()
    assert predictor.probability(mean_embedding) == pytest.approx(expected_at_mean)


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "wrong"),
        ("pooling", "ordered"),
        ("sample_rate", 22050),
        ("model_id", "other"),
    ],
)
def test_rejects_incompatible_representation(field, value):
    saved_checkpoint = checkpoint()
    saved_checkpoint["cache_metadata"][field] = value
    with pytest.raises(ValueError):
        SpeechPredictor(saved_checkpoint)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_bad_statistics(value):
    saved_checkpoint = checkpoint()
    saved_checkpoint["stats"]["scale"][0] = value
    with pytest.raises(ValueError):
        SpeechPredictor(saved_checkpoint)


def test_rejects_nonfinite_weights_and_embeddings():
    saved_checkpoint = checkpoint()
    saved_checkpoint["model_state_dict"]["classifier.bias"][0] = float("nan")
    with pytest.raises(ValueError):
        SpeechPredictor(saved_checkpoint)
    predictor = SpeechPredictor(checkpoint())
    invalid_embeddings = (torch.zeros(4096), torch.full((DIM,), float("inf")))
    for embedding in invalid_embeddings:
        with pytest.raises(ValueError):
            predictor.probability(embedding)


def test_encoder_contract_with_explicit_test_double(monkeypatch):
    # This checks wrapper behavior, not acoustic quality. No toy encoder is shipped.
    from songsored import speech_encoder as module

    class Extractor:
        sampling_rate = 16000

        def __call__(self, waveform, **kwargs):
            assert waveform.dtype == np.float32
            return {"input_values": torch.from_numpy(waveform).unsqueeze(0)}

    class Encoder(torch.nn.Module):
        config = SimpleNamespace(hidden_size=DIM)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, **kwargs):
            assert not self.training and not torch.is_grad_enabled()
            frame_values = torch.arange(3.0).view(1, 3, 1)
            return SimpleNamespace(
                last_hidden_state=frame_values.expand(1, 3, DIM)
            )

    def load_extractor(*args, **kwargs):
        assert kwargs == {"revision": REVISION, "local_files_only": True}
        return Extractor()

    monkeypatch.setattr(module.AutoFeatureExtractor, "from_pretrained", load_extractor)
    monkeypatch.setattr(
        module.Wav2Vec2Model,
        "from_pretrained",
        lambda *args, **kwargs: Encoder(),
    )
    encoder = FrozenSpeechEncoder(revision=REVISION)
    assert not next(encoder.model.parameters()).requires_grad
    encoder.model.train()
    waveform = np.zeros(16000, dtype=np.float16)
    assert torch.equal(encoder.encode(waveform), torch.ones(DIM))
    with pytest.raises(ValueError):
        encoder.encode(np.zeros(8000, dtype="float32"))


def test_smoke_does_not_report_trained_predictions():
    result = smoke()
    assert result["finite"]
    assert result["logit_shape"] == [1, 1]
    assert "random" in result["head"]
    assert "not run" in result["encoder"]
    assert "target_probability" not in result
