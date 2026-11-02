"""Check the vocal runner with generated features, audio and local test weights."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit, logit
import torch

from songsored.vocal_detector import (
    ENCODER_ID, ENCODER_REVISION, KERNELS, STRIDES, VocalDetector,
    correct_scores, four_second_window, persistent_events,
)


class TestBackbone(torch.nn.Module):
    __test__ = False

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(conv_kernel=KERNELS, conv_stride=STRIDES, hidden_size=1024)
        self.calls = 0

    def forward(self, input_values):
        self.calls += 1
        assert input_values.shape[1] == 64000
        features = torch.zeros(len(input_values), 199, 1024)
        features[:, :, 0] = 3
        return SimpleNamespace(last_hidden_state=features)


class TestHead(torch.nn.Module):
    __test__ = False
    embedding_dim = 1024

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, features, mask):
        assert mask.any(1).all()
        assert not mask[:, :50].any() and not mask[:, 150:].any()
        return features[:, :, 0] * self.scale


def make_encoder():
    def prepare(waves, **kwargs):
        assert kwargs == dict(sampling_rate=16000, return_tensors="pt", padding=False)
        return {"input_values": torch.from_numpy(np.stack(waves))}

    return SimpleNamespace(
        model_id=ENCODER_ID, revision=ENCODER_REVISION, device=torch.device("cpu"),
        model=TestBackbone(), feature_extractor=prepare,
    )


def make_detector(**kwargs):
    stats = {"mean": torch.ones(1024), "scale": torch.full((1024,), 2.)}
    return VocalDetector(make_encoder(), TestHead(), stats, np.zeros(1024), threshold=.7, **kwargs)


def test_window_padding_and_real_frame_support():
    wave, valid = four_second_window(np.ones(32000, dtype=np.float32), -1)
    np.testing.assert_array_equal(wave[:16000], 0)
    np.testing.assert_array_equal(wave[16000:48000], 1)
    np.testing.assert_array_equal(wave[48000:], 0)
    assert np.flatnonzero(valid).tolist() == list(range(50, 149))


def test_normalization_center_scoring_and_adjacent_windows():
    detector = make_detector()
    audio = np.full(5 * 16000, .1, dtype=np.float32)
    cells = detector.score_cells(audio, 5)
    # The final 20 ms cell is excluded: its 25 ms receptive field hits padding.
    assert cells.shape == (249, 4)
    np.testing.assert_allclose(cells[:, 0], np.arange(249) * .02, atol=1e-12)
    np.testing.assert_allclose(cells[:, 2], expit(1), atol=1e-7)
    events = detector(audio, 5)
    assert len(events) == 1
    assert events[0]["start"] == 0
    assert events[0]["end"] == pytest.approx(4.98)
    assert events[0]["availability"] == 3
    assert not detector.head.training
    assert not detector.head.scale.requires_grad


def test_correction_gate_and_float16_replay():
    rng = np.random.default_rng(42)
    p = np.array([[.49, .5, .9, 1.]], dtype=np.float32)
    original = p.copy()
    x = rng.normal(size=(1, 4, 1024)).astype(np.float32)
    w = rng.normal(size=1024).astype(np.float32) * .01
    valid = np.array([[True, True, False, True]])
    actual = correct_scores(p, x, valid, w)
    selected = (p >= .5) & valid
    expected = p.copy()
    features = x[selected].astype(np.float16).astype(np.float32)
    delta = np.einsum("ij,j->i", features, w, optimize=False)
    expected[selected] = expit(logit(np.clip(p[selected], 1e-7, 1 - 1e-7)) + 4 * np.tanh(delta))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(p, original)
    assert actual[0, 0] == p[0, 0] and actual[0, 2] == p[0, 2]


def test_runner_applies_correction_before_final_cutoff():
    detector = make_detector()
    audio = np.ones(16000, dtype=np.float32)
    assert detector(audio, 1)
    detector.correction[0] = -1
    assert detector(audio, 1) == []


def test_silence_guard_does_not_reject_quiet_audio():
    detector = make_detector()
    assert detector(np.zeros(16000, dtype=np.float32), 1) == []
    assert detector(np.full(16000, 1e-8, dtype=np.float32), 1)


def test_too_short_audio_has_no_valid_scored_frames():
    detector = make_detector()
    assert detector(np.ones(100, dtype=np.float32), 100 / 16000) == []
    assert detector.encoder.model.calls == 0


def test_persistence_crosses_adjacent_windows_but_not_gaps():
    cells = np.array([[0, .02, .9, 3], [.02, .04, .9, 3], [.04, .06, .1, 3],
                      [.06, .08, .9, 3], [.1, .12, .9, 3], [1.98, 2, .9, 3], [2, 2.02, .9, 5]])
    events = persistent_events(cells, .7, 2)
    assert [(e["start"], e["end"]) for e in events] == [(0, .04), (1.98, 2.02)]
    assert events[-1]["availability"] == 5


def test_batch_size_does_not_change_generated_scores():
    audio = np.ones(16000 * 7, dtype=np.float32)
    np.testing.assert_array_equal(make_detector(batch_size=1).score_cells(audio, 7),
                                  make_detector(batch_size=4).score_cells(audio, 7))


def test_local_artifact_loading_without_model_download(tmp_path, monkeypatch):
    from songsored import speech_encoder
    from songsored.speech_residual_detector import ResidualSpeechDetector

    head_path, correction_path = tmp_path / "head.pt", tmp_path / "correction.npz"
    head = ResidualSpeechDetector()
    torch.save({"head_state": head.state_dict(), "stats": {"mean": torch.zeros(1024), "scale": torch.ones(1024)}}, head_path)
    np.savez(correction_path, weight=np.zeros(1024, dtype=np.float32))

    def local_encoder(model_id, **kwargs):
        assert model_id == ENCODER_ID
        assert kwargs == dict(device="cpu", revision=ENCODER_REVISION, local_files_only=True)
        return make_encoder()

    monkeypatch.setattr(speech_encoder, "FrozenSpeechEncoder", local_encoder)
    detector = VocalDetector.from_files(head_path, correction_path, threshold=.7)
    for name, tensor in head.state_dict().items():
        torch.testing.assert_close(detector.head.state_dict()[name], tensor)
    assert isinstance(detector(np.ones(16000, dtype=np.float32), 1), list)


@pytest.mark.parametrize("problem", ["threshold", "persistence", "batch_size", "revision", "scale", "weight"])
def test_bad_configuration_is_rejected(problem):
    encoder = make_encoder()
    stats = {"mean": torch.zeros(1024), "scale": torch.ones(1024)}
    w = np.zeros(1024)
    kwargs = {"threshold": .7}
    if problem == "threshold": kwargs["threshold"] = 0
    elif problem == "persistence": kwargs["persistence"] = 0
    elif problem == "batch_size": kwargs["batch_size"] = True
    elif problem == "revision": encoder.revision = "different"
    elif problem == "scale": stats["scale"][0] = 0
    elif problem == "weight": w[0] = np.nan
    with pytest.raises(ValueError):
        VocalDetector(encoder, TestHead(), stats, w, **kwargs)


@pytest.mark.parametrize("wave, duration", [(np.ones(16000), 2), (np.array([]), 0), (np.array([np.nan]), 1 / 16000)])
def test_bad_audio_is_rejected(wave, duration):
    with pytest.raises(ValueError):
        make_detector()(wave, duration)
