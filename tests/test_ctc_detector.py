"""Check CTC inference wiring with generated logits, audio and test weights."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from songsored import ctc_detector
from songsored.ctc_detector import CTCDetector, read_tokens


# I use "example" and "examples" here because these tests only check the CTC logic.
ALPHABET = ["<blank>", " ", *"examplsq"]
TOKENS = {char: index for index, char in enumerate([*ALPHABET, "r", "t"])}


class TestSession:
    __test__ = False

    def __init__(self, words=((10, "examples"),)):
        self.logits = np.full((1, 149, len(TOKENS)), -8, dtype=np.float32)
        self.logits[:, :, 0] = 8
        for start, word in words:
            for frame, char in enumerate(word, start):
                self.logits[0, frame, 0] = -8
                self.logits[0, frame, TOKENS[char]] = 8
            boundary = start + len(word)
            self.logits[0, boundary, 0] = -8
            self.logits[0, boundary, TOKENS[" "]] = 8
        self.inputs = []

    def get_inputs(self):
        return [SimpleNamespace(name="waveform")]

    def run(self, output_names, inputs):
        assert output_names is None
        self.inputs.append(inputs["waveform"].copy())
        return [self.logits.copy()]


def checkpoint():
    layer = torch.nn.Linear(len(ALPHABET) + 1, len(ALPHABET) + 1)
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return {"model": "omni", "alphabet": ALPHABET.copy(), "layer": layer.state_dict()}


def detector(session=None, **kwargs):
    return CTCDetector(session or TestSession(), TOKENS, checkpoint(), ["example", "examples"], threshold=.5, **kwargs)


def test_token_reader_keeps_space_and_blank_alias(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("<pad> 0\n  1\ne 2\n", encoding="utf-8")
    assert read_tokens(path) == {"<pad>": 0, " ": 1, "e": 2, "<blank>": 0}


@pytest.mark.parametrize("text", ["", "e 0\ne 1\n", "e 0\nx 0\n", "e 2\n", "<blank> 1\ne 0\n", "invalid\n"])
def test_bad_token_file_is_rejected(tmp_path, text):
    path = tmp_path / "tokens.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_tokens(path)


def test_normalization_and_background_probability_are_preserved():
    session = TestSession()
    rng = np.random.default_rng(42)
    session.logits = rng.normal(size=session.logits.shape).astype(np.float32)
    model = detector(session)
    wave = rng.normal(size=48000).astype(np.float32)
    probabilities = model.frame_probabilities(wave)
    np.testing.assert_array_equal(session.inputs[0][0], (wave - wave.mean()) / np.sqrt(wave.var() + 1e-5))
    full = torch.from_numpy(session.logits[0]).double().softmax(-1).numpy()
    kept_ids = [TOKENS[c] for c in ["<blank>", " ", *"exampls"]]
    np.testing.assert_allclose(probabilities[:, :-1], full[:, kept_ids], atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(probabilities[:, -1], full[:, [TOKENS[c] for c in "qrt"]].sum(1), atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(probabilities.sum(1), 1)


def test_nonzero_saved_adapter_changes_scores_and_is_frozen():
    session = TestSession()
    saved = checkpoint()
    saved["layer"]["bias"][2] = -2
    model = CTCDetector(session, TOKENS, saved, ["example"], threshold=.5)
    p = model.frame_probabilities(np.ones(48000, dtype=np.float32))
    control = detector(session).frame_probabilities(np.ones(48000, dtype=np.float32))
    assert p[10, 2] < control[10, 2]
    assert not model.adapter.training
    assert not any(parameter.requires_grad for parameter in model.adapter.parameters())


def test_three_second_grid_and_clipped_tail():
    session = TestSession()
    model = detector(session)
    wave = np.full(100320, .1, dtype=np.float32)  # 6.27 seconds
    events = model(wave, 6.27)
    np.testing.assert_allclose([(e["start"], e["end"]) for e in events],
                               [(.2, .36), (3.2, 3.36), (6.2, 6.27)])
    assert all(e["score"] >= .5 for e in events)
    assert len(session.inputs) == 3
    assert all(x.shape == (1, 48000) for x in session.inputs)
    tail = np.pad(wave[96000:], (0, 48000 - len(wave[96000:])))
    np.testing.assert_array_equal(session.inputs[-1][0], (tail - tail.mean()) / np.sqrt(tail.var() + 1e-5))


def test_padding_only_detection_is_discarded():
    events = detector()(np.ones(97600, dtype=np.float32), 6.1)
    assert len(events) == 2


def test_only_first_target_on_the_selected_path_is_returned():
    model = detector(TestSession(words=((10, "example"), (40, "examples"))))
    events = model(np.ones(48000, dtype=np.float32), 3)
    assert len(events) == 1
    assert events[0]["start"] == pytest.approx(.2)
    assert events[0]["end"] == pytest.approx(.34)


def test_exact_silence_is_skipped_but_quiet_audio_runs():
    session = TestSession()
    model = detector(session)
    assert model(np.zeros(48000, dtype=np.float32), 3) == []
    assert session.inputs == []
    assert model(np.full(48000, 1e-8, dtype=np.float32), 3)
    assert len(session.inputs) == 1


def test_low_keyword_score_does_not_call_the_locator(monkeypatch):
    def unexpected(*args):
        pytest.fail("Low-scoring audio should not reach the locator.")

    monkeypatch.setattr(ctc_detector, "locate_keyword_frames", unexpected)
    model = detector(TestSession(words=()))
    assert model(np.ones(48000, dtype=np.float32), 3) == []


def test_local_files_use_cpu_session_and_saved_adapter(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    tokens, adapter, acoustic = tmp_path / "tokens.txt", tmp_path / "adapter.pt", tmp_path / "model.onnx"
    tokens.write_text("".join(f"{char} {index}\n" for char, index in TOKENS.items()))
    torch.save(checkpoint(), adapter)
    session = TestSession()

    # I replace only the session loader here; this isn't a trained-model demo.
    def load(path, options, providers):
        assert path == str(acoustic)
        assert providers == ["CPUExecutionProvider"]
        assert options.intra_op_num_threads == 4 and options.inter_op_num_threads == 1
        return session

    monkeypatch.setattr(ort, "InferenceSession", load)
    model = CTCDetector.from_files(acoustic, tokens, adapter, ["example", "examples"], threshold=.5)
    assert model(np.ones(48000, dtype=np.float32), 3)


@pytest.mark.parametrize("problem", ["threshold", "model", "alphabet", "weights", "target", "ids"])
def test_bad_configuration_is_rejected(problem):
    saved, tokens, targets, threshold = checkpoint(), TOKENS.copy(), ["example"], .5
    if problem == "threshold": threshold = 0
    elif problem == "model": saved["model"] = "different"
    elif problem == "alphabet": saved["alphabet"] = ["<blank>", " ", "?"]
    elif problem == "weights": saved["layer"]["bias"][0] = float("nan")
    elif problem == "target": targets = ["missing"]
    elif problem == "ids": tokens["e"] = 99
    with pytest.raises((ValueError, RuntimeError)):
        CTCDetector(TestSession(), tokens, saved, targets, threshold=threshold)


@pytest.mark.parametrize("problem", ["shape", "nan", "dtype"])
def test_invalid_acoustic_output_is_rejected(problem):
    session = TestSession()
    if problem == "shape": session.logits = np.zeros((1, 149, 2), dtype=np.float32)
    elif problem == "nan": session.logits[0, 0, 0] = np.nan
    else: session.logits = session.logits.astype(np.float64)
    with pytest.raises(ValueError, match="acoustic-model"):
        detector(session).frame_probabilities(np.ones(48000, dtype=np.float32))


@pytest.mark.parametrize("wave, duration", [(np.array([]), 0), (np.ones(16000), 2), (np.array([np.nan]), 1 / 16000)])
def test_invalid_audio_is_rejected(wave, duration):
    with pytest.raises(ValueError):
        detector()(wave, duration)
