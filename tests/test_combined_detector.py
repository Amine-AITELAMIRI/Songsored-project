"""Check branch wiring with generated audio and explicit detector stand-ins."""

import json

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from songsored.combined_detector import censor_song, detect_song, load_analysis_audio


def make_inputs(tmp_path):
    source, vocals = tmp_path / "mix.wav", tmp_path / "vocals.wav"
    t = np.arange(48000) / 48000
    mix = np.column_stack([.2 * np.sin(2 * np.pi * 220 * t), .1 * np.cos(2 * np.pi * 330 * t)])
    voice = .15 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
    sf.write(source, mix, 48000, subtype="FLOAT")
    sf.write(vocals, voice, 16000, subtype="FLOAT")
    return source, vocals


def test_ctc_loading_matches_private_polyphase_preprocessing(tmp_path):
    source, _ = make_inputs(tmp_path)
    original, rate = sf.read(source, dtype="float32", always_2d=True)
    wave, duration = load_analysis_audio(source, "polyphase")
    np.testing.assert_array_equal(wave, resample_poly(original.mean(1), 1, 3))
    assert duration == 1
    assert rate == 48000
    assert wave.shape == (16000,)
    assert wave.dtype == np.float32


def test_correct_audio_reaches_each_branch_then_the_original_is_muted(tmp_path):
    source, vocals = make_inputs(tmp_path)
    destination, event_path = tmp_path / "muted.wav", tmp_path / "events.json"
    inputs_before = [p.read_bytes() for p in (source, vocals)]
    expected_mix, _ = load_analysis_audio(source, "polyphase")
    expected_voice, _ = sf.read(vocals, dtype="float32")
    calls = []

    # I supply fixed detections to check the wiring, not model recognition.
    def ctc(wave, duration):
        calls.append("ctc")
        np.testing.assert_array_equal(wave, expected_mix)
        assert duration == 1
        return [{"start": .2, "end": .3, "score": .8}]

    def vocal(wave, duration):
        calls.append("vocals")
        np.testing.assert_array_equal(wave, expected_voice)
        assert duration == 1
        return [{"start": .25, "end": .4, "score": .9}, {"start": .7, "end": .8}]

    result = censor_song(source, vocals, destination, event_path, ctc_detector=ctc, vocal_detector=vocal)
    assert calls == ["ctc", "vocals"]
    assert result["branches"]["combined"] == [{"start": .2, "end": .4}, {"start": .7, "end": .8}]
    assert result["branches"]["ctc"][0]["score"] == .8
    original, rate = sf.read(source, always_2d=True)
    actual, output_rate = sf.read(destination, always_2d=True)
    expected = original.copy()
    expected[9600:19200] = 0
    expected[33600:38400] = 0
    np.testing.assert_array_equal(actual, expected)
    assert output_rate == rate == 48000
    assert [p.read_bytes() for p in (source, vocals)] == inputs_before
    assert json.loads(event_path.read_text()) == result["render"]


@pytest.mark.parametrize("active", ["ctc", "vocals", "neither"])
def test_either_branch_can_contribute_without_agreement(tmp_path, active):
    source, vocals = make_inputs(tmp_path)
    event = {"start": .1, "end": .2}
    result = detect_song(
        source, vocals,
        ctc_detector=lambda wave, duration: [event] if active == "ctc" else [],
        vocal_detector=lambda wave, duration: [event] if active == "vocals" else [],
    )
    assert result["combined"] == ([] if active == "neither" else [event])


def test_duration_mismatch_stops_before_inference(tmp_path):
    source, vocals = make_inputs(tmp_path)
    sf.write(vocals, np.zeros(8000), 16000)

    def unexpected(*args):
        pytest.fail("A detector ran despite the mismatched stem.")

    with pytest.raises(ValueError, match="same timeline"):
        detect_song(source, vocals, ctc_detector=unexpected, vocal_detector=unexpected)


def test_no_partial_output_when_a_branch_fails(tmp_path):
    source, vocals = make_inputs(tmp_path)
    destination, event_path = tmp_path / "muted.wav", tmp_path / "events.json"

    def failed(*args):
        raise RuntimeError("Detector failed")

    with pytest.raises(RuntimeError, match="Detector failed"):
        censor_song(source, vocals, destination, event_path, ctc_detector=lambda *args: [], vocal_detector=failed)
    assert not destination.exists()
    assert not event_path.exists()


def test_output_conflicts_stop_before_inference(tmp_path):
    source, vocals = make_inputs(tmp_path)
    destination, event_path = tmp_path / "muted.wav", tmp_path / "events.json"

    def unexpected(*args):
        pytest.fail("A detector ran despite an output conflict.")

    with pytest.raises(ValueError, match="distinct"):
        censor_song(source, vocals, vocals, event_path, ctc_detector=unexpected, vocal_detector=unexpected)
    destination.write_text("keep")
    with pytest.raises(FileExistsError):
        censor_song(source, vocals, destination, event_path, ctc_detector=unexpected, vocal_detector=unexpected)
    assert destination.read_text() == "keep"


def test_same_input_path_is_rejected(tmp_path):
    source, _ = make_inputs(tmp_path)
    with pytest.raises(ValueError, match="separate vocal stem"):
        detect_song(source, source, ctc_detector=None, vocal_detector=None)


@pytest.mark.parametrize("wave", [np.array([]), np.array([np.nan])])
def test_empty_or_nonfinite_audio_is_rejected(tmp_path, wave):
    path = tmp_path / "invalid.wav"
    sf.write(path, wave, 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="finite audio"):
        load_analysis_audio(path, "polyphase")
