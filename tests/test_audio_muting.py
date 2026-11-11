"""Check combining and muting with generated audio, without trained models."""

import json

import numpy as np
import pytest
import soundfile as sf

from songsored import audio_muting
from songsored.audio_muting import combine_detections, mute_intervals, render_muted_audio


def test_union_keeps_either_detector_and_preserves_gaps():
    first = [{"start": 2, "end": 3, "score": .8}, {"start": 0, "end": 1}]
    second = [{"start": .5, "end": 1.5}, {"start": 1.5, "end": 1.8}, {"start": 2, "end": 3}]
    assert combine_detections(first, second) == [
        {"start": 0, "end": 1.8}, {"start": 2, "end": 3},
    ]
    assert first[0] == {"start": 2, "end": 3, "score": .8}
    assert len(second) == 3
    assert combine_detections() == []


@pytest.mark.parametrize("channels", [1, 2])
def test_muting_clips_edges_and_only_changes_selected_samples(channels):
    audio = np.arange(1, 11, dtype=float)
    if channels == 2:
        audio = np.column_stack([audio, -audio])
    original = audio.copy()
    events = [
        {"start": .41, "end": .49}, {"start": .49, "end": .61},
        {"start": -.2, "end": .1}, {"start": .9, "end": 2},
        {"start": 2, "end": 3}, {"start": -2, "end": -1},
    ]
    muted, spans = mute_intervals(audio, 10, events)
    assert spans == [[0, 1], [4, 7], [9, 10]]
    np.testing.assert_array_equal(muted[[0, 4, 5, 6, 9]], 0)
    np.testing.assert_array_equal(muted[[1, 2, 3, 7, 8]], original[[1, 2, 3, 7, 8]])
    np.testing.assert_array_equal(audio, original)


def test_sample_rounding_can_join_separate_time_intervals():
    events = [{"start": .11, "end": .19}, {"start": .21, "end": .29}]
    _, spans = mute_intervals(np.ones(10), 10, events)
    assert spans == [[1, 3]]


def test_empty_detections_and_silence():
    audio = np.zeros((20, 2))
    muted, spans = mute_intervals(audio, 16000, [])
    assert spans == []
    assert not np.shares_memory(audio, muted)
    np.testing.assert_array_equal(muted, audio)


@pytest.mark.parametrize("rate, channels, subtype", [
    (16000, 1, "PCM_16"), (44100, 2, "PCM_24"), (48000, 2, "DOUBLE"),
])
def test_wav_and_event_list_round_trip(tmp_path, rate, channels, subtype):
    source = tmp_path / "original.wav"
    destination = tmp_path / "muted.wav"
    event_path = tmp_path / "events.json"
    # I generate a tone here so the test doesn't need any song files.
    tone = .25 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
    audio = tone if channels == 1 else np.column_stack([tone, -tone])
    sf.write(source, audio, rate, subtype=subtype)
    source_bytes = source.read_bytes()
    first = [{"start": .2, "end": .3}]
    second = [{"start": .25, "end": .4}, {"start": .8, "end": .9}]

    record = render_muted_audio(source, destination, event_path, first, second)
    original, _ = sf.read(source, always_2d=True)
    actual, actual_rate = sf.read(destination, always_2d=True)
    expected = original.copy()
    expected[rate // 5:2 * rate // 5] = 0
    expected[4 * rate // 5:9 * rate // 10] = 0
    np.testing.assert_array_equal(actual, expected)
    assert actual_rate == rate
    assert actual.shape == (rate, channels)
    assert sf.info(destination).subtype == subtype
    assert source.read_bytes() == source_bytes
    assert json.loads(event_path.read_text()) == record
    assert record["mute_sample_intervals"] == [[rate // 5, 2 * rate // 5], [4 * rate // 5, 9 * rate // 10]]


def test_renderer_refuses_overwrites_and_same_paths(tmp_path):
    source = tmp_path / "original.wav"
    destination = tmp_path / "muted.wav"
    event_path = tmp_path / "events.json"
    sf.write(source, np.ones(20) * .1, 16000)
    with pytest.raises(ValueError):
        render_muted_audio(source, source, event_path)
    with pytest.raises(ValueError):
        render_muted_audio(source, destination, destination)
    with pytest.raises(ValueError):
        render_muted_audio(source, tmp_path / "out.mp3", event_path)
    for existing in (destination, event_path):
        existing.write_bytes(b"keep")
        with pytest.raises(FileExistsError):
            render_muted_audio(source, destination, event_path)
        assert existing.read_bytes() == b"keep"
        existing.unlink()


def test_event_file_collision_rolls_back_new_audio(tmp_path, monkeypatch):
    source = tmp_path / "original.wav"
    destination = tmp_path / "muted.wav"
    event_path = tmp_path / "events.json"
    sf.write(source, np.zeros(20), 16000)
    link = audio_muting.os.link

    def collide(src, dst):
        if dst == event_path:
            event_path.write_text("another writer")
        return link(src, dst)

    monkeypatch.setattr(audio_muting.os, "link", collide)
    with pytest.raises(FileExistsError):
        render_muted_audio(source, destination, event_path)
    assert not destination.exists()
    assert event_path.read_text() == "another writer"


@pytest.mark.parametrize("start, end", [(1, 1), (2, 1), (float("nan"), 1), (0, float("inf"))])
def test_bad_intervals_are_rejected(start, end):
    with pytest.raises(ValueError):
        combine_detections([{"start": start, "end": end}])


@pytest.mark.parametrize("audio, rate", [
    (np.array([]), 16000), (np.array([np.nan]), 16000),
    (np.ones((2, 2, 2)), 16000), (np.ones(2), 0), (np.ones(2), 1.5),
])
def test_bad_audio_or_sample_rate_is_rejected(audio, rate):
    with pytest.raises(ValueError):
        mute_intervals(audio, rate, [])
