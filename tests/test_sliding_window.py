"""Check window boundaries and scores with generated audio and random weights."""

import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from songsored.cnn_inference import CNNPredictor, FEATURE_CONFIG
from songsored.model import WordClassifier
from songsored.sliding_window import iter_audio_windows, score_cnn_windows


def test_complete_windows_and_short_audio():
    audio = np.arange(25, dtype=np.float32)
    windows = list(iter_audio_windows(audio, window_samples=10, hop_samples=4))
    assert [(start, end) for start, end, _ in windows] == [(0, 10), (4, 14), (8, 18), (12, 22)]
    for start, end, window in windows:
        np.testing.assert_array_equal(window, audio[start:end])
    assert list(iter_audio_windows(audio[:9], 10, 4)) == []
    assert len(list(iter_audio_windows(audio[:10], 10, 4))) == 1


@pytest.mark.parametrize("size", [0, -1, True, 2.5])
def test_invalid_window_or_hop(size):
    audio = np.zeros(20, dtype=np.float32)
    with pytest.raises(ValueError):
        list(iter_audio_windows(audio, size, 5))
    with pytest.raises(ValueError):
        list(iter_audio_windows(audio, 10, size))


@pytest.mark.parametrize("pool_size", [(1, 1), (1, 4)])
@pytest.mark.parametrize("normalized", [False, True])
def test_scan_matches_single_clip_predictions(tmp_path, pool_size, normalized):
    torch.manual_seed(42)
    checkpoint = {
        "feature_config": dict(FEATURE_CONFIG),
        "model_config": {"n_mels": 128, "num_classes": 1, "pool_size": pool_size},
        "model_state_dict": WordClassifier(pool_size=pool_size).state_dict(),
        "normalization": (
            {"method": "global_log_mel", "mean": -30.0, "std": 12.0, "eps": 1e-6}
            if normalized else None
        ),
    }
    checkpoint_path = tmp_path / "random.pt"
    torch.save(checkpoint, checkpoint_path)
    predictor = CNNPredictor(checkpoint)

    sample_rate = 22050
    time = np.arange(round(1.25 * sample_rate)) / sample_rate
    tone = (0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    audio_path = tmp_path / "generated.wav"
    sf.write(audio_path, tone, sample_rate, subtype="FLOAT")
    scores = score_cnn_windows(audio_path, checkpoint_path)
    assert len(scores) == 3
    for index, row in enumerate(scores):
        start = index * 2205
        end = start + sample_rate
        assert row["start_s"] == start / sample_rate
        assert row["end_s"] == end / sample_rate
        clip_path = tmp_path / f"window_{index}.wav"
        sf.write(clip_path, tone[start:end], sample_rate, subtype="FLOAT")
        expected = predictor.predict_file(clip_path)["target_probability"]
        assert row["score"] == pytest.approx(expected, abs=1e-7)

    command = [sys.executable, "-m", "songsored.sliding_window",
               "--audio", str(audio_path), "--checkpoint", str(checkpoint_path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == scores
