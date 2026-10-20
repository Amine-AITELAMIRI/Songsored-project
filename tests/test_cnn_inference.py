"""Test inference mechanics with generated audio and random model weights."""

import csv

import numpy as np
import pytest
import soundfile as sf
import torch

from songsored.cnn_inference import CNNPredictor, FEATURE_CONFIG, load_cnn_features
from songsored.dataset import AudioClipDataset
from songsored.model import WordClassifier
from songsored.normalization import normalize_features


def make_checkpoint(pool_size=(1, 1), normalized=True):
    torch.manual_seed(42)
    return {
        "feature_config": dict(FEATURE_CONFIG),
        "model_config": {
            "n_mels": 128,
            "num_classes": 1,
            "pool_size": pool_size,
        },
        "model_state_dict": WordClassifier(pool_size=pool_size).state_dict(),
        "normalization": (
            {
                "method": "global_log_mel",
                "mean": -30.0,
                "std": 12.0,
                "eps": 1e-6,
            }
            if normalized
            else None
        ),
    }


@pytest.mark.parametrize("pool_size", [(1, 1), (1, 4)])
@pytest.mark.parametrize("normalized", [False, True])
def test_inference_matches_dataset_and_manual_forward(
    tmp_path,
    pool_size,
    normalized,
):
    sample_rate = 16000
    sample_times = np.arange(8000) / sample_rate
    waveform = (
        0.1 * np.sin(2 * np.pi * 440 * sample_times)
    ).astype(np.float32)
    audio = tmp_path / "tone.wav"
    stereo_waveform = np.stack([waveform, waveform], axis=1)
    sf.write(audio, stereo_waveform, sample_rate, subtype="FLOAT")

    manifest = tmp_path / "generated.csv"
    with manifest.open("w", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["path", "label", "recording_id", "split"])
        writer.writerow(["tone.wav", 0, "generated", "train"])

    features, _ = AudioClipDataset(manifest, "train")[0]
    torch.testing.assert_close(load_cnn_features(audio), features, rtol=0, atol=0)

    saved_checkpoint = make_checkpoint(pool_size, normalized)
    checkpoint_path = tmp_path / "random.pt"
    torch.save(saved_checkpoint, checkpoint_path)
    predictor = CNNPredictor.from_checkpoint(checkpoint_path)

    comparison_model = WordClassifier(**saved_checkpoint["model_config"])
    comparison_model.load_state_dict(saved_checkpoint["model_state_dict"])
    comparison_model.eval()
    if normalized:
        features = normalize_features(features, saved_checkpoint["normalization"])

    with torch.no_grad():
        expected_probability = (
            comparison_model(features.unsqueeze(0)).sigmoid().item()
        )

    prediction = predictor.predict_file(audio)
    assert prediction["target_probability"] == pytest.approx(
        expected_probability,
        abs=1e-7,
    )
    assert prediction["target_present"] == (expected_probability >= 0.5)
    assert not predictor.model.training
    assert all(parameter.grad is None for parameter in predictor.model.parameters())


@pytest.mark.parametrize("problem", ["features", "missing_stats", "stats", "weights"])
def test_inference_rejects_incompatible_checkpoint(problem):
    saved_checkpoint = make_checkpoint()
    if problem == "features":
        saved_checkpoint["feature_config"]["sample_rate"] = 16000
    elif problem == "missing_stats":
        del saved_checkpoint["normalization"]
    elif problem == "stats":
        saved_checkpoint["normalization"]["std"] = -1
    else:
        saved_checkpoint["model_state_dict"]["classifier.bias"].fill_(float("nan"))
    with pytest.raises(ValueError):
        CNNPredictor(saved_checkpoint)


def test_inference_rejects_overlong_audio(tmp_path):
    audio = tmp_path / "long.wav"
    overlong_audio = np.zeros(24255, dtype=np.float32)
    sf.write(audio, overlong_audio, 22050)
    with pytest.raises(ValueError, match="longer"):
        load_cnn_features(audio)
def test_smoke_is_explicitly_untrained():
    from songsored.cnn_inference import smoke

    result = smoke()
    assert result["trained"] is False
    assert result["mode"] == "smoke"
    assert 0 <= result["forward_output"]["target_probability"] <= 1
