"""Test the CNN pipeline with generated signals and temporary files."""

import csv
from functools import partial
import json
import math

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader, TensorDataset

from songsored.dataset import AudioClipDataset
from songsored.features import extract_mel_spectrogram
from songsored.model import WordClassifier
from songsored.normalization import fit_feature_stats, normalize_features
from songsored.train import train, train_one_epoch, validate_one_epoch


@pytest.fixture(autouse=True)
def cpu_threads():
    """Keep the small tensor tests fast and restore the previous setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_tone(frequency, sample_rate=22050, duration=1.0):
    times = np.arange(round(sample_rate * duration)) / sample_rate
    waveform = 0.1 * np.sin(2 * np.pi * frequency * times)
    return waveform.astype(np.float32)


def write_manifest(folder, rows, header=None):
    path = folder / "generated.csv"
    with path.open("w", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(header or ["path", "label", "recording_id", "split"])
        writer.writerows(rows)
    return path


def test_log_mel_preserves_frequency_and_level_information():
    low = extract_mel_spectrogram(make_tone(440))
    high = extract_mel_spectrogram(make_tone(1760))
    assert low.shape == high.shape == (128, 44)
    assert low.dtype == np.float32 and np.isfinite(low).all()
    # Ignore padded edge frames when comparing where each tone peaks.
    low_band = low[:, 4:-4].mean(axis=1).argmax()
    high_band = high[:, 4:-4].mean(axis=1).argmax()
    bands = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=22050 / 2)
    assert abs(bands[low_band] - 440) < 100
    assert abs(bands[high_band] - 1760) < 200
    assert high_band > low_band
    quieter = extract_mel_spectrogram(make_tone(440) * 0.1)
    # Ten times smaller waveform amplitude means 100 times less power: -20 dB.
    np.testing.assert_allclose(
        quieter[low_band, 4:-4] - low[low_band, 4:-4],
        -20,
        atol=1e-3,
    )
    silence = np.zeros(22050, dtype=np.float32)
    assert np.isfinite(extract_mel_spectrogram(silence)).all()


@pytest.mark.parametrize("pool_size", [(1, 1), (1, 4)])
def test_cnn_accepts_batches_and_backpropagates(pool_size):
    torch.manual_seed(42)
    model = WordClassifier(pool_size=pool_size)
    features = torch.randn(3, 1, 128, 44)
    logits = model(features)
    assert logits.shape == (3, 1) and torch.isfinite(logits).all()
    labels = torch.tensor([[0.], [1.], [0.]])
    torch.nn.BCEWithLogitsLoss()(logits, labels).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert any(torch.count_nonzero(p.grad) for p in model.parameters())


def test_dataset_prepares_audio_and_batches_labels(tmp_path):
    mono = make_tone(440, sample_rate=16000, duration=0.5)
    stereo = np.stack([mono, mono], axis=1)
    sf.write(tmp_path / "generated.wav", stereo, 16000, subtype="FLOAT")
    manifest = write_manifest(
        tmp_path,
        [["generated.wav", 0, "generated-a", "train"]],
    )
    dataset = AudioClipDataset(manifest, "train")
    features, label = dataset[0]
    assert features.shape == (1, 128, 44)
    assert features.dtype == torch.float32
    assert label.shape == (1,)
    assert label.item() == 0
    assert torch.isfinite(features).all()
    # The final frames lie well inside the zero padding and share its floor.
    assert torch.unique(features[:, :, -10:]).numel() == 1
    batch, labels = next(iter(DataLoader(dataset, batch_size=1)))
    assert batch.shape == (1, 1, 128, 44)
    assert labels.shape == (1, 1)
    assert WordClassifier()(batch).shape == labels.shape


@pytest.mark.parametrize(
    "problem",
    ["recording_overlap", "bad_label", "blank_id", "duplicate_header", "empty_split"],
)
def test_manifest_rejects_invalid_examples(tmp_path, problem):
    sf.write(tmp_path / "generated.wav", make_tone(440), 22050)
    rows = [["generated.wav", "0", "a", "train"]]
    header = None
    if problem == "recording_overlap":
        rows.append(["generated.wav", "0", " a ", "val"])
    elif problem == "bad_label":
        rows[0][1] = "2"
    elif problem == "blank_id":
        rows[0][2] = " "
    elif problem == "duplicate_header":
        header = ["path", "label", "recording_id", "split", "label"]
        rows[0].append("0")
    elif problem == "empty_split":
        rows[0][3] = "val"
    with pytest.raises(ValueError):
        AudioClipDataset(write_manifest(tmp_path, rows, header), "train")


def test_dataset_rejects_overlong_audio_instead_of_cropping(tmp_path):
    sf.write(tmp_path / "long.wav", make_tone(440, duration=1.1), 22050)
    manifest = write_manifest(tmp_path, [["long.wav", 0, "a", "train"]])
    with pytest.raises(ValueError, match="longer"):
        AudioClipDataset(manifest, "train")[0]


class SmallFeatureDataset:
    """Controlled tensors for checking statistics, not an audio dataset."""

    split = "train"
    transform = None

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], torch.tensor([0.])


def test_normalization_matches_known_statistics_and_handles_constant_input():
    dataset = SmallFeatureDataset(
        [
            torch.tensor([[[-60.0, -40.0]]]),
            torch.tensor([[[-20.0, 0.0]]]),
        ]
    )
    stats = fit_feature_stats(dataset)
    assert stats["mean"] == -30
    assert stats["count"] == 4
    assert stats["std"] == pytest.approx(math.sqrt(500))
    original_features = dataset.features[0].clone()
    normalized = normalize_features(original_features, stats)
    torch.testing.assert_close(
        normalized, (original_features + 30) / math.sqrt(500)
    )
    assert torch.equal(original_features, dataset.features[0])
    constant_features = torch.full((1, 2, 2), -40.0)
    constant_stats = fit_feature_stats(SmallFeatureDataset([constant_features]))
    assert constant_stats["std"] == 0
    assert torch.equal(
        normalize_features(constant_features, constant_stats),
        torch.zeros_like(constant_features),
    )


@pytest.mark.parametrize("problem", ["validation", "transformed", "empty"])
def test_stats_require_raw_nonempty_training_data(problem):
    dataset = SmallFeatureDataset([torch.ones(1, 2, 2)])
    if problem == "validation":
        dataset.split = "val"
    elif problem == "transformed":
        dataset.transform = lambda features: features
    else:
        dataset.features = []
    with pytest.raises(ValueError):
        fit_feature_stats(dataset)


def test_training_epoch_weights_losses_by_sample_and_updates_parameters():
    torch.manual_seed(42)
    features = torch.randn(5, 1, 128, 44)
    labels = torch.tensor([[0.], [1.], [0.], [1.], [1.]])
    loader = DataLoader(TensorDataset(features, labels), batch_size=3)
    model, loss_fn = WordClassifier(), torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        expected_loss = loss_fn(model(features), labels).item()

    # A zero learning rate lets this check focus on loss averaging.
    measured_loss = train_one_epoch(
        model,
        loader,
        torch.optim.SGD(model.parameters(), lr=0),
        loss_fn,
        "cpu",
    )
    assert measured_loss == pytest.approx(expected_loss, abs=1e-6)

    previous_weights = {
        name: value.clone() for name, value in model.state_dict().items()
    }
    measured_loss = train_one_epoch(
        model,
        loader,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        loss_fn,
        "cpu",
    )
    assert math.isfinite(measured_loss)
    assert any(
        not torch.equal(previous_weights[name], value)
        for name, value in model.state_dict().items()
    )


def test_validation_disables_gradients_and_keeps_model_unchanged():
    torch.manual_seed(42)
    features = torch.randn(5, 1, 128, 44)
    labels = torch.tensor([[0.], [1.], [0.], [1.], [1.]])
    model, loss_fn = WordClassifier(), torch.nn.BCEWithLogitsLoss()
    previous_weights = {
        name: value.clone() for name, value in model.state_dict().items()
    }
    observations = []
    hook = model.register_forward_pre_hook(
        lambda module, args: observations.append(
            (module.training, torch.is_grad_enabled())
        )
    )
    try:
        measured_loss = validate_one_epoch(
            model,
            DataLoader(TensorDataset(features, labels), batch_size=3),
            loss_fn,
            "cpu",
        )
    finally:
        hook.remove()

    with torch.no_grad():
        expected_loss = loss_fn(model(features), labels).item()
    assert measured_loss == pytest.approx(expected_loss, abs=1e-6)
    assert observations == [(False, False), (False, False)]
    assert all(
        torch.equal(previous_weights[name], value)
        for name, value in model.state_dict().items()
    )
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("phase", ["train", "val"])
def test_epoch_rejects_empty_loader(phase):
    empty_features = torch.empty(0, 1, 128, 44)
    empty_labels = torch.empty(0, 1)
    loader = DataLoader(TensorDataset(empty_features, empty_labels))
    model, loss_fn = WordClassifier(), torch.nn.BCEWithLogitsLoss()
    with pytest.raises(ValueError, match="empty"):
        if phase == "train":
            train_one_epoch(
                model,
                loader,
                torch.optim.Adam(model.parameters()),
                loss_fn,
                "cpu",
            )
        else:
            validate_one_epoch(model, loader, loss_fn, "cpu")


def test_training_runner_saves_reloadable_best_checkpoint(tmp_path):
    # Generated tones exercise checkpoint plumbing, not word recognition.
    split_labels = [
        ("train", 0),
        ("train", 1),
        ("train", 0),
        ("val", 0),
        ("val", 1),
    ]
    rows = []
    for index, (split, label) in enumerate(split_labels):
        filename = f"generated_{index}.wav"
        sf.write(
            tmp_path / filename,
            make_tone(220 + index * 110),
            22050,
            subtype="FLOAT",
        )
        rows.append([filename, label, f"generated-{index}", split])

    manifest = write_manifest(tmp_path, rows)
    expected_stats = fit_feature_stats(AudioClipDataset(manifest, "train"))
    checkpoint_path = train(
        manifest,
        tmp_path / "run/best.pt",
        epochs=2,
        batch_size=2,
        normalize_inputs=True,
    )
    history_path = checkpoint_path.with_suffix(".history.json")
    history = json.loads(history_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    best_epoch = min(history["epochs"], key=lambda record: record["val_loss"])

    assert len(history["epochs"]) == 2
    assert checkpoint["epoch"] == history["best_epoch"] == best_epoch["epoch"]
    assert (
        checkpoint["val_loss"]
        == history["best_val_loss"]
        == best_epoch["val_loss"]
    )
    assert checkpoint["normalization"] == expected_stats
    model = WordClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])

    validation_transform = partial(
        normalize_features,
        stats=checkpoint["normalization"],
    )
    validation_dataset = AudioClipDataset(
        manifest,
        "val",
        transform=validation_transform,
    )
    validation_loss = validate_one_epoch(
        model,
        DataLoader(validation_dataset, batch_size=2),
        torch.nn.BCEWithLogitsLoss(),
        "cpu",
    )
    assert validation_loss == pytest.approx(checkpoint["val_loss"], abs=1e-7)

    from matplotlib.image import imread
    image = imread(checkpoint_path.with_suffix(".loss.png"))
    assert image.size > 0 and np.isfinite(image).all()
