"""Helpers for training the audio classifier."""

from __future__ import annotations

from functools import partial
import json
import math
from pathlib import Path

from .augmentation import random_frequency_mask
from .dataset import AudioClipDataset
from .model import WordClassifier
from .normalization import fit_feature_stats, normalize_features

import torch
from torch.utils.data import DataLoader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str | torch.device,
) -> float:
    """Train the model for one epoch and return the mean loss per sample.

    Args:
        model: Classifier being trained.
        loader: Batches of features shaped (batch, 1, n_mels, time_frames)
            and labels shaped (batch, 1).
        optimizer: Optimizer for the model parameters.
        criterion: Mean-reduced loss function, such as
            ``torch.nn.BCEWithLogitsLoss``.
        device: Device for the batches. The model must already be on this device.

    Returns:
        Mean loss across all samples processed in this epoch.

    Raises:
        ValueError: If the loader contains no samples.
    """
    model.train()

    weighted_loss = 0.0
    sample_count = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        # Run one optimization step for this batch.
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        weighted_loss += loss.item() * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError(
            "train_one_epoch received an empty DataLoader; "
            "no samples were processed."
        )

    # Undo the mean reduction from each batch to average across all samples.
    return weighted_loss / sample_count


def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: str | torch.device,
) -> float:
    """Evaluate the model and return the mean loss per sample.

    Evaluation mode and disabled gradient tracking ensure that this pass does
    not update the model. The caller must place the model on ``device`` first.

    Args:
        model: Classifier being evaluated.
        loader: Batches of features shaped (batch, 1, n_mels, time_frames)
            and labels shaped (batch, 1).
        criterion: Mean-reduced loss function, such as
            ``torch.nn.BCEWithLogitsLoss``.
        device: Device for the batches.

    Returns:
        Mean loss across all samples in the validation set.

    Raises:
        ValueError: If the loader contains no samples.
    """
    model.eval()

    weighted_loss = 0.0
    sample_count = 0

    # Validation needs no gradients because it does not update model weights.
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = features.size(0)
            weighted_loss += loss.item() * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError(
            "validate_one_epoch received an empty DataLoader; "
            "no samples were processed."
        )

    # Weight each batch mean by its size so the last, smaller batch counts correctly.
    return weighted_loss / sample_count


def plot_loss_history(
    history_path: str | Path,
    plot_path: str | Path,
) -> Path:
    """Plot training and validation losses from a saved history file.

    Args:
        history_path: Path to the JSON file written by ``train()``.
        plot_path: Destination path for the PNG image.

    Returns:
        Resolved path to the saved plot.

    Raises:
        ValueError: If the history contains no epoch records.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads(Path(history_path).read_text(encoding="utf-8"))

    epoch_records = history["epochs"]
    if not epoch_records:
        raise ValueError(
            f"History file contains no epoch records: {history_path}"
        )

    epoch_numbers = [record["epoch"] for record in epoch_records]
    train_losses = [record["train_loss"] for record in epoch_records]
    val_losses = [record["val_loss"] for record in epoch_records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch_numbers, train_losses, label="Training", marker="o")
    ax.plot(epoch_numbers, val_losses, label="Validation", marker="o")

    best_epoch = history.get("best_epoch")
    if best_epoch is not None:
        ax.axvline(
            best_epoch,
            linestyle="--",
            color="grey",
            linewidth=1,
            label=f"Best val (epoch {best_epoch})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy loss")
    ax.set_title("CNN training and validation loss")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)

    fig.tight_layout()
    plot_path = Path(plot_path).resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path


def _normalized_masked_features(
    features: torch.Tensor,
    *,
    stats: dict,
    max_width: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Standardize a feature batch, then mask random frequency bands."""
    normalized = normalize_features(features, stats)
    return random_frequency_mask(
        normalized,
        max_width=max_width,
        generator=generator,
    )


def train(
    manifest_path: str | Path,
    model_output: str | Path,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 42,
    pool_size: tuple[int, int] = (1, 1),
    normalize_inputs: bool = False,
    weight_decay: float = 0.0,
    freq_mask_width: int = 0,
) -> Path:
    """Train a WordClassifier and save the checkpoint with the best validation loss.

    The CSV manifest provides the train and validation split assignments.
    Training runs for ``epochs`` epochs, saving the checkpoint whenever
    validation loss improves.

    Args:
        manifest_path: Path to the CSV manifest file.
        model_output: Path where the best model checkpoint will be written.
        epochs: Number of training epochs (must be a positive integer).
        batch_size: Mini-batch size (must be a positive integer).
        learning_rate: Adam optimizer learning rate (must be finite and > 0).
        device: Compute device, e.g. ``"cpu"`` or ``"mps"``.
        seed: Random seed for weight initialization and training shuffle.
        pool_size: Adaptive pooling size as (frequency, time). Defaults to
            (1, 1); use (1, 4) to retain four time regions.
        normalize_inputs: Standardize features using statistics fitted on the
            training split only. Defaults to False.
        weight_decay: Nonnegative L2 regularization coefficient used by Adam.
            Defaults to zero.
        freq_mask_width: Maximum frequency-mask width, from 0 to 127.
            Zero disables masking; a positive width enables training-only
            masking and requires ``normalize_inputs=True``.

    Returns:
        Path to the saved best-validation-loss checkpoint.

    Raises:
        ValueError: If an argument is out of range or a loss becomes
            non-finite during training.

    Note:
        With frequency masking enabled, ``train_loss`` is measured on randomly
        masked inputs and is not directly comparable to an unmasked training
        loss. Checkpoint selection still uses unmasked validation loss.
    """
    # Check the configuration before opening datasets or creating output files.
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError(
            f"epochs must be a positive integer, got {epochs!r}."
        )
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size!r}."
        )
    if not (math.isfinite(learning_rate) and learning_rate > 0):
        raise ValueError(
            f"learning_rate must be finite and greater than zero, "
            f"got {learning_rate!r}."
        )
    if (
        not isinstance(pool_size, (tuple, list))
        or len(pool_size) != 2
        or any(type(value) is not int or value < 1 for value in pool_size)
    ):
        raise ValueError(
            f"pool_size must contain two positive integers, got {pool_size!r}."
        )

    pool_size = tuple(pool_size)

    if not isinstance(normalize_inputs, bool):
        raise ValueError(
            f"normalize_inputs must be a bool, got {normalize_inputs!r}."
        )

    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be a finite, nonnegative number")
    weight_decay = float(weight_decay)

    if (
        isinstance(freq_mask_width, bool)
        or not isinstance(freq_mask_width, int)
        or not 0 <= freq_mask_width < 128
    ):
        raise ValueError(
            "freq_mask_width must be an integer between 0 and 127"
        )

    if freq_mask_width > 0 and not normalize_inputs:
        raise ValueError(
            "Frequency masking requires normalize_inputs=True"
        )

    # Set the seed and resolve all output paths once.
    torch.manual_seed(seed)
    device = torch.device(device)
    output_path = Path(model_output).expanduser().resolve()
    history_path = output_path.with_suffix(".history.json")
    plot_path = output_path.with_suffix(".loss.png")

    # Load the training and validation splits from the manifest.
    train_dataset = AudioClipDataset(manifest_path, split="train")
    val_dataset = AudioClipDataset(manifest_path, split="val")

    normalization_stats = None

    if normalize_inputs:
        normalization_stats = fit_feature_stats(train_dataset)

        # Reuse training-only statistics for both splits.
        transform = partial(
            normalize_features,
            stats=normalization_stats,
        )

        train_dataset.transform = transform
        val_dataset.transform = transform

    augmentation_config = None

    if freq_mask_width > 0:
        # Give frequency masking its own reproducible random-number stream.
        augmentation_seed = torch.initial_seed()

        augmentation_generator = torch.Generator(device="cpu")
        augmentation_generator.manual_seed(augmentation_seed)

        train_dataset.transform = partial(
            _normalized_masked_features,
            stats=normalization_stats,
            max_width=freq_mask_width,
            generator=augmentation_generator,
        )

        augmentation_config = {
            "method": "frequency_mask",
            "max_width": freq_mask_width,
            "num_masks": 1,
            "fill_value": 0.0,
            "after_normalization": True,
            "apply_to": "train",
            "seed": augmentation_seed,
        }

    # Shuffle training clips, but keep validation order fixed.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # Set up the classifier and its training objective.
    model = WordClassifier(pool_size=pool_size).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Create the directory before the first history or checkpoint write.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history: dict = {
        "config": {
            "manifest_path": str(train_dataset.manifest_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": str(device),
            "seed": seed,
            "pool_size": list(model.pool_size),
            "normalize_inputs": normalize_inputs,
            "normalization": normalization_stats,
            "optimizer": "Adam",
            "weight_decay": weight_decay,
            "augmentation_config": augmentation_config,
        },
        "epochs": [],
        "best_epoch": None,
        "best_val_loss": None,
    }
    best_val_loss = float("inf")

    # Train, validate, then save the best model and updated history each epoch.
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        if not math.isfinite(train_loss):
            raise ValueError(
                f"Epoch {epoch}: training loss is not finite ({train_loss}). "
                "Check your data and learning rate."
            )
        if not math.isfinite(val_loss):
            raise ValueError(
                f"Epoch {epoch}: validation loss is not finite ({val_loss}). "
                "Check your data and model."
            )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        history["epochs"].append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        # Keep a checkpoint only when validation loss improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            history["best_val_loss"] = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "normalization": normalization_stats,
                    "augmentation_config": augmentation_config,
                    "model_config": {
                        "n_mels": model.n_mels,
                        "num_classes": model.num_classes,
                        "pool_size": model.pool_size,
                    },
                    "feature_config": {
                        "sample_rate": train_dataset.sample_rate,
                        "clip_duration": train_dataset.clip_duration,
                        "n_mels": 128,
                        "n_fft": 2048,
                        "hop_length": 512,
                    },
                    "optimizer_config": {
                        "name": "Adam",
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                    },
                    "manifest_path": str(train_dataset.manifest_path),
                    "seed": seed,
                },
                output_path,
            )

        # Replace the previous history only after the new JSON is fully written.
        temporary_history_path = history_path.with_suffix(".json.tmp")
        temporary_history_path.write_text(
            json.dumps(history, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_history_path.replace(history_path)

    # Save the loss curve and return the best checkpoint path.
    plot_loss_history(history_path, plot_path)
    print(f"History : {history_path}")
    print(f"Plot    : {plot_path}")
    return output_path
