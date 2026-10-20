#!/usr/bin/env python3
"""Train a linear classifier on cached embeddings from a frozen speech model."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import uuid
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import songsored.train_speech_classifier as runner_module
from songsored.speech_classifier import SpeechFeatureClassifier
from songsored.train import validate_one_epoch


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    with path.open("rb") as source_file:
        return hashlib.file_digest(source_file, "sha256").hexdigest()


def _get_source_hashes() -> dict[str, str]:
    """Return hashes for the training script and classifier source files."""
    runner_path = Path(inspect.getfile(runner_module)).resolve()
    classifier_path = Path(inspect.getfile(SpeechFeatureClassifier)).resolve()
    return {
        "train_speech_classifier": _sha256(runner_path),
        "speech_classifier": _sha256(classifier_path),
    }


def fit_embedding_stats(features: torch.Tensor) -> dict[str, Any]:
    """Compute a mean and standard deviation for each embedding dimension.

    Args:
        features: Non-empty float tensor shaped (samples, embedding_dim).

    Returns:
        A dictionary with float32 ``mean`` and ``scale`` tensors, each shaped
        (embedding_dim), plus the ``eps`` value used as the minimum scale.
    """
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(
            f"features must be a 2D torch.Tensor, got {type(features).__name__} with ndim={getattr(features, 'ndim', None)}."
        )
    if features.shape[0] == 0:
        raise ValueError("features must be nonempty.")
    if not torch.isfinite(features).all():
        raise ValueError("features must contain only finite values.")

    # Use double precision for the calculations, then store compact float32 stats.
    features64 = features.to(torch.float64)
    mean = features64.mean(dim=0)
    std = features64.std(dim=0, unbiased=False)
    eps = 1e-6
    scale = std.clamp_min(eps)

    return {
        "mean": mean.to(torch.float32),
        "scale": scale.to(torch.float32),
        "eps": eps,
    }


class BatchStream:
    """Draw seeded shuffled batches, keeping unused indices for the next batch."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        seed: int,
    ) -> None:
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.pending = torch.empty(0, dtype=torch.long)
        self.num_samples = len(features)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.pending) < self.batch_size:
            shuffled_indices = torch.randperm(
                self.num_samples,
                generator=self.generator,
            )
            self.pending = torch.cat([self.pending, shuffled_indices])

        batch_indices = self.pending[: self.batch_size]
        self.pending = self.pending[self.batch_size :]
        return self.features[batch_indices], self.labels[batch_indices]


def validate_cache(cache: dict[str, Any]) -> int:
    """Check cache contents and ensure the train and validation splits are separate.

    Returns:
        The embedding dimension recorded in the cache metadata.
    """
    if not isinstance(cache, dict):
        raise ValueError(f"Cache must be a dictionary, got {type(cache).__name__}.")

    expected_keys = {"metadata", "train", "val"}
    if set(cache.keys()) != expected_keys:
        raise ValueError(
            f"Top-level cache keys must be exactly {sorted(expected_keys)}, got {sorted(cache.keys())}."
        )

    metadata = cache["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Cache metadata must be a dictionary.")

    embedding_dim = metadata.get("embedding_dim")
    if (
        not isinstance(embedding_dim, int)
        or isinstance(embedding_dim, bool)
        or embedding_dim <= 0
    ):
        raise ValueError(
            f"Invalid embedding_dim in cache metadata: {embedding_dim!r}."
        )

    if "split_counts" not in metadata or not isinstance(metadata["split_counts"], dict):
        raise ValueError("Cache metadata must contain a 'split_counts' dictionary.")

    split_counts = metadata["split_counts"]
    for required_split in ("train", "val"):
        if required_split not in split_counts:
            raise ValueError(
                f"metadata['split_counts'] must contain entry for {required_split!r}."
            )
        sample_count = split_counts[required_split]
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count <= 0
        ):
            raise ValueError(
                f"metadata['split_counts'][{required_split!r}] must be a positive "
                f"integer, got {sample_count!r}."
            )

    all_recording_ids: dict[str, set[str]] = {}
    clip_paths_by_split: dict[str, set[str]] = {}

    for split in ("train", "val"):
        split_data = cache[split]
        if not isinstance(split_data, dict):
            raise ValueError(f"Split {split!r} must be a dictionary.")

        for key in ("features", "labels", "rows"):
            if key not in split_data:
                raise ValueError(f"Split {split!r} missing required key {key!r}.")

        features = split_data["features"]
        labels = split_data["labels"]
        rows = split_data["rows"]

        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Split {split!r} features must be a torch.Tensor.")
        if features.dtype != torch.float32:
            raise ValueError(f"Split {split!r} features must be float32, got {features.dtype}.")
        if features.ndim != 2:
            raise ValueError(f"Split {split!r} features must be 2D [N, embedding_dim], got {features.ndim}D.")
        if features.shape[1] != embedding_dim:
            raise ValueError(
                f"Split {split!r} features dim 1 ({features.shape[1]}) does not match "
                f"metadata embedding_dim ({embedding_dim})."
            )
        if not torch.isfinite(features).all():
            raise ValueError(f"Split {split!r} features contain non-finite values.")

        sample_count = features.shape[0]
        if sample_count == 0:
            raise ValueError(f"Split {split!r} is empty.")

        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"Split {split!r} labels must be a torch.Tensor.")
        if labels.dtype != torch.float32:
            raise ValueError(f"Split {split!r} labels must be float32, got {labels.dtype}.")
        if labels.ndim != 2 or labels.shape != (sample_count, 1):
            raise ValueError(
                f"Split {split!r} labels must have shape [{sample_count}, 1], "
                f"got {tuple(labels.shape)}."
            )
        if not torch.isfinite(labels).all():
            raise ValueError(f"Split {split!r} labels contain non-finite values.")

        unique_labels = set(labels.unique().tolist())
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError(
                f"Split {split!r} labels must contain only 0 or 1, got {unique_labels}."
            )

        if not isinstance(rows, list):
            raise TypeError(f"Split {split!r} rows must be a list.")
        if len(rows) != sample_count:
            raise ValueError(
                f"Split {split!r} row count ({len(rows)}) does not match "
                f"feature count ({sample_count})."
            )

        if split_counts[split] != sample_count:
            raise ValueError(
                f"Split {split!r} metadata count ({split_counts[split]}) does not "
                f"match feature count ({sample_count})."
            )

        recording_ids: set[str] = set()
        clip_paths: set[str] = set()
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"Split {split!r} row {row_index} must be a dict.")
            row_label = row.get("label")
            tensor_label = labels[row_index, 0].item()
            if float(row_label) != tensor_label:
                raise ValueError(
                    f"Split {split!r} row {row_index} label ({row_label}) does not match "
                    f"tensor label ({tensor_label})."
                )
            recording_ids.add(row["recording_id"])
            clip_paths.add(row["path"])

        all_recording_ids[split] = recording_ids
        clip_paths_by_split[split] = clip_paths

    overlap_rids = all_recording_ids["train"] & all_recording_ids["val"]
    if overlap_rids:
        raise ValueError(
            f"Train and val splits share recording IDs: {sorted(overlap_rids)}."
        )

    overlap_paths = clip_paths_by_split["train"] & clip_paths_by_split["val"]
    if overlap_paths:
        raise ValueError(
            f"Train and val splits share clip paths: {sorted(overlap_paths)}."
        )

    return embedding_dim


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        temporary_path.write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def train_speech_classifier(
    cache_path: str | Path,
    run_dir: str | Path,
    *,
    seed: int = 42,
    max_steps: int = 24600,
    batch_size: int = 16,
    validation_interval: int = 82,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> Path:
    """Train the linear classifier and return its best checkpoint.

    Args:
        cache_path: Path to the ``.pt`` feature cache.
        run_dir: New directory for checkpoints and training history.
        seed: Seed for model initialization and batch sampling.
        max_steps: Number of optimizer updates to perform.
        batch_size: Number of examples in each training batch.
        validation_interval: Evaluate the validation set every this many steps.
        learning_rate: Learning rate used by Adam.
        device: Device for training, such as ``"cpu"`` or ``"cuda"``.

    Returns:
        Absolute path to the best checkpoint.
    """
    # Reject invalid settings before creating the run directory.
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError(f"max_steps must be a positive integer, got {max_steps!r}")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if (
        not isinstance(validation_interval, int)
        or isinstance(validation_interval, bool)
        or validation_interval <= 0
    ):
        raise ValueError(
            f"validation_interval must be a positive integer, got {validation_interval!r}"
        )
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError(
            f"learning_rate must be a positive finite float, got {learning_rate!r}"
        )

    # Start each run in a new directory rather than overwriting an earlier run.
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    cache_path = Path(cache_path).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Feature cache file not found: {cache_path}")

    cache_sha256 = _sha256(cache_path)

    # Load the cache on CPU and check its structure before training.
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    embedding_dim = validate_cache(cache)

    raw_train_features = cache["train"]["features"]
    raw_train_labels = cache["train"]["labels"]
    raw_val_features = cache["val"]["features"]
    raw_val_labels = cache["val"]["labels"]

    # Fit normalization on training embeddings only to avoid validation leakage.
    stats = fit_embedding_stats(raw_train_features)
    train_features = (raw_train_features - stats["mean"]) / stats["scale"]
    val_features = (raw_val_features - stats["mean"]) / stats["scale"]

    train_labels = raw_train_labels.clone()
    val_labels = raw_val_labels.clone()

    # Set up the classifier, optimizer, loss, and reproducible batch stream.
    torch.manual_seed(seed)
    target_device = torch.device(device)

    model = SpeechFeatureClassifier(embedding_dim=embedding_dim).to(target_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    criterion = nn.BCEWithLogitsLoss()

    batch_stream = BatchStream(
        features=train_features,
        labels=train_labels,
        batch_size=batch_size,
        seed=seed,
    )

    validation_dataset = TensorDataset(val_features, val_labels)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    source_hashes = _get_source_hashes()
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"
    history_path = run_dir / "history.json"

    best_val_loss = float("inf")
    best_step: int | None = None
    history_records: list[dict[str, Any]] = []

    interval_loss_total = 0.0
    interval_step_count = 0

    # Update the model and check validation loss at the requested intervals.
    for step in range(1, max_steps + 1):
        model.train()
        batch_features, batch_labels = batch_stream.next_batch()
        batch_features = batch_features.to(target_device)
        batch_labels = batch_labels.to(target_device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Encountered non-finite training loss at step {step}: {loss.item()}"
            )

        loss.backward()
        optimizer.step()

        interval_loss_total += loss.item()
        interval_step_count += 1

        # Include the final step even when it falls between regular checks.
        if step % validation_interval == 0 or step == max_steps:
            average_train_loss = interval_loss_total / interval_step_count
            val_loss = validate_one_epoch(
                model,
                validation_loader,
                criterion,
                device=target_device,
            )

            if not math.isfinite(val_loss):
                raise FloatingPointError(
                    f"Encountered non-finite validation loss at step {step}: {val_loss}"
                )

            record = {
                "step": step,
                "train_loss": average_train_loss,
                "val_loss": val_loss,
            }
            history_records.append(record)

            # Save a new best checkpoint only for a strictly lower validation loss.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_ckpt = {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": embedding_dim,
                    "stats": {
                        "mean": stats["mean"].clone(),
                        "scale": stats["scale"].clone(),
                        "eps": stats["eps"],
                    },
                    "seed": seed,
                    "step": step,
                    "optimizer": {
                        "name": "Adam",
                        "learning_rate": learning_rate,
                        "weight_decay": 0.0,
                    },
                    "batch_size": batch_size,
                    "validation_interval": validation_interval,
                    "train_loss": average_train_loss,
                    "val_loss": val_loss,
                    "cache_path": str(cache_path),
                    "cache_sha256": cache_sha256,
                    "cache_metadata": cache["metadata"],
                    "source_file_hashes": source_hashes,
                }
                _save_checkpoint(best_checkpoint_path, best_ckpt)

            interval_loss_total = 0.0
            interval_step_count = 0

    # Preserve the final model separately from the best-validation checkpoint.
    last_checkpoint = {
        "model_state_dict": model.state_dict(),
        "embedding_dim": embedding_dim,
        "stats": {
            "mean": stats["mean"].clone(),
            "scale": stats["scale"].clone(),
            "eps": stats["eps"],
        },
        "seed": seed,
        "step": max_steps,
        "optimizer": {
            "name": "Adam",
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
        },
        "batch_size": batch_size,
        "validation_interval": validation_interval,
        "train_loss": history_records[-1]["train_loss"],
        "val_loss": history_records[-1]["val_loss"],
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha256,
        "cache_metadata": cache["metadata"],
        "source_file_hashes": source_hashes,
    }
    _save_checkpoint(last_checkpoint_path, last_checkpoint)

    # Record settings, selected checkpoint, and validation history for this run.
    history_data = {
        "settings": {
            "seed": seed,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "validation_interval": validation_interval,
            "learning_rate": learning_rate,
            "device": str(device),
            "embedding_dim": embedding_dim,
            "cache_path": str(cache_path),
            "cache_sha256": cache_sha256,
        },
        "selected_step": best_step,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "steps": history_records,
        "history": history_records,
        "records": history_records,
    }
    _write_json(history_path, history_data)

    # Reload the best checkpoint and confirm it reproduces its saved validation loss.
    loaded_best = torch.load(best_checkpoint_path, map_location="cpu", weights_only=True)
    eval_model = SpeechFeatureClassifier(embedding_dim=loaded_best["embedding_dim"]).to(
        target_device
    )
    eval_model.load_state_dict(loaded_best["model_state_dict"])
    eval_model.eval()

    saved_mean = loaded_best["stats"]["mean"].to(raw_val_features.device)
    saved_scale = loaded_best["stats"]["scale"].to(raw_val_features.device)
    normalized_val_features = (raw_val_features - saved_mean) / saved_scale
    saved_validation_loader = DataLoader(
        TensorDataset(normalized_val_features, raw_val_labels),
        batch_size=batch_size,
        shuffle=False,
    )
    recomputed_val_loss = validate_one_epoch(
        eval_model,
        saved_validation_loader,
        criterion,
        device=target_device,
    )

    if not math.isclose(
        recomputed_val_loss, loaded_best["val_loss"], rel_tol=1e-5, abs_tol=1e-6
    ):
        raise ValueError(
            f"Recomputed validation loss ({recomputed_val_loss}) does not match "
            f"recorded best validation loss ({loaded_best['val_loss']})."
        )

    return best_checkpoint_path.resolve()


def main() -> None:
    """Parse command-line options and run classifier training."""
    parser = argparse.ArgumentParser(
        description="Train linear speech feature classifier on cached embeddings."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Path to feature cache .pt file",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory to store run checkpoints and history",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=24600,
        help="Total optimizer update steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=82,
        help="Validation interval in steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device ('cpu', 'cuda', 'mps')",
    )
    args = parser.parse_args()

    best_path = train_speech_classifier(
        cache_path=args.cache,
        run_dir=args.run_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        validation_interval=args.val_interval,
        learning_rate=args.lr,
        device=args.device,
    )
    print(f"Training completed successfully. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
