#!/usr/bin/env python3
"""Build a cache of frozen XLS-R embeddings for audio classification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import os
from pathlib import Path
import re
import uuid

import librosa
import numpy as np
import torch
from tqdm import tqdm
import transformers

import songsored.speech_encoder as speech_encoder_module
import songsored.speech_features as speech_features_module
from songsored.speech_encoder import FrozenSpeechEncoder
from songsored.speech_features import load_speech_waveform

_REQUIRED_COLUMNS = {"path", "label", "recording_id", "split"}
_VALID_SPLITS = {"train", "val"}
_VALID_LABELS = {"0", "1"}
_MODEL_ID = "facebook/wav2vec2-xls-r-300m"
_COMMIT_HASH_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    with path.open("rb") as source_file:
        return hashlib.file_digest(source_file, "sha256").hexdigest()


def _get_source_hashes() -> dict[str, str]:
    """Return hashes for the encoder and waveform loader source files."""
    encoder_path = Path(inspect.getfile(speech_encoder_module)).resolve()
    loader_path = Path(inspect.getfile(speech_features_module)).resolve()
    return {
        "speech_encoder": _sha256(encoder_path),
        "speech_features": _sha256(loader_path),
    }


def validate_manifest(manifest_path: Path) -> tuple[list[dict], list[dict]]:
    """Validate the manifest and return its training and validation rows.

    Raises:
        FileNotFoundError: If manifest or any referenced clip file does not exist.
        ValueError: If headers are invalid, fields are blank, labels are not 0/1,
            splits are invalid or include 'test', recordings cross splits,
            duplicate clip paths exist, or either split is empty.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest_dir = manifest_path.parent

    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest CSV has no header row: {manifest_path}")

        headers: set[str] = set()
        for column_name in reader.fieldnames:
            if column_name in headers:
                raise ValueError(
                    f"Manifest has duplicate column header: {column_name!r}. "
                    "Each column header must be unique."
                )
            headers.add(column_name)

        missing = _REQUIRED_COLUMNS - headers
        if missing:
            raise ValueError(
                f"Manifest is missing required column(s): {sorted(missing)}"
            )

        all_rows = list(reader)

    if not all_rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    seen_resolved_paths: set[Path] = set()
    recording_split: dict[str, str] = {}
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    for row_number, row in enumerate(all_rows, start=2):
        for column_name in _REQUIRED_COLUMNS:
            value = (row.get(column_name) or "").strip()
            if not value:
                raise ValueError(
                    f"Row {row_number}: required field {column_name!r} is missing or empty."
                )

        label = row["label"].strip()
        if label not in _VALID_LABELS:
            raise ValueError(
                f"Row {row_number}: label must be '0' or '1', got {label!r}."
            )

        split = row["split"].strip()
        if split == "test":
            raise ValueError(
                f"Row {row_number}: split is 'test', but splits must be exclusively "
                f"'train' or 'val'. Test rows must be excluded prior to cache building."
            )
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"Row {row_number}: invalid split {split!r}. "
                f"Splits must be exclusively 'train' or 'val'."
            )

        recording_id = row["recording_id"].strip()
        if (
            recording_id in recording_split
            and recording_split[recording_id] != split
        ):
            raise ValueError(
                f"Row {row_number}: recording_id {recording_id!r} appears in both "
                f"{recording_split[recording_id]!r} and {split!r} splits. "
                "Each recording must belong to exactly one split."
            )
        recording_split[recording_id] = split

        manifest_path_value = Path(row["path"].strip())
        clip_path = (
            manifest_path_value
            if manifest_path_value.is_absolute()
            else manifest_dir / manifest_path_value
        ).resolve()

        if clip_path in seen_resolved_paths:
            raise ValueError(
                f"Row {row_number}: duplicate clip path detected: {clip_path}"
            )
        seen_resolved_paths.add(clip_path)

        if not clip_path.is_file():
            raise FileNotFoundError(
                f"Row {row_number}: clip file not found: {clip_path} "
                f"(from manifest entry {row['path']!r})"
            )

        entry = {
            "path": clip_path,
            "recording_id": recording_id,
            "label": int(label),
            "split": split,
        }
        if split == "train":
            train_rows.append(entry)
        else:
            val_rows.append(entry)

    if not train_rows:
        raise ValueError("Manifest contains no training examples.")
    if not val_rows:
        raise ValueError("Manifest contains no validation examples.")

    return train_rows, val_rows


def build_feature_cache(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    revision: str,
    device: str = "cpu",
    allow_download: bool = False,
) -> Path:
    """Extract frozen speech embeddings and save them for train and validation.

    Args:
        manifest_path: CSV manifest containing the training and validation clips.
        output_path: Destination for the ``.pt`` feature cache.
        revision: Immutable 40-character commit hash for the pretrained model.
        device: Device used for encoder inference, such as ``"cpu"`` or ``"cuda"``.
        allow_download: Explicit opt-in to fetching the pinned pretrained encoder.

    Returns:
        Absolute path to the saved feature cache.

    Raises:
        FileExistsError: If output_path already exists.
        FileNotFoundError: If manifest or any referenced clip file is missing.
        ValueError: If manifest format is invalid or validation checks fail.
    """
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Output cache file already exists: {output_path}. "
            "Will not overwrite an existing cache."
        )

    if not isinstance(revision, str) or not _COMMIT_HASH_RE.fullmatch(revision):
        raise ValueError(
            f"Model revision must be an immutable 40-character hexadecimal commit hash, got {revision!r}."
        )

    manifest_path = Path(manifest_path).resolve()
    manifest_sha256 = _sha256(manifest_path)

    # Check the manifest before loading the encoder or processing audio.
    train_rows, val_rows = validate_manifest(manifest_path)
    total_clips = len(train_rows) + len(val_rows)

    # Load the encoder once and reuse it for every clip.
    encoder = FrozenSpeechEncoder(
        model_id=_MODEL_ID,
        revision=revision,
        device=device,
        local_files_only=not allow_download,
    )

    # Store embeddings and their labels and manifest details by split.
    cache_splits: dict[str, dict[str, list]] = {
        "train": {"embeddings": [], "labels": [], "rows": []},
        "val": {"embeddings": [], "labels": [], "rows": []},
    }

    progress_bar = tqdm(
        total=total_clips,
        desc="Extracting speech features",
        unit="clip",
    )
    try:
        for split_name, split_rows in (("train", train_rows), ("val", val_rows)):
            for row in split_rows:
                clip_path: Path = row["path"]
                try:
                    waveform = load_speech_waveform(clip_path)
                    embedding = encoder.encode(waveform)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed processing clip {clip_path}: {exc}"
                    ) from exc

                # Check the encoder returned the expected standalone vector.
                if embedding.shape != (encoder.embedding_dim,):
                    raise ValueError(
                        f"Embedding shape {embedding.shape} does not match expected "
                        f"({encoder.embedding_dim},) for {clip_path}"
                    )
                if embedding.device.type != "cpu":
                    raise ValueError(
                        f"Embedding device {embedding.device} is not CPU for {clip_path}"
                    )
                if embedding.dtype != torch.float32:
                    raise ValueError(
                        f"Embedding dtype {embedding.dtype} is not float32 for {clip_path}"
                    )
                if not torch.isfinite(embedding).all():
                    raise ValueError(
                        f"Embedding contains non-finite values for {clip_path}"
                    )
                if embedding.requires_grad or embedding.grad_fn is not None:
                    raise ValueError(
                        f"Embedding has gradient history for {clip_path}"
                    )

                clip_sha256 = _sha256(clip_path)
                cache_splits[split_name]["embeddings"].append(embedding)
                cache_splits[split_name]["labels"].append(row["label"])
                cache_splits[split_name]["rows"].append(
                    {
                        "path": str(clip_path),
                        "recording_id": row["recording_id"],
                        "label": row["label"],
                        "sha256": clip_sha256,
                    }
                )
                progress_bar.update(1)
    finally:
        progress_bar.close()

    # Stack the per-clip outputs into tensors for each split.
    train_features = torch.stack(cache_splits["train"]["embeddings"])
    train_labels = torch.tensor(
        cache_splits["train"]["labels"], dtype=torch.float32
    ).unsqueeze(1)

    val_features = torch.stack(cache_splits["val"]["embeddings"])
    val_labels = torch.tensor(
        cache_splits["val"]["labels"], dtype=torch.float32
    ).unsqueeze(1)

    metadata = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "model_id": _MODEL_ID,
        "revision": revision,
        "embedding_dim": encoder.embedding_dim,
        "sample_rate": 16000,
        "target_samples": 16000,
        "pooling": "last_hidden_state_mean",
        "feature_extractor_config": encoder.feature_extractor.to_dict(),
        "source_file_hashes": _get_source_hashes(),
        "device": str(device),
        "package_versions": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "librosa": str(librosa.__version__),
            "numpy": str(np.__version__),
        },
        "split_counts": {
            "train": len(train_rows),
            "val": len(val_rows),
        },
    }

    cache = {
        "metadata": metadata,
        "train": {
            "features": train_features,
            "labels": train_labels,
            "rows": cache_splits["train"]["rows"],
        },
        "val": {
            "features": val_features,
            "labels": val_labels,
            "rows": cache_splits["val"]["rows"],
        },
    }

    # Keep the feature, label, and row counts aligned in each split.
    for split in ("train", "val"):
        feature_count = cache[split]["features"].shape[0]
        label_count = cache[split]["labels"].shape[0]
        row_count = len(cache[split]["rows"])
        if not (feature_count == label_count == row_count):
            raise ValueError(
                f"Count mismatch in {split} split: {feature_count} features, "
                f"{label_count} labels, {row_count} rows"
            )

    # Write beside the destination so the final publication can be atomic.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp.{uuid.uuid4().hex}")

    try:
        torch.save(cache, temp_path)

        # Reload the file and compare it with the data that was just saved.
        loaded = torch.load(temp_path, map_location="cpu", weights_only=True)

        if loaded.get("metadata") != cache["metadata"]:
            raise ValueError("Cached metadata does not match original after loading.")

        for split in ("train", "val"):
            if not torch.equal(loaded[split]["features"], cache[split]["features"]):
                raise ValueError(
                    f"Features for '{split}' do not match original after loading."
                )
            if not torch.equal(loaded[split]["labels"], cache[split]["labels"]):
                raise ValueError(
                    f"Labels for '{split}' do not match original after loading."
                )
            if loaded[split]["rows"] != cache[split]["rows"]:
                raise ValueError(
                    f"Rows for '{split}' do not match original after loading."
                )

        # A hard link publishes the complete file without overwriting a new output.
        os.link(temp_path, output_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return output_path


def main() -> None:
    """Parse command-line options and build the feature cache."""
    parser = argparse.ArgumentParser(
        description="Build feature cache using frozen XLS-R speech encoder."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to CSV train/val manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for feature cache .pt file",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="1a640f32ac3e39899438a2931f9924c02f080a54",
        help="Model commit revision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for encoder inference ('cpu', 'cuda', 'mps')",
    )
    parser.add_argument("--download-encoder", action="store_true")
    args = parser.parse_args()

    out = build_feature_cache(
        manifest_path=args.manifest,
        output_path=args.output,
        revision=args.revision,
        device=args.device,
        allow_download=args.download_encoder,
    )
    print(f"Feature cache built successfully: {out}")


if __name__ == "__main__":
    main()
