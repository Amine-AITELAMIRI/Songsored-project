"""Dataset for loading labeled audio clips from a CSV manifest."""

from __future__ import annotations

import csv
from pathlib import Path

import librosa
from librosa.util.exceptions import ParameterError
import numpy as np
import torch
from torch.utils.data import Dataset

from .features import extract_mel_spectrogram

_REQUIRED_COLUMNS = {"path", "label", "recording_id", "split"}
_VALID_SPLITS = {"train", "val", "test"}
_VALID_LABELS = {"0", "1"}


class AudioClipDataset(Dataset):
    """Load audio clips and labels from a CSV manifest.

    Each row needs a clip ``path``, a binary ``label``, a ``recording_id``,
    and a ``split`` (``train``, ``val``, or ``test``). Paths are resolved
    relative to the manifest. Clips from one recording must stay in the same
    split.

    Explicitly authorized automatic labels are provisional training data.
    Keep their provenance upstream, and use reviewed labels for validation
    and test.

    Args:
        manifest_path: Path to the CSV manifest file.
        split: Which split to load (``"train"``, ``"val"``, or ``"test"``).
        sample_rate: Target sample rate for loaded audio (Hz).
        clip_duration: Duration of each clip in seconds.
        transform: Optional callable applied to the feature tensor.
    """

    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = 0

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        sample_rate: int = 22050,
        clip_duration: float = 1.0,
        transform=None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"Invalid split {split!r}. Must be one of {sorted(_VALID_SPLITS)}."
            )

        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.transform = transform
        self.samples: list[dict[str, Path | int | str]] = []

        manifest_dir = self.manifest_path.parent

        # Read the manifest and check its header.
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        with self.manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            reader = csv.DictReader(manifest_file)

            if reader.fieldnames is None:
                raise ValueError("Manifest CSV has no header row.")
            # DictReader would overwrite values under duplicate names; reject them.
            columns: set[str] = set()
            for column_name in reader.fieldnames:
                if column_name in columns:
                    raise ValueError(
                        f"Manifest has a duplicate column header: {column_name!r}. "
                        "Each column name must be unique."
                    )
                columns.add(column_name)
            missing = _REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(
                    f"Manifest is missing required column(s): {sorted(missing)}"
                )

            all_rows = list(reader)

        # Check each row's required values.
        for row_number, row in enumerate(all_rows, start=2):
            # Ensure no required field is missing or blank.
            for column_name in _REQUIRED_COLUMNS:
                if not (row.get(column_name) or "").strip():
                    raise ValueError(
                        f"Row {row_number}: required field {column_name!r} is missing or empty."
                    )
            if row["label"].strip() not in _VALID_LABELS:
                raise ValueError(
                    f"Row {row_number}: label must be 0 or 1, got {row['label']!r}."
                )
            if row["split"].strip() not in _VALID_SPLITS:
                raise ValueError(
                    f"Row {row_number}: split must be one of {sorted(_VALID_SPLITS)}, "
                    f"got {row['split']!r}."
                )

        for row in all_rows:
            row["recording_id"] = row["recording_id"].strip()

        # Keep every recording in a single split.
        recording_split: dict[str, str] = {}
        for row_number, row in enumerate(all_rows, start=2):
            recording_id = row["recording_id"]
            row_split = row["split"].strip()
            if (
                recording_id in recording_split
                and recording_split[recording_id] != row_split
            ):
                raise ValueError(
                    f"Row {row_number}: recording_id {recording_id!r} appears in both "
                    f"{recording_split[recording_id]!r} and {row_split!r} splits. "
                    "Each recording must belong to exactly one split."
                )
            recording_split[recording_id] = row_split

        # Keep this split's clips and resolve their paths.
        for row in all_rows:
            if row["split"].strip() != split:
                continue

            clip_path = (manifest_dir / row["path"].strip()).resolve()
            if not clip_path.is_file():
                raise FileNotFoundError(
                    f"Clip file not found: {clip_path} "
                    f"(from manifest entry {row['path']!r})"
                )

            self.samples.append(
                {
                    "path": clip_path,
                    "label": int(row["label"].strip()),
                    "recording_id": row["recording_id"],  # already normalized
                }
            )

        # Require at least one clip in the requested split.
        if not self.samples:
            raise ValueError(
                f"No samples found for split {split!r} in {self.manifest_path}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the clip's mel features and binary label.

        Returns:
            A feature tensor shaped (1, n_mels, time_frames) and a float label
            tensor shaped (1,). The label shape matches the CNN output for
            BCEWithLogitsLoss; the DataLoader adds the batch dimension.
        """
        sample = self.samples[idx]
        path: Path = sample["path"]  # type: ignore[assignment]

        # Load and validate the audio.
        try:
            audio, _ = librosa.load(
                path,
                sr=self.sample_rate,
                mono=True,
                dtype=np.float32,
            )
        except ParameterError as exc:
            raise ValueError(
                f"librosa could not load audio file: {path}"
            ) from exc

        if audio.size == 0:
            raise ValueError(f"Audio file is empty: {path}")
        if not np.isfinite(audio).all():
            raise ValueError(
                f"Audio contains NaN or infinite values: {path}"
            )

        target_length = round(self.sample_rate * self.clip_duration)
        if target_length <= 0:
            raise ValueError(
                f"target_length must be positive, got {target_length} "
                f"(sample_rate={self.sample_rate}, clip_duration={self.clip_duration})"
            )

        # Pad short clips and reject long ones.
        audio_length = len(audio)
        if audio_length < target_length:
            audio = np.pad(audio, (0, target_length - audio_length))
        elif audio_length > target_length:
            raise ValueError(
                f"Clip is longer than the configured duration "
                f"({audio_length} samples > {target_length} at {self.sample_rate} Hz): {path}. "
                "Trim using verified word boundaries before including in the manifest."
            )

        # Extract features and add the channel dimension.
        mel = extract_mel_spectrogram(audio, sample_rate=self.sample_rate)
        feature_tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, T)

        if self.transform is not None:
            feature_tensor = self.transform(feature_tensor)

        label_tensor = torch.tensor([sample["label"]], dtype=torch.float32)
        return feature_tensor, label_tensor
