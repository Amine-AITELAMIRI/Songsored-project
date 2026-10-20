"""Offline tests with synthetic inputs/backbone for build_speech_feature_cache.py."""

import csv
from pathlib import Path
import tempfile
import unittest
import pytest

import numpy as np
import soundfile as sf
import torch

from songsored.build_speech_feature_cache import (
    build_feature_cache,
    validate_manifest,
)


@pytest.mark.usefixtures("stub_speech_backbone")
class BuildSpeechFeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()
        self.clips_dir = self.root / "clips"
        self.clips_dir.mkdir()

    def _create_clip(self, filename: str, duration_samples: int = 16000) -> Path:
        clip_path = self.clips_dir / filename
        data = np.zeros(duration_samples, dtype=np.float32)
        sf.write(str(clip_path), data, 16000)
        return clip_path

    def _write_manifest(self, rows: list[dict], filename: str = "manifest.csv") -> Path:
        manifest_path = self.root / filename
        fieldnames = ["path", "label", "recording_id", "split"]
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return manifest_path

    def test_validate_manifest_success(self):
        c1 = self._create_clip("clip1.wav")
        c2 = self._create_clip("clip2.wav")
        c3 = self._create_clip("clip3.wav")

        rows = [
            {"path": "clips/clip1.wav", "label": "1", "recording_id": "rec1", "split": "train"},
            {"path": "clips/clip2.wav", "label": "0", "recording_id": "rec1", "split": "train"},
            {"path": "clips/clip3.wav", "label": "1", "recording_id": "rec2", "split": "val"},
        ]
        manifest_path = self._write_manifest(rows)

        train_rows, val_rows = validate_manifest(manifest_path)
        self.assertEqual(len(train_rows), 2)
        self.assertEqual(len(val_rows), 1)
        self.assertEqual(train_rows[0]["path"], c1)
        self.assertEqual(train_rows[1]["path"], c2)
        self.assertEqual(val_rows[0]["path"], c3)

    def test_missing_manifest_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            validate_manifest(self.root / "missing.csv")

    def test_test_split_rejected(self):
        self._create_clip("c1.wav")
        self._create_clip("c2.wav")
        rows = [
            {"path": "clips/c1.wav", "label": "1", "recording_id": "rec1", "split": "train"},
            {"path": "clips/c2.wav", "label": "0", "recording_id": "rec2", "split": "test"},
        ]
        manifest_path = self._write_manifest(rows)
        with self.assertRaises(ValueError) as ctx:
            validate_manifest(manifest_path)
        self.assertIn("test", str(ctx.exception))

    def test_invalid_label_rejected(self):
        self._create_clip("c1.wav")
        rows = [
            {"path": "clips/c1.wav", "label": "2", "recording_id": "rec1", "split": "train"},
        ]
        manifest_path = self._write_manifest(rows)
        with self.assertRaises(ValueError) as ctx:
            validate_manifest(manifest_path)
        self.assertIn("label must be '0' or '1'", str(ctx.exception))

    def test_recording_in_multiple_splits_rejected(self):
        self._create_clip("c1.wav")
        self._create_clip("c2.wav")
        rows = [
            {"path": "clips/c1.wav", "label": "1", "recording_id": "rec1", "split": "train"},
            {"path": "clips/c2.wav", "label": "0", "recording_id": "rec1", "split": "val"},
        ]
        manifest_path = self._write_manifest(rows)
        with self.assertRaises(ValueError) as ctx:
            validate_manifest(manifest_path)
        self.assertIn("both 'train' and 'val'", str(ctx.exception))

    def test_missing_clip_file_raises_file_not_found(self):
        rows = [
            {"path": "clips/missing.wav", "label": "1", "recording_id": "rec1", "split": "train"},
            {"path": "clips/missing2.wav", "label": "0", "recording_id": "rec2", "split": "val"},
        ]
        manifest_path = self._write_manifest(rows)
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_manifest(manifest_path)
        self.assertIn("missing.wav", str(ctx.exception))

    def test_duplicate_clip_path_rejected(self):
        self._create_clip("c1.wav")
        rows = [
            {"path": "clips/c1.wav", "label": "1", "recording_id": "rec1", "split": "train"},
            {"path": "clips/c1.wav", "label": "0", "recording_id": "rec1", "split": "train"},
        ]
        manifest_path = self._write_manifest(rows)
        with self.assertRaises(ValueError) as ctx:
            validate_manifest(manifest_path)
        self.assertIn("duplicate clip path", str(ctx.exception))

    def test_empty_split_rejected(self):
        self._create_clip("c1.wav")
        rows = [
            {"path": "clips/c1.wav", "label": "1", "recording_id": "rec1", "split": "train"},
        ]
        manifest_path = self._write_manifest(rows)
        with self.assertRaises(ValueError) as ctx:
            validate_manifest(manifest_path)
        self.assertIn("validation examples", str(ctx.exception))

    def test_build_feature_cache_end_to_end(self):
        c1 = self._create_clip("clip1.wav")
        c2 = self._create_clip("clip2.wav")
        c3 = self._create_clip("clip3.wav")

        rows = [
            {"path": "clips/clip1.wav", "label": "1", "recording_id": "rec1", "split": "train"},
            {"path": "clips/clip2.wav", "label": "0", "recording_id": "rec1", "split": "train"},
            {"path": "clips/clip3.wav", "label": "1", "recording_id": "rec2", "split": "val"},
        ]
        manifest_path = self._write_manifest(rows)
        output_path = self.root / "cache.pt"

        # Output file exists check
        output_path.touch()
        with self.assertRaises(FileExistsError):
            build_feature_cache(
                manifest_path=manifest_path,
                output_path=output_path,
                revision="1a640f32ac3e39899438a2931f9924c02f080a54",
                device="cpu",
            )
        output_path.unlink()

        # Run extraction
        result_path = build_feature_cache(
            manifest_path=manifest_path,
            output_path=output_path,
            revision="1a640f32ac3e39899438a2931f9924c02f080a54",
            device="cpu",
        )
        self.assertEqual(result_path, output_path)
        self.assertTrue(output_path.is_file())

        # Load back cache and verify contents
        cache = torch.load(output_path, map_location="cpu", weights_only=True)

        self.assertIn("metadata", cache)
        self.assertIn("train", cache)
        self.assertIn("val", cache)

        # Check metadata
        meta = cache["metadata"]
        self.assertEqual(meta["schema_version"], 1)
        self.assertEqual(meta["manifest_path"], str(manifest_path.resolve()))
        self.assertIn("manifest_sha256", meta)
        self.assertEqual(meta["model_id"], "facebook/wav2vec2-xls-r-300m")
        self.assertEqual(meta["revision"], "1a640f32ac3e39899438a2931f9924c02f080a54")
        self.assertEqual(meta["embedding_dim"], 1024)
        self.assertEqual(meta["sample_rate"], 16000)
        self.assertEqual(meta["target_samples"], 16000)
        self.assertEqual(meta["pooling"], "last_hidden_state_mean")
        self.assertIn("speech_encoder", meta["source_file_hashes"])
        self.assertIn("speech_features", meta["source_file_hashes"])
        self.assertEqual(meta["split_counts"], {"train": 2, "val": 1})

        # Check train split
        train_features = cache["train"]["features"]
        train_labels = cache["train"]["labels"]
        train_rows_cached = cache["train"]["rows"]
        self.assertEqual(train_features.shape, (2, 1024))
        self.assertEqual(train_features.dtype, torch.float32)
        self.assertEqual(train_labels.shape, (2, 1))
        self.assertEqual(train_labels.dtype, torch.float32)
        self.assertEqual(len(train_rows_cached), 2)
        self.assertEqual(train_rows_cached[0]["path"], str(c1.resolve()))
        self.assertEqual(train_rows_cached[0]["label"], 1)
        self.assertEqual(train_rows_cached[0]["recording_id"], "rec1")
        self.assertIn("sha256", train_rows_cached[0])

        # Check val split
        val_features = cache["val"]["features"]
        val_labels = cache["val"]["labels"]
        val_rows_cached = cache["val"]["rows"]
        self.assertEqual(val_features.shape, (1, 1024))
        self.assertEqual(val_features.dtype, torch.float32)
        self.assertEqual(val_labels.shape, (1, 1))
        self.assertEqual(len(val_rows_cached), 1)
        self.assertEqual(val_rows_cached[0]["path"], str(c3.resolve()))
        self.assertEqual(val_rows_cached[0]["label"], 1)

    def test_immutable_revision_enforced(self):
        c1 = self._create_clip("c1.wav")
        c2 = self._create_clip("c2.wav")
        rows = [
            {"path": "clips/c1.wav", "label": "1", "recording_id": "rec1", "split": "train"},
            {"path": "clips/c2.wav", "label": "0", "recording_id": "rec2", "split": "val"},
        ]
        manifest_path = self._write_manifest(rows)
        output_path = self.root / "cache_bad_rev.pt"

        for bad_rev in ["main", "v1", "1a640f", "z" * 40, "", "1a640f32ac3e39899438a2931f9924c02f080a5"]:
            with self.subTest(bad_rev=bad_rev):
                with self.assertRaises(ValueError) as ctx:
                    build_feature_cache(
                        manifest_path=manifest_path,
                        output_path=output_path,
                        revision=bad_rev,
                    )
                self.assertIn("40-character hexadecimal commit hash", str(ctx.exception))

    def test_atomic_publication_fails_if_destination_created_during_extraction(self):
        self._create_clip("c1.wav")
        self._create_clip("c2.wav")
        rows = [
            {"path": "clips/c1.wav", "label": "1", "recording_id": "rec1", "split": "train"},
            {"path": "clips/c2.wav", "label": "0", "recording_id": "rec2", "split": "val"},
        ]
        manifest_path = self._write_manifest(rows)
        output_path = self.root / "race_cache.pt"

        from unittest.mock import patch
        from songsored.speech_encoder import FrozenSpeechEncoder

        original_encode = FrozenSpeechEncoder.encode

        def competing_encode(self_enc, waveform):
            # Simulate a competing process creating output_path during extraction
            if not output_path.exists():
                output_path.write_bytes(b"competing cache content")
            return original_encode(self_enc, waveform)

        with patch.object(FrozenSpeechEncoder, "encode", side_effect=competing_encode, autospec=True):
            with self.assertRaises(FileExistsError):
                build_feature_cache(
                    manifest_path=manifest_path,
                    output_path=output_path,
                    revision="1a640f32ac3e39899438a2931f9924c02f080a54",
                )

        # Confirm competing file was NOT overwritten
        self.assertEqual(output_path.read_bytes(), b"competing cache content")
        # Confirm no leftover temp files remain
        leftover_tmps = list(self.root.glob(".race_cache.pt.tmp.*"))
        self.assertEqual(leftover_tmps, [])


if __name__ == "__main__":
    unittest.main()
