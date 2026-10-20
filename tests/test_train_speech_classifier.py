"""Unit tests for train_speech_classifier.py."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from songsored.train_speech_classifier import (
    BatchStream,
    fit_embedding_stats,
    train_speech_classifier,
    validate_cache,
)
from songsored.speech_classifier import SpeechFeatureClassifier


class TrainSpeechClassifierTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()

    def _create_synthetic_cache(
        self,
        n_train: int = 25,
        n_val: int = 5,
        embedding_dim: int = 16,
    ) -> tuple[dict, Path]:
        generator = torch.Generator().manual_seed(123)
        train_features = torch.randn(n_train, embedding_dim, generator=generator)
        train_labels = (torch.rand(n_train, 1, generator=generator) > 0.5).to(torch.float32)

        val_features = torch.randn(n_val, embedding_dim, generator=generator)
        val_labels = (torch.rand(n_val, 1, generator=generator) > 0.5).to(torch.float32)

        train_rows = [
            {
                "path": str(self.root / f"train_clip_{i}.wav"),
                "recording_id": f"rec_train_{i // 5}",
                "label": int(train_labels[i, 0].item()),
                "sha256": f"hash_train_{i}",
            }
            for i in range(n_train)
        ]
        val_rows = [
            {
                "path": str(self.root / f"val_clip_{i}.wav"),
                "recording_id": f"rec_val_{i}",
                "label": int(val_labels[i, 0].item()),
                "sha256": f"hash_val_{i}",
            }
            for i in range(n_val)
        ]

        cache = {
            "metadata": {
                "schema_version": 1,
                "manifest_path": str(self.root / "manifest.csv"),
                "manifest_sha256": "dummy_manifest_hash",
                "model_id": "facebook/wav2vec2-xls-r-300m",
                "revision": "1a640f32ac3e39899438a2931f9924c02f080a54",
                "embedding_dim": embedding_dim,
                "sample_rate": 16000,
                "target_samples": 16000,
                "pooling": "last_hidden_state_mean",
                "split_counts": {"train": n_train, "val": n_val},
            },
            "train": {
                "features": train_features,
                "labels": train_labels,
                "rows": train_rows,
            },
            "val": {
                "features": val_features,
                "labels": val_labels,
                "rows": val_rows,
            },
        }

        cache_path = self.root / "test_cache.pt"
        torch.save(cache, cache_path)
        return cache, cache_path

    def test_batch_stream_reproducible_and_carries_leftovers(self):
        x = torch.arange(25).unsqueeze(1).float()
        y = torch.zeros(25, 1)

        stream1 = BatchStream(x, y, batch_size=10, seed=42)
        stream2 = BatchStream(x, y, batch_size=10, seed=42)

        # Batch 1 (10 elements)
        b1_x, _ = stream1.next_batch()
        b2_x, _ = stream2.next_batch()
        self.assertTrue(torch.equal(b1_x, b2_x))
        self.assertEqual(len(b1_x), 10)
        self.assertEqual(len(stream1.pending), 15)  # 25 - 10 = 15

        # Batch 2 (10 elements)
        b1_x, _ = stream1.next_batch()
        b2_x, _ = stream2.next_batch()
        self.assertTrue(torch.equal(b1_x, b2_x))
        self.assertEqual(len(stream1.pending), 5)  # 15 - 10 = 5

        # Batch 3 (5 leftover + 5 from next permutation = 10 elements)
        b1_x, _ = stream1.next_batch()
        b2_x, _ = stream2.next_batch()
        self.assertTrue(torch.equal(b1_x, b2_x))
        self.assertEqual(len(b1_x), 10)
        self.assertEqual(len(stream1.pending), 20)  # 5 + 25 - 10 = 20

    def test_fit_embedding_stats_training_only(self):
        torch.manual_seed(99)
        train_features = torch.randn(50, 8)
        val_features = torch.randn(10, 8) * 100.0  # Wildly different scale

        stats1 = fit_embedding_stats(train_features)
        self.assertEqual(stats1["mean"].shape, (8,))
        self.assertEqual(stats1["scale"].shape, (8,))
        self.assertEqual(stats1["mean"].dtype, torch.float32)
        self.assertEqual(stats1["scale"].dtype, torch.float32)

        # Modifying val_features has zero effect on fit_embedding_stats
        val_features_modified = val_features * 5.0
        stats2 = fit_embedding_stats(train_features)
        self.assertTrue(torch.equal(stats1["mean"], stats2["mean"]))
        self.assertTrue(torch.equal(stats1["scale"], stats2["scale"]))

    def test_fit_embedding_stats_zero_variance_clamped_to_eps(self):
        constant_features = torch.zeros(10, 4)
        stats = fit_embedding_stats(constant_features)
        self.assertTrue(torch.all(stats["scale"] >= 1e-6))
        self.assertTrue(torch.allclose(stats["scale"], torch.tensor([1e-6] * 4)))

    def test_validate_cache_rejects_corrupted_caches(self):
        cache, _ = self._create_synthetic_cache()

        # Missing key
        bad_keys = dict(cache)
        del bad_keys["val"]
        with self.assertRaises(ValueError):
            validate_cache(bad_keys)

        # Non-float32 features
        bad_features = dict(cache)
        bad_features["train"] = dict(cache["train"])
        bad_features["train"]["features"] = cache["train"]["features"].double()
        with self.assertRaises(ValueError):
            validate_cache(bad_features)

        # Shared recording IDs between train and val
        shared_rids = dict(cache)
        shared_rids["val"] = dict(cache["val"])
        shared_rids["val"]["rows"] = [
            dict(r, recording_id=cache["train"]["rows"][0]["recording_id"])
            for r in cache["val"]["rows"]
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_cache(shared_rids)
        self.assertIn("share recording IDs", str(ctx.exception))

    def test_validate_cache_requires_split_counts_dictionary_and_entries(self):
        cache, _ = self._create_synthetic_cache()

        # Missing split_counts dictionary entirely
        no_split_counts = dict(cache)
        no_split_counts["metadata"] = dict(cache["metadata"])
        del no_split_counts["metadata"]["split_counts"]
        with self.assertRaises(ValueError) as ctx:
            validate_cache(no_split_counts)
        self.assertIn("split_counts", str(ctx.exception))

        # split_counts is not a dictionary
        bad_type = dict(cache)
        bad_type["metadata"] = dict(cache["metadata"])
        bad_type["metadata"]["split_counts"] = [25, 5]
        with self.assertRaises(ValueError) as ctx:
            validate_cache(bad_type)
        self.assertIn("split_counts", str(ctx.exception))

        # Missing 'val' split entry
        missing_val = dict(cache)
        missing_val["metadata"] = dict(cache["metadata"])
        missing_val["metadata"]["split_counts"] = {"train": 25}
        with self.assertRaises(ValueError) as ctx:
            validate_cache(missing_val)
        self.assertIn("'val'", str(ctx.exception))

        # Missing 'train' split entry
        missing_train = dict(cache)
        missing_train["metadata"] = dict(cache["metadata"])
        missing_train["metadata"]["split_counts"] = {"val": 5}
        with self.assertRaises(ValueError) as ctx:
            validate_cache(missing_train)
        self.assertIn("'train'", str(ctx.exception))

        # Non-positive integer count
        for bad_count in [0, -5, 25.5, True]:
            with self.subTest(bad_count=bad_count):
                invalid_count = dict(cache)
                invalid_count["metadata"] = dict(cache["metadata"])
                invalid_count["metadata"]["split_counts"] = {"train": bad_count, "val": 5}
                with self.assertRaises(ValueError) as ctx:
                    validate_cache(invalid_count)
                self.assertIn("positive integer", str(ctx.exception))

        # Count mismatch with tensors
        mismatched_count = dict(cache)
        mismatched_count["metadata"] = dict(cache["metadata"])
        mismatched_count["metadata"]["split_counts"] = {"train": 999, "val": 5}
        with self.assertRaises(ValueError) as ctx:
            validate_cache(mismatched_count)
        self.assertIn("metadata count (999) does not match feature count (25)", str(ctx.exception))

    def test_existing_run_dir_rejected(self):
        _, cache_path = self._create_synthetic_cache()
        run_dir = self.root / "existing_run"
        run_dir.mkdir()

        with self.assertRaises(FileExistsError):
            train_speech_classifier(
                cache_path=cache_path,
                run_dir=run_dir,
                max_steps=10,
                validation_interval=5,
            )

    def test_tiny_synthetic_cache_training_end_to_end(self):
        cache, cache_path = self._create_synthetic_cache(n_train=25, n_val=5, embedding_dim=16)
        initial_cache_bytes = cache_path.read_bytes()
        initial_train_features = cache["train"]["features"].clone()

        run_dir = self.root / "test_run"
        best_path = train_speech_classifier(
            cache_path=cache_path,
            run_dir=run_dir,
            seed=42,
            max_steps=15,
            batch_size=8,
            validation_interval=5,
            learning_rate=0.01,
            device="cpu",
        )

        self.assertEqual(best_path, (run_dir / "best.pt").resolve())
        self.assertTrue(best_path.is_file())
        self.assertTrue((run_dir / "last.pt").is_file())
        self.assertTrue((run_dir / "history.json").is_file())

        # Check cache file and tensors remained completely untouched
        self.assertEqual(cache_path.read_bytes(), initial_cache_bytes)
        reloaded_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.assertTrue(torch.equal(reloaded_cache["train"]["features"], initial_train_features))

        # Check history
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        self.assertEqual(len(history["steps"]), 3)  # Steps 5, 10, 15
        self.assertEqual([r["step"] for r in history["steps"]], [5, 10, 15])
        for r in history["steps"]:
            self.assertTrue(math.isfinite(r["train_loss"]))
            self.assertTrue(math.isfinite(r["val_loss"]))

        # Verify best.pt corresponds to lowest recorded validation loss
        min_val_loss = min(r["val_loss"] for r in history["steps"])
        self.assertEqual(history["best_val_loss"], min_val_loss)

        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        self.assertEqual(best_ckpt["val_loss"], min_val_loss)
        self.assertEqual(best_ckpt["embedding_dim"], 16)
        self.assertIn("mean", best_ckpt["stats"])
        self.assertIn("scale", best_ckpt["stats"])
        self.assertEqual(best_ckpt["stats"]["mean"].shape, (16,))
        self.assertEqual(best_ckpt["stats"]["scale"].shape, (16,))
        self.assertIn("train_speech_classifier", best_ckpt["source_file_hashes"])
        self.assertIn("speech_classifier", best_ckpt["source_file_hashes"])

        # Check only head parameters exist in checkpoint
        self.assertEqual(set(best_ckpt["model_state_dict"].keys()), {"classifier.weight", "classifier.bias"})

        # Reloading best checkpoint produces deterministic predictions
        model = SpeechFeatureClassifier(embedding_dim=16)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()

        dummy_input = torch.randn(4, 16)
        with torch.no_grad():
            out1 = model(dummy_input)
            out2 = model(dummy_input)
        self.assertTrue(torch.equal(out1, out2))


if __name__ == "__main__":
    unittest.main()
