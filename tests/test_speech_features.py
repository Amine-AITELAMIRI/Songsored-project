"""Offline tests with synthetic inputs/backbone for load_speech_waveform in speech_features.py."""

from pathlib import Path
import tempfile
import unittest
import pytest

import numpy as np
import soundfile as sf
import torch

from songsored.speech_encoder import FrozenSpeechEncoder
from songsored.speech_features import load_speech_waveform


@pytest.mark.usefixtures("stub_speech_backbone")
class SpeechFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _write_wav(
        self,
        filename: str,
        data: np.ndarray,
        sample_rate: int = 16000,
    ) -> Path:
        path = self.root / filename
        sf.write(str(path), data, sample_rate)
        return path

    def test_nonexistent_file_raises_file_not_found(self):
        nonexistent = self.root / "does_not_exist.wav"
        with self.assertRaises(FileNotFoundError) as ctx:
            load_speech_waveform(nonexistent)
        self.assertIn(str(nonexistent), str(ctx.exception))

    def test_directory_path_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_speech_waveform(self.root)
        self.assertIn(str(self.root), str(ctx.exception))

    def test_exact_16000_sample_audio_loads_cleanly(self):
        rng = np.random.RandomState(42)
        samples = (rng.rand(16000).astype(np.float32) * 2 - 1) * 0.8
        path = self._write_wav("exact.wav", samples, 16000)

        waveform = load_speech_waveform(path)

        self.assertIsInstance(waveform, np.ndarray)
        self.assertEqual(waveform.shape, (16000,))
        self.assertEqual(waveform.dtype, np.float32)
        self.assertTrue(np.isfinite(waveform).all())
        # sf.write/read can have tiny float precision variations
        self.assertTrue(np.allclose(waveform, samples, atol=1e-3))

    def test_resampling_from_different_sample_rate(self):
        # 1.0 second at 22050 Hz
        rng = np.random.RandomState(123)
        samples = (rng.rand(22050).astype(np.float32) * 2 - 1) * 0.8
        path = self._write_wav("resample.wav", samples, 22050)

        waveform = load_speech_waveform(path)

        self.assertEqual(waveform.shape, (16000,))
        self.assertEqual(waveform.dtype, np.float32)
        self.assertTrue(np.isfinite(waveform).all())

    def test_shorter_audio_is_right_padded_with_zeros(self):
        # 0.5 seconds at 16 kHz = 8000 samples
        rng = np.random.RandomState(99)
        half_sec = (rng.rand(8000).astype(np.float32) * 2 - 1) * 0.7
        path = self._write_wav("short.wav", half_sec, 16000)

        waveform = load_speech_waveform(path)

        self.assertEqual(waveform.shape, (16000,))
        self.assertEqual(waveform.dtype, np.float32)
        # First 8000 samples match original
        self.assertTrue(np.allclose(waveform[:8000], half_sec, atol=1e-3))
        # Remaining 8000 samples must be exactly 0.0
        self.assertTrue(np.all(waveform[8000:] == 0.0))

    def test_oversized_audio_raises_value_error_without_truncating(self):
        # 1.25 seconds at 16 kHz = 20000 samples
        oversized = np.zeros(20000, dtype=np.float32)
        path = self._write_wav("oversized.wav", oversized, 16000)

        with self.assertRaises(ValueError) as ctx:
            load_speech_waveform(path)

        err_msg = str(ctx.exception)
        self.assertIn(str(path), err_msg)
        self.assertIn("16000", err_msg)

    def test_silence_audio_accepted(self):
        silence = np.zeros(16000, dtype=np.float32)
        path = self._write_wav("silence.wav", silence, 16000)

        waveform = load_speech_waveform(path)

        self.assertEqual(waveform.shape, (16000,))
        self.assertEqual(waveform.dtype, np.float32)
        self.assertTrue(np.all(waveform == 0.0))

    def test_corrupt_file_raises_value_error_preserving_cause(self):
        corrupt_path = self.root / "corrupt.wav"
        corrupt_path.write_bytes(b"not a valid audio file header")

        with self.assertRaises(ValueError) as ctx:
            load_speech_waveform(corrupt_path)

        self.assertIn(str(corrupt_path), str(ctx.exception))
        self.assertIsNotNone(ctx.exception.__cause__)

    def test_empty_audio_file_raises_value_error(self):
        empty_path = self.root / "empty.wav"
        empty_path.touch()

        with self.assertRaises(ValueError) as ctx:
            load_speech_waveform(empty_path)

        self.assertIn(str(empty_path), str(ctx.exception))

    def test_integration_with_frozen_speech_encoder(self):
        rng = np.random.RandomState(7)
        samples = (rng.rand(12000).astype(np.float32) * 2 - 1) * 0.5
        path = self._write_wav("clip.wav", samples, 16000)

        waveform = load_speech_waveform(path)
        encoder = FrozenSpeechEncoder(device="cpu")
        embedding = encoder.encode(waveform)

        self.assertEqual(embedding.shape, (1024,))
        self.assertEqual(embedding.dtype, torch.float32)
        self.assertTrue(torch.isfinite(embedding).all().item())


if __name__ == "__main__":
    import torch
    unittest.main()
