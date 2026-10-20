"""Offline tests with synthetic inputs/backbone for FrozenSpeechEncoder."""

import unittest
import pytest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from songsored.speech_encoder import FrozenSpeechEncoder, WaveformValidationError


@pytest.mark.usefixtures("stub_speech_backbone")
class SpeechEncoderTests(unittest.TestCase):
    def setUp(self):
        # Test the real wrapper with a small, offline backbone stand-in.
        self.encoder = FrozenSpeechEncoder(
            model_id="facebook/wav2vec2-xls-r-300m",
            device="cpu",
        )

    def test_dimensions_and_attributes(self):
        self.assertEqual(self.encoder.embedding_dim, 1024)
        self.assertEqual(self.encoder.model_id, "facebook/wav2vec2-xls-r-300m")
        self.assertEqual(self.encoder.device, torch.device("cpu"))
        self.assertIsNone(self.encoder.revision)

    def test_parameters_are_frozen(self):
        for name, param in self.encoder.model.named_parameters():
            self.assertFalse(
                param.requires_grad,
                f"Parameter {name} was not frozen (requires_grad is True)",
            )
        self.assertFalse(self.encoder.model.training)

    def test_sampling_rate_mismatch_raises_error(self):
        mock_fe = MagicMock()
        mock_fe.sampling_rate = 8000
        mock_model = MagicMock()
        mock_model.config.hidden_size = 1024

        with patch("songsored.speech_encoder.AutoFeatureExtractor.from_pretrained", return_value=mock_fe), \
             patch("songsored.speech_encoder.Wav2Vec2Model.from_pretrained", return_value=mock_model):
            with self.assertRaises(ValueError) as ctx:
                FrozenSpeechEncoder(model_id="mock-model")
            self.assertIn("16000 Hz is required", str(ctx.exception))

    def test_valid_waveform_encoding(self):
        rng = np.random.RandomState(42)
        waveform = rng.randn(16000).astype(np.float32)

        features = self.encoder.encode(waveform)

        # Output shape and type checks
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape, torch.Size([1024]))
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(features.device.type, "cpu")
        self.assertTrue(torch.isfinite(features).all().item())

        # Gradient history check
        self.assertFalse(features.requires_grad)
        self.assertIsNone(features.grad_fn)

    def test_float16_waveform_regression(self):
        rng = np.random.RandomState(42)
        waveform = rng.randn(16000).astype(np.float16)

        features = self.encoder.encode(waveform)

        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape, torch.Size([1024]))
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(features.device.type, "cpu")
        self.assertTrue(torch.isfinite(features).all().item())

    def test_silence_input_accepted(self):
        silence = np.zeros(16000, dtype=np.float32)
        features = self.encoder.encode(silence)

        self.assertEqual(features.shape, torch.Size([1024]))
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.isfinite(features).all().item())

    def test_repeated_extraction_is_consistent(self):
        rng = np.random.RandomState(1337)
        waveform = rng.randn(16000).astype(np.float32)

        features1 = self.encoder.encode(waveform)
        features2 = self.encoder.encode(waveform)

        self.assertTrue(torch.equal(features1, features2))

    def test_small_head_can_backpropagate(self):
        rng = np.random.RandomState(77)
        waveform = rng.randn(16000).astype(np.float32)
        features = self.encoder.encode(waveform)

        # Classification head: 1024 -> 1
        head = nn.Linear(self.encoder.embedding_dim, 1)
        criterion = nn.BCEWithLogitsLoss()

        logits = head(features.unsqueeze(0))
        target = torch.tensor([[1.0]], dtype=torch.float32)
        loss = criterion(logits, target)
        loss.backward()

        self.assertIsNotNone(head.weight.grad)
        self.assertTrue(torch.isfinite(head.weight.grad).all().item())
        self.assertIsNotNone(head.bias.grad)
        self.assertTrue(torch.isfinite(head.bias.grad).all().item())

        # Confirm encoder parameters remain without gradients
        for param in self.encoder.model.parameters():
            self.assertIsNone(param.grad)

    def test_eval_mode_enforced_during_encode(self):
        # Artificially switch model to train mode
        self.encoder.model.train()
        self.assertTrue(self.encoder.model.training)

        silence = np.zeros(16000, dtype=np.float32)
        _ = self.encoder.encode(silence)

        # Confirm encode switched/kept it in eval mode
        self.assertFalse(self.encoder.model.training)

    def test_invalid_inputs_raise_clear_errors(self):
        # 1. Non-ndarray inputs
        for bad_input in [[0.0] * 16000, torch.zeros(16000), "audio_string"]:
            with self.subTest(bad_input=type(bad_input)):
                with self.assertRaises((ValueError, TypeError)) as ctx:
                    self.encoder.encode(bad_input)
                self.assertIn("numpy.ndarray", str(ctx.exception))

        # 2. Wrong dimension (must be 1D mono)
        with self.subTest("stereo"):
            with self.assertRaises(ValueError) as ctx:
                self.encoder.encode(np.zeros((16000, 2), dtype=np.float32))
            self.assertIn("1D (mono)", str(ctx.exception))

        with self.subTest("batch dimension"):
            with self.assertRaises(ValueError) as ctx:
                self.encoder.encode(np.zeros((1, 16000), dtype=np.float32))
            self.assertIn("1D (mono)", str(ctx.exception))

        # 3. Non-floating dtype
        for bad_dtype in [np.int16, np.int32, np.int64]:
            with self.subTest(bad_dtype=bad_dtype):
                with self.assertRaises((ValueError, TypeError)) as ctx:
                    self.encoder.encode(np.zeros(16000, dtype=bad_dtype))
                self.assertIn("floating-point", str(ctx.exception))

        # 4. Length != 16000 samples
        for bad_len in [0, 8000, 15999, 16001, 32000]:
            with self.subTest(bad_len=bad_len):
                with self.assertRaises(ValueError) as ctx:
                    self.encoder.encode(np.zeros(bad_len, dtype=np.float32))
                self.assertIn("16000 samples", str(ctx.exception))

        # 5. Non-finite values (NaN / Inf)
        with self.subTest("NaN"):
            nan_array = np.zeros(16000, dtype=np.float32)
            nan_array[500] = np.nan
            with self.assertRaises(ValueError) as ctx:
                self.encoder.encode(nan_array)
            self.assertIn("non-finite", str(ctx.exception))

        with self.subTest("Inf"):
            inf_array = np.zeros(16000, dtype=np.float32)
            inf_array[500] = np.inf
            with self.assertRaises(ValueError) as ctx:
                self.encoder.encode(inf_array)
            self.assertIn("non-finite", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
