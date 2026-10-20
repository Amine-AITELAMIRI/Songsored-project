"""Unit tests for SpeechFeatureClassifier."""

import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from songsored.speech_classifier import SpeechFeatureClassifier


class SpeechFeatureClassifierTests(unittest.TestCase):
    def test_default_initialization_and_parameter_count(self):
        model = SpeechFeatureClassifier()
        self.assertEqual(model.embedding_dim, 1024)
        self.assertIsInstance(model.classifier, nn.Linear)
        self.assertEqual(model.classifier.in_features, 1024)
        self.assertEqual(model.classifier.out_features, 1)

        # Default parameter count: 1024 weights + 1 bias = 1025
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 1025)

    def test_invalid_embedding_dim_rejected(self):
        for bad_dim in [0, -1, -1024, 1024.5, "1024", True, False]:
            with self.subTest(bad_dim=bad_dim):
                with self.assertRaises(ValueError):
                    SpeechFeatureClassifier(embedding_dim=bad_dim)

    def test_forward_batch_shape_8(self):
        model = SpeechFeatureClassifier()
        inputs = torch.randn(8, 1024)
        logits = model(inputs)

        self.assertEqual(logits.shape, (8, 1))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_forward_single_example_batch_shape_1(self):
        model = SpeechFeatureClassifier()
        inputs = torch.randn(1, 1024)
        logits = model(inputs)

        # Batch dimension must be preserved
        self.assertEqual(logits.shape, (1, 1))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_custom_embedding_dim(self):
        model = SpeechFeatureClassifier(embedding_dim=32)
        self.assertEqual(model.embedding_dim, 32)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 33)

        inputs = torch.randn(8, 32)
        logits = model(inputs)
        self.assertEqual(logits.shape, (8, 1))

    def test_wrong_rank_or_feature_dimension_raises_clear_error(self):
        model = SpeechFeatureClassifier(embedding_dim=1024)

        # 1D input (rank 1)
        with self.subTest("rank 1"):
            with self.assertRaises(ValueError) as ctx:
                model(torch.randn(1024))
            self.assertIn("Expected features with shape [batch_size, 1024]", str(ctx.exception))
            self.assertIn("(1024,)", str(ctx.exception))

        # 3D input (rank 3)
        with self.subTest("rank 3"):
            with self.assertRaises(ValueError) as ctx:
                model(torch.randn(2, 4, 1024))
            self.assertIn("Expected features with shape [batch_size, 1024]", str(ctx.exception))

        # 2D input with wrong feature dimension (512 instead of 1024)
        with self.subTest("wrong feature dim"):
            with self.assertRaises(ValueError) as ctx:
                model(torch.randn(16, 512))
            self.assertIn("Expected features with shape [batch_size, 1024]", str(ctx.exception))
            self.assertIn("(16, 512)", str(ctx.exception))

        # Non-tensor input
        with self.subTest("non-tensor"):
            with self.assertRaises((TypeError, ValueError)):
                model([[0.0] * 1024])

    def test_bce_loss_and_backward_computes_finite_gradients(self):
        torch.manual_seed(42)
        model = SpeechFeatureClassifier(embedding_dim=1024)
        criterion = nn.BCEWithLogitsLoss()

        features = torch.randn(8, 1024)
        labels = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])

        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()

        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertEqual(model.classifier.weight.grad.shape, (1, 1024))
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all().item())

        self.assertIsNotNone(model.classifier.bias.grad)
        self.assertEqual(model.classifier.bias.grad.shape, (1,))
        self.assertTrue(torch.isfinite(model.classifier.bias.grad).all().item())

    def test_optimizer_step_changes_parameters(self):
        torch.manual_seed(42)
        model = SpeechFeatureClassifier(embedding_dim=1024)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.BCEWithLogitsLoss()

        initial_weight = model.classifier.weight.clone().detach()
        initial_bias = model.classifier.bias.clone().detach()

        features = torch.randn(8, 1024)
        labels = torch.ones(8, 1)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.equal(model.classifier.weight, initial_weight))
        self.assertFalse(torch.equal(model.classifier.bias, initial_bias))


if __name__ == "__main__":
    unittest.main()
