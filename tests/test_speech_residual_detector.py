"""Check temporal and pooled outputs with generated features."""

import pytest
import torch

from songsored.speech_residual_detector import ResidualSpeechDetector


def test_pooled_control_matches_masked_average_of_frame_logits():
    torch.manual_seed(42)
    model = ResidualSpeechDetector()
    features = torch.randn(2, 199, 1024)
    mask = torch.zeros(2, 199, dtype=torch.bool)
    mask[0, 50:150] = True
    mask[1, 60:140:2] = True

    frames = model(features, mask)
    pooled = model(features, mask, pooled=True)
    assert frames.shape == (2, 199)
    assert pooled.shape == (2,)
    expected = (frames * mask).sum(1) / mask.sum(1)
    torch.testing.assert_close(pooled, expected)

    pooled.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_mask_selects_pooling_frames_without_erasing_temporal_context():
    torch.manual_seed(42)
    model = ResidualSpeechDetector(embedding_dim=8, channels=8)
    features = torch.randn(1, 40, 8)
    full_mask = torch.ones(1, 40, dtype=torch.bool)
    center_mask = torch.zeros_like(full_mask)
    center_mask[:, 15:25] = True

    full = model(features, full_mask)
    center = model(features, center_mask)
    torch.testing.assert_close(center, full, rtol=0, atol=0)
    pooled = model(features, center_mask, pooled=True)
    torch.testing.assert_close(pooled, full[:, 15:25].mean(1))


def test_shifting_features_shifts_interior_scores():
    torch.manual_seed(42)
    model = ResidualSpeechDetector(embedding_dim=8, channels=8)
    features = torch.randn(1, 50, 8)
    mask = torch.ones(1, 50, dtype=torch.bool)
    shifted = torch.cat([torch.zeros(1, 3, 8), features[:, :-3]], dim=1)

    original_scores = model(features, mask)
    shifted_scores = model(shifted, mask)
    # Each output uses seven neighboring frames on each side. Compare away
    # from the recording boundaries where padding changes the context.
    torch.testing.assert_close(shifted_scores[:, 10:43], original_scores[:, 7:40])


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_rejects_invalid_dimensions(size):
    with pytest.raises(ValueError):
        ResidualSpeechDetector(embedding_dim=size)
    with pytest.raises(ValueError):
        ResidualSpeechDetector(channels=size)


@pytest.mark.parametrize("problem", ["features", "mask_shape", "mask_type", "empty_row"])
def test_rejects_invalid_inputs(problem):
    model = ResidualSpeechDetector(embedding_dim=8, channels=8)
    features = torch.randn(2, 20, 8)
    mask = torch.ones(2, 20, dtype=torch.bool)
    if problem == "features":
        features = features[:, :, :7]
    elif problem == "mask_shape":
        mask = mask[:, :19]
    elif problem == "mask_type":
        mask = mask.float()
    else:
        mask[1] = False
    with pytest.raises(ValueError):
        model(features, mask)
