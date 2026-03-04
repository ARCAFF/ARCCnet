"""Tests for timeseries focal loss behavior."""

import torch

from arccnet.models.timeseries.focal_loss import FocalLoss


def test_focal_loss_forward_with_weight_and_alpha_vectors():
    """FocalLoss should accept per-class alpha/weight vectors and return finite loss."""
    loss_fn = FocalLoss(alpha=[1.0, 1.0, 2.0, 4.0], gamma=2.0, weight=[1.0, 1.0, 3.0, 10.0])

    # Ensure tensor parameters are registered as buffers for device moves.
    buffer_names = {name for name, _ in loss_fn.named_buffers()}
    assert "weight" in buffer_names
    assert "alpha_tensor" in buffer_names

    logits = torch.randn(8, 4, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 4, (8,), dtype=torch.long)

    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss), f"Expected finite loss, got {loss}"

    loss.backward()
    assert logits.grad is not None


if __name__ == "__main__":
    test_focal_loss_forward_with_weight_and_alpha_vectors()
    print("\n✅ Focal loss tests passed!")
