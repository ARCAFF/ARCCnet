"""Test model forward pass and output shapes."""

import torch

from arccnet.models.timeseries import config as ts_config
from arccnet.models.timeseries.flare_forecaster import FlareForecaster
from arccnet.models.timeseries.spatial_encoder import SpatialEncoder
from arccnet.models.timeseries.temporal_transformer import TemporalTransformer


def test_spatial_encoder_forward():
    """Test spatial encoder forward pass."""
    print("Testing SpatialEncoder forward pass...")

    model = SpatialEncoder(num_channels=10, pretrained=False, feature_dim=512)
    model.eval()

    x = torch.randn(4, 10, 256, 512)  # (B, C, H, W)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, 512), f"Output shape mismatch: {out.shape}"
    print(f"  Input: {x.shape} → Output: {out.shape}")
    print("✓ SpatialEncoder forward test passed!")


def test_temporal_transformer_forward():
    """Test temporal transformer forward pass."""
    print("\nTesting TemporalTransformer forward pass...")

    model = TemporalTransformer(feature_dim=512, num_layers=2, num_heads=8)
    model.eval()

    x = torch.randn(4, 6, 512)  # (B, T, D)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, 512), f"Output shape mismatch: {out.shape}"
    print(f"  Input: {x.shape} → Output: {out.shape}")
    print("✓ TemporalTransformer forward test passed!")


def test_flare_forecaster_forward():
    """Test complete model forward pass."""
    print("\nTesting FlareForecaster forward pass...")
    num_classes = ts_config.NUM_CLASSES

    # Test multiclass model
    model_mc = FlareForecaster(
        task_type="multiclass",
        num_channels=10,
        spatial_feature_dim=512,
        temporal_num_layers=2,
        temporal_num_heads=8,
        output_dim=num_classes,
        pretrained_spatial=False,
        hidden_dims=[128],
    )
    model_mc.eval()

    x = torch.randn(2, 6, 10, 256, 512)  # (B, T, C, H, W)

    with torch.no_grad():
        logits = model_mc(x)
        probs = model_mc.predict_proba(x)

    assert logits.shape == (2, num_classes), f"Logits shape mismatch: {logits.shape}"
    assert probs.shape == (2, num_classes), f"Probs shape mismatch: {probs.shape}"
    assert (probs >= 0).all(), "Probabilities should be >= 0"
    assert (probs <= 1).all(), "Probabilities should be <= 1"
    assert torch.allclose(probs.sum(dim=1), torch.ones(2)), "Probs should sum to 1"

    print("  Multiclass:")
    print(f"    Input: {x.shape}")
    print(f"    Logits: {logits.shape}")
    print(f"    Probs: {probs.shape}, sum: {probs.sum(dim=1)}")

    # Test regression model
    model_reg = FlareForecaster(
        task_type="regression",
        num_channels=10,
        spatial_feature_dim=512,
        temporal_num_layers=2,
        temporal_num_heads=8,
        output_dim=3,
        pretrained_spatial=False,
        hidden_dims=[128],
    )
    model_reg.eval()

    with torch.no_grad():
        preds = model_reg(x)

    assert preds.shape == (2, 3), f"Predictions shape mismatch: {preds.shape}"

    print("  Regression:")
    print(f"    Predictions: {preds.shape}")

    print("✓ FlareForecaster forward test passed!")


def test_model_cuda():
    """Test model on CUDA if available."""
    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available, skipping GPU test")
        return

    print("\nTesting FlareForecaster on CUDA...")

    device = torch.device("cuda")
    model = FlareForecaster(
        task_type="multiclass",
        num_channels=10,
        temporal_num_layers=2,
        output_dim=ts_config.NUM_CLASSES,
        pretrained_spatial=False,
    ).to(device)
    model.eval()

    x = torch.randn(2, 6, 10, 256, 512).to(device)

    with torch.no_grad():
        logits = model(x)

    assert logits.device.type == "cuda", "Output not on CUDA"
    assert logits.shape == (2, ts_config.NUM_CLASSES), f"Shape mismatch: {logits.shape}"

    print(f"  Device: {device}")
    print(f"  Output device: {logits.device}")
    print("✓ CUDA test passed!")


if __name__ == "__main__":
    test_spatial_encoder_forward()
    test_temporal_transformer_forward()
    test_flare_forecaster_forward()
    test_model_cuda()
    print("\n✅ All model tests passed!")
