"""Test data loading and shape validation."""

import pandas as pd

from arccnet.models.timeseries.dataset import SDOTimeseriesDataset


def test_dataset_shapes():
    """Test that dataset returns correct shapes."""
    print("Testing SDOTimeseriesDataset shapes...")

    # Create minimal mock manifest
    mock_manifest = pd.DataFrame(
        {
            "sample_id": ["test_sample"],
            "sample_path": ["/fake/path"],
            "noaa_ar": [12345],
            "date": ["2011-01-01"],
            "hale_class": ["Beta"],
            "mcintosh": ["Dso"],
            "num_timesteps": [6],
            "timestamps": [["t0", "t1", "t2", "t3", "t4", "t5"]],
            "paths": [[[None] * 10] * 6],  # 6 timesteps, 10 channels
            "xb": [0],
            "mb": [0],
            "cb": [0],
            "xa": [0],
            "ma": [0],
            "ca": [1],
            "c_plus": [1],
            "m_plus": [0],
            "x_plus": [0],
            "flare_class": [1],  # C-class (0=No-flare, 1=C, 2=M+)
            "log_ca": [0.301],
            "log_ma": [0.0],
            "log_xa": [0.0],
        }
    )

    # Test multiclass dataset
    dataset_multiclass = SDOTimeseriesDataset(
        mock_manifest,
        split="test",
        task_type="multiclass",
        resize=(256, 512),
        augment=False,
    )

    assert len(dataset_multiclass) == 1, f"Length mismatch: {len(dataset_multiclass)}"

    sample = dataset_multiclass[0]
    assert "x" in sample
    assert "y" in sample
    assert "mask" in sample
    assert "meta" in sample

    x = sample["x"]
    y = sample["y"]
    mask = sample["mask"]

    print(f"  Multiclass - x: {x.shape}, mask: {mask.shape}, y: {y.shape} (scalar class label)")
    assert x.shape == (6, 10, 256, 512), f"x shape mismatch: {x.shape}"
    assert mask.shape == (6,), f"mask shape mismatch: {mask.shape}"
    assert y.shape == (), f"y should be scalar for multiclass, got {y.shape}"

    # Test regression dataset
    dataset_regression = SDOTimeseriesDataset(
        mock_manifest,
        split="test",
        task_type="regression",
        resize=(256, 512),
        augment=False,
    )

    sample_reg = dataset_regression[0]
    y_reg = sample_reg["y"]

    print(f"  Regression - y: {y_reg.shape} (3 regression targets)")
    assert y_reg.shape == (3,), f"y should be (3,) for regression, got {y_reg.shape}"

    print("✓ Dataset shape test passed!")


if __name__ == "__main__":
    test_dataset_shapes()
    print("\n✅ All shape tests passed!")
