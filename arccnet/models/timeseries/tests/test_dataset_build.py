"""Test dataset building and basic functionality."""

import tempfile
from pathlib import Path

from arccnet.models.timeseries.manifest import build_dataset, parse_sample_dirname


def test_parse_sample_dirname():
    """Test directory name parsing."""
    print("Testing parse_sample_dirname...")

    dirname = "2011-01-03_11142_Beta_Dso_Xb0_Mb0_Cb0_Xa0_Ma0_Ca1"
    result = parse_sample_dirname(dirname)

    assert result is not None, "Parsing failed"
    assert result["date"] == "2011-01-03", f"Date mismatch: {result['date']}"
    assert result["noaa_ar"] == 11142, f"NOAA AR mismatch: {result['noaa_ar']}"
    assert result["ca"] == 1, f"Ca mismatch: {result['ca']}"
    assert result["xa"] == 0, f"Xa mismatch: {result['xa']}"

    print("✓ parse_sample_dirname test passed!")


def test_build_dataset_smoke():
    """Smoke test for dataset building."""
    print("\nTesting build_dataset (smoke test)...")

    data_root = Path("/ARCAFF/data/04_final/data")

    if not data_root.exists():
        print("⚠ Data directory not found, skipping test")
        return

    # Build dataset for first 5 samples
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "manifest_test.parq"
        manifest = build_dataset(data_root, output_path=output_path, max_samples=5)

        assert len(manifest) > 0, "No samples found"
        assert "sample_id" in manifest.columns, "Missing sample_id column"
        assert "paths" in manifest.columns, "Missing paths column"
        assert "flare_class" in manifest.columns, "Missing flare_class label"
        assert "log_ca" in manifest.columns, "Missing log_ca regression target"

        print(f"✓ Built dataset with {len(manifest)} samples")
        print(f"  Columns: {manifest.columns.tolist()}")

    print("✓ build_dataset smoke test passed!")


if __name__ == "__main__":
    test_parse_sample_dirname()
    test_build_dataset_smoke()
    print("\n✅ All dataset tests passed!")
