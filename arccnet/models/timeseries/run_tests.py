#!/usr/bin/env python3
"""
Quick test runner for timeseries pipeline.
Runs all component tests to verify installation.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

print("=" * 60)
print("TIMESERIES PIPELINE TEST SUITE")
print("=" * 60)

# Test 1: Dataset building
print("\n[1/5] Testing dataset building...")
try:
    from arccnet.models.timeseries.tests.test_dataset_build import (
        test_build_dataset_smoke,
        test_parse_sample_dirname,
    )

    test_parse_sample_dirname()
    test_build_dataset_smoke()
    print("✅ Dataset building tests passed!")
except Exception as e:
    print(f"❌ Dataset building tests failed: {e}")
    sys.exit(1)

# Test 2: Dataset shapes
print("\n[2/5] Testing dataset shapes...")
try:
    from arccnet.models.timeseries.tests.test_dataset_shapes import test_dataset_shapes

    test_dataset_shapes()
    print("✅ Dataset shape tests passed!")
except Exception as e:
    print(f"❌ Dataset shape tests failed: {e}")
    sys.exit(1)

# Test 3: Splitters + DataModule integration
print("\n[3/5] Testing splitters/datamodule integration...")
try:
    from arccnet.models.timeseries.tests.test_splitters_datamodule import (
        test_datamodule_dataset_wiring,
        test_splitter_contract,
    )

    test_splitter_contract()
    test_datamodule_dataset_wiring()
    print("✅ Splitters/DataModule tests passed!")
except Exception as e:
    print(f"❌ Splitters/DataModule tests failed: {e}")
    sys.exit(1)

# Test 4: Evaluation metrics
print("\n[4/5] Testing evaluation metrics...")
try:
    from arccnet.models.timeseries.tests.test_evaluate_metrics import test_multiclass_metric_contract

    test_multiclass_metric_contract()
    print("✅ Evaluation metric tests passed!")
except Exception as e:
    print(f"❌ Evaluation metric tests failed: {e}")
    sys.exit(1)

# Test 5: Model forward pass
print("\n[5/5] Testing model forward pass...")
try:
    from arccnet.models.timeseries.tests.test_model_forward import (
        test_flare_forecaster_forward,
        test_model_cuda,
        test_spatial_encoder_forward,
        test_temporal_transformer_forward,
    )

    test_spatial_encoder_forward()
    test_temporal_transformer_forward()
    test_flare_forecaster_forward()
    test_model_cuda()
    print("✅ Model forward pass tests passed!")
except Exception as e:
    print(f"❌ Model forward pass tests failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✨")
print("=" * 60)
print("\nYou can now:")
print("  1. Build dataset: python -m arccnet.models.timeseries.manifest --help")
print("  2. Train model:   python -m arccnet.models.timeseries.train --help")
print("  3. Evaluate model: python -m arccnet.models.timeseries.evaluate --help")
print("\nSee README.md for detailed usage instructions.")
