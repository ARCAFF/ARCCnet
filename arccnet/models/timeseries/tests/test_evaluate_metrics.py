"""Tests for evaluation metric contracts."""

import numpy as np

from arccnet.models.timeseries.evaluate import compute_multiclass_metrics


def test_multiclass_metric_contract():
    """compute_multiclass_metrics should support 3-class layout [No-flare, C, M+]."""
    y_true = np.array([0, 1, 2, 1, 0, 2], dtype=np.int64)
    probs = np.array(
        [
            [0.9, 0.08, 0.02],
            [0.1, 0.8, 0.1],
            [0.05, 0.2, 0.75],
            [0.15, 0.7, 0.15],
            [0.75, 0.2, 0.05],
            [0.1, 0.3, 0.6],
        ],
        dtype=np.float32,
    )

    metrics = compute_multiclass_metrics(
        y_true,
        probs,
        threshold=0.5,
        class_names=["No-flare", "C", "M+"],
    )
    required = {"accuracy", "balanced_accuracy", "m_plus_tss", "m_plus_hss", "no_flare_ovr_tss", "m_plus_ovr_tss"}
    assert required.issubset(metrics.keys()), f"Missing expected metric keys: {required - set(metrics.keys())}"
    assert "x_plus_tss" not in metrics, "x_plus metrics should not be emitted without an explicit X class"


def test_multiclass_metric_contract_with_no_flare_class():
    """compute_multiclass_metrics should support 4-class layout [No-flare, C, M, X]."""
    y_true = np.array([0, 1, 2, 3, 0, 2, 1, 3], dtype=np.int64)
    probs = np.array(
        [
            [0.85, 0.10, 0.03, 0.02],
            [0.10, 0.75, 0.10, 0.05],
            [0.05, 0.10, 0.70, 0.15],
            [0.02, 0.03, 0.15, 0.80],
            [0.80, 0.12, 0.05, 0.03],
            [0.08, 0.10, 0.66, 0.16],
            [0.12, 0.70, 0.12, 0.06],
            [0.03, 0.05, 0.17, 0.75],
        ],
        dtype=np.float32,
    )

    metrics = compute_multiclass_metrics(
        y_true,
        probs,
        threshold=0.5,
        class_names=["No-flare", "C", "M", "X"],
    )
    required = {"accuracy", "balanced_accuracy", "class_3_acc", "m_plus_tss", "x_plus_tss", "no_flare_ovr_tss"}
    assert required.issubset(metrics.keys()), f"Missing expected metric keys: {required - set(metrics.keys())}"


if __name__ == "__main__":
    test_multiclass_metric_contract()
    test_multiclass_metric_contract_with_no_flare_class()
    print("\n✅ Evaluation metric tests passed!")
