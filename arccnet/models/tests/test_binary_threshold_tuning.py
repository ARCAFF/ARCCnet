import json

import numpy as np
import pytest
import torch.nn as nn

from arccnet.models.checkpoint_manager import BinaryClassificationCheckpointManager
from arccnet.models.flares.binary_classification import threshold_tuning as tt


def test_tune_binary_threshold_f1_tie_breaks_are_deterministic():
    probs = np.array([0.1, 0.1, 0.1, 0.2], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    result = tt.tune_binary_threshold(
        probs,
        labels,
        default_threshold=0.5,
        objective="f1",
        search_method="unique_probs_exact",
    )

    assert result.status == "ok"
    assert result.candidate_count == 4  # {0.0, 0.1, 0.2, 1.0}
    assert result.threshold == pytest.approx(0.0)
    assert result.objective_value == pytest.approx(2.0 / 3.0)
    assert result.metrics["recall"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("probs", "labels", "expected_status"),
    [
        (np.array([], dtype=np.float64), np.array([], dtype=np.int64), "fallback_empty_labels"),
        (
            np.array([0.2, 0.7, 0.9], dtype=np.float64),
            np.array([0, 0, 0], dtype=np.int64),
            "fallback_single_class_labels",
        ),
    ],
)
def test_tune_binary_threshold_fallback_cases(probs, labels, expected_status):
    result = tt.tune_binary_threshold(probs, labels, default_threshold=0.5)
    assert result.status == expected_status
    assert result.threshold == pytest.approx(0.5)
    assert result.candidate_count == 0


def test_compute_binary_metrics_known_values():
    probs = np.array([0.9, 0.8, 0.4, 0.1], dtype=np.float64)
    labels = np.array([1, 0, 1, 0], dtype=np.int64)

    metrics = tt.compute_binary_metrics(probs, labels, threshold=0.5)

    assert metrics["acc"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["tpr"] == pytest.approx(0.5)
    assert metrics["fpr"] == pytest.approx(0.5)
    assert metrics["tss"] == pytest.approx(0.0)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_tune_binary_threshold_supports_tss_objective():
    probs = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    labels = np.array([1, 1, 0, 0, 1, 0], dtype=np.int64)

    result = tt.tune_binary_threshold(
        probs,
        labels,
        default_threshold=0.2,
        objective="tss",
        search_method="unique_probs_exact",
    )

    assert result.status == "ok"
    assert result.threshold == pytest.approx(0.8)
    assert result.objective == "tss"
    assert result.objective_value == pytest.approx(2.0 / 3.0)
    assert result.metrics["tss"] == pytest.approx(2.0 / 3.0)
    assert result.metrics["fpr"] == pytest.approx(0.0)


def test_model_set_decision_threshold_rebuilds_metrics(monkeypatch):
    pytest.importorskip("timm")
    from arccnet.models.flares.binary_classification import model as binary_model

    class DummyBackbone(nn.Module):
        def __init__(self, num_classes, in_chans):
            super().__init__()
            self.conv = nn.Conv2d(in_chans, 4, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(4, num_classes)

        def forward(self, x):
            x = self.relu(self.conv(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    def fake_create_model(model_name, pretrained, num_classes, in_chans):  # noqa: ARG001
        return DummyBackbone(num_classes=num_classes, in_chans=in_chans)

    monkeypatch.setattr(binary_model.timm, "create_model", fake_create_model)

    classifier = binary_model.FlareClassifier(
        model_name="dummy",
        num_classes=1,
        in_chans=1,
        pretrained=False,
        decision_threshold=0.5,
    )

    previous_val_metric = classifier.val_metrics["val_f1"]
    classifier.set_decision_threshold(0.8)

    assert classifier.decision_threshold == pytest.approx(0.8)
    assert classifier.hparams.decision_threshold == pytest.approx(0.8)
    assert classifier.val_metrics["val_f1"] is not previous_val_metric
    assert classifier.val_metrics["val_f1"].threshold == pytest.approx(0.8)
    assert classifier.test_metrics["test_precision"].threshold == pytest.approx(0.8)
    assert classifier.val_confusion_matrix.threshold == pytest.approx(0.8)


def test_checkpoint_manager_persists_threshold_tuning_artifact(tmp_path):
    manager = BinaryClassificationCheckpointManager(
        data_folder=str(tmp_path),
        model_name="resnet50",
        loss_function="weighted_bce",
    )

    threshold_report = {
        "enabled": True,
        "objective": "f1",
        "split": "val",
        "search_method": "unique_probs_exact",
        "default_threshold": 0.5,
        "selected_threshold": 0.3,
        "status": "ok",
        "candidate_count": 15,
        "objective_value": 0.71,
        "baseline_val_metrics": {"f1": 0.62},
        "tuned_val_metrics": {"f1": 0.71},
        "tuned_test_metrics": {"test_f1": 0.69},
    }
    manager.save_threshold_tuning_report(threshold_report)
    manager.save_training_metadata(
        {
            "decision_threshold": threshold_report["selected_threshold"],
            "threshold_tuning_status": threshold_report["status"],
            "test_results": threshold_report["tuned_test_metrics"],
        }
    )
    manager.save_classification_report(
        {
            "test_metrics": threshold_report["tuned_test_metrics"],
            "decision_threshold": threshold_report["selected_threshold"],
            "threshold_tuning": threshold_report,
        }
    )

    checkpoint_dir = manager.get_checkpoint_path()
    with open(checkpoint_dir / "threshold_tuning.json") as f:
        threshold_json = json.load(f)
    with open(checkpoint_dir / "training_summary.json") as f:
        training_json = json.load(f)
    with open(checkpoint_dir / "classification_report.json") as f:
        report_json = json.load(f)

    assert threshold_json["selected_threshold"] == pytest.approx(0.3)
    assert threshold_json["tuned_test_metrics"]["test_f1"] == pytest.approx(0.69)
    assert training_json["decision_threshold"] == pytest.approx(0.3)
    assert report_json["threshold_tuning"]["status"] == "ok"
