"""Utilities for post-training threshold tuning in binary classification."""

from __future__ import annotations

from typing import Any
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ThresholdTuningResult:
    """Result of a decision-threshold tuning run."""

    threshold: float
    objective: str
    objective_value: float
    status: str
    candidate_count: int
    default_threshold: float
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_binary_outputs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect probabilities and labels from a binary classifier dataloader.

    Returns arrays with shapes ``(N,)`` for probabilities and labels.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    was_training = model.training
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    model.eval()
    try:
        with torch.no_grad():
            for batch in dataloader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError("Expected dataloader batches to be (inputs, labels).")

                inputs, targets = batch[0], batch[1]
                logits = model(inputs.to(device))
                probs = torch.sigmoid(logits.reshape(-1)).detach().cpu().to(dtype=torch.float32)
                target_int = targets.detach().cpu().to(dtype=torch.int64).reshape(-1)

                probabilities.append(probs.reshape(-1))
                labels.append(target_int)
    finally:
        model.train(was_training)

    if not probabilities:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    return (
        torch.cat(probabilities).numpy().astype(np.float32, copy=False),
        torch.cat(labels).numpy().astype(np.int64, copy=False),
    )


def compute_binary_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute standard binary metrics for a given decision threshold."""
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be in [0, 1].")

    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y_true = (np.asarray(labels, dtype=np.int64).reshape(-1) > 0).astype(np.int64)

    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("Probabilities and labels must have the same number of samples.")

    if y_true.size == 0:
        return {
            "threshold": threshold,
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tpr": 0.0,
            "fpr": 0.0,
            "tss": 0.0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "support": 0,
            "positive_support": 0,
            "negative_support": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    y_pred = (probs >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    tpr_denom = tp + fn
    fpr_denom = fp + tn
    tpr = float(tp / tpr_denom) if tpr_denom > 0 else 0.0
    fpr = float(fp / fpr_denom) if fpr_denom > 0 else 0.0
    tss = tpr - fpr

    return {
        "threshold": threshold,
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tpr": tpr,
        "fpr": fpr,
        "tss": tss,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support": int(y_true.size),
        "positive_support": int(y_true.sum()),
        "negative_support": int((1 - y_true).sum()),
        "confusion_matrix": cm.tolist(),
    }


def tune_binary_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    default_threshold: float = 0.5,
    objective: str = "f1",
    search_method: str = "unique_probs_exact",
) -> ThresholdTuningResult:
    """
    Tune decision threshold on validation probabilities and labels.

    Search strategy:
    - Evaluate all unique predicted probabilities plus boundary points 0.0 and 1.0.
    - Maximize requested objective (``f1`` or ``tss``).
    - Deterministic tie-breaks: higher recall, then lower threshold.
    """
    if objective not in {"f1", "tss"}:
        raise ValueError(f"Unsupported objective '{objective}'. Supported objectives: 'f1', 'tss'.")
    if search_method != "unique_probs_exact":
        raise ValueError(f"Unsupported search method '{search_method}'.")
    if not 0.0 <= float(default_threshold) <= 1.0:
        raise ValueError("default_threshold must be in [0, 1].")

    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y_true = (np.asarray(labels, dtype=np.int64).reshape(-1) > 0).astype(np.int64)

    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("Probabilities and labels must have the same number of samples.")

    baseline_metrics = compute_binary_metrics(probs, y_true, default_threshold)

    if y_true.size == 0:
        return ThresholdTuningResult(
            threshold=float(default_threshold),
            objective=objective,
            objective_value=float(baseline_metrics[objective]),
            status="fallback_empty_labels",
            candidate_count=0,
            default_threshold=float(default_threshold),
            metrics=baseline_metrics,
        )

    if np.unique(y_true).size < 2:
        return ThresholdTuningResult(
            threshold=float(default_threshold),
            objective=objective,
            objective_value=float(baseline_metrics[objective]),
            status="fallback_single_class_labels",
            candidate_count=0,
            default_threshold=float(default_threshold),
            metrics=baseline_metrics,
        )

    candidates = np.unique(np.clip(np.concatenate([probs, np.array([0.0, 1.0])]), 0.0, 1.0))

    best_threshold = float(default_threshold)
    best_metrics = baseline_metrics
    best_score = float(baseline_metrics[objective])

    eps = 1e-12
    for candidate in candidates:
        candidate_threshold = float(candidate)
        candidate_metrics = compute_binary_metrics(probs, y_true, candidate_threshold)
        candidate_score = float(candidate_metrics[objective])

        if candidate_score > best_score + eps:
            best_threshold = candidate_threshold
            best_metrics = candidate_metrics
            best_score = candidate_score
            continue

        if abs(candidate_score - best_score) <= eps:
            if candidate_metrics["recall"] > best_metrics["recall"] + eps:
                best_threshold = candidate_threshold
                best_metrics = candidate_metrics
                best_score = candidate_score
            elif (
                abs(candidate_metrics["recall"] - best_metrics["recall"]) <= eps
                and candidate_threshold < best_threshold
            ):
                best_threshold = candidate_threshold
                best_metrics = candidate_metrics
                best_score = candidate_score

    return ThresholdTuningResult(
        threshold=best_threshold,
        objective=objective,
        objective_value=best_score,
        status="ok",
        candidate_count=int(candidates.size),
        default_threshold=float(default_threshold),
        metrics=best_metrics,
    )
