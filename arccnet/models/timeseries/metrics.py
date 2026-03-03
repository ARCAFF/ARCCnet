"""Shared metric helpers for timeseries training/evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def normalize_class_name(name):
    """Normalize class-name strings for key generation and simple matching."""
    normalized = str(name).strip().lower()
    normalized = normalized.replace("+", "_plus")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def class_semantic_tag(name):
    """Map a class name into a coarse semantic bucket."""
    normalized = normalize_class_name(name)

    if normalized.startswith("no_flare") or normalized in {"noflare", "quiet", "none"}:
        return "no_flare"
    if "m_plus" in normalized or normalized in {"mplus"}:
        return "m_plus"
    if normalized.startswith("x"):
        return "x"
    if normalized.startswith("m"):
        return "m"
    if normalized.startswith("c"):
        return "c"
    return "other"


def get_operational_class_layout(num_classes, class_names=None):
    """
    Resolve class-index layout used for M+/X+ operational metrics.

    Supported multiclass layouts:
    - 3-class: [No-flare, C, M+]
    - 4-class: [No-flare, C, M, X]
    - Legacy 3-class fallback: [C, M, X]
    """
    if class_names is not None and len(class_names) == num_classes:
        tags = [class_semantic_tag(name) for name in class_names]

        c_index = next((idx for idx, tag in enumerate(tags) if tag == "c"), None)
        m_plus_direct_index = next((idx for idx, tag in enumerate(tags) if tag == "m_plus"), None)
        x_index = next((idx for idx, tag in enumerate(tags) if tag == "x"), None)

        if m_plus_direct_index is not None:
            m_plus_indices = [m_plus_direct_index]
        else:
            m_plus_indices = [idx for idx, tag in enumerate(tags) if tag in {"m", "x"}]

        if c_index is not None and m_plus_indices:
            return {
                "class_names": list(class_names),
                "c_index": int(c_index),
                "m_plus_indices": [int(idx) for idx in m_plus_indices],
                "x_index": int(x_index) if x_index is not None else None,
            }

    if num_classes == 4:
        return {
            "class_names": ["No-flare", "C", "M", "X"],
            "c_index": 1,
            "m_plus_indices": [2, 3],
            "x_index": 3,
        }
    if num_classes == 3:
        return {
            "class_names": ["C", "M", "X"],
            "c_index": 0,
            "m_plus_indices": [1, 2],
            "x_index": 2,
        }

    # Fallback: assume classes are ordered by severity and last label is X-like.
    m_index = max(0, num_classes - 2)
    x_index = max(0, num_classes - 1)
    return {
        "class_names": [f"Class_{i}" for i in range(num_classes)],
        "c_index": max(0, num_classes - 3),
        "m_plus_indices": sorted(set([m_index, x_index])),
        "x_index": x_index,
    }


def compute_binary_skill(y_true, y_score, threshold=0.5):
    """Compute binary operational skill metrics from probabilistic scores."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    tpr = tp / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    tss = tpr - fpr

    hss_num = 2.0 * (tp * tn - fp * fn)
    hss_den = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn)) + 1e-8
    hss = hss_num / hss_den

    out = {
        "tss": float(tss),
        "hss": float(hss),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None

    return out


def compute_multiclass_metrics(y_true, probs, threshold=0.5, class_names=None):
    """Compute multiclass and operational one-vs-rest metrics."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)
    num_classes = int(probs.shape[1])
    layout = get_operational_class_layout(num_classes, class_names=class_names)
    y_pred = np.argmax(probs, axis=1)
    all_classes = list(range(num_classes))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=all_classes).tolist(),
    }

    cm = np.array(metrics["confusion_matrix"])
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    for idx in range(num_classes):
        metrics[f"class_{idx}_acc"] = float(per_class_acc[idx])

    # One-vs-rest scores for each class.
    for class_idx, class_name in enumerate(layout["class_names"]):
        class_key = normalize_class_name(class_name)
        skill = compute_binary_skill((y_true == class_idx).astype(int), probs[:, class_idx], threshold=threshold)
        metrics[f"{class_key}_ovr_tss"] = skill["tss"]
        metrics[f"{class_key}_ovr_hss"] = skill["hss"]
        metrics[f"{class_key}_ovr_pr_auc"] = skill["pr_auc"]
        metrics[f"{class_key}_ovr_roc_auc"] = skill["roc_auc"]
        metrics[f"{class_key}_ovr_tpr"] = skill["tpr"]
        metrics[f"{class_key}_ovr_fpr"] = skill["fpr"]

        # Backward-compatible aliases for historical keys (C/M/X uppercase).
        if class_name in {"C", "M", "X"}:
            metrics[f"{class_name}_ovr_tss"] = skill["tss"]
            metrics[f"{class_name}_ovr_hss"] = skill["hss"]
            metrics[f"{class_name}_ovr_pr_auc"] = skill["pr_auc"]
            metrics[f"{class_name}_ovr_roc_auc"] = skill["roc_auc"]
            metrics[f"{class_name}_ovr_tpr"] = skill["tpr"]
            metrics[f"{class_name}_ovr_fpr"] = skill["fpr"]

    # Operational targets from multiclass distribution.
    m_plus_indices = layout["m_plus_indices"]
    m_plus_true = np.isin(y_true, m_plus_indices).astype(int)
    m_plus_score = probs[:, m_plus_indices].sum(axis=1)
    m_plus_skill = compute_binary_skill(m_plus_true, m_plus_score, threshold=threshold)

    metrics["m_plus_tss"] = m_plus_skill["tss"]
    metrics["m_plus_hss"] = m_plus_skill["hss"]
    metrics["m_plus_pr_auc"] = m_plus_skill["pr_auc"]
    metrics["m_plus_roc_auc"] = m_plus_skill["roc_auc"]
    metrics["m_plus_tpr"] = m_plus_skill["tpr"]
    metrics["m_plus_fpr"] = m_plus_skill["fpr"]
    metrics["m_plus_confusion_matrix"] = [
        [m_plus_skill["tn"], m_plus_skill["fp"]],
        [m_plus_skill["fn"], m_plus_skill["tp"]],
    ]

    x_index = layout["x_index"]
    if x_index is not None:
        x_plus_true = (y_true == x_index).astype(int)
        x_plus_score = probs[:, x_index]
        x_plus_skill = compute_binary_skill(x_plus_true, x_plus_score, threshold=threshold)
        metrics["x_plus_tss"] = x_plus_skill["tss"]
        metrics["x_plus_hss"] = x_plus_skill["hss"]
        metrics["x_plus_pr_auc"] = x_plus_skill["pr_auc"]
        metrics["x_plus_roc_auc"] = x_plus_skill["roc_auc"]
        metrics["x_plus_tpr"] = x_plus_skill["tpr"]
        metrics["x_plus_fpr"] = x_plus_skill["fpr"]
        metrics["x_plus_confusion_matrix"] = [
            [x_plus_skill["tn"], x_plus_skill["fp"]],
            [x_plus_skill["fn"], x_plus_skill["tp"]],
        ]

    return metrics


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics for log flare-count targets."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)

    metrics = {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
    }

    mse_per_target = ((y_true - y_pred) ** 2).mean(axis=0)
    mae_per_target = np.abs(y_true - y_pred).mean(axis=0)

    for idx in range(y_true.shape[1]):
        metrics[f"target_{idx}_mse"] = float(mse_per_target[idx])
        metrics[f"target_{idx}_mae"] = float(mae_per_target[idx])
        try:
            metrics[f"target_{idx}_r2"] = float(r2_score(y_true[:, idx], y_pred[:, idx]))
        except Exception:
            metrics[f"target_{idx}_r2"] = 0.0

    return metrics
