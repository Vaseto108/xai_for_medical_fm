import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def _to_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def classification_metrics(probs, labels, threshold=0.5):
    probs = _to_numpy(probs)
    labels = _to_numpy(labels)

    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    auc_per_class = []
    valid_aucs = []

    for class_idx in range(labels.shape[1]):
        y_true = labels[:, class_idx]
        y_score = probs[:, class_idx]

        if np.unique(y_true).size < 2:
            auc_per_class.append(None)
            continue

        try:
            auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auc_per_class.append(None)
            continue

        auc_per_class.append(auc)
        valid_aucs.append(auc)

    preds = (probs >= threshold).astype(np.float32)
    labels = labels.astype(np.float32)

    accuracy_per_class = (preds == labels).mean(axis=0).astype(float).tolist()
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0).astype(float).tolist()

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    mean_accuracy = float(np.mean(accuracy_per_class))
    exact_match_accuracy = float((preds == labels).all(axis=1).mean())
    f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    f1_micro = float(f1_score(labels, preds, average="micro", zero_division=0))

    return {
        "mean_auc": mean_auc,
        "mean_accuracy": mean_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "accuracy_per_class": accuracy_per_class,
        "f1_per_class": f1_per_class,
        "auc_per_class": auc_per_class,
    }


def _with_context(row, context):
    if not context:
        return row
    return {**context, **row}


def threshold_metric_table(probs, labels, thresholds, context=None):
    """Evaluate threshold-dependent classification metrics without retraining."""

    rows = []
    for threshold in thresholds:
        metrics = classification_metrics(probs, labels, threshold=threshold)
        rows.append(
            _with_context(
                {
                    "threshold": float(threshold),
                    "mean_auc": metrics["mean_auc"],
                    "mean_accuracy": metrics["mean_accuracy"],
                    "exact_match_accuracy": metrics["exact_match_accuracy"],
                    "f1_macro": metrics["f1_macro"],
                    "f1_micro": metrics["f1_micro"],
                },
                context,
            )
        )
    return pd.DataFrame(rows)


def per_class_metric_table(probs, labels, class_names=None, threshold=0.5, context=None):
    """Build a per-class metric table for one probability set and threshold."""

    probs = _to_numpy(probs)
    labels = _to_numpy(labels)

    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    if class_names is None:
        class_names = [f"class_{idx}" for idx in range(labels.shape[1])]

    metrics = classification_metrics(probs, labels, threshold=threshold)
    preds = (probs >= threshold).astype(np.float32)

    rows = []
    for class_idx, class_name in enumerate(class_names):
        rows.append(
            _with_context(
                {
                    "class_name": class_name,
                    "threshold": float(threshold),
                    "true_positive_rate": float(labels[:, class_idx].mean()),
                    "predicted_positive_rate": float(preds[:, class_idx].mean()),
                    "accuracy": metrics["accuracy_per_class"][class_idx],
                    "f1": metrics["f1_per_class"][class_idx],
                    "auc": metrics["auc_per_class"][class_idx],
                },
                context,
            )
        )
    return pd.DataFrame(rows)


def efficiency_summary(
    adaptation_method,
    total_params,
    trainable_params,
    phase_seconds=None,
    peak_gpu_memory_mb=None,
):
    """Build a phase-aware efficiency record for an adaptation method.

    Different methods spend time in different places: kNN spends most of its
    measured time on feature extraction and neighbor search, while linear probing
    spends it on supervised training and validation. ``phase_seconds`` keeps
    those method-specific timings explicit instead of collapsing everything into
    one ambiguous runtime number.
    """

    phase_seconds = phase_seconds or {}
    total_params = int(total_params)
    trainable_params = int(trainable_params)

    record = {
        "adaptation_method": adaptation_method,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_param_fraction": (
            float(trainable_params / total_params) if total_params else float("nan")
        ),
        "peak_gpu_memory_mb": (
            None if peak_gpu_memory_mb is None else float(peak_gpu_memory_mb)
        ),
    }

    total_runtime_seconds = 0.0
    for phase_name, seconds in phase_seconds.items():
        seconds = float(seconds)
        record[f"{phase_name}_seconds"] = seconds
        total_runtime_seconds += seconds

    record["total_runtime_seconds"] = float(total_runtime_seconds)
    return record
