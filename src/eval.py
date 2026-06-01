import numpy as np
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
