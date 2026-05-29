import numpy as np
from sklearn.metrics import roc_auc_score


def _to_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def classification_metrics(probs, labels):
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

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return {"mean_auc": mean_auc, "auc_per_class": auc_per_class}
