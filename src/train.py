import math
import time

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.eval import classification_metrics
from src.model import get_features


DEFAULT_KNN_FEWSHOT_SETTINGS = [
    {"setting": "1-img", "n_train": 1, "k": 1},
    {"setting": "2-img", "n_train": 2, "k": 1},
    {"setting": "5-img", "n_train": 5, "k": 3},
    {"setting": "10-img", "n_train": 10, "k": 5},
    {"setting": "20-img", "n_train": 20, "k": 5},
    {"setting": "25-img", "n_train": 25, "k": 5},
    {"setting": "50-img", "n_train": 50, "k": 10},
    {"setting": "100-img", "n_train": 100, "k": 20},
    {"setting": "500-img", "n_train": 500, "k": 50},
    {"setting": "1000-img", "n_train": 1000, "k": 50},
    {"setting": "5000-img", "n_train": 5000, "k": 100},
    {"setting": "full", "n_train": None, "k": 2000},
]


def filter_knn_fewshot_settings(settings, available_train_samples):
    return [
        dict(setting)
        for setting in settings
        if setting["n_train"] is None or setting["n_train"] <= available_train_samples
    ]


def evaluate_model(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_probs = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["labels"].float().to(device)

            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, labels)
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                n_samples += batch_size

            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())
            all_ids.extend(list(batch["ids"]))

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = classification_metrics(probs, labels)
    metrics["loss"] = total_loss / n_samples if criterion is not None and n_samples else None
    return metrics, probs, labels, all_ids


def train_model(model, train_loader, val_loader, device, epochs=3, lr=1e-3, weight_decay=0.0):
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    if not trainable_params:
        raise ValueError("No trainable parameters found. Check freeze settings.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        train_probs = []
        train_labels = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for batch in progress:
            images = batch["images"].to(device)
            labels = batch["labels"].float().to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            probs = torch.sigmoid(logits.detach())
            train_probs.append(probs.cpu())
            train_labels.append(labels.cpu())

            batch_accuracy = ((probs >= 0.5) == labels.bool()).float().mean().item()
            progress.set_postfix(
                train_loss=total_loss / n_samples,
                batch_acc=batch_accuracy,
            )

        train_loss = total_loss / n_samples
        train_metrics = classification_metrics(
            torch.cat(train_probs, dim=0),
            torch.cat(train_labels, dim=0),
        )
        val_metrics, _, _, _ = evaluate_model(model, val_loader, device, criterion)

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mean_accuracy": train_metrics["mean_accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "train_f1_micro": train_metrics["f1_micro"],
            "val_loss": val_metrics["loss"],
            "val_mean_auc": val_metrics["mean_auc"],
            "val_mean_accuracy": val_metrics["mean_accuracy"],
            "val_exact_match_accuracy": val_metrics["exact_match_accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_micro": val_metrics["f1_micro"],
            "val_accuracy_per_class": val_metrics["accuracy_per_class"],
            "val_f1_per_class": val_metrics["f1_per_class"],
            "val_auc_per_class": val_metrics["auc_per_class"],
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_mean_acc={train_metrics['mean_accuracy']:.4f} | "
            f"train_f1_macro={train_metrics['f1_macro']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_mean_auc={val_metrics['mean_auc']:.4f} | "
            f"val_mean_acc={val_metrics['mean_accuracy']:.4f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

    return history


def knn_predict(
    train_features,
    train_labels,
    query_features,
    k=20,
    batch_size=256,
    device=None,
):
    if k < 1:
        raise ValueError("k must be at least 1.")

    if device is None:
        device = train_features.device

    k = min(k, train_features.shape[0])
    train_features = F.normalize(train_features.float(), dim=1).to(device)
    train_labels = train_labels.float().to(device)

    all_probs = []
    for start in range(0, query_features.shape[0], batch_size):
        query_batch = query_features[start : start + batch_size]
        query_batch = F.normalize(query_batch.float(), dim=1).to(device)

        similarities = query_batch @ train_features.T
        neighbor_indices = similarities.topk(k=k, dim=1).indices
        neighbor_labels = train_labels[neighbor_indices]
        probs = neighbor_labels.mean(dim=1)
        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)


def sample_feature_indices(n_total, n_train, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(n_total, generator=generator)[:n_train]


def _with_context(df, context):
    if not context:
        return df

    df = df.copy()
    for key, value in reversed(list(context.items())):
        df.insert(0, key, value)
    return df


def summarize_knn_runs(runs_df, context=None):
    summary_df = (
        runs_df.groupby(
            ["setting", "n_train", "k", "threshold", "positive_neighbors_needed"],
            sort=False,
        )
        .agg(
            runs=("setting", "count"),
            classes_with_positive_train_examples_mean=(
                "classes_with_positive_train_examples",
                "mean",
            ),
            mean_auc_mean=("mean_auc", "mean"),
            mean_auc_std=("mean_auc", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            f1_micro_mean=("f1_micro", "mean"),
            f1_micro_std=("f1_micro", "std"),
            mean_accuracy_mean=("mean_accuracy", "mean"),
            exact_match_accuracy_mean=("exact_match_accuracy", "mean"),
        )
        .reset_index()
    )
    return _with_context(summary_df, context)


def run_knn_fewshot_experiment(
    train_features,
    train_labels,
    val_features,
    val_labels,
    class_names,
    settings=None,
    seeds=None,
    threshold=0.05,
    batch_size=256,
    device=None,
    context=None,
):
    if settings is None:
        settings = DEFAULT_KNN_FEWSHOT_SETTINGS
    if seeds is None:
        seeds = list(range(10))
    if class_names is None:
        class_names = [f"class_{idx}" for idx in range(val_labels.shape[1])]

    result_rows = []
    per_class_full_rows = []
    n_total = train_features.shape[0]

    start = time.perf_counter()

    for config in filter_knn_fewshot_settings(settings, n_total):
        setting = config["setting"]
        n_train = config["n_train"]
        requested_k = config["k"]
        run_seeds = ["full"] if n_train is None else seeds

        for seed in run_seeds:
            if n_train is None:
                indices = torch.arange(n_total)
            else:
                indices = sample_feature_indices(n_total, n_train, seed)

            subset_features = train_features[indices]
            subset_labels = train_labels[indices]
            n_train_actual = subset_features.shape[0]
            k_effective = min(requested_k, n_train_actual)
            positive_neighbors_needed = math.ceil(k_effective * threshold)

            probs = knn_predict(
                train_features=subset_features,
                train_labels=subset_labels,
                query_features=val_features,
                k=k_effective,
                batch_size=batch_size,
                device=device,
            )
            metrics = classification_metrics(probs, val_labels, threshold=threshold)
            preds = (probs >= threshold).float()
            positive_counts = subset_labels.sum(dim=0)

            result_rows.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "n_train": n_train_actual,
                    "k": k_effective,
                    "threshold": threshold,
                    "positive_neighbors_needed": positive_neighbors_needed,
                    "neighbor_fraction": k_effective / n_train_actual,
                    "classes_with_positive_train_examples": int((positive_counts > 0).sum().item()),
                    "mean_auc": metrics["mean_auc"],
                    "mean_accuracy": metrics["mean_accuracy"],
                    "exact_match_accuracy": metrics["exact_match_accuracy"],
                    "f1_macro": metrics["f1_macro"],
                    "f1_micro": metrics["f1_micro"],
                }
            )

            if setting == "full":
                for class_idx, class_name in enumerate(class_names):
                    per_class_full_rows.append(
                        {
                            "class_name": class_name,
                            "true_positive_rate": float(val_labels[:, class_idx].mean().item()),
                            "predicted_positive_rate": float(preds[:, class_idx].mean().item()),
                            "accuracy": metrics["accuracy_per_class"][class_idx],
                            "f1": metrics["f1_per_class"][class_idx],
                            "auc": metrics["auc_per_class"][class_idx],
                        }
                    )

    knn_eval_seconds = time.perf_counter() - start

    runs_df = _with_context(pd.DataFrame(result_rows), context)
    summary_df = summarize_knn_runs(pd.DataFrame(result_rows), context=context)
    per_class_full_df = _with_context(pd.DataFrame(per_class_full_rows), context)
    metadata = {"knn_eval_seconds": float(knn_eval_seconds)}

    return {
        "runs": runs_df,
        "summary": summary_df,
        "per_class_full": per_class_full_df,
        "metadata": metadata,
    }


def evaluate_knn_baseline(model, train_loader, val_loader, device, k=20, batch_size=256):
    train_features, train_labels, _ = get_features(model, train_loader, device)
    val_features, val_labels, val_ids = get_features(model, val_loader, device)

    probs = knn_predict(
        train_features=train_features,
        train_labels=train_labels,
        query_features=val_features,
        k=k,
        batch_size=batch_size,
        device=device,
    )
    metrics = classification_metrics(probs, val_labels)
    return metrics, probs, val_labels, val_ids
