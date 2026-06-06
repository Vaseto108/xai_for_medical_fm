import itertools
import math
import time

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.eval import (
    classification_metrics,
    efficiency_summary,
    per_class_metric_table,
    threshold_metric_table,
)
from src.model import get_feature_linear_probe, get_features, model_metadata


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


def make_knn_search_settings(base_settings=None, k_values=None):
    """Expand kNN few-shot settings with candidate ``k`` values.

    ``k`` changes the neighbor vote itself, so it is part of the kNN evaluation
    grid. The original ``k`` from each base setting is always kept.
    """

    if base_settings is None:
        base_settings = DEFAULT_KNN_FEWSHOT_SETTINGS
    if k_values is None:
        return [dict(setting) for setting in base_settings]

    settings = []
    for base_setting in base_settings:
        n_train = base_setting["n_train"]
        candidate_k_values = {int(base_setting["k"])}
        candidate_k_values.update(int(k) for k in k_values)

        for k in sorted(candidate_k_values):
            if k < 1:
                raise ValueError("All k values must be at least 1.")
            if n_train is not None and k > n_train:
                continue

            setting = dict(base_setting)
            setting["k"] = int(k)
            settings.append(setting)

    return settings


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


def _is_cuda_device(device):
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


def _reset_peak_memory(device):
    if _is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device):
    if not _is_cuda_device(device):
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**2


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
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        n_samples = 0
        train_probs = []
        train_labels = []

        train_start = time.perf_counter()
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

        train_epoch_seconds = time.perf_counter() - train_start
        train_loss = total_loss / n_samples
        train_metrics = classification_metrics(
            torch.cat(train_probs, dim=0),
            torch.cat(train_labels, dim=0),
        )

        val_start = time.perf_counter()
        val_metrics, _, _, _ = evaluate_model(model, val_loader, device, criterion)
        val_eval_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start

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
            "train_epoch_seconds": float(train_epoch_seconds),
            "val_eval_seconds": float(val_eval_seconds),
            "epoch_seconds": float(epoch_seconds),
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


def train_linear_probe(
    model,
    train_loader,
    val_loader,
    device,
    epochs=3,
    lr=1e-3,
    weight_decay=0.0,
):
    """Train a linear probe on top of a frozen backbone.

    Build the model with ``get_dino_model(..., freeze_backbone=True)`` before
    calling this helper. The current implementation delegates to ``train_model``
    because the generic loop already optimizes only trainable parameters.
    """

    return train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )


def train_full_finetune(
    model,
    train_loader,
    val_loader,
    device,
    epochs=3,
    lr=1e-5,
    weight_decay=0.0,
):
    """Fine-tune all unfrozen model parameters.

    Build the model with ``get_dino_model(..., freeze_backbone=False)`` before
    calling this helper. A smaller default learning rate is used here because
    backbone parameters are expected to be trainable.
    """

    return train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )


def train_partial_finetune(
    model,
    train_loader,
    val_loader,
    device,
    num_unfrozen_blocks=1,
    epochs=3,
    lr=1e-5,
    weight_decay=0.0,
):
    """Planned partial fine-tuning workflow.

    The future implementation should configure the model with
    ``src.model.unfreeze_last_blocks`` and then call ``train_model``.
    """

    raise NotImplementedError(
        "Partial fine-tuning is planned but train_partial_finetune is not implemented yet."
    )


def train_lora(
    model,
    train_loader,
    val_loader,
    device,
    target_modules=None,
    rank=8,
    alpha=16,
    dropout=0.0,
    epochs=3,
    lr=1e-4,
    weight_decay=0.0,
):
    """Planned LoRA/PEFT training workflow.

    The future implementation should attach adapters with
    ``src.model.apply_lora_adapters`` and then call ``train_model``.
    """

    raise NotImplementedError("LoRA training is planned but train_lora is not implemented yet.")


def make_linear_probe_grid(
    lrs=(1e-4, 3e-4, 1e-3),
    weight_decays=(0.0, 1e-4, 1e-3),
    epochs=None,
):
    """Create a small Cartesian grid for frozen-feature linear probing.

    Epochs are handled as checkpoints in ``run_linear_probe_grid`` so that each
    ``lr``/``weight_decay`` head is trained once to the maximum epoch instead of
    retraining separate 5-, 10-, and 20-epoch runs.
    """

    grid = []
    for lr, weight_decay in itertools.product(lrs, weight_decays):
        config = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        }
        if epochs is not None:
            config["checkpoint_epochs"] = [int(epoch) for epoch in epochs]
        grid.append(config)
    return grid


def _linear_probe_grid_configs(train_grid):
    if train_grid is None:
        return make_linear_probe_grid()

    if isinstance(train_grid, dict):
        keys = list(train_grid)
        values = [
            value if isinstance(value, (list, tuple)) else [value]
            for value in train_grid.values()
        ]
        return [dict(zip(keys, config_values)) for config_values in itertools.product(*values)]

    return [dict(config) for config in train_grid]


def _checkpoint_epochs(configs, checkpoint_epochs):
    if checkpoint_epochs is None:
        checkpoint_epochs = []
        for config in configs:
            if "checkpoint_epochs" in config:
                checkpoint_epochs.extend(config["checkpoint_epochs"])
            elif "epochs" in config:
                checkpoint_epochs.append(config["epochs"])

    if not checkpoint_epochs:
        checkpoint_epochs = [5, 10]

    checkpoint_epochs = sorted({int(epoch) for epoch in checkpoint_epochs})
    if any(epoch < 1 for epoch in checkpoint_epochs):
        raise ValueError("checkpoint epochs must be positive integers.")
    return checkpoint_epochs


def _dedupe_linear_probe_train_configs(configs):
    deduped = []
    seen = set()
    for config in configs:
        key = (float(config["lr"]), float(config["weight_decay"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "lr": float(config["lr"]),
                "weight_decay": float(config["weight_decay"]),
            }
        )
    return deduped


def _set_torch_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def evaluate_feature_classifier(
    model,
    features,
    labels,
    device,
    batch_size=256,
    criterion=None,
    threshold=0.5,
):
    model.eval()
    dataset = TensorDataset(features.float(), labels.float())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=_is_cuda_device(device),
    )

    total_loss = 0.0
    n_samples = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for feature_batch, label_batch in loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)

            logits = model(feature_batch)
            if criterion is not None:
                loss = criterion(logits, label_batch)
                batch_size_actual = feature_batch.size(0)
                total_loss += loss.item() * batch_size_actual
                n_samples += batch_size_actual

            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(label_batch.cpu())

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = classification_metrics(probs, labels, threshold=threshold)
    metrics["loss"] = total_loss / n_samples if criterion is not None and n_samples else None
    return metrics, probs, labels


def train_feature_linear_probe(
    model,
    train_features,
    train_labels,
    val_features,
    val_labels,
    device,
    epochs=5,
    lr=1e-3,
    weight_decay=0.0,
    batch_size=256,
    show_progress=False,
):
    """Train one linear classifier head on precomputed frozen features."""

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_dataset = TensorDataset(train_features.float(), train_labels.float())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=_is_cuda_device(device),
    )

    history = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        n_samples = 0
        train_probs = []
        train_targets = []

        progress = tqdm(
            train_loader,
            desc=f"Feature probe epoch {epoch}/{epochs}",
            leave=False,
            disable=not show_progress,
        )
        for feature_batch, label_batch in progress:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feature_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()

            batch_size_actual = feature_batch.size(0)
            total_loss += loss.item() * batch_size_actual
            n_samples += batch_size_actual

            probs = torch.sigmoid(logits.detach())
            train_probs.append(probs.cpu())
            train_targets.append(label_batch.cpu())

            if show_progress:
                progress.set_postfix(train_loss=total_loss / n_samples)

        train_epoch_seconds = time.perf_counter() - train_start
        train_loss = total_loss / n_samples
        train_metrics = classification_metrics(
            torch.cat(train_probs, dim=0),
            torch.cat(train_targets, dim=0),
        )

        val_start = time.perf_counter()
        val_metrics, _, _ = evaluate_feature_classifier(
            model,
            val_features,
            val_labels,
            device,
            batch_size=batch_size,
            criterion=criterion,
        )
        val_eval_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start

        history.append(
            {
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
                "train_epoch_seconds": float(train_epoch_seconds),
                "val_eval_seconds": float(val_eval_seconds),
                "epoch_seconds": float(epoch_seconds),
            }
        )

    return history


def train_feature_linear_probe_checkpoints(
    model,
    train_features,
    train_labels,
    val_features,
    val_labels,
    device,
    checkpoint_epochs,
    lr=1e-3,
    weight_decay=0.0,
    batch_size=256,
    show_progress=False,
):
    """Train one feature-space linear probe and save validation checkpoints."""

    checkpoint_epochs = sorted({int(epoch) for epoch in checkpoint_epochs})
    max_epochs = max(checkpoint_epochs)
    checkpoint_set = set(checkpoint_epochs)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_dataset = TensorDataset(train_features.float(), train_labels.float())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=_is_cuda_device(device),
    )

    history = []
    checkpoints = {}

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        n_samples = 0
        train_probs = []
        train_targets = []

        progress = tqdm(
            train_loader,
            desc=f"Feature probe epoch {epoch}/{max_epochs}",
            leave=False,
            disable=not show_progress,
        )
        for feature_batch, label_batch in progress:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feature_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()

            batch_size_actual = feature_batch.size(0)
            total_loss += loss.item() * batch_size_actual
            n_samples += batch_size_actual

            probs = torch.sigmoid(logits.detach())
            train_probs.append(probs.cpu())
            train_targets.append(label_batch.cpu())

            if show_progress:
                progress.set_postfix(train_loss=total_loss / n_samples)

        train_epoch_seconds = time.perf_counter() - train_start
        train_loss = total_loss / n_samples
        train_metrics = classification_metrics(
            torch.cat(train_probs, dim=0),
            torch.cat(train_targets, dim=0),
        )

        val_start = time.perf_counter()
        val_metrics, val_probs, val_targets = evaluate_feature_classifier(
            model,
            val_features,
            val_labels,
            device,
            batch_size=batch_size,
            criterion=criterion,
        )
        val_eval_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start

        history.append(
            {
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
                "train_epoch_seconds": float(train_epoch_seconds),
                "val_eval_seconds": float(val_eval_seconds),
                "epoch_seconds": float(epoch_seconds),
            }
        )

        if epoch in checkpoint_set:
            checkpoints[epoch] = {
                "metrics": val_metrics,
                "probs": val_probs,
                "labels": val_targets,
                "val_eval_seconds": float(val_eval_seconds),
            }

    return history, checkpoints


def _best_row(df, metric):
    if metric not in df.columns:
        raise ValueError(f"Selection metric '{metric}' is not in the trial table.")

    sortable = df.copy()
    sortable["_selection_metric"] = pd.to_numeric(sortable[metric], errors="coerce").fillna(
        -float("inf")
    )
    sort_columns = ["_selection_metric"]
    for tie_breaker in ["mean_auc", "f1_macro", "f1_micro", "mean_accuracy"]:
        if tie_breaker in sortable.columns and tie_breaker not in sort_columns:
            sort_columns.append(tie_breaker)

    return sortable.sort_values(sort_columns, ascending=False).iloc[0].drop(
        labels=["_selection_metric"]
    )


def run_linear_probe_grid(
    feature_bank,
    device,
    class_names=None,
    train_grid=None,
    checkpoint_epochs=None,
    thresholds=None,
    batch_size=256,
    selection_metric="f1_macro",
    seed=0,
    backbone_metadata=None,
    context=None,
    show_progress=False,
):
    """Train a grid of linear probes on one frozen feature bank.

    The frozen backbone features are expected to be computed once outside this
    helper. Each ``lr``/``weight_decay`` trial trains one linear classifier head
    to the maximum checkpoint epoch, then evaluates checkpoint epochs and
    thresholds without retraining earlier epochs.
    """

    thresholds = thresholds if thresholds is not None else [0.05, 0.1, 0.2, 0.3, 0.5]
    raw_configs = _linear_probe_grid_configs(train_grid)
    checkpoint_epochs = _checkpoint_epochs(raw_configs, checkpoint_epochs)
    configs = _dedupe_linear_probe_train_configs(raw_configs)

    train_features = feature_bank["train_features"]
    train_labels = feature_bank["train_labels"]
    val_features = feature_bank["val_features"]
    val_labels = feature_bank["val_labels"]
    feature_dim = int(train_features.shape[1])
    num_classes = int(train_labels.shape[1])

    if class_names is None:
        class_names = feature_bank.get("class_names")
    if class_names is None:
        class_names = [f"class_{idx}" for idx in range(num_classes)]

    _reset_peak_memory(device)

    trial_frames = []
    history_frames = []
    probs_by_checkpoint = {}
    labels_by_checkpoint = {}
    total_head_train_seconds = 0.0
    total_checkpoint_eval_seconds = 0.0
    total_threshold_search_seconds = 0.0
    head_metadata = None

    config_progress = tqdm(
        list(enumerate(configs)),
        desc="Linear probe configs",
        leave=True,
        disable=not show_progress,
    )
    for trial_idx, config in config_progress:
        train_trial_id = f"train_trial_{trial_idx:03d}"
        train_context = {
            **(context or {}),
            "train_trial_id": train_trial_id,
            "lr": float(config["lr"]),
            "weight_decay": float(config["weight_decay"]),
        }
        if show_progress:
            config_progress.set_postfix(
                lr=float(config["lr"]),
                weight_decay=float(config["weight_decay"]),
            )

        _set_torch_seed(seed + trial_idx)
        model = get_feature_linear_probe(feature_dim, num_classes).to(device)
        head_metadata = model_metadata(model)

        train_start = time.perf_counter()
        history, checkpoints = train_feature_linear_probe_checkpoints(
            model,
            train_features,
            train_labels,
            val_features,
            val_labels,
            device,
            checkpoint_epochs=checkpoint_epochs,
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            batch_size=batch_size,
            show_progress=False,
        )
        train_wall_seconds = time.perf_counter() - train_start
        head_train_seconds = sum(row["train_epoch_seconds"] for row in history)
        val_eval_seconds = sum(row["val_eval_seconds"] for row in history)
        total_head_train_seconds += head_train_seconds
        total_checkpoint_eval_seconds += val_eval_seconds

        for epoch, checkpoint in checkpoints.items():
            checkpoint_id = f"{train_trial_id}_epoch_{epoch:03d}"
            checkpoint_context = {
                **train_context,
                "trial_id": checkpoint_id,
                "epoch": int(epoch),
                "epochs": int(epoch),
            }
            threshold_start = time.perf_counter()
            trial_df = threshold_metric_table(
                checkpoint["probs"],
                checkpoint["labels"],
                thresholds,
                context=checkpoint_context,
            )
            threshold_search_seconds = time.perf_counter() - threshold_start
            total_threshold_search_seconds += threshold_search_seconds

            trial_df["loss"] = checkpoint["metrics"]["loss"]
            trial_df["head_train_seconds"] = float(head_train_seconds)
            trial_df["train_wall_seconds"] = float(train_wall_seconds)
            trial_df["checkpoint_eval_seconds"] = float(checkpoint["val_eval_seconds"])
            trial_df["threshold_search_seconds"] = float(threshold_search_seconds)
            trial_df["head_total_params"] = int(head_metadata["total_params"])
            trial_df["head_trainable_params"] = int(head_metadata["trainable_params"])
            trial_frames.append(trial_df)

            probs_by_checkpoint[(train_trial_id, int(epoch))] = checkpoint["probs"]
            labels_by_checkpoint[(train_trial_id, int(epoch))] = checkpoint["labels"]

        history_frames.append(_with_context(pd.DataFrame(history), train_context))

        del model

    trials_df = pd.concat(trial_frames, ignore_index=True)
    history_df = pd.concat(history_frames, ignore_index=True)

    best = _best_row(trials_df, selection_metric)
    best_train_trial_id = best["train_trial_id"]
    best_trial_id = best["trial_id"]
    best_epoch = int(best["epoch"])
    best_threshold = float(best["threshold"])
    best_context = {
        **(context or {}),
        "train_trial_id": best_train_trial_id,
        "trial_id": best_trial_id,
        "epoch": best_epoch,
        "epochs": best_epoch,
        "lr": float(best["lr"]),
        "weight_decay": float(best["weight_decay"]),
        "threshold": best_threshold,
        "selection_metric": selection_metric,
    }
    per_class_df = per_class_metric_table(
        probs_by_checkpoint[(best_train_trial_id, best_epoch)],
        labels_by_checkpoint[(best_train_trial_id, best_epoch)],
        class_names=class_names,
        threshold=best_threshold,
        context=best_context,
    )

    backbone_total_params = 0
    if backbone_metadata is not None:
        backbone_total_params = int(backbone_metadata.get("total_params", 0))

    head_total_params = int(head_metadata["total_params"]) if head_metadata else 0
    head_trainable_params = int(head_metadata["trainable_params"]) if head_metadata else 0
    efficiency = efficiency_summary(
        adaptation_method="linear_probe",
        total_params=backbone_total_params + head_total_params,
        trainable_params=head_trainable_params,
        phase_seconds={
            "head_train_grid": total_head_train_seconds,
            "val_eval_grid": total_checkpoint_eval_seconds,
            "threshold_search_grid": total_threshold_search_seconds,
        },
        peak_gpu_memory_mb=_peak_memory_mb(device),
    )

    metadata = {
        **(context or {}),
        **efficiency,
        "selection_metric": selection_metric,
        "num_train_trials": len(configs),
        "num_epoch_checkpoints": len(checkpoint_epochs),
        "num_thresholds": len(thresholds),
        "num_total_metric_rows": len(configs) * len(checkpoint_epochs) * len(thresholds),
        "checkpoint_epochs": checkpoint_epochs,
        "backbone_total_params": backbone_total_params,
        "head_total_params": head_total_params,
        "head_trainable_params": head_trainable_params,
        "selected_trial_id": best_trial_id,
        "selected_train_trial_id": best_train_trial_id,
        "selected_epoch": best_epoch,
        "selected_epochs": best_epoch,
        "selected_lr": float(best["lr"]),
        "selected_weight_decay": float(best["weight_decay"]),
        "selected_threshold": best_threshold,
        "selected_metric_value": float(best[selection_metric]),
    }

    return {
        "history": history_df,
        "trials": trials_df,
        "summary": pd.DataFrame([best.to_dict()]),
        "per_class": per_class_df,
        "metadata": metadata,
    }


def run_linear_probe_experiment(
    model,
    train_loader,
    val_loader,
    device,
    class_names=None,
    epochs=3,
    lr=1e-3,
    weight_decay=0.0,
    threshold=0.5,
    context=None,
):
    """Train and evaluate one frozen-backbone linear probe.

    Returns a dictionary with:
        - ``history``: one row per epoch,
        - ``summary``: one final validation row,
        - ``per_class``: final validation metrics per class,
        - ``metadata``: parameter counts and phase-aware runtime fields.
    """

    _reset_peak_memory(device)
    metadata = model_metadata(model)

    train_start = time.perf_counter()
    history = train_linear_probe(
        model,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    train_loop_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    criterion = nn.BCEWithLogitsLoss()
    metrics, probs, labels, ids = evaluate_model(model, val_loader, device, criterion)
    loss = metrics["loss"]
    metrics = classification_metrics(probs, labels, threshold=threshold)
    metrics["loss"] = loss
    final_eval_seconds = time.perf_counter() - eval_start

    efficiency = efficiency_summary(
        adaptation_method="linear_probe",
        total_params=metadata["total_params"],
        trainable_params=metadata["trainable_params"],
        phase_seconds={
            "train_loop": train_loop_seconds,
            "final_eval": final_eval_seconds,
        },
        peak_gpu_memory_mb=_peak_memory_mb(device),
    )

    summary_row = {
        "adaptation_method": "linear_probe",
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "threshold": threshold,
        "loss": metrics["loss"],
        "mean_auc": metrics["mean_auc"],
        "mean_accuracy": metrics["mean_accuracy"],
        "exact_match_accuracy": metrics["exact_match_accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_micro": metrics["f1_micro"],
    }

    if class_names is None:
        class_names = [f"class_{idx}" for idx in range(labels.shape[1])]

    preds = (probs >= threshold).float()
    per_class_rows = []
    for class_idx, class_name in enumerate(class_names):
        per_class_rows.append(
            {
                "class_name": class_name,
                "true_positive_rate": float(labels[:, class_idx].mean().item()),
                "predicted_positive_rate": float(preds[:, class_idx].mean().item()),
                "accuracy": metrics["accuracy_per_class"][class_idx],
                "f1": metrics["f1_per_class"][class_idx],
                "auc": metrics["auc_per_class"][class_idx],
            }
        )

    return {
        "history": _with_context(pd.DataFrame(history), context),
        "summary": _with_context(pd.DataFrame([summary_row]), context),
        "per_class": _with_context(pd.DataFrame(per_class_rows), context),
        "metadata": {**(context or {}), **efficiency},
        "probs": probs,
        "labels": labels,
        "ids": ids,
    }


def knn_predict(
    train_features,
    train_labels,
    query_features,
    k=20,
    batch_size=256,
    device=None,
    show_progress=False,
):
    if k < 1:
        raise ValueError("k must be at least 1.")

    if device is None:
        device = train_features.device

    k = min(k, train_features.shape[0])
    train_features = F.normalize(train_features.float(), dim=1).to(device)
    train_labels = train_labels.float().to(device)

    all_probs = []
    batch_starts = range(0, query_features.shape[0], batch_size)
    progress = tqdm(
        batch_starts,
        desc="kNN query batches",
        leave=False,
        disable=not show_progress,
    )
    for start in progress:
        query_batch = query_features[start : start + batch_size]
        query_batch = F.normalize(query_batch.float(), dim=1).to(device)

        similarities = query_batch @ train_features.T
        neighbor_indices = similarities.topk(k=k, dim=1).indices
        neighbor_labels = train_labels[neighbor_indices]
        probs = neighbor_labels.mean(dim=1)
        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)


def knn_predict_for_k_values(
    train_features,
    train_labels,
    query_features,
    k_values,
    batch_size=256,
    device=None,
    show_progress=False,
):
    """Predict kNN probabilities for several ``k`` values in one top-k pass."""

    if device is None:
        device = train_features.device

    k_values = sorted({int(k) for k in k_values})
    if not k_values:
        raise ValueError("At least one k value is required.")
    if min(k_values) < 1:
        raise ValueError("All k values must be at least 1.")

    max_available_k = int(train_features.shape[0])
    effective_k_values = sorted({min(k, max_available_k) for k in k_values})
    max_k = max(effective_k_values)

    train_features = F.normalize(train_features.float(), dim=1).to(device)
    train_labels = train_labels.float().to(device)

    probs_by_k = {k: [] for k in effective_k_values}
    batch_starts = range(0, query_features.shape[0], batch_size)
    progress = tqdm(
        batch_starts,
        desc="kNN query batches",
        leave=False,
        disable=not show_progress,
    )
    for start in progress:
        query_batch = query_features[start : start + batch_size]
        query_batch = F.normalize(query_batch.float(), dim=1).to(device)

        similarities = query_batch @ train_features.T
        neighbor_indices = similarities.topk(k=max_k, dim=1).indices
        neighbor_labels = train_labels[neighbor_indices]
        cumulative_neighbor_labels = neighbor_labels.cumsum(dim=1)

        for k in effective_k_values:
            probs = cumulative_neighbor_labels[:, k - 1, :] / k
            probs_by_k[k].append(probs.cpu())

    return {k: torch.cat(parts, dim=0) for k, parts in probs_by_k.items()}


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
    thresholds=None,
    batch_size=256,
    device=None,
    context=None,
    show_progress=False,
):
    if settings is None:
        settings = DEFAULT_KNN_FEWSHOT_SETTINGS
    if seeds is None:
        seeds = list(range(10))
    if class_names is None:
        class_names = [f"class_{idx}" for idx in range(val_labels.shape[1])]
    if thresholds is None:
        thresholds = [threshold]
    thresholds = [float(value) for value in thresholds]
    if not thresholds:
        raise ValueError("At least one threshold is required.")

    result_rows = []
    per_class_full_rows = []
    n_total = train_features.shape[0]

    start = time.perf_counter()
    knn_query_seconds = 0.0
    threshold_search_seconds_total = 0.0

    settings_to_run = filter_knn_fewshot_settings(settings, n_total)
    grouped_settings = []
    grouped_lookup = {}
    for config in settings_to_run:
        key = (config["setting"], config["n_train"])
        if key not in grouped_lookup:
            grouped_lookup[key] = {
                "setting": config["setting"],
                "n_train": config["n_train"],
                "k_values": [],
            }
            grouped_settings.append(grouped_lookup[key])
        grouped_lookup[key]["k_values"].append(int(config["k"]))

    progress = tqdm(
        grouped_settings,
        desc="kNN reference sets",
        leave=True,
        disable=not show_progress,
    )
    for config_group in progress:
        setting = config_group["setting"]
        n_train = config_group["n_train"]
        run_seeds = ["full"] if n_train is None else seeds

        if show_progress:
            progress.set_postfix(setting=setting)

        for seed in run_seeds:
            if n_train is None:
                indices = torch.arange(n_total)
            else:
                indices = sample_feature_indices(n_total, n_train, seed)

            subset_features = train_features[indices]
            subset_labels = train_labels[indices]
            n_train_actual = subset_features.shape[0]
            k_values = sorted({min(k, n_train_actual) for k in config_group["k_values"]})

            query_start = time.perf_counter()
            probs_by_k = knn_predict_for_k_values(
                train_features=subset_features,
                train_labels=subset_labels,
                query_features=val_features,
                k_values=k_values,
                batch_size=batch_size,
                device=device,
                show_progress=False,
            )
            knn_query_seconds += time.perf_counter() - query_start
            positive_counts = subset_labels.sum(dim=0)

            for k_effective, probs in probs_by_k.items():
                threshold_start = time.perf_counter()
                for threshold_value in thresholds:
                    positive_neighbors_needed = math.ceil(k_effective * threshold_value)
                    metrics = classification_metrics(
                        probs,
                        val_labels,
                        threshold=threshold_value,
                    )
                    preds = (probs >= threshold_value).float()

                    result_rows.append(
                        {
                            "setting": setting,
                            "seed": seed,
                            "n_train": n_train_actual,
                            "k": k_effective,
                            "threshold": threshold_value,
                            "positive_neighbors_needed": positive_neighbors_needed,
                            "neighbor_fraction": k_effective / n_train_actual,
                            "classes_with_positive_train_examples": int(
                                (positive_counts > 0).sum().item()
                            ),
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
                                    "setting": setting,
                                    "seed": seed,
                                    "n_train": n_train_actual,
                                    "k": k_effective,
                                    "threshold": threshold_value,
                                    "positive_neighbors_needed": positive_neighbors_needed,
                                    "class_name": class_name,
                                    "true_positive_rate": float(
                                        val_labels[:, class_idx].mean().item()
                                    ),
                                    "predicted_positive_rate": float(
                                        preds[:, class_idx].mean().item()
                                    ),
                                    "accuracy": metrics["accuracy_per_class"][class_idx],
                                    "f1": metrics["f1_per_class"][class_idx],
                                    "auc": metrics["auc_per_class"][class_idx],
                                }
                            )

                threshold_search_seconds_total += time.perf_counter() - threshold_start

    knn_eval_seconds = time.perf_counter() - start

    runs_df = _with_context(pd.DataFrame(result_rows), context)
    summary_df = summarize_knn_runs(pd.DataFrame(result_rows), context=context)
    per_class_full_df = _with_context(pd.DataFrame(per_class_full_rows), context)
    metadata = {
        "knn_eval_seconds": float(knn_eval_seconds),
        "knn_query_seconds": float(knn_query_seconds),
        "threshold_search_seconds": float(threshold_search_seconds_total),
        "num_knn_settings": len(settings_to_run),
        "num_reference_set_groups": len(grouped_settings),
        "num_thresholds": len(thresholds),
        "thresholds": thresholds,
    }

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
