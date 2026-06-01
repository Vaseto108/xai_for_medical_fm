import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.eval import classification_metrics
from src.model import get_features


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


def knn_predict(train_features, train_labels, query_features, k=20, batch_size=256, device=None):
    if k < 1:
        raise ValueError("k must be at least 1.")

    if device is None:
        device = train_features.device

    k = min(k, train_features.shape[0])
    train_features = F.normalize(train_features.float(), dim=1).to(device)
    train_labels = train_labels.float().to(device)

    all_probs = []
    for start in tqdm(range(0, query_features.shape[0], batch_size), desc=f"kNN k={k}", leave=False):
        query_batch = query_features[start : start + batch_size]
        query_batch = F.normalize(query_batch.float(), dim=1).to(device)

        similarities = query_batch @ train_features.T
        neighbor_indices = similarities.topk(k=k, dim=1).indices
        neighbor_labels = train_labels[neighbor_indices]
        probs = neighbor_labels.mean(dim=1)
        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)


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
