import torch
from torch import nn
from tqdm.auto import tqdm

from src.eval import classification_metrics


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
            progress.set_postfix(train_loss=total_loss / n_samples)

        train_loss = total_loss / n_samples
        val_metrics, _, _, _ = evaluate_model(model, val_loader, device, criterion)

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mean_auc": val_metrics["mean_auc"],
            "val_auc_per_class": val_metrics["auc_per_class"],
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_mean_auc={val_metrics['mean_auc']:.4f}"
        )

    return history
