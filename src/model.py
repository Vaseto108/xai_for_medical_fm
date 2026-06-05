import time

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import AutoModel


def _pool_dino_outputs(outputs):
    features = getattr(outputs, "pooler_output", None)
    if features is None:
        features = outputs.last_hidden_state[:, 0]
    return features


class DinoClassifier(nn.Module):
    def __init__(self, num_classes, model_name="facebook/dinov2-small", freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _backbone_is_frozen(self):
        return not any(param.requires_grad for param in self.backbone.parameters())

    def extract_features(self, images):
        if self._backbone_is_frozen():
            with torch.no_grad():
                outputs = self.backbone(pixel_values=images)
        else:
            outputs = self.backbone(pixel_values=images)
        return _pool_dino_outputs(outputs)

    def forward(self, images):
        features = self.extract_features(images)
        return self.classifier(features)


class FeatureLinearProbe(nn.Module):
    """Linear multi-label classifier trained on frozen backbone features."""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.classifier(features)


def get_dino_model(num_classes, model_name="facebook/dinov2-small", freeze_backbone=True):
    return DinoClassifier(
        num_classes=num_classes,
        model_name=model_name,
        freeze_backbone=freeze_backbone,
    )


def get_feature_linear_probe(feature_dim, num_classes):
    return FeatureLinearProbe(feature_dim=feature_dim, num_classes=num_classes)


def get_dino_backbone(model_name="facebook/dinov2-small", freeze=True):
    model = AutoModel.from_pretrained(model_name)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model


def unfreeze_last_blocks(model, num_blocks=1, train_classifier=True):
    """Planned helper for partial fine-tuning of DINO-style models.

    The future implementation should freeze the backbone by default, then
    unfreeze the last ``num_blocks`` transformer blocks. If ``model`` is a
    DinoClassifier, ``train_classifier=True`` should keep its linear head
    trainable.

    Returns:
        The same model with updated ``requires_grad`` flags.
    """

    raise NotImplementedError(
        "Partial fine-tuning is planned but unfreeze_last_blocks is not implemented yet."
    )


def apply_lora_adapters(
    model,
    target_modules=None,
    rank=8,
    alpha=16,
    dropout=0.0,
):
    """Planned PEFT/LoRA modification helper for DINO-style backbones.

    TODO:
    - choose target modules for DINOv2/DINOv3/RAD-DINO attention projections,
    - add PEFT/LoRA adapters without changing the classifier interface,
    - leave only adapter parameters and the classifier head trainable.

    Returns:
        The model with LoRA adapters attached.
    """

    raise NotImplementedError("LoRA adapters are planned but not implemented yet.")


def get_probs(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            images = batch["images"].to(device)
            logits = model(images)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(batch["labels"].float().cpu())
            all_ids.extend(list(batch["ids"]))

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return probs, labels, all_ids


def get_features(model, loader, device, normalize=True):
    model.eval()
    all_features = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features", leave=False):
            images = batch["images"].to(device)

            if hasattr(model, "extract_features"):
                features = model.extract_features(images)
            else:
                outputs = model(pixel_values=images)
                features = _pool_dino_outputs(outputs)

            if normalize:
                features = F.normalize(features, dim=1)

            all_features.append(features.cpu())
            all_labels.append(batch["labels"].float().cpu())
            all_ids.extend(list(batch["ids"]))

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels, all_ids


def model_metadata(model):
    return {
        "total_params": count_total_params(model),
        "trainable_params": count_trainable_params(model),
    }


def _is_cuda_device(device):
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


def extract_feature_bank(model, train_loader, val_loader, device, class_names=None, normalize=True):
    if _is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    train_features, train_labels, train_ids = get_features(
        model,
        train_loader,
        device,
        normalize=normalize,
    )
    train_feature_seconds = time.perf_counter() - start

    start = time.perf_counter()
    val_features, val_labels, val_ids = get_features(
        model,
        val_loader,
        device,
        normalize=normalize,
    )
    val_feature_seconds = time.perf_counter() - start

    peak_gpu_memory_mb = None
    if _is_cuda_device(device):
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    feature_bank = {
        "train_features": train_features,
        "train_labels": train_labels,
        "train_ids": train_ids,
        "val_features": val_features,
        "val_labels": val_labels,
        "val_ids": val_ids,
        "class_names": class_names,
    }
    metadata = {
        "feature_dim": int(train_features.shape[1]),
        "train_feature_seconds": float(train_feature_seconds),
        "val_feature_seconds": float(val_feature_seconds),
        "feature_extraction_seconds": float(train_feature_seconds + val_feature_seconds),
        "peak_gpu_memory_mb": None if peak_gpu_memory_mb is None else float(peak_gpu_memory_mb),
    }
    return feature_bank, metadata


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_total_params(model):
    return sum(param.numel() for param in model.parameters())
