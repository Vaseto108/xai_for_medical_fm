import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoModel


class DinoClassifier(nn.Module):
    def __init__(self, num_classes, model_name="facebook/dinov2-small", freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images):
        outputs = self.backbone(pixel_values=images)
        features = getattr(outputs, "pooler_output", None)
        if features is None:
            features = outputs.last_hidden_state[:, 0]
        return self.classifier(features)


def get_dino_model(num_classes, model_name="facebook/dinov2-small", freeze_backbone=True):
    return DinoClassifier(
        num_classes=num_classes,
        model_name=model_name,
        freeze_backbone=freeze_backbone,
    )


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


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_total_params(model):
    return sum(param.numel() for param in model.parameters())
