import numpy as np
import torch
import matplotlib.pyplot as plt


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _normalize_heatmap(heatmap):
    heatmap = heatmap - heatmap.min()
    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


def make_heatmap(model, image, target_class, device):
    model.eval()
    image = image.detach().to(device).unsqueeze(0)
    image.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = model(image)
        score = logits[0, target_class]
        score.backward()

    heatmap = image.grad.detach().abs().mean(dim=1)[0]
    heatmap = _normalize_heatmap(heatmap)
    return heatmap.cpu()


def make_gradcam_heatmap(model, image, target_class, target_layer=None, device=None):
    model.eval()
    image = image.unsqueeze(0).to(device)

    # Default: hook the last transformer block
    if target_layer is None:
        target_layer = model.backbone.blocks[-1].norm1

    gradients, activations = [], []

    fh = target_layer.register_forward_hook(lambda m, i, o: activations.append(o.detach()))
    bh = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0].detach()))

    logits = model(image)
    model.zero_grad()
    logits[0, target_class].backward()

    fh.remove()
    bh.remove()

    grad = gradients[0]   # (1, seq_len, dim)
    act  = activations[0] # (1, seq_len, dim)

    weights = grad.mean(dim=-1, keepdim=True)      # importance per token
    cam = (weights * act).sum(dim=-1)              # (1, seq_len)
    cam = cam[:, 1:]                               # drop CLS token

    grid = int(cam.shape[1] ** 0.5)
    cam = cam.reshape(1, 1, grid, grid)

    H, W = image.shape[-2:]
    cam = torch.nn.functional.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
    cam = cam.squeeze()
    cam = torch.clamp(cam, min=0)
    return _normalize_heatmap(cam.cpu())


def make_occlusion_heatmap(model, image, target_class, device=None, patch_size=16, stride=8, baseline=0.0):
    model.eval()
    C, H, W = image.shape
    img_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        baseline_score = torch.sigmoid(model(img_batch))[0, target_class].item()

    score_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = img_batch.clone()
            occluded[:, :, y:y + patch_size, x:x + patch_size] = baseline

            with torch.no_grad():
                score = torch.sigmoid(model(occluded))[0, target_class].item()

            drop = baseline_score - score          # positive = region mattered
            score_map[y:y + patch_size, x:x + patch_size] += drop
            count_map[y:y + patch_size, x:x + patch_size] += 1

    count_map = np.maximum(count_map, 1)
    score_map /= count_map
    score_map = torch.tensor(score_map)
    return _normalize_heatmap(score_map)           # reuses existing helper


def make_attention_rollout(model, image, device=None, head_fusion="mean", discard_ratio=0.0):
    model.eval()
    image = image.unsqueeze(0).to(device)

    attention_maps = []

    def hook_fn(module, input, output):
        # DINOv2 attention blocks return (attn_output, attn_weights) or just attn_output
        # depending on the variant — adjust if needed
        attention_maps.append(output.detach())

    hooks = []
    for block in model.backbone.blocks:
        hooks.append(block.attn.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(image)

    for h in hooks:
        h.remove()

    # Rollout
    result = torch.eye(attention_maps[0].shape[-1], device=device)
    for attn in attention_maps:
        if head_fusion == "mean":
            attn = attn.mean(dim=1)
        elif head_fusion == "max":
            attn = attn.max(dim=1).values
        elif head_fusion == "min":
            attn = attn.min(dim=1).values

        # Discard low-attention tokens
        flat = attn.view(attn.shape[0], -1)
        threshold = flat.quantile(discard_ratio, dim=-1, keepdim=True)
        attn = torch.where(attn >= threshold.unsqueeze(-1), attn, torch.zeros_like(attn))

        # Add residual and renormalize
        attn = attn + torch.eye(attn.shape[-1], device=device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        result = attn[0] @ result

    # CLS token row → attention from CLS to all patches
    mask = result[0, 1:]                           # drop CLS-to-CLS
    grid = int(mask.shape[0] ** 0.5)
    mask = mask.reshape(1, 1, grid, grid)

    H, W = image.shape[-2:]
    mask = torch.nn.functional.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
    mask = mask.squeeze()
    return _normalize_heatmap(mask.cpu())


def dice_score(heatmap, mask, threshold=0.5, eps=1e-7):
    pred = (heatmap >= threshold).float()
    mask = mask.float()
    intersection = (pred * mask).sum()
    return (2 * intersection + eps) / (pred.sum() + mask.sum() + eps)


def iou_score(heatmap, mask, threshold=0.5, eps=1e-7):
    pred = (heatmap >= threshold).float()
    mask = mask.float()
    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum() - intersection
    return (intersection + eps) / (union + eps)


def pointing_game_score(heatmap, mask):
    peak = heatmap.argmax()                        # flat index of max activation
    peak_coords = torch.unravel_index(peak, heatmap.shape)
    return float(mask[peak_coords].item() > 0)


def _image_for_display(image):
    if hasattr(image, "detach"):
        image = image.detach().cpu()
    else:
        image = torch.as_tensor(image)

    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def _heatmap_for_display(heatmap):
    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach().cpu().numpy()
    else:
        heatmap = np.asarray(heatmap)
    return np.squeeze(heatmap)


def show_heatmap(image, heatmap):
    image = _image_for_display(image)
    heatmap = _heatmap_for_display(heatmap)

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="magma", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
