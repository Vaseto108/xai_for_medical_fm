import numpy as np
import torch
import matplotlib.pyplot as plt
import time


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
    """Planned Grad-CAM heatmap for a target class.

    Args:
        model: Classification model that produces logits from images.
        image: Single normalized image tensor with shape [3, H, W].
        target_class: Integer class index to explain.
        target_layer: Model layer/module to hook for Grad-CAM activations.
        device: Optional torch device for the forward/backward pass.

    Returns:
        Tensor[H, W] normalized to [0, 1].
    """

    raise NotImplementedError("Grad-CAM heatmap generation is planned but not implemented yet.")


def make_occlusion_heatmap(
    model,
    image,
    target_class,
    device=None,
    patch_size=16,
    stride=8,
    baseline=0.0,
):
    """Planned occlusion sensitivity heatmap for a target class.

    The future implementation should slide an occluding patch over the image and
    measure the target-class score change at each location.

    Returns:
        Tensor[H, W] normalized to [0, 1].
    """

    raise NotImplementedError(
        "Occlusion sensitivity heatmap generation is planned but not implemented yet."
    )


def make_attention_rollout(
    model,
    image,
    device=None,
    head_fusion="mean",
    discard_ratio=0.0,
):
    """Planned attention rollout heatmap for ViT/DINO-style backbones.

    The future implementation should aggregate transformer attention maps into a
    spatial explanation aligned to the input image.

    Returns:
        Tensor[H, W] normalized to [0, 1].
    """

    raise NotImplementedError("Attention rollout is planned but not implemented yet.")


def dice_score(heatmap: torch.Tensor, mask: torch.Tensor, threshold=0.5, eps=1e-7) -> float:
    """Dice coefficient between thresholded heatmap and binary mask."""
    pred = (heatmap >= threshold).float()
    gt = (mask > 0).float()
    intersection = (pred * gt).sum()
    return float((2 * intersection / (pred.sum() + gt.sum() + eps)).item())


def iou_score(heatmap: torch.Tensor, mask: torch.Tensor, threshold=0.5, eps=1e-7) -> float:
    """IoU between thresholded heatmap and binary mask."""
    pred = (heatmap >= threshold).float()
    gt = (mask > 0).float()
    intersection = (pred * gt).sum()
    union = (pred + gt).clamp(max=1).sum()
    return float((intersection / (union + eps)).item())


def pointing_game_score(heatmap: torch.Tensor, mask: torch.Tensor) -> float:
    """Pointing Game: hit if the argmax pixel falls inside the binary mask.

    Args:
        heatmap: Tensor[H, W], values in [0, 1].
        mask: Binary Tensor[H, W] (1 = annotated region).

    Returns:
        1.0 for a hit, 0.0 for a miss.
    """
    heatmap = heatmap.float()
    mask = (mask > 0).float()
    idx = heatmap.argmax()
    row, col = idx // heatmap.shape[1], idx % heatmap.shape[1]
    return float(mask[row, col].item())


def deletion_auc(
    model,
    image,
    heatmap,
    target_class,
    device,
    n_steps=100,
    baseline_value=0.0,
):
    """Deletion AUC faithfulness metric.

    Removes pixels in descending saliency order, measures the target-class
    probability at each step, and returns the area under that curve.
    Lower = more faithful (confidence drops faster).

    Args:
        model: Classification model.
        image: Tensor[3, H, W], normalized.
        heatmap: Tensor[H, W] saliency map aligned to image.
        target_class: Integer class index.
        device: torch device.
        n_steps: Number of deletion steps (default 100).
        baseline_value: Scalar fill value (default 0.0 = normalized mean).

    Returns:
        Float AUC value.
    """
    model.eval()
    _, H, W = image.shape
    n_pixels = H * W

    flat_saliency = heatmap.view(-1)
    sorted_indices = flat_saliency.argsort(descending=True)

    step_fractions = np.linspace(0, 1, n_steps + 1)
    scores = []

    inp = image.clone().to(device)

    with torch.no_grad():
        for frac in step_fractions:
            n_removed = int(frac * n_pixels)
            masked = inp.clone()
            if n_removed > 0:
                remove_idx = sorted_indices[:n_removed]
                rows = remove_idx // W
                cols = remove_idx % W
                masked[:, rows, cols] = baseline_value
            prob = torch.sigmoid(model(masked.unsqueeze(0)))[0, target_class].item()
            scores.append(prob)

    return float(np.trapz(scores, step_fractions))


def insertion_auc(
    model,
    image,
    heatmap,
    target_class,
    device,
    n_steps=100,
    baseline_value=0.0,
):
    """Insertion AUC (complementary to Deletion AUC).

    Reveals pixels in descending saliency order from a fully masked baseline.
    Higher = more faithful.

    Returns:
        Float AUC value.
    """
    model.eval()
    _, H, W = image.shape
    n_pixels = H * W

    flat_saliency = heatmap.view(-1)
    sorted_indices = flat_saliency.argsort(descending=True)

    step_fractions = np.linspace(0, 1, n_steps + 1)
    scores = []

    inp = image.clone().to(device)

    with torch.no_grad():
        for frac in step_fractions:
            n_revealed = int(frac * n_pixels)
            masked = torch.full_like(inp, baseline_value)
            if n_revealed > 0:
                reveal_idx = sorted_indices[:n_revealed]
                rows = reveal_idx // W
                cols = reveal_idx % W
                masked[:, rows, cols] = inp[:, rows, cols]
            prob = torch.sigmoid(model(masked.unsqueeze(0)))[0, target_class].item()
            scores.append(prob)

    return float(np.trapz(scores, step_fractions))


def max_sensitivity(
    model,
    image,
    target_class,
    heatmap_fn,
    device,
    epsilon=0.1,
    n_perturbations=20,
    seed=0,
):
    """Max-Sensitivity stability metric.

    Estimates the maximum change in the saliency map under small input
    perturbations:
        Sens = max_{||delta|| < epsilon} ||S(x+delta) - S(x)|| / ||delta||

    Args:
        heatmap_fn: Callable(model, image, target_class, device) -> Tensor[H, W].
        epsilon: L-inf perturbation bound.
        n_perturbations: Number of random perturbations.

    Returns:
        Float max-sensitivity value.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    base_heatmap = heatmap_fn(model, image, target_class, device).float()

    max_sens = 0.0
    for _ in range(n_perturbations):
        delta = torch.empty_like(image).uniform_(-epsilon, epsilon, generator=rng)
        perturbed = (image + delta).clamp(0, 1)
        pert_heatmap = heatmap_fn(model, perturbed, target_class, device).float()

        heatmap_diff = (pert_heatmap - base_heatmap).norm()
        delta_norm = delta.norm()
        if delta_norm > 0:
            sens = (heatmap_diff / delta_norm).item()
            max_sens = max(max_sens, sens)

    return max_sens


def measure_heatmap_time(
    model,
    image,
    target_class,
    heatmap_fn,
    device,
    n_warmup=2,
    n_runs=5,
):
    """Measure wall-clock time for a single heatmap generation (ms).

    Args:
        heatmap_fn: Callable(model, image, target_class, device) -> Tensor[H, W].

    Returns:
        Float mean time in milliseconds.
    """
    for _ in range(n_warmup):
        heatmap_fn(model, image, target_class, device)

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        heatmap_fn(model, image, target_class, device)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times))


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
