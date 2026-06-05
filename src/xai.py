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


def dice_score(heatmap, mask, threshold=0.5, eps=1e-7):
    """Planned Dice score between a thresholded heatmap and binary mask.

    Args:
        heatmap: Explanation map, typically Tensor[H, W] or Tensor[B, H, W].
        mask: Ground-truth localization mask with a broadcast-compatible shape.
        threshold: Heatmap cutoff used to create a binary prediction mask.
        eps: Small value to avoid division by zero.

    Returns:
        Float Dice score.
    """

    raise NotImplementedError("Dice scoring for XAI masks is planned but not implemented yet.")


def iou_score(heatmap, mask, threshold=0.5, eps=1e-7):
    """Planned IoU score between a thresholded heatmap and binary mask.

    Args:
        heatmap: Explanation map, typically Tensor[H, W] or Tensor[B, H, W].
        mask: Ground-truth localization mask with a broadcast-compatible shape.
        threshold: Heatmap cutoff used to create a binary prediction mask.
        eps: Small value to avoid division by zero.

    Returns:
        Float intersection-over-union score.
    """

    raise NotImplementedError("IoU scoring for XAI masks is planned but not implemented yet.")


def pointing_game_score(heatmap, mask):
    """Planned pointing-game score for XAI localization.

    The future implementation should check whether the maximum heatmap location
    falls inside the ground-truth mask for the explained class.

    Args:
        heatmap: Explanation map, typically Tensor[H, W].
        mask: Binary ground-truth mask aligned to the heatmap.

    Returns:
        1.0 for a hit, 0.0 for a miss, or an aggregate score for batches.
    """

    raise NotImplementedError(
        "Pointing-game scoring for XAI masks is planned but not implemented yet."
    )


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
