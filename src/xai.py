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
