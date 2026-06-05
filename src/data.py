import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from medmnist import ChestMNIST, INFO
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency: medmnist. Install the project requirements in the "
        "same environment as your notebook kernel. From the notebook, run "
        "`%pip install -r ../requirements.txt`; from the project root, run "
        "`python -m pip install -r requirements.txt`."
    ) from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DictChestMNIST(Dataset):
    def __init__(self, dataset, split_name, indices=None, max_items=None):
        self.dataset = dataset
        if indices is None:
            indices = list(range(len(dataset)))

        if max_items is not None and max_items < 0:
            raise ValueError("max_items must be None or a non-negative integer.")

        if max_items is not None:
            indices = indices[:max_items]

        self.indices = list(indices)
        self.split_name = split_name

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        source_idx = self.indices[idx]
        image, label = self.dataset[source_idx]

        return {
            "images": image,
            "labels": torch.as_tensor(label, dtype=torch.float32).view(-1),
            "ids": f"{self.split_name}_{source_idx}",
        }


class CheXpertDataset(Dataset):
    """Planned CheXpert multi-label chest X-ray dataset.

    Expected item format for classification:
        {
            "images": Tensor[3, H, W],
            "labels": Tensor[C],
            "ids": str,
        }

    TODO:
    - read the official CheXpert CSV metadata,
    - resolve image paths relative to the dataset root,
    - map uncertain labels consistently with the experiment plan,
    - apply the same ImageNet normalization used for DINO-style backbones.
    """

    def __init__(
        self,
        root,
        csv_path,
        label_columns=None,
        image_size=224,
        transform=None,
    ):
        raise NotImplementedError(
            "CheXpertDataset is planned for the final CheXpert experiments."
        )

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class CheXlocalizeDataset(Dataset):
    """Planned CheXlocalize dataset for XAI/localization evaluation.

    Expected item format:
        {
            "images": Tensor[3, H, W],
            "labels": Tensor[C],
            "ids": str,
            "masks": Tensor[C, H, W] or Tensor[1, H, W],
        }

    TODO:
    - read image-level labels and localization mask metadata,
    - align mask class order with the classifier class names,
    - resize masks to the heatmap/image resolution needed by XAI metrics.
    """

    def __init__(
        self,
        root,
        metadata_path,
        mask_root=None,
        label_columns=None,
        image_size=224,
        transform=None,
        mask_transform=None,
    ):
        raise NotImplementedError(
            "CheXlocalizeDataset is planned for future XAI mask evaluation."
        )

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


def _class_names():
    label_info = INFO["chestmnist"]["label"]
    if isinstance(label_info, dict):
        return [label_info[key] for key in sorted(label_info, key=lambda x: int(x))]
    return list(label_info)


def _fraction_indices(n_items, data_fraction, seed):
    if data_fraction is None:
        data_fraction = 1.0

    if data_fraction <= 0 or data_fraction > 1:
        raise ValueError("data_fraction must be in the interval (0, 1].")

    if data_fraction == 1.0:
        return list(range(n_items))

    n_selected = max(1, int(round(n_items * data_fraction)))
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n_items, generator=generator)[:n_selected].tolist()


def get_small_data(
    n_train=None,
    n_val=None,
    batch_size=8,
    image_size=224,
    data_fraction=1.0,
    seed=0,
):
    if data_fraction != 1.0 and (n_train is not None or n_val is not None):
        raise ValueError("Use either data_fraction or n_train/n_val caps, not both.")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_dataset = ChestMNIST(
        split="train",
        transform=transform,
        download=True,
        as_rgb=True,
    )
    val_dataset = ChestMNIST(
        split="val",
        transform=transform,
        download=True,
        as_rgb=True,
    )

    train_indices = _fraction_indices(len(train_dataset), data_fraction, seed=seed)
    val_indices = _fraction_indices(len(val_dataset), data_fraction, seed=seed + 1)

    train_dataset = DictChestMNIST(
        train_dataset,
        split_name="train",
        indices=train_indices,
        max_items=n_train,
    )
    val_dataset = DictChestMNIST(
        val_dataset,
        split_name="val",
        indices=val_indices,
        max_items=n_val,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, _class_names()


def get_chexpert_data(
    root,
    train_csv=None,
    val_csv=None,
    batch_size=8,
    image_size=224,
    label_columns=None,
    num_workers=0,
):
    """Planned CheXpert train/validation loaders for final classification.

    Returns:
        train_loader, val_loader, class_names

    Classification batches should keep the shared project format:
        {
            "images": Tensor[B, 3, H, W],
            "labels": Tensor[B, C],
            "ids": list[str],
        }
    """

    raise NotImplementedError(
        "CheXpert loading is not implemented yet. Use get_small_data for ChestMNIST."
    )


def get_chexlocalize_data(
    root,
    metadata_path=None,
    mask_root=None,
    batch_size=8,
    image_size=224,
    label_columns=None,
    num_workers=0,
):
    """Planned CheXlocalize loader for XAI mask evaluation.

    Returns:
        loader, class_names

    Localization batches should include the classification fields and may add:
        {
            "masks": Tensor[B, C, H, W] or Tensor[B, 1, H, W],
        }
    """

    raise NotImplementedError(
        "CheXlocalize loading is not implemented yet. Use this stub as the target interface."
    )
