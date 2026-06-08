import json
import hashlib
import os
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    from medmnist import ChestMNIST, INFO
except ModuleNotFoundError:
    ChestMNIST = None
    INFO = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CHEXPERT_CLASS_NAMES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

CHEXLOCALIZE_CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumothorax",
    "Support Devices",
]

CHEXLOCALIZE_TO_CHEXPERT_INDICES = [
    CHEXPERT_CLASS_NAMES.index(class_name)
    for class_name in CHEXLOCALIZE_CLASS_NAMES
]


def _resolve_root(root, environment_variable):
    root = root or os.environ.get(environment_variable)
    if not root:
        raise ValueError(
            f"Dataset path is not configured. Pass root=... or set the "
            f"{environment_variable} environment variable. Use "
            "scripts/download_datasets.py to create/check the expected local "
            "dataset layout."
        )
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def _default_image_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _validate_view_filter(view_filter):
    view_filter = str(view_filter).lower()
    if view_filter not in {"frontal", "all"}:
        raise ValueError("view_filter must be 'frontal' or 'all'.")
    return view_filter


def _filter_views(metadata, view_filter):
    view_filter = _validate_view_filter(view_filter)
    if view_filter == "all":
        return metadata

    path_is_frontal = (
        metadata["Path"].astype(str).str.lower().str.contains("frontal")
    )
    if "Frontal/Lateral" in metadata.columns:
        view_is_frontal = (
            metadata["Frontal/Lateral"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains("frontal")
        )
        is_frontal = view_is_frontal | path_is_frontal
    else:
        is_frontal = path_is_frontal
    return metadata.loc[is_frontal].reset_index(drop=True)


def _select_metadata_rows(metadata, data_fraction=1.0, seed=0, max_items=None):
    if data_fraction is None:
        data_fraction = 1.0
    if data_fraction != 1.0 and max_items is not None:
        raise ValueError("Use either data_fraction or max_items, not both.")
    indices = _fraction_indices(len(metadata), data_fraction, seed)
    metadata = metadata.iloc[indices].reset_index(drop=True)
    if max_items is not None:
        if max_items < 0:
            raise ValueError("max_items must be None or a non-negative integer.")
        metadata = metadata.iloc[:max_items].reset_index(drop=True)
    return metadata


def _validate_label_columns(metadata, label_columns):
    missing = [column for column in label_columns if column not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata is missing CheXpert label columns: {missing}")


def _u_zero_labels(row, label_columns):
    values = pd.to_numeric(row[label_columns], errors="coerce").fillna(0.0)
    return torch.as_tensor((values.to_numpy(dtype=np.float32) > 0).astype(np.float32))


def _path_parts(path_value):
    return PurePosixPath(str(path_value).replace("\\", "/")).parts


def _image_id_from_path(path_value):
    parts = _path_parts(path_value)
    if len(parts) < 3:
        return Path(parts[-1]).stem
    patient, study, filename = parts[-3:]
    return f"{patient}_{study}_{Path(filename).stem}"


def _patient_id_from_path(path_value):
    parts = _path_parts(path_value)
    for part in parts:
        if part.lower().startswith("patient"):
            return part
    if len(parts) >= 3:
        return parts[-3]
    return str(path_value)


def _normalize_split_name(split):
    split = str(split).strip().lower()
    aliases = {
        "valid": "val",
        "validation": "val",
        "dev": "val",
        "testing": "test",
        "training": "train",
    }
    return aliases.get(split, split)


def _deterministic_split_for_group(group_id, seed):
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big") / 2**64
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "val"
    return "test"


def _prepare_chexpert_512_metadata(metadata, seed):
    """Normalize saved HF metadata and ensure stable train/val/test splits.

    Official split information is used when it provides train, validation, and
    test data. Otherwise, patients are deterministically assigned to an 80/10/10
    train/validation/test split.
    """

    metadata = metadata.copy()
    if "Path" not in metadata.columns:
        raise ValueError("CheXpert-v1.0-512 metadata must contain a 'Path' column.")
    _validate_label_columns(metadata, CHEXPERT_CLASS_NAMES)

    usable_split = None
    for column in ["split", "source_split"]:
        if column not in metadata.columns:
            continue
        normalized = metadata[column].map(_normalize_split_name)
        available = set(normalized.dropna().unique())
        if {"train", "val", "test"}.issubset(available):
            usable_split = normalized
            break

    if usable_split is None:
        groups = metadata["Path"].map(_patient_id_from_path)
        usable_split = groups.map(
            lambda group_id: _deterministic_split_for_group(group_id, seed)
        )

    metadata["split"] = usable_split
    return metadata


def _resolve_image_path(root, path_value):
    parts = _path_parts(path_value)
    candidates = [root.joinpath(*parts)]
    if parts and parts[0].lower().startswith("chexpert"):
        candidates.append(root.joinpath(*parts[1:]))
    if "CheXpert" in root.parts and "CheXpert" in parts:
        chexpert_idx = parts.index("CheXpert")
        candidates.append(root.joinpath(*parts[chexpert_idx + 1 :]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve image '{path_value}' under dataset root '{root}'."
    )


def _resolve_existing_path(root, path, candidates, description):
    if path is not None:
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path
        raise FileNotFoundError(f"{description} does not exist: {path}")

    for candidate in candidates:
        candidate_path = root / candidate
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError(
        f"Could not find {description} under {root}. Tried: {list(candidates)}"
    )


def _chexpert_csv_path(root, split, csv_path=None):
    split = str(split).lower()
    aliases = {"validation": "val", "valid": "val"}
    split = aliases.get(split, split)
    candidates = {
        "train": [
            "train.csv",
            "CheXpert-v1.0/train.csv",
            "CheXpert-v1.0-small/train.csv",
        ],
        "val": [
            "valid.csv",
            "val.csv",
            "val_labels.csv",
            "CheXpert/val_labels.csv",
            "CheXpert-v1.0/valid.csv",
            "CheXpert-v1.0-small/valid.csv",
        ],
        "test": [
            "test.csv",
            "test_labels.csv",
            "CheXpert/test_labels.csv",
        ],
    }
    if split not in candidates:
        raise ValueError("split must be 'train', 'val'/'valid', or 'test'.")
    return _resolve_existing_path(
        root,
        csv_path,
        candidates[split],
        f"CheXpert {split} CSV",
    )


def _chexlocalize_metadata_path(root, split, metadata_path=None):
    return _resolve_existing_path(
        root,
        metadata_path,
        [
            f"CheXpert/{split}_labels.csv",
            f"{split}_labels.csv",
            f"{split}.csv",
            "valid.csv" if split == "val" else "test.csv",
        ],
        f"CheXlocalize {split} label CSV",
    )


def _chexlocalize_segmentation_path(root, split, segmentation_path=None):
    return _resolve_existing_path(
        root,
        segmentation_path,
        [
            f"CheXlocalize/gt_segmentations_{split}.json",
            f"gt_segmentations_{split}.json",
        ],
        f"CheXlocalize {split} segmentation JSON",
    )


def _decode_rle(rle):
    size = tuple(int(value) for value in rle["size"])
    counts = rle["counts"]
    if isinstance(counts, list):
        flat = np.zeros(int(np.prod(size)), dtype=np.uint8)
        offset = 0
        value = 0
        for length in counts:
            length = int(length)
            if value == 1:
                flat[offset : offset + length] = 1
            offset += length
            value = 1 - value
        return flat.reshape(size, order="F")

    try:
        from pycocotools import mask as coco_mask
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Compressed CheXlocalize RLE masks require pycocotools. "
            "Install the project requirements in the active environment."
        ) from exc

    encoded = {"size": list(size), "counts": counts}
    if isinstance(encoded["counts"], str):
        encoded["counts"] = encoded["counts"].encode("ascii")
    mask = coco_mask.decode(encoded)
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    return mask.astype(np.uint8)


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
    """CheXpert image-level classification dataset using the U-Zero policy."""

    def __init__(
        self,
        root,
        csv_path=None,
        metadata=None,
        label_columns=None,
        image_size=224,
        transform=None,
        view_filter="frontal",
        data_fraction=1.0,
        seed=0,
        max_items=None,
    ):
        self.root = Path(root)
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.label_columns = list(label_columns or CHEXPERT_CLASS_NAMES)
        self.class_names = list(self.label_columns)
        self.image_size = int(image_size)
        self.transform = transform or _default_image_transform(self.image_size)

        if metadata is None:
            if self.csv_path is None:
                raise ValueError("Provide csv_path or metadata to CheXpertDataset.")
            metadata = pd.read_csv(self.csv_path)
        else:
            metadata = metadata.copy()
        if "Path" not in metadata.columns:
            raise ValueError("CheXpert metadata must contain a 'Path' column.")
        _validate_label_columns(metadata, self.label_columns)
        metadata = _filter_views(metadata, view_filter)
        self.metadata = _select_metadata_rows(
            metadata,
            data_fraction=data_fraction,
            seed=seed,
            max_items=max_items,
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_path = _resolve_image_path(self.root, row["Path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return {
            "images": image,
            "labels": _u_zero_labels(row, self.label_columns),
            "ids": _image_id_from_path(row["Path"]),
        }


class CheXlocalizeDataset(Dataset):
    """CheXlocalize images and 10-class masks for post-hoc XAI evaluation."""

    def __init__(
        self,
        root,
        metadata_path,
        segmentation_path,
        label_columns=None,
        image_size=224,
        transform=None,
        view_filter="frontal",
        annotated_only=True,
        data_fraction=1.0,
        seed=0,
        max_items=None,
    ):
        self.root = Path(root)
        self.metadata_path = Path(metadata_path)
        self.segmentation_path = Path(segmentation_path)
        self.label_columns = list(label_columns or CHEXPERT_CLASS_NAMES)
        self.class_names = list(self.label_columns)
        self.mask_class_names = list(CHEXLOCALIZE_CLASS_NAMES)
        self.mask_to_label_indices = list(CHEXLOCALIZE_TO_CHEXPERT_INDICES)
        self.image_size = int(image_size)
        self.transform = transform or _default_image_transform(self.image_size)

        with open(self.segmentation_path, "r", encoding="utf-8") as handle:
            self.segmentations = json.load(handle)

        metadata = pd.read_csv(self.metadata_path)
        if "Path" not in metadata.columns:
            raise ValueError(
                f"CheXlocalize label CSV must contain a 'Path' column: {self.metadata_path}"
            )
        _validate_label_columns(metadata, self.label_columns)
        metadata = _filter_views(metadata, view_filter)
        metadata = metadata.copy()
        metadata["_image_id"] = metadata["Path"].map(_image_id_from_path)
        if annotated_only:
            metadata = metadata[metadata["_image_id"].isin(self.segmentations)]
        self.metadata = _select_metadata_rows(
            metadata.reset_index(drop=True),
            data_fraction=data_fraction,
            seed=seed,
            max_items=max_items,
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row["_image_id"]
        image_path = _resolve_image_path(self.root, row["Path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        image_segmentations = self.segmentations.get(image_id, {})
        masks = []
        for class_name in CHEXLOCALIZE_CLASS_NAMES:
            rle = image_segmentations.get(class_name)
            if rle is None:
                mask = torch.zeros((1, self.image_size, self.image_size))
            else:
                mask = torch.from_numpy(_decode_rle(rle)).float().unsqueeze(0)
                mask = TF.resize(
                    mask,
                    [self.image_size, self.image_size],
                    interpolation=InterpolationMode.NEAREST,
                )
            masks.append(mask[0])

        return {
            "images": image,
            "labels": _u_zero_labels(row, self.label_columns),
            "ids": image_id,
            "masks": torch.stack(masks),
        }


def _class_names():
    if INFO is None:
        raise ModuleNotFoundError(
            "Missing dependency: medmnist. Install the project requirements in the "
            "same environment as the notebook kernel."
        )
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
    if ChestMNIST is None:
        raise ModuleNotFoundError(
            "Missing dependency: medmnist. Install the project requirements in the "
            "same environment as the notebook kernel."
        )
    if data_fraction != 1.0 and (n_train is not None or n_val is not None):
        raise ValueError("Use either data_fraction or n_train/n_val caps, not both.")

    transform = _default_image_transform(image_size)

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
    root=None,
    train_csv=None,
    val_csv=None,
    batch_size=8,
    image_size=224,
    label_columns=None,
    num_workers=0,
    view_filter="frontal",
    data_fraction=1.0,
    seed=0,
    max_train=None,
    max_val=None,
):
    """Return CheXpert train/validation loaders for classification.

    Returns:
        train_loader, val_loader, class_names

    Classification batches should keep the shared project format:
        {
            "images": Tensor[B, 3, H, W],
            "labels": Tensor[B, C],
            "ids": list[str],
        }
    """

    root = _resolve_root(root, "CHEXPERT_ROOT")
    class_names = list(label_columns or CHEXPERT_CLASS_NAMES)
    train_dataset = CheXpertDataset(
        root=root,
        csv_path=_chexpert_csv_path(root, "train", train_csv),
        label_columns=class_names,
        image_size=image_size,
        view_filter=view_filter,
        data_fraction=data_fraction,
        seed=seed,
        max_items=max_train,
    )
    val_dataset = CheXpertDataset(
        root=root,
        csv_path=_chexpert_csv_path(root, "val", val_csv),
        label_columns=class_names,
        image_size=image_size,
        view_filter=view_filter,
        data_fraction=data_fraction,
        seed=seed + 1,
        max_items=max_val,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, class_names


def _chexpert_512_metadata(root, metadata_csv=None, split_seed=42):
    metadata_path = _resolve_existing_path(
        root,
        metadata_csv,
        ["metadata.csv"],
        "CheXpert-v1.0-512 metadata CSV",
    )
    metadata = pd.read_csv(metadata_path)
    return _prepare_chexpert_512_metadata(metadata, seed=split_seed)


def _chexpert_512_dataset(
    root,
    metadata,
    split,
    image_size,
    view_filter,
    data_fraction,
    seed,
    max_items,
):
    split = _normalize_split_name(split)
    split_metadata = metadata.loc[metadata["split"] == split].reset_index(drop=True)
    if split_metadata.empty:
        raise ValueError(
            f"CheXpert-v1.0-512 metadata has no examples for split '{split}'."
        )
    return CheXpertDataset(
        root=root,
        metadata=split_metadata,
        label_columns=CHEXPERT_CLASS_NAMES,
        image_size=image_size,
        view_filter=view_filter,
        data_fraction=data_fraction,
        seed=seed,
        max_items=max_items,
    )


def get_chexpert_512_loaders(
    root=None,
    metadata_csv=None,
    batch_size=16,
    image_size=224,
    view_filter="frontal",
    max_train_samples=None,
    max_val_samples=None,
    data_fraction=None,
    seed=42,
    num_workers=0,
):
    """Return train/validation loaders for the saved HF CheXpert-v1.0-512 data.

    The expected root contains ``metadata.csv`` and the images referenced by
    its ``Path`` column. When complete official train/validation/test split
    metadata is unavailable, patients are deterministically split 80/10/10.
    """

    root = _resolve_root(root, "CHEXPERT_ROOT")
    metadata = _chexpert_512_metadata(root, metadata_csv, split_seed=seed)
    train_dataset = _chexpert_512_dataset(
        root,
        metadata,
        split="train",
        image_size=image_size,
        view_filter=view_filter,
        data_fraction=data_fraction,
        seed=seed,
        max_items=max_train_samples,
    )
    val_dataset = _chexpert_512_dataset(
        root,
        metadata,
        split="val",
        image_size=image_size,
        view_filter=view_filter,
        data_fraction=data_fraction,
        seed=seed + 1,
        max_items=max_val_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, list(CHEXPERT_CLASS_NAMES)


def get_chexpert_512_split_loader(
    split,
    root=None,
    metadata_csv=None,
    batch_size=16,
    image_size=224,
    view_filter="frontal",
    max_samples=None,
    data_fraction=None,
    seed=42,
    num_workers=0,
    shuffle=None,
):
    """Return one saved HF CheXpert-v1.0-512 train, validation, or test split."""

    root = _resolve_root(root, "CHEXPERT_ROOT")
    split = _normalize_split_name(split)
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val'/'valid', or 'test'.")
    metadata = _chexpert_512_metadata(root, metadata_csv, split_seed=seed)
    dataset = _chexpert_512_dataset(
        root,
        metadata,
        split=split,
        image_size=image_size,
        view_filter=view_filter,
        data_fraction=data_fraction,
        seed=seed,
        max_items=max_samples,
    )
    if shuffle is None:
        shuffle = split == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader, list(CHEXPERT_CLASS_NAMES)


def get_chexpert_split_data(
    split,
    root=None,
    csv_path=None,
    batch_size=8,
    image_size=224,
    label_columns=None,
    num_workers=0,
    view_filter="frontal",
    data_fraction=1.0,
    seed=0,
    max_items=None,
    shuffle=None,
):
    """Return one CheXpert train, validation, or test split loader."""

    root = _resolve_root(root, "CHEXPERT_ROOT")
    split_name = str(split).lower()
    dataset = CheXpertDataset(
        root=root,
        csv_path=_chexpert_csv_path(root, split_name, csv_path),
        label_columns=label_columns,
        image_size=image_size,
        view_filter=view_filter,
        data_fraction=data_fraction,
        seed=seed,
        max_items=max_items,
    )
    if shuffle is None:
        shuffle = split_name == "train"
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        ),
        list(label_columns or CHEXPERT_CLASS_NAMES),
    )


def get_chexlocalize_data(
    root=None,
    split="val",
    metadata_path=None,
    segmentation_path=None,
    batch_size=8,
    image_size=224,
    label_columns=None,
    num_workers=0,
    view_filter="frontal",
    annotated_only=True,
    data_fraction=1.0,
    seed=0,
    max_items=None,
):
    """Return a CheXlocalize loader for post-hoc XAI mask evaluation.

    Returns:
        loader, class_names

    Localization batches should include the classification fields and may add:
        {
            "masks": Tensor[B, C, H, W] or Tensor[B, 1, H, W],
        }
    """

    root = _resolve_root(root, "CHEXLOCALIZE_ROOT")
    split = str(split).lower()
    if split in {"valid", "validation"}:
        split = "val"
    if split not in {"val", "test"}:
        raise ValueError("CheXlocalize split must be 'val'/'valid' or 'test'.")

    dataset = CheXlocalizeDataset(
        root=root,
        metadata_path=_chexlocalize_metadata_path(root, split, metadata_path),
        segmentation_path=_chexlocalize_segmentation_path(
            root,
            split,
            segmentation_path,
        ),
        label_columns=label_columns,
        image_size=image_size,
        view_filter=view_filter,
        annotated_only=annotated_only,
        data_fraction=data_fraction,
        seed=seed,
        max_items=max_items,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader, list(CHEXLOCALIZE_CLASS_NAMES)


def get_chexlocalize_loader(
    root=None,
    split="all",
    image_size=224,
    batch_size=8,
    max_samples=None,
    num_workers=0,
    view_filter="frontal",
):
    """Return CheXlocalize validation, test, or combined XAI evaluation data."""

    split = str(split).lower()
    if split in {"valid", "validation"}:
        split = "val"
    if split != "all":
        return get_chexlocalize_data(
            root=root,
            split=split,
            image_size=image_size,
            batch_size=batch_size,
            max_items=max_samples,
            num_workers=num_workers,
            view_filter=view_filter,
        )

    loaders = [
        get_chexlocalize_data(
            root=root,
            split=current_split,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            view_filter=view_filter,
        )[0]
        for current_split in ["val", "test"]
    ]
    dataset = ConcatDataset([loader.dataset for loader in loaders])
    if max_samples is not None:
        if max_samples < 0:
            raise ValueError("max_samples must be None or a non-negative integer.")
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader, list(CHEXLOCALIZE_CLASS_NAMES)
