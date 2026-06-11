from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

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


@dataclass(frozen=True)
class SampleRecord:
    split: str
    image_id: str
    image_path: str
    labels: List[float]


def _resolve_root(root: Optional[str | Path], env_var: str = "CHEXLOCALIZE_ROOT") -> Path:
    root = root or os.environ.get(env_var)
    if not root:
        raise ValueError(
            f"Missing dataset root. Pass root=... or set the {env_var} environment variable."
        )
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def _normalize_split_name(split: str) -> str:
    split = str(split).strip().lower()
    aliases = {"valid": "val", "validation": "val", "testing": "test"}
    return aliases.get(split, split)


def _validate_view_filter(view_filter: str) -> str:
    view_filter = str(view_filter).strip().lower()
    if view_filter not in {"frontal", "all"}:
        raise ValueError("view_filter must be 'frontal' or 'all'.")
    return view_filter


def _filter_views(metadata: pd.DataFrame, view_filter: str) -> pd.DataFrame:
    view_filter = _validate_view_filter(view_filter)
    if view_filter == "all":
        return metadata

    path_is_frontal = metadata["Path"].astype(str).str.lower().str.contains("frontal")
    if "Frontal/Lateral" not in metadata.columns:
        return metadata.loc[path_is_frontal].reset_index(drop=True)

    value_is_frontal = (
        metadata["Frontal/Lateral"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains("frontal")
    )
    return metadata.loc[path_is_frontal | value_is_frontal].reset_index(drop=True)


def _metadata_path(root: Path, split: str) -> Path:
    split = _normalize_split_name(split)
    path = root / "CheXpert" / f"{split}_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"CheXlocalize metadata CSV not found: {path}")
    return path


def _segmentation_path(root: Path, split: str) -> Path:
    split = _normalize_split_name(split)
    path = root / "CheXlocalize" / f"gt_segmentations_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"CheXlocalize segmentation JSON not found: {path}")
    return path


def _path_parts(path_value: str) -> Tuple[str, ...]:
    return PurePosixPath(str(path_value).replace("\\", "/")).parts


def _image_id_from_path(path_value: str) -> str:
    parts = _path_parts(path_value)
    if len(parts) < 3:
        return Path(parts[-1]).stem
    patient, study, filename = parts[-3:]
    return f"{patient}_{study}_{Path(filename).stem}"


def _resolve_image_path(root: Path, path_value: str) -> Path:
    parts = list(_path_parts(path_value))
    candidates: List[Path] = []

    candidates.append(root.joinpath(*parts))

    if parts and parts[0].lower().startswith("chexpert"):
        candidates.append(root.joinpath(*parts[1:]))

    normalized = parts[:]
    if normalized:
        first = normalized[0].lower()
        if first in {"chexpert-v1.0", "chexpert-v1.0-small"}:
            normalized[0] = "CheXpert"

    if len(normalized) > 1:
        second = normalized[1].lower()
        if second in {"valid", "validation"}:
            normalized[1] = "val"

    candidates.append(root.joinpath(*normalized))

    if normalized and normalized[0].lower() in {"train", "val", "valid", "test"}:
        prefixed = normalized[:]
        if prefixed[0].lower() in {"valid", "validation"}:
            prefixed[0] = "val"
        candidates.append(root.joinpath("CheXpert", *prefixed))

    if normalized and normalized[0] == "CheXpert":
        candidates.append(root.joinpath(*normalized[1:]))

    unique_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str not in seen:
            seen.add(candidate_str)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image path '{path_value}' under {root}. "
        f"Tried: {[str(c) for c in unique_candidates]}"
    )


def _u_zero_labels(row: pd.Series, label_columns: Sequence[str]) -> List[float]:
    values = pd.to_numeric(row[list(label_columns)], errors="coerce").fillna(0.0)
    return (values.to_numpy(dtype=np.float32) > 0).astype(np.float32).tolist()


def _decode_rle(rle: Dict) -> np.ndarray:
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
            "Compressed CheXlocalize masks require pycocotools. "
            "Install the project requirements in the active environment."
        ) from exc

    encoded = {"size": list(size), "counts": counts}
    if isinstance(encoded["counts"], str):
        encoded["counts"] = encoded["counts"].encode("ascii")
    mask = coco_mask.decode(encoded)
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    return mask.astype(np.uint8)


def _mask_stack_from_entry(
    segmentation_entry: Dict,
    mask_mode: str = "combined",
    class_names: Sequence[str] = CHEXLOCALIZE_CLASS_NAMES,
    fallback_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    mask_mode = str(mask_mode).lower()
    if mask_mode not in {"combined", "per_class"}:
        raise ValueError("mask_mode must be 'combined' or 'per_class'.")

    decoded_masks: List[Optional[np.ndarray]] = []
    image_size: Optional[Tuple[int, int]] = None

    for class_name in class_names:
        rle = segmentation_entry.get(class_name)
        if rle is None:
            decoded_masks.append(None)
            continue

        mask = _decode_rle(rle).astype(np.uint8)
        decoded_masks.append(mask)
        if image_size is None:
            image_size = tuple(mask.shape)

    if image_size is None:
        if "img_size" in segmentation_entry:
            image_size = tuple(int(v) for v in segmentation_entry["img_size"])
        elif fallback_size is not None:
            image_size = tuple(int(v) for v in fallback_size)
        else:
            raise ValueError("Could not infer image size for an empty segmentation entry.")

    masks: List[np.ndarray] = []
    for decoded in decoded_masks:
        if decoded is None:
            masks.append(np.zeros(image_size, dtype=np.uint8))
        else:
            masks.append(decoded)

    mask_stack = np.stack(masks, axis=0)

    if mask_mode == "combined":
        mask_stack = mask_stack.max(axis=0, keepdims=True)

    return mask_stack.astype(np.float32)


def load_chexlocalize_records(
    root: Optional[str | Path] = None,
    splits: Sequence[str] = ("val", "test"),
    view_filter: str = "frontal",
    annotated_only: bool = True,
) -> Tuple[List[SampleRecord], Dict[str, Dict]]:
    root = _resolve_root(root)
    records: List[SampleRecord] = []
    segmentation_lookup: Dict[str, Dict] = {}

    for split_name in splits:
        split = _normalize_split_name(split_name)
        metadata = pd.read_csv(_metadata_path(root, split))
        with open(_segmentation_path(root, split), "r", encoding="utf-8") as handle:
            segmentations = json.load(handle)

        segmentation_lookup[split] = segmentations

        if "Path" not in metadata.columns:
            raise ValueError(f"Metadata for split '{split}' must contain a 'Path' column.")

        missing_columns = [
            column for column in CHEXPERT_CLASS_NAMES if column not in metadata.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Metadata for split '{split}' is missing label columns: {missing_columns}"
            )

        metadata = _filter_views(metadata, view_filter).copy()
        metadata["_image_id"] = metadata["Path"].map(_image_id_from_path)

        if annotated_only:
            metadata = metadata[metadata["_image_id"].isin(segmentations)]

        metadata = metadata.reset_index(drop=True)

        if metadata.empty:
            raise ValueError(f"No aligned rows found for split '{split}'.")

        for _, row in metadata.iterrows():
            image_path = _resolve_image_path(root, row["Path"])
            records.append(
                SampleRecord(
                    split=split,
                    image_id=row["_image_id"],
                    image_path=str(image_path),
                    labels=_u_zero_labels(row, CHEXPERT_CLASS_NAMES),
                )
            )

    if not records:
        raise ValueError("No CheXlocalize records were collected.")

    return records, segmentation_lookup


def create_subset(
    records: Sequence[SampleRecord],
    subset_size: int = 120,
    seed: int = 42,
) -> List[SampleRecord]:
    if subset_size <= 0:
        raise ValueError("subset_size must be greater than 0.")

    records = list(records)
    if subset_size >= len(records):
        return records

    grouped: Dict[str, List[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.split, []).append(record)

    rng = random.Random(seed)
    allocations: Dict[str, int] = {}
    total_records = len(records)

    for split, split_records in grouped.items():
        proportional = subset_size * len(split_records) / total_records
        allocations[split] = max(1, int(round(proportional)))

    while sum(allocations.values()) > subset_size:
        largest_split = max(allocations, key=allocations.get)
        if allocations[largest_split] > 1:
            allocations[largest_split] -= 1
        else:
            break

    while sum(allocations.values()) < subset_size:
        smallest_split = min(
            grouped,
            key=lambda split: len(grouped[split]) - allocations[split],
        )
        if allocations[smallest_split] < len(grouped[smallest_split]):
            allocations[smallest_split] += 1
        else:
            break

    subset: List[SampleRecord] = []
    for split, split_records in grouped.items():
        shuffled = split_records[:]
        rng.shuffle(shuffled)
        subset.extend(shuffled[: allocations[split]])

    subset.sort(key=lambda record: (record.split, record.image_id))
    return subset[:subset_size]


def preprocess_image(
    image_path: str | Path,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(target_size, Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0

    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
    image_array = np.transpose(image_array, (2, 0, 1))
    return image_array.astype(np.float32)


def preprocess_mask_stack(
    mask_stack: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    processed_masks: List[np.ndarray] = []
    for mask in mask_stack:
        mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        mask_image = mask_image.resize(target_size, Image.Resampling.NEAREST)
        resized_mask = (np.asarray(mask_image, dtype=np.uint8) > 0).astype(np.float32)
        processed_masks.append(resized_mask)
    return np.stack(processed_masks, axis=0).astype(np.float32)


def ensure_output_dirs(output_root: str | Path) -> Tuple[Path, Path]:
    output_root = Path(output_root)
    images_dir = output_root / "images"
    masks_dir = output_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, masks_dir


def preprocess_and_save_records(
    records: Sequence[SampleRecord],
    segmentation_lookup: Dict[str, Dict],
    output_root: str | Path,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mask_mode: str = "combined",
    mask_class_names: Sequence[str] = CHEXLOCALIZE_CLASS_NAMES,
) -> List[Dict]:
    output_root = Path(output_root)
    images_dir, masks_dir = ensure_output_dirs(output_root)
    manifest: List[Dict] = []

    mask_to_label_indices = [
        CHEXPERT_CLASS_NAMES.index(class_name)
        for class_name in mask_class_names
    ]

    for index, record in enumerate(records):
        split_segmentations = segmentation_lookup.get(record.split)
        if split_segmentations is None:
            raise KeyError(f"No segmentation lookup found for split '{record.split}'.")

        segmentation_entry = split_segmentations.get(record.image_id)
        if segmentation_entry is None:
            raise KeyError(f"Missing segmentation entry for image '{record.image_id}'.")

        with Image.open(record.image_path) as raw_image:
            fallback_size = (raw_image.height, raw_image.width)

        image_array = preprocess_image(record.image_path, target_size=target_size)
        raw_mask_stack = _mask_stack_from_entry(
            segmentation_entry,
            mask_mode=mask_mode,
            class_names=mask_class_names,
            fallback_size=fallback_size,
        )
        mask_array = preprocess_mask_stack(raw_mask_stack, target_size=target_size)

        sample_id = f"img_{index:03d}"
        image_output = images_dir / f"{sample_id}.npy"
        mask_output = masks_dir / f"{sample_id}_mask.npy"

        np.save(image_output, image_array)
        np.save(mask_output, mask_array)

        manifest.append(
            {
                "sample_id": sample_id,
                "split": record.split,
                "image_id": record.image_id,
                "image_path": record.image_path,
                "image_npy": str(image_output),
                "mask_npy": str(mask_output),
                "labels": record.labels,
                "mask_mode": mask_mode,
                "mask_channels": int(mask_array.shape[0]),
            }
        )

    manifest_payload = {
        "class_names": CHEXPERT_CLASS_NAMES,
        "mask_class_names": list(mask_class_names),
        "mask_mode": mask_mode,
        "mask_to_label_indices": mask_to_label_indices,
        "target_size": list(target_size),
        "num_samples": len(manifest),
        "samples": manifest,
    }

    with open(output_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    return manifest


class PreprocessedCheXlocalizeDataset(Dataset):
    def __init__(self, output_root: str | Path, split: Optional[str] = None):
        output_root = Path(output_root)
        manifest_path = output_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. Run preprocessing first."
            )

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.class_names = payload["class_names"]
        self.mask_class_names = payload["mask_class_names"]
        self.mask_mode = payload["mask_mode"]
        self.mask_to_label_indices = payload.get("mask_to_label_indices")
        self.samples = payload["samples"]

        if split is not None:
            split = _normalize_split_name(split)
            self.samples = [sample for sample in self.samples if sample["split"] == split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = np.load(sample["image_npy"]).astype(np.float32)
        mask = np.load(sample["mask_npy"]).astype(np.float32)
        labels = np.asarray(sample["labels"], dtype=np.float32)

        return {
            "images": torch.from_numpy(image),
            "masks": torch.from_numpy(mask),
            "labels": torch.from_numpy(labels),
            "ids": sample["image_id"],
            "split": sample["split"],
        }


def create_preprocessed_dataloader(
    output_root: str | Path,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    split: Optional[str] = None,
) -> DataLoader:
    dataset = PreprocessedCheXlocalizeDataset(output_root=output_root, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def denormalize_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image, dtype=np.float32)
    image = np.transpose(image, (1, 2, 0))
    image = (image * IMAGENET_STD) + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0)


def summarize_records(records: Sequence[SampleRecord]) -> None:
    split_counts: Dict[str, int] = {}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1

    print(f"Total aligned records: {len(records)}")
    for split, count in sorted(split_counts.items()):
        print(f"  - {split}: {count}")