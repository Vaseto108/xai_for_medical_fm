"""Download a local CheXpert-v1.0-512 export and check CheXlocalize files."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image


DATASET_NAME = "StanfordAIMI/CheXpert-v1.0-512"

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

CHEXLOCALIZE_REQUIRED_FILES = [
    "CheXpert/val_labels.csv",
    "CheXpert/test_labels.csv",
    "CheXlocalize/gt_segmentations_val.json",
    "CheXlocalize/gt_segmentations_test.json",
]


def _normalize_split_name(split):
    aliases = {
        "valid": "val",
        "validation": "val",
        "dev": "val",
        "testing": "test",
        "training": "train",
    }
    split = str(split).strip().lower()
    return aliases.get(split, split)


def _patient_id(path_value):
    parts = PurePosixPath(str(path_value).replace("\\", "/")).parts
    for part in parts:
        if part.lower().startswith("patient"):
            return part
    if len(parts) >= 3:
        return parts[-3]
    return str(path_value)


def _deterministic_split(group_id, seed):
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big") / 2**64
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "val"
    return "test"


def _safe_relative_image_path(original_path, split_name, source_index):
    if original_path:
        parts = [
            part
            for part in PurePosixPath(str(original_path).replace("\\", "/")).parts
            if part not in {"", ".", "..", "/"}
        ]
        if parts:
            path = Path(*parts)
        else:
            path = Path(split_name) / f"example_{source_index:08d}.jpg"
    else:
        path = Path(split_name) / f"example_{source_index:08d}.jpg"
    return path.with_suffix(".jpg")


def _plain_metadata_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_image(item):
    if "image" in item:
        image = item["image"]
    else:
        image = next(
            (value for value in item.values() if isinstance(value, Image.Image)),
            None,
        )
    if image is None:
        raise ValueError("Could not find a decoded image column in the HF example.")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and image.get("path"):
        with Image.open(image["path"]) as opened:
            return opened.convert("RGB")
    raise ValueError(f"Unsupported image value type: {type(image).__name__}")


def _decoded_label_value(value, feature):
    if value is None or not hasattr(feature, "int2str"):
        return value
    try:
        label_name = feature.int2str(int(value)).strip().lower()
    except (TypeError, ValueError):
        return value
    if "uncertain" in label_name:
        return -1
    if "present" in label_name or "positive" in label_name:
        return 1
    if "negative" in label_name or "absent" in label_name:
        return 0
    if "unlabeled" in label_name or "no label" in label_name or "missing" in label_name:
        return None
    return value


def _add_label_columns(row, item, features):
    if all(class_name in row for class_name in CHEXPERT_CLASS_NAMES):
        return

    for key in ["labels", "label"]:
        labels = item.get(key)
        if isinstance(labels, dict):
            for class_name in CHEXPERT_CLASS_NAMES:
                if class_name in labels:
                    row[class_name] = labels[class_name]
            break
        if isinstance(labels, (list, tuple, np.ndarray)) and len(labels) == 14:
            row.update(dict(zip(CHEXPERT_CLASS_NAMES, labels)))
            break

    missing = [name for name in CHEXPERT_CLASS_NAMES if name not in row]
    if missing:
        raise ValueError(
            "The HF examples do not expose the expected 14 CheXpert labels. "
            f"Missing columns: {missing}. Available columns: {list(item)}"
        )
    for class_name in CHEXPERT_CLASS_NAMES:
        row[class_name] = _decoded_label_value(
            row[class_name],
            features.get(class_name) if features is not None else None,
        )


def _selected_dataset(dataset, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("--chexpert_fraction must be in the interval (0, 1].")
    if fraction == 1:
        return dataset
    count = max(1, int(round(len(dataset) * fraction)))
    return dataset.shuffle(seed=seed).select(range(count))


def _assign_saved_splits(metadata, seed):
    source_splits = metadata["source_split"].map(_normalize_split_name)
    available = set(source_splits.unique())
    if {"train", "val", "test"}.issubset(available):
        metadata["split"] = source_splits
        return "Used the official/source train, validation, and test splits."

    metadata["split"] = metadata["Path"].map(
        lambda path: _deterministic_split(_patient_id(path), seed)
    )
    return (
        "Source metadata did not provide complete train/validation/test splits; "
        "assigned a deterministic patient-level 80/10/10 split."
    )


def download_chexpert_512(output_root, fraction, seed):
    try:
        from datasets import get_dataset_split_names, load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: datasets. Run: python -m pip install -r requirements.txt"
        ) from exc

    try:
        split_names = get_dataset_split_names(DATASET_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Could not access {DATASET_NAME}. Accept its conditions on Hugging "
            "Face and run `huggingface-cli login`, then retry."
        ) from exc

    dataset_root = Path(output_root) / "CheXpert-v1.0-512"
    dataset_root.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"Available HF splits: {split_names}")

    for split_offset, split_name in enumerate(split_names):
        try:
            dataset = load_dataset(DATASET_NAME, split=split_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load split '{split_name}' from {DATASET_NAME}. Accept "
                "the dataset conditions and run `huggingface-cli login`, then retry."
            ) from exc

        dataset = dataset.add_column("_download_source_index", list(range(len(dataset))))
        selected = _selected_dataset(dataset, fraction, seed + split_offset)
        print(f"{split_name}: saving {len(selected):,} of {len(dataset):,} examples")
        for selected_index, item in enumerate(selected):
            source_index = item["_download_source_index"]
            original_path = item.get("path") or item.get("Path")
            relative_path = _safe_relative_image_path(
                original_path,
                split_name,
                source_index,
            )
            image_path = dataset_root / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not image_path.exists():
                _extract_image(item).save(
                    image_path,
                    format="JPEG",
                    quality=95,
                    subsampling=0,
                )

            row = {
                "Path": relative_path.as_posix(),
                "original_path": original_path,
                "source_split": split_name,
                "source_index": source_index,
            }
            row.update(
                {
                    key: _plain_metadata_value(value)
                    for key, value in item.items()
                    if key != "image"
                    and key not in {"path", "Path", "_download_source_index"}
                }
            )
            _add_label_columns(row, item, dataset.features)
            rows.append(row)
            if (selected_index + 1) % 1000 == 0:
                print(f"  saved {selected_index + 1:,}/{len(selected):,}")

    metadata = pd.DataFrame(rows)
    split_message = _assign_saved_splits(metadata, seed)
    leading_columns = [
        "Path",
        "original_path",
        "source_split",
        "source_index",
        "split",
    ]
    metadata = metadata[
        leading_columns
        + [column for column in metadata.columns if column not in leading_columns]
    ]
    metadata_path = dataset_root / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    print(split_message)
    print("Saved metadata:", metadata_path)
    print("Saved split counts:")
    print(metadata["split"].value_counts().to_string())


def check_chexlocalize(root):
    root = Path(root)
    missing = [
        relative_path
        for relative_path in CHEXLOCALIZE_REQUIRED_FILES
        if not (root / relative_path).exists()
    ]
    if not missing:
        print(f"CheXlocalize layout is ready: {root}")
        return True

    print("\nCheXlocalize is not automatically downloaded by this script.")
    print("Download the official release using the instructions at:")
    print("https://github.com/rajpurkarlab/cheXlocalize/blob/master/download_instructions.md")
    print(f"Place it under: {root}")
    print("Missing required files:")
    for relative_path in missing:
        print(f"  - {root / relative_path}")
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=Path("data"))
    parser.add_argument("--chexpert_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_chexpert",
        action="store_true",
        help="Only check the expected CheXlocalize folder.",
    )
    parser.add_argument(
        "--chexlocalize_root",
        type=Path,
        default=None,
        help="Defaults to <output_root>/chexlocalize.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.skip_chexpert:
        download_chexpert_512(
            output_root=args.output_root,
            fraction=args.chexpert_fraction,
            seed=args.seed,
        )
    chexlocalize_root = args.chexlocalize_root or args.output_root / "chexlocalize"
    check_chexlocalize(chexlocalize_root)


if __name__ == "__main__":
    main()
