import json
from pathlib import Path

import pandas as pd
import torch


def _format_fraction(data_fraction):
    return f"{data_fraction:g}".replace(".", "p")


def make_run_dir(output_root, dataset_name, run_name, data_fraction=1.0, dataset_seed=0):
    root = Path(output_root) / dataset_name
    if data_fraction == 1.0:
        return root / run_name

    partial_name = f"fraction_{_format_fraction(data_fraction)}_seed_{dataset_seed}"
    return root / "_partial" / partial_name / run_name


def validate_save_request(save_outputs, data_fraction):
    if save_outputs and (data_fraction <= 0 or data_fraction > 1):
        raise ValueError("data_fraction must be in the interval (0, 1].")


def feature_bank_path(run_dir):
    return Path(run_dir) / "features.pt"


def feature_bank_exists(run_dir):
    return feature_bank_path(run_dir).exists()


def save_feature_bank(run_dir, feature_bank, metadata=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "feature_bank": feature_bank,
            "metadata": metadata or {},
        },
        feature_bank_path(run_dir),
    )


def load_feature_bank(run_dir):
    path = feature_bank_path(run_dir)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    return payload["feature_bank"], payload.get("metadata", {})


def save_knn_outputs(run_dir, runs_df, summary_df, per_class_full_df, metadata=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    runs_df.to_csv(run_dir / "runs.csv", index=False)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    per_class_full_df.to_csv(run_dir / "per_class_full.csv", index=False)

    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata or {}, handle, indent=2)


def load_knn_outputs(run_dir):
    run_dir = Path(run_dir)

    with open(run_dir / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    return {
        "runs": pd.read_csv(run_dir / "runs.csv"),
        "summary": pd.read_csv(run_dir / "summary.csv"),
        "per_class_full": pd.read_csv(run_dir / "per_class_full.csv"),
        "metadata": metadata,
    }
