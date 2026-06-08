import json
import shutil
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


def selected_model_dir(run_dir):
    """Return the shared directory for an XAI-ready selected model artifact."""

    return Path(run_dir) / "selected_model"


def save_selected_model_artifact(run_dir, manifest, state=None, source_state_path=None):
    """Save one validation-selected model in the shared XAI artifact format.

    ``state`` contains only the method-specific data needed to reconstruct the
    selected predictor. ``source_state_path`` can copy an existing checkpoint
    without loading it into memory. Exactly one of the two may be provided.
    """

    if state is not None and source_state_path is not None:
        raise ValueError("Provide either state or source_state_path, not both.")

    artifact_dir = selected_model_dir(run_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state_path = artifact_dir / "state.pt"

    if source_state_path is not None:
        shutil.copyfile(source_state_path, state_path)
    elif state is not None:
        torch.save(state, state_path)

    saved_manifest = {
        "artifact_version": 1,
        **manifest,
        "state_path": "state.pt" if state is not None or source_state_path is not None else None,
    }
    with open(artifact_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(saved_manifest, handle, indent=2)
    return artifact_dir


def load_selected_model_artifact(run_dir):
    """Load the manifest and method-specific state for an XAI-ready model."""

    artifact_dir = selected_model_dir(run_dir)
    with open(artifact_dir / "manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    state = None
    if manifest.get("state_path"):
        state_path = artifact_dir / manifest["state_path"]
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")

    return manifest, state


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


def save_linear_probe_outputs(
    run_dir,
    history_df,
    summary_df,
    per_class_df,
    metadata=None,
    trials_df=None,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(run_dir / "history.csv", index=False)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    per_class_df.to_csv(run_dir / "per_class.csv", index=False)
    if trials_df is not None:
        trials_df.to_csv(run_dir / "trials.csv", index=False)

    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata or {}, handle, indent=2)


def load_linear_probe_outputs(run_dir):
    run_dir = Path(run_dir)

    with open(run_dir / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    return {
        "history": pd.read_csv(run_dir / "history.csv"),
        "summary": pd.read_csv(run_dir / "summary.csv"),
        "per_class": pd.read_csv(run_dir / "per_class.csv"),
        "trials": (
            pd.read_csv(run_dir / "trials.csv")
            if (run_dir / "trials.csv").exists()
            else None
        ),
        "metadata": metadata,
    }


def save_partial_finetune_outputs(
    run_dir,
    history_df,
    summary_df,
    per_class_df,
    metadata=None,
    trials_df=None,
):
    """Save compact partial-fine-tuning tables without model checkpoints."""

    save_linear_probe_outputs(
        run_dir,
        history_df=history_df,
        summary_df=summary_df,
        per_class_df=per_class_df,
        metadata=metadata,
        trials_df=trials_df,
    )


def load_partial_finetune_outputs(run_dir):
    """Load compact partial-fine-tuning result tables."""

    return load_linear_probe_outputs(run_dir)


def partial_finetune_progress_dir(run_dir):
    return Path(run_dir) / "_progress"


def load_partial_finetune_progress(run_dir, experiment_signature):
    """Load compatible per-config partial-fine-tuning progress tables."""

    progress_dir = partial_finetune_progress_dir(run_dir)
    metadata_path = progress_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("experiment_signature") != experiment_signature:
        return None

    def read_csv(name):
        path = progress_dir / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    return {
        "history": read_csv("history.csv"),
        "trials": read_csv("trials.csv"),
        "results": read_csv("partial_finetune_results.csv"),
        "metadata": metadata,
    }


def save_partial_finetune_progress(
    run_dir,
    history_df,
    trials_df,
    results_df,
    current_best_df,
    metadata,
):
    """Save cumulative partial-fine-tuning tables after a completed config."""

    progress_dir = partial_finetune_progress_dir(run_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(progress_dir / "history.csv", index=False)
    trials_df.to_csv(progress_dir / "trials.csv", index=False)
    results_df.to_csv(progress_dir / "partial_finetune_results.csv", index=False)
    current_best_df.to_csv(progress_dir / "current_best.csv", index=False)
    current_best_df.to_json(
        progress_dir / "current_best.json",
        orient="records",
        indent=2,
    )

    method_results_path = Path(run_dir).parent / "partial_finetune_results.csv"
    if method_results_path.exists():
        method_results_df = pd.read_csv(method_results_path)
        incoming_model_keys = set(results_df["model_key"])
        method_results_df = method_results_df[
            ~method_results_df["model_key"].isin(incoming_model_keys)
        ]
        method_results_df = pd.concat([method_results_df, results_df], ignore_index=True)
    else:
        method_results_df = results_df.copy()
    method_results_df.to_csv(method_results_path, index=False)

    with open(progress_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def save_partial_finetune_checkpoint(run_dir, checkpoint_name, payload):
    """Save a compact trainable-parameter checkpoint and return its path."""

    checkpoint_dir = partial_finetune_progress_dir(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{checkpoint_name}.pt"
    torch.save(payload, path)
    return path


def load_partial_finetune_checkpoint(path):
    """Load a compact partial-fine-tuning checkpoint."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
