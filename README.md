# Explainable AI with Medical Foundation Models for Medical Imaging

Minimal project skeleton for studying how adaptation strategies for DINO-style foundation models affect classification performance, efficiency, and explanation quality on chest X-ray tasks.

ChestMNIST remains the fast sanity dataset. Final classification experiments use
the local Hugging Face `CheXpert-v1.0-512` export, while CheXlocalize remains a
separate post-hoc XAI evaluation dataset.

## Setup

```bash
python -m pip install -r requirements.txt
```

If you are running the demo in Jupyter, install from inside the notebook kernel if needed:

```python
%pip install -r ../requirements.txt
```

## Current Demo

Open `notebooks/demo.ipynb` and run the cells. The notebook:

- downloads ChestMNIST,
- loads a frozen DINOv2 backbone with a linear multi-label classifier head,
- reports total and trainable parameter counts,
- trains the classifier head for a configurable number of epochs,
- prints training progress and validation metrics after each epoch,
- produces sigmoid probabilities,
- computes mean AUROC over valid classes,
- displays one simple saliency heatmap.

For repeated experiments, copy the demo notebook to a local ignored file:

```bash
cp notebooks/demo.ipynb notebooks/demo_local.ipynb
```

Then run `notebooks/demo_local.ipynb`. Git ignores `*_local.ipynb` notebooks, so saved outputs will not block `git pull`.

## Interfaces

`get_small_data()` uses the full ChestMNIST train and validation splits by default:

```python
train_loader, val_loader, class_names = get_small_data()
```

Pass integers for quick sanity checks:

```python
train_loader, val_loader, class_names = get_small_data(n_train=100, n_val=30)
```

Data loaders return batches with:

```python
{
    "images": Tensor[B, 3, 224, 224],
    "labels": Tensor[B, C],
    "ids": list[str],
}
```

Prediction helpers return:

```python
probs, labels, ids
```

where `probs` and `labels` have shape `[N, C]`.

## Final Datasets

The practical final classification dataset is the gated Hugging Face export
`StanfordAIMI/CheXpert-v1.0-512`. After accepting its conditions and running
`huggingface-cli login`, download a reproducible local subset or the full data:

```bash
python scripts/download_datasets.py --output_root data --chexpert_fraction 0.05 --seed 42
```

Use `--chexpert_fraction 1.0` for the final export. The script writes
`data/CheXpert-v1.0-512/metadata.csv`, preserves each image's relative path,
and checks the expected CheXlocalize layout under `data/chexlocalize/`.
CheXlocalize release files are not reliably downloadable from GitHub, so the
script prints the official instructions and checks for:

```text
data/chexlocalize/CheXpert/val_labels.csv
data/chexlocalize/CheXpert/test_labels.csv
data/chexlocalize/CheXlocalize/gt_segmentations_val.json
data/chexlocalize/CheXlocalize/gt_segmentations_test.json
```

Set roots without hardcoding machine-specific paths:

```bash
set CHEXPERT_ROOT=C:\path\to\data\CheXpert-v1.0-512
set CHEXLOCALIZE_ROOT=C:\path\to\data\chexlocalize
```

CheXpert-v1.0-512 classification uses all 14 observations, frontal images by
default, direct square resizing, ImageNet normalization, and U-Zero labels. If
the downloaded metadata does not provide complete train/validation/test splits,
patients are assigned to a deterministic 80/10/10 split:

```python
from src.data import get_chexpert_512_loaders, get_chexpert_512_split_loader

train_loader, val_loader, class_names = get_chexpert_512_loaders()
test_loader, class_names = get_chexpert_512_split_loader("test")
```

CheXlocalize is separate and should only be used for post-hoc XAI evaluation:

```python
from src.data import (
    CHEXLOCALIZE_TO_CHEXPERT_INDICES,
    get_chexlocalize_loader,
)

xai_loader, mask_class_names = get_chexlocalize_loader(split="all")
```

CheXlocalize batches contain 14-class labels and 10 masks ordered according to
`mask_class_names`. `CHEXLOCALIZE_TO_CHEXPERT_INDICES` maps those masks to the
corresponding classifier outputs. Compressed official masks require
`pycocotools`.

Local ignored notebooks are available for checking the loaders and running a
small CheXpert linear-probe experiment:

```text
notebooks/chexpert_loader_local.ipynb
notebooks/linear_probe_chexpert_local.ipynb
```

## Selected Models for XAI

Official experiment notebooks save the validation-selected predictor under each
run directory:

```text
selected_model/
  manifest.json
  state.pt
```

Load an image-to-logits predictor for explanation work with:

```python
from src.model import load_selected_model

model, manifest = load_selected_model(run_dir, device="cuda")
```

Linear-probe and partial-fine-tuning artifacts support Grad-CAM and occlusion.
The kNN artifact stores its selected full reference bank and supports occlusion,
but not Grad-CAM because neighbor selection is not a differentiable classifier.
These model-state artifacts live under `outputs/` and are intentionally ignored
by Git.

## Notes

- `src/xai.py` currently uses a simple gradient saliency heatmap. Replace this with Grad-CAM, occlusion sensitivity, or attention rollout once the training/adaptation code is in place.
- `src/eval.py` skips classes with only positives or only negatives when computing mean AUROC, which can happen when using small subsets.
- `src/train.py` uses `BCEWithLogitsLoss` for multi-label classification and optimizes only parameters with `requires_grad=True`.
- ChestMNIST is still only a scaffold dataset for this project, so demo metrics should not be treated as final results.
