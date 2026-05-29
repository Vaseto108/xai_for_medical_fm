# Explainable AI with Medical Foundation Models for Medical Imaging

Minimal project skeleton for studying how adaptation strategies for DINO-style foundation models affect classification performance, efficiency, and explanation quality on chest X-ray tasks.

This first version is intentionally small. It uses ChestMNIST from MedMNIST only as a sanity-check dataset because CheXpert and CheXlocalize need separate access/setup. The final project can swap the data module to CheXpert while keeping the shared loader and prediction interfaces.

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

- downloads a small ChestMNIST subset,
- loads a frozen DINOv2 backbone with a linear multi-label classifier head,
- reports total and trainable parameter counts,
- trains the classifier head for a configurable number of epochs,
- prints training progress and validation metrics after each epoch,
- produces sigmoid probabilities,
- computes mean AUROC over valid classes,
- displays one simple saliency heatmap.

## Interfaces

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

## Notes

- `src/xai.py` currently uses a simple gradient saliency heatmap. Replace this with Grad-CAM, occlusion sensitivity, or attention rollout once the training/adaptation code is in place.
- `src/eval.py` skips classes with only positives or only negatives when computing mean AUROC, which can happen in tiny subsets.
- `src/train.py` uses `BCEWithLogitsLoss` for multi-label classification and optimizes only parameters with `requires_grad=True`.
- The dataset subset is tiny, so metrics from the demo are still mostly an interface check.
