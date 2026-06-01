# Explainable AI with Medical Foundation Models for Medical Imaging

Minimal project skeleton for studying how adaptation strategies for DINO-style foundation models affect classification performance, efficiency, and explanation quality on chest X-ray tasks.

This first version uses ChestMNIST from MedMNIST because CheXpert and CheXlocalize need separate access/setup. The final project can swap the data module to CheXpert while keeping the shared loader and prediction interfaces.

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

## Notes

- `src/xai.py` currently uses a simple gradient saliency heatmap. Replace this with Grad-CAM, occlusion sensitivity, or attention rollout once the training/adaptation code is in place.
- `src/eval.py` skips classes with only positives or only negatives when computing mean AUROC, which can happen when using small subsets.
- `src/train.py` uses `BCEWithLogitsLoss` for multi-label classification and optimizes only parameters with `requires_grad=True`.
- ChestMNIST is still only a scaffold dataset for this project, so demo metrics should not be treated as final results.
