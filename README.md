# RSNA Intracranial Aneurysm Detection — 3D ResNet Baseline

This repository contains a reproducible 3D medical imaging pipeline for the RSNA Intracranial Aneurysm Detection challenge. The project focuses on building an end-to-end machine learning workflow for DICOM CT angiography series, including preprocessing, tensor caching, model training (single-split and k-fold), checkpointing, and deployment-oriented model export.

The original version of this project used a compact custom 3D CNN trained from scratch. This version adds an option to start from an ImageNet-pretrained 2D ResNet backbone inflated into 3D, alongside k-fold cross-validation and corrected data augmentation.

## Project Goals

The goal of this project is to build a stronger and cleaner baseline for 3D medical image classification while emphasizing reproducibility, modular code organization, and rigorous model evaluation.

This project demonstrates:

* DICOM series loading
* Hounsfield Unit conversion
* MONOCHROME1 handling
* voxel resampling
* center cropping and padding
* cached tensor generation
* multi-label classification
* class imbalance handling (via `pos_weight` and configurable sampling strategy)
* anatomically-correct flip augmentation (with left/right label swapping)
* 3D ResNet training, from scratch or from an inflated pretrained 2D backbone
* k-fold cross-validation with aggregated out-of-fold macro AUC
* validation macro AUC and per-label AUC tracking
* TorchScript export and inference latency benchmarking
* reproducible configuration files

## Repository Structure

```text
rsna_project/
├── configs/
│   ├── resnet3d.yaml
│   └── resnet3d_smoke.yaml
├── scripts/
│   ├── build_cache.py
│   ├── train_model.py         # single train/valid split
│   ├── train_kfold.py         # k-fold cross-validation
│   ├── export_model.py        # TorchScript export
│   └── benchmark_inference.py # inference latency/throughput benchmarking
├── src/
│   ├── constants.py
│   ├── data_selection.py
│   ├── metrics.py
│   ├── models.py
│   └── preprocess.py
├── notebooks/
├── reports/
├── requirements.txt
└── README.md
```

## Model

Two architectures are available, selected via `training.model_name` in the config:

* **`resnet3d18`** — a compact 3D ResNet trained entirely from scratch.
* **`resnet3d_inflated`** — an ImageNet-pretrained torchvision `resnet18`/`resnet34` (`training.backbone`), inflated into 3D via I3D-style weight inflation (Carreira & Zisserman, 2017). Every 2D conv/batchnorm is converted to its 3D equivalent, with pretrained 2D filters bootstrapped along the new depth axis. This gives the network a head start over random initialization, which matters given how little labeled 3D medical data is available relative to ImageNet. Requires internet access at training time to download pretrained weights (set `training.pretrained: false` to skip this and initialize randomly instead).

Both predict the same 14 labels:

* 13 vascular territory labels (5 of which are Left/Right pairs)
* 1 global `Aneurysm Present` label

## Data

This project uses the RSNA Intracranial Aneurysm Detection Kaggle dataset. Raw DICOM files, cached tensors, and trained model weights are not included in this repository.

Expected Kaggle dataset paths:

```text
/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv
/kaggle/input/rsna-intracranial-aneurysm-detection/series
```

## Preprocessing Pipeline

Each DICOM series is converted into a standardized 3D tensor using the following steps:

1. Load all DICOM slices in a series.
2. Sort slices using `ImagePositionPatient` or `InstanceNumber`.
3. Convert raw pixel values to Hounsfield Units.
4. Handle `MONOCHROME1` images.
5. Clip intensities to a fixed HU window.
6. Scale voxel values.
7. Resample to approximately isotropic voxel spacing.
8. Center crop or pad to a fixed 3D input size.
9. Save the processed volume as a cached `.pt` tensor.

## Training-time Augmentation

Applied only to the training split, in `CachedRSNADataset.augment()`:

* **Depth flip** (superior-inferior) — applied freely, no label change needed.
* **Width flip** (left-right) — mirrors the volume *and* swaps the corresponding Left/Right label pairs (e.g. Left MCA ↔ Right MCA), since flipping laterality without swapping labels would train on incorrect supervision.
* Random intensity scale/shift, and mild additive Gaussian noise.

Height (anterior-posterior) flips are intentionally not used, since brain anatomy isn't front-back symmetric the way it is left-right symmetric.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Build cached tensors:

```bash
python scripts/build_cache.py --config configs/resnet3d.yaml
```

Train on a single train/valid split:

```bash
python scripts/train_model.py --config configs/resnet3d.yaml
```

Train with k-fold cross-validation:

```bash
python scripts/train_kfold.py --config configs/resnet3d.yaml --n_folds 5
```

Export the best checkpoint to TorchScript:

```bash
python scripts/export_model.py --config configs/resnet3d.yaml
```

Benchmark exported model inference latency/throughput:

```bash
python scripts/benchmark_inference.py --model-path <path_to_exported_model.pt>
```

## Configuration

Training and preprocessing settings are controlled through `configs/resnet3d.yaml` (a smaller `configs/resnet3d_smoke.yaml` is provided for quick end-to-end sanity checks).

The config file defines:

* dataset paths and cache directory
* maximum number of series and sampling strategy (`balanced`, `stratified`, or `all`)
* validation split size
* target voxel spacing and input volume size
* HU window
* model architecture (`model_name`, `backbone`, `pretrained`, `dropout`)
* number of cross-validation folds (`n_folds`)
* learning rate, batch size, number of epochs
* checkpoint and metrics output paths

## Evaluation

The training scripts track validation macro AUC and per-label AUC across the 14 prediction labels. Labels that contain only one class in a given split are skipped when computing AUC.

The single-split script (`train_model.py`) selects the best checkpoint by validation macro AUC. The k-fold script (`train_kfold.py`) additionally aggregates out-of-fold predictions across all folds into one overall macro AUC — a more reliable estimate of generalization than a single split, since it's computed over the entire labeled set rather than one held-out slice.

## Current Status

This project is an active v2 refactor of an earlier Kaggle notebook implementation. Recently added:

* pretrained 3D backbone via I3D-style weight inflation
* k-fold cross-validation with out-of-fold evaluation
* corrected flip augmentation (with anatomically-correct label swapping)
* a data-selection bug fix (`max_series: null` no longer crashes)

## Limitations

Intracranial aneurysm detection is a difficult small-lesion 3D medical imaging task. This repository currently uses full-volume center crops, which may miss fine vascular details. The current model should be viewed as a stronger baseline, not a competition-winning solution.

Important current limitations:

* no multi-crop or vessel-localization stage yet — full-volume crops only
* no ensembling of k-fold models at inference time yet
* no ONNX or TensorRT export yet
* no Grad-CAM or other interpretability tooling yet

## Future Work

Planned improvements include:

* ensemble the k-fold models at inference time
* add ONNX export
* add Grad-CAM visualization utilities
* add multi-crop or slab-based training
* add a vessel localization/cropping stage
* test larger training subsets
