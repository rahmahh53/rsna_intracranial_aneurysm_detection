# RSNA Intracranial Aneurysm Detection — 3D ResNet Baseline

This repository contains a reproducible 3D medical imaging pipeline for the RSNA Intracranial Aneurysm Detection challenge. The project focuses on building an end-to-end machine learning workflow for DICOM CT angiography series, including preprocessing, tensor caching, model training, validation, checkpointing, and deployment-oriented model export.

The original version of this project used a compact custom 3D CNN. This version refactors the codebase and upgrades the modeling approach to a deeper 3D ResNet-style architecture for volumetric multi-label classification.

## Project Goals

The goal of this project is to build a stronger and cleaner baseline for 3D medical image classification while emphasizing reproducibility, modular code organization, and model evaluation.

This project demonstrates:

* DICOM series loading
* Hounsfield Unit conversion
* MONOCHROME1 handling
* voxel resampling
* center cropping and padding
* cached tensor generation
* multi-label classification
* class imbalance handling
* validation macro AUC tracking
* 3D ResNet model training
* reproducible configuration files

## Repository Structure

```text
rsna_project/
├── configs/
│   └── resnet3d.yaml
├── scripts/
│   ├── build_cache.py
│   └── train_model.py
├── src/
│   ├── constants.py
│   ├── metrics.py
│   ├── models.py
│   └── preprocess.py
├── notebooks/
├── reports/
├── requirements.txt
└── README.md
```

## Model

The current model is a 3D ResNet-style neural network for multi-label volumetric classification. It uses residual blocks to learn deeper 3D spatial representations from CT angiography volumes.

The model predicts 14 labels:

* 13 vascular territory labels
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

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Build cached tensors:

```bash
python scripts/build_cache.py --config configs/resnet3d.yaml
```

Train the model:

```bash
python scripts/train_model.py --config configs/resnet3d.yaml
```

## Configuration

Training and preprocessing settings are controlled through:

```text
configs/resnet3d.yaml
```

The config file defines:

* dataset paths
* cache directory
* maximum number of series
* validation split size
* target voxel spacing
* input volume size
* HU window
* learning rate
* batch size
* number of epochs
* checkpoint paths

## Evaluation

The training script tracks validation macro AUC across the 14 prediction labels. Labels that contain only one class in a validation split are skipped when computing AUC.

The best checkpoint is selected based on validation macro AUC rather than validation loss.

## Current Status

This project is an active v2 refactor of an earlier Kaggle notebook implementation. The current focus is improving:

* model architecture
* preprocessing quality
* code organization
* validation tracking
* reproducibility
* deployment readiness

## Limitations

Intracranial aneurysm detection is a difficult small-lesion 3D medical imaging task. This repository currently uses full-volume center crops, which may miss fine vascular details. The current model should be viewed as a stronger baseline, not a competition-winning solution.

Important current limitations:

* no multi-crop training yet
* no vessel localization stage
* no k-fold cross-validation yet
* no pretrained 3D backbone yet
* no ensembling
* no ONNX or TensorRT export yet

## Future Work

Planned improvements include:

* add TorchScript export
* add ONNX export
* add inference latency benchmarking
* add Grad-CAM visualization utilities
* add multi-crop or slab-based training
* add k-fold cross-validation
* compare shallow 3D CNN vs 3D ResNet performance
* test larger training subsets
* improve validation reporting with per-label metrics


