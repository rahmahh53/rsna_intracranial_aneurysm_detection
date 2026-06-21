import os
from typing import Dict, Tuple

import numpy as np
import pydicom
import torch
from scipy.ndimage import zoom


def sort_dicom_key(ds) -> float:
    """
    Sort DICOM slices using ImagePositionPatient when available,
    otherwise fall back to InstanceNumber.
    """
    image_position = getattr(ds, "ImagePositionPatient", None)

    if image_position is not None and len(image_position) == 3:
        return float(image_position[2])

    return float(getattr(ds, "InstanceNumber", 0))


def convert_to_hu(ds, pixel_array: np.ndarray) -> np.ndarray:
    """
    Convert raw CT pixel values to Hounsfield Units.
    """
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)

    return pixel_array.astype(np.float32) * slope + intercept


def get_spacing(ds) -> Tuple[float, float, float]:
    """
    Return spacing as (z, y, x).
    """
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])

    spacing_y = float(pixel_spacing[0])
    spacing_x = float(pixel_spacing[1])

    spacing_z = getattr(ds, "SpacingBetweenSlices", None)

    if spacing_z is None:
        spacing_z = getattr(ds, "SliceThickness", 1.0)

    spacing_z = float(spacing_z)

    return spacing_z, spacing_y, spacing_x


def load_dicom_series(series_dir: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load a DICOM series into a 3D volume.

    Returns:
        volume: numpy array with shape (D, H, W)
        spacing: tuple (z, y, x)
    """
    dicom_paths = []

    for root, _, files in os.walk(series_dir):
        for file_name in files:
            if file_name.lower().endswith(".dcm"):
                dicom_paths.append(os.path.join(root, file_name))

    if len(dicom_paths) == 0:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")

    dicoms = [pydicom.dcmread(path, force=True) for path in dicom_paths]
    dicoms.sort(key=sort_dicom_key)

    spacing = get_spacing(dicoms[0])

    frames = []

    for ds in dicoms:
        arr = ds.pixel_array

        # Some DICOMs can be multi-frame: (N, H, W)
        if arr.ndim == 3:
            frame_list = arr
        else:
            frame_list = [arr]

        for frame in frame_list:
            frame = frame.astype(np.float32)

            if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                frame = np.max(frame) - frame

            if (
                getattr(ds, "Modality", "CT") == "CT"
                or hasattr(ds, "RescaleSlope")
                or hasattr(ds, "RescaleIntercept")
            ):
                frame = convert_to_hu(ds, frame)

            frames.append(frame.astype(np.float32))

    volume = np.stack(frames, axis=0)

    return volume, spacing


def window_and_scale(volume: np.ndarray, hu_window: Tuple[float, float]) -> np.ndarray:
    """
    Clip HU values to a fixed window and scale to [0, 1].
    """
    low, high = hu_window

    volume = np.clip(volume, low, high)
    volume = (volume - low) / (high - low)

    return volume.astype(np.float32)


def resample_volume(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    target_spacing: float,
) -> np.ndarray:
    """
    Resample volume to approximately isotropic spacing.

    spacing is (z, y, x).
    """
    spacing_z, spacing_y, spacing_x = spacing

    zoom_factors = (
        spacing_z / target_spacing,
        spacing_y / target_spacing,
        spacing_x / target_spacing,
    )

    return zoom(volume, zoom_factors, order=1).astype(np.float32)


def center_crop_or_pad(volume: np.ndarray, final_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Center crop or pad a 3D volume to final_size = (D, H, W).
    """
    target_d, target_h, target_w = final_size
    depth, height, width = volume.shape

    pad_d = max(0, target_d - depth)
    pad_h = max(0, target_h - height)
    pad_w = max(0, target_w - width)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(
            volume,
            (
                (pad_d // 2, pad_d - pad_d // 2),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
            ),
            mode="constant",
            constant_values=0,
        )

    depth, height, width = volume.shape

    start_d = max(0, (depth - target_d) // 2)
    start_h = max(0, (height - target_h) // 2)
    start_w = max(0, (width - target_w) // 2)

    volume = volume[
        start_d : start_d + target_d,
        start_h : start_h + target_h,
        start_w : start_w + target_w,
    ]

    return volume.astype(np.float32)


def preprocess_series(
    series_dir: str,
    target_spacing: float,
    final_size: Tuple[int, int, int],
    hu_window: Tuple[float, float],
) -> Tuple[torch.Tensor, Dict]:
    """
    Full preprocessing pipeline.

    Returns:
        tensor: shape (1, 1, D, H, W)
        metadata: dictionary of preprocessing metadata
    """
    volume, spacing = load_dicom_series(series_dir)

    original_shape = volume.shape

    volume = window_and_scale(volume, hu_window)
    volume = resample_volume(volume, spacing, target_spacing)
    volume = center_crop_or_pad(volume, final_size)

    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)

    metadata = {
        "original_depth": original_shape[0],
        "original_height": original_shape[1],
        "original_width": original_shape[2],
        "spacing_z": spacing[0],
        "spacing_y": spacing[1],
        "spacing_x": spacing[2],
        "processed_depth": final_size[0],
        "processed_height": final_size[1],
        "processed_width": final_size[2],
    }

    return tensor, metadata
