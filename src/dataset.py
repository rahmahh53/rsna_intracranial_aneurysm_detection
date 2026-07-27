kimport os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.constants import ID_COL, LABEL_COLS, LR_SWAP_INDEX_PAIRS


class CachedRSNADataset(Dataset):
    """
    Loads preprocessed 3D tensors and their multilabel targets.

    Args:
        df:         DataFrame with columns [ID_COL] + LABEL_COLS,
                    one row per series.
        cache_dir:  Directory containing cached .pt tensor files,
                    named <SeriesInstanceUID>.pt
        augment:    Whether to apply training-time augmentation.
    """

    def __init__(self, df: pd.DataFrame, cache_dir: str, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.augment = augment
        self.ids = df[ID_COL].tolist()
        self.labels = df[LABEL_COLS].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        series_uid = self.ids[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        cache_path = os.path.join(self.cache_dir, f"{series_uid}.pt")
        volume = torch.load(cache_path, weights_only=True)  # shape: (1, D, H, W)

        if self.augment:
            volume, label = self._augment(volume, label)

        return volume, label

    def _augment(self, volume: torch.Tensor, label: torch.Tensor):
        """
        Apply training-time augmentation to a (1, D, H, W) volume.

        Augmentations applied:
          - Depth flip (superior-inferior, dim=1): no label change needed
            since brain anatomy is approximately symmetric top-to-bottom
            in this representation.
          - Width flip (left-right, dim=3): mirrors the volume AND swaps
            the corresponding Left/Right label pairs (e.g. Left MCA <->
            Right MCA). Flipping laterality without swapping labels would
            train on incorrect supervision.
          - Random intensity scale and shift: small multiplicative and
            additive perturbations to simulate scanner variability.
          - Additive Gaussian noise: mild noise injection for regularization.

        Height (anterior-posterior) flips are intentionally excluded since
        brain anatomy is not front-back symmetric.
        """
        # Depth flip (dim=1 in C, D, H, W)
        if random.random() < 0.5:
            volume = volume.flip(1)

        # Width flip (dim=3 in C, D, H, W) with anatomically correct label swap
        if random.random() < 0.5:
            volume = volume.flip(3)
            label = label.clone()
            for left_idx, right_idx in LR_SWAP_INDEX_PAIRS:
                label[left_idx], label[right_idx] = label[right_idx].item(), label[left_idx].item()

        # Random intensity scale and shift
        scale = random.uniform(0.9, 1.1)
        shift = random.uniform(-0.05, 0.05)
        volume = volume * scale + shift

        # Additive Gaussian noise
        noise = torch.randn_like(volume) * 0.02
        volume = volume + noise

        return volume, label
