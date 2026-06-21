import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allows "python scripts/train_resnet3d.py" to import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.constants import ID_COL, LABEL_COLS, ANEURYSM_NAME
from src.metrics import multilabel_macro_auc, per_label_auc
from src.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CachedRSNADataset(Dataset):
    """
    Dataset that reads preprocessed .pt tensors from cache.

    Expected tensor shape per file:
        (1, 1, D, H, W)

    Returned x shape after squeezing batch dim:
        (1, D, H, W)
    """

    def __init__(self, df: pd.DataFrame, cache_dir: str, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.train = train

        self.ids = self.df[ID_COL].tolist()
        self.labels = self.df[LABEL_COLS].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        series_id = self.ids[idx]
        tensor_path = os.path.join(self.cache_dir, f"{series_id}.pt")

        x = torch.load(tensor_path, map_location="cpu")

        # Original cache may save (1, 1, D, H, W).
        # DataLoader adds batch dimension, so we return (1, D, H, W).
        if x.ndim == 5:
            x = x.squeeze(0)

        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.train:
            x = self.augment(x)

        return x, y

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lightweight 3D augmentation.

        x shape:
            (1, D, H, W)
        """
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])  # depth

        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])  # height

        if random.random() < 0.5:
            x = torch.flip(x, dims=[3])  # width

        if random.random() < 0.5:
            scale = 1.0 + random.uniform(-0.10, 0.10)
            shift = random.uniform(-0.05, 0.05)
            x = x * scale + shift

        return x


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_labels(cfg: dict) -> pd.DataFrame:
    labels_csv = cfg["data"]["labels_csv"]
    max_series = cfg["data"]["max_series"]

    labels = pd.read_csv(labels_csv)
    labels = labels[[ID_COL] + LABEL_COLS].dropna().reset_index(drop=True)

    if max_series is not None and len(labels) > max_series:
        positives = labels[labels[ANEURYSM_NAME] == 1]
        negatives = labels[labels[ANEURYSM_NAME] == 0]

        if len(positives) >= max_series:
            labels = positives.sample(n=max_series, random_state=cfg["seed"])
        else:
            n_neg = max_series - len(positives)
            negatives = negatives.sample(n=n_neg, random_state=cfg["seed"])
            labels = pd.concat([positives, negatives], axis=0)

        labels = labels.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)

    return labels


def filter_to_cached(labels: pd.DataFrame, cache_dir: str) -> pd.DataFrame:
    cached_ids = {
        Path(p).stem
        for p in Path(cache_dir).glob("*.pt")
    }

    filtered = labels[labels[ID_COL].isin(cached_ids)].reset_index(drop=True)

    print(f"Labels before cache filtering: {len(labels)}")
    print(f"Cached tensors found: {len(cached_ids)}")
    print(f"Labels after cache filtering: {len(filtered)}")

    return filtered


def compute_pos_weight(df_train: pd.DataFrame, device: str) -> torch.Tensor:
    label_means = df_train[LABEL_COLS].mean().values
    pos_weight = (1.0 - label_means) / (label_means + 1e-6)

    # Avoid extreme weights from rare labels dominating training.
    pos_weight = np.clip(pos_weight, 1.0, 20.0)

    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def run_validation(model, valid_loader, criterion, device, use_amp: bool):
    model.eval()

    valid_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            with autocast(enabled=(use_amp and device == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            valid_loss += loss.item() * xb.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(yb.detach().cpu())

    valid_loss /= max(1, len(valid_loader.dataset))

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))

    macro_auc = multilabel_macro_auc(all_targets, all_probs)
    label_aucs = per_label_auc(all_targets, all_probs, LABEL_COLS)

    return valid_loss, macro_auc, label_aucs


def train(cfg: dict):
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    output_dir = cfg["outputs"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    labels = prepare_labels(cfg)
    labels = filter_to_cached(labels, cfg["data"]["cache_dir"])

    if len(labels) == 0:
        raise RuntimeError(
            "No usable cached tensors found. You need to build the cache before training."
        )

    y_strat = labels[ANEURYSM_NAME].astype(int)

    df_train, df_valid = train_test_split(
        labels,
        test_size=cfg["data"]["val_size"],
        random_state=cfg["seed"],
        stratify=y_strat,
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    print("Train size:", len(df_train))
    print("Valid size:", len(df_valid))
    print("Train aneurysm prevalence:", df_train[ANEURYSM_NAME].mean())
    print("Valid aneurysm prevalence:", df_valid[ANEURYSM_NAME].mean())

    train_ds = CachedRSNADataset(
        df=df_train,
        cache_dir=cfg["data"]["cache_dir"],
        train=True,
    )

    valid_ds = CachedRSNADataset(
        df=df_valid,
        cache_dir=cfg["data"]["cache_dir"],
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device == "cuda"),
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device == "cuda"),
    )

    model = build_model(
        model_name=cfg["training"]["model_name"],
        n_outputs=len(LABEL_COLS),
    ).to(device)

    pos_weight = compute_pos_weight(df_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    use_amp = bool(cfg["training"]["use_amp"])
    scaler = GradScaler(enabled=(use_amp and device == "cuda"))

    best_auc = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for xb, yb in pbar:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(use_amp and device == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * xb.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= max(1, len(train_ds))

        valid_loss, valid_auc, label_aucs = run_validation(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step(valid_auc)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_macro_auc": valid_auc,
        }

        history.append(row)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} "
            f"valid_macro_auc={valid_auc:.4f}"
        )

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "best_auc": best_auc,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "label_aucs": label_aucs,
            }

            torch.save(checkpoint, cfg["outputs"]["best_checkpoint"])
            print(f"Saved new best checkpoint: AUC={best_auc:.4f}")

        if epoch - best_epoch >= cfg["training"]["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    metrics_df = pd.DataFrame(history)
    metrics_df.to_csv(cfg["outputs"]["metrics_csv"], index=False)
    print("Saved training metrics to:", cfg["outputs"]["metrics_csv"])
    print("Best validation macro AUC:", best_auc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet3d_baseline.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
