# K-fold cross-validation training for the RSNA aneurysm detection model.

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.constants import ID_COL, LABEL_COLS, ANEURYSM_NAME
from src.data_selection import select_series
from src.metrics import multilabel_macro_auc, per_label_auc

from scripts.train_model import (
    filter_to_cached,
    load_config,
    set_seed,
    train_one_fold,
)


def run_kfold(cfg: dict, n_folds: int):
    set_seed(cfg["seed"])

    output_dir = cfg["outputs"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    labels = select_series(cfg)
    labels = filter_to_cached(labels, cfg["data"]["cache_dir"])

    if len(labels) == 0:
        raise RuntimeError(
            "No usable cached tensors found. You need to build the cache before training."
        )

    labels = labels.reset_index(drop=True)
    y_strat = labels[ANEURYSM_NAME].astype(int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg["seed"])

    model_name = cfg["training"]["model_name"]
    checkpoint_template = os.path.join(output_dir, f"best_{model_name}_fold{{fold}}.pth")
    metrics_template = os.path.join(output_dir, f"metrics_{model_name}_fold{{fold}}.csv")

    fold_results = []
    oof_rows = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(labels, y_strat)):
        df_train = labels.iloc[train_idx]
        df_valid = labels.iloc[valid_idx]

        result = train_one_fold(
            cfg=cfg,
            df_train=df_train,
            df_valid=df_valid,
            checkpoint_path=checkpoint_template.format(fold=fold),
            metrics_csv_path=metrics_template.format(fold=fold),
            fold_label=f"fold {fold}/{n_folds - 1}",
        )

        fold_results.append({"fold": fold, "best_auc": result["best_auc"]})

        if result["oof_probs"] is not None:
            for series_id, probs, targets in zip(
                result["series_ids"], result["oof_probs"], result["oof_targets"]
            ):
                row = {"fold": fold, ID_COL: series_id}
                for i, col in enumerate(LABEL_COLS):
                    row[f"prob_{col}"] = probs[i]
                    row[f"target_{col}"] = targets[i]
                oof_rows.append(row)

    oof_df = pd.DataFrame(oof_rows)
    oof_csv_path = os.path.join(output_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_csv_path, index=False)
    print("Saved out-of-fold predictions to:", oof_csv_path)

    prob_cols = [f"prob_{c}" for c in LABEL_COLS]
    target_cols = [f"target_{c}" for c in LABEL_COLS]

    oof_probs = oof_df[prob_cols].values
    oof_targets = oof_df[target_cols].values

    overall_macro_auc = multilabel_macro_auc(oof_targets, oof_probs)
    overall_label_aucs = per_label_auc(oof_targets, oof_probs, LABEL_COLS)

    print("=" * 60)
    print(f"Overall OOF macro AUC across {n_folds} folds: {overall_macro_auc:.4f}")
    print("Per-label OOF AUC:")
    for label, auc in overall_label_aucs.items():
        print(f"  {label}: {auc:.4f}")
    print("=" * 60)

    summary_rows = fold_results + [
        {"fold": "overall_oof", "best_auc": overall_macro_auc}
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, "kfold_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print("Saved fold summary to:", summary_csv_path)

    return overall_macro_auc, overall_label_aucs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/resnet3d.yaml")
    parser.add_argument(
        "--n_folds",
        type=int,
        default=None,
        help="Overrides training.n_folds in the config if provided.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_folds = args.n_folds or cfg["training"].get("n_folds", 5)

    run_kfold(cfg, n_folds=n_folds)


if __name__ == "__main__":
    main()
