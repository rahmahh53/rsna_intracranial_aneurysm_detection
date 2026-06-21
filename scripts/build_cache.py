import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.constants import ID_COL, LABEL_COLS, ANEURYSM_NAME
from src.preprocess import preprocess_series


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_labels(cfg: dict) -> pd.DataFrame:
    labels = pd.read_csv(cfg["data"]["labels_csv"])
    labels = labels[[ID_COL] + LABEL_COLS].dropna().reset_index(drop=True)

    max_series = cfg["data"].get("max_series", None)

    if max_series is not None and len(labels) > max_series:
        positives = labels[labels[ANEURYSM_NAME] == 1]
        negatives = labels[labels[ANEURYSM_NAME] == 0]

        if len(positives) >= max_series:
            labels = positives.sample(n=max_series, random_state=cfg["seed"])
        else:
            n_negatives = max_series - len(positives)
            negatives = negatives.sample(n=n_negatives, random_state=cfg["seed"])
            labels = pd.concat([positives, negatives], axis=0)

        labels = labels.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)

    return labels


def build_cache(cfg: dict, overwrite: bool = False) -> None:
    labels = prepare_labels(cfg)

    series_root = cfg["data"]["series_root"]
    cache_dir = cfg["data"]["cache_dir"]

    target_spacing = float(cfg["preprocessing"]["target_spacing"])
    final_size = tuple(cfg["preprocessing"]["final_size"])
    hu_window = tuple(cfg["preprocessing"]["hu_window"])

    os.makedirs(cache_dir, exist_ok=True)

    metadata_rows = []
    failed_rows = []

    print("Number of series selected:", len(labels))
    print("Cache directory:", cache_dir)

    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Building cache"):
        series_id = row[ID_COL]
        series_dir = os.path.join(series_root, series_id)
        output_path = os.path.join(cache_dir, f"{series_id}.pt")

        if os.path.exists(output_path) and not overwrite:
            continue

        try:
            tensor, metadata = preprocess_series(
                series_dir=series_dir,
                target_spacing=target_spacing,
                final_size=final_size,
                hu_window=hu_window,
            )

            torch.save(tensor, output_path)

            metadata_row = {
                ID_COL: series_id,
                **metadata,
                ANEURYSM_NAME: row[ANEURYSM_NAME],
            }

            metadata_rows.append(metadata_row)

        except Exception as e:
            failed_rows.append(
                {
                    ID_COL: series_id,
                    "error": str(e),
                }
            )

    metadata_path = os.path.join(cache_dir, "metadata.csv")
    failures_path = os.path.join(cache_dir, "failures.csv")

    if metadata_rows:
        pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
        print("Saved metadata:", metadata_path)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failures_path, index=False)
        print("Saved failures:", failures_path)

    print("Finished cache build.")
    print("Successful:", len(metadata_rows))
    print("Failed:", len(failed_rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet3d.yaml",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    build_cache(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
