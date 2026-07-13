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
from src.data_selection import select_series
from src.preprocess import preprocess_series


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_cache(cfg: dict, overwrite: bool = False) -> None:
    labels = select_series(cfg)

    series_root = cfg["data"]["series_root"]
    cache_dir = cfg["data"]["cache_dir"]

    target_spacing = float(cfg["preprocessing"]["target_spacing"])
    final_size = tuple(cfg["preprocessing"]["final_size"])
    hu_window = tuple(cfg["preprocessing"]["hu_window"])

    os.makedirs(cache_dir, exist_ok=True)

    metadata_path = os.path.join(cache_dir, "metadata.csv")
    failures_path = os.path.join(cache_dir, "failures.csv")

    if os.path.exists(metadata_path):
        existing_metadata = pd.read_csv(metadata_path)
    else:
        existing_metadata = pd.DataFrame()

    if os.path.exists(failures_path):
        existing_failures = pd.read_csv(failures_path)
    else:
        existing_failures = pd.DataFrame()

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


    new_metadata = pd.DataFrame(metadata_rows)
    combined_metadata = pd.concat([existing_metadata, new_metadata], ignore_index=True)
    combined_metadata = combined_metadata.drop_duplicates(subset=[ID_COL], keep="last")
    combined_metadata.to_csv(metadata_path, index=False)

    new_failures = pd.DataFrame(failed_rows)
    combined_failures = pd.concat([existing_failures, new_failures], ignore_index=True)
    combined_failures = combined_failures.drop_duplicates(subset=[ID_COL], keep="last")
    combined_failures.to_csv(failures_path, index=False)

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
