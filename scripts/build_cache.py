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

from src.constants import ID_COL, ANEURYSM_NAME
from src.data_selection import select_series
from src.preprocess import preprocess_series


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    """
    Read an existing CSV.

    Return an empty DataFrame when the file does not exist
    or exists but contains no data.
    """
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_csv_atomically(dataframe: pd.DataFrame, path: Path) -> None:
    """
    Write to a temporary file first, then replace the original file.

    This reduces the chance of leaving behind a damaged CSV if the
    Kaggle session stops while the file is being written.
    """
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def build_cache(
    cfg: dict,
    overwrite: bool = False,
    save_every: int = 10,
) -> None:
    labels = select_series(cfg)

    series_root = Path(cfg["data"]["series_root"])
    cache_dir = Path(cfg["data"]["cache_dir"])

    target_spacing = float(cfg["preprocessing"]["target_spacing"])
    final_size = tuple(cfg["preprocessing"]["final_size"])
    hu_window = tuple(cfg["preprocessing"]["hu_window"])

    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = cache_dir / "metadata.csv"
    failures_path = cache_dir / "failures.csv"

    existing_metadata = read_csv_or_empty(metadata_path)
    existing_failures = read_csv_or_empty(failures_path)

    # A tensor is considered completely cached only when:
    # 1. Its .pt file exists.
    # 2. Its metadata row exists.
    if ID_COL in existing_metadata.columns:
        completed_ids = set(
            existing_metadata[ID_COL].astype(str)
        )
    else:
        completed_ids = set()

    pending_metadata_rows: list[dict] = []
    pending_failure_rows: list[dict] = []

    processed_since_save = 0

    print("Number of series selected:", len(labels))
    print("Cache directory:", cache_dir)
    print("Previously completed series:", len(completed_ids))
    print("Saving progress every", save_every, "processed series")

    def flush_progress() -> None:
        """
        Merge pending rows into the existing CSV data and save them.

        nonlocal allows this inner function to update variables
        created inside build_cache().
        """
        nonlocal existing_metadata
        nonlocal existing_failures
        nonlocal pending_metadata_rows
        nonlocal pending_failure_rows
        nonlocal completed_ids

        successful_ids: set[str] = set()

        # Save successful preprocessing metadata.
        if pending_metadata_rows:
            new_metadata = pd.DataFrame(pending_metadata_rows)

            successful_ids = set(
                new_metadata[ID_COL].astype(str)
            )

            existing_metadata = pd.concat(
                [existing_metadata, new_metadata],
                ignore_index=True,
            )

            existing_metadata = existing_metadata.drop_duplicates(
                subset=[ID_COL],
                keep="last",
            )

            save_csv_atomically(
                existing_metadata,
                metadata_path,
            )

            completed_ids.update(successful_ids)

        # If a previously failed series now succeeded,
        # remove it from failures.csv.
        if successful_ids and ID_COL in existing_failures.columns:
            existing_failures = existing_failures[
                ~existing_failures[ID_COL]
                .astype(str)
                .isin(successful_ids)
            ].copy()

        # Do not keep a pending failure when that same series
        # succeeded during the current batch.
        if successful_ids:
            pending_failure_rows = [
                failure
                for failure in pending_failure_rows
                if str(failure[ID_COL]) not in successful_ids
            ]

        # Save failures.
        if pending_failure_rows:
            new_failures = pd.DataFrame(pending_failure_rows)

            existing_failures = pd.concat(
                [existing_failures, new_failures],
                ignore_index=True,
            )

            existing_failures = existing_failures.drop_duplicates(
                subset=[ID_COL],
                keep="last",
            )

        # Save failures.csv when it has columns.
        # This also records the removal of old failures that later succeeded.
        if len(existing_failures.columns) > 0:
            save_csv_atomically(
                existing_failures,
                failures_path,
            )

        if pending_metadata_rows or pending_failure_rows:
            print(
                "\nProgress saved:",
                f"{len(existing_metadata)} successful,",
                f"{len(existing_failures)} failed",
            )

        pending_metadata_rows = []
        pending_failure_rows = []

    try:
        for _, row in tqdm(
            labels.iterrows(),
            total=len(labels),
            desc="Building cache",
        ):
            series_id = str(row[ID_COL])

            series_dir = series_root / series_id
            output_path = cache_dir / f"{series_id}.pt"

            # The temporary file prevents a partially written tensor
            # from being mistaken for a completed cache file.
            temporary_output_path = cache_dir / f".{series_id}.pt.tmp"

            cache_is_complete = (
                output_path.exists()
                and series_id in completed_ids
            )

            if cache_is_complete and not overwrite:
                continue

            # Remove an incomplete temporary file from an earlier run.
            if temporary_output_path.exists():
                temporary_output_path.unlink()

            try:
                tensor, metadata = preprocess_series(
                    series_dir=str(series_dir),
                    target_spacing=target_spacing,
                    final_size=final_size,
                    hu_window=hu_window,
                )

                # Save to a temporary file first.
                torch.save(tensor, temporary_output_path)

                # Atomically rename it only after torch.save succeeds.
                os.replace(
                    temporary_output_path,
                    output_path,
                )

                metadata_row = {
                    ID_COL: series_id,
                    **metadata,
                    ANEURYSM_NAME: row[ANEURYSM_NAME],
                }

                pending_metadata_rows.append(metadata_row)

            except Exception as error:
                # Do not leave an incomplete tensor behind.
                if temporary_output_path.exists():
                    temporary_output_path.unlink()

                pending_failure_rows.append(
                    {
                        ID_COL: series_id,
                        "error": str(error),
                    }
                )

            processed_since_save += 1

            if processed_since_save >= save_every:
                flush_progress()
                processed_since_save = 0

    finally:
        # This runs when the loop finishes normally and also when you
        # manually interrupt it with Ctrl+C.
        flush_progress()

    print("\nCache building complete.")
    print("Successful series:", len(existing_metadata))
    print("Failed series:", len(existing_failures))
    print("Metadata:", metadata_path)
    print("Failures:", failures_path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet3d.yaml",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess series even when their cache already exists.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save metadata and failure progress after this many series.",
    )

    args = parser.parse_args()

    if args.save_every < 1:
        raise ValueError("--save-every must be at least 1.")

    cfg = load_config(args.config)

    build_cache(
        cfg=cfg,
        overwrite=args.overwrite,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
