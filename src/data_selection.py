import pandas as pd
from sklearn.model_selection import train_test_split
from src.constants import ID_COL, LABEL_COLS, ANEURYSM_NAME


def select_series(cfg: dict) -> pd.DataFrame:
    labels = pd.read_csv(cfg["data"]["labels_csv"])
    labels = labels[[ID_COL] + LABEL_COLS].dropna().reset_index(drop=True)

    max_series = cfg["data"].get("max_series")
    strategy = cfg["data"].get("sampling_strategy", "all")
    seed = cfg["seed"]

    if max_series is None and len(labels) <= max_series:
        return labels

    positives = labels[labels[ANEURYSM_NAME] == 1]
    negatives = labels[labels[ANEURYSM_NAME] == 0]

    if strategy == "balanced":
        n_positive = max_series // 2
        n_negative = max_series - n_positive

        if len(positives) <  n_positive:
           raise ValueError("Not enough positive samples for balanced sampling")
        if len(negatives) < n_negative:
           raise ValueError("Not enough negative samples for balanced sampling")

        positive_sample = positives.sample(n=n_positive, random_state=seed)
        negative_sample = negatives.sample(n=n_negative, random_state=seed)
        selected = pd.concat([positive_sample, negative_sample], axis=0)

        return selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if strategy == "stratified":
        _, selected = train_test_split(labels, test_size=max_series, stratify=labels[ANEURYSM_NAME], random_state=seed)
        return selected.reset_index(drop=True)

    if strategy == "all":
        return labels

    raise ValueError("sampling strategy must be 'balanced', 'stratified', or 'all'.")

