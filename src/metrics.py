import numpy as np
from sklearn.metrics import roc_auc_score


def multilabel_macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute macro AUC across multiple binary labels.

    Some labels may have only 0s or only 1s in a validation split.
    roc_auc_score cannot be computed for those labels, so we skip them.
    """
    aucs = []

    for label_idx in range(y_true.shape[1]):
        targets = y_true[:, label_idx]
        scores = y_prob[:, label_idx]

        if np.unique(targets).size < 2:
            continue

        aucs.append(roc_auc_score(targets, scores))

    if len(aucs) == 0:
        return float("nan")

    return float(np.mean(aucs))


def per_label_auc(y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str]) -> dict:
    """
    Compute AUC separately for each label.

    Returns NaN for labels where AUC cannot be computed.
    """
    results = {}

    for label_idx, label_name in enumerate(label_names):
        targets = y_true[:, label_idx]
        scores = y_prob[:, label_idx]

        if np.unique(targets).size < 2:
            results[label_name] = float("nan")
            continue

        results[label_name] = float(roc_auc_score(targets, scores))

    return results
