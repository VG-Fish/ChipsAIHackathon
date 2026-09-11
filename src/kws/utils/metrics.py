import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_keywords: int, label_names: list[str]) -> dict:
    """Standard classification metrics plus FAR/FRR, the metrics that matter most
    for an always-on wake-word device (false wake-ups vs. missed keywords).

    Assumes label indices [0, num_keywords) are target keywords and the remaining
    indices (unknown, silence) are non-keyword classes.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=range(len(label_names)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))

    is_keyword_true = y_true < num_keywords
    is_keyword_pred = y_pred < num_keywords

    non_keyword_mask = ~is_keyword_true
    false_accepts = np.sum(non_keyword_mask & is_keyword_pred)
    far = float(false_accepts) / max(int(non_keyword_mask.sum()), 1)

    keyword_mask = is_keyword_true
    false_rejects = np.sum(keyword_mask & (y_true != y_pred))
    frr = float(false_rejects) / max(int(keyword_mask.sum()), 1)

    return {
        "accuracy": accuracy,
        "f1_per_class": dict(zip(label_names, f1_per_class.tolist())),
        "confusion_matrix": cm.tolist(),
        "far": far,
        "frr": frr,
    }
