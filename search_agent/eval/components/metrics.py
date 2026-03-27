"""Shared metric helpers for component evals."""
from __future__ import annotations


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator > 0 else default


def accuracy(correct: int, total: int) -> float | None:
    return None if total == 0 else correct / total


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def per_class_metrics(
    results: list[dict],
    label_key: str,
    pred_key: str,
    classes: list[str],
) -> dict[str, dict[str, float]]:
    """Per-class precision/recall/f1 from a list of result dicts."""
    counts: dict[str, dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    for r in results:
        expected = r.get(label_key)
        predicted = r.get(pred_key)
        for c in classes:
            if expected == c and predicted == c:
                counts[c]["tp"] += 1
            elif expected != c and predicted == c:
                counts[c]["fp"] += 1
            elif expected == c and predicted != c:
                counts[c]["fn"] += 1
    return {c: precision_recall_f1(**counts[c]) for c in classes}


def percentile(data: list[float], pct: int) -> float | None:
    if not data:
        return None
    s = sorted(data)
    k = (len(s) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
