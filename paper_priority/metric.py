"""Detailed Evaluation-page metric; this is NOT verification of Kaggle's hidden code."""
from __future__ import annotations

import math
import numpy as np


def score_priorities(ids, labels, priorities) -> dict:
    ids = np.asarray(ids, dtype=str)
    y = np.asarray(labels, dtype=float)
    p = np.asarray(priorities, dtype=float)
    if not (ids.ndim == y.ndim == p.ndim == 1 and len(ids) == len(y) == len(p)):
        raise ValueError("IDs, labels and priorities must be equally sized one-dimensional arrays.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs in metric inputs.")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Priorities must be finite and nonnegative.")
    if not np.all(np.isin(y, [1, 2, 3, 4, 5])):
        raise ValueError("Labels must be integers 1..5.")
    chosen = np.flatnonzero(p > 0)
    chosen = chosen[np.lexsort((ids[chosen], -p[chosen]))]
    if not len(chosen):
        return {"relevance": 0.0, "coverage": 0.0, "score": 0.0, "recommended": 0}
    weights = 1 / np.log2(np.arange(1, len(chosen) + 1) + 1)
    q = float(np.average((y[chosen] - 1) / 4, weights=weights))
    worthwhile = int(np.sum(y > 1))
    r = float(np.sum(y[chosen] > 1) / worthwhile) if worthwhile else 0.0
    score = 2 * q * r / (q + r) if q and r else 0.0
    return {"relevance": q, "coverage": r, "score": score, "recommended": int(len(chosen))}


def make_priorities(ids, gains, fraction: float) -> np.ndarray:
    """Select top fraction; encode order on (0, 10] with text-ID tie breaking."""
    ids, gains = np.asarray(ids, dtype=str), np.asarray(gains, dtype=float)
    if not 0 < fraction <= 1 or len(ids) != len(gains):
        raise ValueError("Invalid selection fraction or prediction length.")
    if not np.all(np.isfinite(gains)):
        raise ValueError("Nonfinite model predictions.")
    order = np.lexsort((ids, -gains))
    count = min(len(ids), math.ceil(fraction * len(ids)))
    result = np.zeros(len(ids), dtype=float)
    if count:
        result[order[:count]] = 10 * np.arange(count, 0, -1) / count
    return result
