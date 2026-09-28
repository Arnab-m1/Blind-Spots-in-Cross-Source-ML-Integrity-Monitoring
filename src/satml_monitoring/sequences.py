"""Cycle-preserving temporal preprocessing helpers."""

from __future__ import annotations

import numpy as np


def cycle_windows(sequences: np.ndarray, length: int, *, predict: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Create within-cycle windows without joining adjacent cycles."""
    count = sequences.shape[1] - length + (0 if predict else 1)
    if count <= 0:
        raise ValueError("window length must be shorter than the cycle")
    values = np.stack([sequences[:, start:start + length] for start in range(count)], axis=1)
    values = values.reshape(-1, length, sequences.shape[2])
    if not predict:
        return values, None
    targets = np.stack([sequences[:, start + length] for start in range(count)], axis=1)
    return values, targets.reshape(-1, sequences.shape[2])
