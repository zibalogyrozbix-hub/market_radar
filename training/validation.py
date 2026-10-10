"""Chronological splitting utilities. Random splits are never used for time series."""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def chronological_split(n: int, train_frac: float = 0.70, calib_frac: float = 0.15, gap: int = 0):
    """Return (train, calib, test) index arrays with a purge `gap` between blocks.

    The gap (>= prediction horizon) stops a training label (which looks `horizon` bars ahead) from
    overlapping the next block's period.
    """
    if not (0 < train_frac < 1 and 0 < calib_frac < 1 and train_frac + calib_frac < 1):
        raise ValueError("invalid fractions")
    a = int(n * train_frac)
    b = int(n * (train_frac + calib_frac))
    train = np.arange(0, a)
    calib = np.arange(a + gap, b)
    test = np.arange(b + gap, n)
    return train, calib, test


def walk_forward_splits(n: int, n_splits: int = 5, min_train: int = 1000, gap: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward: train on [0, k), test on [k+gap, k+gap+size)."""
    if n <= min_train + gap + n_splits:
        return
    test_size = (n - min_train - gap) // n_splits
    if test_size < 20:
        return
    for i in range(n_splits):
        k = min_train + i * test_size
        yield np.arange(0, k), np.arange(k + gap, min(n, k + gap + test_size))


def assert_chronological(train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    if len(train_idx) and len(test_idx) and train_idx.max() >= test_idx.min():
        raise AssertionError("train/test overlap in time (look-ahead)")
