"""Segment continuous IMU streams into fixed-length windows (FR-3).

Defaults (window=120 samples, stride=5) follow the configuration validated
for canine IMU gait classification in the Scientific Reports / arXiv study
cited in docs/PRD.md section 3, at a 100-120 Hz sampling rate.
"""

import numpy as np

DEFAULT_WINDOW_SIZE = 120
DEFAULT_STRIDE = 5


def window_signal(data: np.ndarray, window_size: int = DEFAULT_WINDOW_SIZE, stride: int = DEFAULT_STRIDE) -> np.ndarray:
    """Slide a window over `data` (N, C) -> (n_windows, window_size, C)."""
    n_samples = data.shape[0]
    if n_samples < window_size:
        return np.empty((0, window_size, data.shape[1]))

    starts = range(0, n_samples - window_size + 1, stride)
    windows = np.stack([data[s : s + window_size] for s in starts])
    return windows


def window_dataset(sessions: list[np.ndarray], labels: list[str], window_size: int = DEFAULT_WINDOW_SIZE, stride: int = DEFAULT_STRIDE):
    """Window multiple sessions and broadcast each session's label to its windows.

    `sessions` is a list of (N_i, C) arrays, `labels` a parallel list of
    per-session class labels. Returns (X, y) with X shape
    (total_windows, window_size, C) and y shape (total_windows,).
    """
    all_windows = []
    all_labels = []
    for session, label in zip(sessions, labels):
        windows = window_signal(session, window_size, stride)
        all_windows.append(windows)
        all_labels.extend([label] * windows.shape[0])

    X = np.concatenate(all_windows, axis=0) if all_windows else np.empty((0, window_size, 0))
    y = np.array(all_labels)
    return X, y
