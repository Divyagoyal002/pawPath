"""Load IMU session CSVs into windowed datasets for both ML tracks.

Splits by dog_id (not by window) so evaluation reflects generalization to
unseen individuals, matching the literature's honest evaluation protocol
(docs/PRD.md section 3: 82-85% accuracy on unseen dogs vs. 96% in-sample).
"""

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from signal_processing.filters import preprocess_imu
from signal_processing.windowing import window_dataset, DEFAULT_STRIDE, DEFAULT_WINDOW_SIZE

CLASSES = ["healthy", "orthopedic", "neurological"]
SESSION_FILENAME_RE = re.compile(r"(?P<dog_id>.+)_s\d+\.csv$")


def _dog_id_from_filename(path: str) -> str:
    match = SESSION_FILENAME_RE.search(os.path.basename(path))
    return match.group("dog_id") if match else os.path.basename(path)


def load_sessions(data_dir: str):
    """Return (sessions, labels, dog_ids) for every CSV under data_dir/<class>/."""
    sessions, labels, dog_ids = [], [], []
    for label in CLASSES:
        class_dir = os.path.join(data_dir, label)
        for path in sorted(glob.glob(os.path.join(class_dir, "*.csv"))):
            raw = np.loadtxt(path, delimiter=",", skiprows=1)
            imu = raw[:, 1:7]  # drop millis column
            sessions.append(preprocess_imu(imu))
            labels.append(label)
            dog_ids.append(_dog_id_from_filename(path))
    return sessions, labels, dog_ids


def split_by_dog(dog_ids, labels, test_fraction=0.25, seed=42):
    """Return (train_dog_ids, test_dog_ids), split per-class so every class
    is represented in both splits (a plain pooled shuffle can otherwise
    leave a whole class out of the test set by chance)."""
    dog_to_label = dict(zip(dog_ids, labels))
    rng = np.random.default_rng(seed)

    train_dogs, test_dogs = set(), set()
    for label in CLASSES:
        class_dogs = sorted(d for d, l in dog_to_label.items() if l == label)
        if not class_dogs:
            continue
        rng.shuffle(class_dogs)
        n_test = max(1, int(len(class_dogs) * test_fraction))
        test_dogs.update(class_dogs[:n_test])
        train_dogs.update(class_dogs[n_test:])
    return train_dogs, test_dogs


def build_windowed_dataset(
    data_dir: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    test_fraction: float = 0.25,
    seed: int = 42,
):
    """Load, preprocess, window, and dog-level split the dataset.

    Returns dict with X_train, y_train, X_test, y_test (windows) plus the
    raw class label arrays.
    """
    sessions, labels, dog_ids = load_sessions(data_dir)
    if not sessions:
        raise FileNotFoundError(
            f"No session CSVs found under {data_dir}. Run "
            "data/synthetic_gait_generator.py first, or point --data-dir "
            "at a real dataset with the same <class>/<dog>_<session>.csv layout."
        )

    train_dogs, test_dogs = split_by_dog(dog_ids, labels, test_fraction, seed)

    train_sessions, train_labels = [], []
    test_sessions, test_labels = [], []
    for session, label, dog_id in zip(sessions, labels, dog_ids):
        if dog_id in test_dogs:
            test_sessions.append(session)
            test_labels.append(label)
        else:
            train_sessions.append(session)
            train_labels.append(label)

    X_train, y_train = window_dataset(train_sessions, train_labels, window_size, stride)
    X_test, y_test = window_dataset(test_sessions, test_labels, window_size, stride)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_dogs": sorted(train_dogs),
        "test_dogs": sorted(test_dogs),
    }
