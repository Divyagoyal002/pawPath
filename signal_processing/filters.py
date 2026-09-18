"""Filtering utilities for raw IMU streams.

Removes sensor noise and separates the gravity component from the
dynamic acceleration, per the edge-preprocessing step in docs/PRD.md
section 6.2.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def butter_filter(data: np.ndarray, cutoff_hz, fs_hz: float, order: int = 4, btype: str = "low") -> np.ndarray:
    """Zero-phase Butterworth filter applied along axis 0.

    `data` shape: (n_samples, n_channels) or (n_samples,).
    `cutoff_hz` is a scalar for low/high, or a (low, high) tuple for 'band'.
    """
    nyquist = fs_hz / 2.0
    if btype == "band":
        low, high = cutoff_hz
        wn = [low / nyquist, high / nyquist]
    else:
        wn = cutoff_hz / nyquist
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, data, axis=0)


def remove_gravity(accel: np.ndarray, fs_hz: float, cutoff_hz: float = 0.5) -> np.ndarray:
    """Split accelerometer signal into gravity (low-freq) and body (dynamic) components.

    Returns the dynamic/linear acceleration (gravity subtracted), which is
    what carries gait-cycle information.
    """
    gravity = butter_filter(accel, cutoff_hz, fs_hz, order=2, btype="low")
    return accel - gravity


def denoise(data: np.ndarray, fs_hz: float, cutoff_hz: float = 20.0) -> np.ndarray:
    """Low-pass filter to remove high-frequency sensor/motor noise.

    Canine gait frequency content is dominated by stride harmonics well
    below 20 Hz; this cutoff preserves gait dynamics while cutting noise.
    """
    return butter_filter(data, cutoff_hz, fs_hz, order=4, btype="low")


def preprocess_imu(raw: np.ndarray, fs_hz: float = 100.0) -> np.ndarray:
    """Full preprocessing pipeline for a raw (N, 6) [ax,ay,az,gx,gy,gz] array.

    Returns an (N, 6) array: gravity-removed, denoised accel channels
    (0-2) concatenated with denoised gyro channels (3-5).
    """
    accel = raw[:, :3]
    gyro = raw[:, 3:]

    accel_dynamic = remove_gravity(accel, fs_hz)
    accel_clean = denoise(accel_dynamic, fs_hz)
    gyro_clean = denoise(gyro, fs_hz)

    return np.column_stack([accel_clean, gyro_clean])
