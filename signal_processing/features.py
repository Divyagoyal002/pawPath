"""Statistical + spectral feature extraction for the classical ML track
(docs/PRD.md section 6.3).

Given a windowed dataset (n_windows, window_size, n_channels), produces a
flat feature matrix (n_windows, n_features) suitable for SelectKBest +
Random Forest / Isolation Forest.
"""

import numpy as np

CHANNEL_NAMES = ["ax", "ay", "az", "gx", "gy", "gz"]


def _band_energy(fft_mag: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(fft_mag[mask] ** 2))


def extract_window_features(window: np.ndarray, fs_hz: float = 100.0) -> dict:
    """Extract named features for a single (window_size, n_channels) window."""
    feats = {}
    n = window.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    for ch_idx in range(window.shape[1]):
        name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"ch{ch_idx}"
        signal = window[:, ch_idx]

        # Time-domain statistics.
        feats[f"{name}_mean"] = float(np.mean(signal))
        feats[f"{name}_std"] = float(np.std(signal))
        feats[f"{name}_min"] = float(np.min(signal))
        feats[f"{name}_max"] = float(np.max(signal))
        feats[f"{name}_range"] = feats[f"{name}_max"] - feats[f"{name}_min"]
        feats[f"{name}_rms"] = float(np.sqrt(np.mean(signal ** 2)))
        feats[f"{name}_skew"] = float(_skew(signal))
        feats[f"{name}_kurtosis"] = float(_kurtosis(signal))
        feats[f"{name}_zero_crossings"] = float(np.sum(np.diff(np.sign(signal - np.mean(signal))) != 0))
        # Jerk (derivative) magnitude: proxy for motor-control smoothness,
        # elevated in neurological gait (see synthetic generator noise model).
        jerk = np.diff(signal) * fs_hz
        feats[f"{name}_jerk_rms"] = float(np.sqrt(np.mean(jerk ** 2))) if jerk.size else 0.0

        # Spectral features.
        fft_mag = np.abs(np.fft.rfft(signal - np.mean(signal)))
        total_energy = float(np.sum(fft_mag ** 2)) or 1e-9
        feats[f"{name}_fft_energy"] = total_energy
        feats[f"{name}_fft_dominant_freq"] = float(freqs[np.argmax(fft_mag)]) if len(fft_mag) else 0.0
        # Gait stride harmonics live below ~5 Hz; noise/tremor above ~10 Hz.
        feats[f"{name}_band_low_ratio"] = _band_energy(fft_mag, freqs, 0.0, 5.0) / total_energy
        feats[f"{name}_band_high_ratio"] = _band_energy(fft_mag, freqs, 10.0, fs_hz / 2) / total_energy

    # Cross-axis symmetry proxy (FR-5): correlation between the two
    # dominant gait-plane gyro axes drops when limb loading is asymmetric.
    if window.shape[1] >= 5:
        gy = window[:, 4]
        gx = window[:, 3]
        if np.std(gx) > 1e-6 and np.std(gy) > 1e-6:
            feats["gx_gy_correlation"] = float(np.corrcoef(gx, gy)[0, 1])
        else:
            feats["gx_gy_correlation"] = 0.0

    return feats


def _skew(x: np.ndarray) -> float:
    std = np.std(x)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((x - np.mean(x)) / std) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    std = np.std(x)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((x - np.mean(x)) / std) ** 4) - 3.0)


def extract_feature_matrix(windows: np.ndarray, fs_hz: float = 100.0):
    """Extract features for all windows -> (feature_matrix, feature_names)."""
    if windows.shape[0] == 0:
        return np.empty((0, 0)), []

    rows = [extract_window_features(w, fs_hz) for w in windows]
    feature_names = list(rows[0].keys())
    matrix = np.array([[row[name] for name in feature_names] for row in rows])
    return matrix, feature_names
