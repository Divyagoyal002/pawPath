"""Synthetic canine IMU gait data generator.

No public canine IMU gait dataset exists for direct reuse (docs/PRD.md
section 7), so this generates plausible back/harness-mounted 6-axis IMU
traces to develop and sanity-check the signal-processing and ML pipeline
before real dog data is collected. It is NOT a substitute for real data —
swap `data/real/` in once a clinical partner is onboarded.

Model: a walking/trotting gait is a quasi-periodic signal at a stride
frequency with a few harmonics. Class-specific distortions:

  - healthy:      symmetric, regular stride timing, low jerk noise.
  - orthopedic:   amplitude asymmetry every other stride (the dog
                  unloads a painful limb) + more stride-to-stride
                  timing variability.
  - neurological: irregular stride timing/phase jitter (ataxia) and
                  elevated high-frequency noise (poor motor control),
                  amplitude roughly symmetric.

Output: one CSV per session at data/synthetic/<class>/<dog_id>_<session>.csv
with columns matching the firmware log schema
(millis,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps) plus a `label` column.
"""

import argparse
import os

import numpy as np

CLASSES = ["healthy", "orthopedic", "neurological"]
SAMPLE_RATE_HZ = 100


def _gravity_bias():
    # Back/harness mount close to level with the spine: mostly on Z.
    return np.array([0.02, -0.05, 0.98])


def generate_session(
    duration_s: float,
    dog_size: str,
    label: str,
    rng: np.random.Generator,
):
    """Return an (N, 6) array of [ax,ay,az,gx,gy,gz] at SAMPLE_RATE_HZ."""
    n = int(duration_s * SAMPLE_RATE_HZ)
    t = np.arange(n) / SAMPLE_RATE_HZ

    stride_hz = {"small": 2.4, "medium": 2.0, "large": 1.6}[dog_size]
    base_gyro_amp = {"small": 60.0, "medium": 90.0, "large": 120.0}[dog_size]
    base_acc_amp = {"small": 0.35, "medium": 0.5, "large": 0.65}[dog_size]

    # Stride-to-stride timing variability (phase jitter accumulated over time).
    jitter_std = {"healthy": 0.01, "orthopedic": 0.03, "neurological": 0.08}[label]
    phase_noise = np.cumsum(rng.normal(0, jitter_std, size=n))
    phase = 2 * np.pi * stride_hz * t + phase_noise

    # Amplitude asymmetry: alternate strides scaled differently (limping).
    stride_index = np.floor(phase / (2 * np.pi)).astype(int)
    asymmetry_depth = {"healthy": 0.0, "orthopedic": 0.35, "neurological": 0.08}[label]
    asym = 1.0 - asymmetry_depth * (stride_index % 2)

    # High-frequency jerk/noise floor (poor motor control -> more noise).
    noise_std_acc = {"healthy": 0.02, "orthopedic": 0.04, "neurological": 0.07}[label]
    noise_std_gyro = {"healthy": 2.0, "orthopedic": 4.0, "neurological": 9.0}[label]

    grav = _gravity_bias()
    ax = grav[0] + base_acc_amp * 0.6 * asym * np.sin(phase) + base_acc_amp * 0.15 * np.sin(2 * phase)
    ay = grav[1] + base_acc_amp * 0.4 * asym * np.cos(phase)
    az = grav[2] + base_acc_amp * 0.5 * asym * np.sin(2 * phase + 0.3)

    gx = base_gyro_amp * 0.8 * asym * np.sin(phase + 0.2)
    gy = base_gyro_amp * 1.0 * asym * np.sin(phase)
    # Lateral sway (yaw-ish axis) is elevated for neurological gait (ataxia).
    sway_gain = {"healthy": 1.0, "orthopedic": 1.1, "neurological": 2.2}[label]
    gz = sway_gain * base_gyro_amp * 0.3 * np.sin(phase * 0.5 + 0.1)

    ax += rng.normal(0, noise_std_acc, n)
    ay += rng.normal(0, noise_std_acc, n)
    az += rng.normal(0, noise_std_acc, n)
    gx += rng.normal(0, noise_std_gyro, n)
    gy += rng.normal(0, noise_std_gyro, n)
    gz += rng.normal(0, noise_std_gyro, n)

    return np.column_stack([ax, ay, az, gx, gy, gz])


def write_csv(path, data, start_ms=0):
    header = "millis,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"
    millis = start_ms + np.arange(data.shape[0]) * (1000 // SAMPLE_RATE_HZ)
    out = np.column_stack([millis, data])
    fmt = ["%d"] + ["%.4f"] * 6
    np.savetxt(path, out, delimiter=",", header=header, comments="", fmt=fmt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "synthetic"))
    parser.add_argument("--dogs-per-class", type=int, default=8)
    parser.add_argument("--sessions-per-dog", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sizes = ["small", "medium", "large"]

    for label in CLASSES:
        class_dir = os.path.join(args.out_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        for dog_idx in range(args.dogs_per_class):
            dog_id = f"{label}_{dog_idx:03d}"
            dog_size = sizes[dog_idx % len(sizes)]
            for session_idx in range(args.sessions_per_dog):
                data = generate_session(args.duration_s, dog_size, label, rng)
                fname = f"{dog_id}_s{session_idx}.csv"
                write_csv(os.path.join(class_dir, fname), data)
    print(f"Wrote synthetic sessions to {args.out_dir}")


if __name__ == "__main__":
    main()
