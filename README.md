# PawPath

A non-invasive IMU-based system for early detection of canine spinal injury, lameness and arthritis.

Full requirements and literature review: [docs/PRD.md](docs/PRD.md).

## Repo layout

```
firmware/           ESP32 + MPU6050 sensor node (BLE streaming + SD logging)
data/                synthetic data generator + real-data layout convention
signal_processing/  filtering, windowing, feature extraction (shared by both ML tracks)
ml/
  classical/          Random Forest + SelectKBest track
  deep_learning/      1D-CNN / CNN-LSTM track (PyTorch)
  common/             dataset loading + dog-level train/test split
dashboard/           Streamlit owner/vet report (gait trend, symmetry index, alerts)
docs/                PRD
```

## Status

No real canine gait data has been collected yet (requires vertebrate-animal
approval — see `docs/PRD.md` section 10). Everything here currently runs
end-to-end on a **synthetic** dataset so the pipeline can be validated before
real data exists. Swap `data/synthetic` for `data/real` once collected — the
CSV schema and folder layout are identical (`data/README.md`).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate a synthetic dataset (24 dogs x 3 sessions x 3 classes)
python data/synthetic_gait_generator.py

# 2. Train the classical ML track (Random Forest)
python ml/classical/train_rf.py

# 3. Train the deep learning track (1D-CNN)
python ml/deep_learning/train_cnn.py

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

Both training scripts hold out entire dogs from the test set (not just
windows), so reported accuracy reflects generalization to unseen
individuals — the honest evaluation protocol used in the cited literature
(`docs/PRD.md` section 3).

## Firmware

See [firmware/README.md](firmware/README.md) for wiring, required Arduino
libraries, and the ESP32 + MPU6050 sketch.

## Next steps toward the real system

1. Secure vertebrate-animal approval (Four Rs) before any real dog data collection.
2. Partner with a vet clinic/shelter for baseline + clinically diagnosed sessions.
3. Build the physical harness mount (<=2-3% of body weight) and flash the firmware.
4. Replace `data/synthetic` with `data/real` and re-run both training scripts.
5. Validate model outputs against veterinarian assessment (sensitivity/specificity).
6. Add per-dog calibration (FR-9) and, if pursuing lameness localization, per-limb sensor nodes.
