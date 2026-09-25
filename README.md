# PawPath

A non-invasive IMU-based system for early detection of canine spinal injury, lameness and arthritis.

Full requirements and literature review: [docs/PRD.md](docs/PRD.md).

## Repo layout

```
firmware/           ESP32 + MPU6050 sensor node (BLE streaming + SD logging)
data/                synthetic data generator + real-data layout convention
signal_processing/  filtering, windowing, feature extraction (shared by both ML tracks)
ml/
  classical/          Random Forest + SelectKBest track (this is what the deployed app uses)
  deep_learning/      1D-CNN / CNN-LSTM track (PyTorch) — local-only, not deployed
  common/             dataset loading + dog-level train/test split
api/                 FastAPI + server-rendered HTML dashboard (owner/vet report)
  index.py             app + routes
  templates/           Jinja2 page
  samples/             3 small bundled demo sessions (one per class)
  requirements.txt     deployment-only deps (no torch — see below)
vercel.json          Vercel deployment config for api/index.py
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

# 3. Train the deep learning track (1D-CNN) — optional, local research only
python ml/deep_learning/train_cnn.py

# 4. Launch the dashboard
uvicorn api.index:app --reload
```

Open http://localhost:8000 — pick one of the bundled demo sessions or upload
a session CSV.

Both training scripts hold out entire dogs from the test set (not just
windows), so reported accuracy reflects generalization to unseen
individuals — the honest evaluation protocol used in the cited literature
(`docs/PRD.md` section 3).

## Dashboard architecture & deployment

The dashboard was originally a Streamlit app; it's now a small FastAPI app
(`api/index.py`) rendering a single server-side Jinja2 page with Chart.js for
the gait traces, because Streamlit's dependency footprint doesn't fit on
Vercel and pulls in far more than the dashboard needs.

**PyTorch is intentionally excluded from the deployed app.** The deployed
classifier is the classical Random Forest track (`ml/classical/rf_model.joblib`,
~1.2MB, committed to the repo) — inference only needs scikit-learn + joblib.
The 1D-CNN track stays local/research-only; nothing under `ml/deep_learning/`
is imported by `api/index.py`. This is enforced by two separate requirements
files:
- `requirements.txt` (repo root) — full dev environment: data generation,
  both ML tracks, and running the dashboard locally. Includes `torch`.
- `api/requirements.txt` — deployment-only. No `torch`, no `streamlit`.
  Vercel's Python builder prefers a `requirements.txt` colocated with the
  function over the root one, so `api/index.py` only pulls this lean set in
  production.

**Deploying to Vercel:**
```bash
vercel deploy   # or connect the GitHub repo in the Vercel dashboard
```
`vercel.json` points Vercel at `api/index.py` and explicitly includes
`signal_processing/**` and `ml/classical/rf_model.joblib` in the function
bundle (they live outside `api/`, so they aren't picked up automatically).

**Known limitation — session history persistence.** The per-dog longitudinal
trend log (FR-6) is written to a local CSV. That works when running
`uvicorn` locally or on a host with a persistent disk, but Vercel's
serverless filesystem is read-only outside `/tmp`, and `/tmp` isn't shared
across invocations or instances — so on Vercel, trend history won't
reliably persist between requests. A real deployment would need an external
store (e.g. a small Postgres/SQLite-on-a-volume/Vercel KV) for FR-6; that's
out of scope for the current "small FastAPI app" and noted here so it isn't
a silent surprise.

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
