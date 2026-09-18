# Data

## `synthetic/` (generated, gitignored)

No public canine IMU gait dataset exists for direct reuse (see `docs/PRD.md`
section 7). Generate a placeholder dataset to develop and test the pipeline:

```bash
python data/synthetic_gait_generator.py
```

This writes CSVs to `data/synthetic/<healthy|orthopedic|neurological>/<dog_id>_<session>.csv`,
matching the firmware log schema plus a class folder acting as the label.
It is **not** real gait data — swap it out once real sessions are collected.

## `real/` (not yet collected)

Planned layout once a veterinary clinic/shelter partner is onboarded and
vertebrate-animal approval is secured (`docs/PRD.md` section 7 & 10):

```
data/real/<healthy|orthopedic|neurological>/<dog_id>_<session>.csv
```

Same schema as synthetic data so `ml/common/dataset.py` works unchanged —
just point `--data-dir` at `data/real` instead of `data/synthetic`.

Collection protocol:
- Back/chest IMU placement for general classification; add per-limb sensors
  if pursuing lameness localization (literature: back/chest outperforms neck).
- Record synchronized video during calibration sessions for ground-truth labeling.
- Label by clinically diagnosed condition, confirmed by the partner vet.
