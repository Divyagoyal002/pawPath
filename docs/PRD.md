# PawPath — Project Requirement Document

A Non-Invasive IMU-Based System for Early Detection of Canine Spinal Injury, Lameness and Arthritis

Owner: Anshul Bhatt
Category (IRIS): Biomedical Engineering / Animal Sciences
Prepared for: IRIS National Fair 2026-27
Version: 1.0

See repository README.md for the current implementation status of this PRD.

## 1. Introduction & Background

Canine musculoskeletal and neurological disorders — including spinal injuries, lameness, and osteoarthritis — are among the most common welfare concerns in companion animals, yet current clinical assessment remains largely subjective. Veterinarians typically rely on visual gait observation, which is prone to inter-observer variability and cannot reliably detect subtle, early-stage abnormalities before they progress into chronic or debilitating conditions.

Wearable Inertial Measurement Units (IMUs) — sensors combining a 3-axis accelerometer and 3-axis gyroscope — have emerged in veterinary and human movement science as a low-cost, objective alternative to force-plate and optical motion-capture systems. Because IMUs are compact, inexpensive, and capable of capturing high-frequency 3D motion data outside clinical settings, they are well suited to free-roaming canine gait monitoring.

PawPath proposes a wearable, affordable IMU-based system that continuously or periodically captures a dog's gait dynamics and applies signal processing and machine learning to flag early indicators of spinal injury, lameness, and arthritis — enabling earlier veterinary intervention and better long-term outcomes.

## 2. Problem Statement & Objectives

**Problem:** There is currently no widely accessible, affordable, and objective tool for continuous, at-home or in-clinic monitoring of canine gait health. Existing veterinary-grade motion capture systems (optical cameras, force-plate walkways) are expensive, require a controlled environment, and are impractical for longitudinal, everyday monitoring.

### 2.1 Objectives

- Design and prototype a low-cost, non-invasive wearable IMU harness/collar system for dogs of varying breeds and sizes.
- Collect and process 3D motion data (acceleration, angular velocity) during walking and trotting gaits.
- Develop a signal-processing and machine-learning pipeline to classify gait as healthy vs. abnormal, and where possible, to differentiate between orthopedic (lameness/arthritis) and neurological (spinal) origins.
- Provide longitudinal tracking so gradual degeneration (e.g., early arthritis) can be flagged before it becomes clinically obvious.
- Validate system outputs against veterinarian assessment and, where feasible, a reference gait-measurement method.

## 3. Literature Review & Related Work

- Canine gait analysis using inertial sensors and deep learning for orthopedic and neurological disorders — Scientific Reports — https://www.nature.com/articles/s41598-026-40717-x
- Canine Clinical Gait Analysis for Orthopedic and Neurological Disorders: An Inertial Deep-Learning Approach — arXiv — https://arxiv.org/pdf/2507.05671
- Convolutional neural network for early detection of lameness and irregularity in horses using an IMU sensor — arXiv — https://arxiv.org/abs/2503.13578
- Serial kinematic analysis using inertial measurement units in growing dogs at risk of hip dysplasia — ScienceDirect — https://www.sciencedirect.com/science/article/pii/S2451943X24000528
- Four-limb wireless IMU sensor system for automatic gait detection in canines — PMC — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8938443/
- Gaitkeeper: A system for measuring canine gait — Newcastle University ePrints — https://apps-eprint4-l.ncl.ac.uk/236880
- Machine learning based canine posture estimation using inertial data — PLOS ONE (via IDEAS/RePEc) — https://ideas.repec.org/a/plo/pone00/0286311.html
- Extraction of canine gait characteristics using a mobile gait analysis system based on IMUs — FHNW — https://irf.fhnw.ch/bitstreams/f568b2d9-791d-45f3-9a17-5ccbe65ed870/download
- IEEE DataPort — Gait IMU keyword dataset listings (for transfer-learning pretraining) — https://ieee-dataport.org/keywords/gait-imu

Key takeaways used to drive design decisions in this repo:
- Sub-$100 IMU hardware achieves clinically meaningful accuracy.
- Back/chest sensor placement outperforms neck placement for classification; per-limb placement enables lameness localization.
- A 120-sample window with stride 5 is a validated segmentation configuration for canine IMU gait data.
- Both classical ML (Random Forest, feature-engineered) and deep learning (1D-CNN, CNN-LSTM) are viable; deep learning generalizes better to unseen individuals.

## 4. Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Capture 3-axis accelerometer + 3-axis gyroscope data at >=100 Hz. | Must |
| FR-2 | Support at least one IMU on the back/harness, with optional per-limb sensors. | Must |
| FR-3 | Segment continuous motion data into gait cycles/strides via peak-detection or windowing. | Must |
| FR-4 | Classify gait segments healthy/abnormal, and orthopedic vs. neurological where possible. | Must |
| FR-5 | Compute limb-symmetry / asymmetry indices (step-time, stance-time, swing-time). | Should |
| FR-6 | Store per-dog longitudinal records for trend analysis. | Must |
| FR-7 | Transmit data wirelessly (BLE/Wi-Fi) to a companion app/dashboard. | Should |
| FR-8 | Generate a risk report/alert when abnormal patterns are detected. | Should |
| FR-9 | Allow per-dog calibration/baseline capture. | Could |

## 5. Non-Functional Requirements

- Affordability: BOM under ~$50-60 per unit.
- Non-invasiveness: mount <=2-3% of body weight, no gait disruption.
- Battery life: 6-8 hrs continuous logging, or days of periodic sampling.
- Data privacy: gait/health data stored securely, shared only with owner/vet.
- Robustness: tolerate outdoor use (dust/moisture), varied terrain.

## 6. System Architecture

1. **Sensor layer** — MPU6050/BMX055/MPU9250 + ESP32, 3D-printed harness housing.
2. **Edge preprocessing** — low/high-pass filtering, gravity-bias removal, 120-sample/stride-5 windowing.
3. **Feature extraction & modeling** — classical ML (statistical/spectral features + SelectKBest + Random Forest/Isolation Forest) and deep learning (1D-CNN / CNN-LSTM on raw windows).
4. **Application & reporting layer** — dashboard with gait trend graphs, symmetry scores, flagged anomalies, vet-export.

## 7. Data Requirements

No public canine IMU gait dataset is available for direct reuse. Plan:
- Partner with a vet clinic/shelter for baseline + clinically diagnosed abnormal gait data.
- Back/chest placement for general classification; per-limb for lameness localization.
- Record synchronized video during calibration sessions for ground-truth labeling.
- Use synthetic data (see `data/synthetic_gait_generator.py`) to prototype the pipeline before real data exists.
- **Compliance:** vertebrate-animal approval required before data collection (Four Rs: replace, reduce, refine, respect); confirm no harm results. This is straightforward since the sensor is external/non-invasive, but paperwork must be secured first.

## 8. Tools, Hardware & Software Resources

**Hardware:** MPU6050/MPU9250/BMX055, ESP32 or Arduino Nano 33 BLE Sense, 3D-printed housing/harness, LiPo battery (500-1000mAh), smartphone slow-mo camera for validation.

**Software:** Python (NumPy, SciPy, Pandas), scikit-learn, PyTorch, Arduino IDE/PlatformIO, Matplotlib/Plotly, Streamlit.

## 9. Methodology & Implementation Plan

1. Literature review & hardware selection (2-3 wks)
2. Prototype build (3-4 wks)
3. Ethics/approval & pilot data collection (2-4 wks)
4. Signal processing & model development (3-4 wks)
5. Validation vs. vet assessment (2 wks)
6. Dashboard & reporting, final documentation (2 wks)

## 10. Risks, Constraints & IRIS Compliance Notes

- Vertebrate-animal approval must be secured before any experimentation.
- Small sample sizes risk overfitting — mitigate with augmentation, cross-validation, transfer learning from equine data.
- Breed/size variability — per-dog calibration (FR-9) recommended over a universal threshold.
- Anonymity rule: no school, city, or state disclosed in video/paper/abstract.
- Category ambiguity: confirm Biomedical Engineering vs. Animal Sciences framing early.
