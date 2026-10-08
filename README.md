# Uncertainty-Aware Hip Exoskeleton

Applying the uncertainty-aware control framework of **Tourk et al., _Uncertainty-Aware Ankle
Exoskeleton Control_** ([arXiv:2508.21221](https://arxiv.org/abs/2508.21221)) to a **hip**
exoskeleton, using the dataset from **Molinaro, Kang & Young, _Estimating human joint moments
unifies exoskeleton control and reduces user effort_**, *Science Robotics* 9, eadi8852 (2024).

The ankle paper's premise: a learned exoskeleton controller will happily make a confident,
wrong prediction on a movement it was never trained on, and actuating at the wrong moment is a
safety problem. Their answer is an uncertainty estimator that classifies each window of sensor
data as in-distribution or out-of-distribution and disengages assistance when it is unfamiliar.
This project reproduces that on hip data, then explores whether better input features improve
the uncertainty signal.

---

## Data

**The raw sensor data is not in this repo, and is not the file most people land on first.**
Three different artifacts get confused:

| What | Where | Contents |
|---|---|---|
| **Raw sensor data — what we use** | [Zenodo 10849318](https://zenodo.org/records/10849318) · [Dryad](https://doi.org/10.5061/dryad.8kprr4xsv) | 200 Hz time series, 34 subjects, ~2.4 GB zipped |
| Supplementary figure data | `data/science-robotics/` (in repo) | MATLAB `table` objects behind the paper's figures. No time series. `scipy.io` cannot decode them |
| Nature 2024 dataset — candidate for the moment estimator | [10.35090/gatech/75759](https://doi.org/10.35090/gatech/75759) · `data/nature/` (Complete variant, gitignored, 88 GB) | Belongs to *Task-agnostic exoskeleton control* (s41586-024-08157-7) — a **different paper**. 22 people, 26 tasks, net and biological joint moments; see below |

**Dataset shape:** 34 participants (`AB01`–`AB34`), 5,364 trials, 8.66 M samples, 12.0 hours,
6.1 GB extracted. Each trial is `AB##/<trial>/{angle,exo,gp,grf,moment}.csv`, all sharing an
identical `time` column on a clean 200 Hz grid.

**Collection phases** (hardware differs — this matters):

| Phase | Subjects | Exoskeleton | Notes |
|---|---|---|---|
| 1 | AB01–AB09 | Samsung GEMS | IMU data *transformed* into the custom exo's frames |
| 2 | AB10–AB14 | Custom hip exo | `trq_mea_*` all-NaN (controller software bug) |
| 3 | AB15–AB24 | Custom hip exo | Treadmill LG/RA/RD + overground stairs |
| 4 | AB25–AB34 | Custom hip exo | As phase 3, plus some standing and transitions |

Download with `scripts/download_molinaro.sh` (resumable, parallel, verifies archives — Zenodo
throttles aggressive clients).

**Nature 2024 dataset.** `Phase1And2_Complete.zip` and `Phase3_Complete.zip` from the Georgia Tech repository,
unzipped to `data/nature/phase1-2` and `data/nature/phase3` (verified against the zips' own file index: every
file, size and CRC32). [`notebooks/nature/data_exploration/01_dataset.ipynb`](notebooks/nature/data_exploration/01_dataset.ipynb)
asks whether it can replace the dataset above as the repository for a hip joint-moment estimator and forecaster
(hip encoder, thigh IMU, task label, hip moment). Short answer: yes, with a one-time build — recompute encoder
velocity (the published channel is half scale for most sessions), remap IMU axes (pipeline `(x, y, z)` = Nature
`(y, z, x)`), recover stair and incline direction from joint power, and mask each leg by its own activity flag.
BT04 was never published; BT01, BT02 and BT13 appear in both phases (split by person).

---

## OBJ 1 — Recreate the paper with vanilla methods and vanilla features

Faithful reproduction: the paper's 16 raw channels, its window and stride, its four
architectures, its scoring and thresholding. **No feature engineering.** The point is a
defensible baseline before changing anything.

### 1.1 Data acquisition — done

`scripts/download_molinaro.sh`.

### 1.2 Exploration — done

[`notebooks/sci_rob/data_exploration/01_load_molinaro.ipynb`](notebooks/sci_rob/data_exploration/01_load_molinaro.ipynb)

- Trial inventory: all 5,364 trials parsed into structured metadata (mode, slope/stair height,
  speed, controller, phase, hardware), cached to Parquet.
- `load_trial()` joins the five per-trial CSVs, asserting time alignment.
- **NaN audit.** Onboard exo channels are 0.0–0.4% NaN; ground-truth labels are 44–82% NaN,
  because inverse dynamics was only valid on the force plates. `UC` trials have roughly twice
  the label coverage of `BT`/`ET`.
- **Sensor axis conventions** — the dataset README never defines these, so they were derived by
  correlating each gyro axis against `enc_velo`:
  - `thigh_gyro_y_{l,r}` is the **sagittal** axis (r ≈ −0.90 with hip velocity). Negative sign
    means gyro `y` is positive for *flexion*, opposite the encoder convention.
  - **The IMUs are not mirrored** — same sign on both legs, so no per-side flip is needed. This
    was the detail most likely to silently corrupt training.
  - Convention holds for all 34 subjects; GEMS subjects are measurably noisier (mean |r| 0.75
    vs 0.85) but unambiguously the same axis.
  - **`thigh_gyro_z_r` is unstable across subjects** (r swings −0.78 to +0.63, sign flips on
    5 subjects) — likely IMU mounting rotation varying by session. A suspect if
    held-out-subject generalization underperforms.

### 1.3 Preprocessing — done, pending OOD revision

[`notebooks/sci_rob/data_exploration/02_build_windows.ipynb`](notebooks/sci_rob/data_exploration/02_build_windows.ipynb)

The 16 inputs, in the paper's per-side order (`accel xyz`, `gyro xyz`, angle, velocity):

```
thigh_accel_{x,y,z}_{l,r}   thigh_gyro_{x,y,z}_{l,r}   enc_angle_{l,r}   enc_velo_{l,r}
```

Every one has a direct ankle counterpart; `enc_velo` is already finite-differenced with a
causal 10 Hz Butterworth, matching how the ankle paper derived ankle velocity.

- Windows: 200 samples (1.0 s) at stride 10, channels-first `(N, 16, 200)`, memory-mapped.
- Target: **`sin(2·π·gait_phase)`**, dual output (left/right). The paper predicts the *sine*
  specifically to avoid the discontinuity where phase wraps 100% → 0% at heel strike.
- **Per-leg mask.** Ground truth is per leg; a window can have a real left label and nothing on
  the right. Those are kept with `mask = 0` and `y = 0` as a placeholder — **the mask must be
  applied in the loss**, or the model trains toward a fabricated mid-stride target.
- Splits are **disjoint by subject** (12 / 4 / 4), mirroring the paper's 9 / 3 / 3.
- `StandardScaler` fit on the training split only.

Pending revision: drop the label filter (the AE and GAN need no labels, and OOD windows have
none) and add the OOD splits below.

### 1.4 Split design — planned

Training is ID-only; validation and test subjects contribute both ID and OOD tasks, as in the
paper.

| Split | Modes | Subjects | Windows |
|---|---|---|---|
| train ID | LG, RA, RD | 12 | ~188,000 |
| val ID / OOD | LG,RA,RD / SA,SD,ST,TR | 4 | ~64,400 / ~23,300 |
| test ID / OOD | LG,RA,RD / SA,SD,ST,TR | 4 | ~62,000 / ~23,000 |

Phase 1 (GEMS) subjects are excluded — different hardware with reconstructed IMU frames would
widen the in-distribution manifold for a reason unrelated to human movement.

### 1.5 Models — planned

Four architectures, reproducing Table I of the ankle paper. Architectures in `ml/common/models/`,
training and scoring in `ml/sci_rob/vanilla/`; one notebook each in `notebooks/sci_rob/model_exploration/`.

| Model | Uncertainty score Ψ | Labels? | Their F1 |
|---|---|---|---|
| Ensemble (gait phase) | `(Var(l₁..l₇) + Var(r₁..r₇)) / 2` | yes | **90.3** |
| Ensemble (synthetic target) | branch variance, target = `Σ_{i>j} corr(xᵢ,xⱼ)` | no | 62.5 |
| Autoencoder | `−LOF_k(z)` on the latent space | no | 55.7 |
| GAN | `1 − D(x)` | no | 71.5 |

Two things easy to get wrong: the autoencoder's score is **not** reconstruction error (the
paper tried it — stationary data reconstructs too well) but Local Outlier Factor on the latent
space; and the ensemble is two-stage — LOSO to fix the epoch count, then retrain on all
subjects.

Scoring pipeline: Ψ per window → causal median filter (~0.5 s) → threshold at the **99.5th
percentile of training scores** → ID/OOD label.

### 1.6 Results — placeholder

Metrics: accuracy, precision, recall, F1, J-statistic, AUROC, ECE, Brier.

Our test set is **~27% OOD** where the paper's was **80.1%**. Precision and F1 are therefore
not directly comparable to their numbers — **lead with J-statistic and AUROC**, which is
precisely why the paper adopted J.

### 1.7 Known deviations from the paper

| | Paper | Here | Why |
|---|---|---|---|
| Sample rate | 175 Hz | 200 Hz | native rate of this dataset |
| Window | 175 samples | 200 samples | preserves the 1-second receptive field |
| Joint / IMU site | ankle, shank IMU | hip, thigh IMU | the point of the project |
| OOD tasks | sitting, jumping, lying down | held-out modes (stairs, standing, transitions) | this dataset is all locomotion |
| Dilations | unstated | (1, 2, 8) | so the receptive field covers the full window |
| LOF `k` | unstated | 20 (default) | — |
| Score filter | Table IV says SMA; eval text says causal median | causal median | following the eval procedure; the paper contradicts itself |
| Actuation | actuated **and** unactuated | `UC` only | the raw dataset has no zero-torque condition |

The last row matters for interpretation: the in-distribution set is specifically *walking while
the exoskeleton actively assists under moment-based control*, a narrower notion of "normal"
than the ankle paper's.

---

## OBJ 2 — Feature extraction

Exploratory. OBJ 1 establishes whether raw channels are enough; this asks whether better
features do better, particularly on OOD tasks that sit close to the training distribution —
where the ankle paper's estimator was weakest.

**Lead idea: Kalman filter NIS between encoder and IMU, as an added input channel.**
Fuse the hip encoder (`enc_angle`, `enc_velo`) with the thigh/pelvis IMU under an assumed
kinematic model and feed the **Normalized Innovation Squared**

```
NIS = νᵀ S⁻¹ ν        ν = measurement innovation,  S = innovation covariance
```

alongside the raw channels. Under a correctly specified model NIS is chi-squared distributed
with known degrees of freedom, so it is a *calibrated, physically grounded* novelty statistic:
it spikes exactly when the two sensing modalities stop agreeing with the model — which is what
an unfamiliar movement, or a sensor fault, looks like. It hands the network a feature that
already encodes "this does not match the model," instead of asking it to infer that from raw
signals.

Other directions, open:

- Spectral / time-frequency features over the window (human movement is mostly below ~10 Hz).
- Residuals from a learned forward-dynamics model.
- Per-stride statistics that do **not** presuppose a well-defined gait phase — relevant because
  gait phase is undefined for non-cyclic tasks, a limitation the ankle paper flags as the reason
  its best model can only ever have cyclic actions in its training set.

**Implemented: four engineered feature families** (`ml/common/features.py`, all causal):
jerk and high-frequency content, polar phase space (hip angle against velocity), spectral regularity,
and thigh pitch (gravity fused with the sagittal gyro). Evaluated on the Nature dataset in
[`notebooks/nature/data_exploration/02_features.ipynb`](notebooks/nature/data_exploration/02_features.ipynb):

- The polar phase tracks true gait phase to about a tenth of a stride with one fixed offset, so it is
  a continuous gait phase for the controller without heel-strike detection.
- Stride and step regularity are the best single detectors of cyclic tasks (AUROC 0.89, 0.88), and the
  only features that add to the raw data for task and controller classification. Most of that gain comes
  from the 3 s of history they summarize; raw summaries over 3 s reach nearly the same.
- The rhythm features (stride and step regularity, dominant frequency) report 0 when the window has no peak
  in their search band, instead of the band-edge value. Their 0.5–1.6 s stride search misses slower strides
  (about 1.7–1.8 s), so some slow walking and stair windows read as no rhythm; widening it needs a longer window.
- Jerk and angular acceleration largely re-express raw signal variability. No family materially improves
  the hip-moment probe (R² 0.725 to 0.734).
- In PCA and UMAP space ([`03_feature_embeddings.ipynb`](notebooks/nature/data_exploration/03_feature_embeddings.ipynb))
  the ranking holds: the spectral family is the only one that makes task families cluster more cleanly
  across people, and controller classes form no clusters in any space.

---

## Setup

Machine: Apple M5, 24 GB. The conda base is **x86_64** (Rosetta, no GPU), so training uses a
native **arm64** venv with MPS — measured ~7.3× faster on this workload.

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m venv .venv
.venv/bin/pip install torch numpy pandas pyarrow matplotlib scikit-learn ipykernel tqdm
.venv/bin/python -m ipykernel install --user --name hip-exo --display-name "Python (hip-exo)"
```

Then, in order: download the data, run `01_load_molinaro.ipynb`, run `02_build_windows.ipynb`
to rebuild `data/processed/`.

---

## Layout

Code and notebooks are split by dataset. Anything that does not care which dataset it reads (model
architectures, feature extractors, signal helpers) lives in `ml/common/`; anything that loads, labels,
splits or trains on one dataset lives under that dataset's folder and imports from `ml/common/`. New work on
the Nature data, such as retraining the awareness pipeline, goes in `ml/nature/` (e.g. `ml/nature/awareness/`).

```
data/
  science-robotics/     supplementary figure data (MATLAB tables) — in git
  molinaro-raw/         34 subject folders of raw 200 Hz CSVs — gitignored, 6.1 GB
  nature/               Nature 2024 dataset, Complete variant (phase1-2/, phase3/) — gitignored, 88 GB
  cache/                trial inventory, Nature Parquet caches — gitignored
  processed/            windowed model-ready arrays + scaler + config — gitignored
ml/
  common/               dataset-agnostic
    models/             ConvAutoencoder, ensembles, GAN, ConvVAE
    features.py         engineered features: jerk, polar phase, spectral, thigh pitch (all causal)
    signals.py          bouts, lagged correlation, stride period, IMU frame solver
    test_*.py           one runnable check per module
  sci_rob/              Science Robotics 2024 (Molinaro) dataset
    vanilla/            paper reproduction: data, targets, training, evaluation, CLI (main.py)
    prob_ensemble/      probabilistic ensemble (aleatoric and epistemic split)
    LOTO/               leave-one-task-out holdouts
    VAE/                VAE trainer and beta sweep (model in common/models/vae.py)
    awareness/          awareness pipeline: regime holdout, fault injection
  nature/               Nature 2024 dataset
    nature_data.py      data access and Parquet caches in data/cache/nature/ (test_nature_data.py)
notebooks/
  sci_rob/
    data_exploration/   01 load, 02 build windows, 03 feature plots, 04 NIS, 05 synthetic targets
    model_exploration/  01–10: one notebook per model, then LOTO, UMAP and the awareness holdout
  nature/
    data_exploration/   01 dataset audit, 02 engineered features, 03 feature embeddings (PCA, UMAP)
scripts/
  download_molinaro.sh  resumable parallel download from Zenodo
```

**Cluster checkout.** The cluster copy is rsynced, not cloned, and predates this layout. Move the old
folders once so existing `runs/` stay beside their code, then push as usual:

```bash
cd ~/Uncertainty-Aware-Hip-Exoskeleton && mkdir -p ml/sci_rob && for d in vanilla LOTO VAE awareness prob_ensemble; do [ -d ml/$d ] && mv ml/$d ml/sci_rob/; done
```

## References

- Tourk, Galoaa, Shajan, Young, Everett & Shepherd. *Uncertainty-Aware Ankle Exoskeleton
  Control.* [arXiv:2508.21221](https://arxiv.org/abs/2508.21221)
- Molinaro, Kang & Young. *Estimating human joint moments unifies exoskeleton control and
  reduces user effort.* [Sci. Robot. 9, eadi8852](https://www.science.org/doi/10.1126/scirobotics.adi8852) (2024)
- Molinaro, Scherpereel et al. *Task-agnostic exoskeleton control via biological joint moment
  estimation.* [Nature 635](https://www.nature.com/articles/s41586-024-08157-7) (2024)
- Scherpereel, Molinaro, Inan, Shepherd & Young. *A human lower-limb biomechanics and wearable
  sensors dataset during cyclic and non-cyclic activities.*
  [Sci. Data](https://www.nature.com/articles/s41597-023-02840-6) (2023) — candidate source of
  genuine non-cyclic OOD data
