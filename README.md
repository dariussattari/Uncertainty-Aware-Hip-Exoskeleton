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

**Controller labels from the deployed estimator's error** (`ml/nature/estimator_labels.py`,
[`04_estimator_labels.ipynb`](notebooks/nature/data_exploration/04_estimator_labels.ipynb)): the Nature study's own
estimator (`torque_estimated`) scored against the biological hip moment on the seven Phase 3 participants it never
trained on, with a rule fixed before scoring: an unfamiliar task is OOD-safe when its RMSE is within the worst familiar
task's (stairs, 0.190 Nm/kg), OOD-dangerous above it.

- Only the ball toss is clearly above that envelope (0.279 Nm/kg, 95% interval 0.252 to 0.309). The rule also labels
  step-ups, cutting and lunges dangerous and moves sit-to-stand and tug of war to safe, but eight of the 22 unfamiliar
  tasks, all four changes among them, sit within the bootstrap uncertainty of the boundary.
- Absolute error mostly tracks the moment's size (Spearman 0.81 across tasks); relative to the moment, every
  unfamiliar task (0.25 to 0.56) is within the familiar range (0.31 to 0.64), and the estimator beats switching off
  everywhere. Push has a low RMSE but the second-largest share of wrong-direction assistance (12%).
- The provisional labels (`nature_data.OOD_DANGEROUS_TASKS`) stay in use until the same rule is applied to this
  project's hip-and-thigh estimator; this deployed estimator had more sensors, so its labels are the optimistic bound.

**Nature awareness autoencoders** (`ml/nature/awareness/`, viewed in
[`onboard/01_ae_latent_spaces.ipynb`](notebooks/nature/model_exploration/onboard/01_ae_latent_spaces.ipynb), with standalone
3-D pages in `onboard/interactive/`). The ankle paper's conv
autoencoder (`ml/common/models/autoencoder.py`) trained on the Nature data's ID-safe tasks (normal, incline and weighted
walking, stairs) of 14 participants, 4 for validation and 4 held out, three seeds each, on two input settings: 1 s of
the 16 raw channels, and 3 s of raw channels plus the seven spectral features as constant channels (the setting
nature notebooks 02 and 03 found best; the second pooling stage is 15 so the latent stays 4 x 8 = 32 values). The
input ablation further down adds the other two corners, 1 s + spectral and 3 s raw.

| model | inputs | validation MSE | scale-free L1 (1 = dense) | novelty units (of ~32) |
|---|---|---|---|---|
| vanilla | 1 s raw | 0.247 ± 0.005 | 0.90 | 0–1 |
| sparse, λ 0.3 | 1 s raw | 0.313 ± 0.007 | 0.25 | 2 ± 1 |
| vanilla | 3 s raw + spectral | 0.269 ± 0.002 | 0.84 | 0 |
| sparse, λ 0.3 | 3 s raw + spectral | 0.306 ± 0.014 | 0.10 | 18 ± 7 |
| sparse, λ 0.1 | 3 s raw + spectral | 0.285 ± 0.003 | 0.27 | 4 ± 5 |
| vanilla, 8 x 4 latent | 3 s raw + spectral | 0.353 ± 0.008 | 0.61 | 1 ± 1 |
| sparse, λ 0.1, 8 x 4 latent | 3 s raw + spectral | 0.358 ± 0.006 | 0.14 | 17 ± 8 |

A novelty unit responds at least five times more strongly to some unseen task than to any training task. MSE compares
only within an input setting.

- The sparse variant adds λ x mean(z / RMS_batch(z)) per latent unit. A plain L1 term does not work in this model: the
  decoder's BatchNorm makes reconstruction blind to the latent's scale, so plain L1 shrinks every unit without
  sparsifying any. Validation computes the penalty from moments over the whole set (per batch it depends on window
  order), recomputes BatchNorm statistics over 50 training batches first (running averages made the sparse models'
  validation error jump by up to 0.09), and λ ramps up over 5 epochs (without it one seed silenced 24 of 32 units).
- On 3 s + spectral inputs gait rings collapse into one walking cluster, unseen movement types sit apart, stair and ramp
  direction begin to separate, and the sparse code holds on new people; on 1 s inputs it does not (sparsity 0.15 on
  training people, 0.53 on new ones by epoch 15).
- The λ 0.3 sparse latent's strongest units fire as one block (|r| 1.00 between the top two). A lower penalty
  (λ 0.1) made it more redundant (3.1 against 4.5 effective directions on held-out people) and lost most novelty units;
  reshaping the latent to 8 filters x 4 steps (`--latent 8x4`) tied (4.7) at a higher reconstruction cost. The
  duplication is units firing together, which a per-unit penalty cannot see.
- Novelty detection on the four test participants
  ([`onboard/05_Novelty_Metric_comparison.ipynb`](notebooks/nature/model_exploration/onboard/05_Novelty_Metric_comparison.ipynb), which scores
  every run under `runs/` and every score in `onboard/_novelty_scores.py`): LOF on the vanilla latents ranks unfamiliar
  above familiar windows best (AUROC 0.893 with 3 s + spectral inputs, 0.887 with 1 s; OOD-dangerous tasks 0.968 with
  1 s). The sparse latents score 0.77 to 0.85. At the ankle paper's threshold only 29 to 36% of unfamiliar windows are
  flagged, and quiet tasks (standing poses, the ball toss) are rarely flagged by any latent.
- Operating point on held-out people (notebook 05, section 6; `ml/nature/awareness/encode_split.py` encodes the
  validation participants from each checkpoint, and `--check` reproduces the cluster export to 0.5% relative RMS): a
  1% false-alarm target set on the four validation participants gives 1.7 to 2.0% on the four test participants, so
  the familiar score level is wearer-specific. LOF on 16 principal components of the vanilla 1 s latent flags 34% of
  unfamiliar and 53% of OOD-dangerous test windows at 1.7% false alarms (29% and 45% under the ankle rule; three
  seeds, within their ±6 to ±10 point spread); fewer dimensions do not help the other latents. The false-alarm target
  is the lever: 2% buys 44% and 63%, 5% buys 61% and 81%.
- Gaussian-NLL control (`train_ae.py --variant gaussian`: the encoder ends at its last convolution, so the latent is
  linear, and the objective is the unit-variance Gaussian NLL, under Adam the MSE times a constant): the Science Robotics
  VAE study's best model. On 1 s inputs it matches the plain AE (test AUROC 0.887, OOD-dangerous 0.966) with far less
  spread across seeds; on 3 s + spectral inputs it was the best unfamiliar-movement detector before the input ablation
  (AUROC 0.911 against 0.893; 44% of unfamiliar windows flagged at 1.9% false alarms against 35%), confirmed on the
  validation participants.
  It is the baseline the VAE grid has to beat. Its latent maps are in notebook 01, section 6: the linear latent spreads
  windows along about twice as many directions as the ReLU latent (11.5 against 6.6 on 1 s inputs).
- Run: `python ml/nature/awareness/ae_data.py` (arrays and window tables, about a minute), `bash
  ml/nature/awareness/slurm/push.sh`, then on a login node `bash ml/nature/awareness/slurm/submit.sh [--inputs
  raw_3s_spectral] [--args "--sparsity-weight 0.1 --latent 8x4"]` (a six-task array, one GPU and one core per task, 2 to 8 min each), then `bash
  ml/nature/awareness/slurm/pull.sh` into `notebooks/nature/model_exploration/runs/ae/`. Training uses bf16 autocast on
  CUDA: in fp32, cuDNN runs these convolutions as FFTs, 551 ms per step on an A100 against 23 ms.
- VAE grid (`ml/nature/awareness/train_vae.py`; latent spaces in
  [`onboard/02_vae_latent_spaces.ipynb`](notebooks/nature/model_exploration/onboard/02_vae_latent_spaces.ipynb), scored in
  notebook 05): β in {1, 8, 32, 64, 128} x decoder variance {fixed, learned per channel} x seeds 0-2,
  30 cells on raw 1 s input, trained with the AE's settings (about 3 min each on an A100). β prunes the code by whole
  latent filters: with the fixed variance 32 dimensions at β 1 (KL 60 per window, reconstruction equal to the plain AE),
  19 at β 8 and 8 (one filter over its 8 steps) from β 32. The learned variance needs about four times the β for the
  same pruning, because it reconstructs the hip angle to 0.09 and the sagittal gyro to 0.18 of their SDs against 0.7 to
  0.8 for the out-of-plane axes. Posterior width is nearly flat across tasks in every cell. Run: `push.sh`, then on a
  login node `submit.sh --vae --array 0` (check `jobstats`), `submit.sh --vae --array 1-29`; then `pull.sh`, which
  brings `runs/vae` beside `runs/ae`.
- Every native metric, scored (notebook 05, section 7; `NATIVE` in `onboard/_novelty_scores.py`): LOF (raw, standardized,
  active dimensions), Mahalanobis, out-of-range units, reconstruction error, and for the VAE its KL, decoder NLL,
  negative ELBO and posterior variance; each oriented and thresholded on the validation participants. The best single
  1 s detector is the VAE's KL at β 128 with fixed variance, read downwards (lower KL = unfamiliar): test AUROC 0.901,
  0.989 for OOD-dangerous tasks, 53% and 85% of those windows flagged at 1.7% false alarms. LOF on the posterior means
  never beats the Gaussian-NLL AE, and posterior variance and the likelihood scores stay near chance. A logistic
  regression on a model's native scores (supervised, same tasks) adds 0.03 to 0.09 AUROC.
- VAE on 3 s + spectral input (`train_vae.py --inputs raw_3s_spectral`; β 1, 32, 64, 128 fixed, β 1 learned): the KL
  effect disappears there (β prunes only to 13-18 dimensions and the KL reads upward at 0.69-0.75; the input ablation
  below shows the spectral features, not the window, remove it); the best 3 s + spectral VAE (β 1, fixed, LOF 0.910)
  matches the 3 s Gaussian-NLL AE. Pairing the 1 s VAE KL with the 3 s Gaussian-NLL LOF
  in a supervised classifier gives 0.950 (0.995 dangerous). Onset latency (notebook 05, section 8): the 1 s VAE KL flags
  every OOD-dangerous bout, half within 0.17 s; 3 s + spectral latents alarm on 11-20% of familiar bouts, 1 s ones on
  5-11% (the ablation below traces the difference to the spectral features).

**Sequence VAEs and the full input ablation** (`ml/common/models/seq_vae.py`; latent spaces in
[`onboard/03_LSTM_vae_latent_space.ipynb`](notebooks/nature/model_exploration/onboard/03_LSTM_vae_latent_space.ipynb) and
[`onboard/04_Xformer_vae_latent_space.ipynb`](notebooks/nature/model_exploration/onboard/04_Xformer_vae_latent_space.ipynb),
scores in notebook 05, section 9). `LSTMVAE` has a 2-layer LSTM encoder averaged to the conv VAE's 4 x 8 latent and a
non-autoregressive LSTM decoder (107 to 109 k parameters); `TransformerVAE` cuts the window into 40 patch tokens with 3
self-attention layers each way (217 to 251 k). Both ran on all four inputs at β 1, 32 and 128 with fixed decoder
variance and β 1 with the learned one, three seeds; the conv VAE and the three autoencoders were retrained on the two
new inputs (`ae_data.py --ablation-only` builds them as views of the 3 s window table). Test AUROC for unfamiliar
windows, OOD-dangerous tasks in brackets, with cell and score chosen on the validation participants:

| model type | 1 s raw | 1 s + spectral | 3 s raw | 3 s + spectral |
|---|---|---|---|---|
| plain AE | 0.887 (0.968) | 0.878 (0.933) | 0.950 (0.987) | 0.893 (0.928) |
| sparse AE | 0.846 (0.941) | 0.762 (0.871) | 0.907 (0.937) | 0.811 (0.830) |
| Gaussian-NLL AE | 0.887 (0.966) | 0.868 (0.902) | 0.943 (0.987) | 0.914 (0.956) |
| conv VAE | 0.901 (0.989) | 0.869 (0.967) | 0.939 (0.988) | 0.910 (0.934) |
| LSTM-VAE | 0.859 (0.924) | 0.868 (0.900) | 0.963 (0.993) | 0.906 (0.949) |
| Transformer VAE | 0.896 (0.977) | 0.876 (0.967) | 0.949 (0.985) | 0.889 (0.981) |
| TCN-VAE | 0.863 (0.952) | 0.883 (0.940) | 0.950 (0.995) | 0.904 (0.961) |

- 3 s raw is the best input for every model type; the spectral features lower detection (by 0.03 to 0.10 at 3 s).
  The best detector is the LSTM-VAE on 3 s raw (β 1, fixed variance, LOF): it flags every OOD-dangerous movement, most
  on its first window, at 6% familiar false switches. Mahalanobis distance on the same latent gives 0.950 and its
  posterior variance 0.897 (higher = unfamiliar).
- LOF on the latent is the best score on 23 of the 28 (model type, input) set-ups; the exceptions are the VAEs'
  likelihood on 1 s + spectral and the conv VAE's downward KL on 1 s raw.
- The spectral features, not the 3 s window, set the KL's direction (downward on raw input, upward with them), cause
  most familiar false switches (4 to 6% of familiar movements on 3 s raw against 11 to 15% with them; 7% for the
  LSTM-VAE), and make every architecture learn nearly the same latent (linear CKA 0.68 against 0.37 between raw-input
  latents).
- β 128 collapses the sequence VAEs' posterior on raw input (KL zero): the LSTM-VAE on 1 s and 3 s, the Transformer on 1 s.
- Interactive 3-D pages of each VAE's best fixed- and learned-variance cell (all β 1, 3 s raw):
  `onboard/interactive/{vae,lstm_vae,xformer_vae,tcn_vae}_best_{fixed,learned}_3d.html`. Run: `submit.sh --vae --inputs <input>
  --args "--arch lstm" --array 0-11` (or `--arch transformer`; one core, one GPU, about 4 min per task).
- TCN-VAE (`TCNVAE`, `train_vae.py --arch tcn`; latent spaces in
  [`onboard/08_TCN_vae_latent_space.ipynb`](notebooks/nature/model_exploration/onboard/08_TCN_vae_latent_space.ipynb)):
  residual blocks of dilated causal convolutions whose receptive field covers the window (6 blocks at 1 s, 8 at 3 s),
  the same 4 x 8 latent, 293 to 395 k parameters. It ties the plain AE and the Transformer VAE on 3 s raw (0.950, behind
  the LSTM-VAE's 0.963), ranks the dangerous tasks best of any model (0.995) and does not collapse at β 128. Its training
  step is memory-bound: upsampling through a fixed interpolation matrix and `torch.compile` cut it from 401 to 164 ms on
  an A100 (`slurm/profile_tcn.py`). Its 144 runs (every input, cell, seed and training class) come from
  `train_class_tasks.py --arch tcn > slurm/tcn_tasks.txt` and `submit.sh --train-class --tasks tcn_tasks.txt --time 02:30:00`.

**Training-class ablation** ([`onboard/07_training_class_ablation.ipynb`](notebooks/nature/model_exploration/onboard/07_training_class_ablation.ipynb)).
If a heuristic recognizes ID movement, the novelty model only has to tell OOD-dangerous from OOD-safe movement. Every one
of the 86 configurations above (the TCN-VAE included) was retrained on ID + OOD-safe tasks and on OOD-safe tasks alone (`--train-class` in both
trainers; 420 runs, listed by `ml/nature/awareness/train_class_tasks.py`, run by `slurm/train_class.sbatch`, pulled to
`notebooks/nature/model_exploration/runs_train_class/`). Scoring (`_novelty_scores.evaluate_dangerous`): every native
score, OOD-dangerous against OOD-safe test windows, direction and threshold (1% of OOD-safe windows) set on the validation
participants. Mean test AUROC over the 28 (model type, input) set-ups, cell and score chosen on validation:

| trained on | novelty score, mean (best) | linear read-out of the latent, mean (best) |
|---|---|---|
| ID | 0.763 (0.846) | 0.866 (0.910) |
| ID + OOD-safe | 0.695 (0.786) | 0.882 (0.924) |
| OOD-safe only | 0.699 (0.796) | 0.884 (0.917) |

- Training on safe movement lowers the novelty scores' AUROC on 25 to 26 of the 28 set-ups (by 0.07 on average): dangerous
  movement resembles safe movement (sit-to-stand and squats, step-ups and stairs), so a model of safe movement finds it
  familiar. Step-ups fall to chance once safe movement is trained on (the ID-trained TCN-VAE alone separates them, 0.861).
  No novelty score is a usable dangerous-movement alarm.
- A logistic regression on the same latents (supervised, fitted on the validation participants) separates dangerous from
  safe movement at 0.80 to 0.92 (best: the TCN-VAE trained on ID + OOD-safe, 3 s + spectral, 0.924), and slightly better
  on latents trained on safe movement; the rhythm features help here.
  The dangerous-movement decision belongs in the supervised regime classifier (notebook 06), tested on held-out tasks;
  the ID-trained novelty detector stays for unfamiliar movement.
- Results: `onboard/results/07_train_class_{scores,best,per_task,bouts,probe,cka}.csv`; figures `onboard/figures/07_*.png`;
  3-D page `onboard/interactive/07_train_class_best_3d.html`. Run: `python ml/nature/awareness/train_class_tasks.py >
  ml/nature/awareness/slurm/train_class_tasks.txt`, `push.sh`, `submit.sh --train-class` (about 70 GPU hours), `pull.sh`.

**Baseline: the ankle paper's gait-phase ensemble**
([`onboard/10_gait_phase_ensemble_baseline.ipynb`](notebooks/nature/model_exploration/onboard/10_gait_phase_ensemble_baseline.ipynb)).
Tourk et al.'s best uncertainty estimator (Table I: F1 90.3 against 55.7 for the autoencoder's LOF) is seven gait-phase
TCNs whose disagreement Ψ = (Var(l₁..l₇) + Var(r₁..r₇)) / 2 is thresholded at the 99.5th percentile of training Ψ, an
actuate / do-not-actuate decision. Built here so the awareness models can be judged against the method they extend, and to
show that one anomaly score does not make the three-way controller choice (ID-safe spline, OOD-safe estimator, OOD-dangerous
off): the paper itself notes that gait phase is undefined for non-cyclic tasks, so Ψ ranks tasks by how far they are from a
gait cycle, not by how unsafe they are.
- Targets: the Nature data has no gait phase; `ml/nature/awareness/ensemble_data.py` reads heel strikes off the insoles'
  vertical force in every recording (contact above 50 N, released below 20 N, bounces under 0.1 s dropped, strides of 0.6
  to 2.5 s kept) and writes `data/processed_nature/ae/gait_phase.npy` (phase per leg, NaN off the cycle; `gait_phase.json`
  has the rule and the coverage per task). Coverage: 96 to 98% of walking windows, 81% of stair windows, 2 to 16% of the
  non-cyclic and dangerous tasks.
- Model and training: `ml/common/models/ensemble.py` (the Science Robotics reproduction's class, Table IV) on the 1 s raw
  windows and standardization of `ae_data.py`, `ml/nature/awareness/train_ensemble.py`: masked MSE on sin(2π phase) at the
  window's last sample, Adam 1e-3, batch 1,024, bf16 autocast; `--protocol val` (early stopping on the validation
  participants, the convention of every other Nature model) or `--protocol loso` (the paper's two-stage protocol: each
  training participant held out in turn, then a retrain for the mean best epoch count). Exports `scores.npz` /
  `scores_val.npz` (member predictions, Ψ, target per window). Check: `test_ensemble.py`.
- Notebook 10 scores Ψ with notebooks 05 and 09's protocol (0.5 s causal median; the paper's threshold and the 1%
  false-alarm operating point; two regime cutoffs chosen on the validation participants) and puts its per-task
  median / IQR beside notebook 09's `u`. Results `onboard/results/10_*.csv`, figures `onboard/figures/10_*.png`.
  Run: `python ml/nature/awareness/ensemble_data.py` (20 s), `push.sh`, `submit.sh --ensemble --array 0` then `jobstats`,
  `submit.sh --ensemble --array 1-2` (and `--args "--protocol loso"`), `pull.sh`. Six runs, 3 min each under `val`, 15 under
  `loso` (jobs 53754435, 53755055, 53755061: 1 core, 1.2 GB, GPU 33%); the two protocols agree to the second decimal.
- Results (test participants, mean over seeds). The paper's task: AUROC 0.88 for unfamiliar against familiar (LSTM-VAE
  0.963), 0.992 for OOD-dangerous against familiar (equal to the best latent models); at the paper's threshold 62% of
  unfamiliar and 93% of dangerous windows flagged at 0.5% false alarms. The three-way task: AUROC 0.86 for ID-safe against
  OOD-safe (notebook 09's `u`: 0.98) and 0.78 for OOD-safe against OOD-dangerous (`u`: 0.965); two cutoffs chosen on
  validation give macro F1 0.60 against 0.868, 14% of safe windows switched off (`u`: 2%) and 54% of dangerous windows
  recognised. Per task, Ψ ranks by distance from the gait cycle (Spearman −0.80 against the share of a task's windows with a
  stride): ball toss, sit-to-stand and tug of war top the list interleaved with squats, lift weight, jump and lunges, and
  step-ups sit mid-range. One novelty scale cannot place the estimator band; that is what the regime classifier adds.

**Supplementary latent views** ([`onboard/11_supplementary_latent_views.ipynb`](notebooks/nature/model_exploration/onboard/11_supplementary_latent_views.ipynb)).
The two encoders the regime classifier reads, drawn on the validation participants (seed 0, 400 windows per task) for
slides. The LSTM-VAE detector (β 1, 3 s raw): one UMAP coloured by regime and by each of the classifier's three novelty
inputs (LOF, Mahalanobis, posterior variance) as a percentile among all validation windows, with the familiar 99th
percentile marked; scores are computed in the full latent, UMAP is layout only. All three put OOD-safe and OOD-dangerous in
the same bright periphery (validation AUROC 0.63 to 0.78 between them, against 0.88 to 0.93 for ID-safe against OOD-safe).
The TCN-VAE encoder (β 128, 3 s + spectral, trained on ID-safe + OOD-safe) coloured by regime and task family. Figures
`onboard/figures/11_*.png`, results `onboard/results/11_novelty_percentiles.csv`, movable 3-D pages
`onboard/interactive/11_lstm_vae_novelty_3d.html` (regime, task and the three score lenses) and
`onboard/interactive/11_tcn_vae_regime_3d.html`. `_latent_views.build_3d` now takes continuous lenses (`SCORE_SCALE`).

**What the regime classifier relies on** ([`onboard/12_classifier_importance.ipynb`](notebooks/nature/model_exploration/onboard/12_classifier_importance.ipynb)).
Permutation importance of notebook 06's boosted tree (refitted per seed, reproducing its test scores exactly) on the test
participants: each input group shuffled 10 times, each single input 5 times, three seeds. The two encoders split the job.
Shuffling the LSTM-VAE novelty scores drops ID-safe recall by 0.86 and sends 2.9% of dangerous windows to the spline;
shuffling the TCN-VAE latent drops dangerous recall by 0.78; the rhythm features add 0.24 of dangerous recall. Posterior
variance and Mahalanobis are the most important single inputs; LOF adds little alone (0.013 ± 0.023), so a classifier
without it (no on-device nearest-neighbour search) is worth testing. `_regime_classifier.get_permutation_importance`;
results `onboard/results/12_importance_{groups,features}.csv` (delete both to recompute, about 25 min); figure
`onboard/figures/12_classifier_importance.png`.

**The regime classifier fitted on the training participants** ([`onboard/13_classifier_training_participants.ipynb`](notebooks/nature/model_exploration/onboard/13_classifier_training_participants.ipynb); E39).
`encode_split.py --split train` now encodes VAE runs too, giving every training-participant window (580,140) for the
LSTM-VAE detector and the TCN-VAE encoder. Re-encoded validation windows match the cluster export to within 0.6%.
Notebook 06's classifier was refitted unchanged on 14 people, minus the novelty scores' own in-sample fit windows. The
training participants' features are only mildly in-sample: 0.5–3.0% of their familiar windows sit above the validation
99th percentile, against 0.9–4.6% for new wearers. Cross-fitting the encoders is therefore not needed. With *u* cutoffs
chosen on the now-unused validation participants, the test decision is macro F1 0.875 ± 0.006, no dangerous window on
the spline, and 2.7% of safe windows switched off. Unseen dangerous tasks are switched off 41% of the time (26% for the
4-person fit), but step-ups stay invisible (3%). Results `onboard/results/13_*.csv`, figure
`onboard/figures/13_classifier_training_participants.png`.


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
docs/
  EXPERIMENTS.md        chronological index of every experiment, one-line result each
  experiments/          E01–E38: one file per experiment (hypothesis, method, result, conclusion)
  CLAUDE.md             project rules for Claude Code (imported by the root CLAUDE.md)
deployment/
  deployability_status.md  what is ready for the Jetson, what is missing, and the deployment plan
  docs/                 deployment experiment log: EXPERIMENTS.md index + D01, D02, ... (same format as docs/experiments/)
  bundle.sh             copies the runtime code and checkpoints into deployment/bundle/ (gitignored) to rsync to the Jetson
  awareness.py          on-device awareness layer (numpy/scipy only): novelty scores, regime classifier, causal median, u
  export_awareness.py   fits it as notebook 13 did, checks it reproduces the test scores, saves artifacts/awareness_s*.npz
  test_awareness.py     data-free check of awareness.py (and a saved bundle), runnable on the Jetson
  moment_engine.py      on the Jetson: B1 checkpoint -> ONNX -> TensorRT engine, checked against PyTorch, timed
  send_moment.sh        sends one estimator run to the Jetson, builds its engine there, runs the real-data check
  export_check_windows.py  samples real validation windows + the run's saved predictions for that check
  check_moment_real.py  on the Jetson: engine vs PyTorch and vs the training run's R² on those windows
  scirob_replay.py      Sci-Rob trials -> B1's input conventions (nominal / calibrated IMU) for the replay (D03)
  moment_model.py       the estimator in PyTorch only (loader, standardization, left-leg mirroring), shared
  replay_reference.py   PyTorch reference predictions for the replay (Mac)
  replay_runner.py      on the Jetson: streams the replay sample by sample through the engine, timed
  analyze_replay.py     scores B1 vs inverse dynamics vs the exo's online estimator; Jetson vs reference
  plot_replay.py        D03's figures (accuracy per mode, traces, calibration, lag scan, device) -> docs/figures/
  sync_exo_code.sh      commits exo-code edits made on the Jetson and syncs them with ../HipExoGPUCode (branch uncertainty-aware-hip-exo)
data/
  science-robotics/     supplementary figure data (MATLAB tables) — in git
  molinaro-raw/         34 subject folders of raw 200 Hz CSVs — gitignored, 6.1 GB
  nature/               Nature 2024 dataset, Complete variant (phase1-2/, phase3/) — gitignored, 88 GB
  cache/                trial inventory, Nature Parquet caches — gitignored
  processed_nature/     Nature autoencoder arrays (ml/nature/awareness/ae_data.py) — gitignored
  processed/            windowed model-ready arrays + scaler + config — gitignored
ml/
  common/               dataset-agnostic
    models/             ConvAutoencoder, ensembles, GAN, ConvVAE, LSTMVAE, TransformerVAE and TCNVAE (seq_vae.py)
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
    nature_data.py      data access, task labels and Parquet caches in data/cache/nature/ (test_nature_data.py)
    estimator_labels.py controller classes from the deployed estimator's per-task error (test_estimator_labels.py)
    awareness/          autoencoders: ae_data.py (arrays; four inputs, 1 s / 3 s x with / without spectral), train_ae.py
                        (vanilla, sparse, gaussian), encode_split.py (any split's latents, AE or VAE), train_vae.py (conv, LSTM and
                        Transformer VAEs), ensemble_data.py + train_ensemble.py (the ankle paper's gait-phase ensemble
                        baseline), test_ae.py, test_vae.py, test_ensemble.py, slurm/
notebooks/
  sci_rob/
    data_exploration/   01 load, 02 build windows, 03 feature plots, 04 NIS, 05 synthetic targets
    model_exploration/  01–10: one notebook per model, then LOTO, UMAP and the awareness holdout
  nature/
    data_exploration/   01 dataset audit, 02 engineered features, 03 feature embeddings (PCA, UMAP), 04 estimator labels
    model_exploration/  onboard/ latent spaces (01 autoencoders, 02 conv VAE, 03 LSTM-VAE, 04 Transformer VAE, 08 TCN-VAE;
                        05 novelty metrics; 06 regime classifier; 07 training-class ablation; interactive/ the 3-D pages,
                        results/ the scores), ml_interp/, runs/ (the pulled cluster runs, ID-trained),
                        runs_train_class/ (the ablation's retrained runs)
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
