# Deployability status (Nature data, as of 2026-10-25)

Split: 14 train / 4 validation / 4 test participants. The test participants are held out from every fit.

| Component | Fitted on | Saved? |
|---|---|---|
| LSTM-VAE (novelty encoder, 3 s raw, β 1; E31) | 14 train, 3 seeds | Yes, checkpoints in `notebooks/nature/model_exploration/runs/vae/` |
| TCN-VAE (regime latent, ID + OOD-safe, 3 s + spectral, β 128; E33) | 14 train, 3 seeds | Yes, checkpoints in `notebooks/nature/model_exploration/runs_train_class/` |
| LOF / Mahalanobis / posterior variance on the LSTM-VAE latent | 15,000-window sample of train | Yes, `deployment/artifacts/awareness_s{0,1,2}.pkl` (`deployment/export_awareness.py`) |
| Boosted-tree regime classifier (HistGradientBoosting; E34, E39) | 14 train (E39). E34's original fit used the 4 validation participants | Yes, the 14-train fit, in the same `awareness_s*.pkl`; reproduces notebook 13's test scores exactly |
| *u* cutoffs (0.21 / 0.85; E35, E39) | 4 validation | Yes, in the same `awareness_s*.pkl` (per seed: 0.20 / 0.84, 0.23 / 0.855, 0.19 / 0.855) |
| Joint-moment estimator, OOD-safe controller: **B1** (hip encoder + thigh IMU, trained across IMU rotation and lever-arm perturbations; E41) | 14 train, 3 seeds | Yes, checkpoints in `ml/nature/estimator/runs/hip_thigh_aug_s{0,1,2}/` (Model A reference: `paper_s*`, E40) |
| Per-person IMU orientation calibration (gravity while standing + sagittal axis from gyro against hip encoder velocity; E41 design) | No fit; computed per wearer | Simulated only (E41 moved-fit test); no implementation for the device yet |
| Hip-angle forecaster (reliability signal) | Not trained on Nature | No |

E39 also fitted the classifier on train + val (18 people), but only as a comparison. The recommended fit is train only
with the *u* cutoffs set on validation.

## What goes on the Jetson

`deployment/bundle.sh` copies the runtime files into `deployment/bundle/` (gitignored, about 10 MB) with their repo paths
kept, so the `ml/` imports work unchanged, and writes `VERSION` (git SHA) and `SHA256SUMS`. It stops and lists any
checkpoint missing locally. Edit its file list and this table together.

```bash
deployment/bundle.sh
rsync -a --delete deployment/bundle/ <user>@<jetson>:~/hip-exo/
ssh <user>@<jetson> 'cd ~/hip-exo && sha256sum -c SHA256SUMS'
```

| In the bundle | Role | Source |
|---|---|---|
| `ml/common/models/moment_tcn.py` + `hip_thigh_aug_s{0,1,2}/best.pt` | B1 joint-moment estimator, 8 channels per leg | E41 |
| `ml/common/models/seq_vae.py` + `lstm_vae_raw_3s_b1_fixed_s{0,1,2}/best.pt` | Novelty encoder, 16 channels, 3 s | E31 |
| `seq_vae.py` + `tcn_vae_raw_3s_spectral_b128_fixed_s{0,1,2}/best.pt` | Regime latent, 23 channels (16 raw + spectral), 3 s | E33 |
| `ml/common/features.py` | Causal spectral features for the TCN-VAE input | E21 |
| `ml/common/signals.py`, `ml/nature/estimator/imu_fit.py` | IMU frame solver and rotation / lever-arm model, for per-wearer calibration | E41 |
| `deployment/awareness.py` + `deployment/artifacts/awareness_s{0,1,2}.pkl` | Novelty scores, boosted-tree regime classifier, 0.5 s causal median, *u* and its cutoffs | E39 |
| `deployment/test_awareness.py` | Data-free check, run on the device: `python deployment/test_awareness.py deployment/artifacts/awareness_s0.pkl` | |

Each `best.pt` holds the weights, the input standardization (`mean`, `std`) and the training config, so no dataset
files go on the device. Python dependencies: `torch`, `numpy`, `scipy`, `pandas` (the last two for `features.py`), and `scikit-learn` at
the version that wrote the awareness bundles (`bundle.meta['sklearn']`, 1.9.0): pickled estimators are not portable
across versions.

**Awareness bundles.** `deployment/export_awareness.py` (on the Mac, needs the local data) refits the novelty scores,
classifier and *u* cutoffs exactly as notebook 13 did for the 14-person training fit, runs them through
`awareness.py` window by window as the device will, and saves `deployment/artifacts/awareness_s{seed}.pkl` (gitignored)
only if every seed reproduces the notebook's test results (`13_regime_scores.csv`, `13_uncertainty_scores.csv`) to
1e-9. Each bundle holds the LOF (k = 20, its 15,000-window reference set inside), Mahalanobis mean and Cholesky factor,
the posterior-variance active units, the classifier, the two cutoffs and metadata (runs, git SHA, data version, library
versions). Per window on the device:

```python
import awareness as aw
bundle = aw.load_bundle(path)                          # once
smoother = aw.CausalMedian()                           # once per stream
novelty = aw.score_novelty(bundle.novelty, lstm_z, lstm_logvar)                # (1, 32) x 2 -> (1, 3)
x = aw.get_features(tcn_z, novelty, rhythm)                                     # (1, 32), (1, 3), (1, 7) -> (1, 42)
p = smoother.push(end_sample, aw.get_proba(bundle.classifier, x)[0])           # (3,)
regime = aw.get_regime(aw.get_uncertainty(p), bundle.cutoffs)                  # spline / estimator / exo off
```

**Not in the bundle yet**, so the full uncertainty-aware controller cannot run on the device until they exist:

- the seven rhythm features computed live (the classifier's last 7 inputs; offline they come from `features.py` via the window table);
- the window-building code for live data: the training arrays come from `ml/nature/awareness/ae_data.py` and
  `ml/nature/estimator/moment_data.py`, which read the Nature Parquet caches. The device needs a streaming version
  that produces the same channels, order and units from a ring buffer;
- the torque ramp (step 5), per-wearer calibration (step 7) and switching logic (step 8).

## Deployment plan

Order matters. Items 2–4 change the labels or the inputs, so the classifier and cutoffs are refitted once at the end of
them, then saved (item 6).

| # | Step | Status | Source |
|---|---|---|---|
| 1 | Fit the regime classifier on the training participants | **Done.** Train fit + *u* cutoffs on validation: macro F1 0.875 ± 0.006, 0% dangerous on the spline, 2.7% of safe windows switched off, unseen dangerous tasks 41% switched off. Encoder features only mildly in-sample, so no cross-fitting | E39 |
| 2 | Build the OOD-safe fallback estimator on the project's sensors | **Done, with one addition open.** B1 reaches Phase 3 hip R² 0.70 and keeps it when the IMU moves by up to 15° and 10 cm. Open: add a random hip-encoder zero offset (up to about 15° between sessions, E40) to B1's training randomization | E40, E41 |
| 3 | Re-derive the OOD-safe / OOD-dangerous labels from B1's per-task error (E27 rule) | **Open, now unblocked.** The current labels follow the JME paper's full-sensor estimator (E18). Thigh-only sensing fails on foot-loaded unfamiliar tasks (stair ascent, squats, lunges, step-ups, ball toss), so squats and lunges may move to dangerous. Decide step-ups here too: unseen, they are switched off only 3% of the time (E39). A label change means retraining the TCN-VAE on the new ID + OOD-safe set and refitting the classifier | E27, E39, E41 |
| 4 | Test a classifier without LOF | **Open; a quick refit** on E39's saved features. LOF needs an on-device 20-nearest-neighbour search and adds little (0.013 ± 0.023 F1) | E38, E39 |
| 5 | Fit the torque ramp to B1's measured error against p<sub>dangerous</sub> | **Open, now unblocked.** `JME_FLOOR` and the linear ramp are placeholders | E35 |
| 6 | Save a deployable bundle | **Done for the 14-person fit.** `deployment/export_awareness.py` saves the classifier, novelty fits and cutoffs per seed with seed, git SHA and data version, and checks they reproduce the test scores; `deployment/bundle.sh` adds B1 and the encoders. Still to decide: refit on all 22 with the cutoffs frozen. Steps 3 and 4 change the fit, so rerun the export after them | E39 |
| 7 | Per-wearer calibration at donning | **Open.** One short bout: about 5 s standing, then a few strides of normal walking. It gives the IMU orientation (E41) and the per-wearer alarm threshold, since false-alarm rates differ by about 2× between people (E25). The classifier's F1 spread across test wearers (0.85–0.87) did not shrink with more people (E39) | E25, E39, E41 |
| 8 | Switching logic | **Open.** Dwell time and hysteresis between regimes, the false-alarm target (a controller decision), and a per-movement view of the decisions | E25, E30, E34 |
| 9 | Validation gaps | **Open.** Stair-descent leave-one-task-out holdout; genuinely non-cyclic OOD data; real data from this project's exoskeleton (E41's moved-fit test simulates mounting error, not a different device) | E12, E41 |
