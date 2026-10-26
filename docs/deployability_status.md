# Deployability status (Nature data, as of 2026-10-25)

Split: 14 train / 4 validation / 4 test participants. The test participants are held out from every fit.

| Component | Fitted on | Saved? |
|---|---|---|
| LSTM-VAE (novelty encoder, 3 s raw, β 1; E31) | 14 train, 3 seeds | Yes, checkpoints in `notebooks/nature/model_exploration/runs/vae/` |
| TCN-VAE (regime latent, ID + OOD-safe, 3 s + spectral, β 128; E33) | 14 train, 3 seeds | Yes, checkpoints in `notebooks/nature/model_exploration/runs_train_class/` |
| LOF / Mahalanobis on the LSTM-VAE latent | 15,000-window sample of train | No, refitted inside the notebook on each run |
| Boosted-tree regime classifier (HistGradientBoosting; E34, E39) | 14 train (E39). E34's original fit used the 4 validation participants | **No.** It exists only in memory in `onboard/13_classifier_training_participants.ipynb`; only result CSVs (`onboard/results/13_*.csv`) are written |
| *u* cutoffs (0.21 / 0.85; E35, E39) | 4 validation | Only as values in `onboard/results/13_uncertainty_scores.csv` |
| Joint-moment estimator, OOD-safe controller: **B1** (hip encoder + thigh IMU, trained across IMU rotation and lever-arm perturbations; E41) | 14 train, 3 seeds | Yes, checkpoints in `ml/nature/estimator/runs/hip_thigh_aug_s{0,1,2}/` (Model A reference: `paper_s*`, E40) |
| Per-person IMU orientation calibration (gravity while standing + sagittal axis from gyro against hip encoder velocity; E41 design) | No fit; computed per wearer | Simulated only (E41 moved-fit test); no implementation for the device yet |
| Hip-angle forecaster (reliability signal) | Not trained on Nature | No |

E39 also fitted the classifier on train + val (18 people), but only as a comparison. The recommended fit is train only
with the *u* cutoffs set on validation.

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
| 6 | Save a deployable bundle | **Open.** Classifier, LOF / Mahalanobis fits, cutoffs, feature standardization and B1, with seed, git SHA and data version, plus a check that loads the bundle and reproduces the test scores. Decide first: keep the 14-person fit with validation cutoffs, or refit on all 22 with the cutoffs frozen | E39 |
| 7 | Per-wearer calibration at donning | **Open.** One short bout: about 5 s standing, then a few strides of normal walking. It gives the IMU orientation (E41) and the per-wearer alarm threshold, since false-alarm rates differ by about 2× between people (E25). The classifier's F1 spread across test wearers (0.85–0.87) did not shrink with more people (E39) | E25, E39, E41 |
| 8 | Switching logic | **Open.** Dwell time and hysteresis between regimes, the false-alarm target (a controller decision), and a per-movement view of the decisions | E25, E30, E34 |
| 9 | Validation gaps | **Open.** Stair-descent leave-one-task-out holdout; genuinely non-cyclic OOD data; real data from this project's exoskeleton (E41's moved-fit test simulates mounting error, not a different device) | E12, E41 |
