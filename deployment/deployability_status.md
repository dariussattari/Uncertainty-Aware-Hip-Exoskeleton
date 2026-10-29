# Deployability status (Nature data, as of 2026-10-27)

Deployment experiments (D01, D02, ...) are logged in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

Split: 14 train / 4 validation / 4 test participants. The test participants are held out from every fit.

| Component | Fitted on | Saved? |
|---|---|---|
| LSTM-VAE (novelty encoder, 3 s raw, β 1; E31) | 14 train, 3 seeds | Yes, checkpoints in `notebooks/nature/model_exploration/runs/vae/` |
| TCN-VAE (regime latent, ID + OOD-safe, 3 s + spectral, β 128; E33) | 14 train, 3 seeds | Yes, checkpoints in `notebooks/nature/model_exploration/runs_train_class/` |
| LOF / Mahalanobis / posterior variance on the LSTM-VAE latent | 15,000-window sample of train | Yes, `deployment/artifacts/awareness_s{0,1,2}.npz` (`deployment/export_awareness.py`), as plain arrays |
| Boosted-tree regime classifier (HistGradientBoosting; E34, E39) | 14 train (E39). E34's original fit used the 4 validation participants | Yes, the 14-train fit, in the same `awareness_s*.npz` as tree arrays; reproduces notebook 13's test scores exactly |
| *u* cutoffs (0.21 / 0.85; E35, E39) | 4 validation | Yes, in the same `awareness_s*.npz` (per seed: 0.20 / 0.84, 0.23 / 0.855, 0.19 / 0.855) |
| Joint-moment estimator, OOD-safe controller: **B1** (hip encoder + thigh IMU, trained across IMU rotation and lever-arm perturbations; E41) | 14 train, 3 seeds | Yes, checkpoints in `ml/nature/estimator/runs/hip_thigh_aug_s{0,1,2}/` (Model A reference: `paper_s*`, E40). `hip_thigh_aug_s2` is on the Jetson as a TensorRT engine, checked on real validation windows |
| Per-person IMU orientation calibration (gravity while standing + sagittal axis from gyro against hip encoder velocity; E41 design) | No fit; computed per wearer | Simulated only (E41 moved-fit test); no implementation for the device yet |
| Hip-angle forecaster (reliability signal) | Not trained on Nature | No |

E39 also fitted the classifier on train + val (18 people), but only as a comparison. The recommended fit is train only
with the *u* cutoffs set on validation.

## The Jetson

| | |
|---|---|
| Board | Jetson Orin Nano Developer Kit Super, JetPack 6.2.1 (L4T R36.4.7), Ubuntu 22.04, 7.4 GB RAM, 58 GB SD card |
| Access | USB-C to the Mac: the Jetson is `192.168.55.1`. `ssh jetson` (alias in `~/.ssh/config`, user `uncertainty_aware`, key login); VS Code: Remote-SSH → `jetson` |
| Python | venv `~/venvs/hipexo` (Python 3.10, `--system-site-packages` so the system TensorRT imports). Activate: `. ~/venvs/hipexo/bin/activate` |
| GPU stack | CUDA 12.6, cuDNN 9.3, TensorRT 10.3 (system), torch 2.8.0 (NVIDIA's Jetson build, `pypi.jetson-ai-lab.io/jp6/cu126`), pycuda 2024.1 (built on the board) |
| Pinned packages | The lab's `new_env` versions: numpy 1.26.4, scipy 1.14.0, pandas 2.2.2, scikit-learn 1.5.1, python-can 4.4.2, pyyaml 6.0.1, matplotlib 3.9.1; plus onnx 1.17.0 |
| Lab code | `~/HipExoGPUCode`, branch `uncertainty-aware-hip-exo` (the lab's `dev/lucien` plus your commits) (git clone of a bundle, origin set to GitHub). Its modules import; its `online_deployed/model/current_best_tcn_new.trt` does **not** load (built by an older TensorRT): rebuild it here from the lab's ONNX before running `main.py` |
| This repo | `~/hip-exo` (filled by `bundle.sh` or `send_moment.sh`; code only so far) |
| No internet | Wheels are downloaded on the Mac (`pip download --platform manylinux2014_aarch64 --python-version 3.10 ...`) and installed with `pip install --no-index --find-links ~/wheels`; the cached wheels are in `~/wheels` |
| Clock | Not synced (no internet): it reads 1969. Set it once per boot until it has internet: `ssh -t jetson "sudo date -s '$(date -u +%Y-%m-%dT%H:%M:%SZ)'"` |

`can0` is already up on the board. Nothing here installs services or touches CAN.

**Editing the exo code.** `~/HipExoGPUCode` on the Jetson and `../HipExoGPUCode` on the Mac are both on branch
`uncertainty-aware-hip-exo` (the lab's `dev/lucien` plus your commits), linked by git over SSH: the Mac repo has a
`jetson` remote, and the Jetson repo accepts pushes into its checkout (`receive.denyCurrentBranch updateInstead`). Edit
on the Jetson (VS Code Remote-SSH → `jetson`, open `~/HipExoGPUCode`), then from the Mac:

```bash
deployment/sync_exo_code.sh "What I changed"
```

It commits the Jetson's changes there (as you, dated by the Mac's clock; files over 20 MB are refused), fast-forwards the
Mac's branch, or pushes the Mac's commits to the Jetson if the Mac is ahead, and stops if the two have diverged. It never
pushes to GitHub (the lab's `Harvard-Slade-Lab/HipExoGPUCode`): `git -C ../HipExoGPUCode push origin uncertainty-aware-hip-exo`
when you choose to.

## Sending the joint-moment estimator

B1's best seed on validation is `hip_thigh_aug_s2` (hip R² 0.807, RMSE 0.190 N m/kg; seeds 0 and 1: 0.805, 0.806). One
command from the Mac sends that run's `best.pt` + `config.json` and the code that loads it, then builds and checks the
engine on the Jetson:

```bash
deployment/send_moment.sh                                   # hip_thigh_aug_s2; or: deployment/send_moment.sh <run_dir> [--fp16]
```

On the Jetson, `deployment/moment_engine.py` folds the input standardization into the network (raw units in, N m/kg
out), exports ONNX (static batch 2: left, right leg), builds the engine with `trtexec`, and fails unless it matches
PyTorch on the GPU (fp32: 5e-3 N m/kg, fp16: 3e-2). It writes `moment.onnx`, `moment.engine` and `moment.json` (the
input contract: channel order, units, 249 samples at 200 Hz, left-leg mirroring, 50 ms output delay) into the run
folder. Tested with a randomly initialized stand-in (not a trained model): fp32 engine error 1e-7 N m/kg, 1.85 ms per
two-leg call (p99 2.1 ms) against 5.9 ms for PyTorch; fp16 is no faster for a network this small, so fp32 is the default.

`send_moment.sh` then runs the **real-data check**: `export_check_windows.py` samples 2048 validation windows (all 26
tasks; test participants untouched) with the training run's saved predictions, and `check_moment_real.py` runs the engine
on them on the Jetson, failing unless it matches PyTorch fp32 there and reproduces the run's R². Sent and checked
2026-10-27 for `hip_thigh_aug_s2` (`check_real.json` on the Jetson):

| On 2048 validation windows | Value |
|---|---|
| engine vs PyTorch fp32 on the Jetson, max abs | 7.6e-5 N m/kg |
| engine vs the cluster's saved predictions (bf16), RMS / max | 0.0013 / 0.0078 N m/kg |
| hip R², engine / cluster | 0.7530 / 0.7529 (all 26 tasks; 0.807 is the paper-task subset the run summary reports) |
| Phase 3 windows (738), hip R²: engine / the Nature Estimator | 0.725 / 0.671 |

The live inputs still have to be produced in the Nature conventions the engine expects: hip angle (deg) and its causal
derivative (`features.get_causal_velocity`, deg/s), thigh IMU accel (m/s²) and gyro (deg/s) rotated into the Nature thigh
frame (step 7's calibration), left leg mirrored. The lab's `online_deployed` pipeline uses a different model and input
layout (14 features, 186 samples), so wiring B1 into its CAN loop is a separate change to that repo.

## What goes on the Jetson

`deployment/bundle.sh` copies the runtime files into `deployment/bundle/` (gitignored, about 25 MB) with their repo paths
kept, so the `ml/` imports work unchanged, and writes `VERSION` (git SHA) and `SHA256SUMS`. It stops and lists any
checkpoint missing locally. Edit its file list and this table together. The rsync filters keep engines built on the Jetson.

```bash
deployment/bundle.sh
rsync -a --delete --filter='P *.engine' --filter='P moment.onnx' --filter='P moment.json' deployment/bundle/ jetson:~/hip-exo/
ssh jetson 'cd ~/hip-exo && sha256sum -c SHA256SUMS'
```

| In the bundle | Role | Source |
|---|---|---|
| `ml/common/models/moment_tcn.py` + `hip_thigh_aug_s{0,1,2}/best.pt` | B1 joint-moment estimator, 8 channels per leg | E41 |
| `deployment/moment_engine.py` | Builds and checks B1's TensorRT engine on the Jetson (see above) | |
| `ml/common/models/seq_vae.py` + `lstm_vae_raw_3s_b1_fixed_s{0,1,2}/best.pt` | Novelty encoder, 16 channels, 3 s | E31 |
| `seq_vae.py` + `tcn_vae_raw_3s_spectral_b128_fixed_s{0,1,2}/best.pt` | Regime latent, 23 channels (16 raw + spectral), 3 s | E33 |
| `ml/common/features.py` | Causal spectral features for the TCN-VAE input | E21 |
| `ml/common/signals.py`, `ml/nature/estimator/imu_fit.py` | IMU frame solver and rotation / lever-arm model, for per-wearer calibration | E41 |
| `deployment/awareness.py` + `deployment/artifacts/awareness_s{0,1,2}.npz` | Novelty scores, boosted-tree regime classifier, 0.5 s causal median, *u* and its cutoffs | E39 |
| `deployment/test_awareness.py` | Data-free check, run on the device: `python deployment/test_awareness.py deployment/artifacts/awareness_s0.npz` | |

Each `best.pt` holds the weights, the input standardization (`mean`, `std`) and the training config, so no dataset
files go on the device. Device dependencies: `torch`, `numpy`, `scipy`, `pandas` (for `features.py`), TensorRT. The
awareness bundles need **no scikit-learn**.

**Awareness bundles.** `deployment/export_awareness.py` (on the Mac with scikit-learn 1.9.0, needs the local data) refits
the novelty scores, classifier and *u* cutoffs exactly as notebook 13 did for the 14-person training fit, converts the
fitted LOF (k = 20, its 15,000 reference latents with their k-distances and reachability densities) and the boosted trees
(183 trees, 11,163 nodes) to plain arrays, and saves `deployment/artifacts/awareness_s{seed}.npz` (gitignored, 5–6 MB,
no pickles) only if: the numpy LOF matches the notebook's cached novelty scores (1e-6), the numpy tree walk matches
scikit-learn's probabilities (1e-12), and the result, run window by window as on the device, reproduces the notebook's
test results (`13_regime_scores.csv`, `13_uncertainty_scores.csv`) to 1e-9. Novelty scoring takes about 5.8 ms per window
on the Jetson, nearly all of it the LOF neighbour search (windows arrive every 50 ms).

Why array form: scikit-learn 1.9.0 needs Python ≥ 3.11, and the Jetson's GPU PyTorch is built for 3.10. Refitting under
scikit-learn 1.7.2 (the newest for 3.10) gives a slightly different classifier, within seed noise of the validated one
(test macro F1 on *u* 0.874 ± 0.006 against 0.875 ± 0.006; dangerous recall 0.823 against 0.817; 0.006% of dangerous
windows to the spline against 0%), so the validated fit is kept and shipped as arrays. Per window on the device:

```python
import awareness as aw
bundle = aw.load_bundle(path)                          # once
smoother = aw.CausalMedian()                           # once per stream
novelty = aw.score_novelty(bundle.novelty, lstm_z, lstm_logvar)                # (1, 32) x 2 -> (1, 3)
x = aw.get_features(tcn_z, novelty, rhythm)                                     # (1, 32), (1, 3), (1, 7) -> (1, 42)
p = smoother.push(end_sample, aw.get_proba(bundle.trees, x)[0])                # (3,)
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
