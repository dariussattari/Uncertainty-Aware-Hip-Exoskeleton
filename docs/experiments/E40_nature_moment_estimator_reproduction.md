# E40: Reproducing the Nature paper's joint moment estimator (Model A), the OOD-safe fallback controller

| Field | Value |
|---|---|
| ID | E40 |
| Date | 2026-10-24 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | Molinaro, Scherpereel et al., *Nature* 635 (2024), `docs/papers/s41586-024-08157-7.pdf`: Methods, Extended Data Table 1, Extended Data Fig. 3 |
| Code / notebooks | `ml/nature/estimator/moment_data.py`, `train_moment.py`, `test_moment.py`, `slurm/{train_moment.sbatch,push.sh,pull.sh}`; `ml/common/models/moment_tcn.py`; arrays `data/processed_nature/moment/` (version `099eabcbd8356846`) |

## Hypothesis

The control regime called OOD-safe hands control to the Nature paper's task-agnostic joint moment estimator, which this
project has not yet trained. Model A retrains that estimator as the paper describes it: its sensors, architecture,
hyperparameters and 19 training tasks. Two outcomes were expected:

- On held-out participants, its hip and knee errors should land near the paper's (hip R² of about 0.80 to 0.83).
- It should track the deployed estimator's own output (`torque_estimated` in Exo.csv).

If both hold, the pipeline is a validated reference. Models B0 (hip encoder + thigh IMU) and B1 (B0 with IMU rotation
and lever-arm randomization) then change only the inputs.

## Method

- **Inputs (unilateral, 25 channels per leg):**
  - Hip and knee encoder angle and velocity. Velocity is recomputed causally from the angle, because the published hip
    and knee velocities are both half scale.
  - Thigh, shank and foot IMU (accelerometer and gyro).
  - Insole vertical force and COP x/z.
  - 200 Hz, standardized with training statistics.
- **Left leg mirrored into the right leg's frame,** so one unilateral model serves both legs:
  - Negated: lateral accel z, gyro x and gyro y of all three IMUs, and COP z.
  - Evidence: lagged left/right correlation is negative for exactly these channels in 22 of 22 participants (21 of 22
    for COP z), and positive for every other channel.
- **Targets:** total (biological + exo) moment in N·m/kg, from Joint_Moments_Filt, with OpenSim signs.
  - Hip moment 50 ms (10 samples) behind the window's last sample, as the paper trains it.
  - Knee moment at the window's last sample.
  - Masked where inverse dynamics is undefined; used only where the leg's activity flag is set.
- **Model:** a Bai-style TCN with Extended Data Table 1's selected values:
  - 5 residual blocks of 2 causal convolutions, 80 filters, kernel 5.
  - ReLU, weight normalization, spatial dropout 0.15, no L1/L2 regularization.
  - Two heads at the last time step; 301,842 parameters.
  - Dilations 1, 2, 4, 8, 16 give a 249-sample receptive field, which is used as the window.
- **Training data:** the paper's 19 selected tasks (Extended Data Fig. 3b: level-ground walking plus the next 18).
  - Selected: normal walk, poses, side shuffle (calisthenics), push, jump (in place + across), turn and step, cutting,
    sit-to-stand, walk backward, weighted walk, lift weight, tug of war, tire run, dynamic walk (toe and heel walk),
    twister, meander, incline walk (ascent bouts only) and stairs (descent only).
  - Not selected: lunges, stair ascent, declined walk, start/stop, ball toss, obstacle walk, squats, curb up/down and
    step-ups.
  - Dynamic walk and side shuffle are matched to the paper's names by elimination. Both are selected, so the matching
    does not change the training set.
- **Split:** this project's 14 / 4 / 4 participants (the awareness layer's), not the paper's leave-one-participant-out.
  - Windows every 25 ms.
  - Train: 1.72 M windows (both legs). Validation (19 tasks, picks the epoch): 0.51 M. Test (all 26 tasks): 0.56 M.
- **Not stated in the paper, chosen here:** Adam, batch 256, MSE loss, at most 100 epochs, early stopping at patience
  10, window stride 5. Each is logged in config.json.
- **Planned evaluation:** test participants, scored once.
  - Per-task R², RMSE and normalized MAE at the hip and knee, against Extended Data Fig. 4.
  - Agreement with the deployed estimator's `torque_estimated`.
  - Seeds 0–2, reported as mean ± SD.

## Result

**Cluster set-up:**

- Profile (job 55133795, A100, bf16): 7.1 ms per step. cuDNN ran implicit-GEMM convolutions, not FFT.
- Training jobs 55134594 and 55135221: 1 core, 1 GPU.
- jobstats: CPU 48% of the single core (mostly the 4.4 GB array read), 3.1 GB memory sampled, GPU 60%. `--mem` was then
  set to 6G.

**Validation participants (4 people, 19 paper tasks, pooled over windows), seeds 0–2:**

| | Hip | Knee |
|---|---|---|
| R² | 0.857 ± 0.002 | 0.854 ± 0.002 |
| RMSE (N·m/kg) | 0.164 ± 0.001 | 0.152 ± 0.001 |

- Best epochs were 53, 10 and 24. Validation error flattens after about 10 epochs while training error keeps falling.
- Code `248e43c-dirty`, data `099eabcbd8356846`.
- The paper reports an average R² of 0.83 (hip 0.79–0.81 by task category, online, per task and person). The numbers
  here are offline and pooled, so the per-task test comparison is still needed.

**Test participants (scored once; `score_moment.py`, `ml/nature/estimator/results/test_*.csv`).**

- Test people: BT07 and BT15 (Phase 1–2 sessions), BT19 and BT22 (Phase 3).
- Metrics follow the paper: R², RMSE and normalized MAE per participant and task, averaged over people, then over the
  28 paper tasks (incline walk and stairs split by direction). Mean ± SD over 3 seeds.
- "Deployed" is the study's own online estimator (`torque_estimated`, hip sign flipped). It is scored on Phase 3
  recordings only, where its final model ran.

| Subset | Model | Hip R² | Hip RMSE | Knee R² | Knee RMSE |
|---|---|---|---|---|---|
| Phase 3 (BT19, BT22) | Model A | **0.745 ± 0.006** | 0.168 | **0.771 ± 0.007** | 0.129 |
| Phase 3 (BT19, BT22) | Deployed | 0.596 | 0.223 | 0.812 | 0.121 |
| All 4 people | Model A | 0.569 ± 0.008 | 0.192 | 0.747 ± 0.006 | 0.147 |

RMSE is in N·m/kg.

- **Unseen tasks are no worse than trained ones.** On Phase 3, hip R² is 0.753 on the 9 unseen tasks against 0.741 on
  the 19 trained tasks (knee 0.813 against 0.748). This is the paper's generalization claim, reproduced.
- **Weakest hip tasks** (Phase 3): stair descent 0.42, weighted walk 0.50, ball toss 0.54, twister 0.56, poses 0.63.
  **Strongest:** cutting 0.91, normal walk 0.90, lift weight 0.88, incline ascent 0.87.
- **Model A beats the deployed estimator at the hip** on 25 of 28 tasks (not on poses, tug of war or lunges), by the largest margins on cutting, tire run,
  dynamic walk and normal walk. The deployed numbers include the online pipeline: a 55 Hz loop upsampled to 200 Hz,
  timing jitter, and a model trained on Phase 1–2 data only.
- **BT07 is an outlier at the hip:** per-task hip R² 0.22, against 0.55–0.75 for the other three people. Its knee is
  normal (0.73). Its hip estimates carry a constant bias of −0.16 N·m/kg. The cause is not established: its hip
  encoder-to-motion-capture zero offset (−13°) lies within the range of the other sessions (−17.5° to +1.5°).
- **The paper reports hip R² 0.79–0.81** online (10 Phase 3 people). Model A is 0.745 on 2 Phase 3 people, and lower
  with the Phase 1–2 test people included.

## Conclusion

- **Model A is a usable reference,** with one caveat:
  - On Phase 3 people it comes within about 0.05 hip R² of the paper, and it beats the study's own deployed estimator
    at the hip (0.745 against 0.596).
  - Its error does not grow on the 9 tasks it never saw, which is what the OOD-safe fallback needs.
  - The caveat: the test set holds only 4 people, and one (BT07) carries a hip bias of unknown cause. Per-person spread
    is large next to the seed spread.
- **Hip encoder zero varies up to about 15° between sessions.** A deployment model will see the same thing, so it is
  worth considering a random hip-angle offset in B1 alongside the IMU perturbations.
- **Next:** B0 (hip encoder + thigh IMU), then B1 (B0 plus IMU rotation and lever-arm randomization), scored the same
  way.
