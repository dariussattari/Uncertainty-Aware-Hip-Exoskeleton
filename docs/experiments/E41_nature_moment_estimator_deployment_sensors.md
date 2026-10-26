# E41: The moment estimator on this project's sensors: hip encoder + thigh IMU (B0), and robustness to IMU fit (B1)

| Field | Value |
|---|---|
| ID | E41 |
| Date | 2026-10-24 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | Builds on E40 (Model A); design agreed 2026-10-24 (per-person IMU calibration, residual fit randomization) |
| Code / notebooks | `ml/nature/estimator/train_moment.py` (`--inputs hip_thigh [--augment]`), `imu_fit.py`, `score_moment.py`, `test_moment.py`; runs `ml/nature/estimator/runs/hip_thigh{,_aug}_s{0,1,2}`; results `ml/nature/estimator/results/test_*.csv` |

## Hypothesis

This project's exoskeleton has a hip encoder and a thigh-cuff IMU, and no knee, shank, foot or insole sensing. The IMU's
mounting also differs from the Nature exo's lateral thigh strut, and it varies with each person's fit.

- **(a)** Restricting the paper's estimator to the hip encoder and thigh IMU (B0) costs some hip accuracy. The paper's
  sensor ablation (Extended Data Fig. 1) found the IMUs the most informative sensors and the foot sensing the least.
- **(b)** Deployment calibrates each person's IMU into the Nature frame (gravity while standing, then the sagittal axis
  from gyro against hip velocity). A model trained on one fixed fit (B0) still degrades under the residual mounting
  error. Training across random fits (B1) removes most of that loss at little cost on the recorded fit.

## Method

- **Same data, split, tasks, architecture and hyperparameters as E40,** except:
  - Inputs: 8 channels (hip angle, causal hip velocity, thigh accel xyz, thigh gyro xyz), left leg mirrored as in E40.
  - One output head: total hip moment, 50 ms behind the window's end.
- **B1 moves each training window's thigh IMU to a random fit:**
  - Rotation: uniform random axis, angle uniform in 0–15°.
  - Lever arm: ±10 cm along the thigh (y), ±5 cm across it (x, z); anterior cuff against lateral strut.
  - Rigid-body transfer: `gyro' = R·gyro`, `accel' = R·(accel + α×r + ω×(ω×r))`, with α from the gyro's time
    derivative.
- **Evaluation:** every model with a thigh IMU is also scored on the test windows moved to fits drawn once from a fixed
  seed (2026), the same fits for every model (`@fit`).
- **Scoring as in E40:** per participant and task R², RMSE and normalized MAE, averaged over people, then over the 28
  paper tasks. 3 seeds each.
- **Cluster:**
  - Profiles: 7.0 ms per step for B0 and 8.9 ms for B1 (A100, bf16, implicit-GEMM convolutions).
  - Peak memory 3.9 / 4.6 GB.
  - Seed 0 jobstats: CPU 51–62% of 1 core, GPU 54–57%.
  - Jobs: 55145172 and 55145709 (B0), 55145173 and 55145710 (B1); 16–39 min each.

## Result

Hip, test participants, mean ± SD over 3 seeds. Phase 3 = BT19 and BT22, where the deployed estimator can be compared.

| Model | Phase 3 R² | Trained tasks | Unseen tasks | Moved fit (Phase 3) | All 4 people | All 4, moved fit |
|---|---|---|---|---|---|---|
| A (paper sensors, E40) | **0.745 ± 0.006** | 0.741 | **0.753** | — | 0.569 | — |
| B0 (hip + thigh) | 0.701 ± 0.005 | **0.761** | 0.593 | 0.626 ± 0.006 | 0.523 | 0.445 |
| B1 (B0 + fit randomization) | 0.698 ± 0.007 | 0.752 | 0.602 | **0.685 ± 0.006** | **0.560** | **0.546** |
| Deployed (study's online) | 0.596 | 0.561 | 0.659 | — | — | — |

Phase 3 hip RMSE (N·m/kg): A 0.168, B0 0.187, B1 0.190, deployed 0.223. Validation hip R² (pooled): B0 0.805 and B1
0.806 against A's 0.857.

- **(a) holds, but the loss is concentrated on unseen tasks.**
  - On the 19 trained tasks, B0 matches or beats A (0.761 against 0.741). It is better on weighted walk (+0.22), stair
    descent (+0.18), twister and tug of war.
  - On the 9 unseen tasks it drops 0.16 (0.593 against 0.753). By task: stair ascent 0.08 against A's 0.67, ball toss
    0.24 against 0.54, squats, lunges and step-ups each about 0.2 lower.
  - These are the unseen tasks where the moment is set by loading under the foot (stance on a step, a deep knee bend),
    which the shank, foot and insole sensors see and the thigh does not.
  - Level, inclined and curb walking, obstacle walk and start/stop are unaffected.
- **(b) holds.**
  - Moving the IMU costs B0 0.075 R² on Phase 3 (0.701 → 0.626) and 0.078 over all 4 people.
  - It costs B1 0.013 and 0.014.
  - On the recorded fit, B1 equals B0 (0.698 against 0.701) and is better over all 4 people (0.560 against 0.523).
  - Sit-to-stand shows the effect most: B0 0.76 → 0.53 when moved, B1 0.76 → 0.72.
- **Both B models beat the study's deployed estimator at the hip** (0.70 against 0.60 on Phase 3), as Model A did
  (E40).

## Conclusion

- **B1 is the deployment candidate:**
  - Hip R² is 0.69–0.70 on Phase 3 people and stays there when the thigh IMU moves by up to 15° and 10 cm.
  - The randomization costs nothing measurable on the recorded fit.
  - The per-person calibration plus residual randomization from the agreed design works as intended in simulation.
- **What thigh-only sensing cannot do:**
  - It generalizes much worse to unfamiliar tasks whose hip moment depends on loading at the foot: stair ascent, ball
    toss, squats, lunges and step-ups.
  - That matters for the controller. Squats and lunges are labelled OOD-safe (handled by this estimator), and ball toss
    and step-ups are OOD-dangerous.
  - The OOD-safe/dangerous split came from the paper's full-sensor estimator (E18, E27). It should be re-derived from
    B1's per-task error before B1 is used as the fallback: the open thread "apply the E27 labelling rule to the
    project's own hip-and-thigh estimator".
- **Not tested here:**
  - A hip-encoder zero offset. It varies up to about 15° between sessions (E40) and is a natural addition to B1's
    randomization.
  - Real data from this project's exoskeleton. The moved-fit test is a simulation of mounting error, not of a
    different device.
