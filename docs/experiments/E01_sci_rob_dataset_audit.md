# E01: Science Robotics dataset audit and sensor-axis conventions

| Field | Value |
|---|---|
| ID | E01 |
| Date | 2026-09-03 to 2026-09-04 |
| Dataset | Molinaro, Kang & Young, *Science Robotics* 2024: raw 200 Hz recordings (Zenodo 10849318) |
| Status | Complete |
| Sources | Live_Update.docx, "SEP 2 – 18" §1; Methods_Experiments_Reproduction.docx §2 |
| Code / notebooks | `scripts/download_molinaro.sh`; `notebooks/sci_rob/data_exploration/01_load_molinaro.ipynb` |

## Hypothesis

This audit was descriptive, so it had no formal hypothesis. The question it answered was whether the raw dataset could
supply the reference work's 16 inputs (thigh accelerometer, thigh gyroscope, joint angle and joint velocity, per leg)
without hidden hardware or labelling confounds. The audit also had to establish the sensor axis conventions, because the
dataset never documents them.

## Method

- Indexed every trial and joined its five per-trial CSVs (angle, exo, gait phase, GRF, moment), asserting that their
  time columns align.
- Ran a NaN audit per channel family.
- Correlated each thigh-gyro axis with `enc_velo` across all 34 participants to find the sagittal axis and its sign, and
  to check whether the IMUs are mirrored between legs.
- Mapped the four collection phases to their hardware.

## Result

- 34 participants, 5,364 trials, 8.66 M samples, 12.0 h at 200 Hz, all on an aligned time grid. GRF and joint moments
  are not used as model inputs.
- Onboard channels are 0.0–0.4% NaN. Biomechanical labels are 44–82% NaN because inverse dynamics is valid only on the
  force plates. `UC` trials have about twice the label coverage of `BT`/`ET` trials.
- `thigh_gyro_y` is the sagittal axis (r ≈ −0.90 with hip velocity on both legs): gyro y is positive for flexion, while
  the encoder is positive for extension. The sign is the same on both legs and in every participant, so the IMUs are
  **not mirrored** and no per-side flip is needed.
- `thigh_gyro_z_r` is inconsistent across participants (r from −0.78 to +0.63, positive in 5 of 34). This is most likely
  session-dependent mounting rotation, and it is a candidate source of held-out-subject shift.
- Phases: 1 = AB01–AB09, Samsung GEMS (IMU frames reconstructed post hoc). 2 = AB10–AB14, custom exo (measured torque
  all-NaN, overground). 3 = AB15–AB24, custom exo (treadmill LG/RA/RD plus stairs). 4 = AB25–AB34, the same plus
  standing and transitions.

## Conclusion

Restrict modelling to phases 3–4 (AB15–AB34, n = 20, custom hip exo). Phase-1 IMU frames are reconstructed, so including
them would mix hardware novelty with movement novelty. Phase 2 is overground rather than treadmill. Keep every channel
sign exactly as published. The 16 reference channels all have direct hip counterparts. Every later Science Robotics
experiment uses this participant pool.
