# E04: Kalman-filter innovation (NIS) as a novelty or sensor-health signal

| Field | Value |
|---|---|
| ID | E04 |
| Date | 2026-09-05 to 2026-09-09 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete. Parked as a sensor-health channel |
| Sources | Live_Update.docx, "SEP 2 – 18" §3; README "OBJ 2" (lead idea); Methods §19.3 (cross-reference) |
| Code / notebooks | `notebooks/sci_rob/data_exploration/04_NIS_Exploration.ipynb` |

## Hypothesis

The hip encoder and the thigh-minus-pelvis gyro measure the same joint motion. A Kalman filter that fuses them under a
kinematic model should therefore produce a Normalized Innovation Squared (NIS, $\nu^\top S^{-1}\nu$) that is
chi-squared under a correct model. It should spike when the two sensing modalities stop agreeing, which is what an
unfamiliar movement or a sensor fault looks like. NIS would then be a calibrated, physically grounded novelty statistic.

## Method

- Checked the redundancy assumption: the correlation between encoder rate and the IMU-derived hip rate.
- **Filter v1:** encoder angle and gyro rate both treated as measurements. Angle and rate errors were combined into one
  NIS.
- **Filter v2:** gyro rate used as the process input. Only the error between encoder angle and predicted angle enters
  NIS, which removes the constant-velocity assumption.
- Maximum-likelihood tuning of the noise parameters, and a check of innovation whiteness and its dependence on gait phase
  (participant AB15).
- Matched windowed residual statistics: signed mean (bias), signal correlation (gain), and innovation autocorrelation
  (timing).

## Result

- The redundancy is real: corr(encoder rate, IMU hip rate) = 0.943.
- v1 NIS followed movement intensity, not task identity: **AUROC 0.413** for held-out tasks.
- v2 innovation is **gait-locked**. Its autocorrelation extends well outside the white-noise band, and its size depends
  on position in the gait cycle.
- ML tuning reached calibration only with physically implausible noise levels: the filter "explodes" by claiming the
  sensors are far noisier than they are. That is model mismatch, not poor tuning. Unmodelled effects include sensor
  alignment, soft tissue, hardware motion and gait phase.
- For a linear steady-state filter, NIS ∝ squared innovation. Squaring removes the sign and the temporal structure that
  distinguish bias, gain, dropout and latency faults.
- Matched windowed statistics recover that structure and are useful for timing faults (dropout, latency).
- The statistic collapsed to a function of one measurement and was matched by the absolute hip angular rate, the same
  amplitude collapse the GAN shows in E06.

## Conclusion

Kalman innovation does **not** detect unfamiliar movement. It is kept as a separate, physically interpretable
**sensor-integrity** channel for bias, gain, dropout and latency, using matched residual features rather than NIS.
Better process models (alignment, soft tissue, gait phase) are left to future work.
