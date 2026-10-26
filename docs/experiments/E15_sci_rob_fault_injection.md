# E15: Injected sensor faults: the four-cell grid (ID/OOD × safe/dangerous)

| Field | Value |
|---|---|
| ID | E15 |
| Date | 2026-09-23 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34, DOWN holdout |
| Status | Complete (seed 0) |
| Sources | `ml/sci_rob/awareness/README.md` ("Injected sensor faults"); Live_Update.docx, "SEP 25 – OCT 2" §1 (mentions the toolbox); Live_Update "SEP 18 – 25" pipeline figures (ID-dangerous class) |
| Code / notebooks | `ml/sci_rob/awareness/faults.py`; results `ml/sci_rob/awareness/results/DOWN/faults_*`; `notebooks/sci_rob/model_exploration/10_awareness_regime_holdout.ipynb` |

## Hypothesis

A controller needs a fourth class beyond ID/OOD: **ID-dangerous**, meaning a familiar movement seen through a failing
sensor. The hypothesis was that a novelty axis combined with a forecast-error ("health") axis would resolve four cells:
ID-safe, ID-dangerous, OOD-safe and OOD-dangerous. Novelty would separate ID from OOD, and forecast error would separate
healthy from faulted.

## Method

- Five persistent input faults: a dead left-thigh IMU, a frozen right-thigh IMU, IMU noise at 2× the channel spread,
  gyro spikes, and a 100 ms left-leg delay. A constant encoder offset was tried and dropped, because it is a calibration
  error, not a dynamic fault.
- Faults were applied to level-ground windows (giving ID-dangerous) and ramp-descent windows (giving OOD-dangerous,
  since RD is unseen under DOWN). They were scored alongside the untouched windows (ID-safe, OOD-safe) through the whole
  E14 pipeline.
- Health axis = **routed expert error** (E14), so the regime has its say. Metric: AUROC of each dangerous cell against
  its own safe cell.

## Result

| Fault | ID-dangerous vs ID-safe: LOF / recon / disagreement / routed error | OOD-dangerous vs OOD-safe: LOF / recon / disagreement / routed error |
|---|---|---|
| Dead left IMU | 0.989 / 0.915 / 0.983 / 0.839 | 0.975 / 0.872 / 0.982 / 0.645 |
| Frozen right IMU | 0.989 / 0.931 / 0.973 / 0.920 | 0.998 / 0.830 / 0.955 / 0.643 |
| IMU noise | 0.966 / 0.989 / 0.939 / 0.670 | 0.914 / 1.000 / 0.945 / 0.431 |
| Gyro spikes | 0.989 / 0.989 / 0.988 / 0.804 | 1.000 / 1.000 / 0.999 / 0.323 |
| Left-leg delay | 0.981 / 0.816 / 0.897 / 0.873 | 0.827 / 0.758 / 0.608 / 0.642 |

- **In the novelty × forecast-error plane, the cells do not resolve.** ID-dangerous lands where OOD-safe already is
  (routed error against OOD-safe: 0.16–0.53). A faulted sensor barely hurts a hip-angle forecast that leans on the
  intact channels.
- **In the novelty × reconstruction plane, they do.** LOF separates ID from OOD (0.98), and reconstruction error
  separates dangerous from safe within each column (0.76–1.00). Separation is sharpest for noise and spikes and weakest
  for the leg delay. Disagreement does nearly as well, except for the delay.
- The regime probe becomes *more* confident under faults, so its confidence is anti-informative.

## Conclusion

The grid's two axes are **LOF for familiar against unfamiliar** and **reconstruction error (with disagreement) for
healthy against unhealthy**. Reconstruction error fails as a *movement*-novelty score (E05) but works as a *sensor*-fault
score. Forecast error belongs to the question of what the fault does to the controller, not to fault detection.

## Caveats

- Synthetic faults on two tasks only.
- Seed 0.
- A delay fault is the hardest for every score.
