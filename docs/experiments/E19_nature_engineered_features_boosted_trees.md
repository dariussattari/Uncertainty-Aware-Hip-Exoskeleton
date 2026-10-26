# E19: Engineered feature families, evaluated with gradient-boosted trees

| Field | Value |
|---|---|
| ID | E19 |
| Date | 2026-10-05 to 2026-10-08 |
| Dataset | Nature 2024 |
| Status | Complete (rerun after the E21 fix) |
| Sources | Live_Update.docx, "OCT 5 – 10" §1 (first half); README "OBJ 2 — Feature extraction" |
| Code / notebooks | `ml/common/features.py` (all causal); `notebooks/nature/data_exploration/02_features.ipynb` |

## Hypothesis

Engineered features (jerk and high-frequency content, polar phase space, spectral regularity, orientation) carry
information that raw sensor summaries do not. They should therefore improve prediction of task group, controller class
and hip moment, especially for non-linear models. Feature engineering was overdue in the project.

## Method

- **Families implemented:**
  - Jerk and HF: RMS jerk, peak jerk, accelerometer HF energy ratio, angular acceleration, gyro HF ratio.
  - Polar phase space (hip angle against velocity): continuous gait phase, phase rate (cadence), phase monotonicity,
    ring radius.
  - Spectral: dominant frequency, spectral entropy, stride regularity, step regularity.
  - Orientation: thigh pitch from gravity fused with the sagittal gyro.
- A gradient-boosted tree ensemble predicted three targets:
  - Task **group**: walking; stairs, curbs and step-ups; squat, lift, sit and lunge; jump, run, cut and shuffle;
    standing and unstructured.
  - **Controller class**: ID, OOD-safe or OOD-dangerous.
  - Hip **moment** (R²).
- Each target was compared against raw 1 s summaries, with families added one at a time, all together, engineered-only,
  and 3 s raw summaries with and without spectral features.

## Result

Change against raw 1 s summaries (baseline: group balanced accuracy 0.906, controller class 0.843, moment R² 0.725):

| Feature set | Group | Controller class | Moment R² |
|---|---|---|---|
| Engineered only | −0.03 | ≈ −0.003 | −0.12 |
| Raw + engineered | +0.035 | +0.042 | +0.009 |
| Raw + jerk / HF | 0 | −0.002 | ≈ 0 |
| Raw + polar phase | +0.011 | +0.009 | ≈ 0 |
| Raw + spectral | +0.033 | +0.039 | +0.003 |
| Raw + orientation | −0.002 | +0.002 | +0.004 |
| Raw summaries, 3 s | +0.039 | +0.046 | −0.011 |
| **Raw 3 s + spectral** | **+0.046** | **+0.056** | −0.008 |

- The polar phase tracks true gait phase to about a tenth of a stride with one fixed offset. That makes it a continuous
  gait phase without heel-strike detection.
- Stride and step regularity are the best single detectors of cyclic tasks (AUROC 0.89 and 0.88).
- Jerk and angular acceleration largely re-express raw variability. No family materially improves the moment probe
  (R² 0.725 to 0.734).

## Conclusion

There is a clear non-linear benefit from the **spectral (rhythm)** family and from **3 s of history**. Jerk/HF and
orientation add nothing, presumably because a non-linear model already captures them from raw data. Polar phase is
valuable as a controller phase signal, not for separation. The chosen input recipe was **3 s raw + spectral**, compared
against the ankle paper's 1 s raw. E31 later showed that this recipe helps classification but *hurts* novelty detection.
