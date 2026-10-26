# E09: Krogh–Vedelsby decomposition of the forecast ensemble: disagreement against realized error

| Field | Value |
|---|---|
| ID | E09 |
| Date | 2026-09-16 to 2026-09-18 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete |
| Sources | Methods_Experiments_Reproduction.docx §27.3; Live_Update.docx, "SEP 2 – 18" §11 |
| Code / notebooks | `ml/sci_rob/vanilla/evaluation/ambiguity.py` (identity check kept in code); `notebooks/sci_rob/model_exploration/05_angle_forecast_ensemble.ipynb` |

## Hypothesis

E08's target is available at run time, so the realized error can be measured next to the disagreement. The hypothesis
had three parts:

1. The realized error would rank novelty poorly (a screening exercise predicted about 0.31, the same failure as AE
   reconstruction error).
2. Normalizing disagreement by realized error, $A/(E+\varepsilon)$, would cancel the amplitude component that sank
   three earlier scores.
3. The two-dimensional reading would expose failures that scalar disagreement misses.

## Method

For a uniformly averaged ensemble under squared error, an exact pointwise identity holds:

$$E = \bar{E} - A,\quad E = (\bar f - y)^2,\quad \bar E = \tfrac{1}{M}\sum_m (f_m - y)^2,\quad A = \tfrac{1}{M}\sum_m (f_m - \bar f)^2$$

with M = 7 branches, averaged over both hips. The identity was verified on every test window. Each term was then scored
as a detector, and the test windows were split into quadrants: disagreement at the model's own 99.5th-percentile
threshold, and error at the 95th percentile of ID error (a descriptive cut, not a decision rule).

## Result

- Identity residual: at most 3.81 × 10⁻⁶ over 84,999 windows (floating-point noise). The check is kept in code as a
  correctness test of the masking and averaging.

| Score | AUROC | Best J |
|---|---|---|
| A (disagreement) | 0.893 | 63.2 |
| Ē (mean branch error) | 0.893 | 62.9 |
| E (ensemble error) | **0.867** | 59.1 |
| A / (E + ε) | **0.320** | 0.9 |

- Prediction 1 was wrong. The screening used an *untrained linear baseline*, which has no training manifold, so its
  error ranks by signal complexity. The trained ensemble's per-task median error puts the ID tasks lowest (LG 0.018,
  RA 0.019, RD 0.020) and ST 0.038, TR 0.039, SA 0.072, SD 0.075 above them.
- Prediction 2 was wrong too: the ratio is worse than chance.
- Quadrant occupancy:

| Task | Low A, low E | High A, low E | Low A, high E | High A, high E |
|---|---|---|---|---|
| LG / RA / RD (ID) | 92–93% | 2–3% | 3–4% | 1–3% |
| SA | 33.4% | 8.8% | **29.7%** | 28.1% |
| SD | 20.6% | 16.9% | 17.1% | **45.4%** |
| TR | 40.7% | 36.4% | 3.8% | 19.2% |
| ST | 0% | **100%** | 0% | 0% |

## Conclusion

Disagreement remains the best single score and the only one available without waiting 200 ms. The joint reading is
what a scalar cannot express:

- High A with high E means unfamiliar movement.
- Low A with high E means a confident error ("the branches agreed and were wrong"). This is the signature of a sensor
  fault or disturbance; 29.7% of stair ascent lands here.
- High A with low E is benign; all of standing lands here.

This two-dimensional "safety space" is unavailable to all four reference architectures, because none has a run-time
target. Do not pursue normalizing by error. Two lessons were recorded: an untrained baseline is the wrong instrument
for predicting a trained model's error, and a plausible normalization can fail badly.

## Caveats

- E arrives 200 ms late, so the quadrants are a diagnostic, not a real-time input, unless the controller can tolerate
  the lag on that axis.
- The error cut is descriptive.
