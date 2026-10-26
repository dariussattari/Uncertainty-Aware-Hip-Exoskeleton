# E08: Ensemble forecasting the hip angle 200 ms ahead (Methods "Experiment 5")

| Field | Value |
|---|---|
| ID | E08 |
| Date | 2026-09-16 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete |
| Sources | Methods_Experiments_Reproduction.docx §25–27.2, §27.4, §29; Live_Update.docx, "SEP 2 – 18" §8–10 and §14 |
| Code / notebooks | `ml/sci_rob/vanilla/` (`targets.py`, `pilot_horizon.py`, `training/synthetic.py`); `notebooks/sci_rob/model_exploration/05_angle_forecast_ensemble.ipynb`; `06_all_channel_forecast.ipynb` (16-channel variant, not trained) |

## Hypothesis

A future sensor reading is a target that exists at run time: the encoder reports the true hip angle 200 ms later. If
each branch forecasts the bilateral hip angle at a horizon where linear extrapolation fails, the branches will genuinely
disagree off the training manifold. That makes disagreement a stronger novelty score than gait phase, while the realized
error becomes measurable online. The reference work has no counterpart experiment.

## Method

- **Changed from E03:** only the target. It is the left and right hip angle 40 samples (200 ms) after the window ends,
  read from the row 4 positions later in the same stored array (verified bit-exact). The 0.56% of windows without a
  partner are masked.
- **Horizon selection** (pilot: 2 branches, 3 epochs, before the full protocol):

| Horizon | Linear-baseline skill | Pilot AUROC | Pilot Ψ ratio OOD/ID |
|---|---|---|---|
| 5 ms | +0.99 | – | – |
| 50 ms | +0.83 | 0.674 | 1.44× |
| 100 ms | +0.56 | 0.627 | 1.36× |
| 200 ms | −0.12 | 0.699 | 1.59× |

  Skill = 1 − RMSE / SD(target). It crosses zero near 40 samples, which is where structure has to be learned rather
  than extrapolated.
- **Training:** LOSO best epochs ranged 4–36 (mean 19.9), so stage 2 ran 20 epochs. Validation masked MSE 0.0410.
  Threshold 1.015 × 10⁻², reproduced independently as 0.010138.
- **Amplitude diagnostic,** as in E06: rank correlation, AUROC of the score residualized on amplitude, and a comparison
  of the filtered and unfiltered score.
- A 16-channel forecast variant ("Experiment 6") was registered but never trained. The pilot placed it clearly behind,
  probably because the uncertainty is diluted across 14 uninformative outputs.

## Result

| Metric (shared window set) | E08 forecast | E03 gait phase |
|---|---|---|
| AUROC | **0.899** (0.896 on its own set) | 0.859 |
| Best J | 63.99 | 56.78 |
| J at own threshold | 47.87 | 37.54 |
| Recall / specificity | 52.0% / 95.9% | 39.4% / 98.1% |
| ECE / Brier | 0.056 / 0.116 | 0.048 / 0.133 |

- Flagged at a matched 98.1% specificity: **SA 21.9%** (against E03's 6.9%), SD 47.4% (E03: 67.0%), TR 50.9%,
  ST 100%. AUROC for SA alone against ID: 0.880 (E03: 0.766).
- **Operating-point crossover:** the forecast model leads at moderate specificity (95%: 55.6% against 51.8% recall)
  but trails at strict specificity (99.7%: 8.3% against 27.9%). The crossover is near 98.5%.
- **Amplitude:** the reported score beats amplitude (0.896 against 0.807) and still reaches 0.842 after amplitude is
  regressed out. The rank correlation is −0.29 (the GAN's was −0.86), and +0.004 within ID windows. Caveats: the
  unfiltered score beats amplitude by only 0.009, so the margin comes mostly from the causal median. Within
  transitions the correlation is −0.881.

## Conclusion

The 200 ms forecast target is the strongest estimator on the Science Robotics data on both threshold-free metrics. It
needs no labels and is causal at deployment. It fixes the stair-ascent blind spot best of any model. Gait phase stays
better in the very conservative regime (below about 1.5% false alarms), and the two models are partly complementary
(SA against SD). The forecast also exposes the realized error as a second, online axis (E09). The working position after
this experiment: the 200 ms bilateral forecast ensemble is the primary learned detector.

## Caveats

- Single seed: the 0.040 margin over E03 has no confidence interval.
- The threshold transfers loosely (95.85% specificity realized against a 99.5% target).
- Transition detection should not be credited to learned structure, given the −0.881 amplitude correlation there.
