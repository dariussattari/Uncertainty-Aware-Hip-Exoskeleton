# E34: Supervised regime classifier: which controller should each window get?

| Field | Value |
|---|---|
| ID | E34 |
| Date | 2026-10-15 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | Notebook `06_regime_classifier.ipynb` (markdown findings); Live_Update.docx "OCT 10 – 16" §4.0 ("Stacking ML Models") and the pipeline diagram; Verbose §6.0 (plan) |
| Code / notebooks | `notebooks/nature/model_exploration/onboard/06_regime_classifier.ipynb`, `_regime_classifier.py` (`test_regime_classifier.py`); results `onboard/results/06_regime_*.csv` |

## Hypothesis

The novelty scores separate unfamiliar from familiar movement (E31) but cannot tell dangerous from safe (E32), while a
supervised read-out of the latent can (0.92). A supervised classifier on the encoder's latent, the novelty scores and the
rhythm features should therefore choose the regime: spline for ID-safe, Young's estimator for OOD-safe, exo off for
OOD-dangerous. The real test is a dangerous movement it **never saw**: it should still keep that movement away from the
spline.

## Method

- **Features per window** (no retraining):
  - Rhythm: the 7 spectral features.
  - Latent: the 4 × 8 posterior means of the TCN-VAE trained on ID + OOD-safe, 3 s + spectral, β 128 (E33).
  - Novelty: LOF, Mahalanobis distance and posterior variance of the ID-trained LSTM-VAE on 3 s raw (E31).
- **Protocol** (fixed before fitting):
  - Only the validation (4) and test (4) participants have every window exported.
  - Fit on validation, with the feature set and model chosen by leave-one-participant-out cross-validation within
    them. Test scored once.
  - Three seeds (the encoder seed and classifier seed vary together).
  - Decision: probabilities, then a 0.5 s causal median, then the argmax regime.
- **Candidates:** majority class, multinomial logistic regression (class-balanced) on each feature set, and a
  gradient-boosted tree on the full set.
- **Target ablation:** regime; task family then its regime; task then its regime; dangerous-or-not with ID given to a
  perfect heuristic.
- **Encoder comparison:** each model type's latent as the only input to a logistic regression.
- **Held-out tasks:** each of the 22 OOD tasks left out of fitting in turn. Where do its test windows go?

## Result

- **Chosen: boosted trees on latent + novelty + rhythm.** Test macro F1 **0.868 ± 0.005** (majority 0.274; rhythm
  features alone 0.591).
  - 93% of ID windows get the spline, 92% of OOD-safe windows the estimator, and 79% of dangerous windows are switched
    off.
  - **No dangerous window gets the spline.**
  - 2.8% of safe windows are switched off.
- **The non-linear model and the extra features matter more than the encoder.** The latent alone with a logistic
  regression gives 0.66–0.69 for every encoder (the TCN-VAE trained on ID + safe is first at 0.690, narrowly).
- **Target ablation:** dangerous-or-not with ID heuristic 0.918; regime directly 0.868; task then regime 0.857; family
  0.491 (no family is mostly dangerous, so that classifier never switches off).
- **Unseen dangerous task:** almost never given the spline (under 0.5% of windows), but only 30–38% switched off; the
  rest goes to the estimator. Held-out step-ups are switched off in only 1–5% of windows (they look like stairs and
  curbs).
- **Cost:** false shutdowns on posture tasks. Squats, lifting and lunges are switched off in 29–35% of windows. Obstacle
  and dynamic walking mostly get the spline (68–82%).

## Conclusion

A supervised classifier on the latent plus novelty scores is the right shape for the regime decision. It never sends
dangerous movement to the spline, even movement it never saw, and that is the failure that matters most. Its weakness
runs the other way: unseen dangerous movement mostly gets the estimator instead of being switched off. The target should
be the regime (or dangerous-or-not), not the task family.

Next steps:

- Fit on the training participants too (only 4 people fit it here).
- Add a per-movement view.
- Settle with the E27 labels whether step-ups belong in the dangerous class.

## Caveats

- 4 fitting and 4 test participants.
- One encoder seed per classifier seed.
- The held-out-task test covers this dataset's 22 OOD tasks, not genuinely new movement.
