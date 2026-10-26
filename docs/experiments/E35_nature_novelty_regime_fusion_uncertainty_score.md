# E35: Novelty plus regime-classifier fusion, and a single uncertainty score *u*

| Field | Value |
|---|---|
| ID | E35 |
| Date | 2026-10-15 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | Notebook `09_novelty_regime_fusion.ipynb` (markdown findings); Live_Update.docx "OCT 10 – 16" §5.0 ("Uncertainty Score": pipeline, per-task *u*, torque ramp) |
| Code / notebooks | `notebooks/nature/model_exploration/onboard/09_novelty_regime_fusion.ipynb`; results `onboard/results/09_*.csv`, `09_test_proba.npz` |

## Hypothesis

Three hypotheses:

1. Stacking each model type's best novelty scores onto the regime classifier's probabilities (E34) separates ID-safe,
   OOD-safe and OOD-dangerous better than the classifier alone.
2. The three probabilities collapse into one controller dial, $u = 0\cdot p_\text{ID} + 0.5\cdot p_\text{OOD-safe} + 1\cdot p_\text{dangerous} \in [0, 1]$,
   with no loss for the decision.
3. Inside the estimator band, torque can be graded on a continuous signal.

## Method

- **Stage 1:** E34's classifier, producing out-of-fold probabilities on validation (leave one participant out), so the
  second model never sees probabilities fitted on its own windows.
- **Stage 2:** those probabilities plus, per model type (ID-trained), the best validation score for unfamiliar against
  familiar (from E31) and for dangerous against safe (from E32). Boosted-tree and logistic stackers were compared by
  leave-one-participant-out cross-validation on validation, and test was scored once. Three seeds.
- **Uncertainty score:** two cutoffs on *u*, chosen on validation by macro F1 (0.005 grid), then fixed. In the band,
  torque ramps linearly from full at the first cutoff to `JME_FLOOR` just before the second. From the second cutoff the
  exo is off. The ramp shape is a placeholder.
- **Novelty ramp variant:** the band is set by *u*, but inside it the torque follows a novelty score chosen on
  validation, compared with ramping on $p_\text{dangerous}$.
- **Unseen dangerous tasks:** each removed from both stages in turn.

## Result

- **Stacking does not help.** Test macro F1: classifier alone **0.868 ± 0.005**; with novelty, boosted trees 0.856,
  logistic 0.831; novelty alone 0.824. Logistic stacking buys dangerous recall (0.86 against 0.79) by switching off twice
  as many safe windows (5.6% against 2.8%).
- **The single score *u* works.** ID-safe sits at 0, OOD-safe at 0.5 and dangerous toward 1. Test AUROC: 0.981 for
  ID-safe against OOD-safe, and 0.965 for OOD-safe against dangerous. Cutoffs of **0.25 and 0.825** give the same macro
  F1 (0.868), no dangerous window on the spline, and 2.1% of safe windows switched off.
- **The cost:** dangerous windows the classifier is unsure of spread over 0.5–0.8, inside the estimator band, so 24% of
  dangerous windows still receive 30–70% torque. Getting that to about 5% would mean switching off about 15–20% of safe
  windows.
- **Inside the band *u* barely varies.** OOD-safe task medians are 0.50–0.52 (squats reach 0.80; obstacle walk and curb
  up drop toward ID), so the ramp gives most of them the same ~68% torque.
- **Unseen dangerous tasks:** at most 0.2% reach the spline, but only 16–25% land past the exo-off cutoff (median *u*
  0.58, mean torque 50%). Step-ups are worst (1% switched off) and sit-to-stand best (39%).
- **Novelty ramp:** the LSTM-VAE posterior variance grades OOD-safe widely (IQR 0.70 against 0.01 for *u*) but orders
  safe above dangerous only at 0.74. Ramping on $p_\text{dangerous}$ is the best compromise: OOD-safe gets 0.82 of full
  torque, in-band dangerous 0.43, ordering 0.87. On unseen dangerous tasks the novelty ramp cuts mean torque from 0.50 to
  0.35.

## Conclusion

The regime classifier alone stays the best decision. *u* turns it into one dial with two cutoffs, and the exo-off cutoff
is the safety knob, chosen from the trade-off curve. *u* is the wrong signal for *scaling* the estimator's torque,
because the classifier pushes OOD-safe windows to about 0.5 regardless of how unfamiliar they are. $p_\text{dangerous}$
is the better ramp signal for now. The decisive next measurement is the estimator's own error against that signal, so
the ramp is fitted rather than assumed. Unseen dangerous tasks mostly end up in the estimator band at reduced torque;
shutting them off depends on having seen the task.

## Caveats

Only 4 participants fit both stages, and `JME_FLOOR` and the linear ramp are placeholders.
