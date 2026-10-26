# E39: The regime classifier fitted on the training participants

| Field | Value |
|---|---|
| ID | E39 |
| Date | 2026-10-22 |
| Dataset | Nature 2024 (14 training, 4 validation, 4 test participants) |
| Status | Complete. Not yet written up in the research-update documents |
| Sources | Notebook `13_classifier_training_participants.ipynb` |
| Code / notebooks | `ml/nature/awareness/encode_split.py` (now encodes VAE runs: `--split train`); `notebooks/nature/model_exploration/onboard/_regime_classifier.py` (`get_split_latents`, `get_split_novelty`, `get_score_fit_windows`); `onboard/13_classifier_training_participants.ipynb`; results `onboard/results/13_*.csv`; figure `onboard/figures/13_classifier_training_participants.png` |

## Hypothesis

E34's regime classifier was fitted on only the four validation participants, because only they and the test
participants had every window exported. It is the first item to fix before deployment (E34, E35 and E38 all rest on
those four people). There were two hypotheses:

1. Fitting the same classifier on the 14 training participants (3.5× the people) improves the test decision. That means a
   higher macro F1, more dangerous windows switched off, and more unseen dangerous tasks switched off, while still never
   sending dangerous movement to the spline.
2. The risk: both encoders were trained on those 14 people. The LSTM-VAE saw their ID-safe windows, and the TCN-VAE saw
   their ID-safe and OOD-safe windows. Their features are therefore in-sample, so familiar windows could look *more*
   familiar than a new wearer's, and the classifier could learn boundaries that are too tight. If that shift is large,
   the encoders have to be cross-fitted (retrained per participant fold) before this fit can be trusted.

**Labels.** The regime labels are E18's `controller_class`. OOD-safe and OOD-dangerous follow where the JME paper
(Molinaro, Scherpereel et al., *Nature* 2024) reported its estimator working well or badly per task.

## Method

- **Encoding.** `encode_split.py` gained a VAE path. It uses `train_vae`'s own model and data loader, and asserts that
  the standardization matches the checkpoint's. It then encoded every in-task window of the 14 training participants
  (580,140) with the LSTM-VAE (ID, 3 s raw, β 1) and the TCN-VAE (ID + OOD-safe, 3 s + spectral, β 128), three seeds
  each, on the laptop (MPS; 3 and 5 min per run).
  - Check: re-encoding 4,000 exported validation windows per run reproduces the cluster export to a relative RMS
    difference of ≤ 0.04% for the LSTM-VAE (fp32 on both sides) and ≤ 0.6% for the TCN-VAE (bf16 on the cluster).
- **Features.** The same 42 inputs as E34: 32 TCN-VAE posterior means, the LSTM-VAE's LOF, Mahalanobis and posterior
  variance, and the 7 rhythm features.
  - The novelty scores for the training participants are fitted exactly as before, on the fit part of each run's
    exported training sample. The notebook asserts that the new path gives the validation participants the same scores
    as the old one.
  - The 15,000 windows the novelty scores are fitted on are in-sample by construction (LOF finds each one among its own
    neighbours), so they are left out of fitting.
- **Protocol** (fixed before fitting):
  - Classifier: notebook 06's choice, not re-tuned (HistGradientBoosting, balanced class weights, 300 iterations).
  - Three fits: *val* (4 people, the E34 fit), *train* (14 people) and *train + val* (18 people).
  - The test participants are scored once, over three seeds.
  - Decision: E34's (0.5 s causal median, then argmax).
  - For the train fit, also E35's *u* with two cutoffs chosen on the validation participants. They are now unused by
    fitting, so this is the first time the cutoffs come from people the classifier never saw.
  - Held-out tasks: each dangerous task is left out of every fitting split in turn.
- **In-sample check before fitting.** Of each split's familiar windows, the share above the validation participants'
  familiar 99th percentile (1% on validation by construction), for each novelty score and for the Mahalanobis distance
  of the TCN-VAE latent.

## Result

**In-sample shift** (familiar windows above the validation 99th percentile, %, mean ± SD over seeds):

| Feature | Train (in-sample) | Val | Test (new wearers) |
|---|---|---|---|
| LSTM-VAE LOF | 0.48 ± 0.19 | 1.00 | 0.86 ± 0.36 |
| LSTM-VAE Mahalanobis | 1.92 ± 0.38 | 1.00 | 2.22 ± 0.48 |
| LSTM-VAE posterior variance | 3.04 ± 0.22 | 1.00 | 4.57 ± 0.28 |
| TCN-VAE latent, Mahalanobis | 1.78 ± 0.07 | 1.00 | 1.45 ± 0.09 |

- Training participants look slightly more familiar to the LSTM-VAE than the test participants do. The gap is smaller
  than the drift between two sets of new people (validation 1% against test 2–5%). The TCN-VAE latent shows no in-sample
  advantage, and per-regime medians are nearly identical across the three splits.

**Test decision** (argmax, mean ± SD over 3 seeds):

| Fitted on | Macro F1 | Recall ID / OOD-safe / dangerous | Dangerous to the spline | Safe switched off |
|---|---|---|---|---|
| val, 4 people (E34) | 0.868 ± 0.005 | 0.93 / 0.92 / 0.79 | 0% | 2.8% |
| train, 14 people | 0.860 ± 0.014 | 0.91 / 0.89 / **0.88** | 0% | 5.1% |
| train + val, 18 people | 0.869 ± 0.013 | 0.94 / 0.90 / **0.88** | 0.02% | 4.5% |
| **train, *u* cutoffs on val** | **0.875 ± 0.006** | 0.89 / 0.93 / 0.82 | **0%** | 2.7% |

- More participants move the operating point rather than the macro F1. They catch more danger and switch off more safe
  movement (the extra false shutdowns come from squats, lifting, poses and twister).
- Stairs reach the spline 78–79% of the time, against 59% for the val fit.
- With *u* and cutoffs chosen on validation (0.21 and 0.85), the training fit is the best decision so far. Only 18% of
  dangerous windows land in the estimator band, against 24% in E35.
- **Unseen dangerous tasks** (share of windows switched off; to the spline in brackets):

| Task left out | Val fit | Train fit |
|---|---|---|
| Ball toss | 24% (0%) | **61%** (0.2%) |
| Sit-to-stand | 47% | **62%** |
| Tug of war | 31% | 38% |
| Step-ups | 1% | 3% |
| All four | 26% | **41%** |

- Per test participant, macro F1 stays in 0.85–0.87 for every fit, so the spread between wearers did not shrink.

## Conclusion

Fit the regime classifier on the training participants and set the *u* cutoffs on the validation participants. This
gives macro F1 0.875, no dangerous window on the spline, 2.7% of safe windows switched off, and a larger share of unseen
dangerous movement switched off (41% against 26%). The training participants' encoder features are only mildly
in-sample (smaller than the shift between new wearers), so cross-fitting the encoders is not needed now. Revisit that if
the encoders get larger or train longer.

What more people did *not* fix:

- Step-ups remain invisible when unseen (3%). They move like stairs and curbs, so this is a label question (E27), not a
  data-quantity one.
- Tug of war is switched off only 38–63% of the time even when seen.
- The F1 spread across wearers is unchanged, which points to per-wearer calibration (E25).

## Caveats

- Four test participants. The seed SD of the train fit (0.014) is three times the val fit's.
- The classifier and its features were chosen on the val fit (E34) and not re-tuned for 14 people.
- The argmax and *u* decisions sit at different operating points, so compare them on the trade-off, not on F1 alone.
- The in-sample check uses the novelty scores and a latent distance. The TCN-VAE latent's direct effect on the
  classifier was not isolated (that would need cross-fitted encoders).
