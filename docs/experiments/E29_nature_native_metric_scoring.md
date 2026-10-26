# E29: Scoring the VAE grid with every metric native to each model family

| Field | Value |
|---|---|
| ID | E29 |
| Date | 2026-10-13 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | 12OCT-15OCT Verbose.docx §3.4 and §11.0; README "Every native metric, scored" bullet; Live_Update.docx "OCT 10 – 16" (supporting metric table) |
| Code / notebooks | `notebooks/nature/model_exploration/onboard/05_Novelty_Metric_comparison.ipynb`, section 7; `_novelty_scores.py` (registry `NATIVE`, check `test_novelty_scores.py`) |

## Hypothesis

LOF may not be the best score for every model family. A VAE also offers KL, decoder NLL, negative ELBO and posterior
variance. The hypothesis was that some native score beats LOF on the Gaussian-NLL control's 1 s latent (0.887), and that
native scores combine in a downstream classifier.

The exit criterion was fixed before training: a model keeps its row only if it beats 0.97 AUROC on dangerous tasks at
matched false alarms, or cuts onset latency.

## Method

- 19 models and 57 runs: plain, sparse and Gaussian-NLL AEs, plus the 30-run VAE grid (E28).
- Scores:
  - For all families: LOF (raw and standardized latent), Mahalanobis distance (one matrix-vector product per window),
    out-of-range units (a label-free form of the alarm units), and reconstruction error.
  - VAE only: LOF on active dimensions, KL, decoder NLL, negative ELBO, posterior variance.
- **Protocol:** fit on each run's exported training windows. Each score's direction is set on validation (negated if it
  ranks unfamiliar lower, since an inverted signal is as useful). The threshold flags 1% of familiar validation windows
  after a 0.5 s causal median. Test is scored once.
- A downstream logistic regression on each model's native scores, trained on validation with task labels and scored on
  test.

## Result

| Model (1 s) | Best score (chosen on validation) | AUROC unfamiliar | AUROC dangerous | Unfamiliar / dangerous flagged | False alarms |
|---|---|---|---|---|---|
| Gaussian-NLL AE | LOF, raw | 0.887 ± 0.014 | 0.966 | 34 / 48% | 2.0% |
| Plain AE | LOF, raw | 0.887 ± 0.010 | 0.968 | 32 / 50% | 1.8% |
| Gaussian-NLL AE, 3 s + rhythm | LOF, standardized | 0.914 ± 0.013 | 0.956 | 43 / 45% | 1.7% |
| VAE β 1, fixed | LOF, raw | 0.842 | 0.912 | 28 / 34% | 3.6% |
| VAE β 32, fixed | LOF, standardized | 0.882 | 0.969 | 30 / 58% | 2.1% |
| **VAE β 128, fixed** | **KL, lower = unfamiliar** | **0.901 ± 0.004** | **0.989 ± 0.003** | **53 / 85%** | 1.7% |
| VAE β 1, learned | LOF, raw | 0.862 | 0.970 | 38 / 78% | 2.5% |
| VAE β 128, learned | LOF, standardized | 0.870 | 0.962 | 32 / 60% | 2.6% |

- **The high-β KL points down.** At β 128 (fixed variance), a *lower* KL means unfamiliar. The signal appears only once
  β prunes the code: chance at β 1–8, 0.79 at β 32, 0.87 at β 64. With one filter left, the encoder spends its KL on
  what familiar movement has in common, so an unfamiliar window leaves the posterior nearer the prior.
- LOF on posterior means never beats the control (best 0.88).
- Posterior variance (0.52–0.70) and the likelihood scores (NLL, ELBO, reconstruction error: 0.53–0.67) are near
  chance.
- Generic scores (Mahalanobis, out-of-range units) are weak on 1 s latents (0.47–0.67) and better on 3 s (0.62–0.85).
- **Downstream logistic regression** (test AUROC, unfamiliar): a model's native scores together add 0.03–0.09 over its
  best single score (for example, VAE β 128 fixed: 0.901 → 0.929), except for the 3 s Gaussian-NLL latent. The VAE's
  uncertainty scores alone help only at β ≥ 32 (0.87–0.92, against 0.62–0.75 at β 1–8). For dangerous tasks the
  combination reaches 0.998.

## Conclusion

The VAE's useful uncertainty is the **posterior's distance from the prior at high β**, not its width or likelihood. The
1 s VAE KL at β 128 is the best single 1 s detector so far, and it is seed-stable. Pass the KL, not the posterior
variance, to a downstream classifier, and give that classifier several native scores, because they combine.

## Caveats

The downstream classifiers saw the unfamiliar tasks in training. They show what the scores can give a classifier, not
how a detector meets new movement; held-out-task tests come in E34.
