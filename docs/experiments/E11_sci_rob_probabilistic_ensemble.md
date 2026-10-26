# E11: Probabilistic forecast ensemble, separating aleatoric from epistemic uncertainty (Methods "Experiment 7")

| Field | Value |
|---|---|
| ID | E11 |
| Date | 2026-09-18 to 2026-09-19 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete |
| Sources | Methods_Experiments_Reproduction.docx §30–33; Live_Update.docx, "SEP 18 – 25" §1 |
| Code / notebooks | `ml/sci_rob/prob_ensemble/`; `notebooks/sci_rob/model_exploration/07_probabilistic_ensemble.ipynb` |

## Hypothesis

E08's score (branch variance) is epistemic only. The hip angle 200 ms ahead is also intrinsically stochastic (heel-strike
timing, step-to-step variability). If each branch predicts a Gaussian (mean and variance), the mixture variance splits
exactly into an aleatoric part (intrinsic variability, which should *not* indicate novelty) and an epistemic part
(disagreement, which should). A controller needs that split: novel terrain calls for reduced assistance, inherent gait
variability does not.

Recorded in advance from one preliminary fold: held-out participants would be about 2× over-confident (z² of 1.84
against 0.99 on training participants).

## Method

- **Decomposition** (uniform mixture of M = 7 Gaussians; population variance, because dividing by M − 1 inflates the
  epistemic term by 7/6):

$$\mathrm{Var}[y] = \underbrace{\tfrac{1}{M}\sum_i \sigma_i^2}_{\text{aleatoric}} + \underbrace{\tfrac{1}{M}\sum_i (\mu_i - \bar\mu)^2}_{\text{epistemic}}$$

  The epistemic term is **bit-identical** to E08's score function applied to the mean head.
- **Loss:** β-NLL (Seitzer et al. 2022), $L_\beta = \mathrm{sg}[\sigma_i^{2\beta}] \cdot \big(\tfrac12[\log 2\pi + s_i + (y-\mu_i)^2 e^{-s_i}]\big)$
  with $s_i = \log\sigma_i^2$ and β = 0.5, after 3 epochs of masked-MSE warm-up. Log-variance clamped to [−7, 3],
  gradient clipping 5.0.
- **Model:** E08's trunk (imported, not copied), with the head widened to 30 → 4 (two means, two log-variances).
  321,958 parameters. The same two-stage LOSO, filter and threshold.
- **Verification before running:** the decomposition matched the direct mixture variance to 2.4 × 10⁻⁷; the NLL matched
  `torch.nn.functional.gaussian_nll_loss` exactly; β = 1 gave exactly ½ of the MSE gradient; the calibration
  diagnostic returned 0.993 and 1.007 on synthetic data.
- **Calibration:** $z = (y-\mu)/\sigma$, whose ideal variance is 1. Both the mean and the median are reported.
- **Training:** 12 of 12 folds; best epochs 3–16 (mean 10.2), so stage 2 ran 10 epochs. Validation NLL −0.0862, MSE of
  the mean 0.0402. 0.12% of log-variance entries sat on a clamp bound.

## Result

| Score | AUROC | Unfiltered | J | Best J | Recall | Specificity |
|---|---|---|---|---|---|---|
| Epistemic | **0.914** | 0.840 | 26.99 | **67.99** | 28.2 | 98.8 |
| Aleatoric | 0.781 | 0.714 | 0.04 | 45.08 | 0.1 | 99.95 |
| Total | 0.876 | 0.810 | 1.00 | 62.50 | 1.3 | 99.7 |
| Ratio | 0.824 | 0.746 | 27.25 | 49.16 | 30.8 | 96.5 |

- Epistemic AUROC is 0.914 against E08's 0.896, but J at the fixed threshold is 0.270 against 0.476. The
  99.5th-percentile rule places this model at 98.8% specificity.
- Flagged by epistemic: LG 2.3%, RA 0.2%, RD 0.3%; SA 18.0%, SD 30.2%, TR 48.1%, ST 100%. Aleatoric flags almost
  nothing.
- Calibration (z², ideal 1):

| Population | z² |
|---|---|
| Training participants, stage 2 | 0.992 |
| Held-out participant, mean over 12 folds | **1.968** (range 1.16–4.17; no fold calibrated) |
| Test ID, median | 1.26 |
| Test ID, mean | **16.57**: the worst 0.1% of windows carry 58.4% of the total |

- That tail is concentrated in one test participant on one fast-walking trial whose speed lies inside the training
  range: **confidently wrong**, not uniformly over-confident.

## Conclusion

Training under a likelihood gives branch means whose disagreement is a slightly better novelty signal (0.914). Adding
aleatoric variance does not help (total 0.876), so the components should be read separately, not summed. Aleatoric
uncertainty transfers across participants about 2× worse than the mean does, which matches the prediction recorded in
advance. Takeaway: disagreement is a candidate score, not an automatic replacement for the point model. Interval
coverage and validation-set thresholds still need checking, and realized error is still needed to expose confident
mistakes.

## Caveats

- The Gaussian assumption is lossy around heel strike.
- The threshold transfers poorly (that is why J is lower).
- The ratio score was expected to fail and is reported only to keep the record complete.
- Single seed.
