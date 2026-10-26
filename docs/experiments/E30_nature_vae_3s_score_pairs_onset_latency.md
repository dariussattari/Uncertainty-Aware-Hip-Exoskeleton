# E30: VAE on 3 s + rhythm input, score pairs across models, and onset latency

| Field | Value |
|---|---|
| ID | E30 |
| Date | 2026-10-13 to 2026-10-14 |
| Dataset | Nature 2024 |
| Status | Complete. Two conclusions later corrected by E31 |
| Sources | 12OCT-15OCT Verbose.docx §12.0; README "VAE on 3 s + spectral input" bullet |
| Code / notebooks | `ml/nature/awareness/train_vae.py --inputs raw_3s_spectral`; `notebooks/nature/model_exploration/onboard/05_Novelty_Metric_comparison.ipynb`, sections 7.4 and 8; `02_vae_latent_spaces.ipynb` (3-D pages) |

## Hypothesis

Three hypotheses:

1. The 1 s-only VAE grid was a planning choice, not a finding. On the 3 s + rhythm input (where the Gaussian-NLL AE was
   best, E26), a VAE should match or beat the control, and the KL effect (E29) should carry over.
2. Scores from *different* models, a short-window prior distance and a long-window density, combine better than one
   model's scores.
3. Per-window AUROC understates what a controller sees. **Onset latency** (how soon after a movement starts the alarm
   fires) is the controller-relevant metric.

## Method

- The VAE was generalized to 23 × 600 windows (same 4 × 8 latent). Trained β 1, 32, 64, 128 (fixed) and β 1 (learned),
  three seeds each: 15 runs of 2–8 min. The first passed jobstats on 1 core, 1.1 GB and 71% GPU.
- Scored as in E29. Logistic regression on pairs of scores from different models, aligned window by window.
- Onset latency: bouts of consecutive in-task windows of at least 2 s; time to first alarm at the validation threshold;
  share of movements never flagged.

## Result

| 3 s + rhythm VAE | Active dims | Best score | AUROC unfamiliar | AUROC dangerous | False alarms |
|---|---|---|---|---|---|
| β 1, fixed | 32 | LOF, raw | 0.910 ± 0.030 | 0.934 | 1.6% |
| β 32, fixed | 18 | Mahalanobis | 0.846 | 0.968 | 3.1% |
| β 64, fixed | 13 | LOF, active dims | 0.864 | 0.907 | 3.4% |
| β 128, fixed | 14 | LOF, active dims | 0.859 | 0.903 | 2.9% |
| β 1, learned | 32 | negative ELBO | 0.876 | **0.977** | 3.9% |

- On 3 s input β prunes less (13–18 dimensions), and the KL reads *upward* at only 0.69–0.75. The best 3 s VAE only
  matches the 3 s control (0.910 against 0.914).
- With the learned variance the likelihood finally works (negative ELBO 0.876; dangerous 0.977), but its threshold
  transfers worst between people (3.9% false alarms on a 1% target).

| Pair of scores (logistic regression) | AUROC unfamiliar | AUROC dangerous |
|---|---|---|
| **1 s VAE KL (β 128) + 3 s Gaussian-NLL LOF** | **0.950 ± 0.003** | 0.995 |
| 1 s VAE KL + 1 s Gaussian-NLL LOF | 0.933 | **0.998** |
| 3 s VAE KL + 3 s Gaussian-NLL LOF | 0.914 | 0.970 |
| 1 s VAE KL + 3 s VAE KL | 0.909 | 0.994 |

- **Onset latency:**
  - The 1 s VAE KL flags **every** dangerous bout, half within 0.17 s and all within 0.9 s (ball toss 0.0 s;
    sit-to-stand and tug of war 0.1 s median).
  - The 1 s Gaussian-NLL AE flags 87% of dangerous bouts, half only after 2.3 s (tug of war after 7.4 s).
  - 3 s models alarm on 11–20% of *familiar* bouts, against 5–6% for 1 s models.
- Per movement, recall far exceeds per-window recall: the 1 s control flags 34% of unfamiliar windows but 73% of
  unfamiliar movements.

## Conclusion

At this point the best single alarm was the 1 s VAE KL at β 128: complete and immediate on dangerous tasks. The best
regime-classifier input was that KL together with a 3 s latent's LOF (0.950).

Two interpretations were **corrected in E31**:

1. The KL effect *does* carry over to 3 s windows. It was the rhythm features that removed it.
2. The 3 s models' familiar false switches came from the rhythm features, not from windows straddling a transition.

## Caveats

The pair classifiers saw the unfamiliar tasks. With 13 dangerous bouts per run, per-bout shares move in steps of about
8 points.
