# E24: Can the autoencoders flag unfamiliar movement? (latent LOF at the ankle paper's threshold)

| Field | Value |
|---|---|
| ID | E24 |
| Date | 2026-10-08 to 2026-10-10 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | Live_Update.docx, "OCT 5 – 10" §2, §7, §9 |
| Code / notebooks | `notebooks/nature/model_exploration/onboard/05_Novelty_Metric_comparison.ipynb` (then `05_LOF_comparison.ipynb`), `_novelty_scores.py` |

## Hypothesis

The exoskeleton should run its walking controller only on movement it knows. The hypothesis was that LOF on the latent
of an AE trained on familiar movement (normal, incline and weighted walking, and stairs) recognizes everything else,
and that the 3 s + rhythm recipe (E19) beats the ankle paper's 1 s raw input.

## Method

- The four models of E22 (plain and sparse, on 1 s raw and on 3 s + rhythm), three seeds each. Testing used 4 unseen
  people across all 26 tasks.
- Score: LOF on the 32-number latent (the ankle paper's score). The alarm fires above the 99.5th percentile of familiar
  training windows (the ankle paper's rule), after a 0.5 s causal median.
- Metrics: AUROC for all 22 unfamiliar tasks, AUROC for the 4 dangerous tasks (ball toss, step-ups, tug of war,
  sit-to-stand), share of unfamiliar windows flagged, and false alarms.

## Result

| Model | AUROC, unfamiliar | AUROC, dangerous | Unfamiliar flagged | False alarms |
|---|---|---|---|---|
| **Plain AE, 1 s** | **0.89** | **0.97** | 29% | 1.6% |
| Plain AE, 3 s + rhythm | 0.89 | 0.93 | 36% | 2.2% |
| Sparse AE, 1 s | 0.85 | 0.94 | 17% | 1.0% |
| Sparse AE, 3 s + rhythm | 0.79 | 0.82 | 28% | 2.4% |

- Exact figures (README): plain 1 s 0.887 (dangerous 0.968), plain 3 s + rhythm 0.893.
- The ranking is good, but the threshold is too cautious: only about a third of unfamiliar windows trigger it.
- **Quiet movement is the blind spot.** Standing poses and the ball toss look familiar to every model, the same problem
  the ankle paper had with standing.

## Conclusion

- Use the **plain AE with LOF**, starting with 1 s data. It is the best on dangerous tasks (0.97) and needs no extra 2 s
  of history.
- Next steps:
  - Tune the threshold on held-out people instead of using the fixed rule (E25).
  - Add a supervised classifier for safe against dangerous and for quiet movements (E34).
  - Optionally use the sparse alarm units as a second, simpler warning.
- Further leads: VAE, LSTM and Transformer variants, and perhaps a better metric than LOF (E28–E31).
