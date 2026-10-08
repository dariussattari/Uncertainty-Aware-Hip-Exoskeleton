# Probabilistic ensemble — results (test split)

Experiment 5's target and trunk, trained under Gaussian negative log-likelihood so the uncertainty splits into an aleatoric and an epistemic part.

## Which component detects

| score | AUROC | AUROC (unfiltered) | J | best J | recall | specificity |
|---|---|---|---|---|---|---|
| epistemic | **0.840** | 0.751 | 4.39 | 55.46 | 6.22 | 98.17 |
| aleatoric | **0.791** | 0.717 | 1.38 | 48.15 | 1.40 | 99.98 |
| total | **0.830** | 0.758 | 1.66 | 58.13 | 1.90 | 99.77 |
| ratio | **0.600** | 0.573 | 3.65 | 14.86 | 7.76 | 95.89 |

Experiment 5, the same target scored by branch variance alone, reached AUROC 0.896 and J 47.64. The epistemic component here is that same quantity by construction, so the difference of -0.056 AUROC is a verification rather than a comparison.

## Calibration

- Held-out `Var[z]` averaged over the twelve LOSO folds: **1.800** (target 1.000).
- On test in-distribution windows: median squared z-score **1.322**, mean 12.976, 90th percentile 4.183, 99th 11.206.
- The mean is not the number to read: the worst 0.1% of windows carry **67.1%** of the total. The median says the variance head is calibrated for typical windows; the mean says it has a heavy tail of confident errors. Both are true.
- Mean predicted sigma 0.1638; realised RMSE 0.2555.
- Log-variance on a clamp bound for 0.09% of entries.

## Quadrant occupancy

| task | kind | n | low both | high alea only | high epi only | high both |
|---|---|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 98.6% | 0.0% | 1.4% | 0.0% |
| ramp ascent (RA) | ID | 18,504 | 99.7% | 0.0% | 0.3% | 0.0% |
| ramp descent (RD) | ID | 14,784 | 99.8% | 0.0% | 0.2% | 0.0% |
| stair ascent (SA) | OOD | 10,273 | 93.1% | 0.7% | 5.5% | 0.7% |
| stair descent (SD) | ID | 10,155 | 91.3% | 0.0% | 8.6% | 0.1% |
| transitions (TR) | ID | 1,994 | 99.4% | 0.0% | 0.6% | 0.0% |
| standing (ST) | ID | 621 | 100.0% | 0.0% | 0.0% | 0.0% |

## Training

- Stage 1: 12/12 folds, best epochs [6, 9, 7, 6, 3, 12, 12, 15, 19, 3, 15, 4], mean 9.2 (sd 5.1) -> 9 used.
- Stage 2: 9 epochs on all twelve participants, 3 of them squared-error warm-up.
- beta = 0.5, learning rate 0.001, batch 1024.
- Held-out validation NLL -0.05965, MSE 0.04332, Var[z] 1.680.

## Figures

- `scores_by_task` — all four candidates per task
- `roc_compare` — every score on one ROC, against Experiments 1 and 5
- `detection_by_task` — per-task detection at the calibrated threshold
- `calibration` — Var[z] per fold and through training (target exactly 1)
- `alea_epi_plane` — the two-dimensional space and quadrant occupancy
- `sigma_by_task` — intrinsic uncertainty against disagreement
- `loso_and_training` — fold spread and the stage-2 curve