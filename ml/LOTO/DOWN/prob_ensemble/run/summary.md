# Probabilistic ensemble — results (test split)

Experiment 5's target and trunk, trained under Gaussian negative log-likelihood so the uncertainty splits into an aleatoric and an epistemic part.

## Which component detects

| score | AUROC | AUROC (unfiltered) | J | best J | recall | specificity |
|---|---|---|---|---|---|---|
| epistemic | **0.859** | 0.803 | 23.86 | 58.69 | 24.67 | 99.19 |
| aleatoric | **0.643** | 0.605 | 0.07 | 28.13 | 0.14 | 99.93 |
| total | **0.771** | 0.734 | 0.81 | 47.38 | 1.18 | 99.63 |
| ratio | **0.865** | 0.775 | 33.38 | 56.19 | 35.42 | 97.97 |

Experiment 5, the same target scored by branch variance alone, reached AUROC 0.896 and J 47.64. The epistemic component here is that same quantity by construction, so the difference of -0.037 AUROC is a verification rather than a comparison.

## Calibration

- Held-out `Var[z]` averaged over the twelve LOSO folds: **1.966** (target 1.000).
- On test in-distribution windows: median squared z-score **1.364**, mean 7.207, 90th percentile 4.375, 99th 10.885.
- The mean is not the number to read: the worst 0.1% of windows carry **55.0%** of the total. The median says the variance head is calibrated for typical windows; the mean says it has a heavy tail of confident errors. Both are true.
- Mean predicted sigma 0.1534; realised RMSE 0.2464.
- Log-variance on a clamp bound for 0.06% of entries.

## Quadrant occupancy

| task | kind | n | low both | high alea only | high epi only | high both |
|---|---|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 99.2% | 0.0% | 0.8% | 0.0% |
| ramp ascent (RA) | ID | 18,504 | 100.0% | 0.0% | 0.0% | 0.0% |
| ramp descent (RD) | OOD | 14,784 | 92.5% | 0.0% | 7.4% | 0.1% |
| stair ascent (SA) | ID | 10,273 | 97.7% | 0.1% | 2.1% | 0.2% |
| stair descent (SD) | OOD | 10,155 | 58.2% | 0.0% | 41.5% | 0.3% |
| transitions (TR) | OOD | 1,994 | 58.1% | 0.0% | 41.9% | 0.0% |
| standing (ST) | OOD | 621 | 0.0% | 0.0% | 100.0% | 0.0% |

## Training

- Stage 1: 12/12 folds, best epochs [17, 10, 11, 3, 5, 27, 16, 18, 4, 5, 24, 6], mean 12.2 (sd 7.8) -> 12 used.
- Stage 2: 12 epochs on all twelve participants, 3 of them squared-error warm-up.
- beta = 0.5, learning rate 0.001, batch 1024.
- Held-out validation NLL +0.03135, MSE 0.04763, Var[z] 2.016.

## Figures

- `scores_by_task` — all four candidates per task
- `roc_compare` — every score on one ROC, against Experiments 1 and 5
- `detection_by_task` — per-task detection at the calibrated threshold
- `calibration` — Var[z] per fold and through training (target exactly 1)
- `alea_epi_plane` — the two-dimensional space and quadrant occupancy
- `sigma_by_task` — intrinsic uncertainty against disagreement
- `loso_and_training` — fold spread and the stage-2 curve