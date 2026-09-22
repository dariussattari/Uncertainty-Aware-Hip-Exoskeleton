# Probabilistic ensemble — results (test split)

Experiment 5's target and trunk, trained under Gaussian negative log-likelihood so the uncertainty splits into an aleatoric and an epistemic part.

## Which component detects

| score | AUROC | AUROC (unfiltered) | J | best J | recall | specificity |
|---|---|---|---|---|---|---|
| epistemic | **0.480** | 0.471 | -1.56 | 41.30 | 0.00 | 98.44 |
| aleatoric | **0.029** | 0.059 | -0.07 | 1.71 | 0.00 | 99.93 |
| total | **0.085** | 0.118 | -0.23 | 5.88 | 0.00 | 99.77 |
| ratio | **0.924** | 0.838 | -2.74 | 91.11 | 0.00 | 97.26 |

Experiment 5, the same target scored by branch variance alone, reached AUROC 0.896 and J 47.64. The epistemic component here is that same quantity by construction, so the difference of -0.416 AUROC is a verification rather than a comparison.

## Calibration

- Held-out `Var[z]` averaged over the twelve LOSO folds: **1.850** (target 1.000).
- On test in-distribution windows: median squared z-score **1.279**, mean 5.686, 90th percentile 4.177, 99th 11.156.
- The mean is not the number to read: the worst 0.1% of windows carry **56.8%** of the total. The median says the variance head is calibrated for typical windows; the mean says it has a heavy tail of confident errors. Both are true.
- Mean predicted sigma 0.1648; realised RMSE 0.2436.
- Log-variance on a clamp bound for 0.05% of entries.

## Quadrant occupancy

| task | kind | n | low both | high alea only | high epi only | high both |
|---|---|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 97.9% | 0.0% | 2.1% | 0.1% |
| ramp ascent (RA) | ID | 18,504 | 100.0% | 0.0% | 0.0% | 0.0% |
| ramp descent (RD) | ID | 14,784 | 99.9% | 0.1% | 0.0% | 0.0% |
| stair ascent (SA) | ID | 10,273 | 97.7% | 0.2% | 2.0% | 0.1% |
| stair descent (SD) | ID | 10,155 | 95.6% | 0.0% | 4.4% | 0.0% |
| transitions (TR) | ID | 1,994 | 98.1% | 0.0% | 1.9% | 0.0% |
| standing (ST) | OOD | 621 | 100.0% | 0.0% | 0.0% | 0.0% |

## Training

- Stage 1: 12/12 folds, best epochs [5, 14, 3, 12, 14, 8, 13, 23, 7, 3, 27, 4], mean 11.1 (sd 7.4) -> 11 used.
- Stage 2: 11 epochs on all twelve participants, 3 of them squared-error warm-up.
- beta = 0.5, learning rate 0.001, batch 1024.
- Held-out validation NLL -0.00062, MSE 0.04673, Var[z] 1.831.

## Figures

- `scores_by_task` — all four candidates per task
- `roc_compare` — every score on one ROC, against Experiments 1 and 5
- `detection_by_task` — per-task detection at the calibrated threshold
- `calibration` — Var[z] per fold and through training (target exactly 1)
- `alea_epi_plane` — the two-dimensional space and quadrant occupancy
- `sigma_by_task` — intrinsic uncertainty against disagreement
- `loso_and_training` — fold spread and the stage-2 curve