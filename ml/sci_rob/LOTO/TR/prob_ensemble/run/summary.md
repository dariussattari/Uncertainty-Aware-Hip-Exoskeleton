# Probabilistic ensemble — results (test split)

Experiment 5's target and trunk, trained under Gaussian negative log-likelihood so the uncertainty splits into an aleatoric and an epistemic part.

## Which component detects

| score | AUROC | AUROC (unfiltered) | J | best J | recall | specificity |
|---|---|---|---|---|---|---|
| epistemic | **0.509** | 0.520 | 1.67 | 3.72 | 2.81 | 98.87 |
| aleatoric | **0.239** | 0.242 | -0.14 | 0.00 | 0.00 | 99.86 |
| total | **0.272** | 0.279 | -0.33 | 0.00 | 0.00 | 99.67 |
| ratio | **0.754** | 0.741 | 43.61 | 44.12 | 46.74 | 96.87 |

Experiment 5, the same target scored by branch variance alone, reached AUROC 0.896 and J 47.64. The epistemic component here is that same quantity by construction, so the difference of -0.387 AUROC is a verification rather than a comparison.

## Calibration

- Held-out `Var[z]` averaged over the twelve LOSO folds: **1.834** (target 1.000).
- On test in-distribution windows: median squared z-score **1.229**, mean 4.263, 90th percentile 3.860, 99th 9.915.
- The mean is not the number to read: the worst 0.1% of windows carry **48.4%** of the total. The median says the variance head is calibrated for typical windows; the mean says it has a heavy tail of confident errors. Both are true.
- Mean predicted sigma 0.1730; realised RMSE 0.2441.
- Log-variance on a clamp bound for 0.03% of entries.

## Quadrant occupancy

| task | kind | n | low both | high alea only | high epi only | high both |
|---|---|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 98.0% | 0.0% | 2.0% | 0.0% |
| ramp ascent (RA) | ID | 18,504 | 100.0% | 0.0% | 0.0% | 0.0% |
| ramp descent (RD) | ID | 14,784 | 99.9% | 0.1% | 0.0% | 0.0% |
| stair ascent (SA) | ID | 10,273 | 97.4% | 0.4% | 1.8% | 0.5% |
| stair descent (SD) | ID | 10,155 | 98.7% | 0.0% | 1.3% | 0.0% |
| transitions (TR) | OOD | 1,994 | 97.2% | 0.0% | 2.8% | 0.0% |
| standing (ST) | ID | 621 | 100.0% | 0.0% | 0.0% | 0.0% |

## Training

- Stage 1: 12/12 folds, best epochs [5, 10, 12, 9, 6, 21, 14, 21, 15, 4, 26, 5], mean 12.3 (sd 7.0) -> 12 used.
- Stage 2: 12 epochs on all twelve participants, 3 of them squared-error warm-up.
- beta = 0.5, learning rate 0.001, batch 1024.
- Held-out validation NLL -0.01049, MSE 0.04600, Var[z] 1.721.

## Figures

- `scores_by_task` — all four candidates per task
- `roc_compare` — every score on one ROC, against Experiments 1 and 5
- `detection_by_task` — per-task detection at the calibrated threshold
- `calibration` — Var[z] per fold and through training (target exactly 1)
- `alea_epi_plane` — the two-dimensional space and quadrant occupancy
- `sigma_by_task` — intrinsic uncertainty against disagreement
- `loso_and_training` — fold spread and the stage-2 curve