# Convolutional autoencoder — results (test split)

## Headline

| metric | LOF on latent | reconstruction error | paper (AE) | comparable? |
|---|---|---|---|---|
| J-statistic | **38.716** | -0.500 | 58.5 | yes |
| AUROC | **0.782** | 0.312 | — | yes |
| accuracy | **80.480** | 72.526 | 69.8 | no |
| recall (OOD caught) | **45.064** | 0.000 | 94.7 | no |
| specificity (ID kept) | **93.652** | 99.500 | — | no |
| precision | **72.529** | 0.000 | 40.0 | no |
| F1 | **55.589** | 0.000 | 55.7 | no |
| ECE | 0.042 | — | 0.26 | partly |
| Brier | 0.147 | — | 0.18 | partly |

Test set is **27.1% OOD** against the paper's 80.1%, so accuracy, precision and F1 move with class balance and only J-statistic and AUROC compare directly.

## Per-task

| task | kind | windows | flagged (LOF) | flagged (recon) | median Psi |
|---|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 4.6% | 1.1% | 1.077 |
| ramp ascent (RA) | ID | 18,504 | 6.6% | 0.0% | 1.105 |
| ramp descent (RD) | ID | 14,784 | 9.3% | 0.0% | 1.073 |
| stair ascent (SA) | OOD | 10,273 | 27.0% | 0.0% | 1.136 |
| stair descent (SD) | OOD | 10,155 | 57.6% | 0.0% | 1.284 |
| transitions (TR) | OOD | 1,994 | 57.3% | 0.0% | 1.306 |
| standing (ST) | OOD | 621 | 100.0% | 0.0% | 1.415 |

## Training

- Single stage, 42 epochs run, best at epoch 36 (patience 5).
- Validation participants: AB18, AB28.
- Learning rate 0.0030 (declared deviation; Table IV specifies 0.064, which collapses).
- Latent dimension 32.
- Threshold 1.2427 (99.5th percentile of training Psi).

## Latent space

- 303 trials, 84,999 windows projected.
- `ae_latent_space` — 2-D projection by mode, participant and score.
- `ae_trial_centroids` — one labelled point per trial (the traceable view).
- `ae_latent_dims` — which latent dimensions separate ID from OOD.

## Figures

- `ae_psi_by_task`, `ae_roc`, `ae_detection_by_task` — the paper's reporting
- `ae_training_curve` — single-stage training
- `ae_reconstructions` — per-channel input vs reconstruction
- `ae_score_comparison` — LOF against the paper's rejected reconstruction error