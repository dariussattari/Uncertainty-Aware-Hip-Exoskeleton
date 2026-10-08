# Convolutional autoencoder — results (test split)

## Headline

| metric | LOF on latent | reconstruction error | paper (AE) | comparable? |
|---|---|---|---|---|
| J-statistic | **80.540** | -0.501 | 58.5 | yes |
| AUROC | **0.976** | 0.499 | — | yes |
| accuracy | **93.151** | 67.244 | 69.8 | no |
| recall (OOD caught) | **82.079** | 0.000 | 94.7 | no |
| specificity (ID kept) | **98.461** | 99.499 | — | no |
| precision | **96.238** | 0.000 | 40.0 | no |
| F1 | **88.596** | 0.000 | 55.7 | no |
| ECE | 0.044 | — | 0.26 | partly |
| Brier | 0.051 | — | 0.18 | partly |

Test set is **32.4% OOD** against the paper's 80.1%, so accuracy, precision and F1 move with class balance and only J-statistic and AUROC compare directly.

## Per-task

| task | kind | windows | flagged (LOF) | flagged (recon) | median Psi |
|---|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 2.0% | 1.0% | 1.116 |
| ramp ascent (RA) | ID | 18,504 | 0.7% | 0.0% | 1.120 |
| ramp descent (RD) | OOD | 14,784 | 83.3% | 0.0% | 1.476 |
| stair ascent (SA) | ID | 10,273 | 1.8% | 0.0% | 1.091 |
| stair descent (SD) | OOD | 10,155 | 84.0% | 0.0% | 1.630 |
| transitions (TR) | OOD | 1,994 | 57.3% | 0.0% | 1.408 |
| standing (ST) | OOD | 621 | 100.0% | 0.0% | 1.847 |

## Training

- Single stage, 21 epochs run, best at epoch 15 (patience 5).
- Validation participants: AB18, AB28.
- Learning rate 0.0030 (declared deviation; Table IV specifies 0.064, which collapses).
- Latent dimension 32.
- Threshold 1.3432 (99.5th percentile of training Psi).

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