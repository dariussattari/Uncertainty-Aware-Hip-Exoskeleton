# Convolutional autoencoder — results (test split)

## Headline

| metric | LOF on latent | reconstruction error | paper (AE) | comparable? |
|---|---|---|---|---|
| J-statistic | **-1.684** | -0.498 | 58.5 | yes |
| AUROC | **0.712** | 0.365 | — | yes |
| accuracy | **86.123** | 87.476 | 69.8 | no |
| recall (OOD caught) | **0.409** | 0.000 | 94.7 | no |
| specificity (ID kept) | **97.907** | 99.502 | — | no |
| precision | **2.615** | 0.000 | 40.0 | no |
| F1 | **0.707** | 0.000 | 55.7 | no |
| ECE | 0.040 | — | 0.26 | partly |
| Brier | 0.116 | — | 0.18 | partly |

Test set is **12.1% OOD** against the paper's 80.1%, so accuracy, precision and F1 move with class balance and only J-statistic and AUROC compare directly.

## Per-task

| task | kind | windows | flagged (LOF) | flagged (recon) | median Psi |
|---|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 0.6% | 1.3% | 1.065 |
| ramp ascent (RA) | ID | 18,504 | 0.0% | 0.0% | 1.077 |
| ramp descent (RD) | ID | 14,784 | 0.0% | 0.0% | 1.055 |
| stair ascent (SA) | OOD | 10,273 | 0.4% | 0.0% | 1.130 |
| stair descent (SD) | ID | 10,155 | 0.1% | 0.0% | 1.078 |
| transitions (TR) | ID | 1,994 | 37.7% | 0.0% | 1.247 |
| standing (ST) | ID | 621 | 100.0% | 0.0% | 5.536 |

## Training

- Single stage, 34 epochs run, best at epoch 28 (patience 5).
- Validation participants: AB18, AB28.
- Learning rate 0.0030 (declared deviation; Table IV specifies 0.064, which collapses).
- Latent dimension 32.
- Threshold 1.5435 (99.5th percentile of training Psi).

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