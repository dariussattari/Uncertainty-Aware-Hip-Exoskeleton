# Convolutional autoencoder — results (test split)

## Headline

| metric | LOF on latent | reconstruction error | paper (AE) | comparable? |
|---|---|---|---|---|
| J-statistic | **43.504** | -0.496 | 58.5 | yes |
| AUROC | **0.876** | 0.119 | — | yes |
| accuracy | **97.215** | 97.169 | 69.8 | no |
| recall (OOD caught) | **45.035** | 0.000 | 94.7 | no |
| specificity (ID kept) | **98.469** | 99.504 | — | no |
| precision | **41.402** | 0.000 | 40.0 | no |
| F1 | **43.142** | 0.000 | 55.7 | no |
| ECE | 0.026 | — | 0.26 | partly |
| Brier | 0.026 | — | 0.18 | partly |

Test set is **2.3% OOD** against the paper's 80.1%, so accuracy, precision and F1 move with class balance and only J-statistic and AUROC compare directly.

## Per-task

| task | kind | windows | flagged (LOF) | flagged (recon) | median Psi |
|---|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 1.5% | 1.4% | 1.062 |
| ramp ascent (RA) | ID | 18,504 | 0.0% | 0.0% | 1.057 |
| ramp descent (RD) | ID | 14,784 | 0.3% | 0.0% | 1.052 |
| stair ascent (SA) | ID | 10,273 | 0.3% | 0.0% | 1.075 |
| stair descent (SD) | ID | 10,155 | 1.2% | 0.0% | 1.084 |
| transitions (TR) | OOD | 1,994 | 45.0% | 0.0% | 1.253 |
| standing (ST) | ID | 621 | 100.0% | 0.0% | 12.545 |

## Training

- Single stage, 21 epochs run, best at epoch 15 (patience 5).
- Validation participants: AB18, AB28.
- Learning rate 0.0030 (declared deviation; Table IV specifies 0.064, which collapses).
- Latent dimension 32.
- Threshold 1.3998 (99.5th percentile of training Psi).

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