# Convolutional autoencoder — results (test split)

## Headline

| metric | LOF on latent | reconstruction error | paper (AE) | comparable? |
|---|---|---|---|---|
| J-statistic | **98.258** | -0.500 | 58.5 | yes |
| AUROC | **0.991** | 0.002 | — | yes |
| accuracy | **98.271** | 98.773 | 69.8 | no |
| recall (OOD caught) | **100.000** | 0.000 | 94.7 | no |
| specificity (ID kept) | **98.258** | 99.500 | — | no |
| precision | **29.699** | 0.000 | 40.0 | no |
| F1 | **45.796** | 0.000 | 55.7 | no |
| ECE | 0.018 | — | 0.26 | partly |
| Brier | 0.016 | — | 0.18 | partly |

Test set is **0.7% OOD** against the paper's 80.1%, so accuracy, precision and F1 move with class balance and only J-statistic and AUROC compare directly.

## Per-task

| task | kind | windows | flagged (LOF) | flagged (recon) | median Psi |
|---|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 2.0% | 1.5% | 1.079 |
| ramp ascent (RA) | ID | 18,504 | 0.0% | 0.0% | 1.060 |
| ramp descent (RD) | ID | 14,784 | 0.2% | 0.0% | 1.058 |
| stair ascent (SA) | ID | 10,273 | 0.0% | 0.0% | 1.077 |
| stair descent (SD) | ID | 10,155 | 0.5% | 0.0% | 1.086 |
| transitions (TR) | ID | 1,994 | 41.3% | 0.0% | 1.224 |
| standing (ST) | OOD | 621 | 100.0% | 0.0% | 1.750 |

## Training

- Single stage, 46 epochs run, best at epoch 40 (patience 5).
- Validation participants: AB18, AB28.
- Learning rate 0.0030 (declared deviation; Table IV specifies 0.064, which collapses).
- Latent dimension 32.
- Threshold 1.4729 (99.5th percentile of training Psi).

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