# Experiment 5: hip angle, both legs, 40 samples (200 ms) ahead — results (test split)

Target: `forecast_angle` at a horizon of 40 samples (200 ms). Label-free: computed from the input window, so it is available at run time, unlike gait phase.

## Headline

| metric | ours | no paper row (extension) | comparable? |
|---|---|---|---|
| J-statistic | **1.5** | — | yes |
| AUROC | **0.649** | — | yes |
| accuracy | 95.4% | — | no |
| recall (OOD caught) | 4.0% | — | no |
| specificity (ID kept) | 97.6% | — | no |
| precision | 3.7% | — | no |
| F1 | 3.9% | — | no |
| ECE | 0.032 | — | partly |
| Brier | 0.040 | — | partly |

Test set is **2.3% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 2.0% | 5.74e-03 |
| ramp ascent (RA) | ID | 18,504 | 0.0% | 7.22e-03 |
| ramp descent (RD) | ID | 14,784 | 0.2% | 6.17e-03 |
| stair ascent (SA) | ID | 10,273 | 4.7% | 1.09e-02 |
| stair descent (SD) | ID | 10,155 | 9.4% | 1.10e-02 |
| transitions (TR) | OOD | 1,994 | 4.0% | 9.02e-03 |
| standing (ST) | ID | 621 | 0.0% | 1.08e-02 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 13–32 (mean 23.6).
- Stage 2: retrained on all 12 subjects for 24 epochs.
- Threshold: 2.177e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.04579.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss