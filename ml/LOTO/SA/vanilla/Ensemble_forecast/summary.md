# Experiment 5: hip angle, both legs, 40 samples (200 ms) ahead — results (test split)

Target: `forecast_angle` at a horizon of 40 samples (200 ms). Label-free: computed from the input window, so it is available at run time, unlike gait phase.

## Headline

| metric | ours | no paper row (extension) | comparable? |
|---|---|---|---|
| J-statistic | **9.8** | — | yes |
| AUROC | **0.827** | — | yes |
| accuracy | 86.3% | — | no |
| recall (OOD caught) | 13.5% | — | no |
| specificity (ID kept) | 96.3% | — | no |
| precision | 33.5% | — | no |
| F1 | 19.2% | — | no |
| ECE | 0.059 | — | partly |
| Brier | 0.101 | — | partly |

Test set is **12.1% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 2.5% | 4.71e-03 |
| ramp ascent (RA) | ID | 18,504 | 0.3% | 5.79e-03 |
| ramp descent (RD) | ID | 14,784 | 0.3% | 5.50e-03 |
| stair ascent (SA) | OOD | 10,273 | 13.5% | 1.01e-02 |
| stair descent (SD) | ID | 10,155 | 17.8% | 9.04e-03 |
| transitions (TR) | ID | 1,994 | 5.6% | 5.95e-03 |
| standing (ST) | ID | 621 | 0.0% | 5.50e-03 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 6–39 (mean 18.5).
- Stage 2: retrained on all 12 subjects for 18 epochs.
- Threshold: 1.553e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.04070.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss