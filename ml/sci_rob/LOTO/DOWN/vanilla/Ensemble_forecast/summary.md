# Experiment 5: hip angle, both legs, 40 samples (200 ms) ahead — results (test split)

Target: `forecast_angle` at a horizon of 40 samples (200 ms). Label-free: computed from the input window, so it is available at run time, unlike gait phase.

## Headline

| metric | ours | no paper row (extension) | comparable? |
|---|---|---|---|
| J-statistic | **36.5** | — | yes |
| AUROC | **0.881** | — | yes |
| accuracy | 78.8% | — | no |
| recall (OOD caught) | 38.1% | — | no |
| specificity (ID kept) | 98.4% | — | no |
| precision | 91.9% | — | no |
| F1 | 53.9% | — | no |
| ECE | 0.091 | — | partly |
| Brier | 0.144 | — | partly |

Test set is **32.4% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 2.5% | 5.88e-03 |
| ramp ascent (RA) | ID | 18,504 | 0.0% | 6.64e-03 |
| ramp descent (RD) | OOD | 14,784 | 19.9% | 1.38e-02 |
| stair ascent (SA) | ID | 10,273 | 2.1% | 9.79e-03 |
| stair descent (SD) | OOD | 10,155 | 59.8% | 2.29e-02 |
| transitions (TR) | OOD | 1,994 | 43.4% | 1.45e-02 |
| standing (ST) | OOD | 621 | 100.0% | 4.57e-02 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 6–34 (mean 19.7).
- Stage 2: retrained on all 12 subjects for 20 epochs.
- Threshold: 2.004e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.04488.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss