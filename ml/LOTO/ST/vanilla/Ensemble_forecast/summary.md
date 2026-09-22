# Experiment 5: hip angle, both legs, 40 samples (200 ms) ahead — results (test split)

Target: `forecast_angle` at a horizon of 40 samples (200 ms). Label-free: computed from the input window, so it is available at run time, unlike gait phase.

## Headline

| metric | ours | no paper row (extension) | comparable? |
|---|---|---|---|
| J-statistic | **-3.2** | — | yes |
| AUROC | **0.320** | — | yes |
| accuracy | 96.1% | — | no |
| recall (OOD caught) | 0.0% | — | no |
| specificity (ID kept) | 96.8% | — | no |
| precision | 0.0% | — | no |
| F1 | 0.0% | — | no |
| ECE | 0.038 | — | partly |
| Brier | 0.034 | — | partly |

Test set is **0.7% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 1.9% | 5.21e-03 |
| ramp ascent (RA) | ID | 18,504 | 0.0% | 6.78e-03 |
| ramp descent (RD) | ID | 14,784 | 0.3% | 5.79e-03 |
| stair ascent (SA) | ID | 10,273 | 3.0% | 1.07e-02 |
| stair descent (SD) | ID | 10,155 | 17.8% | 1.12e-02 |
| transitions (TR) | ID | 1,994 | 0.5% | 5.81e-03 |
| standing (ST) | OOD | 621 | 0.0% | 5.45e-03 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 8–38 (mean 20.0).
- Stage 2: retrained on all 12 subjects for 20 epochs.
- Threshold: 2.091e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.04560.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss