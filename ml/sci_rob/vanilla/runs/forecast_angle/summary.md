# Experiment 5: hip angle, both legs, 40 samples (200 ms) ahead — results (test split)

Target: `forecast_angle` at a horizon of 40 samples (200 ms). Label-free: computed from the input window, so it is available at run time, unlike gait phase.

## Headline

| metric | ours | no paper row (extension) | comparable? |
|---|---|---|---|
| J-statistic | **47.6** | — | yes |
| AUROC | **0.896** | — | yes |
| accuracy | 83.8% | — | no |
| recall (OOD caught) | 52.0% | — | no |
| specificity (ID kept) | 95.6% | — | no |
| precision | 81.5% | — | no |
| F1 | 63.5% | — | no |
| ECE | 0.056 | — | partly |
| Brier | 0.116 | — | partly |

Test set is **27.1% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 5.2% | 4.56e-03 |
| ramp ascent (RA) | ID | 18,504 | 3.7% | 5.57e-03 |
| ramp descent (RD) | ID | 14,784 | 3.6% | 5.23e-03 |
| stair ascent (SA) | OOD | 10,273 | 37.9% | 9.15e-03 |
| stair descent (SD) | OOD | 10,155 | 62.7% | 1.19e-02 |
| transitions (TR) | OOD | 1,994 | 55.6% | 1.28e-02 |
| standing (ST) | OOD | 621 | 100.0% | 3.16e-02 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 4–36 (mean 19.9).
- Stage 2: retrained on all 12 subjects for 20 epochs.
- Threshold: 1.015e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.04101.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss