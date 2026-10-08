# Experiment 4: summed pairwise channel correlation (the reference work's own target) — results (test split)

Target: `correlation`. Label-free: computed from the input window, so it is available at run time, unlike gait phase.

## Headline

| metric | ours | paper (synthetic target) | comparable? |
|---|---|---|---|
| J-statistic | **45.9** | — | yes |
| AUROC | **0.884** | — | yes |
| accuracy | 81.5% | — | no |
| recall (OOD caught) | 48.5% | — | no |
| specificity (ID kept) | 97.3% | — | no |
| precision | 89.8% | — | no |
| F1 | 63.0% | 62.5% | no |
| ECE | 0.075 | — | partly |
| Brier | 0.131 | — | partly |

Test set is **32.4% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 3.1% | 1.21e-02 |
| ramp ascent (RA) | ID | 18,504 | 0.0% | 9.52e-03 |
| ramp descent (RD) | OOD | 14,784 | 35.1% | 2.87e-02 |
| stair ascent (SA) | ID | 10,273 | 6.2% | 1.73e-02 |
| stair descent (SD) | OOD | 10,155 | 64.7% | 4.47e-02 |
| transitions (TR) | OOD | 1,994 | 49.6% | 3.70e-02 |
| standing (ST) | OOD | 621 | 100.0% | 1.23e-01 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 10–27 (mean 19.9).
- Stage 2: retrained on all 12 subjects for 20 epochs.
- Threshold: 3.729e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.11871.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss