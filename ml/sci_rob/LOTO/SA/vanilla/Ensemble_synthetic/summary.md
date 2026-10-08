# Experiment 4: summed pairwise channel correlation (the reference work's own target) — results (test split)

Target: `correlation`. Label-free: computed from the input window, so it is available at run time, unlike gait phase.

## Headline

| metric | ours | paper (synthetic target) | comparable? |
|---|---|---|---|
| J-statistic | **-0.2** | — | yes |
| AUROC | **0.626** | — | yes |
| accuracy | 85.2% | — | no |
| recall (OOD caught) | 3.3% | — | no |
| specificity (ID kept) | 96.5% | — | no |
| precision | 11.5% | — | no |
| F1 | 5.1% | 62.5% | no |
| ECE | 0.049 | — | partly |
| Brier | 0.120 | — | partly |

Test set is **12.1% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 2.2% | 1.58e-02 |
| ramp ascent (RA) | ID | 18,504 | 0.1% | 1.41e-02 |
| ramp descent (RD) | ID | 14,784 | 1.6% | 1.96e-02 |
| stair ascent (SA) | OOD | 10,273 | 3.3% | 2.38e-02 |
| stair descent (SD) | ID | 10,155 | 10.7% | 3.47e-02 |
| transitions (TR) | ID | 1,994 | 27.1% | 3.66e-02 |
| standing (ST) | ID | 621 | 15.5% | 5.66e-02 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 6–32 (mean 17.4).
- Stage 2: retrained on all 12 subjects for 17 epochs.
- Threshold: 6.213e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.14688.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss