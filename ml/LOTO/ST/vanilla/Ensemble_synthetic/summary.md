# Experiment 4: summed pairwise channel correlation (the reference work's own target) — results (test split)

Target: `correlation`. Label-free: computed from the input window, so it is available at run time, unlike gait phase.

## Headline

| metric | ours | paper (synthetic target) | comparable? |
|---|---|---|---|
| J-statistic | **-5.5** | — | yes |
| AUROC | **0.722** | — | yes |
| accuracy | 93.9% | — | no |
| recall (OOD caught) | 0.0% | — | no |
| specificity (ID kept) | 94.5% | — | no |
| precision | 0.0% | — | no |
| F1 | 0.0% | 62.5% | no |
| ECE | 0.063 | — | partly |
| Brier | 0.052 | — | partly |

Test set is **0.7% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 1.6% | 1.29e-02 |
| ramp ascent (RA) | ID | 18,504 | 0.5% | 1.24e-02 |
| ramp descent (RD) | ID | 14,784 | 1.8% | 1.77e-02 |
| stair ascent (SA) | ID | 10,273 | 4.4% | 1.94e-02 |
| stair descent (SD) | ID | 10,155 | 29.9% | 3.48e-02 |
| transitions (TR) | ID | 1,994 | 14.4% | 2.13e-02 |
| standing (ST) | OOD | 621 | 0.0% | 2.31e-02 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 6–33 (mean 21.1).
- Stage 2: retrained on all 12 subjects for 21 epochs.
- Threshold: 4.933e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.12674.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss