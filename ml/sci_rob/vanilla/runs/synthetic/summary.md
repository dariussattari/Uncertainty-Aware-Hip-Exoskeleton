# Experiment 4: summed pairwise channel correlation (the reference work's own target) — results (test split)

Target: `correlation`. Label-free: computed from the input window, so it is available at run time, unlike gait phase.

## Headline

| metric | ours | paper (synthetic target) | comparable? |
|---|---|---|---|
| J-statistic | **35.1** | — | yes |
| AUROC | **0.813** | — | yes |
| accuracy | 79.1% | — | no |
| recall (OOD caught) | 42.4% | — | no |
| specificity (ID kept) | 92.7% | — | no |
| precision | 68.4% | — | no |
| F1 | 52.4% | 62.5% | no |
| ECE | 0.036 | — | partly |
| Brier | 0.149 | — | partly |

Test set is **27.1% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 6.2% | 1.37e-02 |
| ramp ascent (RA) | ID | 18,504 | 1.8% | 1.26e-02 |
| ramp descent (RD) | ID | 14,784 | 16.4% | 1.85e-02 |
| stair ascent (SA) | OOD | 10,273 | 18.6% | 2.21e-02 |
| stair descent (SD) | OOD | 10,155 | 60.5% | 3.77e-02 |
| transitions (TR) | OOD | 1,994 | 55.0% | 4.29e-02 |
| standing (ST) | OOD | 621 | 100.0% | 3.88e-01 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 4–43 (mean 24.1).
- Stage 2: retrained on all 12 subjects for 24 epochs.
- Threshold: 3.329e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.08618.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss