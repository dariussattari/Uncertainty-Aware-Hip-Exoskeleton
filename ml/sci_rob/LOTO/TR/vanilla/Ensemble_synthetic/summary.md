# Experiment 4: summed pairwise channel correlation (the reference work's own target) — results (test split)

Target: `correlation`. Label-free: computed from the input window, so it is available at run time, unlike gait phase.

## Headline

| metric | ours | paper (synthetic target) | comparable? |
|---|---|---|---|
| J-statistic | **42.1** | — | yes |
| AUROC | **0.775** | — | yes |
| accuracy | 94.1% | — | no |
| recall (OOD caught) | 46.8% | — | no |
| specificity (ID kept) | 95.3% | — | no |
| precision | 19.2% | — | no |
| F1 | 27.2% | 62.5% | no |
| ECE | 0.065 | — | partly |
| Brier | 0.050 | — | partly |

Test set is **2.3% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 2.5% | 1.45e-02 |
| ramp ascent (RA) | ID | 18,504 | 0.0% | 1.42e-02 |
| ramp descent (RD) | ID | 14,784 | 3.0% | 1.64e-02 |
| stair ascent (SA) | ID | 10,273 | 2.9% | 1.99e-02 |
| stair descent (SD) | ID | 10,155 | 18.5% | 2.88e-02 |
| transitions (TR) | OOD | 1,994 | 46.8% | 4.11e-02 |
| standing (ST) | ID | 621 | 100.0% | 1.01e-01 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 11–36 (mean 20.1).
- Stage 2: retrained on all 12 subjects for 20 epochs.
- Threshold: 4.635e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.11218.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss