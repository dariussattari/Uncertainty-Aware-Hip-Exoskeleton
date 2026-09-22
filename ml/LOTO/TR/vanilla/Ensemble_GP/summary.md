# Vanilla gait-phase ensemble — results (test split)

Target: `gait_phase`

## Headline

| metric | ours | paper (gait phase) | comparable? |
|---|---|---|---|
| J-statistic | **-0.5** | 92.3 | yes |
| AUROC | **0.607** | 0.993 | yes |
| accuracy | 96.1% | 96.1% | no |
| recall (OOD caught) | 0.8% | 96.2% | no |
| specificity (ID kept) | 98.7% | — | no |
| precision | 1.6% | 93.0% | no |
| F1 | 1.1% | 90.3% | no |
| ECE | 0.040 | 0.03 | partly |
| Brier | 0.036 | 0.03 | partly |

Test set is **2.6% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 26,753 | 0.1% | 7.68e-04 |
| ramp ascent (RA) | ID | 17,354 | 0.0% | 1.09e-03 |
| ramp descent (RD) | ID | 13,938 | 0.0% | 1.08e-03 |
| stair ascent (SA) | ID | 6,053 | 5.6% | 5.07e-03 |
| stair descent (SD) | ID | 5,841 | 9.9% | 1.14e-02 |
| transitions (TR) | OOD | 1,889 | 0.8% | 1.55e-03 |
| standing (ST) | ID | 600 | 0.0% | 1.07e-03 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 2–22 (mean 10.2) — the paper found 13 on ankle data.
- Stage 2: retrained on all 12 subjects for 10 epochs.
- Threshold: 3.332e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.02138.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss