# Vanilla gait-phase ensemble — results (test split)

## Headline

| metric | ours | paper (ankle) | comparable? |
|---|---|---|---|
| J-statistic | **37.5** | 92.3 | yes |
| AUROC | **0.859** | 0.993 | yes |
| accuracy | 81.4% | 96.1% | no |
| recall (OOD caught) | 39.4% | 96.2% | no |
| specificity (ID kept) | 98.1% | — | no |
| precision | 89.4% | 93.0% | no |
| F1 | 54.7% | 90.3% | no |
| ECE | 0.048 | 0.03 | partly |
| Brier | 0.133 | 0.03 | partly |

Test set is **28.4% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 26,753 | 1.8% | 3.37e-04 |
| ramp ascent (RA) | ID | 17,354 | 0.5% | 4.53e-04 |
| ramp descent (RD) | ID | 13,938 | 3.8% | 5.01e-04 |
| stair ascent (SA) | OOD | 10,273 | 6.6% | 7.25e-04 |
| stair descent (SD) | OOD | 10,155 | 66.8% | 2.73e-03 |
| transitions (TR) | OOD | 1,994 | 49.9% | 1.77e-03 |
| standing (ST) | OOD | 621 | 100.0% | 1.44e-02 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 7–30 (mean 17.0) — the paper found 13 on ankle data.
- Stage 2: retrained on all 12 subjects for 17 epochs.
- Threshold: 1.817e-03 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.01155.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss