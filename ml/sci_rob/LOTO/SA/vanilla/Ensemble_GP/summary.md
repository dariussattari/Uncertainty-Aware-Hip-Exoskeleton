# Vanilla gait-phase ensemble — results (test split)

Target: `gait_phase`

## Headline

| metric | ours | paper (gait phase) | comparable? |
|---|---|---|---|
| J-statistic | **-1.8** | 92.3 | yes |
| AUROC | **0.698** | 0.993 | yes |
| accuracy | 89.5% | 96.1% | no |
| recall (OOD caught) | 0.5% | 96.2% | no |
| specificity (ID kept) | 97.6% | — | no |
| precision | 2.0% | 93.0% | no |
| F1 | 0.8% | 90.3% | no |
| ECE | 0.037 | 0.03 | partly |
| Brier | 0.091 | 0.03 | partly |

Test set is **8.4% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 26,753 | 0.1% | 6.80e-04 |
| ramp ascent (RA) | ID | 17,354 | 0.0% | 8.41e-04 |
| ramp descent (RD) | ID | 13,938 | 0.0% | 1.15e-03 |
| stair ascent (SA) | OOD | 6,053 | 0.5% | 1.63e-03 |
| stair descent (SD) | ID | 5,841 | 26.1% | 1.73e-02 |
| transitions (TR) | ID | 1,889 | 0.8% | 1.31e-03 |
| standing (ST) | ID | 600 | 0.0% | 1.25e-03 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 2–29 (mean 13.8) — the paper found 13 on ankle data.
- Stage 2: retrained on all 12 subjects for 14 epochs.
- Threshold: 2.550e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.01784.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss