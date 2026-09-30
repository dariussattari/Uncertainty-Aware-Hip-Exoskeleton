# Vanilla gait-phase ensemble — results (test split)

Target: `gait_phase`

## Headline

| metric | ours | paper (gait phase) | comparable? |
|---|---|---|---|
| J-statistic | **2.5** | 92.3 | yes |
| AUROC | **0.878** | 0.993 | yes |
| accuracy | 69.8% | 96.1% | no |
| recall (OOD caught) | 3.0% | 96.2% | no |
| specificity (ID kept) | 99.5% | — | no |
| precision | 73.6% | 93.0% | no |
| F1 | 5.7% | 90.3% | no |
| ECE | 0.149 | 0.03 | partly |
| Brier | 0.195 | 0.03 | partly |

Test set is **30.7% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 26,753 | 0.1% | 7.41e-04 |
| ramp ascent (RA) | ID | 17,354 | 0.0% | 9.63e-04 |
| ramp descent (RD) | OOD | 13,938 | 0.0% | 4.21e-03 |
| stair ascent (SA) | ID | 6,053 | 3.6% | 3.79e-03 |
| stair descent (SD) | OOD | 5,841 | 10.6% | 3.27e-02 |
| transitions (TR) | OOD | 1,889 | 2.1% | 5.24e-03 |
| standing (ST) | OOD | 600 | 0.0% | 3.49e-02 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 3–30 (mean 16.0) — the paper found 13 on ankle data.
- Stage 2: retrained on all 12 subjects for 16 epochs.
- Threshold: 6.584e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.01920.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss