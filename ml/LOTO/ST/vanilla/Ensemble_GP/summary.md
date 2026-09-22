# Vanilla gait-phase ensemble — results (test split)

Target: `gait_phase`

## Headline

| metric | ours | paper (gait phase) | comparable? |
|---|---|---|---|
| J-statistic | **-2.1** | 92.3 | yes |
| AUROC | **0.236** | 0.993 | yes |
| accuracy | 97.1% | 96.1% | no |
| recall (OOD caught) | 0.0% | 96.2% | no |
| specificity (ID kept) | 97.9% | — | no |
| precision | 0.0% | 93.0% | no |
| F1 | 0.0% | 90.3% | no |
| ECE | 0.025 | 0.03 | partly |
| Brier | 0.026 | 0.03 | partly |

Test set is **0.8% OOD** against the paper's **80.1%**. Accuracy, precision and F1 all move with class balance, so only J-statistic and AUROC compare directly. Our pseudo-OOD (held-out ambulation modes) is also far closer to in-distribution than their sitting/jumping/lying-down, which makes this a strictly harder detection problem.

## Per-task

| task | kind | windows | flagged as OOD | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 26,753 | 0.1% | 8.15e-04 |
| ramp ascent (RA) | ID | 17,354 | 0.0% | 1.08e-03 |
| ramp descent (RD) | ID | 13,938 | 0.0% | 1.26e-03 |
| stair ascent (SA) | ID | 6,053 | 5.2% | 4.43e-03 |
| stair descent (SD) | ID | 5,841 | 19.7% | 1.44e-02 |
| transitions (TR) | ID | 1,889 | 1.2% | 1.15e-03 |
| standing (ST) | OOD | 600 | 0.0% | 7.22e-04 |

## Training

- Stage 1: LOSO over 12 subjects, best epoch 0–18 (mean 10.8) — the paper found 13 on ankle data.
- Stage 2: retrained on all 12 subjects for 11 epochs.
- Threshold: 3.013e-02 (99.5th percentile of training Psi).
- Held-out validation masked MSE: 0.02165.

## Figures

- `psi_by_task` — uncertainty distributions per task with the threshold (cf. their Fig. 4)
- `roc` — ROC curve and the operating point the threshold selects
- `detection_by_task` — flagged fraction per task
- `loso` — Stage 1 fold spread
- `training_curve` — Stage 2 loss