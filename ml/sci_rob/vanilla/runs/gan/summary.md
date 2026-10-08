# TCN GAN — results (test split)

## Headline

| metric | GAN | paper (GAN) | comparable? |
|---|---|---|---|
| J-statistic | **10.517** | 63.0 | yes |
| AUROC | **0.727** | — | yes |
| accuracy | **75.597** | 89.4 | no |
| recall (OOD caught) | **10.832** | 68.7 | no |
| specificity (ID kept) | **99.685** | — | no |
| precision | **92.754** | 74.9 | no |
| F1 | **19.398** | 71.5 | no |
| ECE | 0.071 | 0.06 | partly |
| Brier | 0.173 | 0.09 | partly |

Test set is **27.1% OOD** against the paper's 80.1%, so accuracy, precision and F1 move with class balance and only J-statistic and AUROC compare directly.

## Adversarial health

The discriminator *is* the detector here, so this section decides whether the numbers above mean anything. Its training objective is separating real windows from generated ones, and every out-of-distribution window is still real hip data — a discriminator that wins outright scores every real window near zero and detects nothing.

- Final `D(real)` = **0.626**, `D(fake)` = **0.534** (a healthy run keeps these within roughly 0.4–0.7 of each other, not at 1 and 0).
- `Psi` on the test split: median 0.31783, IQR 0.17012, range [0.0324, 0.7728].
- 0.0% of windows score below 1e-3; 51,710 distinct filtered values.
- Verdict: **the score retains usable spread**.

## Per-task

| task | kind | windows | flagged | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 0.7% | 0.29861 |
| ramp ascent (RA) | ID | 18,504 | 0.0% | 0.25078 |
| ramp descent (RD) | ID | 14,784 | 0.0% | 0.42095 |
| stair ascent (SA) | OOD | 10,273 | 0.1% | 0.30420 |
| stair descent (SD) | OOD | 10,155 | 9.2% | 0.48508 |
| transitions (TR) | OOD | 1,994 | 46.7% | 0.53252 |
| standing (ST) | OOD | 621 | 100.0% | 0.75759 |

## Training

- 500 epochs, **fixed** — Table IV specifies 500 with no early stopping, because a GAN has no validation signal that reliably identifies a best epoch.
- 5 generator steps per discriminator step (the paper's ratio, and deliberately the reverse of usual GAN practice — it is what keeps the discriminator weak enough to remain a novelty detector).
- Batch 256, lr G 2.0e-04 / D 5.0e-05, exponential decay 0.99 per epoch.
- Trained on 79,156 windows — the Step-20 subsample (every second stored window). Evaluation scores every test window, unsubsampled, so the cross-model comparison rests on one identical test set.
- Validation participants: AB18, AB28.
- Threshold 0.56893 (99.5th percentile of training Psi; training median 0.30841, IQR 0.14151).
- Generator output scale: mean per-channel sd 0.592 against the real data's 1.025 (both in standardized units, so 1.0 is the target).

## Figures

- `gan_psi_by_task`, `gan_roc`, `gan_detection_by_task` — the paper's reporting
- `gan_training_curve` — losses, the D(real)/D(fake) balance, and the spread of Psi
- `gan_generated` — generator output against real windows, and per-channel scale
- `gan_score_distribution` — ID vs OOD, and the saturation check
- `gan_trial_scores` — one labelled point per trial (303 trials)