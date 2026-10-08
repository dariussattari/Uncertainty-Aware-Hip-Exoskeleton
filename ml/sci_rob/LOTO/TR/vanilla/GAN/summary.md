# TCN GAN — results (test split)

## Headline

| metric | GAN | paper (GAN) | comparable? |
|---|---|---|---|
| J-statistic | **36.263** | 63.0 | yes |
| AUROC | **0.777** | — | yes |
| accuracy | **97.792** | 89.4 | no |
| recall (OOD caught) | **37.011** | 68.7 | no |
| specificity (ID kept) | **99.252** | — | no |
| precision | **54.305** | 74.9 | no |
| F1 | **44.020** | 71.5 | no |
| ECE | 0.012 | 0.06 | partly |
| Brier | 0.019 | 0.09 | partly |

Test set is **2.3% OOD** against the paper's 80.1%, so accuracy, precision and F1 move with class balance and only J-statistic and AUROC compare directly.

## Adversarial health

The discriminator *is* the detector here, so this section decides whether the numbers above mean anything. Its training objective is separating real windows from generated ones, and every out-of-distribution window is still real hip data — a discriminator that wins outright scores every real window near zero and detects nothing.

- Final `D(real)` = **0.653**, `D(fake)` = **0.227** (a healthy run keeps these within roughly 0.4–0.7 of each other, not at 1 and 0).
- `Psi` on the test split: median 0.30731, IQR 0.15345, range [0.0030, 0.7188].
- 0.0% of windows score below 1e-3; 51,464 distinct filtered values.
- Verdict: **the score retains usable spread**.

## Per-task

| task | kind | windows | flagged | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 0.0% | 0.27512 |
| ramp ascent (RA) | ID | 18,504 | 0.0% | 0.25392 |
| ramp descent (RD) | ID | 14,784 | 0.0% | 0.39586 |
| stair ascent (SA) | ID | 10,273 | 0.0% | 0.31659 |
| stair descent (SD) | ID | 10,155 | 0.0% | 0.44653 |
| transitions (TR) | OOD | 1,994 | 37.0% | 0.52565 |
| standing (ST) | ID | 621 | 100.0% | 0.70134 |

## Training

- 500 epochs, **fixed** — Table IV specifies 500 with no early stopping, because a GAN has no validation signal that reliably identifies a best epoch.
- 5 generator steps per discriminator step (the paper's ratio, and deliberately the reverse of usual GAN practice — it is what keeps the discriminator weak enough to remain a novelty detector).
- Batch 256, lr G 2.0e-04 / D 5.0e-05, exponential decay 0.99 per epoch.
- Trained on 103,310 windows — the Step-20 subsample (every second stored window). Evaluation scores every test window, unsubsampled, so the cross-model comparison rests on one identical test set.
- Validation participants: AB18, AB28.
- Threshold 0.64922 (99.5th percentile of training Psi; training median 0.30797, IQR 0.13880).
- Generator output scale: mean per-channel sd 0.582 against the real data's 0.994 (both in standardized units, so 1.0 is the target).

## Figures

- `gan_psi_by_task`, `gan_roc`, `gan_detection_by_task` — the paper's reporting
- `gan_training_curve` — losses, the D(real)/D(fake) balance, and the spread of Psi
- `gan_generated` — generator output against real windows, and per-channel scale
- `gan_score_distribution` — ID vs OOD, and the saturation check
- `gan_trial_scores` — one labelled point per trial (303 trials)