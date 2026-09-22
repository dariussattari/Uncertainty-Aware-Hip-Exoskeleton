# TCN GAN — results (test split)

## Headline

| metric | GAN | paper (GAN) | comparable? |
|---|---|---|---|
| J-statistic | **-1.440** | 63.0 | yes |
| AUROC | **0.982** | — | yes |
| accuracy | **97.840** | 89.4 | no |
| recall (OOD caught) | **0.000** | 68.7 | no |
| specificity (ID kept) | **98.560** | — | no |
| precision | **0.000** | 74.9 | no |
| F1 | **0.000** | 71.5 | no |
| ECE | 0.018 | 0.06 | partly |
| Brier | 0.012 | 0.09 | partly |

Test set is **0.7% OOD** against the paper's 80.1%, so accuracy, precision and F1 move with class balance and only J-statistic and AUROC compare directly.

## Adversarial health

The discriminator *is* the detector here, so this section decides whether the numbers above mean anything. Its training objective is separating real windows from generated ones, and every out-of-distribution window is still real hip data — a discriminator that wins outright scores every real window near zero and detects nothing.

- Final `D(real)` = **0.630**, `D(fake)` = **0.628** (a healthy run keeps these within roughly 0.4–0.7 of each other, not at 1 and 0).
- `Psi` on the test split: median 0.30681, IQR 0.18262, range [0.0007, 0.6829].
- 0.0% of windows score below 1e-3; 47,358 distinct filtered values.
- Verdict: **the score retains usable spread**.

## Per-task

| task | kind | windows | flagged | median Psi |
|---|---|---|---|---|
| level ground (LG) | ID | 28,668 | 0.0% | 0.27569 |
| ramp ascent (RA) | ID | 18,504 | 0.0% | 0.23785 |
| ramp descent (RD) | ID | 14,784 | 0.0% | 0.43593 |
| stair ascent (SA) | ID | 10,273 | 0.0% | 0.31811 |
| stair descent (SD) | ID | 10,155 | 9.0% | 0.49788 |
| transitions (TR) | ID | 1,994 | 14.9% | 0.49823 |
| standing (ST) | OOD | 621 | 0.0% | 0.60099 |

## Training

- 500 epochs, **fixed** — Table IV specifies 500 with no early stopping, because a GAN has no validation signal that reliably identifies a best epoch.
- 5 generator steps per discriminator step (the paper's ratio, and deliberately the reverse of usual GAN practice — it is what keeps the discriminator weak enough to remain a novelty detector).
- Batch 256, lr G 2.0e-04 / D 5.0e-05, exponential decay 0.99 per epoch.
- Trained on 104,814 windows — the Step-20 subsample (every second stored window). Evaluation scores every test window, unsubsampled, so the cross-model comparison rests on one identical test set.
- Validation participants: AB18, AB28.
- Threshold 0.60691 (99.5th percentile of training Psi; training median 0.31483, IQR 0.16212).
- Generator output scale: mean per-channel sd 0.742 against the real data's 0.968 (both in standardized units, so 1.0 is the target).

## Figures

- `gan_psi_by_task`, `gan_roc`, `gan_detection_by_task` — the paper's reporting
- `gan_training_curve` — losses, the D(real)/D(fake) balance, and the spread of Psi
- `gan_generated` — generator output against real windows, and per-channel scale
- `gan_score_distribution` — ID vs OOD, and the saturation check
- `gan_trial_scores` — one labelled point per trial (303 trials)