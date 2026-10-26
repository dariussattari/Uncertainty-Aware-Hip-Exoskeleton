# E06: TCN GAN discriminator, and the amplitude confound (Methods "Experiment 3")

| Field | Value |
|---|---|
| ID | E06 |
| Date | 2026-09-15 to 2026-09-16 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete |
| Sources | Methods_Experiments_Reproduction.docx §17–21; Live_Update.docx, "SEP 2 – 18" §2 (amplitude row) and §7 |
| Code / notebooks | `ml/common/models/` (GAN), `ml/sci_rob/vanilla/` (`training/gan.py`, `evaluation/gan.py`); `notebooks/sci_rob/model_exploration/03_gan.ipynb` |

## Hypothesis

A discriminator trained to separate real ID windows from generated ones assigns a lower probability of being real to
unfamiliar windows, so $1 - D(x)$ is a novelty score. This holds only while the discriminator stays uncertain, which
the reference's 5 generator steps per discriminator step are meant to ensure.

## Method

- **Setup:** as E03 and E05 (the same participants, OOD definition, causal median and 99.5th-percentile rule). Window
  step 20 for training, per the reference (every second stored window, 79,156 windows). The test set is scored in full
  (84,999 windows).
- **Generator:** latent 10 × 40 → ConvTranspose1d 10 → 10 → 12 → 16, kernel 3, LeakyReLU(0.2), BatchNorm on the
  hidden layers, upsampling split (5, 1, 1). 1,318 parameters.
- **Discriminator:** Conv1d 16 → 30 → 30, kernel 5, spectral norm, ReLU, dropout 0.2, global average pooling then a
  linear layer. 6,991 parameters.
- **Training:** 500 fixed epochs, batch 256, Adam β = (0.5, 0.999), lr 2e-4 (G) and 5e-5 (D), decay 0.99 per epoch,
  BCE loss, 1 D step per 5 G steps, 92.7 min. D(real), D(fake) and the IQR of the score were logged every epoch to
  catch saturation.
- **Amplitude diagnostic:** window amplitude = the mean over the 16 channels of each window's SD. Computed: its rank
  correlation with the score, AUROC within amplitude deciles, and raw amplitude used as a detector on its own.

## Result

- The training run was healthy: D(real) stayed between 0.457 and 0.663 for all 500 epochs, and the score IQR grew 12×.
- **AUROC 0.727**, J 10.52 (best achievable J 33.30), recall 10.8%, specificity 99.7%, precision 92.8%.
- Flagged per task: LG 0.68%, RA 0%, RD 0%; SA 0.08%, SD 9.2%, TR 46.7%, ST 100%.
- Amplitude diagnostics:

| Diagnostic | Value |
|---|---|
| Rank correlation, amplitude against score | −0.86 |
| AUROC, raw amplitude alone | **0.807** |
| AUROC, score within amplitude deciles | **0.504** (chance) |
| AUROC, quiet tasks only (ST, TR) | 0.856 |
| AUROC, stair ascent only | 0.515 |

- Mechanism: generator output SD is 0.592 against 1.025 for real data, so the cheapest rule for the discriminator is
  "loud = real, quiet = fake". Standing is flagged because it is motionless, not because it is novel.

## Conclusion

The GAN's score is a **monotone function of signal amplitude**, and a worse one than amplitude itself (0.727 against
0.807). It is last of the four reference architectures here, while the reference ranks it second. That ordering is the
only qualitative divergence from the reference, and it is best explained by this OOD set containing a loud, cyclic task
(stair ascent) that the reference's set lacked. Raw amplitude (AUROC 0.807) becomes a mandatory baseline for every later
score, and the same diagnostic was later applied to E08.

## Caveats

- Generator normalization was inferred, and it is the mechanism behind the finding. No sensitivity analysis was run.
- No hyperparameter search. Single seed.
- The 99.5th-percentile rule discards about two thirds of the usable discrimination for this score.
