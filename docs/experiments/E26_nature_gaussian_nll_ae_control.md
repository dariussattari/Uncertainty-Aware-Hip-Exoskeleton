# E26: Gaussian-NLL autoencoder control with a linear latent (plan step 3.2.2)

| Field | Value |
|---|---|
| ID | E26 |
| Date | 2026-10-13 (commit 248e43c) |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | 12OCT-15OCT Verbose.docx §3.2 (2) and §8.0; README "Gaussian-NLL control" bullet |
| Code / notebooks | `ml/nature/awareness/train_ae.py --variant gaussian` (`linear_latent` option in the shared AE); `notebooks/nature/model_exploration/onboard/05_Novelty_Metric_comparison.ipynb`, section 6.6; `01_ae_latent_spaces.ipynb`, section 6 |

## Hypothesis

The best model of the Science Robotics VAE study (E17) was a deterministic control: the AE encoder ending at its last
convolution (no BatchNorm or ReLU, so the latent is linear and has the same form as a VAE's posterior mean), trained on
the unit-variance Gaussian NLL. It beat the plain AE there by 0.04 AUROC. It should beat the plain AE on Nature too, and
it is the fair baseline that any VAE must beat.

## Method

- Trained on the cluster with the same arrays, participants and settings as the plain AE: three seeds per input (1 s
  raw; 3 s + rhythm).
- Scored exactly as in E25: the threshold and LOF representation were chosen on validation at a 1% false-alarm target,
  and the test set was scored once.
- Under Adam the Gaussian NLL is the MSE times a constant, so the linear latent head is the only substantive change. The
  trainer's check confirms the objective equals the Science Robotics VAE's NLL term.

## Result

| Model | Input | LOF on | AUROC unfamiliar | AUROC dangerous | False alarms | Unfamiliar flagged | Dangerous flagged |
|---|---|---|---|---|---|---|---|
| Plain AE | 1 s | 16 PCs | 0.887 ± 0.010 | 0.968 ± 0.008 | 1.7% | 34.2 ± 6.0% | 52.8 ± 10.0% |
| Gaussian-NLL | 1 s | 24 PCs | 0.887 ± 0.014 | 0.966 ± 0.009 | 1.9% | 35.0 ± 0.8% | 49.8 ± 1.6% |
| Plain AE | 3 s + rhythm | all 32 | 0.893 ± 0.010 | 0.928 ± 0.013 | 2.0% | 34.7 ± 3.8% | 38.2 ± 2.5% |
| **Gaussian-NLL** | **3 s + rhythm** | all 32 | **0.911 ± 0.012** | **0.949 ± 0.018** | 1.9% | **43.9 ± 3.1%** | 46.0 ± 3.6% |

- **1 s input:** no gain, but far more seed-consistent (SD under 2 points against 6–10).
- **3 s + rhythm:** better everywhere, also on validation (0.890 against 0.864) and along the whole recall/false-alarm
  curve. It was the best unfamiliar-movement detector at this point.
- The linear latent spreads windows along about twice as many directions as the ReLU latent (11.5 against 6.6 on 1 s).
- The gain is smaller than on the Science Robotics data (0 on 1 s, about 0.02 on 3 s, against 0.04).

## Conclusion

The Gaussian-NLL control **replaces the plain AE as the baseline**. Its latent has the form of a VAE posterior mean, so
the VAE grid (E28–E29) has to beat it, not the ReLU AE. Which detector to use depends on the alarm's job: the 3 s latent
ranks all unfamiliar movement best, while the 1 s latents (either head) rank dangerous tasks best.
