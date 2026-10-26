# E17: VAE and β-VAE on the Science Robotics data (β sweep, Gaussian-NLL control)

| Field | Value |
|---|---|
| ID | E17 |
| Date | 2026-09-22 to 2026-09-29 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34, original participant-split protocol |
| Status | Complete (first pass: one seed per cell, except β 32 with three) |
| Sources | Live_Update.docx, "SEP 18 – 25" §4 ("VAE Implementation"); 12OCT-15OCT Verbose.docx §3.1 |
| Code / notebooks | `ml/sci_rob/VAE/` (`vae_train.py`, `vae_plots.py`, `slurm/`), `ml/common/models/vae.py`; `notebooks/sci_rob/model_exploration/08_vae_variants.ipynb` |

## Hypothesis

A VAE regularizes the latent toward a continuous Gaussian (KL term weighted by β). The hypothesis was that this would
(a) give a better-behaved latent for LOF, and (b) give probabilistic novelty scores (likelihood, KL), avoiding the
concern that AEs reconstruct too well. β was the key variable to ablate.

## Method

- A VAE with a 32-value Gaussian latent, sharing the AE's first two encoder blocks and decoder (E05). Decoder variance
  fixed at 1 in standardized units.
- β in {0 (no sampling, at lr 0.003/batch 1024 and at lr 0.001/batch 256), 0.25, 1, 2, 4, 8, 16, 32}. One seed per
  cell, plus three seeds at β 32.
- Primary score: LOF on the posterior means. NLL, KL and negative ELBO were scored separately as diagnostics.
- Post hoc: LOF on PCA-12 and PCA-16 of the AE latent (no retraining), and a **deterministic control**: the AE encoder
  ending at its last convolution (linear latent head), trained on the unit-variance Gaussian NLL. This is the β = 0
  control at the AE's settings.
- Latents were inspected with UMAP (validation AUROC and active dimensions per panel).

## Result

| Model (test AUROC) | AUROC |
|---|---|
| AE + LOF (E05) | 0.782 |
| VAE β 1, LOF on means | 0.647 |
| VAE β 4 | 0.709 |
| VAE β 32, three seeds | 0.761 ± 0.014 |
| Likelihood scores (NLL, KL) | about 0.31: unfamiliar windows look *more* likely |
| **Deterministic AE, linear latent + Gaussian NLL (β = 0 control)** | **0.822** |
| AE + LOF on PCA-12 / PCA-16 | 0.810 / 0.823 |

- Validation sweep (UMAP figure): β 0 → 0.776 (lr 0.003) or 0.736 (lr 0.001); β 0.25 → 0.666 (32 active dimensions);
  β 1 → 0.621 (32); β 2 → 0.660 (24); β 4 → 0.706 (18); β 8 → 0.723 (11); β 16 → 0.708 (8); β 32 → 0.798 (8).
- β = 0, 16 and 32 give the most uniform latent shapes, and those cells also have the highest AUROC at the training
  threshold.
- β 16 and β 32 had the same 8 active dimensions yet differed by 0.09 in validation AUROC, so the sweep is noisy at one
  seed.
- With the decoder variance fixed at 1, posterior variance has little spread across tasks.

## Conclusion

On this data **the VAE does not beat the AE**. The best "VAE-family" model is the deterministic linear-latent
Gaussian-NLL control (0.822), and reducing the AE latent to 12–16 PCs gives most of the same gain. Likelihood scores are
inverted, as reconstruction error was. If a VAE is to add uncertainty, the experiment to run is a **learned decoder
variance**, not more β. That idea and the Gaussian-NLL control became steps in the Nature plan (E26, E28).

## Caveats

- One seed per cell except β 32.
- The decoder variance was fixed.
- The original participant-split protocol only.
