# VAE — standard and β-VAE on the autoencoder's trunk

`ml/vanilla/` reproduces the paper's models; `ml/prob_ensemble/` is the first deliberate
departure. This package is the second: the convolutional autoencoder's encoder trunk and
decoder, with a **Gaussian latent** in place of the deterministic bottleneck, trained on the
negative ELBO with a KL weight β. It is explored in
[`notebooks/model_exploration/08_vae_variants.ipynb`](../../notebooks/model_exploration/08_vae_variants.ipynb)
and is not registered in the vanilla CLI.

## What changes, and what does not

| | AE + LOF (`ml/vanilla`) | this package |
|---|---|---|
| encoder trunk | 16×200 → 19×40 → 24×8 | **identical** (imported, not copied) |
| bottleneck | 4×8 ReLU latent | 4×8 **mean** and 4×8 **log-variance** heads, sampled |
| decoder | mirrored, 8 → 40 → 200 | **identical** |
| objective | reconstruction MSE | Gaussian NLL (unit variance) + β·KL, summed per window |
| detection score | LOF on latents, fitted on ID training latents | **identical rule**, on posterior means |
| smoothing, threshold | causal median ×10, 99.5th percentile of training scores | **identical** |
| protocols | participant split; LOTO SA/ST/TR | **identical builders** (`ml/vanilla/dataset.py`, `ml/LOTO/splits.py`) |

Reconstruction NLL, KL and negative ELBO (eight posterior samples) are saved as separate
diagnostic scores with their own thresholds. Posterior log variances are saved beside the
means so the notebook can draw sampled latent vectors and learned variance.

## Files

| file | role |
|---|---|
| `vae_paths.py` | repo lookup; puts `ml/vanilla` and `ml/LOTO` on `sys.path`; no IPython or matplotlib |
| `vae_model.py` | `ConvVAE`, `loss_terms` |
| `vae_train.py` | `VAEConfig`, data builders, `train_experiment`, and the CLI (`--index`, `--variant/--protocol`, `--smoke`, `--list`) |
| `vae_plots.py` | static PCA / score-histogram / trace figures for one run |
| `vae_sweep.py` | summarize the β sweep and controls (active dimensions, KL, validation and test LOF AUROC) into `results/` |
| `posthoc_dims.py` | LOF refitted on fewer latent dimensions (active-only, PCA-k) for the VAE runs and the AE baseline, no retraining |
| `test_vae.py` | wiring tests on tiny synthetic arrays (plus a split-size check when the arrays exist) |
| `slurm/` | `smoke.sbatch`, `vae.sbatch` (matrix, array 0–7), `sweep.sbatch` (β sweep + controls, 0–6), `seeds.sbatch`, `submit.sh`, `push.sh`, `pull.sh` |
| `runs/` | `<protocol>_<variant>_seed<seed>/` — ignored by git, pulled from the cluster |
| `results/` | tracked summaries: `sweep_<protocol>.{csv,md,png}`, `posthoc_dims.{csv,md}` |

Variants are named configuration overrides in `vae_train.py`: the matrix (`standard` β=1, `beta4`
β=4, each under `original` and the `SA`, `ST`, `TR` holdouts; array index `standard` 0–3, `beta4`
4–7), the sweep (`beta0.25`, `beta2`, `beta8`, `beta16`, `beta32`) and two β=0 controls with sampling
off, which run the AE objective through this pipeline: `ae_control` at the VAE optimizer settings and
`ae_control_aehp` at the AE's (lr 0.003, batch 1,024, up to 200 epochs).

## The original protocol on the cluster

The cluster holds only the mode-complete arrays (`processed_all`). `original_protocol_view`
rebuilds the participant-split partition from them at load time: LG/RA/RD in-distribution,
SA/SD/ST/TR out-of-distribution. Restricted to LG/RA/RD, `processed_all` reproduces
`data/processed` window for window with the same scaler (`ml/LOTO/README.md`), and
`test_vae.py` checks the split sizes (187,993 / 64,350 / 61,956 ID windows; 23,043 OOD test
windows). Locally, where `data/processed` exists, the ordinary `AutoencoderData` is used.

## Running

```bash
.venv/bin/python ml/VAE/test_vae.py                  # wiring tests
.venv/bin/python ml/VAE/vae_train.py --list          # cell states
.venv/bin/python ml/VAE/vae_train.py --variant standard --protocol SA --smoke   # local wiring run
```

Cluster (FASRC), reusing `ml/LOTO/slurm/env.sh` for modules, venv, account and node-local
data staging. Run the laptop commands with an authenticated SSH connection open:

```bash
bash ml/VAE/slurm/push.sh                                                   # laptop
ssh login.rc.fas.harvard.edu 'cd ~/Uncertainty-Aware-Hip-Exoskeleton && bash ml/VAE/slurm/submit.sh smoke'
ssh login.rc.fas.harvard.edu 'cd ~/Uncertainty-Aware-Hip-Exoskeleton && bash ml/VAE/slurm/submit.sh matrix'
bash ml/VAE/slurm/pull.sh                                                   # laptop, repeatable
```

```bash
ssh login.rc.fas.harvard.edu 'cd ~/Uncertainty-Aware-Hip-Exoskeleton && bash ml/VAE/slurm/submit.sh sweep'
ssh login.rc.fas.harvard.edu 'cd ~/Uncertainty-Aware-Hip-Exoskeleton && bash ml/VAE/slurm/submit.sh seeds beta4 original'
bash ml/VAE/slurm/pull.sh && .venv/bin/python ml/VAE/vae_sweep.py --protocol original --write
.venv/bin/python ml/VAE/posthoc_dims.py --write        # laptop only, a few minutes
```

Smoke cells write `*_smoke` directories and are never read as results. A finished cell is not
redone without `--force`. Each full cell took 7–13 minutes on an A100 (seed 0, September 2026).

## Results so far (seed 0, latent LOF, test)

| protocol | standard AUROC | β=4 AUROC | AE + LOF AUROC |
|---|---|---|---|
| original | 0.647 | 0.709 | 0.782 |
| SA held out | 0.650 | 0.652 | 0.712 |
| ST held out | 0.984 | 0.992 | 0.991 |
| TR held out | 0.851 | 0.885 | 0.876 |

Neither variant beats the AE on ranking under the original protocol or stair ascent; β=4 is
consistently the better VAE. The likelihood-style scores rank OOD windows *below* ID ones on
most protocols (AUROC well under 0.5), the known failure of generative likelihoods for OOD
detection. One seed per cell: differences of a few hundredths are within noise.

## β sweep, controls and the dimension question (original protocol)

Full tables: [`results/sweep_original.md`](results/sweep_original.md),
[`results/posthoc_dims.md`](results/posthoc_dims.md); figure `results/sweep_original.png`.
Selection is on the validation split; test is reported beside it.

**Posterior variance is not an OOD signal.** σ² is flat across tasks in every run (KL AUROC
0.43–0.49 on validation), because with the decoder variance fixed at one the KL term is under two
percent of the objective and the posterior width is set by that global balance, not by the input.
LOF on the means stays the score.

**The hyperparameters are not the cause of the VAE-versus-AE gap.** The β=0, no-sampling control at
the VAE optimizer settings (`ae_control`) reaches validation 0.736 / test 0.799, above the AE's
0.715 / 0.782. The pipeline is sound; the Gaussian latent with sampling is what costs about 0.15
test AUROC at β=1. The same objective at the AE's own settings (`ae_control_aehp`: lr 0.003,
batch 1,024, stopped at epoch 43 of a 200 budget) does better still, validation **0.776** / test
**0.822**, the best single number in this package. So the optimizer settings are worth about 0.04
on top, and this deterministic pipeline (the AE trunk with a linear rather than ReLU latent head,
unit-variance Gaussian NLL) is a stronger baseline than `ml/vanilla`'s AE. Single seed.

**Fewer dimensions help every model** (`posthoc_dims.py`, LOF refitted, no retraining):

| representation | AE val / test | VAE β=1 val / test | VAE β=4 val / test |
|---|---|---|---|
| all 32 dims | 0.715 / 0.782 | 0.621 / 0.647 | 0.706 / 0.709 |
| active dims only | — | (32 active) | 0.703 / 0.703 (18) |
| PCA-8 | 0.730 / 0.793 | **0.725 / 0.798** | 0.676 / 0.682 |
| PCA-12 | **0.735 / 0.810** | 0.696 / 0.762 | 0.706 / 0.676 |
| PCA-16 | 0.730 / 0.823 | 0.650 / 0.678 | 0.702 / 0.688 |

The β=1 VAE reduced to 8 PCs matches the AE; the AE itself gains 0.02 validation / 0.04 test at
12–16 PCs. So part of β=4's advantage over β=1 is a dimension effect. It is not all of it: β=1
PCA-reduced to 18 dims (0.634 / 0.664) is well below β=4's native 18 active dims (0.706 / 0.709),
so the KL term also shapes the code.

**The β sweep is not an inverted U.** Validation LOF AUROC (seed 0) climbs from 0.62 at β=1
through a noisy plateau of 0.70–0.72 at β=4–16 (active dimensions falling 18 → 11 → 8) to
**0.798 at β=32** with 8 active dimensions; test follows more weakly (0.71–0.72 on the plateau,
0.746 at β=32). β=16 and β=32 have the same 8 active dimensions yet differ by 0.09 on validation,
so seed noise is large. Three seeds at β=32 give validation **0.798 ± 0.012** and test
**0.761 ± 0.014** (mean ± sd, seeds 0–2), which holds up: the strongly regularized VAE is the best
VAE on validation and within 0.02 of the AE on test, but it does not beat the AE, and its recall at
the training threshold is low (25–34 %) against the AE's 45 %.

**What this leaves open.** The plateau-then-jump shape suggests the sweep should be extended
(β = 64, 128) with seeds before choosing; and since the dimension effect is real for the AE too,
the cheapest improvement to the *baseline* is LOF on 12–16 PCs of the AE latent, selected on
validation. Neither is run yet.
