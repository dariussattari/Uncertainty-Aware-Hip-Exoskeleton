# Model exploration notebooks

Open these with the repository's `.venv` Python kernel. Each notebook contains the
question the model answers, equations, an architecture diagram, runnable model setup,
training configuration/commands, saved learning curves and result figures, and limitations.
The notebooks have saved outputs so they can also be read without running training.

| Notebook | Coverage |
|---|---|
| [01_gait_phase_ensemble.ipynb](01_gait_phase_ensemble.ipynb) | Seven-branch TCN, masked gait-phase regression, disagreement, LOTO |
| [02_autoencoder_lof.ipynb](02_autoencoder_lof.ipynb) | Convolutional AE, latent LOF versus reconstruction error, latent figures, LOTO |
| [03_gan.ipynb](03_gan.ipynb) | Generator/discriminator, alternating training, saturation and OOD score, LOTO |
| [04_correlation_ensemble.ipynb](04_correlation_ensemble.ipynb) | Label-free correlation target, denominator floor, disagreement, LOTO |
| [05_angle_forecast_ensemble.ipynb](05_angle_forecast_ensemble.ipynb) | 200 ms bilateral angle forecast, masking, KV error/disagreement decomposition, LOTO |
| [06_all_channel_forecast.ipynb](06_all_channel_forecast.ipynb) | 16-channel forecast architecture and training; explicitly unavailable completed results |
| [07_probabilistic_ensemble.ipynb](07_probabilistic_ensemble.ipynb) | Gaussian forecast heads, beta-NLL, variance decomposition, calibration, four scores, LOTO |
| [08_vae_variants.ipynb](08_vae_variants.ipynb) | New notebook-local standard/beta VAE prototypes, LOF and ELBO diagnostics, opt-in training |

From the repository root:

```bash
.venv/bin/python -m jupyter lab notebooks/model_exploration
```

The notebooks select the existing `hip-exo` kernel, which points to this repository's
`.venv`. On another machine, register its environment first:

```bash
.venv/bin/python -m ipykernel install --user --name hip-exo --display-name "Hip Exo (.venv)"
```

## Safe defaults

- **Run All never trains a model by default.** Set `RUN_TRAINING=True` deliberately.
- Existing models reuse the real training CLI, with new output directories and an
  overwrite guard. Saved figures are displayed, not regenerated in the original run.
- Architecture shape checks use fresh random weights and say so; they are not results.
- Missing artifacts print an explicit message, not invented metrics.
- Code locates the repository from the current directory or its parents. No hardcoded
  user-specific paths are required. Launch Jupyter within this checkout.
- Source arrays/checkpoints are not bundled in these notebooks. Inspection of saved
  reports and figures does not require loading the multi-GB arrays.

## VAE status and boundaries

There was no VAE implementation or trained VAE artifact in this checkout when these
notebooks were created. `_vae_experiment.py` is an **exploratory implementation**, not
a new entry in the production CLI registry. Standard VAE and beta-VAE share a 32-value
Gaussian latent and the existing AE's first two encoder blocks and decoder. The
decoder observation variance is fixed to one in standardized input units.

Training and evaluation are opt-in. Runs go under this folder's ignored `runs/vae/`;
they do not overwrite `ml/vanilla` or `ml/prob_ensemble` outputs. `held_out=None` uses
the original protocol, while `SA`, `ST`, or `TR` uses the existing LOTO data builder.
The primary score is LOF on posterior means. Reconstruction NLL, KL and negative ELBO
are separately evaluated diagnostics. No VAE performance claims are made before training.

## Reading results correctly

OOD is the positive class. AUROC is ranking, not percent detected. The stored JSON
reports recall/specificity/J in percentage units; notebook result tables normalize J
to the range −1 to 1. The original threshold is the 99.5th percentile of causally
median-filtered ID training scores. A low score is not proof a prediction is accurate.

LOSO selects ensemble training duration by leaving out training participants; LOTO
excludes a task while retaining the participant-disjoint splits. Standing has one
test trial, and gait-phase LOTO uses a smaller labelled subset. Different task mixtures
are different benchmarks; do not interpret their AUROC differences as controlled effects.

`_exploration.py` handles presentation; scientific implementations remain in their
source modules. `build_notebooks.py --patch` emits a patch for notebook sources;
regenerating replaces saved outputs, so execute them again afterwards.

Verification:

```bash
.venv/bin/python notebooks/model_exploration/verify_notebooks.py --execute
.venv/bin/python notebooks/model_exploration/test_exploration.py
```

The four unit tests use tiny synthetic arrays only; they are wiring tests, not VAE
benchmark runs. Normal notebook execution never calls the optional full trainers.
