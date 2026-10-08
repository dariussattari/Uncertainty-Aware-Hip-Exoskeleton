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
| [09_umap_all_tasks.ipynb](09_umap_all_tasks.ipynb) | Interactive 2-D/3-D UMAP map of all seven tasks' raw sensor windows; no model, no ID/OOD labels. Variant: the same windows with the raw pelvis IMU added (thigh / pelvis / thigh+pelvis), with leave-one-participant-out k-NN task accuracy per set |
| [10_awareness_regime_holdout.ipynb](10_awareness_regime_holdout.ipynb) | Novelty, regime and reliability signals under the DOWN regime holdout, every score by task, six baselines; injected-fault four-cell grid; pelvis-IMU (22-channel) encoder ablation |

From the repository root:

```bash
.venv/bin/python -m jupyter lab notebooks/sci_rob/model_exploration
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

The VAE trainer lives in its own package, `ml/sci_rob/VAE/` (`vae_train.py`, `vae_plots.py`,
`slurm/`), beside `ml/sci_rob/vanilla` and `ml/sci_rob/prob_ensemble`, with the model in `ml/common/models/vae.py`; it is **not** registered in the vanilla
CLI. Standard VAE and beta-VAE share a 32-value Gaussian latent and the existing AE's first two
encoder blocks and decoder. The decoder observation variance is fixed to one in standardized
input units.

Training is opt-in and normally runs on the cluster (below). Runs live under the ignored
`ml/sci_rob/VAE/runs/`; they never overwrite `ml/sci_rob/vanilla` or `ml/sci_rob/prob_ensemble` outputs. `held_out=None`
uses the original protocol, while `SA`, `ST`, or `TR` uses the existing LOTO data builder. The
primary score is LOF on posterior means. Reconstruction NLL, KL and negative ELBO are separately
evaluated diagnostics. One seed-0 run exists per cell; read those numbers as a first pass.

## Interactive latent views: UMAP, PCA and raw coordinates

Three places offer Plotly maps you can pan, zoom and rotate (scroll = zoom, drag = pan or
rotate, legend click = hide a task in every panel, legend double-click = isolate it):

- `09_umap_all_tasks.ipynb`: raw standardized sensor windows of all seven tasks (PCA-50 then UMAP),
  with the leading principal components as the linear reference. A variant section slices the six
  pelvis IMU channels out of the raw trials for the same windows and shows `pelvis` and `thigh+pelvis`
  maps beside the 16-channel one, with a leave-one-participant-out k-NN task accuracy table per set,
  the cheap check on whether the pelvis IMU is worth the loss of comparable datasets.
- `02_autoencoder_lof.ipynb`: the saved AE test latents. UMAP and PCA are fitted on in-distribution
  latents only and every task transformed into that map (the saved PCA figure's discipline); a
  third panel shows raw latent coordinates.
- `08_vae_variants.ipynb`: for each trained VAE run, posterior means, sampled latent vectors
  (with dense posterior "clouds" around a few windows) and the means coloured by learned
  variance, under the same three projections fitted on the training reference means.

Each cell has `VIEW_DIMS = (2, 3)`, `PROJECTIONS` and, for latents, `LATENT_DIMS` (`None` picks
the highest-variance coordinates on the reference rows). In 2-D the projections are side-by-side
panels of one figure, and so are the 3-D scenes, with `means` / `samples` / `variance` buttons
for the VAE switching every panel. WebGL sets the practical limit: 2-D maps are SVG, but every
3-D panel is a WebGL scene, browsers allow only about 8-16 live contexts per page, and JupyterLab
is one page for every open notebook. A 3-D panel with a legend but a blank scene means the cap was
hit: close the other notebook tabs and reload. Tasks keep one fixed colour and marker symbol
everywhere, never an ID/OOD colour. UMAP preserves neighbourhoods, not distances; PCA is linear
and labelled with explained variance; only the raw-coordinate panel shows true latent distances,
for the two or three coordinates it can show. UMAP embeddings are cached in the ignored `.cache/`
folder; standalone HTML copies go to the ignored `interactive/` folder when `SAVE_HTML = True`.
Figures are stored as the Plotly MIME bundle only (JupyterLab and VS Code render it; the HTML
exports serve other viewers), and `verify_notebooks.py --execute` signs the notebooks it runs as
trusted; if a figure shows nothing, use JupyterLab's *Trust Notebook* (File menu). The optional dependencies are listed
in `requirements-umap.txt` (`umap-learn`, `plotly`); the repository `.venv` already has them.
`_umap_exploration.py` holds the presentation code and `test_umap_exploration.py` its wiring tests.

## Reading results correctly

Every notebook includes a **LOTO score spread and threshold what-if** section for
SA, ST and TR. For completed runs it shows per-task score violins and a threshold
sweep: only the named held-out task is orange; all six ID tasks are blue. The black
dashed line is the saved threshold; black bars/ticks show IQR/median. Change
`LOTO_THRESHOLDS = {'SA': None, 'ST': None, 'TR': None}` to numeric thresholds in
original score units and rerun that cell to add a purple what-if line. Exact recall,
aggregate ID false alarms and per-task flag rates use all saved smoothed scores.
`None` retains the saved threshold. Nothing is retrained or written to the run directory.

The probabilistic notebook defaults to `LOTO_SCORE = 'epistemic'`; change it to
`'aleatoric'`, `'total'`, or `'ratio'` to inspect each score's own threshold. VAE
sections load each variant's SA/ST/TR runs after they exist, with the same score
selector for its four diagnostics. Forecast-all and untrained VAE runs remain
explicitly unavailable. These test-set threshold sweeps are exploratory, **not** a
valid way to select a deployment threshold; use validation data for that choice.

OOD is the positive class. AUROC is ranking, not percent detected. The stored JSON
reports recall/specificity/J in percentage units; notebook result tables normalize J
to the range −1 to 1. The original threshold is the 99.5th percentile of causally
median-filtered ID training scores. A low score is not proof a prediction is accurate.

LOSO selects ensemble training duration by leaving out training participants; LOTO
excludes a task while retaining the participant-disjoint splits. Standing has one
test trial, and gait-phase LOTO uses a smaller labelled subset. Different task mixtures
are different benchmarks; do not interpret their AUROC differences as controlled effects.

`_exploration.py` handles presentation; `_loto_exploration.py` loads and validates
score/task alignment, plots distributions, and computes exact threshold what-ifs;
`_umap_exploration.py` builds the interactive UMAP / PCA / raw-coordinate views. Scientific implementations remain
in their source modules. `build_notebooks.py --patch` emits a patch for notebook sources and
`build_notebooks.py --write [stems]` writes them directly; either way regenerating drops saved
outputs, so execute the notebooks again afterwards (`verify_notebooks.py --execute --notebook '02*'`).

Verification:

```bash
.venv/bin/python notebooks/sci_rob/model_exploration/verify_notebooks.py --execute
.venv/bin/python notebooks/sci_rob/model_exploration/test_exploration.py
.venv/bin/python ml/sci_rob/VAE/test_vae.py
.venv/bin/python notebooks/sci_rob/model_exploration/test_loto_exploration.py
.venv/bin/python notebooks/sci_rob/model_exploration/test_umap_exploration.py
```

The VAE unit tests use tiny synthetic arrays only; they are wiring tests, not benchmark runs. The LOTO tests additionally check saved-run metric agreement,
strict-threshold ties, task labels and figure colors. The UMAP tests use synthetic latents
and check sampling, caching, fixed task colours and graceful handling of missing runs. Normal
notebook execution never calls the optional full trainers.
