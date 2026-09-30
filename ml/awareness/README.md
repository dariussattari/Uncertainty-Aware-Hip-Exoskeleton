# Awareness layer — novelty, regime and reliability under a regime holdout

The controller-facing question is not "is this window out of distribution" but three
questions at once: is it *unfamiliar* (novelty), *which movement regime* does it resemble
(regime), and *is the active prediction trustworthy* (reliability). This package computes
those three signals from one sensor window and lays them side by side, per task, under a
holdout that removes an entire regime from training. It forks nothing:

| stage | signal | implementation | reused from |
|---|---|---|---|
| A `encoder.py` | novelty: LOF on a 32-D latent (32 dims, and 12/16 PCs) | deterministic β=0 encoder at the AE's settings (`ae_control_aehp`) | `ml/VAE` |
| B `regime.py` | regime probabilities, confidence, entropy | logistic and MLP probes on the latent, ID task labels only | scikit-learn |
| C `forecast.py` | reliability: branch disagreement now, realized error 200 ms later, their Krogh–Vedelsby terms | the seven-TCN hip-angle forecast ensemble, plain or with the regime probabilities as extra input channels | `ml/vanilla` |
| C' `experts.py` | reliability with a load-bearing regime: one forecast ensemble per training regime, routed by the probe; routed error, best-expert error, disagreement between experts | the same ensemble, trained per regime | `ml/vanilla` |
| `aw_scores.py` | every per-window score, plus the six existing models' scores from their LOTO leaves | one aligned table, threshold-free summary | `ml/LOTO` |
| `aw_plots.py` | score spread by task, held-out tasks in orange | one violin grid per score group | |

## The DOWN holdout

`ml/LOTO/splits.py` now accepts a holdout *name* mapping to a set of modes. `DOWN` removes
ramp descent, stair descent, standing and transitions, so training sees level ground, ramp
ascent and stair ascent only:

| | ID (LG, RA, SA) | OOD (RD, SD, ST, TR) |
|---|---|---|
| train (12 participants) | 172,527 | — |
| val (4) | 58,813 | 28,881 |
| test (4) | 57,445 | 27,554 (32.4 %) |

Same participants, scaler, causal median filter and evaluators as every other run, so the
six existing models (`ml/LOTO/DOWN/...`) and the three stages here are compared on identical
windows. The regime probes can only name `level` and `ascend`; what is recorded on a
descending window is their probability, confidence and entropy, never a label.

## Data flow

```
window (16 ch, 1 s)
  ├─► encoder ──► latent z ──► LOF (32-D / PCA-12 / PCA-16) ──────────────► novelty
  │                 └──► regime probe ──► p(level), p(ascend), confidence, entropy ─► regime
  └─► forecast ensemble [+ regime probs as 2 constant channels] ──► disagreement, error ─► reliability
```

Row alignment is by construction: every stage iterates the holdout views in the same order
(`aw_paths.SPLITS`), ID and OOD separately, and `scores.py` checks the saved labels against
that order before it aligns anything.

## Running

```bash
.venv/bin/python ml/awareness/encoder.py --holdout DOWN --smoke     # local wiring check
bash ml/awareness/slurm/push.sh
ssh login.rc.fas.harvard.edu 'cd ~/Uncertainty-Aware-Hip-Exoskeleton && bash ml/awareness/slurm/submit.sh DOWN'
ssh login.rc.fas.harvard.edu 'cd ~/Uncertainty-Aware-Hip-Exoskeleton && bash ml/LOTO/slurm/submit.sh all --tasks DOWN'
bash ml/awareness/slurm/pull.sh && bash ml/LOTO/slurm/collect.sh --no-doc
.venv/bin/python ml/awareness/aw_scores.py --holdout DOWN --write
.venv/bin/python ml/awareness/aw_plots.py  --holdout DOWN
```

`submit.sh` chains the stages: the encoder-plus-regime job, the plain forecaster and the regime
experts at once, the regime-conditioned forecaster after stage B (`--only experts` submits just the
expert array). After pulling, `experts.py score --holdout DOWN` runs every expert over all test
windows on the laptop and `experts.py swap-test` records how much the conditioned forecaster used
its regime input. `--folds N` trims the epoch-selection
folds (default 1, the fast first pass; 12 is the full protocol). Runs land under the ignored
`runs/<holdout>/`; tracked summaries and figures under `results/<holdout>/`.

Methods and results are presented in
[`notebooks/model_exploration/10_awareness_regime_holdout.ipynb`](../../notebooks/model_exploration/10_awareness_regime_holdout.ipynb).

## Results so far: the DOWN holdout, seed 0 (test AUROC, OOD versus ID; no thresholds)

| score | OOD vs ID | RD | SD | ST | TR |
|---|---|---|---|---|---|
| vanilla AE, LOF | **0.976** | 0.989 | 0.973 | 0.998 | 0.895 |
| awareness encoder, LOF on 32-D latent | 0.974 | 0.979 | 0.983 | 0.998 | 0.882 |
| GAN | 0.938 | 0.925 | 0.973 | 1.000 | 0.835 |
| forecast ensemble, disagreement (1-fold) | 0.889 | 0.850 | 0.966 | 0.998 | 0.753 |
| correlation ensemble, disagreement | 0.884 | 0.837 | 0.960 | 0.998 | 0.803 |
| forecast ensemble, disagreement (full 12-fold protocol) | 0.881 | 0.841 | 0.961 | 0.998 | 0.727 |
| gait-phase ensemble, disagreement | 0.878 | 0.845 | 0.989 | 0.991 | 0.739 |
| probabilistic ensemble, ratio / epistemic | 0.865 / 0.859 | 0.838 / 0.806 | 0.902 / 0.958 | 0.998 / 0.998 | 0.841 / 0.704 |
| forecast ensemble, realized error at 200 ms | 0.816 | 0.775 | 0.907 | 0.823 | 0.650 |
| regime experts, routed error / best-expert error | 0.810 / 0.811 | 0.784 / 0.817 | 0.851 / 0.791 | 0.840 / 0.905 | 0.780 / 0.836 |
| regime-conditioned forecaster, disagreement / error | 0.862 / 0.792 | 0.816 / 0.788 | 0.962 / 0.899 | 0.922 / 0.134 | 0.673 / 0.490 |
| reconstruction NLL (encoder) | 0.486 | 0.572 | 0.467 | 0.003 | 0.104 |

Full tables and the per-task spread figures: `results/DOWN/`.

- **Density in a learned latent is the detector for a held-out regime.** Both AE-family latents
  with LOF rank the descents, standing and transitions near-perfectly; every disagreement-based
  score sits at 0.86–0.89 and every error-based score lower. The one-fold epoch-selection shortcut
  cost the forecast ensemble nothing (0.889 vs 0.881 with twelve folds).
- **Regime confidence is not novelty.** Both probes read the regime on known windows (89–91 %
  accuracy) and absorb unseen regimes confidently (standing → level at 1.00). The probe's job is
  to name the nearest known regime, which it does; novelty has to come from density.
- **Conditioning the forecaster on the regime did nothing, and experts did not help either.**
  The swap test (`results/DOWN/regime_swap_test.csv`) shows the conditioned model moved its error
  by only 5–7 % under a wrong regime input. Experts make the regime load-bearing (each expert fails
  on the other's regime), yet routed error equals the plain forecaster's, because descending gait
  is predictable from level-and-ascending dynamics 200 ms ahead. The one gain is on transitions
  (error 0.65 → 0.78, disagreement 0.75 → 0.92).
- Transitions are the hardest held-out mode for every score; reconstruction error is inverted
  everywhere, as under the original protocol.
- **A longer forecast horizon does not make the regime matter either.** Plain ensembles at 500 ms
  and 1 s (`forecast_plain_h100`, `forecast_plain_h200`) lose in-distribution accuracy faster than
  they gain sensitivity to the unseen regime: disagreement AUROC 0.889 → 0.825 → 0.787, realized
  error 0.816 → 0.660 → 0.569, validation loss 0.049 → 0.105 → 0.165 (the 1 s model stopped after
  three epochs). Stride-to-stride variability dominates the far horizon for every regime, so the
  separation shrinks instead of growing. The 200 ms target stays.

## Injected sensor faults: all four cells (`faults.py`, `results/DOWN/faults_*`)

Five persistent input faults (dead left-thigh IMU, frozen right-thigh IMU, IMU noise at 2×
channel spread, gyro spikes, 100 ms left-leg delay) applied to level-ground windows (→
ID-dangerous) and to ramp-descent windows (→ OOD-dangerous), scored beside the untouched
windows (ID-safe, OOD-safe) through the whole pipeline. A constant encoder offset was tried and
dropped: a calibration error, not a dynamic fault. Health in the quadrant figure is the **routed
expert error**, so the regime has its say. Test AUROC of each dangerous cell against its own
safe cell:

| fault | ID-dangerous vs ID-safe: LOF / recon / disagreement / routed error | OOD-dangerous vs OOD-safe: LOF / recon / disagreement / routed error |
|---|---|---|
| dead left IMU | 0.989 / 0.915 / 0.983 / 0.839 | 0.975 / 0.872 / 0.982 / 0.645 |
| frozen right IMU | 0.989 / 0.931 / 0.973 / 0.920 | 0.998 / 0.830 / 0.955 / 0.643 |
| IMU noise | 0.966 / 0.989 / 0.939 / 0.670 | 0.914 / 1.000 / 0.945 / 0.431 |
| gyro spikes | 0.989 / 0.989 / 0.988 / 0.804 | 1.000 / 1.000 / 0.999 / 0.323 |
| left-leg delay | 0.981 / 0.816 / 0.897 / 0.873 | 0.827 / 0.758 / 0.608 / 0.642 |

- **In the novelty × forecast-error plane the four cells do not resolve.** ID-dangerous windows
  land where OOD-safe windows already are (routed error against OOD-safe 0.16–0.53), and
  OOD-dangerous sits only slightly above OOD-safe (0.32–0.65). Routing through the regime
  experts did not change that: a faulted sensor barely hurts a hip-angle forecast that leans on
  the intact channels, whereas a genuinely different regime hurts it more.
- **In the novelty × reconstruction plane they do.** LOF separates ID from OOD (0.98) and
  reconstruction error separates dangerous from safe within each column (0.76–1.00), most
  sharply for noise and spikes, least for the leg delay. Disagreement does nearly as well except
  for the delay.
- The regime probe grows *more* confident under faults, so its confidence is anti-informative.

So the two axes of the grid are LOF for familiar-versus-unfamiliar and reconstruction error (with
disagreement) for healthy-versus-unhealthy; forecast error belongs to the controller-consequence
question, not to fault detection.

## Ablation: the pelvis IMU as six more input channels (`pelvis_ablation.py`, `results/DOWN/pelvis_ablation.*`)

The raw trials carry a pelvis-mounted IMU the 16-channel set omits. Notebook 09's raw-space check
found it adds almost no task structure (leave-one-participant-out k-NN task accuracy 0.600 → 0.615).
This is the direct test for novelty: `ml/LOTO/build_data.py --channels pelvis` cuts the same windows
with the six pelvis channels appended (the 16 shared channels keep the identical scaler; the pelvis
channels get their own LG/RA/RD-training scaler; built on the cluster from the raw trials by
`slurm/pelvis_build.sbatch`), and `slurm/pelvis.sbatch` reruns Stage A and Stage B unchanged on it
(`encoder_pelvis/`, `regime_pelvis/`; same β = 0 encoder, settings, seed, LOF reference size).
Test AUROC on identical windows:

| score | 16 ch: OOD vs ID / RD / SD / ST / TR | 22 ch (+ pelvis): OOD vs ID / RD / SD / ST / TR |
|---|---|---|
| LOF, 32-D latent | **0.974** / 0.979 / 0.983 / 0.998 / 0.882 | 0.948 / 0.950 / 0.968 / 0.998 / 0.822 |
| LOF, PCA-12 | 0.957 / 0.953 / 0.983 / 0.998 / 0.838 | 0.947 / 0.948 / 0.965 / 0.999 / 0.843 |
| LOF, PCA-16 | 0.958 / 0.956 / 0.981 / 0.998 / 0.839 | 0.954 / 0.951 / 0.973 / 0.998 / 0.866 |
| reconstruction NLL | 0.486 / 0.572 / 0.467 / 0.003 / 0.104 | 0.611 / 0.699 / 0.623 / 0.000 / 0.093 |

- **The pelvis IMU makes novelty detection worse, not better**: latent-LOF AUROC drops from 0.974
  to 0.948, on every held-out mode except standing (ramp descent 0.979 → 0.950, stair descent
  0.983 → 0.968, transitions 0.882 → 0.822). Reducing the latent to 12/16 PCs recovers part of the
  gap (0.954 at PCA-16) but not all of it: the extra channels add variance the LOF neighbourhoods
  have to absorb without adding a direction that separates descending from ascending gait.
- **Regime probes get easier** (logistic ID accuracy 0.891 → 0.927, MLP 0.911 → 0.946): the
  pelvis signal helps name a *known* regime, which is a different question from flagging an unknown
  one, and consistent with the raw-space gain landing on level ground and ramp ascent.
- Reconstruction NLL moves from 0.49 to 0.61 but stays a non-detector; training ran the same
  length (best epoch 25 of 30 versus 24 of 29, 0.73 h versus 0.66 h).

Together with notebook 09 this settles the question the ablation was run for: the 16 thigh-and-
encoder channels stay the input, and the pelvis IMU is not worth the loss of comparable datasets.
