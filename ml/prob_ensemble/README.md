# Probabilistic ensemble — aleatoric and epistemic uncertainty, separated

`ml/vanilla/` reproduces Tourk et al. with the paper's hyperparameters. This package is the
first deliberate departure from them, which is why it lives in its own directory.

## What changes, and what does not

| | Experiment 5 (`ml/vanilla`) | this package |
|---|---|---|
| target | hip angle, both legs, 200 ms ahead | **identical** |
| trunk | 7 × TCN, kernel 20, dilations (1,2,8) | **identical** (imported, not copied) |
| protocol | two-stage LOSO, patience 10 | **identical** |
| filtering, threshold rule | causal median ×10, 99.5th pct | **identical** |
| output head | `Linear(30, 2)` → mean | `Linear(30, 4)` → mean **and log-variance** |
| loss | masked MSE | masked β-NLL, with MSE warm-up |
| score | variance across branches | **aleatoric / epistemic / total**, scored separately |

Only the head, the loss and the score differ. Everything else is imported from `ml/vanilla`
rather than forked, because the comparison is the point: any difference in result has to be
attributable to the loss, not to a drifted data pipeline.

## Why bother

Experiment 5's score — variance across branches — estimates **epistemic** uncertainty: the
branches disagree where training did not pin them down. It cannot represent **aleatoric**
uncertainty, and the future hip angle is genuinely stochastic even on familiar ground, because
heel-strike timing and step-to-step variability are not recoverable from the preceding second
however much the model knows.

That conflation matters for a controller. A window can be unpredictable because the movement is
unfamiliar (reduce assistance) or because gait is inherently variable at that instant (do not).
Experiment 5 cannot tell those apart. Treating the ensemble as a uniform mixture of Gaussians
splits them without approximation (Lakshminarayanan et al., 2017):

```
Var[y] = (1/M) Σ σᵢ²   +   (1/M) Σ (μᵢ - μ̄)²
         └─ aleatoric ─┘   └─── epistemic ───┘
```

**The epistemic term is Experiment 5's score, exactly.** Same target, same trunk, same
protocol — so it should land near Experiment 5's AUROC of 0.896. That is a built-in correctness
check with a known answer, in the same spirit as the Krogh–Vedelsby identity residual: a large
gap means the implementation is wrong before the idea is. `evaluate.py` prints the difference.

## Running it locally

```bash
source .venv/bin/activate
cd ml/prob_ensemble

python data.py                                    # confirm data + folds resolve
python train.py fold --index 0                    # one LOSO fold
python train.py collect                           # aggregate (needs all 12)
python train.py final                             # stage 2 + threshold calibration
python evaluate.py --split test
python plots.py --split test                      # figures + summary.md
```

Sequentially, stage 1 is twelve folds at roughly 68 s per epoch and ~25 epochs per fold — about
five to six hours. That is the reason for the cluster.

## Running it on the cluster

Stage 1's folds are independent, so they run as a **SLURM job array** and stage 1 collapses
from ~5 hours to the slowest single fold. Each task writes only its own
`runs/prob_forecast/loso/fold_NN_SUBJECT.json`, so there is no shared-file race; `ml/vanilla`
appends all folds to one JSON, which is correct on one machine and would lose results here.

Every stage is idempotent: a fold whose file exists exits immediately. That makes
`--requeue` on a preemptible partition nearly free — a preempted fold costs at most its own
runtime.

### One-time setup

```bash
# from your laptop: code, then data (~4.3 GB)
scp -r ml data_exploration README.md dsattari@boslogin.rc.fas.harvard.edu:~/Uncertainty-Aware-Hip-Exoskeleton/
scp -r data/processed/AE_GAN data/processed/ood dsattari@boslogin.rc.fas.harvard.edu:~/hipexo_data/
```

Then on the cluster, edit the three `EDIT ME` lines at the top of `slurm/env.sh`
(`HIPEXO_ACCOUNT`, `HIPEXO_REPO`, `HIPEXO_DATA`) and run:

```bash
cd ~/Uncertainty-Aware-Hip-Exoskeleton/ml/prob_ensemble
bash slurm/setup.sh          # login node; builds the venv, installs CUDA-matched torch
sbatch slurm/smoke.sbatch    # ~15 min on gpu_test: proves CUDA, data, and all four stages
bash  slurm/submit.sh        # the real run: array + dependent stage 2
```

`submit.sh --requeue` uses the preemptible partition instead.

### Watching it

```bash
squeue -u $USER
ls runs/prob_forecast/loso/                       # one file per completed fold
tail -f slurm/logs/loso_<jobid>_0.out
```

### Getting results back

```bash
scp -r dsattari@boslogin.rc.fas.harvard.edu:~/Uncertainty-Aware-Hip-Exoskeleton/ml/prob_ensemble/runs/prob_forecast ./ml/prob_ensemble/runs/
```

`final.sbatch` prints this line with your paths filled in when it finishes.

## Things worth knowing before you read the output

**β-NLL, not plain NLL.** Gaussian NLL scales the mean's gradient by `1/σ²`, so a network can
reduce its loss more cheaply by inflating log-variance on hard samples than by fitting them —
and then those samples stop contributing gradient. The result is well-calibrated variance
around a worse mean than MSE would have produced. β-NLL (Seitzer et al., 2022) reweights each
sample by `detach(σ²)^β`; at β=1 the `1/σ²` cancels exactly. Default β=0.5, plus three warm-up
epochs of plain MSE. `losses.py` documents this and its self-test verifies that β=1 reproduces
half the MSE gradient to zero error.

**Watch `z_var`.** It is the variance of the standardised residual `(y − μ)/σ`, and a
calibrated model gives exactly **1.0**. Above 1 is over-confident, below 1 over-cautious. It is
logged every epoch and is the most informative single number here, because unlike AUROC it has
a known correct answer. In local testing it converged 0.918 → 0.980 → 0.986 on training data
once NLL engaged, while MSE kept falling — which is what "the variance head is learning and
not cannibalising the mean" looks like.

**Four scores are evaluated, not one.** `epistemic` is primary and is the comparison anchor;
`aleatoric` and `total` are reported alongside; `ratio = epistemic/aleatoric` is screened
because the analogous Krogh–Vedelsby quantity `A/(E+ε)` scored 0.320 — worse than chance — and
the prior is that this one fails too. Reporting it either way keeps the record honest.

**The held-out calibration may not match the training calibration.** In local testing the
training `z_var` reached 0.986 while the held-out participant's was 1.93 — that is, the
aleatoric head is roughly twice over-confident on unseen participants. If that survives the
full run it is a finding, not a bug, and it belongs in the writeup: predicted variance
transfers across participants worse than predicted mean does.

## Files

| file | what it holds |
|---|---|
| `paths.py` | repo/data resolution, puts `ml/vanilla` on the path. `$HIPEXO_DATA` overrides |
| `prob_models.py` | `ProbabilisticEnsemble`, the decomposition. Named to avoid shadowing vanilla's `models/` |
| `losses.py` | masked β-NLL, MSE warm-up, calibration diagnostics. Self-testing |
| `data.py` | Experiment 5's dataset, target and horizon pinned |
| `train.py` | `fold` / `collect` / `final` — the array-safe split of the two-stage protocol |
| `evaluate.py` | the paper's procedure over all four candidate scores |
| `plots.py` | seven figures plus `summary.md`; run after `evaluate.py` |
| `slurm/env.sh` | modules, venv, data staging. **Edit the three `EDIT ME` lines** |
| `slurm/setup.sh` | one-time venv build with CUDA-matched torch |
| `slurm/smoke.sbatch` | cheap end-to-end check on `gpu_test` |
| `slurm/loso_array.sbatch` | stage 1 as a 12-task array |
| `slurm/final.sbatch` | stage 2, dependent on the array |
| `slurm/submit.sh` | submits the chain |

## Caveats to verify on first contact

The module names in `slurm/env.sh` (`python/3.10.13-fasrc01`, `cuda/12.4.1-fasrc01`) and the
partition names (`gpu`, `gpu_test`, `gpu_requeue`) are written for FASRC Cannon but **do drift
between maintenance windows**. Each `module load` falls back to the unversioned name, and
`smoke.sbatch` fails loudly with a specific message if torch cannot see CUDA. Confirm once
with `module avail python`, `module avail cuda`, and
`sacctmgr show assoc where user=$USER format=account,partition%30`.

`HIPEXO_STAGE=1` copies the arrays to node-local disk before training, which is worth it from a
home or networked filesystem since training memory-maps and re-reads them every epoch. On fast
parallel scratch it is a wasted copy. It is off by default.
