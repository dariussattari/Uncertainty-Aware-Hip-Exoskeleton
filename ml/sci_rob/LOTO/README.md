# Leave-one-task-out

Every result so far holds out **participants**. Training saw level ground, ramp ascent and ramp
descent; validation and test participants contributed those same three modes as
in-distribution and four unseen ones — stair ascent, stair descent, standing, transitions — as
pseudo-OOD.

That protocol answers "can the model flag a task it has never seen, having seen only level
walking and ramps?" It cannot answer the sharper question: **can it flag a task whose nearest
neighbour it has already learned?** A model that has trained on stair *descent* and is then
shown stair *ascent* has far less surface to work with than one that has only ever seen flat
ground. Holding out one task at a time is what tests that.

| held out | trained on | what it asks |
|---|---|---|
| `SA` | LG RA RD **SD** ST TR | the hard case — stair descent is in training |
| `ST` | LG RA RD SA SD TR | the easy case, and the smallest (see the caveat below) |
| `TR` | LG RA RD SA SD ST | mode transitions, with every steady-state mode learned |

The participant split is untouched (seed 0, 12/4/4). A LOTO run and its participant-split
counterpart therefore differ in exactly one thing, which is the point.

## The blocker, and what was done about it

**The arrays under `data/processed/` cannot support this.** The in-distribution definition was
baked into the files rather than applied at load time:

```
data/processed/AE_GAN/{train,val,test}   LG, RA, RD only -- all three splits
data/processed/ood/{val,test}            SA, SD, ST, TR  -- val and test only
```

`02_build_windows.ipynb` cell 5 contains `if name == "train" and kind == "OOD": continue`, so
the twelve training participants have **no** SA/SD/ST/TR windows anywhere on disk. Holding out
stair ascent requires training on stair descent, standing and transitions — windows that were
never extracted.

So this is a re-extraction, not a re-split. The raw trials are still under `data/molinaro-raw`,
and `build_data.py` rebuilds them mode-complete into `data/processed_all/`:

```
data/processed_all/ensemble/{train,val,test}_{X,y,mask}.npy + _meta.parquet
data/processed_all/AE_GAN/{train,val,test}_X.npy           + _meta.parquet
```

Every split now holds all seven modes for its own participants, and nothing is labelled ID or
OOD on disk — `splits.py` applies the task partition at load time, which is what lets one build
serve three holdouts.

**The rebuild verifies against the old arrays.** Restricted to LG/RA/RD, each new split
reproduces the existing one window for window:

```
AE_GAN/train:   new ID-subset 187,993 vs existing 187,993   identical=True
AE_GAN/val:     new ID-subset  64,350 vs existing  64,350   identical=True
AE_GAN/test:    new ID-subset  61,956 vs existing  61,956   identical=True
ensemble/train: new ID-subset 168,956 vs existing 168,956   identical=True
ensemble/val:   new ID-subset  59,952 vs existing  59,952   identical=True
ensemble/test:  new ID-subset  58,045 vs existing  58,045   identical=True
```

The scaler is **copied** from `data/processed/`, not refit, so LOTO and participant-split runs
scale their inputs by identical numbers.

```bash
python ml/sci_rob/LOTO/build_data.py --dry-run     # counts, writes nothing
python ml/sci_rob/LOTO/build_data.py               # ~4 min, 9.9 GB  [already run]
python ml/sci_rob/LOTO/build_data.py --verify      # re-check against the existing arrays
```

## What the splits look like

`AE_GAN` family (the AE, GAN and label-free ensembles; the labelled `ensemble` family is
smaller by the gait-phase filter):

| held out | train | val ID | test ID | ood_val | ood_test |
|---|---|---|---|---|---|
| `SA` | 220,691 | 78,327 | 74,726 | 9,367 | 10,273 |
| `ST` | 248,074 | 86,452 | 84,378 | 1,242 | 621 |
| `TR` | 245,065 | 83,613 | 83,005 | 4,081 | 1,994 |

The OOD counts match the existing `data/processed/ood/` arrays exactly, so the held-out task is
scored on precisely the windows the participant-split runs scored it on.

Test is now **11–12% OOD** for `SA`, **2.3%** for `TR` and **0.7%** for `ST`, against 27.1% on
the participant split and 80.1% in the paper. Precision and F1 are even less comparable across
protocols than they already were; read **AUROC and the J-statistic**.

## Caveats worth knowing before reading any of this

**`ST` is thin, on both sides.** The pool has five standing trials in total — two train, two
val, one test. Holding it out removes essentially nothing from training, and the test set is
621 windows from a single trial of a single participant. It also already scored 100% flagged at
the participant-split operating point, so there is very little headroom to detect a change.
Treat the `ST` row as a consistency check, not a result.

**`TR` is thin on the training side.** Nine transition trials across twelve training
participants. When `SA` or `ST` is the holdout, `TR` contributes ~1.7% of training windows, so
"the model has seen transitions" is a weaker statement than the table suggests.

**`SD` is not among the holdouts.** It is the natural fourth run — 517 trials, the same size as
`SA`, and holding it out while `SA` stays in training is the mirror image of the `SA` row. Add
it with `HELD_OUT_TASKS` in `splits.py` and a `SD/` leaf if it turns out to be wanted.

**Gait phase does exist for standing and transitions.** Sampling the raw `gp.csv` files:
≥1 leg is labelled for 93.8% of `ST` samples and 90.4% of `TR` samples, against 41–46% for the
stair modes. So `Ensemble_GP` is not disadvantaged by the holdout the way one might expect —
the labelled-window filter bites hardest on stairs, which are in training.

## A plotting bug that produced no error — and the check that now guards it

**Symptom.** In the `−SA` run, standing and transitions were drawn orange (held out) although
that run trains on both, and the `−ST` reconstruction panel showed a walking trial where it
should have shown standing.

**Cause.** `plots_autoencoder.run()` and `plots_gan.run()` built their own data object —
`AutoencoderData()` / `GanData()` — which reads `data/processed/`, whose ID/OOD partition is the
participant-split one. The model, the windows and every Ψ value were correct; only the ID/OOD
*labelling* came from the wrong protocol. So the figures were plausible, the saved bundle
claimed 23,043 OOD windows where `eval_test.json` said 10,273, the ROC was drawn against the
wrong ground truth, and nothing raised. `plots_ensemble.run()` already took a `data_factory`
for exactly this reason; the other two had not been given one.

**What the metrics said.** `eval_test.json` was correct throughout — `cmd_eval` uses the data
object the registry built. Only `cmd_figures` was affected.

**Fix.** `data_factory` on all three plot modules, and `main.py:cmd_figures` now passes
`parts["data"]` to any figures function that accepts it — so figures are drawn against the same
builder that trained and evaluated the model, by construction rather than by convention. Plot
titles name the modes actually held out in that run, read from the labels. The standalone
`__main__` blocks in `evaluation/{autoencoder,ensemble,gan}.py` still build participant-split
data and now say so. `plots_comparison.py` took hardcoded participant-split run directories;
it now takes a root.

**Guard.** `verify.py` checks each leaf against the one fact that defines it — its directory
name:

```bash
python ml/sci_rob/LOTO/verify.py          # exit 1 if anything disagrees
```

It confirms the evaluation marks exactly the held-out task as OOD, that the window counts match
the rebuilt arrays, that the saved score bundle agrees with the metrics window for window, and
that no figure predates the bundle it should have come from. Re-planting the original bug in a
copied leaf makes it fail with all three relevant messages, so it is known to fire.

## What is being run where

The autoencoder and the GAN run locally — single stage, no LOSO. The four models that use the
12-fold leave-one-subject-out stage are **deferred to the HPC cluster**: that is what the SLURM
job array in `ml/sci_rob/prob_ensemble/slurm/` exists to parallelise, and on one laptop each is a 20-30 h
column against the slowest single fold on the cluster. `run.py --all` skips them and says so;
`--include-hpc` overrides.

| model | where | status |
|---|---|---|
| `vanilla/AE` | local | done, ~0.09 h/cell |
| `vanilla/GAN` | local | running, ~2.9 h/cell |
| `vanilla/Ensemble_GP` | cluster | waiting on access |
| `vanilla/Ensemble_synthetic` | cluster | waiting on access |
| `vanilla/Ensemble_forecast` | cluster | waiting on access |
| `prob_ensemble` | cluster | waiting on access |

## Running it

```bash
python ml/sci_rob/LOTO/run.py --list                               # the matrix and what is done
python ml/sci_rob/LOTO/run.py --held-out SA --model vanilla/AE     # one cell
python ml/sci_rob/LOTO/run.py --held-out SA --model all            # one column
python ml/sci_rob/LOTO/run.py --all                                # all eighteen
```

Add `--stage smoke` for a two-epoch wiring check on a vanilla model, and any extra flags
(`--folds 3`, `--max-epochs 50`) pass straight through to the underlying CLI.

Nothing is reimplemented. `ml/sci_rob/vanilla/main.py` and `ml/sci_rob/prob_ensemble/{train,evaluate,plots}.py`
are driven exactly as they are on the participant split; the one substitution is the data
object. `run.py` rebinds the registry's `data` factory (vanilla) or `data.build`
(prob_ensemble) to `splits.build(...)`, which hands back a normally-constructed data object
whose splits have been swapped for task-restricted views. Downstream, nothing knows a holdout
happened.

**Cost.** Eighteen runs, roughly **124 h sequentially on this laptop** — 5.2 days. Per held-out
task it is about 41 h:

| model | ~h | why |
|---|---|---|
| `vanilla/AE` | 0.5 | single stage, ~42 epochs |
| `vanilla/GAN` | 6 | 500 fixed epochs, Table IV |
| `vanilla/Ensemble_GP` | 8 | 12 LOSO folds + stage 2 |
| `vanilla/Ensemble_forecast` | 9 | as above, more epochs to converge |
| `vanilla/Ensemble_synthetic` | 10 | as above, the slowest to converge |
| `prob_ensemble` | 8 | 12 folds + stage 2, β-NLL |

Estimates scale the measured participant-split epoch counts by the ~1.3× larger training set.
`prob_ensemble`'s twelve folds are independent and already have SLURM scripts — on the cluster
that column collapses to the slowest single fold, and it is the obvious thing to move there
first.

## Running it on the cluster (FASRC Cannon)

Eighteen cells is ~108 h sequentially and about one cell's wall clock as four SLURM arrays.

```bash
# on the cluster, once
bash  ml/sci_rob/LOTO/slurm/setup.sh          # venv at ~/.venvs/hipexo (shared with prob_ensemble)
bash  ml/sci_rob/LOTO/slurm/fetch_data.sh     # pulls the raw data from Zenodo and rebuilds the arrays
sbatch ml/sci_rob/LOTO/slurm/smoke.sbatch     # gpu_test: CUDA, splits, one real cell, two wiring checks
bash  ml/sci_rob/LOTO/slurm/submit.sh         # the matrix

# on the laptop, repeatedly -- safe while the cluster is still working
bash  ml/sci_rob/LOTO/slurm/collect.sh        # pull, verify, summarise, write the docx section
```

| array | jobs | what |
|---|---|---|
| `single` | 6 | AE and GAN × 3 tasks — single stage |
| `vanilla_ens` | 9 | the three vanilla ensembles × 3 tasks; each runs its own 12 LOSO folds in-process |
| `prob_folds` | 36 | `prob_ensemble` stage 1, one fold per task — the only model with a per-fold entry point |
| `prob_final` | 3 | stage 2 + eval + figures, `afterok` on the folds |

### Four things that were measured, not assumed

**Move the recipe, not the result.** Uploading the 9.9 GB of built arrays ran at ~0.3 MB/s and
stalled at 4.7 GB. Zenodo → cluster runs at ~24 MB/s, so `fetch_data.sh` downloads the twenty
raw participants there (~3 min) and rebuilds (~4 min). The rebuilt arrays were confirmed
bit-identical to the laptop's by hashing the same sampled rows on both machines.

**Stage to node-local disk.** Training memory-maps the windows and gathers ~1,000 random rows
per batch. On `$HOME` (NFS) that runs at **4.8 MB/s** — latency-bound, one round trip per
window — and an autoencoder epoch took **221.8 s** against 8 s on a laptop SSD, with the GPU
idle throughout. `/tmp` on a compute node is a 174 GB local disk; staging costs one sequential
copy and brings the epoch back to **8.4 s**. `$HIPEXO_STAGE_FAMILY` limits the copy to the
family a job actually reads.

**TF32 is off.** On an A100 it cuts an autoencoder epoch from 16.4 s to 6.6 s, but it truncates
the mantissa of every convolution and these runs are meant to be read against participant-split
runs trained in full precision. `$HIPEXO_TF32=1` enables it and says so loudly.
`cudnn.benchmark` is on: it changes no arithmetic.

**A cell with an `eval_test.json` is skipped.** That makes the arrays restartable and requeue
safe, and it also means pushing laptop results to the cluster silently prevents those cells from
running there. Pass `--force` to override. (This bit once: three AE cells "finished" in seconds
because their results had been rsynced up beforehand.)

## Files

| file | what it holds |
|---|---|
| `build_data.py` | the mode-complete re-extraction, with the verification against `data/processed/` |
| `splits.py` | task-restricted views over the shared memmaps — no copying |
| `run.py` | dispatch; drives the existing CLIs with the data object substituted |
| `verify.py` | the guard described above — run it after any figures regeneration |
| `summarize.py` | collects every finished leaf into `summary_<split>.md/.csv` plus two figures |
| `timings.json` | measured hours per cell, written as runs finish; `--list` prefers it to the estimates |
| `write_update.py` | replaces the leave-one-task-out section in `Live_Update.docx`, idempotently |
| `slurm/fetch_data.sh` | downloads the raw data on the cluster and rebuilds the arrays there |
| `slurm/collect.sh` | laptop side: pull → verify → summarise → write the doc section |
| `slurm/{single,vanilla_ens,prob_folds,prob_final,smoke}.sbatch` | the arrays |
| `<TASK>/<model>/` | one run's `final.pt`, `report.json`, `eval_<split>.json`, `figures/` |
