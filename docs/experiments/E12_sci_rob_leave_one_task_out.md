# E12: Leave-one-task-out: does any score respond to familiarity itself? (Methods "Experiment 8")

| Field | Value |
|---|---|
| ID | E12 |
| Date | 2026-09-19 to 2026-09-22 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete (3 of 4 possible holdouts; stair descent not run) |
| Sources | Methods_Experiments_Reproduction.docx §34–37; Live_Update.docx, "SEP 18 – 25" §2 |
| Code / notebooks | `ml/sci_rob/LOTO/` (`splits.py`, `build_data.py`, `slurm/`); `notebooks/sci_rob/model_exploration/_loto_exploration.py` and each model notebook's LOTO section |

## Hypothesis

The participant split never puts a task on both sides of the comparison. It therefore cannot tell "this task is
unfamiliar to the model" from "this task simply produces an unusual signal". If one task is held out at a time and the
model trains on the other six, a score that measures **unfamiliarity** must be higher for a task when it is withheld
than when it is trained on. A score that measures an intrinsic signal property will not change.

## Method

- **Holdouts:** SA (trained on LG RA RD SD ST TR; the hard case, because its neighbour SD is in training), ST (trained
  on LG RA RD SA SD TR) and TR (trained on LG RA RD SA SD ST). Participants as before (seed 0, 12/4/4).
- **Data rebuild:** the stored arrays had the ID definition baked in, so the training participants had no
  stair, standing or transition windows. The arrays were re-extracted mode-complete, with the partition applied at load
  time and the original scaler copied over. Restricted to LG/RA/RD, each split reproduces the original arrays window for
  window, and a cluster rebuild was bit-identical by hash.
- **Implementation:** a row-index view over the shared memmap, so no model, trainer or evaluator changed.
- **Execution:** 18 cells (6 models × 3 holdouts) as 4 job arrays on FASRC Cannon. Arrays were staged to node-local
  disk (221.8 → 8.4 s per AE epoch). TF32 was left off to match full-precision baselines.
- **Fault found during verification:** three figure modules built their own data object with the old partition, so the
  figures (only the figures) showed trained tasks as held out. Fixed by injecting the builder. A verification script now
  checks each result directory against its name.

## Result

AUROC for the held-out task:

| Model | Participant split | Hold out SA | Hold out ST | Hold out TR |
|---|---|---|---|---|
| AE + LOF (E05) | 0.782 | 0.712 | **0.991** | **0.876** |
| GAN (E06) | 0.727 | 0.534 | 0.982 | 0.777 |
| Gait-phase ensemble (E03) | 0.859 | 0.698 | 0.236 | 0.607 |
| Correlation ensemble (E07) | 0.813 | 0.626 | 0.722 | 0.775 |
| Forecast ensemble (E08) | 0.896 | 0.827 | 0.320 | 0.649 |
| Probabilistic, epistemic (E11) | 0.914 | **0.840** | 0.480 | 0.509 |

Median score when withheld against when trained on (sign of change):

- **Transitions:** higher when withheld in all 6 models (+1% to +53%). This is a genuine, modest novelty response.
- **Standing:** **lower** when withheld in all 6 models (−3% to −81%). Training on standing forces disagreement about
  it (phase is undefined, little to forecast) and gives the AE a latent region of its own.
- **Stair ascent:** mixed (3 models up, 3 down).
- These changes are small next to the 2–20× spread between tasks within any single run.

Other observations:

- Rankings are model-specific and nearly opposite. The AE and GAN put standing at the top (unusual, near-static
  signal). The forecasters put standing at the bottom (trivial to predict) and stairs at the top.
- Standing's epistemic score fell from 0.036 to 0.004–0.006 once stairs and transitions were in training, for both
  withheld and trained standing.
- The inherited 99.5th-percentile threshold does not transfer. Some runs flag nothing, and some have negative J.

## Conclusion

**No method wins every held-out task, and none of the scores materially responds to familiarity.** A run's AUROC is
set by where the withheld task sits in the score's fixed ranking of intrinsic signal properties. This strengthens the
amplitude findings of E06 and E08. The evidence does not validate a general safety or novelty detector. Complementary
scores should be kept, and tested on whether they predict actual forecast failures, not just task identity.

## Caveats

- Standing has 5 trials in total. The ST holdout is scored on 621 windows from one trial of one participant, so treat
  it as a consistency check.
- Transitions are under 2% of training windows.
- The SD holdout was not run.
- OOD prevalence is 0.7–12%, so only AUROC and best J compare across protocols.
