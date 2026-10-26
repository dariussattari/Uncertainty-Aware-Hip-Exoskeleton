# E36: The ankle paper's gait-phase ensemble as a baseline on the Nature data: anomaly detection is not regime selection

| Field | Value |
|---|---|
| ID | E36 |
| Date | 2026-10-18 |
| Dataset | Nature 2024 |
| Status | Complete |
| Sources | 12OCT-15OCT Verbose.docx §16.0; Live_Update.docx "OCT 10 – 16" (supporting table row); README "Baseline: the ankle paper's gait-phase ensemble" |
| Code / notebooks | `ml/nature/awareness/ensemble_data.py`, `train_ensemble.py`, `slurm/train_ensemble.sbatch`, `test_ensemble.py`, `ml/common/models/ensemble.py`; `notebooks/nature/model_exploration/onboard/10_gait_phase_ensemble_baseline.ipynb`, `_ensemble_baseline.py`; runs `runs/ensemble/`; results `onboard/results/10_*.csv` |

## Hypothesis

Every Nature awareness model had been judged only against the others. Tourk et al.'s best estimator (7 gait-phase TCNs,
disagreement Ψ, F1 90.3) makes a single actuate/do-not-actuate decision. The hypotheses:

- (a) Ψ is a sound unfamiliar-movement alarm on hip data.
- (b) One anomaly scale *cannot* make the three-way controller choice, because gait phase is undefined off the cycle
  (the paper's own stated limit). Ψ should therefore rank tasks by distance from a gait cycle, not by danger.

## Method

- **Targets:** the Nature data publishes no gait phase, so heel strikes were read from the insoles' vertical force.
  Contact above 50 N, released below 20 N; bounces under 0.1 s dropped; strides of 0.6–2.5 s kept. Phase = elapsed time
  over stride duration, undefined elsewhere. Coverage: 96–98% of walking windows, 81% of stairs, 2–16% of non-cyclic
  and dangerous tasks.
- **Model:** the Science Robotics reproduction's class (E03: 3 causal layers, 30 filters, kernel 20, dilations 1/2/8,
  two tanh heads predicting sin 2π phase) on 1 s raw windows with the AEs' standardization. Trained on ID-safe windows
  with masked MSE, Adam 1e-3, batch 1,024, bf16.
- **Two protocols, three seeds each:** `val` (early stopping on validation participants) and `loso` (the paper's
  two-stage procedure, 7–8 epochs). Six runs, 3 min (`val`) and 15 min (`loso`) each, on 1 core and 1.2 GB.
- **Scoring:** 0.5 s causal median by recording. The paper's threshold, plus a 1% false-alarm point on validation, plus
  two regime cutoffs chosen on validation by macro F1. Test scored once. Compared against E35's *u*.

## Result

- Phase estimation works on cyclic tasks: RMSE of sin phase 0.15–0.18 on walking and 0.20 on stairs. The two protocols
  agree to the second decimal.

| Score | AUROC unfamiliar | AUROC dangerous | Unfamiliar flagged | Dangerous flagged | False alarms |
|---|---|---|---|---|---|
| Ψ, paper threshold | 0.882 ± 0.002 | **0.992** | 62% | 93% | 0.5% |
| Ψ, 1% FA on validation | 0.882 | 0.992 | 71% | 96% | 1.9% |
| LSTM-VAE 3 s raw, LOF (E31) | **0.963** | 0.993 | | | |

| Three-way task | ID-safe vs OOD-safe | OOD-safe vs dangerous | Macro F1 | Dangerous given spline | Safe switched off |
|---|---|---|---|---|---|
| Ψ (val) | 0.863 | 0.775 | 0.60 ± 0.02 | 0.5% | 14% |
| Ψ (loso) | 0.860 | 0.779 | 0.61 ± 0.02 | 0.5% | 14% |
| **Regime classifier *u* (E35)** | **0.981** | **0.965** | **0.868** | **0%** | **2.1%** |

- ID-safe and OOD-safe overlap over two decades of Ψ, and OOD-safe and dangerous over the top decade. Only 54% of
  dangerous windows are recognized.
- Per task, Ψ orders by **distance from the gait cycle**: Spearman −0.80 between the task median and the share of
  windows with a stride. Ball toss, sit-to-stand and tug of war top the list, interleaved with squats, lifting, jump and
  lunges (all OOD-safe). Step-ups sit mid-range.
- Ψ and *u* rank tasks alike (Spearman 0.84): both agree on what is unusual. *u* adds the regime axis.

## Conclusion

The ankle paper's estimator is a sound actuate/do-not-actuate alarm on hip data. On dangerous movement it is as good as
the best novelty detector (0.992), though it is weaker on unfamiliar movement overall (0.88 against 0.963). **It is not a
regime selector, and no single novelty scale is.** The OOD-safe band is a property of the estimator, not of novelty.
That is why the supervised regime classifier (E34) carries the three-way decision and the novelty detector keeps the
unfamiliar-movement alarm.
