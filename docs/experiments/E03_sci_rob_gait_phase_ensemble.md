# E03: Gait-phase TCN ensemble, a reproduction of the reference estimator (Methods "Experiment 1")

| Field | Value |
|---|---|
| ID | E03 |
| Date | 2026-09-04 to 2026-09-05 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete |
| Sources | Methods_Experiments_Reproduction.docx §1–11 and §15; Live_Update.docx, "SEP 2 – 18" §4–5 |
| Code / notebooks | `ml/sci_rob/vanilla/` (`main.py` CLI, `dataset.py`, `targets.py`, `training/ensemble.py`, `evaluation/ensemble.py`, `paper_spec.py`), `ml/common/models/ensemble.py`; `notebooks/sci_rob/model_exploration/01_gait_phase_ensemble.ipynb` |

## Hypothesis

The best uncertainty estimator of Tourk et al. (an ensemble of 7 gait-phase TCNs, scored by member disagreement)
transfers from ankle to hip data. Trained only on ID locomotion, its members should agree on familiar windows and
diverge on held-out ambulation modes. The experiment also validates the data and training pipeline end to end and sets
the baseline that later work is measured against.

## Method

- **Inputs:** 16 channels (thigh accelerometer xyz, thigh gyro xyz, `enc_angle`, `enc_velo`, per leg), 1 s windows (200
  samples), stride 10. A window is kept only if it has no NaN. Standardization is fit on the training split only.
- **Splits (participant-disjoint, seeded):** 12 train (ID only: LG, RA, RD; 168,956 windows), 4 validation and 4 test
  (ID plus OOD: SA, SD, ST, TR). The test set is 28.4% OOD, against 80.1% in the reference.
- **Target:** sin(2π · gait phase) per leg at the window's last sample. A per-leg mask is applied inside the loss,
  because the unlabelled leg's placeholder of 0 would otherwise read as mid-stance.
- **Model:** 7 independently initialized TCNs, each with 3 dilated causal conv layers (30 filters, kernel 20, dilations
  (1, 2, 8), receptive field 210 samples), BatchNorm and ReLU, and two tanh heads. 321,524 parameters.
- **Training:** stage 1 is leave-one-subject-out over the 12 training participants (patience 10, cap 60), which gave a
  mean best epoch of 17.0 (range 7–30). Stage 2 retrains on all 12 for 17 epochs. Adam, lr 1e-3, batch 1,024.
- **Score:** $\Psi(x) = [\mathrm{Var}(l_1..l_7) + \mathrm{Var}(r_1..r_7)]/2$ (population variance). Causal median over
  10 consecutive scores (0.5 s), applied within a trial. Threshold = 99.5th percentile of training Ψ (1.817e-3).
- **Declared deviations (6 of 21 parameters):** 200 Hz and 200 samples (forced by the data), dilations inferred (the
  reference does not state them), and a 12/4/4 split instead of 9/3/3. A machine-checked spec file
  (`paper_spec.py`) fails on any undeclared mismatch.

## Result

| Metric | This work (hip) | Reference (ankle) |
|---|---|---|
| Youden J | 37.5 | 92.3 |
| AUROC | 0.859 | 0.993 |
| Recall / specificity | 39.4% / 98.1% | 96.2% / – |
| Precision / F1 | 89.4% / 54.7% | 93.0% / 90.3% |
| ECE / Brier | 0.048 / 0.133 | 0.03 / 0.03 |

Share of each task's windows flagged (for ID tasks, lower is better): LG 1.8%, RA 0.5%, RD 3.8%; **SA 6.6%**,
SD 66.8%, ST 100%, TR 49.9%. Stair ascent makes up 44.6% of OOD windows. Without it, recall would be about 66%.

## Conclusion

The reproduction works and is the strongest of the four reference architectures here (see E10). It has one blind spot,
**stair ascent**, whose hip kinematics resemble ramp ascent, and that blind spot dominates the aggregate. Its main
practical limit is that gait phase is reconstructed offline from force plates, so the target is unavailable on a
deployed exoskeleton. That limit motivates the live targets of E07 and E08. Only AUROC and J are comparable with the
reference, because OOD prevalence differs (28% here against 80%).

## Caveats

- The OOD set is held-out *locomotion*, not the reference's non-cyclic activities (sitting, jumping, lying).
- All training data were collected under the assisting controller. There is no unassisted condition.
- Ensemble diversity comes from initialization only.
- Standing (621 windows) and transitions (1,994) come from few participants.
- Single seed.
