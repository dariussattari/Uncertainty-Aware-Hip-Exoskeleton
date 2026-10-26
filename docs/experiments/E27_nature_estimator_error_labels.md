# E27: OOD-safe against OOD-dangerous labels from the deployed estimator's error (plan step 3.2.3)

| Field | Value |
|---|---|
| ID | E27 |
| Date | 2026-10-13 |
| Dataset | Nature 2024 (phase 3, BT18–BT24) |
| Status | Complete |
| Sources | 12OCT-15OCT Verbose.docx §3.2 (3) and §9.0; README "Controller labels from the deployed estimator's error" |
| Code / notebooks | `ml/nature/estimator_labels.py` (`test_estimator_labels.py`); `notebooks/nature/data_exploration/04_estimator_labels.ipynb` |

## Hypothesis

"OOD-safe" should mean "where Young's estimator's error is acceptable", and that is measurable. Ranking the 26 tasks by
the estimator's per-task error on held-out people, and setting the boundary from data, should reproduce or correct the
hand-made labels of E18. The expected surprises were tug of war and push, because an IMU-only estimator cannot see an
external load.

## Method

- No model was trained. Every `Exo.csv` carries the deployed estimator's output (`torque_estimated`), which was scored
  against the biological hip moment for the 7 phase-3 participants never in its training data (BT18–BT24), in-task
  samples, both legs.
- **Rule written down before scoring:**
  - Compute the RMSE per task and person; a task's score is the mean over people.
  - The envelope is the worst score among the 4 familiar tasks.
  - An unfamiliar task at or below the envelope is OOD-safe; above it, OOD-dangerous.
  - A task is "near the boundary" if the participant-bootstrap 95% CI of its difference from the envelope contains 0.
- Also computed: relative error = RMSE / RMS(moment), and wrong direction = the share of samples with |moment| ≥ 0.1
  Nm/kg where the estimate has the opposite sign.

## Result

| Task | Provisional | From the rule | RMSE Nm/kg [95% CI] | Relative error | Wrong direction |
|---|---|---|---|---|---|
| Ball toss | dangerous | **dangerous** | 0.279 [0.252, 0.309] | 0.39 | 8.5% |
| Step-ups | dangerous | dangerous (near) | 0.240 [0.226, 0.257] | 0.44 | 13.5% |
| Cutting | safe | dangerous (near) | 0.240 [0.213, 0.268] | 0.35 | 5.0% |
| Lunges | safe | dangerous (near) | 0.234 [0.190, 0.297] | 0.37 | 8.6% |
| Tug of war | dangerous | safe (near) | 0.167 [0.133, 0.214] | 0.39 | 5.2% |
| Sit-to-stand | dangerous | safe (near) | 0.165 [0.134, 0.201] | 0.34 | 1.6% |
| Push | safe | safe | 0.115 [0.109, 0.121] | 0.47 | **11.9%** |
| Envelope: stairs (familiar) | | | 0.190 [0.169, 0.241] | 0.64 | 11.1% |

- **Only the ball toss is clearly above the envelope.** 8 of the 22 unfamiliar tasks, including all four label changes,
  sit near the boundary. Adding the three returning participants flips tug of war, jump and tire run to dangerous.
- Absolute error mostly measures the moment's size: Spearman 0.81 between RMSE and moment size, and −0.82 for relative
  error.
- Against switching off, the estimator wins on every task: relative error is 0.25–0.56 on unfamiliar tasks and
  0.31–0.64 on familiar ones.
- The surprise was in **direction**, not RMSE. Push has one of the lowest RMSEs but 11.9% wrong-direction assistance,
  second only to step-ups.

## Conclusion

The estimator's error supports one firm OOD-dangerous label: the ball toss. If sit-to-stand or tug of war should stay
dangerous, the reason is something the error cannot see (fall risk, external load, wrong-direction assistance), and the
thesis should say which. The deployed estimator had more sensors than this project's (hip and knee encoders; thigh, shank
and foot IMUs), so these labels are an optimistic bound. The decisive version is the same rule applied to this project's
own hip-and-thigh estimator. Until then the provisional labels stay in use, with the derived labels reported beside them.
