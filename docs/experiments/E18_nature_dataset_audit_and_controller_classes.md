# E18: Nature dataset audit, and binning its 26 tasks into controller classes

| Field | Value |
|---|---|
| ID | E18 |
| Date | 2026-09-25 to 2026-10-02 |
| Dataset | Molinaro, Scherpereel et al., *Nature* 2024 ("Task-agnostic exoskeleton control"), Complete variant: `Phase1And2_Complete.zip` (25.3 GB), `Phase3_Complete.zip` (11.7 GB) |
| Status | Complete |
| Sources | Live_Update.docx, "SEP 25 – OCT 2" §1–2; README "Data" section |
| Code / notebooks | `ml/nature/nature_data.py`; `notebooks/nature/data_exploration/01_dataset.ipynb` |

## Hypothesis

This was descriptive. The question was whether the Nature dataset can replace the Science Robotics dataset as the basis
for a hip joint-moment estimator and forecaster. It would need hip encoder, thigh IMU, task labels and the biological hip
moment (with the exo moment subtracted), and it should cover enough task variety, transitions included, to define
ID-safe, OOD-safe and OOD-dangerous classes.

## Method

- Downloaded the two Complete packages, unzipped them to `data/nature/phase1-2` and `data/nature/phase3`, and verified
  them against the zips' own file index (every file, size and CRC32).
- Audited channels against the Science Robotics pipeline, and audited the assisted and unassisted recordings.
- Compared hip-moment distributions with the Science Robotics data, and found which raw channel tracks the moment within
  each task group.
- Binned each of the 26 tasks into a controller class with a reason. The per-task R² and RMSE of the joint-moment
  estimator in the binning table are the Nature (JME) paper's reported results: OOD-safe and OOD-dangerous follow where
  that paper reported its estimator working well or badly per task (confirmed 2026-10-22).

## Result

- **Usable with a one-time build:**
  - Recompute encoder velocity (the published channel is half scale for most sessions).
  - Remap the IMU axes (pipeline (x, y, z) = Nature (y, z, x)).
  - Recover stair and incline direction from joint power.
  - Mask each leg by its own activity flag.
  - BT04 was never published. BT01, BT02 and BT13 appear in both phases (split by person).
- **Powered against unpowered:** the 95th percentile of |measured hip torque| per recording separates cleanly (a gap
  near 1.5 Nm). Phase 1–2 has a large unpowered group; phase 3 is all powered. An open question for the advisor was
  whether unassisted data should be ID training data or a separate class (an external check that the controller is
  commanding).
- **Moment spread:** the Nature data has a larger hip-moment spread. `normal_walk` and Science Robotics LG are
  essentially the same task.
- **What drives the moment** (correlation rank by group): hip angle comes first in every group (walking 0.82; stairs,
  curbs and step-ups 0.77; squat, lift, sit and lunge 0.71; jump, run, cut and shuffle 0.63; standing and unstructured
  0.56). Motion drives the moment in cyclic and vigorous tasks; posture (thigh inclination) drives it in squat-type
  tasks.
- **Provisional controller classes** (the controller each class would get):
  - **ID-safe → spline:** normal walk, incline walk, stairs, weighted walk. **5.0 h.**
  - **OOD-safe → joint-moment estimator:** obstacle walk, dynamic walk, tire run, walk backward, side shuffle,
    curb up/down, start-stop, turn-and-step, meander, lift weight, squats, lunges, jump, cutting, push, poses, twister.
    **17.0 h.**
  - **OOD-dangerous → exo off:** ball toss (largest error, R² 0.66 / RMSE 0.32), step-ups (0.69 / 0.26), tug of war
    (0.58 / 0.21, external force the sensors cannot see), sit-to-stand (0.57 / 0.19). **4.0 h.**
  - A fourth class, **ID-dangerous** (fault injection), would reuse the Science Robotics fault toolbox (E15).

## Conclusion

The Nature dataset is adopted as the main dataset. It has the channels, 26 tasks (about 26 h) and natural ID-safe,
OOD-safe and OOD-dangerous classes, which map to the spline, Young's joint-moment estimator and exo-off. The four-task
danger label is provisional; it is tested against the estimator's measured error in E27. Hip angle is the dominant
signal for movement regime.
