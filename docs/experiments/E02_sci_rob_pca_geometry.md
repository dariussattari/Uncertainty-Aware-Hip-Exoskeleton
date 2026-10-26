# E02: Initial geometry of the data (PCA, distance and density scores)

| Field | Value |
|---|---|
| ID | E02 |
| Date | 2026-09-03 to 2026-09-04 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete |
| Sources | Live_Update.docx, "SEP 2 – 18" §1 ("Initial geometry of the data") and §2 ("Do we need machine learning?") |
| Code / notebooks | `notebooks/sci_rob/data_exploration/02_build_windows.ipynb`, `03_features_plot.ipynb` |

## Hypothesis

If held-out tasks lie far from the training windows in a low-dimensional linear projection, then simple distance or
density scores (distance from the PCA centroid, or LOF in 3-D PCA space) can flag unfamiliar movement without a learned
model.

## Method

- PCA fit on in-distribution (ID) training windows only. ID = level ground, ramp ascent and ramp descent.
- ID windows and held-out tasks (stairs, standing, transitions) were projected into that fixed basis.
- Distance from the centre and a 3-D LOF were inspected as novelty scores.

## Result

- PC1–PC3 capture 38.3% of the variance and mainly encode **position within the gait cycle**, not task identity.
- Standing collapses near the **centre** of the walking cloud, so distance-from-centre scores call the clearest held-out
  task normal.
- Decision in the summary table: PCA distance and 3-D LOF were rejected as primary scores. Raw window amplitude
  (AUROC 0.807, measured later in E06) was kept as a mandatory baseline.

## Conclusion

The problem is not finding points far from the training centroid. The choice of representation and the choice of score
both matter. This motivated learned representations (E05 onward) and, later, UMAP views (E13).
