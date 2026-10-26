# E16: Do we need the pelvis IMU? (UMAP check and 22-channel encoder ablation)

| Field | Value |
|---|---|
| ID | E16 |
| Date | 2026-09-23 to 2026-09-24 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete |
| Sources | Live_Update.docx, "SEP 18 – 25" §3 ("Do we need Hip IMU data?"); `ml/sci_rob/awareness/README.md` ("Ablation: the pelvis IMU") |
| Code / notebooks | `notebooks/sci_rob/model_exploration/09_umap_all_tasks.ipynb` (thigh / pelvis / thigh+pelvis variant); `ml/sci_rob/awareness/pelvis_ablation.py`, `ml/sci_rob/LOTO/build_data.py --channels pelvis`; results `ml/sci_rob/awareness/results/DOWN/pelvis_ablation.*` |

## Hypothesis

The raw trials include a pelvis IMU that the 16-channel reference set omits. Adding its 6 channels might separate
ascending from descending movement, and ramps from stairs, better, and so improve novelty detection. The cost is real:
requiring a pelvis IMU limits which offline datasets can be used. So a cheap raw-space check came first, before
committing to retraining about 7 models.

## Method

1. **Raw-space check:** UMAP of the thigh-only, pelvis-only and thigh+pelvis windows. Leave-one-participant-out and
   leave-one-trial-out k-NN task accuracy in PCA-50, UMAP-2D and UMAP-3D, plus silhouette scores.
2. **Direct test:** the E14 awareness encoder (β = 0, the same settings, seed and LOF reference size) retrained on 22
   channels under the DOWN holdout, on identical windows. The 16 shared channels keep their scaler, and the pelvis
   channels get their own LG/RA/RD-fit scaler.

## Result

- **Raw space:** pelvis alone loses ascent/descent direction (SA and SD overlap). Thigh+pelvis separates ascending from
  descending more cleanly, but does not separate ramps from stairs.
- k-NN task accuracy (participant held out, PCA-50): thigh 0.600, pelvis 0.496, **thigh+pelvis 0.615**. Balanced
  accuracy in PCA: 0.708, 0.530, 0.707.
- **Encoder ablation** (test AUROC, DOWN holdout):

| Score | 16 ch: OOD vs ID / RD / SD / ST / TR | 22 ch: OOD vs ID / RD / SD / ST / TR |
|---|---|---|
| LOF, 32-D latent | **0.974** / 0.979 / 0.983 / 0.998 / 0.882 | 0.948 / 0.950 / 0.968 / 0.998 / 0.822 |
| LOF, PCA-12 | 0.957 / 0.953 / 0.983 / 0.998 / 0.838 | 0.947 / 0.948 / 0.965 / 0.999 / 0.843 |
| LOF, PCA-16 | 0.958 / 0.956 / 0.981 / 0.998 / 0.839 | 0.954 / 0.951 / 0.973 / 0.998 / 0.866 |
| Reconstruction NLL | 0.486 | 0.611 |

- Regime probes get easier with pelvis data: logistic accuracy 0.891 → 0.927, MLP 0.911 → 0.946.

## Conclusion

**The pelvis IMU makes novelty detection worse** (latent LOF 0.974 → 0.948, on every held-out mode except standing). It
slightly helps name a *known* regime (by about 0.03 in probe accuracy), which is a different question. The 16 thigh and
encoder channels remain the input, and the pelvis IMU is not worth losing comparable datasets.

## Caveats

- One seed.
- The UMAP check is qualitative.
- The ablation was run under DOWN only.
