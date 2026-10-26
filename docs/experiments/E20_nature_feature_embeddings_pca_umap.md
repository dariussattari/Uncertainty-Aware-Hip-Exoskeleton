# E20: Feature families in PCA and UMAP space (k-NN separability)

| Field | Value |
|---|---|
| ID | E20 |
| Date | 2026-10-05 to 2026-10-08 |
| Dataset | Nature 2024 |
| Status | Complete (rerun after the E21 fix) |
| Sources | Live_Update.docx, "OCT 5 – 10" §1 (second half) |
| Code / notebooks | `notebooks/nature/data_exploration/03_feature_embeddings.ipynb`; `ml/common/features.py` |

## Hypothesis

If a feature family genuinely structures the data, adding it should make task families and controller classes cluster
more cleanly across people in low-dimensional embeddings (PCA, UMAP), not only in a boosted tree's full feature space
(E19). This mirrors the UMAP analysis done on the Science Robotics data (E13).

## Method

- k-NN balanced accuracy for task family and for controller class in three spaces: the full feature space, PCA 3-D and
  UMAP 3-D. Each feature set was compared against raw 1 s summaries (add one family, drop one family, 3 s raw, 3 s raw
  + spectral).
- PCA axes were inspected for what each component encodes.

## Result

- Baseline (raw 1 s): task family 0.855 full / 0.660 PCA / 0.807 UMAP; controller class 0.808 / 0.577 / 0.726.
- **Raw 3 s + spectral** gives the largest gains: task family +0.06 full / +0.08 PCA / +0.11 UMAP; controller class
  +0.06 / +0.11 / +0.14.

| Family | Importance in PCA/UMAP | Agrees with the boosted trees (E19)? |
|---|---|---|
| Spectral | High: +3.5 (full) and +4.4 (UMAP-3D) k-NN points for task family. Dropping it costs about the same. It owns the rhythm axis (PC5) | Yes, and it adds more beyond 3 s context here (+3 against under 1) |
| Polar phase | Low: +0.6 / +0.5; shares PC4–PC5 | Yes. Useful as a phase signal, not for separation |
| Orientation | Low: +0.6 / +0.7; part of the posture axis (PC2) | Yes |
| Jerk / HF | None, slightly negative. Adds to the vigour axis (PC1) that raw spread already spans, and blurs UMAP | Yes |

- **Controller classes form no clusters in any space.**
- After the E21 fix, the spectral gain became +2.9 / +3.2 points instead of +3.5 / +4.4.

## Conclusion

The ranking from E19 holds in embedding space. Raw 3 s data plus spectral properties is the best recipe for
classification, and spectral is the only family that makes task families cluster more cleanly across people. The
absence of controller-class clusters was an early sign that "dangerous" is not a region of movement space, which E32
confirmed.
