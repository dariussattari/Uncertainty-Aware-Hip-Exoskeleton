# E10: Five estimators compared at matched operating points (with a population correction)

| Field | Value |
|---|---|
| ID | E10 |
| Date | 2026-09-16 to 2026-09-18 (tables corrected 2026-09-18) |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete |
| Sources | Methods_Experiments_Reproduction.docx §15, §20, §28; Live_Update.docx, "SEP 2 – 18" §8 ("Integrated result") and §12 |
| Code / notebooks | `ml/sci_rob/vanilla/evaluation/` (`common.py`, `plots_comparison.py`) |

## Hypothesis

The 99.5th-percentile rule places each model at a different specificity, so J at that threshold is not a fair ranking.
Comparing every model at the **same specificity**, on the **same windows**, should give the true ordering, and test
whether E05's apparent J advantage over E03 is real.

## Method

- Every model is scored on one shared set: the 58,045 ID test windows that carry a gait-phase label (E03's population)
  plus the same 23,043 OOD windows.
- **Correction:** earlier versions compared E03 on 58,045 ID windows with the label-free models on 61,956. All tables
  were recomputed on the shared set.
- Reported: AUROC, best achievable J, J at each model's own threshold, and recall at 99.7%, 99.0%, 98.1%, 95.0% and
  90.0% specificity.

## Result

| Comparison | E03 gait phase | E05 AE | E06 GAN | E07 correlation | E08 forecast |
|---|---|---|---|---|---|
| AUROC | 0.859 | 0.782 | 0.727 | 0.817 | **0.899** |
| Best J | 56.78 | 42.61 | 33.29 | 47.24 | **63.99** |
| J at 99.5th-percentile rule | 37.54 | 38.93 | 10.55 | 35.55 | **47.87** |
| Recall at 99.7% spec. | **27.89** | 11.50 | 11.19 | 11.83 | 8.32 |
| Recall at 99.0% spec. | **35.47** | 25.66 | 18.21 | 16.31 | 22.48 |
| Recall at 98.1% spec. | **39.39** | 32.13 | 21.47 | 21.43 | 37.52 |
| Recall at 95.0% spec. | 51.82 | 43.22 | 27.63 | 35.92 | **55.56** |
| Recall at 90.0% spec. | 64.08 | 51.21 | 36.22 | 49.33 | **69.99** |
| Stair ascent at 98.1% spec. | 6.9% | 15.4% | 0.6% | 3.8% | **21.9%** |
| Needs labels | yes | no | no | no | no |

- The correction changed no conclusion: E08 went from 0.896 to 0.899, and E05's J from 38.72 to 38.93.

## Conclusion

1. The forecast target (E08) is the strongest estimator on threshold-free metrics, and at every operating point looser
   than about 98.5% specificity, without labels.
2. Gait phase (E03) is best in the conservative regime (below about 1.5% false alarms), which is the regime a deployed
   controller would probably occupy.
3. E05's J advantage over E03 is a threshold artefact and disappears at matched specificity.
4. The reference's ordering (gait phase > synthetic > AE) is reproduced. The GAN is the one exception (see E06).
5. All five detect standing completely and are weakest on stair ascent. That column is the most informative, because
   it is the only one where the gaps are large and not explained by class balance.

The "integrated result" table in Live_Update records the roles chosen:

- Forecast ensemble: primary movement-novelty score.
- AE + LOF: label-free complementary score.
- GAN: amplitude-confounded baseline.
- Matched Kalman residuals (E04): sensor timing and integrity.

## Caveats

- Single seed per model.
- The 99.5th-percentile rule transfers poorly at this class balance (27% OOD against the reference's 80%).
