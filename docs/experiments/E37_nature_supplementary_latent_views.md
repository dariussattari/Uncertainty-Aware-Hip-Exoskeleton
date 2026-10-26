# E37: Supplementary latent views of the two encoders behind the regime classifier

| Field | Value |
|---|---|
| ID | E37 |
| Date | 2026-10-20 |
| Dataset | Nature 2024 (validation participants) |
| Status | Complete. Not yet written up in the research-update documents |
| Sources | Notebook `11_supplementary_latent_views.ipynb`; README "Supplementary latent views" |
| Code / notebooks | `notebooks/nature/model_exploration/onboard/11_supplementary_latent_views.ipynb`, `_latent_views.py` (`get_views`, `build_3d` with `SCORE_SCALE`); figures `onboard/figures/11_*.png`; results `onboard/results/11_novelty_percentiles.csv`; pages `onboard/interactive/11_lstm_vae_novelty_3d.html`, `11_tcn_vae_regime_3d.html` |

## Hypothesis

This experiment is mainly visual, for slides. The testable expectation was that the ID-trained LSTM-VAE's three novelty
scores (the classifier's novelty inputs) separate ID-safe from everything else, but **not** OOD-safe from
OOD-dangerous. If so, that is the visual counterpart of E32. The TCN-VAE latent (trained on ID + OOD-safe) should then
place the regimes where a classifier can separate them.

## Method

- Validation participants' in-task windows (seed-0 runs), up to 400 per task.
- UMAP (and PCA, plus the raw latent units in the 3-D pages) of each model's standardized active posterior means.
- The LSTM-VAE (β 1, fixed, 3 s raw) is coloured by regime and by each novelty score (LOF, Mahalanobis, posterior
  variance). Each score is oriented so that higher means less familiar and shown as its percentile among all
  validation windows, with the familiar 99th percentile marked. Scores are computed in the full latent; UMAP is layout
  only.
- The TCN-VAE (β 128, fixed, 3 s + spectral, trained on ID + OOD-safe) is coloured by regime and task family.

## Result

- All three novelty scores put OOD-safe and OOD-dangerous in the **same bright periphery**. Validation AUROC between
  those two is 0.63–0.78, against 0.88–0.93 for ID-safe against OOD-safe.
- The TCN-VAE latent is the space in which the classifier separates the regimes. OOD-dangerous is the only class it
  never saw.

## Conclusion

This confirms visually the division of labour the pipeline relies on: the LSTM-VAE novelty scores say *how unfamiliar* a
window is, and the TCN-VAE latent says *which regime* it is in. Novelty alone cannot tell dangerous from safe.
