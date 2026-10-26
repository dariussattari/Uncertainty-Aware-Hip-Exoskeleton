# E14: DOWN regime holdout: novelty, regime and reliability signals side by side

| Field | Value |
|---|---|
| ID | E14 |
| Date | 2026-09-23 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34 |
| Status | Complete (seed 0) |
| Sources | Live_Update.docx, "SEP 18 – 25" §5 (pipeline; gait-phase "DOWN" figure); `ml/sci_rob/awareness/README.md` |
| Code / notebooks | `ml/sci_rob/awareness/` (`encoder.py`, `regime.py`, `forecast.py`, `experts.py`, `aw_scores.py`, `aw_plots.py`); `notebooks/sci_rob/model_exploration/10_awareness_regime_holdout.ipynb`; results `ml/sci_rob/awareness/results/DOWN/` |

## Hypothesis

The controller-facing question is three questions:

- **Novelty:** is the window unfamiliar?
- **Regime:** which known movement regime does it resemble?
- **Reliability:** is the active prediction trustworthy?

The hypothesis was that, with a whole *regime* withheld (all descending movement), these signals separate:

1. Latent density flags the unseen regime.
2. A regime probe names the nearest known regime.
3. Conditioning the forecaster on the regime, or routing to per-regime experts, makes forecast error regime-aware and
   more informative.

## Method

- **The DOWN holdout:** training sees LG, RA and SA only. OOD = RD, SD, ST, TR. The test set is 27,554 OOD windows
  (32.4%). Same participants, scaler, causal median and evaluators as every other run. The six existing models were also
  rerun on DOWN (`ml/sci_rob/LOTO/DOWN/`).
- **Stage A, novelty:** a deterministic β = 0 encoder (E17's Gaussian-NLL control), with LOF on the 32-D latent and on
  12 and 16 PCs.
- **Stage B, regime:** logistic and MLP probes on the latent, trained on ID labels only (level against ascend). Outputs
  are probabilities, confidence and entropy.
- **Stage C, reliability:** the 200 ms forecast ensemble (E08), plain, or with the regime probabilities as 2 extra
  constant channels.
- **Stage C′, experts:** one forecast ensemble per training regime, routed by the probe. A swap test measured how much
  the conditioned forecaster used its regime input.
- **Horizon extension:** plain ensembles at 500 ms and 1 s.
- The gait-phase ensemble (E03) was also retrained on DOWN, to see whether the reference's best detector could serve
  as a regime classifier.

## Result

Test AUROC, OOD against ID (seed 0):

| Score | OOD vs ID | RD | SD | ST | TR |
|---|---|---|---|---|---|
| Vanilla AE, LOF | **0.976** | 0.989 | 0.973 | 0.998 | 0.895 |
| Awareness encoder, LOF 32-D | 0.974 | 0.979 | 0.983 | 0.998 | 0.882 |
| GAN | 0.938 | 0.925 | 0.973 | 1.000 | 0.835 |
| Forecast ensemble, disagreement (1-fold / full 12-fold) | 0.889 / 0.881 | 0.850 | 0.966 | 0.998 | 0.753 |
| Correlation ensemble | 0.884 | 0.837 | 0.960 | 0.998 | 0.803 |
| Gait-phase ensemble | 0.878 | 0.845 | 0.989 | 0.991 | 0.739 |
| Probabilistic, ratio / epistemic | 0.865 / 0.859 | | | | |
| Forecast realized error (200 ms) | 0.816 | 0.775 | 0.907 | 0.823 | 0.650 |
| Regime experts, routed / best-expert error | 0.810 / 0.811 | | | | |
| Regime-conditioned forecaster, disagreement / error | 0.862 / 0.792 | | | | |
| Reconstruction NLL | 0.486 | 0.572 | 0.467 | 0.003 | 0.104 |

- **Gait-phase ensemble on DOWN** (Live_Update figure, threshold 6.58e-2): RD 0% flagged, TR 2%, SD 11%, ST 0%, while
  ID stair ascent itself is 4% flagged. It ranks reasonably but **is not a classifier** of regime.
- **Regime confidence is not novelty.** The probes reach 89–91% accuracy on known regimes but absorb unseen ones
  confidently (standing → level at 1.00).
- **Conditioning did nothing:** a wrong regime input moved the error by only 5–7%. Experts make the regime
  load-bearing, but routed error equals the plain forecaster's, because descending gait is predictable 200 ms ahead from
  level and ascending dynamics. The one gain is on transitions (error AUROC 0.65 → 0.78, disagreement 0.75 → 0.92).
- **Longer horizons hurt:** disagreement AUROC 0.889 (200 ms) → 0.825 (500 ms) → 0.787 (1 s). Realized error 0.816 →
  0.660 → 0.569. Validation loss 0.049 → 0.105 → 0.165.
- The one-fold epoch-selection shortcut cost nothing (0.889 against 0.881).

## Conclusion

**Density in a learned latent is the detector for a held-out regime.** Both AE-family latents with LOF rank the unseen
descents, standing and transitions near-perfectly. Every disagreement score sits at 0.86–0.89, and every error score
lower. Regime probes name the nearest known regime and must not be read as novelty. Making the forecaster regime-aware
did not make its error more informative, so the 200 ms horizon stays.

This experiment set up the proposed pipeline that carried into the Nature work:

- A latent space gives **novelty** (LOF) and a **regime**.
- A forecaster gives **reliability**.
- A classifier then decides between ID, OOD-safe and OOD-dangerous.

The realization that came with it: ID against OOD is not the end goal. Predicting regime, reliability and
(eventually) the joint moment is the more novel framing.

## Caveats

- Seed 0 only.
- Most stage-C runs used 1-fold epoch selection.
- The regime probes have only two classes (level, ascend).
