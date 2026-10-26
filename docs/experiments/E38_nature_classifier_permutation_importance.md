# E38: What the regime classifier relies on: permutation importance of the boosted tree

| Field | Value |
|---|---|
| ID | E38 |
| Date | 2026-10-20 |
| Dataset | Nature 2024 |
| Status | Complete. Not yet written up in the research-update documents |
| Sources | Notebook `12_classifier_importance.ipynb`; README "What the regime classifier relies on" |
| Code / notebooks | `notebooks/nature/model_exploration/onboard/12_classifier_importance.ipynb`, `_regime_classifier.get_permutation_importance`; results `onboard/results/12_importance_{groups,features}.csv`; figure `onboard/figures/12_classifier_importance.png` |

## Hypothesis

E34's boosted tree reads 42 inputs: the TCN-VAE's 32 posterior means, the LSTM-VAE's 3 novelty scores and 7 rhythm
features. The combination beats any single source, but it was not known which inputs the tree uses. The pipeline intends
a split, with the novelty scores recognizing familiar movement and the latent recognizing dangerous movement. Permutation
importance should show whether the tree learned that split. It should also show whether LOF, the costliest score on a
device (a 20-NN search against stored latents), can be dropped.

## Method

- E34's classifier refitted exactly (HistGradientBoosting, balanced weights, 300 iterations) per seed on the four
  validation participants. It reproduces E34's test scores exactly.
- Each input group was shuffled across test windows (one shared permutation per group, so internal correlations
  survive), 10 repeats per group and 5 per single feature, per seed. Measured: the drop in macro F1 and in each regime's
  recall (mean ± SD over three seeds).

## Result

- Baseline: macro F1 0.868; recall ID 0.93, OOD-safe 0.92, dangerous 0.79; no dangerous window on the spline; 2.8% of
  safe windows switched off.
- **The two encoders split the job:**
  - Shuffling the **LSTM-VAE novelty scores** drops ID-safe recall by **0.86 ± 0.02** and sends 2.9% of dangerous
    windows to the spline.
  - Shuffling the **TCN-VAE latent** drops dangerous recall by **0.78 ± 0.01** and leaves ID-safe nearly untouched
    (0.03).
  - Each group alone costs 0.29 of macro F1, for different regimes.
- Rhythm features: dangerous recall −0.24 ± 0.04 (macro F1 −0.06).
- **OOD-safe is the fallback.** Shuffling any group raises OOD-safe recall slightly, the same behaviour as for an unseen
  dangerous task (E34).
- Single inputs: posterior variance (0.149 ± 0.035) and Mahalanobis (0.093 ± 0.009) matter most. **LOF adds little on
  its own (0.013 ± 0.023, within noise).** One latent unit (filter 1, step 0) stands out (0.040); the rest share their
  information.

## Conclusion

The tree learned the intended division of labour: novelty scores recognize familiar movement, and the latent recognizes
dangerous movement. LOF appears redundant given Mahalanobis and posterior variance, so a classifier **without LOF**
(no on-device neighbour search) is worth testing. That needs a refit, not this table.

## Caveats

- Permutation importance understates information that inputs share.
- Shuffled rows can be combinations no real window has.
- Four test participants.
