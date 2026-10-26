# E21: Fix: the rhythm (spectral) features report 0 when there is no rhythm

| Field | Value |
|---|---|
| ID | E21 |
| Date | 2026-10-08 (commit 4acacfc) |
| Dataset | Nature 2024 |
| Status | Complete. E19 and E20 rerun |
| Sources | Live_Update.docx, "OCT 5 – 10" §3; README "OBJ 2" (rhythm features bullet) |
| Code / notebooks | `ml/common/features.py`; `notebooks/nature/data_exploration/02_features.ipynb`, `03_feature_embeddings.ipynb` |

## Hypothesis

When a window has no peak in the search band, the rhythm features (stride regularity, step regularity, dominant
frequency) returned the band-edge value. That invented a rhythm where there was none: a squat or a standing pose read as
a very slow gait. Reporting 0 for "no rhythm" should make the features honest without materially changing the feature
rankings.

## Method

The rhythm features now return 0 when no peak exists in the 0.5–1.6 s stride search band. Notebooks 02 and 03 (E19, E20)
were rerun.

## Result

- The rankings barely moved. The spectral gain in the embeddings became +2.9 / +3.2 k-NN points (from +3.5 / +4.4).
- Known side effect: strides slower than 1.6 s (about 1.7–1.8 s) also read as no rhythm, so some slow walking and stair
  windows are affected.

## Conclusion

The features are now honest, and every downstream conclusion stands. Fixing very slow strides would need a longer window
or a wider search band. In E31 the "rhythm = 0" pattern turned out to dominate latents trained with these features: a
linear readout finds no-rhythm windows at AUROC 0.97–1.00.
