# D01: Getting the awareness layer onto the Jetson without changing it: scikit-learn refit vs version-free arrays

| Field | Value |
|---|---|
| ID | D01 |
| Date | 2026-10-27 |
| Dataset | Nature 2024 (the E39 features: 14 train / 4 validation / 4 test participants) |
| Status | Complete |
| Sources | E39 (the validated classifier and *u* cutoffs, notebook 13) |
| Code / notebooks | `deployment/awareness.py`, `deployment/export_awareness.py`, `deployment/test_awareness.py`; bundles `deployment/artifacts/awareness_s{0,1,2}.npz` (gitignored) |

## Hypothesis

The awareness layer (LSTM-VAE novelty scores, the boosted-tree regime classifier, the *u* cutoffs) was validated in
E39 under scikit-learn 1.9.0. The Jetson runs Python 3.10, fixed by NVIDIA's GPU PyTorch builds for JetPack 6, and
scikit-learn 1.9 needs Python ≥ 3.11.

- **(a)** Refitting under the newest scikit-learn for Python 3.10 (1.7.2) gives a different classifier: gradient-boosting
  internals change between releases.
- **(b)** Storing the fitted LOF and trees as plain arrays and evaluating them with numpy and scipy reproduces the
  validated model exactly, with no scikit-learn on the device.

## Method

- **Refit (a):** notebook 13's training fit (HistGradientBoosting, balanced class weights, 300 iterations, random state =
  seed, 42 features, the novelty-score fit windows left out) and cutoffs on validation, under scikit-learn 1.7.2 /
  numpy 2.2.6 / scipy 1.15.3; test participants scored as in E39. 3 seeds.
- **Array export (b):** `export_awareness.py` refits under 1.9.0 exactly as notebook 13, then converts:
  - LOF (k = 20, Euclidean): the 15,000 deduplicated reference latents, their k-distances and local reachability
    densities; scored with `scipy.spatial.cKDTree` by sklearn's novelty formula.
  - Boosted trees: 61 iterations × 3 classes = 183 trees, 11,163 nodes, as flat node arrays; all trees walked at once,
    raw scores summed per iteration in sklearn's order, softmax.
  - Saved only if: novelty scores match the notebook's cache (rtol 1e-6), probabilities match sklearn's (atol 1e-12),
    and the full pipeline, smoothing window by window as on the device, reproduces E39's test results to 1e-9.

## Result

Test participants, *u* with validation cutoffs, mean ± SD over 3 seeds:

| | Macro F1 | Dangerous recall | Dangerous to the spline | Safe switched off |
|---|---|---|---|---|
| Validated fit (1.9.0, E39) | 0.875 ± 0.006 | 0.817 | 0% | 2.7% |
| Refit under 1.7.2 | 0.874 ± 0.006 | 0.823 | 0.006% | 2.9% |
| Array export (numpy/scipy) | identical to 1.9.0 (1e-9) | | | |

Cutoffs per seed, 1.9.0 → 1.7.2: 0.20/0.84 → 0.18/0.84, 0.23/0.855 → 0.245/0.835, 0.19/0.855 → 0.17/0.84.

- **(a) holds, but the difference is within seed noise.** The refit moves the cutoffs by up to 0.02 and sends a few
  dangerous windows (0.006%) to the spline, where the validated fit sent none.
- **(b) holds.** Every check passes for every seed; bundles are 4.6–6.4 MB of plain arrays (no pickles).
- On the Jetson, novelty scoring takes about 5.8 ms per window, almost all of it the LOF neighbour search over 15,000
  reference points (a window arrives every 50 ms).

## Conclusion

- **Ship the validated fit as arrays.** It is the model E39 scored, it needs only numpy and scipy, and it no longer ties
  the device to a scikit-learn version. A 1.7.2 refit would have been usable but is a different, unvalidated model.
- LOF is the only costly part on the device: a further reason to test the classifier without it (deployment plan
  step 4).
