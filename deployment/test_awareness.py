"""The one check for awareness.py, runnable on the device without data or scikit-learn:
python deployment/test_awareness.py [deployment/artifacts/awareness_s0.npz]."""
from pathlib import Path
import sys

import numpy as np

import awareness as aw


def get_lof_arrays(ref: np.ndarray, k: int) -> dict:
    """Brute-force LOF reference arrays by the textbook definition (each point's neighbours exclude itself)."""
    d = np.linalg.norm(ref[:, None] - ref[None], axis = 2)                    # (n, n)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis = 1)[:, :k]                                        # (n, k)
    kdist = np.take_along_axis(d, nn, axis = 1)[:, -1]                         # (n,)
    reach = np.maximum(np.take_along_axis(d, nn, axis = 1), kdist[nn])
    lrd = 1.0 / (reach.mean(axis = 1) + 1e-10)
    return {'lof_fit': ref, 'lof_kdist': kdist, 'lof_lrd': lrd, 'lof_cap': 10.0}


def check(bundle_path: Path | None) -> None:
    rng = np.random.default_rng(0)
    # novelty: an inlier's LOF is about 1 and a far point's is large; Mahalanobis equals the explicit inverse form
    z, logvar = rng.normal(size = (300, 32)), rng.normal(-2, 0.1, size = (300, 32)).astype(np.float32)
    fit = aw.fit_novelty(z, logvar, **get_lof_arrays(z, aw.LOF_NEIGHBORS))
    q = np.vstack([np.zeros((1, 32)), np.full((1, 32), 8.0)])
    scores = aw.score_novelty(fit, q, logvar[:2])
    assert 0.8 < scores[0, 0] < 1.3 and scores[1, 0] > 3, scores[:, 0]
    diff = q - z.mean(axis = 0)
    explicit = np.einsum('nk,kj,nj->n', diff, np.linalg.inv(np.cov(z, rowvar = False) + 1e-6 * np.eye(32)), diff)
    np.testing.assert_allclose(scores[:, 1], explicit, rtol = 1e-8)
    # trees: two iterations of three one-split stumps on feature 0, a NaN routed by missing_left; softmax sums to 1
    stump = lambda v: [(0, 0.0, True, 1, 2, False, 0.0), (0, 0, False, 0, 0, True, -v), (0, 0, False, 0, 0, True, v)]  # noqa: E731
    nodes = [n for v in (1.0, 2.0, 3.0, 0.5, 0.5, 0.5) for n in stump(v)]
    cols = list(zip(*nodes))
    offsets = np.repeat(np.arange(6) * 3, 3)
    trees = aw.Trees(feature = np.array(cols[0]), threshold = np.array(cols[1], float), missing_left = np.array(cols[2]),
                     left = np.array(cols[3]) + offsets, right = np.array(cols[4]) + offsets, is_leaf = np.array(cols[5]),
                     value = np.array(cols[6], float), roots = np.arange(6).reshape(2, 3) * 3, baseline = np.zeros(3),
                     classes = (aw.ID, aw.DANGEROUS, aw.SAFE))
    x = np.zeros((3, aw.N_FEATURES))
    x[:, 0] = [-1.0, 1.0, np.nan]
    raw_right = np.array([1.5, 2.5, 3.5])                                       # per classifier class, x0 > 0
    expected = np.exp(raw_right) / np.exp(raw_right).sum()
    p = aw.get_proba(trees, x)
    np.testing.assert_allclose(p.sum(axis = 1), 1, rtol = 1e-12)
    np.testing.assert_allclose(p[1], expected[[0, 2, 1]], rtol = 1e-12)         # REGIMES order: ID, OOD-safe, dangerous
    np.testing.assert_allclose(p[2], p[0], rtol = 1e-12)                        # NaN goes left, like x0 <= 0
    # smoothing: a one-window spike is removed; windows older than 0.5 s (100 samples) drop out; a gap empties the history
    smoother = aw.CausalMedian()
    pr = np.tile([0.8, 0.1, 0.1], (20, 1))
    pr[10] = [0.1, 0.1, 0.8]
    out = np.stack([smoother.push(199 + 10 * i, pr[i]) for i in range(20)])
    assert out.argmax(axis = 1).tolist() == [0] * 20 and len(smoother.history) == 10
    np.testing.assert_allclose(smoother.push(10_000, [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
    # u and the regime bands: cutoffs are inclusive on their upper side
    np.testing.assert_allclose(aw.get_uncertainty(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0]])), [0, 0.5, 1, 0.25])
    assert aw.get_regime(np.array([0.1, 0.21, 0.5, 0.85]), (0.21, 0.85)).tolist() == [aw.ID, aw.SAFE, aw.SAFE, aw.DANGEROUS]
    # a saved bundle loads and gives finite novelty scores and probabilities that sum to 1
    if bundle_path is not None:
        bundle = aw.load_bundle(bundle_path)
        zb = bundle.novelty.lof_fit[:4] + 0.01
        nov = aw.score_novelty(bundle.novelty, zb, np.full_like(zb, -2.0, dtype = np.float32))
        xb = aw.get_features(rng.normal(size = (4, 32)), nov, np.ones((4, 7)))
        np.testing.assert_allclose(aw.get_proba(bundle.trees, xb).sum(axis = 1), 1, rtol = 1e-9)
        print(f"{bundle_path.name}: seed {bundle.meta['seed']}, cutoffs {bundle.cutoffs}, {bundle.meta['n_trees']} trees")
    print('awareness.py: ok')


if __name__ == '__main__':
    check(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
