"""The one check for awareness.py, runnable on the device without data: python deployment/test_awareness.py [bundle.pkl]."""
from pathlib import Path
import sys

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

import awareness as aw


def check(bundle_path: Path | None) -> None:
    rng = np.random.default_rng(0)
    # novelty: Mahalanobis equals the explicit inverse-covariance form; the fit set's own centre is the most familiar
    z, logvar = rng.normal(size = (500, 32)), rng.normal(-2, 0.1, size = (500, 32)).astype(np.float32)
    fit = aw.fit_novelty(LocalOutlierFactor(n_neighbors = aw.LOF_NEIGHBORS, novelty = True).fit(z), z, logvar)
    q = rng.normal(size = (5, 32))
    diff = q - z.mean(axis = 0)
    explicit = np.einsum('nk,kj,nj->n', diff, np.linalg.inv(np.cov(z, rowvar = False) + 1e-6 * np.eye(32)), diff)
    scores = aw.score_novelty(fit, q, logvar[:5])
    np.testing.assert_allclose(scores[:, 1], explicit, rtol = 1e-8)
    assert aw.score_novelty(fit, z.mean(axis = 0, keepdims = True), logvar[:1])[0, 1] < scores[:, 1].min()
    # smoothing: a one-window spike is removed; windows older than 0.5 s (100 samples) drop out; a gap empties the history
    smoother = aw.CausalMedian()
    p = np.tile([0.8, 0.1, 0.1], (20, 1))
    p[10] = [0.1, 0.1, 0.8]
    out = np.stack([smoother.push(199 + 10 * i, p[i]) for i in range(20)])
    assert out.argmax(axis = 1).tolist() == [0] * 20 and len(smoother.history) == 10
    np.testing.assert_allclose(smoother.push(10_000, [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
    # u and the regime bands: cutoffs are inclusive on their upper side
    u = aw.get_uncertainty(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0]]))
    np.testing.assert_allclose(u, [0, 0.5, 1, 0.25])
    assert aw.get_regime(np.array([0.1, 0.21, 0.5, 0.85]), (0.21, 0.85)).tolist() == [aw.ID, aw.SAFE, aw.SAFE, aw.DANGEROUS]
    # a saved bundle loads, takes 42 features and gives probabilities that sum to 1
    if bundle_path is not None:
        bundle = aw.load_bundle(bundle_path)
        x = aw.get_features(rng.normal(size = (4, 32)), aw.score_novelty(bundle.novelty, z[:4], logvar[:4]), np.ones((4, 7)))
        np.testing.assert_allclose(aw.get_proba(bundle.classifier, x).sum(axis = 1), 1, rtol = 1e-9)
        print(f"{bundle_path.name}: seed {bundle.meta['seed']}, cutoffs {bundle.cutoffs}, sklearn {bundle.meta['sklearn']}")
    print('awareness.py: ok')


if __name__ == '__main__':
    check(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
