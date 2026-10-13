"""The one check for _latent_views.py: the effective-directions and novelty-unit measures at their known limits."""
import numpy as np
import pandas as pd

import _latent_views as lv


def check() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size = (5000, 1))
    np.testing.assert_allclose(lv.get_effective_directions(np.hstack([a, 2 * a, 3 * a, np.zeros_like(a)])), 1.0, atol = 1e-6)
    np.testing.assert_allclose(lv.get_effective_directions(rng.normal(size = (5000, 4))), 4.0, atol = 0.02)
    scaled = np.hstack([rng.normal(size = (5000, 1)) * 10, rng.normal(size = (5000, 3)) * 0.1])   # one unit holds the variance
    assert lv.get_effective_directions(scaled, standardize = False) < 1.01 < 3.9 < lv.get_effective_directions(scaled)
    # unit 0 fires for an unseen task at 10x its training response; unit 1 peaks on a training task; unit 2 never fires
    means = pd.DataFrame({0: [0.1, 0.0, 1.0], 1: [1.0, 0.2, 0.5], 2: [0.0, 0.0, 0.0]}, index = ['walk', 'stairs', 'jump'])
    assert lv.get_novelty_units(means, ('walk', 'stairs')).tolist() == [True, False, False]
    print('ok')


if __name__ == '__main__':
    check()
