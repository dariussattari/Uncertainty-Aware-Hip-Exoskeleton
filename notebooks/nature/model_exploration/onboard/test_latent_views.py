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
    # a continuous lens is one trace coloured by the score; a categorical lens one trace per category
    pts = rng.normal(size = (6, 3)).astype(np.float32)
    labels = pd.DataFrame({'task': ['a', 'a', 'b', 'b', 'c', 'c'], 'subject': 'P', 'direction': 'none',
                           'controller_class': ['ID-safe', 'OOD-safe'] * 3, 'pct': [0, 20, 40, 60, 80, 100]})
    views = {(v, 3): pts for v in lv.VIEWS}
    fig = lv.build_3d({'m': views}, labels, {'regime': ('controller_class', lv.CLASS_C), 'score': ('pct', lv.SCORE_SCALE)}, {}, 't')
    per_view = len(fig.data) // len(lv.VIEWS)                       # 3 regime traces (one per CLASS_C entry) + 1 score trace
    assert per_view == len(lv.CLASS_C) + 1
    score_trace = fig.data[per_view - 1]
    assert list(score_trace.marker.color) == [0, 20, 40, 60, 80, 100] and len(score_trace.x) == 6
    print('ok')


if __name__ == '__main__':
    check()
