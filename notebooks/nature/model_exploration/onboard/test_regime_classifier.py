"""The one check for _regime_classifier.py: alignment, smoothing, and the regime metrics at known answers."""
import numpy as np
import pandas as pd

import _novelty_scores as ns
import _regime_classifier as rc


def check() -> None:
    # alignment returns the rows named by the keys, in key order, and refuses missing windows
    labels = pd.DataFrame({'rec': [1, 1, 2], 'end': [199, 209, 199]})
    values = np.array([[10.0], [20.0], [30.0]])
    np.testing.assert_allclose(rc.get_aligned(values, labels, pd.DataFrame({'rec': [2, 1], 'end': [199, 209]})), [[30.0], [20.0]])
    try:
        rc.get_aligned(values, labels, pd.DataFrame({'rec': [3], 'end': [199]}))
        raise AssertionError('a missing window should raise')
    except KeyError:
        pass
    # smoothing: a one-window spike in a class probability is removed by the causal median
    lab = pd.DataFrame({'rec': np.zeros(20, int), 'end': 199 + 10 * np.arange(20)})
    p = np.tile([0.8, 0.1, 0.1], (20, 1))
    p[10] = [0.1, 0.1, 0.8]
    assert rc.get_smoothed(p, lab).argmax(axis = 1).tolist() == [0] * 20
    # metrics: a perfect prediction, then dangerous windows sent to the spline
    true = np.array([ns.ID] * 4 + [ns.SAFE] * 4 + [ns.DANGEROUS] * 2)
    m = rc.get_regime_metrics(true, true.copy())
    assert m['macro F1'] == 1 and m['dangerous sent to the spline (%)'] == 0 and m['safe switched off (%)'] == 0
    pred = true.copy()
    pred[-1] = ns.ID
    pred[0] = ns.DANGEROUS
    m = rc.get_regime_metrics(true, pred)
    assert m['dangerous sent to the spline (%)'] == 50 and m['safe switched off (%)'] == 12.5 and m['recall, OOD-dangerous'] == .5
    c = rc.get_confusion(true, pred)
    np.testing.assert_allclose(c.sum(axis = 1), 100)
    assert c.loc[ns.DANGEROUS, ns.ID] == 50
    # targets: a family maps to the regime most of its windows have; 'dangerous or not' fits on OOD windows only
    rows = pd.DataFrame({'controller_class': [ns.ID, ns.ID, ns.SAFE, ns.DANGEROUS, ns.SAFE],
                         'family': ['stairs', 'stairs', 'stairs', 'stairs', 'squat'], 'task': ['stairs', 'stairs', 'curb_up', 'step_ups', 'squats']})
    assert rc.get_regime_map(rows, 'family') == {'stairs': ns.ID, 'squat': ns.SAFE}
    assert rc.get_regime_map(rows, 'task')['step_ups'] == ns.DANGEROUS and rc.get_regime_map(rows, 'regime')[ns.SAFE] == ns.SAFE
    # permutation importance: a classifier that reads only column 0 loses everything when it is shuffled, nothing otherwise
    class ReadsColumnZero:
        classes_ = np.array(rc.REGIMES)

        def predict_proba(self, x):
            return np.eye(3)[np.clip(x[:, 0].astype(int), 0, 2)]

    rng = np.random.default_rng(0)
    true = np.repeat(np.array(rc.REGIMES), 200)
    x = np.stack([np.repeat([0, 1, 2], 200), rng.normal(size = 600)], axis = 1)
    lab = pd.DataFrame({'rec': np.arange(600), 'end': 199})                 # one window per recording: smoothing is a no-op
    imp = rc.get_permutation_importance(ReadsColumnZero(), x, true, lab, {'used': [0], 'unused': [1]}, 3, rng).groupby('group').mean()
    assert imp.loc['unused', 'drop, macro F1'] == 0 and imp.loc['used', 'drop, macro F1'] > 0.5
    assert rc.get_fit_rows(rows, 'dangerous or not').tolist() == [False, False, True, True, True] and rc.get_fit_rows(rows, 'task').all()
    # uncertainty: regimes sit at 0, 0.5, 1; separable scores get cutoffs between them and band back to their regimes
    np.testing.assert_allclose(rc.get_uncertainty(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, .5]])), [0, .5, 1, .5])
    u = np.array([.1] * 5 + [.5] * 4 + [.9] * 3)
    true = np.array([ns.ID] * 5 + [ns.SAFE] * 4 + [ns.DANGEROUS] * 3)
    t1, t2 = rc.get_cutoffs(u, true)
    assert .1 < t1 <= .5 < t2 <= .9 and (rc.get_banded(u, (t1, t2)) == true).all()
    np.testing.assert_allclose(rc.get_assistance(np.array([.1, .3, .5, .69, .7]), (.3, .7), .25), [1, 1, .625, .26875, 0])
    # the novelty ramp: spline, full up to the first anchor, the floor from the second, off from the second cutoff
    np.testing.assert_allclose(rc.get_ramped_assistance(np.array([.1, .5, .5, .5, .5, .9]), np.array([9, 0, 1, 2, 5, 0]), (.25, .8), (1, 3), .2),
                               [1, 1, 1, .6, .2, 0])
    print('ok')


if __name__ == '__main__':
    check()
