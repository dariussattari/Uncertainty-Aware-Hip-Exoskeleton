"""The one check for _novelty_scores.py: causal filter, LOF orientation, evaluation at known answers, run labels."""
import numpy as np
import pandas as pd

import _novelty_scores as ns


def check() -> None:
    # causal median over the last 0.5 s (100 samples) of the same recording, never the future
    out = ns.get_causal_median(np.array([1.0, 5, 3, 100, 7]), np.array([1, 1, 1, 1, 2]), np.array([199, 209, 219, 299, 199]))
    np.testing.assert_allclose(out, [1, 3, 3, 5, 7])
    # LOF: a point far from the fit cloud scores higher than one inside it
    rng = np.random.default_rng(0)
    far, near = ns.score_lof(rng.normal(size = (2000, 4)), [np.full((1, 4), 6.0), np.zeros((1, 4))], 0)
    assert far[0] > 2 * near[0], (far, near)
    # evaluation: perfectly separated scores give AUROC 1, every unfamiliar window flagged, no false alarms
    labels = pd.DataFrame({'task': ['walk'] * 50 + ['jump'] * 50, 'controller_class': ['ID-safe'] * 50 + ['OOD-dangerous'] * 50,
                           'rec': np.repeat([0, 1], 50), 'end': np.tile(199 + 10 * np.arange(50), 2)})
    metrics, per_task = ns.evaluate(rng.uniform(0, 1, 1000), np.r_[rng.uniform(0, 0.9, 50), rng.uniform(5, 6, 50)], labels, ('walk',))
    assert metrics['AUROC, unfamiliar vs familiar'] == 1 and metrics['flagged, unfamiliar (%)'] == 100
    assert metrics['false alarms, familiar (%)'] == 0 and per_task['jump'] == 100
    # fit / calibration split and labels
    fit, cal = ns.get_fit_split(100)
    assert len(fit) == 75 and len(cal) == 25 and not set(fit) & set(cal)
    assert ns.get_model_label({'variant': 'sparse', 'inputs': 'raw_3s_spectral', 'sparsity_weight': 0.1, 'latent': '8x4'}, 'x') \
        == 'sparse · 3 s + spectral · λ 0.1 · 8x4'
    # held-out operating point: the tuned threshold flags the target share of the familiar windows it was set on
    val_labels = pd.DataFrame({'task': ['walk'] * 400 + ['jump'] * 100, 'controller_class': ['ID-safe'] * 400 + ['OOD-dangerous'] * 100,
                               'rec': np.repeat([0, 1, 2, 3, 4], 100), 'end': np.tile(199 + 10 * np.arange(100), 5)})
    filtered = ns.get_filtered(rng.uniform(size = 500), val_labels)
    threshold = ns.tune_threshold(filtered, val_labels, ('walk',), 1.0)
    assert abs(100 * (filtered[:400] > threshold).mean() - 1.0) <= 0.3, threshold
    # a component without variance changes no LOF score: PCA to the rest equals the standardized latent
    base = rng.normal(size = (3000, 5)) * np.arange(1, 6)
    fit = np.c_[base, base @ np.arange(5) + 1e-6 * rng.normal(size = 3000)]
    query = rng.normal(size = (200, 5)) * np.arange(1, 6)
    query = np.c_[query, query @ np.arange(5)]
    every, five = ns.get_representation(fit, None), ns.get_representation(fit, 5)
    (a,), (b,) = ns.score_lof(every(fit), [every(query)], 0), ns.score_lof(five(fit), [five(query)], 0)
    np.testing.assert_allclose(a, b, rtol = 1e-3)
    assert five(query).shape == (200, 5) and every(query).shape == (200, 6)
    # the ankle rule is evaluate_at with the 99.5th percentile of the calibration scores
    cal, test = rng.uniform(0, 1, 1000), np.r_[rng.uniform(0, 0.9, 50), rng.uniform(5, 6, 50)]
    assert ns.evaluate(cal, test, labels, ('walk',))[0] == ns.evaluate_at(float(np.percentile(cal, 99.5)), test, labels, ('walk',))[0]
    print('ok')


if __name__ == '__main__':
    check()
