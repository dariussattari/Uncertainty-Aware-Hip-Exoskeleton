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
    # native metrics: Mahalanobis equals the direct formula; out-of-range is zero inside the fit range
    fit = rng.normal(size = (4000, 3)) @ np.array([[2.0, 0, 0], [0.5, 1.0, 0], [0, 0, 0.5]])
    q = rng.normal(size = (50, 3))
    (d2,) = ns.score_mahalanobis(fit, [q])
    diff = q - fit.mean(axis = 0)
    np.testing.assert_allclose(d2, np.einsum('nk,kj,nj->n', diff, np.linalg.inv(np.cov(fit, rowvar = False) + 1e-6 * np.eye(3)), diff), rtol = 1e-6)
    (inside, outside) = ns.score_out_of_range(fit, [np.zeros((1, 3)), np.full((1, 3), 50.0)])
    assert inside[0] == 0 and outside[0] > 0
    # posterior variance averages exp(logvar) over the dimensions active on the fit set
    lv_fit = np.c_[np.full((100, 2), -2.0), np.zeros((100, 1))]
    (pv,) = ns.get_posterior_variance({'logvar': lv_fit}, [{'logvar': np.c_[np.full((1, 2), np.log(0.5)), np.zeros((1, 1))]}])
    np.testing.assert_allclose(pv, [0.5])
    # orientation: a score that ranks unfamiliar windows lower on validation is negated before the threshold
    lab = pd.DataFrame({'task': ['walk'] * 200 + ['jump'] * 200, 'controller_class': ['ID-safe'] * 200 + ['OOD-dangerous'] * 200,
                        'rec': np.repeat(np.arange(8), 50), 'end': np.tile(199 + 10 * np.arange(50), 8)})
    inverted = np.r_[rng.uniform(5, 6, 200), rng.uniform(0, 1, 200)]
    m, _ = ns.evaluate_oriented(inverted, inverted, lab, lab, ('walk',))
    assert m['direction'] == 'lower = unfamiliar' and m['test AUROC, unfamiliar vs familiar'] == 1.0
    # the combined classifier separates separable features, and VAE runs get a readable label and family
    feats = np.c_[np.r_[rng.normal(0, 1, 200), rng.normal(4, 1, 200)], rng.normal(size = 400)]
    assert ns.get_combined_auroc(feats, lab, feats, lab, ('walk',))['AUROC, unfamiliar vs familiar'] > 0.99
    cfg = {'model': 'vae', 'beta': 8.0, 'decoder_variance': 'learned'}
    assert ns.get_model_label(cfg, 'x') == 'VAE · β 8 · learned variance' and ns.get_family(cfg) == 'vae'
    assert 'KL' in ns.get_native_names('vae') and 'KL' not in ns.get_native_names('vanilla')
    # onset latency: bouts split at gaps; the first flagged window sets the latency, a bout never flagged is NaN
    lab = pd.DataFrame({'rec': [1] * 400 + [2] * 300, 'end': np.r_[199 + 10 * np.arange(400), 199 + 10 * np.arange(300)],
                        'task': ['jump'] * 700, 'controller_class': ['OOD-dangerous'] * 700})
    lab.loc[200:, 'end'] = lab.loc[200:, 'end'] + np.where(lab.loc[200:, 'rec'] == 1, 5000, 0)    # rec 1: two bouts
    flag = np.zeros(700, bool)
    flag[50] = flag[60] = True                                   # bout 1: first hit at window 50, 2.5 s after its start
    bouts = ns.get_onset_latency(flag, lab)
    assert len(bouts) == 3 and bouts['latency_s'].iloc[0] == 2.5 and bouts['latency_s'].iloc[1:].isna().all(), bouts
    # linear CKA: 1 for a rotated and rescaled copy, near 0 for independent representations; model types and labels
    xr = rng.normal(size = (2000, 6))
    rot, _ = np.linalg.qr(rng.normal(size = (6, 6)))
    np.testing.assert_allclose(ns.get_linear_cka(xr, 3.0 * xr @ rot), 1.0, rtol = 1e-9)
    assert ns.get_linear_cka(xr, rng.normal(size = (2000, 6))) < 0.05
    cfg = {'model': 'vae', 'arch': 'lstm', 'beta': 32.0, 'decoder_variance': 'fixed', 'inputs': 'raw_3s'}
    assert ns.get_model_label(cfg, 'x') == 'LSTM-VAE · β 32 · fixed variance · 3 s raw' and ns.get_model_type(cfg) == 'LSTM-VAE'
    assert ns.get_model_type({'variant': 'gaussian'}) == 'Gaussian-NLL AE' and ns.get_family(cfg) == 'vae'
    # lazy exports: the latent and log variance stay on disk until asked for, labels keep the used columns only
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        zz = rng.normal(size = (6, 4)).astype(np.float32)
        np.savez(run / 'latents.npz', rows = np.arange(6), z = zz, logvar = -zz, recon_mse = np.ones(6), nll = np.ones(6),
                 kl = np.ones(6), is_train = np.r_[np.ones(3, bool), np.zeros(3, bool)])
        windows = pd.DataFrame({'rec': np.arange(6), 'end': np.arange(6), 'task': ['walk'] * 6, 'controller_class': ['ID-safe'] * 6,
                                'subject': ['P1'] * 6, 'start': np.arange(6), 'phase': ['p'] * 6})
        e = ns.load_export(run, windows)
        assert 'z' not in e and 'logvar' not in e and list(e['labels'].columns) == ['rec', 'end', 'subject', 'task', 'controller_class']
        np.testing.assert_array_equal(ns.get_array(e, 'z'), zz)
        full = ns.materialize(e)
        np.testing.assert_array_equal(full['logvar'], -zz) and np.testing.assert_array_equal(full['is_train'], e['is_train'])
    # the two-group AUROC equals scikit-learn's, ties included
    pos, neg = rng.integers(0, 20, 300).astype(float), rng.integers(0, 25, 500).astype(float)
    np.testing.assert_allclose(ns.get_auroc(pos, np.sort(neg)), ns.roc_auc_score(np.r_[np.ones(300), np.zeros(500)], np.r_[pos, neg]), rtol = 1e-12)
    # training-class ablation: dangerous vs OOD-safe, oriented and thresholded on the validation participants
    def get_labels(n: int) -> pd.DataFrame:
        cls = ['ID-safe'] * n + ['OOD-safe'] * (2 * n) + ['OOD-dangerous'] * n
        tasks = ['walk'] * n + ['squat'] * (2 * n) + ['ball_toss'] * (n // 2) + ['tug_of_war'] * (n - n // 2)
        return pd.DataFrame({'controller_class': cls, 'task': tasks, 'rec': np.repeat(np.arange(4 * n // 50), 50),
                             'end': np.tile(199 + 10 * np.arange(50), 4 * n // 50)})
    lab_v, lab_t = get_labels(200), get_labels(100)
    make = lambda n, d: np.r_[rng.uniform(0, 1, n), rng.uniform(0, 1, 2 * n), rng.uniform(d, d + 1, n)]   # noqa: E731
    m, flagged = ns.evaluate_dangerous(-make(200, 5), -make(100, 5), lab_v, lab_t)     # dangerous score lower: negated
    assert m['sign'] == -1 and m['direction'] == 'lower = dangerous'
    assert m['test AUROC, dangerous vs OOD-safe'] == 1 and m['test AUROC, tug_of_war vs OOD-safe'] == 1
    assert m['test flagged, dangerous (%)'] == 100 and m['test false alarms, OOD-safe (%)'] <= 3 and flagged['ball_toss'] == 100
    assert abs(m['val false alarms, OOD-safe (%)'] - 1.0) <= 0.5
    print('ok')


if __name__ == '__main__':
    check()
