"""Wiring tests for the interactive latent views: tiny synthetic arrays, no trained models."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import _umap_exploration as ux
from _umap_exploration import (TASKS, TASK_COLORS, TASK_SYMBOLS_2D, stratified_sample, umap_embedding,
                               scatter_by_task, scatter_by_value, sample_latents, project,
                               choose_latent_dims, compose_2d, compose_3d)


def toy_meta(counts, seed=0):
    modes = np.concatenate([[t] * n for t, n in counts.items()])
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'subject': rng.choice(['AB16', 'AB24'], len(modes)),
                         'trial': [f'{m}_trial' for m in modes], 'mode': modes})


class UmapExplorationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(ux, 'CACHE', Path(self.tmp.name) / 'cache'),
                        patch.object(ux, 'INTERACTIVE', Path(self.tmp.name) / 'interactive')]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_stratified_sample_keeps_small_tasks_and_order(self):
        modes = np.array(['LG'] * 5000 + ['ST'] * 60 + ['TR'] * 200)
        idx = stratified_sample(modes, 1000, seed=0, floor=100)
        self.assertTrue(np.all(np.diff(idx) > 0))
        self.assertEqual(np.sum(modes[idx] == 'ST'), 60)
        self.assertEqual(np.sum(modes[idx] == 'TR'), 100)
        self.assertLess(np.sum(modes[idx] == 'LG'), 5000)
        np.testing.assert_array_equal(stratified_sample(modes, None), np.arange(len(modes)))

    def test_embedding_shapes_reference_transform_and_cache(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(300, 32)).astype(np.float32)
        fit = np.arange(0, 200); plot = np.arange(300)
        for dims in (2, 3):
            emb = umap_embedding(features, fit, plot, dims, cache_key='toy', seed=0, n_neighbors=10)
            self.assertEqual(emb.points.shape, (300, dims))
            self.assertTrue(np.isfinite(emb.points).all())
            self.assertTrue(emb.cache_path.exists())
            again = umap_embedding(features, fit, plot, dims, cache_key='toy', seed=0, n_neighbors=10)
            np.testing.assert_array_equal(again.points, emb.points)
        with self.assertRaises(ValueError):
            umap_embedding(features, fit, plot, 4, cache_key='toy')
        with self.assertRaises(ValueError):
            umap_embedding(features, np.arange(0), plot, 2, cache_key='toy')

    def test_pca_and_latent_projections(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(200, 32)).astype(np.float32)
        features[:, 5] *= 10; features[:, 17] *= 5; features[:, 2] *= 3   # make the top-variance dims unambiguous
        fit = np.arange(120); plot = np.arange(200)
        pts, labels = project(features, fit, plot, 2, 'pca', cache_key='toy')
        self.assertEqual(pts.shape, (200, 2)); self.assertTrue(labels[0].startswith('PC1 ('))
        pts, labels = project(features, fit, plot, 3, 'latent', cache_key='toy')
        np.testing.assert_allclose(pts, features[:, [5, 17, 2]], atol=5e-4)  # coordinates are rounded to 3 decimals
        self.assertEqual(labels[0], 'z[5]  (filter 0, t 5)')
        self.assertEqual(choose_latent_dims(features, fit, 2, latent_dims=(1, 4, 9)), (1, 4))
        with self.assertRaises(ValueError):
            choose_latent_dims(features, fit, 3, latent_dims=(1, 4))
        with self.assertRaises(ValueError):
            choose_latent_dims(features, fit, 2, latent_dims=(1, 40))
        with self.assertRaises(ValueError):
            project(features, fit, plot, 2, 'tsne', cache_key='toy')

    def test_task_traces_use_fixed_colours_and_symbols(self):
        meta = toy_meta({'LG': 30, 'ST': 5, 'SA': 10})
        pts2 = np.random.default_rng(1).normal(size=(45, 2))
        fig = scatter_by_task(pts2, meta['mode'], title='t')
        names = [tr.name.split(' ')[0] for tr in fig.data]
        self.assertEqual(names, ['LG', 'SA', 'ST'])  # fixed TASKS order, absent tasks skipped
        for tr in fig.data:
            task = tr.name.split(' ')[0]
            self.assertEqual(tr.marker.color, TASK_COLORS[task])
            self.assertEqual(tr.marker.symbol, TASK_SYMBOLS_2D[task])
            self.assertEqual(tr.legendgroup, task)
            self.assertNotIn('OOD', tr.name); self.assertNotIn('ID', tr.name.split(' ')[0])
        self.assertTrue(all(tr.type == 'scatter' for tr in fig.data))  # 2-D is SVG: no WebGL context
        fig3 = scatter_by_task(np.random.default_rng(2).normal(size=(45, 3)), meta['mode'])
        self.assertTrue(all(tr.type == 'scatter3d' for tr in fig3.data))
        value_fig = scatter_by_value(pts2, np.exp(np.linspace(-3, 3, 45)), meta['mode'], log10=True)
        self.assertEqual(len(value_fig.data), 1)
        self.assertAlmostEqual(float(np.max(value_fig.data[0].marker.color)), 3 / np.log(10), places=5)

    def test_compose_panels_and_selector(self):
        meta = toy_meta({'LG': 30, 'ST': 5, 'SA': 10}); rng = np.random.default_rng(3)
        views = {'a': {}, 'b': {}}; axes = {'umap': ['UMAP 1', 'UMAP 2'], 'pca': ['PC1', 'PC2']}
        for v in views:
            for p in axes:
                views[v][p] = scatter_by_task(rng.normal(size=(45, 2)), meta['mode'], title=f'{v} title')
        fig = compose_2d(views, axes, title='t')
        self.assertEqual(len(fig.data), 12)                      # 2 views x 2 panels x 3 tasks
        self.assertEqual([tr.visible for tr in fig.data], [True] * 6 + [False] * 6)
        self.assertEqual(sum(tr.showlegend is not False for tr in fig.data[:6]), 3)  # legend once per view
        self.assertEqual({tr.xaxis for tr in fig.data}, {'x', 'x2'})
        buttons = fig.layout.updatemenus[0].buttons
        self.assertEqual([b.label for b in buttons], ['a', 'b'])
        self.assertEqual(list(buttons[1].args[0]['visible']), [False] * 6 + [True] * 6)
        self.assertEqual(buttons[1].args[1]['title.text'], 'b title')
        axes3 = {'umap': ['UMAP 1', 'UMAP 2', 'UMAP 3'], 'pca': ['PC1', 'PC2', 'PC3']}
        views3 = {v: {p: scatter_by_task(rng.normal(size=(45, 3)), meta['mode'], title=f'{v} title')
                      for p in axes3} for v in ('a', 'b')}
        fig3 = compose_3d(views3, axes3, title='t')
        self.assertEqual(len(fig3.data), 12)
        self.assertEqual([tr.visible for tr in fig3.data], [True] * 6 + [False] * 6)
        self.assertEqual({tr.scene for tr in fig3.data}, {'scene', 'scene2'})   # one WebGL scene per projection
        self.assertEqual(fig3.layout.scene2.xaxis.title.text, 'PC1')
        buttons = fig3.layout.updatemenus[0].buttons
        self.assertEqual([b.label for b in buttons], ['a', 'b'])
        self.assertEqual(list(buttons[1].args[0]['visible']), [False] * 6 + [True] * 6)
        self.assertEqual(sum(1 for f in (fig, fig3) for tr in f.data if tr.type == 'scatter3d'), 12)

    def test_grid_and_sweep_views(self):
        meta = toy_meta({'LG': 30, 'ST': 5, 'SA': 10}); rng = np.random.default_rng(4)
        figs = {f'run {i}': scatter_by_task(rng.normal(size=(45, 2)), meta['mode'], webgl=True) for i in range(5)}
        grid = ux.compose_grid(figs, cols=3, title='g')
        self.assertEqual(len(grid.data), 15)
        self.assertTrue(all(tr.type == 'scattergl' for tr in grid.data))
        self.assertEqual(sum(tr.showlegend is not False for tr in grid.data), 3)
        self.assertEqual(grid.layout.xaxis5.title.text, 'UMAP 1')
        n = 7 * 40; meta = toy_meta({t: 40 for t in TASKS}); labels = meta['mode'].isin(['SA', 'SD', 'TR', 'ST']).to_numpy().astype(int)
        runs = []
        for beta, sample in ((1., True), (4., True), (0., False)):
            run = Path(self.tmp.name) / f'original_b{beta:g}_seed0'; run.mkdir(); runs.append(run)
            Z = rng.normal(size=(n, 32)).astype(np.float32)
            for split in ('val', 'test'):
                np.savez(run / f'latents_{split}.npz', Z=Z, logvar=np.full_like(Z, -2.), labels=labels, smooth_latent_lof=rng.random(n))
                meta.to_parquet(run / f'latents_{split}.parquet', index=False)
            np.save(run / 'train_reference_mu.npy', rng.normal(size=(100, 32)).astype(np.float32))
            (run / 'report.json').write_text(json.dumps({'config': {'beta': beta, 'sample': sample, 'lr': 1e-3, 'batch_size': 256}}))
            (run / 'eval_val.json').write_text(json.dumps({'scores': {'latent_lof': {'auroc': .7}}}))
        out = ux.sweep_mean_views(runs, dims=(2, 3), max_points=None, reference_points=100, show_figures=False)
        self.assertEqual(out[2]['panels'][0], 'β=1 · val AUROC 0.700 · 32 active dims')
        self.assertTrue(out[2]['panels'][2].startswith('β=0 no sampling'))
        self.assertEqual(len(out[2]['figure'].data), 3 * 7)
        self.assertEqual(len(out[3]['figure'].layout.updatemenus[0].buttons), 3)

    def test_sampled_latents_collapse_to_means_at_zero_variance(self):
        mu = np.random.default_rng(0).normal(size=(4, 32)).astype(np.float32)
        z = sample_latents(mu, np.full_like(mu, -80.), 3, seed=0)
        self.assertEqual(z.shape, (3, 4, 32))
        np.testing.assert_allclose(z, np.broadcast_to(mu, z.shape), atol=1e-6)
        spread = sample_latents(mu, np.zeros_like(mu), 2000, seed=0).std(0)
        self.assertTrue(np.all(np.abs(spread - 1) < .1))

    def test_ae_and_vae_views_on_synthetic_runs(self):
        rng = np.random.default_rng(0)
        meta = toy_meta({t: 40 for t in TASKS})
        n = len(meta); labels = (meta['mode'].isin(['SA', 'SD', 'TR', 'ST'])).to_numpy().astype(int)
        Z = rng.normal(size=(n, 32)).astype(np.float32)
        ae = Path(self.tmp.name) / 'ae'; ae.mkdir()
        np.savez(ae / 'latents_test.npz', Z=Z, labels=labels, psi_smooth=rng.random(n))
        meta.to_parquet(ae / 'latents_test.parquet', index=False)
        views = ux.ae_latent_views(ae, dims=(2, 3), max_points=None, seed=0, reference_points=100,
                                   show_figures=False)
        for d in (2, 3):
            self.assertEqual(set(views[d]['points']), {'umap', 'pca', 'latent'})
            self.assertEqual(views[d]['points']['latent'].shape, (n, d))
            self.assertEqual(len(views[d]['figure'].data), 3 * 7)
        for d in (2, 3):   # a single view: panels only, no buttons
            self.assertEqual(len(views[d]['figure'].layout.updatemenus), 0)
        self.assertEqual({tr.scene for tr in views[3]['figure'].data}, {'scene', 'scene2', 'scene3'})
        self.assertIsNone(ux.ae_latent_views(Path(self.tmp.name) / 'missing', show_figures=False))
        vae = Path(self.tmp.name) / 'vae'; vae.mkdir()
        np.savez(vae / 'latents_test.npz', Z=Z, logvar=np.full_like(Z, -2.), labels=labels,
                 smooth_latent_lof=rng.random(n))
        meta.to_parquet(vae / 'latents_test.parquet', index=False)
        np.save(vae / 'train_reference_mu.npy', rng.normal(size=(120, 32)).astype(np.float32))
        results = ux.vae_latent_views(vae, dims=(2, 3), max_points=None, sample_windows=70,
                                      samples_per_window=2, cloud_windows=1, cloud_samples=20,
                                      show_figures=False)
        for d in (2, 3):
            self.assertEqual(set(results[d]['views']), {'means', 'samples', 'variance'})
            self.assertEqual(results[d]['offsets']['reference'], (0, 120))
            self.assertEqual(results[d]['offsets']['mu'], (120, 120 + n))
            per_projection = 7 + (7 + 7) + 1                       # means, samples + clouds, variance
            self.assertEqual(len(results[d]['figure'].data), 3 * per_projection)
            labels_ = [b.label for b in results[d]['figure'].layout.updatemenus[0].buttons]
            self.assertEqual(labels_, ['means', 'samples', 'variance'])
        # Runs saved before variances were stored degrade gracefully.
        old = Path(self.tmp.name) / 'old'; old.mkdir()
        np.savez(old / 'latents_test.npz', Z=Z, labels=labels)
        meta.to_parquet(old / 'latents_test.parquet', index=False)
        results = ux.vae_latent_views(old, dims=(2,), max_points=None, projections=('pca', 'latent'), show_figures=False)
        self.assertEqual(set(results[2]['views']), {'means'})
        self.assertEqual(len(results[2]['figure'].data), 2 * 7)


class PelvisVariantTests(unittest.TestCase):
    """The pelvis IMU feature sets: raw slicing, scaler, alignment and the separability table."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.raw, self.base = root / 'raw', root / 'processed'
        self.base.mkdir()
        rng = np.random.default_rng(0)
        chans = ['thigh_accel_x_l', 'thigh_accel_y_l', 'thigh_accel_z_l', 'thigh_gyro_x_l', 'thigh_gyro_y_l',
                 'thigh_gyro_z_l', 'enc_angle_l', 'enc_velo_l', 'thigh_accel_x_r', 'thigh_accel_y_r',
                 'thigh_accel_z_r', 'thigh_gyro_x_r', 'thigh_gyro_y_r', 'thigh_gyro_z_r', 'enc_angle_r', 'enc_velo_r']
        # four trials: two training (LG, RA) for the scaler, two test (LG, RD) with windows; window = 20 samples
        self.window = 20
        self.trials = {('AB01', 'LG_1', 'LG'): 120, ('AB01', 'RA_1', 'RA'): 100,
                       ('AB16', 'LG_2', 'LG'): 90, ('AB24', 'RD_1', 'RD'): 80}
        self.raw_values = {}
        for (subject, trial, mode), n in self.trials.items():
            d = self.raw / subject / trial; d.mkdir(parents=True)
            df = pd.DataFrame(rng.normal(size=(n, len(chans) + 6)).astype(np.float32), columns=chans + list(ux.PELVIS_CHANNELS))
            df.insert(0, 'time', np.arange(n) / 200)
            if trial == 'RD_1':
                df.loc[70, 'pelvis_gyro_z'] = np.nan   # one NaN row -> the windows covering it are dropped
            df.to_csv(d / 'exo.csv', index=False)
            self.raw_values[(subject, trial)] = df
        train_meta = pd.DataFrame([dict(subject=s, trial=t, mode=m, end_idx=self.window - 1, row=i)
                                   for i, (s, t, m) in enumerate(k for k in self.trials if k[0] == 'AB01')])
        train_meta.to_parquet(self.base / 'train_meta.parquet')
        rows, X = [], []
        for (subject, trial, mode), n in self.trials.items():
            if subject == 'AB01':
                continue
            for end in range(self.window - 1, n, 10):
                df = self.raw_values[(subject, trial)]
                X.append(df[chans].to_numpy(np.float32)[end - self.window + 1:end + 1].T)
                rows.append(dict(subject=subject, trial=trial, mode=mode, end_idx=end, row=len(rows)))
        pd.DataFrame(rows).to_parquet(self.base / 'test_meta.parquet')
        np.save(self.base / 'test_X.npy', np.stack(X))
        np.savez(self.base / 'scaler.npz', mean=np.zeros(16, np.float32), std=np.ones(16, np.float32), channels=np.array(chans))
        (self.base / 'config.json').write_text(json.dumps({'window_samples': self.window}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_pelvis_windows_are_raw_slices_and_nan_windows_are_marked(self):
        meta = pd.read_parquet(self.base / 'test_meta.parquet')
        out = ux.pelvis_windows(meta, self.raw, window=self.window)
        self.assertEqual(out.shape, (len(meta), 6, self.window))
        r = meta.iloc[3]
        expect = self.raw_values[(r.subject, r.trial)][list(ux.PELVIS_CHANNELS)].to_numpy(np.float32)[r.end_idx - self.window + 1:r.end_idx + 1].T
        np.testing.assert_allclose(out[3], expect)
        nan_rows = np.flatnonzero(np.isnan(out).any(axis=(1, 2)))
        covering = meta.index[(meta.trial == 'RD_1') & (meta.end_idx >= 70) & (meta.end_idx - self.window + 1 <= 70)]
        np.testing.assert_array_equal(nan_rows, covering.to_numpy())
        with self.assertRaises(ValueError):
            ux.pelvis_windows(meta.assign(end_idx=10_000), self.raw, window=self.window)

    def test_pelvis_scaler_uses_training_scaler_trials(self):
        sc = ux.pelvis_scaler(self.base, self.raw)
        rows = pd.concat([self.raw_values[('AB01', 'LG_1')], self.raw_values[('AB01', 'RA_1')]])[list(ux.PELVIS_CHANNELS)].to_numpy(np.float64)
        np.testing.assert_allclose(sc['mean'], rows.mean(0), atol=1e-5)
        np.testing.assert_allclose(sc['std'], rows.std(0), atol=1e-5)
        self.assertEqual((sc['trials'], sc['rows']), (2, 220))

    def test_feature_sets_share_rows_and_drop_nan_windows(self):
        with patch.object(ux, 'stratified_sample', lambda modes, max_points, seed=0, floor=2000: np.arange(len(modes))):
            sets = ux.sensor_window_sets('test', root=self.base, raw_root=self.raw, max_points=None, pca_components=4)
        meta = pd.read_parquet(self.base / 'test_meta.parquet')
        self.assertEqual(set(sets), set(ux.FEATURE_SETS))
        n = len(sets['thigh']['meta'])
        covering = int(((meta.trial == 'RD_1') & (meta.end_idx >= 70) & (meta.end_idx - self.window + 1 <= 70)).sum())
        self.assertGreaterEqual(covering, 1)
        self.assertEqual(sets['thigh']['dropped'], covering)
        self.assertEqual(n, len(meta) - covering)
        for name, s in sets.items():
            self.assertEqual(s['features'].shape, (n, 4))
            self.assertTrue(np.isfinite(s['features']).all())
            pd.testing.assert_frame_equal(s['meta'], sets['thigh']['meta'])
            np.testing.assert_array_equal(s['source_index'], sets['thigh']['source_index'])
        self.assertEqual(len(sets['thigh']['channels']), 16)
        self.assertEqual(len(sets['pelvis']['channels']), 6)
        self.assertEqual(sets['thigh+pelvis']['channels'][16:], list(ux.PELVIS_CHANNELS))
        with self.assertRaises(ValueError):
            ux.sensor_window_sets('test', sets=('knee',), root=self.base, raw_root=self.raw)

    def test_separability_scores_hold_out_participants_or_trials(self):
        rng = np.random.default_rng(1)
        n = 400
        meta = pd.DataFrame({'subject': np.repeat(['AB16', 'AB24', 'AB29', 'AB30'], n // 4),
                             'mode': np.tile(['LG', 'RD', 'ST', 'TR'], n // 4)})
        meta['trial'] = meta['subject'] + '_' + meta['mode'] + '_' + (np.arange(n) % 3).astype(str)
        meta.loc[meta['mode'] == 'ST', 'subject'] = 'AB30'          # standing recorded by one participant only
        # task-informative features: a per-task offset, so k-NN across recordings succeeds
        offsets = {'LG': (0, 0), 'RD': (6, 0), 'ST': (0, 6), 'TR': (6, 6)}
        good = np.array([offsets[m] for m in meta['mode']]) + rng.normal(scale=.5, size=(n, 2))
        noise = rng.normal(size=(n, 2))
        summary, per_task = ux.separability({'good': dict(features=good, umap={2: good}), 'noise': dict(features=noise)}, meta, k=5)
        self.assertEqual(list(summary.index), [('good', 'participant'), ('good', 'trial'), ('noise', 'participant'), ('noise', 'trial')])
        self.assertGreater(summary.loc[('good', 'trial'), 'kNN accuracy, PCA-2'], .95)
        self.assertLess(summary.loc[('noise', 'trial'), 'kNN accuracy, PCA-2'], .5)
        self.assertGreater(summary.loc[('good', 'participant'), 'silhouette, UMAP-2D'], .5)
        self.assertEqual(summary.loc[('good', 'participant'), 'silhouette, UMAP-2D'], summary.loc[('good', 'trial'), 'silhouette, UMAP-2D'])
        self.assertTrue(np.isnan(summary.loc[('noise', 'trial'), 'silhouette, UMAP-2D']))
        self.assertEqual(list(per_task.index), ['LG', 'RD', 'TR', 'ST'])
        # standing has one participant: unscorable when participants are held out, scored when trials are
        self.assertTrue(np.isnan(per_task.loc['ST', ('participant', 'good')]))
        self.assertGreater(per_task.loc['ST', ('trial', 'good')], .95)
        self.assertGreater(per_task.loc['TR', ('participant', 'good')], .95)
        scored = per_task[('participant', 'good')].dropna()
        self.assertAlmostEqual(summary.loc[('good', 'participant'), 'balanced accuracy, PCA'], scored.mean())
        with self.assertRaises(ValueError):
            ux.knn_task_scores(good, meta.assign(subject='AB16'))
        with self.assertRaises(ValueError):
            ux.knn_task_scores(good, meta, held_out='session')
        with self.assertRaises(ValueError):
            ux.separability({'bad': dict(features=good[:10])}, meta)

if __name__ == '__main__':
    unittest.main()
