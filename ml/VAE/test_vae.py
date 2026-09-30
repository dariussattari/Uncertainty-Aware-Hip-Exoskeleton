"""Wiring tests on tiny synthetic arrays, NOT benchmark runs.

    .venv/bin/python ml/VAE/test_vae.py
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vae_train as vae                       # noqa: E402
from vae_model import ConvVAE, loss_terms    # noqa: E402
from vae_paths import ROOT                    # noqa: E402

torch.set_num_threads(2)


class ToySplit:
    def __init__(self, n, subject, mode, seed):
        self.x = np.random.default_rng(seed).normal(size=(n, 16, 200)).astype('float32')
        self.meta = pd.DataFrame({'subject': [subject] * n, 'trial': ['trial1'] * n,
                                  'mode': [mode] * n, 'end_idx': 199 + 10 * np.arange(n)})

    def __len__(self): return len(self.x)
    def take(self, idx): return self.x[idx].copy()


class ToyData:
    n_channels = 16
    window = 200
    channels = ['enc_angle_l'] + [f'channel_{i}' for i in range(15)]

    def __init__(self):
        self.splits = {s: ToySplit(n, s, 'LG', i) for i, (s, n) in
                       enumerate([('train', 32), ('val', 8), ('test', 8)])}
        self.train, self.val, self.test = (self.splits[s] for s in ('train', 'val', 'test'))
        self.ood = {s: ToySplit(8, s, 'SA', 10 + i) for i, s in enumerate(('val', 'test'))}

    def train_val_indices(self, *args): return np.arange(24), np.arange(24, 32), ['internal-validation']

    def loader(self, split, indices=None, **kwargs):
        sp = self.splits[split] if isinstance(split, str) else split
        idx = np.arange(len(sp)) if indices is None else indices
        for i in range(0, len(idx), 8): yield torch.from_numpy(sp.take(idx[i:i + 8]))


class VAETests(unittest.TestCase):
    def test_shapes_gradients_and_eval_determinism(self):
        torch.manual_seed(0); m = ConvVAE(); x = torch.randn(3, 16, 200)
        rec, mu, lv = m(x); self.assertEqual(tuple(mu.shape), (3, 4, 8))
        self.assertEqual(rec.shape, x.shape)
        nll, kl = loss_terms(x, rec, mu, lv)
        self.assertTrue(torch.all(kl >= 0)); (nll + 4 * kl).mean().backward()
        self.assertTrue(torch.isfinite(m.mu_head.weight.grad).all())
        self.assertGreater(m.logvar_head.weight.grad.abs().sum(), 0)
        m.eval()
        with torch.inference_mode(): a = m(x)[0]; b = m(x)[0]
        torch.testing.assert_close(a, b)

    def test_kl_zero_at_prior(self):
        x = torch.zeros(2, 16, 200); mu = torch.zeros(2, 4, 8)
        _, kl = loss_terms(x, x, mu, torch.zeros_like(mu))
        torch.testing.assert_close(kl, torch.zeros(2))

    def test_overwrite_protection_and_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'; out.mkdir(); (out / 'existing').touch()
            with self.assertRaises(FileExistsError): vae.train_experiment(out, vae.VAEConfig())
        self.assertEqual(len(vae.CELLS), 8)
        self.assertEqual(vae.run_dir('beta4', 'SA', 0, root='/r'), Path('/r/SA_beta4_seed0'))
        self.assertEqual(vae.run_dir('standard', 'original', 1, smoke=True, root='/r'),
                         Path('/r/original_standard_seed1_smoke'))

    def test_tiny_training_scoring_and_plot_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(vae, 'data_for', return_value=ToyData()):
            out = Path(tmp) / 'toy'
            cfg = vae.VAEConfig(max_epochs=2, patience=1, mc_samples=2,
                                lof_fit_samples=32, device='cpu', batch_size=8)
            model, dat, report, ev = vae.train_experiment(out, cfg)
            self.assertEqual(set(ev), {'val', 'test'})
            self.assertEqual(set(ev['test']['scores']), {'latent_lof', 'reconstruction_nll', 'kl', 'negative_elbo'})
            for values in ev['test']['scores'].values():
                self.assertTrue(np.isfinite(values['threshold']))
                self.assertEqual(sum(values['counts'].values()), 16)
            saved = np.load(out / 'latents_test.npz')
            self.assertEqual(saved['Z'].shape, (16, 32)); self.assertEqual(saved['logvar'].shape, (16, 32))
            self.assertTrue(np.all(np.abs(saved['logvar']) <= 8))
            self.assertTrue((out / 'final.pt').exists())
            import vae_plots
            with patch.object(vae_plots, 'data_for', return_value=ToyData()):
                vae_plots.plot_representation(out); plt.close('all')

    def test_control_variant_is_deterministic_ae_objective(self):
        self.assertEqual(vae.VARIANTS['ae_control'], dict(beta=0., sample=False))
        self.assertEqual(len(vae.GROUPS['sweep']), 7)
        self.assertTrue(all(p == 'original' for _, p in vae.GROUPS['sweep']))
        with tempfile.TemporaryDirectory() as tmp, patch.object(vae, 'data_for', return_value=ToyData()):
            cfg = vae.VAEConfig(max_epochs=1, patience=1, mc_samples=4, lof_fit_samples=32, device='cpu',
                                batch_size=8, **vae.VARIANTS['ae_control'])
            _, _, report, ev = vae.train_experiment(Path(tmp) / 'ctl', cfg)
            self.assertEqual(report['config']['beta'], 0.); self.assertFalse(report['config']['sample'])
            # with beta=0 and no sampling the objective is the unit-variance Gaussian NLL alone
            self.assertAlmostEqual(report['history'][0]['train_loss'], report['history'][0]['train_nll'], places=6)
            self.assertEqual(set(ev['test']['scores']), {'latent_lof', 'reconstruction_nll', 'kl', 'negative_elbo'})

    @unittest.skipUnless((ROOT / 'data/processed_all/AE_GAN/train_X.npy').exists(),
                         'mode-complete arrays not built here')
    def test_original_protocol_view_matches_documented_split_sizes(self):
        d = vae.original_protocol_view(256)
        self.assertEqual([len(d.splits[s]) for s in ('train', 'val', 'test')], [187_993, 64_350, 61_956])
        self.assertEqual(len(d.ood['test']), 23_043)
        self.assertEqual(sorted(d.train.meta['mode'].unique()), ['LG', 'RA', 'RD'])
        self.assertEqual(sorted(d.ood['test'].meta['mode'].unique()), ['SA', 'SD', 'ST', 'TR'])


if __name__ == '__main__':
    unittest.main()
