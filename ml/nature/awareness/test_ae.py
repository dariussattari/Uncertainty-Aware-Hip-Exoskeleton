"""The one check for the Nature autoencoder code: windows, split, gathering, the scale-free sparsity penalty, and the
gaussian variant (linear latent, Gaussian NLL)."""
import numpy as np
import pandas as pd
import torch

import ae_data as ad
import train_ae as ta
from models.autoencoder import ConvAutoencoder
from models.vae import loss_terms


def check() -> None:
    # a window is valid only if all its samples are finite in every channel
    x = np.zeros((450, 3), dtype = np.float32)
    x[205, 1] = np.nan
    ends = ad.get_valid_ends(x, window = 200, stride = 10)
    assert ends[0] == 199 and not ((ends >= 205) & (ends < 405)).any() and 409 in ends
    # split by person, both strata in test and validation
    inv = pd.DataFrame({'subject': [f'P{i:02d}' for i in range(10)] + ['P00', 'P05'],
                        'phase': ['phase1-2'] * 5 + ['phase3'] * 5 + ['phase3', 'phase1-2']})
    split = ad.get_participant_split(inv)
    assert set(split) == set(inv.subject) and sorted(split.values()).count('test') == 2 * ad.N_TEST
    has3 = inv.groupby('subject').phase.agg(lambda p: 'phase3' in set(p))
    assert {has3[p] for p, s in split.items() if s == 'test'} == {True, False}
    # gathering returns (batch, channels, time) slices starting at each start
    signals = torch.arange(60, dtype = torch.float32).reshape(20, 3)
    w = ta.gather_windows(signals, torch.tensor([0, 7]), 5)
    assert w.shape == (2, 3, 5) and torch.equal(w[1], signals[7:12].T)
    # a batch is the raw channels, then each per-window feature held constant across the window
    data = ta.AeData(signals = signals, features = torch.tensor([[1.0, 2.0], [3.0, 4.0]]), starts = torch.tensor([0, 7]),
                     windows = pd.DataFrame(), train = torch.tensor([0]), val = torch.tensor([1]),
                     mean = np.zeros(5, np.float32), std = np.ones(5, np.float32), window = 5, data_version = 'test')
    b = ta.get_batch(data, torch.tensor([1, 0]))
    assert b.shape == (2, 5, 5) and torch.equal(b[0, :3], signals[7:12].T) and torch.equal(b[0, 3:], torch.tensor([[3.0] * 5, [4.0] * 5]))
    # penalty: 1 for constant units, sqrt(p) for units active in a fraction p, blind to each unit's scale
    z = torch.zeros(100, 4)
    z[:, 0] = 2.0
    z[:25, 1] = 1.0
    z[:4, 2] = 3.0
    torch.testing.assert_close(ta.get_sparsity_penalty(z), torch.tensor((1 + 0.5 + 0.2 + 0) / 4))
    scaled = z * torch.tensor([1e-2, 50.0, 0.1, 7.0])           # invariant above the rms floor sqrt(PENALTY_EPS) = 1e-4
    torch.testing.assert_close(ta.get_sparsity_penalty(scaled), ta.get_sparsity_penalty(z), rtol = 1e-4, atol = 1e-5)
    # whole-set moments give the same penalty as one batch holding every window, in any order
    zz = torch.rand(64, 32) * (torch.rand(64, 32) > 0.7)
    torch.testing.assert_close(ta.get_penalty_from_moments(zz.abs().mean(0), zz.pow(2).mean(0)), ta.get_sparsity_penalty(zz))
    # run names record only what differs from the defaults
    names = [ta.get_run_name(ta.AeConfig(variant = v, seed = 1, inputs = i, sparsity_weight = w, latent = l))
             for v, i, w, l in (('vanilla', 'raw_1s', 0.0, '4x8'), ('sparse', 'raw_3s_spectral', ta.SPARSITY_WEIGHT, '4x8'),
                                ('sparse', 'raw_3s_spectral', 0.1, '8x4'))]
    assert names == ['vanilla_s1', 'sparse_raw_3s_spectral_s1', 'sparse_raw_3s_spectral_l0.1_8x4_s1'], names
    # gaussian variant: the encoder ends at the latent conv, so the latent takes negative values, and the objective is
    # the Science Robotics VAE's Gaussian NLL (per-window sum, averaged over the batch)
    torch.manual_seed(0)
    linear = ConvAutoencoder(16, 200, linear_latent = True).eval()
    assert isinstance(linear.encoder[-1], torch.nn.Conv1d) and len(linear.encoder) == len(ConvAutoencoder(16, 200).encoder) - 2
    xb = torch.randn(8, 16, 200)
    x_hat, zb = linear(xb)
    assert zb.shape == (8, 4, 8) and (zb < 0).any()
    cfg = ta.AeConfig(**ta.VARIANTS['gaussian'])
    nll, _ = loss_terms(xb, x_hat, torch.zeros(8, 4, 8), torch.zeros(8, 4, 8))
    got = ta.get_reconstruction_objective(torch.nn.functional.mse_loss(x_hat, xb), xb[0].numel(), cfg)
    torch.testing.assert_close(got, nll.mean())
    assert ta.get_reconstruction_objective(torch.tensor(0.5), 3200, ta.AeConfig()) == 0.5
    assert ta.get_run_name(ta.AeConfig(variant = 'gaussian', seed = 2, inputs = 'raw_3s_spectral')) == 'gaussian_raw_3s_spectral_s2'
    print('ok')


if __name__ == '__main__':
    check()
