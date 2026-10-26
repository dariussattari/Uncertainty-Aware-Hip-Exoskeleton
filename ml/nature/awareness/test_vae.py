"""The one check for train_vae.py and the learned decoder variance: the grid, run names, and the ELBO terms."""
import math

import torch

import train_vae as tv
from models.seq_vae import LSTMVAE, TCNVAE, TransformerVAE, get_tcn_levels
from models.vae import DECODER_LOGVAR_FLOOR, ConvVAE, get_decoder_logvar, loss_terms


def check() -> None:
    # the grid: 30 distinct cells, 0-2 the standard VAE, variance slowest, then beta, then seed
    cells = [tv.get_grid_cell(i) for i in range(30)]
    assert len(set(cells)) == 30 and cells[:3] == [(1.0, 'fixed', 0), (1.0, 'fixed', 1), (1.0, 'fixed', 2)]
    assert cells[3] == (8.0, 'fixed', 0) and cells[15] == (1.0, 'learned', 0) and cells[29] == (128.0, 'learned', 2)
    try:
        tv.get_grid_cell(30)
        raise AssertionError('index 30 should be outside the grid')
    except IndexError:
        pass
    assert tv.get_run_name(tv.VaeConfig(beta = 8.0, decoder_variance = 'learned', seed = 2)) == 'vae_b8_learned_s2'
    assert 'variant' not in tv.VaeConfig.__dataclass_fields__             # notebook 05 labels VAE runs by their stem
    # the 3 s + spectral grid: five cells, three seeds; the model keeps a 4 x 8 latent on 23 x 600 windows
    cells3 = [tv.get_grid_cell(i, 'raw_3s_spectral') for i in range(15)]
    assert cells3[0] == (1.0, 'fixed', 0) and cells3[12] == (1.0, 'learned', 0) and len(set(cells3)) == 15
    assert tv.get_run_name(tv.VaeConfig(beta = 32.0, seed = 1, inputs = 'raw_3s_spectral')) == 'vae_raw_3s_spectral_b32_fixed_s1'
    m3 = ConvVAE(23, 600, learned_variance = True, pool = (5, 15)).eval()
    xh, mu3, _ = m3(torch.randn(2, 23, 600))
    assert xh.shape == (2, 23, 600) and mu3.shape == (2, 4, 8) and m3.decoder_logvar.shape == (23,)
    # learned variance: the NLL is the Gaussian log likelihood with each channel's variance, and the floor is soft
    torch.manual_seed(0)
    model = ConvVAE(16, 200, learned_variance = True).eval()
    assert model.decoder_logvar.shape == (16,) and ConvVAE(16, 200).decoder_logvar is None
    with torch.no_grad():
        model.decoder_logvar.copy_(torch.linspace(-2, 1, 16))
    x = torch.randn(4, 16, 200)
    x_hat, mu, logvar = model(x)
    assert x_hat.shape == x.shape and mu.shape == (4, 4, 8)
    nll, kl = loss_terms(x, x_hat, mu, logvar, model.decoder_logvar)
    sd = torch.exp(0.5 * get_decoder_logvar(model.decoder_logvar))[None, :, None]
    reference = -torch.distributions.Normal(x_hat, sd).log_prob(x).flatten(1).sum(1)
    torch.testing.assert_close(nll, reference, rtol = 1e-4, atol = 1e-3)
    unit, _ = loss_terms(x, x_hat, mu, logvar)                             # no decoder variance: unit variance
    torch.testing.assert_close(unit, 0.5 * ((x - x_hat) ** 2 + math.log(2 * math.pi)).flatten(1).sum(1))
    raw = torch.tensor([-50.0, DECODER_LOGVAR_FLOOR, 3.0], requires_grad = True)
    floor = get_decoder_logvar(raw)
    floor.sum().backward()
    assert torch.all(floor >= DECODER_LOGVAR_FLOOR) and abs(floor[2] - 3.0) < 1e-3   # floored below, identity above
    torch.testing.assert_close(raw.grad[1], torch.tensor(0.5))                       # smooth at the floor itself
    # sequence VAEs: the same 4 x 8 latent and reconstruction shape on every input setting, finite ELBO, gradients reach
    # the encoder; core cells and run names for the other architectures and inputs
    for arch, cls in (('lstm', LSTMVAE), ('transformer', TransformerVAE)):
        for channels, window in ((16, 200), (23, 200), (16, 600), (23, 600)):
            m = cls(channels, window, learned_variance = True)
            x = torch.randn(3, channels, window)
            xh, mu_s, lv_s = m(x)
            assert xh.shape == x.shape and mu_s.shape == lv_s.shape == (3, 4, 8), (arch, channels, window)
            nll_s, kl_s = loss_terms(x, xh, mu_s, lv_s, m.decoder_logvar)
            (nll_s + kl_s).mean().backward()
            first = next(m.parameters())
            assert torch.isfinite(nll_s).all() and first.grad is not None and first.grad.abs().sum() > 0
        assert tv.get_cells(arch, 'raw_3s') == tv.CORE_CELLS and tv.get_grid_cell(11, 'raw_1s', arch) == (1.0, 'learned', 2)
    assert tv.get_cells('conv', 'raw_1s_spectral') == tv.CORE_CELLS and len(tv.get_cells('conv', 'raw_1s')) == 10
    assert tv.get_run_name(tv.VaeConfig(arch = 'lstm', inputs = 'raw_3s', beta = 32.0)) == 'lstm_vae_raw_3s_b32_fixed_s0'
    assert tv.get_run_name(tv.VaeConfig(arch = 'transformer')) == 'xformer_vae_b1_fixed_s0'
    # TCN-VAE: 4 x 8 latent at both window lengths, the receptive field covers the window, and the encoder is causal
    assert get_tcn_levels(200) == 6 and get_tcn_levels(600) == 8
    torch.manual_seed(0)
    for c, w in ((16, 200), (23, 600)):
        tcn = TCNVAE(c, w, learned_variance = True).eval()
        xh, mu_t, lv_t = tcn(torch.randn(2, c, w))
        assert xh.shape == (2, c, w) and mu_t.shape == (2, 4, 8) and lv_t.shape == (2, 4, 8) and tcn.decoder_logvar.shape == (c,)
    tcn = TCNVAE(16, 200).eval()
    x0 = torch.randn(1, 16, 200)
    x1 = x0.clone(); x1[:, :, 150:] += 5.0                             # change only the last 50 samples
    with torch.no_grad():
        h0, h1 = tcn.encoder(x0), tcn.encoder(x1)
    torch.testing.assert_close(h0[..., :150], h1[..., :150])         # earlier outputs never see later samples
    assert not torch.allclose(h0[..., 150:], h1[..., 150:])
    assert tv.get_run_name(tv.VaeConfig(arch = 'tcn', inputs = 'raw_3s', seed = 1)) == 'tcn_vae_raw_3s_b1_fixed_s1'
    zt = torch.randn(3, 4, 8)                                         # the upsampling matrix is linear interpolation
    torch.testing.assert_close(torch.einsum('bhl,lw->bhw', tcn.latent_in(zt), tcn.upsample),
                               torch.nn.functional.interpolate(tcn.latent_in(zt), size = 200, mode = 'linear', align_corners = False))
    print('ok')


if __name__ == '__main__':
    check()
