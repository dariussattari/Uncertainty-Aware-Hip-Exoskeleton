"""The one check for _vae_runs.py: per-dimension KL against the ELBO's KL, active dimensions, widths, and discovery."""
import json
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

import _vae_runs as vr

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'ml' / 'common').is_dir())
sys.path.insert(0, str(REPO / 'ml' / 'common'))
from models.vae import loss_terms  # noqa: E402


def check() -> None:
    rng = np.random.default_rng(0)
    # per-dimension KL sums to the per-window KL of the ELBO, averaged over windows
    z, lv = rng.normal(size = (500, 32)).astype(np.float32), rng.normal(-1, 0.5, size = (500, 32)).astype(np.float32)
    _, kl = loss_terms(torch.zeros(500, 1, 1), torch.zeros(500, 1, 1), torch.from_numpy(z), torch.from_numpy(lv))
    np.testing.assert_allclose(vr.get_kl_per_dim(z, lv).sum(), kl.double().mean().item(), rtol = 1e-5)
    # active dimensions: posterior variance well below the prior's 1 on the selected windows only
    lv2 = np.zeros((10, 3), np.float32)
    lv2[:5, 0], lv2[5:, 1] = -3.0, -3.0
    sel = np.r_[np.ones(5, bool), np.zeros(5, bool)]
    assert vr.get_active(lv2, sel).tolist() == [True, False, False]
    # width by task: the median over a task's windows of the mean variance over active dimensions
    labels = pd.DataFrame({'task': ['walk'] * 5 + ['jump'] * 5})
    width = vr.get_width_by_task(labels, lv2, np.array([True, True, False]))
    np.testing.assert_allclose(width['walk'], (np.exp(-3) + 1) / 2, rtol = 1e-6)
    # discovery: finished VAE runs only, with beta, decoder variance and seed from config.json
    with tempfile.TemporaryDirectory() as tmp:
        for name, model, finished in (('vae_b8_learned_s1', 'vae', True), ('vae_b1_fixed_s0_smoke', 'vae', True),
                                      ('vae_b1_fixed_s1', 'vae', False), ('vanilla_s0', None, True), ('lstm_vae_b1_fixed_s0', 'vae', True)):
            run = Path(tmp) / name
            run.mkdir()
            (run / 'config.json').write_text(json.dumps({'model': model, 'beta': 8.0, 'decoder_variance': 'learned', 'seed': 1,
                                                         'arch': 'lstm' if name.startswith('lstm') else 'conv'}))
            (run / 'summary.json').write_text('{}')
            if finished:
                np.savez(run / 'latents_val.npz', rows = np.arange(3))
        found = vr.discover_vae_runs(Path(tmp))
        assert found['stem'].tolist() == ['vae_b8_learned'] and found['beta'].iloc[0] == 8.0 and found['seed'].iloc[0] == 1
        assert vr.discover_vae_runs(Path(tmp), 'lstm')['stem'].tolist() == ['lstm_vae_b1_fixed']
    print('ok')


if __name__ == '__main__':
    check()
