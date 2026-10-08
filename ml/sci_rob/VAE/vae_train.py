"""Train and evaluate one VAE cell (variant x protocol); also the SLURM array entry point.

    python ml/sci_rob/VAE/vae_train.py --list
    python ml/sci_rob/VAE/vae_train.py --index 3                          # one SLURM array cell
    python ml/sci_rob/VAE/vae_train.py --variant beta4 --protocol SA --smoke

Cells: ``standard`` (beta=1) and ``beta4`` (beta=4), each under ``original`` (the
participant-split protocol, LG/RA/RD in-distribution) and the ``SA``/``ST``/``TR``
leave-one-task-out holdouts. Results go to ``ml/sci_rob/VAE/runs/<protocol>_<variant>_seed<seed>/``,
which is where ``notebooks/sci_rob/model_exploration/08_vae_variants.ipynb`` reads them.

Protocol and scoring follow the AE baseline: participant-disjoint early stopping inside the
training pool, LOF fitted on in-distribution training posterior means only, the existing
per-trial causal median filter, the 99.5th-percentile training threshold, and the shared
metric functions. Reconstruction NLL, KL and negative ELBO are saved as separate diagnostic
scores. Nothing trains on import.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from vae_paths import RUNS
from models.vae import ConvVAE, loss_terms
from models.autoencoder import LatentLOF
from evaluation.common import classification_metrics, smooth_by_trial
from training.common import pick_device

__all__ = ['VAEConfig', 'VARIANTS', 'PROTOCOLS', 'GROUPS', 'CELLS', 'run_dir', 'data_for',
           'original_protocol_view', 'train_experiment']

# Variant name -> VAEConfig overrides. `standard`/`beta4` are the notebook's matrix; the beta
# sweep and the two beta=0 controls answer whether beta=4's gain came from regularization or
# from a smaller effective code, and whether the VAE hyperparameters (not the Gaussian latent)
# explain the gap to the AE. `sample=False` with beta=0 is the AE objective through this pipeline.
VARIANTS = {
    'standard': dict(beta=1.0), 'beta4': dict(beta=4.0),
    'beta0.25': dict(beta=.25), 'beta2': dict(beta=2.), 'beta8': dict(beta=8.),
    'beta16': dict(beta=16.), 'beta32': dict(beta=32.),
    'ae_control': dict(beta=0., sample=False),
    'ae_control_aehp': dict(beta=0., sample=False, lr=3e-3, batch_size=1024, max_epochs=200, patience=5),
}
PROTOCOLS = ('original', 'SA', 'ST', 'TR')
GROUPS = {
    'matrix': [(v, p) for v in ('standard', 'beta4') for p in PROTOCOLS],              # 8 cells, vae.sbatch
    'sweep': [(v, 'original') for v in ('beta0.25', 'beta2', 'beta8', 'beta16', 'beta32',
                                        'ae_control', 'ae_control_aehp')],              # 7 cells, sweep.sbatch
}
CELLS = GROUPS['matrix']
ORIGINAL_ID = ('LG', 'RA', 'RD')
ORIGINAL_OOD = ('SA', 'SD', 'ST', 'TR')


@dataclass
class VAEConfig:
    beta: float = 1.0
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 5
    seed: int = 0
    val_fraction: float = .2
    held_out: str | None = None   # None: original protocol; or SA / ST / TR
    device: str | None = None
    max_train_windows: int | None = None  # optional SMOKE cap; not a full benchmark
    lof_fit_samples: int = 50_000
    mc_samples: int = 8
    sample: bool = True   # False: z = mu everywhere (with beta=0 this is the AE objective)
    data_root: str | None = None   # alternative mode-complete arrays (the 22-channel pelvis ablation)


def run_dir(variant, protocol, seed=0, smoke=False, root=RUNS):
    return Path(root) / (f'{protocol}_{variant}_seed{seed}' + ('_smoke' if smoke else ''))


# ------------------------------------------------------------------------------ data

def data_for(cfg):
    """held_out=None: the original protocol; SA/ST/TR: the existing LOTO builder."""
    if cfg.held_out:
        from splits import build
        return build('autoencoder', cfg.held_out, batch_size=cfg.batch_size, data_root=cfg.data_root)
    if cfg.data_root:
        raise ValueError('data_root is only supported with a holdout')
    from dataset import AutoencoderData
    try:
        return AutoencoderData(batch_size=cfg.batch_size)
    except FileNotFoundError:
        return original_protocol_view(cfg.batch_size)


def original_protocol_view(batch_size):
    """The original ID/OOD partition (LG/RA/RD versus SA/SD/ST/TR) over the mode-complete arrays.

    Restricted to LG/RA/RD, ``data/processed_all`` reproduces ``data/processed`` window for
    window and copies its scaler (see ml/sci_rob/LOTO/README.md), so on a machine that only holds the
    mode-complete arrays -- the cluster -- this is the same protocol, not a substitute.
    """
    from splits import DATA_ROOT, restrict
    import dataset as D
    dat = D.AutoencoderData(root=DATA_ROOT, batch_size=batch_size)
    ood = {}
    for name, sp in list(dat.splits.items()):
        mode = sp.meta['mode'].to_numpy()
        dat.splits[name] = restrict(sp, np.flatnonzero(np.isin(mode, ORIGINAL_ID)))
        if name != 'train':
            ood[name] = restrict(sp, np.flatnonzero(np.isin(mode, ORIGINAL_OOD)), as_inputs_only=True)
    dat.train, dat.val, dat.test = (dat.splits[s] for s in ('train', 'val', 'test'))
    dat.ood = ood; dat.held_out_task = None; dat.trained_modes = list(ORIGINAL_ID)
    return dat


# ------------------------------------------------------------------------------ training

def _epoch(model, loader, device, beta, opt=None, sample=True):
    model.train(opt is not None)
    totals = np.zeros(3); count = 0
    for x in loader:
        x = x.to(device)
        with torch.set_grad_enabled(opt is not None):
            rec, mu, lv = model(x, sample=sample)
            nll, kl = loss_terms(x, rec, mu, lv)
            loss = (nll + beta * kl).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite VAE objective')
            if opt is not None:
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
        totals += np.array([loss.item(), nll.mean().item(), kl.mean().item()]) * len(x)
        count += len(x)
    return dict(zip(('loss', 'nll', 'kl'), (totals / max(count, 1)).tolist()))


@torch.inference_mode()
def encode_means(model, loader, device):
    model.eval()
    return np.concatenate([model.encode(x.to(device))[0].flatten(1).cpu().numpy() for x in loader])


@torch.inference_mode()
def score_split(model, loader, device, mc_samples, sample=True):
    """Posterior means, posterior log variances and per-window diagnostic scores."""
    model.eval(); means = []; logvars = []; nlls = []; kls = []
    for x in loader:
        x = x.to(device); mu, lv = model.encode(x)
        values = []
        for _ in range(mc_samples if sample else 1):
            z = mu + torch.exp(.5 * lv) * torch.randn_like(mu) if sample else mu
            nll, kl = loss_terms(x, model.decoder(z), mu, lv); values.append(nll)
        means.append(mu.flatten(1).cpu().numpy()); logvars.append(lv.flatten(1).cpu().numpy())
        nlls.append(torch.stack(values).mean(0).cpu().numpy()); kls.append(kl.cpu().numpy())
    nll = np.concatenate(nlls); kl = np.concatenate(kls)
    return (np.concatenate(means), np.concatenate(logvars),
            {'reconstruction_nll': nll, 'kl': kl, 'negative_elbo': nll + kl})


def train_experiment(out, cfg):
    """Explicit opt-in training/evaluation; refuses to overwrite existing runs."""
    if cfg.beta < 0 or cfg.mc_samples < 1 or cfg.max_epochs < 1 or cfg.patience < 1:
        raise ValueError('Invalid beta, Monte Carlo sample count, epoch count or patience')
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    dat = data_for(cfg); device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    tr, va, val_subjects = dat.train_val_indices(cfg.val_fraction, cfg.seed)
    if cfg.max_train_windows is not None:
        tr = np.sort(rng.choice(tr, min(len(tr), cfg.max_train_windows), replace=False))
        va = np.sort(rng.choice(va, min(len(va), cfg.max_train_windows), replace=False))
    model = ConvVAE(dat.n_channels, dat.window).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best = float('inf'); best_state = None; best_epoch = -1; history = []
    for epoch in range(cfg.max_epochs):
        train = _epoch(model, dat.loader('train', indices=tr, shuffle=True, seed=cfg.seed + epoch),
                       device, cfg.beta, opt, cfg.sample)
        # A fixed Monte Carlo stream avoids changing the validation noise every epoch.
        torch.manual_seed(cfg.seed + 10_000)
        val = _epoch(model, dat.loader('train', indices=va), device, cfg.beta, sample=cfg.sample)
        row = {'epoch': epoch, **{f'train_{k}': v for k, v in train.items()},
               **{f'val_{k}': v for k, v in val.items()}}; history.append(row)
        print(f"epoch {epoch + 1}: train={train['loss']:.2f} val={val['loss']:.2f} KL={val['kl']:.2f}", flush=True)
        if val['loss'] < best:
            best = val['loss']; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= cfg.patience: break
    model.load_state_dict(best_state); model.eval()
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'model': best_state, 'config': asdict(cfg)}, out / 'final.pt')
    # Fit LOF on ID train only, just as the AE baseline. Never on test latents.
    fit_idx = np.sort(rng.choice(len(dat.train), min(len(dat.train), cfg.lof_fit_samples), replace=False))
    fit_mu = encode_means(model, dat.loader('train', indices=fit_idx), device)
    lof = LatentLOF(fit_samples=None, seed=cfg.seed).fit(fit_mu)
    thresholds = {}; evaluations = {}
    for split in ('train', 'val', 'test'):
        torch.manual_seed(cfg.seed + 20_000)
        sp = dat.splits[split]; mu, lv, raw = score_split(model, dat.loader(sp), device, cfg.mc_samples, cfg.sample)
        raw['latent_lof'] = lof.score(mu)
        meta = sp.meta.copy().reset_index(drop=True); meta['kind'] = 'ID'
        if split != 'train':
            ood = dat.ood[split]
            om, olv, os = score_split(model, dat.loader(ood), device, cfg.mc_samples, cfg.sample)
            os['latent_lof'] = lof.score(om); mu = np.concatenate([mu, om]); lv = np.concatenate([lv, olv])
            raw = {k: np.r_[v, os[k]] for k, v in raw.items()}
            other = ood.meta.copy(); other['kind'] = 'OOD'; meta = pd.concat([meta, other], ignore_index=True)
        smooth = {k: smooth_by_trial(v, meta, 10, 'median') for k, v in raw.items()}
        if split == 'train':
            thresholds = {k: float(np.percentile(v, 99.5)) for k, v in smooth.items()}
            continue
        lab = (meta.kind == 'OOD').to_numpy().astype(int)
        scores = {k: classification_metrics(lab, v, thresholds[k]) | {'threshold': thresholds[k]}
                  for k, v in smooth.items()}
        rows = []
        for mode, g in meta.groupby('mode', sort=False):
            idx = g.index.to_numpy(); s = smooth['latent_lof'][idx]
            rows.append({'mode': mode, 'kind': g.kind.iloc[0], 'windows': len(idx),
                         'median_psi': float(np.median(s)),
                         'flagged_%': float(100 * np.mean(s > thresholds['latent_lof']))})
        report = {'model': 'conv_vae', 'split': split, 'scores': scores, 'per_mode': rows,
                  'primary_score': 'latent_lof', 'held_out': cfg.held_out,
                  'smoke_run': cfg.max_train_windows is not None}
        (out / f'eval_{split}.json').write_text(json.dumps(report, indent=2))
        # Z = posterior means; logvar = posterior log variances (same rows), for the UMAP views.
        np.savez_compressed(out / f'latents_{split}.npz', Z=mu, logvar=lv, labels=lab,
                            **{f'smooth_{k}': v for k, v in smooth.items()})
        meta.to_parquet(out / f'latents_{split}.parquet', index=False)
        evaluations[split] = report
    report = {'config': asdict(cfg), 'history': history, 'best_epoch': best_epoch,
              'val_subjects': list(val_subjects), 'thresholds': thresholds,
              'smoke_run': cfg.max_train_windows is not None, 'latent_dim': 32}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    np.save(out / 'train_reference_mu.npy', fit_mu)
    return model, dat, report, evaluations


# ------------------------------------------------------------------------------ CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--group', choices=GROUPS, default='matrix', help='which cell list --index refers to')
    parser.add_argument('--index', type=int, help='cell index within --group (SLURM array id)')
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--protocol', choices=PROTOCOLS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-root', default=str(RUNS))
    parser.add_argument('--device', default=None)
    parser.add_argument('--smoke', action='store_true',
                        help='two epochs on 20,000 training windows: wiring only, never a result')
    parser.add_argument('--force', action='store_true', help='delete and redo an existing run')
    parser.add_argument('--list', action='store_true')
    a = parser.parse_args(argv)
    if a.list:
        for group, cells in GROUPS.items():
            print(f'[{group}]')
            for i, (variant, protocol) in enumerate(cells):
                out = run_dir(variant, protocol, a.seed, root=a.out_root)
                state = 'done' if (out / 'eval_test.json').exists() else 'partial' if out.exists() else 'not run'
                print(f'  {i}  {variant:16s} {protocol:9s} {state:8s} {out}')
        extra = sorted(p.name for p in Path(a.out_root).glob('*_seed*') if p.name.endswith(tuple(f'seed{s}' for s in range(1, 10))))
        if extra:
            print('[other seeds]'); [print('  ', n) for n in extra]
        return
    if a.index is not None:
        variant, protocol = GROUPS[a.group][a.index]
    elif a.variant and a.protocol:
        variant, protocol = a.variant, a.protocol
    else:
        parser.error('give --index, or both --variant and --protocol')
    out = run_dir(variant, protocol, a.seed, a.smoke, root=a.out_root)
    if (out / 'eval_test.json').exists() and not a.force:
        print(f'already complete: {out}  (use --force to redo)'); return
    if out.exists():
        print(f'removing incomplete or forced run: {out}'); shutil.rmtree(out)
    overrides = {**VARIANTS[variant],
                 **({'max_epochs': 2, 'patience': 1, 'max_train_windows': 20_000,
                     'lof_fit_samples': 5_000, 'mc_samples': 2} if a.smoke else {})}
    cfg = VAEConfig(seed=a.seed, device=a.device, held_out=None if protocol == 'original' else protocol, **overrides)
    print(f'=== VAE {variant} (beta={cfg.beta}, sample={cfg.sample}, lr={cfg.lr}, batch={cfg.batch_size}) | protocol {protocol} | seed {a.seed}'
          f'{" | SMOKE" if a.smoke else ""} -> {out}', flush=True)
    t0 = time.time()
    _, _, report, evaluations = train_experiment(out, cfg)
    hours = (time.time() - t0) / 3600
    (out / 'timing.json').write_text(json.dumps({'hours': round(hours, 4), 'smoke': a.smoke,
                                                 'best_epoch': report['best_epoch'],
                                                 'epochs_run': len(report['history'])}, indent=2))
    print(f'--- finished in {hours:.2f} h; best epoch {report["best_epoch"] + 1} of {len(report["history"])}')
    for name, m in evaluations['test']['scores'].items():
        print(f'    {name:18s} AUROC {m["auroc"]:.4f}  recall {m["recall"]:6.2f}%  specificity {m["specificity"]:6.2f}%')


if __name__ == '__main__':
    main()
