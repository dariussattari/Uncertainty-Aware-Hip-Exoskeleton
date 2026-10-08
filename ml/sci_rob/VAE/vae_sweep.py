"""Summarize the beta sweep and controls: active latent dimensions, KL and LOF AUROC per cell.

    python ml/sci_rob/VAE/vae_sweep.py --protocol original            # print the table
    python ml/sci_rob/VAE/vae_sweep.py --protocol original --write    # also results/sweep_<protocol>.{csv,md,png}

Selection is on the VALIDATION split (every run saved ``eval_val.json``); the test column is
reported alongside but must not drive the choice. "Active" latent dimensions are those whose
median posterior variance over in-distribution validation windows is below 0.9; a dimension
at the prior (sigma^2 ~ 1, mu ~ 0) carries no information about the input. For the beta=0
controls the variance head receives no gradient, so their active count is meaningless and is
shown as "n/a".
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from vae_paths import HERE, RUNS

RESULTS = HERE / 'results'
ACTIVE_THRESHOLD = .9
NAME = re.compile(r'^(?P<protocol>original|SA|ST|TR)_(?P<variant>.+)_seed(?P<seed>\d+)$')

__all__ = ['active_dimensions', 'summarize', 'markdown_table', 'RESULTS']


def markdown_table(df, float_format='{:.4g}'):
    """A GitHub-style table without the optional `tabulate` dependency."""
    cells = [[('' if pd.isna(v) else float_format.format(v)) if isinstance(v, float) else str(v) for v in row]
             for row in df.itertuples(index=False)]
    head = '| ' + ' | '.join(map(str, df.columns)) + ' |'
    sep = '|' + '|'.join('---' for _ in df.columns) + '|'
    return '\n'.join([head, sep, *('| ' + ' | '.join(r) + ' |' for r in cells)])


def active_dimensions(run, split='val', threshold=ACTIVE_THRESHOLD):
    """(count, boolean mask over the 32 dims, per-dim mean KL) from a run's saved split."""
    run = Path(run)
    with np.load(run / f'latents_{split}.npz') as z:
        if 'logvar' not in z.files:
            return None, None, None
        mu, logvar, labels = z['Z'], z['logvar'], z['labels']
    idx = labels == 0
    var = np.exp(logvar[idx].astype(np.float64))
    active = np.median(var, axis=0) < threshold
    kl_dim = .5 * (mu[idx].astype(np.float64) ** 2 + var - 1 - np.log(var)).mean(0)
    return int(active.sum()), active, kl_dim


def _row(run):
    m = NAME.match(run.name)
    if not m or not (run / 'eval_test.json').exists() or not (run / 'eval_val.json').exists():
        return None
    report = json.loads((run / 'report.json').read_text()); cfg = report['config']
    val = json.loads((run / 'eval_val.json').read_text())['scores']
    test = json.loads((run / 'eval_test.json').read_text())['scores']
    best = report['history'][report['best_epoch']]
    control = not cfg.get('sample', True)
    n_active, _, kl_dim = active_dimensions(run)
    timing = json.loads((run / 'timing.json').read_text()) if (run / 'timing.json').exists() else {}
    return {'variant': m['variant'], 'protocol': m['protocol'], 'seed': int(m['seed']),
            'beta': cfg['beta'], 'sampling': not control, 'lr': cfg['lr'], 'batch': cfg['batch_size'],
            'best epoch': report['best_epoch'] + 1, 'epochs run': len(report['history']),
            'val KL': best['val_kl'], 'val NLL': best['val_nll'],
            'active dims': 'n/a' if control or n_active is None else n_active,
            'KL per active dim': float('nan') if control or kl_dim is None else float(kl_dim[kl_dim > 0].mean()) if (kl_dim > 0).any() else float('nan'),
            'val AUROC (LOF)': val['latent_lof']['auroc'], 'test AUROC (LOF)': test['latent_lof']['auroc'],
            'test recall %': test['latent_lof']['recall'], 'test false alarms %': 100 - test['latent_lof']['specificity'],
            'val AUROC (KL)': val['kl']['auroc'], 'val AUROC (NLL)': val['reconstruction_nll']['auroc'],
            'hours': timing.get('hours', float('nan')), 'run': run.name}


def summarize(protocol='original', root=RUNS):
    rows = [r for r in (_row(p) for p in sorted(Path(root).glob(f'{protocol}_*_seed*')) if not p.name.endswith('_smoke')) if r]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(['sampling', 'beta', 'seed'], ascending=[False, True, True]).reset_index(drop=True)


def figure(df, protocol, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sweep = df[df['sampling'] & (df['seed'] == 0)].sort_values('beta')
    controls = df[~df['sampling'] & (df['seed'] == 0)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(sweep['beta'], sweep['val AUROC (LOF)'], 'o-', color='#2a78d6', label='validation')
    ax.plot(sweep['beta'], sweep['test AUROC (LOF)'], 's--', color='#eb6834', label='test (reported only)')
    for _, r in controls.iterrows():
        ax.axhline(r['val AUROC (LOF)'], color='#4a3aa7', ls=':', lw=1.2)
        ax.text(sweep['beta'].min(), r['val AUROC (LOF)'], f"  {r['variant']} (val)", fontsize=8, va='bottom', color='#4a3aa7')
    ax.set(xscale='log', xlabel='β (KL weight, log scale)', ylabel='latent LOF AUROC', title=f'{protocol}: LOF AUROC against β')
    ax.legend(fontsize=9); ax.grid(alpha=.2)
    ax = axes[1]
    ax.plot(sweep['beta'], pd.to_numeric(sweep['active dims'], errors='coerce'), 'o-', color='#1baf7a')
    ax.set(xscale='log', xlabel='β (KL weight, log scale)', ylabel='active latent dimensions (of 32)',
           ylim=(0, 33), title='Dimensions with median posterior σ² < 0.9 (validation, ID)')
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def write(df, protocol):
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f'sweep_{protocol}.csv', index=False)
    shown = df.drop(columns=['run', 'lr', 'batch'])
    (RESULTS / f'sweep_{protocol}.md').write_text(
        f'# β sweep and controls — protocol `{protocol}`\n\n'
        'Selection metric: validation LOF AUROC. Active dims: median posterior σ² < 0.9 over ID validation windows. '
        'Controls (`sampling = False`, β=0) run the AE objective through this pipeline; `ae_control_aehp` uses the AE\'s '
        'learning rate 0.003, batch 1,024 and 200-epoch budget.\n\n' + markdown_table(shown) + '\n')
    figure(df, protocol, RESULTS / f'sweep_{protocol}.png')
    return RESULTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--protocol', default='original', choices=('original', 'SA', 'ST', 'TR'))
    parser.add_argument('--write', action='store_true', help='write results/sweep_<protocol>.{csv,md,png}')
    a = parser.parse_args(argv)
    df = summarize(a.protocol)
    if df.empty:
        print('no completed runs for', a.protocol); return
    pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
    print(df.drop(columns=['run']).round(4).to_string(index=False))
    if a.write:
        print('wrote', write(df, a.protocol))


if __name__ == '__main__':
    main()
