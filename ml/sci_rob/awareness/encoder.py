"""Stage A: the encoder, trained under a holdout, then every window encoded.

    python ml/sci_rob/awareness/encoder.py --holdout DOWN            # train, evaluate, encode all splits
    python ml/sci_rob/awareness/encoder.py --holdout DOWN --smoke    # two epochs, wiring only

The encoder is ``ml/sci_rob/VAE``'s deterministic configuration (``ae_control_aehp``: β=0, no
sampling, the AE's learning rate 0.003, batch 1,024, up to 200 epochs), which scored the best
latent-LOF AUROC of anything tried under the original protocol. ``train_experiment`` writes the
usual run directory: checkpoint, LOF scores on the 32-D latent, reconstruction NLL, per-split
latents for ID+OOD, and the 50,000 training reference latents the LOF was fitted on.

``encode_all`` then writes ``encoded/<split>_<kind>_mu.npy`` and the matching meta parquet for
every window of every split, ID and OOD separately, in the holdout builder's row order. The
regime probes train on ``train_id`` and every later stage indexes these arrays by row.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from aw_paths import RUNS, SPLITS, run_dir
from vae_train import VAEConfig, VARIANTS, train_experiment, data_for
from models.vae import ConvVAE
from training.common import pick_device

SMOKE = dict(max_epochs=2, patience=1, max_train_windows=20_000, lof_fit_samples=5_000)


def config(holdout: str, smoke: bool = False, data_root: str | None = None) -> VAEConfig:
    return VAEConfig(held_out=holdout, data_root=data_root,
                     **{**VARIANTS['ae_control_aehp'], **(SMOKE if smoke else {})})


def train(holdout: str, out: Path, smoke: bool = False, force: bool = False, data_root: str | None = None) -> VAEConfig:
    cfg = config(holdout, smoke, data_root)
    if (out / 'eval_test.json').exists() and not force:
        print(f'encoder already trained: {out}  (use --force to redo)')
        return cfg
    if out.exists():
        shutil.rmtree(out)
    t0 = time.time()
    _, _, report, evaluations = train_experiment(out, cfg)
    (out / 'timing.json').write_text(json.dumps({'hours': round((time.time() - t0) / 3600, 4), 'smoke': smoke,
                                                 'best_epoch': report['best_epoch'], 'epochs_run': len(report['history'])}, indent=2))
    lof = evaluations['test']['scores']['latent_lof']
    print(f'--- encoder done in {(time.time() - t0) / 3600:.2f} h; test latent-LOF AUROC {lof["auroc"]:.4f} '
          f'(recall {lof["recall"]:.1f}%, specificity {lof["specificity"]:.1f}%)', flush=True)
    return cfg


@torch.inference_mode()
def encode_all(out: Path, cfg: VAEConfig | None = None, device=None) -> Path:
    """Latents for every window of every split, ID and OOD separately, in builder row order."""
    out = Path(out)
    cfg = cfg or VAEConfig(**json.loads((out / 'report.json').read_text())['config'])
    dat = data_for(cfg)
    device = pick_device(device or cfg.device)
    model = ConvVAE(dat.n_channels, dat.window).to(device).eval()
    model.load_state_dict(torch.load(out / 'final.pt', map_location=device, weights_only=True)['model'])
    enc = out / 'encoded'
    enc.mkdir(exist_ok=True)
    for split, kind in SPLITS:
        sp = dat.splits[split] if kind == 'id' else dat.ood[split]
        mu = np.concatenate([model.encode(x.to(device))[0].flatten(1).cpu().numpy() for x in dat.loader(sp)])
        meta = sp.meta.copy().reset_index(drop=True)
        meta['kind'] = kind.upper()
        if len(meta) != len(mu):
            raise RuntimeError(f'{split}/{kind}: {len(mu)} latents for {len(meta)} rows')
        np.save(enc / f'{split}_{kind}_mu.npy', mu.astype(np.float32))
        meta.to_parquet(enc / f'{split}_{kind}_meta.parquet', index=False)
        by = meta['mode'].value_counts().to_dict()
        print(f'  encoded {split}/{kind}: {len(mu):,} windows  ' + '  '.join(f'{m}={n:,}' for m, n in sorted(by.items())), flush=True)
    return enc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--holdout', required=True, help='a name from ml/sci_rob/LOTO/splits.HOLDOUTS, e.g. DOWN')
    parser.add_argument('--out-root', default=str(RUNS))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--device', default=None)
    parser.add_argument('--encode-only', action='store_true', help='skip training; only (re)write encoded/')
    parser.add_argument('--stage', default='encoder', help='run directory name under runs/<holdout>/ (ablations: encoder_pelvis)')
    parser.add_argument('--data-root', default=None,
                        help='alternative arrays directory holding AE_GAN/ (the 22-channel pelvis set); default $HIPEXO_LOTO_DATA')
    a = parser.parse_args(argv)
    out = run_dir(a.holdout, a.stage, a.out_root)
    print(f'=== awareness encoder | holdout {a.holdout}{" | SMOKE" if a.smoke else ""}'
          f'{f" | data {a.data_root}" if a.data_root else ""} -> {out}', flush=True)
    cfg = config(a.holdout, a.smoke, a.data_root) if a.encode_only else train(a.holdout, out, a.smoke, a.force, a.data_root)
    if a.device:
        cfg.device = a.device
    encode_all(out, cfg, a.device)


if __name__ == '__main__':
    main()
