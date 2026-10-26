"""The ankle paper's gait-phase ensemble on the Nature data: the baseline the awareness models are judged against.

    python ml/nature/awareness/ensemble_data.py                          # once: gait-phase targets from the insoles
    python ml/nature/awareness/train_ensemble.py --seed 0                # early stopping on the validation participants
    python ml/nature/awareness/train_ensemble.py --seed 0 --protocol loso   # the paper's two-stage protocol
    python ml/nature/awareness/train_ensemble.py --smoke                 # wiring check on a few thousand windows

Tourk et al. (arXiv:2508.21221) found an ensemble of gait-phase regressors the best of their four uncertainty estimators
(Table I: F1 90.3 against 55.7 for the autoencoder's latent LOF). The model is ``ml/common/models/ensemble.py``'s
GaitPhaseEnsemble, their Table IV: seven TCNs (three causal layers of 30 filters, kernel 20, BatchNorm, ReLU), two tanh
heads predicting sin(2 pi phase) of the left and right leg at the window's last sample, the present instant. The score
is the members' disagreement, Psi = (Var(l_1..l_7) + Var(r_1..r_7)) / 2 (their eq. 2), with no label at run time.

Data: the 1 s raw windows of ``ae_data.py`` (16 channels x 200 samples, the paper's window; the TCN's receptive field,
210 samples, covers it), standardized as the autoencoders' inputs are; targets from ``ensemble_data.py``. Training uses
the in-task ID-safe windows of the training participants whose end sample has a phase for at least one leg; a window
with one labelled leg trains that head only (masked MSE), as in the Science Robotics reproduction. Adam 1e-3, batch 1,024,
bf16 autocast on CUDA (train_ae.get_autocast: fp32 cuDNN picks FFT convolutions for kernel 20).

Protocols (``--protocol``)
  val    early stopping with patience 10 on the validation participants' ID-safe windows, the convention of every other
         Nature awareness model (train_ae.py), so the baseline is picked the way its competitors were
  loso   the paper's: each training participant is held out in turn as the early-stopping set, the model is retrained
         on every training participant for the rounded mean of the folds' best epoch counts

Outputs, under ``<out-root>/<run name>/``: config.json, history.json (the final training; ``loso_folds.json`` for the
folds), summary.json (with ``threshold``, the 99.5th percentile of Psi over every training window: the paper's rule),
best.pt, scores.npz (every in-task window of the test participants, all tasks, plus a sample of training windows) and
scores_val.npz (the validation participants'): ``rows`` index the 1 s window table, ``rec`` and ``end`` name each window,
``members`` (N, 7, 2) are the heads' predictions, ``psi`` (N,) the score, ``target`` (N, 2) sin(2 pi phase) with NaN
where undefined, ``is_train`` the training sample.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'common'))                 # ml/common: dataset-agnostic models
from models.ensemble import N_MEMBERS, THRESHOLD_PERCENTILE, GaitPhaseEnsemble  # noqa: E402
import train_ae as ta  # noqa: E402

LOG = logging.getLogger('train_ensemble')
PROTOCOLS = ('val', 'loso')


@dataclass
class EnsembleConfig:
    """Everything that defines a run; written to config.json."""

    protocol: str = 'val'
    lr: float = 1e-3                             # Table IV
    batch_size: int = 1024                       # Table IV
    max_epochs: int = 150                        # ceiling; the paper's final model took 131 epochs on ankle data
    patience: int = 10                           # Table IV: "10 epochs patience during LOSO"
    n_members: int = N_MEMBERS
    seed: int = 0
    inputs: str = 'raw_1s'                       # the paper's window; the model's receptive field is 210 samples
    train_class: str = 'ID-safe'                 # the paper trains on in-distribution movement only
    bf16_on_cuda: bool = True
    max_train_windows: int | None = None         # smoke runs only
    loso_folds: int | None = None                # smoke runs only: the first few folds
    stats_samples: int = 4_000_000
    export_train_windows: int = 20_000


SMOKE = {'max_epochs': 2, 'patience': 1, 'max_train_windows': 4_096, 'loso_folds': 2, 'stats_samples': 400_000,
         'export_train_windows': 512}


# ----------------------------------------------------------------------------- data

def load_targets(data_dir: Path, data: ta.AeData, device: torch.device) -> tuple[torch.Tensor, str]:
    """sin(2 pi phase) per sample and leg, (n_samples, 2) float32 with NaN where undefined, and the target file's version."""
    path = data_dir / 'gait_phase.npy'
    phase = np.load(path)
    if phase.shape != (len(data.signals), 2):
        raise ValueError(f'{path} is {phase.shape}; signals.npy has {len(data.signals)} samples. Rebuild with ensemble_data.py.')
    version = json.loads(path.with_suffix('.json').read_text()).get('version', 'unknown')
    return torch.from_numpy(np.sin(2 * np.pi * phase).astype(np.float32)).to(device), version


def get_targets(target: torch.Tensor, starts: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The target at each window's last sample, (B, 2) with NaN replaced by 0, and its mask (B, 2) of labelled legs."""
    y = target[starts + (window - 1)]
    mask = torch.isfinite(y)
    return torch.nan_to_num(y, nan = 0.0), mask


def get_labelled_rows(rows: torch.Tensor, data: ta.AeData, target: torch.Tensor) -> torch.Tensor:
    """The window rows whose end sample has a phase for at least one leg."""
    _, mask = get_targets(target, data.starts[rows], data.window)
    return rows[mask.any(dim = 1)]


# ----------------------------------------------------------------------------- loss and evaluation

def get_masked_mse(preds: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over members of the masked MSE: (B, M, 2), (B, 2), (B, 2) -> ()."""
    se = (preds - y[:, None]) ** 2 * mask[:, None]                                # (B, M, 2)
    return se.sum() / (mask.sum() * preds.shape[1]).clamp(min = 1)


@torch.inference_mode()
def evaluate(model: GaitPhaseEnsemble, data: ta.AeData, target: torch.Tensor, rows: torch.Tensor, cfg: EnsembleConfig) -> float:
    """Masked ensemble MSE over some window rows (running BatchNorm statistics)."""
    model.eval()
    total, count = torch.zeros((), device = rows.device), torch.zeros((), device = rows.device)
    for batch in rows.split(cfg.batch_size):
        y, mask = get_targets(target, data.starts[batch], data.window)
        with ta.get_autocast(rows.device, cfg):
            preds = model(ta.get_batch(data, batch))
        n = mask.sum() * preds.shape[1]
        total, count = total + get_masked_mse(preds.float(), y, mask) * n, count + n
    return (total / count.clamp(min = 1)).item()


@torch.inference_mode()
def score_rows(model: GaitPhaseEnsemble, data: ta.AeData, target: torch.Tensor, rows: npt.NDArray[np.int64], cfg: EnsembleConfig
               ) -> dict[str, npt.NDArray]:
    """Member predictions (N, M, 2), Psi (N,) and the target (N, 2, NaN where undefined) of some window rows."""
    model.eval()
    device_rows = torch.from_numpy(rows.astype(np.int64)).to(data.train.device)
    members, targets = [], []
    for batch in device_rows.split(cfg.batch_size):
        with ta.get_autocast(device_rows.device, cfg):
            preds = model(ta.get_batch(data, batch))
        members.append(preds.float().cpu().numpy())
        targets.append(target[data.starts[batch] + (data.window - 1)].cpu().numpy())
    preds = np.concatenate(members).astype(np.float32)                          # (N, M, 2)
    if not np.isfinite(preds).all():
        raise FloatingPointError('non-finite member predictions in the export')
    psi = preds.var(axis = 1, ddof = 0).mean(axis = 1)                           # (N, M, 2) -> (N,), the paper's eq. 2
    return {'members': preds, 'psi': psi.astype(np.float32), 'target': np.concatenate(targets).astype(np.float32)}


# ----------------------------------------------------------------------------- training

def train(model: GaitPhaseEnsemble, data: ta.AeData, target: torch.Tensor, train_rows: torch.Tensor, val_rows: torch.Tensor | None,
          cfg: EnsembleConfig, epochs: int, patience: int | None, generator: torch.Generator, out: Path | None) -> list[dict]:
    """Adam over every member at once (they share no weights). With ``val_rows`` and ``patience``, early stopping on the
    masked validation MSE, the best weights left in ``model`` (saved to ``out/best.pt`` when ``out`` is given); without,
    a fixed number of epochs."""
    opt = torch.optim.Adam(model.parameters(), lr = cfg.lr)
    device = train_rows.device
    history, best, best_state, stale = [], float('inf'), None, 0
    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        order = train_rows[torch.randperm(len(train_rows), generator = generator).to(device)]
        n_full = len(order) // cfg.batch_size * cfg.batch_size                 # drop the ragged last batch (BatchNorm)
        total = torch.zeros((), device = device)
        for batch in order[:n_full].split(cfg.batch_size):
            y, mask = get_targets(target, data.starts[batch], data.window)
            with ta.get_autocast(device, cfg):
                preds = model(ta.get_batch(data, batch))
            loss = get_masked_mse(preds.float(), y, mask)                         # loss in fp32
            opt.zero_grad(set_to_none = True)
            loss.backward()
            opt.step()
            total += loss.detach()
        train_mse = (total / max(n_full // cfg.batch_size, 1)).item()
        if not math.isfinite(train_mse):
            raise FloatingPointError(f'epoch {epoch}: training loss is not finite ({train_mse}); lower --lr or check the arrays.')
        record = {'epoch': epoch, 'train_mse': train_mse, 'seconds': round(time.perf_counter() - t0, 2)}
        if val_rows is not None:
            record['val_mse'] = evaluate(model, data, target, val_rows, cfg)
            if record['val_mse'] < best:
                best, stale = record['val_mse'], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if out is not None:
                    ta.save_atomic({'model': best_state, 'mean': torch.from_numpy(data.mean), 'std': torch.from_numpy(data.std),
                                    'epoch': epoch, 'config': asdict(cfg)}, out / 'best.pt')
            else:
                stale += 1
        history.append(record)
        LOG.info('epoch %3d  train %.5f%s%s  %.1fs', epoch, train_mse,
                 f"  val {record['val_mse']:.5f}" if val_rows is not None else '', '  *' if val_rows is not None and stale == 0 else '',
                 record['seconds'])
        if patience is not None and val_rows is not None and stale >= patience:
            LOG.info('early stop: no improvement for %d epochs', patience)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def get_best_epoch(history: list[dict]) -> int:
    """The epoch with the lowest validation MSE."""
    return min(history, key = lambda h: h['val_mse'])['epoch']


def run_loso(data: ta.AeData, target: torch.Tensor, train_rows: torch.Tensor, cfg: EnsembleConfig, out: Path) -> dict:
    """The paper's stage 1: each training participant held out in turn as the early-stopping set; the epoch count for
    stage 2 is the rounded mean of the folds' best epochs (counted from 1). Folds are logged to loso_folds.json."""
    subjects = data.windows['subject'].astype(str).to_numpy()[train_rows.cpu().numpy()]
    people = sorted(set(subjects))[:cfg.loso_folds]
    folds = []
    for i, person in enumerate(people):
        held = torch.from_numpy(subjects == person).to(train_rows.device)
        torch.manual_seed(cfg.seed * 1000 + i)
        model = GaitPhaseEnsemble(data.n_inputs, num_models = cfg.n_members).to(train_rows.device)
        generator = torch.Generator().manual_seed(cfg.seed * 1000 + i)
        LOG.info('fold %d/%d: hold out %s (%d train, %d held-out windows)', i + 1, len(people), person, int((~held).sum()), int(held.sum()))
        history = train(model, data, target, train_rows[~held], train_rows[held], cfg, cfg.max_epochs, cfg.patience, generator, None)
        best = get_best_epoch(history)
        folds.append({'subject': person, 'best_epoch': best, 'val_mse': history[best]['val_mse'], 'epochs_run': len(history)})
        (out / 'loso_folds.json').write_text(json.dumps(folds, indent = 2))
    n_epochs = max(1, int(round(float(np.mean([f['best_epoch'] for f in folds])) + 1)))
    LOG.info('stage 1: best epochs %s -> %d epochs for the final model', [f['best_epoch'] for f in folds], n_epochs)
    return {'n_epochs': n_epochs, 'folds': folds}


# ----------------------------------------------------------------------------- export

def export_scores(model: GaitPhaseEnsemble, data: ta.AeData, target: torch.Tensor, cfg: EnsembleConfig, rng: np.random.Generator,
                  out: Path) -> dict[str, int]:
    """scores.npz: every in-task test-participant window (all tasks) and a sample of training windows; scores_val.npz:
    every in-task validation-participant window (all tasks)."""
    test_rows = ta.get_set_rows(data.windows, 'test', None)
    train_rows = ta.get_set_rows(data.windows, 'train', cfg.train_class)
    train_rows = np.sort(rng.choice(train_rows, min(cfg.export_train_windows, len(train_rows)), replace = False))
    val_rows = ta.get_set_rows(data.windows, 'val', None)
    counts = {}
    for name, rows, is_train in (('scores.npz', np.concatenate([test_rows, train_rows]),
                                  np.r_[np.zeros(len(test_rows), bool), np.ones(len(train_rows), bool)]),
                                 ('scores_val.npz', val_rows, np.zeros(len(val_rows), bool))):
        scored = score_rows(model, data, target, rows, cfg)
        np.savez(out / name, rows = rows, rec = data.windows['rec'].to_numpy()[rows], end = data.windows['end'].to_numpy()[rows],
                 is_train = is_train, **scored)
        counts[name] = len(rows)
    return counts


def get_run_name(cfg: EnsembleConfig) -> str:
    """ensemble, the protocol if not the default, the seed: e.g. ensemble_loso_s0."""
    protocol = '' if cfg.protocol == 'val' else f'_{cfg.protocol}'
    return f'ensemble{protocol}_s{cfg.seed}'


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--protocol', choices = PROTOCOLS, default = 'val')
    parser.add_argument('--data-dir', type = Path,
                        default = Path(os.environ.get('HIPEXO_NATURE_AE_DATA', HERE.parents[2] / 'data' / 'processed_nature' / 'ae')))
    parser.add_argument('--out-root', type = Path, default = HERE / 'runs' / 'ensemble')
    parser.add_argument('--device', default = None)
    parser.add_argument('--smoke', action = 'store_true')
    parser.add_argument('--force', action = 'store_true', help = 'retrain a run whose summary.json exists')
    a = parser.parse_args(argv)
    cfg = replace(EnsembleConfig(seed = a.seed, protocol = a.protocol), **(SMOKE if a.smoke else {}))
    out = a.out_root / f'{get_run_name(cfg)}{"_smoke" if a.smoke else ""}'
    if (out / 'summary.json').exists() and not a.force:
        print(f'already complete: {out} (--force to retrain)')
        return
    out.mkdir(parents = True, exist_ok = True)
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(message)s', datefmt = '%H:%M:%S',
                        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(out / 'train.log', mode = 'w')])
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    device = ta.resolve_device(a.device)
    t0 = time.time()
    # the autoencoders' arrays, standardization and window sets (train_ae.load_data), then the targets on top
    data = ta.load_data(a.data_dir, ta.AeConfig(inputs = cfg.inputs, train_class = cfg.train_class, max_train_windows = cfg.max_train_windows,
                                                stats_samples = cfg.stats_samples), device, rng)
    target, target_version = load_targets(a.data_dir, data, device)
    train_rows, val_rows = get_labelled_rows(data.train, data, target), get_labelled_rows(data.val, data, target)
    model = GaitPhaseEnsemble(data.n_inputs, num_models = cfg.n_members).to(device)
    provenance = {'git_sha': ta.get_git_sha(), 'data_version': data.data_version, 'gait_phase_version': target_version,
                  'device': str(device), 'torch': torch.__version__, 'n_train': len(train_rows), 'n_val': len(val_rows),
                  'n_train_unlabelled': len(data.train) - len(train_rows)}
    (out / 'config.json').write_text(json.dumps({**asdict(cfg), **provenance}, indent = 2))
    LOG.info('gait-phase ensemble seed %d | %s protocol | %d members x %s params | %s | train %d (%d windows without a phase '
             'dropped), val %d | data %s, targets %s, code %s', cfg.seed, cfg.protocol, cfg.n_members,
             f'{sum(p.numel() for p in model.members[0].parameters()):,}', device, len(train_rows), provenance['n_train_unlabelled'],
             len(val_rows), data.data_version, target_version, provenance['git_sha'])
    stage1 = None
    if cfg.protocol == 'loso':
        stage1 = run_loso(data, target, train_rows, cfg, out)
        torch.manual_seed(cfg.seed)
        model = GaitPhaseEnsemble(data.n_inputs, num_models = cfg.n_members).to(device)
        history = train(model, data, target, train_rows, None, cfg, stage1['n_epochs'], None, generator, None)
        ta.save_atomic({'model': model.state_dict(), 'mean': torch.from_numpy(data.mean), 'std': torch.from_numpy(data.std),
                        'epoch': len(history) - 1, 'config': asdict(cfg)}, out / 'best.pt')
    else:
        history = train(model, data, target, train_rows, val_rows, cfg, cfg.max_epochs, cfg.patience, generator, out)
    (out / 'history.json').write_text(json.dumps(history, indent = 2))
    val_mse = evaluate(model, data, target, val_rows, cfg)
    psi_train = score_rows(model, data, target, data.train.cpu().numpy(), cfg)['psi']   # every training window, labelled or not
    threshold = float(np.percentile(psi_train, THRESHOLD_PERCENTILE))
    counts = export_scores(model, data, target, cfg, rng, out)
    summary = {**provenance, 'protocol': cfg.protocol, 'seed': cfg.seed, 'epochs_run': len(history),
               'best_epoch': len(history) - 1 if cfg.protocol == 'loso' else get_best_epoch(history), 'stage1': stage1,
               'val_mse': val_mse, 'threshold': threshold, 'threshold_percentile': THRESHOLD_PERCENTILE,
               'psi_train_median': float(np.median(psi_train)), 'exported_windows': counts, 'minutes': round((time.time() - t0) / 60, 2)}
    (out / 'summary.json').write_text(json.dumps(summary, indent = 2))
    LOG.info('done in %.1f min: %d epochs, val MSE %.5f, Psi median %.5f, threshold (%.1fth pct) %.5f -> %s', summary['minutes'],
             len(history), val_mse, summary['psi_train_median'], THRESHOLD_PERCENTILE, threshold, out)


if __name__ == '__main__':
    main()
