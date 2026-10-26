"""The Nature paper's joint moment estimator, the OOD-safe fallback controller: Model A (the paper's sensors) and the
deployment-sensor models B0 (hip encoder + thigh IMU) and B1 (B0 trained across random thigh-IMU fits).

    python ml/nature/estimator/moment_data.py                                  # once: the arrays
    python ml/nature/estimator/train_moment.py --seed 0                        # Model A
    python ml/nature/estimator/train_moment.py --seed 0 --inputs hip_thigh     # Model B0
    python ml/nature/estimator/train_moment.py --seed 0 --inputs hip_thigh --augment   # Model B1
    python ml/nature/estimator/train_moment.py --smoke [...]                   # wiring check on a few thousand windows
    python ml/nature/estimator/train_moment.py --profile [...]                 # time training steps, top kernels (CLAUDE.md rule 6)

Molinaro, Scherpereel et al. (Nature 2024). The model is ``ml/common/models/moment_tcn.py``'s MomentTCN at Extended Data
Table 1's selected values (80 filters, 5 residual blocks, kernel 5, ReLU, weight normalization, spatial dropout 0.15,
learning rate 5e-5, no L1/L2 regularization). One unilateral model serves both legs: one leg's channels over 249 samples
(the receptive field) -> total hip moment 50 ms behind the window's last sample (and, for Model A, total knee moment at
it), N m/kg. Inputs standardized with training statistics. Trained on the paper's 19 selected tasks
(``moment_data.PAPER_TASKS``) of the 14 training participants; the 4 validation participants' windows of the same tasks
pick the epoch.

Inputs (``--inputs``, ``INPUTS``)
  paper      the paper's 25 channels: hip and knee encoders, thigh, shank and foot IMUs, insole force and COP; hip + knee
  hip_thigh  this project's exoskeleton: hip encoder angle and velocity and the thigh IMU (8 channels); hip only
B1 (``--augment``) moves each training window's thigh IMU to a random fit (``imu_fit.py``): a rotation up to
``max_rotation_deg`` about a random axis and a lever arm of +-``lever_along_m`` along the thigh and +-``lever_across_m``
across it. Deployment rotates the IMU into the Nature frame with a per-person calibration; this covers what is left.

Not stated in the paper, chosen here (each a field of MomentConfig, logged in config.json): Adam, batch 256, MSE loss,
at most 100 epochs with early stopping at patience 10 on the validation MSE, windows every 25 ms (moment_data.STRIDE).
The split is this project's 14 / 4 / 4 participants (the awareness layer's), not the paper's leave-one-participant-out.

Outputs, under ``<out-root>/<run name>/`` (``paper_s0``, ``hip_thigh_s0``, ``hip_thigh_aug_s0``): config.json,
history.json, summary.json (validation RMSE and R^2 per joint, paper tasks; test participants are exported, not scored
here: ``score_moment.py``), best.pt (weights, standardization, config, epoch), last.pt (weights, optimizer, RNG states,
epoch), preds_val.npz and preds_test.npz (every window of the validation / test participants, all 26 tasks: ``rows``
index windows.parquet, ``pred`` and ``target`` (NaN where undefined), each (N, joints), and ``estimate``, the deployed
estimator's (N, 2) hip, knee). Models with a thigh IMU also write preds_test_perturbed.npz: the test windows at fits
drawn once from ``EVAL_FIT_SEED``, the same for every model.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'common'))                 # ml/common: dataset-agnostic models
from models.moment_tcn import MomentTCN, get_receptive_field  # noqa: E402
import imu_fit  # noqa: E402

LOG = logging.getLogger('train_moment')
JOINTS = ('hip', 'knee')
THIGH_ACCEL = [f'thigh_imu_<leg>_accel_{a}' for a in 'xyz']
THIGH_GYRO = [f'thigh_imu_<leg>_gyro_{a}' for a in 'xyz']
# input setting -> channels (names as in meta.json; None = all of them) and the joints predicted
INPUTS = {'paper': {'channels': None, 'joints': ['hip', 'knee']},
          'hip_thigh': {'channels': ['hip_angle_<leg>', 'hip_angle_<leg> (causal derivative)', *THIGH_ACCEL, *THIGH_GYRO],
                        'joints': ['hip']}}
EVAL_FIT_SEED = 2026             # the fixed fits every model's perturbed test export uses


@dataclass
class MomentConfig:
    """Everything that defines a run; written to config.json."""

    inputs: str = 'paper'
    augment: bool = False                        # B1: train across random thigh-IMU fits
    max_rotation_deg: float = 15.0               # residual mounting error after per-person calibration
    lever_along_m: float = 0.10                  # IMU position along the thigh
    lever_across_m: float = 0.05                 # and across it (anterior cuff against lateral strut)
    lr: float = 5e-5                             # Extended Data Table 1
    batch_size: int = 256                        # not stated in the paper
    max_epochs: int = 100                        # not stated in the paper
    patience: int = 10                           # not stated in the paper
    seed: int = 0
    bf16_on_cuda: bool = True                    # CLAUDE.md rule 6; check the profile before relying on it
    max_train_windows: int | None = None         # smoke runs only
    stats_windows: int = 4_000                   # training windows the standardization is fitted on


SMOKE = {'max_epochs': 2, 'patience': 1, 'max_train_windows': 4_096}


@dataclass
class MomentData:
    """Standardized inputs, targets and estimates on the device, and the window rows of each set."""

    inputs: torch.Tensor                         # (n_samples, 2, C) float32, standardized
    targets: torch.Tensor                        # (n_samples, 2, J) float32, NaN where undefined
    estimate: torch.Tensor                       # (n_samples, 2, 2) float32
    starts: torch.Tensor                         # (n_windows,) int64 first sample of each window
    legs: torch.Tensor                           # (n_windows,) int64
    windows: pd.DataFrame
    train: torch.Tensor                          # (n_train,) int64 rows of ``windows``
    val: torch.Tensor                            # (n_val,) int64
    mean: npt.NDArray[np.float32]                # (C,)
    std: npt.NDArray[np.float32]                 # (C,)
    channels: list[str]
    window: int
    rate_hz: float
    data_version: str

    def get_thigh(self) -> tuple[list[int], list[int]] | None:
        """Column indices of the thigh accelerometer and gyro, or None if the setting has no thigh IMU."""
        result = None
        if all(c in self.channels for c in THIGH_ACCEL + THIGH_GYRO):
            result = [self.channels.index(c) for c in THIGH_ACCEL], [self.channels.index(c) for c in THIGH_GYRO]
        return result


# ----------------------------------------------------------------------------- data

def get_set_rows(windows: pd.DataFrame, split: str, paper_only: bool) -> npt.NDArray[np.int64]:
    """Rows of one participant split, optionally only the paper's training tasks."""
    keep = (windows['split'] == split).to_numpy()
    if paper_only:
        keep = keep & windows['paper_train'].to_numpy()
    return np.flatnonzero(keep)


def load_data(data_dir: Path, cfg: MomentConfig, device: torch.device, rng: np.random.Generator) -> MomentData:
    """Read and check the arrays, keep the setting's channels and joints, fit the standardization on training windows,
    move everything to the device."""
    meta = json.loads((data_dir / 'meta.json').read_text())
    window = int(meta['window'])
    if window != get_receptive_field():
        raise ValueError(f'arrays were built for a {window}-sample window; the TCN sees {get_receptive_field()}. Rebuild.')
    windows = pd.read_parquet(data_dir / 'windows.parquet')
    raw = np.load(data_dir / 'inputs.npy')                                    # one sequential read, about 3.8 GB
    expected = (meta['n_samples'], 2, len(meta['channels']))
    if raw.dtype != np.float32 or raw.shape != expected:
        raise ValueError(f'{data_dir}/inputs.npy is {raw.dtype} {raw.shape}; meta.json expects float32 {expected}.')
    setting = INPUTS[cfg.inputs]
    channels = setting['channels'] or meta['channels']
    if setting['channels'] is not None:
        raw = raw[:, :, [meta['channels'].index(c) for c in channels]]       # (n, 2, 25) -> (n, 2, C)
    targets = np.load(data_dir / 'targets.npy')[:, :, [JOINTS.index(j) for j in setting['joints']]]   # (n, 2, 2) -> (n, 2, J)
    train_rows, val_rows = get_set_rows(windows, 'train', True), get_set_rows(windows, 'val', True)
    if cfg.max_train_windows is not None:
        train_rows = np.sort(rng.choice(train_rows, min(cfg.max_train_windows, len(train_rows)), replace = False))
        val_rows = np.sort(rng.choice(val_rows, min(cfg.max_train_windows, len(val_rows)), replace = False))
    starts, legs = windows['start'].to_numpy(np.int64), windows['leg'].to_numpy(np.int64)
    sample = rng.choice(train_rows, min(cfg.stats_windows, len(train_rows)), replace = False)
    x = raw[starts[sample][:, None] + np.arange(window), legs[sample][:, None]]   # (N, W, C)
    mean, std = x.mean(axis = (0, 1)), x.std(axis = (0, 1))
    if not (np.isfinite(mean).all() and (std > 0).all()):
        raise ValueError(f'standardization is not usable (mean {mean}, std {std}): a training window has a missing or constant channel.')
    inputs = torch.from_numpy(np.ascontiguousarray(raw)).to(device)
    del raw
    inputs.sub_(torch.from_numpy(mean).to(device)).div_(torch.from_numpy(std).to(device))
    as_device = lambda a: torch.from_numpy(np.array(a)).to(device)           # noqa: E731  (a writable copy)
    return MomentData(inputs, as_device(targets), as_device(np.load(data_dir / 'estimate.npy')), as_device(starts), as_device(legs),
                      windows, as_device(train_rows), as_device(val_rows), mean.astype(np.float32), std.astype(np.float32),
                      list(channels), window, float(meta['rate_hz']), meta.get('data_version', 'unknown'))


def apply_fits(data: MomentData, x: torch.Tensor, fits: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """x (B, W, C) standardized, with its thigh IMU moved to each window's fit (rotation (B, 3, 3), lever (B, 3))."""
    accel_i, gyro_i = data.get_thigh()
    mean, std = torch.from_numpy(data.mean).to(x.device), torch.from_numpy(data.std).to(x.device)
    raw = x * std + mean                                                       # back to m/s^2 and deg/s
    accel, gyro = imu_fit.perturb_imu(raw[..., accel_i], raw[..., gyro_i], *fits, data.rate_hz)
    raw = raw.clone()
    raw[..., accel_i], raw[..., gyro_i] = accel, gyro
    return (raw - mean) / std


def get_batch(data: MomentData, rows: torch.Tensor, fits: tuple[torch.Tensor, torch.Tensor] | None = None
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Model input (B, C, W), target (B, J) with NaN replaced by 0, and its mask (B, J) for some window rows, the thigh
    IMU at the given fits if any."""
    starts, legs = data.starts[rows], data.legs[rows]
    offsets = torch.arange(data.window, device = rows.device)
    x = data.inputs[starts[:, None] + offsets, legs[:, None]]                  # (B, W, C)
    if fits is not None:
        x = apply_fits(data, x, fits)
    y = data.targets[starts + data.window - 1, legs]                          # (B, J) at the window's last sample
    mask = torch.isfinite(y)
    return x.transpose(1, 2).contiguous(), torch.nan_to_num(y, nan = 0.0), mask   # (B, W, C) -> (B, C, W)


def draw_fits(n: int, cfg: MomentConfig, generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return imu_fit.sample_fits(n, cfg.max_rotation_deg, cfg.lever_along_m, cfg.lever_across_m, generator, device)


# ----------------------------------------------------------------------------- training

def get_masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over the defined targets: (B, J), (B, J), (B, J) -> ()."""
    return ((pred - y) ** 2 * mask).sum() / mask.sum().clamp(min = 1)


def get_autocast(device: torch.device, cfg: MomentConfig) -> torch.autocast:
    return torch.autocast(device.type, dtype = torch.bfloat16, enabled = cfg.bf16_on_cuda and device.type == 'cuda')


@torch.inference_mode()
def predict(model: MomentTCN, data: MomentData, rows: torch.Tensor, cfg: MomentConfig,
            fits: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Predictions (N, J) and targets (N, J, NaN where undefined) of some window rows, in eval mode; ``fits`` (one per
    row) move the thigh IMU."""
    model.eval()
    preds, targets = [], []
    size = 4 * cfg.batch_size
    for i, batch in enumerate(rows.split(size)):
        part = None if fits is None else tuple(f[i * size:i * size + len(batch)] for f in fits)
        x, _, _ = get_batch(data, batch, part)
        with get_autocast(rows.device, cfg):
            preds.append(model(x).float())
        targets.append(data.targets[data.starts[batch] + data.window - 1, data.legs[batch]])
    pred = torch.cat(preds)
    if not torch.isfinite(pred).all():
        raise FloatingPointError('non-finite predictions')
    return pred, torch.cat(targets)


def get_metrics(pred: torch.Tensor, target: torch.Tensor, joints: list[str]) -> dict[str, float]:
    """MSE over defined targets, and RMSE (N m/kg) and R^2 per joint, pooled over windows."""
    mask = torch.isfinite(target)
    result = {'mse': get_masked_mse(pred, torch.nan_to_num(target), mask).item()}
    for j, joint in enumerate(joints):
        p, y = pred[mask[:, j], j], target[mask[:, j], j]
        result[f'{joint}_rmse'] = torch.sqrt(((p - y) ** 2).mean()).item()
        result[f'{joint}_r2'] = (1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()).item()
    return result


def save_atomic(obj: dict, path: Path) -> None:
    """torch.save through a temporary file, so a crash never leaves a truncated checkpoint."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(model: MomentTCN, data: MomentData, cfg: MomentConfig, out: Path, generator: torch.Generator) -> list[dict]:
    """Adam with early stopping on the validation MSE (validation windows at their recorded fit); the best weights are
    left in ``model`` and saved to best.pt."""
    opt = torch.optim.Adam(model.parameters(), lr = cfg.lr)
    device = data.train.device
    joints = INPUTS[cfg.inputs]['joints']
    history, best, best_state, stale = [], float('inf'), None, 0
    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()
        model.train()
        order = data.train[torch.randperm(len(data.train), generator = generator).to(device)]
        total, steps = torch.zeros((), device = device), 0
        for batch in order.split(cfg.batch_size):
            fits = draw_fits(len(batch), cfg, generator, device) if cfg.augment else None
            x, y, mask = get_batch(data, batch, fits)
            with get_autocast(device, cfg):
                pred = model(x)
            loss = get_masked_mse(pred.float(), y, mask)                         # loss in fp32
            opt.zero_grad(set_to_none = True)
            loss.backward()
            opt.step()
            total, steps = total + loss.detach(), steps + 1
        train_mse = (total / max(steps, 1)).item()
        if not math.isfinite(train_mse):
            raise FloatingPointError(f'epoch {epoch}: training loss is not finite ({train_mse}); check the arrays or lower --lr.')
        metrics = get_metrics(*predict(model, data, data.val, cfg), joints)
        record = {'epoch': epoch, 'train_mse': train_mse, **{f'val_{k}': v for k, v in metrics.items()},
                  'seconds': round(time.perf_counter() - t0, 2)}
        history.append(record)
        if record['val_mse'] < best:
            best, stale = record['val_mse'], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            save_atomic({'model': best_state, 'mean': torch.from_numpy(data.mean), 'std': torch.from_numpy(data.std),
                         'channels': data.channels, 'joints': joints, 'epoch': epoch, 'config': asdict(cfg)}, out / 'best.pt')
        else:
            stale += 1
        save_atomic({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch,
                     'torch_rng': torch.get_rng_state(), 'generator': generator.get_state()}, out / 'last.pt')
        (out / 'history.json').write_text(json.dumps(history, indent = 2))
        per_joint = '  '.join(f"{j} {record[f'val_{j}_rmse']:.3f} (R2 {record[f'val_{j}_r2']:.3f})" for j in joints)
        LOG.info('epoch %3d  train %.5f  val %.5f  %s%s  %.1fs', epoch, train_mse, record['val_mse'], per_joint,
                 '  *' if stale == 0 else '', record['seconds'])
        if stale >= cfg.patience:
            LOG.info('early stop: no improvement for %d epochs', cfg.patience)
            break
    model.load_state_dict(best_state)
    return history


# ----------------------------------------------------------------------------- export and profiling

def export_predictions(model: MomentTCN, data: MomentData, cfg: MomentConfig, out: Path) -> dict[str, int]:
    """preds_val.npz and preds_test.npz: every window of the validation / test participants, all tasks; with a thigh
    IMU, preds_test_perturbed.npz too (the test windows at the fixed evaluation fits)."""
    counts = {}
    device = data.train.device
    for split in ('val', 'test'):
        rows = get_set_rows(data.windows, split, False)
        device_rows = torch.from_numpy(rows).to(device)
        estimate = data.estimate[data.starts[device_rows] + data.window - 1, data.legs[device_rows]].cpu().numpy()
        exports = [(f'preds_{split}.npz', None)]
        if split == 'test' and data.get_thigh() is not None:
            exports.append(('preds_test_perturbed.npz', draw_fits(len(rows), cfg, torch.Generator().manual_seed(EVAL_FIT_SEED), device)))
        for name, fits in exports:
            pred, target = predict(model, data, device_rows, cfg, fits)
            np.savez(out / name, rows = rows, pred = pred.cpu().numpy(), target = target.cpu().numpy(), estimate = estimate)
            counts[name] = len(rows)
    return counts


def profile(data: MomentData, cfg: MomentConfig, device: torch.device, steps: int = 20) -> None:
    """Milliseconds per training step after warm-up, then the top kernels of a few steps (CLAUDE.md rule 6)."""
    model = MomentTCN(len(data.channels), n_outputs = len(INPUTS[cfg.inputs]['joints'])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr = cfg.lr)
    rows = data.train[:cfg.batch_size]
    generator = torch.Generator().manual_seed(0)

    def step() -> None:
        fits = draw_fits(len(rows), cfg, generator, device) if cfg.augment else None
        x, y, mask = get_batch(data, rows, fits)
        with get_autocast(device, cfg):
            pred = model(x)
        loss = get_masked_mse(pred.float(), y, mask)
        opt.zero_grad(set_to_none = True)
        loss.backward()
        opt.step()

    sync = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)
    for _ in range(5):
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync()
    ms = 1000 * (time.perf_counter() - t0) / steps
    print(f'{get_run_name(cfg)} {device} bf16={cfg.bf16_on_cuda and device.type == "cuda"} batch {cfg.batch_size}: {ms:.1f} ms per step, '
          f'{ms * len(data.train) / cfg.batch_size / 60_000:.1f} min per epoch ({len(data.train):,} windows)', flush = True)
    if device.type == 'cuda':
        with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                step()
            sync()
        print(prof.key_averages().table(sort_by = 'cuda_time_total', row_limit = 8), flush = True)


def get_git_sha() -> str:
    """The code version: GIT_SHA written by slurm/push.sh on the cluster, else the local checkout's HEAD."""
    stamp = HERE / 'GIT_SHA'
    result = stamp.read_text().strip() if stamp.exists() else 'unknown'
    if not stamp.exists():
        try:
            result = subprocess.run(['git', 'describe', '--always', '--dirty', '--abbrev=7'], cwd = HERE, capture_output = True,
                                    text = True, timeout = 10, check = True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            result = 'unknown'
    return result


def get_run_name(cfg: MomentConfig) -> str:
    """inputs, _aug for B1, seed: paper_s0, hip_thigh_s0, hip_thigh_aug_s0."""
    return f'{cfg.inputs}{"_aug" if cfg.augment else ""}_s{cfg.seed}'


def resolve_device(name: str | None) -> torch.device:
    """cuda, then mps, then cpu, unless one is named."""
    auto = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.device(name or auto)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--inputs', choices = list(INPUTS), default = 'paper')
    parser.add_argument('--augment', action = 'store_true', help = 'B1: train across random thigh-IMU fits')
    parser.add_argument('--data-dir', type = Path,
                        default = Path(os.environ.get('HIPEXO_NATURE_MOMENT_DATA', HERE.parents[2] / 'data' / 'processed_nature' / 'moment')))
    parser.add_argument('--out-root', type = Path, default = HERE / 'runs')
    parser.add_argument('--device', default = None)
    parser.add_argument('--smoke', action = 'store_true')
    parser.add_argument('--profile', action = 'store_true', help = 'time training steps and exit')
    parser.add_argument('--fp32', action = 'store_true', help = 'no bf16 autocast on CUDA')
    parser.add_argument('--force', action = 'store_true', help = 'retrain a run whose summary.json exists')
    a = parser.parse_args(argv)
    cfg = replace(MomentConfig(seed = a.seed, inputs = a.inputs, augment = a.augment, bf16_on_cuda = not a.fp32),
                  **(SMOKE if a.smoke else {}))
    if cfg.augment and INPUTS[cfg.inputs]['channels'] is None:
        parser.error('--augment perturbs the thigh IMU of a deployment setting; use it with --inputs hip_thigh')
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    device = resolve_device(a.device)
    if a.profile:
        profile(load_data(a.data_dir, cfg, device, rng), cfg, device)
        return
    out = a.out_root / f'{get_run_name(cfg)}{"_smoke" if a.smoke else ""}'
    if (out / 'summary.json').exists() and not a.force:
        print(f'already complete: {out} (--force to retrain)')
        return
    out.mkdir(parents = True, exist_ok = True)
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(message)s', datefmt = '%H:%M:%S',
                        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(out / 'train.log', mode = 'w')])
    t0 = time.time()
    data = load_data(a.data_dir, cfg, device, rng)
    joints = INPUTS[cfg.inputs]['joints']
    model = MomentTCN(len(data.channels), n_outputs = len(joints)).to(device)
    provenance = {'git_sha': get_git_sha(), 'data_version': data.data_version, 'device': str(device), 'torch': torch.__version__,
                  'channels': data.channels, 'joints': joints, 'n_train': len(data.train), 'n_val': len(data.val),
                  'params': sum(p.numel() for p in model.parameters())}
    (out / 'config.json').write_text(json.dumps({**asdict(cfg), **provenance}, indent = 2))
    LOG.info('moment TCN %s | %d channels -> %s | %s params | %s | train %d, val %d windows | data %s, code %s', get_run_name(cfg),
             len(data.channels), '+'.join(joints), f'{provenance["params"]:,}', device, len(data.train), len(data.val),
             data.data_version, provenance['git_sha'])
    history = train(model, data, cfg, out, generator)
    best = min(history, key = lambda h: h['val_mse'])
    counts = export_predictions(model, data, cfg, out)
    summary = {**provenance, 'seed': cfg.seed, 'epochs_run': len(history), 'best_epoch': best['epoch'],
               **{k: v for k, v in best.items() if k.startswith('val_')}, 'exported_windows': counts,
               'minutes': round((time.time() - t0) / 60, 2)}
    (out / 'summary.json').write_text(json.dumps(summary, indent = 2))
    LOG.info('done in %.1f min: best epoch %d, %s -> %s', summary['minutes'], best['epoch'],
             ', '.join(f"val {j} RMSE {best[f'val_{j}_rmse']:.3f} (R2 {best[f'val_{j}_r2']:.3f})" for j in joints), out)


if __name__ == '__main__':
    main()
