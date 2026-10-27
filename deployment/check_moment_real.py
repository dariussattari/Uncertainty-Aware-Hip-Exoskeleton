"""On the Jetson: run a built moment engine on real validation windows and check it is the model that was trained.

    python deployment/check_moment_real.py ml/nature/estimator/runs/hip_thigh_aug_s2

Reads ``<run>/check_val.npz`` (``export_check_windows.py``) and ``<run>/moment.engine`` (``moment_engine.py``), and
compares, on the same windows:

    engine vs PyTorch fp32 on this GPU      the conversion is exact on real data      (fails above moment_engine.TOLERANCE)
    engine vs the cluster's saved preds     the training run's own outputs (bf16 autocast), reported
    R^2 against the inverse-dynamics hip moment, for the engine, the cluster preds and the Nature paper's deployed
    estimator; the engine's must match the cluster's (fails above R2_TOLERANCE)

and writes ``<run>/check_real.json``.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import moment_engine as me

R2_TOLERANCE = 0.005


def get_r2(pred: np.ndarray, target: np.ndarray) -> float:
    return float(1 - np.sum((pred - target) ** 2) / np.sum((target - target.mean()) ** 2))


def get_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__.split('\n')[0])
    parser.add_argument('run', type = Path)
    parser.add_argument('--engine', default = 'moment.engine', help = 'engine file in the run folder')
    args = parser.parse_args()

    data = np.load(args.run / 'check_val.npz')
    x = torch.from_numpy(data['x']).cuda()                                    # (N, C, W) raw units
    if len(x) % me.BATCH:
        raise ValueError(f'{len(x)} windows is not a multiple of the engine batch {me.BATCH}')
    model, _ = me.load_run(args.run)
    model = model.cuda()
    runner = me.EngineRunner(args.run / args.engine)
    with torch.inference_mode():
        engine = torch.cat([runner(b) for b in x.split(me.BATCH)])[:, 0].cpu().numpy()     # (N,)
        torch_fp32 = torch.cat([model(b) for b in x.split(256)])[:, 0].cpu().numpy()
    if not np.isfinite(engine).all():
        raise FloatingPointError('non-finite engine output')

    target, cluster, paper = data['target'], data['cluster_pred'], data['paper_estimate']
    precision = 'fp16' if 'fp16' in args.engine else 'fp32'
    vs_torch = float(np.abs(engine - torch_fp32).max())
    has_paper = np.isfinite(paper)                                            # the deployed estimate is not logged everywhere
    r2 = {'engine': get_r2(engine, target), 'torch_fp32': get_r2(torch_fp32, target), 'cluster_bf16': get_r2(cluster, target)}
    paper_subset = {'windows': int(has_paper.sum()), 'r2_engine': get_r2(engine[has_paper], target[has_paper]),
                    'r2_paper_deployed_estimator': get_r2(paper[has_paper], target[has_paper]),
                    'rmse_engine': get_rmse(engine[has_paper], target[has_paper]),
                    'rmse_paper_deployed_estimator': get_rmse(paper[has_paper], target[has_paper])}
    result = {'windows': len(target), 'tasks': int(len(set(data['tasks']))), 'engine': args.engine,
              'max_abs_engine_vs_torch_fp32': vs_torch, 'tolerance_vs_torch': me.TOLERANCE[precision],
              'max_abs_engine_vs_cluster': float(np.abs(engine - cluster).max()),
              'rms_engine_vs_cluster': get_rmse(engine, cluster), 'r2': r2,
              'rmse': {'engine': get_rmse(engine, target), 'cluster_bf16': get_rmse(cluster, target)},
              'where_the_paper_estimate_exists': paper_subset}
    (args.run / 'check_real.json').write_text(json.dumps(result, indent = 2))
    print(json.dumps(result, indent = 2))
    if vs_torch > me.TOLERANCE[precision]:
        raise AssertionError(f'engine differs from PyTorch fp32 by {vs_torch:.3g} N m/kg on real windows')
    if abs(r2['engine'] - r2['cluster_bf16']) > R2_TOLERANCE:
        raise AssertionError(f"engine R^2 {r2['engine']:.4f} vs the training run's {r2['cluster_bf16']:.4f}")
    print('real-data check: ok')


if __name__ == '__main__':
    main()
