"""On the Jetson: time every per-window component of the uncertainty-aware controller, with stand-in weights.

    python deployment/latency_budget.py ml/nature/estimator/runs/hip_thigh_aug_s2

Latency does not depend on weight values, so the two encoders run with random weights of the deployed architectures
and sizes (no trained awareness model leaves the Mac): LSTM-VAE encoder (16 channels, 3 s = 600 samples) and TCN-VAE
encoder (23 channels: 16 raw + 7 rhythm features held constant, 600 samples), PyTorch on the GPU, fp32 and fp16
autocast. The awareness layer (awareness.py) runs on arrays the size of the real bundles: LOF over 15,000 reference
latents, 183 trees of depth 6. B1 is the real engine. Each component: 500 timed calls after 50 warm-up, batch 1
(one window per call, as the device runs). Writes data/replay/scirob/latency_budget.json and prints a table.

Not timed: building the windows and the 7 rhythm features from live data (no streaming implementation yet).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'ml' / 'common'))
import awareness as aw  # noqa: E402
import moment_engine as me  # noqa: E402
from models.seq_vae import LSTMVAE, TCNVAE  # noqa: E402

WINDOW = 600                                             # 3 s at 200 Hz
N_WARMUP, N_TIMED = 50, 500


def time_ms(fn) -> dict[str, float]:
    """Median, p99 and max wall time of one call (ms), CUDA synchronized."""
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(1e3 * (time.perf_counter() - t0))
    return {'median': float(np.median(times)), 'p99': float(np.percentile(times, 99)), 'max': float(np.max(times))}


def get_stand_in_trees(rng: np.random.Generator, n_iter: int = 61, depth: int = 6) -> aw.Trees:
    """Complete binary trees of ``depth`` with random splits: the real bundle's 183 trees, at a fixed depth."""
    per_tree = 2 ** (depth + 1) - 1
    n_trees = n_iter * len(aw.REGIMES)
    local = np.arange(per_tree)
    is_leaf = np.tile(local >= 2 ** depth - 1, n_trees)
    offsets = np.repeat(np.arange(n_trees) * per_tree, per_tree)
    return aw.Trees(feature = rng.integers(0, aw.N_FEATURES, n_trees * per_tree), threshold = rng.normal(size = n_trees * per_tree),
                    missing_left = np.zeros(n_trees * per_tree, bool),
                    left = np.tile(np.minimum(2 * local + 1, per_tree - 1), n_trees) + offsets,
                    right = np.tile(np.minimum(2 * local + 2, per_tree - 1), n_trees) + offsets,
                    is_leaf = is_leaf, value = rng.normal(scale = 0.1, size = n_trees * per_tree),
                    roots = (np.arange(n_trees) * per_tree).reshape(n_iter, len(aw.REGIMES)), baseline = np.zeros(3),
                    classes = aw.REGIMES)


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__.split('\n')[0])
    parser.add_argument('run', type = Path)
    args = parser.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    results = {}
    with torch.inference_mode():
        for name, model, channels in (('LSTM-VAE encoder', LSTMVAE(16, WINDOW), 16), ('TCN-VAE encoder', TCNVAE(23, WINDOW), 23)):
            model = model.cuda().eval()
            x = torch.randn(1, channels, WINDOW, device = 'cuda')
            results[f'{name}, fp32'] = time_ms(lambda: model.encode(x))
            with torch.autocast('cuda', dtype = torch.float16):
                results[f'{name}, fp16 autocast'] = time_ms(lambda: model.encode(x))
        runner = me.EngineRunner(args.run / 'moment.engine')
        w = torch.randn(*runner.buffers['window'].shape, device = 'cuda')
        results['B1 engine (two legs)'] = time_ms(lambda: runner(w))
    ref = rng.normal(size = (15_000, aw.LATENT_DIM))
    novelty = aw.NoveltyFit(lof_fit = ref, lof_kdist = np.full(15_000, 2.0), lof_lrd = np.full(15_000, 0.5), lof_cap = 10.0,
                            maha_keep = np.ones(aw.LATENT_DIM, bool), maha_mean = np.zeros(aw.LATENT_DIM),
                            maha_chol = np.eye(aw.LATENT_DIM), active = np.ones(aw.LATENT_DIM, bool))
    z, logvar = rng.normal(size = (1, aw.LATENT_DIM)), np.full((1, aw.LATENT_DIM), -2.0, np.float32)
    trees = get_stand_in_trees(rng)
    x = rng.normal(size = (1, aw.N_FEATURES))
    results['novelty scores (LOF + Mahalanobis + variance)'] = time_ms(lambda: aw.score_novelty(novelty, z, logvar))
    results['regime classifier (183 trees)'] = time_ms(lambda: aw.get_proba(trees, x))
    smoother = aw.CausalMedian()
    counter = iter(range(10 ** 9))
    results['causal median + u + regime'] = time_ms(
        lambda: aw.get_regime(aw.get_uncertainty(smoother.push(next(counter) * 10, np.array([0.7, 0.2, 0.1]))), (0.2, 0.85)))
    fp32 = ['LSTM-VAE encoder, fp32', 'TCN-VAE encoder, fp32', 'novelty scores (LOF + Mahalanobis + variance)',
            'regime classifier (183 trees)', 'causal median + u + regime']
    results['awareness total per window (fp32 encoders, medians summed)'] = {'median': sum(results[k]['median'] for k in fp32)}
    out = Path(__file__).resolve().parents[1] / 'data' / 'replay' / 'scirob' / 'latency_budget.json'
    out.write_text(json.dumps(results, indent = 2))
    print(f'{"component":58s} {"median":>8s} {"p99":>8s} {"max":>8s}  (ms)')
    for k, v in results.items():
        print(f'{k:58s} {v["median"]:8.2f} {v.get("p99", float("nan")):8.2f} {v.get("max", float("nan")):8.2f}')
    print('budget: B1 every 10 ms tick; awareness every 50 ms window (stride 10 samples)')


if __name__ == '__main__':
    main()
