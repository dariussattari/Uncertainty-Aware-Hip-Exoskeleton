"""Turn a trained joint-moment estimator run into a TensorRT engine on the Jetson, and check it against PyTorch.

    python deployment/moment_engine.py ml/nature/estimator/runs/hip_thigh_aug_s2            # fp32 (TF32 allowed)
    python deployment/moment_engine.py ml/nature/estimator/runs/hip_thigh_aug_s2 --fp16

Run it on the device that will run the engine: a TensorRT engine only loads under the TensorRT version and GPU that
built it. Writes into the run folder:

    moment.onnx          the MomentTCN with its input standardization folded in: raw channels in, N m/kg out
    moment[_fp16].engine the TensorRT engine (trtexec)
    moment.json          what the device must feed it and what comes out (channels, units, window, rate, mirroring)

and fails unless the engine matches the PyTorch model on the GPU (max abs error in N m/kg, ``TOLERANCE``) on windows
drawn from the training standardization.

Input contract (Nature data conventions, ``ml/nature/estimator/moment_data.py``): one leg per batch row, 249 samples
at 200 Hz (1.245 s) of that leg's channels, oldest first; hip angle in deg, its causal derivative
(``features.get_causal_velocity``) in deg/s, thigh IMU accel in m/s^2 and gyro in deg/s in the Nature thigh frame
(x anterior, y up the thigh, z lateral; per-wearer calibration rotates the exo's IMU into it). The left leg is mirrored
into the right leg's frame: negate its thigh accel z, gyro x and gyro y. Output: total hip flexion moment, N m/kg,
OpenSim signs (flexion positive), estimated 50 ms (10 samples) behind the window's last sample.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import tensorrt as trt
import torch
from torch import nn
from torch.nn.utils import parametrize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'ml' / 'common'))
from models.moment_tcn import MomentTCN  # noqa: E402

TRTEXEC = '/usr/src/tensorrt/bin/trtexec'
BATCH = 2                                                    # left and right leg, one engine call per tick
TOLERANCE = {'fp32': 5e-3, 'fp16': 3e-2}                     # N m/kg; the B1 validation RMSE is ~0.19
MIRRORED = ('accel_z', 'gyro_x', 'gyro_y', 'cop_z')            # channel suffixes negated on the left leg (moment_data.py)
HIP_DELAY_SAMPLES = 10
N_CHECK = 64                                                 # check windows, in batches of BATCH
N_TIMING = 500


class Standardized(nn.Module):
    """The estimator with its training standardization in front: (B, C, T) raw channels -> (B, J) N m/kg."""

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer('mean', mean.float().view(1, -1, 1))                # (1, C, 1)
        self.register_buffer('std', std.float().view(1, -1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def get_unit(channel: str) -> str:
    """Unit of a Nature input channel, from its name."""
    units = {'(causal derivative)': 'deg/s', 'angle': 'deg', 'accel': 'm/s^2', 'gyro': 'deg/s', 'force': 'as published', 'cop': 'as published'}
    return next((u for key, u in units.items() if key in channel), 'unknown')


def load_run(run: Path) -> tuple[Standardized, dict]:
    """The run's best.pt as an eval-mode Standardized model with weight norm folded into plain weights, and the
    checkpoint's metadata."""
    ckpt = torch.load(run / 'best.pt', map_location = 'cpu', weights_only = True)
    model = MomentTCN(len(ckpt['channels']), n_outputs = len(ckpt['joints']))
    model.load_state_dict(ckpt['model'])
    for module in model.modules():
        if parametrize.is_parametrized(module, 'weight'):
            parametrize.remove_parametrizations(module, 'weight', leave_parametrized = True)
    meta = {k: ckpt[k] for k in ('channels', 'joints', 'epoch', 'config')}
    return Standardized(model, ckpt['mean'], ckpt['std']).eval(), meta


def export_onnx(model: Standardized, window: int, n_channels: int, path: Path) -> None:
    """Static shape (BATCH, C, window): TensorRT then needs no optimization profile."""
    dummy = torch.zeros(BATCH, n_channels, window)
    torch.onnx.export(model, (dummy,), str(path), input_names = ['window'], output_names = ['moment'], opset_version = 17,
                      dynamo = False)


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool) -> None:
    cmd = [TRTEXEC, f'--onnx={onnx_path}', f'--saveEngine={engine_path}', '--skipInference']
    if fp16:
        cmd.append('--fp16')
    result = subprocess.run(cmd, capture_output = True, text = True)
    if result.returncode != 0:
        raise RuntimeError(f'trtexec failed ({result.returncode}):\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}')


class EngineRunner:
    """A deserialized TensorRT engine with its execution context and fixed GPU I/O buffers (torch tensors)."""

    def __init__(self, engine_path: Path) -> None:
        logger = trt.Logger(trt.Logger.ERROR)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f'{engine_path} did not deserialize: built by another TensorRT version or GPU? Rebuild it here.')
        self.context = self.engine.create_execution_context()
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = torch.from_numpy(np.empty(0, trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            self.buffers[name] = torch.empty(tuple(self.engine.get_tensor_shape(name)), dtype = dtype, device = 'cuda')
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        """(BATCH, C, T) float32 on the GPU -> (BATCH, J) float32 on the GPU."""
        self.buffers['window'].copy_(window)
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        stream.synchronize()
        return self.buffers['moment'].clone()


def get_latency_ms(fn, window: torch.Tensor) -> tuple[float, float]:
    """Median and 99th-percentile wall time of one call (ms), after a warm-up."""
    for _ in range(50):
        fn(window)
    times = []
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        fn(window)
        torch.cuda.synchronize()
        times.append(1e3 * (time.perf_counter() - t0))
    return float(np.median(times)), float(np.percentile(times, 99))


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__.split('\n')[0])
    parser.add_argument('run', type = Path, help = 'run folder holding best.pt (e.g. ml/nature/estimator/runs/hip_thigh_aug_s2)')
    parser.add_argument('--fp16', action = 'store_true', help = 'build an fp16 engine (looser tolerance)')
    parser.add_argument('--seed', type = int, default = 0, help = 'seed for the check windows')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('no CUDA device: run this on the Jetson')
    precision = 'fp16' if args.fp16 else 'fp32'

    model, meta = load_run(args.run)
    window, n_channels = model.model.receptive_field, len(meta['channels'])
    onnx_path = args.run / 'moment.onnx'
    engine_path = args.run / ('moment_fp16.engine' if args.fp16 else 'moment.engine')
    export_onnx(model, window, n_channels, onnx_path)
    build_engine(onnx_path, engine_path, args.fp16)

    # check: raw windows drawn per channel from the training standardization, engine against PyTorch on the GPU
    model = model.cuda()
    runner = EngineRunner(engine_path)
    gen = torch.Generator(device = 'cuda').manual_seed(args.seed)
    errors = []
    with torch.inference_mode():
        for _ in range(N_CHECK // BATCH):
            z = torch.randn(BATCH, n_channels, window, device = 'cuda', generator = gen)
            x = z * model.std + model.mean                                      # (B, C, T) raw units
            ref, out = model(x), runner(x)
            if not torch.isfinite(out).all():
                raise FloatingPointError('non-finite engine output')
            errors.append((out - ref).abs().max().item())
        max_err = max(errors)
        torch_ms = get_latency_ms(model, x)
        engine_ms = get_latency_ms(runner, x)
    if max_err > TOLERANCE[precision]:
        raise AssertionError(f'{precision} engine differs from PyTorch by {max_err:.3g} N m/kg (> {TOLERANCE[precision]})')

    spec = {'source_run': str(args.run), 'engine': engine_path.name, 'onnx': onnx_path.name, 'precision': precision,
            'tensorrt': trt.__version__, 'torch': torch.__version__, 'batch': BATCH, 'batch_rows': ['left leg', 'right leg'],
            'input': {'name': 'window', 'shape': [BATCH, n_channels, window], 'layout': '(leg, channel, time), oldest first',
                      'channels': meta['channels'], 'units': [get_unit(c) for c in meta['channels']],
                      'rate_hz': 200, 'window_samples': window, 'standardization': 'inside the engine (raw units in)',
                      'left_leg_negate': [c for c in meta['channels'] if c.endswith(MIRRORED)], 'imu_frame': 'Nature thigh frame: x anterior, y up the thigh, z lateral'},
            'output': {'name': 'moment', 'joints': meta['joints'], 'units': 'N m/kg, OpenSim signs (flexion positive)',
                       'delay_samples': HIP_DELAY_SAMPLES},
            'check': {'max_abs_error_n_m_per_kg': max_err, 'tolerance': TOLERANCE[precision], 'windows': N_CHECK},
            'latency_ms': {'engine_median': engine_ms[0], 'engine_p99': engine_ms[1], 'torch_median': torch_ms[0],
                           'torch_p99': torch_ms[1]},
            'checkpoint_epoch': meta['epoch'], 'train_config': meta['config']}
    (args.run / 'moment.json').write_text(json.dumps(spec, indent = 2))
    print(f'{engine_path}: max |engine - torch| {max_err:.2e} N m/kg; latency per {BATCH}-leg call: engine '
          f'{engine_ms[0]:.2f} ms (p99 {engine_ms[1]:.2f}), torch {torch_ms[0]:.2f} ms (p99 {torch_ms[1]:.2f})')


if __name__ == '__main__':
    main()
