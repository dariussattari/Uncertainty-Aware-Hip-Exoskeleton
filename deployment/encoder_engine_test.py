"""On the Jetson: would TensorRT speed up the awareness encoders? Stand-in weights, so no trained model moves.

    python deployment/encoder_engine_test.py

For the LSTM-VAE encoder (16 ch) and the TCN-VAE encoder (23 ch), 600-sample windows, batch 1, random weights of the
deployed architectures: export ``encode`` (mean and log variance) to ONNX, build fp32 and fp16 engines with trtexec,
check each engine against PyTorch fp32 on random windows, and time it exactly as latency_budget.py times PyTorch (500
calls after 50 warm-up, CUDA synchronized). Writes data/replay/scirob/encoder_engines.json and prints a table.
"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import tensorrt as trt
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'ml' / 'common'))
from latency_budget import WINDOW, time_ms  # noqa: E402
from moment_engine import TRTEXEC  # noqa: E402
from models.seq_vae import LSTMVAE, TCNVAE  # noqa: E402

OUT = HERE.parent / 'data' / 'replay' / 'scirob' / 'encoder_engines'


class Encoder(nn.Module):
    """A VAE's encoder alone: (1, C, W) -> mean, log variance (1, 4, 8) each."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vae.encode(x)


class Runner:
    """A deserialized engine with fixed torch I/O buffers on the GPU."""

    def __init__(self, path: Path) -> None:
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.buffers[name] = torch.empty(tuple(self.engine.get_tensor_shape(name)), dtype = torch.float32, device = 'cuda')
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.buffers['window'].copy_(x)
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        stream.synchronize()
        return self.buffers['mu'], self.buffers['logvar']


def main() -> None:
    torch.manual_seed(0)
    OUT.mkdir(parents = True, exist_ok = True)
    results = {}
    for name, vae, channels in (('LSTM-VAE encoder', LSTMVAE(16, WINDOW), 16), ('TCN-VAE encoder', TCNVAE(23, WINDOW), 23)):
        model = Encoder(vae).eval()
        stem = name.split()[0].lower()
        onnx_path = OUT / f'{stem}.onnx'
        torch.onnx.export(model, (torch.zeros(1, channels, WINDOW),), str(onnx_path), input_names = ['window'],
                          output_names = ['mu', 'logvar'], opset_version = 17, dynamo = False)
        model = model.cuda()
        x = torch.randn(1, channels, WINDOW, device = 'cuda')
        with torch.inference_mode():
            results[f'{name}, PyTorch fp32'] = time_ms(lambda: model(x))
            for precision in ('fp32', 'fp16'):
                engine_path = OUT / f'{stem}_{precision}.engine'
                cmd = [TRTEXEC, f'--onnx={onnx_path}', f'--saveEngine={engine_path}', '--skipInference']
                build = subprocess.run(cmd + (['--fp16'] if precision == 'fp16' else []), capture_output = True, text = True)
                if build.returncode != 0:
                    results[f'{name}, TensorRT {precision}'] = {'error': build.stdout[-1500:] + build.stderr[-1500:]}
                    print(f'{name} {precision}: trtexec failed', flush = True)
                    continue
                runner = Runner(engine_path)
                errors = []
                for _ in range(20):
                    xi = torch.randn(1, channels, WINDOW, device = 'cuda')
                    mu_ref, lv_ref = model(xi)
                    mu, lv = runner(xi)
                    errors.append(max((mu - mu_ref).abs().max().item(), (lv - lv_ref).abs().max().item()))
                results[f'{name}, TensorRT {precision}'] = {**time_ms(lambda: runner(x)), 'max_abs_vs_pytorch': max(errors)}
    (OUT.parent / 'encoder_engines.json').write_text(json.dumps(results, indent = 2))
    print(f'{"":42s} {"median":>8s} {"p99":>8s} {"max|diff|":>10s}  (ms)')
    for k, v in results.items():
        if 'error' in v:
            print(f'{k:42s} build failed')
        else:
            print(f'{k:42s} {v["median"]:8.2f} {v["p99"]:8.2f} {v.get("max_abs_vs_pytorch", float("nan")):10.2e}')


if __name__ == '__main__':
    main()
