"""Time one TCN-VAE training step on the cluster GPU (CLAUDE.md rule 6): the decoder's upsampling as F.interpolate or as a
fixed interpolation matrix, with and without torch.compile; bf16 autocast, batch 1,024, 23 x 600 windows."""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'common'))
from models.seq_vae import TCNVAE  # noqa: E402
from models.vae import loss_terms  # noqa: E402

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
x = torch.randn(1024, 23, 600, device = device)
interp = F.interpolate(torch.eye(8)[None], size = 600, mode = 'linear', align_corners = False)[0].to(device)   # (8, W)


def decode_matmul(self, z):
    u = torch.einsum('bhl,lw->bhw', self.latent_in(z), interp.to(z.dtype))
    return self.output(self.decoder(u))


for name, matmul, compiled in (('interpolate', False, False), ('matrix', True, False), ('matrix + compile', True, True)):
    torch.manual_seed(0)
    model = TCNVAE(23, 600).to(device)
    if matmul:
        model.decode = decode_matmul.__get__(model)
    run = torch.compile(model) if compiled else model
    opt = torch.optim.Adam(model.parameters(), lr = 3e-3)

    def step() -> None:
        with torch.autocast('cuda', dtype = torch.bfloat16):
            x_hat, mu, logvar = run(x)
        nll, kl = loss_terms(x, x_hat.float(), mu.float(), logvar.float())
        opt.zero_grad(set_to_none = True)
        (nll + kl).mean().backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    print(f'{name:18s}: {1000 * (time.perf_counter() - t0) / 10:7.1f} ms per step (warm-up {warm:.0f} s)', flush = True)
