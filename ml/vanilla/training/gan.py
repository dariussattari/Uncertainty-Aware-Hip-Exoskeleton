"""Adversarial training for the GAN, following the ankle paper's protocol.

Single stage, like the autoencoder, but with **no early stopping**: Table IV specifies 500
fixed epochs, because a GAN has no validation metric that reliably indicates the best moment to
stop. Checkpoints are written every 5 epochs so a run can be resumed or a different epoch
chosen after the fact.

    from dataset import GanData
    from training.gan import train_paper_protocol

    data = GanData()
    model, threshold, report = train_paper_protocol(data)

**What to watch, and why it matters more here than for the other models.** The discriminator is
the detector, and a discriminator that wins is useless: its objective is separating real from
*generated*, and every out-of-distribution window is still real hip data. If ``D(real)``
climbs toward 1 while ``D(fake)`` falls toward 0, ``Psi = 1 - D(x)`` collapses to ~0 for
everything and there is no detector left, however healthy the losses look. The five-to-one
generator-to-discriminator update ratio exists to prevent exactly that.

Three diagnostics are therefore logged every epoch, and a warning is printed when the
discriminator starts running away:

* ``D(real)`` and ``D(fake)`` — should sit in a band around 0.4-0.7, not diverge.
* ``psi_spread`` — the interquartile range of ``Psi`` over a fixed sample of held-out
  in-distribution windows. This is the one that matters: if it collapses toward zero the score
  has stopped discriminating, whichever way the losses are moving.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import GanData
from models.gan import (ADAM_BETAS, BATCH_SIZE, CHECKPOINT_EVERY, EPOCHS, G_STEPS_PER_D,
                        LR_D, LR_DECAY, LR_G, THRESHOLD_PERCENTILE, create_gan)
from training.common import pick_device, resolve_out

__all__ = ["GanConfig", "train_gan", "score_split", "fit_threshold", "train_paper_protocol"]


@dataclass
class GanConfig:
    """Hyperparameters. Defaults are Table IV's GAN values."""

    lr: float = LR_G                          # generator; --lr maps here
    lr_d: float = LR_D                        # discriminator, deliberately 4x smaller
    lr_decay: float = LR_DECAY                # ExponentialLR gamma, both optimisers
    betas: tuple[float, float] = ADAM_BETAS
    batch_size: int = BATCH_SIZE              # 256
    max_epochs: int = EPOCHS                  # 500, fixed -- no early stopping
    g_steps_per_d: int = G_STEPS_PER_D        # 5; see the module docstring
    checkpoint_every: int = CHECKPOINT_EVERY
    val_fraction: float = 0.2
    seed: int = 0
    device: str | None = None
    threshold_percentile: float = THRESHOLD_PERCENTILE
    monitor_windows: int = 4096               # fixed sample for the psi_spread diagnostic
    out_dir: str = "ml/vanilla/runs/gan"


@torch.no_grad()
def _diagnostics(model, x_real: torch.Tensor) -> dict:
    """D(real), D(fake) and the spread of Psi — the three numbers that reveal a dead run."""
    model.eval()
    d_real = torch.sigmoid(model.discriminator(x_real))
    z = model.generator.noise(len(x_real), x_real.device)
    d_fake = torch.sigmoid(model.discriminator(model.generator(z)))
    psi = (1.0 - d_real).cpu().numpy()
    q1, q3 = np.percentile(psi, [25, 75])
    model.train()
    return {"d_real": float(d_real.mean()), "d_fake": float(d_fake.mean()),
            "psi_median": float(np.median(psi)), "psi_spread": float(q3 - q1)}


def train_gan(model, train_loader, monitor_batch: torch.Tensor, cfg: GanConfig,
              device: torch.device | None = None, checkpoint=None, verbose=True,
              log_every: int = 10):
    """Adversarial training. Returns ``(model, history)``.

    One discriminator step then ``g_steps_per_d`` generator steps per batch. Labels are the
    plain 1/0 targets of the standard loss; ``BCEWithLogitsLoss`` keeps the sigmoid inside the
    loss for numerical stability.
    """
    device = device or pick_device(cfg.device)
    model.to(device)
    monitor_batch = monitor_batch.to(device)

    opt_g = torch.optim.Adam(model.generator.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.lr_d, betas=cfg.betas)
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=cfg.lr_decay)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=cfg.lr_decay)
    bce = nn.BCEWithLogitsLoss()

    history: list[dict] = []
    warned = False

    for epoch in range(cfg.max_epochs):
        model.train()
        t0, gl, dl, n = time.perf_counter(), 0.0, 0.0, 0

        for x in train_loader:
            x = x.to(device)
            b = len(x)
            ones = torch.ones(b, device=device)
            zeros = torch.zeros(b, device=device)

            # --- discriminator: real -> 1, generated -> 0 ---
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = model.generator(model.generator.noise(b, device))
            loss_d = bce(model.discriminator(x), ones) + \
                     bce(model.discriminator(fake), zeros)
            loss_d.backward()
            opt_d.step()

            # --- generator: push D toward calling its output real, several times over ---
            for _ in range(cfg.g_steps_per_d):
                opt_g.zero_grad(set_to_none=True)
                gen = model.generator(model.generator.noise(b, device))
                loss_g = bce(model.discriminator(gen), ones)
                loss_g.backward()
                opt_g.step()

            gl += loss_g.item() * b
            dl += loss_d.item() * b
            n += b

        sched_g.step()
        sched_d.step()

        rec = {"epoch": epoch, "loss_g": gl / max(n, 1), "loss_d": dl / max(n, 1),
               "lr_g": sched_g.get_last_lr()[0], "lr_d": sched_d.get_last_lr()[0],
               "seconds": round(time.perf_counter() - t0, 1),
               **_diagnostics(model, monitor_batch)}
        history.append(rec)

        if verbose and (epoch % log_every == 0 or epoch == cfg.max_epochs - 1):
            print(f"  epoch {epoch:4d}  G {rec['loss_g']:.4f}  D {rec['loss_d']:.4f}  "
                  f"D(real) {rec['d_real']:.3f}  D(fake) {rec['d_fake']:.3f}  "
                  f"psi spread {rec['psi_spread']:.4f}  ({rec['seconds']}s)")

        # the failure mode that matters: D has won, so Psi no longer discriminates
        if not warned and rec["d_real"] > 0.95 and rec["d_fake"] < 0.05:
            warned = True
            print(f"  WARNING at epoch {epoch}: D(real)={rec['d_real']:.3f}, "
                  f"D(fake)={rec['d_fake']:.3f}. The discriminator has separated real from "
                  f"generated, so Psi will collapse toward 0 for every real window and the "
                  f"detector is degenerate. Consider raising --g-steps-per-d or lowering --lr-d.")
        if not np.isfinite(rec["loss_g"]) or not np.isfinite(rec["loss_d"]):
            print(f"  loss is not finite at epoch {epoch} -- stopping.")
            break

        if checkpoint is not None and (epoch % cfg.checkpoint_every == 0
                                       or epoch == cfg.max_epochs - 1):
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                        "history": history}, checkpoint)

    return model, history


@torch.no_grad()
def score_split(model, loader, device: torch.device) -> np.ndarray:
    """``Psi = 1 - D(x)`` for every window a loader yields."""
    model.eval()
    return np.concatenate([model.uncertainty(x.to(device)).cpu().numpy() for x in loader])


def fit_threshold(model, data: GanData, cfg: GanConfig, device=None):
    """Threshold at the 99.5th percentile of Psi over the in-distribution training data."""
    device = device or pick_device(cfg.device)
    psi = score_split(model, data.loader("train", batch_size=cfg.batch_size), device)
    thr = float(np.percentile(psi, cfg.threshold_percentile))
    q1, q3 = np.percentile(psi, [25, 75])
    print(f"Psi over {len(psi):,} training windows: median {np.median(psi):.5f}, "
          f"IQR {q3 - q1:.5f}, {cfg.threshold_percentile}th pct = {thr:.5f}")
    if q3 - q1 < 1e-3:
        print("  warning: Psi is nearly constant across training data. The discriminator has "
              "saturated and this threshold will not separate anything.")
    return thr, {"threshold": thr, "psi_train_median": float(np.median(psi)),
                 "psi_train_iqr": float(q3 - q1)}


def train_paper_protocol(data: GanData, cfg: GanConfig | None = None):
    """Train for the fixed schedule, then calibrate the threshold."""
    cfg = cfg or GanConfig()
    device = pick_device(cfg.device)
    out = resolve_out(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    tr_idx, va_idx, val_subs = data.train_val_indices(cfg.val_fraction, cfg.seed)
    print(f"adversarial training on {device}")
    print(f"  {len(tr_idx):,} train / {len(va_idx):,} val windows "
          f"(Step-20 subsample of the AE_GAN set, split by participant)")
    print(f"  validation participants: {val_subs}")
    print(f"  {cfg.max_epochs} fixed epochs, batch {cfg.batch_size}, "
          f"{cfg.g_steps_per_d} G steps per D step")
    print(f"  lr G {cfg.lr:.1e} / D {cfg.lr_d:.1e}, decay {cfg.lr_decay} per epoch")
    print("  no early stopping -- a GAN has no reliable signal for it (Table IV)\n")

    model = create_gan(data.n_channels, data.window)
    g = sum(p.numel() for p in model.generator.parameters())
    d = sum(p.numel() for p in model.discriminator.parameters())
    print(f"  generator {g:,} params, discriminator {d:,}, "
          f"latent {model.latent_shape[0]}x{model.latent_shape[1]}\n")

    # a fixed held-out batch, so the psi_spread diagnostic is comparable across epochs
    monitor = data.splits["train"].tensors(va_idx[:cfg.monitor_windows])

    model, hist = train_gan(model, data.loader("train", indices=tr_idx,
                                               batch_size=cfg.batch_size, shuffle=True,
                                               seed=cfg.seed, drop_last=True),
                            monitor, cfg, device, checkpoint=out / "final.pt")

    print()
    thr, cal = fit_threshold(model, data, cfg, device)
    torch.save({"model": model.state_dict(), "threshold": thr,
                "config": asdict(cfg)}, out / "final.pt")
    report = {"config": asdict(cfg), "history": hist, "val_subjects": list(val_subs),
              "epochs_run": len(hist), "n_train_windows": int(len(tr_idx)), **cal}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nsaved {out/'final.pt'} and {out/'report.json'}")
    return model, thr, report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train the TCN GAN.")
    p.add_argument("--lr", type=float, default=LR_G, help=f"generator; Table IV: {LR_G}")
    p.add_argument("--lr-d", type=float, default=LR_D, help=f"discriminator; Table IV: {LR_D}")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-epochs", type=int, default=EPOCHS, help="Table IV: 500, fixed")
    p.add_argument("--g-steps-per-d", type=int, default=G_STEPS_PER_D, help="Table IV: 5")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="ml/vanilla/runs/gan")
    p.add_argument("--smoke", action="store_true", help="2 epochs -- proves the wiring only")
    a = p.parse_args()

    cfg = GanConfig(lr=a.lr, lr_d=a.lr_d, batch_size=a.batch_size, max_epochs=a.max_epochs,
                    g_steps_per_d=a.g_steps_per_d, device=a.device, seed=a.seed, out_dir=a.out)
    if a.smoke:
        cfg.max_epochs, cfg.checkpoint_every = 2, 1
        cfg.out_dir = "ml/vanilla/runs/gan_smoke"

    data = GanData(batch_size=cfg.batch_size)
    print(data, "\n")
    train_paper_protocol(data, cfg)
