"""Training for the convolutional autoencoder, following the ankle paper's protocol.

Unlike the ensemble, this is a **single stage** (Table IV, "Training Strategy Comparison"):
train on all subjects with an 80/20 validation split for early stopping, patience 5. No
leave-one-subject-out, so no epoch-count search — it is one training run.

After training, the detector is calibrated in two steps:

1. Encode the in-distribution training windows and fit Local Outlier Factor on those latents.
2. Score the same windows and take the 99.5th percentile of Psi as the ID/OOD threshold.

    from dataset import AutoencoderData
    from train_ae import train_paper_protocol

    data = AutoencoderData()
    model, lof, threshold, report = train_paper_protocol(data)

A note on the learning rate. Table IV specifies ``0.001 * (1024/16)`` = **0.064**. That value
does not train on this data: reconstruction MSE plateaus at 0.959 on standardized inputs, which
is what "predicting the channel mean" looks like — the encoder collapses and 63% of latent
vectors become identical, which in turn makes LOF ill-conditioned. Measured over four epochs on
a fixed subset:

    lr 0.064 -> 0.959 (collapsed)    lr 0.010 -> 0.360
    lr 0.003 -> 0.344 (default)      lr 0.001 -> 0.358

The default here is therefore **0.003**, declared as a deviation in ``paper_spec.py``. Pass
``--lr 0.064`` to reproduce the collapse.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from models.autoencoder import (BASE_LR, BATCH_SIZE, DEFAULT_LR, LOF_NEIGHBORS, PAPER_LR,
                         THRESHOLD_PERCENTILE, LatentLOF, create_autoencoder)
from dataset import AutoencoderData
from training.common import pick_device, resolve_out

__all__ = ["AeConfig", "train_autoencoder", "evaluate_recon", "encode_split",
           "fit_lof_and_threshold", "train_paper_protocol"]


@dataclass
class AeConfig:
    """Hyperparameters. Defaults are Table IV's autoencoder values."""

    lr: float = DEFAULT_LR                    # 0.003; the paper's 0.064 collapses -- see below
    batch_size: int = BATCH_SIZE              # 1024
    max_epochs: int = 200                     # ceiling; patience-5 normally ends far sooner
    patience: int = 5                         # Table IV
    val_fraction: float = 0.2                 # Table IV's 80/20, split by participant
    seed: int = 0
    device: str | None = None
    lof_neighbors: int = LOF_NEIGHBORS
    lof_fit_samples: int = 50_000             # LOF is O(n log n); subsample the fit set
    threshold_percentile: float = THRESHOLD_PERCENTILE
    out_dir: str = "ml/vanilla/runs/autoencoder"


@torch.no_grad()
def evaluate_recon(model, loader, device: torch.device) -> float:
    """Mean reconstruction MSE over a loader, weighted by window count."""
    model.eval()
    total, n = 0.0, 0
    for x in loader:
        x = x.to(device)
        x_hat, _ = model(x)
        total += torch.nn.functional.mse_loss(x_hat, x, reduction="sum").item()
        n += x.numel()
    return total / max(n, 1)


def train_autoencoder(model, train_loader, val_loader, cfg: AeConfig,
                      device: torch.device | None = None, checkpoint=None, verbose=True):
    """Single-stage training with early stopping. Returns ``(model, history)``."""
    device = device or pick_device(cfg.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = torch.nn.MSELoss()

    history: list[dict] = []
    best = {"loss": float("inf"), "epoch": -1, "state": None}

    for epoch in range(cfg.max_epochs):
        model.train()
        t0, total, n = time.perf_counter(), 0.0, 0
        for x in train_loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            x_hat, _ = model(x)
            loss = mse(x_hat, x)
            loss.backward()
            opt.step()
            total += loss.item() * x.shape[0]
            n += x.shape[0]

        rec = {"epoch": epoch, "train_loss": total / max(n, 1),
               "val_loss": evaluate_recon(model, val_loader, device),
               "seconds": round(time.perf_counter() - t0, 1)}
        if rec["val_loss"] < best["loss"] - 1e-9:
            best = {"loss": rec["val_loss"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        rec["best_epoch"] = best["epoch"]
        history.append(rec)

        if verbose:
            print(f"  epoch {epoch:3d}  train {rec['train_loss']:.6f}  "
                  f"val {rec['val_loss']:.6f}  best@{best['epoch']}  ({rec['seconds']}s)")

        if not np.isfinite(rec["train_loss"]):
            print("  train loss is not finite -- the learning rate (0.064) has diverged. "
                  "Lower it with --lr and declare the deviation.")
            break

        if checkpoint is not None:
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": opt.state_dict(), "history": history}, checkpoint)

        if epoch - best["epoch"] >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (no improvement since {best['epoch']})")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, history


@torch.no_grad()
def encode_split(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Latents and per-window reconstruction error for every window a loader yields.

    Returns ``(Z, recon_err)`` with ``Z`` of shape ``(n, latent_size)``.
    """
    model.eval()
    Z, E = [], []
    for x in loader:
        x = x.to(device)
        x_hat, z = model(x)
        Z.append(z.flatten(1).cpu().numpy())
        E.append(model.reconstruction_error(x, x_hat).cpu().numpy())
    return np.concatenate(Z), np.concatenate(E)


def fit_lof_and_threshold(model, data: AutoencoderData, cfg: AeConfig,
                          device: torch.device | None = None):
    """Fit LOF on training latents and set the threshold at the 99.5th percentile of Psi."""
    device = device or pick_device(cfg.device)
    Z, err = encode_split(model, data.loader("train", batch_size=cfg.batch_size), device)
    print(f"encoded {len(Z):,} training windows -> latent dim {Z.shape[1]}")

    lof = LatentLOF(cfg.lof_neighbors, cfg.lof_fit_samples, cfg.seed).fit(Z)
    print(f"LOF fitted on {lof.n_fitted:,} latents (k={cfg.lof_neighbors})")

    psi = lof.score(Z)
    thr = LatentLOF.fit_threshold(psi, cfg.threshold_percentile)
    print(f"Psi on training data: median {np.median(psi):.4f}, "
          f"{cfg.threshold_percentile}th pct = {thr:.4f}")

    # the paper's rejected alternative, kept so the rejection is reproducible
    thr_recon = float(np.percentile(err, cfg.threshold_percentile))
    print(f"reconstruction error (the paper's rejected score): median {np.median(err):.5f}, "
          f"{cfg.threshold_percentile}th pct = {thr_recon:.5f}")
    return lof, thr, {"psi_train_median": float(np.median(psi)), "threshold": thr,
                      "recon_train_median": float(np.median(err)),
                      "threshold_recon": thr_recon, "latent_dim": int(Z.shape[1])}


def train_paper_protocol(data: AutoencoderData, cfg: AeConfig | None = None):
    """Train, then calibrate LOF and the threshold. Returns ``(model, lof, threshold, report)``."""
    import pickle

    cfg = cfg or AeConfig()
    device = pick_device(cfg.device)
    out = resolve_out(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    tr_idx, va_idx, val_subs = data.train_val_indices(cfg.val_fraction, cfg.seed)
    print(f"single-stage training on {device}")
    print(f"  {len(tr_idx):,} train / {len(va_idx):,} val windows "
          f"({100*(1-cfg.val_fraction):.0f}/{100*cfg.val_fraction:.0f} by participant)")
    print(f"  validation participants: {val_subs}")
    print(f"  lr {cfg.lr:.4f}, batch {cfg.batch_size}, patience {cfg.patience}")
    if cfg.lr >= 0.03:
        print(f"  WARNING: lr {cfg.lr} is at or above the value measured to collapse this "
              f"model (paper's {PAPER_LR}). Expect reconstruction MSE to plateau near 1.0.")
    print()

    model = create_autoencoder(data.n_channels, data.window)
    print(f"  {sum(p.numel() for p in model.parameters()):,} params, "
          f"latent {model.latent_filters}x{model.latent_time} = {model.latent_size}\n")

    model, hist = train_autoencoder(
        model,
        data.loader("train", indices=tr_idx, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed),
        data.loader("train", indices=va_idx, batch_size=cfg.batch_size),
        cfg, device, checkpoint=out / "final.pt")

    print()
    lof, thr, cal = fit_lof_and_threshold(model, data, cfg, device)

    torch.save({"model": model.state_dict(), "threshold": thr,
                "config": asdict(cfg), "latent_size": model.latent_size}, out / "final.pt")
    with open(out / "lof.pkl", "wb") as f:
        pickle.dump(lof, f)
    report = {"config": asdict(cfg), "history": hist, "val_subjects": list(val_subs),
              "epochs_run": len(hist), "best_epoch": min(hist, key=lambda r: r["val_loss"])["epoch"],
              **cal}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out/'final.pt'}, {out/'lof.pkl'}, {out/'report.json'}")
    return model, lof, thr, report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train the convolutional autoencoder.")
    p.add_argument("--lr", type=float, default=DEFAULT_LR,
                   help=f"default {DEFAULT_LR}; Table IV says {PAPER_LR} but that collapses")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=5, help="Table IV: 5")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="ml/vanilla/runs/autoencoder")
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs on a small subset -- proves the wiring, not the science")
    a = p.parse_args()

    cfg = AeConfig(lr=a.lr, batch_size=a.batch_size, max_epochs=a.max_epochs,
                   patience=a.patience, device=a.device, seed=a.seed, out_dir=a.out)
    if a.smoke:
        cfg.max_epochs, cfg.patience, cfg.lof_fit_samples = 2, 1, 5_000
        cfg.out_dir = "ml/vanilla/runs/ae_smoke"

    data = AutoencoderData(batch_size=cfg.batch_size)
    print(data, "\n")
    train_paper_protocol(data, cfg)
