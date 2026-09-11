"""Training for the vanilla gait-phase ensemble, following the ankle paper's protocol.

The paper trains this model in **two stages** (Table IV, "Training Strategy Comparison"):

    Stage 1  Leave-one-subject-out. Rotate each training subject out as an early-stopping
             set (patience 10) and record the epoch at which it stopped improving. Average
             those to get an epoch count.
    Stage 2  Retrain from scratch on *all* training subjects for exactly that many epochs.

Then the detector itself is calibrated: run the trained ensemble over the training data,
collect Psi, and take the 99.5th percentile as the in/out-of-distribution threshold.

Stage 1 matters. The paper reports 13 epochs for its gait-phase model, but that number was
found on ankle data with 9 subjects — it is an output of the procedure, not a constant to
copy. :func:`run_loso` determines the equivalent for this dataset.

    from dataset import EnsembleGaitPhase
    from train import train_paper_protocol

    data = EnsembleGaitPhase()
    model, threshold, report = train_paper_protocol(data)

Note on ensemble diversity: all branches see the same batches in the same order, so they
differ only through their random initialisation. That is what the paper's "multi-branch
convolutional predictor" describes, but it is the sole source of the disagreement that
*becomes* the uncertainty signal — worth remembering if Psi turns out too small to threshold.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from dataset import EnsembleGaitPhase
from models.ensemble import THRESHOLD_PERCENTILE, create_ensemble
from training.common import pick_device, resolve_out

__all__ = [
    "masked_ensemble_mse",
    "train_gait_ensemble",
    "evaluate",
    "run_loso",
    "fit_threshold",
    "train_paper_protocol",
    "TrainConfig",
]


def masked_ensemble_mse(preds: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean masked MSE across every branch, computed in one pass.

    ``preds`` is ``(batch, n_branches, 2)``; ``y`` and ``mask`` are ``(batch, 2)`` and
    broadcast over the branch axis. Dividing by ``mask.sum() * n_branches`` makes this exactly
    the mean of the per-branch masked MSEs, so the number stays on the scale of a single
    branch's loss.

    The mask is what keeps unlabelled legs out of the loss: those carry ``y = 0``, which in
    the tanh range reads as mid-stride rather than as "missing".
    """
    n_branches = preds.shape[1]
    se = (preds - y.unsqueeze(1)) ** 2 * mask.unsqueeze(1)
    denom = mask.sum() * n_branches
    return se.sum() / torch.clamp(denom, min=1.0)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> float:
    """Mean masked ensemble MSE over a loader.

    Weighted by each batch's labelled-entry count, so partial final batches and batches with
    differing label coverage do not skew the average.
    """
    model.eval()
    total, weight = 0.0, 0.0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        w = mask.sum().item()
        if w == 0:
            continue
        total += masked_ensemble_mse(model(x), y, mask).item() * w
        weight += w
    return total / max(weight, 1e-9)


@dataclass
class TrainConfig:
    """Hyperparameters. Defaults are the paper's Table IV values for the ensemble."""

    lr: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 60          # ceiling for LOSO; early stopping normally ends sooner
    patience: int = 10            # Table IV: "10 epochs patience during LOSO"
    seed: int = 0
    device: str | None = None
    threshold_percentile: float = THRESHOLD_PERCENTILE
    n_members: int = 7
    out_dir: str = "ml/vanilla/runs"
    loso_folds: int | None = None  # None = every training subject
    history: list = field(default_factory=list)


def train_gait_ensemble(
    model,
    train_loader,
    val_loader=None,
    epochs: int = 15,
    lr: float = 1e-3,
    patience: int | None = None,
    device: torch.device | str | None = None,
    log_every: int = 1,
    checkpoint: str | Path | None = None,
    verbose: bool = True,
):
    """Train the ensemble. One optimizer covers all branches — they share no weights.

    Returns ``(model, history)`` where ``history`` is a list of per-epoch dicts. When
    ``val_loader`` and ``patience`` are both given, training stops after ``patience`` epochs
    without improvement and the model is restored to its best state.
    """
    device = pick_device(device) if not isinstance(device, torch.device) else device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict] = []
    best = {"loss": float("inf"), "epoch": -1, "state": None}

    for epoch in range(epochs):
        model.train()
        t0, run, weight = time.perf_counter(), 0.0, 0.0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)     # cheaper than zeroing in place
            loss = masked_ensemble_mse(model(x), y, mask)
            loss.backward()
            optimizer.step()

            w = mask.sum().item()
            run += loss.item() * w
            weight += w

        rec = {"epoch": epoch, "train_loss": run / max(weight, 1e-9),
               "seconds": round(time.perf_counter() - t0, 1)}

        if val_loader is not None:
            rec["val_loss"] = evaluate(model, val_loader, device)
            if rec["val_loss"] < best["loss"] - 1e-6:
                best = {"loss": rec["val_loss"], "epoch": epoch,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            rec["best_epoch"] = best["epoch"]

        history.append(rec)
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            msg = f"  epoch {epoch:3d}  train {rec['train_loss']:.5f}"
            if "val_loss" in rec:
                msg += f"  val {rec['val_loss']:.5f}  best@{best['epoch']}"
            print(msg + f"  ({rec['seconds']}s)")

        if checkpoint is not None:
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "history": history}, checkpoint)

        if patience and val_loader is not None and epoch - best["epoch"] >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (no improvement since {best['epoch']})")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, history


def run_loso(data: EnsembleGaitPhase, cfg: TrainConfig | None = None) -> dict:
    """Stage 1 — leave-one-subject-out, to determine the epoch count.

    Each fold holds one training subject out as the early-stopping set. The returned
    ``n_epochs`` is the rounded mean of the per-fold best epochs, which Stage 2 then uses.
    """
    cfg = cfg or TrainConfig()
    device = pick_device(cfg.device)
    folds = list(data.loso_folds())
    if cfg.loso_folds:
        folds = folds[: cfg.loso_folds]

    # Resume: Stage 1 is the long half, so completed folds are persisted after each one.
    # An interrupted run (sleep, crash, Ctrl-C) picks up where it left off instead of
    # repeating hours of work.
    out = resolve_out(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "loso_folds.json"
    results = json.loads(ledger.read_text()) if ledger.exists() else []
    done = {r["subject"] for r in results}
    if done:
        print(f"resuming: {len(done)} fold(s) already complete {sorted(done)}")

    print(f"Stage 1: LOSO over {len(folds)} subjects on {device} "
          f"(max {cfg.max_epochs} epochs, patience {cfg.patience})")

    for i, (subject, train_idx, held_idx) in enumerate(folds, 1):
        if subject in done:
            print(f"\n[{i}/{len(folds)}] {subject} already done, skipping")
            continue
        torch.manual_seed(cfg.seed + i)
        model = create_ensemble(data.n_channels, data.n_targets, cfg.n_members)
        print(f"\n[{i}/{len(folds)}] hold out {subject}  "
              f"({len(train_idx):,} train / {len(held_idx):,} held out)")

        _, hist = train_gait_ensemble(
            model,
            data.loader("train", indices=train_idx, batch_size=cfg.batch_size,
                        shuffle=True, seed=cfg.seed + i),
            data.loader("train", indices=held_idx, batch_size=cfg.batch_size),
            epochs=cfg.max_epochs, lr=cfg.lr, patience=cfg.patience, device=device,
        )
        best = min(hist, key=lambda r: r["val_loss"])
        results.append({"subject": subject, "best_epoch": best["epoch"],
                        "val_loss": best["val_loss"], "epochs_run": len(hist)})
        ledger.write_text(json.dumps(results, indent=2))   # persist before the next fold
        print(f"  -> best epoch {best['epoch']}  val {best['val_loss']:.5f}  [saved {len(results)}/{len(folds)}]")

    epochs = [r["best_epoch"] for r in results]
    n_epochs = max(1, int(round(float(np.mean(epochs)))))
    print(f"\nStage 1 result: best epochs {epochs} -> mean {np.mean(epochs):.1f} "
          f"-> using {n_epochs} for the final model")
    return {"n_epochs": n_epochs, "folds": results}


@torch.no_grad()
def fit_threshold(model, data: EnsembleGaitPhase, cfg: TrainConfig | None = None,
                  device: torch.device | None = None) -> tuple[float, np.ndarray]:
    """Calibrate the detector: the 99.5th percentile of Psi over the training data.

    Psi needs no labels, which is the point — the same computation runs at inference time.
    """
    cfg = cfg or TrainConfig()
    device = device or pick_device(cfg.device)
    model.to(device).eval()

    scores = [model.uncertainty(x.to(device)).cpu()
              for x, _, _ in data.loader("train", batch_size=cfg.batch_size)]
    scores = torch.cat(scores).numpy()
    threshold = float(np.percentile(scores, cfg.threshold_percentile))
    print(f"Psi over {len(scores):,} training windows: "
          f"median {np.median(scores):.5f}, {cfg.threshold_percentile}th pct = {threshold:.5f}")
    return threshold, scores


def train_paper_protocol(data: EnsembleGaitPhase, cfg: TrainConfig | None = None):
    """The full recipe: LOSO -> retrain on all subjects -> calibrate the threshold.

    Returns ``(model, threshold, report)`` and writes both to ``cfg.out_dir``.
    """
    cfg = cfg or TrainConfig()
    device = pick_device(cfg.device)
    out = resolve_out(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stage1 = run_loso(data, cfg)

    print(f"\nStage 2: retraining on all {len(data.train.subjects)} subjects "
          f"for {stage1['n_epochs']} epochs")
    torch.manual_seed(cfg.seed)
    model = create_ensemble(data.n_channels, data.n_targets, cfg.n_members)
    model, hist = train_gait_ensemble(
        model,
        data.loader("train", batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed),
        epochs=stage1["n_epochs"], lr=cfg.lr, device=device,
        checkpoint=out / "final.pt",
    )

    threshold, scores = fit_threshold(model, data, cfg, device)
    val_loss = evaluate(model, data.loader("val", batch_size=cfg.batch_size), device)
    print(f"held-out validation loss: {val_loss:.5f}")

    report = {"config": {k: v for k, v in asdict(cfg).items() if k != "history"},
              "stage1": stage1, "stage2_history": hist,
              "threshold": threshold, "val_loss": val_loss,
              "psi_train": {"median": float(np.median(scores)), "mean": float(scores.mean())}}
    torch.save({"model": model.state_dict(), "threshold": threshold,
                "n_epochs": stage1["n_epochs"]}, out / "final.pt")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out / 'final.pt'} and {out / 'report.json'}")
    return model, threshold, report


if __name__ == "__main__":
    # Smoke run, not the real protocol: 2 folds x 2 epochs on a subset, to confirm the
    # pipeline is sound end to end. Call train_paper_protocol(data) for the full thing.
    data = EnsembleGaitPhase()
    cfg = TrainConfig(max_epochs=2, patience=1, loso_folds=2, out_dir="ml/vanilla/runs/smoke")
    print(f"device: {pick_device(cfg.device)}\n")
    model, threshold, report = train_paper_protocol(data, cfg)
    print(f"\nsmoke run complete: threshold={threshold:.5f}, "
          f"n_epochs chosen={report['stage1']['n_epochs']}")
