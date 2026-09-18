r"""Two-stage training, restructured so the twelve LOSO folds run as a SLURM job array.

Experiment 5's protocol is inherited exactly — leave-one-subject-out to choose the epoch count,
then retrain on all twelve participants with that count fixed — but the *execution* is split so
a cluster can exploit it:

    stage 1   twelve independent folds, one per array task, each writing loso/fold_<n>.json
    collect   read the twelve files, average the best epochs, write loso_summary.json
    stage 2   one job: retrain on all twelve participants, calibrate the threshold

``ml/vanilla`` runs the folds sequentially and appends to a single ``loso_folds.json``. That is
correct on one machine and wrong on a cluster: twelve tasks appending to one file race and lose
results. One file per fold makes the array safe, and ``collect`` is the barrier.

Wall clock: Experiment 5's stage 1 took roughly five hours sequentially. Twelve concurrent
tasks reduce that to the slowest single fold — about forty minutes on this data — plus stage 2.

Every stage is idempotent. A fold whose JSON already exists is skipped, so a requeued or
preempted array task costs nothing, which matters on ``gpu_requeue``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import paths  # noqa: F401

import numpy as np
import torch

import data as D
from losses import gaussian_nll_terms, masked_ensemble_beta_nll, masked_ensemble_mse_mu
from prob_models import N_MEMBERS, THRESHOLD_PERCENTILE, create_prob_ensemble
from training.common import pick_device, resolve_out

__all__ = ["ProbConfig", "train_branch_set", "run_fold", "collect_folds",
           "train_final", "fit_threshold"]


@dataclass
class ProbConfig:
    """Experiment 5's hyperparameters, plus the three this experiment introduces.

    Everything inherited is left at Experiment 5's value so the comparison isolates the loss.
    The three new fields all concern Gaussian NLL's instability and are documented in
    ``losses.py``.
    """

    # --- inherited from Experiment 5, do not change without saying so ---
    lr: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 60
    patience: int = 10
    n_members: int = N_MEMBERS
    threshold_percentile: float = THRESHOLD_PERCENTILE
    seed: int = 0

    # --- new here ---
    beta: float = 0.5             # beta-NLL weight; 0 = plain NLL, 1 = MSE-like mean gradients
    warmup_epochs: int = 3        # epochs of masked MSE before switching to beta-NLL
    grad_clip: float = 5.0        # NLL can spike; clip rather than lose a fold to one batch

    device: str | None = None
    out_dir: str = "ml/prob_ensemble/runs/prob_forecast"
    num_workers: int = 0
    history: list = field(default_factory=list)

    @property
    def out(self) -> Path:
        return resolve_out(self.out_dir)


# ---------------------------------------------------------------- one training run

def _epoch_loss(model, loader, device, cfg: ProbConfig, opt=None, use_nll=True):
    """One pass. ``opt=None`` makes it an evaluation pass.

    The reported loss is always the honest masked NLL at ``beta=0``, never the beta-weighted
    training objective, so the number means the same thing in warm-up and afterwards and the
    early-stopping comparison is consistent across the switch.
    """
    train = opt is not None
    model.train(train)
    tot = {"loss": 0.0, "nll": 0.0, "mse": 0.0, "mean_sigma": 0.0, "z_var": 0.0,
           "frac_clamped": 0.0}
    weight = 0.0

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        w = mask.sum().item()
        if w == 0:
            continue
        with torch.set_grad_enabled(train):
            mu, logvar = model(x)
            if use_nll:
                loss = masked_ensemble_beta_nll(mu, logvar, y, mask, beta=cfg.beta)
            else:
                loss = masked_ensemble_mse_mu(mu, y, mask)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        terms = gaussian_nll_terms(mu.detach(), logvar.detach(), y, mask)
        tot["loss"] += float(loss.detach()) * w
        for k in ("nll", "mse", "mean_sigma", "z_var", "frac_clamped"):
            tot[k] += terms[k] * w
        weight += w

    weight = max(weight, 1e-9)
    return {k: v / weight for k, v in tot.items()}


def train_branch_set(model, train_loader, val_loader, cfg: ProbConfig, epochs: int,
                     device, patience: int | None = None, verbose: bool = True,
                     log_every: int = 1, checkpoint: Path | None = None):
    """Train one ensemble. Returns ``(model, history)``.

    Early stopping, when ``val_loader`` and ``patience`` are both given, is on the held-out
    **NLL** rather than the beta-weighted objective. Warm-up epochs are excluded from the
    early-stopping comparison: their loss is a different quantity, and letting an MSE epoch
    win the "best" slot would freeze a model whose variance head is untrained.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: list[dict] = []
    best = {"loss": float("inf"), "epoch": -1, "state": None}

    for epoch in range(epochs):
        warming = epoch < cfg.warmup_epochs
        t0 = time.perf_counter()
        tr = _epoch_loss(model, train_loader, device, cfg, opt=opt, use_nll=not warming)
        rec = {"epoch": epoch, "phase": "warmup" if warming else "nll",
               "train_loss": tr["loss"], "train_nll": tr["nll"], "train_mse": tr["mse"],
               "mean_sigma": tr["mean_sigma"], "z_var": tr["z_var"],
               "frac_clamped": tr["frac_clamped"],
               "seconds": round(time.perf_counter() - t0, 1)}

        if val_loader is not None:
            va = _epoch_loss(model, val_loader, device, cfg, opt=None, use_nll=True)
            rec |= {"val_loss": va["nll"], "val_mse": va["mse"], "val_z_var": va["z_var"]}
            if not warming and rec["val_loss"] < best["loss"] - 1e-6:
                best = {"loss": rec["val_loss"], "epoch": epoch,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
            rec["best_epoch"] = best["epoch"]
        history.append(rec)

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            msg = (f"  epoch {epoch:3d} [{rec['phase']:6s}] train {rec['train_loss']:+.5f} "
                   f"nll {rec['train_nll']:+.5f} mse {rec['train_mse']:.5f} "
                   f"sigma {rec['mean_sigma']:.4f} z_var {rec['z_var']:.3f}")
            if val_loader is not None:
                msg += f" | val nll {rec['val_loss']:+.5f} best@{best['epoch']}"
            print(msg + f"  ({rec['seconds']}s)", flush=True)

        if not np.isfinite(rec["train_loss"]):
            print(f"  loss is not finite at epoch {epoch} -- stopping", flush=True)
            break
        if (patience and val_loader is not None and best["epoch"] >= 0
                and epoch - best["epoch"] >= patience):
            print(f"  early stop at epoch {epoch} (no improvement since {best['epoch']})",
                  flush=True)
            break
        if checkpoint is not None:
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model": model.state_dict(), "history": history},
                       checkpoint)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, history


# ---------------------------------------------------------------- stage 1: one fold

def run_fold(fold_index: int, cfg: ProbConfig | None = None, force: bool = False) -> dict:
    """Train one leave-one-subject-out fold and persist its result.

    Designed to be the body of a SLURM array task. Writes ``loso/fold_<index>_<subject>.json``
    and nothing else, so twelve concurrent tasks never touch the same path.
    """
    cfg = cfg or ProbConfig()
    device = pick_device(cfg.device)
    dat = D.build(batch_size=cfg.batch_size)
    subjects = D.fold_subjects(dat)
    if not 0 <= fold_index < len(subjects):
        raise SystemExit(f"fold index {fold_index} out of range: "
                         f"there are {len(subjects)} folds (0-{len(subjects)-1})")
    subject = subjects[fold_index]

    out = cfg.out / "loso"
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / f"fold_{fold_index:02d}_{subject}.json"
    if ledger.exists() and not force:
        print(f"fold {fold_index} ({subject}) already done -> {ledger.name}; skipping")
        return json.loads(ledger.read_text())

    sp = dat.splits["train"]
    train_idx = sp.indices(exclude=[subject])
    held_idx = sp.indices(subjects=[subject])

    print(f"LOSO fold {fold_index}/{len(subjects)-1}: hold out {subject}")
    print(f"  {len(train_idx):,} train / {len(held_idx):,} held out  on {device}")
    print(f"  beta {cfg.beta}, warmup {cfg.warmup_epochs}, max {cfg.max_epochs} epochs, "
          f"patience {cfg.patience}", flush=True)

    torch.manual_seed(cfg.seed + fold_index + 1)
    model = create_prob_ensemble(dat.n_channels, dat.n_targets, cfg.n_members)
    _, hist = train_branch_set(
        model,
        dat.loader("train", indices=train_idx, batch_size=cfg.batch_size, shuffle=True,
                   seed=cfg.seed + fold_index + 1),
        dat.loader("train", indices=held_idx, batch_size=cfg.batch_size),
        cfg, cfg.max_epochs, device, patience=cfg.patience)

    scored = [r for r in hist if r["phase"] == "nll" and "val_loss" in r]
    if not scored:
        raise SystemExit("no NLL epochs completed -- raise max_epochs above warmup_epochs")
    best = min(scored, key=lambda r: r["val_loss"])
    result = {"fold_index": fold_index, "subject": subject,
              "best_epoch": best["epoch"], "val_nll": best["val_loss"],
              "val_mse": best.get("val_mse"), "val_z_var": best.get("val_z_var"),
              "epochs_run": len(hist), "history": hist,
              "config": {k: v for k, v in asdict(cfg).items() if k != "history"}}
    ledger.write_text(json.dumps(result, indent=2, default=float))
    print(f"\n-> best epoch {best['epoch']}  val NLL {best['val_loss']:+.5f}  "
          f"z_var {best.get('val_z_var', float('nan')):.3f}  [wrote {ledger.name}]", flush=True)
    return result


# ---------------------------------------------------------------- collect

def collect_folds(cfg: ProbConfig | None = None, require_all: bool = True) -> dict:
    """Aggregate the per-fold files into the epoch count stage 2 will use."""
    cfg = cfg or ProbConfig()
    out = cfg.out / "loso"
    files = sorted(out.glob("fold_*.json"))
    if not files:
        raise SystemExit(f"no fold results in {out} -- run stage 1 first")

    dat = D.build(batch_size=cfg.batch_size)
    expected = len(D.fold_subjects(dat))
    folds = [json.loads(f.read_text()) for f in files]
    got = {f["subject"] for f in folds}
    missing = [s for s in D.fold_subjects(dat) if s not in got]
    if missing and require_all:
        raise SystemExit(
            f"only {len(folds)}/{expected} folds present; missing {missing}. "
            f"Re-run those array tasks, or pass --allow-partial to proceed without them "
            f"(the epoch count would then be an average over a subset).")
    if missing:
        print(f"WARNING: proceeding with {len(folds)}/{expected} folds; missing {missing}")

    epochs = [f["best_epoch"] for f in folds]
    n_epochs = max(1, int(round(float(np.mean(epochs)))))
    summary = {"n_epochs": n_epochs, "n_folds": len(folds), "expected_folds": expected,
               "missing": missing,
               "best_epochs": {f["subject"]: f["best_epoch"] for f in folds},
               "val_nll": {f["subject"]: f["val_nll"] for f in folds},
               "val_z_var": {f["subject"]: f.get("val_z_var") for f in folds},
               "mean_best_epoch": float(np.mean(epochs)),
               "sd_best_epoch": float(np.std(epochs))}
    (cfg.out / "loso_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"Stage 1 complete: {len(folds)}/{expected} folds")
    print(f"  best epochs {epochs}")
    print(f"  mean {np.mean(epochs):.1f} (sd {np.std(epochs):.1f}) -> using {n_epochs}")
    zs = [v for v in summary["val_z_var"].values() if v is not None]
    if zs:
        print(f"  held-out z_var across folds: {np.mean(zs):.3f} "
              f"(target 1.000; far from 1 means the variance head is miscalibrated)")
    print(f"  wrote {cfg.out / 'loso_summary.json'}")
    return summary


# ---------------------------------------------------------------- stage 2

@torch.no_grad()
def fit_threshold(model, dat, cfg: ProbConfig, device) -> dict:
    """Calibrate every candidate score at the 99.5th percentile of ID training values.

    All three are calibrated, not just the primary one, so ``evaluate.py`` can compare them
    under the paper's own threshold rule rather than only on threshold-free metrics.
    """
    model.eval()
    acc = {"aleatoric": [], "epistemic": [], "total": []}
    for x, _, _ in dat.loader("train", batch_size=cfg.batch_size):
        u = model.uncertainty_all(x.to(device))
        for k, v in u.items():
            acc[k].append(v.cpu().numpy())
    out = {}
    for k, chunks in acc.items():
        s = np.concatenate(chunks)
        out[k] = {"threshold": float(np.percentile(s, cfg.threshold_percentile)),
                  "median": float(np.median(s)),
                  "iqr": float(np.subtract(*np.percentile(s, [75, 25])))}
        print(f"  {k:10s} median {out[k]['median']:.4e}  IQR {out[k]['iqr']:.4e}  "
              f"{cfg.threshold_percentile}th pct = {out[k]['threshold']:.4e}")
    return out


def train_final(cfg: ProbConfig | None = None, n_epochs: int | None = None) -> dict:
    """Stage 2: retrain on all participants for the collected epoch count, then calibrate."""
    cfg = cfg or ProbConfig()
    device = pick_device(cfg.device)
    cfg.out.mkdir(parents=True, exist_ok=True)

    if n_epochs is None:
        summary_path = cfg.out / "loso_summary.json"
        if not summary_path.exists():
            raise SystemExit(f"no {summary_path} -- run `collect` first, or pass --epochs")
        n_epochs = json.loads(summary_path.read_text())["n_epochs"]

    # A stage-2 run shorter than the warm-up leaves the log-variance head at initialisation
    # and then calibrates thresholds against it -- silently, since nothing errors. Clamp and
    # say so rather than producing a model whose aleatoric term is meaningless.
    if n_epochs <= cfg.warmup_epochs:
        clamped = max(1, n_epochs - 1)
        print(f"WARNING: n_epochs={n_epochs} is not greater than warmup_epochs="
              f"{cfg.warmup_epochs}, so the log-variance head would never train and the "
              f"aleatoric threshold would be calibrated on an untrained head. "
              f"Reducing warmup to {clamped} for this run.")
        cfg = replace(cfg, warmup_epochs=clamped)

    dat = D.build(batch_size=cfg.batch_size)
    print(f"Stage 2: retraining on all {len(D.fold_subjects(dat))} participants "
          f"for {n_epochs} epochs on {device} "
          f"({cfg.warmup_epochs} warm-up, {n_epochs - cfg.warmup_epochs} NLL)")
    print(f"  {dat}", flush=True)

    torch.manual_seed(cfg.seed)
    model = create_prob_ensemble(dat.n_channels, dat.n_targets, cfg.n_members)
    model, hist = train_branch_set(
        model,
        dat.loader("train", batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed),
        None, cfg, n_epochs, device, patience=None,
        checkpoint=cfg.out / "final.pt")

    print("\ncalibrating thresholds on in-distribution training scores:")
    thr = fit_threshold(model, dat, cfg, device)
    val = _epoch_loss(model, dat.loader("val", batch_size=cfg.batch_size), device, cfg)
    print(f"\nheld-out validation: NLL {val['nll']:+.5f}  MSE {val['mse']:.5f}  "
          f"z_var {val['z_var']:.3f} (target 1.000)")
    if not 0.3 < val["z_var"] < 3.0:
        print(f"  WARNING: held-out z_var of {val['z_var']:.3f} is far from 1.0. The variance "
              f"head is badly calibrated, so the aleatoric and total scores are unreliable. "
              f"The epistemic score does not depend on it and remains usable.")

    torch.save({"model": model.state_dict(),
                "thresholds": thr,
                "threshold": thr["epistemic"]["threshold"],   # the primary score
                "n_epochs": n_epochs,
                "config": {k: v for k, v in asdict(cfg).items() if k != "history"}},
               cfg.out / "final.pt")
    report = {"config": {k: v for k, v in asdict(cfg).items() if k != "history"},
              "target": D.TARGET, "horizon": D.HORIZON,
              "target_names": dat.target_names,
              "n_epochs": n_epochs, "stage2_history": hist,
              "thresholds": thr, "val": val}
    (cfg.out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nsaved {cfg.out/'final.pt'} and {cfg.out/'report.json'}")
    return report


def _add_common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--out", default=ProbConfig.out_dir)
    p.add_argument("--batch-size", type=int, default=ProbConfig.batch_size)
    p.add_argument("--lr", type=float, default=ProbConfig.lr)
    p.add_argument("--max-epochs", type=int, default=ProbConfig.max_epochs)
    p.add_argument("--patience", type=int, default=ProbConfig.patience)
    p.add_argument("--beta", type=float, default=ProbConfig.beta)
    p.add_argument("--warmup-epochs", type=int, default=ProbConfig.warmup_epochs)
    p.add_argument("--seed", type=int, default=ProbConfig.seed)
    p.add_argument("--device", default=None)
    return p


def cfg_from(a) -> ProbConfig:
    return ProbConfig(lr=a.lr, batch_size=a.batch_size, max_epochs=a.max_epochs,
                      patience=a.patience, beta=a.beta, warmup_epochs=a.warmup_epochs,
                      seed=a.seed, device=a.device, out_dir=a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Probabilistic ensemble training stages.")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = _add_common(sub.add_parser("fold", help="train one LOSO fold (a SLURM array task)"))
    f.add_argument("--index", type=int, required=True)
    f.add_argument("--force", action="store_true", help="retrain even if the result exists")
    c = _add_common(sub.add_parser("collect", help="aggregate the fold results"))
    c.add_argument("--allow-partial", action="store_true")
    s = _add_common(sub.add_parser("final", help="stage 2 retrain and calibrate"))
    s.add_argument("--epochs", type=int, default=None)
    a = p.parse_args()

    cfg = cfg_from(a)
    print(paths.describe(), "\n")
    if a.cmd == "fold":
        run_fold(a.index, cfg, force=a.force)
    elif a.cmd == "collect":
        collect_folds(cfg, require_all=not a.allow_partial)
    else:
        train_final(cfg, n_epochs=a.epochs)
