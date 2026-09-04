"""Entry point for the vanilla gait-phase ensemble.

    python main.py audit                  # check fidelity against the paper's Table IV
    python main.py smoke                  # 2 folds x 2 epochs, ~5 min, proves the pipeline
    python main.py train                  # the full two-stage protocol (hours -- see below)
    python main.py eval                   # evaluate a trained checkpoint
    python main.py all                    # train then eval

`train` runs the paper's protocol: leave-one-subject-out to determine the epoch count, then a
retrain on all subjects, then threshold calibration. With 12 subjects at roughly 65 s/epoch,
Stage 1 is on the order of 4-7 hours. Everything checkpoints, and `--folds N` trims the fold
count for a cheaper first pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import EnsembleGaitPhase
from eval import evaluate_model, load_checkpoint
from paper_spec import audit
from train import TrainConfig, fit_threshold, pick_device, resolve_out, train_paper_protocol

DEFAULT_RUN = "ml/vanilla/runs/paper"


def _data(args) -> EnsembleGaitPhase:
    data = EnsembleGaitPhase(batch_size=args.batch_size)
    print(data)
    if not data.ood:
        print("  warning: no OOD data found — `eval` needs data/processed/ood/. "
              "Re-run data_exploration/02_build_windows.ipynb.")
    return data


def cmd_audit(args) -> int:
    undeclared = audit()
    return 1 if undeclared else 0


def cmd_train(args) -> int:
    data = _data(args)
    cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        loso_folds=args.folds,
        out_dir=args.out,
    )
    print(f"\ndevice: {pick_device(cfg.device)}")
    print(f"output: {resolve_out(cfg.out_dir)}\n")
    train_paper_protocol(data, cfg)
    return 0


def cmd_eval(args) -> int:
    data = _data(args)
    ckpt = Path(args.checkpoint) if args.checkpoint else resolve_out(args.out) / "final.pt"
    if not ckpt.exists():
        print(f"no checkpoint at {ckpt} — run `python main.py train` first.")
        return 1

    model, threshold = load_checkpoint(ckpt, data, device=pick_device(args.device))
    if not np.isfinite(threshold):
        print("checkpoint has no threshold; recalibrating from training data")
        threshold, _ = fit_threshold(model, data)

    result = evaluate_model(
        model, data, threshold,
        split=args.split,
        device=pick_device(args.device),
        batch_size=args.batch_size,
        filter_kind=args.filter,
        filter_scores=args.filter_scores,
    )

    out = resolve_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{args.split}.json").write_text(json.dumps(
        {"metrics": result.metrics, "per_mode": result.per_mode.to_dict("records"),
         "threshold": result.threshold, "steepness": result.steepness}, indent=2))
    np.savez_compressed(out / f"scores_{args.split}.npz",
                        scores=result.scores, labels=result.labels)
    print(f"\nsaved {out / f'eval_{args.split}.json'}")
    return 0


def cmd_smoke(args) -> int:
    """Cheap end-to-end check: train briefly, then evaluate. Proves the wiring, not the science."""
    data = _data(args)
    cfg = TrainConfig(max_epochs=2, patience=1, loso_folds=2,
                      batch_size=args.batch_size, device=args.device,
                      out_dir="ml/vanilla/runs/smoke")
    print(f"\ndevice: {pick_device(cfg.device)}  (smoke run — 2 folds x 2 epochs)\n")
    model, threshold, _ = train_paper_protocol(data, cfg)
    if data.ood:
        print()
        evaluate_model(model, data, threshold, split="test",
                       device=pick_device(args.device), batch_size=args.batch_size)
    return 0


def cmd_all(args) -> int:
    return cmd_train(args) or cmd_eval(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Vanilla gait-phase ensemble — train and evaluate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--batch-size", type=int, default=1024, help="paper: 1024")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    p.add_argument("--out", default=DEFAULT_RUN, help="run directory")

    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("audit", help="check fidelity against the paper's Table IV").set_defaults(fn=cmd_audit)
    sub.add_parser("smoke", help="quick end-to-end pipeline check").set_defaults(fn=cmd_smoke)

    t = sub.add_parser("train", help="run the full two-stage protocol")
    t.add_argument("--lr", type=float, default=1e-3, help="paper: 0.001")
    t.add_argument("--max-epochs", type=int, default=60, help="ceiling; early stopping usually ends sooner")
    t.add_argument("--patience", type=int, default=10, help="paper: 10 during LOSO")
    t.add_argument("--folds", type=int, default=None, help="limit LOSO folds (default: every subject)")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="evaluate a trained checkpoint")
    e.add_argument("--checkpoint", default=None, help="default: <out>/final.pt")
    e.add_argument("--split", default="test", choices=("val", "test"))
    e.add_argument("--filter", default="median", choices=("median", "mean"),
                   help="paper's eval text says median; Table IV says SMA (mean)")
    e.add_argument("--filter-scores", type=int, default=None,
                   help="scores per filter window (default: 0.5s worth)")
    e.set_defaults(fn=cmd_eval)

    a = sub.add_parser("all", help="train then eval")
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--max-epochs", type=int, default=60)
    a.add_argument("--patience", type=int, default=10)
    a.add_argument("--folds", type=int, default=None)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--checkpoint", default=None)
    a.add_argument("--split", default="test", choices=("val", "test"))
    a.add_argument("--filter", default="median", choices=("median", "mean"))
    a.add_argument("--filter-scores", type=int, default=None)
    a.set_defaults(fn=cmd_all)

    return p


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(getattr(args, "seed", 0))
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
