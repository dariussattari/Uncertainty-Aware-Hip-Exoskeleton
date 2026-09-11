"""One entry point for every model in the vanilla reproduction.

    python main.py models                              # what is registered
    python main.py audit   --model autoencoder         # fidelity vs the paper's Table IV
    python main.py smoke   --model autoencoder         # quick wiring check
    python main.py train   --model autoencoder
    python main.py eval    --model ensemble
    python main.py figures --model autoencoder
    python main.py all     --model autoencoder         # train, eval, figures

``--model`` is required and explicit; there is no default, because silently training the wrong
architecture is expensive. ``registry.py`` maps the name to its model, trainer, evaluator and
plots, so this file contains no model-specific logic beyond the two evaluator signatures.

Checkpoint layout is common to every model: ``<run>/final.pt`` holds the weights and the
calibrated threshold, ``<run>/report.json`` the training record, ``<run>/eval_<split>.json`` the
metrics, ``<run>/figures/`` the plots. Model-specific extras sit alongside — the autoencoder
also writes ``lof.pkl`` and ``latents_<split>.npz``.

Runtimes differ by orders of magnitude: the autoencoder is minutes, the ensemble's full
two-stage protocol is hours. ``--folds`` trims the ensemble's LOSO for a cheaper first pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import registry
from training.common import pick_device, resolve_out


def _spec_and_parts(args):
    spec = registry.get(args.model)
    if not spec.implemented:
        raise SystemExit(
            f"{args.model!r} is registered but not implemented yet.\n"
            f"Implemented: {', '.join(registry.available())}\n"
            f"  reproduces: {spec.paper_row}\n"
            + "".join(f"  - {n}\n" for n in spec.notes))
    return spec, spec.load()


def _run_dir(args, spec) -> Path:
    return resolve_out(getattr(args, "out", None) or spec.default_run)


def _make_config(parts, **overrides):
    """Build the model's config from whichever CLI flags it actually accepts."""
    cls = parts["config"]
    valid = set(cls.__dataclass_fields__)
    return cls(**{k: v for k, v in overrides.items() if k in valid and v is not None})


def cmd_models(args) -> int:
    print("Registered models — the four architectures of the paper's Table I\n")
    print(registry.describe())
    return 0


def cmd_audit(args) -> int:
    _, parts = _spec_and_parts(args)
    return 1 if parts["audit"]() else 0


def cmd_train(args) -> int:
    spec, parts = _spec_and_parts(args)
    data = parts["data"](batch_size=args.batch_size)
    print(data)
    run = _run_dir(args, spec)
    cfg = _make_config(parts, lr=args.lr, batch_size=args.batch_size,
                       max_epochs=args.max_epochs, patience=args.patience,
                       seed=args.seed, device=args.device, out_dir=str(run),
                       loso_folds=getattr(args, "folds", None))
    print(f"\nmodel  : {spec.name} — {spec.description}")
    print(f"device : {pick_device(args.device)}")
    print(f"output : {run}\n")
    parts["train"](data, cfg)
    return 0


def cmd_eval(args) -> int:
    spec, parts = _spec_and_parts(args)
    run = _run_dir(args, spec)
    if not (run / "final.pt").exists():
        print(f"no checkpoint at {run/'final.pt'} — run `train --model {spec.name}` first.")
        return 1
    data = parts["data"](batch_size=args.batch_size)
    print(data, "\n")
    device = pick_device(args.device)

    # The two evaluators differ in signature by necessity: the ensemble scores from a model
    # object plus a threshold, the autoencoder from a run directory (it also needs its LOF).
    if spec.name == "ensemble":
        from evaluation.ensemble import load_checkpoint
        from training.ensemble import fit_threshold
        model, threshold = load_checkpoint(run / "final.pt", data, device=device)
        if not np.isfinite(threshold):
            print("checkpoint has no threshold; recalibrating from training data")
            threshold, _ = fit_threshold(model, data)
        res = parts["evaluate"](model, data, threshold, split=args.split, device=device,
                                batch_size=args.batch_size, filter_kind=args.filter,
                                filter_scores=args.filter_scores)
        payload = {"metrics": res.metrics, "per_mode": res.per_mode.to_dict("records"),
                   "threshold": res.threshold, "steepness": res.steepness}
        np.savez_compressed(run / f"scores_{args.split}.npz",
                            scores=res.scores, labels=res.labels)
    else:
        bundle, metrics = parts["evaluate"](run, data, args.split, device, args.batch_size)
        bundle.save(run / f"latents_{args.split}")
        payload = {k: v for k, v in metrics.items() if k != "recon"}
        payload["recon_score"] = metrics.get("recon")

    run.mkdir(parents=True, exist_ok=True)
    (run / f"eval_{args.split}.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved {run / f'eval_{args.split}.json'}")
    return 0


def cmd_figures(args) -> int:
    spec, parts = _spec_and_parts(args)
    run = _run_dir(args, spec)
    if not (run / "final.pt").exists():
        print(f"no checkpoint at {run/'final.pt'} — run `train --model {spec.name}` first.")
        return 1
    return parts["figures"](out=str(run), split=args.split, batch_size=args.batch_size,
                            device=args.device, reuse=getattr(args, "reuse", False))


def cmd_smoke(args) -> int:
    """Cheap end-to-end check. Proves the wiring, not the science."""
    spec, parts = _spec_and_parts(args)
    data = parts["data"](batch_size=args.batch_size)
    print(data)
    run = resolve_out(f"ml/vanilla/runs/{spec.name}_smoke")
    cfg = _make_config(parts, max_epochs=2, patience=1, device=args.device,
                       batch_size=args.batch_size, out_dir=str(run),
                       loso_folds=2, lof_fit_samples=5_000)
    print(f"\nsmoke: {spec.name} on {pick_device(args.device)} -> {run}\n")
    parts["train"](data, cfg)
    return 0


def cmd_all(args) -> int:
    return cmd_train(args) or cmd_eval(args) or cmd_figures(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Vanilla reproduction of Tourk et al. — train and evaluate any model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--model", required=True, choices=registry.names(),
                        help="which architecture (see `main.py models`)")
        sp.add_argument("--out", default=None, help="run directory (default: per-model)")
        sp.add_argument("--batch-size", type=int, default=1024)
        sp.add_argument("--device", default=None, help="cuda / mps / cpu")
        return sp

    def add_train_flags(sp):
        sp.add_argument("--lr", type=float, default=None, help="default: the model's own")
        sp.add_argument("--max-epochs", type=int, default=None)
        sp.add_argument("--patience", type=int, default=None)
        sp.add_argument("--folds", type=int, default=None, help="ensemble only: limit LOSO folds")
        sp.add_argument("--seed", type=int, default=0)
        return sp

    def add_eval_flags(sp):
        sp.add_argument("--split", default="test", choices=("val", "test"))
        sp.add_argument("--filter", default="median", choices=("median", "mean"),
                        help="paper's eval text says median; Table IV says SMA")
        sp.add_argument("--filter-scores", type=int, default=None)
        return sp

    sub.add_parser("models", help="list the registered models").set_defaults(fn=cmd_models)
    add_common(sub.add_parser("audit", help="fidelity check against Table IV")).set_defaults(fn=cmd_audit)
    add_common(sub.add_parser("smoke", help="quick wiring check")).set_defaults(fn=cmd_smoke)
    add_train_flags(add_common(sub.add_parser("train", help="run the training protocol"))).set_defaults(fn=cmd_train)
    add_eval_flags(add_common(sub.add_parser("eval", help="evaluate a checkpoint"))).set_defaults(fn=cmd_eval)

    f = add_eval_flags(add_common(sub.add_parser("figures", help="regenerate figures")))
    f.add_argument("--reuse", action="store_true", help="redraw without re-evaluating")
    f.set_defaults(fn=cmd_figures)

    a = add_eval_flags(add_train_flags(add_common(
        sub.add_parser("all", help="train, then eval, then figures"))))
    a.add_argument("--reuse", action="store_true")
    a.set_defaults(fn=cmd_all)
    return p


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(getattr(args, "seed", 0))
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
