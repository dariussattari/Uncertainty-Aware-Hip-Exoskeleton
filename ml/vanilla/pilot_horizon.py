"""Horizon pilot — pick the forecast horizon before committing to the full protocols.

    python ml/vanilla/pilot_horizon.py                       # 2 branches, 3 epochs, h in {10,20,40}
    python ml/vanilla/pilot_horizon.py --horizons 20 40 80 --epochs 4

The three full two-stage protocols cost roughly twenty-four hours between them, and a horizon
in the wrong regime produces a near-zero uncertainty signal at the end of it. This script
measures the one quantity the screening notebook cannot: **how much trained branches actually
disagree on out-of-distribution input**, per horizon.

Everything in ``05_synthetic_targets.ipynb`` is computed from baselines and target statistics.
Those establish that a two-tap linear filter predicts hip angle 5 ms ahead with 0.99 skill, so
short horizons *must* fail — but they cannot say how much disagreement a real ensemble produces
at a workable horizon, which is what decides whether Experiments 5 and 6 are worth running.

Deliberately not the real protocol: two branches instead of seven, a few epochs instead of
early-stopped LOSO, and a subsample of the training data. The absolute AUROC will be pessimistic
on all counts. What it is measuring is the *ranking* across horizons, which is cheap and robust
where the absolute value is not.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from dataset import SyntheticEnsembleData
from evaluation.common import smooth_by_trial
from evaluation.ensemble import score_windows
from models.ensemble import create_ensemble
from targets import forecast_skill
from training.common import pick_device, resolve_out
from training.ensemble import train_gait_ensemble

try:
    from sklearn.metrics import roc_auc_score
except ImportError:                                     # pragma: no cover
    raise SystemExit("needs scikit-learn")


def pilot_one(target: str, horizon: int, args, device) -> dict:
    data = SyntheticEnsembleData(batch_size=args.batch_size, target=target, horizon=horizon)
    rng = np.random.default_rng(args.seed)

    tr = data.splits["train"]
    idx = np.sort(rng.choice(len(tr), min(args.train_windows, len(tr)), replace=False))

    torch.manual_seed(args.seed)
    model = create_ensemble(data.n_channels, data.n_targets, args.members,
                            activation="linear")

    t0 = time.perf_counter()
    model, hist = train_gait_ensemble(
        model,
        data.loader("train", indices=idx, batch_size=args.batch_size, shuffle=True,
                    seed=args.seed),
        epochs=args.epochs, lr=args.lr, device=device, verbose=False)
    secs = time.perf_counter() - t0

    # score the held-out ID and OOD sets, smoothing within trials exactly as the real evaluator
    bs = args.batch_size
    sub_id = np.sort(rng.choice(len(data.splits["test"]),
                                min(args.score_windows, len(data.splits["test"])),
                                replace=False))
    sub_ood = np.sort(rng.choice(len(data.ood["test"]),
                                 min(args.score_windows, len(data.ood["test"])),
                                 replace=False))
    psi_id = score_windows(model, data.loader("test", indices=sub_id, batch_size=bs),
                           device, labelled=True)
    psi_ood = score_windows(model, data.loader(data.ood["test"], indices=sub_ood,
                                               batch_size=bs), device, labelled=True)

    m_id = data.splits["test"].meta.iloc[sub_id].reset_index(drop=True)
    m_ood = data.ood["test"].meta.iloc[sub_ood].reset_index(drop=True)
    s_id = smooth_by_trial(psi_id, m_id, 10, "median")
    s_ood = smooth_by_trial(psi_ood, m_ood, 10, "median")

    score = np.concatenate([s_id, s_ood])
    label = np.r_[np.zeros(len(s_id), int), np.ones(len(s_ood), int)]
    auroc = float(roc_auc_score(label, score))

    ch = data.channels.index("enc_angle_l")
    Xs = data.splits["test"].tensors(sub_id[:4000]).numpy()
    floor = forecast_skill(Xs, ch, horizon)["skill"] if target != "correlation" else float("nan")

    per_mode = {}
    for mo in ("LG", "RA", "RD", "SA", "SD", "TR", "ST"):
        sel = np.concatenate([(m_id["mode"] == mo).to_numpy(),
                              (m_ood["mode"] == mo).to_numpy()])
        if sel.any():
            per_mode[mo] = float(np.median(score[sel]))

    return {"target": target, "horizon": horizon, "auroc": auroc,
            "final_train_loss": hist[-1]["train_loss"],
            "psi_median_id": float(np.median(s_id)),
            "psi_median_ood": float(np.median(s_ood)),
            "psi_iqr_id": float(np.subtract(*np.percentile(s_id, [75, 25]))),
            "separation": float(np.median(s_ood) / max(np.median(s_id), 1e-12)),
            "baseline_skill": floor, "seconds": round(secs, 1),
            "epochs": len(hist), "per_mode": per_mode}


def main() -> int:
    p = argparse.ArgumentParser(description="Pick the forecast horizon cheaply.")
    p.add_argument("--targets", nargs="+", default=["forecast_all", "forecast_angle"],
                   choices=("correlation", "forecast_angle", "forecast_all"))
    p.add_argument("--horizons", nargs="+", type=int, default=[10, 20, 40],
                   help="samples; multiples of the window stride (10)")
    p.add_argument("--members", type=int, default=2, help="branches (real protocol uses 7)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--train-windows", type=int, default=40_000)
    p.add_argument("--score-windows", type=int, default=15_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="ml/vanilla/runs/pilot_horizon.json")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"Horizon pilot on {device}")
    print(f"  {args.members} branches, {args.epochs} epochs, "
          f"{args.train_windows:,} training windows, {args.score_windows:,} scored per set")
    print("  NOT the real protocol -- the ranking across horizons is what this measures,")
    print("  not the absolute AUROC, which is pessimistic on every count.\n")

    rows = []
    for target in args.targets:
        hs = [0] if target == "correlation" else args.horizons
        for h in hs:
            r = pilot_one(target, h or 40, args, device)
            rows.append(r)
            lab = target if target == "correlation" else f"{target} h={h}"
            print(f"  {lab:28s} AUROC {r['auroc']:.3f}  "
                  f"Psi ID {r['psi_median_id']:.2e} / OOD {r['psi_median_ood']:.2e}  "
                  f"ratio {r['separation']:5.2f}x  "
                  f"baseline skill {r['baseline_skill']:+.2f}  ({r['seconds']}s)")

    out = resolve_out(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=float))

    best = max(rows, key=lambda r: r["auroc"])
    print(f"\nwrote {out}")
    print(f"\nBest: {best['target']}"
          + (f" at h={best['horizon']}" if best["target"] != "correlation" else "")
          + f", AUROC {best['auroc']:.3f}")
    print("\nWhat to look for, in order of importance:")
    print("  1. Does Psi on OOD exceed Psi on ID at all? A ratio near 1.0 means the branches")
    print("     agree everywhere and the horizon is in the dead regime -- do not train it.")
    print("  2. Does AUROC rise with the horizon? If it plateaus, take the smaller horizon:")
    print("     a shorter lead time is more useful to a controller for the same detection.")
    print("  3. Compare against Experiment 1's AUROC of 0.859. These are 2 branches and a few")
    print("     epochs, so anything above ~0.6 here is promising rather than disappointing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
