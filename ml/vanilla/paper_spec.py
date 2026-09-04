"""Table IV of the ankle paper, encoded — and every deviation from it, declared.

Fidelity to Tourk et al. (arXiv:2508.21221) is something this project has to be able to
*defend*, not just assert. So the paper's specification lives here as data, the live code is
compared against it, and any difference must be listed in :data:`DEVIATIONS` with a category
and a reason. An undeclared mismatch is a failure, not a footnote.

Run it directly to audit the current code:

    python ml/vanilla/paper_spec.py

Categories
----------
``forced``
    The hip dataset makes the paper's value impossible. Not closable.
``unspecified``
    The paper does not state a value, so one had to be inferred. Not closable without the
    original authors' code.
``choice``
    Ours, deliberately. These are the ones worth arguing about — each needs a reason that
    survives a committee asking "why didn't you just do what the paper did?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import model as m

__all__ = ["PAPER", "DEVIATIONS", "observed", "audit"]


# --- Table IV, "COMPLETE SPECIFICATIONS FOR ALL ANOMALY DETECTION MODELS" ----------
# Common Data Preparation + Ensemble Models + Ensemble: Gait Phase Model
PAPER: dict[str, object] = {
    "window_samples": 175,          # "175 samples (1 second at 175 Hz)"
    "sample_rate_hz": 175,
    "n_channels": 16,               # "16 (bilateral sensor data)"
    "stride": 10,                   # "Step: 10"; text: "using every tenth window"
    "n_members": 7,                 # "7 branches"
    "n_layers": 3,                  # "3 layers/branch"
    "n_filters": 30,                # "30 filters/layer"
    "kernel_size": 20,              # "kernel size 20"
    "norm": "BatchNorm1d",          # "ReLU, BatchNorm1d after each convolution"
    "activation": "ReLU",
    "output_activation": "tanh",    # "Tanh activation (range [-1,1])"
    "batch_size": 1024,             # "Batch: 1024"
    "lr": 0.001,                    # "LR: 0.001"
    "optimizer": "Adam",
    "loss": "MSE",
    "patience": 10,                 # "10 epochs patience during LOSO"
    "threshold_percentile": 99.5,   # "99.5th percentile threshold"
    "train_subjects": 9,            # n=9 training set
    "val_subjects": 3,
    "test_subjects": 3,
    "dilations": None,              # not stated anywhere in the paper
}


@dataclass(frozen=True)
class Deviation:
    key: str
    category: str          # forced | unspecified | choice
    reason: str


DEVIATIONS: tuple[Deviation, ...] = (
    Deviation(
        "sample_rate_hz", "forced",
        "The Molinaro dataset is recorded at 200 Hz; the ankle data was 175 Hz.",
    ),
    Deviation(
        "window_samples", "forced",
        "200 samples at 200 Hz preserves the paper's 1-second window, which it describes as "
        "'175 samples, or approximately 1 s'. Keeping 175 samples instead would shorten the "
        "window to 0.875 s. Duration is the physiologically meaningful quantity for gait, so "
        "duration is what is held constant.",
    ),
    Deviation(
        "dilations", "unspecified",
        "Absent from Table IV in both the PDF and HTML renderings; the paper defers to Shetty "
        "et al. (RA-L 2025), which is paywalled. (1, 2, 8) gives a receptive field of 210 "
        "samples, covering the full 200-sample window. Conventional doubling (1, 2, 4) would "
        "reach only 134, leaving the first third of every window unreachable by the model.",
    ),
    Deviation(
        "train_subjects", "choice",
        "12 rather than 9. The Phase 3-4 custom-exo pool has 20 subjects; the paper's n=9 "
        "reflects its recruitment, not a methodological requirement, and LOSO is identical at "
        "any n. Costs ~30% more Stage 1 compute and means the resulting epoch count will not "
        "match their 13.",
    ),
    Deviation(
        "val_subjects", "choice",
        "4 rather than 3, for the same reason as train_subjects.",
    ),
    Deviation(
        "test_subjects", "choice",
        "4 rather than 3. A larger held-out set also makes the per-mode OOD breakdown less "
        "noisy, which matters because some OOD modes come from only one or two subjects.",
    ),
)

_BY_KEY = {d.key: d for d in DEVIATIONS}


def observed(processed: Path | str | None = None) -> dict[str, object]:
    """Read the values the code and data are actually using right now."""
    root = Path(processed) if processed else Path(m.__file__).resolve().parents[2] / "data" / "processed"
    cfg = json.loads((root / "config.json").read_text())
    return {
        "window_samples": cfg["window_samples"],
        "sample_rate_hz": cfg["sample_rate_hz"],
        "n_channels": cfg["n_channels"],
        "stride": cfg["stride_samples"],
        "n_members": m.N_MEMBERS,
        "n_layers": m.N_LAYERS,
        "n_filters": m.N_FILTERS,
        "kernel_size": m.KERNEL_SIZE,
        "norm": "BatchNorm1d",
        "activation": "ReLU",
        "output_activation": "tanh",
        "batch_size": 1024,
        "lr": 0.001,
        "optimizer": "Adam",
        "loss": "MSE",
        "patience": 10,
        "threshold_percentile": m.THRESHOLD_PERCENTILE,
        "train_subjects": len(cfg["splits"]["train"]),
        "val_subjects": len(cfg["splits"]["val"]),
        "test_subjects": len(cfg["splits"]["test"]),
        "dilations": m.DILATIONS,
    }


def audit(processed: Path | str | None = None, verbose: bool = True) -> list[str]:
    """Compare live values against the paper. Returns the keys that differ *undeclared*."""
    obs = observed(processed)
    matches, declared, undeclared = [], [], []

    for key, want in PAPER.items():
        got = obs.get(key)
        if want == got:
            matches.append(key)
        elif key in _BY_KEY:
            declared.append(key)
        else:
            undeclared.append(key)

    if verbose:
        print(f"Fidelity audit against Table IV — {len(PAPER)} parameters\n")
        print(f"  matches the paper exactly ({len(matches)}):")
        for k in matches:
            print(f"    {k:22s} {PAPER[k]}")

        print(f"\n  declared deviations ({len(declared)}):")
        for k in declared:
            d = _BY_KEY[k]
            print(f"    [{d.category:12s}] {k:22s} paper={PAPER[k]!r}  ours={obs[k]!r}")
            for line in _wrap(d.reason, 84):
                print(f"                     {line}")

        if undeclared:
            print(f"\n  UNDECLARED deviations ({len(undeclared)}) -- these need a reason or a fix:")
            for k in undeclared:
                print(f"    {k:22s} paper={PAPER[k]!r}  ours={obs[k]!r}")
        else:
            print("\n  no undeclared deviations")

        by_cat = {c: sum(1 for d in DEVIATIONS if d.category == c and d.key in declared)
                  for c in ("forced", "unspecified", "choice")}
        print(f"\n  summary: {len(matches)} exact, "
              + ", ".join(f"{v} {k}" for k, v in by_cat.items())
              + (f", {len(undeclared)} UNDECLARED" if undeclared else ""))

    return undeclared


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    import sys

    sys.exit(1 if audit() else 0)
