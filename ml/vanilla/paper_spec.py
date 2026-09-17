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
``extension``
    Not a deviation at all: a deliberate departure into territory the paper does not cover.
    Used only by Experiments 5 and 6, which have no counterpart row in Table I. Listed here so
    that the difference between "we could not match the paper" and "we went further than the
    paper" is explicit rather than left to the reader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import models.ensemble as m
from dataset import repo_root

__all__ = ["PAPER", "DEVIATIONS", "observed", "audit",
           "PAPER_AE", "DEVIATIONS_AE", "observed_ae", "audit_ae",
           "PAPER_GAN", "DEVIATIONS_GAN", "observed_gan", "audit_gan",
           "PAPER_SYN", "DEVIATIONS_SYN", "observed_synthetic", "audit_synthetic"]


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
    root = Path(processed) if processed else repo_root() / "data" / "processed"
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


# ============================ Autoencoder =====================================
# Table IV, "Autoencoder Model" + "Training Strategy Comparison"

PAPER_AE: dict[str, object] = {
    "window_samples": 175,
    "sample_rate_hz": 175,
    "n_channels": 16,
    "stride": 10,
    "enc_channels": (19, 24),          # "Conv1d(16->19->24->bottleneck)"
    "kernels": (15, 17, 19),           # "kernel sizes: 15, 17, 19"
    "latent_filters": 4,               # "Filter dimension of latent space: 4"
    "latent_time": 7,                  # "Time dimension of latent space: 7"
    "latent_size": 28,                 # "Latent space size: 28"
    "norm": "BatchNorm1d",             # "ReLU, BatchNorm1d after each layer"
    "activation": "ReLU",
    "dropout": 0.0,                    # "No dropout"
    "batch_size": 1024,
    "lr": 0.064,                       # "LR: 0.001*(1024/16)"
    "optimizer": "Adam",
    "loss": "MSE",
    "patience": 5,                     # "80/20 train/valid split, patience=5"
    "val_fraction": 0.2,
    "scoring": "LOF(latent)",          # "LOF(latent space), 99.5th percentile threshold"
    "threshold_percentile": 99.5,
    "strategy": "single-stage",        # "Train on all subjects with validation split"
    "lof_neighbors": None,             # k is not stated
}

DEVIATIONS_AE: tuple[Deviation, ...] = (
    Deviation("sample_rate_hz", "forced",
              "The Molinaro dataset is recorded at 200 Hz; the ankle data was 175 Hz."),
    Deviation("window_samples", "forced",
              "200 samples at 200 Hz preserves the paper's 1-second window, as for the "
              "ensemble. Duration is what is held constant."),
    Deviation("latent_time", "forced",
              "8 rather than 7. The paper's 7 is 175 samples at 25x temporal compression; "
              "200/25 = 8. The compression factor is what is held constant, and it falls out "
              "exactly as two AvgPool1d(5) stages for both 175 and 200 -- good evidence that "
              "is how the original achieved it."),
    Deviation("latent_size", "forced",
              "32 rather than 28, following directly from latent_time being 8 not 7. The "
              "filter dimension (4) is unchanged."),
    Deviation("lr", "choice",
              "0.003 rather than the specified 0.064. Measured on this data: at 0.064 "
              "reconstruction MSE plateaus at 0.959 on standardized inputs -- the encoder has "
              "collapsed to predicting channel means, and 63% of latent vectors become "
              "identical, which also makes LOF ill-conditioned. 0.01, 0.003 and 0.001 all "
              "train; 0.003 was best over a four-epoch comparison. This is the only place the "
              "specification had to be overridden rather than interpreted."),
    Deviation("lof_neighbors", "unspecified",
              "The paper does not state k for LOF. sklearn's default of 20 is used."),
)

_BY_KEY_AE = {d.key: d for d in DEVIATIONS_AE}


def observed_ae() -> dict[str, object]:
    """What the autoencoder code is actually using right now."""
    import models.autoencoder as ae
    from training.autoencoder import AeConfig

    cfg = AeConfig()
    net = ae.create_autoencoder(16, 200)
    has_dropout = any(isinstance(mod, __import__("torch").nn.Dropout)
                      for mod in net.modules())
    return {
        "window_samples": 200, "sample_rate_hz": 200, "n_channels": 16, "stride": 10,
        "enc_channels": ae.CHANNELS, "kernels": ae.KERNELS,
        "latent_filters": net.latent_filters, "latent_time": net.latent_time,
        "latent_size": net.latent_size,
        "norm": "BatchNorm1d", "activation": "ReLU", "dropout": 1.0 if has_dropout else 0.0,
        "batch_size": cfg.batch_size, "lr": cfg.lr, "optimizer": "Adam", "loss": "MSE",
        "patience": cfg.patience, "val_fraction": cfg.val_fraction,
        "scoring": "LOF(latent)", "threshold_percentile": cfg.threshold_percentile,
        "strategy": "single-stage", "lof_neighbors": cfg.lof_neighbors,
    }


def audit_ae(verbose: bool = True) -> list[str]:
    """Compare the autoencoder against Table IV. Returns undeclared mismatches."""
    obs = observed_ae()
    matches, declared, undeclared = [], [], []
    for key, want in PAPER_AE.items():
        got = obs.get(key)
        if want == got:
            matches.append(key)
        elif key in _BY_KEY_AE:
            declared.append(key)
        else:
            undeclared.append(key)

    if verbose:
        print(f"Autoencoder fidelity audit against Table IV -- {len(PAPER_AE)} parameters\n")
        print(f"  matches the paper exactly ({len(matches)}):")
        for k in matches:
            print(f"    {k:22s} {PAPER_AE[k]}")
        print(f"\n  declared deviations ({len(declared)}):")
        for k in declared:
            d = _BY_KEY_AE[k]
            print(f"    [{d.category:12s}] {k:22s} paper={PAPER_AE[k]!r}  ours={obs[k]!r}")
            for line in _wrap(d.reason, 84):
                print(f"                     {line}")
        if undeclared:
            print(f"\n  UNDECLARED deviations ({len(undeclared)}) -- need a reason or a fix:")
            for k in undeclared:
                print(f"    {k:22s} paper={PAPER_AE[k]!r}  ours={obs[k]!r}")
        else:
            print("\n  no undeclared deviations")
        by_cat = {c: sum(1 for d in DEVIATIONS_AE if d.category == c and d.key in declared)
                  for c in ("forced", "unspecified", "choice")}
        print(f"\n  summary: {len(matches)} exact, "
              + ", ".join(f"{v} {k}" for k, v in by_cat.items())
              + (f", {len(undeclared)} UNDECLARED" if undeclared else ""))
    return undeclared


# ============================ GAN =============================================
# Table IV, "GAN Model"

PAPER_GAN: dict[str, object] = {
    "window_samples": 175,
    "sample_rate_hz": 175,
    "n_channels": 16,
    # "Step: 20" -- twice the other models'. The stored windows are already cut at stride
    # 10, so this is realised as training on every second stored window (GanData.stride_factor)
    # and it applies to TRAINING ONLY: the test split is scored unsubsampled so all four
    # models are compared on one identical set of test windows.
    "stride": 20,
    "g_channels": (10, 12, 16),        # "ConvTranspose1d(10->10->12->16)"
    "g_kernel": 3,
    "g_activation": "LeakyReLU(0.2)",
    "g_norm": None,                    # not stated for the generator
    "d_channels": (30, 30),            # "Conv1d(16->30->30)"
    "d_kernel": 5,
    "d_activation": "ReLU",
    "d_dropout": 0.2,
    "d_spectral_norm": True,
    "d_output": "sigmoid",
    "latent_channels": 10,             # "Filter dimension of latent space: 10"
    "latent_time": 35,                 # "Time dimension of latent space: 35"
    "upsample_factor": 5,              # 175 / 35; the generator's total upsampling
    "batch_size": 256,
    "lr_g": 2e-4,
    "lr_d": 5e-5,
    "lr_decay": 0.99,                  # "exponential decay, gamma=0.99"
    "betas": (0.5, 0.999),
    "optimizer": "Adam",
    "loss": "BCE",
    "max_epochs": 500,                 # "500 epochs, fixed"
    "early_stopping": False,
    "val_fraction": 0.2,
    "g_steps_per_d": 5,                # "5 generator updates per discriminator update"
    "checkpoint_every": 5,
    "scoring": "1-D(x)",
    "threshold_percentile": 99.5,
    "strategy": "single-stage",
    "noise_distribution": None,        # not stated
    "upsample_distribution": None,     # how the 5x splits over three layers is not stated
    "d_head": None,                    # the reduction to a scalar is not stated
}

DEVIATIONS_GAN: tuple[Deviation, ...] = (
    Deviation("sample_rate_hz", "forced",
              "The Molinaro dataset is recorded at 200 Hz; the ankle data was 175 Hz."),
    Deviation("window_samples", "forced",
              "200 samples at 200 Hz preserves the paper's 1-second window. Duration is what "
              "is held constant, as for the ensemble and the autoencoder."),
    Deviation("latent_time", "forced",
              "40 rather than 35. The paper's 35 is 175 samples at a 5x generator upsample; "
              "200/5 = 40. The upsample factor is what is held constant, and it is exactly 5 "
              "for both window lengths -- the same clean scaling the autoencoder's pooling "
              "showed. The filter dimension (10) is unchanged."),
    Deviation("upsample_distribution", "unspecified",
              "Three transpose layers must multiply to 5x and the paper does not say how it "
              "is divided. (5, 1, 1) is used: the stride on the first layer, the remaining "
              "two refining at full resolution. This is the simplest reading and the only one "
              "that needs no non-integer stride."),
    Deviation("d_head", "unspecified",
              "The specification ends at Conv1d(16->30->30) and a sigmoid, with no stated "
              "reduction from 30 channels x 200 timesteps to one number. Global average "
              "pooling then a linear layer is used, which keeps the verdict invariant to "
              "where in the window an anomaly falls."),
    Deviation("g_norm", "unspecified",
              "Table IV names BatchNorm1d for the autoencoder and dropout for the "
              "discriminator, but states no normalisation for the generator. BatchNorm1d on "
              "the two hidden transpose layers is used (output left linear), which is "
              "standard DCGAN practice. Flagging it because it is not neutral here: the "
              "generator's output is under-dispersed (mean per-channel sd 0.59 against the "
              "data's 1.03), and that under-dispersion is what lets the discriminator "
              "separate real from generated on amplitude alone. Generator normalisation is "
              "therefore one of the few levers that could plausibly change this model's "
              "result, and it is an inference rather than a specification."),
    Deviation("noise_distribution", "unspecified",
              "Not stated. A standard normal latent is used, which is the default for "
              "essentially every convolutional GAN."),
    Deviation("d_output", "choice",
              "The discriminator returns a logit and the sigmoid lives inside "
              "BCEWithLogitsLoss rather than in the module. Mathematically identical to the "
              "specified sigmoid output with BCE, and numerically stabler; "
              "Discriminator.probability() and HipGan.uncertainty() apply the sigmoid "
              "explicitly, so D(x) in [0,1] is what the scorer sees."),
)

_BY_KEY_GAN = {d.key: d for d in DEVIATIONS_GAN}


def observed_gan() -> dict[str, object]:
    """What the GAN code is actually using right now."""
    import torch.nn as nn

    import models.gan as g
    from dataset import GanData
    from training.gan import GanConfig

    cfg = GanConfig()
    net = g.create_gan(16, 200)
    d = net.discriminator
    spectral = any("parametrizations" in n for n, _ in d.named_parameters())
    drop = [m.p for m in d.modules() if isinstance(m, nn.Dropout)]
    total_up = 1
    for u in g.UPSAMPLE:
        total_up *= u
    return {
        "window_samples": 200, "sample_rate_hz": 200, "n_channels": 16,
        "stride": 10 * GanData.stride_factor,
        "g_channels": g.G_CHANNELS, "g_kernel": g.G_KERNEL,
        "g_activation": f"LeakyReLU({g.LEAKY_SLOPE})",
        "g_norm": "BatchNorm1d (hidden layers only)",
        "d_channels": g.D_CHANNELS, "d_kernel": g.D_KERNEL,
        "d_activation": "ReLU", "d_dropout": drop[0] if drop else 0.0,
        "d_spectral_norm": spectral, "d_output": "logit (sigmoid in the loss)",
        "latent_channels": net.latent_shape[0], "latent_time": net.latent_shape[1],
        "upsample_factor": total_up,
        "batch_size": cfg.batch_size, "lr_g": cfg.lr, "lr_d": cfg.lr_d,
        "lr_decay": cfg.lr_decay, "betas": cfg.betas, "optimizer": "Adam", "loss": "BCE",
        "max_epochs": cfg.max_epochs, "early_stopping": False,
        "val_fraction": cfg.val_fraction, "g_steps_per_d": cfg.g_steps_per_d,
        "checkpoint_every": cfg.checkpoint_every, "scoring": "1-D(x)",
        "threshold_percentile": cfg.threshold_percentile, "strategy": "single-stage",
        "noise_distribution": "standard normal",
        "upsample_distribution": g.UPSAMPLE,
        "d_head": "AdaptiveAvgPool1d(1) + Linear",
    }


def audit_gan(verbose: bool = True) -> list[str]:
    """Compare the GAN against Table IV. Returns undeclared mismatches."""
    obs = observed_gan()
    matches, declared, undeclared = [], [], []
    for key, want in PAPER_GAN.items():
        got = obs.get(key)
        if want == got:
            matches.append(key)
        elif key in _BY_KEY_GAN:
            declared.append(key)
        else:
            undeclared.append(key)

    if verbose:
        print(f"GAN fidelity audit against Table IV -- {len(PAPER_GAN)} parameters\n")
        print(f"  matches the paper exactly ({len(matches)}):")
        for k in matches:
            print(f"    {k:24s} {PAPER_GAN[k]}")
        print(f"\n  declared deviations ({len(declared)}):")
        for k in declared:
            d = _BY_KEY_GAN[k]
            print(f"    [{d.category:12s}] {k:24s} paper={PAPER_GAN[k]!r}  ours={obs[k]!r}")
            for line in _wrap(d.reason, 84):
                print(f"                       {line}")
        if undeclared:
            print(f"\n  UNDECLARED deviations ({len(undeclared)}) -- need a reason or a fix:")
            for k in undeclared:
                print(f"    {k:24s} paper={PAPER_GAN[k]!r}  ours={obs[k]!r}")
        else:
            print("\n  no undeclared deviations")
        by_cat = {c: sum(1 for d in DEVIATIONS_GAN if d.category == c and d.key in declared)
                  for c in ("forced", "unspecified", "choice")}
        print(f"\n  summary: {len(matches)} exact, "
              + ", ".join(f"{v} {k}" for k, v in by_cat.items())
              + (f", {len(undeclared)} UNDECLARED" if undeclared else ""))
    return undeclared


# ============================ Label-free ensembles ============================
# Table IV, "Ensemble Models" + "Ensemble: Synthetic Target Model".
#
# Experiment 4 reproduces the paper's row. Experiments 5 and 6 are extensions with no
# counterpart in the paper, so for them the architecture rows are still audited against
# Table IV -- they inherit it unchanged -- while the target itself is recorded as an
# extension rather than a deviation.

PAPER_SYN: dict[str, object] = {
    "window_samples": 175,
    "sample_rate_hz": 175,
    "n_channels": 16,
    "stride": 10,
    "n_members": 7,
    "n_layers": 3,
    "n_filters": 30,
    "kernel_size": 20,
    "norm": "BatchNorm1d",
    "activation": "ReLU",
    "output_activation": "linear",       # a single unbounded scalar, unlike the tanh heads
    "n_targets": 1,                      # "sum of pairwise correlations" -- one value
    "target": "summed pairwise correlation",
    "batch_size": 1024,
    "lr": 1e-3,
    "optimizer": "Adam",
    "loss": "MSE",
    "patience": 10,
    "strategy": "two-stage LOSO",
    "scoring": "variance across branches",
    "threshold_percentile": 99.5,
    "dilations": None,                   # not stated
}

_SHARED_SYN = (
    Deviation("sample_rate_hz", "forced",
              "The Molinaro dataset is recorded at 200 Hz; the ankle data was 175 Hz."),
    Deviation("window_samples", "forced",
              "200 samples at 200 Hz preserves the paper's 1-second window, as in every "
              "other experiment. Duration is what is held constant."),
    Deviation("dilations", "unspecified",
              "Table IV gives 3 layers of kernel 20 but no dilation factors. (1, 2, 8) gives "
              "a receptive field of 210 samples, covering the whole window; conventional "
              "doubling (1, 2, 4) would reach only 134 and leave the first third unreachable. "
              "Identical to Experiment 1, deliberately -- these experiments change the target "
              "and nothing else."),
)

DEVIATIONS_SYN: dict[str, tuple[Deviation, ...]] = {
    "correlation": _SHARED_SYN + (
        Deviation("target", "choice",
                  "A standard-deviation floor of 0.01 is applied to the correlation "
                  "denominator. Pearson correlation divides by the product of two channel "
                  "standard deviations, and standing has 89% of its channels below that floor "
                  "in standardized units -- unfloored, its target is the correlation of sensor "
                  "noise, a different random value every window. Flooring shrinks such pairs "
                  "smoothly toward zero rather than introducing a discontinuity, and leaves "
                  "every non-degenerate task bit-identical. Measured in "
                  "05_synthetic_targets.ipynb: the target's spread within standing halves, "
                  "and no other mode moves at any floor tested."),
        Deviation("n_targets", "choice",
                  "The target is additionally standardized using training-split statistics. "
                  "Its raw spread is roughly twice the gait-phase target's, and the learning "
                  "rate is inherited from Table IV rather than retuned, so standardizing "
                  "keeps the effective step size comparable to Experiment 1's. This is "
                  "reported under n_targets only because the paper states no target scaling "
                  "either way."),
    ),
    "forecast_angle": _SHARED_SYN + (
        Deviation("target", "extension",
                  "Not the paper's target. Hip angle for both legs, 40 samples (200 ms) after "
                  "the window ends. The motivation is that gait phase is recovered from force "
                  "plates offline and does not exist at run time, so the paper can only ever "
                  "measure branch disagreement; a future sensor value is available live, which "
                  "makes prediction error measurable online as a second axis. The horizon is "
                  "not free: a two-tap linear filter predicts hip angle 5 ms ahead with 0.99 "
                  "skill and 50 ms ahead with 0.83, so at short horizons every branch learns "
                  "the same extrapolator and the variance collapses. Skill crosses zero near "
                  "40 samples."),
        Deviation("n_targets", "extension",
                  "Two outputs rather than one -- left and right hip angle."),
    ),
    "forecast_all": _SHARED_SYN + (
        Deviation("target", "extension",
                  "Not the paper's target. All sixteen channels at the same 200 ms horizon. "
                  "Differs from forecast_angle only in output width, which makes the pair a "
                  "clean ablation. It is the harder target of the two -- aggregate baseline "
                  "skill -0.36 against -0.12 -- because accelerometers are unpredictable even "
                  "50 ms out while the encoders are smooth, and harder targets produce more "
                  "branch disagreement. Per-channel variance additionally gives fault "
                  "attribution."),
        Deviation("n_targets", "extension",
                  "Sixteen outputs rather than one -- the full sensor state."),
    ),
}


def observed_synthetic(target: str = "correlation") -> dict[str, object]:
    """What one of the label-free ensembles is actually configured to do right now."""
    import models.ensemble as m
    from dataset import SyntheticEnsembleData
    from training.synthetic import CONFIGS

    cfg = CONFIGS[target]()
    net = m.create_ensemble(16, 1, activation="linear")
    member = net.members[0]
    data = SyntheticEnsembleData(batch_size=cfg.batch_size, target=target,
                                 horizon=cfg.horizon)
    names = {"correlation": "summed pairwise correlation",
             "forecast_angle": f"hip angle at +{cfg.horizon} samples",
             "forecast_all": f"all 16 channels at +{cfg.horizon} samples"}
    return {
        "window_samples": 200, "sample_rate_hz": 200, "n_channels": 16, "stride": 10,
        "n_members": cfg.n_members, "n_layers": len(member.dilations),
        "n_filters": m.N_FILTERS, "kernel_size": member.kernel_size,
        "norm": "BatchNorm1d", "activation": "ReLU",
        "output_activation": member.activation,
        "n_targets": data.n_targets, "target": names[target],
        "batch_size": cfg.batch_size, "lr": cfg.lr, "optimizer": "Adam", "loss": "MSE",
        "patience": cfg.patience, "strategy": "two-stage LOSO",
        "scoring": "variance across branches",
        "threshold_percentile": cfg.threshold_percentile,
        "dilations": member.dilations,
    }


def audit_synthetic(target: str = "correlation", verbose: bool = True) -> list[str]:
    """Compare one label-free ensemble against Table IV. Returns undeclared mismatches."""
    obs = observed_synthetic(target)
    by_key = {d.key: d for d in DEVIATIONS_SYN[target]}
    matches, declared, undeclared = [], [], []
    for key, want in PAPER_SYN.items():
        got = obs.get(key)
        if want == got:
            matches.append(key)
        elif key in by_key:
            declared.append(key)
        else:
            undeclared.append(key)

    if verbose:
        from training.synthetic import describe_target
        print(f"{describe_target(target)}")
        print(f"Fidelity audit against Table IV -- {len(PAPER_SYN)} parameters\n")
        print(f"  matches the paper exactly ({len(matches)}):")
        for k in matches:
            print(f"    {k:22s} {PAPER_SYN[k]}")
        print(f"\n  declared deviations ({len(declared)}):")
        for k in declared:
            d = by_key[k]
            print(f"    [{d.category:12s}] {k:22s} paper={PAPER_SYN[k]!r}  ours={obs[k]!r}")
            for line in _wrap(d.reason, 84):
                print(f"                     {line}")
        if undeclared:
            print(f"\n  UNDECLARED deviations ({len(undeclared)}) -- need a reason or a fix:")
            for k in undeclared:
                print(f"    {k:22s} paper={PAPER_SYN[k]!r}  ours={obs[k]!r}")
        else:
            print("\n  no undeclared deviations")
        cats = ("forced", "unspecified", "choice", "extension")
        by_cat = {c: sum(1 for d in DEVIATIONS_SYN[target]
                         if d.category == c and d.key in declared) for c in cats}
        print(f"\n  summary: {len(matches)} exact, "
              + ", ".join(f"{v} {k}" for k, v in by_cat.items() if v)
              + (f", {len(undeclared)} UNDECLARED" if undeclared else ""))
    return undeclared


if __name__ == "__main__":
    import sys

    bad = audit()
    for fn in (audit_ae, audit_gan):
        print("\n" + "=" * 78 + "\n")
        bad += fn()
    for tgt in ("correlation", "forecast_angle", "forecast_all"):
        print("\n" + "=" * 78 + "\n")
        bad += audit_synthetic(tgt)
    sys.exit(1 if bad else 0)
