"""Cross-model comparison figures assembled from completed evaluation outputs.

This module intentionally preserves each model's native Psi axis.  The five uncertainty
scores have different definitions and units, so placing them on one shared axis would imply
a numerical comparability that does not exist.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vanilla-comparison-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/vanilla-comparison-cache")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "runs" / "comparison" / "figures"

# (label, run directory relative to the run root, figure filename). The run root is a
# parameter rather than a constant because the same five panels exist under a different
# protocol in ml/sci_rob/LOTO/<TASK>/, and assembling participant-split panels under a
# leave-one-task-out heading would be a silent, entirely plausible-looking error.
PANEL_SPEC = [
    ("A. Gait-phase TCN ensemble", ("ensemble_gp", "vanilla/Ensemble_GP"), "psi_by_task.png"),
    ("B. Autoencoder with latent LOF", ("autoencoder", "vanilla/AE"), "ae_psi_by_task.png"),
    ("C. TCN GAN discriminator", ("gan", "vanilla/GAN"), "gan_psi_by_task.png"),
    ("D. Summed-correlation TCN ensemble",
     ("synthetic", "vanilla/Ensemble_synthetic"), "psi_by_task.png"),
    ("E. 200 ms bilateral hip-angle forecast ensemble",
     ("forecast_angle", "vanilla/Ensemble_forecast"), "psi_by_task.png"),
]


def panels(root: Path | None = None, loto: bool = False) -> list[tuple[str, Path]]:
    """Resolve the five panel paths under ``root`` (default: the participant-split runs)."""
    base = Path(root) if root is not None else ROOT / "runs"
    i = 1 if loto else 0
    return [(label, base / dirs[i] / "figures" / name) for label, dirs, name in PANEL_SPEC]


PANELS = panels()


def make_combined_psi_by_task(output_dir: Path = OUTPUT_DIR,
                              panel_list: list[tuple[str, Path]] | None = None,
                              subtitle: str | None = None) -> tuple[Path, Path]:
    """Stack the five completed per-task Psi distributions into one figure."""
    PANELS = panel_list if panel_list is not None else panels()
    missing = [str(path) for _, path in PANELS if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source figure(s):\n" + "\n".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)

    # The proportions preserve the source panels' 1.83:1 aspect ratio at nearly native
    # resolution while leaving a compact label strip above each panel.
    height_ratios = [0.34]
    for _ in PANELS:
        height_ratios.extend([0.09, 1.0])

    fig = plt.figure(figsize=(12, 37), facecolor="white")
    grid = fig.add_gridspec(
        nrows=len(height_ratios),
        ncols=1,
        height_ratios=height_ratios,
        left=0.015,
        right=0.985,
        top=0.992,
        bottom=0.008,
        hspace=0.025,
    )

    header = fig.add_subplot(grid[0])
    header.axis("off")
    header.text(
        0.5,
        0.72,
        r"Uncertainty score $\Psi$ by task across all completed methods",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#26313B",
    )
    header.text(
        0.5,
        0.27,
        subtitle or
        "Blue = in-distribution; orange = held-out pseudo-OOD. "
        "Each panel retains its own score scale and 99.5th-percentile ID-training threshold.",
        ha="center",
        va="center",
        fontsize=12,
        color="#59636E",
    )

    for panel_index, (label, source) in enumerate(PANELS):
        label_axis = fig.add_subplot(grid[1 + panel_index * 2])
        label_axis.axis("off")
        label_axis.text(
            0.015,
            0.42,
            label,
            ha="left",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#2B5C88",
        )

        image_axis = fig.add_subplot(grid[2 + panel_index * 2])
        image_axis.imshow(mpimg.imread(source))
        image_axis.axis("off")

    png_path = output_dir / "psi_by_task_all_methods.png"
    pdf_path = output_dir / "psi_by_task_all_methods.pdf"
    fig.savefig(png_path, dpi=165, facecolor="white")
    fig.savefig(pdf_path, dpi=165, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    for generated in make_combined_psi_by_task():
        print(generated)
