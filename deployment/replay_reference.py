"""PyTorch reference predictions for the Sci-Rob replay (stage A): what the Jetson replay must reproduce.

    .venv/bin/python deployment/replay_reference.py ml/nature/estimator/runs/hip_thigh_aug_s2 [--subjects dev|all|AB15,...]

For each participant file of ``scirob_replay.py`` and each variant (nominal, calibrated), predicts the hip moment of
both legs at every ``STRIDE``-th sample (the deployed loop's 100 Hz) whose 249-sample window is complete and finite, the
left leg mirrored into the right leg's frame as the device does. Writes data/replay/scirob/predictions/reference/
<subject>.npz: ``pred_<variant>`` (n, 2) N m/kg, NaN where no prediction was made.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moment_model as mm  # noqa: E402
import scirob_replay as sr  # noqa: E402

STRIDE = 2                                               # samples between predictions: 100 Hz at 200 Hz, as deployed
BATCH = 1024


def get_ends(x: npt.NDArray, length: int, window: int, stride: int = STRIDE) -> npt.NDArray[np.int64]:
    """Within one trial (rows of ``x``, (n, 2, C)), the window ends on the stride grid whose window is all finite."""
    bad = np.r_[0, np.cumsum(~np.isfinite(x).all(axis = (1, 2)))]               # running count of incomplete samples
    ends = np.arange(window - 1, length, stride)
    return ends[(bad[ends + 1] - bad[ends + 1 - window]) == 0]


def get_subjects(choice: str) -> list[str]:
    files = sorted(sr.OUT.glob('AB*.npz'))
    if choice == 'dev':
        result = [f.stem for f in files if f.stem in sr.DEV]
    elif choice == 'all':
        result = [f.stem for f in files]
    else:
        result = choice.split(',')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__.split('\n')[0])
    parser.add_argument('run', type = Path)
    parser.add_argument('--subjects', default = 'dev', help = "'dev', 'all' or a comma-separated list")
    args = parser.parse_args()
    torch.manual_seed(0)
    model, meta = mm.load_run(args.run)
    window = model.model.receptive_field
    sign = torch.from_numpy(mm.get_left_sign(meta['channels'])).view(1, -1, 1)  # (1, C, 1)
    out = sr.OUT / 'predictions' / 'reference'
    out.mkdir(parents = True, exist_ok = True)
    for subject in get_subjects(args.subjects):
        data = np.load(sr.OUT / f'{subject}.npz')
        result = {}
        for variant in sr.VARIANTS:
            x_all = data[f'x_{variant}']                                         # (N, 2, C), legs unmirrored
            pred = np.full((len(x_all), 2), np.nan, np.float32)
            for start, length in zip(data['start'], data['length']):
                x = x_all[start:start + length]
                ends = get_ends(x, length, window)
                for chunk in np.array_split(ends, max(1, len(ends) // BATCH)):
                    idx = chunk[:, None] + np.arange(1 - window, 1)              # (B, W) sample indices
                    w = torch.from_numpy(x[idx]).permute(0, 2, 3, 1)               # (B, W, 2, C) -> (B, 2, C, W)
                    w[:, 0] = w[:, 0] * sign                                       # mirror the left leg
                    with torch.inference_mode():
                        y = model(w.reshape(-1, w.shape[2], window)).reshape(len(chunk), 2)   # (2B, 1) -> (B, 2)
                    pred[start + chunk] = y.numpy()
            result[f'pred_{variant}'] = pred
        np.savez(out / f'{subject}.npz', **result, stride = STRIDE, run = str(args.run))
        print(f'{subject}: {np.isfinite(result["pred_nominal"][:, 0]).sum():,} predictions per leg and variant')


if __name__ == '__main__':
    main()
