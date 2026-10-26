"""Score the moment estimators on the test participants, once, per task as the Nature paper reports them.

    python ml/nature/estimator/score_moment.py          # every run under ml/nature/estimator/runs with preds_test.npz

Per run, participant, task and joint (both legs pooled, every flagged window with a target): R^2, RMSE (N m/kg) and MAE
normalized by the ground truth's peak-to-peak range, the paper's three metrics (Methods, "Phase 3"; Extended Data
Fig. 4). Tasks are the paper's: incline walk and stairs are split by direction. A task's score is the mean over
participants; a model's score the mean over tasks, then mean +- SD over seeds.

``deployed`` is the study's own estimator (``torque_estimated``, hip sign flipped from the exo's extension-positive
convention: on the test participants' Phase 3 recordings target = -0.98 est (hip), 1.01 est (knee)). It is scored on
Phase 3 recordings only, where the final phase 1-2 model ran; every model is also scored on that subset for the
like-for-like comparison. A ``preds_test_perturbed.npz`` (B models: thigh IMU at random fits) scores as ``<family>@fit``.

Writes ``ml/nature/estimator/results/test_per_task.csv`` (family, seed, subset, task, joint, metrics, n participants)
and ``test_summary.csv``.
"""
import json
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moment_data as md  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
RESULTS = HERE / 'results'
DEPLOYED_SIGN = np.array([-1.0, 1.0], np.float32)       # hip extension-positive -> OpenSim flexion-positive; knee as is


def get_paper_task(windows: pd.DataFrame) -> npt.NDArray[np.object_]:
    """The paper's task names: incline walk and stairs carry their direction (inclined / declined, ascent / descent)."""
    task, direction = windows['task'].astype(str).to_numpy(), windows['direction'].astype(str).to_numpy()
    split = np.isin(task, ['incline_walk', 'stairs'])
    return np.where(split, np.char.add(np.char.add(task.astype(str), ':'), direction.astype(str)), task).astype(object)


def get_group_metrics(y: npt.NDArray[np.float32], p: npt.NDArray[np.float32]) -> dict[str, float]:
    """R^2, RMSE and MAE / peak-to-peak range of y over one participant's task."""
    err = p - y
    return {'r2': 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2), 'rmse': np.sqrt(np.mean(err ** 2)),
            'nmae': np.mean(np.abs(err)) / np.ptp(y)}


def score(frame: pd.DataFrame, pred: npt.NDArray[np.float32], target: npt.NDArray[np.float32], joints: list[str]) -> pd.DataFrame:
    """Per task and joint: metrics averaged over participants. frame has subject and paper_task per row."""
    rows = []
    for j, joint in enumerate(joints):
        ok = np.isfinite(target[:, j]) & np.isfinite(pred[:, j])
        df = pd.DataFrame({'subject': frame['subject'].to_numpy()[ok], 'task': frame['paper_task'].to_numpy()[ok],
                           'y': target[ok, j], 'p': pred[ok, j]})
        for (subject, task), g in df.groupby(['subject', 'task']):
            if len(g) > 1 and np.ptp(g['y']) > 0:
                rows.append({'joint': joint, 'subject': subject, 'task': task,
                             **get_group_metrics(g['y'].to_numpy(), g['p'].to_numpy())})
    per_person = pd.DataFrame(rows)
    return per_person.groupby(['task', 'joint']).agg(r2 = ('r2', 'mean'), rmse = ('rmse', 'mean'), nmae = ('nmae', 'mean'),
                                                     n_people = ('subject', 'nunique')).reset_index()


def get_runs() -> list[tuple[str, int, Path, list[str]]]:
    """(family, seed, prediction file, joints) for every finished run, perturbed exports as '<family>@fit'."""
    runs = []
    for summary in sorted(RUNS.glob('*_s[0-9]*/summary.json')):
        folder = summary.parent
        family, seed = folder.name.rsplit('_s', 1)
        joints = json.loads((folder / 'config.json').read_text()).get('joints', ['hip', 'knee'])
        runs.append((family, int(seed), folder / 'preds_test.npz', joints))
        if (folder / 'preds_test_perturbed.npz').exists():
            runs.append((f'{family}@fit', int(seed), folder / 'preds_test_perturbed.npz', joints))
    return runs


def main() -> None:
    windows = pd.read_parquet(md.OUT / 'windows.parquet')
    windows['paper_task'] = get_paper_task(windows)
    trained = set(get_paper_task(windows[windows['paper_train']]))
    tables, deployed_done = [], False
    for family, seed, path, joints in get_runs():
        npz = np.load(path)
        frame = windows.iloc[npz['rows']]
        phase3 = (frame['phase'] == 'phase3').to_numpy()
        for subset, keep in (('all', np.ones(len(frame), bool)), ('phase3', phase3)):
            t = score(frame[keep], npz['pred'][keep], npz['target'][keep], joints)
            tables.append(t.assign(family = family, seed = seed, subset = subset))
        if not deployed_done and joints == ['hip', 'knee']:                       # the same windows for every run; needs both targets
            est = npz['estimate'][phase3] * DEPLOYED_SIGN
            t = score(frame[phase3], est, npz['target'][phase3, :len(DEPLOYED_SIGN)], ['hip', 'knee'])
            tables.append(t.assign(family = 'deployed', seed = 0, subset = 'phase3'))
            deployed_done = True
    per_task = pd.concat(tables, ignore_index = True)
    per_task['trained_task'] = per_task['task'].isin(trained)
    RESULTS.mkdir(exist_ok = True)
    per_task.to_csv(RESULTS / 'test_per_task.csv', index = False)
    # mean over tasks per seed (all tasks, the 19 trained, the unseen), then mean +- SD over seeds
    groups = {'all tasks': per_task, 'trained tasks': per_task[per_task['trained_task']],
              'unseen tasks': per_task[~per_task['trained_task']]}
    summary = pd.concat([g.groupby(['family', 'subset', 'joint', 'seed'])[['r2', 'rmse', 'nmae']].mean().assign(tasks = name)
                         for name, g in groups.items()]).reset_index()
    summary = summary.groupby(['family', 'subset', 'joint', 'tasks'])[['r2', 'rmse', 'nmae']].agg(['mean', 'std', 'count'])
    summary.to_csv(RESULTS / 'test_summary.csv')
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(summary.round(3))


if __name__ == '__main__':
    main()
