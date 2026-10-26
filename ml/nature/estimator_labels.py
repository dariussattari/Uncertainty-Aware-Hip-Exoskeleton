"""Controller classes from the deployed estimator's error, per task, on people it never saw (nature notebook 04).

The awareness layer hands unfamiliar movement to the joint-moment estimator (OOD-safe) or switches the exo off
(OOD-dangerous). Which tasks belong where was set by hand (``nature_data.OOD_DANGEROUS_TASKS``). This module measures
it instead, from ``torque_estimated``, the output of the estimator deployed in the Nature study (Molinaro et al. 2024),
against the biological hip moment.

Rule (fixed before any task was scored)
  people      Phase 3 participants who are not in Phase 1-2, i.e. never in the deployed estimator's training data
  samples     in-task samples of each leg (its own activity flag), both legs pooled per person, both values finite
  score       per task and person: RMSE of the estimate against the moment (Nm/kg); a task's score is the mean over
              people, so every person weighs the same
  envelope    the largest task score among the familiar tasks (the four the awareness models train on): the worst
              error the estimator is known to run with on movement the pipeline treats as normal
  label       an unfamiliar task is OOD-safe if its score is at or below the envelope, OOD-dangerous above it;
              'near the boundary' if the participant-bootstrap 95% interval of (score - envelope) contains zero
Context, not part of the rule: the error relative to the moment's own size (RMSE / RMS of the moment: 1 means the
estimator delivers torque no closer to the intended assistance than switching off), and the share of samples with
|moment| >= ``SIGN_FLOOR`` where the estimate has the opposite sign (assistance in the wrong direction).
"""
import numpy as np
import pandas as pd

import nature_data as nt

SIGN_FLOOR = 0.1           # Nm/kg: below this the moment's sign is not meaningful for wrong-direction assistance
N_BOOT = 4_000
CI = (2.5, 97.5)


def get_held_out_subjects(inv: pd.DataFrame) -> list[str]:
    """Phase 3 participants absent from Phase 1-2: never in the deployed estimator's training data."""
    trained = set(inv.loc[inv['phase'] == 'phase1-2', 'subject'])
    return sorted(set(inv.loc[inv['phase'] == 'phase3', 'subject']) - trained)


def load_pairs(inv: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    """In-task (moment, estimate) pairs of both legs for some Phase 3 participants, extension positive, Nm/kg.

    The OpenSim biological moment is flexion positive and is negated; the exo's estimate is extension positive.
    """
    recs = inv.loc[(inv['phase'] == 'phase3') & inv['subject'].isin(subjects), ['rec', 'subject', 'task']]
    cols = [f'{k}_{s}' for s in nt.SIDES for k in ('flag', 'bio', 'tq_est')]
    samples = nt.load_samples(cols, recs['rec']).merge(recs, on = 'rec')
    parts = []
    for s in nt.SIDES:
        y, p = -samples[f'bio_{s}'].to_numpy(np.float64), samples[f'tq_est_{s}'].to_numpy(np.float64)
        keep = (samples[f'flag_{s}'].to_numpy() > 0) & np.isfinite(y) & np.isfinite(p)
        parts.append(pd.DataFrame({'subject': samples['subject'].to_numpy()[keep], 'task': samples['task'].to_numpy()[keep],
                                   'side': s, 'y': y[keep], 'p': p[keep]}))
    return pd.concat(parts, ignore_index = True)


def get_person_task_errors(pairs: pd.DataFrame, sign_floor: float = SIGN_FLOOR) -> pd.DataFrame:
    """Per (task, subject): samples, RMSE, RMS of the moment, their ratio, and the wrong-direction share (%)."""
    err = pairs['p'].to_numpy() - pairs['y'].to_numpy()
    big = np.abs(pairs['y'].to_numpy()) >= sign_floor
    frame = pd.DataFrame({'task': pairs['task'].astype(str).to_numpy(), 'subject': pairs['subject'].astype(str).to_numpy(),
                          'se': err ** 2, 'y2': pairs['y'].to_numpy() ** 2, 'big': big,
                          'wrong': big & (np.sign(pairs['p'].to_numpy()) != np.sign(pairs['y'].to_numpy()))})
    g = frame.groupby(['task', 'subject'])
    out = pd.DataFrame({'samples': g.size(), 'rmse': np.sqrt(g['se'].mean()), 'rms_moment': np.sqrt(g['y2'].mean()),
                        'wrong_direction_pct': 100 * g['wrong'].sum() / g['big'].sum().clip(lower = 1)})
    out['relative_error'] = out['rmse'] / out['rms_moment']
    return out.reset_index()


def get_task_table(errors: pd.DataFrame, familiar: tuple[str, ...], rng: np.random.Generator,
                   n_boot: int = N_BOOT) -> tuple[pd.DataFrame, dict]:
    """Per task: mean over people of each error measure, the label, and bootstrap intervals; and the envelope.

    The bootstrap resamples people (with replacement) and recomputes every task's mean and the envelope in each
    resample, so the interval of (score - envelope) carries the envelope's own uncertainty. A person missing a task
    is skipped for that task's mean.
    """
    rmse = errors.pivot(index = 'subject', columns = 'task', values = 'rmse')         # (people, tasks)
    missing = [t for t in familiar if t not in rmse.columns]
    if missing:
        raise ValueError(f'familiar tasks without scores: {missing}')
    values = rmse.to_numpy()                                                         # (P, T), NaN where absent
    fam = np.array([t in familiar for t in rmse.columns])
    score = np.nanmean(values, axis = 0)                                             # (T,)
    envelope = float(score[fam].max())
    pick = rng.integers(0, len(values), size = (n_boot, len(values)))               # (B, P) people per resample
    with np.errstate(invalid = 'ignore'):
        boot = np.nanmean(values[pick], axis = 1)                                    # (B, P, T) -> (B, T)
    boot_env = np.nanmax(boot[:, fam], axis = 1, keepdims = True)                    # (B, 1)
    diff_lo, diff_hi = np.nanpercentile(boot - boot_env, CI, axis = 0)               # (T,), (T,)
    score_lo, score_hi = np.nanpercentile(boot, CI, axis = 0)
    means = errors.groupby('task')[['rms_moment', 'relative_error', 'wrong_direction_pct']].mean()
    table = pd.DataFrame({'people': rmse.notna().sum().to_numpy(), 'rmse': score, 'rmse_lo': score_lo, 'rmse_hi': score_hi,
                          'minus_envelope_lo': diff_lo, 'minus_envelope_hi': diff_hi}, index = rmse.columns)
    table = table.join(means)
    table['familiar'] = fam
    table['label'] = np.where(fam, 'ID-safe', np.where(table['rmse'] <= envelope, 'OOD-safe', 'OOD-dangerous'))
    table['near_boundary'] = ~fam & (table['minus_envelope_lo'] <= 0) & (table['minus_envelope_hi'] >= 0)
    table['provisional'] = [nt.get_controller_class(t) for t in table.index]
    table = table.sort_values('rmse')
    table.index.name = 'task'
    return table, {'envelope': envelope, 'envelope_task': str(rmse.columns[fam][np.argmax(score[fam])]),
                   'envelope_lo': float(np.percentile(boot_env, CI[0])), 'envelope_hi': float(np.percentile(boot_env, CI[1]))}

