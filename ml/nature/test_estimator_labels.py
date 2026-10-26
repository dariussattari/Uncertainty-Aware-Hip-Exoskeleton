"""The one check for estimator_labels.py: errors at known answers, the envelope rule, and the bootstrap interval."""
import numpy as np
import pandas as pd

import estimator_labels as el


def check() -> None:
    rng = np.random.default_rng(0)
    # per (task, subject) errors: constant offset 0.2 on a +-1 square wave gives RMSE 0.2, RMS 1, ratio 0.2; an estimate of
    # the wrong sign on half the samples gives 50% wrong direction
    y = np.tile([1.0, -1.0], 500)
    pairs = pd.DataFrame({'subject': 'A', 'task': 'walk', 'side': 'l', 'y': y, 'p': y + 0.2})
    flip = pd.DataFrame({'subject': 'A', 'task': 'push', 'side': 'l', 'y': y, 'p': np.where(np.arange(1000) < 500, -y, y)})
    errors = el.get_person_task_errors(pd.concat([pairs, flip])).set_index('task')
    np.testing.assert_allclose(errors.loc['walk', ['rmse', 'rms_moment', 'relative_error']].to_numpy(float), [0.2, 1.0, 0.2])
    np.testing.assert_allclose(errors.loc['push', 'wrong_direction_pct'], 50.0)
    # envelope = the worst familiar task; an unfamiliar task clearly above it is dangerous, one clearly below is safe,
    # one at the envelope's level is near the boundary
    people = [f'P{i}' for i in range(12)]
    level = {'walk': 0.10, 'stairs': 0.20, 'squat': 0.12, 'push': 0.50, 'lunge': 0.20}
    rows = [{'task': t, 'subject': s, 'rmse': m + rng.normal(0, 0.01), 'rms_moment': 1.0, 'relative_error': m,
             'wrong_direction_pct': 0.0, 'samples': 100} for t, m in level.items() for s in people]
    table, env = el.get_task_table(pd.DataFrame(rows), ('walk', 'stairs'), rng, n_boot = 2000)
    assert env['envelope_task'] == 'stairs' and abs(env['envelope'] - 0.20) < 0.01, env
    assert table.loc['push', 'label'] == 'OOD-dangerous' and not table.loc['push', 'near_boundary']
    assert table.loc['squat', 'label'] == 'OOD-safe' and not table.loc['squat', 'near_boundary']
    assert table.loc['lunge', 'near_boundary'] and table.loc['walk', 'label'] == 'ID-safe'
    assert table.loc['push', 'rmse_lo'] < table.loc['push', 'rmse'] < table.loc['push', 'rmse_hi'] < table.loc['push', 'rmse'] + 0.02
    print('ok')


if __name__ == '__main__':
    check()
