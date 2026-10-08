"""The one check for nature_data.py: stair direction from the recording name, and the shared-time-axis guard."""
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import nature_data as nt


def check():
    assert [nt.get_stair_direction(r) for r in ('stairs_2_9', 'stairs_1_1_10', 'jump_1_1')] == ['up', 'down', None]
    assert nt.parse_task('stairs_2_10') == 'stairs' and nt.parse_task('lift_weight_1_2') == 'lift_weight'
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        t = np.arange(5) / nt.RATE_HZ
        pd.DataFrame({'time': t, 'a': np.arange(5.)}).to_csv(folder / 'A.csv', index = False)
        pd.DataFrame({'time': t, 'b': -np.arange(5.)}).to_csv(folder / 'B.csv', index = False)
        df = nt.read_columns(folder, {'A': {'a': 'x'}, 'B': {'b': 'y'}})
        assert list(df) == ['x', 'y'] and df['y'].iloc[-1] == -4
        pd.DataFrame({'time': t + 1, 'b': np.zeros(5)}).to_csv(folder / 'B.csv', index = False)
        failed = False
        try:
            nt.read_columns(folder, {'A': {'a': 'x'}, 'B': {'b': 'y'}})
        except ValueError:
            failed = True
        assert failed, 'misaligned time axes must raise'
    print('ok')


if __name__ == '__main__':
    check()
