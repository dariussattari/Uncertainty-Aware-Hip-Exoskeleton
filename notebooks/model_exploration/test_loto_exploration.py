"""Check exact threshold arithmetic, saved-run alignment and holdout figure semantics."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

from _loto_exploration import (LOTO, HOLDOUTS, TASKS, ORANGE, BLUE, LotoScores,
                               load_loto_scores, threshold_rates, plot_loto_scores)


class LotoExplorationTests(unittest.TestCase):
    def test_strict_threshold_ties_and_endpoints(self):
        b = LotoScores('test', 'ST', 'primary', np.array([1., 2., 2., 3.]),
                       np.array([0, 0, 1, 1]), np.array(['LG', 'LG', 'ST', 'ST']), 2., Path('.'))
        rates = threshold_rates(b, [0., 2., 3.])
        np.testing.assert_array_equal(rates['OOD recall (%)'], [100., 50., 0.])
        np.testing.assert_array_equal(rates['ID false alarms (%)'], [100., 0., 0.])
        self.assertTrue(rates['SA flagged (%)'].isna().all())
        with self.assertRaises(ValueError):
            threshold_rates(b, [np.nan])

    def test_all_saved_runs_and_probabilistic_scores_reproduce_metrics(self):
        for key in LOTO:
            scores = ('epistemic', 'aleatoric', 'total', 'ratio') if key == 'prob_ensemble' else ('primary',)
            for task in HOLDOUTS:
                for score in scores:
                    with self.subTest(key=key, task=task, score=score):
                        b = load_loto_scores(key, task, score)
                        self.assertEqual(set(b.modes), set(TASKS))
                        np.testing.assert_array_equal(b.labels, b.modes == task)

    def test_gan_standing_what_if(self):
        b = load_loto_scores('gan', 'ST')
        rates = threshold_rates(b, [b.threshold, .60, .59])
        self.assertEqual(rates['OOD recall (%)'][0], 0.)
        self.assertAlmostEqual(rates['OOD recall (%)'][1], 78.421900161, places=6)
        self.assertEqual(rates['OOD recall (%)'][2], 100.)
        self.assertTrue(np.all(np.diff(rates['ID false alarms (%)']) > 0))

    def test_misaligned_metadata_rejected(self):
        b = load_loto_scores('gan', 'ST')
        fake = pd.read_parquet(b.run / 'scores_test.parquet')
        fake.loc[fake['mode'].eq('ST'), 'mode'] = 'SA'
        with patch('_loto_exploration.pd.read_parquet', return_value=fake):
            with self.assertRaisesRegex(ValueError, 'ONLY ST'):
                load_loto_scores('gan', 'ST')

    def test_missing_run_and_invalid_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_loto_scores('vae', 'SA', run=tmp)
        with self.assertRaises(ValueError):
            load_loto_scores('gan', 'LG')

    def test_only_heldout_is_orange_and_candidate_is_shown(self):
        for task in HOLDOUTS:
            b = load_loto_scores('gan', task)
            fig = plot_loto_scores(b, candidate=.60)
            labels = fig.axes[0].get_yticklabels()
            for mode, label in zip(TASKS, labels):
                self.assertEqual(to_rgba(label.get_color()), to_rgba(ORANGE if mode == task else BLUE))
                self.assertEqual('(held out)' in label.get_text(), mode == task)
            self.assertIn(f'Held out: {task}', fig._suptitle.get_text())
            self.assertEqual(len(fig.axes[0].get_legend().get_texts()), 2)
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
