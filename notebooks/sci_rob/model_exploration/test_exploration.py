"""Small wiring tests for the notebook helpers, NOT trained-model benchmarks.

The VAE tests live with the package: ml/sci_rob/VAE/test_vae.py.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _exploration import run_commands


class ExplorationTests(unittest.TestCase):
    def test_opt_in_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run';out.mkdir();(out/'existing').touch()
            with patch('_exploration.subprocess.run') as call:
                run_commands([['python','never-run.py']],out,enabled=False)
                call.assert_not_called()
                with self.assertRaises(FileExistsError):
                    run_commands([['python','never-run.py']],out,enabled=True)
                call.assert_not_called()


if __name__=='__main__':unittest.main()
