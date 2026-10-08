"""Put ml/common (the dataset-agnostic models, features and signal helpers) on sys.path for the Science Robotics code.

Imported for its side effect by ``training/__init__.py``, ``evaluation/__init__.py`` and the flat scripts that import
``models`` directly; everything else in ml/sci_rob reaches it through its own paths module.
"""
import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parents[2] / 'common'
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
