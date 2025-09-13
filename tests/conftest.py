"""Pytest configuration to ensure the local package is importable.

Adds the repository root to sys.path so tests can import `mrtk` without
requiring a formal package installation.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

