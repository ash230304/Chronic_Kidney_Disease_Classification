"""
conftest.py — Root-level pytest configuration.

Adds the project root to sys.path so that `from src.xxx import ...`
works in all test modules without needing to set PYTHONPATH manually.
"""
import sys
from pathlib import Path

# Insert the project root (the directory containing 'src/') into sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
