"""pytest configuration: add scripts/ to sys.path so `scaffold` package is importable."""

import sys
from pathlib import Path

# Make the scaffold package importable from anywhere
SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

INPUT = Path(__file__).parent.parent / "input"
TEST_STL = INPUT / "test.stl"
TEST_SMALL_STL = INPUT / "test_small.stl"
