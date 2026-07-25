"""The installed-wheel examples remain runnable and deterministic."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest


EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*.py"))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_python_example(example: Path) -> None:
    """Execute each documented example against the installed extension."""
    runpy.run_path(str(example), run_name="__main__")
