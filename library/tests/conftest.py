"""Keep pytest's temporary files inside the repo.

Without this hook, `tmp_path` is under the system temp dir (pytest-of-<user>/).
Nothing that a test writes goes outside the repo. The base temp dir is
library/data/pytest-tmp/, inside the git-ignored data/ folder; pytest empties it
at the start of every run. An explicit --basetemp on the command line still wins.
The property tests (Hypothesis) keep their cache in library/data/hypothesis/, unless
HYPOTHESIS_STORAGE_DIRECTORY gives another folder.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

BASETEMP = Path(__file__).resolve().parents[1] / "data" / "pytest-tmp"
HYPOTHESIS_DIR = Path(__file__).resolve().parents[1] / "data" / "hypothesis"


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    os.environ.setdefault("HYPOTHESIS_STORAGE_DIRECTORY", str(HYPOTHESIS_DIR))
    # tryfirst: pytest's own tmp_path plugin reads the option in its configure hook
    if not config.option.basetemp:
        BASETEMP.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(BASETEMP)
