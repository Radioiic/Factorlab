"""factorlab: formula definitions of cross-sectional crypto factors.

The package generates the factor definitions, evaluates them on market panels
(time x symbol) and keeps them in a SQLite catalog. The main names:

* ``FactorDB``: open and query a factor database.
* ``Evaluator``: compute a factor expression on a dict of panels.
* ``parse`` and ``raw_dependencies``: check an expression and list its raw fields.
* ``generate_factors``: make the list of candidate definitions.
"""

from pathlib import Path as _Path

from .database import FactorDB
from .expr import Evaluator, parse, raw_dependencies
from .generator import generate_factors

__all__ = ["FactorDB", "Evaluator", "parse", "raw_dependencies", "generate_factors"]


def _read_version() -> str:
    """Return the version in the file VERSION at the root of the repository.

    A copy of the package without that file gets the version "0+unknown"."""
    path = _Path(__file__).resolve().parent.parent.parent / "VERSION"
    try:
        return path.read_text(encoding="ascii").strip() or "0+unknown"
    except OSError:
        return "0+unknown"


__version__ = _read_version()
