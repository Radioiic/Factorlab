"""Where the factor catalog is, and the guard that prevents changes to it.

``catalog/factors.db`` holds the factor definitions. ``./setup.sh`` builds it with
``python -m factorlab build`` (about 40 s). ``build`` writes the catalog when it does
not exist and refuses to replace an existing one without ``--overwrite-catalog``.
A rebuild renumbers ``factor_id`` (ids follow the generation order), and an
``evaluate``/``compute`` run adds rows to the same file. Thus the other commands
that write refuse ``catalog/`` unless the caller gives ``--overwrite-catalog``.
Work files go in ``data/`` next to the package. Git ignores ``catalog/`` and ``data/``.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = PACKAGE_ROOT / "catalog"
CATALOG_DB = CATALOG_DIR / "factors.db"
LOCAL_DATA = PACKAGE_ROOT / "data"

NO_CATALOG = "No catalog. Run ./setup.sh first."
COPY_HINT = (
    "copy it first (`mkdir -p data && cp catalog/factors.db data/factors.db`) "
    "and use data/factors.db (git ignores data/)"
)


class CatalogWriteError(RuntimeError):
    """A command would have written into the catalog folder without --overwrite-catalog."""


def in_catalog(path: str | Path) -> bool:
    """True if ``path`` is the catalog folder or anything inside it."""
    p = Path(path).expanduser().resolve()
    c = CATALOG_DIR.resolve()
    return p == c or c in p.parents


def is_catalog_db(path: str | Path) -> bool:
    """True if ``path`` is the catalog database ``catalog/factors.db``."""
    return Path(path).expanduser().resolve() == CATALOG_DB.resolve()


def refuse_catalog(*paths: str | Path | None, allow: bool = False, what: str = "write", option: bool = True) -> None:
    """Raise CatalogWriteError if any of ``paths`` lies in the catalog folder (unless ``allow``).

    ``option`` tells if the command has ``--overwrite-catalog``. Without it, the
    message names only another folder.
    """
    if allow:
        return
    hits = [str(p) for p in paths if p is not None and in_catalog(p)]
    if not hits:
        return
    head = f"refusing to {what} inside the catalog folder ({', '.join(hits)}): it holds the built factor catalog."
    if option:
        raise CatalogWriteError(
            f"{head} To work on a copy, {COPY_HINT}. To replace the catalog on purpose, add --overwrite-catalog."
        )
    raise CatalogWriteError(f"{head} Write to another folder, for example data/ (git ignores data/).")
