"""Factor Finder: the label-free generator tools for cross-sectional factors.

Modules:
    config_io   load, validate and version-hash eval_config.toml
    fieldmap    the generator's terminals on the engine's point-in-time panels
    libops      the library's operators as a generator registry, and the tree limits
    novelty     the novelty index: every existing factor's G-zone values
    probe       the label-free generator probe (finder.py --generator-probe)
    incumbents  the incumbents of the factor lab and AlphaForge, and the number for the library

The package reads data and writes nothing. Its command line is finder.py.

This module holds the shared helpers:
    data_through()  the UTC date of the newest closed daily bar on disk
    utc_today()     the UTC wall-clock date
    repo_label(p)   a repository-relative label for a path, never an absolute path

An import of the package creates no directory and no file. Nothing here imports
ingest.py, ccxt or alphaforge_bridge.py. Thus an import cannot reach the network or run
the miner. test_finder_isolation.py checks sys.modules in a new process.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import config as C


class FinderError(RuntimeError):
    """A Finder helper cannot continue (for example: no daily bars on disk)."""


# --------------------------------------------------------------------------- time


def data_through() -> date:
    """UTC open-time date of the newest closed daily bar on disk.

    A bar stamped D closes at D+1 00:00 UTC; the still-forming bar is never
    written (an engine invariant), so the newest stamp on disk IS the last
    closed bar. The newest bar of any universe symbol counts."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    newest = None
    for sym in C.UNIVERSE:
        p = C.RAW_OHLCV / f"{C.sym_to_file(sym)}.parquet"
        if not p.exists():
            continue
        ts = pc.max(pq.read_table(p, columns=["ts"]).column("ts")).as_py()
        if ts is not None and (newest is None or ts > newest):
            newest = ts
    if newest is None:
        raise FinderError("no daily bars on disk: data-through date is undefined")
    if newest.tzinfo is not None:
        newest = newest.astimezone(UTC)
    return newest.date()


def utc_today() -> date:
    return datetime.now(UTC).date()


# --------------------------------------------------------------------------- labels


def repo_label(p) -> str:
    """Repo-relative label for a path, or a placeholder; never an absolute path
    (output may be pasted into reports, and home paths are private)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(C.ROOT.parent.resolve()).as_posix()
    except ValueError:
        return "<outside this repo>"
