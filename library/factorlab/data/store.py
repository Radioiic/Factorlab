"""Persist raw panels as one compressed CSV per field (time x symbol)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd


def save_panels(panels: dict[str, pd.DataFrame], directory: str | Path, *, gzip_mtime: int | None = None) -> None:
    """Write ``<directory>/<field>.csv.gz`` for every panel.

    gzip stores a modification time in each file's header; by default that is the write
    time, so writing the same panels twice gives the same CSV content but not the same
    bytes. ``gzip_mtime`` fixes that header field (for example 0), which makes the files
    byte-for-byte reproducible from the same panels (the header also holds the file name,
    which is the same for the same field)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    compression: Literal["infer"] | dict[str, Any] = (
        "infer" if gzip_mtime is None else {"method": "gzip", "mtime": int(gzip_mtime)}
    )
    for name, df in panels.items():
        df.to_csv(d / f"{name}.csv.gz", float_format="%.10g", compression=compression)


def load_panels(directory: str | Path) -> dict[str, pd.DataFrame]:
    """Read every ``<field>.csv.gz`` in ``directory`` and return the panels (field -> time x symbol).

    All panels get the union of the timestamps and of the symbols (missing cells are NaN).
    A folder without panel files gives FileNotFoundError.
    """
    d = Path(directory)
    panels = {}
    for f in sorted(d.glob("*.csv.gz")):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        panels[f.name[: -len(".csv.gz")]] = df.astype(float)
    if not panels:
        raise FileNotFoundError(f"no *.csv.gz panels found in {d}")
    # align every field on the union index / columns
    index = sorted(set().union(*(p.index for p in panels.values())))
    columns = sorted(set().union(*(p.columns for p in panels.values())))
    return {k: v.reindex(index=index, columns=columns) for k, v in panels.items()}
