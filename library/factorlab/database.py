"""SQLite storage for the factor library, factor values and evaluation statistics.

``FactorDB`` opens one database file. ``SCHEMA`` gives its tables: ``factors`` (the
definitions), ``raw_fields``, ``derived_fields``, ``operators``, ``meta`` (build
information), ``factor_values`` and ``factor_stats``.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .fields import DERIVED_FIELDS, RAW_FIELDS
from .operators import OPERATOR_DOCS

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS raw_fields (
    name TEXT PRIMARY KEY, grp TEXT, source TEXT, description TEXT);
CREATE TABLE IF NOT EXISTS derived_fields (
    name TEXT PRIMARY KEY, expression TEXT, grp TEXT, description TEXT, stationary INTEGER);
CREATE TABLE IF NOT EXISTS operators (
    name TEXT PRIMARY KEY, kind TEXT, description TEXT);
CREATE TABLE IF NOT EXISTS factors (
    factor_id      TEXT PRIMARY KEY,
    name           TEXT UNIQUE NOT NULL,
    family         TEXT NOT NULL,
    category       TEXT NOT NULL,
    expression     TEXT NOT NULL,
    description    TEXT,
    window         INTEGER,
    lookback       INTEGER,
    raw_inputs     TEXT,      -- comma separated raw fields
    needs_derivs   INTEGER,   -- 1 if perp/derivatives data is required
    params         TEXT,      -- JSON
    expr_hash      TEXT,
    coverage       REAL,      -- share of non-NaN values in validation data
    cs_dispersion  REAL,      -- mean cross-sectional std of ranks (sanity)
    rank_autocorr  REAL       -- mean 1-bar autocorrelation of CS ranks (1 - turnover)
);
CREATE INDEX IF NOT EXISTS ix_factors_family ON factors(family);
CREATE INDEX IF NOT EXISTS ix_factors_category ON factors(category);
CREATE TABLE IF NOT EXISTS factor_values (
    factor_id TEXT NOT NULL, ts TEXT NOT NULL, symbol TEXT NOT NULL, value REAL,
    PRIMARY KEY (factor_id, ts, symbol));
CREATE TABLE IF NOT EXISTS factor_stats (
    factor_id TEXT NOT NULL, dataset TEXT NOT NULL, horizon INTEGER NOT NULL,
    start TEXT, end TEXT, n_periods INTEGER,
    ic_mean REAL, ic_std REAL, icir REAL, ic_tstat REAL, ic_hit_rate REAL,
    ls_mean REAL, ls_sharpe REAL, turnover REAL, coverage REAL,
    computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (factor_id, dataset, horizon));
"""

FACTOR_COLUMNS = [
    "factor_id",
    "name",
    "family",
    "category",
    "expression",
    "description",
    "window",
    "lookback",
    "raw_inputs",
    "needs_derivs",
    "params",
    "expr_hash",
    "coverage",
    "cs_dispersion",
    "rank_autocorr",
]
STATS_COLUMNS = [
    "factor_id",
    "dataset",
    "horizon",
    "start",
    "end",
    "n_periods",
    "ic_mean",
    "ic_std",
    "icir",
    "ic_tstat",
    "ic_hit_rate",
    "ls_mean",
    "ls_sharpe",
    "turnover",
    "coverage",
]


class FactorDB:
    """Access to one factor database.

    ``readonly=True`` opens the file with SQLite's ``mode=ro`` URI: every write
    is refused by SQLite itself, no schema script runs, and a missing file is an
    error instead of a new empty database. The browsing commands of the CLI use
    it, so they never change ``catalog/factors.db``.
    """

    def __init__(self, path: str | Path, readonly: bool = False):
        """Open the database file ``path``.

        Without ``readonly``, the file is made when it does not exist, and the schema
        script adds the tables that are missing. With ``readonly``, a missing file gives
        FileNotFoundError.
        """
        self.path = Path(path)
        self.readonly = readonly
        if readonly:
            if not self.path.is_file():
                raise FileNotFoundError(f"factor database not found: {self.path}")
            self.conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            self.conn = sqlite3.connect(self.path)
            self.conn.executescript(SCHEMA)

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        """Return the database for a ``with`` block."""
        return self

    def __exit__(self, *exc):
        """Close the connection at the end of a ``with`` block."""
        self.close()

    # -- catalog ---------------------------------------------------------------
    def write_catalog(self) -> None:
        """Replace the tables of raw fields, derived fields and operators with the package's definitions."""
        c = self.conn
        c.execute("DELETE FROM raw_fields")
        c.execute("DELETE FROM derived_fields")
        c.execute("DELETE FROM operators")
        c.executemany(
            "INSERT INTO raw_fields VALUES (?,?,?,?)", [(f.name, f.group, f.source, f.description) for f in RAW_FIELDS]
        )
        c.executemany(
            "INSERT INTO derived_fields VALUES (?,?,?,?,?)",
            [(d.name, d.expression, d.group, d.description, int(d.stationary)) for d in DERIVED_FIELDS],
        )
        ops = [(k, kind, doc) for k, (kind, doc) in sorted(OPERATOR_DOCS.items())]
        ops.append(("bench", "cross_section", "Benchmark (BTC) series broadcast to all symbols"))
        c.executemany("INSERT INTO operators VALUES (?,?,?)", ops)
        c.commit()

    def set_meta(self, **kv) -> None:
        """Store each keyword argument as a JSON value in the meta table (an existing key gets the new value)."""
        self.conn.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)", [(k, json.dumps(v)) for k, v in kv.items()])
        self.conn.commit()

    def replace_meta(self, **kv) -> None:
        """Remove every row of the meta table, then store each keyword argument as a JSON value."""
        self.conn.execute("DELETE FROM meta")
        self.set_meta(**kv)

    def meta(self) -> dict:
        """Return the meta table as a dict of decoded JSON values."""
        return {k: json.loads(v) for k, v in self.conn.execute("SELECT key, value FROM meta")}

    def replace_factors(self, rows: Iterable[dict]) -> int:
        """Replace the factor table with ``rows`` (dicts with the keys of FACTOR_COLUMNS). Return the row count."""
        self.conn.execute("DELETE FROM factors")
        rows = list(rows)
        self.conn.executemany(
            f"INSERT INTO factors ({','.join(FACTOR_COLUMNS)}) VALUES ({','.join('?' * len(FACTOR_COLUMNS))})",  # nosec B608  # constant column names; the values are parameters
            [tuple(r.get(c) for c in FACTOR_COLUMNS) for r in rows],
        )
        self.conn.commit()
        return len(rows)

    # -- queries ---------------------------------------------------------------
    def factors(
        self,
        family: str | None = None,
        category: str | None = None,
        available_fields: Iterable[str] | None = None,
        max_lookback: int | None = None,
        like: str | None = None,
    ) -> pd.DataFrame:
        """Return the factors that agree with all given filters, in factor_id order.

        ``family`` and ``category`` must be equal, ``like`` is a text in the name or the
        description (SQL LIKE), ``max_lookback`` is the largest lookback in bars, and
        ``available_fields`` keeps only factors whose raw inputs are all in it.
        """
        sql = "SELECT * FROM factors WHERE 1=1"
        args: list[str | int] = []
        if family:
            sql += " AND family = ?"
            args.append(family)
        if category:
            sql += " AND category = ?"
            args.append(category)
        if max_lookback:
            sql += " AND lookback <= ?"
            args.append(max_lookback)
        if like:
            sql += " AND (name LIKE ? OR description LIKE ?)"
            args += [f"%{like}%", f"%{like}%"]
        df = pd.read_sql_query(sql + " ORDER BY factor_id", self.conn, params=args)
        if available_fields is not None:
            have = set(available_fields)
            df = df[df["raw_inputs"].map(lambda s: set(s.split(",")) <= have)]
        return df.reset_index(drop=True)

    def get(self, factor: str) -> dict:
        """Return one factor row by factor_id or by name. An unknown factor gives KeyError."""
        cur = self.conn.execute("SELECT * FROM factors WHERE factor_id = ? OR name = ?", (factor, factor))
        row = cur.fetchone()
        if row is None:
            raise KeyError(factor)
        return dict(zip([d[0] for d in cur.description], row))

    def count(self) -> int:
        """Return the number of factors."""
        return self.conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]

    def summary(self) -> pd.DataFrame:
        """Return the number of factors for each family and category (columns family, category, n)."""
        return pd.read_sql_query(
            "SELECT family, category, COUNT(*) AS n FROM factors GROUP BY family, category ORDER BY family, n DESC",
            self.conn,
        )

    # -- values ---------------------------------------------------------------
    def write_values(self, factor_id: str, values: pd.DataFrame) -> int:
        """Replace the stored values of one factor with ``values`` (time x symbol; NaN is not stored).

        Return the number of stored values.
        """
        long = values.stack().dropna()
        if long.empty:
            return 0
        ts = long.index.get_level_values(0)
        ts = ts.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(ts, "strftime") else ts.astype(str)
        rows = list(
            zip([factor_id] * len(long), ts, long.index.get_level_values(1).astype(str), long.to_numpy(dtype=float))
        )
        self.conn.execute("DELETE FROM factor_values WHERE factor_id = ?", (factor_id,))
        self.conn.executemany("INSERT INTO factor_values VALUES (?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def read_values(self, factor_id: str) -> pd.DataFrame:
        """Return the stored values of one factor as a panel (time x symbol), or an empty frame."""
        df = pd.read_sql_query(
            "SELECT ts, symbol, value FROM factor_values WHERE factor_id = ?", self.conn, params=[factor_id]
        )
        if df.empty:
            return df
        out = df.pivot(index="ts", columns="symbol", values="value")
        out.index = pd.to_datetime(out.index)
        return out

    # -- stats ----------------------------------------------------------------
    def write_stats(self, rows: Iterable[dict]) -> None:
        """Store evaluation statistics (dicts with the keys of STATS_COLUMNS); a row with the same key replaces the old one."""
        self.conn.executemany(
            f"INSERT OR REPLACE INTO factor_stats ({','.join(STATS_COLUMNS)}) VALUES ({','.join('?' * len(STATS_COLUMNS))})",
            [tuple(_py(r.get(c)) for c in STATS_COLUMNS) for r in rows],
        )
        self.conn.commit()

    def stats(self, dataset: str | None = None) -> pd.DataFrame:
        """Return the stored statistics with the name, family and category of each factor (one dataset, or all)."""
        sql = "SELECT s.*, f.name, f.family, f.category FROM factor_stats s JOIN factors f USING(factor_id)"
        args = []
        if dataset:
            sql += " WHERE dataset = ?"
            args.append(dataset)
        return pd.read_sql_query(sql, self.conn, params=args)


def _py(v):
    """Return ``v`` as a Python number for SQLite; a value that is not finite gives None."""
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v
