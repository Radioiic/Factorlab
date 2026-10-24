"""Ad-hoc SQL over the data folder. DuckDB reads the parquet files in
place — there is no server, no import step and no copy to update.

Usage:
    python db.py "SELECT symbol, avg(fwd_ret_7) AS mean_fwd_ret_7
                  FROM labels GROUP BY 1 ORDER BY 2 DESC"
    python db.py --help

    # or interactively:
    python -i db.py
    >>> q("SELECT count(*) FROM labels")

Registered views (a view exists only when its files exist):
    labels      — the label table (labels.py; long format: date, symbol, fwd_ret_1, fwd_ret_7)
    ohlcv       — all raw daily bars, with a `symbol` column derived from filename
    funding     — all raw funding prints, likewise
    funding_fb  — the fallback-venue funding prints (deep history), likewise
"""

import argparse
from pathlib import Path

import duckdb

import config as C

_con = None


def _sql_text(path: Path | str) -> str:
    """``path`` as an SQL string literal: in quotes, with each quote in the path doubled."""
    return "'" + str(path).replace("'", "''") + "'"


# The SELECT of each view. The only value in the SQL text is a configuration path, quoted by
# _sql_text.
_TABLE_SELECT = "SELECT * FROM read_parquet({})"
_FILES_SELECT = "SELECT parse_filename(filename, true) AS symbol, * FROM read_parquet({}, filename=true)"


def _views() -> list[tuple[str, str]]:
    """(view name, SELECT) for each view whose files exist. DuckDB checks the files of a view
    when it makes the view, so a view without files would stop every query."""
    out = []
    labels = C.LABELS / "labels.parquet"
    if labels.is_file():
        out.append(("labels", _TABLE_SELECT.format(_sql_text(labels))))
    for name, folder in (("ohlcv", C.RAW_OHLCV), ("funding", C.RAW_FUNDING), ("funding_fb", C.RAW_FUNDING_FB)):
        if any(folder.glob("*.parquet")):
            out.append((name, _FILES_SELECT.format(_sql_text(folder / "*.parquet"))))
    return out


def _connect() -> "duckdb.DuckDBPyConnection":
    """The shared in-memory connection, created with its views on first use.

    Lazy: an import of this module has no side effects. A view whose files are absent is not
    made (for example labels before labels.py runs); the other views work.
    """
    global _con
    if _con is None:
        con = duckdb.connect()
        # Timestamps are stored as UTC instants; without this DuckDB prints them in
        # the machine's local zone, which shifts calendar days west of UTC.
        con.execute("SET TimeZone = 'UTC'")
        for name, select in _views():
            con.execute(f"CREATE VIEW {name} AS {select}")
        _con = con
    return _con


def __getattr__(name: str):
    """Keep ``db.con`` working for callers that use the connection directly."""
    if name == "con":
        return _connect()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def q(sql: str):
    """Run SQL, return a pandas DataFrame."""
    return _connect().execute(sql).df()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="db.py",
        description="Run SQL over the data folder (DuckDB, in memory). The views are labels, ohlcv, "
        "funding and funding_fb (the fallback funding prints). A view exists only when its files exist. "
        "Without SQL, the program only makes the views: use `python -i db.py` and then q(\"SELECT ...\").",
    )
    # REMAINDER keeps words that start with "-" (for example SELECT -fwd_ret_7 FROM labels)
    p.add_argument("sql", nargs=argparse.REMAINDER, help="the SQL statement; the words are joined with spaces")
    return p


if __name__ == "__main__":
    args = _parser().parse_args()
    con = _connect()  # bound eagerly for `python -i db.py`
    if args.sql:
        print(q(" ".join(args.sql)).to_string(index=False))
