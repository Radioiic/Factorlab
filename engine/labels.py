"""The label table: the forward returns fwd_ret_1 and fwd_ret_7 of each coin and day.

Usage (from engine/):
    python -B labels.py             # build data/labels/labels.parquet from the OKX daily bars
    python -B labels.py --status    # show the label table and compare it with the bars; write nothing

THE LABELS. The row of date D and one coin holds:
    fwd_ret_1    close(D + 1) / close(D) - 1: the return of the bar after D
    fwd_ret_7    close(D + 7) / close(D) - 1: the return of the seven bars after D, the span
                 that a book earns from a Monday trade on the values of Sunday D
A label is a TARGET: it uses bars after D. It is never an input of a factor value of D. A
label is NaN until its last bar is on disk (the last row has no fwd_ret_1, the last seven rows
have no fwd_ret_7).

THE ROWS. The table reads only the OKX perpetual closes (data/raw/ohlcv). A coin with fewer
than MIN_BARS bars has no row. A row exists where close.pct_change(WARMUP_ROWS) is formed: the
bar of D and the bar WARMUP_ROWS rows before it have a close. The shifts are row shifts on the
coin's own bars: when a bar is missing, a label spans more calendar days. eligible() gives the
same rows as a wide True/False frame; the books trade only the eligible coins.

Each run builds the table again from the bars. Columns: date (UTC bar-open time), symbol (the
file name of the coin, for example BTC-USDT), fwd_ret_1, fwd_ret_7. The rows are sorted by date,
then by symbol.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

import config as C
import data_io

LABEL_COLUMNS = ("fwd_ret_1", "fwd_ret_7")
LABEL = "fwd_ret_7"  # the label of every weekly statistic
HORIZON_DAYS = 7
MIN_BARS = 91  # a coin with fewer bars in total has no row
WARMUP_ROWS = 90  # a row exists when close.pct_change(90) is formed (a ROW shift)
FILE_NAME = "labels.parquet"
COLUMNS = ("date", "symbol", *LABEL_COLUMNS)


def path() -> Path:
    """data/labels/labels.parquet (config.LABELS)."""
    return Path(C.LABELS) / FILE_NAME


def _keep(close: pd.Series) -> pd.Series:
    """True on the rows that have a label row: close.pct_change(WARMUP_ROWS) is formed."""
    return close.pct_change(WARMUP_ROWS).notna()


def symbol_rows(sym_file: str) -> pd.DataFrame | None:
    """The label rows of one coin (index ts, columns fwd_ret_1, fwd_ret_7), or None when the coin
    has no bar file or fewer than MIN_BARS bars."""
    px = data_io.ohlcv(sym_file)
    if px is None or len(px) < MIN_BARS:
        return None
    close = px["close"]
    out = pd.DataFrame(index=px.index)
    r = close.pct_change()
    out["fwd_ret_1"] = r.shift(-1)
    out["fwd_ret_7"] = close.shift(-HORIZON_DAYS) / close - 1
    return out[_keep(close)]


def compute() -> pd.DataFrame:
    """The label table in memory. Pure: it reads data/raw/ohlcv and writes nothing."""
    frames = []
    for symbol in C.UNIVERSE:
        sym_file = C.sym_to_file(symbol)
        f = symbol_rows(sym_file)
        if f is None:
            continue
        f = f.reset_index().rename(columns={"ts": "date"})
        f.insert(1, "symbol", sym_file)
        frames.append(f)
    if not frames:
        raise SystemExit("No usable symbols. Run ingest.py first.")
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])
    return df.reset_index(drop=True)[list(COLUMNS)]


def build(out_path: Path | None = None) -> pd.DataFrame:
    """Compute the label table and write it (default data/labels/labels.parquet).

    The target is checked before the computation (config.guard_write). The table goes to a
    temporary file first, and a rename then replaces the target."""
    target = path() if out_path is None else Path(out_path)
    C.guard_write(target)
    df = compute()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    C.guard_write(tmp)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, target)
    print(
        f"Wrote {len(df):,} label rows of {df['symbol'].nunique()} symbols "
        f"({df['date'].min().date()} to {df['date'].max().date()}) to {target.name}"
    )
    return df


def eligible(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """A wide True/False frame (date x the 20 engine symbols, in config.UNIVERSE order): True
    exactly where compute() has a row. It is computed from the bars, not from the file.

    index: the dates of the frame (default: every bar date, data_io.close_matrix().index)."""
    if index is None:
        index = pd.DatetimeIndex(data_io.close_matrix().index)
    cols = [C.sym_to_file(s) for s in C.UNIVERSE]
    out = pd.DataFrame(False, index=index, columns=cols)
    for sym in cols:
        px = data_io.ohlcv(sym)
        if px is None or len(px) < MIN_BARS:
            continue
        keep = _keep(px["close"])
        rows = keep.index[keep.to_numpy()]
        out[sym] = out.index.isin(rows)
    return out


def status() -> int:
    """Print the label table and whether it equals a new computation from the bars (read-only)."""
    p = path()
    if not p.is_file():
        print(f"No label table ({p.name} is not in data/labels). Run labels.py to build it.")
        return 0
    df = pd.read_parquet(p)
    print(f"Label table: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    if len(df):
        print(f"  dates {df['date'].min().date()} to {df['date'].max().date()}")
        for col in LABEL_COLUMNS:
            print(f"  {col}: {int(df[col].notna().sum()):,} values")
    try:
        new = compute()
    except SystemExit as exc:
        print(f"  The bars cannot be read: {exc}")
        return 0
    same = list(df.columns) == list(new.columns) and df.equals(new)
    print(
        "  The table is equal to a new computation from the bars on disk."
        if same
        else "  The table is not equal to a new computation from the bars on disk. Run labels.py."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the label table data/labels/labels.parquet (fwd_ret_1, fwd_ret_7) from the OKX daily bars."
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="show the label table and compare it with a new computation; write nothing",
    )
    a = ap.parse_args(argv)
    if a.status:
        return status()
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
