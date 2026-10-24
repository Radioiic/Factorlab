"""The one place that reads the parquet lake. labels.py, backtest.py and the other
engine modules read the raw data from here and nowhere else.

One loader for all modules keeps the daily-funding resample, a POINT-IN-TIME
convention, in one place. The convention:

    Funding prints (8-hourly) are summed per UTC day with label='left',
    closed='left': the sum of everything paid during bar D is stamped on
    D's OPEN, matching the OHLCV bars. So funding[D] is fully known at
    D+1 00:00 UTC, exactly like every other column of bar D.

Funding depth: OKX's public API serves only ~3 months of prints, so
daily_funding() splices two sources — OKX (the PRIMARY venue) and the
fallback venue (config.FUNDING_FALLBACK_EXCHANGE_ID, Binance USDT-M) for the
deep past. Fallback rates are a close proxy, not the same number OKX
charged; fine for research, worth remembering when a carry backtest looks
too clean.

THE SPLICE POLICY (settlement level):

    1. Each settlement time is rounded to the minute (the fallback venue
       stamps settlements a few ms off the hour) and assigned to the 8-hour
       funding slot it closes: (k*8h - 8h, k*8h] on the UTC 00/08/16 grid,
       i.e. slot = ceil(settlement, 8h).
    2. A slot belongs to the primary venue if the primary has ANY settlement
       in it; otherwise every fallback settlement in that slot is used.
    3. The kept settlements are summed per UTC day by their settlement time,
       with the same label='left', closed='left' convention as above.

Why slots and not days: a per-DAY splice (daily sums, then
primary.combine_first(fallback)) lets one primary settlement hide the
fallback for the whole day. A primary file that starts mid-day would then
lose the fallback's earlier settlements on that day. Why not a plain
timestamp union: a venue on a shorter interval (the fallback settled SOL
every 2 hours from 2022-11-09 to 2022-11-18) would then be summed ON TOP of
the primary's 8-hour rate for the same hours. One venue per slot means no
period is paid twice. A slot the primary only partly covers keeps the
primary's prints alone (under-count, not double count). Days covered by one
venue only are the same under this policy.

The midnight boundary can REVISE a day: the slot (D 16:00, D+1 00:00]
closes at D+1 00:00, so when the primary owns it, its funding is booked on
day D+1 (the primary's settlement time). A fallback print
inside that slot but on day D -- e.g. a 20:00 print from a venue settling
every 4 hours -- is then dropped from day D, not moved: the primary's 00:00
print pays that whole period, on D+1. Until the primary's D+1 00:00 print is
on disk the slot is still the fallback's, so day D's sum includes the 20:00
print, and ingesting that 00:00 print later revises day D downward. This is
not lookahead (the 00:00 settlement is known at D's close, 00:00 UTC), but it
is a revision of an already-written day. It can only happen where a fallback
print falls inside a primary-owned slot on the slot's previous day, which
needs a fallback settling off the 00/08/16 grid.
test_engine_rules.py tests both states.

Provenance is kept, not inferred: funding_settlements() returns the kept
settlements with their venue, and daily_funding_detail() adds per-day
n_primary / n_fallback counts and a source label (primary / fallback /
mixed / none). "fallback" and "mixed" days carry proxy rates.

All loaders return None (or skip the symbol) when the underlying file is
missing — callers decide whether that's fatal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

# --- raw, per symbol ----------------------------------------------------------


def ohlcv(sym: str) -> pd.DataFrame | None:
    """Daily bars for one symbol file ('BTC-USDT'), indexed by ts (UTC)."""
    p = C.RAW_OHLCV / f"{sym}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p).set_index("ts").sort_index()


def spot_ohlcv(sym: str) -> pd.DataFrame | None:
    """Same-venue spot daily bars for basis measurement."""
    p = C.RAW_SPOT_OHLCV / f"{sym}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p).set_index("ts").sort_index()


FUNDING_SLOT = "8h"  # the settlement grid both venues use (UTC 00/08/16)
SETTLEMENT_ROUND = "min"  # absorbs the fallback venue's few-ms timestamp jitter
FUNDING_SOURCES = ("primary", "fallback", "mixed", "none")


def _daily_funding_from(path) -> pd.Series | None:
    """One venue's raw daily sums (no splice). Kept for per-venue coverage
    counts; daily_funding() is the series every consumer should use."""
    if not path.exists():
        return None
    fr = pd.read_parquet(path).set_index("ts").sort_index()["funding_rate"]
    return fr.resample("1D", label="left", closed="left").sum(min_count=1)


def _settlements_from(path) -> pd.DataFrame | None:
    """One venue's settlements: ts (as stored), settle (rounded to the
    minute), slot (the 8h slot it closes), funding_rate. NaN rates are not
    settlements and are dropped, so they can never claim a slot."""
    if not path.exists():
        return None
    fr = pd.read_parquet(path, columns=["ts", "funding_rate"])
    fr = fr.dropna(subset=["funding_rate"]).sort_values("ts")
    fr["settle"] = fr["ts"].dt.round(SETTLEMENT_ROUND)
    # two stored prints for one settlement (only possible via jitter): keep
    # the newest, as ingest's upsert does for an exact duplicate
    fr = fr.drop_duplicates(subset="settle", keep="last")
    fr["slot"] = fr["settle"].dt.ceil(FUNDING_SLOT)
    return fr.reset_index(drop=True)


def funding_settlements(sym: str) -> pd.DataFrame | None:
    """The spliced settlement list for one symbol, with provenance.

    Columns: ts, settle, slot, venue ('primary' | 'fallback'), funding_rate.
    Primary settlements are all kept; a fallback settlement is kept only if
    the primary has no settlement in its slot (see the module docstring).
    None when neither venue has a file.
    """
    primary = _settlements_from(C.RAW_FUNDING / f"{sym}.parquet")
    fallback = _settlements_from(C.RAW_FUNDING_FB / f"{sym}.parquet")
    if primary is None and fallback is None:
        return None
    parts = []
    if primary is not None:
        parts.append(primary.assign(venue="primary"))
    if fallback is not None:
        owned = primary["slot"] if primary is not None else pd.Series([], dtype=object)
        parts.append(fallback[~fallback["slot"].isin(owned)].assign(venue="fallback"))
    out = pd.concat(parts, ignore_index=True).sort_values(["settle", "venue"])
    return out[["ts", "settle", "slot", "venue", "funding_rate"]].reset_index(drop=True)


def daily_funding_detail(sym: str) -> pd.DataFrame | None:
    """Daily funding sums plus provenance, indexed by day (UTC, bar-open stamp).

    Columns: funding_rate (NaN on days with no settlement), n_primary,
    n_fallback (settlements used that day) and source, one of
    FUNDING_SOURCES. Days between the first and last settlement with none
    are present with source 'none', exactly as the daily resample has them.
    """
    s = funding_settlements(sym)
    if s is None:
        return None
    x = s.set_index("settle")
    x = x.assign(
        n_primary=(x["venue"] == "primary").astype("int64"), n_fallback=(x["venue"] == "fallback").astype("int64")
    )
    g = x.resample("1D", label="left", closed="left")
    out = pd.DataFrame(
        {
            "funding_rate": g["funding_rate"].sum(min_count=1),
            "n_primary": g["n_primary"].sum().astype("int64"),
            "n_fallback": g["n_fallback"].sum().astype("int64"),
        }
    )
    has_p, has_f = out["n_primary"] > 0, out["n_fallback"] > 0
    out["source"] = np.select([has_p & has_f, has_p, has_f], ["mixed", "primary", "fallback"], default="none")
    out.index.name = "ts"
    return out


def daily_funding(sym: str) -> pd.Series | None:
    """Daily funding sums for one symbol, spliced per 8h settlement slot:
    the primary venue where it settled, the fallback venue for slots the
    primary lacks (the deep past, and the primary's partial first day).
    NaN on days neither venue covers. Provenance: daily_funding_detail()."""
    detail = daily_funding_detail(sym)
    if detail is None:
        return None
    out = detail["funding_rate"]
    out.name = "funding_rate"
    return out


def daily_oi(sym: str) -> pd.Series | None:
    """Daily open-interest value snapshots (last print of the day)."""
    p = C.RAW_OI / f"{sym}.parquet"
    if not p.exists():
        return None
    oi = pd.read_parquet(p).set_index("ts").sort_index()["oi_value"]
    return oi.resample("1D", label="left", closed="left").last()


# --- wide matrices for the backtester -----------------------------------------


def close_matrix() -> pd.DataFrame:
    """Wide close-price frame: index=date (UTC), one column per universe symbol."""
    cols = {}
    for symbol in C.UNIVERSE:
        sym = C.sym_to_file(symbol)
        px = ohlcv(sym)
        if px is not None:
            cols[sym] = px["close"]
    if not cols:
        raise SystemExit("No raw OHLCV data. Run ingest.py first.")
    return pd.DataFrame(cols)


def observed_funding_matrix(index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """Daily funding sums aligned to prices; uncovered observations stay NaN."""
    out = pd.DataFrame(index=index, columns=columns, dtype=float)
    for sym in columns:
        f = daily_funding(sym)
        if f is not None:
            out[sym] = f.reindex(index)
    return out


def funding_matrix(index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """Daily funding sums aligned to a price frame; uncovered days -> 0.

    Zero (not NaN) because the caller is accruing pnl: no data means we model
    no funding flow, and the settlement-slot splice in daily_funding() keeps
    that regime as small as the venues allow.
    """
    return observed_funding_matrix(index, columns).fillna(0.0)


# --- derived ------------------------------------------------------------------


def label_table(columns=None, dates=None) -> pd.DataFrame:
    """The label table data/labels/labels.parquet (labels.py): date, symbol, fwd_ret_1, fwd_ret_7.

    columns: the columns to read (default: all). dates: read only the rows of these dates (UTC
    days). Raises SystemExit("No label table. ...") when the file is absent."""
    import labels  # labels.py imports this module: import it when the function runs

    p = labels.path()
    if not p.exists():
        raise SystemExit("No label table. Run labels.py first.")
    filters = None
    if dates is not None:
        days = pd.DatetimeIndex(pd.to_datetime(list(dates), utc=True)).normalize().as_unit("ms")
        filters = [("date", "in", list(days))]
    return pd.read_parquet(p, columns=None if columns is None else list(columns), filters=filters)
