"""Incremental ingestion of daily OHLCV, funding rates, and open interest.

Usage:
    python ingest.py --check    # print row counts / date ranges, fetch nothing
    FACTORLAB_OFFLINE=0 python ingest.py   # pull everything new since last run

OFFLINE BY DEFAULT: unless FACTORLAB_OFFLINE=0 is set, every mode except
--check exits before any network request (this is the only module in the
engine that talks to an exchange). The value is parsed strictly by
config.offline_from: unset or 1 = offline, exactly 0 = fetch, anything else
("", "true", "yes", ...) = refuse to run.

Properties that matter more than speed here:
- IDEMPOTENT: safe to re-run any time. Each run re-fetches a small overlap
  window, then de-duplicates on (timestamp) before writing. Running it twice
  produces identical files.
- APPEND-ONLY in spirit: we never delete history. If the exchange restates a
  bar inside the overlap window, newest fetch wins for that timestamp.
- DROPS THE LIVE BAR: the most recent, still-forming daily candle is excluded.
  A half-formed bar in your factor file is lookahead bias waiting to happen.
- POLITE: respects ccxt's built-in rate limiter. A full backfill of 20 symbols
  takes a few minutes; a later run that adds a few days takes seconds.

Schemas (all timestamps UTC, ms since epoch in raw, stored as timestamp):
  ohlcv:          ts, open, high, low, close, volume
  funding:        ts, funding_rate                (typically every 8h)
  open_interest:  ts, oi_contracts, oi_value      (daily snapshot history)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime

import ccxt
import pandas as pd

import config as C

OVERLAP_MS = 3 * 24 * 60 * 60 * 1000  # re-fetch trailing 3 days for restatements
DAY_MS = 24 * 60 * 60 * 1000


# --- generic parquet upsert ---------------------------------------------------


def _load(path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def _upsert(path, new: pd.DataFrame) -> int:
    """Merge new rows into parquet on 'ts'; newest data wins. Returns rows added."""
    if new.empty:
        return 0
    new = new.sort_values("ts")
    old = _load(path)
    before = 0 if old is None else len(old)
    if old is not None:
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset="ts", keep="last").sort_values("ts")
    else:
        merged = new.drop_duplicates(subset="ts", keep="last")
    C.guard_write(path)  # refused while the data folder is protected
    merged.to_parquet(path, index=False)
    return len(merged) - before


def _since_for(path) -> int:
    """Resume point: last stored ts minus overlap, or the configured backfill start."""
    old = _load(path)
    if old is None or old.empty:
        return int(pd.Timestamp(C.BACKFILL_START).timestamp() * 1000)
    return int(old["ts"].max().timestamp() * 1000) - OVERLAP_MS


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


# --- fetchers -----------------------------------------------------------------


def fetch_ohlcv(ex, symbol: str, folder=None) -> int:
    """Fetch completed daily bars into the perp or an explicit OHLCV folder."""
    folder = C.RAW_OHLCV if folder is None else folder
    path = folder / f"{C.sym_to_file(symbol)}.parquet"
    since = _since_for(path)
    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe="1d", since=since, limit=100)
        if not batch:
            # OKX (and others) return an EMPTY page when the requested window
            # predates the instrument's listing, not the earliest bars. With
            # no data found, jump forward to hunt for the listing
            # date instead of giving up (bounded: <20 extra requests to scan
            # from 2022 to today). Once we do have rows, empty means done.
            if not rows and since < _now_ms() - DAY_MS:
                since += 100 * DAY_MS
                continue
            break
        rows.extend(batch)
        last = batch[-1][0]
        if last <= since:  # no progress -> stop
            break
        since = last + 1
        if last >= _now_ms() - DAY_MS:
            break
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    # Drop the still-forming bar: keep only bars whose close time is in the past.
    cutoff = pd.Timestamp.now(tz="UTC").floor("D")  # today's 00:00 UTC open
    df = df[df["ts"] < cutoff]
    return _upsert(path, df)


def fetch_funding(ex, symbol: str, folder=None, limit: int = 100) -> int:
    """Funding prints into <folder>/<sym>.parquet (default: the primary venue
    folder). Also used for the fallback venue, which allows bigger pages."""
    folder = C.RAW_FUNDING if folder is None else folder
    path = folder / f"{C.sym_to_file(symbol)}.parquet"
    since = _since_for(path)
    rows = []
    while True:
        batch = ex.fetch_funding_rate_history(symbol, since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1]["timestamp"]
        if last <= since:
            break
        since = last + 1
        if last >= _now_ms():
            break
    if not rows:
        return 0
    df = pd.DataFrame({"ts": [r["timestamp"] for r in rows], "funding_rate": [r["fundingRate"] for r in rows]})
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return _upsert(path, df)


def fetch_open_interest(ex, symbol: str) -> int:
    """OI history, best-effort. Factors degrade gracefully if missing.

    OKX's normal fetchOpenInterestHistory rejects daily queries, but the
    "rubik" statistics API serves ~6 months of daily OI (USD, aggregated
    across ALL of the ccy's contracts — a broader proxy than our specific
    USDT perp, fine for a 7d-change factor). ~6 months is a serving cap,
    not a storage cap: the upsert appends, so local history grows one day
    per day from first ingest. Other venues fall back to the generic ccxt
    endpoint if they support it.
    """
    path = C.RAW_OI / f"{C.sym_to_file(symbol)}.parquet"
    if C.EXCHANGE_ID == "okx":
        ccy = symbol.split("/")[0]
        try:
            r = ex.publicGetRubikStatContractsOpenInterestVolume({"ccy": ccy, "period": "1D"})
        except ccxt.BaseError as e:
            print(f"    OI unavailable for {symbol}: {type(e).__name__}", file=sys.stderr)
            return 0
        rows = r.get("data", [])
        if not rows:
            return 0
        df = pd.DataFrame(
            {
                "ts": [int(x[0]) for x in rows],
                "oi_contracts": [None] * len(rows),  # rubik is USD-only
                "oi_value": [float(x[1]) for x in rows],
            }
        )
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        # drop the still-forming day, same convention as OHLCV bars
        df = df[df["ts"] < pd.Timestamp.now(tz="UTC").floor("D")]
        return _upsert(path, df)

    if not ex.has.get("fetchOpenInterestHistory"):
        return 0
    since = _since_for(path)
    rows = []
    try:
        while True:
            batch = ex.fetch_open_interest_history(symbol, timeframe="1d", since=since, limit=100)
            if not batch:
                break
            rows.extend(batch)
            last = batch[-1]["timestamp"]
            if last <= since:
                break
            since = last + 1
            if last >= _now_ms() - DAY_MS:
                break
    except ccxt.BaseError as e:
        print(f"    OI unavailable for {symbol}: {type(e).__name__}", file=sys.stderr)
        return 0
    if not rows:
        return 0
    df = pd.DataFrame(
        {
            "ts": [r["timestamp"] for r in rows],
            "oi_contracts": [r.get("openInterestAmount") for r in rows],
            "oi_value": [r.get("openInterestValue") for r in rows],
        }
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return _upsert(path, df)


# --- status -------------------------------------------------------------------


def check():
    for name, folder in [
        ("ohlcv", C.RAW_OHLCV),
        ("spot_ohlcv", C.RAW_SPOT_OHLCV),
        ("funding", C.RAW_FUNDING),
        (f"funding fallback ({C.FUNDING_FALLBACK_EXCHANGE_ID})", C.RAW_FUNDING_FB),
        ("open_interest", C.RAW_OI),
    ]:
        print(f"\n[{name}]")
        files = sorted(folder.glob("*.parquet"))
        if not files:
            print("  (empty)")
            continue
        for f in files:
            df = pd.read_parquet(f, columns=["ts"])
            print(f"  {f.stem:<12} {len(df):>6} rows   {df.ts.min().date()} -> {df.ts.max().date()}")


# --- main ---------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Download the new OKX daily bars (perp and spot), funding and open "
        "interest, and the Binance funding archive, into data/raw. Each mode except "
        "--check needs FACTORLAB_OFFLINE=0."
    )
    ap.add_argument("--check", action="store_true", help="show status, fetch nothing")
    ap.add_argument("--spot-only", action="store_true", help="fetch only same-venue spot reference bars")
    args = ap.parse_args()

    # Parsed on every run (even --check, which never fetches) by the strict
    # parser config.offline_from: only exactly "0" may reach the exchange code
    # below; an unexpected value is an error, never a silent "online".
    offline = C.offline_from(os.environ.get("FACTORLAB_OFFLINE", "1"))
    if args.check:
        check()
        return
    if offline:
        raise SystemExit(
            "FACTORLAB_OFFLINE=1 (the default): network ingest disabled; set FACTORLAB_OFFLINE=0 to fetch market data"
        )

    # Every folder this run can write is checked BEFORE an exchange client exists
    # or a request is sent. _upsert() checks each file again, but only after the
    # fetch. A protected data folder is refused here, before any network traffic.
    targets = (
        (C.RAW_SPOT_OHLCV,)
        if args.spot_only
        else (C.RAW_OHLCV, C.RAW_SPOT_OHLCV, C.RAW_FUNDING, C.RAW_OI, C.RAW_FUNDING_FB)
    )
    for folder in targets:
        C.guard_write(folder)

    ex = getattr(ccxt, C.EXCHANGE_ID)({"enableRateLimit": True})
    ex.load_markets()
    if args.spot_only:
        print(f"Ingesting same-venue spot references from {C.EXCHANGE_ID}")
        t0 = time.time()
        for perp_symbol in C.UNIVERSE:
            spot_symbol = C.perp_to_spot(perp_symbol)
            if spot_symbol not in ex.markets:
                print(f"  {spot_symbol:<18} NOT LISTED — skipped", file=sys.stderr)
                continue
            rows = fetch_ohlcv(ex, spot_symbol, folder=C.RAW_SPOT_OHLCV)
            print(f"  {spot_symbol:<18} +{rows} spot ohlcv")
        print(f"Done in {time.time() - t0:.0f}s.")
        return

    # Fallback venue exists ONLY because OKX serves ~3 months of funding
    # history. Data only: prices and OI stay on EXCHANGE_ID. It is optional:
    # where it cannot be reached (Binance refuses some regions), the run goes
    # on with the primary funding only (data_io splices whatever is on disk).
    ex_fb = getattr(ccxt, C.FUNDING_FALLBACK_EXCHANGE_ID)({"enableRateLimit": True})
    try:
        ex_fb.load_markets()
    except ccxt.BaseError as exc:
        print(
            f"  funding fallback {C.FUNDING_FALLBACK_EXCHANGE_ID} not reachable "
            f"({type(exc).__name__}): primary funding only",
            file=sys.stderr,
        )
        ex_fb = None
    print(
        f"Ingesting {len(C.UNIVERSE)} symbols from {C.EXCHANGE_ID} "
        f"(funding fallback: {C.FUNDING_FALLBACK_EXCHANGE_ID}; "
        f"UTC now: {datetime.now(UTC):%Y-%m-%d %H:%M})"
    )

    t0 = time.time()
    for symbol in C.UNIVERSE:
        if symbol not in ex.markets:
            print(f"  {symbol:<18} NOT LISTED on {C.EXCHANGE_ID} — skipped", file=sys.stderr)
            continue
        a = fetch_ohlcv(ex, symbol)
        spot_symbol = C.perp_to_spot(symbol)
        s = fetch_ohlcv(ex, spot_symbol, folder=C.RAW_SPOT_OHLCV) if spot_symbol in ex.markets else 0
        b = fetch_funding(ex, symbol)
        c = fetch_open_interest(ex, symbol)
        if ex_fb is None:
            d = 0
        elif symbol in ex_fb.markets:
            # Binance allows 1000-row pages; a full backfill is ~6 requests.
            d = fetch_funding(ex_fb, symbol, folder=C.RAW_FUNDING_FB, limit=1000)
        else:
            d = 0
            print(f"    no fallback funding: {symbol} not on {C.FUNDING_FALLBACK_EXCHANGE_ID}", file=sys.stderr)
        print(f"  {symbol:<18} +{a} ohlcv, +{s} spot, +{b} funding, +{d} funding_fb, +{c} oi")
    print(f"Done in {time.time() - t0:.0f}s. Then run 'python labels.py'.")


if __name__ == "__main__":
    main()
