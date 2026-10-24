"""Synthetic demo data: a market data set that the engine can build every output from.

Usage (from the repository root):
    engine/.venv/bin/python -B engine/demo_data.py --write     # write the demo data set
    engine/.venv/bin/python -B engine/demo_data.py --check     # can the demo write here? (writes nothing)
    engine/.venv/bin/python -B engine/demo_data.py --remove    # remove the demo data set
    engine/.venv/bin/python -B engine/update.py --demo         # write it, then build every output

Synthetic demo data. Not market data. A random generator makes every value; no value
comes from an exchange. The data set holds the 20 coins of config.UNIVERSE from 2022-01-01
to 2026-09-30 (1,734 daily bars), in the formats of the engine and the library:
    <data>/raw/ohlcv/<COIN>.parquet                   perpetual daily bars (ts, open, high, low, close, volume)
    <data>/raw/spot_ohlcv/<COIN>.parquet              spot daily bars, the same columns
    <data>/raw/funding/<COIN>.parquet                 primary funding prints (8-hourly), the last 92 days
    <data>/raw/funding_binanceusdm/<COIN>.parquet     fallback funding prints (8-hourly), every day
    <data>/raw/open_interest/<COIN>.parquet           open interest value, stamped 16:00 UTC, the last 180 days
    <data>/demo.json                                  the demo marker (MARKER)
    library/data/binance_um_1d/<field>.csv.gz         the Binance panel: 16 fields in the format of
                                                      library/scripts/merge_panels.py, and its own demo.json
Three coins start after the first day, as in the universe: OP on 2022-06-01, APT on 2022-10-19 and ARB on
2023-03-23. Before that day a coin has no row (no file row, an empty panel cell).

The generator is library/factorlab/data/synthetic.py (make_synthetic_panel): a common market
factor, stochastic volatility, volume that follows absolute returns, mean-reverting funding and
premium, and open interest. This module takes 20 coins of that panel and adds the spot bars, the
8-hour funding prints and the venue differences. It also sets a realistic volume scale and
plants two signals in the returns (market() gives the rules), so the library screen has
something to find. The same seed gives byte-identical files.

Rules:
- The demo writes only into an empty data folder (empty folders are allowed) or into a folder
  that holds the demo marker. A folder with any other file can hold market data: the demo
  refuses it. A protected folder (the file .read_only) is refused too.
- The demo panel goes into library/data/binance_um_1d only when that folder is absent, empty
  or a demo panel. A folder with other panel files holds Binance data: the demo refuses it.
  Use --no-panel (update.py --demo --skip-binance) to make no panel.
- --remove deletes the files of the data folder and of the panel folder, each only when its
  demo marker is there.
- update.py refuses to download market data into a demo folder (see update.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import config as C

LIBRARY = C.ROOT.parent / "library"
PANEL_DIR = LIBRARY / "data" / "binance_um_1d"
MARKER = "demo.json"
BANNER = "Synthetic demo data. Not market data."
SEED = 20220101
FIRST_DAY = "2022-01-01"
# 2022-01-01 to 2026-09-30: the library screen's fixed zones (G from 2022-04-03 to 2024-06-09)
# and its point-in-time cuts (to 2026-09-13) need this span
N_DAYS = 1734
# the first daily bar of the coins that start after FIRST_DAY (all other coins start on FIRST_DAY)
LISTINGS = {"OP/USDT:USDT": "2022-06-01", "APT/USDT:USDT": "2022-10-19", "ARB/USDT:USDT": "2023-03-23"}
PRIMARY_FUNDING_DAYS = 92  # the primary venue serves about 3 months of funding prints
OI_DAYS = 180  # the open-interest source serves about 6 months
OI_STAMP_HOUR = 16  # each open-interest print is stamped 16:00 UTC
SETTLEMENT_HOURS = (0, 8, 16)  # the 8-hour funding grid (UTC)
# the generator makes more coins than the universe; the demo takes the coins listed on the first day
POOL = 40
DOLLAR_VOLUME = 1e10  # the median daily dollar volume of the first two coins
DOLLAR_VOLUME_RANGE = (5e7, 3e9)  # the range of the median daily dollar volume of the other coins
# the planted signals: (field, days of the mean, daily return per unit of z-score)
SIGNALS = (("clv", 7, 0.0016), ("long_short_ratio", 7, 0.0022))
VWAP_MARGIN = 1e-6  # each average price stays this share of the bar's range inside [low, high]
FUNDING_DECIMALS = 8  # a funding print has 8 decimals, as the exchanges publish it
FORMAT = 1  # the version of the demo data layout (in the marker)
# the Binance panel fields (library/factorlab/fields.py RAW_FIELDS, the same 16 names)
PANEL_FIELDS = (
    "close",
    "funding_rate",
    "high",
    "long_short_ratio",
    "low",
    "open",
    "open_interest",
    "open_interest_value",
    "premium",
    "quote_volume",
    "taker_buy_quote_volume",
    "taker_buy_sell_ratio",
    "taker_buy_volume",
    "top_long_short_ratio",
    "trades",
    "volume",
)
RAW_FOLDERS = ("ohlcv", "spot_ohlcv", "funding", f"funding_{C.FUNDING_FALLBACK_EXCHANGE_ID}", "open_interest")


class DemoRefused(SystemExit):
    """The demo does not write into this folder (exit status 1 with the message)."""

    def __init__(self, message: str):
        super().__init__(message)


# --------------------------------------------------------------------------- folder checks


def is_demo(data: Path | None = None) -> bool:
    """True when the data folder holds the demo marker."""
    return (Path(C.DATA if data is None else data) / MARKER).is_file()


def is_demo_panel(panel: Path = PANEL_DIR) -> bool:
    """True when the panel folder holds the demo marker."""
    return (Path(panel) / MARKER).is_file()


def marker(data: Path | None = None) -> dict | None:
    """The demo marker of the data folder, or None when there is none (or it does not parse)."""
    p = Path(C.DATA if data is None else data) / MARKER
    try:
        out = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return out if isinstance(out, dict) else None


# files that a file manager writes by itself: they are not data
IGNORED_FILES = (".DS_Store",)


def _first_file(folder: Path) -> Path | None:
    """The first file found below `folder` (a symbolic link counts as a file), or None."""
    folder = Path(folder)
    if not folder.exists():
        return None
    for p in folder.rglob("*"):
        if (p.is_file() or p.is_symlink()) and p.name not in IGNORED_FILES:
            return p
    return None


def refusal(data: Path | None = None, panel: Path | None = PANEL_DIR) -> str | None:
    """Why the demo cannot write into `data` (and `panel`, None for no panel), or None when it can."""
    data = Path(C.DATA if data is None else data)
    if (data / C.PROTECT_MARKER).exists():
        return (
            f"the data folder is protected (it has {C.PROTECT_MARKER}). Use another folder: FACTORLAB_DATA=<new folder>"
        )
    if not is_demo(data):
        other = _first_file(data)
        if other is not None:
            shown = other.relative_to(data).as_posix()
            return (
                f"the data folder holds files that are not demo data (for example {shown}). "
                "The demo writes only into an empty data folder or into a demo folder. "
                "Use another folder: FACTORLAB_DATA=<new folder>"
            )
    if panel is not None and not is_demo_panel(panel) and _first_file(panel) is not None:
        return (
            "library/data/binance_um_1d holds a Binance panel that is not demo data. "
            "Use --skip-binance to build the demo without a Binance panel"
        )
    return None


# --------------------------------------------------------------------------- the generator


def file_names() -> list[str]:
    """The engine's coin file names in config.UNIVERSE order ('BTC-USDT', ...)."""
    return [C.sym_to_file(s) for s in C.UNIVERSE]


def panel_symbols() -> list[str]:
    """The Binance USD-M symbols of the coins, in config.UNIVERSE order ('BTCUSDT', ...)."""
    return [n.replace("-", "") for n in file_names()]


def _zscore(x):
    """Cross-sectional z-score of each row (NaN cells stay NaN; a row with one value gives 0)."""
    import warnings

    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a row without a value (before the first mean)
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


def _rolling_mean(x, n: int):
    """The mean of the last n rows up to and including each row (NaN until n rows exist)."""
    import pandas as pd

    return pd.DataFrame(x).rolling(n, min_periods=n).mean().to_numpy()


def market(seed: int = SEED) -> dict:
    """The synthetic market: {field: DataFrame(index = daily UTC bar open, columns = coin file names)}.

    The fields are the 16 raw fields of make_synthetic_panel (funding_rate is the daily sum of
    three 8-hour prints) plus basis (spot below perpetual, as a fraction) and spot_share (spot
    volume over perpetual volume). The 20 coins are the first 20 coins of the generator pool that
    are listed on the first day; the coins of LISTINGS get no value before their first bar.

    Changes to the generator panel, so that the demo has a realistic scale and something to find:
    - each coin's volumes and open interest are scaled to a median daily dollar volume
      (DOLLAR_VOLUME for the first two coins, log-uniform in DOLLAR_VOLUME_RANGE for the others);
    - the taker-buy quote volume is the taker-buy volume times the bar's average price, so each
      average price stays inside the bar's [low, high];
    - two planted signals (SIGNALS) move the next day's return: the 7-day mean close location value
      ((2 close - high - low) / (high - low), from the perpetual bars) and the 7-day mean of
      log(long_short_ratio) (a Binance-only field), each as a cross-sectional z-score known at the
      bar's close. The prices are scaled by the cumulative change, base-unit volumes by its inverse.
      The change of scale keeps each close location value as it is."""
    import numpy as np
    import pandas as pd

    if str(LIBRARY) not in sys.path:
        sys.path.append(str(LIBRARY))
    from factorlab.data.synthetic import make_synthetic_panel

    raw = make_synthetic_panel(n_symbols=POOL, n_bars=N_DAYS, seed=seed)
    listed = [c for c in raw["close"].columns if raw["close"][c].notna().all()]
    names = file_names()
    if len(listed) < len(names):
        raise RuntimeError(f"the generator gave {len(listed)} coins with a full history, not {len(names)}")
    src = listed[: len(names)]
    a = {f: df[src].to_numpy(dtype=float).copy() for f, df in raw.items()}
    rng = np.random.default_rng(seed + 1)
    n = len(names)
    # 1. the scale of the volumes and of open interest
    target = np.exp(rng.uniform(np.log(DOLLAR_VOLUME_RANGE[0]), np.log(DOLLAR_VOLUME_RANGE[1]), n))
    target[:2] = DOLLAR_VOLUME
    k = target / np.median(a["quote_volume"], axis=0)
    for f in (
        "volume",
        "quote_volume",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "trades",
        "open_interest",
        "open_interest_value",
    ):
        a[f] = a[f] * k
    a["trades"] = np.maximum(np.round(a["trades"]), 1.0)
    # 2. the planted signals (known at the close of bar t, they move the return of bar t + 1)
    sig = np.zeros_like(a["close"])
    for field, weeks_mean, weight in SIGNALS:
        if field == "clv":
            x = (2 * a["close"] - a["high"] - a["low"]) / (a["high"] - a["low"])
        else:
            x = np.log(a[field]) if field == "long_short_ratio" else a[field]
        z = np.nan_to_num(_zscore(_rolling_mean(x, weeks_mean)))
        sig += weight * z
    ret = a["close"] / np.vstack([a["close"][:1], a["close"][:-1]]) - 1.0
    new = ret.copy()
    new[1:] += sig[:-1]
    factor = np.cumprod((1.0 + new) / (1.0 + ret), axis=0)
    for f in ("open", "high", "low", "close"):
        a[f] = a[f] * factor
    for f in ("volume", "taker_buy_volume", "open_interest"):
        a[f] = a[f] / factor
    # 3. average prices inside the bar (with a margin, so the CSV text of the panel keeps them inside)
    margin = VWAP_MARGIN * (a["high"] - a["low"])
    vwap = np.clip(a["quote_volume"] / a["volume"], a["low"] + margin, a["high"] - margin)
    a["volume"] = a["quote_volume"] / vwap
    a["taker_buy_quote_volume"] = a["taker_buy_volume"] * vwap
    a["open_interest_value"] = a["open_interest"] * a["close"]
    index = pd.date_range(FIRST_DAY, periods=N_DAYS, freq="D", tz="UTC", name="ts")
    out = {f: pd.DataFrame(v, index=index, columns=names) for f, v in a.items()}
    shape = (N_DAYS, n)
    basis = np.zeros(shape)
    share = np.zeros(shape)
    share0 = rng.uniform(0.08, 0.4, n)
    for t in range(1, N_DAYS):
        basis[t] = 0.8 * basis[t - 1] + rng.normal(0, 0.0003, n)
        share[t] = 0.9 * share[t - 1] + rng.normal(0, 0.05, n)
    out["basis"] = pd.DataFrame(basis + 0.0004, index=index, columns=names)
    out["spot_share"] = pd.DataFrame(share0 * np.exp(share), index=index, columns=names)
    for symbol, first in LISTINGS.items():
        name = C.sym_to_file(symbol)
        before = index < pd.Timestamp(first, tz="UTC")
        for f in out:
            out[f].loc[before, name] = np.nan
    return out


def _bars(m: dict, name: str, spot: bool = False):
    """The daily bars of one coin (ts, open, high, low, close, volume) with ts in milliseconds."""
    import pandas as pd

    k = (1.0 - m["basis"][name]) if spot else 1.0
    df = pd.DataFrame(
        {
            "open": m["open"][name] * k,
            "high": m["high"][name] * k,
            "low": m["low"][name] * k,
            "close": m["close"][name] * k,
            "volume": m["volume"][name] * (m["spot_share"][name] if spot else 1.0),
        }
    )
    df = df[df["close"].notna()]
    df.index = pd.DatetimeIndex(df.index).as_unit("ms")
    return df.rename_axis("ts").reset_index()


def _prints(daily, days, rng, jitter_ms: int = 0, noise: float = 0.0):
    """8-hour funding prints from daily sums: three prints on each day, a third of the sum each."""
    import numpy as np
    import pandas as pd

    rows_ts, rows_v = [], []
    for day, value in zip(days, daily):
        if np.isnan(value):
            continue
        for h in SETTLEMENT_HOURS:
            rows_ts.append(day + pd.Timedelta(hours=h))
            rows_v.append(value / 3.0)
    ts = pd.DatetimeIndex(rows_ts)
    v = np.asarray(rows_v, dtype=float)
    if jitter_ms:
        ts = ts + pd.to_timedelta(rng.integers(0, jitter_ms + 1, len(ts)), unit="ms")
    if noise:
        v = v + rng.normal(0, noise, len(v))
    return pd.DataFrame({"ts": ts.as_unit("ms"), "funding_rate": np.round(v, FUNDING_DECIMALS)})


def funding_prints(m: dict, seed: int = SEED) -> dict:
    """{coin: (fallback prints, primary prints)}. The fallback venue has every day, with the few
    milliseconds of jitter of a real archive; the primary venue has the last PRIMARY_FUNDING_DAYS
    days, and its first day starts at 08:00 UTC (a partial day, as on a real first download)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed + 2)
    index = m["close"].index
    primary_from = index[-1] - pd.Timedelta(days=PRIMARY_FUNDING_DAYS - 1)
    out = {}
    for name in file_names():
        fr = m["funding_rate"][name]
        fallback = _prints(fr.to_numpy(), index, rng, jitter_ms=12)
        recent = fr[index >= primary_from]
        prim = _prints(recent.to_numpy(), recent.index, rng, noise=0.00002)
        prim = prim[prim["ts"] > pd.Timestamp(primary_from)].reset_index(drop=True)
        out[name] = (fallback, prim)
    return out


def write_raw(data: Path, m: dict, seed: int = SEED) -> list[Path]:
    """Write the raw folders of the engine (RAW_FOLDERS) into `data`. Returns the written files."""
    import pandas as pd

    data = Path(data)
    raw = data / "raw"
    index = m["close"].index
    oi_from = index[-1] - pd.Timedelta(days=OI_DAYS - 1)
    prints = funding_prints(m, seed)
    written = []

    def put(folder: str, name: str, df) -> None:
        d = raw / folder
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.parquet"
        C.guard_write(p)
        df.to_parquet(p, index=False)
        written.append(p)

    for name in file_names():
        put("ohlcv", name, _bars(m, name))
        put("spot_ohlcv", name, _bars(m, name, spot=True))
        fallback, primary = prints[name]
        put(f"funding_{C.FUNDING_FALLBACK_EXCHANGE_ID}", name, fallback)
        put("funding", name, primary)
        oiv = m["open_interest_value"][name]
        oiv = oiv[(oiv.index >= oi_from) & oiv.notna()]
        ts = (oiv.index + pd.Timedelta(hours=OI_STAMP_HOUR)).as_unit("ms")
        put(
            "open_interest",
            name,
            pd.DataFrame({"ts": ts, "oi_contracts": [None] * len(oiv), "oi_value": (oiv * 1.15).to_numpy()}),
        )
    return written


def panel_frames(m: dict, seed: int = SEED) -> dict:
    """The 16 Binance panel fields: {field: DataFrame(index open_time, columns BTCUSDT, ...)}.

    The prices differ a little from the primary venue (one factor per coin and day, so high, low
    and close stay in order); each open is the venue's own previous close times its own small
    gap, so the overnight gaps of the two venues are not correlated; the volumes are 1.3 times
    the primary venue's; funding_rate is the daily sum of the fallback prints, as on the real panel."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed + 3)
    names = file_names()
    index = m["close"].index
    eps = 1.0 + rng.normal(0, 0.0004, (len(index), len(names)))
    scale = {
        "open": eps,
        "high": eps,
        "low": eps,
        "close": eps,
        "volume": 1.3,
        "quote_volume": 1.3 * eps,
        "taker_buy_volume": 1.3,
        "taker_buy_quote_volume": 1.3 * eps,
        "trades": 1.3,
        "open_interest": 1.0,
        "open_interest_value": eps,
    }
    # funding_rate: the per-day sum of the fallback prints (settlement time in [D, D + 1))
    prints = funding_prints(m, seed)
    daily = {}
    for name in names:
        fb = prints[name][0]
        daily[name] = fb.groupby(fb["ts"].dt.floor("D"))["funding_rate"].sum().reindex(index)
    funding = pd.DataFrame(daily, index=index)[names].to_numpy()
    out = {}
    for f in PANEL_FIELDS:
        v = funding if f == "funding_rate" else m[f][names].to_numpy() * scale.get(f, 1.0)
        if f == "trades":
            v = np.round(v)
        df = pd.DataFrame(v, index=index.rename("open_time"), columns=panel_symbols())
        out[f] = df
    # the venue's own overnight gap: open = previous close x (1 + noise), where a previous close exists
    gap = 1.0 + rng.normal(0, 0.0015, (len(index), len(names)))
    prev = out["close"].shift(1).to_numpy()
    opened = np.where(np.isfinite(prev), prev * gap, out["open"].to_numpy())
    out["open"] = pd.DataFrame(opened, index=out["open"].index, columns=out["open"].columns)
    out["high"] = np.maximum(out["high"], out["open"])
    out["low"] = np.minimum(out["low"], out["open"])
    return out


def write_panel(panel: Path, m: dict, seed: int = SEED) -> list[Path]:
    """Write the Binance panel (merge_panels format, gzip header time 0) and its demo marker."""
    if str(LIBRARY) not in sys.path:
        sys.path.append(str(LIBRARY))
    from factorlab.data import save_panels

    panel = Path(panel)
    panel.mkdir(parents=True, exist_ok=True)
    # the marker comes first: a write that stops part way leaves a panel that is marked as demo data
    (panel / MARKER).write_text(json.dumps(_marker_body(seed, m), indent=2) + "\n", encoding="utf-8")
    save_panels(panel_frames(m, seed), panel, gzip_mtime=0)
    return sorted(panel.glob("*.csv.gz")) + [panel / MARKER]


def _marker_body(seed: int, m: dict) -> dict:
    index = m["close"].index
    return {
        "demo": True,
        "banner": BANNER,
        "format": FORMAT,
        "seed": seed,
        "first_bar": str(index[0].date()),
        "last_bar": str(index[-1].date()),
        "coins": len(file_names()),
        "generator": "library/factorlab/data/synthetic.py make_synthetic_panel",
    }


def write(data: Path | None = None, panel: Path | None = PANEL_DIR, seed: int = SEED) -> dict:
    """Write the demo data set into `data` (and the Binance panel into `panel`; None: no panel).

    Raises DemoRefused when refusal() gives a reason. Returns a summary."""
    data = Path(C.DATA if data is None else data)
    why = refusal(data, panel)
    if why:
        raise DemoRefused(f"demo_data: {why}.")
    m = market(seed)
    data.mkdir(parents=True, exist_ok=True)
    # the marker comes first: a write that stops part way leaves a folder that is marked as demo
    # data, so a market build refuses it and remove() can delete it
    body = _marker_body(seed, m)
    (data / MARKER).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    files = write_raw(data, m, seed)
    if panel is not None:
        files += write_panel(panel, m, seed)
    files.append(data / MARKER)
    return {**body, "files": len(files), "panel": panel is not None}


def _remove_content(folder: Path) -> None:
    """Delete each entry of `folder`, the demo marker last."""
    for p in sorted(folder.iterdir()):
        if p.name == MARKER:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
    (folder / MARKER).unlink()


def remove(data: Path | None = None, panel: Path = PANEL_DIR) -> list[str]:
    """Delete the demo data set: the content of the data folder and the panel folder, each only
    when it holds the demo marker. Returns what it removed (names only).

    The marker goes last: a removal that stops part way leaves a folder that is still marked as
    demo data, so a market build refuses it and a second remove() completes it."""
    data = Path(C.DATA if data is None else data)
    done = []
    if is_demo(data):
        _remove_content(data)
        done.append("the demo data folder content")
    if is_demo_panel(panel):
        _remove_content(Path(panel))
        Path(panel).rmdir()
        done.append("library/data/binance_um_1d (demo panel)")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic demo data (not market data).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="write the demo data set into the data folder")
    g.add_argument("--check", action="store_true", help="tell if the demo can write here; write nothing")
    g.add_argument("--remove", action="store_true", help="remove the demo data set (only demo folders)")
    ap.add_argument("--no-panel", action="store_true", help="make no Binance panel")
    ap.add_argument("--seed", type=int, default=SEED, help=f"generator seed (default {SEED})")
    a = ap.parse_args(argv)
    panel = None if a.no_panel else PANEL_DIR
    if a.check:
        why = refusal(C.DATA, panel)
        print(f"demo_data: {why}." if why else "demo_data: the demo can write here.")
        return 1 if why else 0
    if a.remove:
        done = remove(C.DATA)
        print("demo_data: removed " + " and ".join(done) + "." if done else "demo_data: no demo data found.")
        return 0
    try:
        s = write(C.DATA, panel, a.seed)
    except DemoRefused as exc:
        print(exc.code, file=sys.stderr)
        return 1
    print(
        f"demo_data: {BANNER} {s['coins']} coins, {s['first_bar']} to {s['last_bar']}, seed {s['seed']}, "
        f"{s['files']} files" + ("" if s["panel"] else ", no Binance panel") + "."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
