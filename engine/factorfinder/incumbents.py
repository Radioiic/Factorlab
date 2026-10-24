"""The incumbents of the factor lab and AlphaForge, and the number for the library.

An incumbent is a factor that a new formula is compared with. The factor lab
(factor_lab.py) and the AlphaForge miner (alphaforge_bridge.py) use the same set:

    user    each registered test of the trial ledger (data/trials/) that is not withdrawn
            (trials.registered_values): a formula on the panels of its source, a values
            file on its Sundays
    zoo     each active AlphaForge formula: its column of data/research/alphaforge_daily.parquet,
            and the flag "active" of data/research/alphaforge_library.json (format ZOO_FORMAT; a
            file of another format stops with ZooFormatError)
    proxy   the three risk regressors pit_symbol_tilt, btc_beta_90d and size_proxy
            (userfactor.risk_proxies on the OKX panels)

The largest absolute correlation with the library definitions is a separate number
(library_corr). It uses the novelty index of factorfinder/novelty.py: the library members on
the G Sundays, each one measured exactly.

The statistic of nearest() is the mean within-date Spearman correlation on the common dates,
with at least MIN_COMMON_DATES such dates. n_incumbents is the number of incumbents with this
number. An empty set gives n_incumbents = 0 and no correlation (None), never 0.0.

The provider only reads. It writes no file.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

PROXIES = ("pit_symbol_tilt", "btc_beta_90d", "size_proxy")
SOURCES = ("user", "zoo", "proxy")
ZOO_LIBRARY = "alphaforge_library.json"
ZOO_TABLE = "alphaforge_daily.parquet"
ZOO_FORMAT = 3  # the format of the zoo file that alphaforge_bridge.py writes and reads
MIN_COMMON_DATES = 20  # nearest(): the fewest dates with a correlation


@dataclass(frozen=True)
class Incumbent:
    name: str  # "user:FT-0001", "zoo:af_07", "proxy:btc_beta_90d"
    source: str  # "user" | "zoo" | "proxy"
    values: pd.DataFrame  # date x the 20 engine symbols, UTC daily index


def symbols() -> list[str]:
    """The engine's symbol names in universe order ('BTC-USDT', ...)."""
    return [C.sym_to_file(s) for s in C.UNIVERSE]


def _utc(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.as_unit("ns")


def _daily(frame: pd.DataFrame, through: date | None = None) -> pd.DataFrame:
    """A value frame on a UTC date index and the engine symbols (float64), cut at `through`."""
    out = frame.copy()
    out.index = _utc(out.index)
    out = out.reindex(columns=symbols()).astype(np.float64)
    if through is not None:
        out = out[out.index <= pd.Timestamp(through, tz="UTC")]
    return out.sort_index()


def _research() -> Path:
    return Path(C.RESEARCH)


# --------------------------------------------------------------------------- the members


def user_members(*, through: date | None = None) -> list[Incumbent]:
    """The registered tests that are not withdrawn, in the order of registration."""
    import trials

    values = trials.registered_values(trials.read(), daily=True, through=through)
    return [Incumbent(f"user:{tid}", "user", _daily(v, through)) for tid, v in values.items()]


class ZooFormatError(ValueError):
    """The zoo file does not have the format ZOO_FORMAT."""


def check_zoo_format(zoo, where: str = f"data/research/{ZOO_LIBRARY}") -> dict:
    """`zoo` when it is a JSON object with "format" equal to ZOO_FORMAT; else ZooFormatError."""
    fmt = zoo.get("format") if isinstance(zoo, dict) else None
    if not isinstance(fmt, int) or isinstance(fmt, bool) or fmt != ZOO_FORMAT:
        raise ZooFormatError(f"the zoo file {where} has the format {fmt!r}; Factorlab reads only format {ZOO_FORMAT}")
    return zoo


def check_zoo_file() -> None:
    """Stop with ZooFormatError when data/research/alphaforge_library.json exists and does not have
    the format ZOO_FORMAT. A file that is not JSON also stops. Without the file: no check."""
    lib = _research() / ZOO_LIBRARY
    if not lib.is_file():
        return
    try:
        zoo = json.loads(lib.read_text(encoding="utf-8"))
    except ValueError as exc:  # also a text that is not UTF-8
        raise ZooFormatError(f"the zoo file data/research/{ZOO_LIBRARY} is not a JSON file") from exc
    check_zoo_format(zoo)


def zoo_active_names(zoo: dict) -> list[str]:
    """The names of the active formulas of a zoo file (the flag "active"), in the order of the file.
    A zoo of another format stops with ZooFormatError."""
    check_zoo_format(zoo)
    return [str(m.get("name")) for m in zoo.get("factors") or [] if m.get("active") is True]


def zoo_members(*, through: date | None = None, active_only: bool = True) -> list[Incumbent]:
    """The AlphaForge formulas of the zoo table (only the active ones by default). An empty list
    without the zoo file or the table."""
    lib, table = _research() / ZOO_LIBRARY, _research() / ZOO_TABLE
    if not lib.is_file() or not table.is_file():
        return []
    import pyarrow.parquet as pq

    zoo = json.loads(lib.read_text(encoding="utf-8"))
    active = set(zoo_active_names(zoo))
    names = [m["name"] for m in zoo.get("factors", []) if not active_only or m.get("name") in active]
    have = set(pq.read_schema(table).names)
    names = [n for n in names if n in have]
    if not names:
        return []
    df = pd.read_parquet(table, columns=["date", "symbol", *names])
    out = []
    for n in names:
        wide = df.pivot(index="date", columns="symbol", values=n)
        out.append(Incumbent(f"zoo:{n}", "zoo", _daily(wide, through)))
    return out


def proxy_members(*, through: date | None = None) -> list[Incumbent]:
    """The three risk regressors on each day, from the OKX bars up to that day. None without the
    bars of the benchmark coin (BTC-USDT)."""
    import libfactors as lf
    import userfactor as uf

    try:
        raw = lf.load_raw()
    except SystemExit:
        return []
    panels = lf.build_panels(raw, through=through)
    syms = list(panels["close"].columns)
    if lf.BENCHMARK not in syms:
        return []
    idx = panels["close"].index
    regs = uf.risk_proxies(panels, syms, idx)
    return [
        Incumbent(f"proxy:{name}", "proxy", _daily(pd.DataFrame(regs[key], index=idx, columns=syms), through))
        for key, name in zip(lf.REG_KEYS, PROXIES)
    ]


def incumbents(
    *,
    sources: tuple[str, ...] = SOURCES,
    through: date | None = None,
    zoo_values: Mapping[str, pd.DataFrame] | None = None,
) -> list[Incumbent]:
    """The incumbents of the given sources, in the order user, zoo, proxy.

    zoo_values: the values of the active AlphaForge formulas from the caller (the miner computes
    them on its own store); when None, zoo_members() reads them from the zoo files."""
    bad = [s for s in sources if s not in SOURCES]
    if bad:
        raise ValueError(f"unknown incumbent source(s) {bad}")
    out: list[Incumbent] = []
    for s in SOURCES:
        if s not in sources:
            continue
        if s == "user":
            out += user_members(through=through)
        elif s == "zoo":
            if zoo_values is None:
                out += zoo_members(through=through)
            else:
                out += [Incumbent(f"zoo:{k}", "zoo", _daily(v, through)) for k, v in zoo_values.items()]
        else:
            out += proxy_members(through=through)
    return out


def counts(members: list[Incumbent]) -> dict[str, int]:
    """{source: number of members} for each source."""
    return {s: sum(1 for m in members if m.source == s) for s in SOURCES}


# --------------------------------------------------------------------------- ranks and correlations


def rank_frames(
    members: list[Incumbent], dates: pd.DatetimeIndex, columns: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """{name: within-date percentile ranks on `dates` x `columns` (default: the engine symbols)}.
    The index of each frame is `dates` as given (naive or UTC)."""
    idx = pd.DatetimeIndex(dates)
    aware = _utc(idx)
    cols = symbols() if columns is None else list(columns)
    out = {}
    for m in members:
        r = m.values.reindex(index=aware, columns=cols).rank(axis=1, pct=True)
        r.index = idx
        out[m.name] = r
    return out


def nearest(values: pd.DataFrame, frames: Mapping[str, pd.DataFrame], *, min_dates: int = MIN_COMMON_DATES) -> dict:
    """The incumbent with the largest absolute mean within-date Spearman correlation with `values`.

    {"corr": that signed mean (None without an incumbent), "name": its name (None),
    "n_incumbents": the number of incumbents with at least `min_dates` dates with a correlation}."""
    best: float | None = None
    name: str | None = None
    n = 0
    for k, f in frames.items():
        common = values.index.intersection(f.index)
        if len(common) < min_dates:
            continue
        # a constant or empty row has no correlation (NaN); its warnings are silenced here only
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            cc = values.loc[common].corrwith(f.loc[common], axis=1, method="spearman").dropna()
        if len(cc) < min_dates:
            continue
        n += 1
        m = float(cc.mean())
        if math.isfinite(m) and (best is None or abs(m) > abs(best)):
            best, name = m, k
    return {"corr": best, "name": name, "n_incumbents": n}


# --------------------------------------------------------------------------- the library number

_LIBRARY_INDEX: dict = {}


def _library_index():
    """The library members of the novelty index, built once for each data folder and input state.
    FileNotFoundError when the libfactors values or the label table are absent."""
    import labels
    import libfactors as lf
    from factorfinder import novelty as N

    status = lf.out_dir() / "status.parquet"
    if not status.is_file() or not lf.has_values():
        raise FileNotFoundError("the library values are absent: run libfactors.py --compute")
    if not labels.path().is_file():
        raise FileNotFoundError("the label table is absent: run labels.py")
    key = (str(C.DATA), status.stat().st_mtime_ns, labels.path().stat().st_mtime_ns)
    if _LIBRARY_INDEX.get("key") != key:
        _LIBRARY_INDEX.clear()
        _LIBRARY_INDEX.update(key=key, ix=N.build_index(sources=("library",)))
    return _LIBRARY_INDEX["ix"]


def library_corr(values: pd.DataFrame) -> dict | None:
    """The largest absolute correlation of `values` (dates x symbols) with a library definition on
    the G Sundays: {"max_abs", "corr" (signed), "nearest" (the definition id), "n_members"}.
    max_abs is None when no member has a correlation. None when the libfactors values are absent."""
    from factorfinder import novelty as N

    try:
        ix = _library_index()
    except FileNotFoundError:
        return None
    v = values.copy()
    v.index = _utc(v.index)
    arr = v.reindex(index=_utc(ix.sundays), columns=ix.symbols).to_numpy(dtype=np.float64)
    r = N.max_corr(ix, arr)
    ok = bool(np.isfinite(r["max_abs"]))
    return {
        "max_abs": float(r["max_abs"]) if ok else None,
        "corr": float(r["corr"]) if ok else None,
        "nearest": r["nearest"] if ok else None,
        "n_members": int(ix.size),
    }
