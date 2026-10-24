"""The generator's field map: its terminals on the engine's own point-in-time panels.

LABEL-FREE. This module reads perp bars, funding and same-venue spot bars, and nothing
else: no open-interest file, no factor table, no label, no screen result, no library test
result. The raw files are read whole (through the last bar on disk) and every series is
cut at the G cutoff (the 2024-06-09 Sunday row, the last row of the generator zone) as
soon as it is read, so no value dated later is used by a terminal.

ONE PANEL ADAPTER. The perp panels come from the library screen's own adapter,
`libfactors.build_panels()` (imported, never copied), fed with the same per-symbol reads
as `libfactors.load_raw()` except open interest: `data_io.ohlcv` and
`data_io.daily_funding`, the calls load_raw makes, through `read_raw()` below.
test_finder_fieldmap.py pins those reads equal to load_raw's, cell for cell, so a
generator formula and a library definition see identical inputs: OKX perp bars (volume
in base units), quote_volume = close x volume (the engine's dollar-volume convention) and
funding_rate = `data_io.daily_funding` (the per-8h-slot splice, NaN on a day without a
settlement). This module never reads open interest: it has no G-zone history, and
load_raw opens every open-interest file. Thus the adapter's open-interest fields are
empty, and this module drops them. No terminal and no library evaluation can reach them.

THE TERMINALS (returns, ranges, volume change, illiquidity and funding from the library,
plus perp-vs-spot basis, perp-over-spot range, perp/spot volume shift and return after
funding; names in eval_config.toml [generator.terminals], read from there and checked
against the constants here):
    library-named   21 stationary derived fields of library/factorlab/fields.py
                    (LIBRARY_TERMINALS): ret, ex_ret, ex_btc_ret, oc_ret, gap, abs_ret,
                    hl_range, log_hl, body, upper_shadow, lower_shadow, clv, tr_pct,
                    parkinson_var, gk_var, rs_var, dvol_chg, amihud, funding, mkt_ret,
                    btc_ret. The library's `basis` needs the premium index, which
                    Factorlab's data does not have; the Factorlab-only `sbasis` takes its
                    role. Each is evaluated by the library's own Evaluator on
                    those panels with the engine's BTC perp ("BTC-USDT") as the benchmark,
                    so `btc_ret` is its return and `ex_btc_ret` the excess over it, and
                    `mkt_ret` is the equal-weight mean return of the perps with a bar that
                    day and `ex_ret` the excess over it
    market-wide     mkt_ret and btc_ret (MARKET_WIDE) hold one value for every coin with a
                    return on a date (NaN where a coin has no return, including a
                    late-listed coin's first bar), so a formula built from them
                    alone is constant per date and the label-free G0 check sets it aside
                    (R_G0_CONSTANT; probe.py). A recursive operator such as ema can split
                    the coins by listing date on some dates, into at most as many values as
                    there are listing dates, fewer than G0's minimum of 10 distinct values
                    on this universe. They carry information only next to a coin's own
                    series, for example in ts_beta, ts_corr, ts_cov or ts_resid
    Factorlab-only  four series only Factorlab has, built from the same perp panels, the
                    same-venue OKX spot bars (`data_io.spot_ohlcv`) and funding, with the
                    library's own operators (div, log, delta: a zero or a non-positive
                    input gives NaN, never an infinity):
        sbasis             perp close / spot close - 1 (named so it never collides with the
                           library's premium-based `basis`). It is the perp-spot basis of
                           the same venue
        perp_range_excess  perp hl range - spot hl range, hl range = (high - low) / close
                           (the library's `hl_range`)
        deriv_share_chg    1-day change of log(perp dollar volume / spot dollar volume),
                           each dollar volume = close x base volume
        carry_ret          ret - funding_rate (the library's `ret`, close / close(D-1) - 1)
    Not listed (eval_config [generator.terminals] not_listed): the other stationary library
    series the panels supply (logret, sq_ret, up_ret, down_ret, close_loc, mkt_ret_vw,
    signed_dvol): logret, sq_ret and close_loc order the coins exactly like ret, abs_ret
    and clv; up_ret and down_ret are the positive and the negative part of ret; mkt_ret_vw
    is one value per date like mkt_ret; signed_dvol is a level.
    Excluded: open interest (no G-zone history), vwap and vwap_dev (vwap
    equals close here up to floating-point rounding, since quote_volume is close x volume,
    so vwap_dev is zero up to rounding), price and volume levels (open, high, low, close,
    volume, quote_volume, typical, log_dvol) and log_amount (a size level).

Every terminal is a (UTC day x perp) frame stamped with the bar's open time, NaN where an
input is missing (before a listing, a day without funding, a day without a spot bar). A
value on day D uses bars and funding of days <= D only; test_finder_fieldmap.py proves it
by overwriting every perp, spot and funding value after a cut.

ONE DAY GRID. libfactors.build_panels refuses a perp bar not stamped 00:00 UTC, and
spot_panels() refuses a spot bar that is not, before reindexing spot onto the perp grid:
a spot file stamped at another hour (for example 16:00 UTC, the start of a UTC+8 day)
would otherwise become NaN everywhere and silently empty sbasis, perp_range_excess and
deriv_share_chg. test_finder_fieldmap.py also pins the alignment: the same-day median
|sbasis| is far below its value with spot shifted one day either way.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config as C
import data_io
from factorfinder import config_io as K

#: The library-named terminals, in the config's order.
LIBRARY_TERMINALS = (
    "ret",
    "ex_ret",
    "ex_btc_ret",
    "oc_ret",
    "gap",
    "abs_ret",
    "hl_range",
    "log_hl",
    "body",
    "upper_shadow",
    "lower_shadow",
    "clv",
    "tr_pct",
    "parkinson_var",
    "gk_var",
    "rs_var",
    "dvol_chg",
    "amihud",
    "funding",
    "mkt_ret",
    "btc_ret",
)
#: The Factorlab-only terminals, in the config's order.
FACTORLAB_TERMINALS = ("sbasis", "perp_range_excess", "deriv_share_chg", "carry_ret")
#: Terminals with one value for every coin with a bar on a date: alone they are constant
#: per date (G0 sets such a formula aside); they matter next to a coin's own series.
MARKET_WIDE = ("mkt_ret", "btc_ret")
#: Perp panel fields a terminal may depend on: the adapter's fields minus open interest.
PERP_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume", "funding_rate")
OI_FIELDS = ("open_interest", "open_interest_value")
SPOT_FIELDS = ("open", "high", "low", "close", "volume")
#: Excluded names (open interest and its derived fields, vwap and vwap_dev,
#: price and volume levels, log_amount). None may be a terminal.
EXCLUDED = (
    "open_interest",
    "open_interest_value",
    "log_oi",
    "oi_chg",
    "oi_coin_chg",
    "oi_turnover",
    "vwap",
    "vwap_dev",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "typical",
    "log_dvol",
    "log_amount",
)


class FieldMapError(ValueError):
    """The terminal list in eval_config.toml does not match this module or the library."""


def _lib():
    """libfactors, imported on first use: it puts the repository's library/ on sys.path
    (and checks that the imported factorlab package is that one), and it holds the panel
    adapter this module shares with the library screen."""
    import libfactors

    return libfactors


# --------------------------------------------------------------------------- config


def g_cutoff(cfg: dict | None = None) -> pd.Timestamp:
    """The last G Sunday row (eval_config [zones] g_eval_sundays end): 2024-06-09, UTC."""
    cfg = K.load() if cfg is None else cfg
    return pd.Timestamp(cfg["zones"]["g_eval_sundays"][1], tz="UTC")


def terminal_config(cfg: dict | None = None) -> dict:
    """{"library": names, "factorlab": names, "definitions": {name: text}} from
    eval_config.toml [generator.terminals], checked against this module and the library."""
    _lib()
    from factorlab.expr import raw_dependencies
    from factorlab.fields import DERIVED_BY_NAME

    cfg = K.load() if cfg is None else cfg
    t = cfg["generator"]["terminals"]
    lib, fl, defs = tuple(t["library"]), tuple(t["factorlab"]), dict(t["definitions"])
    if lib != LIBRARY_TERMINALS:
        raise FieldMapError(f"[generator.terminals] library is {lib}, expected {LIBRARY_TERMINALS}")
    if fl != FACTORLAB_TERMINALS:
        raise FieldMapError(f"[generator.terminals] factorlab is {fl}, expected {FACTORLAB_TERMINALS}")
    if set(defs) != set(FACTORLAB_TERMINALS):
        raise FieldMapError("[generator.terminals.definitions] must define exactly the four Factorlab terminals")
    if len(set(lib)) != len(lib) or set(lib) & set(fl):
        raise FieldMapError("terminal names must be unique")
    for name in lib:
        d = DERIVED_BY_NAME.get(name)
        if d is None:
            raise FieldMapError(f"{name!r} is not a derived field of the library")
        if not d.stationary or name in EXCLUDED:
            raise FieldMapError(f"{name!r} is a level or an excluded field")
        deps = set(raw_dependencies(name))
        if not deps <= set(PERP_FIELDS):
            raise FieldMapError(
                f"{name!r} needs {sorted(deps - set(PERP_FIELDS))}, which the engine's "
                "panels do not supply (or that is excluded)"
            )
        if deps & {"open_interest", "open_interest_value"}:
            raise FieldMapError(f"{name!r} uses open interest, which is excluded")
    return {"library": lib, "factorlab": fl, "definitions": defs}


def terminal_names(cfg: dict | None = None) -> tuple[str, ...]:
    """Every terminal, library-named first, then the four Factorlab-only ones."""
    t = terminal_config(cfg)
    return t["library"] + t["factorlab"]


# --------------------------------------------------------------------------- inputs


def _cut(obj, through: pd.Timestamp):
    if obj is None:
        return None
    if obj.index.tz is None:
        raise ValueError("bars must carry a UTC index")
    return obj.loc[obj.index <= through]


def read_raw() -> dict:
    """The library screen's per-symbol reads without open interest: libfactors.load_raw()
    minus its data_io.daily_oi call, so no open-interest file is opened.

    {sym: {"ohlcv": DataFrame(open, high, low, close, volume), "funding": daily Series or
    None, "oi": None}}, the symbols in universe order (test_finder_fieldmap.py pins every
    frame equal to load_raw's)."""
    raw = {}
    for symbol in C.UNIVERSE:
        sym = C.sym_to_file(symbol)
        px = data_io.ohlcv(sym)
        if px is None:
            continue
        raw[sym] = {
            "ohlcv": px[["open", "high", "low", "close", "volume"]].astype(float),
            "funding": data_io.daily_funding(sym),
            "oi": None,
        }
    if not raw:
        raise SystemExit("fieldmap: no perp bars on disk")
    return raw


def load_inputs(through=None) -> dict:
    """The raw reads, each cut at `through` (default: the G cutoff) as soon as it is read.

    {"through": Timestamp, "raw": {sym: {"ohlcv", "funding", "oi": None}} (read_raw(): the
    library screen's reads without open interest), "spot": {sym: DataFrame(open, high,
    low, close, volume)}}."""
    through = g_cutoff() if through is None else pd.Timestamp(through)
    if through.tz is None:
        through = through.tz_localize("UTC")
    raw = {}
    for sym, r in read_raw().items():
        raw[sym] = {"ohlcv": _cut(r["ohlcv"], through), "funding": _cut(r["funding"], through), "oi": None}
    spot = {}
    for sym in raw:
        s = data_io.spot_ohlcv(sym)
        if s is not None:
            spot[sym] = _cut(s[list(SPOT_FIELDS)].astype(float), through)
    return {"through": through, "raw": raw, "spot": spot}


def perp_panels(inputs: dict) -> dict[str, pd.DataFrame]:
    """The library's raw panels from the shared adapter, its (empty) open-interest fields
    removed."""
    panels = _lib().build_panels(inputs["raw"], through=inputs["through"])
    return {k: v for k, v in panels.items() if k not in OI_FIELDS}


def _check_day_stamps(sym: str, index: pd.DatetimeIndex, what: str) -> None:
    """Refuse daily bars that are not stamped with their UTC open at 00:00 (the perp rule
    of libfactors.build_panels)."""
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or (index != index.normalize()).any():
        raise ValueError(f"{sym}: {what} bars must be stamped with their UTC open at 00:00")


def spot_panels(spot: dict, index: pd.DatetimeIndex, columns) -> dict[str, pd.DataFrame]:
    """Same-venue spot bars on the perp grid: NaN where a coin has no spot bar. Refuses a
    spot bar not stamped 00:00 UTC (the module docstring, ONE DAY GRID)."""
    for sym in columns:
        if sym in spot:
            _check_day_stamps(sym, spot[sym].index, "spot")
    out = {}
    for k in SPOT_FIELDS:
        f = pd.DataFrame(np.nan, index=index, columns=list(columns))
        for sym in columns:
            if sym in spot:
                f[sym] = spot[sym][k].reindex(index).astype(float)
        out[k] = f
    bar = out["close"].notna()
    return {k: v.where(bar) for k, v in out.items()}


# --------------------------------------------------------------------------- terminals


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).astype(float)


def build_terminals(inputs: dict | None = None, through=None) -> dict[str, pd.DataFrame]:
    """{terminal name: (day x perp) frame} for every terminal of [generator.terminals].

    `inputs` from load_inputs() (default: read now, cut at the G cutoff). Library-named
    terminals are the library Evaluator's own series on the shared panels; the four
    Factorlab-only ones are built with the library's operators."""
    LF = _lib()
    from factorlab.expr import Evaluator
    from factorlab.operators import OPERATORS as O

    inputs = load_inputs(through) if inputs is None else inputs
    names = terminal_config()
    perp = perp_panels(inputs)
    idx, cols = perp["close"].index, list(perp["close"].columns)
    ev = Evaluator(perp, benchmark=LF.BENCHMARK, cache_size=64)
    sp = spot_panels(inputs["spot"], idx, cols)
    out: dict[str, pd.DataFrame] = {}
    # NaN windows, logs of 0 and 0/0 are expected and become NaN; their RuntimeWarnings
    # are silenced here only
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for n in names["library"]:
            out[n] = _clean(ev(n).reindex(index=idx, columns=cols))
        out["sbasis"] = _clean(O["div"](perp["close"], sp["close"]) - 1.0)
        out["perp_range_excess"] = _clean(
            O["div"](perp["high"] - perp["low"], perp["close"]) - O["div"](sp["high"] - sp["low"], sp["close"])
        )
        out["deriv_share_chg"] = _clean(
            O["delta"](O["log"](O["div"](perp["quote_volume"], sp["close"] * sp["volume"])), 1)
        )
        out["carry_ret"] = _clean(ev("ret").reindex(index=idx, columns=cols) - perp["funding_rate"])
    if tuple(out) != names["library"] + names["factorlab"]:
        raise FieldMapError("terminal order drifted from the config")
    return out


def market_only(fields) -> bool:
    """True when every leaf is a market-wide terminal (mkt_ret, btc_ret): such a formula
    has one value for every coin on a date (up to the listing artefacts of recursive
    operators such as ema), so it cannot rank coins."""
    fields = set(fields)
    return bool(fields) and fields <= set(MARKET_WIDE)


def symbols() -> list[str]:
    """The engine's symbol files in universe order ('BTC-USDT', ...)."""
    return [C.sym_to_file(s) for s in C.UNIVERSE]
