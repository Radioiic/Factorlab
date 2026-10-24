"""Book specifications and the book of a factor test.

THE BOOK SPECIFICATION. A TOML file (format 1) gives the book that tests a factor:

    format = 1                       required
    name = "long-short"              required: 1 to 64 letters, digits, "_", "." or "-"
    version = 1                      required: a whole number from 1 to 1000000
    [book]
    direction = "long_short"         "long_short" or "long_only"
    sign = 1                         1: a high value is long; -1: a low value is long
    top_fraction = 0.2               the long side: more than 0, not more than 0.5
    bottom_fraction = 0.2            the short side (long_short only; a long_only file has no such key)
    exit_fraction = 0.4              a name in the book stays while it is in this top (or bottom) fraction
                                     (long_short: from the larger fraction to 0.5; long_only: from
                                     top_fraction to 1.0)
    rebalance = "weekly-monday"      each Monday 00:00 UTC, on the values of the Sunday before
    [costs]
    scenario = "taker"               "taker" (5 + 5 bps), "maker" (2 + 2 bps) or "custom"
    fee_bps = 5.0                    optional for taker and maker (it must then be the scenario value);
    slippage_bps = 5.0               necessary for custom (0 to 100)
    [control]
    kind = "zero"                    "zero" (no position; the default of long_short) or
                                     "equal_weight" (the eligible coins of the 20; the default of long_only)
    [forward]
    sigma_floor_fallback = 0.03      0.001 to 1.0 (weekly-return units): the sigma floor of the forward
                                     test when the history has fewer than 52 weeks with a factor position

Every key other than format, name and version can be absent: it then gets its default. An
unknown key or table is refused, and a true or false value is never a number. The identity of a
specification is the sha256 of the canonical JSON of its normalized content (every key present,
whole numbers of the float keys as floats, the costs filled from the scenario). Comments and
layout do not change it. book_key is the same hash without name, version, book.sign,
costs.scenario and the table [forward]: two specifications with the same book_key give the same
book of the signed factor values.
engine/bookspecs/ holds two files: long_short.toml (the default) and long_only.toml.

THE BOOK (BandedBook). At each signal Sunday, the names are the eligible coins (labels.eligible)
with a finite value. A long_short book holds the top fraction of the signed values long and the
bottom fraction short, at least MIN_PER_SIDE names on each side, each side equal-weighted at half
of the gross. A long_only book holds the top fraction long at the full gross. A name in the book
stays while it is in the exit fraction. With fewer than MIN_NAMES names (also none), the book holds
the control portfolio: no position for the control "zero", the equal weights of the eligible coins
for the control "equal_weight". At a signal Sunday without an eligible coin, the book and the
control book hold no position.

THE RUN (run_book). backtest.run_backtest runs the book from a seed Sunday (no trade) to an end
Sunday: OKX perpetual closes and funding, a trade each Monday 00:00 UTC on the Sunday values,
fees and slippage on each traded dollar, accounting reconciled-v2, a start equity of 10,000. The
equal-weight control is the equal-weight book of the eligible coins with the same rebalance and
the same costs. BTC hold (price only) is context. The weekly return of a signal Sunday S is
E(S + 7) / E(S) - 1 (E: the daily equity mark), and the weekly net excess return is the book's
weekly return minus the control's (0 for the control "zero").
"""

from __future__ import annotations

import hashlib
import math
import re
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import config as C
import data_io
import labels
import ledger

SPEC_FORMAT = 1
SPEC_DIR = C.ROOT / "bookspecs"
DEFAULT_SPEC = SPEC_DIR / "long_short.toml"
SHIPPED_SPECS = ("long_short", "long_only")
COST_SCENARIOS = {"taker": (5.0, 5.0), "maker": (2.0, 2.0)}  # (fee_bps, slippage_bps)
MIN_NAMES, MIN_PER_SIDE, GROSS, INITIAL_EQUITY = 5, 2, 1.0, 10_000.0
MAX_SPEC_BYTES = 65_536
BENCHMARK = "BTC-USDT"  # context only: no rule reads it
ACCOUNTING = backtest.ACCOUNTING_RECONCILED
DIRECTIONS = ("long_short", "long_only")
CONTROLS = ("zero", "equal_weight")
REBALANCES = ("weekly-monday",)
SCENARIOS = ("taker", "maker", "custom")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAX_VERSION = 1_000_000
MAX_CUSTOM_BPS = 100.0
DEFAULTS = {"top_fraction": 0.2, "bottom_fraction": 0.2, "exit_fraction": 0.4, "sigma_floor_fallback": 0.03}
TABLE_KEYS = {
    "book": ("direction", "sign", "top_fraction", "bottom_fraction", "exit_fraction", "rebalance"),
    "costs": ("scenario", "fee_bps", "slippage_bps"),
    "control": ("kind",),
    "forward": ("sigma_floor_fallback",),
}
TOP_KEYS = ("format", "name", "version", *TABLE_KEYS)
WEEK = pd.Timedelta(days=7)
# A name that is not a coin. The engine calls a book only at a date with a row. So at a signal
# Sunday without an eligible coin, run_book gives the book and the control one row of this name.
# Both then hold no position: BandedBook and equal_weight give this name no weight, and the
# engine ignores a name that is not a column of the prices.
NO_COIN = "-"


class SpecError(ValueError):
    """A book specification that is not valid. Its text names the key and the rule."""


# --------------------------------------------------------------------------- the specification


def _is_int(x) -> bool:
    return type(x) is int


def _is_num(x) -> bool:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:
        return math.isfinite(float(x))
    except OverflowError:  # a TOML integer that is too large for a float
        return False


def _q(text: str) -> str:
    return f'"{text}"'


def _choice(value, key: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise SpecError(f"{key} must be {' or '.join(_q(a) for a in allowed)}.")
    return value


def _number(value, key: str, lo: float, hi: float, lo_open: bool = False) -> float:
    low = "more than" if lo_open else "at least"
    rule = f"{key} must be a number {low} {lo:g} and not more than {hi:g}."
    if not _is_num(value):
        raise SpecError(rule)
    x = float(value) + 0.0  # + 0.0: -0.0 becomes 0.0, so the canonical JSON has one form of zero
    if (x <= lo if lo_open else x < lo) or x > hi:
        raise SpecError(rule)
    return x


def normalize(raw: Mapping) -> dict:
    """The validated, normalized body of a specification (module docstring). SpecError when a key
    or a value is not valid."""
    if not isinstance(raw, Mapping):
        raise SpecError("the specification must be a table of keys.")
    for k in raw:
        if k not in TOP_KEYS:
            raise SpecError(f"{k!r} is not a key of the book specification (format 1).")
    tables: dict[str, Mapping] = {}
    for name, keys in TABLE_KEYS.items():
        t = raw.get(name, {})
        if not isinstance(t, Mapping):
            raise SpecError(f"[{name}] must be a table.")
        for k in t:
            if k not in keys:
                raise SpecError(f"{name}.{k} is not a key of the book specification (format 1).")
        tables[name] = t
    for k in ("format", "name", "version"):
        if k not in raw:
            raise SpecError(f"the key {k} is necessary.")
    if not _is_int(raw["format"]) or raw["format"] != SPEC_FORMAT:
        raise SpecError(f"format must be {SPEC_FORMAT}.")
    name = raw["name"]
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise SpecError('name must have 1 to 64 characters: letters, digits, "_", "." or "-".')
    version = raw["version"]
    if not _is_int(version) or not 1 <= version <= MAX_VERSION:
        raise SpecError(f"version must be a whole number from 1 to {MAX_VERSION}.")

    b = tables["book"]
    direction = _choice(b.get("direction", "long_short"), "book.direction", DIRECTIONS)
    sign = b.get("sign", 1)
    if not _is_int(sign) or sign not in (1, -1):
        raise SpecError("book.sign must be 1 or -1.")
    top = _number(b.get("top_fraction", DEFAULTS["top_fraction"]), "book.top_fraction", 0.0, 0.5, lo_open=True)
    if direction == "long_short":
        bottom: float | None = _number(
            b.get("bottom_fraction", DEFAULTS["bottom_fraction"]), "book.bottom_fraction", 0.0, 0.5, lo_open=True
        )
        lo_exit, hi_exit = max(top, float(bottom or 0.0)), 0.5
        exit_key = "book.exit_fraction (long_short: from the larger of top_fraction and bottom_fraction)"
    else:
        if b.get("bottom_fraction") is not None:
            raise SpecError("book.bottom_fraction is not a key of a long_only book. Remove it.")
        bottom = None
        lo_exit, hi_exit = top, 1.0
        exit_key = "book.exit_fraction (long_only: from top_fraction)"
    exit_value = b.get("exit_fraction", DEFAULTS["exit_fraction"])
    if not _is_num(exit_value) or not lo_exit <= float(exit_value) <= hi_exit:
        raise SpecError(f"{exit_key} must be a number from {lo_exit:g} to {hi_exit:g}.")
    rebalance = _choice(b.get("rebalance", REBALANCES[0]), "book.rebalance", REBALANCES)

    c = tables["costs"]
    scenario = _choice(c.get("scenario", "taker"), "costs.scenario", SCENARIOS)
    costs: dict[str, object] = {"scenario": scenario}
    for i, key in enumerate(("fee_bps", "slippage_bps")):
        if scenario == "custom":
            if key not in c:
                raise SpecError(f"costs.{key} is necessary for the scenario {_q('custom')}.")
            costs[key] = _number(c[key], f"costs.{key}", 0.0, MAX_CUSTOM_BPS)
        else:
            want = COST_SCENARIOS[scenario][i]
            if key in c and (not _is_num(c[key]) or float(c[key]) != want):
                raise SpecError(f"costs.{key} must be {want:g} for the scenario {_q(scenario)} (or remove the key).")
            costs[key] = want

    kind = _choice(
        tables["control"].get("kind", "zero" if direction == "long_short" else "equal_weight"), "control.kind", CONTROLS
    )
    floor = _number(
        tables["forward"].get("sigma_floor_fallback", DEFAULTS["sigma_floor_fallback"]),
        "forward.sigma_floor_fallback",
        0.001,
        1.0,
    )
    return {
        "format": SPEC_FORMAT,
        "name": name,
        "version": version,
        "book": {
            "direction": direction,
            "sign": sign,
            "top_fraction": top,
            "bottom_fraction": bottom,
            "exit_fraction": float(exit_value),
            "rebalance": rebalance,
        },
        "costs": costs,
        "control": {"kind": kind},
        "forward": {"sigma_floor_fallback": floor},
    }


def spec_sha256(body: Mapping) -> str:
    """The identity of a specification: sha256 of the canonical JSON of its normalized body."""
    return hashlib.sha256(ledger.canonical(body).encode("ascii")).hexdigest()


def book_key(body: Mapping) -> str:
    """sha256 of the canonical JSON of the body without name, version, book.sign, costs.scenario
    and the table [forward]. The table [forward] holds a setting of the forward test, not of the
    book. The book uses only the two numbers of the costs (fee_bps and slippage_bps), so the name
    of the scenario does not change the key: "taker" and "custom" with 5 + 5 bps give the same key."""
    rest = {k: v for k, v in body.items() if k not in ("name", "version", "forward")}
    rest["book"] = {k: v for k, v in body["book"].items() if k != "sign"}
    rest["costs"] = {k: v for k, v in body["costs"].items() if k != "scenario"}
    return hashlib.sha256(ledger.canonical(rest).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ForwardSettings:
    sigma_floor_fallback: float


@dataclass(frozen=True, eq=False)
class BookSpec:
    body: dict  # the normalized body
    sha256: str
    source: str  # the file name, for display only (not hashed)

    @property
    def name(self) -> str:
        return self.body["name"]

    @property
    def version(self) -> int:
        return self.body["version"]

    @property
    def direction(self) -> str:
        return self.body["book"]["direction"]

    @property
    def long_only(self) -> bool:
        return self.direction == "long_only"

    @property
    def sign(self) -> int:
        return self.body["book"]["sign"]

    @property
    def top(self) -> float:
        return self.body["book"]["top_fraction"]

    @property
    def bottom(self) -> float | None:
        return self.body["book"]["bottom_fraction"]

    @property
    def exit(self) -> float:
        return self.body["book"]["exit_fraction"]

    @property
    def rebalance(self) -> str:
        return self.body["book"]["rebalance"]

    @property
    def scenario(self) -> str:
        return self.body["costs"]["scenario"]

    @property
    def fee_bps(self) -> float:
        return self.body["costs"]["fee_bps"]

    @property
    def slippage_bps(self) -> float:
        return self.body["costs"]["slippage_bps"]

    @property
    def control(self) -> str:
        return self.body["control"]["kind"]

    @property
    def forward(self) -> ForwardSettings:
        return ForwardSettings(self.body["forward"]["sigma_floor_fallback"])

    @property
    def book_key(self) -> str:
        return book_key(self.body)


def spec_from_body(body: Mapping, source: str = "ledger") -> BookSpec:
    """A BookSpec from a normalized body (for example the body of a registration line). It
    validates the body again; a body that is not normalized is refused."""
    try:
        norm = normalize(body)
    except SpecError as exc:
        raise SpecError(f"The book specification {source} is not valid: {exc}") from None
    if ledger.canonical(norm) != ledger.canonical(body):
        raise SpecError(f"The book specification {source} is not a normalized body.")
    return BookSpec(body=norm, sha256=spec_sha256(norm), source=source)


def load_spec(spec: str | Path | None = None) -> BookSpec:
    """The specification of a file. None: the default (bookspecs/long_short.toml);
    "long_short" or "long_only": the shipped file; else the path of a file."""
    if spec is None:
        p = DEFAULT_SPEC
    elif isinstance(spec, str) and spec in SHIPPED_SPECS:
        p = SPEC_DIR / f"{spec}.toml"
    else:
        p = Path(spec)
    source = p.name

    def bad(text: str) -> SpecError:
        return SpecError(f"The book specification {source} is not valid: {text}")

    if not p.is_file():
        raise SpecError(f"The book specification {source} is not a file.")
    if p.stat().st_size > MAX_SPEC_BYTES:
        raise bad(f"the file has more than {MAX_SPEC_BYTES:,} bytes.")
    data = p.read_bytes()
    if len(data) > MAX_SPEC_BYTES:
        raise bad(f"the file has more than {MAX_SPEC_BYTES:,} bytes.")
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except UnicodeDecodeError:
        raise bad("the file is not UTF-8 text.") from None
    except tomllib.TOMLDecodeError as exc:
        raise bad(f"the file is not valid TOML: {exc}.") from None
    try:
        body = normalize(raw)
    except SpecError as exc:
        raise bad(str(exc)) from None
    return BookSpec(body=body, sha256=spec_sha256(body), source=source)


# --------------------------------------------------------------------------- the book construction


class BandedBook:
    """The banded book (module docstring). Stateful: it keeps the names that it holds, so each
    run needs a new instance (book_weights_fn)."""

    def __init__(
        self,
        score_fn: Callable[[pd.DataFrame], pd.Series],
        *,
        long_only: bool,
        top: float,
        bottom: float | None,
        exit: float,
        flat: str = "cash",
        min_names: int = MIN_NAMES,
        min_side: int = MIN_PER_SIDE,
        gross: float = GROSS,
    ) -> None:
        if flat not in ("cash", "equal_weight"):
            raise ValueError(f"flat must be 'cash' or 'equal_weight', not {flat!r}")
        if not long_only and bottom is None:
            raise ValueError("a long-short book needs a bottom fraction")
        self.score_fn = score_fn
        self.long_only, self.top, self.bottom, self.exit = long_only, top, bottom, exit
        self.flat, self.min_names, self.min_side, self.gross = flat, min_names, min_side, gross
        self.longs: set = set()
        self.shorts: set = set()

    def __call__(self, cs: pd.DataFrame) -> pd.Series:
        s = self.score_fn(cs).dropna()
        if len(s) < self.min_names:
            # too few names with a value (also none): the control portfolio
            self.longs, self.shorts = set(), set()
            if self.flat == "cash" or len(cs) == 0:
                return pd.Series(dtype=float)
            return equal_weight(cs)
        r = s.rank(pct=True)  # 1.0 = the strongest long
        n_long = max(self.min_side, math.floor(len(s) * self.top + 1e-9))
        top_n = set(s.nlargest(n_long).index)
        self.longs = {x for x in self.longs if x in r.index and r[x] >= 1 - self.exit} | top_n
        w = pd.Series(0.0, index=s.index)
        if self.long_only:
            w[list(self.longs)] = self.gross / len(self.longs)
            return w
        n_short = max(self.min_side, math.floor(len(s) * float(self.bottom or 0.0) + 1e-9))
        bot_n = set(s.nsmallest(n_short).index)
        self.shorts = {x for x in self.shorts if x in r.index and r[x] <= self.exit} | bot_n
        self.shorts -= self.longs  # a name is never on both sides
        self.longs -= bot_n
        # with ties a side can be empty (for example: every value is equal); it then holds nothing
        if self.longs:
            w[list(self.longs)] = self.gross / 2 / len(self.longs)
        if self.shorts:
            w[list(self.shorts)] = -self.gross / 2 / len(self.shorts)
        return w


def equal_weight(cs: pd.DataFrame) -> pd.Series:
    """1 / n on each of the n coin rows of cs: the equal-weight control (backtest.demo_market). No
    position when cs has no coin row (only the row NO_COIN)."""
    coins = cs[cs.index != NO_COIN]
    return backtest.demo_market(coins) if len(coins) else pd.Series(dtype=float)


def book_weights_fn(spec: BookSpec) -> BandedBook:
    """A new BandedBook for one run of the specification: score = sign x value."""
    sign = spec.sign

    def score(cs: pd.DataFrame) -> pd.Series:
        return sign * cs["value"]

    return BandedBook(
        score,
        long_only=spec.long_only,
        top=spec.top,
        bottom=spec.bottom,
        exit=spec.exit,
        flat="cash" if spec.control == "zero" else "equal_weight",
    )


# --------------------------------------------------------------------------- frames for the engine


def _symbols() -> list[str]:
    return [C.sym_to_file(s) for s in C.UNIVERSE]


def _utc_index(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def factor_frame(values: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """The long factor table [date, symbol, value] of the values (Sunday rows x the symbols): one
    row for EVERY eligible cell of each date, value NaN where the value is absent or not finite.
    Sorted by date, then by the config.UNIVERSE order. Thus the engine calls the book at each
    Sunday with an eligible coin."""
    cols = _symbols()
    v = values.copy()
    v.index = _utc_index(v.index)
    v = v.sort_index().reindex(columns=cols).astype(np.float64)
    arr = v.to_numpy()
    arr = np.where(np.isfinite(arr), arr, np.nan)
    el = eligible.copy()
    el.index = _utc_index(el.index)
    mask = el.reindex(index=v.index, columns=cols, fill_value=False).to_numpy(dtype=bool)
    ii, jj = np.nonzero(mask)
    return pd.DataFrame(
        {"date": v.index[ii], "symbol": [cols[j] for j in jj], "value": arr[ii, jj] if len(ii) else np.array([])}
    )


def universe_frame(eligible: pd.DataFrame, sundays) -> pd.DataFrame:
    """The long table [date, symbol] of every eligible cell of the Sundays (the control's universe)."""
    cols = _symbols()
    days = _utc_index(sundays)
    el = eligible.copy()
    el.index = _utc_index(el.index)
    mask = el.reindex(index=days, columns=cols, fill_value=False).to_numpy(dtype=bool)
    ii, jj = np.nonzero(mask)
    return pd.DataFrame({"date": days[ii], "symbol": [cols[j] for j in jj]})


def _every_sunday(frame: pd.DataFrame, sundays: pd.DatetimeIndex) -> pd.DataFrame:
    """The frame with one row of NO_COIN (value NaN) at each Sunday without a row."""
    absent = sundays[~sundays.isin(pd.DatetimeIndex(frame["date"]))]
    if len(absent) == 0:
        return frame
    pad = pd.DataFrame({"date": absent, "symbol": NO_COIN})
    if "value" in frame.columns:
        pad["value"] = np.nan
    out = pd.concat([frame, pad], ignore_index=True)
    return out.sort_values("date", kind="stable", ignore_index=True)


# --------------------------------------------------------------------------- the run


@dataclass
class BookRun:
    spec: BookSpec
    start: pd.Timestamp  # the seed Sunday (row 0, no trade)
    end: pd.Timestamp  # the last end mark (a Sunday)
    book: backtest.Result
    control: backtest.Result | None  # None for the control "zero"
    missed: list[str]  # the signal Sundays at which the book has the control portfolio


def _sunday(x, what: str) -> pd.Timestamp:
    t = pd.Timestamp(x)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    if t != t.normalize() or t.weekday() != 6:
        raise ValueError(f"{what} must be a Sunday (UTC midnight), not {x!r}")
    return t


def _pad_bust(res: backtest.Result, window: pd.DataFrame) -> backtest.Result:
    """After a bust (equity <= 0) the engine stops: the rows up to the end get equity 0.0 (no
    position, no cost), so the weekly series has every week."""
    full = pd.DatetimeIndex(window.index)
    if len(res.daily) >= len(full):
        return res
    rest = full[len(res.daily) :]
    pad = pd.DataFrame(0.0, index=rest, columns=res.daily.columns)
    daily = pd.concat([res.daily, pad])
    daily.index.name = res.daily.index.name
    eq = pd.concat([res.equity["equity"], pd.Series(0.0, index=rest)])
    if BENCHMARK in window.columns:
        bpx = window[BENCHMARK].ffill()
        bench = INITIAL_EQUITY * bpx / bpx.iloc[0]
    else:
        bench = pd.Series(INITIAL_EQUITY, index=full)
    equity = pd.DataFrame({"equity": eq, "benchmark": bench.reindex(eq.index)})
    return backtest.Result(equity=equity, trades=res.trades, daily=daily, stats=res.stats)


def run_book(
    spec: BookSpec,
    values: pd.DataFrame,
    *,
    start,
    end,
    prices: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    eligible: pd.DataFrame | None = None,
    schedule: Mapping[pd.Timestamp, pd.Timestamp] | None = None,
) -> BookRun:
    """Run the book of a specification on the values (Sunday rows x the 20 symbols) from the seed
    Sunday `start` to the end Sunday `end` (at least 7 days later), and its control book.

    schedule: the trade dates of the book (backtest.run_backtest); the control always trades on
    Mondays. A Sunday with fewer than MIN_NAMES finite values is in `missed`. At a Sunday without
    an eligible coin, the book and the control go to no position (NO_COIN)."""
    s0 = _sunday(start, "start")
    e0 = _sunday(end, "end")
    if e0 < s0 + WEEK:
        raise ValueError(f"end {e0.date()} must be at least 7 days after start {s0.date()}")
    if prices is None:
        prices = data_io.close_matrix()
    if funding is None:
        funding = data_io.funding_matrix(pd.DatetimeIndex(prices.index), prices.columns)
    if eligible is None:
        eligible = labels.eligible(pd.DatetimeIndex(prices.index))
    sundays = pd.date_range(s0, e0 - WEEK, freq="7D")
    v = values.copy()
    v.index = _utc_index(v.index)
    v = v[v.index.isin(sundays)]
    ff = factor_frame(v.reindex(sundays), eligible)

    def run(weights_fn, frame: pd.DataFrame, sched) -> backtest.Result:
        return backtest.run_backtest(
            weights_fn,
            prices=prices,
            funding=funding,
            factors=frame,
            start=str(s0.date()),
            end=str(e0.date()),
            initial_equity=INITIAL_EQUITY,
            fee_bps=spec.fee_bps,
            slippage_bps=spec.slippage_bps,
            gross_cap=GROSS,
            rebalance_weekday=0,
            benchmark=BENCHMARK,
            accounting=ACCOUNTING,
            schedule=sched,
        )

    idx = pd.DatetimeIndex(prices.index)
    window = prices[(idx >= s0) & (idx <= e0)]
    book = _pad_bust(run(book_weights_fn(spec), _every_sunday(ff, sundays), schedule), window)
    control = None
    if spec.control == "equal_weight":
        universe = _every_sunday(universe_frame(eligible, sundays), sundays)
        control = _pad_bust(run(equal_weight, universe, None), window)
    finite = ff[np.isfinite(ff["value"].to_numpy(dtype=np.float64))].groupby("date").size()
    counts = finite.reindex(sundays, fill_value=0)
    missed = [d.strftime("%Y-%m-%d") for d, n in zip(sundays, counts.to_numpy()) if n < MIN_NAMES]
    return BookRun(spec=spec, start=s0, end=e0, book=book, control=control, missed=missed)


# --------------------------------------------------------------------------- weekly series


def weekly_returns(equity: pd.Series, start, end) -> pd.Series:
    """E(S + 7) / E(S) - 1 for the signal Sundays S = start, start + 7, ..., end - 7 (E: the daily
    marks). 0.0 when E(S) <= 0 (after a bust). ValueError when a Sunday row is absent."""
    s0, e0 = _sunday(start, "start"), _sunday(end, "end")
    sundays = pd.date_range(s0, e0 - WEEK, freq="7D")
    eq = equity.copy()
    eq.index = _utc_index(eq.index)
    need = sundays.append(sundays + WEEK).unique()
    absent = [d for d in need if d not in eq.index]
    if absent:
        raise ValueError(f"the equity has no row for the Sunday {absent[0].date()}")
    a = eq.reindex(sundays).to_numpy(dtype=np.float64)
    b = eq.reindex(sundays + WEEK).to_numpy(dtype=np.float64)
    out = np.zeros(len(sundays))
    ok = a > 0
    out[ok] = b[ok] / a[ok] - 1.0
    return pd.Series(out, index=sundays)


WEEKLY_COLUMNS = (
    "sunday",
    "end",
    "book_ret",
    "control_ret",
    "excess",
    "btc_ret",
    "book_equity",
    "control_equity",
    "btc_equity",
    "book_cost",
    "book_funding_usd",
    "book_turnover",
    "rebalanced",
    "held",
)


def weekly_table(run: BookRun) -> pd.DataFrame:
    """One row for each signal Sunday of the run (the index), the columns WEEKLY_COLUMNS.

    control_ret is 0.0 and control_equity NaN for the control "zero". book_cost and
    book_funding_usd are the sums of the days S + 1 .. S + 7, book_turnover the traded dollars of
    those days over E(S). held is False at a Sunday of run.missed."""
    s0, e0 = run.start, run.end
    sundays = pd.date_range(s0, e0 - WEEK, freq="7D")
    ends = sundays + WEEK
    be = run.book.equity["equity"]
    bench = run.book.equity["benchmark"]
    book_ret = weekly_returns(be, s0, e0)
    if run.control is not None:
        ce = run.control.equity["equity"]
        control_ret = weekly_returns(ce, s0, e0)
        control_eq = ce.reindex(ends).to_numpy(dtype=np.float64)
    else:
        control_ret = pd.Series(0.0, index=sundays)
        control_eq = np.full(len(sundays), np.nan)
    daily = run.book.daily
    didx = _utc_index(daily.index)
    # the week of a daily row t: the signal Sunday S with S < t <= S + 7
    wk = np.asarray((didx - s0).days - 1, dtype=np.int64) // 7
    inside = (didx > s0) & (didx <= e0)
    cost = pd.Series(daily["cost"].to_numpy()[inside]).groupby(wk[inside]).sum()
    fund = pd.Series(daily["funding_usd"].to_numpy()[inside]).groupby(wk[inside]).sum()
    n = len(sundays)
    trades = run.book.trades
    notional = np.zeros(n)
    traded = np.zeros(n, dtype=bool)
    if len(trades):
        tidx = _utc_index(trades["date"])
        tw = np.asarray((tidx - s0).days - 1, dtype=np.int64) // 7
        ok = (tidx > s0) & (tidx <= e0)
        sums = pd.Series(trades["notional"].to_numpy(dtype=np.float64)[ok]).groupby(tw[ok]).sum()
        weeks = sums.index.to_numpy(dtype=np.int64)
        notional[weeks] = sums.to_numpy(dtype=np.float64)
        traded[weeks] = True
    e_start = be.reindex(sundays).to_numpy(dtype=np.float64)
    turnover = np.divide(notional, e_start, out=np.zeros(n), where=e_start > 0)
    missed = set(run.missed)
    out = pd.DataFrame(
        {
            "sunday": sundays,
            "end": ends,
            "book_ret": book_ret.to_numpy(),
            "control_ret": control_ret.to_numpy(),
            "excess": book_ret.to_numpy() - control_ret.to_numpy(),
            "btc_ret": weekly_returns(bench, s0, e0).to_numpy(),
            "book_equity": be.reindex(ends).to_numpy(dtype=np.float64),
            "control_equity": control_eq,
            "btc_equity": bench.reindex(ends).to_numpy(dtype=np.float64),
            "book_cost": cost.reindex(range(n), fill_value=0.0).to_numpy(),
            "book_funding_usd": fund.reindex(range(n), fill_value=0.0).to_numpy(),
            "book_turnover": turnover,
            "rebalanced": traded,
            "held": [d.strftime("%Y-%m-%d") not in missed for d in sundays],
        },
        index=sundays,
    )
    return out[list(WEEKLY_COLUMNS)]


def _daily_returns(equity: pd.Series) -> pd.Series:
    r = equity.pct_change()
    prev = equity.shift(1)
    return r.where(~(prev <= 0), 0.0)


def daily_excess(run: BookRun) -> pd.Series:
    """The daily net excess return: the book's daily return minus the control's (0 for the
    control "zero"), without the first row. A day after a bust has the return 0.0."""
    d = _daily_returns(run.book.equity["equity"])
    if run.control is not None:
        d = d - _daily_returns(run.control.equity["equity"]).reindex(d.index)
    return d.iloc[1:]
