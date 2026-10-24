"""AlphaForge expression miner (part of Factorlab), connected to the engine behind a config switch.

Usage::

    python alphaforge_bridge.py --cycle           # forward-score the active formulas, mine when there is a
                                                  # free place and a mining run is due, write
    python alphaforge_bridge.py --mine            # a cycle that mines now when there is a free place
    python alphaforge_bridge.py --refresh         # recompute the table from the zoo, nothing else
    python alphaforge_bridge.py --report          # print the last report, run nothing
    python alphaforge_bridge.py --mine --force    # run although config.ALPHAFORGE_ENABLED is False
    python alphaforge_bridge.py --mine --pop 30 --gens 3   # cheaper search

THE ZOO (data/research/alphaforge_library.json, format 3)
    The zoo keeps each mined formula with its own selection date and its own forward
    record. A zoo file of another format stops each command that reads it, with an error
    (factorfinder/incumbents.py, ZooFormatError). The numbers only, no rule on them:

        forward window   each formula has a fixed forward window of `forward_window_weeks`
                         (52) Sundays, from the first Sunday after its selection date.
                         A forward week is a window Sunday S whose 7-day label is complete:
                         S + 7 <= complete_through (the oldest last bar over the coins that
                         are live, see complete_through()). Its number is the Spearman IC
                         of the formula with fwd_ret_7 on that Sunday.
        active           a formula is active while it has fewer forward weeks than the
                         window. When a cycle first sees the full window, the formula
                         leaves the active list (active False, left_on = the cycle date).
                         Its record stays in the file and is final.
        capacity         at most `max_factors` (6) formulas are active. A cycle mines only
                         when it has a free place and the last mining run is at least
                         `mine_every_days` (7) days before the cycle date; it adds at most
                         as many formulas as there are free places.

    Forward weeks are data that the search never saw.

THE SWITCH (config.py)
    ALPHAFORGE_ENABLED     master switch, default False. Off: --cycle, --mine and
                           --refresh exit 3. On: update.py runs --cycle on each update.
    ALPHAFORGE             dict of miner settings (population, generations, ...).

WHAT IT IS
    alphaforge/ is a genetic-programming expression miner:
    a typed expression language over ts_* / cs_* operators, a fitness function
    (|ICIR| - turnover - redundancy - complexity, TRAIN dates only), a hall of
    fame scored again on VALID dates, greedy de-correlation, and a factor library.
    This module feeds it the project's own point-in-time bars instead of its
    synthetic market, and translates the result into the project's conventions:
    the fwd_ret_7 label and Sunday cross-sections.

THE INCUMBENTS (factorfinder/incumbents.py)
    A new formula is compared with your registered tests (not withdrawn), the active
    AlphaForge formulas and the three risk proxies. The fitness redundancy term uses them,
    and the admission needs |corr| <= max_corr with each of them on the dates up to the end
    of VALID and a mean Sunday |corr| below max_corr after TRAIN. The largest absolute
    correlation with the library definitions is a separate number of each added formula.

ISOLATION
    Writes ONLY data/research/alphaforge_{daily.parquet,library.json,registry.parquet,
    cycles.jsonl,latest.txt} (plus a dated copy of the previous library file).
    test_alphaforge_bridge.py proves it: hash, run, hash again.

POINT-IN-TIME
    Every alphaforge operator is windowed on rows <= t (its own suite proves
    no-look-ahead per operator); cross-sectional operators use row t only.
    The label is exactly fwd_ret_7 of the label table -- close(D+7)/close(D) - 1
    stamped on D -- which is alphaforge's fwd_return(horizon=7, lag=0): bar
    D's close IS the D+1 00:00 UTC price the weekly book trades at.
    test_alphaforge_bridge.py rewrites the future of every raw column and
    requires every earlier row of every terminal and of the table to be
    bit-identical.

STATISTICS
    The miner, the VALID rule and every weekly number use the SPEARMAN rank IC.
    The VALID rule is two-fold: rank ICIR (annualised with 365 // 7 = 52
    independent periods, because daily ICs on a 7-day label overlap) >=
    min_valid_icir_ann, AND the t-stat of the Sunday rank IC on VALID >=
    min_valid_t. The weekly numbers (IC, t, p, q_bh, persistence, coverage, the
    incumbent correlation) are computed on Sundays AFTER the TRAIN block only.
    Hypotheses are counted by VALUE, not spelling (two expressions whose Sunday
    panels correlate > 0.999 are one hypothesis); p_bonferroni multiplies p by the
    larger of the distinct hypotheses and the expressions the search evaluated.
    Mining selects on the whole window up to the selection date, so the in-sample
    numbers of a formula are not out-of-sample numbers; its forward record is.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import alphaforge.ops  # noqa: F401  (registers the operator library)
import config as C
import data_io
from alphaforge.data import DataStore
from alphaforge.evaluation import evaluate_factor, rowwise_corr
from alphaforge.expr import Call, Evaluator, Node, parse, to_string
from alphaforge.library import FactorLibrary
from alphaforge.mining import (
    ExpressionGenerator,
    FitnessFunction,
    GeneticMiner,
    select_uncorrelated,
    spearman_corr_panel,
)
from alphaforge.preprocess import preprocess_factor
from factorfinder import incumbents as INC

# --- the only paths this module writes ----------------------------------------
RESEARCH = C.RESEARCH
TABLE = RESEARCH / "alphaforge_daily.parquet"
LIBRARY = RESEARCH / "alphaforge_library.json"
REGISTRY = RESEARCH / "alphaforge_registry.parquet"
LATEST = RESEARCH / "alphaforge_latest.txt"
CYCLES = RESEARCH / "alphaforge_cycles.jsonl"  # append-only, one line per --cycle
ZOO_FORMAT = INC.ZOO_FORMAT  # 3: the one format that this module writes and reads
LABEL_TEXT = "fwd_ret_7 = close(D+7)/close(D) - 1 stamped on D (the label table)"
STATISTIC_TEXT = "Spearman rank IC everywhere; ICIRs annualised with 365 // horizon periods"

# --- conventions ----------------------------------------------------------------
LAG = 0  # the label table convention: row D is acted on at bar D's close
PREFIX = "af"  # output columns: af_01, af_02, ...
GRACE_DAYS = 7  # a coin with no bar for more than this many days does not hold complete_through back
#: store fields offered to the miner as terminals -- SCALE-FREE ones only.
#: Raw price and coin-volume levels are not comparable across symbols, so a
#: cross-sectional rank of `volume` (in coins) or `close` (in dollars) ranks the
#: coins by their unit size. Levels stay in the store for the label; the miner
#: sees ratios, changes and rates.
MINING_FIELDS = ("returns", "range", "gap", "hl_pos", "log_amount", "amount_ratio_30", "funding")
#: no industry / market-cap neutralisation on a 20-name crypto cross-section;
#: missing stays missing so the coverage number means something
PREPROCESS = {"winsor": "mad", "fill": None, "neutralize": (), "standardize": True}
SAME_HYPOTHESIS_CORR = 0.999  # two expressions this correlated are one hypothesis
#: the settings of the zoo that its file records
ZOO_SETTINGS = ("max_factors", "mine_every_days", "forward_window_weeks")

DEFAULTS = {
    "horizon": 7,  # = fwd_ret_7, one Monday-to-Monday holding week
    "population": 80,
    "generations": 10,
    "max_depth": 3,
    "hall_of_fame": 40,
    "max_factors": 6,  # at most 6 active formulas
    "max_corr": 0.5,  # the largest |corr| with an incumbent that a new formula may have
    "min_valid_icir_ann": 0.5,  # VALID rule 1: Spearman rank ICIR, 52-period annualised
    "min_valid_t": 1.5,  # VALID rule 2: t-stat of the Sunday rank IC on VALID
    "seed": 42,
    "n_jobs": 1,
    "train_frac": 0.6,
    "valid_frac": 0.2,
    "min_coverage": 0.8,
    # FitnessFunction annualises its |ICIR| term with 252 periods; on a 7-day label
    # the honest count is 52, so the turnover and complexity penalties are scaled by
    # sqrt(252 / 52) ~ 2.2 to keep the churn penalty at full strength.
    "w_turnover": 1.1,
    "parsimony": 0.022,
    "n_quantiles": 5,  # quintiles, like the project's own books (20 names)
    # --- the zoo: cadence and the forward window ---
    "mine_every_days": 7,  # at most one mining run in this many days, and only with a free place
    "forward_window_weeks": 52,  # the fixed forward window of each formula
}

log = logging.getLogger("factorlab.alphaforge_bridge")


def _count(n, one: str, many: str | None = None) -> str:
    """'1 formula', '2 formulas': the number and the noun in the form that agrees with it."""
    return f"{n} {one if n == 1 else (many or one + 's')}"


def periods_per_year(horizon: int) -> int:
    """Independent label periods per year for annualising an ICIR.

    Daily ICs on an h-day forward return overlap h-fold and are strongly
    autocorrelated, so annualising with 365 overstates the ICIR by ~sqrt(h).
    Using 365 // h (52 for the weekly label) makes `min_valid_icir_ann`
    comparable to a weekly-frequency ICIR.
    """
    return max(1, 365 // int(horizon))


# --- switch -------------------------------------------------------------------


def enabled() -> bool:
    """The master switch, read from config.py at call time."""
    return bool(getattr(C, "ALPHAFORGE_ENABLED", False))


def settings(**overrides) -> dict:
    """DEFAULTS <- config.ALPHAFORGE <- explicit overrides (None values ignored)."""
    s = dict(DEFAULTS)
    s.update(getattr(C, "ALPHAFORGE", None) or {})
    s.update({k: v for k, v in overrides.items() if v is not None})
    return s


# --- data: project bars -> alphaforge DataStore -------------------------------


def _naive(idx) -> pd.DatetimeIndex:
    """UTC-aware -> naive UTC, ns resolution (alphaforge's grid)."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return pd.DatetimeIndex(idx.as_unit("ns"), name="date")


def _aware(values) -> pd.Series:
    """naive UTC -> the project's on-disk convention, datetime64[ms, UTC]."""
    return pd.to_datetime(values).dt.tz_localize("UTC").dt.as_unit("ms")


def build_store(prices: pd.DataFrame | None = None) -> DataStore:
    """Wrap the project's raw bars in an alphaforge DataStore (point-in-time by construction).

    Fields: open/high/low/close/volume from data_io.ohlcv, amount = close*volume,
    returns = close/close[-1] - 1, funding = the spliced daily funding sum
    (NaN where neither venue covers the day), tradable = 1.0 where a close and
    positive volume exist, plus the scale-free terminals the miner is allowed
    to use: range = (high-low)/close, gap = open/close[-1] - 1,
    hl_pos = (close-low)/(high-low), log_amount = log(dollar volume),
    amount_ratio_30 = amount / its trailing 30-day mean.
    Everything at row D is bar D, i.e. known at D+1 00:00 UTC. Pass ``prices``
    to restrict the grid (e.g. a truncated close matrix for a point-in-time test).
    """
    if prices is None:
        prices = data_io.close_matrix()
    prices = prices.sort_index()
    syms = list(prices.columns)
    idx = prices.index

    def per(col: str) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=idx, columns=syms, dtype=float)
        for s in syms:
            o = data_io.ohlcv(s)
            if o is not None and col in o.columns:
                out[s] = o[col].reindex(idx).astype(float)
        return out

    close = prices.astype(float)
    open_, high, low, volume = per("open"), per("high"), per("low"), per("volume")
    amount = close * volume
    spread = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        fields = {
            "close": close,
            "open": open_,
            "high": high,
            "low": low,
            "volume": volume,
            "amount": amount,
            "returns": close / close.shift(1) - 1.0,
            "range": spread / close,
            "gap": open_ / close.shift(1) - 1.0,
            "hl_pos": ((close - low) / spread).where(spread > 0),
            "log_amount": np.log(amount.where(amount > 0)),
            "amount_ratio_30": amount / amount.rolling(30, min_periods=30).mean(),
            "funding": data_io.observed_funding_matrix(idx, syms).astype(float),
            "tradable": (close.notna() & (volume > 0)).astype(float),
        }
    naive = _naive(idx)
    for name, df in fields.items():
        df = df.copy()
        df.index = naive
        fields[name] = df
    return DataStore(fields, meta={"source": "factorlab data_io", "tz": "UTC", "data_through": str(naive[-1].date())})


def _universe(store: DataStore) -> pd.DataFrame:
    tr = store.get("tradable")
    return tr.where(tr > 0)


def factor_values(node: Node | str, evaluator: Evaluator, store: DataStore, universe: pd.DataFrame) -> pd.DataFrame:
    """Evaluate an expression and pre-process it: universe mask -> MAD winsor -> z-score.

    Same treatment for a hall-of-fame candidate and for the zoo, so the table
    is a slice of the panel the miner scored.
    """
    if isinstance(node, str):
        node = parse(node)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw = evaluator.eval(node)
        out = preprocess_factor(raw, store, universe=universe, **PREPROCESS)
    arr = out.to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(arr).any():
        out = out.where(np.isfinite(arr))
    return out


def _negate(node: Node) -> Node:
    if isinstance(node, Call) and node.op == "neg" and len(node.args) == 1:
        return node.args[0]
    return Call("neg", (node,))


def hypothesis_key(expr: str) -> str:
    """Spelling-level identity of a hypothesis: the expression up to sign.

    Value-level identity (two spellings whose panels correlate > 0.999) is
    assigned on top of this by :func:`dedup_hypotheses`.
    """
    e = expr.strip()
    return e[4:-1] if e.startswith("neg(") and e.endswith(")") else e


def dedup_hypotheses(
    rows: list[dict],
    panels: dict[str, pd.DataFrame],
    sundays: pd.DatetimeIndex,
    threshold: float = SAME_HYPOTHESIS_CORR,
) -> None:
    """Give value-identical expressions one `hypothesis` key (the first seen wins).

    `max(x, x)`, `cs_zscore(x)` around an already z-scored value, `abs` of a
    positive quantity ... are the same bet spelled differently; counting them
    as separate hypotheses would both inflate the Bonferroni count and hand BH
    several correlated p-values for one idea. Rows arrive in fitness order, so
    the representative is the best-fitting spelling.
    """
    reps: list[tuple[str, pd.DataFrame]] = []
    for r in rows:
        v = panels[r["expr"]].reindex(sundays)
        for key, pv in reps:
            c = spearman_corr_panel(v, pv)
            if math.isfinite(c) and abs(c) > threshold:
                r["hypothesis"] = key
                break
        else:
            reps.append((r["hypothesis"], v))


def _wide(long: pd.DataFrame, col: str, store: DataStore) -> pd.DataFrame:
    w = long.pivot_table(index="date", columns="symbol", values=col, aggfunc="first")
    w.index = _naive(w.index)
    return w.reindex(index=store.dates, columns=store.assets).astype(float)


def _aware_frame(values: pd.DataFrame) -> pd.DataFrame:
    """A store-grid panel (naive UTC dates) on a UTC-aware index."""
    out = values.copy()
    idx = pd.DatetimeIndex(out.index)
    out.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return out


# --- the incumbents (factorfinder/incumbents.py) -----------------------------------


def incumbent_ranks(
    store: DataStore, active: list[dict], evaluator: Evaluator, universe: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict]:
    """The incumbents on the store grid as within-date ranks ({name: panel}) and the count of each
    source.

    The provider gives your registered tests and the three risk proxies. The active zoo
    formulas are computed on this store and handed to the provider (zoo_values), so a formula
    that left the active list in this cycle is not an incumbent. A within-date rank keeps the
    row-wise Spearman correlation of the raw values."""
    zoo_values = {m["name"]: _aware_frame(factor_values(m["expr"], evaluator, store, universe)) for m in active}
    members = INC.incumbents(sources=INC.SOURCES, zoo_values=zoo_values)
    ranks = INC.rank_frames(members, store.dates, columns=list(store.assets))
    return ranks, INC.counts(members)


# --- weekly numbers: Sunday cross-sections, fwd_ret_7 --------------------------------


def sunday_ic(values: pd.DataFrame, fwd: pd.DataFrame, sundays: pd.DatetimeIndex) -> pd.Series:
    """Per-Sunday Spearman IC of ``values`` against the 7-day forward return."""
    return rowwise_corr(values.reindex(sundays), fwd.reindex(sundays), method="spearman").dropna()


def _t_of(ics: pd.Series) -> float:
    n = int(len(ics))
    if n < 2:
        return float("nan")
    sd = float(ics.std(ddof=1))
    return float(ics.mean() / sd * math.sqrt(n)) if sd > 0 else 0.0


MIN_COMMON_SUNDAYS = 20  # weekly_stats(): the fewest Sundays with a correlation to an incumbent


def weekly_stats(
    values: pd.DataFrame,
    fwd: pd.DataFrame,
    sundays: pd.DatetimeIndex,
    incumbents: dict[str, pd.DataFrame],
    n_universe: int,
) -> dict:
    """IC / t / p / persistence / coverage / the incumbent correlation on the given Sundays.

    Spearman IC per Sunday against fwd_ret_7. max_incumbent_corr is the signed mean Sunday
    Spearman correlation with the incumbent of the largest absolute mean (NaN without an
    incumbent with MIN_COMMON_SUNDAYS such Sundays); n_incumbents counts the incumbents with
    them. The caller selects WHICH Sundays to pass."""
    v = values.reindex(sundays)
    ics = sunday_ic(values, fwd, sundays)
    weeks = int(len(ics))
    ic = float(ics.mean()) if weeks else float("nan")
    tstat = _t_of(ics)
    pval = float(2 * student_t.sf(abs(tstat), df=weeks - 1)) if weeks > 1 and math.isfinite(tstat) else float("nan")
    ac = rowwise_corr(v, v.shift(1), method="spearman").dropna()
    persistence = float(ac.mean()) if len(ac) else float("nan")
    coverage = float(v.notna().sum(axis=1).mean() / max(n_universe, 1)) if len(v) else 0.0
    best_corr, best_name, n_inc = float("nan"), "", 0
    for name, r in incumbents.items():
        cc = rowwise_corr(v, r.reindex(sundays), method="spearman").dropna()
        if len(cc) < MIN_COMMON_SUNDAYS:
            continue
        n_inc += 1
        c = float(cc.mean())
        if math.isfinite(c) and (not math.isfinite(best_corr) or abs(c) > abs(best_corr)):
            best_corr, best_name = c, name
    turnover = 1.0 - persistence if math.isfinite(persistence) else float("nan")
    return {
        "weeks": weeks,
        "ic": ic,
        "t": tstat,
        "p": pval,
        "persistence": persistence,
        "turnover": turnover,
        "ic_per_turnover": abs(ic) / max(turnover, 0.01) if math.isfinite(turnover) else float("nan"),
        "coverage": coverage,
        "max_incumbent_corr": best_corr,
        "closest_incumbent": best_name,
        "n_incumbents": n_inc,
    }


def _split(dates: pd.DatetimeIndex, s: dict):
    n = len(dates)
    te = int(round(n * float(s["train_frac"])))
    ve = int(round(n * (float(s["train_frac"]) + float(s["valid_frac"]))))
    te = min(max(te, 1), max(n - 2, 1))
    ve = min(max(ve, te + 1), max(n - 1, te))
    return dates[:te], dates[te:ve], dates[ve:]


def _fin(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _num_or_none(x) -> float | None:
    v = _fin(x)
    return v if math.isfinite(v) else None


# --- the forward window ---------------------------------------------------------------


def first_forward_sunday(selected_on) -> pd.Timestamp:
    """The first Sunday strictly after the selection date (naive UTC)."""
    d = pd.Timestamp(selected_on)
    d = (d.tz_convert("UTC").tz_localize(None) if d.tz is not None else d).normalize()
    return d + pd.Timedelta(days=(6 - d.weekday()) % 7 or 7)


def window_of(selected_on, weeks: int) -> dict:
    """The fixed forward window of a formula: {weeks, first_sunday, last_sunday}."""
    first = first_forward_sunday(selected_on)
    last = first + pd.Timedelta(weeks=int(weeks) - 1)
    return {"weeks": int(weeks), "first_sunday": str(first.date()), "last_sunday": str(last.date())}


def complete_through(store: DataStore) -> pd.Timestamp | None:
    """The oldest last bar over the live coins (naive UTC): a coin is live when its last close is at
    most GRACE_DAYS days before the newest last close. None without a bar."""
    close = store.get("close")
    last = pd.Series({c: close[c].last_valid_index() for c in close.columns}).dropna()
    if last.empty:
        return None
    last = pd.to_datetime(last)
    newest = last.max()
    return pd.Timestamp(last[last >= newest - pd.Timedelta(days=GRACE_DAYS)].min())


def forward_record(
    values: pd.DataFrame,
    fwd: pd.DataFrame,
    selected_on,
    complete: pd.Timestamp | None,
    window_weeks: int,
) -> dict:
    """The forward record on the window Sundays with a complete 7-day label.

    weeks = the window Sundays S of the calendar with S + 7 <= complete (complete_through);
    series = the Spearman IC of each of them that has one; ic, t, p from the series. A Sunday
    on which no coin has a bar is a window week with no IC, so the window always ends."""
    w = window_of(selected_on, window_weeks)
    first, last = pd.Timestamp(w["first_sunday"]), pd.Timestamp(w["last_sunday"])
    wsun = pd.date_range(first, last, freq="7D")
    fsun = wsun[wsun + pd.Timedelta(days=7) <= complete] if complete is not None else wsun[:0]
    ics = sunday_ic(values, fwd, fsun) if len(fsun) else pd.Series(dtype=float)
    k = int(len(ics))
    t = _t_of(ics) if k >= 2 else float("nan")
    ic = float(ics.mean()) if k else float("nan")
    p = float(2 * student_t.sf(abs(t), df=k - 1)) if k > 1 and math.isfinite(t) else float("nan")
    return {
        "weeks": int(len(fsun)),
        "ic": _num_or_none(ic),
        "t": _num_or_none(t),
        "p": _num_or_none(p),
        "first": str(fsun[0].date()) if len(fsun) else None,
        "last": str(fsun[-1].date()) if len(fsun) else None,
        "series": [[str(d.date()), round(float(v), 5)] for d, v in ics.items()],
    }


EMPTY_FORWARD = {"weeks": 0, "ic": None, "t": None, "p": None, "first": None, "last": None, "series": []}


# --- mining -------------------------------------------------------------------


def mine(
    overrides: dict | None = None,
    *,
    store: DataStore | None = None,
    log_fn=print,
    zoo: dict | None = None,
    today: pd.Timestamp | None = None,
) -> dict:
    """Search on TRAIN, apply the VALID rule, de-correlate, score post-train Sundays -> one new mining run.

    The incumbents (incumbent_ranks) are the redundancy anchors of the fitness and of the
    admission; new names continue the zoo's numbering. `max_factors` is the number of free
    places. Returns {"cohort": dict (run metadata), "new_factors": [member dicts],
    "results": DataFrame (one row per valid hall-of-fame member, the trials for the
    registry), "store": DataStore}. Nothing is written here; see run_cycle()."""
    s = settings(**(overrides or {}))
    zoo = zoo or empty_zoo()
    active = active_members(zoo)
    next_no = len(zoo["factors"]) + 1
    cohort_no = len(zoo.get("cohorts") or []) + 1
    today = pd.Timestamp(today) if today is not None else pd.Timestamp(datetime.now(UTC).date())
    t0 = time.perf_counter()
    if store is None:
        store = build_store()
    H = int(s["horizon"])
    ppy = periods_per_year(H)
    ann = math.sqrt(ppy)
    max_corr = float(s["max_corr"])
    universe = _universe(store)
    evaluator = Evaluator(store, cache_size=512)
    train, valid, test = _split(store.dates, s)
    train_store = store.slice(None, train[-1])
    valid_store = store.slice(None, valid[-1])
    fwd = store.fwd_return(H, LAG)
    fwd_valid = valid_store.fwd_return(H, LAG)  # NaN at the VALID tail: no peeking into TEST
    sundays = store.dates[store.dates.weekday == 6]
    valid_sundays = sundays[(sundays >= valid[0]) & (sundays <= valid[-1])]
    post_sundays = sundays[sundays > train[-1]]  # the weekly numbers never see the miner's TRAIN weeks
    log_fn(
        f"alphaforge: {len(store.assets)} names x {len(store.dates)} days "
        f"({store.dates[0].date()} -> {store.dates[-1].date()}); "
        f"train {train[0].date()}..{train[-1].date()} valid ..{valid[-1].date()} test ..{test[-1].date()}; "
        f"weekly numbers on {_count(len(post_sundays), 'post-train Sunday')}"
    )

    # the incumbents: your registered tests, the active zoo formulas, the risk proxies
    ranks, inc_counts = incumbent_ranks(store, active, evaluator, universe)
    library = FactorLibrary(root=None, max_corr=max_corr)
    for name, panel in ranks.items():
        vals = preprocess_factor(panel, store, universe=universe, **PREPROCESS)
        if vals.notna().any().any():
            library.add(name, f"<{name}>", vals, tags=("incumbent",), force=True)
    log_fn(
        f"alphaforge: {_count(len(library), 'incumbent')} loaded as redundancy anchors "
        f"({_count(inc_counts['user'], 'registered test')}, {_count(inc_counts['zoo'], 'active zoo formula')}, "
        f"{_count(inc_counts['proxy'], 'risk proxy', 'risk proxies')})"
    )

    generator = ExpressionGenerator(
        list(MINING_FIELDS), group_fields=(), max_depth=int(s["max_depth"]), seed=int(s["seed"])
    )
    fitness = FitnessFunction(
        store,
        evaluator=evaluator,
        library=library if len(library) else None,
        horizon=H,
        lag=LAG,
        start=train[0],
        end=train[-1],
        universe=universe,
        max_corr=max_corr,
        n_quantiles=int(s["n_quantiles"]),
        min_coverage=float(s["min_coverage"]),
        w_turnover=float(s["w_turnover"]),
        parsimony=float(s["parsimony"]),
        preprocess_kwargs=dict(PREPROCESS),
    )
    miner = GeneticMiner(
        generator,
        fitness,
        population_size=int(s["population"]),
        n_generations=int(s["generations"]),
        hall_of_fame=int(s["hall_of_fame"]),
        seed=int(s["seed"]),
        n_jobs=int(s["n_jobs"]),
    )

    def _on_gen(gen: int, hof) -> None:
        h = miner.history[-1]
        log_fn(
            f"  gen {gen + 1}/{s['generations']}: best {_fin(h['best']):.3f} mean {_fin(h['mean']):.3f} "
            f"valid {h['n_valid']}/{s['population']} hof {len(hof)} [{h['elapsed_s']:.1f}s]  {h['best_expr']}"
        )

    hof = miner.run(callback=_on_gen)
    log_fn(f"alphaforge: {_count(miner.n_evaluations, 'expression')} evaluated, hall of fame {len(hof)}")

    # --- VALID (used once): orientation from TRAIN only; Spearman throughout ------
    rows: list[dict] = []
    oriented: dict[str, pd.DataFrame] = {}
    cands, cvals, cvalid_icir = [], {}, {}
    key_of: dict[str, str] = {}
    for m in hof:
        if not m.result.valid:
            continue
        node = m.node
        vals = factor_values(node, evaluator, store, universe)
        flipped = math.isfinite(_fin(m.result.ic_mean)) and _fin(m.result.ic_mean) < 0.0
        if flipped:
            node = _negate(node)
            vals = -vals
        expr = to_string(node)
        key_of[m.expr] = expr
        rep_v = evaluate_factor(
            vals,
            valid_store,
            name=expr,
            expr=expr,
            horizon=H,
            lag=LAG,
            n_quantiles=int(s["n_quantiles"]),
            start=valid[0],
            end=valid[-1],
            periods_per_year=ppy,
        )
        valid_icir = _fin(rep_v.rank_icir) * ann
        valid_t = _t_of(sunday_ic(vals, fwd_valid, valid_sundays))
        meets = (
            math.isfinite(valid_icir)
            and valid_icir >= float(s["min_valid_icir_ann"])
            and math.isfinite(valid_t)
            and valid_t >= float(s["min_valid_t"])
        )
        rows.append(
            {
                "expr": expr,
                "hypothesis": hypothesis_key(expr),
                "generation": int(m.generation),
                "fitness": _fin(m.fitness),
                "complexity": int(m.result.complexity),
                "flipped": bool(flipped),
                "train_ic_mean": abs(_fin(m.result.ic_mean)),
                "train_icir_ann": abs(_fin(m.result.icir)) * ann,
                "train_turnover": _fin(m.result.turnover),
                "valid_ic_mean": _fin(rep_v.rank_ic_mean),
                "valid_icir_ann": valid_icir,
                "valid_t": valid_t,
                "valid_passed": bool(meets),
                "test_icir_ann": float("nan"),
                "selected": False,
                "library_name": "",
                "max_library_corr": float("nan"),
                "closest_library": "",
            }
        )
        oriented[expr] = vals
        if meets:
            cands.append(m)
            cvals[m.expr] = vals
            cvalid_icir[m.expr] = valid_icir

    # --- the weekly numbers on post-train Sundays; all-window numbers kept aside ---
    row_by_expr = {r["expr"]: r for r in rows}
    for r in rows:
        r.update(weekly_stats(oriented[r["expr"]], fwd, post_sundays, ranks, len(store.assets)))
        w_all = weekly_stats(oriented[r["expr"]], fwd, sundays, {}, len(store.assets))
        r.update(
            {
                "weeks_all": w_all["weeks"],
                "ic_all": w_all["ic"],
                "t_all": w_all["t"],
                "p_all": w_all["p"],
                "persistence_all": w_all["persistence"],
            }
        )
    dedup_hypotheses(rows, oriented, post_sundays)

    # --- admission: the VALID rule, then |corr| < max_corr with each incumbent after TRAIN,
    # --- then greedy de-correlation
    def _low(m) -> bool:
        c = _fin(row_by_expr[key_of[m.expr]]["max_incumbent_corr"])
        return not math.isfinite(c) or abs(c) < max_corr

    low_corr = [m for m in cands if _low(m)]
    for m in cands:
        if m not in low_corr:
            r = row_by_expr[key_of[m.expr]]
            r["library_name"] = f"(|corr| {abs(_fin(r['max_incumbent_corr'])):.2f} with {r['closest_incumbent']})"
    # selection looks only at dates the miner was allowed to see (<= VALID end)
    sel_vals = {k: v.loc[: valid[-1]] for k, v in cvals.items()}
    selected = select_uncorrelated(low_corr, sel_vals, max_corr=max_corr, key=lambda m: cvalid_icir[m.expr])
    chosen: list[tuple[str, pd.DataFrame]] = []
    for m in selected:
        expr = key_of[m.expr]
        if len(chosen) >= int(s["max_factors"]):
            row_by_expr[expr]["library_name"] = "(no free place)"
            continue
        if len(library):
            name_c, c = library.max_corr(sel_vals[m.expr])
            if name_c is not None and abs(_fin(c)) > max_corr:
                row_by_expr[expr]["library_name"] = f"(|corr| {abs(c):.2f} with {name_c} up to the end of VALID)"
                continue
        chosen.append((expr, cvals[m.expr]))
    log_fn(
        f"alphaforge: {_count(len(rows), 'valid hall-of-fame member')} -> {_count(len(cands), 'meets', 'meet')} "
        f"the VALID rule "
        f"(rank ICIR >= {s['min_valid_icir_ann']} and Sunday t >= {s['min_valid_t']}) -> {len(low_corr)} with "
        f"post-train |corr| < {max_corr} to each incumbent -> {len(selected)} uncorrelated -> "
        f"{len(chosen)} added (free places {s['max_factors']})"
    )

    factors_out: list[dict] = []
    run_id = today.strftime("%Y-%m-%d") + "T" + datetime.now(UTC).strftime("%H:%M:%SZ")
    window_weeks = int(s["forward_window_weeks"])
    for i, (expr, vals) in enumerate(chosen, next_no):
        name = f"{PREFIX}_{i:02d}"
        r = row_by_expr[expr]
        r["selected"], r["library_name"] = True, name
        rep_t = evaluate_factor(
            vals,
            train_store,
            name=name,
            expr=expr,
            horizon=H,
            lag=LAG,
            n_quantiles=int(s["n_quantiles"]),
            start=train[0],
            end=train[-1],
            periods_per_year=ppy,
        )
        rep_oos = evaluate_factor(
            vals,
            store,
            name=name,
            expr=expr,
            horizon=H,
            lag=LAG,
            n_quantiles=int(s["n_quantiles"]),
            start=test[0],
            end=test[-1],
            periods_per_year=ppy,
        )
        r["test_icir_ann"] = _fin(rep_oos.rank_icir) * ann
        lib = INC.library_corr(_aware_frame(vals))
        if lib is not None and lib["max_abs"] is not None:
            r["max_library_corr"], r["closest_library"] = float(lib["max_abs"]), str(lib["nearest"])
        factors_out.append(
            {
                "name": name,
                "expr": expr,
                "selected_on": str(today.date()),
                "run_id": run_id,
                "cohort": cohort_no,
                "data_through": str(store.dates[-1].date()),
                "flipped_from_train_sign": bool(r["flipped"]),
                "active": True,
                "left_on": None,
                "window": window_of(today, window_weeks),
                "forward": dict(EMPTY_FORWARD),
                "gates": {
                    "valid_icir": r["valid_icir_ann"],
                    "valid_t": r["valid_t"],
                    "persistence": r["persistence"],
                    "coverage": r["coverage"],
                    "max_incumbent_corr": r["max_incumbent_corr"],
                    "max_library_corr": r["max_library_corr"],
                },
                "library": lib,
                "train": {
                    "rank_ic_mean": _fin(rep_t.rank_ic_mean),
                    "rank_icir_ann": _fin(rep_t.rank_icir) * ann,
                    "turnover": _fin(rep_t.turnover_mean),
                },
                "valid": {
                    "rank_ic_mean": r["valid_ic_mean"],
                    "rank_icir_ann": r["valid_icir_ann"],
                    "sunday_t": r["valid_t"],
                },
                "test": {
                    "rank_ic_mean": _fin(rep_oos.rank_ic_mean),
                    "rank_icir_ann": r["test_icir_ann"],
                    "ls_sharpe": _fin(rep_oos.ls_sharpe),
                },
                "weekly_post_train": {
                    k: r[k]
                    for k in (
                        "weeks",
                        "ic",
                        "t",
                        "p",
                        "persistence",
                        "turnover",
                        "ic_per_turnover",
                        "coverage",
                        "max_incumbent_corr",
                        "closest_incumbent",
                        "n_incumbents",
                    )
                },
                "weekly_all_window": {
                    "weeks": r["weeks_all"],
                    "ic": r["ic_all"],
                    "t": r["t_all"],
                    "p": r["p_all"],
                    "persistence": r["persistence_all"],
                },
            }
        )
        log_fn(f"  {name}: {expr}")
        log_fn(
            f"      VALID rank ICIR {r['valid_icir_ann']:+.2f} t {r['valid_t']:+.2f}  TEST rank ICIR {r['test_icir_ann']:+.2f}  "
            f"post-train Sundays: IC {r['ic']:+.4f} t {r['t']:+.2f} persistence {r['persistence']:.3f} "
            f"coverage {r['coverage']:.2f} incumbent corr {_fin(r['max_incumbent_corr']):+.2f} "
            f"({r['closest_incumbent'] or 'none'})"
        )

    results = pd.DataFrame(rows)
    cohort = {
        "cohort": cohort_no,
        "selected_on": str(today.date()),
        "run_id": run_id,
        "data_through": str(store.dates[-1].date()),
        "horizon": H,
        "lag": LAG,
        "label": LABEL_TEXT,
        "fields": list(MINING_FIELDS),
        "preprocess": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PREPROCESS.items()},
        "settings": s,
        "split": {
            "train": [str(train[0].date()), str(train[-1].date()), int(len(train))],
            "valid": [str(valid[0].date()), str(valid[-1].date()), int(len(valid))],
            "test": [str(test[0].date()), str(test[-1].date()), int(len(test))],
        },
        "post_train_weeks": {
            "frame": "Sundays after the TRAIN block (VALID + TEST)",
            "from": str(post_sundays[0].date()) if len(post_sundays) else None,
            "n": int(len(post_sundays)),
        },
        "statistic": STATISTIC_TEXT,
        "incumbents": sorted(ranks),
        "incumbent_counts": inc_counts,
        "n_evaluations": int(miner.n_evaluations),
        "hall_of_fame": int(len(hof)),
        "n_valid_hof": int(len(rows)),
        "n_valid_rule": int(len(cands)),
        "n_low_corr": int(len(low_corr)),
        "n_uncorrelated": int(len(selected)),
        "n_selected": int(len(chosen)),
        "added": [f["name"] for f in factors_out],
        "history": [
            {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in h.items()}
            for h in miner.history
        ],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.system(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    return {"cohort": cohort, "new_factors": factors_out, "results": results, "store": store}


# --- registry (append-only, distinct-hypothesis count) --------------------------

REGISTRY_COLUMNS = [
    "run_date",
    "run_id",
    "source",
    "n_evaluated_in_run",
    "expr",
    "hypothesis",
    "generation",
    "fitness",
    "complexity",
    "flipped",
    "train_ic_mean",
    "train_icir_ann",
    "train_turnover",
    "valid_ic_mean",
    "valid_icir_ann",
    "valid_t",
    "valid_passed",
    "test_icir_ann",
    "selected",
    "library_name",
    "weeks",
    "ic",
    "t",
    "p",
    "persistence",
    "turnover",
    "ic_per_turnover",
    "coverage",
    "weeks_all",
    "ic_all",
    "t_all",
    "p_all",
    "persistence_all",
    "first_seen",
    "max_incumbent_corr",
    "closest_incumbent",
    "n_incumbents",
    "max_library_corr",
    "closest_library",
]
_TEXT_COLUMNS = ("run_date", "run_id", "source", "expr", "hypothesis", "library_name")
_TEXT_COLUMNS += ("closest_incumbent", "closest_library")
_BOOL_COLUMNS = ("flipped", "valid_passed", "selected", "first_seen")


def load_registry() -> pd.DataFrame:
    if REGISTRY.exists():
        return pd.read_parquet(REGISTRY)
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def _registry_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Every registry column in REGISTRY_COLUMNS order, with one dtype each (an absent column is
    null)."""
    out = rows.copy()
    for col in REGISTRY_COLUMNS:
        if col not in out.columns:
            out[col] = None
    for col in REGISTRY_COLUMNS:
        if col in _TEXT_COLUMNS:
            out[col] = out[col].astype(object).where(out[col].notna(), None)
        elif col in _BOOL_COLUMNS:
            out[col] = out[col].astype("boolean")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out[REGISTRY_COLUMNS]


def record(results: pd.DataFrame, run_date: str, *, run_id: str, n_evaluated: int) -> pd.DataFrame:
    """Append every trial of this run; first_seen marks the hypotheses that are new."""
    reg = load_registry()
    seen = set(reg["hypothesis"]) if len(reg) else set()
    rows = results.copy()
    rows["run_date"] = run_date
    rows["run_id"] = run_id
    rows["source"] = "alphaforge"
    rows["n_evaluated_in_run"] = int(n_evaluated)
    rows["first_seen"] = [h not in seen for h in rows["hypothesis"]]
    new = _registry_frame(rows)
    out = pd.concat([_registry_frame(reg), new], ignore_index=True) if len(reg) else new
    C.guard_write(REGISTRY)  # refused while the data folder is protected
    RESEARCH.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY.with_name(f".{REGISTRY.name}.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(REGISTRY)
    return out


def searched_so_far(registry: pd.DataFrame, cohorts: list[dict] | None = None) -> int:
    """Expressions the search evaluated over all recorded mining runs (an upper bound: runs overlap).

    Each run counts once: from its registry rows (n_evaluated_in_run), or, for a run with no
    registry row (no valid hall-of-fame member), from its record in the zoo (cohorts[*].n_evaluations)."""
    per_run: dict[str, float] = {}
    if len(registry) and {"run_id", "n_evaluated_in_run"} <= set(registry.columns):
        first = registry.dropna(subset=["run_id"]).groupby("run_id")["n_evaluated_in_run"].first()
        per_run.update({str(k): float(v) for k, v in pd.to_numeric(first, errors="coerce").fillna(0).items()})
    for c in cohorts or []:
        rid, n = c.get("run_id"), _fin(c.get("n_evaluations"))
        if rid and str(rid) not in per_run and math.isfinite(n):
            per_run[str(rid)] = n
    return int(sum(per_run.values()))


def registry_counts(registry: pd.DataFrame, results: pd.DataFrame, cohorts: list[dict] | None = None) -> dict:
    """The cumulative counts of the zoo record: value-distinct hypotheses (registry and this run) and
    expressions searched (searched_so_far)."""
    keys: set = set()
    for frame in (registry, results):
        if len(frame) and "hypothesis" in frame.columns:
            keys |= set(frame["hypothesis"].dropna())
    return {"n_distinct_hypotheses": len(keys), "n_expressions_searched": searched_so_far(registry, cohorts)}


def _bh_q(p: np.ndarray) -> np.ndarray:
    """The Benjamini-Hochberg q-values (the library's code)."""
    import libfactors

    p = np.asarray(p, dtype=float)
    return libfactors.bh_qvalues(p) if p.size else np.zeros(0)


def assess(
    results: pd.DataFrame, registry: pd.DataFrame, *, searched_now: int = 0, cohorts: list[dict] | None = None
) -> pd.DataFrame:
    """Numbers only on the post-train weekly numbers: q_bh and p_bonferroni.

    q_bh: Benjamini-Hochberg over the VALUE-distinct hypotheses of this run (duplicates share
    the q-value of their hypothesis). p_bonferroni: min(1, p x max(cumulative distinct
    hypotheses, expressions searched)). The expressions searched: searched_so_far(registry,
    cohorts) + searched_now. Order: added first, then p ascending."""
    if results.empty:
        return results
    r = results.copy()
    counts = registry_counts(registry, r, cohorts)
    n_distinct = counts["n_distinct_hypotheses"]
    n_searched = counts["n_expressions_searched"] + int(searched_now)
    uniq = r.drop_duplicates("hypothesis")
    p_uniq = pd.to_numeric(uniq["p"], errors="coerce").to_numpy(dtype=float)
    p_uniq = np.where(np.isfinite(p_uniq), p_uniq, 1.0)
    q = dict(zip(uniq["hypothesis"], _bh_q(p_uniq)))
    r["q_bh"] = r["hypothesis"].map(q).astype(float)
    divisor = max(n_distinct, n_searched, 1)
    p_all = pd.to_numeric(r["p"], errors="coerce").fillna(1.0)
    r["p_bonferroni"] = np.minimum(1.0, p_all.to_numpy(dtype=float) * divisor)
    r.attrs["n_distinct"] = n_distinct
    r.attrs["n_searched"] = n_searched
    r.attrs["bonferroni_divisor"] = divisor
    return r.sort_values(["selected", "p"], ascending=[False, True], kind="stable")


# --- the zoo file -----------------------------------------------------------------


def empty_zoo() -> dict:
    s = settings()
    return {
        "format": ZOO_FORMAT,
        "run_id": None,
        "data_through": None,
        "horizon": int(s["horizon"]),
        "lag": LAG,
        "label": LABEL_TEXT,
        "fields": list(MINING_FIELDS),
        "statistic": STATISTIC_TEXT,
        "settings": {k: s[k] for k in ZOO_SETTINGS},
        "factors": [],
        "cohorts": [],
        "registry": {},
    }


def load_zoo() -> dict:
    """The zoo of data/research/alphaforge_library.json; an empty zoo if the file does not exist.
    A file of another format than ZOO_FORMAT stops with ZooFormatError."""
    if not LIBRARY.exists():
        return empty_zoo()
    return INC.check_zoo_format(json.loads(LIBRARY.read_text(encoding="utf-8")))


def load_library() -> dict:
    if not LIBRARY.exists():
        raise SystemExit("no AlphaForge zoo file in data/research -- run `python alphaforge_bridge.py --mine`")
    return load_zoo()


def active_members(zoo: dict) -> list[dict]:
    return [m for m in zoo.get("factors", []) if m.get("active") is True]


def update_forward(zoo: dict, store: DataStore, s: dict, *, today: pd.Timestamp) -> list[str]:
    """Compute the forward record of each active formula again; a formula with the full window
    leaves the active list. Returns the names that left."""
    fwd = store.fwd_return(int(zoo.get("horizon", s["horizon"])), LAG)
    complete = complete_through(store)
    universe = _universe(store)
    evaluator = Evaluator(store, cache_size=64)
    left: list[str] = []
    for m in zoo["factors"]:
        if not m.get("active"):
            continue  # a formula that left the active list has a final record
        window = m.get("window") or window_of(m["selected_on"], int(s["forward_window_weeks"]))
        m["window"] = window
        vals = factor_values(m["expr"], evaluator, store, universe)
        m["forward"] = forward_record(vals, fwd, m["selected_on"], complete, int(window["weeks"]))
        if m["forward"]["weeks"] >= int(window["weeks"]):
            m["active"], m["left_on"] = False, str(today.date())
            left.append(m["name"])
    return left


def save_zoo(zoo: dict) -> None:
    """Write the zoo atomically, keeping a dated backup of the previous file."""
    C.guard_write(LIBRARY)  # refused while the data folder is protected
    RESEARCH.mkdir(parents=True, exist_ok=True)
    if LIBRARY.exists():
        try:
            old = json.loads(LIBRARY.read_text(encoding="utf-8"))
            stamp = str(old.get("run_id") or "prev").replace(":", "")
        except (OSError, ValueError):
            stamp = "prev"
        backup = RESEARCH / f"alphaforge_library_{stamp}.json"
        if not backup.exists():
            C.guard_write(backup)
            backup.write_bytes(LIBRARY.read_bytes())
    _atomic_text(LIBRARY, json.dumps(_json_safe(zoo), indent=2, ensure_ascii=False))


def refresh(lib: dict | None = None, *, store: DataStore | None = None, write: bool = True) -> pd.DataFrame:
    """Compute data/research/alphaforge_daily.parquet from the expressions of every formula of the zoo.

    One af_NN column for each formula, active or not. No selection happens here; new bars
    extend every row-D value the same way. A zoo with no formula gives an empty table (the
    columns date and symbol)."""
    if lib is None:
        lib = load_library()
    if store is None:
        store = build_store()
    universe = _universe(store)
    evaluator = Evaluator(store, cache_size=64)
    members = list(lib.get("factors", []))
    names = [f["name"] for f in members]
    long: pd.DataFrame | None = None
    for f in members:
        vals = factor_values(f["expr"], evaluator, store, universe)
        m = vals.reset_index().melt(id_vars="date", var_name="symbol", value_name=f["name"])
        long = m if long is None else long.merge(m, on=["date", "symbol"], how="outer")
    if long is None:
        long = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "symbol": pd.Series(dtype=object)})
    long["date"] = _aware(long["date"])
    long["symbol"] = long["symbol"].astype(str)
    if names:
        long = long.dropna(subset=names, how="all")
    long = long.sort_values(["date", "symbol"]).reset_index(drop=True)
    if long.duplicated(["date", "symbol"]).any():
        raise AssertionError("alphaforge table has duplicate (date, symbol) rows")
    if write:
        C.guard_write(TABLE)  # refused while the data folder is protected
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        tmp = TABLE.with_name(f".{TABLE.name}.tmp")
        long.to_parquet(tmp, index=False)
        tmp.replace(TABLE)
    return long


def table_through() -> pd.Timestamp | None:
    """Last date covered by the table (None if absent or empty)."""
    if not TABLE.exists():
        return None
    d = pd.read_parquet(TABLE, columns=["date"])["date"]
    return None if d.empty else d.max()


# --- outputs & report ------------------------------------------------------------


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, Path)):
        return str(o)
    return o


def _atomic_text(path: Path, text: str) -> None:
    C.guard_write(path)  # refused while the data folder is protected
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _cycle_seed(s: dict, today: pd.Timestamp) -> int:
    """A different, deterministic seed per ISO week so mining runs explore different regions."""
    iso = today.isocalendar()
    return int(s["seed"]) + int(iso[0]) * 100 + int(iso[1])


def run_cycle(
    overrides: dict | None = None,
    *,
    store: DataStore | None = None,
    force_mine: bool = False,
    today=None,
    write: bool = True,
    log_fn=print,
) -> dict:
    """One cycle: forward-score the active formulas -> the window rule -> (maybe) mine -> write.

    Mining happens only with a free place (room = max_factors - active formulas > 0), and
    when it is forced, when the zoo has no mining run, or when the last mining run is at least
    `mine_every_days` old. It adds at most `room` formulas. Nothing about an existing formula
    changes except its forward record and, at the end of its window, its place in the
    active list."""
    s = settings(**(overrides or {}))
    today = pd.Timestamp(today) if today is not None else pd.Timestamp(datetime.now(UTC).date())
    if store is None:
        store = build_store()
    zoo = load_zoo()
    zoo["settings"] = {k: s[k] for k in ZOO_SETTINGS}
    left = update_forward(zoo, store, s, today=today)
    for name in left:
        log_fn(f"alphaforge: {name} has {_count(s['forward_window_weeks'], 'forward week')} and leaves the active list")

    last = zoo["cohorts"][-1]["selected_on"] if zoo["cohorts"] else None
    age = (today - pd.Timestamp(last)).days if last else None
    due = bool(force_mine or last is None or (age is not None and age >= int(s["mine_every_days"])))
    room = int(s["max_factors"]) - len(active_members(zoo))
    mined, res, assessed, reg, seed = False, None, None, load_registry(), None
    if due and room > 0:
        seed = int(overrides["seed"]) if overrides and overrides.get("seed") is not None else _cycle_seed(s, today)
        res = mine(
            {**(overrides or {}), "seed": seed, "max_factors": room}, store=store, log_fn=log_fn, zoo=zoo, today=today
        )
        mined = True
        if write and len(res["results"]):
            reg = record(
                res["results"],
                res["cohort"]["selected_on"],
                run_id=res["cohort"]["run_id"],
                n_evaluated=res["cohort"]["n_evaluations"],
            )
        # the run's record in the zoo: searched_so_far counts it once, from the registry or from here
        zoo["cohorts"].append({k: v for k, v in res["cohort"].items() if k != "history"})
        assessed = assess(res["results"], reg, cohorts=zoo["cohorts"])
        by_name = (
            {r["library_name"]: r for r in assessed.to_dict("records") if r.get("selected")} if len(assessed) else {}
        )
        for m in res["new_factors"]:
            r = by_name.get(m["name"])
            if r is not None:
                m["weekly_post_train"]["q_bh"] = r["q_bh"]
                m["weekly_post_train"]["p_bonferroni"] = r["p_bonferroni"]
            zoo["factors"].append(m)
        zoo["registry"] = registry_counts(reg, res["results"], zoo["cohorts"])
        zoo["run_id"], zoo["data_through"] = res["cohort"]["run_id"], res["cohort"]["data_through"]
    elif due:
        log_fn(
            f"alphaforge: mining is due, but no place is free ({_count(len(active_members(zoo)), 'active formula')}, "
            f"at most {s['max_factors']})"
        )
    else:
        log_fn(f"alphaforge: last mining run {last} ({age} d ago); next mining in {int(s['mine_every_days']) - age} d")

    added = [m["name"] for m in (res["new_factors"] if res else [])]
    # the counts of the cycle line are the counts after the cycle: active + room = max_factors
    n_active = len(active_members(zoo))
    room_after = max(int(s["max_factors"]) - n_active, 0)
    table = refresh(zoo, store=store, write=write)
    text = render(
        zoo,
        assessed,
        cycle={
            "today": str(today.date()),
            "mined": mined,
            "due": due,
            "room_before": room,
            "room": room_after,
            "active": n_active,
            "added": added,
            "left": left,
        },
    )
    line = {
        "date": str(today.date()),
        "data_through": str(store.dates[-1].date()),
        "mined": mined,
        "added": added,
        "left": left,
        "active": n_active,
        "room": room_after,
        "n_evaluated": int(res["cohort"]["n_evaluations"]) if res else 0,
        "seed": seed,
    }
    if write:
        save_zoo(zoo)
        _atomic_text(LATEST, text)
        C.guard_write(CYCLES)
        RESEARCH.mkdir(parents=True, exist_ok=True)
        with CYCLES.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_json_safe(line), ensure_ascii=False) + "\n")
    return {
        "zoo": zoo,
        "results": res["results"] if res else pd.DataFrame(),
        "assessed": assessed,
        "table": table,
        "mined": mined,
        "added": added,
        "left": left,
        "report": text,
        "line": line,
    }


def _zoo_lines(zoo: dict) -> list[str]:
    st = zoo.get("settings", {})
    act = active_members(zoo)
    L = [
        f"=== ALPHAFORGE -- {_count(len(act), 'active formula')} (at most {st.get('max_factors')}), "
        f"{_count(len(zoo.get('factors', [])), 'formula')} in the zoo, "
        f"{_count(len(zoo.get('cohorts', [])), 'mining run')} ===",
        "",
        f"  forward window: {_count(st.get('forward_window_weeks'), 'week')} for each formula; at most one mining "
        f"run in {_count(st.get('mine_every_days'), 'day')}, and only with a free place",
        "  forward week: a window Sunday after the selection date with a complete 7-day label (data that the "
        "search never saw)",
        "",
    ]
    L.append(
        f"{'name':<6}{'active':>7}{'selected':>12}{'window end':>12}{'fwd wk':>7}{'fwd IC':>9}{'fwd t':>7}"
        f"{'  | post-train: IC':>19}{'t':>7}{'persist':>9}{'inc corr':>9}  expr"
    )
    L.append("-" * 118)
    for m in zoo.get("factors", []):
        fw = m.get("forward", {}) or {}
        w = m.get("weekly_post_train", {}) or {}
        win = m.get("window") or {}
        L.append(
            f"{m['name']:<6}{('yes' if m.get('active') else 'no'):>7}{str(m.get('selected_on')):>12}"
            f"{str(win.get('last_sunday')):>12}{int(fw.get('weeks') or 0):>7}"
            f"{_fin(fw.get('ic')):>+9.3f}{_fin(fw.get('t')):>+7.2f}{_fin(w.get('ic')):>+19.4f}{_fin(w.get('t')):>+7.2f}"
            f"{_fin(w.get('persistence')):>9.3f}{_fin(w.get('max_incumbent_corr')):>+9.2f}  {m['expr']}"
        )
        if not m.get("active"):
            L.append(f"{'':<6}  left the active list on {m.get('left_on')}")
    if not zoo.get("factors"):
        L.append("  (empty -- no mining run)")
    return L


def render(zoo: dict, r: pd.DataFrame | None, cycle: dict | None = None) -> str:
    """The report of a cycle: numbers only."""
    L = _zoo_lines(zoo)
    cycle = cycle or {}
    L += [""]
    if cycle:
        state = "ran" if cycle.get("mined") else ("due, no free place" if cycle.get("due") else "not due")
        L.append(
            f"  cycle {cycle.get('today')}: mining {state} · added {cycle.get('added') or []} · "
            f"left {cycle.get('left') or []} · after the cycle: {cycle.get('active')} active, "
            f"{_count(cycle.get('room'), 'free place')}"
        )
    coh = zoo["cohorts"][-1] if zoo.get("cohorts") else None
    if r is not None and len(r) and coh is not None:
        s = coh.get("settings", {})
        pw = coh.get("post_train_weeks", {})
        inc = coh.get("incumbent_counts", {})
        reg = zoo.get("registry", {})
        L += [
            "",
            f"--- newest mining run {coh.get('selected_on')} (run {coh.get('run_id')}), data through {coh.get('data_through')} ---",
            f"  split train {coh['split']['train'][0]}..{coh['split']['train'][1]} · valid ..{coh['split']['valid'][1]} · test ..{coh['split']['test'][1]}"
            f" · pop {s.get('population')} x {s.get('generations')} gens, seed {s.get('seed')} · {_count(coh.get('n_evaluations'), 'expression')} in {coh.get('elapsed_s', 0):.0f}s",
            f"  hall of fame {coh.get('hall_of_fame')} -> {coh.get('n_valid_hof')} valid -> "
            f"{_count(coh.get('n_valid_rule'), 'meets', 'meet')} the VALID rule "
            f"(rank ICIR >= {s.get('min_valid_icir_ann')} and Sunday t >= {s.get('min_valid_t')}) -> {coh.get('n_low_corr')} with "
            f"|corr| < {s.get('max_corr')} to each incumbent -> {coh.get('n_uncorrelated')} uncorrelated -> {coh.get('n_selected')} added "
            f"(free places before the run {cycle.get('room_before', '?')})",
            f"  incumbents: {_count(inc.get('user', 0), 'registered test')}, "
            f"{_count(inc.get('zoo', 0), 'active zoo formula')}, {_count(inc.get('proxy', 0), 'risk proxy', 'risk proxies')}",
            f"  weekly numbers on {_count(pw.get('n', '?'), 'post-train Sunday')} from {pw.get('from', '?')} · value-distinct hypotheses "
            f"{reg.get('n_distinct_hypotheses')} · expressions searched {reg.get('n_expressions_searched')}",
            "",
        ]
        L.append(
            f"{'name':<6}{'V':>2}{'valIR':>7}{'valT':>6}{'tstIR':>7}{'IC':>8}{'t':>7}{'q_bh':>7}{'p_bonf':>8}"
            f"{'persist':>9}{'inc corr':>9}  expr"
        )
        L.append("-" * 104)
        for row in r.head(14).itertuples():
            name = row.library_name if row.selected else "-"
            L.append(
                f"{name:<6}{('V' if row.valid_passed else '-'):>2}{_fin(row.valid_icir_ann):>+7.2f}{_fin(row.valid_t):>+6.1f}"
                f"{_fin(row.test_icir_ann):>+7.2f}{_fin(row.ic):>+8.4f}{_fin(row.t):>+7.2f}{_fin(row.q_bh):>7.3f}"
                f"{_fin(row.p_bonferroni):>8.3f}{_fin(row.persistence):>9.3f}{_fin(row.max_incumbent_corr):>+9.2f}  {row.expr}"
            )
        L.append("  V: the formula meets the VALID rule.")
    L += [
        "",
        "  writes: data/research/alphaforge_daily.parquet and data/research/alphaforge_*.",
        f"  switch: ALPHAFORGE_ENABLED={enabled()}",
    ]
    return "\n".join(L)


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--cycle",
        action="store_true",
        help="forward-score the active formulas, mine when a run is due and a place is free, write",
    )
    g.add_argument(
        "--mine",
        action="store_true",
        help="a cycle that mines now when a place is free; the window rule applies as in --cycle, "
        "and the zoo keeps the record of each formula",
    )
    g.add_argument("--refresh", action="store_true", help="rebuild the table from the zoo only")
    g.add_argument("--report", action="store_true", help="print the last report, run nothing")
    ap.add_argument("--force", action="store_true", help="run even while config.ALPHAFORGE_ENABLED is False")
    ap.add_argument("--no-write", action="store_true", help="compute and print; write no file")
    ap.add_argument("--pop", type=int, default=None, help="population size of the miner (default: config.ALPHAFORGE)")
    ap.add_argument("--gens", type=int, default=None, help="generations of the miner (default: config.ALPHAFORGE)")
    ap.add_argument("--seed", type=int, default=None, help="random seed of the miner (default: config.ALPHAFORGE)")
    ap.add_argument("--max-factors", type=int, default=None, help="the largest number of active formulas (default 6)")
    ap.add_argument("--today", default=None, help="override the cycle date (YYYY-MM-DD); for tests and replays")
    ap.add_argument("--verbose", action="store_true", help="show the progress messages of the miner")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    if not a.verbose:
        logging.getLogger("alphaforge").setLevel(logging.WARNING)

    if a.report:
        if LATEST.exists():
            print(LATEST.read_text(encoding="utf-8"))
            return 0
        print("no AlphaForge report -- run `python alphaforge_bridge.py --cycle`")
        return 0

    if not enabled() and not a.force:
        print(
            "AlphaForge is off (config.ALPHAFORGE_ENABLED = False). It ran nothing and wrote nothing.\n"
            "  To use it, set ALPHAFORGE_ENABLED = True in config.py.\n"
            "  To run it one time, add --force."
        )
        return 3

    try:
        load_zoo()  # a zoo file of another format stops the command here, before any computation
    except INC.ZooFormatError as exc:
        print(f"alphaforge: {exc}", file=sys.stderr)
        return 1

    if a.refresh:
        zoo = load_library()
        table = refresh(zoo, write=not a.no_write)
        n = int(table["date"].nunique()) if len(table) else 0
        last = str(table["date"].max().date()) if len(table) else "n/a"
        print(
            f"alphaforge refresh: {_count(len(zoo.get('factors', [])), 'formula')} ({len(active_members(zoo))} active), "
            f"{len(table):,} {'row' if len(table) == 1 else 'rows'} over {_count(n, 'date')}, latest {last}"
            + ("" if a.no_write else f" -> data/research/{TABLE.name}")
        )
        return 0

    out = run_cycle(
        {"population": a.pop, "generations": a.gens, "seed": a.seed, "max_factors": a.max_factors},
        force_mine=bool(a.mine),
        today=a.today,
        write=not a.no_write,
    )
    print()
    print(out["report"])
    if not a.no_write:
        print(f"\nwrote data/research/: {LIBRARY.name}, {CYCLES.name}, {REGISTRY.name}, {TABLE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
