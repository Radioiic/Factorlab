"""Combinatorial generator for the crypto factor library.

Each *family* below is a grammar of economically-motivated templates crossed
with input series and look-back windows.  ``generate_factors()`` returns a
de-duplicated list of :class:`FactorSpec`; the build step (``build.py``)
then evaluates every candidate on data and drops degenerate or redundant
ones.

Windows are expressed in bars, so the same library works for daily, 4h or
1h data (on daily bars ``w=7`` is one week, on hourly bars it is 7 hours).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .expr import parse

WINDOWS = [3, 5, 7, 10, 14, 20, 30, 45, 60, 90, 120, 180, 240, 360]
CORR_WINDOWS = [5, 7, 10, 14, 20, 30, 45, 60, 90, 120, 180, 240, 360]


@dataclass
class FactorSpec:
    """One candidate factor: name, family, category, expression, description, main window and parameters."""

    name: str
    family: str
    category: str
    expression: str
    description: str
    window: int | None = None
    params: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# input series
# --------------------------------------------------------------------------
# stationary per-bar series: (name, category, signed?)
STAT_SERIES = [
    ("ret", "momentum", True),
    ("oc_ret", "momentum", True),
    ("gap", "momentum", True),
    ("ex_ret", "momentum", True),
    ("ex_btc_ret", "momentum", True),
    ("abs_ret", "volatility", False),
    ("hl_range", "volatility", False),
    ("tr_pct", "volatility", False),
    ("gk_var", "volatility", False),
    ("rs_var", "volatility", False),
    ("body", "price_pattern", False),
    ("upper_shadow", "price_pattern", False),
    ("lower_shadow", "price_pattern", False),
    ("close_loc", "price_pattern", False),
    ("clv", "price_pattern", True),
    ("vwap_dev", "price_pattern", True),
    ("dvol_chg", "liquidity", True),
    ("amihud", "liquidity", False),
    ("signed_dvol", "liquidity", True),
    ("taker_imb", "order_flow", True),
    ("net_taker_dvol", "order_flow", True),
    ("funding", "funding", True),
    ("funding_premium_gap", "funding", True),
    ("oi_chg", "open_interest", True),
    ("oi_coin_chg", "open_interest", True),
    ("oi_turnover", "open_interest", True),
    ("ls_ratio", "positioning", True),
    ("top_ls_ratio", "positioning", True),
    ("ls_divergence", "positioning", True),
    ("taker_bs", "positioning", True),
    ("basis", "basis", True),
]

# level series: (expression of the level, short label, category)
LEVEL_SERIES = [
    ("close", "close", "trend"),
    ("vwap", "vwap", "trend"),
    ("high", "high", "trend"),
    ("low", "low", "trend"),
    ("quote_volume", "dvol", "liquidity"),
    ("volume", "volume", "liquidity"),
    ("trades", "trades", "liquidity"),
    ("trade_size", "tsize", "liquidity"),
    ("open_interest_value", "oi", "open_interest"),
    ("open_interest", "oi_coin", "open_interest"),
]

CATEGORY_OF = {name: cat for name, cat, _ in STAT_SERIES}


_MIN_WINDOW = {"skew": 10, "kurt": 14, "autocorr": 14, "slope": 5, "std": 5, "zscore": 5, "ir": 5}


class _Builder:
    """Collect specs and skip a spec whose expression is already in the list."""

    def __init__(self) -> None:
        self.specs: list[FactorSpec] = []
        self._seen_expr: set[str] = set()
        self._seen_name: set[str] = set()

    def add(self, name, family, category, expression, description, window=None, **params):
        """Add one spec, unless its expression is already in the list. A repeated name gives ValueError."""
        canon = _canonical(expression)
        if canon in self._seen_expr:
            return
        if name in self._seen_name:
            raise ValueError(f"duplicate factor name {name}")
        self._seen_expr.add(canon)
        self._seen_name.add(name)
        self.specs.append(FactorSpec(name, family, category, expression, description, window, params))


def _canonical(expression: str) -> str:
    """Return the dump of the parsed expression: the key that finds a repeated expression."""
    import ast

    return ast.dump(parse(expression))


# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------
STAT_OPS = [
    # op, template, description
    ("mean", "ts_mean({x}, {w})", "rolling mean of {x}"),
    ("ema", "ema({x}, {w})", "exponential moving average of {x}"),
    ("decay", "decay_linear({x}, {w})", "linearly decayed mean of {x}"),
    ("median", "ts_median({x}, {w})", "rolling median of {x}"),
    ("std", "ts_std({x}, {w})", "rolling volatility of {x}"),
    ("zscore", "ts_zscore({x}, {w})", "time-series z-score of {x}"),
    ("demean", "{x} - ts_mean({x}, {w})", "{x} minus its rolling mean"),
    ("rank", "ts_rank({x}, {w})", "time-series percentile rank of {x}"),
    ("minmax", "ts_minmax({x}, {w})", "position of {x} in its rolling range"),
    ("max", "ts_max({x}, {w})", "rolling maximum of {x}"),
    ("min", "ts_min({x}, {w})", "rolling minimum of {x}"),
    ("ir", "div(ts_mean({x}, {w}), ts_std({x}, {w}))", "rolling mean/std (information ratio) of {x}"),
    ("skew", "ts_skew({x}, {w})", "rolling skewness of {x}"),
    ("kurt", "ts_kurt({x}, {w})", "rolling kurtosis of {x}"),
    ("autocorr", "ts_autocorr({x}, {w}, 1)", "lag-1 autocorrelation of {x}"),
    ("slope", "ts_slope({x}, {w})", "linear trend slope of {x}"),
    ("argmax", "ts_argmax({x}, {w})", "bars since rolling max of {x}"),
    ("argmin", "ts_argmin({x}, {w})", "bars since rolling min of {x}"),
    ("posfrac", "ts_pos_frac({x}, {w})", "fraction of bars with {x} > 0"),
]
_SIGNED_ONLY = {"posfrac", "ir"}


def family_stat(b: _Builder):
    """Add the stat_rolling family: rolling statistics (STAT_OPS) of the stationary series over WINDOWS."""
    for x, cat, signed in STAT_SERIES:
        for op, tpl, desc in STAT_OPS:
            if op in _SIGNED_ONLY and not signed:
                continue
            for w in WINDOWS:
                if w < _MIN_WINDOW.get(op, 3):
                    continue
                b.add(
                    f"stat.{op}.{x}.w{w}",
                    "stat_rolling",
                    cat,
                    tpl.format(x=x, w=w),
                    desc.format(x=x) + f" over {w} bars",
                    w,
                    series=x,
                    op=op,
                )


LEVEL_OPS = [
    ("mom", "log(div({x}, delay({x}, {w})))", "log change of {x}"),
    ("ma_dev", "div({x}, ts_mean({x}, {w})) - 1", "{x} relative to its moving average"),
    ("ema_dev", "div({x}, ema({x}, {w})) - 1", "{x} relative to its EMA"),
    ("zscore", "ts_zscore(log({x}), {w})", "z-score of log {x}"),
    ("rank", "ts_rank({x}, {w})", "time-series rank of {x}"),
    ("minmax", "ts_minmax({x}, {w})", "position of {x} inside its rolling range"),
    ("hi_dev", "div({x}, ts_max({x}, {w})) - 1", "distance of {x} from rolling high"),
    ("lo_dev", "div({x}, ts_min({x}, {w})) - 1", "distance of {x} from rolling low"),
    ("slope", "ts_slope(log({x}), {w})", "log-linear trend slope of {x}"),
    ("rsq", "ts_rsq(log({x}), {w})", "trend R-squared of log {x}"),
    (
        "tstat",
        "ts_slope(log({x}), {w}) * sqrt_w_{w} / ts_std(log(div({x}, delay({x}, 1))), {w})",
        "trend slope scaled by volatility of {x}",
    ),
    ("vol", "ts_std(log(div({x}, delay({x}, 1))), {w})", "volatility of log changes of {x}"),
    ("argmax", "ts_argmax({x}, {w})", "bars since rolling high of {x}"),
    ("argmin", "ts_argmin({x}, {w})", "bars since rolling low of {x}"),
    ("maxdd", "ts_maxdd({x}, {w})", "maximum drawdown of {x}"),
    ("maxru", "ts_maxru({x}, {w})", "maximum run-up of {x}"),
]


def family_level(b: _Builder):
    """Add the level_transform family: changes, MA distances, range positions, trends and drawdowns of levels."""
    for x, label, cat in LEVEL_SERIES:
        for op, tpl, desc in LEVEL_OPS:
            for w in WINDOWS:
                if op in ("rsq",) and w < 7:
                    continue
                if op in ("zscore", "vol", "tstat", "slope") and w < 5:
                    continue
                expr = tpl.format(x=x, w=w).replace(f"sqrt_w_{w}", f"{math.sqrt(w):.6f}")
                b.add(
                    f"level.{op}.{label}.w{w}",
                    "level_transform",
                    cat,
                    expr,
                    desc.format(x=label) + f" over {w} bars",
                    w,
                    series=x,
                    op=op,
                )


def _window_pairs(windows, ratio=2):
    """Return the (short, long) window pairs with long >= ratio x short."""
    return [(s, lw) for s in windows for lw in windows if lw >= ratio * s]


def family_ma_cross(b: _Builder):
    """Add the ma_crossover family: short/long SMA and EMA ratios of levels and short-minus-long means of series."""
    pairs = _window_pairs(WINDOWS)
    for x, label, cat in LEVEL_SERIES:
        if label in ("high", "low"):
            continue
        for s, lw in pairs:
            b.add(
                f"cross.sma.{label}.{s}_{lw}",
                "ma_crossover",
                cat,
                f"log(div(ts_mean({x}, {s}), ts_mean({x}, {lw})))",
                f"SMA({s})/SMA({lw}) crossover of {label}",
                lw,
                series=x,
                short=s,
                long=lw,
            )
            b.add(
                f"cross.ema.{label}.{s}_{lw}",
                "ma_crossover",
                cat,
                f"log(div(ema({x}, {s}), ema({x}, {lw})))",
                f"EMA({s})/EMA({lw}) crossover of {label}",
                lw,
                series=x,
                short=s,
                long=lw,
            )
    stat_cross = [
        "ret",
        "taker_imb",
        "net_taker_dvol",
        "funding",
        "oi_chg",
        "ls_ratio",
        "top_ls_ratio",
        "taker_bs",
        "basis",
        "vwap_dev",
        "clv",
        "hl_range",
        "amihud",
        "dvol_chg",
        "abs_ret",
    ]
    for x in stat_cross:
        for s, lw in pairs:
            b.add(
                f"cross.diff.{x}.{s}_{lw}",
                "ma_crossover",
                CATEGORY_OF[x],
                f"ts_mean({x}, {s}) - ts_mean({x}, {lw})",
                f"short minus long average of {x} ({s} vs {lw})",
                lw,
                series=x,
                short=s,
                long=lw,
            )


def family_momentum(b: _Builder):
    """Add the momentum family: skip-period, risk-adjusted, excess and weighted momentum, and efficiency ratios."""
    skips = [0, 1, 2, 3, 5, 7, 10, 14, 20, 30]
    for w in WINDOWS:
        if w < 5:
            continue
        for k in skips:
            if k == 0 or k > w // 2:
                continue
            n = w - k
            for p in ("close", "vwap"):
                b.add(
                    f"mom.skip.{p}.w{w}_k{k}",
                    "momentum",
                    "momentum",
                    f"log(div(delay({p}, {k}), delay({p}, {w})))",
                    f"{p} momentum from t-{w} to t-{k}",
                    w,
                    price=p,
                    skip=k,
                )
                b.add(
                    f"mom.riskadj.{p}.w{w}_k{k}",
                    "momentum",
                    "momentum",
                    f"div(log(div(delay({p}, {k}), delay({p}, {w}))), ts_std(logret, {w}) * {math.sqrt(n):.6f})",
                    f"volatility-adjusted {p} momentum t-{w}..t-{k}",
                    w,
                    price=p,
                    skip=k,
                )
            for r, lbl in (("ex_ret", "market"), ("ex_btc_ret", "btc")):
                b.add(
                    f"mom.excess_{lbl}.w{w}_k{k}",
                    "momentum",
                    "momentum",
                    f"ts_sum(delay({r}, {k}), {n})",
                    f"cumulative return in excess of {lbl}, t-{w}..t-{k}",
                    w,
                    skip=k,
                )
        for r, lbl in (("ex_ret", "market"), ("ex_btc_ret", "btc")):
            b.add(
                f"mom.excess_{lbl}.w{w}_k0",
                "momentum",
                "momentum",
                f"ts_sum({r}, {w})",
                f"cumulative return in excess of {lbl} over {w} bars",
                w,
                skip=0,
            )
        b.add(
            f"mom.efficiency.w{w}",
            "momentum",
            "trend",
            f"div(abs(delta(close, {w})), ts_sum(abs(delta(close, 1)), {w}))",
            f"Kaufman efficiency ratio over {w} bars",
            w,
        )
        b.add(
            f"mom.signed_eff.w{w}",
            "momentum",
            "trend",
            f"div(delta(close, {w}), ts_sum(abs(delta(close, 1)), {w}))",
            f"signed efficiency ratio over {w} bars",
            w,
        )
        b.add(
            f"mom.vol_weighted.w{w}",
            "momentum",
            "momentum",
            f"div(ts_sum(ret * quote_volume, {w}), ts_sum(quote_volume, {w}))",
            f"dollar-volume weighted mean return over {w} bars",
            w,
        )
        b.add(
            f"mom.range_weighted.w{w}",
            "momentum",
            "momentum",
            f"div(ts_sum(ret * hl_range, {w}), ts_sum(hl_range, {w}))",
            f"range-weighted mean return over {w} bars",
            w,
        )
        b.add(
            f"mom.sharpe_down.w{w}",
            "momentum",
            "momentum",
            f"div(ts_mean(ret, {w}), ssqrt(ts_mean(square(down_ret), {w})))",
            f"Sortino-style mean return over downside deviation, {w} bars",
            w,
        )
        b.add(
            f"mom.intraday.w{w}",
            "momentum",
            "momentum",
            f"ts_sum(oc_ret, {w})",
            f"cumulative intrabar (open-to-close) return over {w} bars",
            w,
        )
        b.add(
            f"mom.overnight.w{w}",
            "momentum",
            "momentum",
            f"ts_sum(gap, {w})",
            f"cumulative bar-open gap return over {w} bars",
            w,
        )


CORR_SERIES = [
    "ret",
    "abs_ret",
    "oc_ret",
    "hl_range",
    "body",
    "close_loc",
    "upper_shadow",
    "lower_shadow",
    "vwap_dev",
    "dvol_chg",
    "amihud",
    "signed_dvol",
    "taker_imb",
    "net_taker_dvol",
    "funding",
    "oi_chg",
    "oi_turnover",
    "ls_ratio",
    "top_ls_ratio",
    "taker_bs",
    "basis",
    "log_dvol",
    "log_trades",
]
LEVEL_CORR_PAIRS = [
    ("close", "volume"),
    ("close", "quote_volume"),
    ("vwap", "volume"),
    ("high", "volume"),
    ("low", "volume"),
    ("close", "trades"),
    ("close", "open_interest_value"),
    ("close", "open_interest"),
    ("volume", "open_interest"),
    ("open", "volume"),
    ("high", "low"),
    ("close", "trade_size"),
    ("quote_volume", "open_interest_value"),
    ("close", "funding_rate"),
    ("close", "long_short_ratio"),
    ("close", "premium"),
]


def family_correlation(b: _Builder):
    """Add the correlation family: rolling correlations between series, lead-lag correlations and autocorrelations."""
    for x, y in itertools.combinations(CORR_SERIES, 2):
        for w in CORR_WINDOWS:
            b.add(
                f"corr.{x}.{y}.w{w}",
                "correlation",
                "correlation",
                f"ts_corr({x}, {y}, {w})",
                f"rolling correlation of {x} and {y}",
                w,
                x=x,
                y=y,
            )
    lead_windows = [10, 20, 30, 60, 120, 240]
    for x, y in itertools.permutations(CORR_SERIES, 2):
        for w in lead_windows:
            b.add(
                f"leadlag.{x}.{y}.w{w}",
                "lead_lag",
                "correlation",
                f"ts_corr(delay({x}, 1), {y}, {w})",
                f"correlation of lagged {x} with current {y}",
                w,
                leader=x,
                follower=y,
            )
    for x, y in LEVEL_CORR_PAIRS:
        for w in CORR_WINDOWS:
            b.add(
                f"corr_level.{x}.{y}.w{w}",
                "correlation",
                "price_volume",
                f"ts_corr({x}, {y}, {w})",
                f"rolling correlation of {x} and {y} levels",
                w,
                x=x,
                y=y,
            )
            b.add(
                f"corr_rank.{x}.{y}.w{w}",
                "correlation",
                "price_volume",
                f"ts_corr(cs_rank({x}), cs_rank({y}), {w})",
                f"correlation of cross-sectional ranks of {x} and {y}",
                w,
                x=x,
                y=y,
            )
    for x in CORR_SERIES:
        for lag in (2, 3, 5, 10):
            for w in (20, 60, 120, 240):
                if w < 4 * lag:
                    continue
                b.add(
                    f"autocorr.{x}.lag{lag}.w{w}",
                    "correlation",
                    "correlation",
                    f"ts_autocorr({x}, {w}, {lag})",
                    f"lag-{lag} autocorrelation of {x}",
                    w,
                    series=x,
                    lag=lag,
                )


BENCHMARKS = [("mkt_ret", "mkt"), ("mkt_ret_vw", "mktvw"), ("btc_ret", "btc")]


def family_beta(b: _Builder):
    """Add the market_beta family: beta, downside and upside beta, idiosyncratic volatility, residual momentum and alpha."""
    for bm, lbl in BENCHMARKS:
        for w in WINDOWS:
            if w < 10:
                continue
            tpl = {
                "beta": (f"ts_beta(ret, {bm}, {w})", "market beta"),
                "corr": (f"ts_corr(ret, {bm}, {w})", "correlation with market"),
                "rsq": (f"square(ts_corr(ret, {bm}, {w}))", "R-squared vs market"),
                "idio_vol": (
                    f"ts_std(ret, {w}) * ssqrt(1 - square(ts_corr(ret, {bm}, {w})))",
                    "idiosyncratic volatility",
                ),
                "idio_share": (f"1 - square(ts_corr(ret, {bm}, {w}))", "share of idiosyncratic variance"),
                "resid": (f"ts_resid(ret, {bm}, {w})", "current CAPM residual"),
                "resid_mom": (
                    f"ts_mean(ret - ts_beta(ret, {bm}, {w}) * {bm}, {w})",
                    "residual (beta-adjusted) momentum",
                ),
                "alpha": (f"ts_mean(ret, {w}) - ts_beta(ret, {bm}, {w}) * ts_mean({bm}, {w})", "rolling Jensen alpha"),
                "down_beta": (f"ts_beta(ret, emin({bm}, 0), {w})", "downside beta"),
                "up_beta": (f"ts_beta(ret, emax({bm}, 0), {w})", "upside beta"),
                "beta_asym": (
                    f"ts_beta(ret, emin({bm}, 0), {w}) - ts_beta(ret, emax({bm}, 0), {w})",
                    "downside minus upside beta",
                ),
                "coskew": (f"ts_corr(ret, square({bm}), {w})", "co-skewness proxy"),
                "lag_beta": (f"ts_beta(ret, delay({bm}, 1), {w})", "beta to lagged market (delayed reaction)"),
                "lead_corr": (f"ts_corr(delay(ret, 1), {bm}, {w})", "does the coin lead the market"),
                "beta_vol": (f"ts_corr(abs_ret, abs({bm}), {w})", "volatility co-movement with market"),
                "tracking": (f"ts_std(ret - {bm}, {w})", "tracking error vs market"),
            }
            for k, (expr, desc) in tpl.items():
                b.add(
                    f"beta.{k}.{lbl}.w{w}",
                    "market_beta",
                    "beta",
                    expr,
                    f"{desc} vs {lbl} over {w} bars",
                    w,
                    benchmark=bm,
                )


def family_technical(b: _Builder):
    """Add the technical family: classic indicators, volatility estimators and spread or liquidity measures."""
    up = "emax(delta(close, 1), 0)"
    dn = "emax(-delta(close, 1), 0)"
    tr = "(emax(high, delay(close, 1)) - emin(low, delay(close, 1)))"
    dmp = "(emax(delta(high, 1), 0) * gt(delta(high, 1), -delta(low, 1)))"
    dmm = "(emax(-delta(low, 1), 0) * gt(-delta(low, 1), delta(high, 1)))"
    mid = "((high + low) / 2)"
    for w in WINDOWS:
        t = {
            "rsi": (
                f"div(ts_mean({up}, {w}), ts_mean({up}, {w}) + ts_mean({dn}, {w}))",
                "momentum",
                "RSI (simple averages)",
            ),
            "rsi_wilder": (
                f"div(ema({up}, {2 * w - 1}), ema({up}, {2 * w - 1}) + ema({dn}, {2 * w - 1}))",
                "momentum",
                "RSI (Wilder smoothing)",
            ),
            "stoch_k": (
                f"div(close - ts_min(low, {w}), ts_max(high, {w}) - ts_min(low, {w}))",
                "trend",
                "Stochastic %K",
            ),
            "stoch_d": (
                f"ts_mean(div(close - ts_min(low, {w}), ts_max(high, {w}) - ts_min(low, {w})), 3)",
                "trend",
                "Stochastic %D",
            ),
            "cci": (
                f"div(typical - ts_mean(typical, {w}), 0.015 * ts_mean(abs(typical - ts_mean(typical, {w})), {w}))",
                "trend",
                "Commodity Channel Index",
            ),
            "mfi": (
                f"div(ts_sum(typical * volume * gt(delta(typical, 1), 0), {w}), ts_sum(typical * volume, {w}))",
                "price_volume",
                "Money Flow Index",
            ),
            "natr": (f"div(ts_mean({tr}, {w}), close)", "volatility", "normalised ATR"),
            "natr_ema": (f"div(ema({tr}, {w}), close)", "volatility", "normalised ATR (EMA)"),
            "di_diff": (f"div(ts_sum({dmp}, {w}) - ts_sum({dmm}, {w}), ts_sum({tr}, {w}))", "trend", "+DI minus -DI"),
            "dx": (
                f"div(abs(ts_sum({dmp}, {w}) - ts_sum({dmm}, {w})), ts_sum({dmp}, {w}) + ts_sum({dmm}, {w}))",
                "trend",
                "Directional movement index",
            ),
            "aroon_osc": (f"ts_argmin(low, {w}) - ts_argmax(high, {w})", "trend", "Aroon oscillator"),
            "trix": (f"delta(ema(ema(ema(log(close), {w}), {w}), {w}), 1)", "trend", "TRIX"),
            "uo": (
                f"(4 * div(ts_sum(close - emin(low, delay(close, 1)), {w}), ts_sum({tr}, {w})) + 2 * div(ts_sum(close - emin(low, delay(close, 1)), {2 * w}), ts_sum({tr}, {2 * w})) + div(ts_sum(close - emin(low, delay(close, 1)), {4 * w}), ts_sum({tr}, {4 * w}))) / 7",
                "trend",
                "Ultimate oscillator",
            ),
            "cmf": (f"div(ts_sum(clv * volume, {w}), ts_sum(volume, {w}))", "price_volume", "Chaikin money flow"),
            "force": (
                f"div(ema(ret * quote_volume, {w}), ts_mean(quote_volume, {w}))",
                "price_volume",
                "normalised force index",
            ),
            "eom": (
                f"ts_mean(div({mid}, delay({mid}, 1)) - 1, {w}) * div(ts_mean(quote_volume, {w}), quote_volume)",
                "price_volume",
                "ease of movement (normalised)",
            ),
            "vwap_dist": (
                f"div(close, div(ts_sum(quote_volume, {w}), ts_sum(volume, {w}))) - 1",
                "price_volume",
                "distance from rolling VWAP",
            ),
            "obv_flow": (
                f"div(ts_sum(sign(ret) * quote_volume, {w}), ts_sum(quote_volume, {w}))",
                "price_volume",
                "normalised on-balance volume flow",
            ),
            "keltner": (f"div(close - ema(close, {w}), 2 * ema({tr}, {w}))", "trend", "Keltner channel position"),
            "donchian_width": (
                f"div(ts_max(high, {w}) - ts_min(low, {w}), close)",
                "volatility",
                "Donchian channel width",
            ),
            "boll_width": (f"div(ts_std(close, {w}), ts_mean(close, {w}))", "volatility", "Bollinger band width"),
            "vortex": (
                f"div(ts_sum(abs(high - delay(low, 1)), {w}) - ts_sum(abs(low - delay(high, 1)), {w}), ts_sum({tr}, {w}))",
                "trend",
                "Vortex indicator difference",
            ),
            "chop": (
                f"log(div(ts_sum({tr}, {w}), ts_max(high, {w}) - ts_min(low, {w})))",
                "trend",
                "Choppiness (log form)",
            ),
            "ulcer": (f"ssqrt(ts_mean(square(div(close, ts_max(close, {w})) - 1), {w}))", "volatility", "Ulcer index"),
            "parkinson_vol": (f"ssqrt(ts_mean(parkinson_var, {w}))", "volatility", "Parkinson volatility"),
            "gk_vol": (f"ssqrt(ts_mean(gk_var, {w}))", "volatility", "Garman-Klass volatility"),
            "rs_vol": (f"ssqrt(ts_mean(rs_var, {w}))", "volatility", "Rogers-Satchell volatility"),
            "yz_vol": (
                f"ssqrt(square(ts_std(gap, {w})) + 0.34 * square(ts_std(oc_ret, {w})) + 0.66 * ts_mean(rs_var, {w}))",
                "volatility",
                "Yang-Zhang volatility",
            ),
            "down_vol": (f"ssqrt(ts_mean(square(down_ret), {w}))", "volatility", "downside semi-volatility"),
            "up_vol": (f"ssqrt(ts_mean(square(up_ret), {w}))", "volatility", "upside semi-volatility"),
            "semivol_ratio": (
                f"log(div(ts_mean(square(up_ret), {w}), ts_mean(square(down_ret), {w})))",
                "volatility",
                "log upside/downside variance ratio",
            ),
            "vol_of_vol": (
                f"div(ts_std(log_hl, {w}), ts_mean(log_hl, {w}))",
                "volatility",
                "volatility of range volatility",
            ),
            "range_vs_close_vol": (
                f"log(div(ssqrt(ts_mean(parkinson_var, {w})), ts_std(logret, {w})))",
                "volatility",
                "range-based vs close-based volatility",
            ),
            "roll_spread": (
                f"2 * ssqrt(emax(-ts_cov(ret, delay(ret, 1), {w}), 0))",
                "liquidity",
                "Roll implied spread",
            ),
            "kyle_lambda": (f"ts_beta(ret, signed_dvol, {w})", "liquidity", "Kyle lambda (price impact)"),
            "flow_toxicity": (f"ts_mean(abs(taker_imb), {w})", "order_flow", "VPIN-style flow toxicity"),
            "flow_impact": (f"ts_beta(ret, taker_imb, {w})", "order_flow", "price impact of taker imbalance"),
            "up_volume_share": (
                f"div(ts_sum(quote_volume * gt(ret, 0), {w}), ts_sum(quote_volume, {w}))",
                "price_volume",
                "share of volume on up bars",
            ),
            "up_trade_share": (
                f"div(ts_sum(trades * gt(ret, 0), {w}), ts_sum(trades, {w}))",
                "price_volume",
                "share of trades on up bars",
            ),
            "cs_spread": (
                f"ts_mean(emax(2 * (exp(ssqrt(2 * (square(log(high / low)) + square(log(delay(high, 1) / delay(low, 1))))) / 0.171573 - ssqrt(square(log(high / low)) + square(log(delay(high, 1) / delay(low, 1)))) / 0.171573 - ssqrt(div(square(log(div(emax(high, delay(high, 1)), emin(low, delay(low, 1))))), 0.171573))) - 1) / (1 + exp(ssqrt(2 * (square(log(high / low)) + square(log(delay(high, 1) / delay(low, 1))))) / 0.171573 - ssqrt(square(log(high / low)) + square(log(delay(high, 1) / delay(low, 1)))) / 0.171573 - ssqrt(div(square(log(div(emax(high, delay(high, 1)), emin(low, delay(low, 1))))), 0.171573)))), 0), {w})",
                "liquidity",
                "Corwin-Schultz high-low spread",
            ),
        }
        if w >= 5:
            for q in (2, 3, 5, 10):
                if w >= 6 * q:
                    t[f"var_ratio_q{q}"] = (
                        f"div(square(ts_std(log(div(close, delay(close, {q}))), {w})), {q} * square(ts_std(logret, {w})))",
                        "trend",
                        f"variance ratio (q={q})",
                    )
        for k, (expr, cat, desc) in t.items():
            b.add(f"tech.{k}.w{w}", "technical", cat, expr, f"{desc} over {w} bars", w, indicator=k)


def family_derivatives(b: _Builder):
    """Add the crypto_derivatives family: funding, open interest, positioning and basis measures."""
    for w in WINDOWS:
        t = {
            "oi_price_div": (
                f"log(div(open_interest_value, delay(open_interest_value, {w}))) - log(div(close, delay(close, {w})))",
                "open_interest",
                "OI change minus price change",
            ),
            "oi_coin_price_div": (
                f"log(div(open_interest, delay(open_interest, {w}))) - log(div(close, delay(close, {w})))",
                "open_interest",
                "OI (contracts) change minus price change",
            ),
            "oi_confirm": (
                f"sign(delta(log(close), {w})) * delta(log_oi, {w})",
                "open_interest",
                "OI change confirming price trend",
            ),
            "leverage": (
                f"log(div(open_interest_value, ts_mean(quote_volume, {w})))",
                "open_interest",
                "OI relative to average dollar volume (leverage proxy)",
            ),
            "liq_proxy": (
                f"ts_sum(emin(oi_chg, 0) * abs_ret, {w})",
                "open_interest",
                "OI flush on large moves (liquidation proxy)",
            ),
            "oi_ret_beta": (f"ts_beta(oi_chg, ret, {w})", "open_interest", "sensitivity of OI changes to returns"),
            "funding_carry": (f"ts_sum(funding_rate, {w})", "funding", "cumulative funding paid by longs"),
            "funding_vw": (
                f"div(ts_sum(funding_rate * quote_volume, {w}), ts_sum(quote_volume, {w}))",
                "funding",
                "volume-weighted funding",
            ),
            "funding_flip": (
                f"ts_mean(abs(delta(sign(funding_rate), 1)), {w})",
                "funding",
                "funding sign flip frequency",
            ),
            "funding_vs_mkt": (
                f"ts_mean(funding_rate - cs_mean(funding_rate), {w})",
                "funding",
                "funding relative to market average",
            ),
            "funding_minus_mom": (
                f"ts_zscore(funding_rate, {w}) - ts_zscore(log(close), {w})",
                "funding",
                "funding crowding vs price trend",
            ),
            "funding_ret_corr": (f"ts_corr(funding_rate, ret, {w})", "funding", "funding-return co-movement"),
            "basis_mom": (f"delta(premium, {w})", "basis", "change in perp basis"),
            "basis_vs_funding": (
                f"ts_mean(premium, {w}) - ts_mean(funding_rate, {w})",
                "basis",
                "average basis minus average funding",
            ),
            "basis_vs_mkt": (f"ts_mean(premium - cs_mean(premium), {w})", "basis", "basis relative to market"),
            "ls_mom": (f"delta(ls_ratio, {w})", "positioning", "change in crowd long/short ratio"),
            "top_ls_mom": (f"delta(top_ls_ratio, {w})", "positioning", "change in top-trader long/short ratio"),
            "smart_dumb": (
                f"delta(top_ls_ratio, {w}) - delta(ls_ratio, {w})",
                "positioning",
                "top traders vs crowd positioning change",
            ),
            "ls_vs_price": (
                f"ts_zscore(ls_ratio, {w}) - ts_zscore(log(close), {w})",
                "positioning",
                "crowd positioning vs price trend",
            ),
            "fut_spot_flow": (
                f"ts_mean(taker_bs - log(div(taker_ratio, 1 - taker_ratio)), {w})",
                "order_flow",
                "futures vs spot taker aggressiveness",
            ),
            "ls_vs_mkt": (
                f"ts_mean(ls_ratio - cs_mean(ls_ratio), {w})",
                "positioning",
                "long/short ratio relative to market",
            ),
            "oi_vs_mkt": (f"ts_sum(oi_chg - cs_mean(oi_chg), {w})", "open_interest", "OI growth relative to market"),
        }
        for k, (expr, cat, desc) in t.items():
            b.add(f"deriv.{k}.w{w}", "crypto_derivatives", cat, expr, f"{desc} over {w} bars", w, indicator=k)


CONDITIONS = [
    ("up", "gt(ret, 0)", "up bars"),
    ("down", "lt(ret, 0)", "down bars"),
    ("hivol", "gt(quote_volume, ts_mean(quote_volume, 20))", "above-average volume bars"),
    ("lovol", "lt(quote_volume, ts_mean(quote_volume, 20))", "below-average volume bars"),
    ("mktup", "gt(mkt_ret, 0)", "market up bars"),
    ("mktdown", "lt(mkt_ret, 0)", "market down bars"),
    ("btcup", "gt(btc_ret, 0)", "BTC up bars"),
    ("btcdown", "lt(btc_ret, 0)", "BTC down bars"),
    ("fundpos", "gt(funding_rate, 0)", "positive funding bars"),
    ("fundneg", "lt(funding_rate, 0)", "negative funding bars"),
    ("oiup", "gt(oi_chg, 0)", "rising-OI bars"),
    ("oidown", "lt(oi_chg, 0)", "falling-OI bars"),
    ("bigmove", "gt(abs_ret, ts_mean(abs_ret, 20))", "large-move bars"),
    ("buyflow", "gt(taker_imb, 0)", "net taker-buy bars"),
]
COND_SERIES = [
    "ret",
    "ex_ret",
    "abs_ret",
    "hl_range",
    "taker_imb",
    "dvol_chg",
    "vwap_dev",
    "close_loc",
    "oi_chg",
    "funding",
    "basis",
    "signed_dvol",
    "amihud",
    "upper_shadow",
    "lower_shadow",
]


def family_conditional(b: _Builder):
    """Add the conditional family: means of a series on bars that meet a condition (CONDITIONS)."""
    for x in COND_SERIES:
        for cname, cexpr, cdesc in CONDITIONS:
            if x in ("ret", "ex_ret", "signed_dvol") and cname in ("up", "down"):
                continue
            for w in (10, 20, 30, 60, 90, 120, 240):
                b.add(
                    f"cond.{x}.{cname}.w{w}",
                    "conditional",
                    CATEGORY_OF.get(x, "conditional"),
                    f"div(ts_mean({x} * {cexpr}, {w}), ts_mean({cexpr}, {w}))",
                    f"mean of {x} on {cdesc} over {w} bars",
                    w,
                    series=x,
                    condition=cname,
                )
    for x in ("ret", "ex_ret", "abs_ret", "taker_imb", "oi_chg", "funding", "hl_range", "vwap_dev"):
        for (c1, e1, _), (c2, e2, _) in [
            (CONDITIONS[0], CONDITIONS[1]),
            (CONDITIONS[2], CONDITIONS[3]),
            (CONDITIONS[4], CONDITIONS[5]),
            (CONDITIONS[8], CONDITIONS[9]),
            (CONDITIONS[10], CONDITIONS[11]),
        ]:
            if x in ("ret", "ex_ret") and c1 == "up":
                continue
            for w in (20, 60, 120):
                b.add(
                    f"condspread.{x}.{c1}_{c2}.w{w}",
                    "conditional",
                    CATEGORY_OF.get(x, "conditional"),
                    f"div(ts_mean({x} * {e1}, {w}), ts_mean({e1}, {w})) - div(ts_mean({x} * {e2}, {w}), ts_mean({e2}, {w}))",
                    f"{x} on {c1} minus on {c2} bars over {w} bars",
                    w,
                    series=x,
                )


SIGNAL_COMPONENTS = [
    ("price", "log(close)", "trend"),
    ("dvol", "log_dvol", "liquidity"),
    ("trades", "log_trades", "liquidity"),
    ("tsize", "log(trade_size)", "liquidity"),
    ("range", "hl_range", "volatility"),
    ("flow", "taker_imb", "order_flow"),
    ("netflow", "net_taker_dvol", "order_flow"),
    ("funding", "funding", "funding"),
    ("oi", "log_oi", "open_interest"),
    ("ls", "ls_ratio", "positioning"),
    ("topls", "top_ls_ratio", "positioning"),
    ("takerbs", "taker_bs", "positioning"),
    ("basis", "basis", "basis"),
    ("closeloc", "close_loc", "price_pattern"),
    ("vwapdev", "vwap_dev", "price_pattern"),
    ("illiq", "amihud", "liquidity"),
]


def family_interaction(b: _Builder):
    """Add the interaction family: products, differences and sign gates of z-scores, and horizon spreads."""
    windows = [5, 10, 20, 30, 60, 120, 240]
    for (n1, e1, _), (n2, e2, _) in itertools.combinations(SIGNAL_COMPONENTS, 2):
        for w in windows:
            z1, z2 = f"ts_zscore({e1}, {w})", f"ts_zscore({e2}, {w})"
            b.add(
                f"inter.prod.{n1}.{n2}.w{w}",
                "interaction",
                "interaction",
                f"{z1} * {z2}",
                f"product of {n1} and {n2} z-scores ({w} bars)",
                w,
                a=n1,
                b=n2,
            )
            b.add(
                f"inter.diff.{n1}.{n2}.w{w}",
                "interaction",
                "interaction",
                f"{z1} - {z2}",
                f"{n1} z-score minus {n2} z-score ({w} bars)",
                w,
                a=n1,
                b=n2,
            )
            b.add(
                f"inter.gate.{n1}.{n2}.w{w}",
                "interaction",
                "interaction",
                f"{z1} * sign({z2})",
                f"{n1} z-score signed by {n2} ({w} bars)",
                w,
                a=n1,
                b=n2,
            )
            b.add(
                f"inter.gate.{n2}.{n1}.w{w}",
                "interaction",
                "interaction",
                f"{z2} * sign({z1})",
                f"{n2} z-score signed by {n1} ({w} bars)",
                w,
                a=n2,
                b=n1,
            )
    for n, e, cat in SIGNAL_COMPONENTS:
        for s, lw in _window_pairs(windows, 2):
            b.add(
                f"inter.horizon.{n}.{s}_{lw}",
                "interaction",
                cat,
                f"ts_zscore({e}, {s}) - ts_zscore({e}, {lw})",
                f"short vs long horizon z-score of {n} ({s} vs {lw})",
                lw,
                a=n,
            )


SECOND_ORDER_SERIES = [
    "ret",
    "abs_ret",
    "hl_range",
    "taker_imb",
    "net_taker_dvol",
    "dvol_chg",
    "amihud",
    "vwap_dev",
    "clv",
    "funding",
    "oi_chg",
    "ls_ratio",
    "top_ls_ratio",
    "taker_bs",
    "basis",
    "log_dvol",
    "log_trades",
    "log_oi",
]


def family_second_order(b: _Builder):
    """Add the second_order family: z-score, rank, range position or change of smoothed statistics."""
    inner = [("mean", "ts_mean"), ("std", "ts_std"), ("ema", "ema")]
    outer = [("zscore", "ts_zscore"), ("rank", "ts_rank"), ("minmax", "ts_minmax")]
    for x in SECOND_ORDER_SERIES:
        cat = CATEGORY_OF.get(x, "liquidity" if x.startswith("log_") and x != "log_oi" else "open_interest")
        for (iname, ifn), iw in itertools.product(inner, (3, 5, 10, 20)):
            for (oname, ofn), ow in itertools.product(outer, (20, 60, 120, 240)):
                if ow <= iw:
                    continue
                b.add(
                    f"so.{oname}_{iname}.{x}.i{iw}_o{ow}",
                    "second_order",
                    cat,
                    f"{ofn}({ifn}({x}, {iw}), {ow})",
                    f"{oname} over {ow} bars of {iname}({x}, {iw})",
                    ow + iw,
                    series=x,
                    inner=iw,
                    outer=ow,
                )
            for k in (1, 3, 5, 10):
                b.add(
                    f"so.delta_{iname}.{x}.i{iw}_k{k}",
                    "second_order",
                    cat,
                    f"delta({ifn}({x}, {iw}), {k})",
                    f"{k}-bar change of {iname}({x}, {iw})",
                    iw + k,
                    series=x,
                    inner=iw,
                    lag=k,
                )


ALPHA_TEMPLATES = [
    (
        "a001",
        "cs_rank(ts_argmax(square(emax(ret, 0) + ts_std(ret, {w}) * lt(ret, 0)), {d})) - 0.5",
        "Alpha#1 analogue: timing of extreme return/vol",
    ),
    (
        "a002",
        "-ts_corr(cs_rank(delta(log(volume), 2)), cs_rank(div(close - open, open)), {w})",
        "Alpha#2: volume change vs intrabar return correlation",
    ),
    ("a003", "-ts_corr(cs_rank(open), cs_rank(volume), {w})", "Alpha#3: open vs volume rank correlation"),
    ("a004", "-ts_rank(cs_rank(low), {w})", "Alpha#4: time-series rank of low's cross-sectional rank"),
    ("a005", "cs_rank(open - ts_mean(vwap, {w})) * -abs(cs_rank(close - vwap))", "Alpha#5: open vs average vwap"),
    ("a006", "-ts_corr(open, volume, {w})", "Alpha#6: open/volume correlation"),
    (
        "a007",
        "-ts_rank(abs(delta(close, {d})), {w}) * sign(delta(close, {d})) * gt(quote_volume, ts_mean(quote_volume, {w}))",
        "Alpha#7: reversal on high volume",
    ),
    (
        "a008",
        "-cs_rank(ts_sum(open, {d}) * ts_sum(ret, {d}) - delay(ts_sum(open, {d}) * ts_sum(ret, {d}), {w}))",
        "Alpha#8",
    ),
    (
        "a009",
        "ts_min(delta(close, 1), {d}) * gt(ts_min(delta(close, 1), {d}), 0) + ts_max(delta(close, 1), {d}) * lt(ts_max(delta(close, 1), {d}), 0) - delta(close, 1) * gt(ts_max(delta(close, 1), {d}), 0) * lt(ts_min(delta(close, 1), {d}), 0)",
        "Alpha#9: trend-persistence switch",
    ),
    ("a012", "sign(delta(volume, {d})) * -delta(close, {d}) / close", "Alpha#12: volume-confirmed reversal"),
    ("a013", "-cs_rank(ts_cov(cs_rank(close), cs_rank(volume), {w}))", "Alpha#13"),
    ("a014", "-cs_rank(delta(ret, {d})) * ts_corr(open, volume, {w})", "Alpha#14"),
    ("a015", "-ts_sum(cs_rank(ts_corr(cs_rank(high), cs_rank(volume), {d})), {d})", "Alpha#15"),
    ("a016", "-cs_rank(ts_cov(cs_rank(high), cs_rank(volume), {w}))", "Alpha#16"),
    (
        "a017",
        "-cs_rank(ts_rank(close, {w})) * cs_rank(delta(delta(close, 1), 1)) * cs_rank(ts_rank(div(volume, ts_mean(volume, {w})), {d}))",
        "Alpha#17",
    ),
    (
        "a018",
        "-cs_rank(ts_std(abs(close - open), {w}) + (close - open) / close + ts_corr(close, open, {w}))",
        "Alpha#18",
    ),
    ("a019", "-sign(close / delay(close, {d}) - 1) * (1 + cs_rank(1 + ts_sum(ret, {w})))", "Alpha#19"),
    (
        "a020",
        "-cs_rank(open / delay(high, 1) - 1) * cs_rank(open / delay(close, 1) - 1) * cs_rank(open / delay(low, 1) - 1)",
        "Alpha#20",
    ),
    ("a022", "-delta(ts_corr(high, volume, {d}), {d}) * cs_rank(ts_std(close, {w}) / close)", "Alpha#22"),
    ("a023", "-delta(high, 2) / close * gt(high, ts_mean(high, {w}))", "Alpha#23"),
    ("a025", "cs_rank(-ret * ts_mean(quote_volume, {w}) * vwap * (high - close) / close / close)", "Alpha#25"),
    ("a026", "-ts_max(ts_corr(ts_rank(volume, {d}), ts_rank(high, {d}), {d}), {d})", "Alpha#26"),
    ("a028", "cs_zscore(ts_corr(ts_mean(volume, {w}), low, {d}) + (high + low) / 2 / close - 1)", "Alpha#28"),
    (
        "a030",
        "(1 - cs_rank(sign(delta(close, 1)) + sign(delay(delta(close, 1), 1)) + sign(delay(delta(close, 1), 2)))) * div(ts_sum(volume, {d}), ts_sum(volume, {w}))",
        "Alpha#30",
    ),
    (
        "a032",
        "cs_zscore(ts_mean(close, {d}) / close - 1) + 20 * cs_zscore(ts_corr(vwap, delay(close, {d}), {w}))",
        "Alpha#32",
    ),
    ("a033", "cs_rank(-(1 - open / close))", "Alpha#33"),
    (
        "a034",
        "cs_rank(1 - cs_rank(div(ts_std(ret, {d}), ts_std(ret, {w}))) + 1 - cs_rank(delta(close, 1) / close))",
        "Alpha#34",
    ),
    ("a035", "ts_rank(volume, {w}) * (1 - ts_rank(close + high - low, {d})) * (1 - ts_rank(ret, {w}))", "Alpha#35"),
    ("a037", "cs_rank(ts_corr(delay(open - close, 1), close, {w})) + cs_rank(open - close)", "Alpha#37"),
    ("a038", "-cs_rank(ts_rank(close, {d})) * cs_rank(close / open)", "Alpha#38"),
    ("a040", "-cs_rank(ts_std(high, {d}) / close) * ts_corr(high, volume, {w})", "Alpha#40"),
    ("a041", "ssqrt(high * low) / vwap - 1", "Alpha#41"),
    ("a042", "div(cs_rank(vwap - close), cs_rank(vwap + close))", "Alpha#42"),
    ("a043", "ts_rank(div(volume, ts_mean(volume, {w})), {w}) * ts_rank(-delta(close, {d}), {d})", "Alpha#43"),
    ("a044", "-ts_corr(high, cs_rank(volume), {d})", "Alpha#44"),
    (
        "a045",
        "-cs_rank(ts_mean(delay(close, {d}), {w})) * ts_corr(close, volume, {d}) * cs_rank(ts_corr(ts_sum(close, {d}), ts_sum(close, {w}), {d}))",
        "Alpha#45",
    ),
    ("a053", "-delta(div((close - low) - (high - close), close - low), {d})", "Alpha#53"),
    ("a054", "-div((low - close) * square(open), (low - high) * square(close))", "Alpha#54"),
    (
        "a055",
        "-ts_corr(cs_rank(div(close - ts_min(low, {w}), ts_max(high, {w}) - ts_min(low, {w}))), cs_rank(volume), {d})",
        "Alpha#55",
    ),
    (
        "a060",
        "-cs_rank(div((close - low) - (high - close), high - low) * volume) + cs_rank(ts_argmax(close, {w}))",
        "Alpha#60",
    ),
    (
        "a083",
        "div(cs_rank(delay((high - low) / ts_mean(close, {d}), 2)) * cs_rank(cs_rank(volume)), div((high - low) / ts_mean(close, {d}), vwap - close))",
        "Alpha#83",
    ),
    ("a101", "div(close - open, high - low + 0.001 * close)", "Alpha#101: intrabar trend"),
]


def family_alpha(b: _Builder):
    """Add the alpha101_style family: crypto versions of the Alpha101 formulas (ALPHA_TEMPLATES) for three window pairs."""
    for key, tpl, desc in ALPHA_TEMPLATES:
        variants = [(3, 10), (5, 20), (10, 60)] if ("{w}" in tpl or "{d}" in tpl) else [(None, None)]
        for d, w in variants:
            expr = tpl.format(d=d, w=w) if d else tpl
            suffix = f".d{d}_w{w}" if d else ""
            b.add(
                f"alpha.{key}{suffix}",
                "alpha101_style",
                "alpha101",
                expr,
                f"{desc}" + (f" (d={d}, w={w})" if d else ""),
                w,
                d=d,
                w=w,
            )


FAMILIES = [
    family_stat,
    family_level,
    family_ma_cross,
    family_momentum,
    family_correlation,
    family_beta,
    family_technical,
    family_derivatives,
    family_conditional,
    family_interaction,
    family_second_order,
    family_alpha,
]


def generate_factors() -> list[FactorSpec]:
    """Return every candidate factor of all families, in generation order, without repeated expressions.

    Two specs with the same parsed expression count once (the first one stays). The same
    code gives the same list.
    """
    b = _Builder()
    for fam in FAMILIES:
        fam(b)
    return b.specs
