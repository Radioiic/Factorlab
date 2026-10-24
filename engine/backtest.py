"""Weekly-rebalance portfolio backtester for perp books. book.py runs it for each test.

Design: one plain loop.
    The loop books perp funding payments together with fees and returns.
    Thus every dollar of fees and funding is visible in one place. The loop
    runs the full universe in less than a second. At a weekly cadence,
    auditability is more important than speed.

TIMING CONVENTION (matches labels.py — read this before touching anything):
    A daily bar stamped D spans D 00:00 -> D+1 00:00 UTC. The factor row
    stamped D is fully known at D+1 00:00. The engine therefore:
      - on a rebalance date t (default: Mondays), reads the factor
        cross-section stamped t-1 (the bar that closed at t 00:00),
      - trades at t 00:00 (execution reference price = close of bar t-1),
      - and the new weights first earn the return of bar t
        (close[t]/close[t-1] - 1).
    Nothing decided at t can see bar t's own outcome. test_backtest.py
    enforces this with a synthetic price-jump test.

COST MODEL:
    - fees + slippage are charged on the DOLLARS traded, per leg:
      cost_i = (fee_bps + slippage_bps)/1e4 * |h_new_i - h_prev_i|, paid out
      of equity at the open before returns accrue (exact convention below).
      Default fee is OKX taker (5 bps) — conservative; the maker scenario is
      fee_bps=2, slippage_bps=2 (book.COST_SCENARIOS).
    - funding: the raw 8h prints are summed per UTC day (stamped on the day's
      open, same convention as labels.py). Sign: positive = longs PAY.
      Holding weight w through bar t costs w * funding[t] of equity — negative
      w (a short) therefore RECEIVES positive funding.
    - weights drift between rebalances with relative performance; turnover is
      computed against the *drifted* weight, not last week's target, so the
      trade log reflects what a rebalance actually trades.

ACCOUNTING CONVENTION ("reconciled-v2"):
    ONE set of dollar holdings drives both the trade log and the P&L. At a
    rebalance on t, with pre-trade equity E_pre (the mark at t 00:00 = the
    close of bar t-1), drifted weights w and drifted holdings h_prev = w*E_pre:

        E_post  = E_pre - sum_i cost_i          fees leave equity first
        h_new_i = target_w_i * E_post           sizing is % of POST-fee equity
        cost_i  = rate * |h_new_i - h_prev_i|   rate = (fee+slippage)/1e4

    E_post sits on both sides, so the fee is a small fixed point in the ratio
    x = E_post/E_pre:   x = 1 - rate * sum_i |target_w_i * x - w_i|.
    The right side is a contraction with factor rate * gross (<= 1e-3 at
    taker costs and gross 1.0), so iterating from x = 1 reaches machine
    precision in about five steps (_fill_ratio). Each leg is then logged with
    the SAME dollars the P&L uses:

        notional_i = |h_new_i - h_prev_i|      side = sign(h_new_i - h_prev_i)
        cost_i     = rate * notional_i          sum_i cost_i = the daily cost
        contracts  = notional_i / ref_price_i  (ref_price = execution price)

    and bar t is earned by exactly those holdings:
        E_end = E_post + sum_i h_new_i * r_i - sum_i h_new_i * funding_i.
    test_accounting.py rebuilds the equity curve from the trade log and the
    market data alone (contracts x prices, minus logged fees and funding) and
    requires it to match the daily frame of the Result to 1e-9.

    Three consequences worth knowing:
      * a held name whose target weight is UNCHANGED still trades a sliver:
        keeping weights exact while fees shrink equity means every position
        sells target_w * fees. Those legs are real and appear in the log.
      * dust: a leg whose dollar trade is within TRADE_EPS * E_pre is neither
        traded, charged nor logged — its holdings stay exactly as drifted (a
        weight within TRADE_EPS of target). A rebalance with no live leg is no
        rebalance at all.
      * delta_w keeps its meaning target_w - prev_w (a WEIGHT change); notional
        is the DOLLAR change and differs from |delta_w| * E_pre by
        target_w * fees. Reconcile dollars with notional/side, not delta_w.

    The stats of each run carry `"accounting": "reconciled-v2"`.

POSITION SIZING:
    Weights are fractions of current equity — the same run is valid at $1k
    and $1M (test_backtest.py asserts scale invariance). Gross exposure is
    capped at gross_cap (default 1.0 = unlevered); weights beyond the cap
    are scaled down proportionally. Shorts are allowed (negative weights).
    Targets are fractions of POST-fee equity, so the cap holds after costs
    (up to dust legs, each within TRADE_EPS).

OUTPUT (a Result):
    equity   date, equity, benchmark (BTC buy&hold, same start equity)
    trades   every trade: date, symbol, side, prev/target/delta weight,
             notional (dollars traded), cost, execution reference price
    daily    per-day ledger: returns, funding, costs, gross exposure
    stats    summary statistics with the accounting tag

demo_market (equal-weight long every coin with a factor row) is the weights function
of the equal-weight control book (book.py).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

import data_io

TRADE_EPS = 1e-6  # a leg trading less than this fraction of equity is dust — not logged or charged
BENCHMARK = "BTC-USDT"

# The accounting convention tag (see the module docstring); the only mode.
ACCOUNTING_RECONCILED = "reconciled-v2"
ACCOUNTING = ACCOUNTING_RECONCILED
ACCOUNTING_MODES = (ACCOUNTING_RECONCILED,)

# Fee fixed point: the map contracts by rate * gross per step (~1e-3), so ~5
# steps reach machine precision. The caps only bound pathological inputs.
_FIXED_POINT_MAX_ITER = 200
_DUST_MAX_PASSES = 8


# --- data loading -------------------------------------------------------------
# All parquet access goes through data_io (single home of the daily-funding
# resample + venue-splice conventions). run_backtest still accepts injected
# frames so the integrity tests never touch disk.


# --- result container ----------------------------------------------------------


@dataclass
class Result:
    equity: pd.DataFrame  # date, equity, benchmark
    trades: pd.DataFrame  # trade-by-trade log
    daily: pd.DataFrame  # per-day ledger
    stats: dict = field(default_factory=dict)


# --- rebalance execution ---------------------------------------------------------


def _fill_ratio(t: np.ndarray, p: np.ndarray, rate: float) -> float:
    """Solve x = E_post/E_pre = 1 - rate * sum|t*x - p| over the traded legs.

    t are target weights (fractions of POST-fee equity), p the drifted weights
    (fractions of PRE-trade equity). The map is a contraction with factor
    rate * sum|t| < 1 (checked by the caller), so plain iteration from x = 1
    converges; it stops once a step moves x by no more than a few ulps.
    """
    x = 1.0
    for _ in range(_FIXED_POINT_MAX_ITER):
        nxt = 1.0 - rate * float(np.abs(t * x - p).sum())
        if abs(nxt - x) <= 4.0 * np.finfo(float).eps:
            return nxt
        x = nxt
    raise RuntimeError("fee fixed point did not converge — rate * gross too close to 1")


def _solve_fill(t: np.ndarray, p: np.ndarray, rate: float) -> tuple[float, np.ndarray]:
    """Post-fee equity ratio x and the mask of legs that actually trade.

    A leg trades when its dollar trade |t*x - p| * E_pre exceeds TRADE_EPS of
    E_pre. Which legs trade depends on x and x depends on which legs pay fees,
    so the dust split is re-checked after each solve (it settles on the first
    or second pass). Whatever split is finally used, x is solved FOR that
    split, so the fees charged always equal rate * the dollars logged.
    """
    gross = float(np.abs(t).sum())
    if rate * gross >= 1.0:
        raise ValueError(
            f"cost rate {rate:g} x target gross {gross:g} >= 1: fees would consume the whole traded notional"
        )
    live = np.abs(t - p) > TRADE_EPS  # first guess: the pre-fee weight change
    for _ in range(_DUST_MAX_PASSES):
        x = _fill_ratio(t[live], p[live], rate)
        nxt = np.abs(t * x - p) > TRADE_EPS
        if np.array_equal(nxt, live):
            return x, live
        live = nxt
    return _fill_ratio(t[live], p[live], rate), live


def _rebalance_reconciled(date, target: pd.Series, w: pd.Series, equity: float, rate: float, ref_px: pd.Series):
    """reconciled-v2 execution: (weights held during bar t, cost, legs, traded).

    Holdings are the primitive: h_new = h_prev + logged dollar trade, the cost
    is rate * logged notional, E_post = E_pre - cost, and the weights handed to
    the P&L are h_new / E_post (== target_w for every traded leg, to rounding).
    So the trade log, the fees, the holdings and the equity are one ledger.
    """
    tv = target.to_numpy(dtype=float)
    pv = w.to_numpy(dtype=float)
    x, live = _solve_fill(tv, pv, rate)
    if not live.any():
        return w, 0.0, [], False  # nothing above dust: holdings untouched
    dollars = np.where(live, tv * x - pv, 0.0) * equity  # h_new - h_prev, signed
    notional = np.abs(dollars)
    leg_cost = notional * rate
    cost = float(leg_cost.sum())
    e_post = equity - cost
    h_new = pv * equity + dollars  # a closed leg lands on exactly 0.0
    legs = [
        {
            "date": date,
            "symbol": sym,
            "side": "buy" if dollars[k] > 0 else "sell",
            "prev_w": float(pv[k]),
            "target_w": float(tv[k]),
            "delta_w": float(tv[k] - pv[k]),
            "notional": float(notional[k]),
            "cost": float(leg_cost[k]),
            "ref_price": ref_px[sym],
        }
        for k, sym in enumerate(target.index)
        if live[k]
    ]
    return pd.Series(h_new / e_post, index=w.index), cost, legs, True


# --- the engine -----------------------------------------------------------------


def _utc_midnight(x) -> bool:
    return (
        isinstance(x, pd.Timestamp)
        and x.tzinfo is not None
        and x.utcoffset() == pd.Timedelta(0)
        and str(x.tz) in ("UTC", "Etc/UTC")
        and x == x.normalize()
    )


def _check_schedule(schedule: Mapping[pd.Timestamp, pd.Timestamp]) -> dict[pd.Timestamp, pd.Timestamp]:
    """The trade schedule as a dict {trade date: signal date}; ValueError for a bad entry."""
    out: dict[pd.Timestamp, pd.Timestamp] = {}
    for key, value in schedule.items():
        if not (_utc_midnight(key) and _utc_midnight(value)):
            raise ValueError(f"schedule entry {key!r}: {value!r}: each date must be a UTC midnight Timestamp")
        if not value < key:
            raise ValueError(f"schedule entry {key!r}: {value!r}: the signal date must be before the trade date")
        out[key] = value
    return out


def run_backtest(
    weights_fn,
    *,
    factors: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
    initial_equity: float = 10_000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
    gross_cap: float = 1.0,
    rebalance_weekday: int = 0,  # 0 = Monday 00:00 UTC
    benchmark: str | None = BENCHMARK,
    accounting: str = ACCOUNTING,
    schedule: Mapping[pd.Timestamp, pd.Timestamp] | None = None,
) -> Result:
    """Run one backtest. weights_fn(cross_section) -> Series of target weights.

    cross_section is the factor table's rows for ONE date (the signal date),
    indexed by symbol. Returned weights are fractions of (post-fee) equity;
    symbols not returned are treated as target 0 (i.e. every call is a full
    statement of the desired portfolio).

    factors: the factor rows, long format (date, symbol, value columns).

    accounting: ACCOUNTING_RECONCILED, the only mode (module docstring).

    schedule: None (default) trades on each day t with t.weekday() ==
    rebalance_weekday, on the factor row of t - 1 day. A mapping
    {trade date: signal date} trades only on its keys, each on the factor row
    of its value; rebalance_weekday is then not used. Each key and value must
    be a UTC midnight Timestamp, and each value must be before its key.
    """
    if accounting not in ACCOUNTING_MODES:
        raise ValueError(f"accounting must be one of {ACCOUNTING_MODES}, got {accounting!r}")
    trade_on = None if schedule is None else _check_schedule(schedule)
    if prices is None:
        prices = data_io.close_matrix()
    if start:
        prices = prices[prices.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        prices = prices[prices.index <= pd.Timestamp(end, tz="UTC")]
    if prices.empty:
        raise SystemExit(f"No completed bars in [{start}, {end}] — nothing to backtest.")
    if funding is None:
        funding = data_io.funding_matrix(cast(pd.DatetimeIndex, prices.index), prices.columns)
    funding = funding.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

    # bar t return: close[t]/close[t-1]-1. NaN (symbol not listed on that bar, or the
    # first bar) contributes 0 pnl — factor coverage rules mean no strategy
    # should be holding such a symbol anyway.
    rets = prices.pct_change(fill_method=None)

    by_date = {d: g.set_index("symbol") for d, g in factors.groupby("date")}

    cost_rate = (fee_bps + slippage_bps) / 1e4
    w = pd.Series(0.0, index=prices.columns)  # weights held during current bar
    equity = float(initial_equity)
    trades, daily_rows = [], []
    n_rebalances = 0

    dates = prices.index
    for i, t in enumerate(dates):
        if i == 0:
            daily_rows.append((t, equity, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue

        # -- 1) rebalance at t 00:00 using the factor row stamped t-1 --------
        # (with a schedule: only on its trade dates, on the row of their signal date)
        cost = 0.0
        if trade_on is None:
            signal = t - pd.Timedelta(days=1) if t.weekday() == rebalance_weekday else None
        else:
            signal = trade_on.get(t)
        if signal is not None:
            cs = by_date.get(signal)
            if cs is not None and not cs.empty:
                target = pd.Series(weights_fn(cs), dtype=float).reindex(prices.columns).fillna(0.0)
                gross = target.abs().sum()
                if gross > gross_cap:
                    target *= gross_cap / gross
                ref_px = prices.iloc[i - 1]  # price at t 00:00 = close of bar t-1
                w_new, cost, legs, traded = _rebalance_reconciled(t, target, w, equity, cost_rate, ref_px)
                if traded:
                    n_rebalances += 1
                    trades.extend(legs)
                    w = w_new

        # -- 2) accrue bar t: costs at the open, then price + funding pnl ----
        # Reconciled: equity - cost is E_post, and w * E_post are exactly the
        # holdings the trade log implies (h_prev + signed notional).
        equity -= cost
        r = rets.iloc[i].fillna(0.0)
        r_p = float((w * r).sum())
        f_p = float(-(w * funding.iloc[i]).sum())  # positive funding: longs pay
        pnl_growth = 1.0 + r_p + f_p
        funding_dollars = equity * f_p  # equity here = post-cost open equity
        equity *= pnl_growth
        daily_rows.append((t, equity, r_p, f_p, funding_dollars, cost, w.abs().sum()))
        if equity <= 0:
            print(f"BUST on {t.date()} — equity wiped out. Stopping.")
            break

        # -- 3) drift weights: positions grow with their own return ----------
        if pnl_growth > 0:
            w = w * (1.0 + r) / pnl_growth

    daily = pd.DataFrame(
        daily_rows, columns=["date", "equity", "ret", "funding", "funding_usd", "cost", "gross"]
    ).set_index("date")
    trades_df = pd.DataFrame(
        trades, columns=["date", "symbol", "side", "prev_w", "target_w", "delta_w", "notional", "cost", "ref_price"]
    )

    # benchmark: BTC buy & hold, same start equity, no fees (it's a yardstick,
    # not a strategy)
    eq = daily["equity"]
    if benchmark is not None and benchmark in prices.columns:
        bpx = prices[benchmark].ffill()
        bench = initial_equity * bpx / bpx.iloc[0]
    else:
        bench = pd.Series(initial_equity, index=daily.index)
    equity_df = pd.DataFrame({"equity": eq, "benchmark": bench.reindex(eq.index)})

    stats = _stats(equity_df, daily, trades_df, initial_equity)
    stats["accounting"] = accounting  # the version tag of the accounting convention
    return Result(equity=equity_df, trades=trades_df, daily=daily, stats=stats)


def _stats(equity_df, daily, trades, initial_equity) -> dict:
    eq, bench = equity_df["equity"], equity_df["benchmark"]
    n_days = len(eq)
    years = max(n_days / 365.0, 1e-9)
    dr = eq.pct_change().dropna()

    def max_dd(s):
        return float((s / s.cummax() - 1.0).min()) if len(s) else 0.0

    def cagr(s):
        return float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) if s.iloc[0] > 0 else np.nan

    rebal_turnover = (
        trades.groupby("date")["notional"].sum() / daily["equity"].reindex(trades["date"].unique())
        if len(trades)
        else pd.Series(dtype=float)
    )
    return {
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "n_days": n_days,
        "initial_equity": float(initial_equity),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "cagr": cagr(eq),
        "ann_vol": float(dr.std() * np.sqrt(365)) if len(dr) > 1 else 0.0,
        "sharpe": float(dr.mean() / dr.std() * np.sqrt(365)) if len(dr) > 1 and dr.std() > 0 else 0.0,
        "max_dd": max_dd(eq),
        "bench_total_return": float(bench.iloc[-1] / bench.iloc[0] - 1),
        "bench_cagr": cagr(bench),
        "bench_max_dd": max_dd(bench),
        "n_rebalances": int(trades["date"].nunique()) if len(trades) else 0,
        "n_trades": len(trades),
        "avg_turnover": float(rebal_turnover.mean()) if len(rebal_turnover) else 0.0,
        "fees_total": float(daily["cost"].sum()),
        "funding_total": float(daily["funding_usd"].sum()),
    }


# --- the equal-weight weights ----------------------------------------------------


def demo_market(cs: pd.DataFrame) -> pd.Series:
    """Equal-weight long every coin with a factor row: the weights of the equal-weight control
    book (book.py)."""
    return pd.Series(1.0 / len(cs), index=cs.index)
