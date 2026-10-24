"""Cost-aware daily backtest of a target-weight panel.

Timing convention (identical to the label convention of ``DataStore.fwd_return``):
weights decided at the close of decision date ``t`` are traded at the close of ``t + lag``
and held over the next day, so

    ret_t = sum_i w_{i,t} * (close_{i,t+lag+1} / close_{i,t+lag} - 1)

**All result series are indexed by the decision date ``t``**, not by the date on which
the P&L is realised.  This keeps weights, returns and turnover on one index so they can
be joined with the factor panels that produced them; shift by ``lag + 1`` if you need a
calendar-accurate equity curve.

The backtest adds realistic timing (no fill at the signal close), transaction costs and
trading halts, so it shows whether a ranking survives them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("alphaforge.portfolio.backtest")

__all__ = ["BacktestResult", "backtest"]


@dataclass
class BacktestResult:
    """Container for backtest output.

    All series share one index: the **decision dates** (see module docstring).

    Attributes:
        returns: daily portfolio return net of transaction costs.
        gross_returns: daily portfolio return before costs.
        turnover: ``sum_i |w_{i,t} - w_{i,t-1}|`` per day (first day: ``sum_i |w_{i,0}|``).
        equity: cumulative product of ``1 + returns`` (starts at ``1 + returns[0]``).
        ann_return: mean daily net return x ``periods_per_year``.
        ann_vol: sample std of daily net returns x ``sqrt(periods_per_year)``.
        sharpe: ``ann_return / ann_vol`` (``NaN`` when vol is 0).
        max_dd: maximum peak-to-trough drawdown of ``equity`` as a **non-positive** number
            (e.g. ``-0.12`` = -12%).
        calmar: ``ann_return / |max_dd|`` (``NaN`` when there is no drawdown).
        avg_turnover: mean daily turnover.
        total_cost: sum of daily costs, in return units (``cost_bps / 1e4 * turnover``).
        hit_rate: share of days with a strictly positive net return.
        costs: daily cost series (extra).
        drawdown: daily drawdown series ``equity / cummax(equity) - 1`` (extra).
        positions: the effective weights actually used -- after zeroing untradable names
            and renormalising each leg (extra; useful for exposure diagnostics).
        n_days: number of days in the backtest.
        start: first decision date.
        end: last decision date.
        cost_bps: cost assumption used.
        lag: entry lag used.
        periods_per_year: annualisation factor used.
    """

    returns: pd.Series = field(repr=False)
    gross_returns: pd.Series = field(repr=False)
    turnover: pd.Series = field(repr=False)
    equity: pd.Series = field(repr=False)
    ann_return: float
    ann_vol: float
    sharpe: float
    max_dd: float
    calmar: float
    avg_turnover: float
    total_cost: float
    hit_rate: float
    costs: pd.Series | None = field(default=None, repr=False)
    drawdown: pd.Series | None = field(default=None, repr=False)
    positions: pd.DataFrame | None = field(default=None, repr=False)
    n_days: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    cost_bps: float = math.nan
    lag: int = 1
    periods_per_year: int = 252

    @property
    def total_return(self) -> float:
        """Total compounded net return over the backtest (``equity[-1] - 1``)."""
        if self.equity is None or len(self.equity) == 0:
            return math.nan
        return float(self.equity.iloc[-1] - 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Scalar (numeric) statistics only -- safe for CSV/JSON tables.

        Values are Python floats/ints; ``sharpe``/``calmar`` may be ``NaN`` when undefined
        (zero volatility / no drawdown), which callers writing strict JSON should handle.
        """
        return {
            "ann_return": float(self.ann_return),
            "ann_vol": float(self.ann_vol),
            "sharpe": float(self.sharpe),
            "max_dd": float(self.max_dd),
            "calmar": float(self.calmar),
            "avg_turnover": float(self.avg_turnover),
            "total_cost": float(self.total_cost),
            "hit_rate": float(self.hit_rate),
            "total_return": float(self.total_return),
            "n_days": int(self.n_days),
            "cost_bps": float(self.cost_bps),
            "lag": int(self.lag),
        }

    def summary(self) -> str:
        """Multi-line, human-readable summary."""
        start = self.start.date() if isinstance(self.start, pd.Timestamp) else self.start
        end = self.end.date() if isinstance(self.end, pd.Timestamp) else self.end
        lines = [
            f"Backtest summary: {self.n_days} days, {start} -> {end} (decision dates), "
            f"lag={self.lag}, cost={self.cost_bps:.1f} bps",
            f"  ann. return    : {self.ann_return * 100:8.2f}%",
            f"  ann. vol       : {self.ann_vol * 100:8.2f}%",
            f"  Sharpe         : {self.sharpe:8.2f}",
            f"  max drawdown   : {self.max_dd * 100:8.2f}%",
            f"  Calmar         : {self.calmar:8.2f}",
            f"  total return   : {self.total_return * 100:8.2f}%",
            f"  hit rate       : {self.hit_rate * 100:8.1f}%",
            f"  avg turnover   : {self.avg_turnover:8.3f}  (daily sum|dw|)",
            f"  total cost     : {self.total_cost * 100:8.2f}%  (cumulative, return units)",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _forward_returns(close: pd.DataFrame, lag: int) -> pd.DataFrame:
    """One-day return from close ``t+lag`` to ``t+lag+1``, aligned to decision date ``t``.

    Name contains "forward" on purpose: this is a *label*, it looks ahead by design.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd = close.shift(-(lag + 1)) / close.shift(-lag) - 1.0
    return fwd.replace([np.inf, -np.inf], np.nan)


def _effective_weights(w: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    """Zero blocked names and rescale each leg so its gross exposure is preserved.

    If an entire leg is blocked there is nothing to rescale and the leg is dropped.
    ``NaN`` weights are treated as 0 (not in the book).
    """
    w = np.where(np.isfinite(w), w, 0.0)
    w_eff = np.where(blocked, 0.0, w)

    long_before = np.where(w > 0, w, 0.0).sum(axis=1)
    long_after = np.where(w_eff > 0, w_eff, 0.0).sum(axis=1)
    short_before = np.where(w < 0, -w, 0.0).sum(axis=1)
    short_after = np.where(w_eff < 0, -w_eff, 0.0).sum(axis=1)

    f_long = np.divide(long_before, long_after, out=np.ones_like(long_before), where=long_after > 0)
    f_short = np.divide(short_before, short_after, out=np.ones_like(short_before), where=short_after > 0)
    return np.where(w_eff > 0, w_eff * f_long[:, None], w_eff * f_short[:, None])


def _max_drawdown(equity: np.ndarray) -> tuple[float, np.ndarray]:
    """Drawdown series and its minimum (a non-positive number)."""
    if equity.size == 0:
        return math.nan, equity
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, equity / peak - 1.0, 0.0)
    return float(dd.min()), dd


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def backtest(
    weights: pd.DataFrame,
    store: Any,
    *,
    cost_bps: float = 10.0,
    lag: int = 1,
    start: Any = None,
    end: Any = None,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Daily, cost-aware backtest of a target-weight panel against a ``DataStore``.

    Mechanics (per decision date ``t``):

    1. ``r_{i,t} = close_{i,t+lag+1} / close_{i,t+lag} - 1`` (forward, aligned to ``t``).
    2. Names that are untradable at ``t+lag`` (``tradable == 0``) or whose forward return is
       undefined (missing price) get weight ``0``; each leg is then rescaled so that its
       gross exposure is unchanged (you cannot trade a halted name, so its capital is
       spread over the rest of the leg).  ``NaN`` weights mean "not in the book" (``0``).
    3. ``gross_t = sum_i w_eff_{i,t} r_{i,t}``;
       ``turnover_t = sum_i |w_eff_{i,t} - w_eff_{i,t-1}|`` with ``w_{-1} = 0`` (the first day
       pays for building the whole book); ``cost_t = cost_bps / 1e4 * turnover_t``;
       ``net_t = gross_t - cost_t``; ``equity = cumprod(1 + net)``.
    4. Statistics: ``ann_return = mean(net) * periods_per_year``,
       ``ann_vol = std(net, ddof=1) * sqrt(periods_per_year)``, ``sharpe = ann_return / ann_vol``,
       ``max_dd`` = min of ``equity / cummax(equity) - 1`` (non-positive),
       ``calmar = ann_return / |max_dd|``, ``hit_rate`` = share of days with ``net > 0``.

    Rows handled specially:

    * ``start``/``end`` slice the *decision* dates (labels inclusive); prices after ``end``
      are still used for the forward returns of the last decision dates, if the store has
      them.
    * Leading rows without any position -- all ``NaN`` *or* all zero, e.g. a factor's
      warm-up period, or the rows :func:`hold_weights` emits before the first rebalance
      -- are dropped so the statistics start when the strategy goes live.  Interior
      all-``NaN`` rows are treated as flat (zero weights: the book is unwound and rebuilt,
      and turnover is charged for both).
    * Dates whose forward return is ``NaN`` for every asset -- in practice the trailing
      ``lag + 1`` dates of the store -- are dropped.

    The linear cost on turnover and the refusal to trade halted names show when a
    high-turnover signal loses money after costs.

    Args:
        weights: target weight panel (dates x assets); decision-date convention.
        store: object exposing ``.dates``, ``.assets``, ``.get(name)`` and ``.has(name)``
            (a ``DataStore``); ``close`` is required, ``tradable`` optional.
        cost_bps: one-way linear transaction cost in basis points of traded notional.
        lag: entry lag in days (``1`` = trade at the next close after the signal).
        start: first decision date (inclusive), or ``None``.
        end: last decision date (inclusive), or ``None``.
        periods_per_year: annualisation factor (252 trading days).

    Returns:
        :class:`BacktestResult` whose series are indexed by decision date.

    Raises:
        ValueError: on ``lag < 0``, negative ``cost_bps``, or if no usable rows remain.
    """
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas DataFrame panel")
    if int(lag) < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    lag = int(lag)

    close = store.get("close")
    fwd = _forward_returns(close, lag)
    tradable_fwd = store.get("tradable").shift(-lag) if store.has("tradable") else None

    w = weights.loc[start:end]
    missing_assets = w.columns.difference(close.columns)
    if len(missing_assets):
        logger.warning(
            "backtest: %d weight columns are not in the store and are untradable (e.g. %s)",
            len(missing_assets),
            list(missing_assets[:3]),
        )

    # Drop leading rows with no position at all -- all-NaN or all-zero (the strategy holds no position).
    any_position = (w.notna() & (w != 0.0)).any(axis=1).to_numpy()
    if not any_position.any():
        raise ValueError("backtest: weights contain no non-zero finite values in the requested window")
    first_live = int(np.argmax(any_position))
    if first_live:
        logger.debug("backtest: dropping %d leading rows without any position", first_live)
        w = w.iloc[first_live:]

    fwd = fwd.reindex(index=w.index, columns=w.columns)
    has_ret = fwd.notna().any(axis=1).to_numpy()
    if not has_ret.all():
        n_drop = int((~has_ret).sum())
        logger.debug("backtest: dropping %d dates with no forward return (trailing %d expected)", n_drop, lag + 1)
        w = w.loc[has_ret]
        fwd = fwd.loc[has_ret]
    if len(w.index) == 0:
        raise ValueError("backtest: no dates with a defined forward return (store too short for lag)")

    W = w.to_numpy(dtype=float)
    R = fwd.to_numpy(dtype=float)
    blocked = ~np.isfinite(R)
    if tradable_fwd is not None:
        T = tradable_fwd.reindex(index=w.index, columns=w.columns).to_numpy(dtype=float)
        blocked |= T == 0.0  # NaN tradability (unknown) is not treated as a halt

    W_eff = _effective_weights(W, blocked)
    R0 = np.where(np.isfinite(R), R, 0.0)
    gross = (W_eff * R0).sum(axis=1)

    prev = np.vstack([np.zeros((1, W_eff.shape[1])), W_eff[:-1]])
    turnover = np.abs(W_eff - prev).sum(axis=1)
    costs = cost_bps / 1e4 * turnover
    net = gross - costs
    equity = np.cumprod(1.0 + net)

    n = int(net.size)
    ann_return = float(net.mean() * periods_per_year)
    ann_vol = float(net.std(ddof=1) * math.sqrt(periods_per_year)) if n > 1 else math.nan
    sharpe = ann_return / ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else math.nan
    max_dd, dd = _max_drawdown(equity)
    calmar = ann_return / abs(max_dd) if np.isfinite(max_dd) and max_dd < 0 else math.nan
    hit_rate = float((net > 0).mean())
    avg_turnover = float(turnover.mean())
    total_cost = float(costs.sum())

    idx = w.index
    result = BacktestResult(
        returns=pd.Series(net, index=idx, name="net"),
        gross_returns=pd.Series(gross, index=idx, name="gross"),
        turnover=pd.Series(turnover, index=idx, name="turnover"),
        equity=pd.Series(equity, index=idx, name="equity"),
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=float(sharpe),
        max_dd=max_dd,
        calmar=float(calmar),
        avg_turnover=avg_turnover,
        total_cost=total_cost,
        hit_rate=hit_rate,
        costs=pd.Series(costs, index=idx, name="cost"),
        drawdown=pd.Series(dd, index=idx, name="drawdown"),
        positions=pd.DataFrame(W_eff, index=idx, columns=w.columns),
        n_days=n,
        start=idx[0],
        end=idx[-1],
        cost_bps=float(cost_bps),
        lag=lag,
        periods_per_year=int(periods_per_year),
    )
    logger.info(
        "backtest: %d days %s->%s | ann_ret=%.2f%% vol=%.2f%% sharpe=%.2f max_dd=%.2f%% "
        "turnover=%.3f cost=%.1fbps total_cost=%.2f%%",
        n,
        idx[0].date(),
        idx[-1].date(),
        ann_return * 100,
        ann_vol * 100 if np.isfinite(ann_vol) else float("nan"),
        sharpe,
        max_dd * 100,
        avg_turnover,
        cost_bps,
        total_cost * 100,
    )
    return result
