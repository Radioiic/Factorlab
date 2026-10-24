"""One-call factor evaluation producing a :class:`FactorReport`.

Each candidate, written by hand or mined, gets the same scorecard: IC, ICIR,
quantile spread, turnover, coverage and decay.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.evaluation.ic import ic_decay, ic_series, ic_stats
from alphaforge.evaluation.quantile import monotonicity, quantile_returns
from alphaforge.evaluation.turnover import quantile_turnover, rank_autocorr

logger = logging.getLogger("alphaforge.evaluation.report")

__all__ = ["FactorReport", "evaluate_factor"]

_TRADING_DAYS = 252


@dataclass
class FactorReport:
    """Scorecard of one factor over one evaluation window.

    Scalar fields are annualised as documented in :func:`evaluate_factor`.
    ``ic_*`` fields use the **Pearson** IC, ``rank_ic_*`` the **Spearman**
    (rank) IC.  ``ls_max_dd`` is reported as a non-positive number
    (``-0.12`` = a 12% drawdown of cumulative daily-equivalent return).
    """

    name: str
    expr: str | None
    horizon: int
    lag: int
    n_quantiles: int
    start: pd.Timestamp
    end: pd.Timestamp
    ic_mean: float
    ic_std: float
    icir: float
    icir_ann: float
    ic_t: float
    ic_p: float
    ic_pos_ratio: float
    rank_ic_mean: float
    rank_icir: float
    ls_ann_return: float
    ls_ann_vol: float
    ls_sharpe: float
    ls_max_dd: float
    top_q_ann_return: float
    bottom_q_ann_return: float
    monotonicity: float
    turnover_mean: float
    autocorr_mean: float
    coverage: float
    ic_series: pd.Series
    rank_ic_series: pd.Series
    quantile_returns: pd.DataFrame
    ls_series: pd.Series
    ic_decay: pd.Series

    # names of the non-scalar (series / frame) fields
    _SERIES_FIELDS = ("ic_series", "rank_ic_series", "quantile_returns", "ls_series", "ic_decay")

    @classmethod
    def scalar_fields(cls) -> list[str]:
        """Names of the scalar fields, in declaration order."""
        return [f.name for f in dataclasses.fields(cls) if f.name not in cls._SERIES_FIELDS]

    def to_dict(self, *, json_safe: bool = False) -> dict[str, Any]:
        """Scalar fields only (no series / frames).

        Args:
            json_safe: If True, ``start``/``end`` become ISO strings and
                non-finite floats become ``None`` so the dict can be passed
                straight to ``json.dumps``.
        """
        d = {k: getattr(self, k) for k in self.scalar_fields()}
        if json_safe:
            for k, v in list(d.items()):
                if isinstance(v, pd.Timestamp):
                    d[k] = v.isoformat()
                elif isinstance(v, (float, np.floating)) and not math.isfinite(float(v)):
                    d[k] = None
                elif isinstance(v, np.generic):
                    d[k] = v.item()
        return d

    def summary(self) -> str:
        """Multi-line human-readable scorecard."""
        n_dates = int(self.ic_series.notna().sum())
        lines = [
            f"FactorReport: {self.name}" + (f"    expr: {self.expr}" if self.expr else ""),
            f"  window     : {self.start.date()} -> {self.end.date()}  "
            f"({n_dates} IC obs)  horizon={self.horizon}  lag={self.lag}  "
            f"quantiles={self.n_quantiles}",
            f"  IC         : mean {self.ic_mean:+.4f}  std {self.ic_std:.4f}  "
            f"ICIR {self.icir:+.3f}  ICIR_ann {self.icir_ann:+.2f}  "
            f"t {self.ic_t:+.2f}  p {self.ic_p:.2e}  pos {self.ic_pos_ratio:.2%}",
            f"  Rank IC    : mean {self.rank_ic_mean:+.4f}  ICIR {self.rank_icir:+.3f}",
            f"  Long-short : ann.ret {self.ls_ann_return:+.2%}  ann.vol {self.ls_ann_vol:.2%}  "
            f"Sharpe {self.ls_sharpe:+.2f}  maxDD {self.ls_max_dd:+.2%}",
            f"  Quantiles  : top {self.top_q_ann_return:+.2%}  bottom {self.bottom_q_ann_return:+.2%}  "
            f"monotonicity {self.monotonicity:+.3f}",
            f"  Stability  : top-Q turnover {self.turnover_mean:.3f}  "
            f"rank autocorr {self.autocorr_mean:.3f}  coverage {self.coverage:.2%}",
            "  IC decay   : " + "  ".join(f"h{int(h)} {v:+.4f}" for h, v in self.ic_decay.items()),
        ]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()

    @staticmethod
    def table(reports: list[FactorReport]) -> pd.DataFrame:
        """One row per report (indexed by ``name``), scalar columns only."""
        cols = [c for c in FactorReport.scalar_fields() if c != "name"]
        if not reports:
            return pd.DataFrame(columns=cols, index=pd.Index([], name="name"))
        rows = [r.to_dict() for r in reports]
        return pd.DataFrame(rows).set_index("name")[cols]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _slice_dates(index: pd.DatetimeIndex, start: Any, end: Any) -> pd.DatetimeIndex:
    """Sub-range of ``index`` with labels in ``[start, end]`` (both inclusive)."""
    lo = pd.Timestamp(start) if start is not None else None
    hi = pd.Timestamp(end) if end is not None else None
    mask = np.ones(len(index), dtype=bool)
    if lo is not None:
        mask &= index >= lo
    if hi is not None:
        mask &= index <= hi
    return index[mask]


def _max_drawdown(ls: pd.Series, horizon: int) -> float:
    """Max drawdown of the cumulative-sum equity of ``ls / horizon`` (<= 0)."""
    x = ls.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    equity = np.cumsum(x / horizon)
    peak = np.maximum(np.maximum.accumulate(equity), 0.0)  # equity starts at 0
    return float((equity - peak).min())


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if (math.isfinite(den) and den > 0.0 and math.isfinite(num)) else float("nan")


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def evaluate_factor(
    factor: pd.DataFrame,
    store: Any,
    *,
    name: str = "factor",
    expr: str | None = None,
    horizon: int = 5,
    lag: int = 1,
    n_quantiles: int = 10,
    start: Any = None,
    end: Any = None,
    decay_horizons: Iterable[int] = (1, 2, 3, 5, 10, 20),
    periods_per_year: int = _TRADING_DAYS,
) -> FactorReport:
    """Evaluate a factor against ``store.fwd_return(horizon, lag)`` on ``[start, end]``.

    Args:
        factor: Factor panel; may cover a date sub-range of the store (it is
            inner-aligned with the label on dates and assets).
        store: Object exposing ``fwd_return(horizon, lag)`` (a ``DataStore``).
        name: Label used in tables and summaries.
        expr: Optional expression string that produced the factor.
        horizon: Holding period ``h`` in days.
        lag: Entry lag (signal at close ``t`` traded at close ``t + lag``).
        n_quantiles: Number of quantile buckets.
        start: First date (inclusive) of the evaluation window, or None.
        end: Last date (inclusive) of the evaluation window, or None.
        decay_horizons: Horizons for the IC-decay curve.
        periods_per_year: Trading days per year used for annualisation.

    Returns:
        :class:`FactorReport`.

    Annualisation (overlap caveat): ``ls_series`` and the quantile returns are
    *h*-day returns observed every day, so consecutive observations overlap
    and are autocorrelated.  ``ann_return = mean * 252 / h`` and
    ``ann_vol = std * sqrt(252 / h)`` treat them as ``252 / h`` independent
    periods per year, which is the right scale for the mean but only an
    approximation for the vol (it ignores the overlap-induced
    autocorrelation).  ``ls_sharpe = ann_return / ann_vol`` and
    ``ls_max_dd`` is taken on the cumulative sum of ``ls_series / h`` (a
    daily-equivalent additive equity curve starting at 0; reported <= 0).
    ``coverage`` is the mean over dates of
    ``count(non-NaN factor) / count(non-NaN fwd_ret)``; dates with no known
    label are excluded from that mean.

    Raises:
        ValueError: on invalid parameters or an empty evaluation window.

    One scorecard for each factor and window makes the train, validation
    and test results comparable.
    """
    horizon = int(horizon)
    lag = int(lag)
    n_quantiles = int(n_quantiles)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    if not isinstance(factor, pd.DataFrame):
        raise TypeError("factor must be a pd.DataFrame panel")

    fwd_full = store.fwd_return(horizon, lag)
    dates = factor.index.intersection(fwd_full.index)
    dates = _slice_dates(pd.DatetimeIndex(dates).sort_values(), start, end)
    if len(dates) == 0:
        raise ValueError(
            f"evaluate_factor({name!r}): no dates in window [{start}, {end}] shared by the factor and the store"
        )
    assets = factor.columns.intersection(fwd_full.columns)
    if len(assets) == 0:
        raise ValueError(f"evaluate_factor({name!r}): factor and store share no assets")
    f = factor.reindex(index=dates, columns=assets).astype(float)
    r = fwd_full.reindex(index=dates, columns=assets).astype(float)

    # --- information coefficients -------------------------------------- #
    ic = ic_series(f, r, method="pearson")
    ric = ic_series(f, r, method="spearman")
    st = ic_stats(ic, periods_per_year=periods_per_year)
    rst = ic_stats(ric, periods_per_year=periods_per_year)

    # --- quantile / long-short ------------------------------------------ #
    qret = quantile_returns(f, r, n_quantiles)
    ls = qret[n_quantiles] - qret[1]
    ls.name = "long_short"
    scale_mean = periods_per_year / horizon
    scale_vol = math.sqrt(periods_per_year / horizon)
    ls_valid = ls.dropna()
    ls_ann_return = float(ls_valid.mean() * scale_mean) if len(ls_valid) else float("nan")
    ls_ann_vol = float(ls_valid.std(ddof=1) * scale_vol) if len(ls_valid) >= 2 else float("nan")
    ls_sharpe = _safe_div(ls_ann_return, ls_ann_vol)
    ls_max_dd = _max_drawdown(ls, horizon)
    top_ann = float(qret[n_quantiles].mean() * scale_mean)
    bottom_ann = float(qret[1].mean() * scale_mean)
    mono = monotonicity(qret)

    # --- stability / coverage ------------------------------------------- #
    turnover_mean = float(quantile_turnover(f, n_quantiles).mean())
    autocorr_mean = float(rank_autocorr(f, 1).mean())
    n_f = f.notna().sum(axis=1).to_numpy(dtype=float)
    n_r = r.notna().sum(axis=1).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov_ratio = np.where(n_r > 0, n_f / n_r, np.nan)
    coverage = float(np.nanmean(cov_ratio)) if np.isfinite(cov_ratio).any() else float("nan")

    decay = ic_decay(f, store, horizons=decay_horizons, lag=lag, method="spearman")

    report = FactorReport(
        name=name,
        expr=expr,
        horizon=horizon,
        lag=lag,
        n_quantiles=n_quantiles,
        start=pd.Timestamp(dates[0]),
        end=pd.Timestamp(dates[-1]),
        ic_mean=st["mean"],
        ic_std=st["std"],
        icir=st["ir"],
        icir_ann=st["ir_ann"],
        ic_t=st["t_stat"],
        ic_p=st["p_value"],
        ic_pos_ratio=st["positive_ratio"],
        rank_ic_mean=rst["mean"],
        rank_icir=rst["ir"],
        ls_ann_return=ls_ann_return,
        ls_ann_vol=ls_ann_vol,
        ls_sharpe=ls_sharpe,
        ls_max_dd=ls_max_dd,
        top_q_ann_return=top_ann,
        bottom_q_ann_return=bottom_ann,
        monotonicity=float(mono),
        turnover_mean=turnover_mean,
        autocorr_mean=autocorr_mean,
        coverage=coverage,
        ic_series=ic,
        rank_ic_series=ric,
        quantile_returns=qret,
        ls_series=ls,
        ic_decay=decay,
    )
    logger.debug(
        "evaluate_factor(%s): %s -> %s, n=%d, IC %.4f, ICIR_ann %.2f, rankIC %.4f, LS Sharpe %.2f, "
        "turnover %.3f, coverage %.3f",
        name,
        report.start.date(),
        report.end.date(),
        st["n"],
        st["mean"],
        st["ir_ann"],
        rst["mean"],
        ls_sharpe,
        turnover_mean,
        coverage,
    )
    return report
