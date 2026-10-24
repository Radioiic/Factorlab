"""AlphaForge evaluation: IC / ICIR, quantile spreads, turnover and factor reports.

Public API::

    rowwise_corr, ic_series, ic_stats, ic_decay
    assign_quantiles, quantile_returns, long_short_returns, monotonicity
    rank_autocorr, quantile_turnover
    FactorReport, evaluate_factor
"""

from alphaforge.evaluation.ic import ic_decay, ic_series, ic_stats, rowwise_corr
from alphaforge.evaluation.quantile import (
    assign_quantiles,
    long_short_returns,
    monotonicity,
    quantile_returns,
)
from alphaforge.evaluation.report import FactorReport, evaluate_factor
from alphaforge.evaluation.turnover import quantile_turnover, rank_autocorr

__all__ = [
    "rowwise_corr",
    "ic_series",
    "ic_stats",
    "ic_decay",
    "assign_quantiles",
    "quantile_returns",
    "long_short_returns",
    "monotonicity",
    "rank_autocorr",
    "quantile_turnover",
    "FactorReport",
    "evaluate_factor",
]
