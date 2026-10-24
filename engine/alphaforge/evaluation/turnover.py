"""Factor stability: rank autocorrelation and quantile turnover.

Rank autocorrelation and top-bucket turnover measure the trading cost of a
signal before a backtest, and they feed the turnover penalty of the miner.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alphaforge.evaluation.ic import rowwise_corr
from alphaforge.evaluation.quantile import assign_quantiles

logger = logging.getLogger("alphaforge.evaluation.turnover")

__all__ = ["rank_autocorr", "quantile_turnover"]


def rank_autocorr(factor: pd.DataFrame, lag: int = 1) -> pd.Series:
    """Per-date Spearman correlation between ``factor[t]`` and ``factor[t-lag]``.

    Args:
        factor: Factor panel.
        lag: Number of dates to look back (>= 1).

    Returns:
        ``pd.Series`` named ``"rank_autocorr"`` on the factor's date index;
        the first ``lag`` dates are NaN (no earlier row), as are dates with
        fewer than 5 assets valid at both times.

    The autocorrelation of the cross-sectional ranking shows how much of a
    rank-based book changes each period.

    No look-ahead: row ``t`` uses rows ``t`` and ``t - lag`` only.
    """
    lag = int(lag)
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    s = rowwise_corr(factor, factor.shift(lag), method="spearman")
    s.name = "rank_autocorr"
    return s


def quantile_turnover(
    factor: pd.DataFrame,
    n_quantiles: int = 10,
    quantile: int | None = None,
) -> pd.Series:
    """Fraction of a quantile's members on ``t`` that were not members on ``t-1``.

    Args:
        factor: Factor panel.
        n_quantiles: Number of buckets used by :func:`assign_quantiles`.
        quantile: Bucket to track, ``1..n``; default ``n`` (the top bucket,
            i.e. the long leg).

    Returns:
        ``pd.Series`` in ``[0, 1]`` named ``"turnover_q<k>"`` on the factor's
        date index.  NaN on the first date and on any date where the bucket is
        empty on ``t`` or on ``t-1`` (there is no previous membership to
        compare against, e.g. during a factor's warm-up window).

    The share of new names in the top quantile each period is close to
    the round-trip trading cost of the long leg.

    No look-ahead: row ``t`` uses rows ``t`` and ``t - 1`` only.
    """
    n = int(n_quantiles)
    k = n if quantile is None else int(quantile)
    if not 1 <= k <= n:
        raise ValueError(f"quantile must be in 1..{n}, got {quantile!r}")
    member = assign_quantiles(factor, n).to_numpy() == float(k)
    prev = np.zeros_like(member)
    prev[1:] = member[:-1]
    cur_cnt = member.sum(axis=1)
    prev_cnt = prev.sum(axis=1)
    new_cnt = (member & ~prev).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        turn = np.where((cur_cnt > 0) & (prev_cnt > 0), new_cnt / cur_cnt, np.nan)
    turn[0] = np.nan
    return pd.Series(turn, index=factor.index, name=f"turnover_q{k}")
