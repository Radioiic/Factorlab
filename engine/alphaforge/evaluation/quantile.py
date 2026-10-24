"""Quantile (bucket) analysis of a cross-sectional factor.

On every date the universe is ranked by the factor and split into
``n_quantiles`` equal-count buckets; the equal-weight forward return of each
bucket, and the spread between the top and bottom bucket, show whether the
signal is *tradable* rather than merely correlated.

The quantile returns show where the signal is, whether it is monotone, and
how much of the spread comes from the short leg.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from alphaforge.evaluation.ic import _align_pair

logger = logging.getLogger("alphaforge.evaluation.quantile")

__all__ = ["assign_quantiles", "quantile_returns", "long_short_returns", "monotonicity"]


def _check_n(n_quantiles: int) -> int:
    n = int(n_quantiles)
    if n < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles!r}")
    return n


def assign_quantiles(factor: pd.DataFrame, n_quantiles: int = 10) -> pd.DataFrame:
    """Assign each asset to a factor quantile ``1..n`` on every date.

    Ranks are computed among the non-NaN assets of each row (average rank for
    ties) and the bucket is ``ceil(rank * n / count)`` clipped to ``[1, n]``,
    so the highest factor value lands in bucket ``n`` and the lowest in
    bucket ``1``.  Tied values share a bucket (they are never split
    arbitrarily), which can leave a bucket empty when the factor has massive
    ties; ``NaN`` inputs stay ``NaN``.

    Args:
        factor: Factor panel (dates x assets).
        n_quantiles: Number of buckets (>= 2).

    Returns:
        Float panel of bucket labels ``1.0 .. n`` with the input's NaN pattern.

    No look-ahead: purely cross-sectional -- row ``t`` depends only on row ``t``.
    """
    n = _check_n(n_quantiles)
    ranks = factor.rank(axis=1, method="average")
    counts = factor.notna().sum(axis=1).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.ceil(ranks.to_numpy(dtype=float) * n / counts[:, None])
    q = np.clip(q, 1.0, float(n))  # NaN passes through np.clip
    return pd.DataFrame(q, index=factor.index, columns=factor.columns)


def quantile_returns(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """Equal-weight mean forward return of each factor quantile per date.

    Bucket membership comes from :func:`assign_quantiles` on the factor alone;
    within a bucket the mean is taken over the members whose forward return is
    finite, so a name with a known factor but an unknown label is simply left
    out of that date's average.  A bucket with no valid member on a date is
    NaN.

    Args:
        factor: Factor panel.
        fwd_ret: Forward-return panel aligned with ``factor`` (misaligned
            inputs are inner-aligned).
        n_quantiles: Number of buckets.

    Returns:
        DataFrame indexed by date with integer columns ``1..n`` (column index
        named ``"quantile"``); values are raw ``h``-day returns.

    Implementation: vectorised through boolean masks -- one pass per bucket,
    never a loop over dates.
    """
    n = _check_n(n_quantiles)
    factor, fwd_ret = _align_pair(factor, fwd_ret)
    q = assign_quantiles(factor, n).to_numpy()
    r = fwd_ret.to_numpy(dtype=float)
    r_valid = np.isfinite(r)
    r0 = np.where(r_valid, r, 0.0)
    out = np.full((len(factor.index), n), np.nan)
    for k in range(1, n + 1):
        member = (q == k) & r_valid
        cnt = member.sum(axis=1)
        total = np.where(member, r0, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, k - 1] = np.where(cnt > 0, total / cnt, np.nan)
    cols = pd.Index(range(1, n + 1), name="quantile")
    return pd.DataFrame(out, index=factor.index, columns=cols)


def long_short_returns(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quantiles: int = 10,
) -> pd.Series:
    """Top-minus-bottom quantile spread per date (``Q_n - Q_1``).

    Args:
        factor: Factor panel.
        fwd_ret: Forward-return panel.
        n_quantiles: Number of buckets.

    Returns:
        ``pd.Series`` named ``"long_short"`` of raw ``h``-day returns, NaN on
        dates where either extreme bucket is empty.

    The long-short spread is the return of the simplest dollar-neutral
    book of the factor, before costs and risk limits.
    """
    n = _check_n(n_quantiles)
    qret = quantile_returns(factor, fwd_ret, n)
    ls = qret[n] - qret[1]
    ls.name = "long_short"
    return ls


def monotonicity(qret: pd.DataFrame) -> float:
    """Spearman correlation between the quantile index and mean quantile return.

    A value of ``1.0`` means the time-averaged return rises strictly with the
    factor bucket; ``-1.0`` means it falls strictly.  Buckets whose mean is
    NaN are ignored; NaN is returned when fewer than two buckets remain or the
    means are all identical.

    Args:
        qret: Output of :func:`quantile_returns` (columns are quantile labels).

    Returns:
        float in ``[-1, 1]`` or NaN.

    A profile that is not monotone, or a spread that comes only from
    ``Q_1``, shows a non-linear effect, a data error or a short-leg
    result.
    """
    means = qret.mean(axis=0, skipna=True)
    means = means[np.isfinite(means.to_numpy(dtype=float))]
    if len(means) < 2:
        return float("nan")
    order = np.asarray(means.index, dtype=float)
    x = np.asarray(pd.Series(order).rank().to_numpy(), dtype=float)
    y = np.asarray(means.rank().to_numpy(), dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if den <= 0.0:
        return float("nan")
    return float(np.clip((x * y).sum() / den, -1.0, 1.0))
