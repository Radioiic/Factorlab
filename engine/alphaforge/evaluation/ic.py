"""Information-coefficient (IC) analytics for cross-sectional factors.

The IC is the per-date correlation, across assets, between a factor value
known at the close of ``t`` and the forward return earned after ``t``.  Every
function here is fully vectorised over dates (no Python loop over rows) and
NaN-aware: a pair is only used when *both* the factor and the label are
finite on that date.

The IC time series gives the mean, the stability (ICIR) and the decay of the
information in a signal.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("alphaforge.evaluation.ic")

__all__ = ["rowwise_corr", "ic_series", "ic_stats", "ic_decay"]

_METHODS = ("pearson", "spearman")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _align_pair(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(a, b)`` on their common index/columns.

    Inputs sharing the store's index and columns (the normal case) pass
    through untouched; otherwise both are inner-aligned so that stray
    misalignment can never silently pair the wrong asset or date.
    """
    if a.index.equals(b.index) and a.columns.equals(b.columns):
        return a, b
    a2, b2 = a.align(b, join="inner", axis=None)
    if a2.shape != a.shape or b2.shape != b.shape:
        logger.debug("aligned panels on intersection: %s & %s -> %s", a.shape, b.shape, a2.shape)
    return a2, b2


def _rank_rows(x: np.ndarray) -> np.ndarray:
    """Average-rank each row of ``x`` ignoring NaN (NaN stays NaN)."""
    if x.shape[1] == 0:
        return x.copy()
    return stats.rankdata(x, axis=1, nan_policy="omit")


def _pearson_rows(a: np.ndarray, b: np.ndarray, valid: np.ndarray, min_periods: int) -> np.ndarray:
    """Row-wise Pearson correlation of two 2-D arrays with a joint validity mask.

    Rows with fewer than ``min_periods`` valid pairs, or where either side is
    constant over the valid pairs, return NaN (mirrors ``scipy.stats.pearsonr``,
    which refuses constant input).
    """
    n = valid.sum(axis=1)
    a0 = np.where(valid, a, 0.0)
    b0 = np.where(valid, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = a0.sum(axis=1) / n
        mean_b = b0.sum(axis=1) / n
        da = np.where(valid, a0 - mean_a[:, None], 0.0)
        db = np.where(valid, b0 - mean_b[:, None], 0.0)
        cov = (da * db).sum(axis=1)
        var_a = (da * da).sum(axis=1)
        var_b = (db * db).sum(axis=1)
        corr = cov / np.sqrt(var_a * var_b)

    # exact constant-row detection (max == min over the valid entries)
    const_a = np.where(valid, a, -np.inf).max(axis=1) == np.where(valid, a, np.inf).min(axis=1)
    const_b = np.where(valid, b, -np.inf).max(axis=1) == np.where(valid, b, np.inf).min(axis=1)
    bad = (n < min_periods) | const_a | const_b | ~(var_a > 0.0) | ~(var_b > 0.0)
    corr = np.where(bad, np.nan, corr)
    # guard against 1.0000000000000002-style rounding
    return np.clip(corr, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def rowwise_corr(
    a: pd.DataFrame,
    b: pd.DataFrame,
    method: str = "pearson",
    *,
    min_periods: int = 5,
) -> pd.Series:
    """Per-date (row-wise) correlation across assets between two panels.

    Both panels are joint-NaN masked: an asset only contributes on a date
    where *both* values are finite (``inf`` is treated as missing).  For
    ``method="spearman"`` each row is average-ranked (ties share a rank)
    after masking and the Pearson correlation of the ranks is returned,
    exactly as ``scipy.stats.spearmanr`` would compute it.

    Args:
        a: Panel (dates x assets), e.g. a factor.
        b: Panel aligned with ``a``, e.g. forward returns.  Misaligned inputs
            are inner-aligned on both axes.
        method: ``"pearson"`` or ``"spearman"``.
        min_periods: Minimum number of valid pairs on a date; rows with fewer
            return NaN (default 5).

    Returns:
        ``pd.Series`` indexed by date, NaN where the correlation is undefined
        (too few pairs or a constant row).

    The cross-sectional correlation is scale-free, so factors from prices,
    volumes and other fields compare on one axis.

    No look-ahead: row ``t`` of the output depends only on row ``t`` of the inputs.
    """
    method = str(method).lower()
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if min_periods < 2:
        raise ValueError("min_periods must be >= 2")
    a, b = _align_pair(a, b)
    av = a.to_numpy(dtype=float, copy=True)
    bv = b.to_numpy(dtype=float, copy=True)
    valid = np.isfinite(av) & np.isfinite(bv)
    av[~valid] = np.nan
    bv[~valid] = np.nan
    if method == "spearman":
        av = _rank_rows(av)
        bv = _rank_rows(bv)
    corr = _pearson_rows(av, bv, valid, min_periods)
    return pd.Series(corr, index=a.index, name=f"rowwise_{method}")


def ic_series(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """Information coefficient per date: corr across assets of factor vs forward return.

    Args:
        factor: Factor panel computed with data known at the close of ``t``.
        fwd_ret: Forward-return panel (label) aligned with ``factor``; see
            ``DataStore.fwd_return`` for the entry-lag convention.
        method: ``"spearman"`` (rank IC, the default -- robust to outliers) or
            ``"pearson"`` (classic IC).

    Returns:
        ``pd.Series`` named ``"ic"`` on the factor's full date index; dates
        where the IC is undefined (e.g. the last ``lag + horizon`` days, whose
        labels are not known) are NaN rather than dropped, so the series
        stays aligned with the panel.

    The rank IC is the screening statistic, because a small number of
    extreme returns cannot control it.
    """
    s = rowwise_corr(factor, fwd_ret, method=method)
    s.name = "ic"
    return s


def ic_stats(ic: pd.Series, periods_per_year: int = 252) -> dict[str, Any]:
    """Summary statistics of an IC series.

    Args:
        ic: Per-date IC series (NaN entries are ignored).
        periods_per_year: Number of IC observations per year used to annualise
            the information ratio (252 for daily ICs).

    Returns:
        dict with keys ``mean``, ``std`` (ddof=1), ``ir`` (mean/std),
        ``ir_ann`` (ir * sqrt(periods_per_year)), ``t_stat`` (mean/std*sqrt(n)),
        ``p_value`` (two-sided, Student-t with n-1 dof, NaN when n < 3),
        ``positive_ratio`` (share of ICs > 0), ``n`` (valid observations) and
        ``abs_mean``.  All float values are NaN when undefined.

    The ICIR and its t-statistic measure the stability of the IC, not
    only its size.
    """
    x = pd.Series(ic).to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    nan = float("nan")
    mean = float(x.mean()) if n else nan
    std = float(x.std(ddof=1)) if n >= 2 else nan
    ir = mean / std if (n >= 2 and std > 0.0) else nan
    ir_ann = ir * math.sqrt(periods_per_year) if math.isfinite(ir) else nan
    t_stat = ir * math.sqrt(n) if math.isfinite(ir) else nan
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1)) if n >= 3 and math.isfinite(t_stat) else nan
    return {
        "mean": mean,
        "std": std,
        "ir": ir,
        "ir_ann": ir_ann,
        "t_stat": t_stat,
        "p_value": p_value,
        "positive_ratio": float((x > 0).mean()) if n else nan,
        "n": n,
        "abs_mean": float(np.abs(x).mean()) if n else nan,
    }


def ic_decay(
    factor: pd.DataFrame,
    store: Any,
    horizons: Iterable[int] = (1, 2, 3, 5, 10, 20),
    lag: int = 1,
    method: str = "spearman",
) -> pd.Series:
    """Mean IC of a factor against forward returns of increasing horizon.

    Args:
        factor: Factor panel (may be a date sub-range of the store; it is
            inner-aligned with each label panel).
        store: Any object exposing ``fwd_return(horizon, lag) -> panel``
            (a ``DataStore``).
        horizons: Holding periods in days.
        lag: Entry lag passed to ``store.fwd_return``.
        method: Correlation method, see :func:`rowwise_corr`.

    Returns:
        ``pd.Series`` named ``"ic"`` indexed by horizon (index name
        ``"horizon"``).

    The decay of the IC over the horizons shows the holding period and the
    turnover that a signal needs.
    """
    hs = [int(h) for h in horizons]
    if any(h < 1 for h in hs):
        raise ValueError("horizons must be >= 1")
    values = [float(ic_series(factor, store.fwd_return(h, lag), method=method).mean()) for h in hs]
    out = pd.Series(values, index=pd.Index(hs, name="horizon"), name="ic", dtype=float)
    logger.debug("ic_decay(lag=%d, %s): %s", lag, method, out.round(4).to_dict())
    return out
