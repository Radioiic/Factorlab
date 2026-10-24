"""Time-series (trailing-window) operators.

Conventions shared by every ``ts_*`` operator:

* **Strictly causal.**  Row ``t`` of the output depends only on rows ``<= t`` of
  the inputs (trailing windows ``[t-d+1, t]``).  The tests perturb rows ``> t``
  and assert row ``t`` is unchanged for every operator in this module.
* **``min_periods = window``.**  A windowed statistic is ``NaN`` until ``d``
  observations are available *and* whenever any of the ``d`` values in the
  window is ``NaN`` - identical to ``pandas`` rolling with ``min_periods=d``.
  (No "partial" statistic comes from fewer points: a 5-day mean of 2 points
  is a different, noisier feature.)
* Output is a *new* ``float64`` panel with the caller's index/columns; ``±inf``
  is mapped to ``NaN``.
* Windows are ``int >= 1``; integral floats (``5.0``, as produced by the
  expression engine's ``Const``) are accepted.

Performance: every operator runs in well under 0.1 s on a 1000 x 300 panel with
``d = 60`` (target 0.5 s).  Nothing here uses ``rolling.apply`` or a
JIT-compiled kernel, so there is no warm-up cost and timings are predictable.

All windowing conventions are in this one module, so each mined expression
uses the same causal rules.
"""

from __future__ import annotations

import logging
import numbers
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from alphaforge.ops.elementwise import _aligned, _check_panel, _finite, _values, _wrap

logger = logging.getLogger("alphaforge.ops.ts")

__all__ = [
    "ts_delay",
    "ts_delta",
    "ts_pct_change",
    "ts_mean",
    "ts_sum",
    "ts_std",
    "ts_min",
    "ts_max",
    "ts_argmax",
    "ts_argmin",
    "ts_rank",
    "ts_zscore",
    "ts_corr",
    "ts_cov",
    "ts_decay_linear",
    "ts_skew",
    "ts_kurt",
    "ts_product",
]

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _window(d: object, name: str = "d") -> int:
    """Validate a window length and return it as a Python ``int`` (``>= 1``)."""
    if isinstance(d, bool):
        raise TypeError(f"{name} must be an integer window, got bool")
    if (
        isinstance(d, (numbers.Integral, np.integer))
        or isinstance(d, (numbers.Real, np.floating))
        and float(d).is_integer()
    ):
        w = int(d)
    else:
        raise TypeError(f"{name} must be an integer window, got {d!r}")
    if w < 1:
        raise ValueError(f"{name} must be >= 1, got {w}")
    return w


def _rolling(x: pd.DataFrame, d: int) -> pd.core.window.rolling.Rolling:
    """Trailing window of length ``d`` with ``min_periods=d``."""
    return x.rolling(window=d, min_periods=d)


def _rolling_higher_moment(x: pd.DataFrame, d: int, which: str) -> pd.DataFrame:
    """``rolling.skew()`` / ``rolling.kurt()`` with pandas' internal centring
    warning silenced: on an all-``NaN`` column pandas computes ``nanmean`` of an
    empty slice and emits ``RuntimeWarning: All-NaN slice encountered`` even
    though the (correct) result is simply ``NaN``.  Only that message is muted."""
    roll = _rolling(x, d)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return roll.skew() if which == "skew" else roll.kurt()


def _from_pandas(res: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """Normalise a pandas result: ``like``'s labels, ``float64``, no ``inf``."""
    if not (res.index.equals(like.index) and res.columns.equals(like.columns)):
        res = res.reindex(index=like.index, columns=like.columns)
    return _wrap(_finite(res.to_numpy(dtype=np.float64, na_value=np.nan)), like)


def _window_has_nan(a: np.ndarray, d: int) -> np.ndarray:
    """Boolean mask: ``True`` where the trailing ``d``-window ending at ``t`` is
    incomplete (``t < d-1``) or contains a ``NaN``.  O(n*m) via a cumulative count."""
    n = a.shape[0]
    mask = np.ones(a.shape, dtype=bool)
    if n < d:
        return mask
    c = np.cumsum(np.isnan(a), axis=0)
    cnt = c[d - 1 :].copy()
    cnt[1:] -= c[:-d]
    mask[d - 1 :] = cnt > 0
    return mask


def _shift_sum(a: np.ndarray, d: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted trailing-window sum via ``d`` shifted adds (``weights[k]`` applies to
    lag ``k``, ``k = 0`` = today).  ``NaN`` propagates naturally, so the result is
    ``NaN`` exactly where the window is incomplete or contains a ``NaN``."""
    n, m = a.shape
    out = np.full((n, m), np.nan)
    if n < d:
        return out
    acc = np.zeros((n - d + 1, m))
    for k in range(d):
        block = a[d - 1 - k : n - k]
        acc += block if weights is None else weights[k] * block
    out[d - 1 :] = acc
    return out


# ----------------------------------------------------------------------------
# lags and differences
# ----------------------------------------------------------------------------


def ts_delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Value ``d`` rows ago: ``x.shift(d)``.  First ``d`` rows are ``NaN``.

    The lag is an explicit operator, so the expression string shows the as-of
    rule.
    """
    d = _window(d)
    return _from_pandas(_check_panel(x).shift(d), x)


def ts_delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Change over ``d`` rows: ``x - x.shift(d)``."""
    d = _window(d)
    _check_panel(x)
    return _from_pandas(x - x.shift(d), x)


def ts_pct_change(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Relative change over ``d`` rows: ``x / x.shift(d) - 1`` (``0`` base -> ``NaN``)."""
    d = _window(d)
    a = _values(x)
    n = a.shape[0]
    out = np.full(a.shape, np.nan)
    if n > d:
        with np.errstate(all="ignore"):
            out[d:] = a[d:] / a[:-d] - 1.0
    return _wrap(_finite(out), x)


# ----------------------------------------------------------------------------
# rolling moments and extremes (pandas Cython kernels)
# ----------------------------------------------------------------------------


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Trailing ``d``-row mean (``min_periods=d``)."""
    d = _window(d)
    return _from_pandas(_rolling(_check_panel(x), d).mean(), x)


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Trailing ``d``-row sum (``min_periods=d``)."""
    d = _window(d)
    return _from_pandas(_rolling(_check_panel(x), d).sum(), x)


def ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Trailing ``d``-row sample standard deviation (``ddof=1``; ``NaN`` for ``d=1``).

    Realised volatility is a factor (the low-volatility effect) and also a
    scale for other signals.
    """
    d = _window(d)
    return _from_pandas(_rolling(_check_panel(x), d).std(ddof=1), x)


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Trailing ``d``-row minimum (``min_periods=d``)."""
    d = _window(d)
    return _from_pandas(_rolling(_check_panel(x), d).min(), x)


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Trailing ``d``-row maximum (``min_periods=d``)."""
    d = _window(d)
    return _from_pandas(_rolling(_check_panel(x), d).max(), x)


# ----------------------------------------------------------------------------
# rolling extremes location / rank (numpy sliding windows)
# ----------------------------------------------------------------------------


def _ts_argext(x: pd.DataFrame, d: int, *, largest: bool) -> pd.DataFrame:
    """Shared kernel for :func:`ts_argmax` / :func:`ts_argmin`."""
    d = _window(d)
    a = _values(x)
    n, m = a.shape
    out = np.full((n, m), np.nan)
    if n < d:
        return _wrap(out, x)
    fill = -np.inf if largest else np.inf
    filled = np.where(np.isnan(a), fill, a)
    win = sliding_window_view(filled, d, axis=0)  # (n-d+1, m, d), oldest -> newest
    pos = np.argmax(win, axis=-1) if largest else np.argmin(win, axis=-1)
    out[d - 1 :] = (d - 1) - pos
    out[_window_has_nan(a, d)] = np.nan
    return _wrap(out, x)


def ts_argmax(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Days since the trailing ``d``-window maximum (``0`` = the max is today).

    Ties resolve to the *earliest* occurrence (the larger "days since"), the same
    convention as ``d - 1 - np.argmax(window)``.

    The distance in days to the window high is scale-free, but the level of the
    high is not.
    """
    return _ts_argext(x, d, largest=True)


def ts_argmin(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Days since the trailing ``d``-window minimum (``0`` = the min is today).

    Ties resolve to the earliest occurrence (see :func:`ts_argmax`).
    """
    return _ts_argext(x, d, largest=False)


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Percentile rank of today's value within its trailing ``d``-window, in ``(0, 1]``.

    Uses the average-rank convention (``scipy.stats.rankdata`` / pandas
    ``rank(pct=True)``): ``rank = (1 + #{x_{t-k} < x_t} + 0.5 * #{x_{t-k} == x_t}) / d``
    over ``k = 1..d-1``.  The window minimum therefore scores ``1/d``, the maximum
    ``1.0`` and ``d = 1`` gives ``1.0``.  ``NaN`` if any of the ``d`` values is ``NaN``.

    Implemented with ``d-1`` vectorised shift-comparisons (no ``rolling.apply``),
    O(n*m*d) but with tiny constants: ~20 ms on 1000 x 300, ``d = 60``.

    The time-series rank gives the place of today's value in its recent range: a
    bounded value that compares across names and regimes.
    """
    d = _window(d)
    a = _values(x)
    n, m = a.shape
    gt = np.zeros((n, m))
    eq = np.zeros((n, m))
    with np.errstate(invalid="ignore"):
        for k in range(1, min(d, n)):
            gt[k:] += a[k:] > a[:-k]
            eq[k:] += a[k:] == a[:-k]
    out = (1.0 + gt + 0.5 * eq) / d
    out[_window_has_nan(a, d)] = np.nan
    return _wrap(out, x)


def ts_zscore(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Trailing z-score ``(x - ts_mean(x, d)) / ts_std(x, d)``; zero std -> ``NaN``.

    The result is today's value in units of its own recent volatility, so a
    2-sigma move has the same meaning for each name.
    """
    d = _window(d)
    _check_panel(x)
    a = _values(x)
    mean = _rolling(x, d).mean().to_numpy(dtype=np.float64)
    std = _rolling(x, d).std(ddof=1).to_numpy(dtype=np.float64)
    with np.errstate(all="ignore"):
        out = (a - mean) / std
        out = np.where(std == 0.0, np.nan, out)
    return _wrap(_finite(out), x)


# ----------------------------------------------------------------------------
# rolling bivariate statistics
# ----------------------------------------------------------------------------


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """Trailing ``d``-window Pearson correlation of ``x`` and ``y``, column by column.

    Uses ``DataFrame.rolling(d, min_periods=d).corr(other)`` (``pairwise=False``),
    so the window is ``NaN`` if *either* series has a ``NaN`` in it.  Zero
    variance in either series -> ``NaN``; results are clipped to ``[-1, 1]`` to
    remove floating-point overshoot.

    Rolling correlations of price and volume are common factors, for example
    ``-ts_corr(cs_rank(open), cs_rank(volume), 10)``.
    """
    d = _window(d)
    _check_panel(x)
    y = _aligned(y, x, "y")
    with np.errstate(all="ignore"):
        res = _rolling(x, d).corr(y)
    if not (res.index.equals(x.index) and res.columns.equals(x.columns)):
        res = res.reindex(index=x.index, columns=x.columns)  # pandas unions (sorts) columns
    arr = _finite(res.to_numpy(dtype=np.float64, na_value=np.nan))
    return _wrap(np.clip(arr, -1.0, 1.0), x)  # np.clip keeps NaN; returns a new array


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """Trailing ``d``-window sample covariance (``ddof=1``) of ``x`` and ``y``, column by column.

    ``NaN`` if either series has a ``NaN`` in the window, or ``d = 1``.
    """
    d = _window(d)
    _check_panel(x)
    y = _aligned(y, x, "y")
    with np.errstate(all="ignore"):
        res = _rolling(x, d).cov(y, ddof=1)
    return _from_pandas(res, x)


# ----------------------------------------------------------------------------
# weighted / higher-moment / multiplicative windows
# ----------------------------------------------------------------------------


def ts_decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Linearly-decayed trailing mean: weights ``d, d-1, ..., 1`` (newest = ``d``),
    normalised to sum to one.  ``NaN`` if any value in the window is ``NaN``.

    The linear decay gives recent values more weight and has a finite memory,
    so the start date of the history does not change the result.
    """
    d = _window(d)
    a = _values(x)
    w = np.arange(d, 0, -1, dtype=np.float64)  # w[k] for lag k: d, d-1, ..., 1
    w /= w.sum()
    return _wrap(_finite(_shift_sum(a, d, w)), x)


def ts_skew(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Trailing ``d``-window sample skewness (pandas, bias-corrected; ``NaN`` for ``d < 3``
    or a zero-variance window)."""
    d = _window(d)
    return _from_pandas(_rolling_higher_moment(_check_panel(x), d, "skew"), x)


def ts_kurt(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Trailing ``d``-window *excess* kurtosis (pandas, Fisher, bias-corrected;
    ``NaN`` for ``d < 4`` or a zero-variance window).

    High kurtosis of recent returns marks names with a recent large move; it
    is also a simple crash-risk measure.
    """
    d = _window(d)
    return _from_pandas(_rolling_higher_moment(_check_panel(x), d, "kurt"), x)


def ts_product(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Trailing ``d``-window product (e.g. compounding ``1 + returns``).

    Computed by ``d`` shifted multiplications (no ``rolling.apply``); ``NaN`` if
    any value in the window is ``NaN``; overflow (``inf``) -> ``NaN``.
    """
    d = _window(d)
    a = _values(x)
    n, m = a.shape
    out = np.full((n, m), np.nan)
    if n >= d:
        acc = np.ones((n - d + 1, m))
        with np.errstate(all="ignore"):
            for k in range(d):
                acc *= a[d - 1 - k : n - k]
        out[d - 1 :] = acc
    return _wrap(_finite(out), x)
