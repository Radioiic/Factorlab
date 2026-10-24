"""Vectorised operators used by factor expressions.

Every operator works on *panels*: ``pandas.DataFrame`` objects indexed by
timestamp (rows) with one column per symbol.  Scalars are accepted wherever
it makes sense so expressions such as ``close / delay(close, 1) - 1`` work.

Conventions
-----------
* Time-series operators (``ts_*``, ``delay``, ``delta`` ...) look backwards
  only -- no operator ever uses future information.
* Rolling windows require a full window of valid observations
  (``min_periods == window``) so values are never computed from a partial
  history right after a coin is listed.
* Divisions by zero and logs of non-positive numbers yield ``NaN`` instead of
  ``inf``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

Panel = pd.DataFrame

OPERATORS: dict[str, Callable] = {}
OPERATOR_DOCS: dict[str, tuple[str, str]] = {}


def operator(kind: str, doc: str):
    """Register an operator under its function name."""

    def deco(fn):
        OPERATORS[fn.__name__] = fn
        OPERATOR_DOCS[fn.__name__] = (kind, doc)
        return fn

    return deco


def _clean(x):
    """Return x with infinite values set to NaN (a panel), or x unchanged (a number)."""
    if isinstance(x, pd.DataFrame):
        return x.replace([np.inf, -np.inf], np.nan)
    return x


def _w(window) -> int:
    """Return the window as an int; a window below 1 gives ValueError."""
    w = int(window)
    if w < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return w


# --------------------------------------------------------------------------
# element-wise
# --------------------------------------------------------------------------
@operator("elementwise", "Safe division; x / 0 -> NaN")
def div(x, y):
    """Return x / y; a zero divisor gives NaN and an infinite result gives NaN. x, y: panels or numbers."""
    if isinstance(y, pd.DataFrame):
        y = y.where(y != 0)
    elif y == 0:
        return x * np.nan
    return _clean(x / y)


@operator("elementwise", "Natural log, NaN for x <= 0")
def log(x):
    """Return the natural log of x; x <= 0 gives NaN. x: panel or number."""
    if isinstance(x, pd.DataFrame):
        return np.log(x.where(x > 0))
    return math.log(x) if x > 0 else np.nan


@operator("elementwise", "Exponential")
def exp(x):
    """Return e ** x for each value. x: panel or number."""
    return np.exp(x)


@operator("elementwise", "Signed log: sign(x) * log(1 + |x|)")
def slog(x):
    """Return sign(x) * log(1 + |x|) for each value. x: panel or number."""
    return np.sign(x) * np.log1p(np.abs(x))


@operator("elementwise", "Signed square root: sign(x) * sqrt(|x|)")
def ssqrt(x):
    """Return sign(x) * sqrt(|x|) for each value. x: panel or number."""
    return np.sign(x) * np.sqrt(np.abs(x))


@operator("elementwise", "Absolute value")
def abs_(x):
    """Return |x| for each value (the expression name is ``abs``). x: panel or number."""
    return np.abs(x)


OPERATORS["abs"] = abs_
OPERATOR_DOCS["abs"] = OPERATOR_DOCS.pop("abs_")
del OPERATORS["abs_"]


@operator("elementwise", "Sign (-1, 0, 1)")
def sign(x):
    """Return the sign of each value: -1, 0 or 1. x: panel or number."""
    return np.sign(x)


@operator("elementwise", "Square")
def square(x):
    """Return x * x for each value. x: panel or number."""
    return x * x


@operator("elementwise", "Element-wise maximum of two inputs")
def emax(x, y):
    """Return the larger of x and y for each cell; NaN in a panel input gives NaN. x, y: panels or numbers."""
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        return x.where(x >= y, y).where(x.notna() & y.notna())
    if isinstance(x, pd.DataFrame):
        return x.clip(lower=y)
    if isinstance(y, pd.DataFrame):
        return y.clip(lower=x)
    return max(x, y)


@operator("elementwise", "Element-wise minimum of two inputs")
def emin(x, y):
    """Return the smaller of x and y for each cell; NaN in a panel input gives NaN. x, y: panels or numbers."""
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        return x.where(x <= y, y).where(x.notna() & y.notna())
    if isinstance(x, pd.DataFrame):
        return x.clip(upper=y)
    if isinstance(y, pd.DataFrame):
        return y.clip(upper=x)
    return min(x, y)


@operator("elementwise", "1.0 where x > y else 0.0 (NaN propagates)")
def gt(x, y):
    """Return 1.0 where x > y, else 0.0; NaN in a panel input gives NaN. x, y: panels or numbers."""
    out = (x > y).astype(float)
    nan = pd.isna(x) if isinstance(x, pd.DataFrame) else None
    if isinstance(y, pd.DataFrame):
        nan = pd.isna(y) if nan is None else (nan | pd.isna(y))
    return out.mask(nan) if nan is not None else out


@operator("elementwise", "1.0 where x < y else 0.0 (NaN propagates)")
def lt(x, y):
    """Return 1.0 where x < y, else 0.0; NaN in a panel input gives NaN. x, y: panels or numbers."""
    return gt(y, x)


@operator("elementwise", "Clip to [lo, hi]")
def clip(x, lo, hi):
    """Return x limited to the interval [lo, hi]. x: panel; lo, hi: numbers."""
    return x.clip(lower=lo, upper=hi)


# --------------------------------------------------------------------------
# time-series
# --------------------------------------------------------------------------
@operator("time_series", "Value d bars ago")
def delay(x, d):
    """Return the value of x d bars before each bar. x: panel; d: number of bars."""
    return x.shift(int(d))


@operator("time_series", "x - delay(x, d)")
def delta(x, d):
    """Return x minus its value d bars before. x: panel; d: number of bars."""
    return x - x.shift(int(d))


@operator("time_series", "x / delay(x, d) - 1")
def pct_change(x, d):
    """Return x / delay(x, d) - 1; a zero earlier value gives NaN. x: panel; d: number of bars."""
    return div(x, x.shift(int(d))) - 1.0


@operator("time_series", "Rolling sum")
def ts_sum(x, w):
    """Return the sum of the last w bars of x (NaN until w valid values). x: panel; w: window in bars."""
    w = _w(w)
    return x.rolling(w, min_periods=w).sum()


@operator("time_series", "Rolling mean")
def ts_mean(x, w):
    """Return the mean of the last w bars of x (NaN until w valid values). x: panel; w: window in bars."""
    w = _w(w)
    return x.rolling(w, min_periods=w).mean()


@operator("time_series", "Rolling standard deviation")
def ts_std(x, w):
    """Return the sample standard deviation of the last w bars of x. x: panel; w: window in bars."""
    w = _w(w)
    return x.rolling(w, min_periods=w).std()


@operator("time_series", "Rolling median")
def ts_median(x, w):
    """Return the median of the last w bars of x. x: panel; w: window in bars."""
    w = _w(w)
    return x.rolling(w, min_periods=w).median()


@operator("time_series", "Rolling maximum")
def ts_max(x, w):
    """Return the maximum of the last w bars of x. x: panel; w: window in bars."""
    w = _w(w)
    return x.rolling(w, min_periods=w).max()


@operator("time_series", "Rolling minimum")
def ts_min(x, w):
    """Return the minimum of the last w bars of x. x: panel; w: window in bars."""
    w = _w(w)
    return x.rolling(w, min_periods=w).min()


@operator("time_series", "Rolling skewness")
def ts_skew(x, w):
    """Return the skewness of the last w bars of x; an infinite value gives NaN. x: panel; w: window in bars."""
    w = _w(w)
    return _clean(x.rolling(w, min_periods=w).skew())


@operator("time_series", "Rolling excess kurtosis")
def ts_kurt(x, w):
    """Return the excess kurtosis of the last w bars of x; an infinite value gives NaN. x: panel; w: window in bars."""
    w = _w(w)
    return _clean(x.rolling(w, min_periods=w).kurt())


@operator("time_series", "(x - rolling mean) / rolling std")
def ts_zscore(x, w):
    """Return (x - rolling mean) / rolling standard deviation over w bars. x: panel; w: window in bars."""
    return div(x - ts_mean(x, w), ts_std(x, w))


@operator("time_series", "Position of x inside its rolling [min, max] range, 0..1")
def ts_minmax(x, w):
    """Return the position of x in its rolling [min, max] range over w bars, 0..1. x: panel; w: window in bars."""
    lo, hi = ts_min(x, w), ts_max(x, w)
    return div(x - lo, hi - lo)


@operator("time_series", "Exponential moving average with span w")
def ema(x, w):
    """Return the exponential moving average of x with span w (NaN for the first w - 1 bars). x: panel; w: span in bars."""
    w = _w(w)
    return x.ewm(span=w, min_periods=w, adjust=False).mean()


@operator("time_series", "Linearly decaying weighted mean (newest weight w)")
def decay_linear(x, w):
    """Return the weighted mean of the last w bars with weights 1..w (the newest bar has weight w). x: panel; w: window."""
    w = _w(w)
    weights = np.arange(1, w + 1, dtype=float)
    weights /= weights.sum()
    return _rolling_np(x, w, lambda win: win @ weights)


@operator("time_series", "Percentile rank of current value within the window, 0..1")
def ts_rank(x, w):
    """Return the percentile rank of the current value of x within its last w bars, 0..1. x: panel; w: window."""
    w = _w(w)
    if w == 1:
        return x * 0 + 0.5

    def f(win):
        last = win[:, -1:]
        less = (win < last).sum(axis=1)
        equal = (win == last).sum(axis=1)
        return (less + 0.5 * (equal - 1)) / (w - 1)

    return _rolling_np(x, w, f)


@operator("time_series", "Bars since the rolling maximum, scaled to 0..1")
def ts_argmax(x, w):
    """Return the bars since the maximum of the last w bars, divided by w - 1 (0..1). x: panel; w: window."""
    w = _w(w)
    return _rolling_np(x, w, lambda win: (w - 1 - np.argmax(win, axis=1)) / max(w - 1, 1))


@operator("time_series", "Bars since the rolling minimum, scaled to 0..1")
def ts_argmin(x, w):
    """Return the bars since the minimum of the last w bars, divided by w - 1 (0..1). x: panel; w: window."""
    w = _w(w)
    return _rolling_np(x, w, lambda win: (w - 1 - np.argmin(win, axis=1)) / max(w - 1, 1))


@operator("time_series", "OLS slope of x against time over the window")
def ts_slope(x, w):
    """Return the OLS slope of x against time over the last w bars. x: panel; w: window in bars."""
    w = _w(w)
    t = np.arange(w, dtype=float)
    t -= t.mean()
    denom = (t * t).sum()
    return _rolling_np(x, w, lambda win: (win @ t) / denom)


@operator("time_series", "R-squared of x regressed on time (trend strength)")
def ts_rsq(x, w):
    """Return the R-squared of x regressed on time over the last w bars. x: panel; w: window in bars."""
    w = _w(w)
    t = np.arange(w, dtype=float)
    t -= t.mean()
    tt = (t * t).sum()

    def f(win):
        xc = win - win.mean(axis=1, keepdims=True)
        sxx = (xc * xc).sum(axis=1)
        sxt = xc @ t
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sxx > 0, sxt * sxt / (sxx * tt), np.nan)

    return _rolling_np(x, w, f)


@operator("time_series", "Rolling Pearson correlation of x and y")
def ts_corr(x, y, w):
    """Return the rolling Pearson correlation of x and y over w bars, limited to [-1, 1]. x, y: panels; w: window."""
    w = _w(w)
    return _clean(x.rolling(w, min_periods=w).corr(y)).clip(-1.0, 1.0)


@operator("time_series", "Rolling covariance of x and y")
def ts_cov(x, y, w):
    """Return the rolling covariance of x and y over w bars. x, y: panels; w: window in bars."""
    w = _w(w)
    return x.rolling(w, min_periods=w).cov(y)


@operator("time_series", "Rolling OLS beta of y on x")
def ts_beta(y, x, w):
    """Return the rolling OLS beta of y on x over w bars. y, x: panels (x must be a panel); w: window in bars."""
    if not isinstance(x, pd.DataFrame):
        raise TypeError("ts_beta regressor must be a panel")
    xy = ts_mean(x * y, w) - ts_mean(x, w) * ts_mean(y, w)
    xx = ts_mean(x * x, w) - ts_mean(x, w) ** 2
    return div(xy, xx)


@operator("time_series", "Current residual of the rolling OLS of y on x")
def ts_resid(y, x, w):
    """Return the current residual of the rolling OLS of y on x over w bars. y, x: panels; w: window in bars."""
    b = ts_beta(y, x, w)
    a = ts_mean(y, w) - b * ts_mean(x, w)
    return y - a - b * x


@operator("time_series", "Fraction of the window where x > 0")
def ts_pos_frac(x, w):
    """Return the fraction of the last w bars with x > 0. x: panel; w: window in bars."""
    return ts_mean(gt(x, 0.0), w)


@operator("time_series", "Rolling autocorrelation of x at lag `lag`")
def ts_autocorr(x, w, lag=1):
    """Return the rolling correlation of x and x lagged by ``lag`` bars, over w bars. x: panel; w, lag: bars."""
    return ts_corr(x, x.shift(int(lag)), w)


@operator("time_series", "Maximum drawdown of price x inside the window (<= 0)")
def ts_maxdd(x, w):
    """Return the maximum drawdown of the price x within the last w bars (<= 0). x: panel; w: window in bars."""
    w = _w(w)

    def f(win):
        peak = np.maximum.accumulate(win, axis=1)
        return (win / peak - 1.0).min(axis=1)

    return _rolling_np(x, w, f)


@operator("time_series", "Maximum run-up of price x inside the window (>= 0)")
def ts_maxru(x, w):
    """Return the maximum run-up of the price x within the last w bars (>= 0). x: panel; w: window in bars."""
    w = _w(w)

    def f(win):
        trough = np.minimum.accumulate(win, axis=1)
        return (win / trough - 1.0).max(axis=1)

    return _rolling_np(x, w, f)


def _rolling_np(x: Panel, w: int, func) -> Panel:
    """Apply ``func`` to every full window; ``func`` maps (n, w) -> (n,)."""
    values = x.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    if values.shape[0] < w:
        return pd.DataFrame(out, index=x.index, columns=x.columns)
    for j in range(values.shape[1]):
        col = values[:, j]
        windows = sliding_window_view(col, w)
        valid = ~np.isnan(windows).any(axis=1)
        if not valid.any():
            continue
        res = np.full(windows.shape[0], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            res[valid] = func(windows[valid])
        out[w - 1 :, j] = res
    return _clean(pd.DataFrame(out, index=x.index, columns=x.columns))


# --------------------------------------------------------------------------
# cross-sectional
# --------------------------------------------------------------------------
@operator("cross_section", "Cross-sectional percentile rank per timestamp, 0..1")
def cs_rank(x):
    """Return the cross-sectional percentile rank of x at each timestamp, 0..1. x: panel."""
    return x.rank(axis=1, pct=True)


@operator("cross_section", "Cross-sectional z-score per timestamp")
def cs_zscore(x):
    """Return the cross-sectional z-score of x at each timestamp; zero dispersion gives NaN. x: panel."""
    sd = x.std(axis=1)
    return _clean(x.sub(x.mean(axis=1), axis=0).div(sd.where(sd > 0), axis=0))


@operator("cross_section", "Subtract the cross-sectional mean")
def cs_demean(x):
    """Return x minus its cross-sectional mean at each timestamp. x: panel."""
    return x.sub(x.mean(axis=1), axis=0)


@operator("cross_section", "Equal-weight cross-sectional mean broadcast to all symbols")
def cs_mean(x):
    """Return the cross-sectional mean of x at each timestamp, in each cell where x has a value. x: panel."""
    m = x.mean(axis=1)
    return pd.DataFrame(np.repeat(m.to_numpy()[:, None], x.shape[1], axis=1), index=x.index, columns=x.columns).where(
        x.notna()
    )


@operator("cross_section", "Weighted cross-sectional mean (weights w) broadcast to all symbols")
def cs_wmean(x, wts):
    """Return the cross-sectional mean of x with weights wts at each timestamp, where x has a value. x, wts: panels."""
    wts = wts.where(x.notna())
    m = (x * wts).sum(axis=1) / wts.sum(axis=1).replace(0, np.nan)
    return pd.DataFrame(np.repeat(m.to_numpy()[:, None], x.shape[1], axis=1), index=x.index, columns=x.columns).where(
        x.notna()
    )
