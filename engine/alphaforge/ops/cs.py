"""Cross-sectional (row-wise, across assets) and group operators.

Conventions shared by every ``cs_*`` / ``group_*`` operator:

* Each *date* (row) is processed independently, so these operators can never
  look ahead - there is no time dimension inside them at all.
* ``NaN`` in -> ``NaN`` out; an all-``NaN`` row stays all-``NaN``.  The output
  ``NaN`` pattern equals the input pattern except in the documented degenerate
  cases (zero dispersion for a z-score, an all-zero row for :func:`cs_scale`, a
  ``NaN`` group id for the ``group_*`` ops, and the joint mask of the two inputs
  for :func:`cs_regress_out` / :func:`cs_neutralize`).
* Output is a *new* ``float64`` panel with the caller's index/columns.

Everything is vectorised over the whole panel: row statistics are numpy
reductions, and the group operators key every cell by ``(row, group)`` and use
``np.bincount`` / a single ``groupby.rank`` - there is no per-date Python loop.

Cross-sectional operators turn a raw signal into a relative score for each date,
and neutralisation removes industry and size exposure.
"""

from __future__ import annotations

import logging
import numbers

import numpy as np
import pandas as pd

from alphaforge.ops.elementwise import _aligned, _check_panel, _finite, _values, _wrap

logger = logging.getLogger("alphaforge.ops.cs")

__all__ = [
    "cs_rank",
    "cs_zscore",
    "cs_demean",
    "cs_scale",
    "cs_winsorize",
    "cs_quantile_clip",
    "group_neutralize",
    "group_rank",
    "group_zscore",
    "cs_regress_out",
    "cs_neutralize",
]

#: Consistency constant turning a median absolute deviation into a Gaussian sigma.
MAD_TO_SIGMA: float = 1.4826

#: Relative tolerance below which a within-group dispersion is treated as zero
#: (guards against ``std ~ 1e-17`` artefacts on numerically-constant groups).
_REL_EPS: float = 1e-12

# ----------------------------------------------------------------------------
# row-wise helpers (no per-row Python loops, no numpy warnings)
# ----------------------------------------------------------------------------


def _row_count_mean(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row count of valid cells and their mean (``NaN`` for an empty row)."""
    valid = ~np.isnan(a)
    cnt = valid.sum(axis=1)
    s = np.where(valid, a, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / cnt
    return cnt, mean


def _row_std(a: np.ndarray, mean: np.ndarray, cnt: np.ndarray, ddof: int = 0) -> np.ndarray:
    """Per-row standard deviation about ``mean`` (two-pass); exactly ``0.0`` for a
    numerically-constant row (``max == min``), ``NaN`` when ``cnt <= ddof``."""
    if a.shape[1] == 0:
        return np.full(a.shape[0], np.nan)
    dev = np.where(np.isnan(a), 0.0, a - mean[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (dev * dev).sum(axis=1) / (cnt - ddof)
        std = np.sqrt(var)
    constant = np.fmax.reduce(a, axis=1) == np.fmin.reduce(a, axis=1)  # False for all-NaN rows
    std = np.where(constant, 0.0, std)
    std[cnt <= ddof] = np.nan
    return std


def _row_quantile(a: np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    """Per-row ``NaN``-ignoring quantiles with linear interpolation (matches
    ``np.nanquantile(a, qs, axis=1)``) - fully vectorised via one sort.

    Returns:
        Array of shape ``(len(qs), n_rows)``; ``NaN`` for all-``NaN`` rows.
    """
    n = a.shape[0]
    if a.shape[1] == 0:
        return np.full((len(qs), n), np.nan)
    s = np.sort(a, axis=1)  # NaN sorts last
    cnt = (~np.isnan(a)).sum(axis=1)
    last = np.maximum(cnt - 1, 0)
    out = np.empty((len(qs), n))
    for i, q in enumerate(qs):
        pos = q * (cnt - 1)
        lo = np.floor(pos).astype(np.int64)
        frac = pos - lo
        lo_i = np.clip(lo, 0, last)
        hi_i = np.clip(lo + 1, 0, last)
        v_lo = np.take_along_axis(s, lo_i[:, None], axis=1)[:, 0]
        v_hi = np.take_along_axis(s, hi_i[:, None], axis=1)[:, 0]
        v = v_lo + (v_hi - v_lo) * frac
        v[cnt == 0] = np.nan
        out[i] = v
    return out


def _group_keys(a: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Key every jointly-valid cell by ``row * n_groups + group_code``.

    Returns:
        ``(valid, key, n_groups)`` where ``valid`` is the boolean cell mask
        (value and group both non-``NaN``), ``key`` the ``int64`` key of each
        valid cell in row-major order and ``n_groups`` the number of distinct
        group ids.  Group ids are matched on their float value, so ``3.0`` and
        ``3`` are the same industry.
    """
    valid = ~np.isnan(a) & ~np.isnan(g)
    rows = np.nonzero(valid)[0]
    codes, uniques = pd.factorize(g[valid])
    n_groups = len(uniques)
    key = rows.astype(np.int64) * n_groups + codes.astype(np.int64)
    return valid, key, n_groups


def _group_mean(vals: np.ndarray, key: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-key mean and count (``NaN`` mean for empty keys) via ``np.bincount``."""
    s = np.bincount(key, weights=vals, minlength=size)
    c = np.bincount(key, minlength=size).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / c
    return mean, c


def _group_input(x: pd.DataFrame, g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Validate and align the ``(x, g)`` pair, returning float arrays."""
    a = _values(x, "x")
    gv = _values(_aligned(g, x, "g"), "g")
    return a, gv


# ----------------------------------------------------------------------------
# plain cross-sectional operators
# ----------------------------------------------------------------------------


def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank across assets per date, in ``(0, 1]`` (ties averaged).

    The row maximum scores ``1.0``, the minimum ``1/n_valid``; ``NaN`` preserved.

    Ranks limit the effect of outliers and make signals comparable across
    dates; they are also the input of the rank IC.
    """
    _check_panel(x)
    res = x.rank(axis=1, method="average", pct=True, numeric_only=False)
    return _wrap(res.to_numpy(dtype=np.float64, na_value=np.nan), x)


def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score ``(x - mean_row) / std_row`` with ``ddof=0``.

    A row with zero dispersion (all valid values equal, including a single valid
    value) becomes all-``NaN`` - there is no information to standardise.
    """
    a = _values(x)
    cnt, mean = _row_count_mean(a)
    std = _row_std(a, mean, cnt, ddof=0)
    with np.errstate(all="ignore"):
        out = (a - mean[:, None]) / std[:, None]
        out = np.where((std == 0.0)[:, None], np.nan, out)
    return _wrap(_finite(out), x)


def cs_demean(x: pd.DataFrame) -> pd.DataFrame:
    """Subtract the per-date cross-sectional mean (``NaN`` ignored, preserved)."""
    a = _values(x)
    _, mean = _row_count_mean(a)
    return _wrap(a - mean[:, None], x)


def cs_scale(x: pd.DataFrame) -> pd.DataFrame:
    """Scale each row so that ``sum(|x|) == 1`` (a "dollar-neutral book" scaling).

    A row whose absolute sum is zero (all valid values are ``0``) becomes
    all-``NaN``; every other row keeps its ``NaN`` pattern.

    Division by gross exposure gives weights per unit of capital, so alphas of
    different scale can be compared and combined.
    """
    a = _values(x)
    denom = np.nansum(np.abs(a), axis=1)
    with np.errstate(all="ignore"):
        out = a / denom[:, None]
        out = np.where((denom == 0.0)[:, None], np.nan, out)
    return _wrap(_finite(out), x)


def cs_winsorize(x: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Clip each row to ``median ± k * (1.4826 * MAD)``.

    Rows whose MAD is zero (more than half the row is identical) are left
    unchanged; ``NaN`` preserved.

    Args:
        x: Panel.
        k: Number of robust sigmas to keep (``> 0``).

    The median and the MAD do not move with the outliers, but the mean and
    the standard deviation do.
    """
    if isinstance(k, bool) or not isinstance(k, (numbers.Real, np.number)) or not k > 0:
        raise ValueError(f"k must be a positive number, got {k!r}")
    a = _values(x)
    med = _row_quantile(a, (0.5,))[0]
    with np.errstate(invalid="ignore"):
        mad = _row_quantile(np.abs(a - med[:, None]), (0.5,))[0] * MAD_TO_SIGMA
    lo = med - float(k) * mad
    hi = med + float(k) * mad
    with np.errstate(invalid="ignore"):
        clipped = np.clip(a, lo[:, None], hi[:, None])  # NaN cells stay NaN
    usable = (mad > 0.0)[:, None]  # False for mad == 0 or NaN (all-NaN rows)
    return _wrap(np.where(usable, clipped, a), x)


def cs_quantile_clip(x: pd.DataFrame, lo: float = 0.01, hi: float = 0.99) -> pd.DataFrame:
    """Clip each row to its ``[lo, hi]`` empirical quantiles (linear interpolation).

    Args:
        x: Panel.
        lo: Lower quantile in ``[0, 1]``.
        hi: Upper quantile in ``[lo, 1]``.
    """
    lo_f, hi_f = float(lo), float(hi)
    if not (0.0 <= lo_f <= hi_f <= 1.0):
        raise ValueError(f"require 0 <= lo <= hi <= 1, got lo={lo}, hi={hi}")
    a = _values(x)
    q_lo, q_hi = _row_quantile(a, (lo_f, hi_f))
    with np.errstate(invalid="ignore"):
        out = np.clip(a, q_lo[:, None], q_hi[:, None])
    return _wrap(out, x)


# ----------------------------------------------------------------------------
# group (industry) operators
# ----------------------------------------------------------------------------


def group_neutralize(x: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Subtract the per-date *group* mean: ``x - mean(x | date, g)``.

    Cells whose value **or** group id is ``NaN`` are ``NaN`` in the output.
    Implemented with one ``np.bincount`` over ``(date, group)`` keys - ~5 ms on
    1000 x 300 - rather than a per-date ``groupby``.

    The result has no net industry exposure, so the signal is only the
    selection in each industry.
    """
    a, gv = _group_input(x, g)
    valid, key, n_groups = _group_keys(a, gv)
    mean, _ = _group_mean(a[valid], key, a.shape[0] * n_groups)
    out = np.full(a.shape, np.nan)
    out[valid] = a[valid] - mean[key]
    return _wrap(out, x)


def group_zscore(x: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Per-date z-score *within* each group (``ddof=0``).

    Groups with zero dispersion on a date (including single-member groups) give
    ``NaN``; cells whose value or group id is ``NaN`` are ``NaN``.
    """
    a, gv = _group_input(x, g)
    valid, key, n_groups = _group_keys(a, gv)
    size = a.shape[0] * n_groups
    vals = a[valid]
    mean, cnt = _group_mean(vals, key, size)
    dev = vals - mean[key]
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.sqrt(np.bincount(key, weights=dev * dev, minlength=size) / cnt)
    std = np.where(std <= _REL_EPS * np.abs(mean), 0.0, std)  # numerically constant -> 0
    with np.errstate(all="ignore"):
        z = dev / std[key]
        z = np.where(std[key] == 0.0, np.nan, z)
    out = np.full(a.shape, np.nan)
    out[valid] = z
    return _wrap(_finite(out), x)


def group_rank(x: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Per-date percentile rank *within* each group, in ``(0, 1]`` (ties averaged).

    Cells whose value or group id is ``NaN`` are ``NaN``.  Uses a single Cython
    ``groupby(...).rank`` over ``(date, group)`` keys.
    """
    a, gv = _group_input(x, g)
    valid, key, _ = _group_keys(a, gv)
    ranks = pd.Series(a[valid]).groupby(key, sort=False).rank(method="average", pct=True)
    out = np.full(a.shape, np.nan)
    out[valid] = ranks.to_numpy(dtype=np.float64)
    return _wrap(out, x)


# ----------------------------------------------------------------------------
# regression-based neutralisation
# ----------------------------------------------------------------------------


def cs_regress_out(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Per-date OLS residual of ``y`` on ``[1, x]`` in closed form.

    For each date the regression uses the cells where both ``y`` and ``x`` are
    valid; the residual is ``(y - ȳ) - β (x - x̄)`` with ``β = cov(x, y) / var(x)``.
    Output is ``NaN`` where either input is ``NaN`` and for dates with fewer than
    three valid pairs.  If ``x`` is constant on a date (``var(x) = 0``) the slope
    is undefined and the residual is simply the demeaned ``y``.

    The daily residual is the part of the factor that the regressor (for example
    log market cap or beta) does not explain.
    """
    a = _values(y, "y")
    b = _values(_aligned(x, y, "x"), "x")
    valid = ~np.isnan(a) & ~np.isnan(b)
    cnt = valid.sum(axis=1)
    a0 = np.where(valid, a, 0.0)
    b0 = np.where(valid, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = a0.sum(axis=1) / cnt
        mean_b = b0.sum(axis=1) / cnt
    da = np.where(valid, a - mean_a[:, None], 0.0)
    db = np.where(valid, b - mean_b[:, None], 0.0)
    sxy = (da * db).sum(axis=1)
    sxx = (db * db).sum(axis=1)
    if a.shape[1] == 0:
        return _wrap(np.full(a.shape, np.nan), y)
    b_masked = np.where(valid, b, np.nan)
    constant = np.fmax.reduce(b_masked, axis=1) == np.fmin.reduce(b_masked, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(constant | (sxx <= 0.0), 0.0, sxy / sxx)
    resid = da - beta[:, None] * db
    resid[~valid] = np.nan
    resid[cnt < 3] = np.nan
    return _wrap(_finite(resid), y)


def cs_neutralize(x: pd.DataFrame, g: pd.DataFrame, size: pd.DataFrame) -> pd.DataFrame:
    """Residual of ``x`` after a per-date regression on group dummies **and** ``size``.

    Implemented exactly (Frisch-Waugh-Lovell) as :func:`group_neutralize` of both
    ``x`` and ``size`` on ``g``, followed by :func:`cs_regress_out` of the two
    residuals - which equals the joint OLS on ``[dummies, size]``.  The joint
    ``NaN`` mask of ``(x, g, size)`` is applied first so the two partial
    regressions see the same sample; the output is ``NaN`` wherever any of the
    three inputs is, and on dates with fewer than three valid names.

    The result is neutral to industry and size, the two exposures that
    most risk models price.
    """
    a = _values(x, "x")
    gv = _values(_aligned(g, x, "g"), "g")
    sv = _values(_aligned(size, x, "size"), "size")
    joint = np.isnan(a) | np.isnan(gv) | np.isnan(sv)
    xm = _wrap(np.where(joint, np.nan, a), x)
    sm = _wrap(np.where(joint, np.nan, sv), x)
    gm = _wrap(gv, x)
    return cs_regress_out(group_neutralize(xm, gm), group_neutralize(sm, gm))
