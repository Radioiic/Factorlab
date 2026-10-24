"""Cross-sectional factor evaluation: rank IC, ICIR, long-short spread, turnover."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def forward_returns(close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Return from bar t close to bar t+horizon close (aligned at t)."""
    return close.shift(-horizon) / close - 1


def _row_rank(a: np.ndarray) -> np.ndarray:
    """Return the rank of each value within its row (NaN stays NaN)."""
    return pd.DataFrame(a).rank(axis=1).to_numpy()


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 5) -> pd.Series:
    """Spearman correlation per timestamp between factor and forward return."""
    f, r = factor.align(fwd, join="inner")
    fv, rv = f.to_numpy(dtype=float), r.to_numpy(dtype=float)
    valid = np.isfinite(fv) & np.isfinite(rv)
    fv = np.where(valid, fv, np.nan)
    rv = np.where(valid, rv, np.nan)
    fr, rr = _row_rank(fv), _row_rank(rv)
    n = valid.sum(axis=1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        fr = fr - np.nanmean(fr, axis=1, keepdims=True)
        rr = rr - np.nanmean(rr, axis=1, keepdims=True)
        num = np.nansum(fr * rr, axis=1)
        den = np.sqrt(np.nansum(fr * fr, axis=1) * np.nansum(rr * rr, axis=1))
        ic = num / den
    ic[n < min_names] = np.nan
    return pd.Series(ic, index=f.index)


def evaluate_factor(
    factor: pd.DataFrame, close: pd.DataFrame, horizon: int = 1, quantile: float = 0.2, periods_per_year: int = 365
) -> dict:
    """Return the evaluation statistics of one factor panel against the close panel.

    Inputs: ``factor`` and ``close`` (time x symbol), the forward ``horizon`` in bars, the
    ``quantile`` of each long-short leg and the bars per year. The result has ``horizon``
    and ``n_periods``; with 10 or more IC values also the IC mean, standard deviation, ICIR,
    t statistic (scaled for overlapping horizons), hit rate, coverage, the one-bar
    long-short mean and Sharpe ratio, and the turnover of the long leg.
    """
    fwd = forward_returns(close, horizon)
    ic = rank_ic(factor, fwd).dropna()
    out: dict[str, object] = {"horizon": horizon, "n_periods": int(ic.size)}
    if ic.size < 10:
        return out
    std = ic.std()
    out.update(
        start=str(ic.index[0]),
        end=str(ic.index[-1]),
        ic_mean=ic.mean(),
        ic_std=std,
        icir=ic.mean() / std if std > 0 else np.nan,
        # overlapping horizons inflate the t-stat; scale by sqrt(horizon)
        ic_tstat=ic.mean() / std * np.sqrt(ic.size / horizon) if std > 0 else np.nan,
        ic_hit_rate=(np.sign(ic) == np.sign(ic.mean())).mean(),
        coverage=float(factor.notna().to_numpy().mean()),
    )
    ranks = factor.rank(axis=1, pct=True)
    long = ranks >= 1 - quantile
    short = ranks <= quantile
    fwd1 = forward_returns(close, 1)
    ls = (fwd1.where(long).mean(axis=1) - fwd1.where(short).mean(axis=1)).dropna()
    if ls.size > 10:
        out["ls_mean"] = ls.mean()
        out["ls_sharpe"] = ls.mean() / ls.std() * np.sqrt(periods_per_year) if ls.std() > 0 else np.nan
    w = long.astype(float).div(long.sum(axis=1).replace(0, np.nan), axis=0)
    out["turnover"] = float(w.diff().abs().sum(axis=1).mean() / 2)
    return out
