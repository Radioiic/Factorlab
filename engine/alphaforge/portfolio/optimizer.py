"""Simple closed-form portfolio optimiser: diagonal mean-variance with constraints.

The optimiser solves ``max_w  s'w - (lambda/2) w' D w`` with ``D = diag(risk^2)``, whose
unconstrained solution is ``w_i = s_i / (lambda * risk_i^2)`` (inverse-variance scaled
scores).  Constraints are then imposed in closed form, in this order:

1. industry-demean (industry-neutral book) if ``industry`` is given;
2. dollar-neutral: subtract the cross-sectional mean so ``sum_i w_i = 0``;
3. size each leg to the gross target (``sum |w| = gross``);
4. single-name cap ``|w| <= max_weight``, enforced *exactly* with the cap-and-redistribute
   loop of :func:`alphaforge.portfolio.construct.cap_and_redistribute` on each leg so the
   leg sums (and hence gross exposure and dollar neutrality) are preserved whenever the
   cap is feasible.

Inverse-variance sizing makes the risk of each name approximately equal, and it needs no
covariance estimate.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.portfolio.construct import cap_and_redistribute

logger = logging.getLogger("alphaforge.portfolio.optimizer")

__all__ = ["optimize_weights", "rolling_risk"]


def rolling_risk(store: Any, window: int = 60) -> pd.DataFrame:
    """Per-asset daily volatility: rolling sample std of returns over ``window`` days.

    Uses only rows ``<= t`` for row ``t`` (``min_periods = window``, so the first
    ``window - 1`` rows are ``NaN``).  Falls back to ``close.pct_change()`` if the store
    has no ``returns`` field.

    Args:
        store: object exposing ``.get(name)`` / ``.has(name)`` (a ``DataStore``).
        window: trailing window length in trading days.

    Returns:
        Volatility panel aligned to the store's panels.
    """
    if int(window) < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if store.has("returns"):
        ret = store.get("returns")
    else:
        close = store.get("close")
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = close / close.shift(1) - 1.0
        ret = ret.replace([np.inf, -np.inf], np.nan)
    return ret.rolling(int(window), min_periods=int(window)).std()


def _group_demean(w: np.ndarray, g: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Subtract the (date, industry) mean from ``w`` on valid cells; long-format groupby."""
    n_rows, n_cols = w.shape
    out = w.copy()
    pos = np.flatnonzero(valid.ravel())
    if pos.size == 0:
        return out
    df = pd.DataFrame({"d": pos // n_cols, "g": g.ravel()[pos], "w": w.ravel()[pos]})
    means = df.groupby(["d", "g"], sort=False)["w"].transform("mean").to_numpy()
    flat = out.ravel()
    flat[pos] -= means
    return flat.reshape(n_rows, n_cols)


def optimize_weights(
    scores: pd.DataFrame,
    *,
    risk: pd.DataFrame | None = None,
    industry: pd.DataFrame | None = None,
    risk_aversion: float = 10.0,
    max_weight: float | None = 0.02,
    gross: float = 2.0,
    dollar_neutral: bool = True,
    store: Any = None,
    risk_window: int = 60,
    risk_floor_frac: float = 0.1,
    scale_to_gross: bool = True,
) -> pd.DataFrame:
    """Diagonal mean-variance weights: ``w ~ score / risk^2``, neutralised, capped, scaled.

    Args:
        scores: expected-return proxy panel (dates x assets), e.g. a z-scored composite.
            ``NaN`` = not in universe -> weight ``0.0``.
        risk: per-asset daily volatility panel (same shape).  ``None`` -> if ``store`` is
            given, ``rolling_risk(store, risk_window)`` (trailing std of returns, no
            look-ahead); otherwise **unit risk** (``1.0`` everywhere), which reduces the
            optimiser to score-proportional weights.  Callers typically pass the rolling
            60-day std of ``store.get("returns")``.  Non-positive / ``NaN`` risk on a scored
            name -> that name gets weight ``0`` (no risk estimate, no position).
        industry: optional float-code panel; weights are demeaned within each (date,
            industry) bucket, making the book industry-neutral before capping.  Names with
            a ``NaN`` industry code are excluded from an industry-neutral book.
        risk_aversion: the mean-variance ``lambda``.  Because the book is rescaled to
            ``gross`` afterwards (``scale_to_gross=True``), ``lambda`` does not change the
            *shape* of the weights -- it is kept for fidelity with the closed-form solution
            and matters only when ``scale_to_gross=False``.
        max_weight: single-name cap on ``|w|``; ``None`` disables it.  A hard limit: when
            it is infeasible (``n_names * max_weight < leg target``) the book runs
            under-invested and a warning is logged.
        gross: target gross exposure ``sum_i |w_i|`` (``2.0`` = 100% long / 100% short).
        dollar_neutral: subtract the cross-sectional mean so ``sum_i w_i = 0``; with a
            binding cap on one leg, both legs are shrunk to the same feasible size so
            neutrality is preserved (gross then falls below the target).
        store: optional ``DataStore`` used only to derive ``risk`` when it is ``None``.
        risk_window: window for the derived rolling risk (default 60 days).
        risk_floor_frac: floor each name's risk at this fraction of the cross-sectional
            median risk on that date (guards inverse-variance sizing against near-zero
            vol names, e.g. after a halt).  ``0`` disables the floor.
        scale_to_gross: ``True`` (default) rescales to exactly ``gross``; ``False``
            keeps the natural mean-variance size ``score / (lambda * risk^2)`` and uses
            ``gross`` only as an upper limit.

    Returns:
        Weight panel (same index/columns as ``scores``): ``0.0`` for unscored names, an
        all-``NaN`` row on dates without any valid (score, risk) pair.  Row-wise, hence
        free of look-ahead.  Never mutates its inputs.

    Raises:
        ValueError: on non-positive ``risk_aversion``/``gross``/``max_weight``.
    """
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("scores must be a pandas DataFrame panel")
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be > 0, got {risk_aversion}")
    if gross <= 0:
        raise ValueError(f"gross must be > 0, got {gross}")
    if max_weight is not None and max_weight <= 0:
        raise ValueError(f"max_weight must be > 0 or None, got {max_weight}")
    if risk_floor_frac < 0:
        raise ValueError(f"risk_floor_frac must be >= 0, got {risk_floor_frac}")

    s = scores.to_numpy(dtype=float)
    n_rows, n_cols = s.shape

    # ---- risk panel ---------------------------------------------------------------
    if risk is None and store is not None:
        risk = rolling_risk(store, risk_window)
        logger.debug("optimize_weights: risk derived from store as rolling %d-day std of returns", risk_window)
    if risk is None:
        r = np.ones_like(s)
        logger.debug("optimize_weights: no risk panel given; using unit risk (score-proportional weights)")
    else:
        r = risk.reindex(index=scores.index, columns=scores.columns).to_numpy(dtype=float)
        r = np.where(np.isfinite(r) & (r > 0), r, np.nan)
        if risk_floor_frac > 0:
            with warnings.catch_warnings():
                # all-NaN rows (no scored names) legitimately have no median
                warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)
                med = np.nanmedian(np.where(np.isfinite(s), r, np.nan), axis=1)
            floor = np.where(np.isfinite(med), risk_floor_frac * med, 0.0)
            r = np.maximum(r, floor[:, None])

    valid = np.isfinite(s) & np.isfinite(r)
    has_scores = np.isfinite(s).any(axis=1)

    if industry is not None:
        g = industry.reindex(index=scores.index, columns=scores.columns).to_numpy(dtype=float)
        n_missing = int((valid & ~np.isfinite(g)).sum())
        if n_missing:
            logger.debug("optimize_weights: %d scored cells have no industry code and are excluded", n_missing)
        valid &= np.isfinite(g)

    # ---- unconstrained diagonal MV solution ------------------------------------------
    w = np.zeros_like(s)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_mv = s / (float(risk_aversion) * r**2)
    w[valid] = w_mv[valid]

    # ---- neutralisation ---------------------------------------------------------------
    if industry is not None:
        w = _group_demean(w, g, valid)
    if dollar_neutral:
        n_valid = valid.sum(axis=1)
        mean = np.divide(w.sum(axis=1), n_valid, out=np.zeros(n_rows), where=n_valid > 0)
        w = np.where(valid, w - mean[:, None], 0.0)

    # ---- leg sizing -------------------------------------------------------------------
    pos = np.where(w > 0, w, 0.0)
    neg = np.where(w < 0, -w, 0.0)
    long_sum = pos.sum(axis=1)
    short_sum = neg.sum(axis=1)
    total = long_sum + short_sum

    if scale_to_gross:
        scale = np.divide(np.full(n_rows, float(gross)), total, out=np.zeros(n_rows), where=total > 0)
    else:
        scale = np.where(total > gross, gross / np.where(total > 0, total, 1.0), 1.0)
    t_long = long_sum * scale
    t_short = short_sum * scale

    if max_weight is not None and np.isfinite(max_weight):
        cap_long = (pos > 0).sum(axis=1) * max_weight
        cap_short = (neg > 0).sum(axis=1) * max_weight
        if dollar_neutral:
            # keep the legs equal under a binding cap so sum(w) stays 0
            feasible = np.minimum(np.minimum(t_long, cap_long), np.minimum(t_short, cap_short))
            infeasible = (t_long > cap_long + 1e-12) | (t_short > cap_short + 1e-12)
            t_long = np.where(infeasible, feasible, t_long)
            t_short = np.where(infeasible, feasible, t_short)
        else:
            infeasible = (t_long > cap_long + 1e-12) | (t_short > cap_short + 1e-12)
        n_bad = int((infeasible & has_scores).sum())
        if n_bad:
            logger.warning(
                "optimize_weights: max_weight=%.4f cannot reach gross=%.2f on %d/%d dates "
                "(too few names); book runs under-invested",
                max_weight,
                gross,
                n_bad,
                int(has_scores.sum()),
            )

    pos = cap_and_redistribute(pos, max_weight, target=t_long)
    neg = cap_and_redistribute(neg, max_weight, target=t_short)
    w = pos - neg

    no_valid = ~valid.any(axis=1)
    if no_valid.any():
        w[no_valid, :] = np.nan

    logger.debug(
        "optimize_weights: %d dates x %d assets, gross target=%.2f, dollar_neutral=%s, industry=%s, "
        "avg realised gross=%.3f",
        n_rows,
        n_cols,
        gross,
        dollar_neutral,
        industry is not None,
        float(np.nanmean(np.abs(w).sum(axis=1))) if n_rows else 0.0,
    )
    return pd.DataFrame(w, index=scores.index, columns=scores.columns, dtype=float)
