"""Portfolio construction: turn cross-sectional scores into target weights.

Construction rules turn ranks into a book with explicit properties: dollar
neutrality, equal weights, industry balance and single-name caps.

Conventions: every panel is a ``pd.DataFrame`` indexed by ``date`` with
asset ids as columns; ``NaN`` means "not in the universe".  Every function in this
module is purely cross-sectional (row-wise) or uses only rows ``<= t`` to produce row
``t``, so nothing here can look ahead.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("alphaforge.portfolio.construct")

__all__ = [
    "weights_from_scores",
    "rebalance_schedule",
    "hold_weights",
    "cap_and_redistribute",
]

# Relative tolerance used when deciding whether a weight exceeds the cap.
_CAP_TOL = 1e-12


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _as_float_array(panel: pd.DataFrame, name: str) -> np.ndarray:
    """Return a float64 ndarray view/copy of ``panel`` (validated)."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame panel, got {type(panel).__name__}")
    return panel.to_numpy(dtype=float)


def _leg_counts(frac: float, n_valid: np.ndarray) -> np.ndarray:
    """Number of names selected for one leg per row: ``ceil(frac * n_valid)``.

    ``ceil`` guarantees at least one name whenever the row (or industry bucket) has at
    least one valid score and ``frac > 0``; a tiny ``1e-9`` slack absorbs floating-point
    noise so that e.g. ``0.1 * 10`` yields exactly 1.
    """
    n_valid = np.asarray(n_valid, dtype=np.int64)
    if frac <= 0:
        return np.zeros_like(n_valid)
    return np.maximum(np.ceil(frac * n_valid - 1e-9).astype(np.int64), 0)


def _select_wide(s: np.ndarray, top_frac: float, bottom_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise top/bottom selection on a 2-D score array (no industry grouping).

    Returns ``(long_mask, short_mask)`` boolean arrays of the same shape.  Ties are broken
    by column order (``method="first"`` semantics) so the selected counts are exact and the
    two legs never overlap.
    """
    valid = np.isfinite(s)
    n_valid = valid.sum(axis=1)
    n_long = _leg_counts(top_frac, n_valid)
    n_short = np.minimum(_leg_counts(bottom_frac, n_valid), n_valid - n_long)

    # Descending score order; NaN / non-finite sort last; stable => ties by column order.
    key = np.where(valid, -s, np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(1, s.shape[1] + 1), s.shape), axis=1)

    long_mask = valid & (rank <= n_long[:, None])
    short_mask = valid & (rank > (n_valid - n_short)[:, None])
    return long_mask, short_mask


def _select_grouped(s: np.ndarray, g: np.ndarray, top_frac: float, bottom_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Top/bottom selection *within each (date, industry) bucket* via a long-format groupby.

    Names with a valid score but a ``NaN`` industry code are excluded (they cannot be
    neutralised).  Returns ``(long_mask, short_mask)`` in the original 2-D shape.
    """
    n_rows, n_cols = s.shape
    s_flat = s.ravel()
    g_flat = g.ravel()
    valid = np.isfinite(s_flat) & np.isfinite(g_flat)
    pos = np.flatnonzero(valid)

    long_mask = np.zeros(s_flat.shape, dtype=bool)
    short_mask = np.zeros(s_flat.shape, dtype=bool)
    if pos.size:
        df = pd.DataFrame({"d": pos // n_cols, "g": g_flat[pos], "s": s_flat[pos]})
        grp = df.groupby(["d", "g"], sort=False)
        rank = grp["s"].rank(method="first", ascending=False).to_numpy()
        n = grp["s"].transform("size").to_numpy().astype(np.int64)
        n_long = _leg_counts(top_frac, n)
        n_short = np.minimum(_leg_counts(bottom_frac, n), n - n_long)
        long_mask[pos] = rank <= n_long
        short_mask[pos] = rank > (n - n_short)
    return long_mask.reshape(n_rows, n_cols), short_mask.reshape(n_rows, n_cols)


def _equal_weight(mask: np.ndarray) -> np.ndarray:
    """Equal weights (1/n per selected name) per row; rows with no selection are all-zero."""
    n = mask.sum(axis=1).astype(float)
    w = mask.astype(float)
    np.divide(w, n[:, None], out=w, where=n[:, None] > 0)
    return w


def cap_and_redistribute(
    w: np.ndarray,
    max_weight: float | None,
    target: np.ndarray | None = None,
    *,
    max_iter: int = 100,
) -> np.ndarray:
    """Cap non-negative leg weights at ``max_weight`` and redistribute the excess pro rata.

    This is the classic "cap-and-redistribute" (water-filling) loop used by portfolio
    construction desks: names above the cap are pinned to it, and the capital they give
    up is spread over the remaining names *in proportion to their original weights*;
    repeat until no new name breaches the cap.  The loop is exact (it terminates after at
    most ``n_names`` rounds) and fully vectorised over rows.

    When the cap applies, the excess weight goes to the other names of the same leg; the
    function does not discard it.

    Args:
        w: 2-D array ``(n_rows, n_names)`` of **non-negative** weights (pass ``|w|`` of one
            leg).  ``NaN``/inf entries are treated as 0.
        max_weight: cap on every entry.  ``None`` or ``inf`` disables the cap (weights are
            only rescaled to ``target``).
        target: per-row target sum (shape ``(n_rows,)``).  Defaults to the current row sums
            (i.e. pure capping, no rescaling).  ``NaN`` targets are treated as 0.
        max_iter: safety bound on the number of redistribution rounds.

    Returns:
        A new array with ``0 <= out <= max_weight`` everywhere.  Rows sum to ``target``
        whenever that is feasible (``n_positive * max_weight >= target``); otherwise the
        cap wins and the row sums to ``n_positive * max_weight`` (the leg runs
        under-invested -- the cap is a *hard* limit).

    Raises:
        ValueError: if ``w`` contains negative entries or ``max_weight <= 0``.
    """
    w = np.array(w, dtype=float, copy=True)
    if w.ndim != 2:
        raise ValueError(f"cap_and_redistribute expects a 2-D array, got ndim={w.ndim}")
    w[~np.isfinite(w)] = 0.0
    if (w < 0).any():
        raise ValueError("cap_and_redistribute expects non-negative weights (pass |w| per leg)")

    row_sum = w.sum(axis=1)
    tgt = row_sum if target is None else np.asarray(target, dtype=float).reshape(-1)
    if tgt.shape[0] != w.shape[0]:
        raise ValueError("target must have one entry per row of w")
    tgt = np.where(np.isfinite(tgt), tgt, 0.0)
    if (tgt < 0).any():
        raise ValueError("target sums must be non-negative")

    # 1) scale each row to its target
    factor = np.divide(tgt, row_sum, out=np.zeros_like(tgt), where=row_sum > 0)
    w *= factor[:, None]

    if max_weight is None or (isinstance(max_weight, float) and np.isinf(max_weight)):
        return w
    if max_weight <= 0:
        raise ValueError(f"max_weight must be > 0, got {max_weight}")

    # 2) water-filling: pin breaching names to the cap, spread the remainder pro rata
    capped = np.zeros(w.shape, dtype=bool)
    out = w
    limit = max_weight * (1.0 + _CAP_TOL)
    for _ in range(max_iter):
        over = (out > limit) & ~capped
        if not over.any():
            break
        capped |= over
        free = (w > 0) & ~capped
        free_sum = np.where(free, w, 0.0).sum(axis=1)
        remaining = np.maximum(tgt - max_weight * capped.sum(axis=1), 0.0)
        factor = np.divide(remaining, free_sum, out=np.zeros_like(remaining), where=free_sum > 0)
        out = np.where(capped, max_weight, w * factor[:, None])
    else:  # pragma: no cover - only reachable with a pathological max_iter
        logger.debug("cap_and_redistribute: hit max_iter=%d before converging; applying hard clip", max_iter)
    return np.minimum(out, max_weight)


def _warn_if_cap_binds(leg: str, w_leg: np.ndarray, max_weight: float | None) -> None:
    """Log once per call when the single-name cap makes a leg under-invested."""
    if max_weight is None or not np.isfinite(max_weight):
        return
    active = w_leg.sum(axis=1) > 0
    if not active.any():
        return
    under = active & (w_leg.sum(axis=1) < 1.0 - 1e-9)
    if under.any():
        n_sel = (w_leg > 0).sum(axis=1)[active]
        logger.warning(
            "%s leg under-invested on %d/%d dates: max_weight=%.4f needs >= %d names but "
            "only %.1f were selected on average (raise max_weight or top/bottom_frac)",
            leg,
            int(under.sum()),
            int(active.sum()),
            max_weight,
            int(np.ceil(1.0 / max_weight - 1e-9)),
            float(n_sel.mean()),
        )


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def weights_from_scores(
    scores: pd.DataFrame,
    *,
    top_frac: float = 0.1,
    bottom_frac: float | None = None,
    industry: pd.DataFrame | None = None,
    max_weight: float | None = 0.02,
    long_only: bool = False,
) -> pd.DataFrame:
    """Equal-weight long/short quantile portfolio from a cross-sectional score panel.

    Per date the non-NaN scores are ranked; the top ``ceil(top_frac * n)`` names form the
    long leg and the bottom ``ceil(bottom_frac * n)`` names the short leg (``bottom_frac``
    defaults to ``top_frac``).  Each leg is equal-weighted and normalised so that the long
    leg sums to ``+1`` and the short leg to ``-1`` (gross exposure 2, dollar-neutral).
    With ``industry`` given, selection happens *within* each (date, industry) bucket, so
    every industry contributes the same number of longs and shorts and the book is
    industry-neutral by construction.  Finally ``|w| <= max_weight`` is enforced with
    :func:`cap_and_redistribute` (the cap is a hard limit: if a leg has too few names to
    reach a sum of 1 under the cap it runs under-invested and a warning is logged).

    The quantile book is dollar-neutral with a fixed gross and equal weights; the
    industry-neutral selection and the single-name cap are options.

    Args:
        scores: score panel (dates x assets); ``NaN``/inf = not in universe on that date.
        top_frac: fraction of valid names (per date, or per industry bucket) to hold long.
        bottom_frac: fraction to hold short; ``None`` -> same as ``top_frac``.  ``0`` -> no
            short leg.
        industry: optional panel of float industry codes aligned to ``scores`` (reindexed if
            not).  Names with a ``NaN`` code are excluded from an industry-neutral book.
        max_weight: single-name cap on ``|w|``; ``None`` disables it.
        long_only: keep only the long leg (weights sum to ``+1``).

    Returns:
        Float panel with the same index/columns as ``scores``: ``0.0`` for unselected
        names (including NaN-scored names on dates that have valid scores) and an all-NaN
        row on dates with no valid score at all.  Never mutates its inputs.

    Raises:
        ValueError: on invalid fractions (``top_frac`` outside ``(0, 1]``, ``bottom_frac``
            outside ``[0, 1]``, ``top_frac + bottom_frac > 1``) or ``max_weight <= 0``.

    Notes:
        Ties are broken by column order (``rank(method="first")``), so the selected counts
        are exact and the legs never overlap.  The function is row-wise, hence free of
        look-ahead by construction.
    """
    if bottom_frac is None:
        bottom_frac = top_frac
    if not 0.0 < top_frac <= 1.0:
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")
    if not 0.0 <= bottom_frac <= 1.0:
        raise ValueError(f"bottom_frac must be in [0, 1], got {bottom_frac}")
    if not long_only and top_frac + bottom_frac > 1.0 + 1e-12:
        raise ValueError(f"top_frac + bottom_frac must be <= 1, got {top_frac + bottom_frac}")
    if max_weight is not None and max_weight <= 0:
        raise ValueError(f"max_weight must be > 0 or None, got {max_weight}")

    s = _as_float_array(scores, "scores")
    short_frac = 0.0 if long_only else float(bottom_frac)

    if industry is None:
        long_mask, short_mask = _select_wide(s, float(top_frac), short_frac)
    else:
        g = _as_float_array(industry.reindex(index=scores.index, columns=scores.columns), "industry")
        n_missing = int((np.isfinite(s) & ~np.isfinite(g)).sum())
        if n_missing:
            logger.debug("weights_from_scores: %d scored cells have no industry code and are excluded", n_missing)
        long_mask, short_mask = _select_grouped(s, g, float(top_frac), short_frac)

    w_long = _equal_weight(long_mask)
    w_long = cap_and_redistribute(w_long, max_weight, target=long_mask.any(axis=1).astype(float))
    _warn_if_cap_binds("long", w_long, max_weight)

    if long_only:
        w = w_long
    else:
        w_short = _equal_weight(short_mask)
        w_short = cap_and_redistribute(w_short, max_weight, target=short_mask.any(axis=1).astype(float))
        _warn_if_cap_binds("short", w_short, max_weight)
        w = w_long - w_short

    no_scores = ~np.isfinite(s).any(axis=1)
    if no_scores.any():
        w[no_scores, :] = np.nan

    logger.debug(
        "weights_from_scores: %d dates, avg longs=%.1f shorts=%.1f, industry_neutral=%s, long_only=%s",
        s.shape[0],
        float(long_mask.sum(axis=1).mean()) if s.size else 0.0,
        float(short_mask.sum(axis=1).mean()) if s.size else 0.0,
        industry is not None,
        long_only,
    )
    return pd.DataFrame(w, index=scores.index, columns=scores.columns, dtype=float)


def rebalance_schedule(dates: pd.DatetimeIndex, every: int = 1) -> pd.DatetimeIndex:
    """Every ``every``-th date starting from the first one.

    A fixed calendar (for example weekly) controls turnover and makes the cost profile
    predictable.

    Args:
        dates: sorted ``DatetimeIndex`` (typically ``store.dates``).
        every: rebalance period in trading days (``1`` = every date).

    Returns:
        ``dates[::every]`` as a ``DatetimeIndex`` (name preserved).
    """
    if int(every) < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    idx = pd.DatetimeIndex(dates)
    return idx[:: int(every)]


def hold_weights(weights: pd.DataFrame, schedule: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill weights between rebalance dates.

    The weights row of each *schedule* date is taken as the new target book and held
    unchanged until the next schedule date.  Rows before the first rebalance are ``0.0``
    (not invested).  ``NaN`` cells in a rebalance row become ``0.0``; a schedule date
    whose weights row is entirely ``NaN`` (no scores that day) is skipped and the previous
    book is carried forward -- "if you cannot rebalance, you keep what you have".

    The backtest sees the held weights, so it charges turnover and costs only on
    rebalance days; the weights do not drift with prices.

    Args:
        weights: weight panel (dates x assets), e.g. from :func:`weights_from_scores`.
        schedule: rebalance dates, e.g. from :func:`rebalance_schedule`; dates that are not
            in ``weights.index`` are ignored.

    Returns:
        Float panel with the same index/columns; uses only rows ``<= t`` for row ``t``.
    """
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas DataFrame panel")
    sched = pd.DatetimeIndex(schedule)
    on_schedule = weights.index.isin(sched)
    row_valid = weights.notna().any(axis=1).to_numpy()
    rebalance = on_schedule & row_valid
    if not rebalance.any():
        logger.warning("hold_weights: no schedule date carries valid weights; returning an all-zero panel")
        return pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    held = weights.loc[rebalance].fillna(0.0).reindex(weights.index).ffill().fillna(0.0)
    logger.debug("hold_weights: %d rebalances over %d dates", int(rebalance.sum()), len(weights.index))
    return held.astype(float)
