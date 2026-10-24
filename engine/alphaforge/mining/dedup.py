"""Greedy correlation-based de-duplication of mined factors.

The filter keeps only the candidates that are novel, in rank correlation, relative to the
candidates that it kept before.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from alphaforge.evaluation.ic import rowwise_corr

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids an import cycle with genetic.py)
    from alphaforge.mining.genetic import MinedFactor

logger = logging.getLogger("alphaforge.mining.dedup")

__all__ = ["spearman_corr_panel", "select_uncorrelated"]


def spearman_corr_panel(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Mean over dates of the row-wise Spearman correlation of two panels.

    The panels are aligned on the intersection of their dates and assets, so a
    train-window factor can be compared with a full-history one.  Dates on which the
    correlation is undefined (fewer than five joint pairs, constant row) are ignored.

    Returns:
        The mean correlation in ``[-1, 1]``, or ``NaN`` when no date has a defined value.

    Rank correlation does not change under a monotone transform, so ``x`` and
    ``exp(x)`` give ``1.0``.
    """
    if not isinstance(a, pd.DataFrame) or not isinstance(b, pd.DataFrame):
        raise TypeError("spearman_corr_panel expects two pandas DataFrames")
    if a.index.equals(b.index) and a.columns.equals(b.columns):
        a2, b2 = a, b
    else:
        dates = a.index.intersection(b.index)
        assets = a.columns.intersection(b.columns)
        if len(dates) == 0 or len(assets) == 0:
            return float("nan")
        a2 = a.reindex(index=dates, columns=assets)
        b2 = b.reindex(index=dates, columns=assets)
    corr = rowwise_corr(a2, b2, method="spearman").to_numpy(dtype=float)
    corr = corr[np.isfinite(corr)]
    return float(corr.mean()) if corr.size else float("nan")


def select_uncorrelated(
    candidates: Sequence[MinedFactor],
    values: Mapping[str, pd.DataFrame],
    *,
    max_corr: float = 0.7,
    key: Callable[[Any], float] = lambda m: m.fitness,
) -> list[MinedFactor]:
    """Greedy novelty filter: keep the best candidates that are mutually uncorrelated.

    Candidates are visited in descending ``key`` order (ties broken by expression string
    for determinism); one is kept when ``|spearman_corr_panel|`` with *every* factor kept
    so far is ``<= max_corr``.

    Args:
        candidates: :class:`~alphaforge.mining.genetic.MinedFactor` objects (anything
            with ``.expr`` works).
        values: ``{expr: panel}`` giving the value panel of every candidate, keyed by
            ``candidate.expr``.  A missing key raises ``KeyError`` (a silently dropped
            candidate would be a bug that is hard to see).
        max_corr: Correlation threshold (inclusive: exactly ``max_corr`` is kept).
        key: Ranking function, larger is better.  Candidates whose key is not finite
            (e.g. ``fitness = -inf``) are dropped.

    Returns:
        The kept candidates in visiting (descending ``key``) order.

    Notes:
        * An all-NaN value panel is dropped with a warning (it cannot be compared).
        * A correlation that is ``NaN`` (no overlapping valid dates) is treated as ``0``,
          i.e. the pair is considered independent.
        * Duplicate expressions are collapsed (the first occurrence wins).
    """
    max_corr = float(max_corr)
    if not (0.0 <= max_corr <= 1.0):
        raise ValueError(f"max_corr must be in [0, 1], got {max_corr}")
    scored: list[tuple[float, str, Any]] = []
    seen: set[str] = set()
    for cand in candidates:
        k = float(key(cand))
        if not math.isfinite(k):
            continue
        if cand.expr in seen:
            continue
        seen.add(cand.expr)
        scored.append((k, cand.expr, cand))
    scored.sort(key=lambda t: (-t[0], t[1]))

    kept: list[Any] = []
    kept_values: list[pd.DataFrame] = []
    for k, expr, cand in scored:
        try:
            v = values[expr]
        except KeyError:
            raise KeyError(f"select_uncorrelated: no value panel for candidate {expr!r}") from None
        if not np.isfinite(v.to_numpy(dtype=float, na_value=np.nan)).any():
            logger.warning("select_uncorrelated: dropping %r - value panel is all-NaN", expr)
            continue
        worst = 0.0
        redundant = False
        for other in kept_values:
            c = spearman_corr_panel(v, other)
            c = 0.0 if not math.isfinite(c) else c
            worst = max(worst, abs(c))
            if abs(c) > max_corr:
                redundant = True
                break
        if redundant:
            logger.debug("select_uncorrelated: dropped %r (|corr| %.3f > %.2f, key %.4f)", expr, worst, max_corr, k)
            continue
        kept.append(cand)
        kept_values.append(v)
    logger.info("select_uncorrelated: kept %d of %d candidates (max_corr=%.2f)", len(kept), len(scored), max_corr)
    return kept
