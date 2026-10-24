"""Factor pre-processing pipeline.

A raw factor panel (the output of an expression such as
``ts_delta(close, 5) / close``) is turned into a *tradeable* cross-sectional
score in five fixed steps::

    universe mask -> winsorize -> fill -> neutralize -> standardize

Every step is a documented, vectorised cross-sectional operator from
:mod:`alphaforge.ops.cs`; this module only orchestrates them, so the pipeline
inherits their NaN semantics and can never look ahead (each date is processed
independently).

The steps clip outliers, fill missing names at a neutral score, remove industry
and size exposure, and standardise, so all factors share one scale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from alphaforge.data.universe import apply_universe
from alphaforge.ops.cs import (
    cs_neutralize,
    cs_quantile_clip,
    cs_regress_out,
    cs_winsorize,
    cs_zscore,
    group_neutralize,
)
from alphaforge.ops.elementwise import log as _log_op

if TYPE_CHECKING:  # pragma: no cover - typing only
    from alphaforge.data.store import DataStore

logger = logging.getLogger("alphaforge.preprocess")

__all__ = ["preprocess_factor", "log_market_cap"]

#: Fill strategies accepted by :func:`preprocess_factor`.
FILL_METHODS: tuple[str, ...] = ("cs_median", "zero")

#: Winsorisation strategies accepted by :func:`preprocess_factor` (besides a
#: ``(lo_q, hi_q)`` tuple).
WINSOR_METHODS: tuple[str, ...] = ("mad",)

#: Store field used as the size exposure (``log`` of it is regressed out).
SIZE_FIELD: str = "market_cap"

#: Store field holding the industry (group) codes.
INDUSTRY_FIELD: str = "industry"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _store_panel(store: DataStore, name: str, like: pd.DataFrame) -> pd.DataFrame:
    """Fetch ``store.get(name)`` re-indexed onto ``like``'s dates x assets.

    A factor may legitimately cover a date sub-range (a train slice) or a
    subset of assets; aligning the store panel onto the factor - never the
    other way round - keeps the caller's grid intact.

    Raises:
        KeyError: if the store lacks the field (with the available names).
    """
    panel = store.get(name)
    if panel.index.equals(like.index) and panel.columns.equals(like.columns):
        return panel
    return panel.reindex(index=like.index, columns=like.columns)


def log_market_cap(store: DataStore, like: pd.DataFrame | None = None) -> pd.DataFrame:
    """``log(market_cap)`` aligned to ``like`` (or the store grid) - the size exposure.

    Non-positive or missing market caps give ``NaN``.

    Market cap is log-normal, so the regression uses its log; the raw
    level would give the largest names too much effect on the slope.
    """
    mcap = store.get(SIZE_FIELD)
    if like is not None:
        mcap = _store_panel(store, SIZE_FIELD, like)
    return _log_op(mcap)


def _validate_winsor(winsor: Any) -> tuple[str, tuple[float, float] | None]:
    """Normalise the ``winsor`` argument to ``(kind, quantiles)``."""
    if winsor is None:
        return "none", None
    if isinstance(winsor, str):
        key = winsor.lower()
        if key not in WINSOR_METHODS:
            raise ValueError(f"winsor must be one of {WINSOR_METHODS}, a (lo, hi) tuple or None; got {winsor!r}")
        return key, None
    if isinstance(winsor, (tuple, list)) and len(winsor) == 2:
        lo, hi = float(winsor[0]), float(winsor[1])
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"winsor quantiles must satisfy 0 <= lo <= hi <= 1, got {winsor!r}")
        return "quantile", (lo, hi)
    raise ValueError(f"winsor must be 'mad', a (lo_q, hi_q) tuple or None; got {winsor!r}")


def _validate_fill(fill: Any) -> str | None:
    if fill is None:
        return None
    if not isinstance(fill, str) or fill.lower() not in FILL_METHODS:
        raise ValueError(f"fill must be one of {FILL_METHODS} or None; got {fill!r}")
    return fill.lower()


def _validate_neutralize(neutralize: Any) -> tuple[str, ...]:
    if neutralize is None:
        return ()
    if isinstance(neutralize, str):
        neutralize = (neutralize,)
    try:
        items = tuple(str(n) for n in neutralize)
    except TypeError:
        raise ValueError(f"neutralize must be a tuple of field names, got {neutralize!r}") from None
    if len(set(items)) != len(items):
        raise ValueError(f"neutralize contains duplicates: {items!r}")
    return items


def _fill_target(x: pd.DataFrame, store: DataStore, universe: pd.DataFrame | None) -> np.ndarray:
    """Boolean cell mask of *where* a missing factor value may be filled.

    * ``universe`` given -> cells that are in the universe;
    * otherwise, if the store has a ``close`` field -> cells whose close is
      known (the asset exists / is listed on that date);
    * otherwise every cell.

    Filling a name that does not exist on a date would manufacture a position
    in a non-tradeable asset, which is why the target is restricted.
    """
    if universe is not None:
        u = universe.reindex(index=x.index, columns=x.columns)
        return (u.notna() & (u != 0.0)).to_numpy()
    if store.has("close"):
        return _store_panel(store, "close", x).notna().to_numpy()
    return np.ones(x.shape, dtype=bool)


def _fill(x: pd.DataFrame, method: str, target: np.ndarray) -> pd.DataFrame:
    """Fill missing cells inside ``target`` on rows that have *some* coverage.

    ``"cs_median"`` uses the row median of the (already masked / winsorised)
    factor, ``"zero"`` uses ``0.0``.  Rows without a single valid value stay
    all-``NaN`` - there is nothing to be neutral *relative to*.
    """
    a = x.to_numpy(dtype=np.float64, na_value=np.nan)
    missing = np.isnan(a)
    has_cov = ~missing.all(axis=1)
    if method == "cs_median":
        fill_val = np.full(a.shape[0], np.nan)
        if has_cov.any():  # nanmedian only on rows with data -> no "All-NaN slice" warning
            fill_val[has_cov] = np.nanmedian(a[has_cov], axis=1)
    elif method == "zero":
        fill_val = np.where(has_cov, 0.0, np.nan)
    else:  # pragma: no cover - guarded by _validate_fill
        raise ValueError(f"unknown fill method {method!r}")
    out = np.where(missing & target & has_cov[:, None], fill_val[:, None], a)
    n_filled = int((missing & target & has_cov[:, None]).sum())
    logger.debug("fill=%s: filled %d cells", method, n_filled)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _neutralize(x: pd.DataFrame, store: DataStore, items: tuple[str, ...]) -> pd.DataFrame:
    """Apply the requested neutralisations.

    ``"industry"`` and ``"size"`` together are handled by the exact joint
    regression :func:`~alphaforge.ops.cs.cs_neutralize` (industry dummies +
    ``log(market_cap)``) so the result has *both* zero industry means and zero
    size slope - sequential one-at-a-time regressions would leave a small
    residual exposure to whichever was removed first.  Any other name is
    treated as a store field to regress out (e.g. ``"beta"``), applied after
    industry/size in the order given.
    """
    out = x
    want_ind = INDUSTRY_FIELD in items
    want_size = "size" in items
    if want_ind and want_size:
        out = cs_neutralize(out, _store_panel(store, INDUSTRY_FIELD, x), log_market_cap(store, x))
    elif want_ind:
        out = group_neutralize(out, _store_panel(store, INDUSTRY_FIELD, x))
    elif want_size:
        out = cs_regress_out(out, log_market_cap(store, x))
    for name in items:
        if name in (INDUSTRY_FIELD, "size"):
            continue
        if not store.has(name):
            raise ValueError(
                f"neutralize item {name!r} is neither 'industry'/'size' nor a store field (available: {store.fields})"
            )
        out = cs_regress_out(out, _store_panel(store, name, x))
    return out


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def preprocess_factor(
    x: pd.DataFrame,
    store: DataStore,
    *,
    universe: pd.DataFrame | None = None,
    winsor: str | tuple | None = "mad",
    fill: str | None = "cs_median",
    neutralize: tuple[str, ...] = ("industry", "size"),
    standardize: bool = True,
    winsor_k: float = 3.0,
) -> pd.DataFrame:
    """Turn a raw factor panel into a universe-masked, robust, neutral z-score.

    Order of operations::

        universe mask -> winsorize -> fill -> neutralize -> standardize

    Args:
        x: Raw factor panel (dates x assets).  May be a date sub-range or an
            asset subset of the store; store fields are aligned onto it.
        store: :class:`~alphaforge.data.DataStore` providing ``industry`` and
            ``market_cap`` (only fetched when the matching neutralisation is
            requested) and, optionally, ``close`` (used to bound the fill).
        universe: Optional 1.0/NaN investability mask (see
            :func:`~alphaforge.data.universe.build_universe`).  Cells outside
            the universe are ``NaN`` in the output, always.
        winsor: ``"mad"`` clips each row to ``median ± winsor_k * 1.4826 * MAD``
            (:func:`~alphaforge.ops.cs.cs_winsorize`); a ``(lo_q, hi_q)`` tuple
            clips to row quantiles (:func:`~alphaforge.ops.cs.cs_quantile_clip`);
            ``None`` skips.
        fill: ``"cs_median"`` fills missing in-universe cells with the row
            median of the masked (winsorised) factor, ``"zero"`` with ``0.0``,
            ``None`` leaves them missing.  Only rows with at least one valid
            value are filled; when ``universe`` is ``None`` the fill is limited
            to cells whose ``close`` is known (if the store has ``close``).
        neutralize: Exposures to remove.  ``"industry"`` demeans within
            ``store.get("industry")``; ``"size"`` regresses out
            ``log(store.get("market_cap"))``.  Both together are one joint
            regression (:func:`~alphaforge.ops.cs.cs_neutralize`).  Any other
            name must be a store field and is regressed out afterwards.  ``()``
            skips.
        standardize: Apply :func:`~alphaforge.ops.cs.cs_zscore` (``ddof=0``)
            last, so each row has mean 0 and std 1.
        winsor_k: Robust-sigma multiplier for ``winsor="mad"`` (default 3).

    Returns:
        A new ``float64`` panel on ``x``'s index/columns.  ``NaN`` where the
        input (after the universe mask) had no coverage at all on a date, where
        the cell is outside the universe, and where a requested neutralisation
        exposure (industry code / market cap) is missing.  A date with zero
        dispersion after neutralisation becomes all-``NaN`` when
        ``standardize`` is on.

    Raises:
        TypeError: if ``x`` is not a DataFrame.
        ValueError: on an unknown ``winsor`` / ``fill`` / ``neutralize`` option.
        KeyError: if a required store field is missing.

    No look-ahead: every step is a per-date cross-sectional operation, so row
    ``t`` of the output depends only on row ``t`` of the inputs.

    Hand-written and mined factors go through the same five steps, so their
    IC and ICIR are comparable.
    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"x must be a pandas DataFrame panel, got {type(x).__name__}")
    winsor_kind, winsor_q = _validate_winsor(winsor)
    fill_method = _validate_fill(fill)
    neut_items = _validate_neutralize(neutralize)

    # 1. universe mask (also a defensive copy / float64 cast: x does not change)
    out = x.astype(np.float64, copy=True)
    if universe is not None:
        if not isinstance(universe, pd.DataFrame):
            raise TypeError("universe must be a pandas DataFrame mask or None")
        out = apply_universe(out, universe)
    coverage = out.notna().to_numpy()
    n_dates_with_cov = int(coverage.any(axis=1).sum())

    # 2. winsorize
    if winsor_kind == "mad":
        out = cs_winsorize(out, k=float(winsor_k))
    elif winsor_kind == "quantile":
        if winsor_q is None:
            raise ValueError("quantile winsorizing needs winsor_q (lo, hi)")
        out = cs_quantile_clip(out, lo=winsor_q[0], hi=winsor_q[1])

    # 3. fill
    if fill_method is not None:
        out = _fill(out, fill_method, _fill_target(out, store, universe))

    # 4. neutralize
    if neut_items:
        out = _neutralize(out, store, neut_items)

    # 5. standardize
    if standardize:
        out = cs_zscore(out)

    if universe is not None:  # belt and braces: nothing outside the universe survives
        out = apply_universe(out, universe)

    logger.debug(
        "preprocess_factor: %d x %d, winsor=%s, fill=%s, neutralize=%s, standardize=%s; "
        "%d/%d dates with coverage, output coverage %.1f%%",
        out.shape[0],
        out.shape[1],
        winsor,
        fill_method,
        neut_items,
        standardize,
        n_dates_with_cov,
        out.shape[0],
        100.0 * out.notna().to_numpy().mean() if out.size else float("nan"),
    )
    return out
