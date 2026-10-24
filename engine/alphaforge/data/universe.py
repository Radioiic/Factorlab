"""Investable-universe construction.

The miner sees only the names of the tradable universe, so illiquid or halted names cannot
give a factor its result.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .store import DataStore

logger = logging.getLogger("alphaforge.data.universe")

__all__ = ["build_universe", "apply_universe"]


def build_universe(
    store: DataStore,
    min_price: float = 1.0,
    min_adv_quantile: float = 0.1,
    adv_window: int = 20,
    require_tradable: bool = True,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return a 1.0/NaN mask panel: 1.0 where the asset is investable at t.
    ADV = rolling mean of `amount` over adv_window using only data <= t; assets below the
    cross-sectional `min_adv_quantile` of ADV at t are excluded.  Why: research on
    untradeable names produces alphas that vanish in production.

    Filters applied at each date ``t`` (all backward-looking, so no look-ahead):
        1. ``close`` is not NaN and ``close >= min_price`` (skip when ``min_price <= 0``).
        2. ``tradable == 1`` when ``require_tradable`` (raises ``KeyError`` if the store has
           no ``tradable`` field).
        3. ADV filter: ADV is ``amount.rolling(adv_window, min_periods).mean()`` (NaN days,
           e.g. halts, are skipped inside the window); names with ADV strictly below the
           row-wise ``min_adv_quantile`` quantile, or with undefined ADV, are excluded.
           Skipped when ``min_adv_quantile <= 0``.

    Args:
        store: data store with ``close``, ``amount`` (and ``tradable``) fields.
        min_price: minimum close price.
        min_adv_quantile: cross-sectional quantile of ADV below which names are dropped.
        adv_window: trailing window (days) for ADV.
        require_tradable: exclude halted names.
        min_periods: minimum observations for ADV (default 1, so the first
            ``adv_window - 1`` dates are not empty and single halts do not evict a name).

    Returns:
        Float panel aligned to the store: 1.0 (in universe) or NaN.
    """
    close = store.get("close")
    mask = close.notna()
    if min_price is not None and min_price > 0:
        mask &= close >= float(min_price)
    if require_tradable:
        mask &= store.get("tradable") == 1.0
    if min_adv_quantile is not None and min_adv_quantile > 0:
        if adv_window < 1:
            raise ValueError("adv_window must be >= 1")
        if not 0 < min_adv_quantile < 1:
            raise ValueError("min_adv_quantile must be in (0, 1)")
        mp = 1 if min_periods is None else int(min_periods)
        adv = store.get("amount").rolling(adv_window, min_periods=mp).mean()
        threshold = adv.quantile(float(min_adv_quantile), axis=1)
        mask &= adv.ge(threshold, axis=0)  # NaN adv or NaN threshold -> False

    universe = pd.DataFrame(np.where(mask.to_numpy(), 1.0, np.nan), index=store.dates, columns=store.assets)
    n_dates = len(store.dates)
    if n_dates:
        per_date = universe.notna().sum(axis=1)
        logger.info(
            "universe: mean %.1f of %d names per date (min %d, max %d)",
            per_date.mean(),
            len(store.assets),
            per_date.min(),
            per_date.max(),
        )
    return universe


def apply_universe(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Mask ``panel`` to the universe: NaN wherever the asset is not investable.

    ``universe`` is aligned to ``panel`` (missing cells count as *not* in universe); any
    non-NaN, non-zero value is treated as "in".  Returns a new panel (inputs untouched).
    """
    u = universe.reindex(index=panel.index, columns=panel.columns)
    mask = u.notna() & (u != 0.0)
    return panel.where(mask)
