"""Linear factor combination.

Each factor is cross-sectionally z-scored, oriented so that its train-window
IC is positive, and averaged with ``"equal"`` / ``"ic"`` / ``"icir"`` / explicit
weights.  The weighted mean is NaN-aware: on every cell the weights are
re-normalised over the factors that actually have a value there.

The composite weights come from the train-window ICIR only, and the same
weights apply unchanged out of sample.
"""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.ops.cs import cs_zscore

logger = logging.getLogger("alphaforge.combine.linear")

__all__ = ["orient", "combine_linear", "resolve_weights", "WEIGHT_SCHEMES"]

#: Named weighting schemes accepted by :func:`combine_linear`.
WEIGHT_SCHEMES: tuple[str, ...] = ("equal", "ic", "icir")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _ic_table(ic_table: Any) -> pd.DataFrame | None:
    """Normalise an IC table to a frame indexed by factor name (or ``None``)."""
    if ic_table is None:
        return None
    if not isinstance(ic_table, pd.DataFrame):
        raise TypeError("ic_table must be a pandas DataFrame indexed by factor name")
    if "name" in ic_table.columns:
        ic_table = ic_table.set_index("name")
    return ic_table


def _lookup(table: pd.DataFrame | None, name: str, col: str) -> float:
    """``table.loc[name, col]`` as a float; ``NaN`` when absent."""
    if table is None or col not in table.columns or name not in table.index:
        return float("nan")
    v = table.loc[name, col]
    if isinstance(v, pd.Series):  # duplicated index label
        v = v.iloc[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def orientation_signs(names: list[str], ic_table: pd.DataFrame | None) -> pd.Series:
    """``+1`` / ``-1`` per factor: ``-1`` iff its ``ic_mean`` in ``ic_table`` is negative.

    Factors missing from the table (or with a non-finite ``ic_mean``) get ``+1``.
    """
    table = _ic_table(ic_table)
    signs = []
    for n in names:
        ic = _lookup(table, n, "ic_mean")
        if table is not None and not math.isfinite(ic):
            logger.debug("orient: no finite ic_mean for %r; keeping its sign", n)
        signs.append(-1.0 if (math.isfinite(ic) and ic < 0.0) else 1.0)
    return pd.Series(signs, index=pd.Index(names, name="name"), name="sign", dtype=float)


def resolve_weights(
    names: list[str],
    weights: Mapping[str, float] | str = "equal",
    ic_table: pd.DataFrame | None = None,
) -> pd.Series:
    """Non-negative factor weights normalised to sum to one.

    Args:
        names: Factor names (order of the result).
        weights: ``"equal"``; ``"ic"`` (``|ic_mean|``) or ``"icir"`` (``|icir|``)
            read from ``ic_table``; or a ``{name: weight}`` mapping of
            non-negative magnitudes (sign is handled by orientation, never by
            the weights).  Names absent from a mapping get weight ``0``.
        ic_table: Frame indexed by factor name with columns ``ic_mean`` and
            ``icir`` (e.g. :meth:`FactorReport.table`), computed on TRAIN.
            Required for ``"ic"`` / ``"icir"``; every factor must be present.
            A non-finite statistic gives weight ``0`` with a warning.

    Returns:
        ``pd.Series`` of weights summing to one, indexed by ``names``.

    Raises:
        ValueError: unknown scheme, missing table / names, negative or
            all-zero weights.
    """
    if len(names) == 0:
        raise ValueError("no factors to weight")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate factor names: {names}")
    table = _ic_table(ic_table)
    w = np.zeros(len(names), dtype=float)

    if isinstance(weights, str):
        scheme = weights.lower()
        if scheme not in WEIGHT_SCHEMES:
            raise ValueError(f"weights must be one of {WEIGHT_SCHEMES} or a mapping, got {weights!r}")
        if scheme == "equal":
            w[:] = 1.0
        else:
            col = "ic_mean" if scheme == "ic" else "icir"
            if table is None:
                raise ValueError(f"weights={scheme!r} requires ic_table")
            if col not in table.columns:
                raise ValueError(f"ic_table lacks column {col!r} (has {list(table.columns)})")
            missing = [n for n in names if n not in table.index]
            if missing:
                raise ValueError(f"ic_table has no row for factors {missing}")
            for i, n in enumerate(names):
                v = _lookup(table, n, col)
                if not math.isfinite(v):
                    logger.warning("weights=%r: non-finite %s for %r -> weight 0", scheme, col, n)
                    v = 0.0
                w[i] = abs(v)
    elif isinstance(weights, Mapping):
        extra = [k for k in weights if k not in names]
        if extra:
            logger.warning("weights mapping has keys not among the factors (ignored): %s", extra)
        for i, n in enumerate(names):
            v = weights.get(n, 0.0)
            if isinstance(v, bool) or not isinstance(v, (numbers.Real, np.number)):
                raise TypeError(f"weight for {n!r} must be a real number, got {v!r}")
            v = float(v)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"weight for {n!r} must be finite and >= 0 (sign comes from orientation), got {v}")
            if n not in weights:
                logger.warning("weights mapping has no entry for %r -> weight 0", n)
            w[i] = v
    else:
        raise TypeError("weights must be a scheme name or a {name: weight} mapping")

    total = w.sum()
    if not total > 0.0:
        raise ValueError("all factor weights are zero; nothing to combine")
    return pd.Series(w / total, index=pd.Index(names, name="name"), name="weight", dtype=float)


def _common_grid(factors: Mapping[str, pd.DataFrame]) -> tuple[pd.Index, pd.Index]:
    """Union of the factors' dates (sorted) and assets (first-seen order)."""
    panels = list(factors.values())
    index = panels[0].index
    columns = panels[0].columns
    for p in panels[1:]:
        if not p.index.equals(index):
            index = index.union(p.index)
        if not p.columns.equals(columns):
            columns = columns.append(p.columns.difference(columns, sort=False))
    if not index.is_monotonic_increasing:
        index = index.sort_values()
    return index, columns


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def orient(values: pd.DataFrame, ic_mean: float) -> pd.DataFrame:
    """Flip the sign of a factor so that its IC is positive.

    Args:
        values: Factor panel.
        ic_mean: Mean IC of the factor (from the TRAIN window).  Negative ->
            the panel is negated; non-negative or non-finite -> unchanged.

    Returns:
        A new panel (never the input object).

    The sign flip uses in-sample evidence only, so each composite reads a high
    score as long.
    """
    if not isinstance(values, pd.DataFrame):
        raise TypeError("values must be a pandas DataFrame panel")
    try:
        ic = float(ic_mean) if ic_mean is not None else float("nan")
    except (TypeError, ValueError):
        ic = float("nan")
    out = values.astype(np.float64, copy=True)
    if math.isfinite(ic) and ic < 0.0:
        return -out
    return out


def combine_linear(
    factors: dict[str, pd.DataFrame],
    *,
    weights: dict[str, float] | str = "equal",
    ic_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Weighted, NaN-aware mean of z-scored, IC-oriented factors.

    Steps per factor: :func:`~alphaforge.ops.cs.cs_zscore` -> :func:`orient`
    by the sign of its ``ic_mean`` in ``ic_table`` (missing -> unchanged) ->
    weighted mean.  On each cell only the factors with a finite value take
    part and the weights are re-normalised over them, so a single missing
    factor does not drop a name from the composite; cells where *no* factor
    has a value are ``NaN``.

    Args:
        factors: ``{name: panel}``.  Panels are aligned onto the union of
            their dates and assets.
        weights: ``"equal"`` | ``"ic"`` | ``"icir"`` | ``{name: weight}`` -
            see :func:`resolve_weights`.  Always normalised to sum to one.
        ic_table: Frame indexed by factor name with ``ic_mean`` / ``icir``
            columns computed on the TRAIN window (e.g. ``FactorReport.table``
            or ``FactorLibrary.to_frame()``).  Needed for orientation and for
            the ``"ic"`` / ``"icir"`` schemes.

    Returns:
        Composite score panel (``float64``).  It is *not* re-standardised;
        with equal weights on identical factors it equals their z-score.

    No look-ahead: every operation is per date; the only cross-date
    information is the (train-window) table the caller supplies.
    """
    if not isinstance(factors, Mapping) or len(factors) == 0:
        raise ValueError("factors must be a non-empty {name: panel} mapping")
    for n, p in factors.items():
        if not isinstance(p, pd.DataFrame):
            raise TypeError(f"factor {n!r} must be a pandas DataFrame, got {type(p).__name__}")
    names = [str(n) for n in factors]
    w = resolve_weights(names, weights, ic_table)
    signs = orientation_signs(names, ic_table)
    index, columns = _common_grid(factors)

    num = np.zeros((len(index), len(columns)), dtype=np.float64)
    den = np.zeros_like(num)
    for name, (_key, panel) in zip(names, factors.items()):
        wi = float(w[name])
        if wi == 0.0:
            continue
        if not (panel.index.equals(index) and panel.columns.equals(columns)):
            panel = panel.reindex(index=index, columns=columns)
        z = cs_zscore(panel).to_numpy(dtype=np.float64, na_value=np.nan) * float(signs[name])
        valid = np.isfinite(z)
        num += wi * np.where(valid, z, 0.0)
        den += wi * valid
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0.0, num / den, np.nan)
    logger.info(
        "combine_linear: %d factors, weights=%s -> %s; flipped=%s; coverage %.1f%%",
        len(names),
        weights if isinstance(weights, str) else "mapping",
        {n: round(float(v), 4) for n, v in w.items()},
        [n for n in names if signs[n] < 0],
        100.0 * np.isfinite(out).mean() if out.size else float("nan"),
    )
    return pd.DataFrame(out, index=index, columns=columns)
