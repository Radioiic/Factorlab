"""Element-wise operators.

Every operator maps ``panel -> panel`` (plus scalar parameters) and obeys the
shared operator contract:

* the output is a *new* ``float64`` panel with the caller's index/columns,
* inputs are never mutated,
* ``NaN`` in -> ``NaN`` out, and any non-finite result (``inf``, ``-inf`` from a
  division by zero, an overflow, ``log(0)`` ...) is mapped to ``NaN``.

This module also hosts the tiny helpers (:func:`_values`, :func:`_wrap`,
:func:`_finite`, :func:`_aligned`) that the ``ts`` and ``cs`` modules share, so
the whole operator library speaks one dialect.

Each operator maps ``inf`` to ``NaN``, so one bad cell drops out of the
cross-section for that date and does not change the other cells.
"""

from __future__ import annotations

import logging
import numbers
from collections.abc import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("alphaforge.ops.elementwise")

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs_",
    "sign",
    "log",
    "signed_log",
    "sqrt",
    "signed_sqrt",
    "power",
    "inverse",
    "max_",
    "min_",
    "clip",
]

# ----------------------------------------------------------------------------
# shared helpers (used by ts.py / cs.py as well)
# ----------------------------------------------------------------------------


def _check_panel(x: object, name: str = "x") -> pd.DataFrame:
    """Raise ``TypeError`` unless ``x`` is a :class:`pandas.DataFrame`."""
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame panel, got {type(x).__name__}")
    return x


def _values(x: pd.DataFrame, name: str = "x") -> np.ndarray:
    """Return the panel's values as a ``float64`` ndarray.

    The array may be a *view* of the frame's storage, so callers must never
    write into it - every operator allocates its own output.
    """
    return _check_panel(x, name).to_numpy(dtype=np.float64, na_value=np.nan)


def _wrap(values: np.ndarray, like: pd.DataFrame) -> pd.DataFrame:
    """Wrap a fresh ndarray into a panel carrying ``like``'s index/columns."""
    if values.dtype != np.float64:
        values = values.astype(np.float64)
    return pd.DataFrame(values, index=like.index, columns=like.columns)


def _finite(values: np.ndarray) -> np.ndarray:
    """Map ``±inf`` to ``NaN`` without mutating ``values`` (it may alias an input)."""
    bad = np.isinf(values)
    if bad.any():
        values = np.where(bad, np.nan, values)
    return values


def _aligned(y: pd.DataFrame, like: pd.DataFrame, name: str = "y") -> pd.DataFrame:
    """Return ``y`` re-indexed onto ``like``'s index/columns (no-op when equal).

    Operators can *assume* aligned inputs.  The binary ops still check the
    alignment, because a silent positional mismatch can look like a signal.
    Realignment is logged at DEBUG level.
    """
    _check_panel(y, name)
    if y.index.equals(like.index) and y.columns.equals(like.columns):
        return y
    logger.debug("%s re-indexed onto the primary panel (index/columns differed)", name)
    return y.reindex(index=like.index, columns=like.columns)


def _scalar(v: object, name: str) -> float:
    """Validate a real scalar parameter and return it as ``float``."""
    if isinstance(v, bool) or not isinstance(v, (numbers.Real, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(v).__name__}")
    return float(v)


def _unary(x: pd.DataFrame, fn: Callable[[np.ndarray], np.ndarray]) -> pd.DataFrame:
    """Apply ``fn`` to the values of ``x`` with numpy warnings silenced."""
    a = _values(x)
    with np.errstate(all="ignore"):
        out = fn(a)
    return _wrap(_finite(out), x)


def _binary(
    x: pd.DataFrame | float,
    y: pd.DataFrame | float,
    fn: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray],
) -> pd.DataFrame:
    """Apply a binary ufunc-like ``fn`` to two panels (or a panel and a scalar).

    At least one operand must be a panel; the other may be a real scalar which
    is broadcast.  The output takes the index/columns of the first panel operand.
    """
    if isinstance(x, pd.DataFrame):
        like = x
        a: np.ndarray | float = _values(x, "x")
        b: np.ndarray | float = _values(_aligned(y, x, "y"), "y") if isinstance(y, pd.DataFrame) else _scalar(y, "y")
    elif isinstance(y, pd.DataFrame):
        like = y
        a = _scalar(x, "x")
        b = _values(y, "y")
    else:
        raise TypeError("at least one operand of a binary operator must be a DataFrame panel")
    with np.errstate(all="ignore"):
        out = fn(a, b)
    return _wrap(_finite(np.asarray(out, dtype=np.float64)), like)


# ----------------------------------------------------------------------------
# arithmetic
# ----------------------------------------------------------------------------


def add(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise ``x + y``.

    Args:
        x: Panel.
        y: Panel aligned with ``x`` (a real scalar is also accepted and broadcast).

    Returns:
        New ``float64`` panel; ``NaN`` wherever either input is ``NaN``.
    """
    return _binary(x, y, np.add)


def sub(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise ``x - y`` (see :func:`add` for conventions)."""
    return _binary(x, y, np.subtract)


def mul(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise ``x * y`` (see :func:`add` for conventions)."""
    return _binary(x, y, np.multiply)


def div(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise ``x / y``; division by zero yields ``NaN`` (never ``inf``).

    A zero denominator gives ``NaN``, so the name drops out of the cross-section for
    that date.
    """
    return _binary(x, y, np.divide)


def neg(x: pd.DataFrame) -> pd.DataFrame:
    """Element-wise ``-x``."""
    return _unary(x, np.negative)


def abs_(x: pd.DataFrame) -> pd.DataFrame:
    """Element-wise ``|x|`` (registered as ``abs``)."""
    return _unary(x, np.abs)


def sign(x: pd.DataFrame) -> pd.DataFrame:
    """Element-wise sign: ``-1``, ``0`` or ``+1`` (``NaN`` stays ``NaN``)."""
    return _unary(x, np.sign)


# ----------------------------------------------------------------------------
# non-linear transforms
# ----------------------------------------------------------------------------


def log(x: pd.DataFrame) -> pd.DataFrame:
    """Natural logarithm; ``x <= 0`` -> ``NaN``.

    Logs turn heavy-tailed, multiplicative fields (prices, volumes) into
    near-additive, symmetric values for ranks and z-scores.
    """
    return _unary(x, np.log)


def signed_log(x: pd.DataFrame) -> pd.DataFrame:
    """Sign-preserving log compression ``sign(x) * log1p(|x|)``.

    Defined for every real ``x`` (unlike :func:`log`) and monotone, so it
    compresses outliers without changing the ordering or the sign.
    """
    return _unary(x, lambda a: np.sign(a) * np.log1p(np.abs(a)))


def sqrt(x: pd.DataFrame) -> pd.DataFrame:
    """Square root; ``x < 0`` -> ``NaN``."""
    return _unary(x, np.sqrt)


def signed_sqrt(x: pd.DataFrame) -> pd.DataFrame:
    """Sign-preserving square root ``sign(x) * sqrt(|x|)``."""
    return _unary(x, lambda a: np.sign(a) * np.sqrt(np.abs(a)))


def power(x: pd.DataFrame, p: float) -> pd.DataFrame:
    """Element-wise ``x ** p``.

    Args:
        x: Panel.
        p: Real exponent (a panel exponent is also accepted for direct use, but
            the registry types this slot as ``"float"``).

    Returns:
        ``NaN`` where the result is undefined or non-finite (negative base with a
        fractional exponent, ``0 ** p`` for ``p < 0``, overflow).
    """
    if isinstance(p, pd.DataFrame):
        return _binary(x, p, np.power)
    pf = _scalar(p, "p")
    return _unary(x, lambda a: np.power(a, pf))


def inverse(x: pd.DataFrame) -> pd.DataFrame:
    """Element-wise ``1 / x``; ``x == 0`` -> ``NaN``."""
    return _unary(x, np.reciprocal)


def max_(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise maximum (registered as ``max``); ``NaN`` if either side is ``NaN``."""
    return _binary(x, y, np.maximum)


def min_(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise minimum (registered as ``min``); ``NaN`` if either side is ``NaN``."""
    return _binary(x, y, np.minimum)


def clip(x: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Clip values into ``[lo, hi]`` (``NaN`` preserved).

    Args:
        x: Panel.
        lo: Lower bound (``None`` = unbounded below).
        hi: Upper bound (``None`` = unbounded above).

    Raises:
        ValueError: if ``lo > hi``.

    Hard clips are a utility, not a search primitive (``mineable=False``), so the
    miner cannot learn the extremes of the training sample.
    """
    lo_f = -np.inf if lo is None else _scalar(lo, "lo")
    hi_f = np.inf if hi is None else _scalar(hi, "hi")
    if lo_f > hi_f:
        raise ValueError(f"clip requires lo <= hi, got lo={lo_f}, hi={hi_f}")
    return _unary(x, lambda a: np.clip(a, lo_f, hi_f))
