"""Operator registry.

The registry is the single source of truth about *which* operators exist, their
signatures (``arg_types``) and how the miner may use them (``int_choices``,
``float_choices``, ``commutative``, ``mineable``).  The expression engine
(``alphaforge.expr``) validates and evaluates trees against it and the genetic
miner (``alphaforge.mining``) samples from it.

Registered names are the function names **without a trailing underscore**:
``abs`` -> :func:`~alphaforge.ops.elementwise.abs_`, ``max`` -> ``max_``,
``min`` -> ``min_``.

The typed table gives well-formed generation and canonical printing, and it marks
the utilities that the miner cannot use.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from alphaforge.ops import cs, elementwise, ts

logger = logging.getLogger("alphaforge.ops.registry")

__all__ = [
    "OpSpec",
    "REGISTRY",
    "ARG_TYPES",
    "CATEGORIES",
    "register",
    "get_op",
    "list_ops",
    "register_all_builtin",
]

#: Allowed entries of :attr:`OpSpec.arg_types`.
ARG_TYPES: frozenset[str] = frozenset({"panel", "int", "float", "group"})
#: Allowed values of :attr:`OpSpec.category`.
CATEGORIES: frozenset[str] = frozenset({"elementwise", "ts", "cs", "group"})


@dataclass(frozen=True)
class OpSpec:
    """Immutable description of one operator.

    Attributes:
        name: Registry name (``"abs"``, ``"ts_corr"`` ...).
        func: The callable; positional arguments follow ``arg_types``.
        arg_types: One entry per positional argument, each in
            ``{"panel", "int", "float", "group"}``.
        category: ``"elementwise" | "ts" | "cs" | "group"``.
        int_choices: Values the miner samples for ``"int"`` slots.
        float_choices: Values the miner samples for ``"float"`` slots.
        commutative: ``True`` if swapping the two panel arguments is a no-op
            (lets the printer/miner canonicalise and de-duplicate).
        mineable: ``False`` => the miner must not use this op (utility only).
    """

    name: str
    func: Callable[..., pd.DataFrame] = field(repr=False)
    arg_types: tuple[str, ...]
    category: str
    int_choices: tuple[int, ...] = (3, 5, 10, 20, 40, 60)
    float_choices: tuple[float, ...] = (0.5, 2.0, 3.0)
    commutative: bool = False
    mineable: bool = True

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("OpSpec.name must be a non-empty string")
        if not callable(self.func):
            raise TypeError(f"OpSpec.func for {self.name!r} must be callable")
        bad = [t for t in self.arg_types if t not in ARG_TYPES]
        if bad:
            raise ValueError(f"OpSpec {self.name!r}: unknown arg types {bad}; allowed {sorted(ARG_TYPES)}")
        if self.category not in CATEGORIES:
            raise ValueError(f"OpSpec {self.name!r}: unknown category {self.category!r}; allowed {sorted(CATEGORIES)}")
        # dataclass(frozen=True) still lets us normalise tuples via object.__setattr__
        object.__setattr__(self, "arg_types", tuple(self.arg_types))
        object.__setattr__(self, "int_choices", tuple(int(v) for v in self.int_choices))
        object.__setattr__(self, "float_choices", tuple(float(v) for v in self.float_choices))
        if self.commutative and not (len(self.arg_types) == 2 and self.arg_types[0] == self.arg_types[1] == "panel"):
            raise ValueError(f"OpSpec {self.name!r}: commutative ops must take exactly (panel, panel)")

    @property
    def arity(self) -> int:
        """Number of positional arguments."""
        return len(self.arg_types)

    @property
    def n_panel_args(self) -> int:
        """Number of ``"panel"``/``"group"`` arguments (tree children in an expression)."""
        return sum(t in ("panel", "group") for t in self.arg_types)

    def __call__(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Apply the operator (convenience: ``get_op("ts_mean")(x, 5)``)."""
        return self.func(*args, **kwargs)


#: Global name -> :class:`OpSpec` table.  Populated by :func:`register_all_builtin`.
REGISTRY: dict[str, OpSpec] = {}

_BUILTIN_DONE: bool = False


def register(spec: OpSpec, *, overwrite: bool = True) -> None:
    """Add ``spec`` to :data:`REGISTRY`.

    Args:
        spec: The operator description.
        overwrite: If ``False``, raise ``ValueError`` when the name already exists.
    """
    if not isinstance(spec, OpSpec):
        raise TypeError(f"register() expects an OpSpec, got {type(spec).__name__}")
    if spec.name in REGISTRY:
        if not overwrite:
            raise ValueError(f"operator {spec.name!r} is already registered")
        if REGISTRY[spec.name] != spec:
            logger.info("operator %r re-registered with a different spec", spec.name)
    REGISTRY[spec.name] = spec
    logger.debug("registered operator %r %s", spec.name, spec.arg_types)


def get_op(name: str) -> OpSpec:
    """Look up an operator by registry name.

    Raises:
        KeyError: if ``name`` is unknown (the message lists the closest names).
    """
    try:
        return REGISTRY[name]
    except KeyError:
        hint = name.rstrip("_") if isinstance(name, str) else name
        if hint in REGISTRY:
            raise KeyError(
                f"unknown operator {name!r}; did you mean {hint!r}? (registry names drop the trailing underscore)"
            ) from None
        raise KeyError(f"unknown operator {name!r}; known operators: {sorted(REGISTRY)}") from None


def list_ops(category: str | None = None, mineable_only: bool = False) -> list[OpSpec]:
    """Return registered operators sorted by name, optionally filtered.

    Args:
        category: Keep only this category (``None`` = all).
        mineable_only: Drop operators with ``mineable=False``.

    Raises:
        ValueError: on an unknown ``category``.
    """
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; allowed {sorted(CATEGORIES)}")
    specs = [
        s
        for s in REGISTRY.values()
        if (category is None or s.category == category) and (not mineable_only or s.mineable)
    ]
    return sorted(specs, key=lambda s: s.name)


# ----------------------------------------------------------------------------
# built-in operator table
# ----------------------------------------------------------------------------

_P, _PP, _PI, _PPI, _PF = (
    ("panel",),
    ("panel", "panel"),
    ("panel", "int"),
    ("panel", "panel", "int"),
    ("panel", "float"),
)
_LAG_CHOICES = (1, 2, 3, 5, 10, 20, 40, 60)  # lags/differences are useful at 1-2 days too


def _builtin_specs() -> list[OpSpec]:
    """The complete built-in operator table (45 operators)."""
    ew, t, c = elementwise, ts, cs
    specs: list[OpSpec] = [
        # ---- elementwise -------------------------------------------------
        OpSpec("add", ew.add, _PP, "elementwise", commutative=True),
        OpSpec("sub", ew.sub, _PP, "elementwise"),
        OpSpec("mul", ew.mul, _PP, "elementwise", commutative=True),
        OpSpec("div", ew.div, _PP, "elementwise"),
        OpSpec("neg", ew.neg, _P, "elementwise"),
        OpSpec("abs", ew.abs_, _P, "elementwise"),
        OpSpec("sign", ew.sign, _P, "elementwise"),
        OpSpec("log", ew.log, _P, "elementwise"),
        OpSpec("signed_log", ew.signed_log, _P, "elementwise"),
        OpSpec("sqrt", ew.sqrt, _P, "elementwise"),
        OpSpec("signed_sqrt", ew.signed_sqrt, _P, "elementwise"),
        OpSpec("power", ew.power, _PF, "elementwise", float_choices=(0.5, 2.0, 3.0)),
        OpSpec("inverse", ew.inverse, _P, "elementwise"),
        OpSpec("max", ew.max_, _PP, "elementwise", commutative=True),
        OpSpec("min", ew.min_, _PP, "elementwise", commutative=True),
        OpSpec("clip", ew.clip, ("panel", "float", "float"), "elementwise", mineable=False),
        # ---- time-series -------------------------------------------------
        OpSpec("ts_delay", t.ts_delay, _PI, "ts", int_choices=_LAG_CHOICES),
        OpSpec("ts_delta", t.ts_delta, _PI, "ts", int_choices=_LAG_CHOICES),
        OpSpec("ts_pct_change", t.ts_pct_change, _PI, "ts", int_choices=_LAG_CHOICES),
        OpSpec("ts_mean", t.ts_mean, _PI, "ts"),
        OpSpec("ts_sum", t.ts_sum, _PI, "ts"),
        OpSpec("ts_std", t.ts_std, _PI, "ts"),
        OpSpec("ts_min", t.ts_min, _PI, "ts"),
        OpSpec("ts_max", t.ts_max, _PI, "ts"),
        OpSpec("ts_argmax", t.ts_argmax, _PI, "ts"),
        OpSpec("ts_argmin", t.ts_argmin, _PI, "ts"),
        OpSpec("ts_rank", t.ts_rank, _PI, "ts"),
        OpSpec("ts_zscore", t.ts_zscore, _PI, "ts"),
        OpSpec("ts_corr", t.ts_corr, _PPI, "ts"),
        OpSpec("ts_cov", t.ts_cov, _PPI, "ts"),
        OpSpec("ts_decay_linear", t.ts_decay_linear, _PI, "ts"),
        OpSpec("ts_skew", t.ts_skew, _PI, "ts"),
        OpSpec("ts_kurt", t.ts_kurt, _PI, "ts"),
        OpSpec("ts_product", t.ts_product, _PI, "ts"),
        # ---- cross-sectional ---------------------------------------------
        OpSpec("cs_rank", c.cs_rank, _P, "cs"),
        OpSpec("cs_zscore", c.cs_zscore, _P, "cs"),
        OpSpec("cs_demean", c.cs_demean, _P, "cs"),
        OpSpec("cs_scale", c.cs_scale, _P, "cs"),
        OpSpec("cs_winsorize", c.cs_winsorize, _PF, "cs", float_choices=(2.0, 3.0, 5.0)),
        OpSpec("cs_quantile_clip", c.cs_quantile_clip, ("panel", "float", "float"), "cs", mineable=False),
        OpSpec("cs_regress_out", c.cs_regress_out, _PP, "cs", mineable=False),
        OpSpec("cs_neutralize", c.cs_neutralize, ("panel", "group", "panel"), "cs", mineable=False),
        # ---- group ---------------------------------------------------------
        OpSpec("group_neutralize", c.group_neutralize, ("panel", "group"), "group"),
        OpSpec("group_rank", c.group_rank, ("panel", "group"), "group"),
        OpSpec("group_zscore", c.group_zscore, ("panel", "group"), "group"),
    ]
    return specs


def register_all_builtin(*, force: bool = False) -> None:
    """Register every built-in operator.  Idempotent: a second call is a no-op
    unless ``force=True`` (which re-registers the built-ins, restoring any that
    were overridden or removed).

    Called automatically by ``import alphaforge.ops``.
    """
    global _BUILTIN_DONE
    if _BUILTIN_DONE and not force:
        return
    specs = _builtin_specs()
    for spec in specs:
        register(spec, overwrite=True)
    _BUILTIN_DONE = True
    logger.debug("registered %d built-in operators", len(specs))
