"""Expression evaluator with an LRU sub-tree cache.

:class:`Evaluator` turns a tree (or its text) into a panel by looking fields up in a
:class:`~alphaforge.data.store.DataStore` and applying registry operators bottom-up.
Every :class:`~alphaforge.expr.ast.Call` sub-tree is cached under its canonical string
(:func:`~alphaforge.expr.printer.to_string`) in a bounded LRU, so
``cs_rank(ts_mean(close, 20))`` evaluated after ``ts_mean(close, 20)`` costs one op.

The cache keys are canonical strings, so a sub-tree that many candidates share (for
example ``ts_mean(close, 20)``) is computed once.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import pandas as pd

from alphaforge.expr.ast import REGISTRY, Call, Const, ExprValidationError, Field, Node, validate
from alphaforge.expr.parser import parse
from alphaforge.expr.printer import to_string

logger = logging.getLogger("alphaforge.expr.evaluator")

__all__ = ["Evaluator", "ExprEvalError"]


class ExprEvalError(RuntimeError):
    """Evaluation failed.  ``node``/``expr`` identify the failing sub-tree.

    Raised for every failure inside :meth:`Evaluator.eval` - an ill-formed tree
    (unknown operator, wrong arity, constant in a panel slot ...), an unknown field, or
    an exception thrown by an operator - so callers have one type to catch.  The
    original exception is chained as ``__cause__``.
    """

    def __init__(self, message: str, node: Node | None = None) -> None:
        super().__init__(message)
        self.node = node
        self.expr: str | None = None if node is None else to_string(node)


class Evaluator:
    """Evaluate expression trees against a store with an LRU cache of sub-tree panels.

    Args:
        store: a :class:`~alphaforge.data.store.DataStore` (anything with ``.get(name)``
            raising ``KeyError`` for unknown names works; a plain ``Mapping`` of panels is
            accepted too, mainly for tests).
        cache_size: maximum number of cached ``Call`` sub-tree panels (``0`` disables
            caching; the ``stats`` counters still work).
        registry: operator table (``name -> OpSpec``); defaults to the global registry.

    Attributes:
        stats: ``{"hits": int, "misses": int}`` counted over ``Call`` sub-trees (field
            look-ups are not counted - they are free).  Cumulative until :meth:`reset_stats`.

    Notes:
        * **No copies are made.**  A ``Field`` evaluates to the store's own panel object and
          cached panels are returned as-is on later hits, exactly like ``DataStore.get``.
          Treat every returned frame as read-only (operators never mutate inputs).
          Copy before in-place edits.
        * The cache key is the canonical string, so it is only valid for one store.  If
          you overwrite a field in the store (``store.add(..., overwrite=True)``) call
          :meth:`clear_cache`.
        * Evaluation validates the whole tree first (:func:`~alphaforge.expr.ast.validate`
          with ``group_fields=None`` - any field may serve as a group id at run time) and
          then evaluates bottom-up; a failure anywhere raises :class:`ExprEvalError`.
    """

    def __init__(self, store: Any, cache_size: int = 256, *, registry: Mapping[str, Any] = REGISTRY) -> None:
        cache_size = int(cache_size)
        if cache_size < 0:
            raise ValueError(f"cache_size must be >= 0, got {cache_size}")
        self.store = store
        self.cache_size = cache_size
        self.registry = registry
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.stats: dict[str, int] = {"hits": 0, "misses": 0}

    # ------------------------------------------------------------------ public
    def eval(self, node: Node | str) -> pd.DataFrame:
        """Evaluate ``node`` (a tree or expression text) to a ``dates x assets`` panel.

        Args:
            node: :class:`~alphaforge.expr.ast.Node`, or text passed through
                :func:`~alphaforge.expr.parser.parse`.

        Returns:
            The resulting panel (shared object - see the class notes).

        Raises:
            ExprSyntaxError: if ``node`` is text that does not parse.
            ExprEvalError: for any evaluation failure (bad structure, unknown field or
                operator, constant where a panel is expected, operator exception).
            TypeError: if ``node`` is neither a ``Node`` nor a ``str``.
        """
        if isinstance(node, str):
            node = parse(node)
        elif not isinstance(node, Node):
            raise TypeError(f"eval() expects a Node or str, got {type(node).__name__}")
        try:
            validate(node, self.registry, None, group_fields=None, require_panel=True)
        except ExprValidationError as exc:
            raise ExprEvalError(f"invalid expression: {exc}", exc.node if exc.node is not None else node) from exc
        return self._eval(node)

    def clear_cache(self) -> None:
        """Drop every cached panel (``stats`` are kept; see :meth:`reset_stats`)."""
        self._cache.clear()

    def reset_stats(self) -> None:
        """Zero the ``hits`` / ``misses`` counters."""
        self.stats["hits"] = 0
        self.stats["misses"] = 0

    @property
    def cache_len(self) -> int:
        """Number of cached sub-tree panels."""
        return len(self._cache)

    def cached_expressions(self) -> list[str]:
        """Canonical strings currently cached, least-recently used first."""
        return list(self._cache)

    def __repr__(self) -> str:
        return (
            f"Evaluator(cache={len(self._cache)}/{self.cache_size}, "
            f"hits={self.stats['hits']}, misses={self.stats['misses']})"
        )

    # ------------------------------------------------------------------ internals
    def _lookup_field(self, node: Field) -> pd.DataFrame:
        try:
            panel = self.store[node.name] if isinstance(self.store, Mapping) else self.store.get(node.name)
        except KeyError as exc:
            raise ExprEvalError(f"unknown field {node.name!r}: {exc.args[0] if exc.args else exc}", node) from exc
        if not isinstance(panel, pd.DataFrame):
            raise ExprEvalError(f"field {node.name!r} is not a DataFrame panel ({type(panel).__name__})", node)
        return panel

    def _eval(self, node: Node) -> pd.DataFrame:
        if isinstance(node, Field):
            return self._lookup_field(node)
        if isinstance(node, Const):
            # Unreachable after validate(); kept as a guard for direct _eval callers.
            raise ExprEvalError(f"constant {node!s} used where a panel is expected", node)
        if not isinstance(node, Call):
            raise ExprEvalError(f"unknown node type {type(node).__name__}")
        key = to_string(node)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.stats["hits"] += 1
            return cached
        self.stats["misses"] += 1
        result = self._apply(node, key)
        if self.cache_size > 0:
            self._cache[key] = result
            while len(self._cache) > self.cache_size:
                evicted, _ = self._cache.popitem(last=False)
                logger.debug("cache evicted %s", evicted)
        return result

    def _apply(self, node: Call, key: str) -> pd.DataFrame:
        spec = self.registry[node.op]  # validated: exists with matching arity
        call_args: list[Any] = []
        for arg, typ in zip(node.args, spec.arg_types):
            if typ == "int":
                call_args.append(int(arg.value))  # type: ignore[union-attr]
            elif typ == "float":
                call_args.append(float(arg.value))  # type: ignore[union-attr]
            else:  # "panel" / "group"
                call_args.append(self._eval(arg))
        logger.debug("evaluating %s", key)
        try:
            out = spec.func(*call_args)
        except Exception as exc:  # noqa: BLE001 - deliberately wrap everything the op raises
            raise ExprEvalError(f"operator {node.op!r} failed in {key}: {type(exc).__name__}: {exc}", node) from exc
        if not isinstance(out, pd.DataFrame):
            raise ExprEvalError(f"operator {node.op!r} returned {type(out).__name__}, not a DataFrame, in {key}", node)
        self._check_shape(out, node, key)
        return out

    def _check_shape(self, out: pd.DataFrame, node: Call, key: str) -> None:
        """Guard against an operator silently changing the grid (only when the store has one)."""
        dates = getattr(self.store, "dates", None)
        assets = getattr(self.store, "assets", None)
        if dates is None or assets is None:
            return
        if out.shape != (len(dates), len(assets)):
            raise ExprEvalError(
                f"operator {node.op!r} returned shape {out.shape}, expected {(len(dates), len(assets))}, in {key}",
                node,
            )
