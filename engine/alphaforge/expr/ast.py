"""Expression AST for the AlphaForge factor language.

An alpha expression such as ``cs_rank(div(ts_delta(close, 5), close))`` is represented as
an immutable tree of three node kinds:

* :class:`Field`  - a leaf naming a :class:`~alphaforge.data.store.DataStore` panel
  (``close``, ``volume``, ``industry`` ...).
* :class:`Const`  - a numeric leaf (``5``, ``0.5``).  Always stored as ``float``; the printer
  emits it as an ``int`` when the value is integral.
* :class:`Call`   - an operator application ``op(arg, ...)`` where ``op`` is a name in the
  operator registry (:data:`alphaforge.ops.REGISTRY`).

All nodes are frozen dataclasses: hashable, comparable by value, and safe to share
between trees (``replace_subtree`` builds new trees without copying untouched branches).
``str(node)`` is the canonical prefix form produced by :func:`alphaforge.expr.printer.to_string`.

Expressions are trees, so the miner can change and combine sub-trees, the cache can share
panels, and the library can find duplicates by canonical string.
"""

from __future__ import annotations

import logging
import math
import numbers
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from alphaforge.ops import REGISTRY  # importing alphaforge.ops registers every built-in operator

logger = logging.getLogger("alphaforge.expr.ast")

__all__ = [
    "Node",
    "Field",
    "Const",
    "Call",
    "Path",
    "IDENT_RE",
    "DEFAULT_GROUP_FIELDS",
    "SCALAR_SLOTS",
    "PANEL_SLOTS",
    "ExprValidationError",
    "depth",
    "size",
    "iter_nodes",
    "get_subtree",
    "replace_subtree",
    "fields_used",
    "validate",
]

#: Regular expression every field and operator name must match (so that the printed
#: form is always re-parseable: ``parse(to_string(n)) == n``).
IDENT_RE: re.Pattern[str] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Field names accepted in a ``"group"`` argument slot by :func:`validate` by default.
DEFAULT_GROUP_FIELDS: frozenset[str] = frozenset({"industry"})

#: Registry slot types that must be filled by a :class:`Const`.
SCALAR_SLOTS: frozenset[str] = frozenset({"int", "float"})
#: Registry slot types that must be filled by a :class:`Field` or :class:`Call`.
PANEL_SLOTS: frozenset[str] = frozenset({"panel", "group"})

#: A path from the root to a node: the sequence of child indices (``()`` is the root).
Path = tuple[int, ...]


class ExprValidationError(ValueError):
    """Raised by :func:`validate` for an unknown operator, wrong arity or a type mismatch.

    Subclasses ``ValueError``; ``node`` is the offending subtree.
    """

    def __init__(self, message: str, node: Node | None = None) -> None:
        super().__init__(message)
        self.node = node


# ----------------------------------------------------------------------------
# node classes
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    """Base class of every expression node.  Immutable, hashable, value-comparable."""

    def __str__(self) -> str:
        # Lazy import: the printer imports this module.
        from alphaforge.expr.printer import to_string

        return to_string(self)


@dataclass(frozen=True)
class Field(Node):
    """Leaf referring to a DataStore panel by name (``close``, ``industry`` ...).

    Args:
        name: identifier-shaped field name (``[A-Za-z_][A-Za-z0-9_]*``).

    Raises:
        ValueError: if ``name`` is not identifier-shaped (it could not be printed and
            parsed back, which would break the canonical-string contract).
    """

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not IDENT_RE.fullmatch(self.name):
            raise ValueError(f"Field name must match {IDENT_RE.pattern!r}, got {self.name!r}")


@dataclass(frozen=True)
class Const(Node):
    """Numeric leaf.  The value is always stored as a finite ``float``.

    ``Const(5)`` and ``Const(5.0)`` are equal and hash identically; the printer emits
    integral values without a decimal point (``5``), so ``ts_mean(close, 5)`` round-trips.

    Raises:
        TypeError: for non-numeric values (``bool`` is rejected deliberately).
        ValueError: for ``NaN`` / ``±inf`` (they have no literal in the grammar).
    """

    value: float

    def __post_init__(self) -> None:
        v = self.value
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise TypeError(f"Const value must be a real number, got {type(v).__name__}")
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError(f"Const value must be finite, got {v!r}")
        object.__setattr__(self, "value", fv)

    @property
    def is_integral(self) -> bool:
        """``True`` when the value has no fractional part (printed as an ``int``)."""
        return self.value.is_integer()


@dataclass(frozen=True)
class Call(Node):
    """Operator application ``op(*args)``.

    Args:
        op: registry name of the operator (``"ts_mean"``, ``"add"`` ...).
        args: child nodes in positional order (a list is converted to a tuple).

    Construction is deliberately *structural*: the operator is not looked up and arity is
    not checked here (the parser must be able to build a tree before it is validated).
    Use :func:`validate` for that.
    """

    op: str
    args: tuple[Node, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.op, str) or not IDENT_RE.fullmatch(self.op):
            raise ValueError(f"Call op must match {IDENT_RE.pattern!r}, got {self.op!r}")
        if isinstance(self.args, Node) or not isinstance(self.args, (tuple, list)):
            raise TypeError(f"Call args must be a tuple of Node, got {type(self.args).__name__}")
        args = tuple(self.args)
        for i, a in enumerate(args):
            if not isinstance(a, Node):
                raise TypeError(f"Call {self.op!r}: argument {i} is not a Node ({type(a).__name__})")
        object.__setattr__(self, "args", args)


def _children(node: Node) -> tuple[Node, ...]:
    return node.args if isinstance(node, Call) else ()


def _check_node(node: Any) -> Node:
    if not isinstance(node, Node):
        raise TypeError(f"expected an expression Node, got {type(node).__name__}")
    return node


# ----------------------------------------------------------------------------
# tree utilities
# ----------------------------------------------------------------------------
def depth(node: Node) -> int:
    """Height of the tree: leaves (``Field``/``Const``) are 1, a ``Call`` is 1 + max child depth.

    A zero-argument call also has depth 1.

    The miner limits the depth (``max_depth``), because deep trees overfit and
    are hard to read.
    """
    _check_node(node)
    kids = _children(node)
    return 1 + max((depth(k) for k in kids), default=0)


def size(node: Node) -> int:
    """Number of nodes in the tree (used as the parsimony penalty by the miner)."""
    _check_node(node)
    return 1 + sum(size(k) for k in _children(node))


def iter_nodes(node: Node) -> Iterator[tuple[Node, Path]]:
    """Yield ``(node, path)`` pairs in pre-order (root first, children left to right).

    ``path`` is the tuple of child indices from the root, so
    ``get_subtree(root, path) is node`` for every pair yielded.
    """
    _check_node(node)
    stack: list[tuple[Node, Path]] = [(node, ())]
    while stack:
        cur, path = stack.pop()
        yield cur, path
        kids = _children(cur)
        # push right-to-left so the leftmost child is popped (and yielded) first
        for i in range(len(kids) - 1, -1, -1):
            stack.append((kids[i], path + (i,)))


def get_subtree(node: Node, path: Iterable[int]) -> Node:
    """Return the subtree at ``path`` (a sequence of child indices; ``()`` is the root).

    Raises:
        IndexError: if the path leaves the tree (index out of range or descending into a leaf).
    """
    cur = _check_node(node)
    path = tuple(path)
    for k, i in enumerate(path):
        kids = _children(cur)
        if not kids:
            raise IndexError(f"path {path!r} descends into leaf {cur!s} at step {k}")
        if not (-len(kids) <= i < len(kids)):
            raise IndexError(f"path {path!r}: child index {i} out of range for {cur.op}/{len(kids)} at step {k}")
        cur = kids[i]
    return cur


def replace_subtree(node: Node, path: Iterable[int], new: Node) -> Node:
    """Return a copy of ``node`` with the subtree at ``path`` replaced by ``new``.

    The original tree is untouched (nodes are immutable); branches not on ``path`` are
    shared with the result rather than copied.  ``replace_subtree(n, (), new)`` is ``new``.

    Raises:
        IndexError: if ``path`` does not exist in the tree.
        TypeError: if ``new`` is not a :class:`Node`.
    """
    _check_node(node)
    _check_node(new)
    path = tuple(path)

    def _rebuild(cur: Node, k: int) -> Node:
        if k == len(path):
            return new
        kids = _children(cur)
        i = path[k]
        if not kids:
            raise IndexError(f"path {path!r} descends into leaf {cur!s} at step {k}")
        if not (-len(kids) <= i < len(kids)):
            raise IndexError(f"path {path!r}: child index {i} out of range for {cur.op}/{len(kids)} at step {k}")
        i %= len(kids)
        new_kids = kids[:i] + (_rebuild(kids[i], k + 1),) + kids[i + 1 :]
        return Call(cur.op, new_kids)  # type: ignore[union-attr]  (cur is a Call when kids is non-empty)

    return _rebuild(node, 0)


def fields_used(node: Node) -> set[str]:
    """Set of DataStore field names referenced anywhere in the tree."""
    return {n.name for n, _ in iter_nodes(node) if isinstance(n, Field)}


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def validate(
    node: Node,
    registry: Mapping[str, Any] = REGISTRY,
    fields: set[str] | None = None,
    *,
    group_fields: Iterable[str] | None = DEFAULT_GROUP_FIELDS,
    require_panel: bool = False,
) -> None:
    """Check a tree against the operator registry; raise ``ValueError`` if it is ill-formed.

    Rules:

    * every ``Call.op`` must be a registry name and ``len(args)`` must equal its arity;
    * an ``"int"`` slot must hold a :class:`Const` with an integral value ``>= 1``
      (every ``int`` slot in the built-in registry is a window / lag);
    * a ``"float"`` slot must hold a :class:`Const` (any finite value);
    * a ``"panel"`` slot must hold a :class:`Field` or :class:`Call` - a bare number where
      a panel is expected (``close * 2``) is an error;
    * a ``"group"`` slot must hold a :class:`Call` (accepted unconditionally - the
      grouping is computed) or a :class:`Field` whose name is in ``group_fields``
      (default ``{"industry"}``); pass ``group_fields=None`` to accept any field there.
      Group-id panels are ordinary panels, so a group field is also accepted
      in a ``"panel"`` slot.
    * if ``fields`` is given, every :class:`Field` name must be in it.

    Args:
        node: tree to check.
        registry: mapping ``op name -> OpSpec`` (anything exposing ``arg_types``).
        fields: if given, the allowed field names (e.g. ``set(store.fields)``).
        group_fields: field names allowed in a ``"group"`` slot (``None`` = any field).
        require_panel: if ``True``, additionally reject a bare :class:`Const` root (a
            constant is a valid tree but not a panel expression; the evaluator uses this).

    Raises:
        ExprValidationError: (a ``ValueError``) describing the first problem found,
            with the offending subtree in ``.node``.
        TypeError: if ``node`` is not a :class:`Node`.

    The type check runs before any data computation, so a malformed candidate fails
    at once and costs no panel computation.
    """
    _check_node(node)
    field_set: set[str] | None = None if fields is None else set(fields)
    group_set: frozenset[str] | None = None if group_fields is None else frozenset(group_fields)
    if require_panel and isinstance(node, Const):
        raise ExprValidationError(f"expression {node!s} is a bare constant, not a panel expression", node)
    _validate(node, None, registry, field_set, group_set, None)


def _validate(
    node: Node,
    slot: str | None,
    registry: Mapping[str, Any],
    field_set: set[str] | None,
    group_set: frozenset[str] | None,
    parent: Call | None,
) -> None:
    where = "" if parent is None else f" in {parent!s}"
    if isinstance(node, Field):
        if slot in SCALAR_SLOTS:
            raise ExprValidationError(f"field {node.name!r} used where an {slot} constant is expected{where}", node)
        if field_set is not None and node.name not in field_set:
            raise ExprValidationError(f"unknown field {node.name!r}{where}; known fields: {sorted(field_set)}", node)
        if slot == "group" and group_set is not None and node.name not in group_set:
            raise ExprValidationError(
                f"field {node.name!r} used in a group slot{where}; allowed group fields: {sorted(group_set)}", node
            )
        return
    if isinstance(node, Const):
        if slot in PANEL_SLOTS:
            raise ExprValidationError(f"constant {node!s} used where a {slot} is expected{where}", node)
        if slot == "int" and not (node.value.is_integer() and node.value >= 1):
            raise ExprValidationError(f"int slot requires an integer >= 1, got {node!s}{where}", node)
        return
    if isinstance(node, Call):
        if slot in SCALAR_SLOTS:
            raise ExprValidationError(f"call {node!s} used where an {slot} constant is expected{where}", node)
        try:
            spec = registry[node.op]
        except KeyError:
            raise ExprValidationError(f"unknown operator {node.op!r} in {node!s}", node) from None
        arg_types = tuple(spec.arg_types)
        if len(node.args) != len(arg_types):
            raise ExprValidationError(
                f"operator {node.op!r} expects {len(arg_types)} argument(s) {arg_types}, "
                f"got {len(node.args)} in {node!s}",
                node,
            )
        for arg, typ in zip(node.args, arg_types):
            _validate(arg, typ, registry, field_set, group_set, node)
        return
    raise TypeError(f"unsupported node type {type(node).__name__}")
