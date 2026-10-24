"""Canonical printer for expression trees.

:func:`to_string` renders a tree in *canonical prefix form* - ``op(arg, arg)`` with a
``", "`` separator, integral constants printed as ints - and guarantees that
``parse(to_string(node)) == node`` for every tree built from identifier-shaped names.
Infix sugar accepted by the parser (``a / b``) is never emitted: the canonical form of
``ts_delta(close, 5) / close`` is ``div(ts_delta(close, 5), close)``.

The canonical string of a tree is the cache key, the identity for de-duplication
and the stored text that re-creates the signal.
"""

from __future__ import annotations

import logging

from alphaforge.expr.ast import Call, Const, Field, Node

logger = logging.getLogger("alphaforge.expr.printer")

__all__ = ["to_string", "format_const"]

#: Integral magnitudes below this print as plain ints (``5``); larger ones fall back to
#: ``repr(float)`` (``1e+20``) so the text stays short and exactly re-parseable.
_INT_PRINT_LIMIT = 1e15


def format_const(value: float) -> str:
    """Shortest exactly-round-tripping literal for a :class:`Const` value.

    Integral values print without a decimal point (``5``, ``-2``); everything else uses
    Python's shortest-repr (``0.5``, ``1e-05``, ``0.3333333333333333``), which
    ``float()`` reads back bit-for-bit.
    """
    v = float(value)
    if v.is_integer() and abs(v) < _INT_PRINT_LIMIT:
        return str(int(v))
    return repr(v)


def to_string(node: Node) -> str:
    """Render ``node`` in canonical prefix form (``cs_rank(div(ts_delta(close, 5), close))``).

    Args:
        node: any :class:`~alphaforge.expr.ast.Node`.

    Returns:
        The canonical string.  ``Node.__str__`` delegates here.

    Raises:
        TypeError: for objects that are not expression nodes.
    """
    if isinstance(node, Field):
        return node.name
    if isinstance(node, Const):
        return format_const(node.value)
    if isinstance(node, Call):
        return f"{node.op}({', '.join(to_string(a) for a in node.args)})"
    raise TypeError(f"to_string expects an expression Node, got {type(node).__name__}")
