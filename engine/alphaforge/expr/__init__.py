"""``alphaforge.expr`` - the factor expression language.

* :mod:`~alphaforge.expr.ast` - immutable ``Field`` / ``Const`` / ``Call`` trees and
  tree utilities (``depth``, ``size``, ``iter_nodes``, ``get_subtree``,
  ``replace_subtree``, ``fields_used``, ``validate``).
* :mod:`~alphaforge.expr.parser` - ``parse(text)``: call syntax plus infix ``+ - * /``
  sugar, desugared to ``add/sub/mul/div/neg``.
* :mod:`~alphaforge.expr.printer` - ``to_string(node)``: canonical prefix form, with
  ``parse(to_string(n)) == n``.
* :mod:`~alphaforge.expr.evaluator` - ``Evaluator(store).eval(node_or_text)`` with an
  LRU sub-tree cache keyed by canonical string.

Example::

    >>> from alphaforge.expr import parse, to_string, Evaluator
    >>> tree = parse("cs_rank(ts_delta(close, 5) / close)")
    >>> to_string(tree)
    'cs_rank(div(ts_delta(close, 5), close))'
    >>> Evaluator(store).eval(tree)   # a dates x assets panel in (0, 1]
"""

from alphaforge.expr.ast import (
    DEFAULT_GROUP_FIELDS,
    IDENT_RE,
    PANEL_SLOTS,
    SCALAR_SLOTS,
    Call,
    Const,
    ExprValidationError,
    Field,
    Node,
    Path,
    depth,
    fields_used,
    get_subtree,
    iter_nodes,
    replace_subtree,
    size,
    validate,
)
from alphaforge.expr.evaluator import Evaluator, ExprEvalError
from alphaforge.expr.parser import ExprSyntaxError, Token, parse, tokenize
from alphaforge.expr.printer import format_const, to_string

__all__ = [
    # ast
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
    # parser
    "parse",
    "tokenize",
    "Token",
    "ExprSyntaxError",
    # printer
    "to_string",
    "format_const",
    # evaluator
    "Evaluator",
    "ExprEvalError",
]
