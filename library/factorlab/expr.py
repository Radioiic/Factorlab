"""Safe parser / evaluator for factor expressions.

Expressions are a small subset of Python syntax::

    ts_zscore(log(close / delay(close, 5)), 20) * -1

Only numeric constants, field names, registered operators, unary +/-, and
binary + - * / ** are allowed -- nothing is ever passed to ``eval``.

Limits: an expression has at most MAX_EXPRESSION_LENGTH characters and at most
MAX_EXPRESSION_DEPTH nested nodes. A power of two constants must have an
exponent of at most MAX_CONSTANT_EXPONENT in magnitude and a finite result.
An expression outside these limits gives ExpressionError.
"""

from __future__ import annotations

import ast
import math
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import pandas as pd

from .fields import DERIVED_BY_NAME, RAW_FIELD_NAMES
from .operators import OPERATORS, div

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: div,
    ast.Pow: lambda a, b: a**b,
}


# The longest catalog expression has 608 characters and 17 nested nodes. These limits
# are much larger, and they keep the parser and the evaluator far from the recursion limit.
MAX_EXPRESSION_LENGTH = 4_000
MAX_EXPRESSION_DEPTH = 64
# A power of two constants is computed exactly (Python int), so a large exponent can use
# much CPU time and memory.
MAX_CONSTANT_EXPONENT = 100


class ExpressionError(ValueError):
    """An expression is not valid: a syntax error, a name or an operation that is not allowed, or a limit.

    The message is plain text for the user: it names no class of the Python parser."""


# The plain names of the Python syntax that an expression cannot use (the message of ExpressionError)
_SYNTAX_WORDS = {
    ast.Compare: "a comparison (<, >, ==)",
    ast.BoolOp: "and / or",
    ast.IfExp: "if / else",
    ast.Subscript: "an index in square brackets",
    ast.Attribute: "a dot (.)",
    ast.Lambda: "lambda",
    ast.List: "a list",
    ast.Tuple: "a list",
    ast.Set: "braces ({})",
    ast.Dict: "braces ({})",
    ast.Starred: "a star (*)",
    ast.JoinedStr: "text in quotation marks",
    ast.NamedExpr: "an assignment (:=)",
    ast.ListComp: "a loop",
    ast.SetComp: "a loop",
    ast.DictComp: "a loop",
    ast.GeneratorExp: "a loop",
}
_OPERATOR_SIGNS = {
    ast.Not: "not",
    ast.Invert: "~",
    ast.Mod: "%",
    ast.FloorDiv: "//",
    ast.MatMult: "@",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
}


def _syntax_problem(exc: SyntaxError, expression: str) -> str:
    """The plain text of a syntax error: what is wrong and where, without the hints of the parser."""
    msg = exc.msg or ""
    if not expression.strip():
        return "the expression is empty"
    if "never closed" in msg:
        what = "a bracket is not closed"
    elif msg.startswith("unmatched") or "does not match" in msg:
        what = "a closing bracket has no opening bracket"
    elif "unterminated" in msg:
        what = "a quotation mark is not closed"
    elif "null" in msg:
        what = "the expression contains a null character"
    else:
        what = "the syntax is not correct"
    if exc.lineno and exc.lineno > 1:
        return f"{what} (line {exc.lineno}, column {exc.offset})"
    if exc.offset:
        return f"{what} (column {exc.offset})"
    return what


@lru_cache(maxsize=200_000)
def parse(expression: str) -> ast.AST:
    """Check ``expression`` and return its syntax tree (the body of a Python ``eval`` tree).

    An expression that is not valid, or that is outside the limits, gives ExpressionError.
    The result is cached, so the same text gives the same tree object.
    """
    if not isinstance(expression, str):
        raise ExpressionError("an expression must be a string")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"expression has {len(expression)} characters (limit {MAX_EXPRESSION_LENGTH})")
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        # what is wrong and where, without the parser's file name "<unknown>" and its hints
        raise ExpressionError(f"syntax error in {expression!r}: {_syntax_problem(exc, expression)}") from None
    except ValueError:
        raise ExpressionError(f"the expression contains a character that is not allowed: {expression!r}") from None
    except (RecursionError, MemoryError):
        raise ExpressionError(f"the expression is too deep or too long to parse: {expression!r}") from None
    depth = _depth(tree)
    if depth > MAX_EXPRESSION_DEPTH:
        raise ExpressionError(f"expression has {depth} nested nodes (limit {MAX_EXPRESSION_DEPTH})")
    try:
        _validate(tree, expression)
    except RecursionError:  # pragma: no cover - the depth limit comes first
        raise ExpressionError(f"expression is too deep: {expression!r}") from None
    return tree


def _depth(tree: ast.AST) -> int:
    """The number of nested expression nodes. Iterative, so a deep tree cannot cause a RecursionError."""
    best, stack = 0, [(tree, 1)]
    while stack:
        node, d = stack.pop()
        if isinstance(node, ast.expr):
            best = max(best, d)
            stack.extend((c, d + 1) for c in ast.iter_child_nodes(node))
    return best


def _constant_value(node: ast.AST) -> float | None:
    """The float value of a subtree of constants only, or None when the subtree reads data.

    Only for the limit check of constant powers: the evaluator computes these values with
    Python numbers. An overflow gives inf, and a division by zero gives nan (as ``div``)."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            return None  # _validate refuses this constant
        try:
            return float(node.value)
        except OverflowError:
            return math.inf
    if isinstance(node, ast.UnaryOp):
        v = _constant_value(node.operand)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.BinOp):
        a, b = _constant_value(node.left), _constant_value(node.right)
        if a is None or b is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b if b != 0 else math.nan
            if isinstance(node.op, ast.Pow):
                return abs(a) ** b
        except (OverflowError, ZeroDivisionError):
            return math.inf
    return None


def _check_constant_power(node: ast.BinOp, expression: str) -> None:
    """Refuse a power of two constants with a large exponent or a result that is not finite."""
    base, exponent = _constant_value(node.left), _constant_value(node.right)
    if base is None or exponent is None:
        return
    if not math.isfinite(exponent) or abs(exponent) > MAX_CONSTANT_EXPONENT:
        raise ExpressionError(
            f"constant power with an exponent above {MAX_CONSTANT_EXPONENT} in magnitude in {expression!r}"
        )
    try:
        value = abs(base) ** exponent
    except (OverflowError, ZeroDivisionError):
        value = math.inf
    if math.isinf(value):
        raise ExpressionError(f"constant power without a finite result in {expression!r}")


def _validate(node: ast.AST, expression: str) -> None:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ExpressionError(f"only numeric constants allowed in {expression!r}")
    elif isinstance(node, ast.Name):
        if node.id not in RAW_FIELD_NAMES and node.id not in DERIVED_BY_NAME:
            raise ExpressionError(f"unknown field {node.id!r} in {expression!r}")
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            sign = _OPERATOR_SIGNS.get(type(node.op), "this operator")
            raise ExpressionError(f"the operator {sign} is not allowed in {expression!r} (allowed: unary - and +)")
        _validate(node.operand, expression)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            sign = _OPERATOR_SIGNS.get(type(node.op), "this operator")
            raise ExpressionError(f"the operator {sign} is not allowed in {expression!r} (allowed: + - * / **)")
        if isinstance(node.op, ast.Pow):
            _check_constant_power(node, expression)
        _validate(node.left, expression)
        _validate(node.right, expression)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError(f"only the name of an operator can come before an argument list in {expression!r}")
        if node.func.id not in OPERATORS and node.func.id != "bench":
            raise ExpressionError(f"unknown operator {node.func.id!r} in {expression!r}")
        if node.keywords:
            raise ExpressionError(f"an argument with a name (name=value) is not allowed in {expression!r}")
        for a in node.args:
            _validate(a, expression)
    else:
        what = _SYNTAX_WORDS.get(type(node), "this syntax")
        raise ExpressionError(f"{what} is not allowed in {expression!r}")


def raw_dependencies(expression: str) -> frozenset[str]:
    """Raw data fields an expression needs (derived fields are expanded)."""
    return _deps(expression)


@lru_cache(maxsize=200_000)
def _deps(expression: str) -> frozenset[str]:
    out: set[str] = set()
    tree = parse(expression)
    funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) not in funcs:
            if node.id in RAW_FIELD_NAMES:
                out.add(node.id)
            else:
                out |= _deps(DERIVED_BY_NAME[node.id].expression)
    return frozenset(out)


@lru_cache(maxsize=200_000)
def max_lookback(expression: str) -> int:
    """Rough upper bound of bars of history the expression consumes."""

    def walk(node) -> int:
        if isinstance(node, ast.Name):
            if node.id in DERIVED_BY_NAME:
                return max_lookback(DERIVED_BY_NAME[node.id].expression)
            return 0
        if isinstance(node, ast.Call):
            inner = max((walk(a) for a in node.args), default=0)
            ints = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, int)]
            name = node.func.id if isinstance(node.func, ast.Name) else ""
            own = max(ints) if ints and name not in ("clip",) else 0
            return inner + own
        return max((walk(c) for c in ast.iter_child_nodes(node)), default=0)

    return walk(parse(expression))


class Evaluator:
    """Evaluate expressions against a dict of raw panels with sub-expression caching."""

    def __init__(self, panels: dict[str, pd.DataFrame], benchmark: str = "BTCUSDT", cache_size: int = 512):
        """Use ``panels`` (raw field name -> time x symbol frame) as the data.

        ``benchmark`` is the column that ``bench(...)`` reads; without that column the
        equal-weight mean is the benchmark. ``cache_size`` is the number of sub-expression
        results that the evaluator keeps. An empty ``panels`` gives ValueError.
        """
        if not panels:
            raise ValueError("no panels supplied")
        self.panels = panels
        self.benchmark = benchmark
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._cache_size = cache_size
        ref = next(iter(panels.values()))
        self.index, self.columns = ref.index, ref.columns

    def available(self, expression: str) -> bool:
        """Return True when the panels have every raw field that ``expression`` needs."""
        return raw_dependencies(expression) <= set(self.panels)

    def __call__(self, expression: str) -> pd.DataFrame:
        """Return the values of ``expression`` as a panel (time x symbol); infinite values become NaN."""
        out = self._eval(parse(expression))
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(float(out), index=self.index, columns=self.columns)
        return out.replace([np.inf, -np.inf], np.nan)

    # -- internals ---------------------------------------------------------
    def _eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        key = ast.dump(node)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        val = self._compute(node)
        if isinstance(val, pd.DataFrame):
            self._cache[key] = val
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return val

    def _compute(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.panels:
                return self.panels[node.id].astype(float)
            if node.id in DERIVED_BY_NAME:
                return self._eval(parse(DERIVED_BY_NAME[node.id].expression))
            raise ExpressionError(f"field {node.id!r} not available in data")
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.Call):
            args = [self._eval(a) for a in node.args]
            if node.func.id == "bench":
                return self._bench(args[0])
            return OPERATORS[node.func.id](*args)
        raise ExpressionError(f"cannot evaluate {ast.unparse(node)!r}")

    def _bench(self, x: pd.DataFrame) -> pd.DataFrame:
        # without the benchmark column, the equal-weight market is the benchmark
        s = x[self.benchmark] if self.benchmark in x.columns else x.mean(axis=1)
        return pd.DataFrame(
            np.repeat(s.to_numpy()[:, None], x.shape[1], axis=1), index=x.index, columns=x.columns
        ).where(x.notna())
