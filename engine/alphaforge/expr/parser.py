"""Parser for the AlphaForge expression language.

Grammar (case-sensitive identifiers)::

    expr    := term (("+" | "-") term)*          # left-associative -> add / sub
    term    := unary (("*" | "/") unary)*        # left-associative -> mul / div
    unary   := "-" unary | primary               # -x -> neg(x); -2 -> Const(-2)
    primary := NUMBER
             | IDENT "(" [expr ("," expr)*] ")"  # operator call
             | IDENT                             # DataStore field
             | "(" expr ")"

``NUMBER`` is ``\\d+(\\.\\d*)?`` / ``\\.\\d+`` with an optional exponent; ``IDENT`` is
``[A-Za-z_][A-Za-z0-9_]*``.  Infix operators are pure sugar and are desugared to the
registry names ``add`` / ``sub`` / ``mul`` / ``div`` / ``neg`` with the usual precedence
(``*`` ``/`` bind tighter than ``+`` ``-``; unary minus binds tightest), so
``parse("a / b") == Call("div", (Field("a"), Field("b")))`` and the tree is identical to
``parse("div(a, b)")``.  Unary minus applied to a numeric literal folds into the
constant (``-2`` is ``Const(-2.0)``, not ``neg(2)``), which is what the printer emits.

The parser is purely syntactic: it does not consult the operator registry, so an
unknown operator or a constant in a panel slot parses fine and is rejected later by
:func:`alphaforge.expr.ast.validate` / the evaluator.

The parser accepts infix and prefix text and keeps one canonical form, so the library
has no duplicates that differ only in spelling.

Limit: the parser nests at most :data:`MAX_DEPTH` levels (operator calls, parentheses
and unary minus signs), and the tree that it returns has a depth of at most
:data:`MAX_DEPTH`. A deeper text gives :class:`ExprSyntaxError`, not a ``RecursionError``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from alphaforge.expr.ast import Call, Const, Field, Node

logger = logging.getLogger("alphaforge.expr.parser")

__all__ = ["parse", "ExprSyntaxError", "tokenize", "Token", "MAX_DEPTH"]

# The miner makes trees of depth 4 or less by default (ResearchConfig.max_depth). The limit is much larger,
# and it keeps the recursive parser, printer and evaluator far from the Python recursion limit.
MAX_DEPTH = 100


class ExprSyntaxError(ValueError):
    """Malformed expression text.

    Attributes:
        pos: 0-based character offset of the problem in ``text``.
        text: the full source text.
    """

    def __init__(self, message: str, pos: int, text: str) -> None:
        self.pos = pos
        self.text = text
        snippet = text if len(text) <= 80 else text[:77] + "..."
        super().__init__(f"{message} at position {pos} in {snippet!r}")


@dataclass(frozen=True)
class Token:
    """One lexical token: ``kind`` in ``{"number", "ident", "op", "eof"}``."""

    kind: str
    text: str
    pos: int


_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>[-+*/(),])
    """,
    re.VERBOSE,
)

_ADDITIVE = {"+": "add", "-": "sub"}
_MULTIPLICATIVE = {"*": "mul", "/": "div"}


def tokenize(text: str) -> list[Token]:
    """Split ``text`` into tokens (whitespace dropped), ending with an ``"eof"`` token.

    Raises:
        ExprSyntaxError: on a character outside the grammar's alphabet.
    """
    tokens: list[Token] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise ExprSyntaxError(f"unexpected character {text[pos]!r}", pos, text)
        kind = m.lastgroup
        if kind != "ws":
            tokens.append(Token(kind, m.group(), pos))  # type: ignore[arg-type]
        pos = m.end()
    tokens.append(Token("eof", "", n))
    return tokens


class _Parser:
    """Recursive-descent parser over a token list (one instance per :func:`parse` call)."""

    def __init__(self, text: str, tokens: list[Token]) -> None:
        self.text = text
        self.tokens = tokens
        self.i = 0
        self.depth = 0

    # ---- token helpers ---------------------------------------------------
    def peek(self) -> Token:
        return self.tokens[self.i]

    def advance(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def at_op(self, *chars: str) -> bool:
        tok = self.peek()
        return tok.kind == "op" and tok.text in chars

    def expect_op(self, char: str, context: str) -> Token:
        tok = self.peek()
        if not self.at_op(char):
            raise ExprSyntaxError(f"expected {char!r} {context}, found {self._describe(tok)}", tok.pos, self.text)
        return self.advance()

    @staticmethod
    def _describe(tok: Token) -> str:
        return "end of input" if tok.kind == "eof" else f"{tok.text!r}"

    # ---- grammar ----------------------------------------------------------
    def parse_expr(self) -> Node:
        node = self.parse_term()
        while self.at_op("+", "-"):
            op = _ADDITIVE[self.advance().text]
            rhs = self.parse_term()
            node = Call(op, (node, rhs))
        return node

    def parse_term(self) -> Node:
        node = self.parse_unary()
        while self.at_op("*", "/"):
            op = _MULTIPLICATIVE[self.advance().text]
            rhs = self.parse_unary()
            node = Call(op, (node, rhs))
        return node

    def parse_unary(self) -> Node:
        # every nesting level (call, parenthesis, unary minus) passes through here
        self.depth += 1
        try:
            if self.depth > MAX_DEPTH:
                tok = self.peek()
                raise ExprSyntaxError(f"expression nests deeper than {MAX_DEPTH} levels", tok.pos, self.text)
            if self.at_op("-"):
                self.advance()
                operand = self.parse_unary()
                if isinstance(operand, Const):
                    return Const(-operand.value)
                return Call("neg", (operand,))
            return self.parse_primary()
        finally:
            self.depth -= 1

    def parse_primary(self) -> Node:
        tok = self.peek()
        if tok.kind == "number":
            self.advance()
            try:
                return Const(float(tok.text))
            except (TypeError, ValueError) as exc:  # e.g. 1e999 -> inf
                raise ExprSyntaxError(f"invalid number literal {tok.text!r} ({exc})", tok.pos, self.text) from None
        if tok.kind == "ident":
            self.advance()
            if self.at_op("("):
                return self.parse_call(tok)
            return Field(tok.text)
        if self.at_op("("):
            self.advance()
            node = self.parse_expr()
            self.expect_op(")", "to close parenthesis")
            return node
        raise ExprSyntaxError(f"expected an expression, found {self._describe(tok)}", tok.pos, self.text)

    def parse_call(self, name: Token) -> Node:
        self.expect_op("(", f"after operator name {name.text!r}")
        args: list[Node] = []
        if not self.at_op(")"):
            args.append(self.parse_expr())
            while self.at_op(","):
                self.advance()
                args.append(self.parse_expr())
        self.expect_op(")", f"to close call to {name.text!r}")
        return Call(name.text, tuple(args))


def parse(text: str) -> Node:
    """Parse expression ``text`` into a canonical :class:`~alphaforge.expr.ast.Node` tree.

    Args:
        text: source such as ``"cs_rank(ts_delta(close, 5) / close)"``.  Surrounding
            whitespace (including newlines) is ignored.

    Returns:
        The tree; infix operators are already desugared, so the result equals what
        parsing the canonical prefix form would give.

    Raises:
        ExprSyntaxError: (a ``ValueError``) with ``.pos`` on any syntax problem, including
            empty input, trailing garbage and a text deeper than :data:`MAX_DEPTH`.
        TypeError: if ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse() expects a str, got {type(text).__name__}")
    tokens = tokenize(text)
    parser = _Parser(text, tokens)
    if parser.peek().kind == "eof":
        raise ExprSyntaxError("empty expression", 0, text)
    node = parser.parse_expr()
    tail = parser.peek()
    if tail.kind != "eof":
        raise ExprSyntaxError(f"unexpected {parser._describe(tail)} after complete expression", tail.pos, text)
    # a long chain such as a + a + ... + a makes a deep tree without deep nesting in the text
    height = _height(node)
    if height > MAX_DEPTH:
        raise ExprSyntaxError(f"expression tree has depth {height} (limit {MAX_DEPTH})", 0, text)
    return node


def _height(node: Node) -> int:
    """Return the depth of the tree (a leaf is 1), without recursion."""
    best, stack = 0, [(node, 1)]
    while stack:
        cur, d = stack.pop()
        best = max(best, d)
        if isinstance(cur, Call):
            stack.extend((arg, d + 1) for arg in cur.args)
    return best
