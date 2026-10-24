"""Limits of the factor expression parser: length, depth and constant powers.

An expression outside a limit gives ExpressionError (a ValueError), never a RecursionError,
a MemoryError or a long computation. Valid expressions keep their parse tree.
"""

from __future__ import annotations

import ast
import random

import numpy as np
import pandas as pd
import pytest

from factorlab import expr
from factorlab.data.synthetic import make_synthetic_panel
from factorlab.expr import (
    MAX_CONSTANT_EXPONENT,
    MAX_EXPRESSION_DEPTH,
    MAX_EXPRESSION_LENGTH,
    Evaluator,
    ExpressionError,
    _depth,
    max_lookback,
    parse,
    raw_dependencies,
)
from factorlab.fields import DERIVED_BY_NAME
from factorlab.generator import generate_factors


@pytest.fixture(scope="module")
def expressions():
    return [s.expression for s in generate_factors()] + [d.expression for d in DERIVED_BY_NAME.values()]


def test_error_is_a_value_error():
    assert issubclass(ExpressionError, ValueError)


def test_every_library_expression_is_far_inside_the_limits(expressions):
    longest = max(len(e) for e in expressions)
    deepest = max(_depth(ast.parse(e, mode="eval").body) for e in expressions)
    assert longest * 4 <= MAX_EXPRESSION_LENGTH, longest
    assert deepest * 2 <= MAX_EXPRESSION_DEPTH, deepest


def test_parse_keeps_the_tree_of_valid_expressions(expressions):
    for e in random.Random(1).sample(expressions, 500):
        assert ast.dump(parse(e)) == ast.dump(ast.parse(e, mode="eval").body), e


def test_length_limit():
    at_limit = "close" + " " * (MAX_EXPRESSION_LENGTH - len("close"))
    assert len(at_limit) == MAX_EXPRESSION_LENGTH
    parse(at_limit)
    with pytest.raises(ExpressionError, match="characters"):
        parse(at_limit + " ")


def test_depth_limit():
    parse("-" * (MAX_EXPRESSION_DEPTH - 1) + "close")  # depth: the unary nodes and the field
    with pytest.raises(ExpressionError, match="nested nodes"):
        parse("-" * MAX_EXPRESSION_DEPTH + "close")
    with pytest.raises(ExpressionError, match="nested nodes"):
        parse("abs(" * 100 + "close" + ")" * 100)


@pytest.mark.parametrize(
    "text",
    [
        "-" * 5000 + "close",  # longer than the limit
        "-" * 3000 + "close",  # inside the length limit: the depth limit applies
        "abs(" * 1500 + "close" + ")" * 1500,
        "(" * 300 + "close" + ")" * 300,  # too many parentheses for the Python parser
        "1" * 5000,
        "close\x00",
    ],
)
def test_deep_or_long_input_gives_expression_error(text):
    with pytest.raises(ExpressionError):
        parse(text)


def test_parser_failures_become_expression_error(monkeypatch):
    """Without the length limit, the Python parser itself fails (MemoryError or SyntaxError).
    The result is still ExpressionError."""
    monkeypatch.setattr(expr, "MAX_EXPRESSION_LENGTH", 10**7)
    for text in ["-" * 200_000 + "close", "abs(" * 3000 + "close" + ")" * 3000]:
        with pytest.raises(ExpressionError):
            parse(text)


def test_recursion_error_in_validation_becomes_expression_error(monkeypatch):
    def deep(node, expression):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(expr, "_validate", deep)
    with pytest.raises(ExpressionError, match="too deep"):
        parse("close * 3 + 1")


def test_non_string_input():
    for bad in (None, 3, b"close"):
        with pytest.raises(ExpressionError, match="must be a string"):
            parse(bad)


@pytest.mark.parametrize(
    "text",
    [
        "9**9**9",
        f"2**{MAX_CONSTANT_EXPONENT + 1}",
        f"2**-{MAX_CONSTANT_EXPONENT + 1}",
        "(2**100)**100",
        "((2**60)**60)**60",
        "1e308**2",
        "0**-1",
        "2**(1/0)",
        "close + 10**(10**3)",
        "ts_mean(close, 5) * (3**99999999)",
    ],
)
def test_large_constant_power_is_refused(text):
    with pytest.raises(ExpressionError, match="constant power"):
        parse(text)


@pytest.mark.parametrize("text", ["'a'**2", "2**'a'", "1j**2", "True**2", "(2**None)**2"])
def test_power_of_a_constant_that_is_not_a_number(text):
    with pytest.raises(ExpressionError, match="only numeric constants"):
        parse(text)


@pytest.mark.parametrize(
    "text",
    [
        "2**10",
        f"2**{MAX_CONSTANT_EXPONENT}",
        "10**-30",
        "(-2)**3",
        "-2**0.5",
        "close ** 2",
        "close ** 99999999",
        "2 ** close",
        "close ** (2**10)",
        "ts_mean(close, 5) ** 3",
    ],
)
def test_other_powers_are_allowed(text):
    parse(text)


def test_allowed_constant_power_gives_the_same_value():
    panels = make_synthetic_panel(n_symbols=4, n_bars=60, seed=5)
    ev = Evaluator(panels)
    a = ev("close * 2**10")
    b = ev("close * 1024")
    pd.testing.assert_frame_equal(a, b)
    assert np.isfinite(ev(f"close * 2**{MAX_CONSTANT_EXPONENT}").to_numpy()[-1]).all()


def test_dependencies_and_lookback_after_the_limits():
    assert raw_dependencies("ts_mean(ret, 5) ** 2") == {"close"}
    assert max_lookback("ts_mean(delay(close, 5), 20) * 2**3") == 25


def test_syntax_error_names_the_place_not_the_parser_file():
    with pytest.raises(ExpressionError) as exc:
        parse("ts_mean(ret, 5")
    text = str(exc.value)
    assert text.startswith("syntax error in 'ts_mean(ret, 5': a bracket is not closed (column 8)"), text
    assert "<unknown>" not in text
    with pytest.raises(ExpressionError) as exc:
        parse("ts_mean(ret,\n 5))")
    assert "(line 2, column" in str(exc.value) and "<unknown>" not in str(exc.value), str(exc.value)


# the words of the Python parser that a message must not show: class names of the syntax tree,
# exception names and the hints of the parser
PARSER_WORDS = (
    "Compare", "BoolOp", "Subscript", "Attribute", "Lambda", "IfExp", "JoinedStr", "Starred", "Tuple",
    "List", "Dict", "Set", "Call", "Name", "BinOp", "UnaryOp", "Mod", "FloorDiv", "Not", "Invert",
    "Error", "Perhaps", "Did you mean", "Maybe you meant", "literal", "never closed", "unmatched",
    "invalid syntax", "unknown>", "op ", "keyword", "NoneType", "int", "bytes",
)  # fmt: skip


@pytest.mark.parametrize(
    "text, words",
    [
        ("close > 1", "a comparison"),
        ("close and ret", "and / or"),
        ("close[1]", "an index in square brackets"),
        ("close.shift", "a dot (.)"),
        ("lambda: close", "lambda"),
        ("close if ret else 1", "if / else"),
        ("(close, ret)", "a list"),
        ("{close}", "braces"),
        ("f'{close}'", "text in quotation marks"),
        ("ts_mean(*close)", "a star (*)"),
        ("close % 2", "the operator % is not allowed"),
        ("close // 2", "the operator // is not allowed"),
        ("not close", "the operator not is not allowed"),
        ("~close", "the operator ~ is not allowed"),
        ("foo(close)", "unknown operator 'foo'"),
        ("close.mean()", "only the name of an operator"),
        ("ts_mean(close, w=5)", "an argument with a name"),
        ("ts_mean(close 5)", "the syntax is not correct (column"),
        ("ts_mean(ret, 5", "a bracket is not closed (column 8)"),
        ("ts_mean(ret, 5))", "a closing bracket has no opening bracket"),
        ("'abc", "a quotation mark is not closed"),
        ("1abc", "the syntax is not correct"),
        ("close\x00", "the expression contains a null character"),
        ("", "the expression is empty"),
        (3, "an expression must be a string"),
    ],
)
def test_messages_are_plain_text(text, words):
    """A message says what is wrong in plain words. It names no class or hint of the Python parser."""
    with pytest.raises(ExpressionError) as exc:
        parse(text)
    msg = str(exc.value)
    assert words in msg, msg
    shown = msg.replace(repr(text), "") if isinstance(text, str) else msg
    hits = [w for w in PARSER_WORDS if w in shown]
    assert not hits, (hits, msg)
