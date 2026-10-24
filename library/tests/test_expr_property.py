"""Property tests: random text into the factor expression parser.

Each input either parses, or gives ExpressionError. No other exception (RecursionError,
MemoryError, TypeError, OverflowError ...) is allowed. The tests use a fixed seed
(derandomize) and no example database, so they give the same result on each run.
Hypothesis keeps its cache in the git-ignored data/hypothesis/ (tests/conftest.py).
"""

from __future__ import annotations

import ast

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from factorlab.expr import ExpressionError, max_lookback, parse, raw_dependencies  # noqa: E402
from factorlab.operators import OPERATORS  # noqa: E402

SETTINGS = settings(max_examples=400, deadline=None, derandomize=True, database=None)

TOKENS = (
    ["close", "volume", "ret", "funding_rate", "open_interest", "bench", "x", "__import__", "lambda"]
    + sorted(OPERATORS)[:40]
    + ["(", ")", ",", " ", "+", "-", "*", "/", "**", "%", "//", ".", "[", "]", "'", '"', "=", ":", "if", "else"]
    + ["0", "1", "2", "5", "20", "9", "1e308", "0.5", "-1", "10**", "j", "\\", "\n", "\x00"]
)


def _check(text: str) -> None:
    try:
        tree = parse(text)
    except ExpressionError:
        return
    assert isinstance(tree, ast.AST)
    # a parsed expression also has its dependencies and its lookback
    assert isinstance(raw_dependencies(text), frozenset)
    assert isinstance(max_lookback(text), int)


@SETTINGS
@given(st.text(max_size=300))
def test_random_text(text):
    _check(text)


@SETTINGS
@given(st.lists(st.sampled_from(TOKENS), max_size=200).map("".join))
def test_random_token_strings(text):
    _check(text)


@SETTINGS
@given(
    st.integers(min_value=0, max_value=5000),
    st.sampled_from(["-", "+", "abs(", "(", "ts_mean(", "-(", "2**"]),
    st.sampled_from(["close", "1", "2**2", "9**9"]),
)
def test_deep_input(n, prefix, core):
    close = ")" * prefix.count("(")
    suffix = ", 5" if prefix == "ts_mean(" else ""
    _check(prefix * n + core + (suffix + close) * n)


@SETTINGS
@given(st.integers(min_value=0, max_value=10**12), st.integers(min_value=-(10**12), max_value=10**12))
def test_constant_powers(base, exponent):
    _check(f"{base}**{exponent}")
    _check(f"close * ({base}**{exponent})")


@SETTINGS
@given(
    st.one_of(st.integers(), st.floats(), st.text(max_size=5), st.complex_numbers(), st.booleans(), st.none()),
    st.one_of(st.integers(), st.floats(), st.text(max_size=5), st.complex_numbers(), st.booleans(), st.none()),
)
def test_powers_of_any_constant(left, right):
    _check(f"({left!r})**({right!r})")
    _check(f"close * ({left!r})**({right!r})")
