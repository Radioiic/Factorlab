"""The depth limit of the expression parser.

A text that nests deeper than MAX_DEPTH levels, or that gives a tree deeper than
MAX_DEPTH, gives ExprSyntaxError (a ValueError), not a RecursionError. The property
tests send random text to the parser: each input parses or gives ExprSyntaxError. They
use a fixed seed (derandomize) and no example database. Hypothesis keeps its cache in
.hypothesis/ in the current folder (git ignores engine/.hypothesis/).
"""

from __future__ import annotations

import pytest

from alphaforge.expr.ast import depth
from alphaforge.expr.parser import MAX_DEPTH, ExprSyntaxError, parse
from alphaforge.expr.printer import to_string


def _nested(op: str, n: int) -> str:
    return f"{op}(" * n + "close" + ")" * n


def test_depth_at_the_limit_parses():
    node = parse(_nested("abs", MAX_DEPTH - 1))
    assert depth(node) == MAX_DEPTH
    assert parse(to_string(node)) == node


@pytest.mark.parametrize(
    "text",
    [
        _nested("abs", MAX_DEPTH),
        _nested("abs", 3000),
        "(" * 3000 + "close" + ")" * 3000,
        "-" * 5000 + "close",
        "-(" * 2000 + "close" + ")" * 2000,
        "+".join(["close"] * 5000),
        "*".join(["close"] * (MAX_DEPTH + 1)),
    ],
)
def test_deep_text_gives_a_syntax_error(text):
    with pytest.raises(ExprSyntaxError):
        parse(text)


def test_a_long_chain_at_the_limit_parses():
    node = parse("+".join(["close"] * MAX_DEPTH))
    assert depth(node) == MAX_DEPTH


def test_unary_minus_on_a_constant_folds_within_the_limit():
    assert parse("-" * (MAX_DEPTH - 1) + "2").value in (2.0, -2.0)


def _check(text: str) -> None:
    try:
        node = parse(text)
    except ExprSyntaxError:
        return
    assert depth(node) <= MAX_DEPTH
    assert parse(to_string(node)) == node


try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ImportError:  # the test packages without hypothesis: the deterministic tests above still run
    given = None

if given is None:

    @pytest.mark.skip(reason="hypothesis is not installed")
    def test_random_text():
        pass

else:
    SETTINGS = settings(max_examples=400, deadline=None, derandomize=True, database=None)
    TOKENS = [
        "close",
        "volume",
        "abs",
        "ts_mean",
        "cs_rank",
        "neg",
        "(",
        ")",
        ",",
        " ",
        "+",
        "-",
        "*",
        "/",
        "0",
        "1",
        "5",
        "1e308",
        "1e999",
        ".5",
        "x_",
        "\n",
        "#",
        "%",
    ]

    @SETTINGS
    @given(st.text(max_size=300))
    def test_random_text(text):
        _check(text)

    @SETTINGS
    @given(st.lists(st.sampled_from(TOKENS), max_size=300).map("".join))
    def test_random_token_strings(text):
        _check(text)

    @SETTINGS
    @given(st.integers(min_value=0, max_value=4000), st.sampled_from(["abs(", "(", "-", "-(", "ts_mean("]))
    def test_deep_input(n, prefix):
        suffix = ", 5" if prefix == "ts_mean(" else ""
        _check(prefix * n + "close" + (suffix + ")" * prefix.count("(")) * n)
