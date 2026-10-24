"""Tests for ``alphaforge.expr``.

Coverage:

* parse / print round trip over hand-written expressions (call syntax, infix precedence,
  unary minus, numeric literals) and over random registry-typed trees;
* the AST contract: frozen, hashable, value-equal nodes; ``Const`` stores float;
  ``depth`` / ``size`` / ``iter_nodes`` / ``get_subtree`` / ``replace_subtree`` /
  ``fields_used`` consistency;
* ``validate``: unknown op, arity, int/float/panel/group slot rules, field whitelist;
* ``Evaluator``: equality with direct operator calls, LRU cache hit/miss accounting and
  eviction, error wrapping (unknown field / op, constant in a panel slot, op exception),
  and the ``cs_rank(ts_delta(close, 5) / close)`` NaN prefix on a simulated market.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import alphaforge.ops as ops
from alphaforge.data import DataStore, SimConfig, simulate_market
from alphaforge.expr import (
    Call,
    Const,
    Evaluator,
    ExprEvalError,
    ExprSyntaxError,
    ExprValidationError,
    Field,
    Node,
    depth,
    fields_used,
    get_subtree,
    iter_nodes,
    parse,
    replace_subtree,
    size,
    to_string,
    validate,
)
from alphaforge.ops import REGISTRY, OpSpec

# ----------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_store() -> DataStore:
    return simulate_market(SimConfig(n_assets=30, n_days=120))


def _panel(rng: np.random.Generator, dates: pd.DatetimeIndex, assets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rng.standard_normal((len(dates), len(assets))), index=dates, columns=assets)


@pytest.fixture
def tiny_store() -> DataStore:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=30, name="date")
    assets = [f"S{i:04d}" for i in range(8)]
    a, b, c = (_panel(rng, dates, assets) for _ in range(3))
    b.iloc[3, 2] = np.nan
    c = c.abs() + 0.5  # safe divisor
    g = pd.DataFrame(rng.integers(0, 3, (30, 8)).astype(float), index=dates, columns=assets)
    return DataStore({"a": a, "b": b, "c": c, "g": g})


# ----------------------------------------------------------------------------
# parse / print round trip
# ----------------------------------------------------------------------------

ROUND_TRIP: list[tuple[str, str]] = [
    ("close", "close"),
    ("5", "5"),
    ("-2", "-2"),
    ("- 2", "-2"),
    ("2.5", "2.5"),
    ("2.0", "2"),
    ("-0.5", "-0.5"),
    ("1e-05", "1e-05"),
    (".5", "0.5"),
    ("a + b * c", "add(a, mul(b, c))"),
    ("(a - b) / c", "div(sub(a, b), c)"),
    ("a - b - c", "sub(sub(a, b), c)"),
    ("a / b / c", "div(div(a, b), c)"),
    ("a * b + c * d", "add(mul(a, b), mul(c, d))"),
    ("a + (b + c)", "add(a, add(b, c))"),
    ("a * (b + c)", "mul(a, add(b, c))"),
    ("-a", "neg(a)"),
    ("-(a + b)", "neg(add(a, b))"),
    ("--a", "neg(neg(a))"),
    ("-a * b", "mul(neg(a), b)"),
    ("a - -2", "sub(a, -2)"),
    ("a-2", "sub(a, 2)"),
    ("ts_delta(close, 5) / close", "div(ts_delta(close, 5), close)"),
    ("cs_rank(ts_delta(close, 5) / close)", "cs_rank(div(ts_delta(close, 5), close))"),
    ("ts_corr(close, volume, 20)", "ts_corr(close, volume, 20)"),
    ("power(close, 0.5)", "power(close, 0.5)"),
    ("power(close, 2.0)", "power(close, 2)"),
    ("clip(x, -3.0, 3)", "clip(x, -3, 3)"),
    ("group_neutralize(cs_zscore(returns), industry)", "group_neutralize(cs_zscore(returns), industry)"),
    ("  ts_mean( close ,\n 5 ) ", "ts_mean(close, 5)"),
    ("f()", "f()"),
]


@pytest.mark.parametrize("src,canonical", ROUND_TRIP, ids=[s for s, _ in ROUND_TRIP])
def test_round_trip(src: str, canonical: str) -> None:
    node = parse(src)
    text = to_string(node)
    assert text == canonical
    assert parse(text) == node
    assert str(node) == text  # Node.__str__ delegates to the printer
    assert to_string(parse(text)) == text  # printing is idempotent


def test_parse_is_canonical() -> None:
    assert parse("a/b") == Call("div", (Field("a"), Field("b")))
    assert parse("a/b") == parse("div(a, b)")
    assert parse("a + b * c") == Call("add", (Field("a"), Call("mul", (Field("b"), Field("c")))))
    assert parse("-a") == Call("neg", (Field("a"),))
    assert parse("-2") == Const(-2.0)
    assert parse("f()") == Call("f", ())


def test_const_numeric_literals() -> None:
    n = parse("5")
    assert isinstance(n, Const)
    assert isinstance(n.value, float) and n.value == 5.0
    assert parse("5") == parse("5.0") == Const(5)
    assert parse("1e3") == Const(1000.0)
    assert parse("2.5E-1") == Const(0.25)
    assert to_string(Const(1 / 3)) == repr(1 / 3)
    assert parse(to_string(Const(1 / 3))) == Const(1 / 3)
    assert to_string(Const(-0.0)) == "0"
    assert to_string(Const(1e20)) == "1e+20" and parse("1e+20") == Const(1e20)


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "a +", "(a", "a)", "a b", "ts_mean(close,)", "f(,a)", "a ? b", "1..2", "2a", "+a", "a * * b", "close,"],
)
def test_syntax_errors(bad: str) -> None:
    with pytest.raises(ExprSyntaxError) as info:
        parse(bad)
    assert isinstance(info.value, ValueError)
    assert isinstance(info.value.pos, int) and 0 <= info.value.pos <= len(bad)
    assert "position" in str(info.value)


def test_parse_rejects_non_str() -> None:
    with pytest.raises(TypeError):
        parse(Field("a"))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# random registry-typed trees round trip and validate
# ----------------------------------------------------------------------------

FIELDS = ["open", "high", "low", "close", "volume", "returns", "industry"]


def _random_tree(rng: np.random.Generator, max_depth: int, slot: str = "panel") -> Node:
    if slot == "group":
        return Field("industry")
    ops_ = [s for s in REGISTRY.values() if s.mineable]
    if max_depth <= 1 or rng.random() < 0.25:
        return Field(str(rng.choice(FIELDS)))
    spec = ops_[int(rng.integers(len(ops_)))]
    args: list[Node] = []
    for typ in spec.arg_types:
        if typ == "int":
            args.append(Const(int(rng.choice(spec.int_choices))))
        elif typ == "float":
            args.append(Const(float(rng.choice(spec.float_choices))))
        else:
            args.append(_random_tree(rng, max_depth - 1, typ))
    return Call(spec.name, tuple(args))


def test_random_trees_round_trip_and_validate() -> None:
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(300):
        tree = _random_tree(rng, max_depth=5)
        text = to_string(tree)
        assert parse(text) == tree
        assert hash(parse(text)) == hash(tree)
        validate(tree)
        validate(tree, fields=set(FIELDS))
        assert depth(tree) <= 5
        assert size(tree) == sum(1 for _ in iter_nodes(tree))
        seen.add(tree)
    assert len(seen) > 100  # trees are usable as set / dict keys


# ----------------------------------------------------------------------------
# AST contract
# ----------------------------------------------------------------------------


def test_nodes_are_frozen_hashable_and_value_equal() -> None:
    a1, a2 = Field("close"), Field("close")
    assert a1 == a2 and hash(a1) == hash(a2) and a1 is not a2
    assert Const(5) == Const(5.0) and hash(Const(5)) == hash(Const(5.0))
    assert Field("a") != Const(1) and Field("a") != Call("neg", (Field("a"),))
    assert Call("add", (a1, Const(1))) == Call("add", [a2, Const(1.0)])  # list args coerced to tuple
    assert isinstance(Call("add", [a1, a2]).args, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a1.name = "open"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Const(1).value = 2  # type: ignore[misc]
    assert len({a1, a2, Const(5), Const(5.0)}) == 2


def test_node_constructor_validation() -> None:
    with pytest.raises(TypeError):
        Const(True)
    with pytest.raises(TypeError):
        Const("5")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Const(float("nan"))
    with pytest.raises(ValueError):
        Const(float("inf"))
    with pytest.raises(ValueError):
        Field("1abc")
    with pytest.raises(ValueError):
        Field("my-field")
    with pytest.raises(ValueError):
        Call("bad name", (Field("a"),))
    with pytest.raises(TypeError):
        Call("add", (Field("a"), 5))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Call("neg", Field("a"))  # type: ignore[arg-type]
    assert Const(np.int64(3)) == Const(3.0) and isinstance(Const(np.float32(0.5)).value, float)


def test_depth_and_size() -> None:
    tree = parse("cs_rank(ts_delta(close, 5) / close)")
    assert depth(tree) == 4
    assert size(tree) == 6
    assert depth(Field("a")) == depth(Const(1)) == 1
    assert size(Field("a")) == size(Const(1)) == 1
    assert depth(Call("f", ())) == 1 and size(Call("f", ())) == 1
    assert depth(parse("a + b")) == 2 and size(parse("a + b")) == 3


def test_iter_nodes_preorder_paths() -> None:
    tree = parse("cs_rank(ts_delta(close, 5) / close)")
    got = list(iter_nodes(tree))
    expected = [
        (tree, ()),
        (parse("div(ts_delta(close, 5), close)"), (0,)),
        (parse("ts_delta(close, 5)"), (0, 0)),
        (Field("close"), (0, 0, 0)),
        (Const(5), (0, 0, 1)),
        (Field("close"), (0, 1)),
    ]
    assert got == expected
    for node, path in got:
        assert get_subtree(tree, path) is node


def test_get_and_replace_subtree_consistency() -> None:
    tree = parse("ts_corr(cs_rank(close), ts_mean(volume, 20), 10)")
    new = parse("log(amount)")
    for node, path in iter_nodes(tree):
        # replacing a subtree by itself is the identity
        assert replace_subtree(tree, path, node) == tree
        out = replace_subtree(tree, path, new)
        assert get_subtree(out, path) == new
        assert tree == parse("ts_corr(cs_rank(close), ts_mean(volume, 20), 10)")  # original untouched
        # every node that is neither an ancestor nor a descendant of the path is
        # unchanged *and shared* (ancestors are rebuilt, nothing is deep-copied)
        for other, other_path in iter_nodes(tree):
            is_descendant = other_path[: len(path)] == path
            is_ancestor = path[: len(other_path)] == other_path
            if not (is_descendant or is_ancestor):
                assert get_subtree(out, other_path) is other
        assert parse(to_string(out)) == out
    assert replace_subtree(tree, (), new) is new
    assert replace_subtree(tree, (1, 0), Field("amount")) == parse("ts_corr(cs_rank(close), ts_mean(amount, 20), 10)")
    with pytest.raises(IndexError):
        get_subtree(tree, (3,))
    with pytest.raises(IndexError):
        get_subtree(tree, (0, 0, 0))  # descends into a leaf
    with pytest.raises(IndexError):
        replace_subtree(tree, (0, 5), new)
    with pytest.raises(TypeError):
        replace_subtree(tree, (0,), "close")  # type: ignore[arg-type]


def test_fields_used() -> None:
    assert fields_used(parse("ts_corr(cs_rank(close), ts_mean(volume, 20), 10) + close")) == {"close", "volume"}
    assert fields_used(Const(1)) == set()
    assert fields_used(Field("x")) == {"x"}


# ----------------------------------------------------------------------------
# validate
# ----------------------------------------------------------------------------


def test_validate_accepts_well_formed_trees() -> None:
    validate(parse("cs_rank(ts_delta(close, 5) / close)"))
    validate(parse("cs_rank(ts_delta(close, 5) / close)"), fields={"close"})
    validate(parse("group_rank(close, industry)"))
    validate(parse("group_rank(close, cs_rank(volume))"))  # a Call is accepted in a group slot
    validate(parse("cs_rank(industry)"))  # group ids are ordinary panels
    validate(parse("cs_neutralize(returns, industry, log(market_cap))"))
    validate(parse("power(close, 2)"))  # integral Const in a float slot
    validate(parse("ts_delay(close, 1)"))
    validate(Field("close"))
    validate(Const(5))  # a bare constant is a valid tree ...
    with pytest.raises(ValueError):
        validate(Const(5), require_panel=True)  # ... but not a panel expression


@pytest.mark.parametrize(
    "text",
    [
        "nope(close)",  # unknown op
        "ts_mean(close)",  # arity too low
        "cs_rank(close, volume)",  # arity too high
        "close * 2",  # Const in panel slot
        "2 + 3",
        "neg(5)",
        "ts_mean(close, volume)",  # Field in int slot
        "ts_mean(close, cs_rank(volume))",  # Call in int slot
        "ts_mean(close, 2.5)",  # non-integral window
        "ts_mean(close, 0)",  # window < 1
        "ts_mean(close, -5)",
        "power(close, volume)",  # Field in float slot
        "cs_rank(nope(close))",  # nested unknown op
        "group_rank(close, volume)",  # non-group field in group slot (default group_fields)
    ],
)
def test_validate_rejects(text: str) -> None:
    with pytest.raises(ValueError) as info:
        validate(parse(text))
    assert isinstance(info.value, ExprValidationError)
    assert isinstance(info.value.node, Node)


def test_validate_field_whitelist_and_group_fields() -> None:
    tree = parse("ts_mean(close, 5)")
    validate(tree, fields={"close", "open"})
    with pytest.raises(ValueError, match="unknown field"):
        validate(tree, fields={"open"})
    g = parse("group_rank(close, sector)")
    with pytest.raises(ValueError, match="group"):
        validate(g)
    validate(g, group_fields={"sector"})
    validate(g, group_fields=None)
    with pytest.raises(TypeError):
        validate("close")  # type: ignore[arg-type]


def test_validate_with_custom_registry() -> None:
    reg = {"twice": OpSpec("twice", lambda x: x * 2, ("panel",), "elementwise")}
    validate(Call("twice", (Field("a"),)), registry=reg)
    with pytest.raises(ValueError):
        validate(parse("cs_rank(a)"), registry=reg)


# ----------------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------------


def test_eval_matches_direct_op_calls(sim_store: DataStore) -> None:
    ev = Evaluator(sim_store)
    close, volume, ind = (sim_store.get(f) for f in ("close", "volume", "industry"))
    assert_frame_equal(ev.eval("ts_mean(close, 5)"), ops.ts_mean(close, 5))
    assert_frame_equal(
        ev.eval("cs_rank(ts_delta(close, 5) / close)"),
        ops.cs_rank(ops.div(ops.ts_delta(close, 5), close)),
    )
    assert_frame_equal(ev.eval("ts_corr(close, volume, 20)"), ops.ts_corr(close, volume, 20))
    assert_frame_equal(
        ev.eval("group_neutralize(returns, industry)"), ops.group_neutralize(sim_store.get("returns"), ind)
    )
    assert_frame_equal(ev.eval("-close"), ops.neg(close))
    assert_frame_equal(ev.eval("power(close, 0.5)"), ops.power(close, 0.5))
    assert_frame_equal(
        ev.eval("abs(returns) * volume - close"), ops.sub(ops.mul(ops.abs_(sim_store.get("returns")), volume), close)
    )
    # Node input and str input agree; output sits on the store grid
    out = ev.eval(parse("ts_zscore(close, 10)"))
    assert_frame_equal(out, ev.eval("ts_zscore(close, 10)"))
    assert out.index.equals(sim_store.dates) and out.columns.equals(sim_store.assets)
    assert (out.dtypes == np.float64).all()


def test_eval_field_returns_store_panel(sim_store: DataStore) -> None:
    ev = Evaluator(sim_store)
    assert_frame_equal(ev.eval("close"), sim_store.get("close"))
    assert ev.stats == {"hits": 0, "misses": 0}  # field look-ups are not counted


def test_eval_infix_precedence_numerically(tiny_store: DataStore) -> None:
    ev = Evaluator(tiny_store)
    a, b, c = (tiny_store.get(k) for k in "abc")
    assert_frame_equal(ev.eval("a + b * c"), a + b * c)
    assert_frame_equal(ev.eval("(a - b) / c"), (a - b) / c)
    assert_frame_equal(ev.eval("a - b - c"), a - b - c)
    assert_frame_equal(ev.eval("-a * b"), -a * b)
    assert_frame_equal(ev.eval("group_rank(a, g)"), ops.group_rank(a, tiny_store.get("g")))


def test_cache_hit_counting(sim_store: DataStore) -> None:
    ev = Evaluator(sim_store)
    ev.eval("ts_mean(close, 5)")
    assert ev.stats == {"hits": 0, "misses": 1}
    ev.eval("ts_mean(close, 5)")
    assert ev.stats == {"hits": 1, "misses": 1}
    # shared sub-tree is reused; only the new outer call is a miss
    ev.eval("cs_rank(ts_mean(close, 5))")
    assert ev.stats == {"hits": 2, "misses": 2}
    # infix and prefix spellings share one cache entry
    ev.eval("cs_rank(ts_delta(close, 5) / close)")
    ev.eval("cs_rank(div(ts_delta(close, 5), close))")
    assert ev.stats == {"hits": 3, "misses": 5}
    assert ev.cache_len == 5
    assert "ts_mean(close, 5)" in ev.cached_expressions()
    r1 = ev.eval("ts_mean(close, 5)")
    r2 = ev.eval("ts_mean(close, 5)")
    assert r1 is r2  # cached object is shared, not copied
    ev.clear_cache()
    assert ev.cache_len == 0
    ev.eval("ts_mean(close, 5)")
    assert ev.stats["misses"] == 6
    ev.reset_stats()
    assert ev.stats == {"hits": 0, "misses": 0}


def test_cache_lru_eviction_and_disable(sim_store: DataStore) -> None:
    ev = Evaluator(sim_store, cache_size=2)
    ev.eval("ts_mean(close, 5)")  # [m5]
    ev.eval("ts_mean(close, 10)")  # [m5, m10]
    ev.eval("ts_mean(close, 5)")  # hit -> [m10, m5]
    ev.eval("ts_mean(close, 20)")  # evicts m10 -> [m5, m20]
    assert ev.cached_expressions() == ["ts_mean(close, 5)", "ts_mean(close, 20)"]
    ev.eval("ts_mean(close, 10)")  # miss again
    assert ev.stats == {"hits": 1, "misses": 4}
    assert ev.cache_len == 2
    off = Evaluator(sim_store, cache_size=0)
    off.eval("ts_mean(close, 5)")
    off.eval("ts_mean(close, 5)")
    assert off.stats == {"hits": 0, "misses": 2} and off.cache_len == 0
    with pytest.raises(ValueError):
        Evaluator(sim_store, cache_size=-1)


@pytest.mark.parametrize(
    "text",
    [
        "nope(close)",  # unknown op
        "cs_rank(nope(close))",
        "no_such_field",  # unknown field
        "ts_mean(no_such_field, 5)",
        "close * 2",  # constant in a panel slot
        "5",  # bare constant root
        "ts_mean(close, volume)",  # panel where an int is expected
        "ts_mean(close, 2.5)",
        "ts_mean(close)",  # arity
    ],
)
def test_eval_errors(sim_store: DataStore, text: str) -> None:
    ev = Evaluator(sim_store)
    with pytest.raises(ExprEvalError) as info:
        ev.eval(text)
    assert isinstance(info.value, RuntimeError)
    assert info.value.node is not None and info.value.expr == to_string(info.value.node)
    assert info.value.expr in str(info.value)  # the message names the failing subtree
    assert ev.cache_len == 0  # nothing partially cached from a failed evaluation
    with pytest.raises(TypeError):
        ev.eval(5)  # type: ignore[arg-type]
    with pytest.raises(ExprSyntaxError):
        ev.eval("close +")


def test_eval_wraps_operator_exceptions(sim_store: DataStore) -> None:
    def boom(x: pd.DataFrame) -> pd.DataFrame:
        raise ZeroDivisionError("kaboom")

    reg = dict(REGISTRY)
    reg["boom"] = OpSpec("boom", boom, ("panel",), "elementwise")
    reg["bad_shape"] = OpSpec("bad_shape", lambda x: x.iloc[:5], ("panel",), "elementwise")
    reg["not_panel"] = OpSpec("not_panel", lambda x: x.to_numpy(), ("panel",), "elementwise")
    ev = Evaluator(sim_store, registry=reg)
    with pytest.raises(ExprEvalError) as info:
        ev.eval("cs_rank(boom(ts_mean(close, 5)))")
    err = info.value
    assert "boom(ts_mean(close, 5))" in str(err)  # names the failing subtree ...
    assert err.expr == "boom(ts_mean(close, 5))" and err.node == parse("boom(ts_mean(close, 5))")
    assert isinstance(err.__cause__, ZeroDivisionError)  # ... and chains the original
    assert ev.cached_expressions() == ["ts_mean(close, 5)"]  # the inner result was cached before the failure
    with pytest.raises(ExprEvalError, match="shape"):
        ev.eval("bad_shape(close)")
    with pytest.raises(ExprEvalError, match="DataFrame"):
        ev.eval("not_panel(close)")
    assert "boom" not in REGISTRY  # the global registry was not touched


def test_eval_on_simulated_store_nan_prefix_and_range(sim_store: DataStore) -> None:
    ev = Evaluator(sim_store)
    out = ev.eval("cs_rank(ts_delta(close, 5) / close)")
    assert out.shape == (120, 30)
    assert out.iloc[:5].isna().all().all()  # ts_delta(., 5) needs 5 prior rows
    body = out.iloc[5:]
    assert body.notna().mean().mean() > 0.95  # close is never NaN in the simulator
    vals = body.to_numpy()
    finite = vals[np.isfinite(vals)]
    assert finite.size > 0 and (finite > 0).all() and (finite <= 1).all()
    assert np.allclose(body.max(axis=1), 1.0)  # the best-ranked name each day scores 1
    assert_frame_equal(out, ops.cs_rank(ops.div(ops.ts_delta(sim_store.get("close"), 5), sim_store.get("close"))))


def test_evaluator_accepts_mapping_store(tiny_store: DataStore) -> None:
    panels = {name: tiny_store.get(name) for name in tiny_store.fields}
    ev = Evaluator(panels)
    assert_frame_equal(ev.eval("ts_mean(a, 3)"), ops.ts_mean(panels["a"], 3))
    with pytest.raises(ExprEvalError, match="unknown field"):
        ev.eval("zzz")
    assert "cache=" in repr(ev)
