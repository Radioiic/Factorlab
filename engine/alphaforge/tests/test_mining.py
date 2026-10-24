"""Tests for alphaforge.mining.

Uses a small simulated market (60 assets x 300 days) - the simulator, expression engine,
pre-processing and evaluation are all earlier build stages.
"""

from __future__ import annotations

import math
import pickle
import time

import numpy as np
import pandas as pd
import pytest

from alphaforge.data import DataStore, SimConfig, simulate_market
from alphaforge.expr import Call, Const, Evaluator, Field, depth, parse, size, to_string, validate
from alphaforge.library import FactorLibrary
from alphaforge.mining import (
    MUTATION_KINDS,
    ExpressionGenerator,
    FitnessFunction,
    FitnessResult,
    GeneticMiner,
    MinedFactor,
    select_uncorrelated,
    spearman_corr_panel,
)

PANEL_FIELDS = ["open", "high", "low", "close", "volume", "amount", "vwap", "returns", "market_cap", "bp"]
REVERSAL = "neg(cs_rank(ts_pct_change(close, 5)))"
NOISE = ["cs_rank(ts_corr(open, volume, 3))", "ts_skew(high, 10)", "cs_rank(ts_argmax(low, 20))"]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def store() -> DataStore:
    return simulate_market(SimConfig(n_assets=60, n_days=300, seed=1))


@pytest.fixture(scope="module")
def all_fields(store) -> set[str]:
    return set(store.fields)


@pytest.fixture
def gen() -> ExpressionGenerator:
    return ExpressionGenerator(PANEL_FIELDS, max_depth=4, seed=0)


@pytest.fixture(scope="module")
def fitness(store) -> FitnessFunction:
    d = store.dates
    return FitnessFunction(store, start=d[40], end=d[240])


def _check_tree(node, gen, all_fields) -> None:
    validate(node, fields=all_fields)
    assert depth(node) <= gen.max_depth
    assert parse(to_string(node)) == node


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
def test_random_trees_valid_and_bounded(gen, all_fields):
    for i in range(200):
        method = "grow" if i % 2 == 0 else "full"
        d = 2 + (i % 3)
        tree = gen.random_tree(d, method)
        _check_tree(tree, gen, all_fields)
        assert depth(tree) <= d
        assert isinstance(tree, Call), "a random tree of depth >= 2 is always an operator call"
        if method == "full":
            assert depth(tree) == d
    assert isinstance(gen.random_tree(1), Field)
    with pytest.raises(ValueError):
        gen.random_tree(gen.max_depth + 1)
    with pytest.raises(ValueError):
        gen.random_tree(3, "bushy")


def test_random_terminal_kinds(gen):
    for _ in range(20):
        t = gen.random_terminal("panel")
        assert isinstance(t, Field) and t.name in PANEL_FIELDS
    assert gen.random_terminal("group") == Field("industry")
    assert isinstance(gen.random_terminal("int"), Const)
    assert isinstance(gen.random_terminal("float"), Const)
    with pytest.raises(ValueError):
        gen.random_terminal("matrix")
    with pytest.raises(ValueError):
        ExpressionGenerator(PANEL_FIELDS, group_fields=()).random_terminal("group")


def test_generator_pool_respects_registry_flags_and_weights():
    g = ExpressionGenerator(PANEL_FIELDS)
    names = {s.name for s in g.ops}
    assert "clip" not in names and "cs_neutralize" not in names  # mineable=False
    assert "group_rank" in names
    g2 = ExpressionGenerator(PANEL_FIELDS, group_fields=())
    assert not any("group" in s.arg_types for s in g2.ops)
    g3 = ExpressionGenerator(PANEL_FIELDS, op_weights={"ts": 0.0, "cs_rank": 5.0})
    assert not any(s.category == "ts" for s in g3.ops)
    probs = dict(zip([s.name for s in g3.ops], g3.op_probs))
    assert probs["cs_rank"] > probs["add"]
    with pytest.raises(ValueError):
        ExpressionGenerator(PANEL_FIELDS, op_weights={"no_such_op": 1.0})
    with pytest.raises(ValueError):
        ExpressionGenerator([])


def test_generator_is_deterministic():
    a = ExpressionGenerator(PANEL_FIELDS, seed=7)
    b = ExpressionGenerator(PANEL_FIELDS, seed=7)
    assert [to_string(a.random_tree()) for _ in range(20)] == [to_string(b.random_tree()) for _ in range(20)]


@pytest.mark.parametrize("kind", MUTATION_KINDS)
def test_mutate_outputs_valid(gen, all_fields, kind):
    for _ in range(60):
        parent = gen.random_tree(4, "grow")
        child = gen.mutate(parent, kind)
        _check_tree(child, gen, all_fields)
        if kind == "point" and child != parent:
            assert depth(child) == depth(parent) and size(child) == size(parent)
        if kind == "hoist" and any(isinstance(a, Call) for a in parent.args):
            assert size(child) < size(parent)  # hoisting a proper operator sub-tree always shrinks
        # (a parent without an operator sub-tree falls back to a subtree mutation, which may grow)


def test_mutate_specific_kinds(gen):
    tree = parse("cs_rank(ts_mean(close, 10))")
    consts = {to_string(gen.mutate(tree, "const")) for _ in range(30)}
    assert consts <= {"cs_rank(ts_mean(close, 5))", "cs_rank(ts_mean(close, 20))"}
    points = {to_string(gen.mutate(tree, "point")) for _ in range(60)}
    assert all(p != to_string(tree) for p in points)
    assert any(p.startswith("cs_rank(ts_") for p in points) or any(
        p.startswith(("cs_zscore", "cs_demean", "cs_scale", "neg", "abs")) for p in points
    )
    hoisted = {to_string(gen.mutate(tree, "hoist")) for _ in range(10)}
    assert hoisted == {"ts_mean(close, 10)"}
    # kinds that do not apply fall back to a subtree mutation and still return a valid tree
    leaf = Field("close")
    for kind in ("hoist", "const", "point"):
        out = gen.mutate(leaf, kind)
        validate(out)
        assert depth(out) <= gen.max_depth
    with pytest.raises(ValueError):
        gen.mutate(tree, "teleport")


def test_crossover_outputs_valid(gen, all_fields):
    changed = 0
    for _ in range(100):
        a, b = gen.random_tree(4, "grow"), gen.random_tree(4, "full")
        ca, cb = gen.crossover(a, b)
        _check_tree(ca, gen, all_fields)
        _check_tree(cb, gen, all_fields)
        changed += int(ca != a or cb != b)
    assert changed > 50


def test_crossover_returns_parents_when_infeasible():
    g = ExpressionGenerator(["close"], group_fields=(), max_depth=2, seed=1)
    a, b = parse("cs_rank(close)"), parse("cs_zscore(close)")
    for _ in range(10):
        ca, cb = g.crossover(a, b)
        assert depth(ca) <= 2 and depth(cb) <= 2
        validate(ca)
        validate(cb)


@pytest.mark.parametrize(
    "src, expected",
    [
        ("cs_rank(cs_rank(close))", "cs_rank(close)"),
        ("cs_zscore(cs_zscore(close))", "cs_zscore(close)"),
        ("abs(abs(close))", "abs(close)"),
        ("neg(neg(close))", "close"),
        ("neg(neg(neg(close)))", "neg(close)"),
        ("ts_delay(ts_delay(close, 1), 2)", "ts_delay(close, 3)"),
        ("ts_delay(ts_delay(close, 40), 60)", "ts_delay(ts_delay(close, 40), 60)"),  # 100 not a choice
        ("cs_rank(cs_zscore(close))", "cs_rank(close)"),
        ("cs_zscore(cs_demean(close))", "cs_zscore(close)"),
        ("abs(neg(close))", "abs(close)"),
        ("abs(sqrt(close))", "sqrt(close)"),
        ("group_rank(group_rank(close, industry), industry)", "group_rank(close, industry)"),
        ("add(volume, close)", "add(close, volume)"),
        ("mul(cs_rank(cs_rank(volume)), close)", "mul(close, cs_rank(volume))"),
        ("sub(close, close)", "sub(close, close)"),  # never a Const in a panel slot
        ("close", "close"),
    ],
)
def test_simplify_rules(gen, src, expected):
    out = gen.simplify(parse(src))
    assert to_string(out) == expected
    validate(out)
    assert gen.simplify(out) == out  # idempotent


def test_simplify_preserves_values(store, gen):
    ev = Evaluator(store)
    for src in [
        "cs_rank(cs_zscore(ts_mean(close, 5)))",
        "neg(neg(ts_delta(volume, 3)))",
        "abs(neg(returns))",
        "add(volume, close)",
    ]:
        a = ev.eval(parse(src))
        b = ev.eval(gen.simplify(parse(src)))
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)


# --------------------------------------------------------------------------- #
# fitness
# --------------------------------------------------------------------------- #
def test_fitness_known_good_beats_noise(fitness):
    good = fitness(parse(REVERSAL))
    assert good.valid and math.isfinite(good.fitness) and good.reason == ""
    assert good.expr == REVERSAL and good.complexity == size(parse(REVERSAL))
    # low past return -> high factor value -> outperforms (embedded reversal alpha): positive IC.
    # The fitness uses |icir_ann|, so the sign would not matter either way.
    assert good.rank_ic_mean > 0
    assert 0.0 <= good.turnover <= 1.0 and 0.5 <= good.coverage <= 1.0
    noise = [fitness(parse(e)) for e in NOISE]
    finite = [n.fitness for n in noise if n.valid]
    assert finite, "at least one noise factor should be evaluable"
    assert good.fitness > float(np.mean(finite))
    assert abs(good.icir_ann) > max(abs(n.icir_ann) for n in noise if n.valid)


def test_fitness_formula_matches_fields(store):
    d = store.dates
    ff = FitnessFunction(store, start=d[40], end=d[240], w_icir=1.0, w_ic=0.5, w_turnover=0.3, parsimony=0.02)
    r = ff(parse(REVERSAL))
    expected = 1.0 * abs(r.icir_ann) + 0.5 * abs(r.ic_mean) * 100 - 0.3 * r.turnover - 0.02 * r.complexity
    assert r.fitness == pytest.approx(expected)
    assert r.ic_mean == pytest.approx(r.rank_ic_mean)  # default ic_method is spearman
    assert r.icir_ann == pytest.approx(r.icir * math.sqrt(252))
    assert math.isnan(r.max_corr) and r.max_corr_name is None  # no library


def test_fitness_invalid_cases(store, fitness):
    r = fitness(parse("ts_mean(no_such_field, 5)"))
    assert not r.valid and r.fitness == -math.inf and "no_such_field" in r.reason
    r = fitness(parse("sub(close, close)"))  # constant -> all-NaN after z-scoring
    assert not r.valid and r.fitness == -math.inf and r.reason
    r = fitness(parse("ts_mean(close, 5)"))  # arity ok but used as text -> fine; now a bad tree:
    assert r.valid
    r = fitness(Call("ts_mean", (Field("close"), Field("close"))))  # panel in an int slot
    assert not r.valid
    strict = FitnessFunction(store, start=store.dates[40], end=store.dates[240], min_coverage=1.01)
    r = strict(parse(REVERSAL))
    assert not r.valid and "coverage" in r.reason
    strict_ic = FitnessFunction(store, start=store.dates[40], end=store.dates[240], min_ic_abs=0.99)
    r = strict_ic(parse(REVERSAL))
    assert not r.valid and "min_ic_abs" in r.reason
    with pytest.raises(ValueError):
        FitnessFunction(store, start="2030-01-01")  # empty window


def test_fitness_honours_train_window(store):
    d = store.dates
    start, end = d[60], d[200]
    node = parse(REVERSAL)
    ref = FitnessFunction(store, start=start, end=end)(node)
    values = FitnessFunction(store, start=start, end=end).evaluate_factor_values(node)
    assert values.index.equals(d[60:201]) and list(values.columns) == list(store.assets)

    # (a) dropping everything after `end` changes nothing
    cut = store.slice(None, end)
    r_cut = FitnessFunction(cut, start=start, end=end)(node)
    assert r_cut.fitness == ref.fitness and r_cut.rank_ic_mean == ref.rank_ic_mean and r_cut.turnover == ref.turnover

    # (b) perturbing prices after `end` (a "different future") changes nothing either
    rng = np.random.default_rng(0)
    fields = {name: store.get(name).copy() for name in store.fields}
    after = fields["close"].index > end
    for name in ("open", "high", "low", "close", "volume", "amount", "vwap"):
        fields[name].loc[after] *= np.exp(rng.normal(0, 0.5, size=(int(after.sum()), len(store.assets))))
    perturbed = DataStore(fields, meta={})
    r_pert = FitnessFunction(perturbed, start=start, end=end)(node)
    assert r_pert.fitness == ref.fitness

    # (c) perturbing prices inside the window does change it
    inside = (fields["close"].index >= start) & (fields["close"].index <= end)
    fields["close"].loc[inside] *= np.exp(rng.normal(0, 0.05, size=(int(inside.sum()), len(store.assets))))
    r_in = FitnessFunction(DataStore(fields, meta={}), start=start, end=end)(node)
    assert r_in.fitness != ref.fitness

    # (d) labels never use closes after `end`: the tail of the window has no label
    ff = FitnessFunction(store, start=start, end=end)
    assert ff.labels.iloc[-6:].isna().all().all() and ff.labels.iloc[:-6].notna().any().any()
    assert ff.window.equals(d[60:201])


def test_fitness_cache_and_pickle(store):
    d = store.dates
    ff = FitnessFunction(store, start=d[40], end=d[240])
    node = parse(REVERSAL)
    r1 = ff(node)
    r2 = ff(REVERSAL)  # text form, same canonical key
    assert r2 is r1 and ff.stats["hits"] == 1 and ff.stats["evaluated"] == 1 and ff.cache_len == 1
    clone = pickle.loads(pickle.dumps(ff))
    assert clone.cache_len == 0
    r3 = clone(node)
    assert r3.fitness == r1.fitness and r3.turnover == r1.turnover
    ff.clear_cache()
    assert ff.cache_len == 0


def test_fitness_with_library_and_full_report(store):
    d = store.dates
    ff = FitnessFunction(store, start=d[40], end=d[240], full_report=True)
    rev = parse(REVERSAL)
    base = ff(rev)
    assert base.report is not None and base.report.expr == REVERSAL
    assert base.report.start == d[40] and base.report.end == d[240]
    assert base.report.rank_ic_mean == pytest.approx(base.rank_ic_mean)
    lib = FactorLibrary(max_corr=0.7)
    assert lib.add("reversal", REVERSAL, ff.evaluate_factor_values(rev))
    ff_lib = FitnessFunction(store, start=d[40], end=d[240], library=lib, max_corr=0.5, w_corr=1.0)
    twin = ff_lib(parse("neg(cs_rank(ts_delta(close, 5)))"))  # near-identical ranking
    assert twin.valid and twin.max_corr_name == "reversal" and abs(twin.max_corr) > 0.5
    penalty = (abs(twin.max_corr) - 0.5) * 10
    assert twin.fitness == pytest.approx(abs(twin.icir_ann) - 0.5 * twin.turnover - penalty - 0.01 * twin.complexity)
    unrelated = ff_lib(parse("cs_rank(ts_std(volume, 20))"))
    assert unrelated.valid and abs(unrelated.max_corr) < 0.5


def test_fitness_overflowing_candidate_is_quiet(store):
    """FP overflow inside a garbage candidate must neither warn nor be mistaken for a failure."""
    import warnings

    d = store.dates
    ff = FitnessFunction(store, start=d[40], end=d[240])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # a leaked warning would surface as an exception
        for src in ["ts_product(amount, 20)", "ts_product(volume, 60)", "power(ts_product(amount, 10), 3)"]:
            r = ff(parse(src))
            assert "RuntimeWarning" not in r.reason, r.reason
            v = ff.evaluate_factor_values(parse(src))
            assert not np.isinf(v.to_numpy(dtype=float, na_value=np.nan)).any()


def test_fitness_without_preprocess(store):
    d = store.dates
    ff = FitnessFunction(store, start=d[40], end=d[240], preprocess=False)
    v = ff.evaluate_factor_values(parse("cs_rank(close)"))
    assert v.index.equals(ff.window) and float(np.nanmax(v.to_numpy())) <= 1.0
    assert ff(parse(REVERSAL)).valid


# --------------------------------------------------------------------------- #
# miner
# --------------------------------------------------------------------------- #
def test_miner_small_run(store):
    d = store.dates
    ff = FitnessFunction(store, start=d[40], end=d[240])
    g = ExpressionGenerator(PANEL_FIELDS, max_depth=4, seed=3)
    seen: list[tuple[int, int]] = []
    miner = GeneticMiner(g, ff, population_size=20, n_generations=3, hall_of_fame=10, seed=3)
    t0 = time.perf_counter()
    hof = miner.run(callback=lambda gen, h: seen.append((gen, len(h))))
    assert time.perf_counter() - t0 < 60
    assert hof and all(isinstance(m, MinedFactor) for m in hof)
    fits = [m.fitness for m in hof]
    assert fits == sorted(fits, reverse=True) and all(math.isfinite(f) for f in fits)
    exprs = [m.expr for m in hof]
    assert len(set(exprs)) == len(exprs) and all(m.expr == to_string(m.node) for m in hof)
    assert all(m.result.valid and m.fitness == m.result.fitness for m in hof)
    assert len(miner.history) == 3 and [h["gen"] for h in miner.history] == [0, 1, 2]
    for h in miner.history:
        assert {"gen", "best", "mean", "median", "n_valid", "n_unique", "best_expr", "elapsed_s"} <= set(h)
        assert h["n_valid"] >= 1 and h["n_unique"] <= 20 and h["best_expr"]
    assert [s[0] for s in seen] == [0, 1, 2]
    assert miner.best is hof[0] and miner.hof == hof and len(miner.population) == 20
    assert all(validate(m.node) is None and depth(m.node) <= 4 for m in hof)
    # the hall of fame is the best of everything ever seen: never worse than any generation's best
    assert hof[0].fitness >= max(h["best"] for h in miner.history)
    assert hof[0].fitness >= miner.history[0]["best"]
    assert miner.n_evaluations == ff.stats["evaluated"]


@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork:DeprecationWarning"
)  # loky on py3.13/macOS
def test_miner_parallel_matches_serial(store):
    d = store.dates
    g_kwargs = dict(max_depth=3, seed=5)
    ff = FitnessFunction(store, start=d[40], end=d[240])
    serial = GeneticMiner(
        ExpressionGenerator(PANEL_FIELDS, **g_kwargs),
        ff,
        population_size=8,
        n_generations=2,
        hall_of_fame=5,
        seed=5,
        n_jobs=1,
    ).run()
    parallel = GeneticMiner(
        ExpressionGenerator(PANEL_FIELDS, **g_kwargs),
        ff,
        population_size=8,
        n_generations=2,
        hall_of_fame=5,
        seed=5,
        n_jobs=2,
    ).run()
    assert [m.expr for m in serial] == [m.expr for m in parallel]
    assert [m.fitness for m in serial] == pytest.approx([m.fitness for m in parallel])


def test_miner_early_stopping_and_plain_callable():
    g = ExpressionGenerator(PANEL_FIELDS, max_depth=3, seed=1)
    flat = lambda node: FitnessResult(fitness=1.0, valid=True, expr=to_string(node))  # noqa: E731
    miner = GeneticMiner(g, flat, population_size=10, n_generations=10, early_stop_rounds=2, seed=1, hall_of_fame=5)
    hof = miner.run()
    assert miner.stopped_early and len(miner.history) == 3 and len(hof) == 5

    # a fitness that raises for some trees is contained
    def flaky(node):
        if "ts_" in to_string(node):
            raise RuntimeError("boom")
        return FitnessResult(fitness=float(size(node)), valid=True)

    miner = GeneticMiner(g, flaky, population_size=10, n_generations=2, seed=2, hall_of_fame=5)
    hof = miner.run()
    assert all("ts_" not in m.expr for m in hof)


def test_miner_hof_max_corr_filters(store):
    d = store.dates
    ff = FitnessFunction(store, start=d[40], end=d[240])
    g = ExpressionGenerator(["close", "volume"], max_depth=3, seed=11, op_weights={"elementwise": 0.0})
    miner = GeneticMiner(g, ff, population_size=16, n_generations=2, hall_of_fame=6, hof_max_corr=0.5, seed=11)
    hof = miner.run()
    assert 1 <= len(hof) <= 6
    vals = {m.expr: ff.evaluate_factor_values(m.node) for m in hof}
    for i, a in enumerate(hof):
        for b in hof[i + 1 :]:
            assert abs(spearman_corr_panel(vals[a.expr], vals[b.expr])) <= 0.5 + 1e-12


def test_miner_argument_validation(store):
    g = ExpressionGenerator(PANEL_FIELDS)
    ff = lambda n: FitnessResult(1.0, True)  # noqa: E731
    with pytest.raises(ValueError):
        GeneticMiner(g, ff, population_size=1)
    with pytest.raises(ValueError):
        GeneticMiner(g, ff, crossover_p=0.8, mutation_p=0.5)
    with pytest.raises(ValueError):
        GeneticMiner(g, ff, n_jobs=0)
    with pytest.raises(TypeError):
        GeneticMiner(g, "not callable")


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #
def _mf(expr: str, fit: float) -> MinedFactor:
    return MinedFactor(
        expr=expr, node=Field("close"), fitness=fit, generation=0, result=FitnessResult(fit, math.isfinite(fit))
    )


def test_select_uncorrelated_drops_monotone_transform():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=60, name="date")
    cols = [f"S{i:04d}" for i in range(30)]
    a = pd.DataFrame(rng.standard_normal((60, 30)), index=idx, columns=cols)
    a.iloc[::7, 3] = np.nan
    b = np.exp(a) + 1.0  # monotone transform of a -> spearman corr 1
    c = pd.DataFrame(rng.standard_normal((60, 30)), index=idx, columns=cols)
    assert spearman_corr_panel(a, b) == pytest.approx(1.0)
    assert spearman_corr_panel(a, -a) == pytest.approx(-1.0)
    assert abs(spearman_corr_panel(a, c)) < 0.2
    values = {"a": a, "b": b, "c": c, "neg": -a}
    cands = [_mf("c", 0.5), _mf("b", 2.0), _mf("a", 1.0), _mf("neg", 0.9), _mf("bad", -math.inf)]
    kept = select_uncorrelated(cands, values, max_corr=0.7)
    assert [m.expr for m in kept] == ["b", "c"]  # a and neg are monotone transforms of b
    kept = select_uncorrelated(cands, values, max_corr=0.7, key=lambda m: -m.fitness)
    assert [m.expr for m in kept] == ["c", "neg"]
    assert select_uncorrelated(cands, values, max_corr=1.0) == [
        m for m in sorted(cands, key=lambda m: -m.fitness) if math.isfinite(m.fitness)
    ]
    with pytest.raises(KeyError):
        select_uncorrelated([_mf("missing", 1.0)], values)
    with pytest.raises(ValueError):
        select_uncorrelated(cands, values, max_corr=1.5)
    assert select_uncorrelated([], values) == []
