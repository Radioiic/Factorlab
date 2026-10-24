"""Tests for FINDER_REGISTRY (factorfinder/libops.py): the library's operators for the generator.

Usage (from engine/):
    python -B test_finder_libops.py

Reads the data folder read-only; nothing is written. Without the perp bars every test is
skipped (exit 0); the test that compares with the library's stored values is skipped
without data/library_on_engine/values.

  coverage            every library operator (46) plus the library's arithmetic is
                      registered; exp, clip, pct_change, ts_maxdd, ts_maxru, cs_mean and
                      cs_wmean are registered but not mineable; windows are the library's
                      own within 3-90 days (with its per-operator minimums), lags 1-30
  settings            defaults() reads [generator.settings]: depth 3, 20 nodes, nested
                      lookback 180 days; it refuses windows that differ from WINDOWS
  future mutation     every mineable operator, at its shortest and longest window or lag:
                      input VALUES overwritten after a cut (later NaN kept where it was)
                      leave every earlier output value bit-identical; the output keeps the
                      input's shape, dates and coins, and nothing appears before a coin's
                      first value
  truncation, NaNs    every registered operator (all 50), every int choice, three cuts:
                      with the later rows cut off, or later cells emptied and filled, the
                      earlier outputs keep their finite mask and agree within the library
                      screen's PIT tolerance (rtol 1e-9, atol 1e-12); only decay_linear,
                      ts_slope and ts_rsq may differ in the last bits (BLAS rounding of
                      _rolling_np's matrix-vector product over the complete windows)
  alphaforge          alphaforge.ops.REGISTRY has the same names and the same spec hash
                      before and after FINDER_REGISTRY is built and used; the import is
                      lazy (importing the Finder modules imports no alphaforge module)
  generated trees     thousands of draws, several seeds: no non-mineable operator, depth
                      <= 3, at most 20 nodes, nested lookback <= 180 days
  nested lookback     hand-computed cases, and the library's own max_lookback for trees
                      without ts_autocorr
  same DSL            a tree printed with to_library() parses in the library and its
                      Evaluator gives bit-identical values; fourteen library definitions
                      written as trees (amihud, and mkt_ret and btc_ret inside ts_beta,
                      ts_resid and ts_corr among them) reproduce the screen's stored values
                      on the G Sundays
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from factorfinder import fieldmap as FM
from factorfinder import libops as LO

ENGINE = Path(__file__).resolve().parent
_CACHE: dict = {}


def _terms():
    if "terms" not in _CACHE:
        _CACHE["inputs"] = FM.load_inputs()
        _CACHE["terms"] = FM.build_terminals(_CACHE["inputs"])
    return _CACHE["terms"]


def _A():
    from alphaforge.expr import ast as A

    return A


def _panels(seed: int, n: int = 260, m: int = 7) -> list[pd.DataFrame]:
    """Three synthetic panels: positive price-like levels, signed returns-like values and a
    ragged one, with coins listed late (NaN before) and a few holes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    cols = [f"C{i}" for i in range(m)]
    lvl = np.exp(np.cumsum(rng.normal(0, 0.03, (n, m)), axis=0)) * 100
    sig = rng.normal(0, 1, (n, m))
    rag = np.round(rng.normal(0, 1, (n, m)), 1)
    out = []
    for a in (lvl, sig, rag):
        a = a.copy()
        for j, start in enumerate((0, 0, 0, 30, 75, 120, 0)[:m]):
            a[:start, j] = np.nan
        a[rng.random(a.shape) < 0.01] = np.nan
        out.append(pd.DataFrame(a, index=idx, columns=cols))
    return out


def _mutate(p: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = p.to_numpy().copy()
    late = a[k:]
    a[k:] = np.where(np.isfinite(late), rng.normal(0, 5, late.shape) + np.abs(np.nan_to_num(late)), np.nan)
    return pd.DataFrame(a, index=p.index, columns=p.columns)


# --------------------------------------------------------------------------- tests


def test_registry_covers_the_library_operators():
    """Every library operator plus +, -, * and unary minus; the not-mineable list from the
    config; windows and lags from the library generator, inside 3-90 and 1-30."""
    import libfactors  # noqa: F401
    from factorlab import generator as LG
    from factorlab.operators import OPERATORS

    reg = LO.registry()
    assert set(reg) == set(OPERATORS) | set(LO.ARITHMETIC), sorted(set(reg) ^ (set(OPERATORS) | set(LO.ARITHMETIC)))
    assert len(OPERATORS) == 46 and len(reg) == 50
    nm = {k for k, s in reg.items() if not s.mineable}
    assert nm == {"exp", "clip", "pct_change", "ts_maxdd", "ts_maxru", "cs_mean", "cs_wmean"} == set(LO.non_mineable())
    assert "bench" not in reg and "pow" not in reg
    for name, s in reg.items():
        if "int" not in s.arg_types:
            continue
        if name in ("delay", "delta", "pct_change"):
            assert s.int_choices == LO.LAGS and min(LO.LAGS) == 1 and max(LO.LAGS) == 30
        else:
            assert set(s.int_choices) <= set(LG.WINDOWS) and min(s.int_choices) >= 3 and max(s.int_choices) == 90, name
    assert reg["ts_skew"].int_choices[0] == 10 and reg["ts_kurt"].int_choices[0] == 14
    assert reg["ts_autocorr"].int_choices[0] == 14 and reg["ts_std"].int_choices[0] == 5
    assert reg["ts_corr"].int_choices[0] == 5 and reg["ts_mean"].int_choices[0] == 3
    assert reg["ts_autocorr"].arg_types == ("panel", "int"), "the lag is the library default, 1"
    assert {k for k, s in reg.items() if s.commutative} == {"add", "mul", "emax", "emin"}


def test_settings_are_read_and_windows_are_checked():
    """defaults() reads [generator.settings] (depth 3, 20 nodes, nested lookback 180 days, the
    non-mineable operators); a changed value changes the generator version; windows that
    differ from libops.WINDOWS are refused."""
    from factorfinder import config_io as K

    cfg = K.load()
    d = LO.defaults(cfg)
    assert (d["max_depth"], d["max_nodes"], d["max_nested_lookback"]) == (3, 20, 180)
    assert d["non_mineable"] == tuple(cfg["generator"]["settings"]["non_mineable"])
    deeper = json.loads(json.dumps(cfg))
    deeper["generator"]["settings"]["max_depth"] = 4
    assert K.versions(deeper)["generator_config_version"] != K.versions(cfg)["generator_config_version"]
    assert K.versions(deeper)["eval_config_version"] == K.versions(cfg)["eval_config_version"]
    assert LO.defaults(deeper)["max_depth"] == 4
    bad = json.loads(json.dumps(cfg))
    bad["generator"]["settings"]["window_max_days"] = 120
    try:
        LO.defaults(bad)
        raise AssertionError("windows other than libops.WINDOWS were accepted")
    except ValueError as exc:
        assert "windows" in str(exc), exc


def test_every_mineable_operator_passes_future_mutation_and_keeps_its_shape():
    """Outputs on rows <= a cut are bit-identical when every input VALUE after it is noise
    (a later NaN stays NaN: value mutation, the case bit-identity is guaranteed for); the
    output has the input's dates and coins; no value before a coin's first input value."""
    reg = LO.registry()
    p1, p2, p3 = _panels(1)
    k = 170
    q1, q2, q3 = _mutate(p1, k, 2), _mutate(p2, k, 3), _mutate(p3, k, 4)
    first = (p1.notna() | p2.notna()).cummax().to_numpy()
    reached = 0
    tried = 0
    for name, s in sorted(reg.items()):
        if not s.mineable:
            continue
        ints = sorted({s.int_choices[0], s.int_choices[-1]}) if "int" in s.arg_types else [None]
        for w in ints:
            for a, b, qa, qb in ((p1, p2, q1, q2), (p2, p3, q2, q3)):
                args, margs = [], []
                pan = iter([(a, qa), (b, qb)])
                for t in s.arg_types:
                    if t == "panel":
                        x, qx = next(pan)
                        args.append(x)
                        margs.append(qx)
                    else:
                        args.append(w)
                        margs.append(w)
                with np.errstate(all="ignore"):
                    out, mout = s.func(*args), s.func(*margs)
                tried += 1
                assert isinstance(out, pd.DataFrame), name
                assert out.shape == a.shape and out.index.equals(a.index) and list(out.columns) == list(a.columns), name
                o, mo = out.to_numpy(dtype=float), mout.to_numpy(dtype=float)
                assert np.array_equal(o[:k], mo[:k], equal_nan=True), f"{name}({w}) changed a value before the cut"
                assert not (np.isfinite(o) & ~first).any(), f"{name}({w}) produced a value before a coin's first input"
                reached += not np.array_equal(o[k:], mo[k:], equal_nan=True)
    assert tried >= 100 and reached >= 0.9 * tried, (reached, tried)


#: the operators whose earlier values may differ in the last bits when later rows are cut
#: off or later cells change from missing to present (libops.py docstring)
NOT_BIT_IDENTICAL_UNDER_TRUNCATION = {"decay_linear", "ts_slope", "ts_rsq"}


def _mutate_nan_pattern(p: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Every value after row k replaced, and a quarter of the later cells emptied: cells
    that were missing get values and cells that had values go missing."""
    rng = np.random.default_rng(seed)
    a = p.to_numpy().copy()
    late = a[k:]
    new = rng.normal(0, 5, late.shape) + np.abs(np.nan_to_num(late, nan=100.0))
    new[rng.random(late.shape) < 0.25] = np.nan
    a[k:] = new
    return pd.DataFrame(a, index=p.index, columns=p.columns)


def _call(s, panels, w):
    """The operator on `panels` (one per panel argument), int argument w, floats 0 and 1."""
    pan, fl, args = iter(panels), iter((0.0, 1.0)), []
    for t in s.arg_types:
        args.append(next(pan) if t == "panel" else w if t == "int" else next(fl))
    with np.errstate(all="ignore"):
        return s.func(*args)


def test_every_operator_is_causal_under_truncation_and_nan_pattern_changes():
    """All 50 registered operators, every int choice, cuts at rows 40, 130 and 200: the
    later rows cut off, or later cells emptied and filled, leave every earlier output with
    the same finite mask and within the PIT tolerance; bit-identity fails only for the
    three _rolling_np matrix-vector operators."""
    import libfactors as LF

    reg = LO.registry()
    p1, p2, p3 = _panels(1)
    checks, not_bit, worst = 0, set(), 0.0
    for name, s in sorted(reg.items()):
        ints = list(s.int_choices) if "int" in s.arg_types else [None]
        for w in ints:
            for k in (40, 130, 200):
                for i, (a, b) in enumerate(((p1, p2), (p2, p3))):
                    full = _call(s, (a, b), w).to_numpy(dtype=float)[:k]
                    for mode, (qa, qb) in (
                        ("truncation", (a.iloc[:k], b.iloc[:k])),
                        ("nan_pattern", (_mutate_nan_pattern(a, k, 10 + i), _mutate_nan_pattern(b, k, 20 + i))),
                    ):
                        out = _call(s, (qa, qb), w)
                        assert out.index[:k].equals(a.index[:k]), (name, w, k, mode)
                        got = out.to_numpy(dtype=float)[:k]
                        checks += 1
                        assert np.array_equal(np.isfinite(full), np.isfinite(got)), f"{name}({w}) {mode} at {k}: mask"
                        assert np.isclose(got, full, rtol=LF.PIT_RTOL, atol=LF.PIT_ATOL, equal_nan=True).all(), (
                            f"{name}({w}) {mode} at {k}: an earlier value moved beyond the PIT tolerance"
                        )
                        if not np.array_equal(full, got, equal_nan=True):
                            not_bit.add(name)
                            fin = np.isfinite(full)
                            worst = max(worst, float(np.abs(full[fin] - got[fin]).max()))
    assert len(reg) == 50 and checks > 3000, checks
    assert not_bit <= NOT_BIT_IDENTICAL_UNDER_TRUNCATION, sorted(not_bit - NOT_BIT_IDENTICAL_UNDER_TRUNCATION)
    assert worst < 1e-12, worst


def test_alphaforge_registry_is_untouched_and_the_import_is_lazy():
    """alphaforge.ops.REGISTRY keeps its names and spec hash while FINDER_REGISTRY is built,
    a generator draws from it and trees are evaluated; importing the Finder modules imports
    no alphaforge module."""
    code = r"""
import json, sys
import factorfinder, factorfinder.libops as LO, factorfinder.fieldmap, factorfinder.novelty, factorfinder.probe
lazy = sorted(m for m in sys.modules if m.split('.')[0] == 'alphaforge')
from alphaforge.ops import REGISTRY
before_names, before = sorted(REGISTRY), LO.spec_digest(REGISTRY)
reg = LO.registry()
from factorfinder import fieldmap as FM
gen = LO.generator(FM.terminal_names(), seed=5)
trees, _ = LO.draw_trees(gen, 200)
terms = FM.build_terminals()
ev = LO.evaluator(terms)
for t in trees[:40]:
    LO.evaluate(gen.simplify(t), ev=ev)
print(json.dumps({"lazy": lazy, "same_names": sorted(REGISTRY) == before_names, "same_hash": LO.spec_digest(REGISTRY) == before,
                  "separate": reg is not REGISTRY, "finder_only_absent": [n for n in ("ts_pos_frac", "emax", "slog") if n in REGISTRY],
                  "n_alphaforge": len(REGISTRY)}))
"""
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=ENGINE, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["lazy"] == [], f"importing the Finder modules imported {out['lazy']}"
    assert out["same_names"] and out["same_hash"] and out["separate"], out
    assert out["finder_only_absent"] == [] and out["n_alphaforge"] == 45, out


def test_generated_trees_respect_the_limits_and_never_use_a_non_mineable_operator():
    """Several seeds, thousands of ramped half-and-half draws: within depth 3, 20 nodes and a
    180-day nested lookback, every operator mineable, every leaf a generator terminal."""
    A = _A()
    names = FM.terminal_names()
    nm = set(LO.non_mineable())
    seen_ops: set = set()
    for seed in (1, 2, 3):
        gen = LO.generator(names, seed=seed)
        assert {s.name for s in gen.ops} == {k for k, s in LO.registry().items() if s.mineable}
        trees, info = LO.draw_trees(gen, 1500)
        assert info["draws"] >= 1500
        for t in trees:
            for node in (t, gen.simplify(t)):
                st = LO.tree_stats(node)
                assert st["depth"] <= 3 and st["nodes"] <= 20 and st["lookback"] <= 180, (LO.to_string(node), st)
                assert not set(st["ops"]) & nm, LO.to_string(node)
                assert set(st["fields"]) <= set(names)
                seen_ops |= set(st["ops"])
                A.validate(node, LO.registry(), set(names), group_fields=None)
    assert seen_ops == {k for k, s in LO.registry().items() if s.mineable}, sorted(
        {k for k, s in LO.registry().items() if s.mineable} - seen_ops
    )


def test_nested_lookback_hand_computed_cases():
    """The library's max_lookback convention: windows and lags add along a path, the leaf's
    own lookback included; ts_autocorr adds its window plus its lag of 1."""
    A = _A()
    F_, C_, K_ = A.Field, A.Call, A.Const
    cases = [
        (F_("funding"), 0),
        (F_("ret"), 1),
        (F_("tr_pct"), 1),
        (F_("dvol_chg"), 1),
        (F_("hl_range"), 0),
        (F_("amihud"), 1),
        (F_("mkt_ret"), 1),
        (F_("btc_ret"), 1),
        (F_("sbasis"), 0),
        (F_("perp_range_excess"), 0),
        (F_("deriv_share_chg"), 1),
        (F_("carry_ret"), 1),
        (C_("ts_mean", (F_("ret"), K_(20))), 21),
        (C_("cs_rank", (F_("sbasis"),)), 0),
        (C_("delta", (F_("deriv_share_chg"), K_(30))), 31),
        (C_("ts_autocorr", (F_("carry_ret"), K_(30))), 32),
        (C_("add", (C_("ts_max", (F_("gap"), K_(90))), C_("ts_min", (F_("hl_range"), K_(90))))), 91),
        (C_("ts_corr", (C_("ts_std", (F_("hl_range"), K_(30))), C_("delay", (F_("ret"), K_(5))), K_(60))), 90),
        (C_("ts_mean", (C_("ts_std", (F_("ret"), K_(90))), K_(90))), 181),
        (C_("ts_mean", (C_("ts_std", (F_("funding"), K_(90))), K_(90))), 180),
    ]
    for node, want in cases:
        assert LO.nested_lookback(node) == want, (LO.to_string(node), LO.nested_lookback(node), want)
    try:
        LO.check_limits(cases[-2][0])
        raise AssertionError("a 181-day nested lookback passed the limits")
    except LO.TreeLimitError as exc:
        assert exc.code == "nested_lookback"
    LO.check_limits(cases[-1][0])
    # the library's own count, for library-only trees without ts_autocorr
    import libfactors  # noqa: F401
    from factorlab.expr import max_lookback

    gen = LO.generator(FM.terminal_config()["library"], seed=9)
    trees, _ = LO.draw_trees(gen, 600)
    n = 0
    for t in trees:
        if "ts_autocorr" in LO.tree_stats(t)["ops"]:
            continue
        assert LO.nested_lookback(t) == max_lookback(LO.to_library(t)), LO.to_string(t)
        n += 1
    assert n > 400


def test_trees_print_and_evaluate_like_library_definitions():
    """to_library() prints a tree in the library's language; the library parses it and its
    Evaluator gives bit-identical values to the Finder evaluation."""
    import libfactors as LF
    from factorlab.expr import Evaluator, parse

    terms = _terms()
    ref = Evaluator(FM.perp_panels(_CACHE["inputs"]), benchmark=LF.BENCHMARK)
    ev = LO.evaluator(terms)
    gen = LO.generator(FM.terminal_config()["library"], seed=21)
    trees, _ = LO.draw_trees(gen, 120)
    ok = 0
    for t in trees:
        t = gen.simplify(t)
        text = LO.to_library(t)
        parse(text)
        with np.errstate(all="ignore"):
            want = ref(text).to_numpy(dtype=float)
        got = LO.evaluate(t, ev=ev).to_numpy(dtype=float)
        assert np.array_equal(want, got, equal_nan=True), text
        ok += np.isfinite(got).any()
    assert ok > 80


# library definitions that are trees of the grammar: (factor id, catalog expression, tree)
def _library_cases():
    A = _A()
    F_, C_, K_ = A.Field, A.Call, A.Const
    return [
        ("CF00006", "ts_mean(ret, 20)", C_("ts_mean", (F_("ret"), K_(20)))),
        ("CF00061", "ts_std(ret, 20)", C_("ts_std", (F_("ret"), K_(20)))),
        ("CF00102", "ts_rank(ret, 20)", C_("ts_rank", (F_("ret"), K_(20)))),
        ("CF00169", "ts_skew(ret, 30)", C_("ts_skew", (F_("ret"), K_(30)))),
        ("CF00189", "ts_autocorr(ret, 30, 1)", C_("ts_autocorr", (F_("ret"), K_(30)))),
        ("CF00508", "ts_mean(gap, 20)", C_("ts_mean", (F_("gap"), K_(20)))),
        ("CF01253", "ts_max(abs_ret, 14)", C_("ts_max", (F_("abs_ret"), K_(14)))),
        ("CF01422", "ts_zscore(hl_range, 20)", C_("ts_zscore", (F_("hl_range"), K_(20)))),
        ("CF03420", "ts_mean(dvol_chg, 5)", C_("ts_mean", (F_("dvol_chg"), K_(5)))),
        ("CF04635", "ts_mean(funding, 7)", C_("ts_mean", (F_("funding"), K_(7)))),
        ("CF03675", "ts_mean(amihud, 20)", C_("ts_mean", (F_("amihud"), K_(20)))),
        # the market-wide terminals next to a coin's own series (alone they are constant
        # per date; test_finder_probe.py)
        ("CF19254", "ts_beta(ret, mkt_ret, 10)", C_("ts_beta", (F_("ret"), F_("mkt_ret"), K_(10)))),
        ("CF19258", "ts_resid(ret, mkt_ret, 10)", C_("ts_resid", (F_("ret"), F_("mkt_ret"), K_(10)))),
        ("CF19560", "ts_corr(ret, btc_ret, 10)", C_("ts_corr", (F_("ret"), F_("btc_ret"), K_(10)))),
    ]


def test_library_definitions_as_trees_reproduce_the_stored_values():
    """Fourteen catalog definitions written as trees (among them amihud, and mkt_ret and
    btc_ret inside ts_beta, ts_resid and ts_corr): the same text from to_library(), and
    their G-Sunday values equal the library screen's stored float64 values (rows read with
    the novelty reader, cut at the G cutoff)."""
    import libfactors as LF
    from factorfinder import novelty as N

    if not ((LF.out_dir() / "status.parquet").is_file() and LF.has_values()):
        print("        (skipped: no library values in data/library_on_engine)")
        return

    cat = LF.load_catalog().set_index("factor_id")["expression"]
    st = pd.read_parquet(LF.out_dir() / "status.parquet", columns=["factor_id", "values_shard"]).set_index("factor_id")
    terms = _terms()
    ev = LO.evaluator(terms)
    sundays = N.g_sundays()
    cut = FM.g_cutoff()
    for fid, expr, tree in _library_cases():
        assert cat[fid] == expr and LO.to_library(tree) == expr, (fid, cat[fid], LO.to_library(tree))
        df = N.read_columns(
            LF.out_dir() / LF.VALUES_NAME / st.loc[fid, "values_shard"], ["symbol", fid], through=cut, date_col="sunday"
        )
        df["sunday"] = pd.DatetimeIndex(df["sunday"]).as_unit("ns")
        stored = df.pivot(index="sunday", columns="symbol", values=fid).reindex(index=sundays, columns=FM.symbols())
        got = LO.evaluate(tree, ev=ev).reindex(index=sundays, columns=FM.symbols()).to_numpy(dtype=float)
        assert np.isclose(got, stored.to_numpy(), rtol=LF.PIT_RTOL, atol=LF.PIT_ATOL, equal_nan=True).all(), fid
        assert np.isfinite(got).sum() > 1500, fid


if __name__ == "__main__":
    import config as C

    if not any((C.RAW_OHLCV / f"{C.sym_to_file(s)}.parquet").exists() for s in C.UNIVERSE):
        print("  SKIP  every finder operator-registry test skipped: no perp bars in the data folder (run ./setup.sh)")
        raise SystemExit(0)
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} finder operator-registry tests passed")
    raise SystemExit(1 if failed else 0)
