"""Acceptance tests for the label-free generator probe (finder.py --generator-probe).

Usage (from engine/):
    python -B test_finder_probe.py
    TMPDIR=$PWD/.tmp python -B test_finder_probe.py   # keep the scratch copy repo-local

The probe reads the data folder read-only; the first test runs it in a child process on a
scratch copy of the data folder (FACTORLAB_DATA, under TMPDIR, removed at exit). Without the
raw bars, the label table or the library values (data/library_on_engine), every test is
skipped (exit 0).

  writes nothing      the scratch copy is content-hashed before and after: no file
                      changes, appears or vanishes; the report says "writes: none"
  reads no label      an audit hook sees every Python-level open and database connect,
                      and pyarrow's and pandas' readers are wrapped (pyarrow opens files in
                      C++): no screen_g.parquet, nothing in library/catalog, no fwd_*
                      column and no unprojected read of a file that holds one
  command line        it prints its report; --n and --seed belong to it alone
  deterministic       the same seed gives the same report; another seed another draw
  rules               the label-free G0 checks, the future-perturbation lint (a test-only
                      leaking operator is caught; so is a value before a coin's listing),
                      the value key (a sign flip or a rank transform is the same signal)
                      and the bands of the largest absolute correlation
  numbers only        the report, its keys and its printed lines hold no verdict, tier,
                      stage or gate-pass word; the index line names the four sources
  market-wide         mkt_ret and btc_ret hold one value for every coin on a date (the
                      same for all coins): a formula built from them alone fails G0
                      (R_G0_CONSTANT for the hand-written cases, ema's listing artefact
                      included), and so does every such formula of the default draw, each
                      with fewer distinct values on any Sunday than G0's minimum of 10;
                      next to a coin's own series they pass G0 (checked here with ts_beta,
                      ts_corr, ts_cov and ts_resid), and the draw contains such formulas;
                      the probe's report counts them and none passes
  truncation          the lint perturbs later VALUES; the real-time case cuts later rows
                      off. Every distinct formula of the default draw (seed 1, 2,000
                      formulas), on terminals built from inputs cut at 2022-10-16,
                      2023-07-30 and 2024-01-07: every earlier value keeps its finite
                      mask and agrees within the PIT tolerance (rtol 1e-9, atol 1e-12),
                      the tie-snapped ranks on every G Sunday are identical, and a formula
                      that is not bit-identical uses decay_linear, ts_slope or ts_rsq
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import finder
import labels
from factorfinder import fieldmap as FM
from factorfinder import libops as LO
from factorfinder import novelty as N
from factorfinder import probe as P

ENGINE = Path(__file__).resolve().parent
_CACHE: dict = {}


def _index():
    if "ix" not in _CACHE:
        _CACHE["ix"] = N.build_index()
    return _CACHE["ix"]


def _skip_binance_outputs(src, names) -> list:
    """copytree ignore: the library's Binance outputs (data/library_on_engine/binance, about
    1 GB of git-ignored values), which this suite never reads, and the marker file
    .read_only; the OKX library outputs stay in the copy."""
    if Path(src).name == "library_on_engine" and "binance" in names:
        return ["binance"]
    return [C.PROTECT_MARKER] if C.PROTECT_MARKER in names else []


def tree_hash(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


CHILD = r"""
import contextlib, io, json, os, sys
opened, sql = [], []
def hook(event, args):
    if event == "open":
        opened.append(str(args[0]))
    elif event == "sqlite3.connect":
        sql.append(str(args[0]))
sys.addaudithook(hook)
import pyarrow.parquet as pq, pandas as pd
reads = []
def _where(path):
    return path if isinstance(path, (str, os.PathLike)) else getattr(path, "name", repr(path))
def wrap(fn, kind):
    def inner(path, *a, **k):
        reads.append([kind, str(_where(path)), k.get("columns")])
        return fn(path, *a, **k)
    return inner
pq.read_table = wrap(pq.read_table, "read_table")
pq.read_schema = wrap(pq.read_schema, "read_schema")
pq.read_metadata = wrap(pq.read_metadata, "read_metadata")
pd.read_parquet = wrap(pd.read_parquet, "read_parquet")
_PF = pq.ParquetFile
class PF(_PF):
    def __init__(self, source, *a, **k):
        reads.append(["ParquetFile", str(_where(source)), None])
        super().__init__(source, *a, **k)
pq.ParquetFile = PF
import config, finder
out = io.StringIO()
with contextlib.redirect_stdout(out):
    rc = finder.main(["--generator-probe", "--n", "40", "--seed", "2"])
print(json.dumps({"rc": rc, "stdout": out.getvalue(), "opened": opened, "sql": sql, "reads": reads}))
"""


def test_probe_writes_nothing_and_reads_no_label():
    """Child process, scratch copy: byte-identical tree, 'writes: none', and no open of
    screen_g.parquet or library/catalog, no fwd_* column."""
    tmp = tempfile.mkdtemp(prefix="finder_probe_")
    atexit.register(shutil.rmtree, tmp, True)
    data = Path(tmp) / "data"
    shutil.copytree(C.DATA, data, ignore=_skip_binance_outputs)
    assert (data / "library_on_engine" / "screen_g.parquet").exists()
    before = tree_hash(data)
    r = subprocess.run(
        [sys.executable, "-B", "-c", CHILD],
        cwd=ENGINE,
        capture_output=True,
        text=True,
        timeout=900,
        env=dict(os.environ, FACTORLAB_DATA=str(data)),
    )
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["rc"] == 0
    assert tree_hash(data) == before, "the probe changed a file"
    assert "writes: none" in out["stdout"] and "0 label column(s)" in out["stdout"], out["stdout"][-800:]
    assert "Traceback" not in r.stderr
    seen = out["opened"] + [p for _, p, _ in out["reads"]] + out["sql"]
    data_s, repo = str(data.resolve()), str(ENGINE.parent.resolve())
    assert any(str(Path(p).resolve()).startswith(data_s) for p in seen), "the hooks saw no data file"
    for p in seen:
        rp = str(Path(p).resolve()) if not p.startswith("<") else p
        assert "screen_g.parquet" not in rp, rp
        assert not rp.startswith(os.path.join(repo, "library", "catalog")), rp
    assert out["sql"] == [], "the probe connected to a database"
    import pyarrow.parquet as pq

    for kind, p, cols in out["reads"]:
        if kind in ("read_schema", "read_metadata") or not p.endswith(".parquet") or not Path(p).exists():
            continue
        names = pq.read_schema(p).names
        if cols is None:
            assert not any(n.startswith("fwd_") for n in names), f"an unprojected read of {Path(p).name}"
        else:
            assert not any(str(c).startswith("fwd_") for c in cols), (Path(p).name, cols)


def test_same_seed_gives_the_same_report():
    """Two runs with one seed agree on every count and every formula; another seed draws
    other formulas."""
    ix = _index()
    a = P.run(n=60, seed=7, index=ix, null_perms=2)
    b = P.run(n=60, seed=7, index=ix, null_perms=2)
    c = P.run(n=60, seed=8, index=ix, null_perms=2)
    dump = lambda r: json.dumps({"report": r["report"], "rows": r["rows"]}, sort_keys=True, default=str)  # noqa: E731
    assert dump(a) == dump(b)
    assert [x["expr"] for x in a["rows"]] != [x["expr"] for x in c["rows"]]
    assert a["journal"] == ["writes: none"]
    s = a["report"]["summary"]["all"]
    assert s["drawn"] == 60 == sum(s["status_counts"].values())
    assert s["value_unique_valid"] == sum(s["novelty_counts"].values())
    assert s["distinct"] == s["drawn"] - s["status_counts"].get("R_DUP_STRUCT", 0)
    assert s["passed_g0_and_lint"] == s["status_counts"].get("valid", 0) + s["status_counts"].get("R_DUP_VALUE", 0)
    g = a["report"]["summary"]
    assert g["library_only"]["drawn"] + g["uses_factorlab"]["drawn"] == 60
    assert a["report"]["label_columns_read"] == 0
    assert a["report"]["index"]["pool_hash"] == ix.pool_hash
    assert s["market_only_passed"] == 0 and s["market_only_distinct"] == sum(
        1 for x in a["rows"] if x["market_only"] and x["status"] != "R_DUP_STRUCT"
    )
    # shares are kept at full precision (lines() rounds once)
    assert s["share_valid"] == s["passed_g0_and_lint"] / s["distinct"]


def test_probe_command_line():
    """--generator-probe runs and prints its report; --n and --seed are refused without
    --generator-probe."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = finder.main(["--generator-probe", "--n", "25", "--seed", "3"])
    text = out.getvalue()
    assert rc == 0, text[-800:]
    assert "label-free generator probe" in text and "writes: none" in text and "share valid" in text
    assert re.search(r"Index: [\d,]+ library, \d+ user, \d+ zoo, \d+ proxy members\.", text), text[:1500]
    for label in P.CLASS_LABELS.values():
        assert label in text, label
    assert "/Users/" not in text and "/home/" not in text
    for bad in (["--check", "--n", "5"], ["--check", "--seed", "1"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                finder.main(bad)
                raise AssertionError(f"{bad} was accepted")
            except SystemExit as exc:
                assert exc.code == 2


def test_g0_lint_value_key_and_classes():
    """The label-free rules on hand-made cases, and the lint against a test-only leaking
    operator."""
    rng = np.random.default_rng(0)
    T, m = 114, 20
    el = np.ones((T, m), bool)
    g0 = {"min_coverage": 0.8, "min_names_per_sunday": 15, "min_finite_share": 0.95, "min_distinct_values_per_date": 10}
    good = rng.normal(size=(T, m))
    assert P.g0_label_free(good, el, g0)[0] is None
    x = good.copy()
    x[:30] = np.nan
    assert P.g0_label_free(x, el, g0)[0] == "R_G0_COVERAGE"
    x = good.copy()
    x[:, :2] = np.nan  # 18 of 20 names: counted, finite share 0.9
    assert P.g0_label_free(x, el, g0)[0] == "R_G0_FINITE"
    x = good.copy()
    x[5] = 1.0
    assert P.g0_label_free(x, el, g0)[0] == "R_G0_CONSTANT"
    assert P.g0_label_free(np.sign(good), el, g0)[0] == "R_G0_DISTINCT"
    x = good.copy()
    x[7, :3] = x[7, 3] * (1 + 1e-15)  # floating-point noise is a tie, not a value
    assert P.g0_label_free(x, el, g0)[1]["min_distinct"] == 17

    # the lint: a leaking operator (tomorrow's value) and a listing leak (a value before the first bar)
    from alphaforge.expr import ast as A
    from alphaforge.ops.registry import OpSpec

    inputs = FM.load_inputs()
    terms = FM.build_terminals(inputs)
    reg = dict(LO.registry())
    reg["leak_next"] = OpSpec("leak_next", lambda a: a.shift(-1), ("panel",), "ts")
    reg["fill_zero"] = OpSpec("fill_zero", lambda a: a.fillna(0.0), ("panel",), "elementwise")
    from alphaforge.expr.evaluator import Evaluator

    cut = pd.Timestamp(P.LINT_CUT, tz="UTC")
    pert = P.perturbed_terminals(terms, cut, 1)
    ev, evp = Evaluator(terms, registry=reg), Evaluator(pert, registry=reg)
    k = int(terms["ret"].index.searchsorted(cut, side="right"))
    first = FM.perp_panels(inputs)["close"].notna().cummax().to_numpy()

    def lint(tree):
        return P.lint_reason(LO.evaluate(tree, ev=ev).to_numpy(), LO.evaluate(tree, ev=evp).to_numpy(), k, first)

    ok_tree = A.Call("ts_mean", (A.Field("ret"), A.Const(20)))
    assert lint(ok_tree) is None
    assert lint(A.Call("leak_next", (A.Field("ret"),))) == "R_LINT_FUTURE"
    assert lint(A.Call("fill_zero", (A.Field("ret"),))) == "R_LINT_LISTING"
    assert set(reg) - set(LO.registry()) == {"leak_next", "fill_zero"}, "the test registry is a copy"
    assert "leak_next" not in LO.registry()

    # value key: the same signal under a sign flip or a rank transform; another signal differs
    ix = _index()
    evf = LO.evaluator(terms)

    def vals(tree):
        return LO.evaluate(tree, ev=evf).reindex(index=ix.sundays, columns=ix.symbols).to_numpy()

    base = vals(ok_tree)
    keys = {
        P.value_key(N.candidate(ix, v))
        for v in (base, vals(A.Call("neg", (ok_tree,))), vals(A.Call("cs_rank", (ok_tree,))))
    }
    assert len(keys) == 1
    assert P.value_key(N.candidate(ix, vals(A.Call("ts_mean", (A.Field("ret"), A.Const(30)))))) not in keys

    th = {"max_abs_corr": 0.5, "duplicate": 0.85, "same": 0.999}
    assert P.CLASSES == ("below_050", "from_050_to_085", "from_085_library", "from_085_other", "no_overlap")
    assert P.novelty_class({"max_abs": 0.49, "source": "library"}, th) == "below_050"
    assert P.novelty_class({"max_abs": 0.5, "source": "library"}, th) == "from_050_to_085"
    assert P.novelty_class({"max_abs": 0.85, "source": "library"}, th) == "from_085_library"
    for src in ("user", "zoo", "proxy"):
        assert P.novelty_class({"max_abs": 0.9, "source": src}, th) == "from_085_other"
    assert P.novelty_class({"max_abs": float("nan"), "source": None}, th) == "no_overlap"
    assert P.derived_seed(1) != P.derived_seed(2) and P.derived_seed(1) == P.derived_seed(1)


# the words that a find tool must not write (numbers only): verdicts, tiers, stages, gate-pass words
VERDICT_WORDS = re.compile(
    r"\b(useful|useless|unproven|effective|survivors?|promotable|novel|confirmed|retired|verdicts?|tiers?|stages?)\b"
    r"|\bpass(es|ed)?\s+(the\s+|all\s+|every\s+)?(gates?|screen)\b|\bgates?\s+pass(es|ed)?\b",
    re.IGNORECASE,
)


def _texts(x) -> list:
    """Every string value and every key of a JSON-like object, at each depth."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [str(k) for k in x] + [t for v in x.values() for t in _texts(v)]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _texts(v)]
    return []


def test_report_has_no_verdict_words():
    """The report (its keys and texts), the rows and the printed lines give numbers only. The
    report keys of the bands are share_<band>; the sources of the nearest member are the four
    sources of the index."""
    ix = _index()
    res = P.run(n=40, seed=5, index=ix, null_perms=1)
    s = res["report"]["summary"]["all"]
    for c in P.CLASSES:
        assert f"share_{c}" in s and c in s["novelty_counts"], c
    assert set(s["nearest_sources"]) == set(N.SOURCES) == {"library", "user", "zoo", "proxy"}
    for old in ("share_novel", "share_R_CORR", "share_R_DUP_LIB", "share_R_DUP_ZOO"):
        assert old not in s, old
    texts = _texts(res["report"]) + _texts(res["rows"]) + P.lines(res)
    hits = [t for t in texts if VERDICT_WORDS.search(str(t).replace("_", " "))]
    assert not hits, hits[:10]
    assert {r.get("cls") for r in res["rows"] if r["status"] == "valid"} <= set(P.CLASSES)


PAIR_OPS = ("ts_beta", "ts_corr", "ts_cov", "ts_resid")


def test_market_wide_formulas_fail_g0_alone_and_pass_next_to_a_coin_series():
    """mkt_ret and btc_ret hold one value for every coin on a date, so a formula whose
    leaves are all market-wide cannot rank coins. Hand-written ones (bare, windowed,
    cross-sectional, two market series together, and ema, whose recursion splits coins by
    listing date) fail G0 with R_G0_CONSTANT; every such formula of the default draw fails
    G0, with fewer distinct values on every Sunday than G0's minimum of 10, so R_G0_DISTINCT
    would set it aside as well. Next to a coin's own series in a two-series operator they
    pass G0, and the default draw contains such formulas that do."""
    import libfactors as LF
    from factorfinder import config_io as K

    A = LO._ast()
    F_, C_, K_ = A.Field, A.Call, A.Const
    cfg = K.load()
    g0 = cfg["g0"]
    floor, min_distinct = int(g0["min_names_per_sunday"]), int(g0["min_distinct_values_per_date"])
    terms = FM.build_terminals(FM.load_inputs())
    ev = LO.evaluator(terms)
    sundays, syms = N.g_sundays(), FM.symbols()
    el = N.eligibility(sundays, syms, FM.g_cutoff(cfg))

    def g0_of(tree):
        """(G0 reason, the most distinct tie-snapped values on any counted Sunday)."""
        try:
            v = LO.evaluate(tree, ev=ev).reindex(index=sundays, columns=syms).to_numpy(dtype=float)
        except Exception:  # noqa: BLE001 - the probe records these as R_EVAL_ERROR: set aside too
            return "R_EVAL_ERROR", 0
        reason, _ = P.g0_label_free(v, el, g0)
        x = LF.snap_ties(v)
        fin = np.isfinite(x) & el
        most = max((len(np.unique(x[t][fin[t]])) for t in range(len(x)) if fin[t].sum() >= floor), default=0)
        return reason, most

    alone = [
        F_("mkt_ret"),
        F_("btc_ret"),
        C_("ts_mean", (F_("mkt_ret"), K_(20))),
        C_("ts_std", (F_("btc_ret"), K_(30))),
        C_("ts_kurt", (F_("mkt_ret"), K_(90))),
        C_("cs_rank", (F_("btc_ret"),)),
        C_("ts_corr", (F_("mkt_ret"), F_("btc_ret"), K_(30))),
        C_("sub", (F_("mkt_ret"), F_("btc_ret"))),
        C_("ema", (F_("btc_ret"), K_(90))),
    ]
    for t in alone:
        reason, most = g0_of(t)
        assert reason == "R_G0_CONSTANT", (LO.to_library(t), reason)
        assert most < min_distinct, (LO.to_library(t), most)
    # ema's recursion starts at each coin's listing, so the coins listed during G differ
    # from the rest on some Sundays: a few values, never 10
    assert 1 < g0_of(alone[-1])[1] < min_distinct
    paired = [
        C_("ts_beta", (F_("ret"), F_("mkt_ret"), K_(30))),
        C_("ts_corr", (F_("ret"), F_("btc_ret"), K_(30))),
        C_("ts_resid", (F_("carry_ret"), F_("mkt_ret"), K_(30))),
        C_("ts_cov", (F_("abs_ret"), F_("btc_ret"), K_(30))),
    ]
    for t in paired:
        assert g0_of(t)[0] is None, LO.to_library(t)

    gen = LO.generator(FM.terminal_names(), seed=P.derived_seed(P.DEFAULT_SEED))
    trees, _ = LO.draw_trees(gen, P.DEFAULT_N)
    seen: set = set()
    n_alone = n_pair = n_pair_pass = 0
    reasons: dict = {}
    for raw in trees:
        t = gen.simplify(raw)
        text = LO.to_string(t)
        if text in seen:
            continue
        seen.add(text)
        fields = set(LO.tree_stats(t)["fields"])
        if FM.market_only(fields):
            n_alone += 1
            reason, most = g0_of(t)
            reasons[reason] = reasons.get(reason, 0) + 1
            assert reason is not None, f"{text}: market-wide leaves only, but it passed G0"
            assert most < min_distinct, (text, most)
        elif (
            isinstance(t, A.Call)
            and t.op in PAIR_OPS
            and fields & set(FM.MARKET_WIDE)
            and all(isinstance(a, (A.Field, A.Const)) for a in t.args)
        ):
            n_pair += 1
            n_pair_pass += g0_of(t)[0] is None
    assert n_alone >= 20, n_alone
    assert reasons.get("R_G0_CONSTANT", 0) >= 0.5 * n_alone, reasons
    assert n_pair >= 3 and n_pair_pass >= 1, (n_pair, n_pair_pass)


TRUNCATION_CUTS = ("2022-10-16", "2023-07-30", "2024-01-07")
NOT_BIT_IDENTICAL_OPS = {"decay_linear", "ts_slope", "ts_rsq"}


def test_lint_verdict_holds_when_later_rows_are_cut_off():
    """Every distinct formula of the default draw, evaluated on terminals cut at three
    earlier dates (the real-time case) and on the G-cut terminals: same finite mask, values
    within the PIT tolerance, identical tie-snapped ranks on the G Sundays, and bit-identity
    lost only through decay_linear, ts_slope or ts_rsq (libops.py docstring)."""
    import libfactors as LF

    ix_sundays = N.g_sundays()
    terms = FM.build_terminals(FM.load_inputs())
    ev = LO.evaluator(terms)
    cuts = [pd.Timestamp(c, tz="UTC") for c in TRUNCATION_CUTS]
    evs = [LO.evaluator(FM.build_terminals(FM.load_inputs(c))) for c in cuts]
    gen = LO.generator(FM.terminal_names(), seed=P.derived_seed(P.DEFAULT_SEED))
    trees, _ = LO.draw_trees(gen, P.DEFAULT_N)
    seen: set = set()

    def ranks(a):
        return pd.DataFrame(LF.snap_ties(a)).rank(axis=1).to_numpy()

    checked = not_bit = 0
    for raw in trees:
        t = gen.simplify(raw)
        text = LO.to_string(t)
        if text in seen:
            continue
        seen.add(text)
        try:
            full = LO.evaluate(t, ev=ev)
        except Exception:  # noqa: BLE001 - the probe records these as R_EVAL_ERROR
            continue
        for cut, e in zip(cuts, evs):
            cut_vals = LO.evaluate(t, ev=e)
            k = len(cut_vals)
            assert cut_vals.index.equals(full.index[:k]) and cut_vals.index.max() == cut, text
            a, b = full.to_numpy(dtype=float)[:k], cut_vals.to_numpy(dtype=float)
            checked += 1
            if np.array_equal(a, b, equal_nan=True):
                continue
            not_bit += 1
            assert np.array_equal(np.isfinite(a), np.isfinite(b)), f"{text}: the finite mask moved at {cut.date()}"
            assert np.isclose(b, a, rtol=LF.PIT_RTOL, atol=LF.PIT_ATOL, equal_nan=True).all(), text
            g = full.index[:k].isin(ix_sundays)
            assert np.array_equal(ranks(a[g]), ranks(b[g]), equal_nan=True), f"{text}: a G-Sunday rank moved"
            assert set(LO.tree_stats(t)["ops"]) & NOT_BIT_IDENTICAL_OPS, text
    assert len(seen) > 1500 and checked > 4000, (len(seen), checked)
    assert not_bit < 0.1 * checked, not_bit


def missing_data() -> str | None:
    """Why the suite cannot run on this data folder, or None."""
    import libfactors as LF

    if not any((C.RAW_OHLCV / f"{C.sym_to_file(s)}.parquet").exists() for s in C.UNIVERSE):
        return "no perp bars in the data folder (run ./setup.sh)"
    if not labels.path().is_file():
        return "no label table in the data folder (run ./setup.sh)"
    if not ((LF.out_dir() / "status.parquet").is_file() and LF.has_values()):
        return "no library values in data/library_on_engine (run libfactors.py --compute)"
    return None


if __name__ == "__main__":
    why = missing_data()
    if why:
        print(f"  SKIP  every finder probe test skipped: {why}")
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
    print(f"\n{len(tests) - failed}/{len(tests)} finder probe tests passed")
    raise SystemExit(1 if failed else 0)
