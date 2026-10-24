"""Tests of the incumbent provider (factorfinder/incumbents.py).

Usage:
    python -B test_incumbents.py

The suite reads no data folder of the repository. At the start it sets FACTORLAB_DATA to a new
folder under the git-ignored engine/.tmp/ and removes that folder at the end. It writes the
synthetic demo market there, builds the label table, registers three tests (a formula and two
values files; one of them is withdrawn) and writes a small AlphaForge zoo with two active
formulas and one formula that left the active list.

The properties:
    1. the sources and the counts: your registered tests, the active zoo formulas and the three
       risk proxies, each on a UTC date index and the 20 engine symbols, with the values of their
       own code
    2. a withdrawn test is not an incumbent (and not a member of the probe's novelty index)
    3. a zoo formula that left the active list is not an incumbent, but it is a member of the
       probe's novelty index. A zoo file of another format than 3 stops with ZooFormatError. It
       stops the factor lab with exit 1.
    4. an empty set reports n_incumbents = 0 and no correlation, never 0.0; the factor lab shows it;
       a registry row without a count (null) does not show it
    5. without the libfactors values, library_corr gives None; the factor lab keeps the largest
       absolute library correlation
    6. with libfactors values (written by libfactors' own writer), library_corr equals the exact
       brute-force correlation with each library member on the G Sundays, for a frame on a UTC
       nanosecond or millisecond index (the factor-lab pivot), on a naive index and with the
       columns in another order; the factor lab gives the same number
    7. the provider writes nothing
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

ENGINE = Path(__file__).resolve().parent
(ENGINE / ".tmp").mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="test_incumbents_", dir=ENGINE / ".tmp"))
atexit.register(shutil.rmtree, TMP, True)
os.environ["FACTORLAB_DATA"] = str(TMP / "data")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import book  # noqa: E402
import config as C  # noqa: E402
import demo_data  # noqa: E402
import labels  # noqa: E402
import ledger as LG  # noqa: E402
import libfactors as lf  # noqa: E402
import trials as T  # noqa: E402
import userfactor as uf  # noqa: E402
from factorfinder import incumbents as INC  # noqa: E402

DATA = Path(C.DATA)
SYMS = [C.sym_to_file(s) for s in C.UNIVERSE]
FORMULA = "ts_mean(clv, 7)"
QUIET = lambda *a, **k: None  # noqa: E731
_STATE: dict = {}


# --------------------------------------------------------------------------- the fixture


def values_csv(rows: dict, name: str) -> Path:
    """A CSV values file: rows {sunday: 20 values}."""
    lines = ["date,symbol,value"]
    for s in sorted(rows):
        for sym, v in zip(SYMS, rows[s]):
            lines.append(f"{s},{sym},{'' if not np.isfinite(v) else repr(float(v))}")
    d = Path(tempfile.mkdtemp(prefix="vf_", dir=TMP))
    p = d / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def world() -> dict:
    """The demo market, the label table, three registered tests (FT-0003 withdrawn) and a zoo."""
    if _STATE:
        return _STATE
    m = demo_data.market(demo_data.SEED)
    demo_data.write_raw(DATA, m)
    labels.build()
    last = pd.Timestamp(m["close"].index[-1])
    LG.CLOCK = lambda: datetime(last.year, last.month, last.day, 10, tzinfo=UTC) + pd.Timedelta(days=1)
    sundays = [d for d in m["close"].index if d.dayofweek == 6]
    rng = np.random.default_rng(7)
    rows_a = {str(d.date()): rng.normal(size=len(SYMS)) for d in sundays}
    rows_b = {str(d.date()): rng.normal(size=len(SYMS)) for d in sundays}
    spec = book.load_spec()
    for inp, name in (
        ({"formula": FORMULA, "source": None}, None),
        ({"values_file": str(values_csv(rows_a, "a.csv"))}, "file-a"),
        ({"values_file": str(values_csv(rows_b, "b.csv"))}, "file-b"),
    ):
        assert T.register(inp, spec, name=name, note=None, log=QUIET) == 0
    assert T.withdraw("FT-0003", "not needed", log=QUIET) == 0
    # the zoo: af_01 and af_02 active, af_03 left the active list; the table has all three
    research = Path(C.RESEARCH)
    research.mkdir(parents=True, exist_ok=True)
    zoo = {
        "format": 3,
        "factors": [
            {"name": "af_01", "expr": "ts_mean(returns, 5)", "active": True, "left_on": None},
            {"name": "af_02", "expr": "ts_std(returns, 20)", "active": True, "left_on": None},
            {"name": "af_03", "expr": "ts_mean(range, 10)", "active": False, "left_on": "2026-01-04"},
        ],
    }
    (research / INC.ZOO_LIBRARY).write_text(json.dumps(zoo), encoding="utf-8")
    idx = m["close"].index
    long = pd.DataFrame(
        {
            "date": np.repeat(idx.as_unit("ms"), len(SYMS)),
            "symbol": np.tile(SYMS, len(idx)),
            **{f"af_0{i}": rng.normal(size=len(idx) * len(SYMS)) for i in (1, 2, 3)},
        }
    )
    long.to_parquet(research / INC.ZOO_TABLE, index=False)
    _STATE.update(market=m, rows_a=rows_a, rows_b=rows_b, table=long, last=last)
    return _STATE


def tree(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------- tests


def test_sources_and_counts():
    """A formula test, a values-file test, 2 active zoo formulas and 3 risk proxies, in the order
    user, zoo, proxy; each a UTC daily frame on the 20 engine symbols with the values of its code."""
    w = world()
    members = INC.incumbents()
    names = [x.name for x in members]
    assert names == [
        "user:FT-0001",
        "user:FT-0002",
        "zoo:af_01",
        "zoo:af_02",
        "proxy:pit_symbol_tilt",
        "proxy:btc_beta_90d",
        "proxy:size_proxy",
    ], names
    assert INC.counts(members) == {"user": 2, "zoo": 2, "proxy": 3}
    assert [x.source for x in members] == ["user", "user", "zoo", "zoo", "proxy", "proxy", "proxy"]
    for x in members:
        assert list(x.values.columns) == SYMS and str(x.values.index.tz) == "UTC", x.name
        assert x.values.index.is_monotonic_increasing and x.values.dtypes.eq(np.float64).all()
    by = {x.name: x.values for x in members}
    # the formula: its values on the full OKX panels
    okx = uf.load_source("okx")
    full = uf.full_values(FORMULA, okx["panels"])
    got = by["user:FT-0001"].reindex(okx["panels"]["close"].index.tz_convert("UTC").as_unit("ns"))
    assert np.array_equal(got.to_numpy(), full, equal_nan=True)
    # the values file: its history values on the Sundays (a coin without a bar on that day has none)
    vf = by["user:FT-0002"]
    assert (vf.index.dayofweek == 6).all() and len(vf) > 200
    for d in vf.index:
        row = vf.loc[d].to_numpy()
        ok = np.isfinite(row)
        # the values file reader parses the CSV text: equal to the last bits of the float
        assert ok.any() and np.allclose(row[ok], np.asarray(w["rows_a"][str(d.date())])[ok], rtol=1e-12, atol=0), d
    assert np.isfinite(vf.iloc[-1].to_numpy()).all()
    # the zoo: the columns of the table
    t = w["table"]
    piv = t.pivot(index="date", columns="symbol", values="af_02").reindex(columns=SYMS)
    assert np.array_equal(by["zoo:af_02"].to_numpy(), piv.to_numpy())
    # the proxies: userfactor.risk_proxies on the OKX panels
    panels = lf.build_panels(lf.load_raw())
    regs = uf.risk_proxies(panels, SYMS, panels["close"].index)
    for key, name in zip(lf.REG_KEYS, INC.PROXIES):
        assert np.array_equal(by[f"proxy:{name}"].to_numpy(), regs[key], equal_nan=True), name
    # a subset of the sources, a cut at a date, an unknown source
    only = INC.incumbents(sources=("proxy",), through=w["last"].date() - pd.Timedelta(days=30))
    assert [x.source for x in only] == ["proxy"] * 3
    assert all(x.values.index.max() <= w["last"] - pd.Timedelta(days=30) for x in only)
    try:
        INC.incumbents(sources=("library",))
        raise AssertionError("an unknown source was accepted")
    except ValueError:
        pass
    # the numbers: rank frames and the mean within-date Spearman correlation
    sundays = pd.DatetimeIndex([d for d in w["market"]["close"].index if d.dayofweek == 6])
    frames = INC.rank_frames(members, sundays)
    assert set(frames) == set(names) and all(f.index.equals(sundays) for f in frames.values())
    v = by["user:FT-0001"].reindex(sundays)
    near = INC.nearest(v, {k: f for k, f in frames.items() if k != "user:FT-0001"})
    want = {}
    for k, f in frames.items():
        if k == "user:FT-0001":
            continue
        cs = []
        for d in sundays:
            a, b = v.loc[d].to_numpy(), f.loc[d].to_numpy()
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 1 and np.unique(a[ok]).size > 1 and np.unique(b[ok]).size > 1:
                cs.append(spearmanr(a[ok], b[ok]).statistic)
        if len(cs) >= INC.MIN_COMMON_DATES:
            want[k] = float(np.mean(cs))
    best = max(want, key=lambda k: abs(want[k]))
    assert near["name"] == best and abs(near["corr"] - want[best]) < 1e-9, (near, best, want[best])
    assert near["n_incumbents"] == len(want) == 6


def test_withdrawn_tests_are_not_incumbents():
    """FT-0003 is withdrawn: neither the provider nor the probe's novelty index holds it."""
    world()
    rec = T.read()
    assert [t["test_id"] for t in T.registered(rec)] == ["FT-0001", "FT-0002", "FT-0003"]
    assert T.registered(rec)[2]["withdrawn"] is not None
    assert [x.name for x in INC.user_members()] == ["user:FT-0001", "user:FT-0002"]
    from factorfinder import novelty as N

    ix = N.build_index(sources=("user", "zoo", "proxy"))
    users = [i for i, s in zip(ix.ids, ix.sources) if s == "user"]
    assert users == ["user:FT-0001", "user:FT-0002"], users
    assert ix.counts["user"] == 2 and ix.counts["library"] == 0


def test_inactive_zoo_formulas_are_not_incumbents():
    """af_03 left the active list: it is not an incumbent, but it is a member of the probe index."""
    world()
    assert [x.name for x in INC.zoo_members()] == ["zoo:af_01", "zoo:af_02"]
    assert [x.name for x in INC.zoo_members(active_only=False)] == ["zoo:af_01", "zoo:af_02", "zoo:af_03"]
    from factorfinder import novelty as N

    ix = N.build_index(sources=("user", "zoo", "proxy"))
    zoo = sorted(i for i, s in zip(ix.ids, ix.sources) if s == "zoo")
    assert zoo == ["af_01", "af_02", "af_03"], zoo
    assert sorted(i for i, s in zip(ix.ids, ix.sources) if s == "proxy") == sorted(INC.PROXIES)


def test_zoo_of_another_format_is_refused():
    """The provider reads only a zoo file of format 3. A file of another format, or with no format,
    stops with ZooFormatError. The message names the file and the two formats. The factor lab
    stops with exit 1 and that message, before it computes."""
    world()
    lib = Path(C.RESEARCH) / INC.ZOO_LIBRARY
    saved = lib.read_bytes()
    zoo = json.loads(saved)
    assert INC.ZOO_FORMAT == 3 and INC.check_zoo_format(zoo) is zoo
    assert INC.zoo_active_names(zoo) == ["af_01", "af_02"]
    try:
        for fmt in (2, 4, "3", None, True):
            bad = {**zoo, "format": fmt} if fmt is not None else {k: v for k, v in zoo.items() if k != "format"}
            lib.write_text(json.dumps(bad), encoding="utf-8")
            for call in (
                INC.zoo_members,
                lambda: INC.zoo_members(active_only=False),
                lambda bad=bad: INC.zoo_active_names(bad),
            ):
                try:
                    call()
                except INC.ZooFormatError as exc:
                    text = str(exc)
                    assert INC.ZOO_LIBRARY in text and f"format {fmt!r}" in text, text
                    assert "reads only format 3" in text, text
                else:
                    raise AssertionError(f"a zoo file with the format {fmt!r} was read")
        for bad in ([], "zoo", None):
            try:
                INC.check_zoo_format(bad)
            except INC.ZooFormatError:
                pass
            else:
                raise AssertionError(f"{bad!r} was accepted as a zoo")
        # the factor lab: exit 1 with one line on stderr, before it computes a spec or writes the registry
        import factor_lab as FLAB

        for text, want in (
            (json.dumps({**zoo, "format": 2}), "has the format 2;"),
            ("{not json", "is not a JSON file"),
        ):
            lib.write_text(text, encoding="utf-8")
            reg = Path(C.RESEARCH) / FLAB.REGISTRY.name
            before = reg.read_bytes() if reg.is_file() else None
            for argv in ([], ["--no-write"]):
                out, err = io.StringIO(), io.StringIO()
                with (
                    mock.patch.object(FLAB, "evaluate", side_effect=AssertionError("evaluate ran")),
                    contextlib.redirect_stdout(out),
                    contextlib.redirect_stderr(err),
                ):
                    try:
                        FLAB.main(argv)
                    except SystemExit as exc:
                        code = exc.code
                    else:
                        code = 0
                line = f"factor_lab: the zoo file data/research/{INC.ZOO_LIBRARY} {want}"
                assert code == 1 and err.getvalue().startswith(line), (argv, code, err.getvalue())
                assert err.getvalue().count("\n") == 1 and out.getvalue() == "", (err.getvalue(), out.getvalue())
            assert (reg.read_bytes() if reg.is_file() else None) == before, "the factor lab changed its registry"
    finally:
        lib.write_bytes(saved)
    assert [x.name for x in INC.zoo_members()] == ["zoo:af_01", "zoo:af_02"], "the format-3 file again"
    INC.check_zoo_file()  # a format-3 file passes


_EMPTY_CHILD = r"""
import json
import numpy as np, pandas as pd
from factorfinder import incumbents as INC
m = INC.incumbents()
idx = pd.date_range("2024-01-07", periods=40, freq="7D", tz="UTC")
v = pd.DataFrame(np.random.default_rng(1).normal(size=(40, 20)), index=idx, columns=INC.symbols())
print(json.dumps({"members": len(m), "counts": INC.counts(m), "near": INC.nearest(v, INC.rank_frames(m, idx)),
                  "lib": INC.library_corr(v)}))
"""


def test_empty_set_reports_zero_members():
    """No ledger, no zoo, no bars: no incumbent, n_incumbents 0 and no correlation (not 0.0); the
    factor lab writes the line for an empty set."""
    empty = TMP / "empty" / "data"
    empty.mkdir(parents=True)
    r = subprocess.run(
        [sys.executable, "-B", "-c", _EMPTY_CHILD],
        cwd=ENGINE,
        env=dict(os.environ, FACTORLAB_DATA=str(empty)),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-1500:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["members"] == 0 and out["counts"] == {"user": 0, "zoo": 0, "proxy": 0}
    assert out["near"] == {"corr": None, "name": None, "n_incumbents": 0}, out["near"]
    assert out["lib"] is None
    # the factor lab with an empty set: every row n_incumbents 0, no correlation, and the line
    world()
    import factor_lab as FLAB

    # the library number is the largest absolute correlation (library_corr's max_abs), not the signed one
    lib = {"max_abs": 0.27, "corr": -0.27, "nearest": "CF00001", "n_members": 9}
    with (
        __import__("contextlib").redirect_stdout(__import__("io").StringIO()),
        mock.patch.object(INC, "library_corr", return_value=lib),
    ):
        res = FLAB.evaluate(FLAB.default_grid(1), members=[])
    assert len(res) == 1 and int(res["n_incumbents"].iloc[0]) == 0
    assert res["max_incumbent_corr"].iloc[0] is None or not np.isfinite(float(res["max_incumbent_corr"].iloc[0]))
    assert res["closest_incumbent"].iloc[0] is None
    assert res["max_library_corr"].iloc[0] == 0.27 and res["closest_library"].iloc[0] == "CF00001"
    text = FLAB.render(FLAB.assess(res, res.assign(run_date="2026-10-01")), res.assign(run_date="2026-10-01"))
    assert "No incumbent: no registered test, no active zoo formula and no risk proxy." in text, text
    assert "Numbers only." in text and "+0.00 (0)" not in text and "-0.27" not in text


def test_row_without_an_incumbent_count():
    """A registry row with no n_incumbents (null) shows "-" in the report and not the line for an
    empty incumbent set; a row with a count of 0 shows the line."""
    import factor_lab as FLAB

    row = {"run_date": "2026-10-20", "factor": "dif_signed_corr5_w90_t1.5_d0.7_l5", "params": "{}", "weeks": 80}
    row.update(ic=0.01, t=1.1, p=0.27, persistence=0.4, turnover=0.6, ic_per_turnover=0.02, coverage=1.0)
    row.update(first_seen=True)
    reg = FLAB._registry_frame(pd.DataFrame([row]))
    assert reg["n_incumbents"].isna().all() and list(reg.columns) == FLAB.REGISTRY_COLUMNS
    text = FLAB.render(FLAB.assess(reg.copy(), reg), reg)
    assert "No incumbent" not in text and "- (-)" in text, text
    reg0 = FLAB._registry_frame(pd.DataFrame([{**row, "n_incumbents": 0}]))
    text0 = FLAB.render(FLAB.assess(reg0.copy(), reg0), reg0)
    assert "No incumbent: no registered test, no active zoo formula and no risk proxy." in text0, text0


def test_library_corr_is_none_without_values():
    """Without data/library_on_engine (the libfactors values), library_corr gives None."""
    w = world()
    assert not (DATA / "library_on_engine").exists()
    v = w["market"]["close"].pct_change(fill_method=None)
    v.index = v.index.tz_convert("UTC")
    assert INC.library_corr(v) is None
    try:
        INC._library_index()
        raise AssertionError("an index without the library values was built")
    except FileNotFoundError:
        pass


# a few library-language formulas as library members; CF90001 is the formula of the test FT-0001
LIB_FORMULAS = {
    "CF90001": FORMULA,
    "CF90002": "ts_std(ret, 20)",
    "CF90003": "ts_mean(hl_range, 10)",
    "CF90004": "ts_mean(abs_ret, 30)",
    "CF90005": "ts_mean(gap, 5)",
}


def _write_library_values() -> list[str]:
    """data/library_on_engine/: the Sunday values of LIB_FORMULAS on the OKX panels (libfactors'
    own writer, the layout of --compute) and a status.parquet that lists them as computed and PIT
    proven. Returns the ids."""
    panels = uf.load_source("okx")["panels"]
    idx = panels["close"].index
    pos = lf.sunday_positions(idx)
    ids = list(LIB_FORMULAS)
    vals = np.stack([uf.full_values(e, panels)[pos] for e in LIB_FORMULAS.values()])
    shard_of = lf.write_values(ids, vals, idx[pos], list(panels["close"].columns))
    status = pd.DataFrame(
        {
            "factor_id": ids,
            "family": "fixture",
            "compute": "ok",
            "vwap_proxy": False,
            "pit_finite_t1": np.int32(1000),
            "pit_diff_cells_t1": np.int32(0),
            "pit_error_t1": "",
            "values_shard": [shard_of[i] for i in ids],
        }
    )
    status.to_parquet(lf.out_dir() / "status.parquet", index=False)
    return ids


def _expected(ix, frame: pd.DataFrame) -> dict:
    """The exact reference: the frame's rows on the G Sundays, matched by the date text (so neither
    the time zone nor the unit of the index matters) and by the symbol names, then
    novelty.brute_force over every member."""
    from factorfinder import novelty as N

    by_day = {str(pd.Timestamp(d).date()): i for i, d in enumerate(frame.index)}
    rows = np.full((len(ix.sundays), len(ix.symbols)), np.nan)
    for t, d in enumerate(ix.sundays):
        i = by_day.get(str(d.date()))
        if i is not None:
            rows[t] = frame.iloc[i].reindex(ix.symbols).to_numpy(dtype=np.float64)
    bf = N.brute_force(ix, rows)
    k = int(np.nanargmax(np.abs(bf)))
    exact = N.max_corr(ix, rows)
    assert exact["nearest"] == ix.ids[k] and abs(exact["max_abs"] - abs(bf[k])) < 1e-6, (exact, bf)
    return {"max_abs": float(abs(bf[k])), "corr": float(bf[k]), "nearest": ix.ids[k], "n_members": ix.size}


def _same(got: dict | None, want: dict) -> bool:
    return (
        got is not None
        and got["nearest"] == want["nearest"]
        and got["n_members"] == want["n_members"]
        and abs(got["max_abs"] - want["max_abs"]) < 1e-6
        and abs(got["corr"] - want["corr"]) < 1e-6
    )


def test_library_corr_equals_the_novelty_index():
    """With library values on disk: library_corr is the exact largest |corr| with a library member
    on the G Sundays, for each form of the date index that a caller gives (UTC ns, UTC ms, naive,
    Sundays only, columns in another order, a frame that ends inside G); the factor lab writes the
    same number for its spec."""
    world()
    root = DATA / "library_on_engine"
    assert not root.exists()
    try:
        ids = _write_library_values()
        INC._LIBRARY_INDEX.clear()
        ix = INC._library_index()
        assert sorted(ix.ids) == ids and ix.size == len(ids), ix.ids
        assert set(ix.sources) == {"library"} and str(ix.sundays.max().date()) == "2024-06-09"
        assert INC._library_index() is ix, "the index is built once for each input state"
        by = {x.name: x.values for x in INC.user_members()}
        v = by["user:FT-0001"]  # daily, UTC, nanoseconds
        want = _expected(ix, v)
        assert want["nearest"] == "CF90001" and want["max_abs"] > 0.999 and want["corr"] > 0, want
        ms = v.copy()
        ms.index = v.index.as_unit("ms")  # the unit of the factor-lab pivot
        naive = v.copy()
        naive.index = v.index.tz_localize(None)  # the store grid of the miner
        aware = naive.copy()
        aware.index = pd.DatetimeIndex(naive.index).tz_localize("UTC")  # alphaforge_bridge._aware_frame
        sundays_ms = ms[ms.index.dayofweek == 6]
        flipped = ms[list(reversed(SYMS))]
        for name, frame in (("ns", v), ("ms", ms), ("naive", naive), ("aware", aware)):
            assert _same(INC.library_corr(frame), want), name
        assert _same(INC.library_corr(sundays_ms), want), "Sundays only"
        assert _same(INC.library_corr(flipped), want), "columns in another order"
        # a frame that ends inside G, and a values file (random values on the Sundays)
        part = ms[ms.index <= pd.Timestamp("2023-09-30", tz="UTC")]
        assert _same(INC.library_corr(part), _expected(ix, part)), "a frame that ends inside G"
        vf = by["user:FT-0002"]
        got = INC.library_corr(vf)
        assert _same(got, _expected(ix, vf)) and got is not None and got["max_abs"] < 0.5, got
        # the factor lab: its number is library_corr of the spec's Sunday pivot (UTC ms)
        import data_io
        import factor_lab as FLAB
        from diffusion import build as dbuild

        specs = FLAB.default_grid(1)
        with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
            res = FLAB.evaluate(specs, members=[])
            prices = data_io.close_matrix()
            diff = dbuild(prices, specs)
        col = f"{specs[0].name()}_rank"
        wk = diff[diff.date.map(lambda d: pd.Timestamp(d).weekday() == 6)]
        piv = wk.pivot_table(index="date", columns="symbol", values=col).reindex(columns=list(prices.columns))
        ref = _expected(ix, piv)
        assert len(res) == 1 and res["closest_library"].iloc[0] == ref["nearest"], (res, ref)
        assert abs(float(res["max_library_corr"].iloc[0]) - ref["max_abs"]) < 1e-6, (res, ref)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        INC._LIBRARY_INDEX.clear()
    assert INC.library_corr(v) is None, "without the values again: None"


def test_provider_writes_nothing():
    """Every function of the provider leaves the data folder byte-identical."""
    world()
    before = tree(DATA)
    members = INC.incumbents()
    sundays = pd.DatetimeIndex([d for d in members[0].values.index if d.dayofweek == 6])
    frames = INC.rank_frames(members, sundays)
    INC.nearest(frames[members[0].name], frames)
    INC.library_corr(members[0].values)
    INC.zoo_members(active_only=False)
    assert tree(DATA) == before, "the provider changed the data folder"
    src = (ENGINE / "factorfinder" / "incumbents.py").read_text(encoding="utf-8")
    for bad in ("to_parquet", "write_text", "write_bytes", "guard_write", ".open(", "import factor_lab"):
        assert bad not in src, bad
    assert "import alphaforge_bridge" not in src


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} incumbent tests passed")
    raise SystemExit(1 if failed else 0)
