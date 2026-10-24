"""Tests for Factor Finder's field map (factorfinder/fieldmap.py): the generator's terminals.

Usage (from engine/):
    python -B test_finder_fieldmap.py

Reads the data folder read-only (perp and spot bars, funding; the shared-adapter test also
reads open interest through libfactors.load_raw(), to compare); every write goes to a
temporary folder under TMPDIR (point it inside the git-ignored engine/.tmp/). Without the
perp bars every test is skipped (exit 0).

  terminal set        exactly the 21 library-named series (ret, ex_ret,
                      ex_btc_ret, oc_ret, gap, abs_ret, hl_range, log_hl, body,
                      upper_shadow, lower_shadow, clv, tr_pct, parkinson_var, gk_var,
                      rs_var, dvol_chg, amihud, funding, mkt_ret, btc_ret) plus sbasis,
                      perp_range_excess, deriv_share_chg and carry_ret; no vwap, vwap_dev,
                      open-interest, level field, log_amount or premium-based basis; a
                      config with another list is refused; not_listed names exactly the
                      other stationary library series the panels supply
  market-wide         mkt_ret and btc_ret hold one value for every coin with a bar on a
                      date (bit-identical), mkt_ret the mean of ret, btc_ret BTC's ret, and
                      both are NaN where a coin has no return
  shared adapter      fieldmap.read_raw() equals libfactors.load_raw() cell for cell
                      without its open interest, and each library-named terminal is
                      cell-equal to the library Evaluator's series on the library
                      screen's own panels
  point in time       every perp, spot and funding value after a cut overwritten with
                      noise: every terminal value on or before the cut is bit-identical
                      (and the mutation reaches the terminals after it)
  Factorlab terminals each equals its definition, recomputed from the raw bars, and is
                      finite exactly where its inputs are
  units               perp / spot base-volume ratio inside the band test_libfactors pins
                      (deriv_share_chg compares the two venues' dollar volumes)
  one day grid        a spot file stamped at another hour (16:00 UTC) is refused, not
                      reindexed to NaN; the same-day median |sbasis| is under a tenth of
                      its value with spot shifted one day either way
  G cutoff            inputs and terminals end at the 2024-06-09 row
  no label            building the terminals opens only raw bar and funding files: no
                      open-interest file, no factor table, no library output, no
                      catalog, no Finder folder
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from factorfinder import config_io as K
from factorfinder import fieldmap as FM

ENGINE = Path(__file__).resolve().parent
_CACHE: dict = {}


def _base():
    if not _CACHE:
        _CACHE["inputs"] = FM.load_inputs()
        _CACHE["terms"] = FM.build_terminals(_CACHE["inputs"])
    return _CACHE["inputs"], _CACHE["terms"]


def _mutate(inputs: dict, cut: pd.Timestamp, seed: int) -> dict:
    """A copy of the inputs with every perp bar, spot bar and funding value after `cut`
    replaced by positive noise (dates unchanged)."""
    rng = np.random.default_rng(seed)
    raw, spot = {}, {}
    for sym, r in inputs["raw"].items():
        px = r["ohlcv"].copy()
        late = px.index > cut
        px.loc[late] = rng.uniform(0.5, 2.0, size=(int(late.sum()), px.shape[1])) * px.loc[late].to_numpy()[::-1]
        f = r["funding"].copy() if r["funding"] is not None else None
        if f is not None:
            f[f.index > cut] = rng.normal(0, 1e-3, size=int((f.index > cut).sum()))
        raw[sym] = {"ohlcv": px, "funding": f, "oi": None}
    for sym, s in inputs["spot"].items():
        s = s.copy()
        late = s.index > cut
        s.loc[late] = rng.uniform(0.5, 2.0, size=(int(late.sum()), s.shape[1])) * s.loc[late].to_numpy()[::-1]
        spot[sym] = s
    return {"through": inputs["through"], "raw": raw, "spot": spot}


# --------------------------------------------------------------------------- tests


LIBRARY_TERMINALS = (
    "ret",
    "ex_ret",
    "ex_btc_ret",
    "oc_ret",
    "gap",
    "abs_ret",
    "hl_range",
    "log_hl",
    "body",
    "upper_shadow",
    "lower_shadow",
    "clv",
    "tr_pct",
    "parkinson_var",
    "gk_var",
    "rs_var",
    "dvol_chg",
    "amihud",
    "funding",
    "mkt_ret",
    "btc_ret",
)


def test_terminal_set():
    """Exactly the 21 library-named series plus the four Factorlab-only terminals, read
    from eval_config.toml; none of the exclusions; not_listed names exactly the other
    stationary library series the panels supply."""
    cfg = K.load()
    t = FM.terminal_config(cfg)
    assert t["library"] == tuple(cfg["generator"]["terminals"]["library"]) == LIBRARY_TERMINALS
    assert FM.LIBRARY_TERMINALS == LIBRARY_TERMINALS and len(t["library"]) == 21
    assert set(FM.MARKET_WIDE) == {"mkt_ret", "btc_ret"} <= set(t["library"])
    assert t["factorlab"] == ("sbasis", "perp_range_excess", "deriv_share_chg", "carry_ret")
    assert set(t["definitions"]) == set(t["factorlab"])
    names = FM.terminal_names(cfg)
    for bad in (
        "vwap",
        "vwap_dev",
        "open_interest",
        "open_interest_value",
        "log_oi",
        "oi_chg",
        "oi_coin_chg",
        "oi_turnover",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "typical",
        "log_dvol",
        "log_amount",
        "basis",
        "premium",
    ):
        assert bad not in names, bad
    excluded = " ".join(cfg["generator"]["terminals"]["excluded"])
    for word in ("open interest", "vwap", "vwap_dev", "level", "log_amount"):
        assert word in excluded, word
    import libfactors  # noqa: F401 - library on sys.path
    from factorlab.expr import raw_dependencies
    from factorlab.fields import DERIVED_BY_NAME, DERIVED_FIELDS

    for n in t["library"]:
        assert DERIVED_BY_NAME[n].stationary, n
        assert set(raw_dependencies(n)) <= set(FM.PERP_FIELDS), n
    # not_listed: one "name: reason" entry for every other stationary library series the
    # engine's panels supply (the terminals and the exclusions aside), and nothing else
    supplied = {d.name for d in DERIVED_FIELDS if d.stationary and set(raw_dependencies(d.name)) <= set(FM.PERP_FIELDS)}
    entries = cfg["generator"]["terminals"]["not_listed"]
    listed = [e.split(":", 1)[0].strip() for e in entries]
    assert len(listed) == len(set(listed)) and all(":" in e and e.split(":", 1)[1].strip() for e in entries)
    assert set(listed) == supplied - set(t["library"]) - set(FM.EXCLUDED), sorted(
        set(listed) ^ (supplied - set(t["library"]) - set(FM.EXCLUDED))
    )
    assert set(listed) == {"logret", "sq_ret", "up_ret", "down_ret", "close_loc", "mkt_ret_vw", "signed_dvol"}
    # the library's basis is the premium index, which the engine's panels do not supply
    assert "premium" in raw_dependencies("basis") and "premium" not in FM.PERP_FIELDS
    _, terms = _base()
    assert tuple(terms) == names
    assert not any(k.startswith("fwd_") for k in terms)
    cols = {tuple(f.columns) for f in terms.values()}
    assert cols == {tuple(FM.symbols())}, "every terminal covers the 20 perps in universe order"
    # a config that names an excluded field or drops a terminal is refused
    bad = json.loads(json.dumps(cfg))
    bad["generator"]["terminals"]["library"] = list(t["library"]) + ["vwap_dev"]
    for broken in (bad,):
        try:
            FM.terminal_config(broken)
            raise AssertionError("an excluded field was accepted as a terminal")
        except FM.FieldMapError:
            pass
    bad = json.loads(json.dumps(cfg))
    bad["generator"]["terminals"]["factorlab"] = ["sbasis", "carry_ret"]
    try:
        FM.terminal_config(bad)
        raise AssertionError("a terminal list without the four Factorlab terminals was accepted")
    except FM.FieldMapError:
        pass
    # another library list (a 23-name list with not-listed series, or one without amihud) is refused
    other = [
        "ret",
        "logret",
        "oc_ret",
        "gap",
        "ex_ret",
        "ex_btc_ret",
        "abs_ret",
        "sq_ret",
        "up_ret",
        "down_ret",
        "hl_range",
        "log_hl",
        "body",
        "upper_shadow",
        "lower_shadow",
        "close_loc",
        "clv",
        "tr_pct",
        "parkinson_var",
        "gk_var",
        "rs_var",
        "dvol_chg",
        "funding",
    ]
    for lib in (other, [n for n in LIBRARY_TERMINALS if n != "amihud"]):
        bad = json.loads(json.dumps(cfg))
        bad["generator"]["terminals"]["library"] = lib
        try:
            FM.terminal_config(bad)
            raise AssertionError(f"another library list was accepted: {lib}")
        except FM.FieldMapError:
            pass


def test_market_wide_terminals_hold_one_value_per_date():
    """mkt_ret and btc_ret are equal (bit for bit) for every coin with a bar on a date:
    mkt_ret is the mean of ret over the coins with one, btc_ret is BTC's ret; both are NaN
    where a coin has no return, so they never reach a coin before its listing."""
    _, terms = _base()
    ret = terms["ret"].to_numpy(dtype=float)
    has = np.isfinite(ret)
    btc = list(terms["ret"].columns).index("BTC-USDT")
    for name in FM.MARKET_WIDE:
        a = terms[name].to_numpy(dtype=float)
        fin = np.isfinite(a)
        assert not (fin & ~has).any(), f"{name} has a value where the coin has no return"
        rows = 0
        for t in range(len(a)):
            v = a[t][fin[t]]
            if v.size:
                assert (v == v[0]).all(), f"{name}: more than one value on {terms[name].index[t].date()}"
                want = ret[t][has[t]].mean() if name == "mkt_ret" else ret[t, btc]
                assert np.isclose(v[0], want, rtol=1e-12, atol=0), (name, terms[name].index[t].date())
                rows += 1
        assert rows > 800, (name, rows)
    assert np.array_equal(np.isfinite(terms["mkt_ret"].to_numpy()), has)


def test_library_terminals_equal_the_library_evaluator():
    """fieldmap's reads are libfactors.load_raw()'s without open interest (same symbols, same
    bars, same funding, cell for cell), and each library-named terminal is cell-equal to the
    library's own Evaluator on the screen's panels (benchmark BTC-USDT)."""
    import libfactors as LF
    from factorlab.expr import Evaluator

    _, terms = _base()
    cut = FM.g_cutoff()
    lib_raw, own = LF.load_raw(), FM.read_raw()
    assert list(own) == list(lib_raw), "read_raw() and load_raw() cover different symbols"
    for sym, r in lib_raw.items():
        assert own[sym]["oi"] is None, sym
        assert own[sym]["ohlcv"].equals(r["ohlcv"]), f"{sym}: bars differ from load_raw()"
        if r["funding"] is None:
            assert own[sym]["funding"] is None, sym
        else:
            assert own[sym]["funding"].equals(r["funding"]), f"{sym}: funding differs from load_raw()"
    ref = Evaluator(LF.build_panels(lib_raw, through=cut), benchmark=LF.BENCHMARK)
    with np.errstate(all="ignore"):
        for n in FM.terminal_config()["library"]:
            want = ref(n).to_numpy(dtype=float)
            got = terms[n].to_numpy(dtype=float)
            assert want.shape == got.shape and np.array_equal(want, got, equal_nan=True), n
            assert np.isfinite(got).sum() > 0.9 * np.isfinite(ref("close").to_numpy()).sum(), n


def test_future_mutation_leaves_earlier_terminals_bit_identical():
    """Every perp, spot and funding value after a cut is noise; every terminal value on a
    day <= cut is bit-identical, and the mutation reaches every terminal after the cut."""
    inputs, terms = _base()
    for cut, seed in (
        (pd.Timestamp("2022-10-19", tz="UTC"), 1),
        (pd.Timestamp("2023-07-30", tz="UTC"), 2),
        (pd.Timestamp("2024-03-01", tz="UTC"), 3),
    ):
        mut = FM.build_terminals(_mutate(inputs, cut, seed))
        changed = 0
        for n, f in terms.items():
            a, b = f.to_numpy(dtype=float), mut[n].to_numpy(dtype=float)
            k = int(f.index.searchsorted(cut, side="right"))
            assert np.array_equal(a[:k], b[:k], equal_nan=True), f"{n}: a value on or before {cut.date()} changed"
            assert np.isfinite(a[:k]).any(), n
            changed += not np.array_equal(a[k:], b[k:], equal_nan=True)
        assert changed == len(terms), f"the mutation reached only {changed}/{len(terms)} terminals"


def test_factorlab_terminals_follow_their_definitions_and_are_finite_where_inputs_are():
    """sbasis, perp_range_excess, deriv_share_chg and carry_ret recomputed from the raw bars
    and funding: equal, and finite exactly where every input they need is."""
    inputs, terms = _base()
    idx, cols = terms["sbasis"].index, list(terms["sbasis"].columns)

    def wide(get):
        return pd.DataFrame({s: get(s) for s in cols}).reindex(index=idx, columns=cols).to_numpy(dtype=float)

    pc, ph, pl, pv = (wide(lambda s, k=k: inputs["raw"][s]["ohlcv"][k]) for k in ("close", "high", "low", "volume"))
    sc, sh, sl, sv = (wide(lambda s, k=k: inputs["spot"][s][k]) for k in ("close", "high", "low", "volume"))
    fr = wide(lambda s: inputs["raw"][s]["funding"])
    with np.errstate(all="ignore"):
        want = {
            "sbasis": pc / sc - 1.0,
            "perp_range_excess": (ph - pl) / pc - (sh - sl) / sc,
            "deriv_share_chg": None,
            "carry_ret": None,
        }
        share = np.log((pc * pv) / (sc * sv))
        share[~np.isfinite(share)] = np.nan
        dsc = share.copy()
        dsc[1:] = share[1:] - share[:-1]
        dsc[0] = np.nan
        want["deriv_share_chg"] = dsc
        ret = np.full_like(pc, np.nan)
        ret[1:] = pc[1:] / pc[:-1] - 1.0
        want["carry_ret"] = ret - fr
    need = {
        "sbasis": np.isfinite(pc) & np.isfinite(sc) & (sc != 0),
        "perp_range_excess": np.isfinite(pc + ph + pl + sc + sh + sl) & (pc != 0) & (sc != 0),
        "deriv_share_chg": None,
        "carry_ret": np.isfinite(ret) & np.isfinite(fr),
    }
    ok = np.isfinite(share)
    need["deriv_share_chg"] = np.r_[np.zeros((1, ok.shape[1]), bool), ok[1:] & ok[:-1]]
    for n in FM.FACTORLAB_TERMINALS:
        got = terms[n].to_numpy(dtype=float)
        assert np.array_equal(np.isfinite(got), need[n]), f"{n}: finite where an input is missing, or the reverse"
        assert np.allclose(got[need[n]], want[n][need[n]], rtol=1e-12, atol=1e-15), n
        assert need[n].mean() > 0.9, n
    # spot and funding missing on some perp days? then the terminal is missing there too
    assert (~np.isfinite(terms["carry_ret"].to_numpy()) | np.isfinite(fr)).all()


def test_perp_spot_volume_ratio_stays_in_the_pinned_band():
    """deriv_share_chg compares the two venues' dollar volumes, so their units must match:
    perp / spot base volume stays in test_libfactors' 1-50 band for every coin and year."""
    inputs, _ = _base()
    for sym, r in inputs["raw"].items():
        j = r["ohlcv"]["volume"] / inputs["spot"][sym]["volume"].reindex(r["ohlcv"].index)
        by_year = j.groupby(j.index.year).median()
        assert ((by_year > 1.0) & (by_year < 50.0)).all(), (sym, by_year.round(2).to_dict())


def test_spot_bars_share_the_perp_day_grid():
    """A spot file stamped 16:00 UTC (the start of a UTC+8 day) is refused before it is
    reindexed onto the perp grid, where it would silently become NaN; spot is aligned to
    the same UTC day: median |sbasis| is under a tenth of its value with spot shifted one
    day earlier or later."""
    inputs, terms = _base()
    sym = "BTC-USDT"
    for hours in (16, -8):
        bad = dict(inputs, spot=dict(inputs["spot"]))
        bad["spot"][sym] = inputs["spot"][sym].set_axis(inputs["spot"][sym].index + pd.Timedelta(hours=hours))
        try:
            FM.build_terminals(bad)
            raise AssertionError(f"spot bars stamped {hours:+d}h off midnight were accepted")
        except ValueError as exc:
            assert "00:00" in str(exc) and sym in str(exc), exc
    naive = dict(inputs, spot=dict(inputs["spot"]))
    naive["spot"][sym] = inputs["spot"][sym].tz_localize(None)
    try:
        FM.build_terminals(naive)
        raise AssertionError("spot bars without a UTC index were accepted")
    except ValueError:
        pass
    same = float(np.nanmedian(np.abs(terms["sbasis"].to_numpy(dtype=float))))
    for days in (-1, 1):
        moved = dict(inputs, spot={s: f.set_axis(f.index + pd.Timedelta(days=days)) for s, f in inputs["spot"].items()})
        off = float(np.nanmedian(np.abs(FM.build_terminals(moved)["sbasis"].to_numpy(dtype=float))))
        assert same < 0.1 * off, (same, days, off)
    assert 0 < same < 0.002, same


def test_inputs_and_terminals_end_at_the_g_cutoff():
    """No input row and no terminal row is dated after the 2024-06-09 row."""
    inputs, terms = _base()
    cut = pd.Timestamp("2024-06-09", tz="UTC")
    assert FM.g_cutoff() == cut == inputs["through"]
    for sym, r in inputs["raw"].items():
        assert r["ohlcv"].index.max() == cut, sym
        assert r["funding"].index.max() <= cut and r["oi"] is None, sym
        assert inputs["spot"][sym].index.max() <= cut, sym
    for n, f in terms.items():
        assert f.index.max() == cut and f.index.tz is not None, n
        assert f.index.min() == min(r["ohlcv"].index.min() for r in inputs["raw"].values())
    first = pd.Series({s: r["ohlcv"].index.min() for s, r in inputs["raw"].items()})
    for n, f in terms.items():
        for s in f.columns:
            assert not f.loc[f.index < first[s], s].notna().any(), f"{n} has a value before {s}'s first bar"


def test_no_label_column_is_ever_loaded():
    """Building the terminals opens raw bar and funding files only: an audit hook (Python
    opens) and wrapped pyarrow/pandas readers (pyarrow opens files in C++) see no
    open-interest file, no factor table, no library output or catalog, no Finder folder and
    no fwd_* column."""
    code = r"""
import json, sys
seen = []
def hook(event, args):
    if event == "open":
        seen.append(str(args[0]))
    elif event == "sqlite3.connect":
        seen.append("sqlite:" + str(args[0]))
sys.addaudithook(hook)
import pyarrow.parquet as pq, pandas as pd
cols_seen = []
_rt, _rp = pq.read_table, pd.read_parquet
def rt(path, *a, **k):
    cols_seen.append((str(path), k.get("columns")))
    return _rt(path, *a, **k)
def rp(path, *a, **k):
    cols_seen.append((str(path), k.get("columns")))
    return _rp(path, *a, **k)
pq.read_table, pd.read_parquet = rt, rp
from factorfinder import fieldmap as FM
t = FM.build_terminals()
print(json.dumps({"open": seen, "parquet": [[p, c] for p, c in cols_seen], "keys": list(t)}))
"""
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=ENGINE, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    data = str(Path(C.DATA).resolve()) + os.sep
    paths = [p for p in out["open"] + [p for p, _ in out["parquet"]] if str(Path(p).resolve()).startswith(data)]
    assert paths, "no data file was seen: the hooks did not fire"
    allowed = ("raw/ohlcv/", "raw/spot_ohlcv/", "raw/funding/", "raw/funding_binanceusdm/")
    rels = [str(Path(p).resolve())[len(data) :] for p in paths]
    for rel in rels:
        assert rel.startswith(allowed), f"building the terminals opened {rel}"
    assert not any(rel.startswith("raw/open_interest") for rel in rels), "an open-interest file was opened"
    assert any(rel.startswith("raw/funding/") for rel in rels) and any(rel.startswith("raw/ohlcv/") for rel in rels)
    repo, venv = str(ENGINE.parent.resolve()), str((ENGINE / ".venv").resolve())
    for p in out["open"]:
        if p.startswith("sqlite:"):
            raise AssertionError(f"building the terminals opened a database: {p}")
        if not p.startswith(repo) or p.startswith(venv):
            continue
        rel = p[len(repo) + 1 :]
        for bad in ("library/catalog", "screen_g", "library_on_engine"):
            assert bad not in rel, rel
    for p, c in out["parquet"]:
        assert c is None or not any(str(x).startswith("fwd_") for x in c), (p, c)
    assert not any(k.startswith("fwd_") for k in out["keys"])


if __name__ == "__main__":
    if not any((C.RAW_OHLCV / f"{C.sym_to_file(s)}.parquet").exists() for s in C.UNIVERSE):
        print("  SKIP  every finder field-map test skipped: no perp bars in the data folder (run ./setup.sh)")
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
    print(f"\n{len(tests) - failed}/{len(tests)} finder field-map tests passed")
    raise SystemExit(1 if failed else 0)
