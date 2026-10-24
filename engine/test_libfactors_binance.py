"""Integrity tests for libfactors.py's binance source (Binance public data, as data only).

Script-style like the other engine suites:  python -B test_libfactors_binance.py

Reads the data folder read-only, the git-ignored Binance panel (library/data/binance_um_1d;
the tests that need it say "skipped" when it is absent) and the outputs in
data/library_on_engine/ and data/library_on_engine/binance/ when they exist. The weekly
values (values/ in both folders) are git-ignored too: the checks that read them say
"skipped" when a folder has no shard. Without the raw bars, the label table or the
catalog, every test is skipped (exit 0). The only writes go to temporary folders under
TMPDIR (point it inside the git-ignored engine/.tmp/).

  okx unchanged        okx is the default source with its own folder, panels and
                       classification; a sample of the OKX screen's stored rows is
                       recomputed with this code and matches; binance writes never touch
                       the OKX files (hashes), and each source's writer refuses the other's
                       folder, a symlinked folder and a protected folder
  symbol mapping       BTCUSDT -> BTC-USDT for the 20 coins, in the engine's order; a panel
                       with a missing or an extra coin, or a missing field, is refused; the
                       benchmark is Binance's BTCUSDT
  one venue            the binance adapter and the factor evaluation run with every engine
                       data reader disabled, and every panel cell is the Binance file's,
                       except the cells the cleaning rules set missing
  cleaning             the three label-free rules (open interest <= 0; a base volume whose
                       average price leaves the bar's low-high range, for volume and for
                       taker_buy_volume) clear exactly their cells, per cell and point in
                       time; on the panel they clear exactly the pinned archive errors
                       (2022-03-07 for the 17 coins then listed, BTC 2024-07-10 and
                       2024-07-12, ADA 2024-07-14; DOT 2023-11-14; XRP 2023-11-30), and the
                       manifest lists the same cells
  cleaning freeze      the pinned hash equals the hash of the rules in the code; a changed
                       field, text, tolerance or code moves the hash and stops the compute;
                       the manifest records the freeze. The hash uses the tokens of the code.
                       The token list does not depend on the Python version.
  appends              the new closed days that merge_panels.py --append added: read from
                       appends.jsonl, each append consecutive to the one before it; the
                       panel ends on the last append's last bar, and each appended bar was
                       closed when the append ran
  field conventions    UTC open-time daily bars, all closed; funding = the sum of the
                       settlements in [D, D + 1) (equal to the engine's own Binance prints);
                       the downloader's rules (label and closed left; sum for funding, last
                       snapshot for metrics; premium = the premium kline's close; metrics
                       create_time read as UTC); the metrics snapshot sits at the bar's end;
                       vwap is a true VWAP here and collapses onto close on OKX
  future mutation      every Binance value after a cut overwritten with noise: every factor
                       value on dates <= cut unchanged, for a sample covering every library
                       operator and all 16 raw fields, at the three PIT cuts
  sealed I zone        a label after 2024-06-09 raises LabelLeak; with every label and every
                       Binance and OKX bar after it mutated, the screen output of Binance
                       values is byte-identical (and a change before it is not)
  new family           never holds a factor the OKX screen screened (rule and outputs)
  cross-check          no BH, no q-value, no column of a q-level set or a signal; its statistics are
                       factor_stats on the stored Binance values; on G the OKX-panel funding
                       is the Binance panel's on every coin-day (so funding readers share
                       that input), and the manifest says so beside the mechanical-agreement
                       caveat, with the agreement recomputed without the funding readers
  outputs              status / screen_g / crosscheck / manifest reconcile; panel fingerprints
                       are names and hashes only; no path or private string; no verdict
                       column; values/ git-ignored; the sensitivity notes are the
                       module's own
  numbers only         the outputs, the manifest texts and --status have no verdict, tier,
                       stage or gate word; --status prints counts and the states of the
                       code and the inputs only
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import data_io
import libfactors as L
from labels import path as label_path

_CACHE: dict = {}
SRC = "binance"


def _have_panel() -> bool:
    return L.BINANCE_PANEL_DIR.is_dir() and any(L.BINANCE_PANEL_DIR.glob("*.csv.gz"))


def _demo_panel() -> bool:
    """True when the panel is the synthetic demo panel (demo_data.py): it has no archive errors."""
    import demo_data

    return demo_data.is_demo_panel(L.BINANCE_PANEL_DIR)


# "(skipped: ...)" is the line that engine/run_tests.py counts as a skipped test
def _skipped(what: str) -> None:
    print(f"    (skipped: {what}: no Binance panel in library/data/)")


def _no_values(what: str) -> None:
    print(f"    (skipped: {what}: no Binance values/ (git-ignored; written by --source binance --compute))")


def _bin():
    if "braw" not in _CACHE:
        _CACHE["braw"] = L.load_raw_binance()
        _CACHE["bpanels"] = L.build_panels_binance(_CACHE["braw"])
    return _CACHE["braw"], _CACHE["bpanels"]


def _okx():
    if "oraw" not in _CACHE:
        _CACHE["oraw"] = L.load_raw()
        _CACHE["opanels"] = L.build_panels(_CACHE["oraw"])
    return _CACHE["oraw"], _CACHE["opanels"]


def _cat() -> pd.DataFrame:
    if "cat" not in _CACHE:
        _CACHE["cat"] = L.load_catalog()
    return _CACHE["cat"]


def _eval_all(panels: dict, exprs: list) -> list:
    ev = L._evaluator(panels)
    return [L._values(ev, e) for e in exprs]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _mutate(raw: dict, cut: pd.Timestamp, seed: int) -> dict:
    """A copy of the Binance frames with EVERY field after `cut` replaced by positive noise
    (dates and calendar unchanged)."""
    rng = np.random.default_rng(seed)
    out = {}
    for sym, df in raw.items():
        x = df.copy()
        late = x.index > cut
        x.loc[late] = rng.uniform(0.5, 2.0, size=(int(late.sum()), x.shape[1])) * x.loc[late].to_numpy()[::-1]
        out[sym] = x
    return out


def _mutate_okx(raw: dict, cut: pd.Timestamp, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for sym, r in raw.items():
        px = r["ohlcv"].copy()
        late = px.index > cut
        px.loc[late] = rng.uniform(0.5, 2.0, size=(int(late.sum()), px.shape[1])) * px.loc[late].to_numpy()[::-1]
        f = None if r["funding"] is None else r["funding"].copy()
        if f is not None:
            f[f.index > cut] = rng.normal(0, 1e-3, size=int((f.index > cut).sum()))
        oi = None if r["oi"] is None else r["oi"].copy()
        if oi is not None:
            oi[oi.index > cut] = rng.uniform(1e8, 1e10, size=int((oi.index > cut).sum()))
        out[sym] = {"ohlcv": px, "funding": f, "oi": oi}
    return out


# operators no library definition uses (covered by these, which exist only for the mutation test)
EXTRA_EXPRESSIONS = (
    "clip(pct_change(close, 5), -0.2, 0.2)",
    "cs_demean(ret) ** 2 - ts_mean(funding_rate, 3)",
    "-ts_mean(div(open_interest, delay(open_interest, 1)), 5)",
)


def _operators():
    """The names of the library operators (libfactors puts library/ on sys.path)."""
    from factorlab.operators import OPERATORS

    return OPERATORS


def _cover_sample() -> list:
    """(id, expression): a greedy cover of every operator and every one of the 16 raw fields
    by library definitions (all computable on Binance), the extra expressions, and 40 more
    definitions drawn with a fixed seed."""
    if "cover" in _CACHE:
        return _CACHE["cover"]
    cat = _cat()
    need = (
        set(_operators())
        | {"bench", "binop_Add", "binop_Sub", "binop_Mult", "binop_Div", "binop_Pow", "unary_USub"}
        | set(L.BINANCE_FIELDS)
    )
    chosen, have = [], set()
    for fid, e in zip(cat["factor_id"], cat["expression"]):
        ops, names = L.expr_atoms(e)
        got = (ops | names | set(L.raw_dependencies(e))) & need
        if got - have:
            chosen.append((fid, e))
            have |= got
    for i, e in enumerate(EXTRA_EXPRESSIONS):
        ops, names = L.expr_atoms(e)
        have |= (ops | names) & need
        chosen.append((f"extra_{i}", e))
    extra = cat.sample(40, random_state=17)
    chosen += [(f, e) for f, e in zip(extra["factor_id"], extra["expression"]) if f not in dict(chosen)]
    _CACHE["cover"], _CACHE["cover_missing"] = chosen, need - have
    return chosen


# --------------------------------------------------------------------------- the OKX source is unchanged


def test_okx_is_the_default_source_and_its_stored_rows_reproduce():
    """okx is the default everywhere (folder, panels, classification); the stored OKX status
    classification equals classify() today, and 40 of the OKX screen's stored rows are
    recomputed from fresh OKX values with this code."""
    assert L.DEFAULT_SOURCE == "okx" and L.SOURCES == ("okx", "binance")
    assert L.out_dir() == L.out_dir("okx") == Path(C.DATA) / "library_on_engine"
    assert L.out_dir("binance") == Path(C.DATA) / "library_on_engine" / "binance"
    try:
        L.out_dir("bybit")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown source was accepted")
    raw, panels = _okx()
    assert list(panels) == list(L.SUPPLIED) and list(L.source_panels(raw)) == list(L.SUPPLIED)
    # --status names the code files that changed since a computation (read-only)
    now = L.code_fingerprint()["files"]
    assert L.changed_code_files({"code": {"files": now}}) == [] and L.changed_code_files({}) == []
    assert L.changed_code_files({"code": {"files": {**now, "engine/libfactors.py": "0" * 64}}}) == [
        "engine/libfactors.py"
    ]
    # a recorded code file that does not exist counts as changed (and does not stop --status)
    assert L.changed_code_files({"code": {"files": {**now, "library/factorlab/research.py": "0" * 64}}}) == [
        "library/factorlab/research.py"
    ]
    out = L.out_dir()
    if not (out / "status.parquet").is_file():
        print("    (no OKX outputs: the stored-row check skipped)")
        return
    st = pd.read_parquet(out / "status.parquet")
    fresh = L.classify(_cat())
    cols = ["factor_id", "group", "missing_fields", "oi_short_history", "vwap_proxy", "oi_base_proxy", "lookback"]
    pd.testing.assert_frame_equal(
        fresh[cols].reset_index(drop=True), st[cols].reset_index(drop=True), check_dtype=False
    )
    if not (out / "screen_g.parquet").is_file():
        return
    sg = pd.read_parquet(out / "screen_g.parquet")
    sample = sg.sample(40, random_state=3)
    expr = _cat().set_index("factor_id")["expression"]
    sundays, _ = L.g_sundays()
    pos = panels["close"].index.get_indexer(sundays)
    vals = np.stack([v[pos] for v in _eval_all(panels, list(expr[sample["factor_id"]]))])
    stats, _ = L.factor_stats(
        vals, list(panels["close"].columns), L.g_labels(), L.build_panels(raw, through=L.LABEL_CUTOFF)
    )
    for c in ("ic_n", "ic_t", "ric_t", "ric_mean", "ric_fit_t", "ric_eval_t", "persistence", "coverage"):
        a, b = stats[c].to_numpy(dtype=float), sample[c].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True), c
    for c in ("ric_p", "ic_p"):
        assert np.allclose(
            stats[c].to_numpy(dtype=float), sample[c].to_numpy(dtype=float), rtol=1e-9, atol=1e-12, equal_nan=True
        ), c
    assert np.allclose(sg["ric_q"], L.bh(sg["ric_p"]), equal_nan=True)


def test_binance_writes_never_touch_the_okx_folder():
    """In a scratch data root holding copies of the OKX outputs: the binance writers (shards,
    status, manifest, removal) write only data/library_on_engine/binance/, and every OKX file
    keeps its hash; each source's guard refuses the other's folder, a symlinked folder and a
    protected target."""
    saved = (C.DATA, C.SHIPPED_DATA)
    real_okx = L.out_dir("okx")
    with tempfile.TemporaryDirectory() as td:
        data = Path(td) / "data"
        okx = data / "library_on_engine"
        (okx / "values").mkdir(parents=True)
        for name in ("status.parquet", "screen_g.parquet", "manifest.json"):
            if (real_okx / name).is_file():
                shutil.copy2(real_okx / name, okx / name)
            else:
                (okx / name).write_text("{}\n")
        (okx / "values" / "values_000.parquet").write_bytes(b"shard")
        before = {p.relative_to(okx).as_posix(): _sha(p) for p in okx.rglob("*") if p.is_file()}
        C.DATA = data
        try:
            refused = lambda fn: _refused(fn)  # noqa: E731
            days = pd.date_range("2022-04-03", periods=3, freq="7D", tz="UTC")
            vals = np.arange(2 * 3 * 2, dtype=np.float64).reshape(2, 3, 2)
            so = L.write_values(["CFA", "CFB"], vals, days, ["BTC-USDT", "ETH-USDT"], source=SRC)
            st = pd.DataFrame({"factor_id": ["CFA", "CFB"], "values_shard": [so["CFA"], so["CFB"]]})
            L._write_frame(st, L.out_dir(SRC) / "status.parquet", source=SRC)
            L._write_json({"source": SRC}, L.out_dir(SRC) / "manifest.json", source=SRC)
            (L.out_dir(SRC) / "screen_g.parquet").write_bytes(b"x")
            L._remove(L.out_dir(SRC) / "screen_g.parquet", SRC)
            v, syms, _ = L.read_values(["CFB"], source=SRC)
            assert np.array_equal(v[0], vals[1]) and syms == ["BTC-USDT", "ETH-USDT"]
            after = {
                p.relative_to(okx).as_posix(): _sha(p)
                for p in okx.rglob("*")
                if p.is_file() and not p.relative_to(okx).as_posix().startswith("binance/")
            }
            assert after == before, "a binance write changed the OKX folder"
            assert sorted(p.relative_to(okx).as_posix() for p in (okx / "binance").rglob("*") if p.is_file()) == [
                "binance/manifest.json",
                "binance/status.parquet",
                "binance/values/values_000.parquet",
            ]
            assert refused(lambda: L._guard(okx / "status.parquet", SRC)), "binance wrote the OKX folder"
            assert refused(lambda: L._guard(okx / "binance" / "status.parquet", "okx")), "okx wrote binance/"
            assert refused(lambda: L._guard(okx / "binance" / "sub" / "x.parquet", SRC))
            shutil.rmtree(okx / "binance")
            (okx / "binance").symlink_to(okx)
            assert refused(lambda: L._guard(okx / "binance" / "status.parquet", SRC)), "a symlinked binance/"
        finally:
            C.DATA = saved[0]
        # a protected folder is refused before anything else (a scratch root with the marker file)
        data2 = Path(td) / "protected" / "data"
        for name in C.PROTECTED_DIRS:
            (data2 / name).mkdir(parents=True)
        (data2 / C.PROTECT_MARKER).write_text("protected\n")
        C.DATA, C.SHIPPED_DATA = data2, data2
        try:
            assert _refused(lambda: L._guard(data2 / "research" / "x.parquet", SRC))
            (data2 / "library_on_engine").symlink_to(data2 / "research")
            assert _refused(lambda: L._guard(L.out_dir(SRC) / "status.parquet", SRC)), "a symlinked library folder"
            assert not any((data2 / "research").iterdir())
        finally:
            C.DATA, C.SHIPPED_DATA = saved


def _refused(fn) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    return False


# --------------------------------------------------------------------------- the adapter


def test_symbol_mapping_and_the_benchmark():
    m = L.binance_symbol_map()
    assert list(m.values()) == [C.sym_to_file(s) for s in C.UNIVERSE] and len(m) == 20
    assert all(k == v.replace("-", "") for k, v in m.items()) and m["BTCUSDT"] == "BTC-USDT"
    assert L.binance_symbol("AAVE-USDT") == "AAVEUSDT"
    if not _have_panel():
        return _skipped("panel mapping")
    raw, p = _bin()
    _, op = _okx()
    assert list(raw) == list(p["close"].columns) == list(op["close"].columns)
    # missing coin, extra coin, missing field: refused
    from factorlab.data import load_panels, save_panels

    src = load_panels(L.BINANCE_PANEL_DIR)
    for case in ("missing_coin", "extra_coin", "missing_field"):
        with tempfile.TemporaryDirectory() as td:
            q = {f: df.copy() for f, df in src.items()}
            if case == "missing_coin":
                q = {f: df.drop(columns=["AAVEUSDT"]) for f, df in q.items()}
            elif case == "extra_coin":
                q = {f: df.assign(BNBUSDT=1.0) for f, df in q.items()}
            else:
                q.pop("premium")
            save_panels(q, td)
            assert _refused(lambda: L.load_raw_binance(Path(td))), case
    # the benchmark: btc_ret is Binance's BTCUSDT return, read straight from the file
    close = pd.read_csv(L.BINANCE_PANEL_DIR / "close.csv.gz", index_col=0, parse_dates=True)["BTCUSDT"]
    ref = (close / close.shift(1) - 1).to_numpy()
    btc = L._values(L._evaluator(p), "btc_ret")
    idx = p["close"].index
    j = list(p["close"].columns).index("SOL-USDT")
    assert np.allclose(btc[:, j], ref, equal_nan=True, rtol=1e-12, atol=0)
    # the OKX data can run past the Binance panel: compare on the Binance panel's dates
    okx_btc = pd.Series(L._values(L._evaluator(op), "btc_ret")[:, j], index=op["close"].index).reindex(idx).to_numpy()
    assert not np.allclose(btc[:, j], okx_btc, equal_nan=True, rtol=0, atol=1e-9), "the benchmark is not Binance's"


def test_binance_adapter_reads_no_engine_data():
    """One venue per formula: with every engine data reader (data_io and pandas' parquet
    reader) disabled, the Binance panels build and the cover sample evaluates; every panel
    cell equals the Binance file's value on bars the coin has, and is NaN elsewhere."""
    if not _have_panel():
        return _skipped("venue isolation")
    names = (
        "ohlcv",
        "spot_ohlcv",
        "daily_funding",
        "daily_oi",
        "funding_settlements",
        "daily_funding_detail",
        "_daily_funding_from",
        "label_table",
        "close_matrix",
        "funding_matrix",
    )
    saved = {n: getattr(data_io, n) for n in names}
    saved_rp = pd.read_parquet

    def boom(*a, **k):
        raise AssertionError("the binance adapter read engine data")

    for n in names:
        setattr(data_io, n, boom)
    pd.read_parquet = boom
    try:
        raw = L.load_raw_binance()
        p = L.build_panels_binance(raw)
        vals = _eval_all(p, [e for _, e in _cover_sample()])
    finally:
        for n, f in saved.items():
            setattr(data_io, n, f)
        pd.read_parquet = saved_rp
    assert len(vals) == len(_cover_sample()) and any(np.isfinite(v).any() for v in vals)
    from factorlab.data import load_panels

    src = load_panels(L.BINANCE_PANEL_DIR)
    bar = src["close"].notna()
    file_cells = {
        f: src[f].where(bar).rename(columns=L.binance_symbol_map())[list(p[f].columns)] for f in L.BINANCE_FIELDS
    }
    gone = _cleared_cells(file_cells)
    for f in L.BINANCE_FIELDS:
        want = file_cells[f].mask(gone[f]) if f in gone else file_cells[f]
        got = p[f]
        assert len(got) == len(want) and bool((got.index == want.index).all()), f
        assert np.array_equal(got.to_numpy(), want.to_numpy(), equal_nan=True), f
    assert set(p) == set(L.BINANCE_FIELDS) and len(L.BINANCE_FIELDS) == 16
    # the downloaded panel: the pinned archive errors (OI_ZERO_DAY, OI_ZERO_LATER, VOLUME_ERRORS,
    # TAKER_ERRORS) clear 2 x 20 + 1 + 1 cells; the demo panel has no archive error
    want = 0 if _demo_panel() else 2 * 20 + 1 + 1
    assert sum(int(m.to_numpy().sum()) for m in gone.values()) == want


def _cleared_cells(p: dict) -> dict:
    """The cleaning rules written out independently of libfactors: {field: mask of the cells
    set missing}. Open interest or its value <= 0 clears both; an average price (quote / base
    volume) outside [low, high], relative tolerance 1e-9, clears the base volume."""

    def outside(q, b):
        lo, hi = p["low"].to_numpy(), p["high"].to_numpy()
        with np.errstate(all="ignore"):
            px = p[q].to_numpy() / p[b].to_numpy()
            fire = ~np.isnan(px) & ((px < lo * (1 - 1e-9)) | (px > hi * (1 + 1e-9)))
        return pd.DataFrame(fire, index=p["low"].index, columns=p["low"].columns)

    oi = (p["open_interest"] <= 0) | (p["open_interest_value"] <= 0)
    return {
        "open_interest": oi,
        "open_interest_value": oi,
        "volume": outside("quote_volume", "volume"),
        "taker_buy_volume": outside("taker_buy_quote_volume", "taker_buy_volume"),
    }


def test_binance_field_conventions():
    """UTC open-time daily bars, all closed; funding = the per-bar sum of settlements in
    [D, D + 1); the downloader's resample rules; the metrics snapshot at the bar's end; a
    true vwap (not degenerate) on Binance, a collapsing one on OKX."""
    from factorlab.data import binance as B

    # the downloader's rules, offline: funding summed by settlement time, label and closed left
    t0 = pd.Timestamp("2023-05-01", tz="UTC")
    st = pd.Series(
        [1.0, 2.0, 4.0, 8.0],
        index=[
            t0 + pd.Timedelta(milliseconds=3),
            t0 + pd.Timedelta(hours=8),
            t0 + pd.Timedelta(hours=16, milliseconds=5),
            t0 + pd.Timedelta(days=1, milliseconds=2),
        ],
    )
    bars = B.to_bar(st, "1d", "sum")
    assert bars[t0] == 7.0 and bars[t0 + pd.Timedelta(days=1)] == 8.0
    snaps = pd.DataFrame(
        {"open_interest": [1.0, 2.0, 3.0]},
        index=[t0 + pd.Timedelta(hours=12), t0 + pd.Timedelta(hours=23, minutes=55), t0 + pd.Timedelta(days=1)],
    )
    last = B.to_bar(snaps, "1d", "last")
    assert last.loc[t0, "open_interest"] == 2.0 and last.loc[t0 + pd.Timedelta(days=1), "open_interest"] == 3.0
    src = inspect.getsource(B.download_metrics_archive)
    assert 'resample(_PANDAS_FREQ[interval], label="left", closed="left").last()' in src
    assert 'r.sum(min_count=1) if how == "sum" else r.last()' in inspect.getsource(B.to_bar)
    assert 'resample(_PANDAS_FREQ[interval], label="left", closed="left")' in inspect.getsource(B.to_bar)
    prem = inspect.getsource(B.fetch_premium)
    assert "float(r[4])" in prem and "r[0] for r in rows" in prem, (
        "premium = the premium kline's close at its open time"
    )
    k = B.parse_klines(
        [
            [
                int(t0.timestamp() * 1000),
                "1",
                "2",
                "0.5",
                "1.5",
                "10",
                int(t0.timestamp() * 1000) + 86_399_999,
                "15",
                "7",
                "4",
                "6",
                "0",
            ]
        ]
    )
    assert k.index[0] == t0 and k.loc[t0, "close"] == 1.5 and k.loc[t0, "quote_volume"] == 15.0
    # the metrics archive's create_time is read as UTC (a synthetic daily file, no network)
    csv = (
        "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
        "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2023-05-01 23:55:00,BTCUSDT,10,300000,1.1,1.2,1.3,0.9\n"
    )
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as z:
        z.writestr("BTCUSDT-metrics-2023-05-01.csv", csv)
    mz = B.parse_metrics_zip(blob.getvalue())
    assert mz.index[0] == t0 + pd.Timedelta(hours=23, minutes=55) and str(mz.index.tz) == "UTC"
    assert (
        mz.iloc[0]["open_interest"],
        mz.iloc[0]["top_long_short_ratio"],
        mz.iloc[0]["long_short_ratio"],
        mz.iloc[0]["taker_buy_sell_ratio"],
    ) == (10.0, 1.2, 1.3, 0.9)
    if not _have_panel():
        return _skipped("panel conventions")
    raw, p = _bin()
    oraw, op = _okx()
    idx = p["close"].index
    assert str(idx.tz) == "UTC" and (idx == idx.normalize()).all()
    assert pd.Series(idx).diff().dropna().eq(pd.Timedelta(days=1)).all()
    okx_last = max(r["ohlcv"].index.max() for r in oraw.values())
    # the panel starts on its first bar; each append (merge_panels.py --append) adds the next days,
    # and the panel then ends on the last append's last bar
    appends = L.binance_panel_appends()
    assert idx[0] == pd.Timestamp(L.BINANCE_FIRST_BAR, tz="UTC")
    if appends:
        assert idx[-1] == pd.Timestamp(appends[-1]["last_bar"], tz="UTC")
    # the Binance panel ends on a closed bar, no later than the OKX data
    assert idx[-1] <= okx_last
    assert idx[-1] + pd.Timedelta(days=1) <= pd.Timestamp.now(tz="UTC").normalize(), "the last bar is still forming"
    for a in appends:  # every appended bar was closed when the append ran
        assert pd.Timestamp(a["last_bar"], tz="UTC") + pd.Timedelta(days=1) <= pd.Timestamp(a["appended_at"]), a
        assert a["old_cells"] == "byte-equal" and a["bars"] == len(pd.date_range(a["first_bar"], a["last_bar"])), a
    # funding: the per-day sum of the engine's own Binance settlement prints, cell for cell
    n = 0
    for sym in p["close"].columns:
        prints = pd.read_parquet(C.RAW_FUNDING_FB / f"{sym}.parquet", columns=["ts", "funding_rate"]).dropna()
        day = prints.groupby(prints["ts"].dt.floor("D"))["funding_rate"].sum()
        mine = p["funding_rate"][sym].dropna()
        common = mine.index.intersection(day.index)
        assert len(common) == len(mine), sym
        assert np.allclose(mine.reindex(common), day.reindex(common), rtol=0, atol=1e-15), sym
        n += len(common)
    assert n > 0, "no funding day-cell was compared"
    # the metrics snapshot sits at the bar's end: implied mark price next to D's close
    with np.errstate(all="ignore"):
        imp = p["open_interest_value"] / p["open_interest"]
        d0 = np.abs((imp / p["close"] - 1).to_numpy())
        d1 = np.abs((imp / p["close"].shift(1) - 1).to_numpy())
    ok = np.isfinite(d0) & np.isfinite(d1)
    assert np.median(d0[ok]) < 0.002 and np.median(d1[ok]) > 10 * np.median(d0[ok]), (
        np.median(d0[ok]),
        np.median(d1[ok]),
    )
    # vwap: a true VWAP on Binance, a copy of close on OKX
    assert not L.vwap_collapses(p) and L.vwap_collapses(op)
    with np.errstate(all="ignore"):
        dev = np.abs(p["close"] / (p["quote_volume"] / p["volume"]) - 1).to_numpy()
    assert np.nanmedian(dev) > 1e-3
    # every field NaN before a coin's first kline bar
    for sym, df in raw.items():
        first = df["close"].first_valid_index()
        for f in L.BINANCE_FIELDS:
            assert p[f].loc[p[f].index < first, sym].isna().all(), (sym, f)


# the archive errors the cleaning rules clear on the downloaded panel (pinned; manifest panels.checks.cleaning).
# The dates and values below are facts from the Binance Vision archive (data.binance.vision).
# Credit: Binance Vision.
OI_ZERO_DAY = "2022-03-07"  # every coin listed then (all but the three late listings)
LATE_LISTINGS = ("APT-USDT", "OP-USDT", "ARB-USDT")
OI_ZERO_LATER = (("2024-07-10", "BTC-USDT"), ("2024-07-12", "BTC-USDT"), ("2024-07-14", "ADA-USDT"))
VOLUME_ERRORS = (("2023-11-14", "DOT-USDT"),)  # quote_volume / volume 7.515 against a 4.96-5.526 bar
TAKER_ERRORS = (("2023-11-30", "XRP-USDT"),)  # taker_buy_quote / taker_buy_volume 1.431 against 0.5973-0.6098


def _synthetic_raw(days: int = 6) -> dict:
    """Two coins with every field filled and consistent (average prices inside [low, high])."""
    idx = pd.date_range("2023-01-01", periods=days, freq="D", tz="UTC")
    out = {}
    for k, sym in enumerate(("BTC-USDT", "ETH-USDT")):
        b = 100.0 * (k + 1)
        df = pd.DataFrame(index=idx, columns=list(L.BINANCE_FIELDS), dtype=float)
        df[["open", "high", "low", "close"]] = [b, 1.1 * b, 0.9 * b, 1.02 * b]
        df[["volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]] = [1e3, 1e3 * b, 400.0, 404.0 * b]
        df[["trades", "funding_rate", "premium", "open_interest", "open_interest_value"]] = [
            50.0,
            1e-4,
            2e-4,
            5e3,
            5e3 * b,
        ]
        df[["long_short_ratio", "top_long_short_ratio", "taker_buy_sell_ratio"]] = [1.5, 1.2, 0.9]
        out[sym] = df
    return out


def test_binance_cleaning_rules():
    """The three label-free rules clear exactly their cells (and nothing else), each from one
    coin's fields on one bar; on the downloaded panel they clear exactly the pinned archive
    errors, and the manifest lists the same cells."""
    raw = _synthetic_raw()
    d = raw["BTC-USDT"].index
    btc, eth = raw["BTC-USDT"], raw["ETH-USDT"]
    btc.loc[d[1], "open_interest"] = 0.0  # OI rule: both OI fields
    eth.loc[d[2], "open_interest_value"] = -1.0  # OI rule
    btc.loc[d[3], "volume"] = 10.0  # average price 100 x the bar: volume rule
    eth.loc[d[3], "quote_volume"] = 1e3 * 220.0 * (1 + 1e-10)  # on the high, inside the tolerance: kept
    eth.loc[d[4], "taker_buy_volume"] = 10.0  # taker rule
    btc.loc[d[4], "volume"] = 0.0  # quote volume over zero volume: volume rule
    btc.loc[d[5], "volume"] = np.nan  # a missing input fires nothing
    eth.loc[d[5], "open_interest"] = np.nan
    u = L.build_panels_binance(raw, clean=False)
    p = L.build_panels_binance(raw)
    cleared = {
        ("open_interest", d[1], "BTC-USDT"),
        ("open_interest_value", d[1], "BTC-USDT"),
        ("open_interest", d[2], "ETH-USDT"),
        ("open_interest_value", d[2], "ETH-USDT"),
        ("volume", d[3], "BTC-USDT"),
        ("taker_buy_volume", d[4], "ETH-USDT"),
        ("volume", d[4], "BTC-USDT"),
    }
    for f in L.BINANCE_FIELDS:
        assert u[f].equals(pd.DataFrame({s: raw[s][f] for s in raw}).rename_axis("ts").astype(float)), f
        for day in d:
            for sym in raw:
                a, b = u[f].at[day, sym], p[f].at[day, sym]
                if (f, day, sym) in cleared:
                    assert np.isfinite(a) and np.isnan(b), (f, day, sym)
                else:
                    assert (np.isnan(a) and np.isnan(b)) or a == b, (f, day, sym)
    assert not any(m.to_numpy().any() for m in L.binance_glitches(p).values()), "a rule fires after cleaning"
    rec = L.binance_cleaning_record(u)
    got = {r: [(w["date"], w["symbol"]) for w in x["where"]] for r, x in rec.items()}
    assert got == {
        "open_interest_nonpositive": [(str(d[1].date()), "BTC-USDT"), (str(d[2].date()), "ETH-USDT")],
        "volume_vwap_outside_low_high": [(str(d[3].date()), "BTC-USDT"), (str(d[4].date()), "BTC-USDT")],
        "taker_buy_volume_vwap_outside_low_high": [(str(d[4].date()), "ETH-USDT")],
    }, got
    assert [rec[r]["fields_set_missing"] for r in rec] == [list(L.CLEANING_RULES[r][0]) for r in rec]
    # per bar and point in time: errors after a cut change nothing up to it, and a cut panel is
    # the full panel's head
    late = {s: df.copy() for s, df in raw.items()}
    late["ETH-USDT"].loc[d[5], ["volume", "open_interest"]] = [1.0, 0.0]
    cut = d[2]
    a, b = L.build_panels_binance(raw), L.build_panels_binance(late)
    t = L.build_panels_binance(raw, through=cut)
    for f in L.BINANCE_FIELDS:
        assert a[f].loc[:cut].equals(b[f].loc[:cut]) and t[f].equals(a[f].loc[:cut]), f
    assert np.isnan(b["volume"].at[d[5], "ETH-USDT"]) and np.isnan(b["open_interest"].at[d[5], "ETH-USDT"])
    assert "cleaning" in L.BINANCE_CONVENTIONS and "no value is carried" in L.BINANCE_CONVENTIONS["cleaning"].lower()
    if not _have_panel():
        return _skipped("cleaning on the panel")
    braw, bp = _bin()
    u = L.build_panels_binance(braw, clean=False)
    rec = L.binance_cleaning_record(u)
    if _demo_panel():
        assert all(x["where"] == [] for x in rec.values()), rec
        print("        (part skipped: the demo panel has no archive errors, so no pinned error to check)")
        return None
    listed = [s for s in u["close"].columns if np.isfinite(u["close"].at[pd.Timestamp(OI_ZERO_DAY, tz="UTC"), s])]
    assert len(listed) == 17 and not set(listed) & set(LATE_LISTINGS)
    want = {
        "open_interest_nonpositive": [(OI_ZERO_DAY, s) for s in listed] + list(OI_ZERO_LATER),
        "volume_vwap_outside_low_high": list(VOLUME_ERRORS),
        "taker_buy_volume_vwap_outside_low_high": list(TAKER_ERRORS),
    }
    got = {r: [(w["date"], w["symbol"]) for w in x["where"]] for r, x in rec.items()}
    assert got == want, got
    assert all(
        w["open_interest"] == 0 and w["open_interest_value"] == 0 for w in rec["open_interest_nonpositive"]["where"]
    )
    dot, xrp = (
        rec["volume_vwap_outside_low_high"]["where"][0],
        rec["taker_buy_volume_vwap_outside_low_high"]["where"][0],
    )
    assert abs(dot["average_price"] - 7.515) < 1e-3 and (dot["low"], dot["high"]) == (4.96, 5.526)
    assert abs(xrp["average_price"] - 1.431) < 1e-3 and (xrp["low"], xrp["high"]) == (0.5973, 0.6098)
    # where they sit: the zero-OI day inside the lookbacks of the first G_fit Sundays, the two volume
    # errors inside G_eval, the later zero-OI cells after the G labels (I zone)
    assert pd.Timestamp(OI_ZERO_DAY) < pd.Timestamp(L.G_FIT[0])
    assert all(L.G_EVAL[0] <= x <= L.G_EVAL[1] for x, _ in VOLUME_ERRORS + TAKER_ERRORS)
    assert all(pd.Timestamp(x, tz="UTC") > L.LABEL_CUTOFF for x, _ in OI_ZERO_LATER)
    # the two base-volume errors: the quote-unit taker share agrees with the days around it,
    # the base-unit one does not
    for (day, sym), (_base, _quote) in (
        (VOLUME_ERRORS[0], ("volume", "quote_volume")),
        (TAKER_ERRORS[0], ("taker_buy_volume", "taker_buy_quote_volume")),
    ):
        t0 = pd.Timestamp(day, tz="UTC")
        around = [t0 - pd.Timedelta(days=1), t0 + pd.Timedelta(days=1)]
        sb = (u["taker_buy_volume"] / u["volume"])[sym]
        sq = (u["taker_buy_quote_volume"] / u["quote_volume"])[sym]
        assert all(abs(sq[t0] - sq[x]) < 0.05 for x in around) and abs(sb[t0] - sq[t0]) > 0.15, (day, sym)
    assert int(bp["open_interest"].notna().to_numpy().sum()) == int(u["open_interest"].notna().to_numpy().sum()) - 20
    out = L.out_dir(SRC)
    if (out / "manifest.json").is_file():
        man = json.loads((out / "manifest.json").read_text())
        ch = man["panels"]["checks"]
        assert ch["cleaning"] == json.loads(json.dumps(rec)), "the manifest's cleaning record differs from the panel"
        if not L.changed_code_files(man):
            assert man["panels"]["conventions"]["cleaning"] == L.BINANCE_CONVENTIONS["cleaning"]
        assert set(ch["cleaning_after"]["cells_firing_after_cleaning"].values()) == {0}
        assert ch["cleaning_after"]["nonpositive_open_interest_cells"] == 0
        assert ch["coverage"]["open_interest"]["cells"] == int(bp["open_interest"].notna().to_numpy().sum())
        assert ch["vwap"]["share_inside_low_high"] == 1.0


# The pinned hash of the cleaning rules (the token form, the same on each Python version). The
# weekly log of the formula tests (judge.py log-week) reads only this set of rules.
FROZEN_CLEANING_SHA256 = "ec8b739ac9182fbb4b1d3d69faf9d017f9981b32637ac64f7573c22591462880"


class _SameCodeA:  # three versions of one function f (the syntax tree holds the name)
    def f(x):
        """One docstring."""
        return x + 1  # one comment


class _SameCodeB:
    def f(x):
        """Another docstring, with other words."""

        return x + 1


class _OtherCode:
    def f(x):
        """One docstring."""
        return x + 2


def _other_code(x):
    return x + 2


def test_token_form_of_the_freeze():
    """The token form ignores the docstring, the comments, blank lines and the indent width. It
    does not ignore the code or the block structure. An f-string is one item."""
    t = L._source_tokens
    one = 'def f(x):\n    """Doc."""\n    if x:\n        a = 1  # note\n    b = 2\n    return max(a, b)\n'
    same = "def f(x):\n  if x:\n\n     a = 1\n  b = 2\n  return max(a,\n             b)\n"
    moved = 'def f(x):\n    """Doc."""\n    if x:\n        a = 1\n        b = 2\n    return max(a, b)\n'
    assert t(one) == t(same), "a docstring, a comment, a blank line or the indent width changed the form"
    assert t(one) != t(moved), "a statement moved into the if block, and the form did not change"
    assert "Doc" not in " ".join(t(one)) and t('def g():\n    x = "Doc."\n')[-3] == '"Doc."'
    assert t('def h(x):\n    return f"a{x!r:>4}b"\n')[-3] == 'f"a{x!r:>4}b"', "an f-string is one item"
    assert L._code_tokens(_SameCodeA.f) == L._code_tokens(_SameCodeB.f)
    assert L._code_tokens(_SameCodeA.f) != L._code_tokens(_OtherCode.f)
    assert L.CLEANING_FREEZE["sha256"] == FROZEN_CLEANING_SHA256


def test_cleaning_rules_are_frozen():
    """The rules in the code give the pinned hash. A change to a rule's fields, its text, the
    tolerance or the code gives another hash, and the compute's first check then stops. A
    change to a docstring, a comment or the layout does not count. A manifest from the same
    code records the same freeze."""
    fz = L.CLEANING_FREEZE
    assert set(fz) == {"version", "sha256", "form"} and fz["version"] == "1"
    fp = L.cleaning_rules_fingerprint()
    assert fp["sha256"] == fz["sha256"] == FROZEN_CLEANING_SHA256, "the cleaning rules changed after the freeze"
    assert (
        fp["rules"]
        == list(L.CLEANING_RULES)
        == ["open_interest_nonpositive", "volume_vwap_outside_low_high", "taker_buy_volume_vwap_outside_low_high"]
    )
    assert L.cleaning_freeze_record()["matches_frozen_set"] is True
    assert "CLEANING_FREEZE" in L.BINANCE_CONVENTIONS["cleaning_frozen"]
    # the compute checks the freeze before it reads the panel
    src = inspect.getsource(L.run_compute_binance)
    assert src.index("cleaning_freeze_record()") < src.index("load_raw_binance()")
    rule = "volume_vwap_outside_low_high"
    saved_rule, saved_tol, saved_fn = L.CLEANING_RULES[rule], L.CLEAN_TOL, L.binance_glitches
    changes = {
        "fields": lambda: L.CLEANING_RULES.__setitem__(rule, (("volume", "quote_volume"), saved_rule[1])),
        "text": lambda: L.CLEANING_RULES.__setitem__(rule, (saved_rule[0], saved_rule[1] + ".")),
        "tolerance": lambda: setattr(L, "CLEAN_TOL", 1e-8),
        "code": lambda: setattr(L, "binance_glitches", _other_code),
    }
    try:
        for name, change in changes.items():
            change()
            assert L.cleaning_rules_fingerprint()["sha256"] != FROZEN_CLEANING_SHA256, name
            try:
                L.cleaning_freeze_record()
                raise AssertionError(f"a {name} change passed the freeze check")
            except SystemExit as exc:
                assert "cleaning rules differ" in str(exc), name
            L.CLEANING_RULES[rule], L.CLEAN_TOL, L.binance_glitches = saved_rule, saved_tol, saved_fn
    finally:
        L.CLEANING_RULES[rule], L.CLEAN_TOL, L.binance_glitches = saved_rule, saved_tol, saved_fn
    assert L.cleaning_rules_fingerprint()["sha256"] == FROZEN_CLEANING_SHA256
    out = L.out_dir(SRC)
    if not (out / "manifest.json").is_file():
        return
    man = json.loads((out / "manifest.json").read_text())
    if L.changed_code_files(man):
        print("    (the code changed after the computation of the Binance manifest: freeze record not checked)")
        return
    assert man["panels"]["cleaning_freeze"] == L.cleaning_freeze_record()
    assert man["panels"]["conventions"]["cleaning_frozen"] == L.BINANCE_CONVENTIONS["cleaning_frozen"]


def test_appends_log_must_be_consecutive():
    """libfactors reads the appends of new closed days (merge_panels.py --append) from
    appends.jsonl in the panel folder. Without the file, it finds no append. Each append must
    start on the day after the end of the append before it."""
    d = Path(tempfile.mkdtemp(prefix="binance_appends_"))
    try:
        assert L.binance_panel_appends(d) == []
        end = pd.Timestamp("2025-01-05")
        rows = [
            {
                "first_bar": str((end + pd.Timedelta(days=1)).date()),
                "last_bar": str((end + pd.Timedelta(days=6)).date()),
            },
            {
                "first_bar": str((end + pd.Timedelta(days=7)).date()),
                "last_bar": str((end + pd.Timedelta(days=13)).date()),
            },
        ]
        (d / L.BINANCE_APPEND_LOG).write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert L.binance_panel_appends(d) == rows
        rows[1]["first_bar"] = str((end + pd.Timedelta(days=8)).date())  # a missing day
        (d / L.BINANCE_APPEND_LOG).write_text("".join(json.dumps(r) + "\n" for r in rows))
        try:
            L.binance_panel_appends(d)
            raise AssertionError("a gap between two appends was accepted")
        except SystemExit as exc:
            assert "starts on" in str(exc)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # the compute checks that the panel's span equals the provenance's span
    src = inspect.getsource(L.run_compute_binance)
    assert "binance_panel_appends()" in src and "but its provenance" in src


def test_future_mutation_through_the_binance_adapter():
    """Every Binance value of every field after a cut replaced by noise, through the whole
    adapter: every factor value on a date <= cut is unchanged, at the three PIT cuts, for a
    sample covering every operator and all 16 raw fields; the mutation reaches the factors."""
    sample = _cover_sample()
    assert not _CACHE["cover_missing"], f"not covered: {sorted(_CACHE['cover_missing'])}"
    if not _have_panel():
        return _skipped("future mutation")
    raw, panels = _bin()
    exprs = [e for _, e in sample]
    full = _eval_all(panels, exprs)
    for seed, cut in enumerate((pd.Timestamp(c, tz="UTC") for c in L.PIT_CUTS), start=1):
        mut = _eval_all(L.build_panels_binance(_mutate(raw, cut, seed)), exprs)
        k = int(panels["close"].index.searchsorted(cut, side="right"))
        changed, finite = 0, 0
        for (fid, e), a, b in zip(sample, full, mut):
            same = np.isclose(a[:k], b[:k], rtol=L.PIT_RTOL, atol=L.PIT_ATOL, equal_nan=True)
            assert same.all(), f"{fid} {e}: {int((~same).sum())} cells on or before {cut.date()} changed"
            finite += int(np.isfinite(a[:k]).sum())
            changed += not np.array_equal(a[k:], b[k:], equal_nan=True)
        assert finite > 0 and changed >= 0.9 * len(sample), (cut, changed, len(sample))
    # a lookahead factor is caught by the same comparison
    ahead = "delay(premium, -1) - premium"
    a = _eval_all(panels, [ahead])[0]
    b = _eval_all(L.build_panels_binance(_mutate(raw, L.LABEL_CUTOFF, 9)), [ahead])[0]
    k = int(panels["close"].index.searchsorted(L.LABEL_CUTOFF, side="right"))
    assert not np.isclose(a[:k], b[:k], equal_nan=True).all()


# --------------------------------------------------------------------------- the sealed I zone


def _family_sample(n: int = 30) -> list:
    st = L.binance_status(
        _cat(), pd.read_parquet(L.out_dir("okx") / "status.parquet", columns=["factor_id", "group", "screen"]), False
    )
    pool = st[(st["role"] == "new_family") & (st["lookback"] < 300)]
    expr = _cat().set_index("factor_id")["expression"]
    picks = []
    for origin in ("binance_only_fields", "oi", "vwap"):
        g = pool[pool["origin"] == origin]
        picks += list(g.sample(min(len(g), n // 3), random_state=5)["factor_id"])
    return [(f, expr[f]) for f in picks]


def _bin_screen(raw_b: dict, raw_o: dict, ft: pd.DataFrame, sample: list):
    panels = L.build_panels_binance(raw_b)
    sundays, _ = L.g_sundays()
    pos = panels["close"].index.get_indexer(sundays)
    vals = np.stack([v[pos] for v in _eval_all(panels, [e for _, e in sample])])
    return L.screen_frame(
        vals,
        [f for f, _ in sample],
        list(panels["close"].columns),
        L.g_labels(ft),
        L.build_panels(raw_o, through=L.LABEL_CUTOFF),
    )


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf)
    return buf.getvalue()


def test_binance_screen_keeps_the_i_zone_sealed():
    if not _have_panel() or not (L.out_dir("okx") / "status.parquet").is_file():
        return _skipped("sealed I zone")
    sample = _family_sample()
    assert len(sample) >= 20
    raw_b, _ = _bin()
    raw_o, _ = _okx()
    ft = data_io.label_table()
    lab = L.g_labels(ft)
    late = pd.concat(
        [lab, pd.DataFrame({"date": [L.LABEL_CUTOFF + pd.Timedelta(days=7)], "symbol": ["BTC-USDT"], L.LABEL: [0.0]})],
        ignore_index=True,
    )
    sundays, _ = L.g_sundays()
    vals = np.zeros((1, len(sundays), len(raw_b)))
    try:
        L.screen_frame(vals, ["x"], list(raw_b), late, L.build_panels(raw_o, through=L.LABEL_CUTOFF))
    except L.LabelLeak:
        pass
    else:
        raise AssertionError("a label after the cutoff was accepted")
    df0, s0, w0 = _bin_screen(raw_b, raw_o, ft, sample)
    rng = np.random.default_rng(7)
    ft_m = ft.copy()
    after = ft_m["date"] > L.LABEL_CUTOFF
    num = ft_m.select_dtypes("number").columns
    ft_m.loc[after, num] = rng.normal(0, 0.2, size=(int(after.sum()), len(num)))
    df1, s1, w1 = _bin_screen(_mutate(raw_b, L.LABEL_CUTOFF, 8), _mutate_okx(raw_o, L.LABEL_CUTOFF, 8), ft_m, sample)
    assert len(df0) >= 15 and s0 == s1 and np.array_equal(w0, w1)
    assert _parquet_bytes(df0) == _parquet_bytes(df1), "the Binance screen read something after the cutoff"
    ft_e = ft.copy()
    hit = (ft_e["date"] == L.LABEL_CUTOFF) & (ft_e["symbol"] == "ETH-USDT")
    ft_e.loc[hit, L.LABEL] = ft_e.loc[hit, L.LABEL] + 0.5
    df2, _, _ = _bin_screen(raw_b, raw_o, ft_e, sample)
    assert _parquet_bytes(df0) != _parquet_bytes(df2)


# --------------------------------------------------------------------------- family and cross-check rules


def test_new_family_never_holds_an_okx_screened_factor():
    st = pd.DataFrame(
        {
            "factor_id": ["A", "B", "C", "D", "E"],
            "okx_screen": ["screened", "not_computable", "vwap_proxy", "screened", "no_g_values"],
            "compute": ["ok"] * 5,
            "pit_check": ["pass", "pass", "pass", "pass", "fail"],
            "vwap_proxy": [False] * 5,
        }
    )
    assert L.new_family_ids(st) == ["B", "C"] and L.crosscheck_ids(st) == ["A", "D"]
    okx_path = L.out_dir("okx") / "status.parquet"
    if not okx_path.is_file():
        return
    okx = pd.read_parquet(okx_path, columns=["factor_id", "group", "screen"])
    bs = L.binance_status(_cat(), okx, False)
    scr = set(okx.loc[okx["screen"] == "screened", "factor_id"])
    assert set(bs.loc[bs["role"] == "crosscheck", "factor_id"]) == scr
    assert not set(bs.loc[bs["role"] == "new_family", "factor_id"]) & scr
    assert (bs["group"] == "computable").all() and set(L.BINANCE_FIELDS) == set(L.RAW_FIELD_NAMES)
    fam = bs[bs["role"] == "new_family"]
    assert (fam.loc[fam["okx_group"] != "computable", "origin"] == "binance_only_fields").all()
    assert (fam.loc[fam["origin"] == "binance_only_fields", "binance_only_fields"] != "").all()
    assert (bs.loc[bs["role"] == "crosscheck", "new_fields"] == "").all()
    out = L.out_dir(SRC)
    if (out / "screen_g.parquet").is_file():
        sg = pd.read_parquet(out / "screen_g.parquet")
        assert not set(sg["factor_id"]) & scr, "an OKX-screened factor is in the Binance new family"
        stored = pd.read_parquet(out / "status.parquet")
        assert (stored.loc[stored["screen"] == "screened", "role"] == "new_family").all()
        assert set(stored.loc[stored["role"] == "crosscheck", "factor_id"]) == scr


def test_crosscheck_applies_no_bh_and_publishes_no_q_set():
    """The cross-check gives agreement numbers only: no Binance q-value and no set of factors at a Binance q level."""
    stats = pd.DataFrame({c: [1.0, -2.0] for c in L.CROSS_STATS})
    okx_sg = pd.DataFrame(
        {
            "factor_id": ["A", "B"],
            "name": ["a", "b"],
            "family": ["f", "f"],
            "category": ["c", "c"],
            **{c: [0.5, -1.0] for c in ("ic_n", "ic_t", "ric_mean", "ric_t", "ric_fit_t", "ric_eval_t", "ric_q")},
        }
    )
    fr = L.crosscheck_frame(stats, ["A", "B"], okx_sg, np.array([0.9, 0.8]), np.array([10, 12]))
    bad = re.compile(r"(^binance_.*_(p|q)$)|signal|rep_|survivor|_q\d\d")
    assert not [c for c in fr.columns if bad.search(c)], fr.columns
    out = L.out_dir(SRC)
    if not (out / "crosscheck.parquet").is_file():
        return
    cr = pd.read_parquet(out / "crosscheck.parquet")
    assert not [c for c in cr.columns if bad.search(c)], list(cr.columns)
    assert [c for c in cr.columns if c.endswith("_q")] == ["okx_ric_q"], "only the OKX screen's own q is shown"
    man = json.loads((out / "manifest.json").read_text())
    x = man["crosscheck"]
    flat = json.dumps(x)
    for word in ("n_tests", "fit_selection", "binance_count", "binance_ric_q", "distinct_signals", "_rep_"):
        assert word not in flat, word
    # the only sets at a q level are the OKX screen's own (okx_ric_q05, okx_ric_q10)
    assert [k for k in x if re.search(r"_q\d\d$", k)] == [f"okx_ric_{L._tag(lev)}" for lev in L.Q_LEVELS]
    assert "no Benjamini-Hochberg" in x["rule"]
    # the stored Binance statistics are factor_stats on the stored values (25 factors); the
    # values are git-ignored, so the check is skipped without them
    if not L.has_values(SRC):
        return _no_values("the cross-check recomputation")
    ids = list(cr["factor_id"].sample(25, random_state=2))
    sundays, _ = L.g_sundays()
    vals, syms, _ = L.read_values(ids, sundays, source=SRC)
    raw_o, _ = _okx()
    stats, _ = L.factor_stats(vals, syms, L.g_labels(), L.build_panels(raw_o, through=L.LABEL_CUTOFF))
    look = cr.set_index("factor_id").loc[ids]
    for c in ("ic_t", "ric_t", "ric_fit_t", "ric_eval_t", "persistence", "ic_n"):
        assert np.allclose(
            look[f"binance_{c}"].to_numpy(dtype=float),
            stats[c].to_numpy(dtype=float),
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
        ), c


# --------------------------------------------------------------------------- the written outputs


def test_written_binance_outputs_reconcile_and_carry_no_path():
    out = L.out_dir(SRC)
    if not (out / "manifest.json").is_file():
        print("    (no data/library_on_engine/binance outputs: skipped)")
        return
    from webexport.common import privacy_hits

    text = (out / "manifest.json").read_text()
    man = json.loads(text)
    st = pd.read_parquet(out / "status.parquet")
    for s in (text, "\n".join(st.select_dtypes(exclude="number").astype(str).to_numpy().ravel())):
        assert not privacy_hits(s)
        assert str(C.ROOT) not in s and str(Path.home()) not in s and "/Users/" not in s and str(C.ROOT.parent) not in s
    fp = man["inputs"]["binance_panel"]
    assert fp["panel_set"] == "binance_um_1d" and all("/" not in k for k in fp["files"])
    assert sorted(fp["files"]) == sorted(f"{f}.csv.gz" for f in L.BINANCE_FIELDS)
    if _have_panel():
        assert fp == L.binance_panel_fingerprint(), "the panel changed since --compute"
    assert man["source"] == SRC and man["code"]["files"].keys() == set(L.BINANCE_CODE_FILES)
    assert len(st) == 28904 and "verdict" not in st.columns and (st["group"] == "computable").all()
    c = man["compute"]["counts"]
    assert c["roles"] == st["role"].value_counts().sort_index().to_dict()
    assert c["computed_ok"] == int((st["compute"] == "ok").sum())
    fam = st["role"] == "new_family"
    assert c["new_family_by_origin"] == {k: int(st.loc[fam, "origin"].eq(k).sum()) for k in L.ORIGINS}
    mp = man["compute"]["pit"]
    for k in ("pass", "no_finite_cells", "fail"):
        assert mp[k] == int((st["pit_check"] == k).sum()), k
    assert mp["pass"] + mp["no_finite_cells"] + mp["fail"] == c["computed_ok"]
    cuts = range(1, len(L.PIT_CUTS) + 1)
    fin = st[[f"pit_finite_t{j}" for j in cuts]].sum(axis=1)
    assert (fin[st["pit_check"] == "pass"] > 0).all()
    chk = man["panels"]["checks"]
    assert chk["funding_vs_engine_binance_prints"]["differing_above_1e-12"] == 0
    assert chk["vwap"]["collapses_onto_close"] is False and c["vwap_proxy_degenerate"] == 0
    # the stored values equal a fresh computation, bit for bit (float64); both the panel and
    # values/ are git-ignored, so each may be absent
    if not L.has_values(SRC):
        print("    (part skipped: the stored-values check: no Binance values/ (git-ignored))")
    elif _have_panel():
        _, panels = _bin()
        ids = list(st["factor_id"].sample(25, random_state=9))
        vals, symbols, days = L.read_values(ids, source=SRC)
        assert symbols == list(panels["close"].columns)
        pos = panels["close"].index.get_indexer(days)
        fresh = _eval_all(panels, list(_cat().set_index("factor_id").loc[ids, "expression"]))
        for i, f in enumerate(fresh):
            assert vals.dtype == np.float64 and np.array_equal(vals[i], f[pos], equal_nan=True), ids[i]
    repo = C.ROOT.parent
    if C.DATA.resolve() == (C.ROOT / "data").resolve() and (repo / ".git").exists():
        # values/ is git-ignored (checked on a shard path, which need not exist)
        shard = out / "values" / "values_000.parquet"
        assert subprocess.run(["git", "check-ignore", "-q", str(shard.relative_to(repo))], cwd=repo).returncode == 0
    if "screen" not in man:
        return
    sg = pd.read_parquet(out / "screen_g.parquet")
    k = man["screen"]["counts"]
    assert len(sg) == k["screened"] == int((st["screen"] == "screened").sum())
    assert set(sg["factor_id"]) == set(st.loc[st["screen"] == "screened", "factor_id"])
    assert (sg["ic_n"] >= L.MIN_WEEKS).all() and "verdict" not in sg.columns
    assert k["candidates"] == len(L.new_family_ids(st))
    for name in ("ic", "ric"):
        p = sg[f"{name}_p"].to_numpy()
        ok = np.isfinite(p)
        assert k[f"n_tests_{name}"] == int(ok.sum())
        assert np.allclose(sg[f"{name}_q"].to_numpy()[ok], L.bh_qvalues(p[ok])), "BH over this family only"
        for lev in L.Q_LEVELS:
            tag = L._tag(lev)
            surv = sg[f"{name}_q"] <= lev
            assert k[f"{name}_count_{tag}"] == int(surv.sum())
            assert k[f"{name}_distinct_signals_{tag}"] == int(sg[f"{name}_rep_{tag}"].sum())
            b = k["q_sets"][f"{name}_{tag}"]
            assert b["factors"] == int(surv.sum()) and sum(b["by_origin"].values()) == int(surv.sum())
            assert sum(b["by_family"].values()) == int(surv.sum())
            ov = k["overlap_with_okx_q_sets"][f"{name}_{tag}"]
            col = f"okx_{name}_{tag}_max_rho"
            assert ov["overlapping"] == int((sg.loc[surv, col] >= L.DEDUP_RHO).sum())
            assert ov["new_family_count"] == int(surv.sum())
    assert k["fit_selection"] == L.fit_selection(sg)
    # the Binance screen's own description and sensitivity notes (the counts cover the new family;
    # the rows other than label_mean have the OKX screen's notes), when the same code wrote them
    assert not L.ric_description()["deviations"][-1].endswith("covers the new family only.")
    if L.changed_code_files(man):
        print("    (the code changed after the computation of the manifest: its texts are not compared)")
    else:
        assert man["screen"]["ric_regressors"] == json.loads(json.dumps(L.ric_description(SRC)))
        assert list(man["screen"]["sensitivity"]) == list(L.VARIANTS)
        dev = man["screen"]["ric_regressors"]["deviations"][-1]
        fr = man["screen"]["sensitivity"]["label_mean"]["note"]
        assert dev.endswith("covers the new family only.") and "label_mean" in dev
        assert fr == L.BINANCE_VARIANT_NOTES["label_mean"] and fr.endswith("applied to the new family")
        assert {k_: v["note"] for k_, v in man["screen"]["sensitivity"].items() if k_ != "label_mean"} == {
            k_: v for k_, v in L.VARIANT_NOTES.items() if k_ != "label_mean"
        }
    gw = st.set_index("factor_id")["g_weeks"]
    assert (gw[sg["factor_id"]].to_numpy() == sg["ic_n"].to_numpy()).all()
    few = st[fam & (st["screen"] == "too_few_weeks")]
    assert (few["g_weeks"] < L.MIN_WEEKS).all() and len(few) == k["dropped_too_few_weeks"]
    top = man["screen"]["top_by_abs_ric_t"]
    assert [r["factor_id"] for r in top] == list(
        sg.reindex(sg["ric_t"].abs().sort_values(ascending=False).index).head(15)["factor_id"]
    )
    cr = pd.read_parquet(out / "crosscheck.parquet")
    x = man["crosscheck"]
    assert len(cr) == x["counts"]["with_g_values"] == int((st["screen"] == "crosscheck").sum())
    a, b = cr["okx_ric_t"].to_numpy(dtype=float), cr["binance_ric_t"].to_numpy(dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    from scipy.stats import spearmanr

    assert abs(x["ric_t_spearman"] - spearmanr(a[both], b[both]).statistic) < 1e-4
    s05 = cr["okx_ric_q"] <= 0.05
    same = s05 & (np.sign(cr["okx_ric_t"]) == np.sign(cr["binance_ric_t"]))
    assert x["okx_ric_q05"]["okx_count"] == int(s05.sum())
    assert x["okx_ric_q05"]["binance_same_sign"] == int(same.sum())
    assert x["okx_ric_q05"]["binance_same_sign_abs_t_ge_2"] == int((same & (cr["binance_ric_t"].abs() >= 2)).sum())
    assert "verdict" not in cr.columns
    # what the agreement can show: the mechanical-agreement and same-funding caveats, and the
    # agreement without the definitions that read funding_rate, recomputed
    cat = _cat().set_index("factor_id")
    fund = np.array(["funding_rate" in cat.at[f, "raw_inputs"].split(",") for f in cr["factor_id"]])
    assert x["funding_readers"] == int(fund.sum()) == sum(v["funding_readers"] for v in x["by_input_class"].values())
    assert abs(x["ric_t_spearman_without_funding"] - spearmanr(a[both & ~fund], b[both & ~fund]).statistic) < 1e-4
    sm = (np.sign(a) == np.sign(b))[both & ~fund].mean()
    assert abs(x["same_ric_sign_share_without_funding"] - sm) < 1e-4
    assert x["agreement_note"] == L.AGREEMENT_NOTE.format(rho=x["value_rho"]["median"])
    assert (
        "not an independent replication" in x["agreement_note"] and "the labels, the regressors" in x["agreement_note"]
    )
    f = x["series_agreement"]["funding"]
    assert f["identical_share"] == 1.0
    if not L.changed_code_files(man):
        assert x["funding_note"] == L.FUNDING_NOTE.format(
            same=f["identical_share"], cells=f["cells"], n=x["funding_readers"], rho=x["ric_t_spearman_without_funding"]
        )
    assert "not cross-venue evidence" in x["funding_note"]
    assert all(v["identical_share"] < 0.2 for k_, v in x["series_agreement"].items() if k_ != "funding")


def test_funding_is_the_same_input_on_both_venues_on_g():
    """The funding caveat, checked on the panels: through the last G label row the engine's OKX-panel
    funding (data_io's splice) is the Binance panel's funding on every coin-day both have (it is
    Binance's archive there), and in the last 30 days (OKX's own prints) it mostly is not."""
    if not _have_panel():
        return _skipped("funding identity")
    _, bp = _bin()
    _, op = _okx()
    fo, fb = op["funding_rate"], bp["funding_rate"]
    idx = fo.index.intersection(fb.index)
    fo, fb = fo.reindex(idx), fb.reindex(idx)
    g = idx <= L.LABEL_CUTOFF
    x, y = fo[g].to_numpy(), fb[g].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    g_cells = int(m.sum())
    assert g_cells > 0 and (np.isfinite(x) == np.isfinite(y)).all()
    assert (np.abs(x[m] - y[m]) <= L.SAME_VALUE_ATOL).all(), "OKX-panel funding on G is not Binance's"
    late = idx >= idx[-1] - pd.Timedelta(days=30)
    x, y = fo[late].to_numpy(), fb[late].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    assert m.sum() > 300 and (np.abs(x[m] - y[m]) <= L.SAME_VALUE_ATOL).mean() < 0.5, "OKX's own prints are not used"
    # the stored series row says the same
    out = L.out_dir(SRC)
    if (out / "manifest.json").is_file() and "crosscheck" in json.loads((out / "manifest.json").read_text()):
        f = json.loads((out / "manifest.json").read_text())["crosscheck"]["series_agreement"]["funding"]
        assert f["cells"] == g_cells and f["identical_share"] == 1.0


# the words that a find tool must not write (numbers only): verdicts, tiers, stages, gate-pass words
VERDICT_WORDS = re.compile(
    r"\b(useful|useless|unproven|effective|survivors?|promotable|novel|confirmed|retired|tiers?|stages?)\b"
    r"|\bpass(es|ed)?\s+(the\s+|all\s+|every\s+)?(gates?|screen)\b|\bgates?\s+pass(es|ed)?\b|forward-only",
    re.IGNORECASE,
)


def _texts(x) -> list:
    """Every string value in a JSON-like object."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [t for v in x.values() for t in _texts(v)]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _texts(v)]
    return []


def _names(x) -> list:
    """Every key of a JSON-like object, at each depth."""
    if isinstance(x, dict):
        return [str(k) for k in x] + [t for v in x.values() for t in _names(v)]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _names(v)]
    return []


def _word_hits(texts) -> list:
    """The texts that hold a word of VERDICT_WORDS. An underscore counts as a space, so the
    check also finds a word inside a key or a column name (ric_xyz_q05 reads as ric xyz q05)."""
    return [t for t in texts if VERDICT_WORDS.search(str(t).replace("_", " "))]


def test_outputs_are_numbers_only():
    """The Binance outputs and --status give numbers only: no verdict column and no verdict, tier or stage word.

    The check reads the manifest texts and keys, the column names and the string cells of the
    screen and cross-check outputs, and the --status lines. --status prints the counts and the
    states of the code and the inputs. It prints no ranked table."""
    import contextlib

    assert not hasattr(L, "VERDICT")
    own = [
        L.BINANCE_ABOUT,
        L.EVIDENCE,
        L.CROSSCHECK_EVIDENCE,
        L.POOLED_G_NOTE,
        L.AGREEMENT_NOTE,
        L.FUNDING_NOTE,
        *L.BINANCE_VARIANT_NOTES.values(),
        *_texts(L.BINANCE_CONVENTIONS),
        *_texts(L.ric_description(SRC)),
    ]
    assert not _word_hits(own), _word_hits(own)
    assert L.CROSSCHECK_EVIDENCE == (
        "Agreement of the two venues on the same labels and regressors. It is not an independent replication."
    )
    stats = pd.DataFrame({c: [1.0, -2.0] for c in L.CROSS_STATS})
    okx_sg = pd.DataFrame(
        {
            "factor_id": ["A", "B"],
            "name": ["a", "b"],
            "family": ["f", "f"],
            "category": ["c", "c"],
            **{c: [0.5, -1.0] for c in ("ic_n", "ic_t", "ric_mean", "ric_t", "ric_fit_t", "ric_eval_t", "ric_q")},
        }
    )
    fr = L.crosscheck_frame(stats, ["A", "B"], okx_sg, np.array([0.9, 0.8]), np.array([10, 12]))
    assert "verdict" not in fr.columns and not _word_hits(list(fr.columns))
    out = L.out_dir(SRC)
    if not (out / "manifest.json").is_file():
        print("    (part skipped: no data/library_on_engine/binance outputs)")
        return
    man = json.loads((out / "manifest.json").read_text())
    assert not _word_hits(_texts(man) + _names(man)), _word_hits(_texts(man) + _names(man))
    if not L.changed_code_files(man):
        assert man["about"] == L.BINANCE_ABOUT
        if "screen" in man:
            assert man["screen"]["evidence"] == L.EVIDENCE and man["screen"]["pooled_g_note"] == L.POOLED_G_NOTE
        if "crosscheck" in man:
            assert man["crosscheck"]["evidence"] == L.CROSSCHECK_EVIDENCE
    for name in ("status.parquet", "screen_g.parquet", "crosscheck.parquet"):
        if (out / name).is_file():
            t = pd.read_parquet(out / name)
            cells = t.select_dtypes(exclude=["number", "bool"]).astype(str).to_numpy().ravel().tolist()
            assert "verdict" not in t.columns and not _word_hits(list(t.columns) + sorted(set(cells))), name
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert L.show_status_binance() == 0
    text = buf.getvalue()
    assert not _word_hits(text.splitlines()), _word_hits(text.splitlines())
    assert "\n  code: " in text and "\n  inputs: " in text and " top " not in text and "factor_id" not in text


def test_status_without_the_panel_says_not_checked():
    """--source binance --status without the panel prints "inputs: not checked", not a changed input."""
    import contextlib

    saved = L.BINANCE_PANEL_DIR
    with tempfile.TemporaryDirectory() as td:
        try:
            # no folder, and a folder without a merged file: the same rule as load_raw_binance
            for d in (Path(td) / "absent", Path(td)):
                L.BINANCE_PANEL_DIR = d
                assert L.changed_inputs({"inputs": {}}, SRC) is None, d
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                L._print_states({"inputs": {}}, SRC, "--source binance --compute and --screen")
                if L._read_manifest(SRC):
                    assert L.show_status_binance() == 0
        finally:
            L.BINANCE_PANEL_DIR = saved
    text = buf.getvalue()
    assert "  inputs: not checked (an input is not on disk)" in text and "inputs: changed" not in text, text


def missing_data() -> str | None:
    """Why the suite cannot run on this data folder, or None."""
    if not L.CATALOG.is_file():
        return "no library catalog (run ./setup.sh)"
    if not any((C.RAW_OHLCV / f"{C.sym_to_file(s)}.parquet").exists() for s in C.UNIVERSE):
        return "no perp bars in the data folder (run ./setup.sh)"
    if not label_path().is_file():
        return "no label table in the data folder (run ./setup.sh)"
    return None


if __name__ == "__main__":
    why = missing_data()
    if why:
        print(f"  SKIP  every libfactors binance test skipped: {why}")
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
    print(f"\n{len(tests) - failed}/{len(tests)} libfactors binance tests passed")
    raise SystemExit(1 if failed else 0)
