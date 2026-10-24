"""Tests of the synthetic demo data (demo_data.py), update.py --demo and --offline, and the banner.

    python test_demo_data.py

Checks:
    1. the same seed gives byte-identical files; another seed gives other values;
    2. data_io reads the raw folders: 20 coins, the late coins, closed daily bars with valid
       OHLC, primary and fallback funding prints, open interest stamped 16:00 UTC;
    3. the library reads the Binance panel (merge_panels format, gzip header time 0), and
       libfactors builds its 16 panels from it; its funding is the daily sum of the fallback prints;
    4. the demo refuses a folder with other files (market data), a protected folder and a
       Binance panel that is not demo data, and writes nothing there; update.py refuses a
       download or an offline build into a demo folder and a demo panel; --remove deletes only
       demo folders; a write or a removal that stops part way keeps the demo markers;
    5. update.py plans --demo without a network step and without a factor-test step; no step
       gets an extra environment variable; the demo writes no other file into the data folder;
    6. meta.js carries demo=true for the demo folder only, and the dashboard shows the banner
       "Synthetic demo data. Not market data." on every page (needs node for the page part).
The tests write only into temporary folders.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

import config as C
import demo_data as D

ENGINE = Path(__file__).resolve().parent
PY = sys.executable


def _hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _run(args, data: Path, **env):
    e = dict(os.environ, FACTORLAB_DATA=str(data), FACTORLAB_OFFLINE="1", **env)
    return subprocess.run([PY, "-B", *args], cwd=ENGINE, env=e, capture_output=True, text=True, timeout=300)


def test_same_seed_gives_identical_files():
    with tempfile.TemporaryDirectory(prefix="demo_") as tmp:
        t = Path(tmp)
        D.write(t / "a", t / "pa")
        D.write(t / "b", t / "pb")
        ha, hb = _hashes(t / "a"), _hashes(t / "b")
        assert ha == hb and len(ha) == 5 * 20 + 1, (len(ha), sorted(set(ha) ^ set(hb)))
        assert sorted(k for k in ha if "/" not in k) == [D.MARKER], "the demo writes one file outside raw/"
        assert _hashes(t / "pa") == _hashes(t / "pb") and len(_hashes(t / "pa")) == 16 + 1
        # a second write into the same demo folder gives the same bytes again
        D.write(t / "a", t / "pa")
        assert _hashes(t / "a") == ha
        D.write(t / "c", None, seed=D.SEED + 1)
        hc = _hashes(t / "c")
        assert hc["raw/ohlcv/BTC-USDT.parquet"] != ha["raw/ohlcv/BTC-USDT.parquet"]
        assert json.loads((t / "c" / D.MARKER).read_text())["seed"] == D.SEED + 1


def test_formats_readable_by_data_io():
    code = r"""
import json, data_io as io, pandas as pd, pyarrow.parquet as pq, config as C
out = {}
cm = io.close_matrix()
out["coins"] = list(cm.columns)
out["first"], out["last"] = str(cm.index[0].date()), str(cm.index[-1].date())
out["listing"] = {c: str(cm[c].first_valid_index().date()) for c in ("OP-USDT", "APT-USDT", "ARB-USDT", "BTC-USDT")}
bad = 0
for s in C.UNIVERSE:
    px = io.ohlcv(C.sym_to_file(s))
    bad += int(((px["low"] > px[["open", "close"]].min(axis=1)) | (px["high"] < px[["open", "close"]].max(axis=1))
                | (px["volume"] <= 0)).sum())
    bad += int((px.index != px.index.normalize()).sum())
out["bad_bars"] = bad
out["spot"] = [C.sym_to_file(s) for s in C.UNIVERSE if io.spot_ohlcv(C.sym_to_file(s)) is not None] == list(cm.columns)
d = io.daily_funding_detail("BTC-USDT")
out["sources"] = sorted(set(d["source"]))
out["primary_days"] = int((d["n_primary"] > 0).sum())
oi = pq.read_table(C.RAW_OI / "BTC-USDT.parquet")
out["oi_schema"] = [str(f.type) for f in oi.schema]
out["oi_hours"] = sorted(set(oi.to_pandas()["ts"].dt.hour))
out["oi_days"] = int(io.daily_oi("BTC-USDT").notna().sum())
out["ts_type"] = str(pq.read_schema(C.RAW_OHLCV / "BTC-USDT.parquet").field("ts").type)
out["funding_cells"] = int(io.observed_funding_matrix(cm.index, cm.columns).notna().sum().sum())
print(json.dumps(out))
"""
    with tempfile.TemporaryDirectory(prefix="demo_") as tmp:
        data = Path(tmp) / "data"
        D.write(data, None)
        r = _run(["-c", code], data)
        assert r.returncode == 0, r.stderr[-1500:]
        out = json.loads(r.stdout)
    assert out["coins"] == [C.sym_to_file(s) for s in C.UNIVERSE]
    assert (out["first"], out["last"]) == (D.FIRST_DAY, "2026-09-30"), out
    assert out["listing"] == {
        "OP-USDT": "2022-06-01",
        "APT-USDT": "2022-10-19",
        "ARB-USDT": "2023-03-23",
        "BTC-USDT": D.FIRST_DAY,
    }, out["listing"]
    assert out["bad_bars"] == 0 and out["spot"] is True
    assert set(out["sources"]) <= {"primary", "fallback", "mixed"} and {"primary", "fallback"} <= set(out["sources"])
    assert out["primary_days"] == D.PRIMARY_FUNDING_DAYS, out["primary_days"]
    assert out["oi_schema"] == ["timestamp[ms, tz=UTC]", "null", "double"] and out["oi_hours"] == [D.OI_STAMP_HOUR]
    assert out["oi_days"] == D.OI_DAYS and out["ts_type"] == "timestamp[ms, tz=UTC]"
    assert out["funding_cells"] > 0


def test_panel_readable_by_the_library():
    import libfactors as L
    from factorlab.data import load_panels

    with tempfile.TemporaryDirectory(prefix="demo_") as tmp:
        data, panel = Path(tmp) / "data", Path(tmp) / "panel"
        D.write(data, panel)
        files = sorted(panel.glob("*.csv.gz"))
        assert [f.name[:-7] for f in files] == sorted(L.BINANCE_FIELDS)
        assert all(int.from_bytes(f.read_bytes()[4:8], "little") == 0 for f in files), "gzip header time is 0"
        head = gzip.decompress((panel / "close.csv.gz").read_bytes()).decode().splitlines()
        assert head[0] == "open_time," + ",".join(D.panel_symbols()), head[0]
        assert head[1].startswith("2022-01-01 00:00:00+00:00,")
        p = load_panels(panel)
        assert set(p) == set(L.BINANCE_FIELDS) and list(p["close"].columns) == sorted(D.panel_symbols())
        raw = L.load_raw_binance(panel)
        assert list(raw) == [C.sym_to_file(s) for s in C.UNIVERSE]
        panels = L.build_panels_binance(raw)
        assert set(panels) == set(L.BINANCE_FIELDS) and panels["close"].shape == (D.N_DAYS, 20)
        assert panels["close"]["ARB-USDT"].first_valid_index() == pd.Timestamp("2023-03-23", tz="UTC")
        # the panel's funding is the daily sum of the fallback venue's prints
        fb = pd.read_parquet(data / "raw" / f"funding_{C.FUNDING_FALLBACK_EXCHANGE_ID}" / "ETH-USDT.parquet")
        daily = fb.set_index("ts")["funding_rate"].resample("1D", label="left", closed="left").sum(min_count=1)
        got = raw["ETH-USDT"]["funding_rate"].reindex(daily.index)
        assert ((got - daily).abs() <= 1e-9 * daily.abs().clip(lower=1e-6)).all()
        assert L.binance_panel_fingerprint(panel)["files"].keys() == {f.name for f in files}, (
            "the marker is no panel file"
        )


def test_refuses_market_data_and_writes_nothing():
    import update as U

    with tempfile.TemporaryDirectory(prefix="demo_") as tmp:
        t = Path(tmp)
        # a data folder with one file that is not demo data
        real = t / "real"
        (real / "raw" / "ohlcv").mkdir(parents=True)
        (real / "raw" / "ohlcv" / "BTC-USDT.parquet").write_bytes(b"market data")
        before = _hashes(real)
        assert "not demo data" in (D.refusal(real, None) or "")
        try:
            D.write(real, None)
            raise AssertionError("the demo wrote into a folder with market data")
        except D.DemoRefused as exc:
            assert "FACTORLAB_DATA" in str(exc.code)
        assert _hashes(real) == before
        r = _run(["update.py", "--demo", "--skip-binance"], real)
        assert r.returncode == 1 and "not demo data" in r.stderr, (r.returncode, r.stderr[-600:])
        assert _hashes(real) == before
        # a protected folder
        prot = t / "prot"
        prot.mkdir()
        (prot / C.PROTECT_MARKER).write_text("", encoding="utf-8")
        assert "protected" in (D.refusal(prot, None) or "")
        # an empty folder (empty sub-folders are allowed) is fine; a Binance panel that is not demo is not
        empty, panel = t / "empty", t / "panel"
        (empty / "raw" / "ohlcv").mkdir(parents=True)
        panel.mkdir()
        assert D.refusal(empty, panel) is None
        (panel / "close.csv.gz").write_bytes(b"x")
        assert "--skip-binance" in (D.refusal(empty, panel) or "") and D.refusal(empty, None) is None
        assert U.demo_check(True, False, empty, panel) and U.demo_check(True, True, empty, panel) is None
        assert U.demo_check(False, False, empty, panel) is None, "a build from market data keeps a real panel"
        # a demo folder: the demo writes again; a build from market data refuses it and the demo panel
        demo, dpanel = t / "demo", t / "dpanel"
        D.write(demo, dpanel)
        assert D.refusal(demo, dpanel) is None and D.is_demo(demo) and D.is_demo_panel(dpanel)
        assert "demo_data.py --remove" in U.demo_check(False, True, demo, dpanel)
        assert "demo_data.py --remove" in U.demo_check(False, False, empty, dpanel)
        assert U.demo_check(False, True, empty, dpanel) is None
        before = _hashes(demo)
        for args in (["update.py", "--offline", "--skip-binance"], ["update.py", "--skip-binance"]):
            r = _run(args, demo)
            assert r.returncode == 1 and "synthetic demo data" in r.stderr, (args, r.stderr[-600:])
        assert _hashes(demo) == before
        # --remove deletes only demo folders
        assert D.remove(real, panel) == [] and _hashes(real) and (panel / "close.csv.gz").exists()
        assert len(D.remove(demo, dpanel)) == 2
        assert not any(demo.iterdir()) and not dpanel.exists()


def test_stopped_write_or_remove_keeps_the_marker():
    """A demo write or removal that stops part way leaves the folders marked as demo data, so a
    market build refuses them and a second removal completes."""
    from unittest import mock

    import update as U

    with tempfile.TemporaryDirectory(prefix="demo_") as tmp:
        t = Path(tmp)
        data, panel = t / "data", t / "panel"
        # the panel write stops after the raw files: both markers are on disk
        with mock.patch("factorlab.data.save_panels", side_effect=OSError("disk full")):
            try:
                D.write(data, panel)
                raise AssertionError("the stopped panel write did not raise")
            except OSError:
                pass
        assert D.is_demo(data) and D.is_demo_panel(panel)
        assert any((data / "raw").rglob("*.parquet")), "the raw files come before the panel"
        assert "demo_data.py --remove" in U.demo_check(False, True, data, panel)
        assert "demo_data.py --remove" in U.demo_check(False, False, t / "empty", panel)
        # a removal that stops part way keeps the marker; a second removal completes
        real_rmtree = D.shutil.rmtree
        calls = []

        def stop_once(p, *a, **k):
            calls.append(p)
            if len(calls) == 1:
                raise PermissionError(f"locked: {p}")
            return real_rmtree(p, *a, **k)

        with mock.patch.object(D.shutil, "rmtree", side_effect=stop_once):
            try:
                D.remove(data, panel)
                raise AssertionError("the stopped removal did not raise")
            except PermissionError:
                pass
        assert D.is_demo(data), "the marker goes last"
        assert "demo_data.py --remove" in U.demo_check(False, True, data, panel)
        assert len(D.remove(data, panel)) == 2
        assert not any(data.iterdir()) and not panel.exists()


def test_update_plan_demo():
    import datetime as dt
    from unittest import mock

    import update as U

    today = dt.date(2025, 1, 8)
    with tempfile.TemporaryDirectory(prefix="demo_") as tmp, mock.patch.object(C, "DATA", Path(tmp)):
        demo = U.plan(today, demo=True)
        names = [s.name for s in demo]
        assert names[0] == U.DEMO_STEP and not any(s.online for s in demo), names
        assert demo[0].argv[-1] == "--write" and U.plan(today, demo=True, skip_binance=True)[0].argv[-1] == "--no-panel"
        assert "library on Binance data, compute" in names
        assert "library on Binance data, compute" not in [s.name for s in U.plan(today, demo=True, skip_binance=True)]
        # no factor test (the demo registers no factor); the factor lab after the library steps
        assert U.FACTOR_TEST_STEP not in names and names[1] == "labels", names
        assert names.index("library on Binance data, screen") < names.index("factor lab") < names.index("dashboard")
        for steps in (demo, U.plan(today, offline=True), U.plan(today, skip_binance=True)):
            assert all(not s.env for s in steps), "no step gets an extra environment variable"
            assert not [s.name for s in steps if "paper" in s.name], [s.name for s in steps]
        # the text after a failed step
        online = next(s for s in U.plan(today, skip_binance=True) if s.name == "OKX download")
        assert "Run the same command again" in U.failure_hint(online)
        labels = next(s for s in demo if s.name == "labels")
        assert '--from "labels"' in U.failure_hint(labels)
        assert U.start_lines(demo)[0].startswith("Factorlab ") and len(U.start_lines(demo)) == 1
        assert U.start_lines(U.plan(today, skip_binance=True))[1] == U.DATA_TERMS


def test_meta_demo_flag():
    from unittest import mock

    import export_web as E

    with tempfile.TemporaryDirectory(prefix="demo_") as tmp:
        t = Path(tmp)
        demo, plain, out = t / "demo", t / "plain", t / "out"
        D.write(demo, None)
        plain.mkdir()
        out.mkdir()
        for data, want in ((demo, True), (plain, False)):
            with mock.patch.object(E.C, "DATA", data):
                meta = E.build_meta(out)
            assert meta["demo"] is want, (data.name, meta["demo"])
            assert isinstance(meta["version"], str) and meta["version"]


_DEMO_BODY = r"""
(async () => {
  for (const f of ["assets/fl-core.js", "assets/fl-charts.js", "data/meta.js", "assets/fl-app.js"]) run(f);
  await tick();
  const routes = ["library", "research", "mine", "system", "factor/FT-0001"];
  const out = {};
  for (const p of routes) {
    await go("#/" + p);
    const b = main().querySelectorAll(".fl-demo-banner");
    out[p] = b.length ? b[0].querySelector(".fl-callout-title").textContent : null;
  }
  out.snapshot = D.documentElement.querySelector(".fl-snapshot").textContent;
  process.stdout.write(JSON.stringify(out));
})().catch(e => { console.error(e && e.stack || e); process.exit(1); });
"""


def _node() -> str | None:
    import shutil

    n = os.environ.get("FACTORLAB_NODE") or os.environ.get("NODE") or shutil.which("node")
    return n if n and Path(n).exists() else None


def test_banner_on_every_page():
    """The page shell in node, with the page fixture of test_export_web.py: meta.demo = true puts
    the banner on every route (also a route without data and the page of a test id that the export
    does not have) and in the header;
    meta.demo = false shows no banner."""
    import test_export_web as TW
    from webexport import common as WC

    node = _node()
    if node is None:
        print("        (skipped the node run: set NODE or FACTORLAB_NODE)")
        return
    stub = TW._SHELL_STUB[: TW._SHELL_STUB.index("(async () => {")] + _DEMO_BODY
    for flag in (True, False):
        with tempfile.TemporaryDirectory(prefix="demo_") as tmp:
            root = Path(tmp) / "web"
            root.mkdir()
            TW._shell_fixture(root)
            meta = TW._payload((root / "data" / "meta.js").read_text(encoding="ascii"))[1]
            meta["demo"] = flag
            WC.write_section(root / "data", "meta", meta)
            r = subprocess.run([node, "-e", stub, str(root)], capture_output=True, text=True, timeout=120)
            assert r.returncode == 0, r.stderr[-1500:]
            out = json.loads(r.stdout)
        snap = out.pop("snapshot")
        if flag:
            assert set(out.values()) == {D.BANNER}, out
            assert D.BANNER in snap and "Data is " not in snap, snap
        else:
            assert set(out.values()) == {None}, out
            assert D.BANNER not in snap
    app = (ENGINE.parent / "web" / "assets" / "fl-app.js").read_text(encoding="utf-8")
    assert f'"{D.BANNER}"' in app, "the page text equals demo_data.BANNER"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} demo data tests passed")
    raise SystemExit(1 if failed else 0)
