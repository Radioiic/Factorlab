"""Integrity tests for the label table (labels.py) and data_io.label_table.

Usage:
    python -B test_labels.py

The suite reads no data folder of the repository. At the start it sets FACTORLAB_DATA to a new
folder under the git-ignored engine/.tmp/ and removes that folder at the end. Each test writes
its own world of daily bars into a new data folder and points the config paths at it.

The properties:
    1. alignment: fwd_ret_1 is the return of the next bar, fwd_ret_7 the return of the next
       seven bars, both stamped on the bar of the signal date
    2. no fabricated target: the last row has no fwd_ret_1, the last seven rows no fwd_ret_7
    3. a settled label never changes when bars are added (a label is final once its bars exist)
    4. a coin gets its first row on the bar where close.pct_change(90) is formed, and the rows
       of the other coins do not change
    5. ingest's upsert is idempotent and keeps the newest restatement
    6. the shifts are row shifts on each coin's own bars (a missing bar moves a label span)
    7. eligible() is True exactly on the label rows
    8. the label table is equal to an independent computation from the bars on a demo fixture
    9. a digest of the label table on exact input prices is pinned (equal on every platform)
   10. build() writes only its file, and refuses a protected folder before it computes
   11. an absent label table is "no data" for the exporter
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
(ENGINE / ".tmp").mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="test_labels_", dir=ENGINE / ".tmp"))
atexit.register(shutil.rmtree, TMP, True)
os.environ["FACTORLAB_DATA"] = str(TMP / "data")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
import data_io  # noqa: E402
import labels  # noqa: E402

SYMS = ["AAA-USDT", "BBB-USDT", "CCC-USDT"]
ENGINE_UNIVERSE = list(C.UNIVERSE)  # the 20 coins of config.py
N_DAYS = 200
PATHS = ("DATA", "RAW_OHLCV", "RAW_SPOT_OHLCV", "RAW_FUNDING", "RAW_FUNDING_FB", "RAW_OI", "LABELS", "UNIVERSE")
# the sha256 of the label table of the exact-price fixture (test_label_digest_on_literal_prices)
LABEL_DIGEST = "00d323e1f19fe7eb335de7f3ef53e91b83e03cef7df1d47b2fe2164d97d4eab2"


@contextmanager
def sandbox(syms: list[str] | None = SYMS):
    """A new data folder: the config paths and the universe point at it; restored on exit.
    syms=None keeps the 20 coins of config.py."""
    saved = {n: getattr(C, n) for n in PATHS}
    data = Path(tempfile.mkdtemp(prefix="world_", dir=TMP)) / "data"
    raw = data / "raw"
    C.DATA = data
    C.RAW_OHLCV = raw / "ohlcv"
    C.RAW_SPOT_OHLCV = raw / "spot_ohlcv"
    C.RAW_FUNDING = raw / "funding"
    C.RAW_FUNDING_FB = raw / f"funding_{C.FUNDING_FALLBACK_EXCHANGE_ID}"
    C.RAW_OI = raw / "open_interest"
    C.LABELS = data / "labels"
    for p in (C.RAW_OHLCV, C.RAW_SPOT_OHLCV, C.RAW_FUNDING, C.RAW_FUNDING_FB, C.RAW_OI):
        p.mkdir(parents=True)
    C.UNIVERSE = ENGINE_UNIVERSE if syms is None else [s.replace("-", "/") + ":USDT" for s in syms]
    try:
        yield data
    finally:
        for name, value in saved.items():
            setattr(C, name, value)


def write_bars(sym: str, close: np.ndarray, idx: pd.DatetimeIndex, volume=None) -> None:
    close = np.asarray(close, dtype=np.float64)
    df = pd.DataFrame(
        {
            "ts": idx.as_unit("ms"),
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(close), 2e6) if volume is None else volume,
        }
    )
    df.to_parquet(C.RAW_OHLCV / f"{sym}.parquet", index=False)


def write_world(seed=0, n_days=N_DAYS, syms=SYMS) -> pd.DatetimeIndex:
    """Random-walk daily bars for each symbol (gapless)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")
    for sym in syms:
        write_bars(sym, 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, n_days))), idx)
    return idx


def append_bar(sym: str, factor: float = 1.01) -> pd.Timestamp:
    df = pd.read_parquet(C.RAW_OHLCV / f"{sym}.parquet")
    last = df.iloc[-1]
    new_day = last["ts"] + pd.Timedelta(days=1)
    c = float(last["close"]) * factor
    row = pd.DataFrame([{"ts": new_day, "open": c, "high": c * 1.02, "low": c * 0.98, "close": c, "volume": 2e6}])
    out = pd.concat([df, row], ignore_index=True)
    out["ts"] = out["ts"].dt.as_unit("ms")  # the unit of the bar files
    out.to_parquet(C.RAW_OHLCV / f"{sym}.parquet", index=False)
    return new_day


def table() -> pd.DataFrame:
    return labels.compute().set_index(["date", "symbol"]).sort_index()


def tree(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --- 1. alignment -------------------------------------------------------------------


def test_target_alignment():
    with sandbox():
        write_world()
        f = table()
        for sym in SYMS:
            close = pd.read_parquet(C.RAW_OHLCV / f"{sym}.parquet").set_index("ts")["close"]
            g = f.xs(sym, level="symbol")
            want1 = close.pct_change().shift(-1).reindex(g.index)
            want7 = (close.shift(-7) / close - 1).reindex(g.index)
            assert np.array_equal(g["fwd_ret_1"].to_numpy(), want1.to_numpy(), equal_nan=True), sym
            assert np.array_equal(g["fwd_ret_7"].to_numpy(), want7.to_numpy(), equal_nan=True), sym
            # the row of D holds close(D + 7) / close(D) - 1: one Monday-to-Monday span for a Sunday D
            d = g.index[10]
            assert g.loc[d, "fwd_ret_7"] == close[d + pd.Timedelta(days=7)] / close[d] - 1
        assert list(labels.compute().columns) == ["date", "symbol", "fwd_ret_1", "fwd_ret_7"]
        assert labels.LABEL == "fwd_ret_7" and labels.LABEL_COLUMNS == ("fwd_ret_1", "fwd_ret_7")


# --- 2. no fabricated latest target -----------------------------------------------------


def test_no_fabricated_latest_target():
    with sandbox():
        idx = write_world()
        f = table()
        for sym in SYMS:
            g = f.xs(sym, level="symbol")
            assert g.index.max() == idx[-1], "the newest bar has no label row"
            assert np.isnan(g.loc[idx[-1], "fwd_ret_1"]), "the newest row has a fwd_ret_1: the next bar does not exist"
            assert g["fwd_ret_7"].tail(7).isna().all(), "the last 7 rows have a fwd_ret_7: the week is not complete"
            assert g["fwd_ret_7"].iloc[:-7].notna().all()
        assert int(f["fwd_ret_1"].isna().sum()) == len(SYMS)
        assert int(f["fwd_ret_7"].isna().sum()) == 7 * len(SYMS)


# --- 3. settled labels --------------------------------------------------------------------


def test_settled_labels_never_change_when_bars_are_added():
    with sandbox():
        idx = write_world()
        before = table()
        for sym in SYMS:
            append_bar(sym, factor=3.0)  # a violent move on the next bar
        after = table()
        last = idx[-1]
        prior = before["fwd_ret_1"].drop(index=last, level="date")
        pd.testing.assert_series_equal(prior, after.loc[prior.index, "fwd_ret_1"])
        for sym in SYMS:
            assert np.isclose(after.loc[(last, sym), "fwd_ret_1"], 2.0, rtol=1e-12), "the new bar fills the last label"
        dates = before.index.get_level_values("date")
        settled = before["fwd_ret_7"][dates <= last - pd.Timedelta(days=7)]
        pd.testing.assert_series_equal(settled, after.loc[settled.index, "fwd_ret_7"])
        newly = last - pd.Timedelta(days=6)
        for sym in SYMS:
            assert np.isnan(before.loc[(newly, sym), "fwd_ret_7"]) and not np.isnan(
                after.loc[(newly, sym), "fwd_ret_7"]
            )
        # the rows that existed are still there, and the new bar adds exactly one row per coin
        assert before.index.isin(after.index).all() and len(after) == len(before) + len(SYMS)


# --- 4. a newly eligible symbol -----------------------------------------------------------


def test_newly_eligible_symbol_rule():
    new_sym = "NEW-USDT"
    with sandbox(SYMS + [new_sym]):
        idx = write_world()
        rng = np.random.default_rng(1)
        n_new = labels.MIN_BARS - 1  # one bar short
        write_bars(new_sym, 50 * np.exp(np.cumsum(rng.normal(0.01, 0.05, n_new))), idx[-n_new:])
        before = table()
        assert new_sym not in before.index.get_level_values("symbol"), "a coin below MIN_BARS has a row"
        assert not labels.eligible()[new_sym].any()
        for sym in SYMS + [new_sym]:
            new_day = append_bar(sym)
        after = table()
        # the rows of the other coins before the new bar are unchanged (labels settle; no row moves)
        old = before.index[before.index.get_level_values("date") < idx[-1] - pd.Timedelta(days=7)]
        pd.testing.assert_frame_equal(before.loc[old], after.loc[old])
        rows = after.xs(new_sym, level="symbol")
        assert list(rows.index) == [new_day], list(rows.index)  # pct_change(90) is formed on bar 91 only
        close = pd.read_parquet(C.RAW_OHLCV / f"{new_sym}.parquet")["close"]
        assert len(close) == labels.MIN_BARS and close.pct_change(labels.WARMUP_ROWS).notna().sum() == 1
        el = labels.eligible()
        assert el[new_sym].sum() == 1 and bool(el.loc[new_day, new_sym])


# --- 5. upsert idempotency ----------------------------------------------------------------


def test_upsert_idempotency():
    from ingest import _upsert

    with sandbox() as data:
        path = data / "u.parquet"
        idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        a = pd.DataFrame({"ts": idx, "close": np.arange(10.0)})
        assert _upsert(path, a) == 10
        assert _upsert(path, a) == 0, "an upsert of the same rows adds rows"
        one = pd.read_parquet(path)
        assert _upsert(path, a) == 0
        pd.testing.assert_frame_equal(one, pd.read_parquet(path))
        restated = pd.DataFrame({"ts": idx[-2:], "close": [99.0, 98.0]})
        assert _upsert(path, restated) == 0
        three = pd.read_parquet(path)
        assert len(three) == 10 and three.set_index("ts")["close"].iloc[-2:].tolist() == [99.0, 98.0]


# --- 6. row shifts ------------------------------------------------------------------------


def test_shifts_are_row_shifts():
    """A coin without its bar of one day: the labels are row shifts on the coin's own bars. The
    row before the gap spans two calendar days in fwd_ret_1 and eight in fwd_ret_7. The engine keeps
    this rule; the forward test needs every bar of a week (or a final gap)."""
    with sandbox():
        idx = write_world()
        gap = idx[150]
        df = pd.read_parquet(C.RAW_OHLCV / "AAA-USDT.parquet")
        df[df["ts"] != gap].to_parquet(C.RAW_OHLCV / "AAA-USDT.parquet", index=False)
        close = df.set_index("ts")["close"]
        g = table().xs("AAA-USDT", level="symbol")
        assert gap not in g.index
        before_gap = idx[149]
        assert g.loc[before_gap, "fwd_ret_1"] == close[idx[151]] / close[before_gap] - 1  # two calendar days
        d = idx[145]
        assert g.loc[d, "fwd_ret_7"] == close[idx[153]] / close[d] - 1  # seven rows = eight calendar days
        assert labels.eligible().loc[gap, "AAA-USDT"] == np.False_


# --- 7. the eligible matrix ---------------------------------------------------------------


def test_eligible_matrix_equals_the_label_rows():
    new_sym = "LATE-USDT"
    with sandbox(SYMS + [new_sym]):
        idx = write_world()
        write_bars(new_sym, np.linspace(10, 20, 120), idx[-120:])
        df = pd.read_parquet(C.RAW_OHLCV / "BBB-USDT.parquet")
        df[df["ts"] != idx[120]].to_parquet(C.RAW_OHLCV / "BBB-USDT.parquet", index=False)
        el = labels.eligible()
        assert list(el.columns) == [C.sym_to_file(s) for s in C.UNIVERSE]
        assert el.index.equals(data_io.close_matrix().index) and el.dtypes.eq(bool).all()
        rows = labels.compute()
        got = {(d, s) for d in el.index for s in el.columns if el.loc[d, s]}
        assert got == set(zip(rows["date"], rows["symbol"])), "eligible() is not the set of label rows"
        assert el[new_sym].sum() == 120 - labels.WARMUP_ROWS
        # another index: the same cells, False for a date without a bar
        other = pd.DatetimeIndex([idx[100], idx[120], idx[-1] + pd.Timedelta(days=30)])
        sub = labels.eligible(other)
        assert sub.loc[idx[100]].equals(el.loc[idx[100]]) and not sub.iloc[2].any()
        assert not bool(sub.loc[idx[120], "BBB-USDT"]) and bool(sub.loc[idx[120], "AAA-USDT"])


# --- 8. an independent computation ---------------------------------------------------------


def test_labels_equal_an_independent_recomputation():
    """On a demo fixture built in this process: labels.compute() is bit-equal to the labels that
    this test computes from the bar files with pandas alone: close.shift(-7) / close - 1 and
    close.pct_change().shift(-1), on the rows where close.pct_change(90) is formed."""
    import demo_data

    with sandbox(None) as data:
        m = demo_data.market(demo_data.SEED)
        demo_data.write_raw(data, m)
        parts = []
        for symbol in C.UNIVERSE:
            name = C.sym_to_file(symbol)
            bars = pd.read_parquet(data / "raw" / "ohlcv" / f"{name}.parquet").sort_values("ts")
            close = bars["close"].astype("float64").reset_index(drop=True)
            keep = close.pct_change(90).notna()
            part = pd.DataFrame(
                {
                    "date": pd.to_datetime(bars["ts"], utc=True).reset_index(drop=True),
                    "symbol": name,
                    "fwd_ret_1": close.pct_change().shift(-1),
                    "fwd_ret_7": close.shift(-7) / close - 1,
                }
            )
            parts.append(part[keep.to_numpy()])
        want = pd.concat(parts).sort_values(["date", "symbol"]).reset_index(drop=True)
        got = labels.compute()
        assert list(got.columns) == ["date", "symbol", "fwd_ret_1", "fwd_ret_7"]
        assert len(got) > 30_000 and got["symbol"].nunique() == 20
        pd.testing.assert_frame_equal(
            got[["date", "symbol"]].astype({"symbol": str}),
            want[["date", "symbol"]].astype({"date": got["date"].dtype, "symbol": str}),
            check_exact=True,
        )
        for col in ("fwd_ret_1", "fwd_ret_7"):
            a, b = got[col].to_numpy(dtype=np.float64), want[col].to_numpy(dtype=np.float64)
            assert np.array_equal(a, b, equal_nan=True), col


# --- 9. the digest on exact prices ----------------------------------------------------------


def literal_closes(coin: int, n: int = 120) -> np.ndarray:
    """Exact binary floats (quarters of whole numbers), not results of exp or cumprod: the input
    bits are the same on every platform."""
    return np.array([(1000 + (k * k * (coin + 3) + 7 * k + 31 * coin) % 251) / 4.0 for k in range(n)])


def label_digest(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(pd.DatetimeIndex(df["date"]).as_unit("ns").asi8, dtype="<i8").tobytes())
    h.update("\t".join(df["symbol"].astype(str)).encode("ascii"))
    a = df[list(labels.LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    a = np.where(np.isnan(a), np.nan, a) + 0.0
    h.update(np.ascontiguousarray(a, dtype="<f8").tobytes())
    return h.hexdigest()


def test_label_digest_on_literal_prices():
    """IEEE division and subtraction are correctly rounded, so the digest is equal on every
    platform. (A digest of the demo market is not pinned: its prices come from exp and cumprod.)"""
    with sandbox():
        idx = pd.date_range("2024-01-07", periods=120, freq="D", tz="UTC")
        for i, sym in enumerate(SYMS):
            write_bars(sym, literal_closes(i), idx)
        df = labels.compute()
        assert len(df) == 3 * (120 - labels.WARMUP_ROWS)
        digest = label_digest(df)
        assert digest == LABEL_DIGEST, digest


# --- 10. the write --------------------------------------------------------------------------


def test_build_writes_only_its_file_and_refuses_a_protected_folder():
    with sandbox() as data:
        write_world()
        before = tree(data)
        df = labels.build()
        after = tree(data)
        new = sorted(set(after) - set(before))
        assert new == ["labels/labels.parquet"], new
        assert all(after[k] == v for k, v in before.items()), "build() changed another file"
        pd.testing.assert_frame_equal(pd.read_parquet(labels.path()), df)
        pd.testing.assert_frame_equal(df, labels.compute())
        assert labels.main([]) == 0 and tree(data) == after, "a second build changed the file"
        assert labels.main(["--status"]) == 0 and tree(data) == after, "--status wrote a file"
        # the marker file protects the folder labels (and raw): build() refuses before it computes
        (data / C.PROTECT_MARKER).write_text("protected\n")
        saved = C.PROTECTED_DIRS
        C.PROTECTED_DIRS = tuple(dict.fromkeys((*saved, "labels")))
        env = os.environ.pop(C.PROTECT_OVERRIDE_ENV, None)
        try:
            for target in (None, data / "raw" / "labels.parquet"):
                try:
                    labels.build(target)
                except C.WriteRefused as exc:
                    assert "refusing to write" in str(exc)
                else:
                    raise AssertionError(f"build({target}) was not refused")
        finally:
            C.PROTECTED_DIRS = saved
            if env is not None:
                os.environ[C.PROTECT_OVERRIDE_ENV] = env
        assert {k: v for k, v in tree(data).items() if k != C.PROTECT_MARKER} == after


# --- 11. no data ----------------------------------------------------------------------------


def test_label_table_absent_is_no_data():
    from webexport.common import is_no_data

    with sandbox():
        try:
            data_io.label_table()
        except SystemExit as exc:
            assert str(exc) == "No label table. Run labels.py first.", str(exc)
            assert is_no_data(exc), "the exporter does not treat an absent label table as no data"
        else:
            raise AssertionError("an absent label table was read")
        idx = write_world()
        labels.build()
        full = data_io.label_table()
        pd.testing.assert_frame_equal(full, labels.compute())
        sundays = [d for d in idx if d.weekday() == 6][20:23]
        part = data_io.label_table(columns=["date", "symbol", "fwd_ret_7"], dates=sundays)
        assert list(part.columns) == ["date", "symbol", "fwd_ret_7"]
        assert sorted(set(part["date"])) == sundays and len(part) == 3 * len(SYMS)
        want = full[full["date"].isin(sundays)][["date", "symbol", "fwd_ret_7"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(part.reset_index(drop=True), want)
        assert data_io.label_table(dates=[idx[0] - pd.Timedelta(days=9)]).empty


# --- runner -------------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001 - report every test, then fail at the end
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} label-table tests passed")
    raise SystemExit(1 if failed else 0)
