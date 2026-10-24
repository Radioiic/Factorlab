"""Tests of userfactor.py: the formula check, the values at each Sunday cut, the values file, the
risk proxies and the factor numbers.

Usage:
    python -B test_userfactor.py

The suite reads no data folder of the repository. At the start it sets FACTORLAB_DATA to a new
folder under the git-ignored engine/.tmp/ and removes that folder at the end. It writes the
synthetic demo market (demo_data.market, demo_data.write_raw) and the label table into that
folder, and a demo Binance panel into a temporary folder.

The properties:
    1. a formula that the parser refuses, and a window or lag that is not a whole number from 1
       to 1000, give an InputError before any computation
    2. a formula that reads a later row fails the Sunday-cut check; causal formulas are proven;
       the official values are the cut values; a formula without a value is "empty"
    3. the fields of each source: OKX has no premium and its open interest is refused; the notes
    4. the canonical text and its sha256 do not change with spacing or extra parentheses
    5. the values file: CSV and Parquet give the same frame; the refusals; the symbol names; the
       dropped rows are counted
    6. a library id gives its definition from the catalog
    7. the factor numbers equal an independent computation; the risk proxies equal the
       regressors of the library screen on the G Sundays
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import sqlite3
import tempfile
from datetime import date
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
(ENGINE / ".tmp").mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="test_userfactor_", dir=ENGINE / ".tmp"))
atexit.register(shutil.rmtree, TMP, True)
os.environ["FACTORLAB_DATA"] = str(TMP / "data")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
import data_io  # noqa: E402
import demo_data  # noqa: E402
import labels  # noqa: E402
import libfactors as lf  # noqa: E402
import userfactor as uf  # noqa: E402

SYMS = [C.sym_to_file(s) for s in C.UNIVERSE]
_STATE: dict = {}


def world() -> dict:
    """The demo market in this suite's data folder (written once), its OKX source and the label table."""
    if not _STATE:
        m = demo_data.market(demo_data.SEED)
        demo_data.write_raw(C.DATA, m)
        labels.build()
        _STATE.update(market=m, okx=uf.load_source("okx"))
    return _STATE


def binance_source() -> dict:
    if "binance" not in _STATE:
        w = world()
        panel = TMP / "panel"
        demo_data.write_panel(panel, w["market"])
        _STATE["binance_raw"] = lf.load_raw_binance(panel)
        _STATE["binance"] = uf.load_source("binance", _STATE["binance_raw"])
    return _STATE["binance"]


def refused(fn, *words) -> str:
    try:
        fn()
    except uf.InputError as exc:
        text = str(exc)
        for w in words:
            assert w in text, (w, text)
        return text
    raise AssertionError(f"not refused: {words}")


def csv_file(text: str, name: str = "values.csv") -> Path:
    d = Path(tempfile.mkdtemp(prefix="vf_", dir=TMP))
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p


def sundays(n: int, last: str = "2026-09-27") -> list[str]:
    return [str(d.date()) for d in pd.date_range(end=last, periods=n, freq="7D")]


# --------------------------------------------------------------------------- 1. the static check


def test_parse_errors_become_input_errors():
    long = "close" + " + close" * 600
    assert len(long) > 4000
    deep = "abs(" * 64 + "close" + ")" * 64
    for expr in ("close +", "x", "foo(close)", "close.mean()", "'a'", "True", "close % 2", "ts_mean(close, w=5)",
                 long, deep, "2 ** 1000"):  # fmt: skip
        refused(lambda e=expr: uf.validate_formula(e), "The formula is not valid")
    refused(lambda: uf.validate_formula(3), "must be a text")  # type: ignore[arg-type]


def test_refusals_of_a_formula_are_plain_text():
    """The text of a refused formula says what is wrong in plain words: no class name of the Python
    syntax tree, no exception name and no hint of the Python parser. A count of 1 has the singular noun."""
    internals = ("Compare", "BoolOp", "Subscript", "Attribute", "IfExp", "JoinedStr", "Starred", "Tuple",
                 "Error", "Perhaps", "Maybe you meant", "never closed", "unmatched", "literal", "invalid syntax",
                 "unknown>", "keyword", "op in")  # fmt: skip
    cases = {
        "close > 1": "a comparison",
        "close and ret": "and / or",
        "close[1]": "an index in square brackets",
        "close if ret else 1": "if / else",
        "f'{close}'": "text in quotation marks",
        "close % 2": "the operator % is not allowed",
        "ts_mean(close 5)": "the syntax is not correct",
        "ts_mean(ret, 5": "a bracket is not closed",
        "ts_mean(ret, 5))": "a closing bracket has no opening bracket",
        "'abc": "a quotation mark is not closed",
        "ts_mean(close, w=5)": "an argument with a name",
        "close.mean()": "only the name of an operator",
        "": "the expression is empty",
    }
    for expr, words in cases.items():
        text = refused(lambda e=expr: uf.validate_formula(e), "The formula is not valid", words)
        shown = text.replace(repr(expr), "")
        hits = [w for w in internals if w in shown]
        assert not hits, (hits, text)
    refused(lambda: uf.validate_formula("abs(close, 5)"), "abs takes 1 argument, not 2.")


def test_lookahead_formulas_are_refused_statically():
    for expr in (
        "delay(close, -1)",
        "delay(close, 1 - 2)",
        "delta(close, -3)",
        "pct_change(close, -1)",
        "pct_change(close, 0)",
        "ts_autocorr(ret, 10, -2)",
        "ts_mean(close, 0)",
        "ts_mean(close, 2.7)",
        "ts_mean(close, 1001)",
        "ts_mean(close, ts_mean(close, 5))",
    ):
        refused(lambda e=expr: uf.validate_formula(e, "okx"), "must be an integer from 1 to 1000")
    refused(lambda: uf.validate_formula("ts_mean(close)"), "takes 2 arguments")
    refused(lambda: uf.validate_formula("ts_autocorr(ret)"), "takes 2 or 3 arguments")
    refused(lambda: uf.validate_formula("ts_mean(5, 10)"), "argument 1 must read data")
    refused(lambda: uf.validate_formula("cs_rank(2)"), "argument 1 must read data")
    # whole numbers written as floats, and the largest window, are accepted
    assert uf.validate_formula("ts_mean(close, 5.0)")["canonical"] == "ts_mean(close, 5.0)"
    assert uf.validate_formula("ts_autocorr(ret, 10, 2)")["source"] == "okx"
    assert uf.validate_formula("ts_mean(close, 1000)")["max_lookback"] == 1000


# --------------------------------------------------------------------------- 2. the Sunday cuts


def test_lookahead_formula_fails_the_sunday_cut_check():
    src = world()["okx"]
    r = uf.formula_values("delay(close, -1) / close - 1", src)  # the static check is not called here
    assert r["pit"]["status"] == "fail", r["pit"]
    assert r["pit"]["first_diff_sunday"] == str(r["sundays"][0].date()), r["pit"]
    assert r["pit"]["diff_cells"] > 0
    # the cut value of a lookahead formula has no value on its last row
    assert not np.isfinite(r["values"]).any() and np.isfinite(r["full"]).any()


def test_causal_formulas_are_proven():
    src = world()["okx"]
    for expr in (
        "-ts_mean(ex_ret, 20)",
        "ts_rank(close, 365)",
        "cs_rank(ts_corr(ret, btc_ret, 60)) * ts_zscore(decay_linear(hl_range, 30), 90)",
    ):
        r = uf.formula_values(uf.validate_formula(expr)["canonical"], src)
        p = r["pit"]
        assert p["status"] == "proven" and p["diff_cells"] == 0 and p["finite_cells"] > 0, (expr, p)
        assert p["sundays"] == len(r["sundays"]) and p["cells"] == r["values"].size
        assert all(d.dayofweek == 6 and d.date() <= src["complete_through"] for d in r["sundays"])


def test_cut_values_are_the_official_values():
    src = world()["okx"]
    expr = "ts_zscore(ret, 30)"
    r = uf.formula_values(expr, src)
    idx = src["panels"]["close"].index
    for k in (5, 60, len(r["sundays"]) - 1):
        s = r["sundays"][k]
        part = {f: v.loc[:s] for f, v in src["panels"].items()}
        want = lf._values(lf._evaluator(part), expr)[-1]
        assert np.array_equal(np.where(np.isfinite(want), want, np.nan), r["values"][k], equal_nan=True), s
        one = uf.sunday_value(expr, src, s.date())
        assert one["equal"] and np.array_equal(one["values"], r["values"][k], equal_nan=True)
        assert idx[idx.get_loc(s)] == s
    # `last` stops the Sundays
    short = uf.formula_values(expr, src, last=date(2023, 1, 1))
    assert short["sundays"][-1] <= pd.Timestamp("2023-01-01", tz="UTC") and len(short["sundays"]) < len(r["sundays"])


def test_empty_formula_values_are_refused():
    raw = lf.load_raw()
    cut = {s: {**r, "ohlcv": r["ohlcv"].iloc[:300]} for s, r in raw.items()}
    src = uf.load_source("okx", cut)
    r = uf.formula_values("ts_mean(close, 1000)", src)
    assert r["pit"]["status"] == "empty" and r["pit"]["finite_cells"] == 0, r["pit"]


def test_compute_errors_become_input_errors():
    src = world()["okx"]
    saved = lf._values

    def boom(ev, expr):
        raise ZeroDivisionError("a test error")

    lf._values = boom
    try:
        refused(lambda: uf.formula_values("ts_mean(close, 5)", src), "cannot be computed", "ZeroDivisionError")
    finally:
        lf._values = saved


# --------------------------------------------------------------------------- 3. sources and notes


def test_venue_fields_and_notes():
    refused(lambda: uf.validate_formula("premium", "okx"), "The formula reads premium", "Use --source binance")
    for expr in (
        "open_interest",
        "open_interest_value",
        "log_oi",
        "oi_chg",
        "oi_coin_chg",
        "ts_mean(oi_turnover, 5)",
    ):
        refused(lambda e=expr: uf.validate_formula(e, "okx"), "the time of the open-interest value is not proven")
        assert uf.validate_formula(expr, "binance")["source"] == "binance"
    assert uf.validate_formula("premium", "binance")["raw_fields"] == ["premium"]
    assert uf.validate_formula("vwap_dev", "okx")["notes"] == ["vwap_close"]
    assert uf.validate_formula("ts_mean(funding, 7)", "okx")["notes"] == ["funding_splice"]
    assert uf.validate_formula("ts_mean(funding, 7)", "binance")["notes"] == []
    assert uf.validate_formula("taker_bs", "binance")["notes"] == ["taker_last_5m"]
    assert set(uf.NOTES) == {"vwap_close", "funding_splice", "taker_last_5m"}
    refused(lambda: uf.validate_formula("close", "bybit"), "The source must be okx or binance")


def test_auto_source():
    assert uf.auto_source("-ts_mean(ex_ret, 20)") == "okx"
    assert uf.auto_source("ts_mean(funding, 7)") == "okx"
    for expr in ("premium", "log_oi", "vwap", "vwap_dev", "taker_imb", "ls_ratio"):
        assert uf.auto_source(expr) == "binance", expr
    assert uf.validate_formula("ls_ratio")["source"] == "binance"


def test_binance_needs_the_frozen_cleaning_rules():
    src = binance_source()
    r = uf.formula_values("ts_mean(ls_ratio, 7)", src)
    assert r["pit"]["status"] == "proven", r["pit"]
    saved = dict(lf.CLEANING_FREEZE)
    lf.CLEANING_FREEZE["sha256"] = "0" * 64
    try:
        for fn in (
            lambda: uf.formula_values("ts_mean(ls_ratio, 7)", src),
            lambda: uf.sunday_value("ts_mean(ls_ratio, 7)", src, r["sundays"][-1].date()),
        ):
            try:
                fn()
            except SystemExit as exc:
                assert "cleaning rules differ" in str(exc), str(exc)
            else:
                raise AssertionError("other cleaning rules were accepted")
    finally:
        lf.CLEANING_FREEZE.clear()
        lf.CLEANING_FREEZE.update(saved)
    # the gaps of Binance read the cells before the cleaning rules
    assert src["presence"] is not src["panels"]


# --------------------------------------------------------------------------- 4. the identity


def test_canonical_text_and_hash():
    a = uf.validate_formula("-ts_mean(ex_ret,20)")
    b = uf.validate_formula("  -( ts_mean( (ex_ret) , 20 ) )  ")
    assert a["canonical"] == b["canonical"] == "-ts_mean(ex_ret, 20)"
    assert a["expression_sha256"] == b["expression_sha256"] == hashlib.sha256(b"-ts_mean(ex_ret, 20)").hexdigest()
    c = uf.validate_formula("ts_mean(ex_ret, 20) * -1")
    assert c["expression_sha256"] != a["expression_sha256"]  # another text is another identity


# --------------------------------------------------------------------------- 5. the values file


def _values_text(rows: list[tuple[str, str, str]]) -> str:
    return "date,symbol,value\n" + "".join(f"{d},{s},{v}\n" for d, s, v in rows)


def test_values_file_csv_and_parquet():
    world()
    days = sundays(3)
    rows = [(d, s, str(i + 0.5 * j)) for i, d in enumerate(days) for j, s in enumerate(SYMS)]
    rows[3] = (rows[3][0], rows[3][1], "")  # a missing value
    p = csv_file(_values_text(rows))
    frame, rep = uf.read_values_file(p, through=date(2026, 9, 30))
    assert list(frame.columns) == SYMS and len(frame) == 3 and str(frame.index.tz) == "UTC"
    assert np.isnan(frame.iloc[0, 3]) and frame.iloc[2, 1] == 2.5
    assert rep["rows"] == 60 and rep["sunday_rows"] == 60 and rep["used_cells"] == 59
    assert rep["first_date"] == days[0] and rep["last_date"] == days[-1]
    assert rep["file_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest() and rep["file_name"] == "values.csv"
    df = pd.DataFrame(
        {
            "value": [None if v == "" else float(v) for _, _, v in rows],
            "symbol": [s for _, s, _ in rows],
            "date": pd.to_datetime([d for d, _, _ in rows]).date,
        }
    )
    q = p.with_suffix(".parquet")
    df.to_parquet(q, index=False)
    f2, rep2 = uf.read_values_file(q, through=date(2026, 9, 30))
    pd.testing.assert_frame_equal(frame, f2, check_exact=True)
    assert rep2["used_cells"] == 59
    # a timestamp column at midnight UTC (naive or with a zero offset) is a day
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(q, index=False)
    pd.testing.assert_frame_equal(frame, uf.read_values_file(q, through=date(2026, 9, 30))[0], check_exact=True)
    df["date"] = df["date"].dt.tz_localize("UTC")
    df.to_parquet(q, index=False)
    pd.testing.assert_frame_equal(frame, uf.read_values_file(q, through=date(2026, 9, 30))[0], check_exact=True)
    # ISO timestamps at 00:00:00 UTC in a CSV file
    iso = [(d + "T00:00:00Z" if i % 2 else d + "T00:00:00.000+00:00", s, v) for i, (d, s, v) in enumerate(rows)]
    pd.testing.assert_frame_equal(
        frame, uf.read_values_file(csv_file(_values_text(iso)), through=date(2026, 9, 30))[0], check_exact=True
    )


def test_values_file_refusals():
    world()
    d = sundays(1)[0]
    ok = [(d, "BTC-USDT", "1.0"), (d, "ETH-USDT", "2.0")]
    through = date(2026, 9, 30)
    cases = [
        ("time,symbol,value\n" + f"{d},BTC-USDT,1\n", 'must be "date,symbol,value"'),
        ("date,symbol,value,extra\n" + f"{d},BTC-USDT,1,2\n", 'must be "date,symbol,value"'),
        (_values_text(ok) + f"{d},BTC-USDT\n", "do not have 3 fields"),
        (_values_text([*ok, (d, "XYZ-USDT", "1")]), "not coins of Factorlab: XYZ-USDT"),
        (_values_text([*ok, (d, "BTC-USDT", "3.0")]), f"more than one time: {d} BTC-USDT"),
        (_values_text([*ok, (d, "SOL-USDT", "inf")]), "finite number"),
        (_values_text([*ok, (d, "SOL-USDT", "-Infinity")]), "finite number"),
        (_values_text([*ok, (d, "SOL-USDT", "true")]), "not a number"),
        (_values_text([*ok, (d, "SOL-USDT", "abc")]), "not a number"),
        (_values_text([(d + "T01:00:00Z", "SOL-USDT", "1")]), "must be 00:00:00"),
        (_values_text([(d + "T00:00:00+02:00", "SOL-USDT", "1")]), "must be UTC"),
        (_values_text([("2026-10-04", "SOL-USDT", "1")]), "a date after 2026-09-30"),
        (_values_text([("2026-02-30", "SOL-USDT", "1")]), "not a valid calendar date"),
        (_values_text([("4 Oct 2026", "SOL-USDT", "1")]), "do not have a date"),
    ]
    for text, word in cases:
        refused(lambda t=text: uf.read_values_file(csv_file(t), through=through), word)
    refused(lambda: uf.read_values_file(csv_file(_values_text(ok), "values.txt"), through=through), ".csv or .parquet")
    refused(lambda: uf.read_values_file(TMP / "absent.csv", through=through), "is not a file")
    saved = uf.MAX_FILE_ROWS
    uf.MAX_FILE_ROWS = 1
    try:
        refused(lambda: uf.read_values_file(csv_file(_values_text(ok)), through=through), "more than 1 rows")
    finally:
        uf.MAX_FILE_ROWS = saved
    # Parquet: types
    q = TMP / "bad.parquet"
    base = {"date": [date(2026, 9, 27)], "symbol": ["BTC-USDT"], "value": [1.0]}
    for change, word in (
        ({"value": [True]}, "number column"),
        ({"value": ["1.0"]}, "number column"),
        ({"date": ["2026-09-27"]}, "a date or a timestamp"),
        ({"date": [pd.Timestamp("2026-09-27 05:00")]}, "00:00:00"),
        ({"date": [pd.Timestamp("2026-09-27", tz="Asia/Tokyo")]}, "must be UTC"),
        ({"value": [float("inf")]}, "finite number"),
    ):
        pd.DataFrame({**base, **change}).to_parquet(q, index=False)
        refused(lambda: uf.read_values_file(q, through=through), word)
    pd.DataFrame({**base, "x": [1]}).to_parquet(q, index=False)
    refused(lambda: uf.read_values_file(q, through=through), "exactly the columns")
    # a file without a data row: a CSV file with only the header (also with empty lines), a
    # Parquet file with 0 rows
    for text in ("date,symbol,value\n", "date,symbol,value\n\n\n"):
        refused(lambda t=text: uf.read_values_file(csv_file(t), through=through), "the file has no data row")
    import pyarrow as pa
    import pyarrow.parquet as pq

    empty = pa.table(
        {"date": pa.array([], pa.date32()), "symbol": pa.array([], pa.string()), "value": pa.array([], pa.float64())}
    )
    pq.write_table(empty, q)
    refused(lambda: uf.read_values_file(q, through=through), "the file has no data row")


def test_values_file_aliases():
    world()
    d = sundays(1)[0]
    p = csv_file(_values_text([(d, "BTCUSDT", "1"), (d, "ETH/USDT:USDT", "2"), (d, "SOL-USDT", "3")]))
    frame, _ = uf.read_values_file(p, through=date(2026, 9, 30))
    assert list(frame.iloc[0, :3]) == [1.0, 2.0, 3.0] and frame.iloc[0, 3:].isna().all()
    # two names of one coin on one date are one pair: a duplicate
    p = csv_file(_values_text([(d, "BTCUSDT", "1"), (d, "BTC-USDT", "2")]))
    refused(lambda: uf.read_values_file(p, through=date(2026, 9, 30)), "more than one time")


def test_values_file_non_sunday_and_no_bar_rows_are_counted():
    world()
    sun = "2026-09-27"
    rows = [(sun, "BTC-USDT", "1"), ("2026-09-28", "BTC-USDT", "2"), ("2026-09-29", "ETH-USDT", "3")]
    # ARB has no bar before its listing in the demo market
    first_arb = data_io.ohlcv("ARB-USDT").index.min()
    early = (first_arb - pd.Timedelta(days=(first_arb.dayofweek + 1) % 7 or 7)).date()
    rows += [(str(early), "ARB-USDT", "4"), (str(early), "BTC-USDT", "5")]
    p = csv_file(_values_text(rows))
    frame, rep = uf.read_values_file(p, through=date(2026, 9, 30))
    assert rep["rows"] == 5 and rep["non_sunday_rows"] == 2 and rep["no_bar_rows"] == 1 and rep["sunday_rows"] == 2
    assert list(frame.index.date) == [early, date(2026, 9, 27)]
    frame2, rep2 = uf.read_values_file(p, through=date(2026, 9, 30), drop_no_bar=False)
    assert (
        rep2["no_bar_rows"] == 0
        and rep2["sunday_rows"] == 3
        and frame2.loc[pd.Timestamp(early, tz="UTC"), "ARB-USDT"] == 4.0
    )
    # with drop_no_bar=False a Sunday after the last bar keeps its rows (a forward log on time)
    late = csv_file(_values_text([("2026-10-04", s, "1") for s in SYMS]))
    f3, r3 = uf.read_values_file(late, through=date(2026, 10, 4), drop_no_bar=False)
    assert r3["used_cells"] == 20 and len(f3) == 1
    f4, r4 = uf.read_values_file(late, through=date(2026, 10, 4))
    assert r4["used_cells"] == 0 and r4["no_bar_rows"] == 20 and len(f4) == 0


# --------------------------------------------------------------------------- 6. library ids


def test_library_formula_lookup():
    db = TMP / "catalog.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE factors (factor_id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, family TEXT NOT NULL, "
        "category TEXT NOT NULL, expression TEXT NOT NULL, raw_inputs TEXT, lookback INTEGER)"
    )
    con.executemany(
        "INSERT INTO factors VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("CF00001", "mom_5", "momentum", "price", "ts_mean(ret, 5)", "close", 5),
            ("CF00002", "lsr_7", "positioning", "derivatives", "ts_mean(ls_ratio, 7)", "long_short_ratio", 7),
        ],
    )
    con.commit()
    con.close()
    got = uf.library_formula("CF00002", catalog=db)
    assert got == {
        "factor_id": "CF00002",
        "name": "lsr_7",
        "family": "positioning",
        "expression": "ts_mean(ls_ratio, 7)",
    }
    assert uf.validate_formula(got["expression"])["source"] == "binance"
    refused(lambda: uf.library_formula("CF09999", catalog=db), "No library definition has the id CF09999")
    refused(lambda: uf.library_formula("CF00001", catalog=TMP / "absent.db"), "catalog is not on disk")


# --------------------------------------------------------------------------- 7. numbers and proxies


def test_factor_numbers_match_an_independent_reference():
    from scipy.stats import spearmanr

    src = world()["okx"]
    r = uf.formula_values("-ts_mean(ex_ret, 20)", src)
    values = pd.DataFrame(r["values"], index=r["sundays"], columns=r["symbols"])
    got = uf.factor_numbers(values, src)
    lab = data_io.label_table()
    el = labels.eligible(src["panels"]["close"].index)
    ics = []
    last = pd.Timestamp(src["complete_through"], tz="UTC") - pd.Timedelta(days=7)
    for s in values.index[values.index <= last]:
        lab_s = lab[lab["date"] == s].set_index("symbol")["fwd_ret_7"].reindex(SYMS)
        v = values.loc[s]
        ok = np.isfinite(v.to_numpy()) & np.isfinite(lab_s.to_numpy()) & el.loc[s].to_numpy()
        if ok.sum() >= lf.MIN_NAMES:
            ics.append(spearmanr(v.to_numpy()[ok], lab_s.to_numpy()[ok])[0])
    ics = np.array(ics)
    assert got["available"] and got["weeks"] == int((values.index <= last).sum())
    assert abs(got["ic_mean"] - ics.mean()) < 1e-9, (got["ic_mean"], ics.mean())
    t = ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics))
    assert abs(got["ic_t"] - t) < 1e-6, (got["ic_t"], t)
    for k in ("ric_mean", "ric_t", "persistence", "coverage"):
        assert got[k] is not None and np.isfinite(got[k]), k
    assert 0 < got["coverage"] <= 1 and -1 <= got["persistence"] <= 1
    # without the label table: no numbers, and the reason
    p = labels.path()
    keep = p.read_bytes()
    p.unlink()
    try:
        none = uf.factor_numbers(values, src)
        assert none == {"available": False, "reason": "No label table. Run labels.py first."}, none
    finally:
        p.write_bytes(keep)


def test_risk_proxies_equal_the_screen_regressors_on_g():
    world()
    days, _ = lf.g_sundays()
    raw = lf.load_raw()
    want = lf.regressors(lf.g_labels(data_io.label_table()), lf.build_panels(raw, through=lf.LABEL_CUTOFF), SYMS, days)
    got = uf.risk_proxies(lf.build_panels(raw), SYMS, days)
    for k in ("tilt", "beta", "size"):
        assert np.array_equal(want[k], got[k], equal_nan=True), k
        assert np.isfinite(got[k]).any(), k


def test_source_dates_and_gaps():
    w = world()
    src = w["okx"]
    assert src["data_through"] == src["complete_through"] == date(2026, 9, 30) and len(src["live"]) == 20
    raw = lf.load_raw()
    # one coin ends 3 days early: it is live and holds complete_through
    raw["ETH-USDT"] = {**raw["ETH-USDT"], "ohlcv": raw["ETH-USDT"]["ohlcv"].loc[:"2026-09-27"]}
    # one coin ends 20 days early: it is not live
    raw["SOL-USDT"] = {**raw["SOL-USDT"], "ohlcv": raw["SOL-USDT"]["ohlcv"].loc[:"2026-09-10"]}
    part = uf.load_source("okx", raw)
    assert part["data_through"] == date(2026, 9, 30) and part["complete_through"] == date(2026, 9, 27)
    assert "SOL-USDT" not in uf.live_coins(part) and uf.complete_through(part) == date(2026, 9, 27)
    g = uf.gaps(part, "2026-09-21", "2026-09-30", ["close"])
    assert ("ETH-USDT", "close", "2026-09-28") in g["pending"] and not any(c == "SOL-USDT" for c, _, _ in g["pending"])
    g2 = uf.gaps(part, "2026-09-08", "2026-09-14", ["close"])
    assert [x for x in g2["final"] if x[0] == "SOL-USDT"] == [
        ("SOL-USDT", "close", f"2026-09-{d}") for d in range(11, 15)
    ]
    assert uf.gap_start(part, "SOL-USDT", "close", "2026-09-13") == "2026-09-11"
    # a coin whose first bar is in the span: each day of the span before it is a gap too
    late = lf.load_raw()
    coin = SYMS[-1]
    late[coin] = {**late[coin], "ohlcv": late[coin]["ohlcv"].loc["2026-09-25":]}
    part2 = uf.load_source("okx", late)
    g3 = uf.gaps(part2, "2026-09-21", "2026-09-27", ["close"])
    assert [x for x in g3["final"] if x[0] == coin] == [(coin, "close", f"2026-09-{d}") for d in (21, 22, 23)], g3
    assert [x for x in g3["pending"] if x[0] == coin] == [(coin, "close", "2026-09-24")], g3
    first = str(part2["presence"]["close"].index.min().date())
    assert {uf.gap_start(part2, coin, "close", f"2026-09-{d}") for d in (21, 22, 23)} == {first}
    # a coin with no value in the whole span has no gap
    g4 = uf.gaps(part2, "2026-09-14", "2026-09-17", ["close"])
    assert g4 == {"pending": [], "final": []}, g4
    assert uf.closed_sundays(date(2026, 9, 6), date(2026, 9, 27)) == [date(2026, 9, d) for d in (6, 13, 20, 27)]
    assert uf.closed_sundays(date(2026, 9, 6), date(2026, 9, 27), date(2026, 9, 19)) == [
        date(2026, 9, 6),
        date(2026, 9, 13),
    ]
    assert uf.read_fields(["ts_corr(delay(log_dvol, 1), funding, 20)"]) == ["close", "funding_rate", "quote_volume"]
    h = uf.field_sha256(uf.cut(src["panels"], "2026-09-27"))
    assert set(h) == set(src["panels"]) and all(len(x) == 64 for x in h.values())


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
    print(f"\n{len(tests) - failed}/{len(tests)} userfactor tests passed")
    raise SystemExit(1 if failed else 0)
