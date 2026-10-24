"""Integrity tests for the AlphaForge bridge: the switch, the zoo format, the forward window,
the capacity, the incumbents, isolation, point-in-time and numbers-only outputs.

Usage:
    python test_alphaforge_bridge.py          # or: pytest test_alphaforge_bridge.py
    TMPDIR=$PWD/.tmp python test_alphaforge_bridge.py   # keep the sandbox repo-local

Most tests run on synthetic parquet files in a temp directory (no network). The sandbox
points the data folder, the raw folders, the trial ledger and the research folder of
config.py and the bridge's output paths at the temp directory, so a cycle writes nowhere
else. One test, when the real zoo file exists, truncates the REAL bars and computes it again.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import alphaforge_bridge as AB
import config as C
import labels
from factorfinder import incumbents as INC

SYMS = ["BTC-USDT", "BBB-USDT", "CCC-USDT", "DDD-USDT", "EEE-USDT", "FFF-USDT"]
N_DAYS = 420
#: tiny miner settings: a cycle takes seconds, and the VALID rule admits every valid formula
TINY = {
    "population": 12,
    "generations": 2,
    "hall_of_fame": 8,
    "min_valid_icir_ann": -100.0,
    "min_valid_t": -100.0,
    "seed": 1,
}
# the words that a find tool must not write (numbers only): verdicts, tiers, stages, gate-pass words
VERDICT_WORDS = re.compile(
    r"\b(useful|useless|unproven|effective|survivors?|promotable|novel|confirmed|retired|probation|"
    r"healthy|honesty|tiers?|stages?)\b"
    r"|\bpass(es|ed)?\s+(the\s+|all\s+|every\s+)?(gates?|screen)\b|\bgates?\s+pass(es|ed)?\b",
    re.IGNORECASE,
)


# --- synthetic world ------------------------------------------------------------


@contextmanager
def sandbox():
    """Point config paths AND the bridge's output paths at a temp dir; restore on exit."""
    names_c = (
        "DATA",
        "RAW_OHLCV",
        "RAW_FUNDING",
        "RAW_FUNDING_FB",
        "RAW_OI",
        "LABELS",
        "TRIALS",
        "RESEARCH",
        "UNIVERSE",
    )
    saved_c = {n: getattr(C, n) for n in names_c}
    saved_ab = {n: getattr(AB, n) for n in ("RESEARCH", "TABLE", "LIBRARY", "REGISTRY", "LATEST", "CYCLES")}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        C.DATA = root
        for n in ("RAW_OHLCV", "RAW_FUNDING", "RAW_FUNDING_FB", "RAW_OI"):
            p = root / "raw" / n.lower()
            p.mkdir(parents=True)
            setattr(C, n, p)
        C.LABELS, C.TRIALS, C.RESEARCH = root / "labels", root / "trials", root / "research"
        C.UNIVERSE = [s.replace("-", "/") + ":USDT" for s in SYMS]
        AB.RESEARCH = root / "research"
        AB.TABLE = AB.RESEARCH / "alphaforge_daily.parquet"
        AB.LIBRARY = AB.RESEARCH / "alphaforge_library.json"
        AB.REGISTRY = AB.RESEARCH / "alphaforge_registry.parquet"
        AB.LATEST = AB.RESEARCH / "alphaforge_latest.txt"
        AB.CYCLES = AB.RESEARCH / "alphaforge_cycles.jsonl"
        try:
            yield root
        finally:
            for n, v in saved_c.items():
                setattr(C, n, v)
            for n, v in saved_ab.items():
                setattr(AB, n, v)


def write_world(seed: int = 0, n_days: int = N_DAYS, rewrite_after=None, short: dict | None = None):
    """OHLCV with independent open/high/low/volume noise + 8-hourly funding, per symbol.

    Every raw column has its own randomness, so every derived terminal (range,
    gap, hl_pos, log_amount, amount_ratio_30, funding) varies day to day. With
    ``rewrite_after`` set, EVERY raw column strictly after that date is redrawn
    from a separate per-symbol generator: the same history, a genuinely
    different future -- which is what the point-in-time tests need. ``short``
    ({symbol: days}) ends the bars of a symbol that many days before the others.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    for i, sym in enumerate(SYMS):
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.03, n_days))), index=idx)
        hi = np.abs(rng.normal(0.0, 0.015, n_days))
        lo = np.abs(rng.normal(0.0, 0.015, n_days))
        gap = rng.normal(0.0, 0.01, n_days)
        vol = rng.uniform(1e6, 5e6, n_days)
        prints = pd.date_range(idx[0], idx[-1] + pd.Timedelta(hours=16), freq="8h")
        fund = rng.normal(1e-4, 2e-4, len(prints))
        if rewrite_after is not None:
            alt = np.random.default_rng(10_000 + i)
            future = close.index > rewrite_after
            n_f = int(future.sum())
            close.loc[future] = float(close.loc[~future].iloc[-1]) * np.exp(np.cumsum(alt.normal(0.0, 0.05, n_f)))
            hi[future] = np.abs(alt.normal(0.0, 0.03, n_f))
            lo[future] = np.abs(alt.normal(0.0, 0.03, n_f))
            gap[future] = alt.normal(0.0, 0.02, n_f)
            vol[future] = alt.uniform(2e5, 9e6, n_f)
            f_future = prints > rewrite_after + pd.Timedelta(hours=23)
            fund[f_future] = alt.normal(5e-4, 4e-4, int(f_future.sum()))
        c = close.to_numpy()
        open_ = np.r_[c[0], c[:-1] * (1 + gap[1:])]
        high = np.maximum.reduce([c * (1 + hi), c, open_])
        low = np.minimum.reduce([c * (1 - lo), c, open_])
        bars = pd.DataFrame({"ts": idx, "open": open_, "high": high, "low": low, "close": c, "volume": vol})
        if short and sym in short:
            bars = bars.iloc[: n_days - int(short[sym])]
        bars.to_parquet(C.RAW_OHLCV / f"{sym}.parquet", index=False)
        pd.DataFrame({"ts": prints, "funding_rate": fund}).to_parquet(C.RAW_FUNDING / f"{sym}.parquet", index=False)
    return idx


#: one expression per mining field (plus the operators the real library likes)
EXPRS = [
    "neg(cs_rank(ts_pct_change(close, 5)))",
    "cs_rank(ts_std(returns, 20))",
    "cs_rank(ts_mean(funding, 10))",
    "div(ts_argmax(range, 10), max(gap, range))",
    "ts_max(ts_kurt(gap, 20), 60)",
    "ts_mean(hl_pos, 5)",
    "ts_zscore(log_amount, 20)",
    "ts_max(amount_ratio_30, 10)",
]
AF_VALUE_COLS = [f"af_{i:02d}" for i in range(1, len(EXPRS) + 1)]


def member(i: int, expr: str, selected_on: str, *, active: bool = True, left_on: str | None = None) -> dict:
    """A format-3 zoo member."""
    return {
        "name": f"af_{i:02d}",
        "expr": expr,
        "selected_on": selected_on,
        "run_id": f"{selected_on}T00:00:00Z",
        "cohort": 1,
        "active": active,
        "left_on": left_on,
        "window": AB.window_of(selected_on, 52),
        "forward": dict(AB.EMPTY_FORWARD),
        "gates": {},
    }


def zoo_of(members: list[dict], last_mining: str = "2024-12-01") -> dict:
    z = AB.empty_zoo()
    z["factors"] = members
    z["cohorts"] = [{"cohort": 1, "selected_on": last_mining, "run_id": f"{last_mining}T00:00:00Z", "added": []}]
    return z


FROZEN = zoo_of([member(i + 1, e, "2024-12-01") for i, e in enumerate(EXPRS)])


def _snapshot(roots) -> dict:
    out = {}
    for root in roots:
        if not root.exists():
            continue
        for f in sorted(root.rglob("*")):
            if f.is_file():
                out[str(f)] = hashlib.md5(f.read_bytes()).hexdigest()
    return out


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return bool(np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True))


def _texts(x) -> list:
    """Every string value and every key of a JSON-like object, at each depth."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [str(k) for k in x] + [t for v in x.values() for t in _texts(v)]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _texts(v)]
    return []


def _word_hits(texts) -> list:
    """The texts with a word of VERDICT_WORDS (an underscore counts as a space)."""
    return [t for t in texts if VERDICT_WORDS.search(str(t).replace("_", " "))]


# --- 1. the switch --------------------------------------------------------------


def test_bridge_refuses_while_switched_off():
    """ALPHAFORGE_ENABLED decides. While it is False, the bridge refuses to run (exit 3)."""
    on = bool(getattr(C, "ALPHAFORGE_ENABLED", False))
    assert AB.enabled() == on
    if not on:
        assert AB.main(["--refresh"]) == 3, "bridge must refuse to run while the switch is off"


# --- 2. conventions -------------------------------------------------------------


def test_label_is_exactly_the_label_table():
    """The store's label equals fwd_ret_7 of labels.compute(), cell by cell, on the label rows."""
    with sandbox():
        write_world()
        store = AB.build_store()
        fwd = store.fwd_return(7, AB.LAG)
        table = labels.compute()
        theirs = AB._wide(table, "fwd_ret_7", store)
        both = fwd.notna() & theirs.notna()
        assert both.to_numpy().sum() > 1000
        assert np.allclose(fwd.to_numpy()[both.to_numpy()], theirs.to_numpy()[both.to_numpy()], rtol=1e-12, atol=0)
        # where the label table has a row, the two NaN patterns agree exactly
        rows = theirs.notna().any(axis=1)
        assert fwd.loc[rows].isna().equals(theirs.loc[rows].isna())
        assert fwd.iloc[-7:].isna().all().all(), "the last 7 labels must be unknown"
        assert store.dates.tz is None and store.dates[0] == pd.Timestamp("2024-01-01")
        assert set(AB.MINING_FIELDS) <= set(store.fields)


def test_hypothesis_key_ignores_sign():
    assert AB.hypothesis_key("neg(cs_rank(x))") == AB.hypothesis_key("cs_rank(x)") == "cs_rank(x)"
    assert AB.hypothesis_key("cs_rank(neg(x))") == "cs_rank(neg(x))"


def test_value_identical_expressions_are_one_hypothesis():
    with sandbox():
        write_world()
        store = AB.build_store()
        ev = AB.Evaluator(store)
        uni = AB._universe(store)
        panels = {
            e: AB.factor_values(e, ev, store, uni)
            for e in (
                "ts_max(range, 10)",
                "cs_zscore(ts_max(range, 10))",
                "max(ts_max(range, 10), ts_max(range, 10))",
                "ts_mean(hl_pos, 5)",
            )
        }
        rows = [{"expr": e, "hypothesis": AB.hypothesis_key(e)} for e in panels]
        sundays = store.dates[store.dates.weekday == 6]
        AB.dedup_hypotheses(rows, panels, sundays)
        keys = [r["hypothesis"] for r in rows]
        assert keys[0] == keys[1] == keys[2] == "ts_max(range, 10)"
        assert keys[3] == "ts_mean(hl_pos, 5)"


def test_table_path_is_research():
    """The table, the zoo file and every other output are in data/research, and the table holds
    one af_NN column for each formula and no composite."""
    assert C.RESEARCH == C.DATA / "research"
    for p in (AB.TABLE, AB.LIBRARY, AB.REGISTRY, AB.LATEST, AB.CYCLES):
        assert p.parent == C.RESEARCH, p
    assert AB.TABLE.name == "alphaforge_daily.parquet"
    src = Path(AB.__file__).read_text(encoding="utf-8")
    assert "composite" not in src.replace("composite date", "")
    with sandbox():
        write_world()
        zoo = zoo_of([member(1, EXPRS[0], "2024-12-01"), member(2, EXPRS[1], "2024-03-03", active=False)])
        t = AB.refresh(zoo, write=True)
        assert AB.TABLE.is_file() and AB.TABLE.parent.name == "research"
        assert list(t.columns) == ["date", "symbol", "af_01", "af_02"], list(t.columns)
        assert str(t["date"].dtype).startswith("datetime64[ms, UTC]")
        assert not t.duplicated(["date", "symbol"]).any()
        day = t[t.date == t.date.max()]
        assert len(day) == len(SYMS), "every symbol with data carries a row on the last date"
        empty = AB.refresh(zoo_of([]), write=False)
        assert len(empty) == 0 and list(empty.columns) == ["date", "symbol"]


# --- 3. the forward window and the capacity -------------------------------------


def test_forward_window_is_52_weeks():
    """Each formula has a fixed window of 52 Sundays from the first Sunday after its selection;
    with 52 complete forward weeks it leaves the active list, its record stays and is final."""
    s = AB.settings()
    assert s["forward_window_weeks"] == 52 == C.ALPHAFORGE["forward_window_weeks"]
    # the settings: the miner's search, the admission rules, the forward window and the capacity; no other key
    want = {
        "forward_window_weeks",
        "generations",
        "hall_of_fame",
        "horizon",
        "max_corr",
        "max_depth",
        "max_factors",
        "min_coverage",
        "min_valid_icir_ann",
        "min_valid_t",
        "mine_every_days",
        "n_jobs",
        "n_quantiles",
        "parsimony",
        "population",
        "seed",
        "train_frac",
        "valid_frac",
        "w_turnover",
    }
    assert set(s) == want, f"settings keys: only in settings {sorted(set(s) - want)}; missing {sorted(want - set(s))}"
    assert set(C.ALPHAFORGE) <= want, (
        f"config.ALPHAFORGE keys that are not settings: {sorted(set(C.ALPHAFORGE) - want)}"
    )
    w = AB.window_of("2024-01-20", 52)  # a Saturday
    assert w == {"weeks": 52, "first_sunday": "2024-01-21", "last_sunday": "2025-01-12"}
    assert AB.window_of("2024-01-21", 52)["first_sunday"] == "2024-01-28", "never the selection day itself"
    with sandbox():
        write_world()
        store = AB.build_store()
        old, new = member(1, EXPRS[1], "2024-02-03"), member(2, EXPRS[2], "2024-09-01")
        zoo = zoo_of([old, new])
        left = AB.update_forward(zoo, store, s, today=pd.Timestamp("2025-02-20"))
        assert left == ["af_01"] and old["active"] is False and old["left_on"] == "2025-02-20"
        fw = old["forward"]
        assert fw["weeks"] == 52 and len(fw["series"]) == 52, fw["weeks"]
        assert fw["first"] == "2024-02-04" and fw["last"] == "2025-01-26"
        assert all("2024-02-04" <= d <= "2025-01-26" for d, _ in fw["series"])
        assert new["active"] is True and 0 < new["forward"]["weeks"] < 52 and new["left_on"] is None
        # the record of a formula that left is final: a later cycle does not compute it again
        old["forward"]["ic"] = 123.0
        assert AB.update_forward(zoo, store, s, today=pd.Timestamp("2025-02-27")) == []
        assert old["forward"]["ic"] == 123.0 and old["left_on"] == "2025-02-20"


def test_window_ends_when_no_coin_has_a_bar_on_a_window_sunday():
    """The window Sundays come from the calendar: a Sunday on which no coin has a bar is a window
    week with no IC. The formula leaves the active list after 52 weeks, and its place is free."""
    s = AB.settings()
    gap = pd.Timestamp("2024-06-02", tz="UTC")
    assert gap.dayofweek == 6
    with sandbox():
        write_world()
        for sym in SYMS:  # one day without a bar for every coin
            path = C.RAW_OHLCV / f"{sym}.parquet"
            bars = pd.read_parquet(path)
            bars[bars["ts"] != gap].to_parquet(path, index=False)
        store = AB.build_store()
        assert gap.tz_convert(None) not in store.dates, "the fixture needs a date without a row"
        m = member(1, EXPRS[1], "2024-02-03")
        zoo = zoo_of([m])
        assert m["window"]["first_sunday"] <= "2024-06-02" <= m["window"]["last_sunday"]
        left = AB.update_forward(zoo, store, s, today=pd.Timestamp("2025-02-20"))
        fw = m["forward"]
        assert left == ["af_01"] and m["active"] is False and m["left_on"] == "2025-02-20"
        assert fw["weeks"] == 52 and len(fw["series"]) == 51, (fw["weeks"], len(fw["series"]))
        assert "2024-06-02" not in {d for d, _ in fw["series"]}
        assert fw["first"] == "2024-02-04" and fw["last"] == "2025-01-26"


def test_forward_week_needs_complete_labels():
    """A Sunday counts only when S + 7 <= complete_through (the oldest last bar over the live coins):
    a Sunday whose label only 5 coins have is not a forward week; a coin with no bar for more than
    7 days does not hold the date back."""
    with sandbox():
        idx = write_world(short={"FFF-USDT": 3})
        store = AB.build_store()
        newest = idx[-1].tz_convert(None)
        assert AB.complete_through(store) == newest - pd.Timedelta(days=3)
        fwd = store.fwd_return(7, AB.LAG)
        sundays = store.dates[store.dates.weekday == 6]
        late = [d for d in sundays if newest - pd.Timedelta(days=3) < d + pd.Timedelta(days=7) <= newest]
        assert late, "the fixture needs a Sunday whose label only 5 coins have"
        assert all(int(fwd.loc[d].notna().sum()) == 5 for d in late)
        ev, uni = AB.Evaluator(store), AB._universe(store)
        vals = AB.factor_values(EXPRS[1], ev, store, uni)
        assert all(np.isfinite(AB.sunday_ic(vals, fwd, pd.DatetimeIndex([d]))).all() for d in late), "IC on 5 coins"
        rec = AB.forward_record(vals, fwd, "2024-09-01", AB.complete_through(store), 52)
        assert not {str(d.date()) for d in late} & {d for d, _ in rec["series"]}
        assert rec["last"] < str(late[0].date())
        # a coin with no bar for 10 days is not live and does not hold complete_through back
        write_world(short={"FFF-USDT": 10})
        store2 = AB.build_store()
        assert AB.complete_through(store2) == newest


def test_capacity_is_six_active():
    """At most 6 formulas are active: with 6 active formulas a due (or forced) cycle mines nothing;
    with 5 it mines with exactly one free place."""
    s = AB.settings()
    assert s["max_factors"] == 6 == C.ALPHAFORGE["max_factors"]
    with sandbox():
        write_world()
        six = [member(i + 1, e, "2024-12-01") for i, e in enumerate(EXPRS[:6])]
        AB.save_zoo(zoo_of(six, last_mining="2024-12-01"))
        calls = []
        with mock.patch.object(AB, "mine", side_effect=lambda *a, **k: calls.append(k) or None):
            out = AB.run_cycle(dict(TINY), today="2025-01-05", force_mine=True, log_fn=lambda *_: None)
        assert not calls and not out["mined"] and out["line"]["room"] == 0 and out["line"]["active"] == 6
        assert not AB.REGISTRY.exists(), "a cycle with no free place logs no trial"
        five = [member(i + 1, e, "2024-12-01") for i, e in enumerate(EXPRS[:5])]
        AB.LIBRARY.unlink()
        AB.save_zoo(zoo_of(five, last_mining="2024-12-01"))
        out = AB.run_cycle(dict(TINY), today="2025-01-05", log_fn=lambda *_: None)
        assert out["mined"] and len(out["added"]) <= 1
        # the cycle line gives the counts after the cycle: active + free places = 6
        assert out["line"]["active"] == 5 + len(out["added"]) and out["line"]["room"] == 1 - len(out["added"])
        assert out["line"]["active"] + out["line"]["room"] == 6 and len(AB.active_members(out["zoo"])) <= 6


def test_mining_fills_free_places_only():
    """A cycle mines only with a free place and when the last mining run is at least mine_every_days
    old; it adds at most room = max_factors - active formulas. A formula that left the active list
    frees its place and stays in the zoo."""
    with sandbox():
        write_world()
        members = [member(i + 1, e, "2024-12-01") for i, e in enumerate(EXPRS[:4])]
        members.append(member(5, EXPRS[4], "2024-03-03", active=False, left_on="2025-03-03"))
        AB.save_zoo(zoo_of(members, last_mining="2024-12-01"))
        seen = []
        real = AB.mine

        def spy(overrides=None, **k):
            seen.append(dict(overrides or {}))
            return real(overrides, **k)

        # 3 days after the last mining run: not due
        with mock.patch.object(AB, "mine", side_effect=spy):
            out = AB.run_cycle(dict(TINY), today="2024-12-04", log_fn=lambda *_: None)
        assert not seen and not out["mined"] and out["line"]["room"] == 2
        with mock.patch.object(AB, "mine", side_effect=spy):
            out = AB.run_cycle(dict(TINY), today="2024-12-08", log_fn=lambda *_: None)
        assert out["mined"] and seen[-1]["max_factors"] == 2, seen
        zoo = out["zoo"]
        assert 1 <= len(out["added"]) <= 2, out["added"]
        assert len(AB.active_members(zoo)) == 4 + len(out["added"]) <= 6
        assert [m["name"] for m in zoo["factors"][:5]] == [f"af_{i:02d}" for i in range(1, 6)]
        assert zoo["factors"][4]["active"] is False, "the formula that left stays in the zoo and stays out"
        assert [m["name"] for m in zoo["factors"][5:]] == out["added"] and out["added"][0] == "af_06"
        assert all(m["window"]["first_sunday"] == "2024-12-15" for m in zoo["factors"][5:])
        line = json.loads(AB.CYCLES.read_text().splitlines()[-1])
        assert set(line) == {"date", "data_through", "mined", "added", "left", "active", "room", "n_evaluated", "seed"}
        assert line["added"] == out["added"] and line["active"] == len(AB.active_members(zoo)) == 4 + len(out["added"])
        assert line["room"] == 2 - len(out["added"]) and line["active"] + line["room"] == 6, "counts after the cycle"
        places = "1 free place" if line["room"] == 1 else f"{line['room']} free places"
        after = f"after the cycle: {line['active']} active, {places}"
        assert after in out["report"], out["report"]
        assert "free places before the run 2" in out["report"], out["report"]


def test_a_count_of_one_has_the_singular_noun():
    """The report gives the number and the noun in the form that agrees with it."""
    one = zoo_of([member(1, EXPRS[0], "2024-12-01")])
    cycle = {"today": "2025-01-05", "mined": False, "due": False, "active": 1, "room": 1}
    text = AB.render(one, None, cycle)
    for want in ("1 active formula (at most", "1 formula in the zoo", "1 mining run ===", "1 free place"):
        assert want in text, (want, text)
    assert "1 free places" not in text and "1 mining runs" not in text, text
    two = zoo_of([member(i + 1, e, "2024-12-01") for i, e in enumerate(EXPRS[:2])])
    text = AB.render(two, None, {**cycle, "active": 2, "room": 4})
    assert "2 active formulas (at most" in text and "2 formulas in the zoo" in text and "4 free places" in text, text


def test_zoo_of_another_format_is_refused():
    """The bridge reads only a zoo file of format 3. A file of another format, or with no format,
    stops load_zoo, load_library and a cycle with ZooFormatError. The command line gives exit 1
    and one line. The line names the file and the formats. Nothing is written."""
    assert AB.ZOO_FORMAT == INC.ZOO_FORMAT == 3 and AB.empty_zoo()["format"] == 3
    with sandbox():
        write_world()
        AB.RESEARCH.mkdir(parents=True)
        good = zoo_of([member(1, EXPRS[1], "2024-12-01")])
        AB.LIBRARY.write_text(json.dumps(good))
        assert AB.load_zoo() == json.loads(json.dumps(good)), "a format-3 file reads as written"
        for fmt in (2, 4, "3", None):
            bad = {k: v for k, v in good.items() if k != "format"}
            if fmt is not None:
                bad["format"] = fmt
            AB.LIBRARY.write_text(json.dumps(bad))
            before = _snapshot([AB.RESEARCH])
            calls = (
                AB.load_zoo,
                AB.load_library,
                lambda: AB.run_cycle(dict(TINY), today="2025-01-05", log_fn=lambda *_: None),
            )
            for call in calls:
                try:
                    call()
                except INC.ZooFormatError as exc:
                    assert f"format {fmt!r}" in str(exc) and "reads only format 3" in str(exc), str(exc)
                else:
                    raise AssertionError(f"a zoo file with the format {fmt!r} was read")
            for argv in (["--cycle", "--force", "--today", "2025-01-05"], ["--refresh", "--force"]):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    code = AB.main(argv)
                text = err.getvalue()
                assert code == 1 and text.startswith("alphaforge: the zoo file "), (argv, code, text)
                assert AB.LIBRARY.name in text and f"format {fmt!r}" in text, text
            assert _snapshot([AB.RESEARCH]) == before, f"a refused zoo file (format {fmt!r}) changed data/research"


# --- 4. the incumbents -------------------------------------------------------------


def test_incumbents_come_from_the_provider():
    """The miner takes its incumbents from factorfinder/incumbents.py: your registered tests, the
    active zoo formulas (computed on its own store) and the three risk proxies. A formula that left
    the active list is not an incumbent."""
    with sandbox():
        idx = write_world()
        rng = np.random.default_rng(5)
        user = INC.Incumbent("user:FT-0001", "user", pd.DataFrame(rng.normal(size=(len(idx), len(SYMS))), idx, SYMS))
        calls = []
        real = INC.incumbents

        def spy(**k):
            calls.append(k)
            return real(**k)

        zoo = zoo_of([member(1, EXPRS[1], "2024-10-06"), member(2, EXPRS[2], "2024-03-03", active=False)])
        with (
            mock.patch.object(INC, "user_members", return_value=[user]),
            mock.patch.object(INC, "incumbents", side_effect=spy),
        ):
            res = AB.mine({**TINY, "max_factors": 2}, zoo=zoo, today=pd.Timestamp("2025-01-05"), log_fn=lambda *_: None)
        assert len(calls) == 1 and tuple(calls[0]["sources"]) == ("user", "zoo", "proxy")
        assert sorted(calls[0]["zoo_values"]) == ["af_01"], "only the active formulas are incumbents"
        names = res["cohort"]["incumbents"]
        assert names == sorted(
            ["user:FT-0001", "zoo:af_01", "proxy:pit_symbol_tilt", "proxy:btc_beta_90d", "proxy:size_proxy"]
        ), names
        assert res["cohort"]["incumbent_counts"] == {"user": 1, "zoo": 1, "proxy": 3}
        rows = res["results"]
        assert len(rows) and (rows["n_incumbents"] <= 5).all() and "max_incumbent_corr" in rows
        for m in res["new_factors"]:
            assert set(m["gates"]) == {
                "valid_icir",
                "valid_t",
                "persistence",
                "coverage",
                "max_incumbent_corr",
                "max_library_corr",
            }
            assert m["library"] is None, "the sandbox has no library values"


# --- 5. point-in-time -----------------------------------------------------------


def test_every_terminal_and_the_table_are_point_in_time():
    """Rewrite the future of every raw column; every earlier row must be bit-identical."""
    with sandbox():
        idx = write_world()
        cut = idx[-60]
        cut_naive = pd.Timestamp(cut).tz_convert(None)
        store_a = AB.build_store()
        a = AB.refresh(FROZEN, write=False)
        write_world(rewrite_after=cut)
        store_b = AB.build_store()
        b_full = AB.refresh(FROZEN, write=False)
        # 1) every terminal the miner may use, plus the label inputs
        for f in AB.MINING_FIELDS + ("close", "tradable", "amount"):
            pa, pb = store_a.get(f).loc[:cut_naive], store_b.get(f).loc[:cut_naive]
            assert _same(pa, pb), f"terminal {f}: future data leaked into past rows"
            if f == "tradable":
                continue  # constant 1.0 in the synthetic world by construction
            fa, fb = (
                store_a.get(f).loc[cut_naive + pd.Timedelta(days=1) :],
                store_b.get(f).loc[cut_naive + pd.Timedelta(days=1) :],
            )
            assert not _same(fa, fb), f"terminal {f}: the rewritten future did not change (test has no teeth)"
        # 2) the table
        b = b_full[b_full.date <= cut]
        m = a[a.date <= cut].merge(b, on=["date", "symbol"], suffixes=("_a", "_b"))
        assert len(m) == len(a[a.date <= cut]) > 0
        for c in AF_VALUE_COLS:
            d = (m[f"{c}_a"] - m[f"{c}_b"]).abs().max()
            assert not (d > 1e-12), f"{c}: future data leaked into past rows ({d:.2e})"
        # 3) and every value column moved after the cut, so each expression was exercised
        after = a[a.date > cut].merge(b_full[b_full.date > cut], on=["date", "symbol"], suffixes=("_a", "_b"))
        for c in AF_VALUE_COLS:
            assert (after[f"{c}_a"] - after[f"{c}_b"]).abs().max() > 1e-9, f"{c}: future unchanged, no teeth"


def test_real_library_is_point_in_time_on_real_bars():
    """Read-only: truncate the REAL bars and compute the real zoo again; earlier rows must match."""
    # "(skipped: ...)" is the line that engine/run_tests.py counts as a skipped test
    if not (C.RAW_OHLCV / "BTC-USDT.parquet").exists():
        print("        (skipped: no raw bars in the data folder)")
        return
    if not AB.LIBRARY.exists():
        print("        (skipped: no AlphaForge formulas in data/research; AlphaForge is off by default)")
        return
    lib = AB.load_zoo()
    if not lib.get("factors"):
        print("        (skipped: the AlphaForge file in data/research has no formula)")
        return
    import data_io

    px = data_io.close_matrix()
    cut = px.index[-45]
    full = AB.refresh(lib, store=AB.build_store(px), write=False)
    trunc = AB.refresh(lib, store=AB.build_store(px[px.index <= cut]), write=False)
    m = full[full.date <= cut].merge(trunc, on=["date", "symbol"], suffixes=("_a", "_b"))
    assert len(m) == len(trunc) == len(full[full.date <= cut]) > 0
    for c in [c for c in trunc.columns if c.startswith("af_")]:
        d = (m[f"{c}_a"] - m[f"{c}_b"]).abs().max()
        assert not (d > 1e-12), f"{c}: real library is not point-in-time ({d:.2e})"
        assert m[f"{c}_a"].isna().equals(m[f"{c}_b"].isna()), f"{c}: NaN pattern changed"


# --- 6. cycles end to end, in the sandbox, writing nowhere else -------------------


def test_cycles_add_formulas_keep_clocks_and_write_only_research():
    with sandbox() as root:
        write_world()
        tiny = {**TINY, "max_factors": 2}
        # cycle 1 (no zoo -> mining is due without force)
        c1 = AB.run_cycle(tiny, today="2025-01-05", log_fn=lambda *_: None)
        zoo1 = c1["zoo"]
        assert c1["mined"] and len(c1["results"]) > 0
        for p in (AB.LIBRARY, AB.REGISTRY, AB.LATEST, AB.TABLE, AB.CYCLES):
            assert p.exists(), p
        assert zoo1["format"] == AB.ZOO_FORMAT and len(zoo1["cohorts"]) == 1
        assert all(m["active"] and m["selected_on"] == "2025-01-05" for m in zoo1["factors"])
        assert (len(c1["table"]) > 0) == (len(zoo1["factors"]) > 0)
        reg1 = AB.load_registry()
        assert len(reg1) == len(c1["results"]) and reg1["first_seen"].all()
        assert (reg1["n_evaluated_in_run"] == zoo1["cohorts"][0]["n_evaluations"]).all()
        assert list(reg1.columns) == AB.REGISTRY_COLUMNS
        a1 = c1["assessed"]
        assert (a1["q_bh"].between(0, 1)).all() and (a1["p_bonferroni"] <= 1).all()
        assert a1.attrs["bonferroni_divisor"] >= zoo1["cohorts"][0]["n_evaluations"]
        n1 = len(zoo1["factors"])
        file1 = AB.LIBRARY.read_bytes()
        assert not _backups(), "the first cycle replaces no zoo file"
        # the backup of a replaced zoo file: alphaforge_library_<its run_id without ':'>.json
        rid1 = zoo1["run_id"]
        assert isinstance(rid1, str) and rid1.startswith("2025-01-05T") and ":" in rid1, rid1
        backup1 = f"alphaforge_library_{rid1.replace(':', '')}.json"
        # cycle 2, two days later: not due (weekly) -> nothing mined, forward updated
        c2 = AB.run_cycle(tiny, today="2025-01-07", log_fn=lambda *_: None)
        assert c2["zoo"]["run_id"] == rid1, "a cycle without mining keeps the run_id"
        assert _backups() == [backup1], (_backups(), backup1)
        assert (AB.RESEARCH / backup1).read_bytes() == file1, "the backup is the replaced file"
        assert not c2["mined"] and len(AB.load_registry()) == len(reg1)
        assert [m["selected_on"] for m in c2["zoo"]["factors"]] == ["2025-01-05"] * n1, "clocks must not move"
        # cycle 3, a week later with more places: adds formulas, old ones keep their clocks
        tiny3 = {**TINY, "max_factors": 4, "seed": 2}
        c3 = AB.run_cycle(tiny3, today="2025-01-12", log_fn=lambda *_: None)
        zoo3 = c3["zoo"]
        assert c3["mined"] and len(zoo3["cohorts"]) == 2
        assert [m["selected_on"] for m in zoo3["factors"][:n1]] == ["2025-01-05"] * n1
        assert all(m["selected_on"] == "2025-01-12" and m["cohort"] == 2 for m in zoo3["factors"][n1:])
        assert len(set(m["name"] for m in zoo3["factors"])) == len(zoo3["factors"]), "names never reused"
        if c3["added"]:
            assert set(c3["added"]) <= set(c3["table"].columns)
        reg3 = AB.load_registry()
        assert len(reg3) == len(reg1) + len(c3["results"])
        assert reg3.iloc[: len(reg1)].drop(columns=["first_seen"]).equals(reg1.drop(columns=["first_seen"])), (
            "the registry must be append-only"
        )
        # cycle 3 replaced a file with the same run_id: no second backup, the first one is unchanged
        assert zoo3["run_id"] != rid1, (zoo3["run_id"], rid1)
        assert _backups() == [backup1], (_backups(), backup1)
        assert (AB.RESEARCH / backup1).read_bytes() == file1, "a backup is written one time for each run_id"
        lines = [json.loads(line) for line in AB.CYCLES.read_text().splitlines()]
        assert [line["mined"] for line in lines] == [True, False, True] and lines[-1]["date"] == "2025-01-12"
        for m in zoo3["factors"]:
            fw = m["forward"]
            assert all(pd.Timestamp(d) > pd.Timestamp(m["selected_on"]) for d, _ in fw["series"])
        written = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        written = {w for w in written if not w.startswith("raw/")}
        assert written and all(w.startswith("research/") for w in written), written


def _backups() -> list[str]:
    """The names of the zoo backups in data/research/, sorted."""
    return sorted(p.name for p in AB.RESEARCH.glob("alphaforge_library_*.json"))


def test_backup_of_a_zoo_without_run_id_is_prev():
    """save_zoo copies the file that it replaces to alphaforge_library_<run_id without ':'>.json, one time
    for each run_id. A zoo file with no run_id, or a file that is not JSON, gives alphaforge_library_prev.json."""
    with sandbox():
        AB.RESEARCH.mkdir(parents=True)
        AB.save_zoo(zoo_of([]))
        first = AB.LIBRARY.read_bytes()
        assert json.loads(first)["run_id"] is None and not _backups()
        second = {**zoo_of([member(1, EXPRS[1], "2024-12-01")]), "run_id": "2024-12-01T10:20:30Z"}
        AB.save_zoo(second)
        assert _backups() == ["alphaforge_library_prev.json"], _backups()
        assert (AB.RESEARCH / "alphaforge_library_prev.json").read_bytes() == first
        AB.save_zoo({**second, "data_through": "2024-12-02"})
        AB.save_zoo({**second, "data_through": "2024-12-03"})
        want = ["alphaforge_library_2024-12-01T102030Z.json", "alphaforge_library_prev.json"]
        assert _backups() == want, (_backups(), want)
        backup = json.loads((AB.RESEARCH / want[0]).read_text())
        assert backup["data_through"] is None, "the first replaced file of a run_id is kept"
        assert (AB.RESEARCH / want[1]).read_bytes() == first, "a later save keeps the earlier backup"
        AB.LIBRARY.write_text("not json")
        (AB.RESEARCH / want[1]).unlink()
        AB.save_zoo(second)
        assert (AB.RESEARCH / want[1]).read_text() == "not json", "a file that is not JSON gives the prev backup"


def test_counts_keep_a_run_without_a_valid_member():
    """A mining run whose hall of fame has no valid member writes no registry row. The zoo keeps its
    cumulative counts, and the expressions of that run count in the next Bonferroni divisor. The
    library number of an added formula is the largest absolute correlation."""
    lib = {"max_abs": 0.31, "corr": -0.31, "nearest": "CF00001", "n_members": 9}
    with sandbox():
        write_world()
        tiny = {**TINY, "max_factors": 2}
        with mock.patch.object(INC, "library_corr", return_value=lib):
            c1 = AB.run_cycle(tiny, today="2025-01-05", log_fn=lambda *_: None)
        assert c1["added"], "the fixture needs an added formula"
        for m in c1["zoo"]["factors"]:
            assert m["gates"]["max_library_corr"] == 0.31 and m["library"]["nearest"] == "CF00001"
        reg1 = AB.load_registry()
        sel = reg1[reg1["selected"].fillna(False).astype(bool)]
        assert (sel["max_library_corr"] == 0.31).all() and (sel["closest_library"] == "CF00001").all()
        n1 = int(c1["zoo"]["cohorts"][0]["n_evaluations"])
        want1 = {"n_distinct_hypotheses": int(reg1["hypothesis"].nunique()), "n_expressions_searched": n1}
        assert c1["zoo"]["registry"] == want1, c1["zoo"]["registry"]
        real = AB.mine

        def no_valid_member(overrides=None, **k):
            r = real(overrides, **k)
            r["results"], r["new_factors"] = r["results"].iloc[0:0], []
            r["cohort"]["added"] = []
            return r

        with mock.patch.object(AB, "mine", side_effect=no_valid_member):
            c2 = AB.run_cycle({**TINY, "max_factors": 6, "seed": 3}, today="2025-01-12", log_fn=lambda *_: None)
        assert c2["mined"] and not c2["added"] and len(AB.load_registry()) == len(reg1), "no registry row"
        zoo = json.loads(AB.LIBRARY.read_text())
        n2 = int(zoo["cohorts"][1]["n_evaluations"])
        assert n2 > 0 and zoo["registry"] == {**want1, "n_expressions_searched": n1 + n2}, zoo["registry"]
        assert AB.searched_so_far(AB.load_registry(), zoo["cohorts"]) == n1 + n2
        assert AB.searched_so_far(AB.load_registry()) == n1, "the registry alone does not see the run"
        # the next run with rows: its divisor counts the expressions of every run
        c3 = AB.run_cycle({**TINY, "max_factors": 6, "seed": 4}, today="2025-01-19", log_fn=lambda *_: None)
        if len(c3["results"]):
            n3 = int(c3["zoo"]["cohorts"][2]["n_evaluations"])
            assert c3["assessed"].attrs["n_searched"] == n1 + n2 + n3
            assert c3["zoo"]["registry"]["n_expressions_searched"] == n1 + n2 + n3


def test_outputs_are_numbers_only():
    """The zoo file, the cycle log, the registry, the report and the module hold no verdict, tier,
    stage or status word. The zoo of the test has a formula that left the active list."""
    with sandbox():
        write_world()
        AB.RESEARCH.mkdir(parents=True)
        zoo = zoo_of(
            [
                member(1, EXPRS[1], "2024-06-02", active=False, left_on="2024-12-20"),
                member(2, EXPRS[2], "2024-12-01"),
            ]
        )
        AB.LIBRARY.write_text(json.dumps(zoo))
        out = AB.run_cycle({**TINY, "max_factors": 3}, today="2025-01-05", log_fn=lambda *_: None)
        assert out["mined"]
        zoo = json.loads(AB.LIBRARY.read_text())
        texts = _texts(zoo)
        texts += [t for line in AB.CYCLES.read_text().splitlines() for t in _texts(json.loads(line))]
        texts += AB.LATEST.read_text().splitlines()
        reg = AB.load_registry()
        texts += list(reg.columns) + [str(x) for c in reg.columns if reg[c].dtype == object for x in reg[c].dropna()]
        texts += out["report"].splitlines()
        hits = _word_hits(texts)
        assert not hits, hits[:10]
        # the module itself: no line of its code, its texts or its comments has such a word
        src = Path(AB.__file__).read_text(encoding="utf-8").splitlines()
        hits = [f"line {i}: {line.strip()}" for i, line in enumerate(src, 1) if _word_hits([line])]
        assert not hits, hits[:10]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} alphaforge bridge tests passed")
    raise SystemExit(1 if failed else 0)
