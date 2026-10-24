"""Tests of the trial ledger (trials.py) and of its command line (judge.py).

Usage:
    python -B test_trials.py

The suite reads no data folder of the repository. At the start it sets FACTORLAB_DATA to a new
folder under the git-ignored engine/.tmp/ and removes that folder at the end. It writes the
synthetic demo market once into a master folder. Each test copies the bars up to a date into
its data folder (the market as it was on that date), and adds the later days back to show an
update. The tests replace ledger.CLOCK in this process, so each command reads a known time; the
bars end on the day before that time, as a download writes them.

The properties:
    1. check writes nothing; register appends one line and writes the result files; a factor
       that the historical test calls "useless" is registered, and a forward decision replaces
       its historical word
    2. the refusals of register: no point-in-time proof, a duplicate (by content: another name or
       version of the book, the mirror formula with sign -1, the same values as CSV or Parquet,
       in another row order), code that the ledger does not hold, a history without a complete
       week; the related tests
    3. one clock read for each command; the first forward Sunday; a switch of config.py is not a
       code change
    4. log-week: the cut values and the PIT check; a pending gap or a PIT mismatch holds only its
       own test; a week is complete when its data is complete or the gap is final (an alarm)
    5. the forward replay: the frozen weeks equal two independent runs of the book and the
       control; foresight values give "useful" at week 26, negated foresight "useless" at week
       26; random values follow rules.forward_test; no week after a decision
    6. values files: the log-time rule, a log before the bar, a week without a used value holds
       the control portfolio, a changed value is refused with an alarm, a future date is
       refused, a week counts without a log
    7. the hash chain, the write boundary, the alarms (each recorded once, exit 3), code drift,
       --accept-release, the withdrawal rules, the expected number of false "useful" verdicts,
       verify, a folder of the data folder that the ledger does not use
    8. the command line: help texts, refusals, an empty status; the point-in-time line of check;
       absolute file paths in the printed register command; the imports
    9. an end-to-end run of 30 forward weeks writes only under data/trials
   10. a day without any bar, a source that ends before the OKX data, a Sunday logged after a
       later Sunday, a BTC hold without its first close (unknown, not 0.0), the excess
       volatility, and verify --write-results gives the result files of the registration
"""

from __future__ import annotations

import atexit
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
(ENGINE / ".tmp").mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="test_trials_", dir=ENGINE / ".tmp"))
atexit.register(shutil.rmtree, TMP, True)
os.environ["FACTORLAB_DATA"] = str(TMP / "data")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import book  # noqa: E402
import config as C  # noqa: E402
import demo_data  # noqa: E402
import labels  # noqa: E402
import ledger as LG  # noqa: E402
import libfactors as lf  # noqa: E402
import rules  # noqa: E402
import trials as T  # noqa: E402
import userfactor as uf  # noqa: E402

DATA = Path(C.DATA)
MASTER = TMP / "master"
SYMS = [C.sym_to_file(s) for s in C.UNIVERSE]
CUT = "2026-02-20"  # the bars of a registration end on this Friday
F = date(2026, 2, 22)  # the first forward Sunday of a registration on Saturday 2026-02-21
LAST = "2026-09-30"  # the last bar of the demo market
FORMULA = "-ts_mean(ex_ret, 20)"  # the historical test calls it useless on the demo market
PASSING = "ts_mean(clv, 7)"  # the historical test gives a pass on the demo market
QUIET = lambda *a, **k: None  # noqa: E731
_STATE: dict = {}


# --------------------------------------------------------------------------- the fixture


def master() -> dict:
    """The demo market, written once into MASTER, and its frames in memory."""
    if not _STATE:
        m = demo_data.market(demo_data.SEED)
        demo_data.write_raw(MASTER, m)
        frames = {}
        for p in sorted((MASTER / "raw").rglob("*.parquet")):
            df = pd.read_parquet(p)
            frames[p.relative_to(MASTER)] = (df, pd.to_datetime(df["ts"], utc=True).dt.floor("D"))
        close = pd.DataFrame({s: frames[Path("raw/ohlcv") / f"{s}.parquet"][0].set_index("ts")["close"] for s in SYMS})
        close.index = pd.DatetimeIndex(close.index).tz_convert("UTC").as_unit("ns")
        _STATE.update(market=m, frames=frames, close=close)
    return _STATE


def cut_to(day, *, ends: dict | None = None, drop: dict | None = None, label_table: bool = True) -> None:
    """The raw bars up to `day` in the data folder. ends {coin: last day} ends a coin earlier;
    drop {(folder, coin): [days]} removes the rows of these days (for example a funding day)."""
    st = master()
    lim = pd.Timestamp(str(day), tz="UTC")
    for rel, (df, days) in st["frames"].items():
        coin = rel.stem
        last = lim
        if ends and coin in ends:
            last = min(last, pd.Timestamp(str(ends[coin]), tz="UTC"))
        keep = days <= last
        for (folder, c), gone in (drop or {}).items():
            if c == coin and rel.parent.name == folder:
                keep &= ~days.isin([pd.Timestamp(str(g), tz="UTC") for g in gone])
        out = DATA / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        df[keep.to_numpy()].to_parquet(out, index=False)
    if label_table:
        labels.build()


def fresh(day=CUT, **kw) -> None:
    """A data folder without a trial ledger, with the bars up to `day`."""
    for name in ("trials", "notes"):
        shutil.rmtree(DATA / name, ignore_errors=True)
    cut_to(day, **kw)


def at(day, hh: int = 10, mm: int = 0, ss: int = 0) -> datetime:
    d = LG.as_day(day)
    return datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=UTC)


class Clock:
    """ledger.CLOCK for a test: the given times in turn, then the last one again."""

    def __init__(self, *times: datetime) -> None:
        self.times = list(times)
        self.calls = 0

    def __call__(self) -> datetime:
        t = self.times[min(self.calls, len(self.times) - 1)]
        self.calls += 1
        return t


def clock(*times: datetime) -> Clock:
    c = Clock(*times)
    LG.CLOCK = c
    return c


def spec_file(text: str, name: str = "spec.toml") -> Path:
    d = Path(tempfile.mkdtemp(prefix="spec_", dir=TMP))
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p


def spec_of(extra: str = "", name: str = "long-short", version: int = 1) -> book.BookSpec:
    return book.load_spec(spec_file(f'format = 1\nname = "{name}"\nversion = {version}\n{extra}'))


def values_csv(rows: dict, name: str = "values.csv", order: list | None = None) -> Path:
    """A CSV values file: rows {sunday: 20 values} (NaN is an empty value)."""
    lines = ["date,symbol,value"]
    for s in sorted(rows):
        for sym, v in zip(SYMS, rows[s]):
            lines.append(f"{s},{sym},{'' if not np.isfinite(v) else repr(float(v))}")
    if order is not None:
        lines = [lines[0]] + [lines[1 + i] for i in order]
    d = Path(tempfile.mkdtemp(prefix="vf_", dir=TMP))
    p = d / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def foresight(s, sign: float = 1.0) -> np.ndarray:
    """The return of each coin over the 7 days after the Sunday s (a value that no one knows on s)."""
    close = master()["close"]
    t = pd.Timestamp(str(s), tz="UTC")
    return sign * (close.loc[t + pd.Timedelta(days=7)].to_numpy() / close.loc[t].to_numpy() - 1.0)


def random_values(s, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + int(pd.Timestamp(str(s)).value // 86_400_000_000_000))
    return rng.normal(0.0, 1.0, len(SYMS))


def history(first: str, last: str, fn) -> dict:
    return {str(d.date()): fn(d.date()) for d in pd.date_range(first, last, freq="7D")}


def register(inp: dict, spec: book.BookSpec | None = None, name: str | None = None, note: str | None = None) -> str:
    rc = T.register(inp, spec or book.load_spec(), name=name, note=note, log=QUIET)
    assert rc == 0
    return T.registered(T.read())[-1]["test_id"]


def refusal(fn, *words) -> str:
    try:
        fn()
    except SystemExit as exc:
        text = str(exc.code if exc.code is not None else "")
        for w in words:
            assert w in text, (w, text)
        return text
    raise AssertionError(f"not refused: {words}")


def tree(root: Path, skip: str | None = None) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not (skip and p.relative_to(root).parts[0] == skip)
    }


def weeks_of(test_id: str) -> list[dict]:
    return [w for w in T.read()["weeks"]["records"] if w["test_id"] == test_id]


def log_on_time(test_id: str, sundays: list, fn) -> None:
    """Log the value of each Sunday at 23:00 of that Sunday (on time)."""
    for s in sundays:
        clock(at(s, 23))
        assert T.log_values(test_id, values_csv({str(s): fn(s)}), log=QUIET) == 0


def write_chain(r: Path, objs: list[dict], key: str = "registrations") -> None:
    """A chain file of the given objects and a matching head.json in the folder r."""
    (r / "values").mkdir(parents=True, exist_ok=True)
    prev, lines = LG.GENESIS, []
    for i, obj in enumerate(objs, start=1):
        line = LG.canonical({**obj, "seq": i, "prev_sha256": prev})
        prev = LG.sha_text(line)
        lines.append(line)
    (r / T.CHAINS[key]).write_text("".join(x + "\n" for x in lines), encoding="ascii")
    head = {"about": "test"}
    for k in T.CHAINS:
        head[k] = {"lines": len(lines), "sha256": prev} if k == key else {"lines": 0, "sha256": LG.GENESIS}
    (r / LG.HEAD).write_text(json.dumps(head), encoding="ascii")


def rewrite_chain(key: str, edit) -> None:
    """Apply edit(records) to one chain of the ledger and write it again with correct links."""
    rec = T.read()
    objs = [{k: v for k, v in r.items() if k not in ("seq", "prev_sha256")} for r in rec[key]["records"]]
    objs = edit(objs)
    r = T.root()
    prev, lines = LG.GENESIS, []
    for i, obj in enumerate(objs, start=1):
        line = LG.canonical({**obj, "seq": i, "prev_sha256": prev})
        prev = LG.sha_text(line)
        lines.append(line)
    (r / T.CHAINS[key]).write_text("".join(x + "\n" for x in lines), encoding="ascii")
    head = json.loads((r / LG.HEAD).read_text())
    head[key] = {"lines": len(lines), "sha256": prev}
    (r / LG.HEAD).write_text(json.dumps(head), encoding="ascii")


def run_cli(*args: str, data: Path | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "FACTORLAB_DATA": str(data or DATA), "TMPDIR": str(TMP)}
    return subprocess.run(
        [sys.executable, "-B", str(ENGINE / "judge.py"), *args],
        cwd=cwd or ENGINE,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


# --------------------------------------------------------------------------- 1. check and register


def test_check_writes_nothing():
    fresh()
    clock(at("2026-02-21"))
    before = tree(DATA)
    r = T.check({"formula": PASSING, "source": None}, book.load_spec())
    p = values_csv(history("2025-06-01", "2026-02-15", lambda s: random_values(s, 1)))
    v = T.check({"values_file": p}, book.load_spec("long_only"))
    assert tree(DATA) == before, "check wrote a file"
    assert not (DATA / "trials").exists()
    assert r["historical"]["result"] == "pass" and r["pit"]["status"] == "proven"
    assert v["pit"]["status"] == "not_proven" and v["historical"]["result"] in ("pass", "useless")
    assert r["first_forward_sunday"] == str(F) and v["first_forward_sunday"] == str(F)
    json.dumps(T.public(r), allow_nan=False)  # the JSON part is plain JSON


def test_register_formula_appends_one_line_and_writes_results():
    fresh()
    clock(at("2026-02-21"))
    msgs: list[str] = []
    rc = T.register({"formula": PASSING, "source": None}, book.load_spec(), name="clv", note=None, log=msgs.append)
    tid = T.registered(T.read())[-1]["test_id"]
    # the next steps: one "cd engine" line, then the commands (each line works after the one before)
    i = msgs.index("Next steps (run these commands from the repository root):")
    assert rc == 0 and msgs[i + 1 :] == [
        "  cd engine",
        "  .venv/bin/python -B judge.py log-week",
        "  .venv/bin/python -B judge.py status FT-0001",
    ], msgs
    rec = T.read()
    assert tid == "FT-0001" and [r["kind"] for r in rec["registrations"]["records"]] == ["registration"]
    line = rec["registrations"]["records"][0]
    for key in ("test_id", "name", "registered_at", "input", "factor", "pit", "book_spec", "signed_values_sha256",
                "related", "rule", "historical", "sigma_floor", "sigma_floor_source", "data_through",
                "complete_through", "first_forward_sunday", "code", "version"):  # fmt: skip
        assert key in line, key
    assert line["registered_at"] == "2026-02-21T10:00:00Z" and line["first_forward_sunday"] == str(F)
    assert line["factor"]["expression"] == PASSING and line["pit"]["status"] == "proven"
    assert line["rule"] == rules.rule_record() and line["book_spec"]["sha256"] == book.load_spec().sha256
    assert line["data_through"] == {"okx": CUT, "binance": None}
    res = T.root() / "results" / "ft-0001"
    assert sorted(p.name for p in res.iterdir()) == [
        "historical.json",
        "historical_daily.parquet",
        "historical_weekly.parquet",
    ]
    # the numbers of the line equal a computation from the daily marks of the result file
    daily = pd.read_parquet(res / "historical_daily.parquet").set_index("date")
    weekly = pd.read_parquet(res / "historical_weekly.parquet")
    d = daily["book_equity"].pct_change().iloc[1:]
    h = rules.historical_test(d, weekly.set_index("sunday")["excess"], held=weekly.set_index("sunday")["held"],
                              fallback_floor=0.03)  # fmt: skip
    for k in ("total_return", "sharpe", "share_positive"):
        assert line["historical"][k] == LG.num(h[k]), k
    assert line["historical"]["n_windows"] == h["n_windows"] and line["sigma_floor"] == LG.num(h["sigma_floor"])
    doc = json.loads((res / "historical.json").read_text())
    assert doc["test_id"] == "FT-0001" and len(doc["historical"]["windows"]) == h["n_windows"]
    assert T.verdict_of(rec, T.registered(rec)[0]) == {
        "word": "unproven (historical pass)",
        "by": "historical",
        "final": False,
    }


def test_register_accepts_a_historically_useless_factor():
    fresh()
    clock(at("2026-02-21"))
    tid = register({"formula": FORMULA, "source": None})
    rec = T.read()
    t = T.registered(rec)[0]
    assert t["historical"]["result"] == "useless" and t["historical"]["reasons"], t["historical"]
    assert T.verdict_of(rec, t)["word"] == "useless (historical)"
    d = T.status_data()
    assert d["tests"][0]["verdict"] == {"word": "useless (historical)", "by": "historical"} and tid == "FT-0001"
    # a forward decision replaces the historical word (the foresight scenario: test B)
    sc = foresight_scenario()
    b = next(x for x in sc["status"]["tests"] if x["id"] == sc["B"])
    assert b["historical"]["result"] == "useless" and b["verdict"] == {"word": "useful", "by": "forward"}, b


def test_register_refuses_a_pit_failure_and_a_duplicate():
    fresh()
    clock(at("2026-02-21"))
    saved = uf._cut_row

    def shifted(expr, panels, pos):
        return saved(expr, panels, pos) + 1.0

    uf._cut_row = shifted
    try:
        refusal(lambda: T.register({"formula": PASSING, "source": None}, book.load_spec(), name=None, note=None),
                "not proven point in time")  # fmt: skip
    finally:
        uf._cut_row = saved
    assert not (DATA / "trials" / "registrations.jsonl").exists()
    register({"formula": PASSING, "source": None})
    before = tree(DATA / "trials")
    refusal(lambda: register({"formula": PASSING, "source": None}), "Test FT-0001 has the same book")
    refusal(lambda: register({"formula": "1 * ts_mean(clv, 7)", "source": None}), "Test FT-0001")
    assert tree(DATA / "trials") == before


def test_register_refuses_code_drift():
    fresh()
    clock(at("2026-02-21"))
    register({"formula": PASSING, "source": None})
    saved = T.code_record

    def other():
        c = saved()
        return {**c, "sha256": "f" * 64, "files": {**c["files"], "engine/rules.py": "0" * 64}}

    T.code_record = other
    try:
        refusal(lambda: register({"formula": FORMULA, "source": None}), "Record the code change first")
    finally:
        T.code_record = saved
    assert len(T.registered(T.read())) == 1


def test_short_history_uses_the_fallback_floor():
    fresh()
    clock(at("2026-02-21"))
    p = values_csv(history("2025-07-20", "2026-02-08", foresight))
    tid = register({"values_file": p})
    t = T.registered(T.read())[0]
    h = t["historical"]
    assert tid == "FT-0001" and h["weeks"] == 30 and h["held_weeks"] == 30 and h["result"] == "pass", h
    assert h["n_windows"] == 0 and h["total_return"] > 0
    assert t["sigma_floor_source"] == "fallback" and t["sigma_floor"] == 0.03
    assert t["pit"]["status"] == "not_proven" and t["factor"]["file_name"] == "values.csv"
    assert len(t["factor"]["history"]) == 30


def test_short_history_useless():
    fresh()
    clock(at("2026-02-21"))
    p = values_csv(history("2025-07-20", "2026-02-08", lambda s: foresight(s, -1.0)))
    register({"values_file": p})
    t = T.registered(T.read())[0]
    assert t["historical"]["result"] == "useless", t["historical"]
    assert t["historical"]["reasons"][0] == "The total return after costs is 0 or less."
    assert T.verdict_of(T.read(), t)["word"] == "useless (historical)"


def test_no_history_is_refused():
    fresh()
    clock(at("2026-02-21"))
    p = values_csv({"2026-02-15": foresight("2026-02-15")})
    text = refusal(lambda: register({"values_file": p}), "Not registered. The history has no complete week of values.")
    assert "Give values for at least one Sunday before the registration week." in text
    r = T.check({"values_file": p}, book.load_spec())
    assert r["historical"]["result"] == "none" and r["historical"]["weeks"] == 0
    assert not (DATA / "trials" / "registrations.jsonl").exists()


def test_duplicate_by_content_is_refused():
    fresh()
    clock(at("2026-02-21"))
    register({"formula": FORMULA, "source": None})
    for spec, inp in (
        (spec_of(name="renamed"), {"formula": FORMULA, "source": None}),
        (spec_of(version=2), {"formula": FORMULA, "source": None}),
        (
            spec_of('[costs]\nscenario = "custom"\nfee_bps = 5\nslippage_bps = 5\n'),
            {"formula": FORMULA, "source": None},
        ),
        (spec_of("[book]\nsign = -1\n"), {"formula": "ts_mean(ex_ret, 20)", "source": None}),
        (spec_of("[forward]\nsigma_floor_fallback = 0.05\n"), {"formula": FORMULA, "source": None}),
    ):
        refusal(lambda s=spec, i=inp: register(i, s), "Test FT-0001 has the same book and the same factor values")
    # the mirror sign and another book content are other tests
    assert register({"formula": FORMULA, "source": None}, spec_of("[book]\nsign = -1\n")) == "FT-0002"
    assert register({"formula": FORMULA, "source": None}, spec_of("[book]\ntop_fraction = 0.3\n")) == "FT-0003"
    # values files: the same values as CSV, Parquet, in another row order, with another layout
    rows = history("2025-01-05", "2026-02-15", lambda s: random_values(s, 7))
    p = values_csv(rows)
    assert register({"values_file": p}) == "FT-0004"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    q = p.with_suffix(".parquet")
    df.sample(frac=1.0, random_state=3).to_parquet(q, index=False)
    n = len(df)
    order = list(np.random.default_rng(5).permutation(n))
    p2 = values_csv(rows, order=order)
    p3 = p.with_name("layout.csv")
    p3.write_text(p.read_text() + "\n", encoding="utf-8")
    for f in (q, p2, p3):
        refusal(lambda f=f: register({"values_file": f}), "Test FT-0004 has the same book")
    rel = T.related(T.read())
    assert set(rel["FT-0001"]) >= {"FT-0002", "FT-0003"}, rel


def test_related_count():
    fresh()
    clock(at("2026-02-21"))
    a = register({"formula": FORMULA, "source": None})
    b = register({"formula": "-ts_mean(ex_ret, 21)", "source": None}, book.load_spec("long_only"))
    c = register({"formula": PASSING, "source": None})
    rec = T.read()
    rel = T.related(rec)
    assert rel[a] == [b] and rel[b] == [a] and rel[c] == [], rel
    line_b = T.registered(rec)[1]
    assert line_b["related"] == [a]
    d = T.status_data()
    by = {x["id"]: x for x in d["tests"]}
    assert by[a]["related"] == [b] and by[b]["related"] == [a] and by[c]["related"] == []
    r = T.check({"formula": "-ts_mean(ex_ret, 22)", "source": None}, book.load_spec())
    assert set(r["related"]) == {a, b} and r["largest"]["test_id"] in (a, b) and r["largest"]["abs_corr"] >= 0.85


# --------------------------------------------------------------------------- 3. clock, F and code


def test_clock_is_read_once():
    fresh()
    c = clock(at("2026-02-21", 23, 59, 59), at("2026-02-22", 0, 0, 1))
    register({"values_file": values_csv(history("2025-06-01", "2026-02-15", lambda s: random_values(s, 2)))})
    t = T.registered(T.read())[0]
    assert t["registered_at"] == "2026-02-21T23:59:59Z" and t["first_forward_sunday"] == "2026-02-22"
    assert c.calls >= 2  # the head of the ledger read the clock again; the line did not
    clock(at("2026-02-22", 23, 59, 59), at("2026-02-23", 0, 0, 1))
    T.log_values(t["test_id"], values_csv({"2026-02-22": random_values("2026-02-22", 2)}), log=QUIET)
    r = T.read()["values_file"]["records"][0]
    assert r["logged_at"] == "2026-02-22T23:59:59Z" and r["counts_from"] == {"2026-02-22": "2026-02-23"}
    clock(at("2026-02-25"))
    assert T.problems(T.read()) == [] and T.verify(log=QUIET) == 0


def test_alphaforge_switch_is_not_code_drift():
    assert "engine/config.py" not in T.CODE_FILES
    a = T.code_record()
    saved = C.ALPHAFORGE_ENABLED, list(C.UNIVERSE)
    try:
        C.ALPHAFORGE_ENABLED = not saved[0]
        assert T.code_record() == a
        C.UNIVERSE = [*saved[1][:-1], "DOT/USDT:USDT"]
        b = T.code_record()
        assert b["sha256"] != a["sha256"] and b["files"] == a["files"] and b["config"] != a["config"]
    finally:
        C.ALPHAFORGE_ENABLED, C.UNIVERSE = saved[0], saved[1]
    assert T.code_record() == a


def test_first_forward_sunday_rule():
    fresh()
    inp = {"formula": PASSING, "source": None}
    spec = book.load_spec()
    assert T.check(inp, spec, now=at("2026-02-21"))["first_forward_sunday"] == "2026-02-22"
    assert T.check(inp, spec, now=at("2026-02-22", 9))["first_forward_sunday"] == "2026-03-01"  # never the same day
    assert T.check(inp, spec, now=at("2026-02-25"))["first_forward_sunday"] == "2026-03-01"
    cut_to("2026-02-22")  # the bar of Sunday 02-22 is on disk (Monday after 00:00)
    assert T.check(inp, spec, now=at("2026-02-23", 1))["first_forward_sunday"] == "2026-03-01"
    assert LG.next_sunday_after(date(2026, 2, 22)) == date(2026, 3, 1)


# --------------------------------------------------------------------------- 4. log-week and its rules


def test_log_week_logs_cut_values_with_the_pit_check():
    fresh()
    clock(at("2026-02-21"))
    tid = register({"formula": PASSING, "source": None})
    cut_to("2026-03-10", label_table=False)
    clock(at("2026-03-11"))
    assert T.log_week(log=QUIET) == 0
    rec = T.read()
    lines = rec["values_okx"]["records"]
    assert [x["sunday"] for x in lines] == ["2026-02-22", "2026-03-01", "2026-03-08"]
    src = uf.load_source("okx")
    want = uf.formula_values(PASSING, src)
    idx = {str(d.date()): i for i, d in enumerate(want["sundays"])}
    for x in lines:
        assert (
            x["kind"] == "week_values" and x["pit_check"]["equal"] is True and x["logged_at"] == "2026-03-11T10:00:00Z"
        )
        assert set(x["inputs"]["fields_sha256"]) == {"close", "high", "low"}, x["inputs"]
        got = LG.from_list(x["values"][tid])
        assert np.array_equal(got, want["values"][idx[x["sunday"]]], equal_nan=True)
    assert [w["week"] for w in weeks_of(tid)] == [1, 2]  # the week of 03-08 ends after the data
    assert T.log_week(log=QUIET) == 0 and len(T.read()["values_okx"]["records"]) == 3  # a second run adds nothing


def test_closed_rule_holds_the_log():
    # a late coin bar: complete_through is the oldest last bar of the live coins
    fresh()
    clock(at("2026-02-21"))
    tid = register({"formula": PASSING, "source": None})
    cut_to("2026-03-08", ends={"ETH-USDT": "2026-03-07"}, label_table=False)
    clock(at("2026-03-09"))
    T.log_week(log=QUIET)
    assert [x["sunday"] for x in T.read()["values_okx"]["records"]] == ["2026-02-22", "2026-03-01"]
    # the other coins have the bar of 2026-03-08, so the late coin holds the test (status shows it)
    held = T.status_data()["tests"][0]["forward"]["held"]
    assert held == {"sunday": "2026-03-08", "reason": T.HELD_GAP}, held
    # a late funding value holds a test that reads funding
    fresh()
    clock(at("2026-02-21"))
    tf = register({"formula": "ts_mean(funding, 7)", "source": None})
    fb = f"funding_{C.FUNDING_FALLBACK_EXCHANGE_ID}"
    cut_to(
        "2026-03-08",
        drop={("funding", "BTC-USDT"): ["2026-03-08"], (fb, "BTC-USDT"): ["2026-03-08"]},
        label_table=False,
    )
    clock(at("2026-03-09"))
    T.log_week(log=QUIET)
    assert [x["sunday"] for x in T.read()["values_okx"]["records"]] == ["2026-02-22", "2026-03-01"]
    held = T.status_data()["tests"][0]["forward"]["held"]
    assert held == {"sunday": "2026-03-08", "reason": "a value is missing; the week waits up to 7 days"}, held
    assert tf == "FT-0001" and tid == "FT-0001"
    # a late Binance field holds a Binance test
    fresh()
    with binance_to("2026-02-20"):
        clock(at("2026-02-21"))
        tb = register({"formula": "ts_mean(ls_ratio, 7)", "source": "binance"})
    raw = binance_raw("2026-03-08")
    raw["ETH-USDT"].loc[pd.Timestamp("2026-03-08", tz="UTC"), "long_short_ratio"] = np.nan
    cut_to("2026-03-08", label_table=False)
    with binance_to(raw=raw):
        clock(at("2026-03-09"))
        T.log_week(log=QUIET)
    lines = T.read()["values_binance"]["records"]
    assert [x["sunday"] for x in lines] == ["2026-02-22", "2026-03-01"] and tb in lines[0]["values"]
    assert lines[0]["inputs"]["cleaning_freeze_sha256"] == lf.CLEANING_FREEZE["sha256"]
    # a Binance panel that is not updated: the test stays at its last bar, and status shows "held"
    fresh()
    with binance_to("2026-02-20"):
        clock(at("2026-02-21"))
        tb = register({"formula": "ts_mean(ls_ratio, 7)", "source": "binance"})
    cut_to("2026-03-08", label_table=False)
    with binance_to("2026-02-28"):
        clock(at("2026-03-09"))
        T.log_week(log=QUIET)
        assert [x["sunday"] for x in T.read()["values_binance"]["records"]] == ["2026-02-22"]
        held = T.status_data()["tests"][0]["forward"]["held"]
    assert held == {
        "sunday": "2026-03-01",
        "reason": "the Binance data ends on 2026-02-28; the test waits for its update",
    }, held
    # an OKX test that waits for the next Sunday is not held
    fresh()
    clock(at("2026-02-21"))
    register({"formula": PASSING, "source": None})
    cut_to("2026-03-04", label_table=False)
    clock(at("2026-03-05"))
    T.log_week(log=QUIET)
    assert T.status_data()["tests"][0]["forward"]["held"] is None


class binance_to:  # noqa: N801 - a context manager used like a function
    """In this process the Binance panel ends on `day` (or is `raw`): libfactors.load_raw_binance
    gives the demo panel cut at that day."""

    def __init__(self, day=None, raw=None) -> None:
        self.raw = raw if raw is not None else binance_raw(day)

    def __enter__(self):
        self.saved = lf.load_raw_binance
        raw = self.raw
        lf.load_raw_binance = lambda panel_dir=None: {k: v.copy() for k, v in raw.items()}
        return self

    def __exit__(self, *exc):
        lf.load_raw_binance = self.saved


def binance_raw(day) -> dict:
    if "binance" not in _STATE:
        panel = TMP / "panel"
        demo_data.write_panel(panel, master()["market"])
        _STATE["binance"] = lf.load_raw_binance(panel)
    lim = pd.Timestamp(str(day), tz="UTC")
    return {k: v.loc[:lim].copy() for k, v in _STATE["binance"].items()}


def test_late_field_holds_only_its_test():
    fresh()
    clock(at("2026-02-21"))
    a = register({"formula": "ts_mean(funding, 7)", "source": None})
    b = register({"formula": PASSING, "source": None})
    fb = f"funding_{C.FUNDING_FALLBACK_EXCHANGE_ID}"
    cut_to(
        "2026-03-08",
        drop={("funding", "BTC-USDT"): ["2026-03-08"], (fb, "BTC-USDT"): ["2026-03-08"]},
        label_table=False,
    )
    clock(at("2026-03-09"))
    T.log_week(log=QUIET)
    lines = T.read()["values_okx"]["records"]
    assert [(x["sunday"], sorted(x["values"])) for x in lines] == [
        ("2026-02-22", [a, b]),
        ("2026-03-01", [a, b]),
        ("2026-03-08", [b]),
    ]
    assert set(lines[2]["closed_check"]["fields"]) == {"close", "high", "low"}  # the fields of b only
    cut_to("2026-03-09", label_table=False)  # the funding of 03-08 arrives
    clock(at("2026-03-10"))
    T.log_week(log=QUIET)
    lines = T.read()["values_okx"]["records"]
    assert [(x["sunday"], sorted(x["values"])) for x in lines[3:]] == [("2026-03-08", [a])]
    assert T.problems(T.read()) == []


def test_pit_mismatch_holds_only_its_test():
    fresh()
    clock(at("2026-02-21"))
    a = register({"formula": PASSING, "source": None})
    b = register({"formula": "ts_mean(ret, 5)", "source": None})
    cut_to("2026-03-02", label_table=False)
    clock(at("2026-03-03"))
    saved = uf._cut_row

    def shifted(expr, panels, pos):
        out = saved(expr, panels, pos)
        hit = expr == "ts_mean(ret, 5)" and panels["close"].index[pos] == pd.Timestamp("2026-03-01", tz="UTC")
        return out + 1.0 if hit else out

    uf._cut_row = shifted
    try:
        assert T.log_week(log=QUIET) == T.EXIT_ALARM
    finally:
        uf._cut_row = saved
    rec = T.read()
    assert [(x["sunday"], sorted(x["values"])) for x in rec["values_okx"]["records"]] == [
        ("2026-02-22", [a, b]),
        ("2026-03-01", [a]),
    ]
    al = [x for x in rec["alarms"]["records"] if x["type"] == "pit_mismatch"]
    assert len(al) == 1 and al[0]["test_id"] == b and al[0]["detail"]["sunday"] == "2026-03-01"
    held = {x["id"]: x["forward"]["held"] for x in T.status_data()["tests"]}
    assert held[b] == {"sunday": "2026-03-01", "reason": "pit_mismatch"} and held[a] is None


def test_week_complete_rule():
    fresh("2026-03-15")
    raw = lf.load_raw()
    okx = uf.load_source("okx", raw)
    s = date(2026, 3, 8)
    assert T._week_complete(okx, s, None)[0] is True
    assert T._week_complete(okx, date(2026, 3, 15), None)[0] is False  # the week ends after the data

    def with_bars(coin, keep) -> dict:
        r = dict(raw)
        px = raw[coin]["ohlcv"]
        r[coin] = {**raw[coin], "ohlcv": px[keep(px.index)]}
        return uf.load_source("okx", r)

    for name, keep in (
        ("a gap on the end day", lambda ix: ix <= pd.Timestamp("2026-03-14", tz="UTC")),
        ("a two-day gap at the end", lambda ix: ix <= pd.Timestamp("2026-03-13", tz="UTC")),
        ("a gap inside the week", lambda ix: ix != pd.Timestamp("2026-03-11", tz="UTC")),
    ):
        src = with_bars("ETH-USDT", keep)
        ok, final, why = T._week_complete(src, s, None)
        assert ok is False and final == [] and "ETH-USDT close" in why, (name, why)
    src = with_bars("ETH-USDT", lambda ix: ix <= pd.Timestamp("2026-03-13", tz="UTC"))
    assert src["complete_through"] == date(2026, 3, 13) and src["data_through"] == date(2026, 3, 15)


def test_final_gap_completes_the_week_with_an_alarm():
    fresh()
    clock(at("2026-02-21"))
    tid = register({"formula": PASSING, "source": None})
    lo = register({"formula": FORMULA, "source": None}, book.load_spec("long_only"))
    cut_to("2026-04-30", ends={"ETH-USDT": "2026-03-04"}, label_table=False)
    clock(at("2026-05-01"))
    assert T.log_week(log=QUIET) == T.EXIT_ALARM
    rec = T.read()
    gaps = [a for a in rec["alarms"]["records"] if a["type"] == "data_gap"]
    assert len(gaps) == 1 and gaps[0]["detail"] == {"coin": "ETH-USDT", "field": "close", "first_day": "2026-03-05"}, (
        gaps
    )
    wk = weeks_of(tid)
    assert len(wk) == 9, [w["sunday"] for w in wk]  # weeks to 2026-04-26
    assert wk[1]["final_gaps"][0] == ["ETH-USDT", "close", "2026-03-05"]
    # ETH is not eligible after its last bar: no target weight at the next rebalance
    t = T.registered(rec)[0]
    mk = T.market()
    rp = T.replay(t, rec, mk, date(2026, 4, 19))
    targets = T._targets(rp["spec"], rp["values"], pd.Timestamp(F, tz="UTC"), pd.Timestamp("2026-04-26", tz="UTC"),
                         mk["eligible"], None)  # fmt: skip
    after = targets[pd.Timestamp("2026-03-08", tz="UTC")]
    assert "ETH-USDT" not in after.index or after.get("ETH-USDT", 0.0) == 0.0
    assert len(weeks_of(lo)) == 9
    # a missing funding day: one alarm of funding_rate, and the week counts
    fresh()
    clock(at("2026-02-21"))
    tid = register({"formula": PASSING, "source": None})
    fb = f"funding_{C.FUNDING_FALLBACK_EXCHANGE_ID}"
    cut_to(
        "2026-03-31",
        drop={("funding", "SOL-USDT"): ["2026-03-03"], (fb, "SOL-USDT"): ["2026-03-03"]},
        label_table=False,
    )
    clock(at("2026-04-01"))
    assert T.log_week(log=QUIET) == T.EXIT_ALARM
    gaps = [a["detail"] for a in T.read()["alarms"]["records"] if a["type"] == "data_gap"]
    assert gaps == [{"coin": "SOL-USDT", "field": "funding_rate", "first_day": "2026-03-03"}], gaps
    assert len(weeks_of(tid)) == 5  # the weeks to 2026-03-29
    assert T.log_week(log=QUIET) == 0  # an alarm is recorded once


# --------------------------------------------------------------------------- 5. the forward replay


def test_book_week_equals_two_replays():
    import backtest as B

    fresh()
    clock(at("2026-02-21"))
    tid = register({"formula": PASSING, "source": None}, book.load_spec("long_only"))
    cut_to("2026-05-31", label_table=False)
    clock(at("2026-06-01"))
    T.log_week(log=QUIET)
    rec = T.read()
    wk = weeks_of(tid)
    assert len(wk) == 14
    t = T.registered(rec)[0]
    vals = {s: x["values"] for s, x in T.formula_logs(rec, t).items()}
    values = T._frame(vals)
    mk = T.market()
    start, end = pd.Timestamp(F, tz="UTC"), pd.Timestamp(wk[-1]["end"], tz="UTC")
    spec = book.load_spec("long_only")
    sundays = pd.date_range(start, end - pd.Timedelta(days=7), freq="7D")
    common = dict(prices=mk["prices"], funding=mk["funding"], start=str(start.date()), end=str(end.date()),
                  initial_equity=10_000.0, fee_bps=5.0, slippage_bps=5.0, gross_cap=1.0, rebalance_weekday=0,
                  benchmark="BTC-USDT", accounting=B.ACCOUNTING_RECONCILED)  # fmt: skip
    b = B.run_backtest(
        book.book_weights_fn(spec), factors=book.factor_frame(values.reindex(sundays), mk["eligible"]), **common
    )
    c = B.run_backtest(book.equal_weight, factors=book.universe_frame(mk["eligible"], sundays), **common)
    eb, ec = b.equity["equity"], c.equity["equity"]
    for w in wk:
        s, e = pd.Timestamp(w["sunday"], tz="UTC"), pd.Timestamp(w["end"], tz="UTC")
        rb, rc = eb[e] / eb[s] - 1.0, ec[e] / ec[s] - 1.0
        assert abs(w["book_ret"] - rb) < 1e-12 and abs(w["control_ret"] - rc) < 1e-12, w["week"]
        assert w["excess"] == w["book_ret"] - w["control_ret"]
        assert w["book"]["n_long"] >= 2 and w["book"]["n_short"] == 0 and w["held"] is True
    assert wk[0]["book"]["trade_date"] == "2026-02-23" and wk[0]["control"]["cost"] > 0


def foresight_scenario() -> dict:
    """Three values-file tests registered on 2026-02-21, logged on time for 26 weeks:
    A foresight (history and forward), B a historically useless history with forward foresight
    (another book), C negated foresight. log-week then runs on the full data. Made once."""
    saved = TMP / "foresight_trials"
    if "foresight" in _STATE:
        shutil.rmtree(DATA / "trials", ignore_errors=True)
        shutil.copytree(saved, DATA / "trials")
        cut_to(LAST, label_table=False)
        return _STATE["foresight"]
    fresh()
    clock(at("2026-02-21"))
    hist = ("2022-04-03", "2026-02-15")
    a = register({"values_file": values_csv(history(*hist, foresight))})
    neg = values_csv(history(*hist, lambda s: foresight(s, -1.0)))
    b = register({"values_file": neg}, spec_of("[book]\ntop_fraction = 0.3\nbottom_fraction = 0.3\n"))
    c = register({"values_file": neg})
    sundays = [F + timedelta(days=7 * k) for k in range(28)]
    log_on_time(a, sundays, foresight)
    log_on_time(b, sundays, foresight)
    log_on_time(c, sundays, lambda s: foresight(s, -1.0))
    cut_to(LAST, label_table=False)
    clock(at("2026-10-01"))
    rc = T.log_week(log=QUIET)
    out = {"A": a, "B": b, "C": c, "rc": rc, "status": T.status_data(), "trials": tree(DATA / "trials")}
    shutil.copytree(DATA / "trials", saved)
    _STATE["foresight"] = out
    return out


def test_foresight_values_are_useful_at_week_26():
    sc = foresight_scenario()
    for tid in (sc["A"], sc["B"]):
        wk = weeks_of(tid)
        assert len(wk) == 26 and [w["cs"]["state"] for w in wk[:25]] == ["running"] * 25, [w["cs"]["state"] for w in wk]
        assert wk[-1]["cs"]["state"] == "useful" and wk[-1]["cs"]["lower"] > 0
        row = next(x for x in sc["status"]["tests"] if x["id"] == tid)
        assert row["forward"]["state"] == "useful" and row["forward"]["decided_week"] == 26
        assert row["verdict"] == {"word": "useful", "by": "forward"}
        assert all(w["held"] for w in wk) and wk[0]["book"]["trade_date"] == "2026-02-23"
    assert sc["rc"] == 0


def test_negated_foresight_values_are_useless_at_week_26():
    sc = foresight_scenario()
    wk = weeks_of(sc["C"])
    assert len(wk) == 26 and wk[-1]["cs"]["state"] == "useless" and wk[-1]["cs"]["upper"] < 0
    row = next(x for x in sc["status"]["tests"] if x["id"] == sc["C"])
    assert row["verdict"] == {"word": "useless", "by": "forward"} and row["forward"]["decided_week"] == 26


def test_no_decision_after_a_decision():
    sc = foresight_scenario()
    assert T.decision(T.read(), sc["A"])["week"] == 26
    clock(at("2026-10-05"))
    assert T.log_week(log=QUIET) == 0
    assert tree(DATA / "trials") == sc["trials"], "a run after the decisions changed the ledger"
    assert T.problems(T.read()) == []
    refusal(lambda: T.log_values(sc["A"], values_csv({"2026-10-04": foresight("2026-09-20")}), log=QUIET),
            "has its forward decision")  # fmt: skip


def test_random_values_follow_the_rule():
    fresh("2023-09-22")
    clock(at("2023-09-23"))
    tid = register({"values_file": values_csv(history("2022-04-03", "2023-09-17", lambda s: random_values(s, 11)))})
    t = T.registered(T.read())[0]
    assert t["first_forward_sunday"] == "2023-09-24" and t["sigma_floor_source"] == "history"
    sundays = [date(2023, 9, 24) + timedelta(days=7 * k) for k in range(157)]
    log_on_time(tid, sundays, lambda s: random_values(s, 11))
    cut_to(LAST, label_table=False)
    clock(at("2026-10-01"))
    T.log_week(log=QUIET)
    wk = weeks_of(tid)
    assert 26 <= len(wk) <= rules.MAX_WEEKS, len(wk)
    xs = [w["excess"] for w in wk]
    fwd = rules.forward_test(xs, t["sigma_floor"])
    for w, p in zip(wk, fwd["path"]):
        assert w["cs"] == T._cs_record(p), w["week"]
    assert fwd["state"] in ("running", "useful", "useless")
    if fwd["state"] in ("useful", "useless"):
        assert fwd["decided_week"] == len(wk)
    assert T.problems(T.read()) == []


# --------------------------------------------------------------------------- 6. values files


def _values_test(spec: book.BookSpec | None = None, seed: int = 3) -> str:
    fresh()
    clock(at("2026-02-21"))
    return register(
        {"values_file": values_csv(history("2025-06-01", "2026-02-15", lambda s: random_values(s, seed)))}, spec
    )


def test_values_file_late_rule():
    tid = _values_test()
    s1, s2, s3 = F, F + timedelta(days=7), F + timedelta(days=14)
    msgs: list[str] = []
    clock(at(s1, 23))  # Sunday 23:00: the bars end on the Saturday before
    T.log_values(tid, values_csv({str(s1): random_values(s1, 3)}), log=msgs.append)
    clock(at(s2 + timedelta(days=1), 10))  # Monday 10:00
    T.log_values(tid, values_csv({str(s2): random_values(s2, 3)}), log=msgs.append)
    clock(at(s3 + timedelta(days=7), 0))  # the next Sunday 00:00
    T.log_values(tid, values_csv({str(s3): random_values(s3, 3)}), log=msgs.append)
    assert msgs == ["2026-02-22: counts from Monday 2026-02-23.", "2026-03-01: counts from 2026-03-03.",
                    "2026-03-08: too late: not used."], msgs  # fmt: skip
    recs = T.read()["values_file"]["records"]
    assert [r["used"] for r in recs] == [{"2026-02-22": True}, {"2026-03-01": True}, {"2026-03-08": False}]
    cut_to("2026-03-22", label_table=False)
    clock(at("2026-03-23"))
    T.log_week(log=QUIET)
    wk = weeks_of(tid)
    assert [w["book"]["trade_date"] for w in wk[:3]] == ["2026-02-23", "2026-03-03", "2026-03-09"], wk
    assert [w["held"] for w in wk[:3]] == [True, True, False]
    assert wk[2]["book"]["n_long"] == 0 and wk[2]["book"]["n_short"] == 0  # the zero control: no position
    assert wk[0]["values_ref"] == {"chain": "values/file.jsonl", "seq": 1} and wk[2]["values_ref"] is None


def test_values_file_logged_before_the_bar_keeps_all_rows():
    tid = _values_test()
    cut_to("2026-02-21", label_table=False)  # Sunday 23:00: the bars end on Saturday
    clock(at(F, 23))
    T.log_values(tid, values_csv({str(F): random_values(F, 3)}), log=QUIET)
    rec = T.read()
    line = rec["values_file"]["records"][0]
    assert sum(v is not None for v in line["values"][str(F)]) == 20
    clock(at(F + timedelta(days=1), 9))
    msgs: list[str] = []
    assert T.log_values(tid, values_csv({str(F): random_values(F, 3)}), log=msgs.append) == 0
    rec2 = T.read()
    assert len(rec2["values_file"]["records"]) == 1 and rec2["alarms"]["records"] == []
    assert any("already logged with the same values" in m for m in msgs), msgs


def test_unlogged_values_file_week_holds_the_control():
    # a long-only values test against the equal-weight control, with no logged value, in a
    # falling market: the equal-weight book of the demo market falls from 2024-09 to 2025-04,
    # so a book in cash would beat the control in each week
    fresh("2024-09-06")
    clock(at("2024-09-07"))
    hist = values_csv(history("2024-01-07", "2024-09-01", lambda s: random_values(s, 3)))
    tid = register({"values_file": hist}, book.load_spec("long_only"))
    t = T.registered(T.read())[0]
    assert t["first_forward_sunday"] == "2024-09-08" and t["sigma_floor_source"] == "fallback"
    cut_to("2025-04-10", label_table=False)
    clock(at("2025-04-11"))
    T.log_week(log=QUIET)
    wk = weeks_of(tid)
    assert len(wk) == 30, len(wk)
    control = float(np.prod([1.0 + w["control_ret"] for w in wk]) - 1.0)
    assert control < -0.3, control  # the market falls
    assert all(w["held"] is False for w in wk)
    # the book has the control portfolio: the excess is 0 (the entry costs are equal), never -R_control
    assert all(abs(w["excess"]) < 1e-12 for w in wk), [w["excess"] for w in wk]
    assert all(w["cs"]["state"] == "running" for w in wk)
    fwd = T.forward_of(T.read(), t)
    assert fwd["mean"] == 0.0 and fwd["state"] == "running"
    # with an excess of 0 in each week, the test ends "useless" at week 156 at the latest
    end = rules.forward_test([0.0] * rules.MAX_WEEKS, float(t["sigma_floor"]))
    assert end["state"] == "useless" and end["decided_week"] == rules.MAX_WEEKS, end


def test_values_file_sunday_logged_after_a_later_sunday():
    tid = _values_test()
    s1, s2, s3 = F, F + timedelta(days=7), F + timedelta(days=14)
    clock(at(s1, 23))
    T.log_values(tid, values_csv({str(s1): random_values(s1, 3)}), log=QUIET)
    msgs: list[str] = []
    clock(at(s3, 23))  # the file has no value for s2: s2 is not written
    T.log_values(tid, values_csv({str(s2): np.full(len(SYMS), np.nan), str(s3): random_values(s3, 3)}),
                 log=msgs.append)  # fmt: skip
    clock(at(s3 + timedelta(days=1), 10))  # the value of s2 comes later: it is logged and not used
    T.log_values(tid, values_csv({str(s2): random_values(s2, 3)}), log=msgs.append)
    assert msgs == ["2026-03-08: counts from Monday 2026-03-09.", "2026-03-01: too late: not used."], msgs
    rec = T.read()
    assert [sorted(r["values"]) for r in rec["values_file"]["records"]] == [[str(s1)], [str(s3)], [str(s2)]]
    assert T.problems(rec) == []
    cut_to("2026-03-22", label_table=False)
    clock(at("2026-03-23"))
    T.log_week(log=QUIET)
    wk = weeks_of(tid)
    assert [w["week"] for w in wk] == [1, 2, 3, 4] and [w["held"] for w in wk[:3]] == [True, False, True]
    assert T.verify(log=QUIET) == 0
    # a Sunday that a test logs two times is a problem for verify
    rewrite_chain("values_file", lambda objs: [*objs, {**objs[1], "logged_at": objs[2]["logged_at"]}])
    assert any("logs 2026-03-08 a second time" in x for x in T.problems(T.read())), T.problems(T.read())


def test_values_file_restatement_is_refused_and_alarmed():
    tid = _values_test()
    clock(at(F, 23))
    v = random_values(F, 3)
    T.log_values(tid, values_csv({str(F): v}), log=QUIET)
    before = T.read()["values_file"]["records"]
    clock(at(F + timedelta(days=2)))
    w = v.copy()
    w[4] += 1.0
    refusal(lambda: T.log_values(tid, values_csv({str(F): w}), log=QUIET), "another value for a logged cell")
    rec = T.read()
    assert rec["values_file"]["records"] == before
    al = rec["alarms"]["records"]
    assert len(al) == 1 and al[0]["type"] == "file_value_changed" and al[0]["test_id"] == tid


def test_values_file_future_date_is_refused():
    tid = _values_test()
    clock(at(F, 10))
    try:
        T.log_values(tid, values_csv({str(F + timedelta(days=7)): random_values(F, 3)}), log=QUIET)
    except uf.InputError as exc:
        assert "a date after 2026-02-22" in str(exc), str(exc)
    else:
        raise AssertionError("a future date was accepted")
    assert T.read()["values_file"]["records"] == []


def test_selective_logging_does_not_skip_weeks():
    tid = _values_test()
    s1, s3 = F, F + timedelta(days=14)
    log_on_time(tid, [s1, s3], lambda s: random_values(s, 3))
    cut_to("2026-03-22", label_table=False)
    clock(at("2026-03-23"))
    T.log_week(log=QUIET)
    wk = weeks_of(tid)
    assert [w["week"] for w in wk] == [1, 2, 3, 4] and [w["held"] for w in wk[:3]] == [True, False, True]


# --------------------------------------------------------------------------- 7. the ledger


def test_hash_chain_finds_edits_removals_and_moves():
    shutil.rmtree(DATA / "trials", ignore_errors=True)
    r = DATA / "trials"
    write_chain(r, [{"kind": "test", "n": i, "sign": 1 if i % 2 else -1} for i in range(1, 13)])
    reg = r / "registrations.jsonl"
    good = reg.read_text()
    lines = good.splitlines()
    head_good = (r / "head.json").read_text()
    T.read()

    def broken(text: str | None = None, head: str | None = None, word: str = "") -> None:
        if text is not None:
            reg.write_text(text)
        if head is not None:
            (r / "head.json").write_text(head)
        try:
            T.read()
        except SystemExit as exc:
            assert word in str(exc) and str(exc).startswith("judge: "), (word, str(exc))
        else:
            raise AssertionError(f"the change was not found ({word})")
        finally:
            reg.write_text(good)
            if head is not None:
                (r / "head.json").write_text(head_good)

    edit = lines.copy()
    edit[9] = (
        edit[9].replace('"sign":1', '"sign":-1') if '"sign":1' in edit[9] else edit[9].replace('"sign":-1', '"sign":1')
    )
    broken("\n".join(edit) + "\n", word="line 11 does not hold the hash of line 10")
    broken("\n".join(lines[:9] + lines[10:]) + "\n", word="seq")
    broken("\n".join(lines[:3] + [lines[4], lines[3]] + lines[5:]) + "\n", word="seq")  # a moved line
    edit = lines.copy()
    last = json.loads(edit[-1])
    last["edited"] = True
    edit[-1] = LG.canonical(last)
    broken("\n".join(edit) + "\n", word="does not have the hash")
    broken("\n".join(lines[:-1]) + "\n", word="a line was removed")
    broken(good.rstrip("\n"), word="no line end")
    broken(good.replace('{"', '{ "', 1), word="canonical")
    # a head.json with other types: a RecordBroken that names head.json, never another error
    sha_last = json.loads(head_good)["registrations"]["sha256"]
    for bad_head in (
        [1],
        "text",
        {"registrations": 1},
        {"registrations": [len(lines), sha_last]},
        {"registrations": {"lines": "one", "sha256": sha_last}},
        {"registrations": {"lines": float(len(lines)), "sha256": sha_last}},
        {"registrations": {"lines": True, "sha256": sha_last}},
        {"registrations": {"lines": -1, "sha256": LG.GENESIS}},
        {"registrations": {"sha256": sha_last}},
        {"registrations": {"lines": len(lines)}},
        {"registrations": {"lines": len(lines), "sha256": None}},
    ):
        broken(head=json.dumps(bad_head), word="head.json")
    (r / "head.json").unlink()
    try:
        T.read()
        raise AssertionError("an absent head.json was not found")
    except T.RecordBroken as exc:
        assert "head.json" in str(exc)
    (r / "head.json").write_text(head_good)
    # a correct line after the head (a run that stopped between its two writes) is accepted
    extra = LG.canonical({"kind": "test", "n": 99, "seq": len(lines) + 1, "prev_sha256": LG.sha_text(lines[-1])})
    reg.write_text(good + extra + "\n")
    assert len(T.read()["registrations"]["records"]) == len(lines) + 1
    # an append of no line writes nothing; it reads and checks the record first
    files = sorted(f for f in r.rglob("*") if f.is_file())
    data = [f.read_bytes() for f in files]
    assert T.ledger().append("alarms", []) == []
    assert sorted(f for f in r.rglob("*") if f.is_file()) == files and [f.read_bytes() for f in files] == data
    reg.write_text(good.replace('{"', '{ "', 1))
    try:
        T.ledger().append("alarms", [])
    except SystemExit as exc:
        assert "canonical" in str(exc), str(exc)
    else:
        raise AssertionError("an empty append on a broken record was accepted")
    finally:
        reg.write_text(good)


def test_append_refuses_a_key_that_is_not_a_string():
    shutil.rmtree(DATA / "trials", ignore_errors=True)
    r = DATA / "trials"
    write_chain(r, [{"kind": "test", "n": i} for i in range(1, 4)])
    files = sorted(f for f in r.rglob("*") if f.is_file())
    before = [f.read_bytes() for f in files]
    for obj in (
        {"kind": "book_week", "by_week": {9: 0.1, 10: 0.2}},
        {"kind": "book_week", "weeks": [{"n": 1, "by_coin": {1.5: "x"}}]},
        {"kind": "book_week", 7: "seven"},
    ):
        try:
            T.ledger().append("alarms", [{"kind": "alarm"}, obj])
        except ValueError as exc:
            assert "string keys only" in str(exc), str(exc)
        else:
            raise AssertionError(f"a record with a key that is not a string was accepted: {obj}")
    assert sorted(f for f in r.rglob("*") if f.is_file()) == files and [f.read_bytes() for f in files] == before
    assert len(T.read()["registrations"]["records"]) == 3


def test_record_with_other_texts_reads_and_checks():
    """A registration line with fields that this code does not write reads and checks."""
    fresh()
    clock(at("2026-02-21"))
    register({"formula": PASSING, "source": None}, note="a note")
    rewrite_chain(
        "registrations", lambda objs: [{**objs[0], "extra_text": "a field that this code does not write", "x": [1]}]
    )
    rec = T.read()
    assert T.problems(rec) == [] and T.registered(rec)[0]["extra_text"]
    assert T.status_data(data=False)["tests"][0]["id"] == "FT-0001"


def test_write_boundary():
    fresh()
    clock(at("2026-02-21"))
    register({"formula": PASSING, "source": None})
    led = T.ledger()
    r = T.root()
    for bad in (r / "sub" / "x.json", r.parent / "x.json", r / "results" / "FT-0001" / "x.json"):
        refusal(lambda b=bad: led._guard(b), "write refused")
    refusal(lambda: led.write_file("ft-0001\n/x.json", b"{}"), "is not <id>/<file name>")
    os.link(r / "registrations.jsonl", r / "second_link")
    refusal(lambda: led.append("alarms", [{"kind": "alarm"}]), "not a plain file with one link")
    os.unlink(r / "second_link")
    (r / "values").mkdir(exist_ok=True)
    (r / "values").rename(r / "values_real")
    (r / "values").symlink_to(r / "values_real", target_is_directory=True)
    refusal(lambda: led._guard(r / "values" / "okx.jsonl"), "not a real folder")
    (r / "values").unlink()
    (r / "values_real").rename(r / "values")
    # a symlinked trial folder, to a folder in the data folder and to a folder outside it
    real = DATA / "trials_real"
    outside = TMP / "outside_trials"
    r.rename(real)
    try:
        shutil.copytree(real, outside)
        for target in (real, outside):
            r.symlink_to(target, target_is_directory=True)
            before = tree(target)
            refusal(T.read, "is a symlink")
            refusal(lambda: led.append("alarms", [{"kind": "alarm"}]), "is a symlink")
            refusal(lambda: led.write_file("ft-0001/x.json", b"{}"), "write refused")
            refusal(lambda: led._guard(r / "registrations.jsonl"), "write refused")
            assert tree(target) == before, target
            r.unlink()
    finally:
        if r.is_symlink():
            r.unlink()
        real.rename(r)
    assert len(T.registered(T.read())) == 1
    # a protected folder (the marker file and trials among the protected folders)
    saved = C.PROTECTED_DIRS
    (DATA / C.PROTECT_MARKER).write_text("protected\n")
    C.PROTECTED_DIRS = (*saved, "trials")
    before = tree(r)
    try:
        try:
            led.append("alarms", [{"kind": "alarm"}])
        except C.WriteRefused as exc:
            assert "refusing to write" in str(exc)
        else:
            raise AssertionError("a protected folder was written")
    finally:
        C.PROTECTED_DIRS = saved
        (DATA / C.PROTECT_MARKER).unlink()
    assert tree(r) == before


def test_inputs_changed_alarm_names_the_fields():
    fresh()
    clock(at("2026-02-21"))
    register({"formula": PASSING, "source": None})
    cut_to("2026-03-10", label_table=False)
    clock(at("2026-03-11"))
    assert T.log_week(log=QUIET) == 0
    # a restated volume: no test reads it, so no alarm
    vol = DATA / "raw" / "ohlcv" / "BTC-USDT.parquet"
    df = pd.read_parquet(vol)
    df.loc[df.index[-20], "volume"] *= 2.0
    df.to_parquet(vol, index=False)
    assert T.log_week(log=QUIET) == 0 and T.read()["alarms"]["records"] == []
    # a restated low of 2026-02-25 (a read field): inputs_changed names it
    p = DATA / "raw" / "ohlcv" / "DOT-USDT.parquet"
    df = pd.read_parquet(p)
    day = pd.to_datetime(df["ts"], utc=True) == pd.Timestamp("2026-02-25", tz="UTC")
    df.loc[day, "low"] *= 0.9
    df.to_parquet(p, index=False)
    assert T.log_week(log=QUIET) == T.EXIT_ALARM
    al = {a["type"]: a for a in T.read()["alarms"]["records"]}
    d = al["inputs_changed"]["detail"]
    assert d["changed_fields"] == ["low"] and d["sundays"] == ["2026-03-01", "2026-03-08"], d
    assert "value_changed" in al and "Changed fields: low" in T.alarm_text(al["inputs_changed"])


def test_value_changed_and_book_week_changed_alarms():
    fresh()
    clock(at("2026-02-21"))
    tid = register({"formula": PASSING, "source": None})
    cut_to("2026-03-20", label_table=False)
    clock(at("2026-03-21"))
    T.log_week(log=QUIET)
    frozen = weeks_of(tid)
    assert len(frozen) == 3
    saved = uf.full_values

    def other(expr, panels):
        return saved(expr, panels) * 1.5

    uf.full_values = other
    try:
        assert T.log_week(log=QUIET) == T.EXIT_ALARM
    finally:
        uf.full_values = saved
    al = [a for a in T.read()["alarms"]["records"] if a["type"] == "value_changed"]
    assert len(al) == 1 and al[0]["detail"]["tests"]["2026-02-22"] == [tid]
    # a restated close in week 2 changes the replay of a frozen week (found with the next new week)
    cut_to("2026-03-27", label_table=False)
    p = DATA / "raw" / "ohlcv" / "BTC-USDT.parquet"
    df = pd.read_parquet(p)
    day = pd.to_datetime(df["ts"], utc=True) == pd.Timestamp("2026-03-03", tz="UTC")
    df.loc[day, "close"] *= 1.2
    df.to_parquet(p, index=False)
    clock(at("2026-03-28"))
    assert T.log_week(log=QUIET) == T.EXIT_ALARM
    types = {a["type"] for a in T.read()["alarms"]["records"]}
    assert "book_week_changed" in types, types
    bw = next(a for a in T.read()["alarms"]["records"] if a["type"] == "book_week_changed")
    assert bw["test_id"] == tid and 2 in bw["detail"]["weeks"]
    wk = weeks_of(tid)
    assert len(wk) == 4 and wk[1]["book_ret"] == frozen[1]["book_ret"]  # the frozen number stays


def test_an_alarm_is_recorded_once_and_exit_3():
    fresh()
    clock(at("2026-02-21"))
    register({"formula": PASSING, "source": None})
    cut_to("2026-03-31", ends={"ETH-USDT": "2026-03-04"}, label_table=False)
    clock(at("2026-04-01"))
    assert T.log_week(log=QUIET) == T.EXIT_ALARM
    n = len(T.read()["alarms"]["records"])
    assert n == 1
    assert T.log_week(log=QUIET) == 0 and len(T.read()["alarms"]["records"]) == n
    clock(at("2026-04-02"))
    assert T.verify(log=QUIET) == 0


def test_code_drift_stops_log_week_until_recorded():
    fresh()
    clock(at("2026-02-21"))
    register({"formula": PASSING, "source": None})
    cut_to("2026-03-10", label_table=False)
    clock(at("2026-03-11"))
    saved = T.code_record

    def other():
        c = saved()
        files = {**c["files"], "engine/rules.py": "0" * 64}
        return {**c, "files": files, "sha256": "e" * 64}

    T.code_record = other
    try:
        refusal(lambda: T.log_week(log=QUIET), "Record the code change first")
        refusal(lambda: T.verify(log=QUIET), " in the trial ledger")
        assert T.read()["values_okx"]["records"] == []
        refusal(lambda: T.record_code_change("", log=QUIET), "needs a reason")
        assert T.record_code_change("a new rule file", ["first note"], log=QUIET) == 0
        rec = T.read()
        line = rec["registrations"]["records"][-1]
        assert line["kind"] == "code_change" and list(line["changed_files"]) == ["engine/rules.py"]
        assert line["notes"] == ["first note"] and line["config_changed"] is False
        assert T.log_week(log=QUIET) == 0 and len(T.read()["values_okx"]["records"]) == 3
        refusal(lambda: T.record_code_change("again", log=QUIET), "Nothing to record")
        assert T.problems(T.read()) == []
    finally:
        T.code_record = saved


def test_accept_release_records_a_version_change():
    fresh()
    clock(at("2026-02-21"))
    register({"formula": PASSING, "source": None})
    saved_code, saved_version = T.code_record, T.version.VERSION

    def other():
        c = saved_code()
        return {**c, "files": {**c["files"], "engine/book.py": "1" * 64}, "sha256": "d" * 64}

    T.code_record = other
    try:
        refusal(lambda: T.log_week(accept_release=True, log=QUIET), "Record the code change first")  # same VERSION
        T.version.VERSION = "9.9.9"
        assert T.log_week(accept_release=True, log=QUIET) == 0
        line = T.read()["registrations"]["records"][-1]
        assert line["kind"] == "code_change" and line["reason"] == "Factorlab 9.9.9" and line["version"] == "9.9.9"
    finally:
        T.code_record, T.version.VERSION = saved_code, saved_version


def test_withdrawal_rules():
    fresh()
    clock(at("2026-02-21", 10))
    register({"formula": PASSING, "source": None})
    refusal(lambda: T.withdraw("FT-0001", "  ", log=QUIET), "needs a reason")
    clock(at("2026-02-21", 12))
    assert T.withdraw("FT-0001", "a test withdrawal", log=QUIET) == 0
    refusal(lambda: T.withdraw("FT-0001", "again", log=QUIET), "already withdrawn")
    refusal(lambda: T.withdraw("FT-0099", "x", log=QUIET), "not a registered test")
    register({"formula": FORMULA, "source": None})
    clock(at(F, 1))
    refusal(lambda: T.withdraw("FT-0002", "too late", log=QUIET), "cannot be withdrawn now")
    rec = T.read()
    assert [r["kind"] for r in rec["registrations"]["records"]] == ["registration", "withdrawal", "registration"]
    assert T.problems(rec) == []
    d = T.status_data(data=False)
    assert [x["withdrawn"] for x in d["tests"]] == [True, False] and d["totals"]["n"] == 1
    # a withdrawn test gets no value line
    cut_to("2026-03-10", label_table=False)
    clock(at("2026-03-11"))
    T.log_week(log=QUIET)
    assert all("FT-0001" not in x["values"] for x in T.read()["values_okx"]["records"])
    # a withdrawal line on or after the first forward Sunday is a problem for verify
    rewrite_chain(
        "registrations",
        lambda objs: [objs[0], {**objs[1], "withdrawn_at": "2026-02-22T01:00:00Z"}, objs[2]],
    )
    assert any("withdrawn on or after its first forward Sunday" in p for p in T.problems(T.read()))


def test_expected_false_useful_counts_every_registration():
    sc = foresight_scenario()
    d = sc["status"]
    assert d["totals"]["n"] == 3 and d["totals"]["states"]["useful"] == 2 and d["totals"]["states"]["useless"] == 1
    e = d["expected_false_useful"]
    assert e["n"] == 3 and abs(e["value"] - 0.3) < 1e-12 and e["alpha"] == 0.10
    assert e["text"] == (
        'Expected number of false "useful" verdicts, for a false-pass rate of 10 % for each test: '
        "0.1 x N = 0.3 (N = 3 registered tests that are not withdrawn)."
    )
    assert T.expected_line(0)["value"] == 0.0


def test_verify_recomputes_and_finds_a_changed_line():
    fresh()
    clock(at("2026-02-21"))
    tid = register({"formula": PASSING, "source": None})
    cut_to("2026-03-31", label_table=False)
    clock(at("2026-04-01"))
    T.log_week(log=QUIET)
    msgs: list[str] = []
    assert T.verify(log=msgs.append) == 0 and any("No new alarm" in m for m in msgs)
    assert T.verify(write_results=True, log=QUIET) == 0
    assert (T.root() / "results" / tid.lower() / "forward_weekly.parquet").is_file()
    # a changed line with correct links: the content check finds it
    good = (T.root() / "weeks.jsonl").read_text()
    good_head = (T.root() / "head.json").read_text()
    rewrite_chain(
        "weeks", lambda objs: [{**o, "cs": {**o["cs"], "state": "useful"}} if o["week"] == 2 else o for o in objs]
    )
    refusal(lambda: T.verify(log=QUIET), " in the trial ledger")
    assert any("week 2" in p for p in T.problems(T.read()))
    # an edited byte breaks the chain
    (T.root() / "weeks.jsonl").write_text(good.replace('"week":2', '"week":3', 1))
    (T.root() / "head.json").write_text(good_head)
    refusal(lambda: T.verify(log=QUIET), "weeks.jsonl")
    (T.root() / "weeks.jsonl").write_text(good)
    assert T.verify(log=QUIET) == 0


def test_other_folder_is_not_read_or_changed():
    """A folder of the data folder that the ledger does not use, here with a JSON-lines file that is
    not a chain: register, log-week and verify do not read it and do not change it."""
    fresh()
    other = DATA / "notes"
    (other / "values").mkdir(parents=True)
    (other / "registrations.jsonl").write_text('{"kind":"batch","not":"a chain"}\n')
    before = tree(other)
    clock(at("2026-02-21"))
    register({"formula": PASSING, "source": None})
    cut_to("2026-03-10", label_table=False)
    clock(at("2026-03-11"))
    assert T.log_week(log=QUIET) == 0 and T.verify(log=QUIET) == 0
    assert T.status_data()["totals"]["registered"] == 1 and tree(other) == before


# --------------------------------------------------------------------------- 8. the command line


def test_cli_help_refusals_and_empty_status():
    empty = TMP / "empty_data"
    empty.mkdir(exist_ok=True)
    p = run_cli("--help", data=empty)
    assert p.returncode == 0 and "check" in p.stdout and "log-week" in p.stdout
    for sub in (
        "check",
        "register",
        "log-values",
        "log-week",
        "status",
        "verify",
        "withdraw",
        "record-code-change",
        "spec",
    ):
        p = run_cli(sub, "--help", data=empty)
        assert p.returncode == 0 and "usage:" in p.stdout, (sub, p.stderr)
    p = run_cli("status", data=empty)
    assert p.returncode == 0 and "No test is registered." in p.stdout, p.stdout
    assert "Expected number of false" in p.stdout and rules.RULE_TEXT[0] in p.stdout
    p = run_cli("check", "--formula", "delay(close, -1)", data=empty)
    assert p.returncode == 1 and "argument 2 must be an integer from 1 to 1000" in p.stderr, p.stderr
    p = run_cli("check", "--formula", "close", "--spec", "nothing.toml", data=empty)
    assert p.returncode == 1 and "is not a file" in p.stderr, p.stderr
    p = run_cli("check", data=empty)
    assert p.returncode == 2
    p = run_cli("spec", "long_only", "--json", data=empty)
    assert p.returncode == 0 and json.loads(p.stdout)["sha256"] == book.load_spec("long_only").sha256
    assert sorted(x.name for x in empty.iterdir() if x.name != "raw") in ([], ["backtests", "factors"])
    assert not (empty / "trials").exists()
    # check and register on the fixture data through the command line
    fresh()
    p = run_cli("check", "--formula", PASSING)
    assert (
        p.returncode == 0
        and "Historical test, 2022-04-03" in p.stdout
        and "Result: unproven (historical pass)" in p.stdout
    )
    assert (
        "Register (run from the repository root): cd engine && .venv/bin/python -B judge.py register --formula="
        in p.stdout
    )
    # a formula that starts with "-" and has no space: the printed register command works
    import judge

    sp = spec_file('format = 1\nname = "mirror"\nversion = 1\n[book]\nsign = -1\n')
    p = run_cli("check", "--formula=-ts_mean(clv,7)", f"--spec={sp}")
    assert p.returncode == 0 and "Result: unproven (historical pass)" in p.stdout, (p.stdout, p.stderr)
    prefix = "Register (run from the repository root): cd engine && .venv/bin/python -B judge.py "
    line = next(x for x in p.stdout.splitlines() if x.startswith(prefix))
    assert "--formula='-ts_mean(clv,7)'" in line, line
    args = shlex.split(line[len(prefix) :])
    assert judge.parser().parse_args(args).formula == "-ts_mean(clv,7)"
    p = run_cli(*args, "--dry-run")
    assert p.returncode == 0 and "Dry run: nothing is written." in p.stdout, (p.stdout, p.stderr)
    # a values file without a data row is refused with a text, not a traceback
    p = run_cli("check", "--values", str(values_csv({})))
    assert p.returncode == 1 and "the file has no data row" in p.stderr and "Traceback" not in p.stderr, p.stderr
    p = run_cli("register", "--formula", PASSING, "--name", "clv 7")
    assert p.returncode == 1 and "the name must have" in p.stderr
    p = run_cli("register", "--formula", PASSING, "--name", "clv7", "--dry-run")
    assert p.returncode == 0 and "Dry run: nothing is written." in p.stdout and not (DATA / "trials").exists()


def test_check_text_names_the_pit_status():
    """check shows "proven" only for a proven formula. A formula that is not proven gets no
    register command, also when its history passes."""
    import contextlib

    import judge

    fresh()
    p = run_cli("check", "--formula", "ts_mean(ts_mean(close, 1000), 1000)")
    assert p.returncode == 0, p.stderr
    assert "Point in time: not proven: the formula has no value at any Sunday." in p.stdout, p.stdout
    assert "Point in time: proven" not in p.stdout and "Register (" not in p.stdout, p.stdout
    r = T.check({"formula": PASSING, "source": None}, book.load_spec(), now=at("2026-02-21"))
    a = judge.parser().parse_args(["check", "--formula", PASSING])

    def text(result: dict) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            judge.print_check(result, a)
        return out.getvalue()

    good = text(r)
    assert f"Point in time: proven at each Sunday cut ({T.count_text(r['pit']['sundays'], 'Sunday')})." in good
    one = text({**r, "pit": {**r["pit"], "sundays": 1}, "historical": {**r["historical"], "weeks": 1, "held_weeks": 1}})
    assert "proven at each Sunday cut (1 Sunday)." in one and "(1 week, 1 with a factor position)" in one, one
    assert "Register (run from the repository root)" in good, good
    fail = {"status": "fail", "sundays": r["pit"]["sundays"], "first_diff_sunday": "2025-01-05"}
    bad = text({**r, "pit": {**r["pit"], "status": "fail"}, "pit_detail": {**r["pit_detail"], **fail}})
    want = "Point in time: not proven: the value at the Sunday cut 2025-01-05 is not equal to the full computation."
    assert want in bad and "Register (" not in bad and "Result: unproven (historical pass)" in bad, bad
    # a historically "useless" test can be registered: the command and the verdict note are shown
    useless = text({**r, "historical": {**r["historical"], "result": "useless"}})
    assert "Register (run from the repository root)" in useless, useless
    assert 'Its verdict stays "useless (historical)" until the forward test decides.' in useless, useless
    none = text({**r, "historical": {**r["historical"], "result": "none"}})
    assert "Register (" not in none, none


def test_printed_register_command_has_absolute_paths():
    """check in another folder with relative --values and --spec paths: the printed register
    command names the same files after its "cd engine"."""
    fresh()
    r = T.check({"formula": PASSING, "source": None}, book.load_spec(), now=at("2026-02-21"))
    work = Path(tempfile.mkdtemp(prefix="work_", dir=TMP))
    v = r["_values"]
    lines = ["date,symbol,value"]
    for d, row in zip(v.index, v.to_numpy()):
        lines += [f"{d.date()},{c},{float(x)!r}" for c, x in zip(SYMS, row) if np.isfinite(x)]
    (work / "mine.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(book.SPEC_DIR / "long_short.toml", work / "mine.toml")
    p = run_cli("check", "--values", "mine.csv", "--spec", "mine.toml", cwd=work)
    assert p.returncode == 0 and "Result: unproven (historical pass)" in p.stdout, (p.stdout, p.stderr)
    prefix = "Register (run from the repository root): cd engine && .venv/bin/python -B judge.py "
    line = next(x for x in p.stdout.splitlines() if x.startswith(prefix))
    args = shlex.split(line[len(prefix) :])
    import judge

    ns = judge.parser().parse_args(args)
    assert ns.values == (work / "mine.csv").resolve() and Path(ns.spec) == (work / "mine.toml").resolve(), args
    p = run_cli(*args, "--dry-run")  # from engine/, as the printed command runs
    assert p.returncode == 0 and "Dry run: nothing is written." in p.stdout, (p.stdout, p.stderr)
    # the name of a shipped specification stays a name
    shipped = judge._options(judge.parser().parse_args(["check", "--formula", PASSING, "--spec", "long_only"]))
    assert shipped == f"--formula={shlex.quote(PASSING)} --spec=long_only", shipped


def test_imports():
    """judge.py, trials.py and userfactor.py load no exchange client, no download code and no find tool."""
    code = (
        "import sys, judge, trials, userfactor\n"
        "bad = ['ccxt', 'ingest', 'factor_lab', 'alphaforge_bridge']\n"
        "print([m for m in bad if m in sys.modules])\n"
    )
    p = subprocess.run([sys.executable, "-B", "-c", code], cwd=ENGINE, capture_output=True, text=True, timeout=300,
                       env={**os.environ, "FACTORLAB_DATA": str(TMP / "empty_data")})  # fmt: skip
    assert p.returncode == 0 and p.stdout.strip() == "[]", (p.stdout, p.stderr[-1000:])
    for name in ("judge.py", "trials.py", "userfactor.py"):
        src = (ENGINE / name).read_text()
        for word in ("FACTORLAB_OFFLINE", "g_labels(", "factor_stats("):
            assert word not in src, (name, word)


def test_ledger_texts_are_short_sentences():
    for text in (T.LOG_RULE, T.WEEK_RULE, T.PIT_FILE_RULE, uf.PIT_RULE, T.DRIFT_TEXT):
        for sentence in re.split(r"(?<=\.)\s+", text):
            assert len(sentence.split()) <= 25, sentence
        assert "pending" not in text and "judging" not in text, text


# --------------------------------------------------------------------------- 9. end to end


def test_end_to_end_run_writes_only_trials():
    """check, register (a formula and a values file), 30 forward weeks with a replaced clock and
    bars that end the day before it, a values file logged on Sunday 23:00 and later, a coin whose
    bars end, verify exit 0, a broken line found. Only data/trials changes."""
    fresh()
    clock(at("2026-02-21"))

    def judged(fn):
        before = tree(DATA, skip="trials")
        out = fn()
        assert tree(DATA, skip="trials") == before, "a judge command wrote outside data/trials"
        return out

    judged(lambda: T.check({"formula": PASSING, "source": None}, book.load_spec()))
    a = judged(lambda: register({"formula": PASSING, "source": None}))
    v = judged(lambda: register({"values_file": values_csv(history("2025-01-05", "2026-02-15", lambda s: random_values(s, 9)))},
                                book.load_spec("long_only")))  # fmt: skip
    ends = {"ETH-USDT": "2026-05-13"}
    for k in range(30):
        s = F + timedelta(days=7 * k)
        cut_to(s - timedelta(days=1), ends=ends, label_table=False)
        when = at(s, 23) if k % 3 else at(s + timedelta(days=1), 10)  # on Sunday 23:00, or on Monday 10:00
        clock(when)
        judged(lambda s=s: T.log_values(v, values_csv({str(s): random_values(s, 9)}), log=QUIET))
        if k % 5 == 4:
            cut_to(s, ends=ends, label_table=False)
            clock(at(s + timedelta(days=1), 10))
            judged(lambda: T.log_week(log=QUIET))
    cut_to(F + timedelta(days=7 * 30), ends=ends, label_table=False)
    clock(at(F + timedelta(days=7 * 30 + 1), 10))
    judged(lambda: T.log_week(log=QUIET))
    assert len(weeks_of(a)) == 30 and len(weeks_of(v)) == 30
    gaps = [x["detail"] for x in T.read()["alarms"]["records"] if x["type"] == "data_gap"]
    assert gaps == [{"coin": "ETH-USDT", "field": "close", "first_day": "2026-05-14"}], gaps
    assert judged(lambda: T.verify(log=QUIET)) == 0
    d = judged(T.status_data)
    assert [x["forward"]["weeks"] for x in d["tests"]] == [30, 30]
    # a broken line is found
    p = T.root() / "registrations.jsonl"
    good = p.read_bytes()
    p.write_bytes(good.replace(b'"name":', b'"name" :', 1))
    refusal(lambda: T.verify(log=QUIET), "registrations.jsonl line 1 is not a canonical JSON object")
    p.write_bytes(good)


# --------------------------------------------------------------------------- 10. data edge cases


def test_a_day_without_any_bar():
    """A day on which no coin has a bar: the week completes after 7 days with an alarm (no error),
    and the history of check runs over such a day."""
    gone = {("ohlcv", c): ["2026-03-01"] for c in SYMS}
    fresh()
    clock(at("2026-02-21"))
    tid = register({"formula": PASSING, "source": None})
    cut_to("2026-03-20", drop=gone, label_table=False)
    clock(at("2026-03-21"))
    assert T.log_week(log=QUIET) == T.EXIT_ALARM
    wk = weeks_of(tid)
    assert [w["sunday"] for w in wk] == ["2026-02-22", "2026-03-01", "2026-03-08"], [w["sunday"] for w in wk]
    assert sorted(g[0] for g in wk[0]["final_gaps"]) == sorted(SYMS)
    gaps = [a["detail"] for a in T.read()["alarms"]["records"] if a["type"] == "data_gap"]
    assert sorted(g["coin"] for g in gaps) == sorted(SYMS), gaps
    assert {(g["field"], g["first_day"]) for g in gaps} == {("close", "2026-03-01")}, gaps
    assert T.log_week(log=QUIET) == 0 and len(weeks_of(tid)) == 3  # the alarms are recorded once
    clock(at("2026-03-22"))
    assert T.verify(log=QUIET) == 0
    # the history: a day without any bar inside the history span
    fresh(drop={("ohlcv", c): ["2025-06-01"] for c in SYMS})
    r = T.check({"formula": PASSING, "source": None}, book.load_spec(), now=at("2026-02-21"))
    assert r["historical"]["result"] in ("pass", "useless") and r["historical"]["start"] < "2025-06-01"
    assert pd.Timestamp("2025-06-01", tz="UTC") in r["_weekly"]["sunday"].tolist()


def test_btc_without_a_first_close_is_unknown():
    """The engine scales the BTC hold by its first close. Without the BTC close of the first day,
    the BTC numbers are unknown (None), never 0.0: in the history of check and in the weeks."""
    fresh()
    r = T.check({"formula": PASSING, "source": None}, book.load_spec(), now=at("2026-02-21"))
    s0 = r["historical"]["start"]
    assert r["historical"]["btc"]["sharpe"] is not None and np.isfinite(r["_weekly"]["btc_ret"]).all()
    fresh(drop={("ohlcv", "BTC-USDT"): [s0]})
    r = T.check({"formula": PASSING, "source": None}, book.load_spec(), now=at("2026-02-21"))
    assert r["historical"]["start"] == s0, r["historical"]["start"]
    assert r["historical"]["btc"] == {"total_return": None, "sharpe": None, "max_dd": None}, r["historical"]["btc"]
    assert r["stats"]["btc"]["ann_vol"] is None and r["_weekly"]["btc_ret"].isna().all()
    # the forward weeks: no BTC close on the first forward Sunday
    fresh()
    clock(at("2026-02-21"))
    tid = register({"formula": PASSING, "source": None})
    cut_to("2026-03-20", drop={("ohlcv", "BTC-USDT"): [str(F)]}, label_table=False)
    clock(at("2026-03-21"))
    assert T.log_week(log=QUIET) == T.EXIT_ALARM  # the data_gap alarm of the BTC bar
    wk = weeks_of(tid)
    assert len(wk) >= 3 and all(w["btc_ret"] is None for w in wk), [w["btc_ret"] for w in wk]
    table = pd.read_parquet(T.root() / "results" / tid.lower() / "forward_weekly.parquet")
    assert table["btc_ret"].isna().all(), table["btc_ret"]
    assert T.verify(log=QUIET) == 0


def test_excess_vol_of_the_zero_control_is_the_book_vol():
    """With the control "zero", the daily excess return is the daily book return. Thus the two
    annual volatilities are equal, also for a history of one week."""
    fresh()
    r = T.check({"formula": PASSING, "source": None}, book.load_spec(), now=at("2026-02-21"))
    st = r["stats"]
    assert abs(st["excess"]["ann_vol"] - st["book"]["ann_vol"]) <= 1e-12 * st["book"]["ann_vol"], st
    s0 = r["historical"]["start"]
    one = r["_values"].loc[[pd.Timestamp(s0, tz="UTC")]]
    r = T.check({"values_file": values_csv({s0: one.iloc[0].to_numpy()})}, book.load_spec(), now=at("2026-02-21"))
    st = r["stats"]
    assert r["historical"]["weeks"] == 1 and st["book"]["ann_vol"] is not None, r["historical"]
    assert abs(st["excess"]["ann_vol"] - st["book"]["ann_vol"]) <= 1e-12 * st["book"]["ann_vol"], st


def test_write_results_gives_the_files_of_the_registration():
    fresh()
    clock(at("2026-02-21"))
    a = register({"formula": PASSING, "source": None})
    v = register({"values_file": values_csv(history("2025-01-05", "2026-02-15", lambda s: random_values(s, 4)))},
                 book.load_spec("long_only"))  # fmt: skip
    names = ("historical.json", "historical_daily.parquet", "historical_weekly.parquet")

    def files(tid: str) -> dict:
        d = T.root() / "results" / tid.lower()
        return {n: (d / n).read_bytes() for n in names}

    before = {tid: files(tid) for tid in (a, v)}
    for tid in (a, v):
        doc = json.loads(before[tid]["historical.json"])
        assert set(doc["stats"]) == {"book", "control", "excess", "btc"}, doc["stats"]
        assert doc["factor_numbers"]["weeks"] == doc["historical"]["weeks"] > 0, doc["factor_numbers"]
    cut_to("2026-03-10", label_table=False)
    clock(at("2026-03-11"))
    T.log_week(log=QUIET)
    for p in (T.root() / "results").rglob("historical*"):
        p.unlink()
    assert T.verify(write_results=True, log=QUIET) == 0
    for tid in (a, v):
        after = files(tid)
        assert after["historical.json"] == before[tid]["historical.json"], tid
        for n in names[1:]:
            pd.testing.assert_frame_equal(
                pd.read_parquet(io.BytesIO(after[n])), pd.read_parquet(io.BytesIO(before[tid][n]))
            )


# --- runner -------------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        saved_clock = LG.CLOCK
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001 - report every test, then fail at the end
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        finally:
            LG.CLOCK = saved_clock
    print(f"\n{len(tests) - failed}/{len(tests)} trial tests passed")
    raise SystemExit(1 if failed else 0)
