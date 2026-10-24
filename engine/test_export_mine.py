"""Tests for the section "mine" of the web export (webexport/mine.py, pages/mine.js, pages/factor.js).

Script-style like the other engine tests:  python -B test_export_mine.py

The suite reads no data folder of the repository. At the start it sets FACTORLAB_DATA to a new
folder under the git-ignored engine/.tmp/ and removes that folder at the end. It writes the
synthetic demo market there and builds a trial ledger with judge's own procedures (trials.py)
and a replaced clock: four registrations (a values file, two formula tests on OKX data that are
related to each other, and a values file on the long-only book that is withdrawn), 26 frozen
weeks of the values-file test (a decided test), and a weekly log of the formula tests over 9
weeks, in which a late funding value holds one test and a final gap gives a data_gap alarm. The
fixture lives in the temporary folder only; nothing in the repository holds a judged factor.

Each number that the pages show is computed again here from the chain files and the result
files with plain pandas and numpy (not through the builder or rules.py):

  rows        each row of the payload from the registration lines: id, name, input, source,
              book, the historical numbers, the forward week, the bounds, the state, the verdict,
              the related tests, the alarms; the held test; the counts
  historical  the total return, the Sharpe ratio, the 1-year windows and their share, the weekly
              marks and the statistics of the book, the control, BTC hold and the net excess, from
              historical_daily.parquet
  forward     the bounds and the states of the confidence sequence of each frozen week, written as
              a plain loop; the equity marks; the week-26 and week-156 dates
  checks      a broken ledger (the problems, no row); result files that disagree with the ledger
              (the export fails); absent result files (an empty historical block); result files
              that judge.py wrote again after a restatement of the market data (a close inside a
              frozen forward week, a close of the history): the export works, the row and the part
              get "results_differ", and the rows and the forward path still come from the ledger
  files       the part names, the size of each part (at most 200,000 bytes, outside the total),
              the privacy guard
  page        the file rules of mine.js and factor.js, and node renders of the page with rows,
              empty and with a broken ledger, the page of each test (also a card route), of a test
              with a broken ledger and of an unknown id (needs node on PATH, NODE or
              FACTORLAB_NODE; skipped otherwise)
"""

from __future__ import annotations

import ast
import atexit
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
REPO = ENGINE.parent
(ENGINE / ".tmp").mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="test_export_mine_", dir=ENGINE / ".tmp"))
atexit.register(shutil.rmtree, TMP, True)
os.environ["FACTORLAB_DATA"] = str(TMP / "data")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import book  # noqa: E402
import config as C  # noqa: E402
import demo_data  # noqa: E402
import export_web as E  # noqa: E402
import labels  # noqa: E402
import ledger as LG  # noqa: E402
import rules  # noqa: E402
import trials as T  # noqa: E402
import userfactor as uf  # noqa: E402
from webexport import common as WC  # noqa: E402
from webexport import mine as M  # noqa: E402

DATA = Path(C.DATA)
WEB = REPO / "web"
MASTER = TMP / "master"
SYMS = [C.sym_to_file(s) for s in C.UNIVERSE]
CUT = "2026-02-20"  # the bars of the registrations end on this Friday
F = date(2026, 2, 22)  # the first forward Sunday of a registration on Saturday 2026-02-21
LOG_DAY = "2026-04-26"  # the bars of the weekly log end on this Sunday
QUIET = lambda *a, **k: None  # noqa: E731
FB = f"funding_{C.FUNDING_FALLBACK_EXCHANGE_ID}"
# the words that a page must not hold as its own text (the verdict words come from the data)
VERDICT_WORD = re.compile(r"\b(useful|useless|unproven)\b", re.I)
_STATE: dict = {}


# --------------------------------------------------------------------------- the fixture


def _frames() -> dict:
    if "frames" not in _STATE:
        demo_data.write_raw(MASTER, demo_data.market(demo_data.SEED))
        frames = {}
        for p in sorted((MASTER / "raw").rglob("*.parquet")):
            df = pd.read_parquet(p)
            frames[p.relative_to(MASTER)] = (df, pd.to_datetime(df["ts"], utc=True).dt.floor("D"))
        _STATE["frames"] = frames
    return _STATE["frames"]


def cut_to(day, drop: dict | None = None) -> None:
    """The raw bars up to `day` in the data folder; drop {(folder, coin): [days]} removes those rows."""
    lim = pd.Timestamp(str(day), tz="UTC")
    for rel, (df, days) in _frames().items():
        keep = days <= lim
        for (folder, coin), gone in (drop or {}).items():
            if coin == rel.stem and rel.parent.name == folder:
                keep &= ~days.isin([pd.Timestamp(str(g), tz="UTC") for g in gone])
        out = DATA / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        df[keep.to_numpy()].to_parquet(out, index=False)
    with contextlib.redirect_stdout(io.StringIO()):
        labels.build()


def at(day, hh: int = 10) -> datetime:
    d = LG.as_day(day)
    return datetime(d.year, d.month, d.day, hh, tzinfo=UTC)


def clock(t: datetime) -> None:
    LG.CLOCK = lambda: t


def values_csv(seed: int, name: str) -> Path:
    """A CSV values file of random values on the Sundays 2022-04-03 .. 2026-02-15."""
    rng = np.random.default_rng(seed)
    lines = ["date,symbol,value"]
    for d in pd.date_range("2022-04-03", "2026-02-15", freq="7D"):
        for s, v in zip(SYMS, rng.normal(0.0, 1.0, len(SYMS))):
            lines.append(f"{d.date()},{s},{float(v)!r}")
    p = TMP / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def build_fixture() -> None:
    """The trial ledger of the module docstring, in DATA (once)."""
    if _STATE.get("built"):
        return
    shutil.rmtree(DATA / "trials", ignore_errors=True)
    cut_to(CUT)
    clock(at("2026-02-21"))
    ls, lo = book.load_spec("long_short"), book.load_spec("long_only")
    maker = TMP / "maker.toml"
    maker.write_text(
        'format = 1\nname = "long-short-maker"\nversion = 1\n[costs]\nscenario = "maker"\n', encoding="utf-8"
    )
    reg = [
        ({"values_file": values_csv(1, "a.csv")}, ls, "values-a", None),
        ({"formula": "ts_mean(clv, 7)", "source": None}, ls, None, None),
        (
            {"formula": "ts_mean(clv, 7) + 0 * ts_mean(funding, 3)", "source": None},
            book.load_spec(maker),
            "clv-funding",
            "a note",
        ),
        ({"values_file": values_csv(2, "b.csv")}, lo, "values-b", None),
    ]
    for inp, spec, name, note in reg:
        assert T.register(inp, spec, name=name, note=note, log=QUIET) == 0
    assert T.withdraw("FT-0004", "a withdrawal", log=QUIET) == 0
    # 26 frozen weeks of FT-0001 with a positive mean: the confidence sequence decides at week 26
    rec = T.read()
    t1 = next(t for t in T.registered(rec) if t["test_id"] == "FT-0001")
    xs = list(0.05 + np.random.default_rng(7).normal(0.0, 0.01, 26))
    fwd = rules.forward_test(xs, float(t1["sigma_floor"]))
    assert (fwd["state"], fwd["decided_week"]) == ("useful", 26), fwd["state"]
    lines = []
    for i, (x, p) in enumerate(zip(xs, fwd["path"]), start=1):
        s = F + timedelta(days=7 * (i - 1))
        lines.append(
            {
                "kind": "book_week",
                "test_id": "FT-0001",
                "week": i,
                "sunday": str(s),
                "end": str(s + timedelta(days=7)),
                "logged_at": "2026-08-24T10:00:00Z",
                "book_ret": float(x),
                "control_ret": 0.0,
                "excess": float(x),
                "btc_ret": 0.01,
                "book": {
                    "cost": 1.0,
                    "funding_usd": 0.0,
                    "turnover": 0.1,
                    "gross": 1.0,
                    "n_long": 4,
                    "n_short": 4,
                    "trade_date": None,
                },
                "control": None,
                "values_ref": None,
                "held": True,
                "final_gaps": [],
                "inputs": {"prices_sha256": "0" * 64, "funding_sha256": "0" * 64},
                "code_sha256": T.code_record()["sha256"],
                "registrations": {
                    "lines": len(rec["registrations"]["records"]),
                    "sha256": rec["registrations"]["shas"][-1],
                },
                "cs": T._cs_record(p),
            }
        )
    T.ledger().append("weeks", lines)
    # the weekly log: BTC funding of the last Sunday is late (it holds FT-0003, which reads
    # funding), and ETH funding of 2026-03-10 never comes (a final gap: a data_gap alarm)
    cut_to(
        LOG_DAY,
        drop={
            ("funding", "BTC-USDT"): [LOG_DAY],
            (FB, "BTC-USDT"): [LOG_DAY],
            ("funding", "ETH-USDT"): ["2026-03-10"],
            (FB, "ETH-USDT"): ["2026-03-10"],
        },
    )
    clock(at("2026-04-27"))
    assert T.log_week(log=QUIET) == T.EXIT_ALARM
    assert T.problems(T.read()) == []
    _STATE["built"] = True


@contextlib.contextmanager
def ledger_variant():
    """Change the ledger or its result files inside the block; the fixture comes back after it."""
    build_fixture()
    backup = TMP / "trials_backup"
    shutil.rmtree(backup, ignore_errors=True)
    shutil.copytree(DATA / "trials", backup)
    try:
        yield DATA / "trials"
    finally:
        shutil.rmtree(DATA / "trials")
        shutil.copytree(backup, DATA / "trials")


def built() -> dict:
    """The payload of the fixture (as export_web writes it) and its parts."""
    if "payload" not in _STATE:
        build_fixture()
        b = dict(M.build())
        parts = b.pop(WC.PARTS_KEY, {}) or {}
        b["parts"] = sorted(parts)
        _STATE["payload"] = json.loads(WC.dumps(b))
        _STATE["parts"] = {k: json.loads(WC.dumps(v)) for k, v in parts.items()}
    return _STATE["payload"]


def part(name: str) -> dict:
    built()
    return _STATE["parts"][name]


def chain(rel: str) -> list[dict]:
    """The lines of a chain file, read with json only."""
    p = DATA / "trials" / rel
    return [json.loads(x) for x in p.read_text(encoding="ascii").splitlines()] if p.is_file() else []


def close(a, b, tol=1e-9, what=""):
    if a is None or b is None:
        assert a is None and b is None, (what, a, b)
        return
    assert abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b))), (what, a, b)


def sig6(a, b, what=""):
    """a equals b within the 6 significant digits of a ledger number."""
    if a is None or b is None:
        assert a is None and b is None, (what, a, b)
        return
    assert abs(float(a) - float(b)) <= 5e-6 * max(abs(float(a)), abs(float(b))) + 1e-12, (what, a, b)


def tree(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------- independent statistics


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(365))


def daily_returns(eq: np.ndarray) -> np.ndarray:
    eq = np.asarray(eq, dtype=float)
    out = np.empty(len(eq))
    out[0] = np.nan
    for i in range(1, len(eq)):
        out[i] = 0.0 if eq[i - 1] <= 0 else eq[i] / eq[i - 1] - 1.0
    return out


def historical_from_daily(daily: pd.DataFrame) -> dict:
    """The historical numbers from the daily marks, written out here."""
    dates = pd.DatetimeIndex(pd.to_datetime(daily["date"], utc=True))
    b = daily_returns(daily["book_equity"].to_numpy())
    c = daily["control_equity"].to_numpy(dtype=float)
    d = b - daily_returns(c) if np.isfinite(c).any() else b
    d = d[1:]
    x = np.concatenate([[1.0], np.cumprod(1.0 + d)])
    xs = pd.Series(x, index=dates)
    wins = []
    s, end = dates[0], dates[-1]
    while s + pd.Timedelta(days=365) <= end:
        e = s + pd.Timedelta(days=365)
        seg = xs[(xs.index >= s) & (xs.index <= e)].to_numpy()
        wins.append((str(s.date()), str(e.date()), sharpe(seg[1:] / seg[:-1] - 1.0)))
        s += pd.Timedelta(days=91)
    npos = sum(1 for w in wins if w[2] > 0)
    return {
        "d": d,
        "x": xs,
        "total_return": float(x[-1] - 1.0),
        "sharpe": sharpe(d),
        "windows": wins,
        "n_windows": len(wins),
        "n_positive": npos,
        "share_positive": npos / len(wins) if wins else None,
        "max_dd": float(np.min(x / np.maximum.accumulate(x) - 1.0)),
    }


def equity_stats(eq: np.ndarray) -> dict:
    eq = np.asarray(eq, dtype=float)
    r = eq[1:] / eq[:-1] - 1.0
    return {
        "total_return": float(eq[-1] / eq[0] - 1.0),
        "sharpe": sharpe(r),
        "ann_vol": float(np.std(r, ddof=1) * math.sqrt(365)),
        "max_dd": float(np.min(eq / np.maximum.accumulate(eq) - 1.0)),
    }


def mixture_r(alpha: float = 0.10, t_star: int = 26) -> float:
    from scipy.optimize import brentq

    return float(brentq(lambda r: math.log((t_star + r) / (r * alpha * alpha)) - t_star / r, 1e-6, 1e6))


def plain_band(xs: list[float], floor: float, alpha: float = 0.10) -> list[dict]:
    """The confidence sequence of the rule, as a plain loop: mean, the sample SD and the Bartlett
    long-run SD (4 lags, divisor t - 1), sigma = the largest of the two and the floor, the bounds,
    the states (from week 26: useful when lower > 0; useless when upper < 0 or at week 156)."""
    r = mixture_r(alpha)
    out, decided = [], None
    for t in range(1, len(xs) + 1):
        x = np.asarray(xs[:t], dtype=float)
        mean = float(x.mean())
        if t == 1:
            sd = lr = float("nan")
            sigma = floor
        else:
            dev = x - mean
            g = [float(np.sum(dev[k:] * dev[: t - k]) / (t - 1)) for k in range(0, min(4, t - 1) + 1)]
            sd = math.sqrt(max(g[0], 0.0))
            lr = math.sqrt(max(g[0] + 2 * sum((1 - k / 5) * g[k] for k in range(1, len(g))), 0.0))
            sigma = max(sd, lr, floor)
        half = sigma * math.sqrt((t + r) * math.log((t + r) / (r * alpha * alpha))) / t
        lower, upper = mean - half, mean + half
        if decided is None:
            if t >= 26 and lower > 0:
                decided = "useful"
            elif t >= 26 and (upper < 0 or t >= 156):
                decided = "useless"
        out.append({"mean": mean, "sigma": sigma, "lower": lower, "upper": upper, "state": decided or "running"})
    return out


# --------------------------------------------------------------------------- tests


def test_no_ledger_is_an_empty_state():
    """A data folder without a trial ledger (every demo build): the section is available, the
    ledger is absent, no row, no part, N = 0, the rule text, the two shipped book specifications
    and the commands."""
    if not (DATA / "raw").exists():
        cut_to(CUT)
    saved = None
    if (DATA / "trials").exists():
        saved = TMP / "trials_saved"
        shutil.rmtree(saved, ignore_errors=True)
        shutil.move(str(DATA / "trials"), saved)
    try:
        p = json.loads(WC.dumps(M.build()))
    finally:
        if saved is not None:
            shutil.move(str(saved), DATA / "trials")
    assert p["available"] is True and p["schema"] == 1 and p["as_of"] is None
    assert p["ledger"]["present"] is False and p["ledger"]["ok"] is True and p["ledger"]["problems"] == []
    assert p["rows"] == [] and WC.PARTS_KEY not in p and "parts" not in p and p["alarms"] == []
    assert p["counts"] == {
        "registered": 0,
        "withdrawn": 0,
        "n": 0,
        "states": {"waiting": 0, "running": 0, "useful": 0, "useless": 0},
        "verdicts": {"useful": 0, "useless": 0, "unproven": 0},
        "inputs": {"formula": 0, "values_file": 0},
    }
    assert p["expected_false_useful"]["n"] == 0 and p["expected_false_useful"]["value"] == 0.0
    rec = rules.rule_record()
    assert {k: p["rule"][k] for k in rec} == json.loads(WC.dumps(rec)) and p["rule"]["text"] == list(rules.RULE_TEXT)
    specs = {s["file"]: s for s in p["specs"]}
    assert list(specs) == ["engine/bookspecs/long_short.toml", "engine/bookspecs/long_only.toml"]
    for name, s in zip(("long_short", "long_only"), p["specs"]):
        sp = book.load_spec(name)
        assert (s["sha256"], s["book_key"], s["body"]) == (sp.sha256, sp.book_key, json.loads(WC.dumps(sp.body)))
    assert p["commands"]["where"] == "Run these commands from the repository root."
    assert p["commands"]["check_formula"] == 'cd engine && .venv/bin/python -B judge.py check --formula="<formula>"'
    assert (
        p["commands"]["register_values"]
        == "cd engine && .venv/bin/python -B judge.py register --values <file.csv> --name <name>"
    )
    assert (
        p["commands"]["log_values"]
        == "cd engine && .venv/bin/python -B judge.py log-values <FT id> --values <file.csv>"
    )
    assert p["commands"]["spec"] == "cd engine && .venv/bin/python -B judge.py spec long_only"
    assert p["notes"] == uf.NOTES


def test_build_is_read_only_and_deterministic():
    build_fixture()
    before = tree(DATA)
    a = json.loads(WC.dumps(M.build()))
    b = json.loads(WC.dumps(M.build()))
    assert tree(DATA) == before, "build() changed a file of the data folder"
    assert a == b, "build() is not deterministic"


def test_rows_recomputed_from_the_ledger_and_the_results():
    """Each row from the registration lines (read with json only): the identity, the book, the
    historical numbers, the forward week, the bounds and state of the last frozen week, the
    related tests (from the "related" field of each line, both ways), the held test, the alarms
    and the counts."""
    p = built()
    regs = [x for x in chain("registrations.jsonl") if x["kind"] == "registration"]
    wd = {x["test_id"] for x in chain("registrations.jsonl") if x["kind"] == "withdrawal"}
    weeks = chain("weeks.jsonl")
    alarms = chain("alarms.jsonl")
    assert [r["id"] for r in p["rows"]] == [x["test_id"] for x in regs] == ["FT-0001", "FT-0002", "FT-0003", "FT-0004"]
    rel: dict[str, set] = {x["test_id"]: set() for x in regs}
    for x in regs:
        for o in x["related"]:
            rel[x["test_id"]].add(o)
            rel[o].add(x["test_id"])
    assert rel["FT-0002"] == {"FT-0003"} and rel["FT-0003"] == {"FT-0002"}, rel
    words = {"pass": "unproven (historical pass)", "useless": "useless (historical)"}
    for row, x in zip(p["rows"], regs):
        tid = x["test_id"]
        f, body = x["factor"], x["book_spec"]["body"]
        assert row["name"] == x["name"] and row["input"] == x["input"] and row["registered_at"] == x["registered_at"]
        assert row["source"] == f.get("source") and row["library_id"] == f.get("library_id")
        assert row["expression"] == f.get("expression") and row["pit"] == x["pit"]["status"]
        assert row["file"] == (
            {"name": f["file_name"], "sha256": f["file_sha256"]} if x["input"] == "values_file" else None
        )
        assert row["notes"] == list(f.get("notes") or [])
        assert row["spec"] == {
            "name": body["name"],
            "version": body["version"],
            "sha256": x["book_spec"]["sha256"],
            "direction": body["book"]["direction"],
            "sign": body["book"]["sign"],
            "control": body["control"]["kind"],
            "fee_bps": body["costs"]["fee_bps"],
            "slippage_bps": body["costs"]["slippage_bps"],
        }
        assert row["first_forward_sunday"] == x["first_forward_sunday"] == str(F)
        assert row["withdrawn"] is (tid in wd) and row["line"] == x["seq"] and row["part"] == tid.lower()
        h = row["historical"]
        for k in (
            "result",
            "word",
            "reasons",
            "total_return",
            "sharpe",
            "n_windows",
            "n_positive",
            "share_positive",
            "weeks",
            "held_weeks",
        ):
            assert h[k] == x["historical"][k], (tid, k)
        assert h["sigma_floor"] == x["sigma_floor"] and h["sigma_floor_source"] == x["sigma_floor_source"]
        mine = [w for w in weeks if w["test_id"] == tid]
        fw = row["forward"]
        assert fw["weeks"] == len(mine)
        state = mine[-1]["cs"]["state"] if mine else "waiting"
        assert fw["state"] == state, tid
        if mine:
            sig6(fw["lower"], mine[-1]["cs"]["lower"], tid)
            sig6(fw["upper"], mine[-1]["cs"]["upper"], tid)
        decided = state in ("useful", "useless")
        want_word = state if decided else words[x["historical"]["result"]]
        assert row["verdict"]["word"] == want_word and row["verdict"]["by"] == (
            "forward" if decided else "historical"
        ), tid
        assert row["verdict"]["badge"] == M.VERDICT_BADGES[want_word] and fw["badge"] == M.STATE_BADGES[state]
        nxt = date.fromisoformat(x["first_forward_sunday"]) + timedelta(days=7 * len(mine))
        assert fw["next_week_end"] == (None if decided else str(nxt + timedelta(days=7))), tid
        assert row["related"] == sorted(rel[tid]) and row["alarms"] == sum(1 for a in alarms if a.get("test_id") == tid)
        assert row["results"] is True and row["results_differ"] == [], tid
        assert part(row["part"])["results_differ"] == [], tid
    held = {r["id"]: r["forward"]["held"] for r in p["rows"]}
    assert held == {
        "FT-0001": None,
        "FT-0002": None,
        "FT-0003": {"sunday": LOG_DAY, "reason": T.HELD_GAP},
        "FT-0004": None,
    }, held
    rows = {r["id"]: r for r in p["rows"]}
    assert rows["FT-0001"]["forward"]["state"] == "useful" and rows["FT-0001"]["forward"]["decided_week"] == 26
    assert rows["FT-0002"]["forward"]["weeks"] == rows["FT-0003"]["forward"]["weeks"] == 8
    c = p["counts"]
    live = [r for r in p["rows"] if not r["withdrawn"]]
    assert (c["registered"], c["withdrawn"], c["n"]) == (4, 1, 3)
    assert c["states"] == {s: sum(1 for r in live if r["forward"]["state"] == s) for s in rules.FORWARD_STATES}
    assert c["verdicts"] == {
        "useful": sum(1 for r in live if r["verdict"]["word"] == "useful"),
        "useless": sum(1 for r in live if r["verdict"]["word"].startswith("useless")),
        "unproven": sum(1 for r in live if r["verdict"]["word"].startswith("unproven")),
    }
    assert c["inputs"] == {"formula": 2, "values_file": 1}
    # the alarms of the data: the data_gap alarm of the ETH funding day
    assert [a["type"] for a in p["alarms"]] == ["data_gap"]
    assert (
        p["alarms"][0]["text"]
        == "data_gap (okx): missing data: ETH-USDT funding_rate from 2026-03-10. The week counts."
    )
    assert p["ledger"]["present"] and p["ledger"]["ok"] and p["ledger"]["problems"] == []
    for key, rel_path in T.CHAINS.items():
        lines = (
            (DATA / "trials" / rel_path).read_text(encoding="ascii").splitlines()
            if (DATA / "trials" / rel_path).is_file()
            else []
        )
        assert p["ledger"]["lines"][key] == len(lines), key
        want = hashlib.sha256(lines[-1].encode("ascii")).hexdigest() if lines else "0" * 64
        assert p["ledger"]["head_sha256"][key] == want, key
    assert p["as_of"] == LOG_DAY and p["code"] == {"line": 4, "matches": True, "changed": []}


def test_historical_numbers_recomputed():
    """From historical_daily.parquet, written out here: the total return, the Sharpe ratio, the
    1-year windows and their share (equal to the registration line within its 6 digits), the
    weekly marks and the statistics of the book, the control, BTC hold and the net excess."""
    p = built()
    for row in p["rows"]:
        tid = row["id"]
        d = DATA / "trials" / "results" / tid.lower()
        daily = pd.read_parquet(d / "historical_daily.parquet")
        doc = json.loads((d / "historical.json").read_text())
        h = historical_from_daily(daily)
        reg = row["historical"]
        for k in ("total_return", "sharpe", "share_positive"):
            sig6(reg[k], h[k], (tid, k))
        assert (reg["n_windows"], reg["n_positive"]) == (h["n_windows"], h["n_positive"]), tid
        ph = part(row["part"])["historical"]
        assert [(w["start"], w["end"]) for w in ph["windows"]] == [w[:2] for w in h["windows"]], tid
        for w, (_, _, s) in zip(ph["windows"], h["windows"]):
            close(w["sharpe"], s, 1e-6, tid)
        assert ph["reasons"] == reg["reasons"] and ph["missed"] == doc["missed"]
        # the weekly marks: each Sunday S0 .. E as a multiple of the mark of S0
        s0 = pd.Timestamp(ph["start"], tz="UTC")
        sundays = pd.date_range(s0, pd.Timestamp(ph["end"], tz="UTC"), freq="7D")
        assert ph["weekly"]["sundays"] == [str(s.date()) for s in sundays]
        di = pd.DatetimeIndex(pd.to_datetime(daily["date"], utc=True))
        for col, key in (("book_equity", "book"), ("btc_equity", "btc"), ("control_equity", "control")):
            v = pd.Series(daily[col].to_numpy(dtype=float), index=di).reindex(sundays).to_numpy()
            if not np.isfinite(v).any():
                assert ph["weekly"][key] is None, (tid, key)
                continue
            for a, b in zip(ph["weekly"][key], v / v[0]):
                close(a, b, 1e-6, (tid, key))
        for a, b in zip(ph["weekly"]["excess"], h["x"].reindex(sundays).to_numpy()):
            close(a, b, 1e-6, (tid, "excess"))
        # the statistics matrix
        st = ph["stats"]
        for key, col in (("book", "book_equity"), ("btc", "btc_equity"), ("control", "control_equity")):
            eq = daily[col].to_numpy(dtype=float)
            if not np.isfinite(eq).any():
                assert st[key] is None, (tid, key)
                continue
            want = equity_stats(eq)
            for k in ("total_return", "sharpe", "ann_vol", "max_dd"):
                close(st[key][k], want[k], 1e-9, (tid, key, k))
        for k in ("total_return", "sharpe", "max_dd"):
            close(st["excess"][k], h[k], 1e-9, (tid, "excess", k))
        close(st["excess"]["ann_vol"], float(np.std(h["d"], ddof=1) * math.sqrt(365)), 1e-9, tid)
        assert ph["factor_numbers"] == json.loads(WC.dumps(doc["factor_numbers"])), tid
    # the factor numbers of the values-file test from its history values in the registration line
    t1 = next(x for x in chain("registrations.jsonl") if x.get("test_id") == "FT-0001")
    frame = pd.DataFrame(
        [[np.nan if v is None else v for v in t1["factor"]["history"][s]] for s in sorted(t1["factor"]["history"])],
        index=pd.DatetimeIndex([pd.Timestamp(s, tz="UTC") for s in sorted(t1["factor"]["history"])]),
        columns=SYMS,
    )
    okx = {**uf.load_source("okx"), "complete_through": date.fromisoformat(t1["complete_through"]["okx"])}
    want = json.loads(WC.dumps(uf.factor_numbers(frame, okx)))
    got = part("ft-0001")["historical"]["factor_numbers"]
    for k in ("weeks", "ic_mean", "ic_t", "ric_mean", "ric_t", "persistence", "coverage"):
        close(got[k], want[k], 1e-9, k)
    last = date.fromisoformat(t1["complete_through"]["okx"]) - timedelta(days=7)
    assert got["weeks"] == sum(1 for s in t1["factor"]["history"] if date.fromisoformat(s) <= last)


def test_forward_path_recomputed():
    """The bounds and the state of each frozen week, from the frozen weekly net excess returns and
    the registered sigma floor (a plain loop of the rule); the equity marks of the frozen weeks; the
    dates of week 26 and week 156; the final gaps of each week."""
    p = built()
    weeks = chain("weeks.jsonl")
    regs = {x["test_id"]: x for x in chain("registrations.jsonl") if x["kind"] == "registration"}
    for row in p["rows"]:
        tid = row["id"]
        mine = [w for w in weeks if w["test_id"] == tid]
        fp = part(row["part"])["forward"]
        assert fp["first_forward_sunday"] == str(F)
        assert fp["next_week_end"] == row["forward"]["next_week_end"], tid
        assert fp["milestones"] == {
            "week26_end": str(F + timedelta(days=182)),
            "week156_end": str(F + timedelta(days=1092)),
        }
        band = plain_band([w["excess"] for w in mine], float(regs[tid]["sigma_floor"]))
        assert [x["week"] for x in fp["path"]] == [w["week"] for w in mine] == list(range(1, len(mine) + 1))
        for x, w, b in zip(fp["path"], mine, band):
            assert (x["sunday"], x["end"]) == (w["sunday"], w["end"])
            for k in ("mean", "sigma", "lower", "upper"):
                close(x[k], b[k], 1e-6, (tid, w["week"], k))
            assert x["state"] == b["state"] == w["cs"]["state"], (tid, w["week"])
            for k in ("book_ret", "control_ret", "excess", "btc_ret"):
                close(x[k], w[k], 1e-6, (tid, w["week"], k))
            assert x["final_gaps"] == w["final_gaps"] and x["held"] == w["held"]
        eq = fp["equity"]
        assert eq["sundays"] == [str(F)] + [w["end"] for w in mine]
        for key in ("book_ret", "btc_ret"):
            want = np.concatenate([[1.0], np.cumprod([1.0 + w[key] for w in mine])])
            for a, b in zip(eq["book" if key == "book_ret" else "btc"], want):
                close(a, b, 1e-6, (tid, key))
        assert eq["control"] is None if regs[tid]["book_spec"]["body"]["control"]["kind"] == "zero" else eq["control"]
    # FT-0001: useful at week 26; FT-0002: a final gap of the ETH funding in week 3
    assert part("ft-0001")["forward"]["path"][-1]["state"] == "useful"
    gaps = [x["final_gaps"] for x in part("ft-0002")["forward"]["path"]]
    assert gaps[2] == [["ETH-USDT", "funding_rate", "2026-03-10"]] and not any(gaps[:2] + gaps[3:]), gaps
    # the withdrawn test: its withdrawal line, no forward week
    fw = part("ft-0004")
    assert fw["withdrawal"]["reason"] == "a withdrawal" and fw["forward"]["path"] == []


def test_verdict_words_come_from_rules_verdict():
    """The verdict of each row and part is rules.verdict of the historical result and the forward
    test of the frozen weeks. The words of the badges and of the filter come from the data."""
    p = built()
    weeks = chain("weeks.jsonl")
    regs = {x["test_id"]: x for x in chain("registrations.jsonl") if x["kind"] == "registration"}
    for row in p["rows"]:
        x = regs[row["id"]]
        fwd = rules.forward_test([w["excess"] for w in weeks if w["test_id"] == row["id"]], float(x["sigma_floor"]))
        v = rules.verdict(x["historical"]["result"], fwd)
        assert (row["verdict"]["word"], row["verdict"]["by"]) == (v["word"], v["by"])
        assert part(row["part"])["verdict"] == {"word": v["word"], "by": v["by"]}
    assert [w["key"] for w in p["words"]["verdicts"]] == list(M.VERDICT_BADGES)
    assert [w["key"] for w in p["words"]["states"]] == list(rules.FORWARD_STATES)
    for name in ("mine.js", "factor.js"):
        code = _code(WEB / "pages" / name)
        lits = re.findall(r'"((?:[^"\\\n]|\\.)*)"', code)
        hits = [x for x in lits if VERDICT_WORD.search(x)]
        assert not hits, (name, hits)


def test_expected_false_useful_number():
    """alpha x N over the registered tests that are not withdrawn (all states), with its text."""
    p = built()
    ex = p["expected_false_useful"]
    n = sum(1 for r in p["rows"] if not r["withdrawn"])
    assert (ex["alpha"], ex["n"]) == (0.1, n) and n == 3
    close(ex["value"], 0.1 * n, 1e-12)
    assert ex["text"] == (
        'Expected number of false "useful" verdicts, for a false-pass rate of 10 % for each test: '
        "0.1 x N = 0.3 (N = 3 registered tests that are not withdrawn)."
    )
    assert ex["label"] == 'Expected false "useful" verdicts'


def test_broken_ledger_shows_the_problems():
    """A changed line: the ledger is broken (the problems, no row, no part). A line check that fails:
    the same. Result files that disagree with the ledger fail the section; absent result files give
    an empty historical block."""
    with ledger_variant() as root:
        p = root / "weeks.jsonl"
        lines = p.read_text(encoding="ascii").splitlines()
        x = json.loads(lines[3])
        x["excess"] = 0.5
        lines[3] = LG.canonical(x)
        p.write_text("\n".join(lines) + "\n", encoding="ascii")
        b = json.loads(WC.dumps(M.build()))
        assert b["ledger"]["present"] is True and b["ledger"]["ok"] is False and b["rows"] == []
        assert b["ledger"]["problems"] and "weeks.jsonl" in b["ledger"]["problems"][0], b["ledger"]["problems"]
        assert b["counts"] is None and b["expected_false_useful"] is None and "parts" not in b
        assert b["rule"]["text"] == list(rules.RULE_TEXT) and len(b["specs"]) == 2
    with ledger_variant() as root:
        # the historical numbers of the result files differ from the ledger: the section fails
        f = root / "results" / "ft-0002" / "historical_daily.parquet"
        d = pd.read_parquet(f)
        d.loc[d.index[-1], "book_equity"] *= 1.01
        d.to_parquet(f, index=False)
        try:
            M.build()
        except WC.DataMismatch as exc:
            assert "FT-0002" in str(exc), exc
        else:
            raise AssertionError("a changed result file was not found")
    with ledger_variant() as root:
        f = root / "results" / "ft-0003" / "forward_weekly.parquet"
        d = pd.read_parquet(f)
        d.loc[d.index[0], "excess"] += 0.01
        d.to_parquet(f, index=False)
        try:
            M.build()
        except WC.DataMismatch as exc:
            assert "FT-0003" in str(exc) and "forward_weekly" in str(exc), exc
        else:
            raise AssertionError("a changed forward_weekly.parquet was not found")
    with ledger_variant() as root:
        (root / "results" / "ft-0003" / "historical.json").unlink()
        b = dict(M.build())
        parts = b.pop(WC.PARTS_KEY)
        row = next(r for r in b["rows"] if r["id"] == "FT-0003")
        assert row["results"] is False and parts["ft-0003"]["historical"] is None
        assert parts["ft-0003"]["forward"]["path"], "the forward block comes from the ledger"


def _restate_closes(day: str) -> None:
    """Multiply the close of each coin on `day` by a small factor (a restatement of the data)."""
    d = pd.Timestamp(day, tz="UTC")
    for i, f in enumerate(sorted((DATA / "raw" / "ohlcv").glob("*.parquet"))):
        df = pd.read_parquet(f)
        m = (pd.to_datetime(df["ts"], utc=True).dt.floor("D") == d).to_numpy()
        df.loc[m, "close"] = df.loc[m, "close"] * (1 + 0.02 * (i % 7 - 3))
        df.to_parquet(f, index=False)


def _write_results_again() -> None:
    """The history part of judge.py verify --write-results (trials.verify) for each test that is not
    withdrawn: the alarm history_changed when the history values changed, then the historical result
    files from the data on disk. (A full verify replays the frozen weeks too; the synthetic weeks of
    FT-0001 end after the data of the fixture.)"""
    rec = T.read()
    cache = T._sources()
    okx = T._source(cache, "okx")
    mk = T.market()
    now = T._now()
    alarms = []
    for t in T.active(rec):
        hv = T._history_values_of(t, cache)
        if T.values_sha256(hv) != t["historical"]["values_sha256"]:
            detail = {"start": t["historical"]["start"], "end": t["historical"]["end"]}
            source = t["factor"].get("source") or "file"
            alarms.append(T._alarm("history_changed", t["test_id"], source, detail, now, okx["data_through"]))
        spec = book.spec_from_body(t["book_spec"]["body"], source=t["book_spec"].get("source") or "ledger")
        h = T.run_history(spec, hv, T._ts(t["historical"]["start"]), T._ts(t["historical"]["end"]), mk)
        okx_reg = {**okx, "complete_through": T._day(t["complete_through"]["okx"])}
        result = {
            "_run": h["run"],
            "_weekly": h["weekly"],
            "historical_full": h["hist"],
            "historical": t["historical"],
            **T.history_numbers(h, okx_reg),
        }
        T.write_history_results(t["test_id"], result)
    if alarms:
        T.ledger().append("alarms", alarms)


def test_restated_data_marks_the_result_files():
    """judge.py writes the result files again from the data on disk; the ledger keeps the frozen
    numbers. (1) A restated close inside a frozen forward week: log-week records an alarm
    book_week_changed and writes forward_weekly.parquet from a new replay. The export works, the
    row and the part get a "results_differ" entry with the weeks of the alarm, and the forward path
    is the chain. A change in a week that no alarm names still fails the section. (2) A restated
    close of the history and verify --write-results: the export works, each rewritten test gets an
    entry (with the alarm history_changed for a formula that reads the close), and the historical
    numbers of each row are the numbers of the registration line."""
    gap = {("funding", "ETH-USDT"): ["2026-03-10"], (FB, "ETH-USDT"): ["2026-03-10"]}
    fixture = {**gap, ("funding", "BTC-USDT"): [LOG_DAY], (FB, "BTC-USDT"): [LOG_DAY]}  # build_fixture()
    try:
        with ledger_variant() as root:
            cut_to("2026-05-03", drop=gap)
            _restate_closes("2026-03-03")  # inside the frozen week 2 of FT-0002 and FT-0003
            clock(at("2026-05-04"))
            assert T.log_week(log=QUIET) == T.EXIT_ALARM
            assert T.problems(T.read()) == []
            named = {
                a["test_id"]: a["detail"]["weeks"] for a in chain("alarms.jsonl") if a["type"] == "book_week_changed"
            }
            assert named.get("FT-0002"), named
            b = dict(M.build())
            parts = json.loads(WC.dumps(b.pop(WC.PARTS_KEY)))
            b = json.loads(WC.dumps(b))
            weeks = chain("weeks.jsonl")
            for row in b["rows"]:
                tid = row["id"]
                want = []
                if tid in named:
                    got = pd.read_parquet(root / "results" / tid.lower() / "forward_weekly.parquet")
                    have = dict(zip(pd.to_datetime(got["sunday"]).dt.strftime("%Y-%m-%d"), got["excess"]))
                    bad = [
                        w["week"]
                        for w in weeks
                        if w["test_id"] == tid and not np.isclose(have[w["sunday"]], w["excess"], rtol=1e-9, atol=1e-12)
                    ]
                    assert bad and set(bad) <= set(named[tid]), (tid, bad, named[tid])
                    s_ = "s" if len(bad) > 1 else ""
                    want = [
                        {
                            "block": "forward",
                            "alarm": "book_week_changed",
                            "weeks": bad,
                            "text": f"forward_weekly.parquet differs from the frozen week{s_} "
                            f"{', '.join(str(x) for x in bad)}. The alarm book_week_changed records the change. "
                            "The verdict uses the frozen weeks of the ledger, and the page shows them.",
                        }
                    ]
                assert row["results_differ"] == want and parts[row["part"]]["results_differ"] == want, tid
                # the forward path is the chain, not the result file
                mine = [w for w in weeks if w["test_id"] == tid]
                for x, w in zip(parts[row["part"]]["forward"]["path"], mine):
                    close(x["excess"], w["excess"], 1e-6, (tid, w["week"]))
            if _node() is not None:
                web = _web()
                f2 = _render("factor", web, {"param": "FT-0002"})
                assert "The result files differ from the ledger" in f2["text"]
                assert "The alarm book_week_changed records the change." in f2["text"]
                m = _render("mine", web)
                assert "The result files differ from the ledger" in m["text"] and "FT-0002: forward_weekly" in m["text"]
            # a week that no alarm names: the section fails
            first = next(w for w in weeks if w["test_id"] == "FT-0002" and w["week"] not in named["FT-0002"])
            f = root / "results" / "ft-0002" / "forward_weekly.parquet"
            d = pd.read_parquet(f)
            i = int(np.flatnonzero(pd.to_datetime(d["sunday"]).dt.strftime("%Y-%m-%d") == first["sunday"])[0])
            d.loc[d.index[i], "excess"] += 0.01
            d.to_parquet(f, index=False)
            try:
                M.build()
            except WC.DataMismatch as exc:
                assert "FT-0002" in str(exc) and "forward_weekly" in str(exc), exc
            else:
                raise AssertionError("a change in a week without an alarm was not found")
        with ledger_variant() as root:
            cut_to(LOG_DAY, drop=fixture)
            _restate_closes("2025-06-03")  # inside the history of each test
            clock(at("2026-04-27"))
            _write_results_again()
            assert T.problems(T.read()) == []
            alarms = chain("alarms.jsonl")
            regs = {x["test_id"]: x for x in chain("registrations.jsonl") if x["kind"] == "registration"}
            b = dict(M.build())
            parts = json.loads(WC.dumps(b.pop(WC.PARTS_KEY)))
            b = json.loads(WC.dumps(b))
            for row in b["rows"]:
                tid = row["id"]
                hl = regs[tid]["historical"]
                for k in ("result", "word", "total_return", "sharpe", "n_windows", "n_positive", "share_positive"):
                    assert row["historical"][k] == hl[k], (tid, k)
                doc = json.loads((root / "results" / tid.lower() / "historical.json").read_text())
                same = all(
                    abs(doc["historical"][k] - hl[k]) <= 5e-6 * max(abs(doc["historical"][k]), abs(hl[k])) + 1e-12
                    for k in ("total_return", "sharpe")
                )
                if row["withdrawn"] or same:
                    assert row["results_differ"] == [], tid
                    continue
                has = any(a["type"] == "history_changed" and a["test_id"] == tid for a in alarms)
                text = (
                    "The historical result files differ from the registration line. judge.py verify --write-results "
                    "wrote them again after a change of the data."
                    + (" The alarm history_changed records the change." if has else "")
                    + " The verdict uses the numbers of the registration line. The charts and the statistics "
                    "show the result files."
                )
                want = [
                    {"block": "historical", "alarm": "history_changed" if has else None, "weeks": None, "text": text}
                ]
                assert row["results_differ"] == want and parts[row["part"]]["results_differ"] == want, tid
            differ = {r["id"]: r["results_differ"][0]["alarm"] for r in b["rows"] if r["results_differ"]}
            # the values file has no new values (no alarm); the formula on the close has the alarm
            assert differ.get("FT-0001", "x") is None and differ.get("FT-0002") == "history_changed", differ
    finally:
        cut_to(LOG_DAY, drop=fixture)
        clock(at("2026-04-27"))


def test_part_names_sizes_and_privacy():
    """One part for each registered test (a withdrawn one too), named by its id in lower case; each
    part is at most 200,000 bytes and is not in the total of the size budget; no path and no
    private string; the registration line of a values file gives the count of its history values."""
    p = built()
    assert p["parts"] == ["ft-0001", "ft-0002", "ft-0003", "ft-0004"]
    for name in p["parts"]:
        text = WC.render_section("mine", part(name), part=name)  # raises on a private string
        assert len(text.encode("ascii")) <= WC.MINE_PART_MAX_BYTES == 200_000, name
        assert not re.search(r"/Users/|/home/|/private/|/tmp/|\.tmp/|[A-Za-z]:\\\\", text), name
        assert str(TMP) not in text and str(REPO) not in text
        # the registration block: the line without seq and prev_sha256, its line number and its hash
        reg = part(name)["registration"]
        assert "prev_sha256" not in reg and "seq" not in reg and "withdrawn" not in reg
        lines = (DATA / "trials" / "registrations.jsonl").read_text(encoding="ascii").splitlines()
        src = json.loads(lines[reg["line"] - 1])
        assert src["kind"] == "registration" and src["test_id"] == part(name)["id"] and src["seq"] == reg["line"]
        assert reg["sha256"] == hashlib.sha256(lines[reg["line"] - 1].encode("ascii")).hexdigest()
        for k, v in src.items():
            if k not in ("seq", "prev_sha256", "factor"):
                assert reg[k] == v, (name, k)
    hist = part("ft-0001")["registration"]["factor"]["history"]
    assert hist == {"sundays": len(chain("registrations.jsonl")[0]["factor"]["history"]), "in_ledger": True}
    text = WC.render_section("mine", built())
    assert len(text.encode("ascii")) <= WC.SECTION_MAX_BYTES
    # the size rule: a part of mine has its own limit and is not in the total
    assert WC.check_sizes({"mine.ft-0001": 200_001}) == ["mine.ft-0001.js is 200,001 bytes > 200,000"]
    assert WC.check_sizes({"mine": 10, **{f"mine.ft-{i:04d}": 199_000 for i in range(40)}}) == []
    assert WC.check_sizes({"library": 1_000_000, "library.expr-0": 1_000_000, "x": 4_500_000})
    # the exporter writes the parts and lists them in meta.js
    with tempfile.TemporaryDirectory(dir=TMP) as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            assert E.main(["--out", tmp, "--only", "mine"]) == 0
        files = sorted(f.name for f in Path(tmp).iterdir())
        assert files == [
            "meta.js",
            "mine.ft-0001.js",
            "mine.ft-0002.js",
            "mine.ft-0003.js",
            "mine.ft-0004.js",
            "mine.js",
        ], files
        meta = E.read_data_file(Path(tmp) / "meta.js")
        assert (
            sorted(meta["sections"]["mine"]["parts"]) == p["parts"] and meta["sections"]["mine"]["state"] == "written"
        )
        assert meta["sections"]["mine"]["as_of"] == LOG_DAY


def _code(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    code = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", code)


def test_page_rules_and_wiring():
    """mine.js and factor.js: classic ES2019 scripts, data through textContent, the route wiring,
    the texts of the states (5.7), the entity colors and no fixed verdict text."""
    mine, factor = (
        (WEB / "pages" / "mine.js").read_text(encoding="utf-8"),
        (WEB / "pages" / "factor.js").read_text(encoding="utf-8"),
    )
    for name, js in (("mine.js", mine), ("factor.js", factor)):
        code = _code(WEB / "pages" / name)
        assert '"use strict"' in js and "FL.pages.register" in js, name
        for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
            assert bad not in code, (name, bad)
        assert "?." not in re.sub(r"\"[^\"]*\"|'[^']*'|/[^/\n]+/[gimsuy]*", "", code) and "??" not in code, name
        assert not re.search(r"(?m)^\s*(?:import|export)\s", js), name
        assert "window.addEventListener" not in code, name
    reg = mine[mine.index("FL.pages.register") :]
    assert re.search(r'id: "mine",\s*title: "My factors"', reg) and 'section: "mine"' in reg
    reg = factor[factor.index("FL.pages.register") :]
    assert re.search(r'id: "factor",\s*title: "Factor test"', reg)
    assert 'section: "mine"' in reg and 'data: ["mine"]' in reg and "requiresData: false" in reg
    for text in (
        "Give a formula or a values file. Factorlab tests it in a book: first a historical test, then a forward test with the weeks after the registration.",
        '"No factor is registered. "',
        "The trial ledger does not pass its checks",
        "Registered tests (N)",
        "The rule",
        "Book specifications",
        "Commands",
    ):
        assert text in mine, text
    for text in (
        "Point in time: proven at each Sunday cut.",
        "Point in time: not proven. A forward value counts only from the first daily close after its log time.",
        "Waiting for the first forward week. It ends on ",
        '"Held at "',
        '"Missing data in this week: "',
        "The result files are absent. Run judge.py verify --write-results. ",
        "Withdrawn before the first forward week.",
        "No registered test has this id.",
        "BTC hold gives context only.",
        "refLines: [{ y: 0 }]",
        "log: true",
        # the log time of a values file (trials.counts_from: the first daily close after the log)
        "Log the values of each Sunday before 24:00 UTC on that Sunday. Then the book uses them from Monday 00:00 UTC.",
        "This test page failed to render",
        "The result files differ from the ledger",
        '"A library definition."',
    ):
        assert text in factor, text
    assert "The result files differ from the ledger" in mine and "seed Sunday" not in factor
    assert "Monday close" not in factor and "Loading" not in factor and "judging code" not in mine
    for ent in ('entity: "factor_book"', 'entity: "control"', 'entity: "excess"', 'entity: "btc"'):
        assert ent in factor, ent
    core = (WEB / "assets" / "fl-core.js").read_text(encoding="utf-8")
    # the entities of the charts: the three series of a factor test and the BTC benchmark, no context override
    start = core.index("var ENTITY = {")
    entity = core[start : core.index("};", start) + 2]
    want = (
        "var ENTITY = {\n"
        "    factor_book: 1, control: 2, excess: 3,\n"
        '    btc: "btc", btc_hold: "btc", BTC: "btc", benchmark: "btc", bench: "btc"\n'
        "  };"
    )
    assert entity == want, f"the entity map of fl-core.js: {entity!r}; want {want!r}"
    assert "var CONTEXT = {};" in core, "fl-core.js has a context override of the entity colors"
    # the links: each id of the table opens its page; the page of a library id links to the search
    assert "ctx.nav.factorHref(id)" in mine and 'href(ctx, "library", "search")' in factor


def _oklab_distances(css: str, theme: str) -> dict:
    """The smallest OKLab distance (x 100) of each pair of entity colors that share a chart, over
    normal vision and simulated protanopia, deuteranopia and tritanopia (the physiologically
    based simulation model of 2009, severity 1)."""
    if theme == "light":
        block = re.search(r"(?m)^:root\s*\{([^{}]*)\}", css).group(1)
    else:
        block = re.search(r'(?m)^:root\[data-theme="dark"\]\s*\{([^{}]*)\}', css).group(1)
    tok = dict(re.findall(r"(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{6})", block))
    light = dict(
        re.findall(r"(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{6})", re.search(r"(?m)^:root\s*\{([^{}]*)\}", css).group(1))
    )
    col = {
        k: tok.get(f"--{v}", light[f"--{v}"]) for k, v in (("factor_book", "c1"), ("control", "c2"), ("btc", "muted"))
    }

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    def oklab(rl):
        m1 = np.array(
            [
                [0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005],
            ]
        )
        m2 = np.array(
            [
                [0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660],
            ]
        )
        return m2 @ np.cbrt(m1 @ rl)

    sims = [
        np.eye(3),
        np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
        np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
        np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
    ]
    rgb = {k: np.array([lin(int(v[i : i + 2], 16) / 255) for i in (1, 3, 5)]) for k, v in col.items()}
    out = {}
    for a, b in (("factor_book", "control"), ("factor_book", "btc"), ("control", "btc")):
        out[(a, b)] = round(
            min(
                float(np.linalg.norm(oklab(np.clip(m @ rgb[a], 0, 1)) - oklab(np.clip(m @ rgb[b], 0, 1)))) for m in sims
            )
            * 100,
            1,
        )
    return out


def test_entity_colors_are_apart():
    """The entity colors of the pages of a test stay apart for a viewer with a color-vision
    deficiency, in both themes; the table of web/README.md gives the same numbers."""
    css = (WEB / "assets" / "app.css").read_text(encoding="utf-8")
    readme = (WEB / "README.md").read_text(encoding="utf-8")
    light, dark = _oklab_distances(css, "light"), _oklab_distances(css, "dark")
    rows = {
        ("factor_book", "control"): "| `factor_book` and `control` |",
        ("factor_book", "btc"): "| `factor_book` and `btc` (also dashed) |",
        ("control", "btc"): "| `control` and `btc` (also dashed) |",
    }
    for pair, head in rows.items():
        assert light[pair] >= 9.0 and dark[pair] >= 9.0, (pair, light[pair], dark[pair])
        assert f"{head} {light[pair]} | {dark[pair]} |" in readme, (head, light[pair], dark[pair])


def _stub(page: str) -> str:
    """The node DOM stand-in of test_export_web.py (read with ast, not imported), set to a page that
    reads the section mine and loads its part files through ctx.loadPart; the render waits for them."""
    tree_ = ast.parse((ENGINE / "test_export_web.py").read_text(encoding="utf-8"))
    stub = next(
        n.value.value
        for n in tree_.body
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "_DOM_STUB" for x in n.targets)
    )
    pairs = (
        ('"data/PAGE.js", "pages/PAGE.js"]', '"data/mine.js", "pages/PAGE.js"]'),
        ("section: FL.data.PAGE,", "section: FL.data.mine, loadPart: LOADPART, cardReady: CARDREADY,"),
        ("const TITLE = [], SUBTITLE = [];", "const TITLE = [], SUBTITLE = [], READY = [];"),
        (
            "subtitle: SUBTITLE.length ? SUBTITLE[SUBTITLE.length - 1] : null }",
            "subtitle: SUBTITLE.length ? SUBTITLE[SUBTITLE.length - 1] : null, ready: READY }",
        ),
        ('root.querySelectorAll(".fl-chart-toggle")', 'WAIT root.querySelectorAll(".fl-chart-toggle")'),
    )
    for a, b in pairs:
        assert stub.count(a) == 1, a
        stub = stub.replace(a, b)
    stub = stub.replace("PAGE", page)
    load = (
        'n => { const f = path.join(WEB, "data", "mine." + n + ".js"); if (!fs.existsSync(f)) return Promise.reject({ state: "missing" });'
        ' vm.runInContext(fs.readFileSync(f, "utf8"), C, { filename: f }); return Promise.resolve(W.FL.data["mine." + n]); }'
    )
    stub = stub.replace("LOADPART", load)
    # ctx.cardReady: the page calls it when its cards are on the page (the shell then scrolls to the
    # card of the route); the stand-in records the card of the route and whether it is on the page
    ready = "() => { const c = NAV.card || null; READY.push({ card: c, found: !!(c && D.getElementById(c)) }); return true; }"
    stub = stub.replace("CARDREADY", ready)
    stub = stub.replace(
        "WAIT ", "(async () => { for (let k = 0; k < 8; k++) await new Promise(r => setImmediate(r));\n"
    )
    return stub.rstrip() + "\n})().catch(e => { console.error(e && e.stack || e); process.exit(1); });\n"


def _node() -> str | None:
    n = os.environ.get("FACTORLAB_NODE") or os.environ.get("NODE") or shutil.which("node")
    return n if n and Path(n).exists() else None


def _web(payload_override: dict | None = None) -> Path:
    """A web folder with the real assets and pages and the export of the fixture (or a payload)."""
    root = Path(tempfile.mkdtemp(prefix="web_", dir=TMP))
    shutil.copytree(WEB / "assets", root / "assets")
    shutil.copytree(WEB / "pages", root / "pages")
    data = root / "data"
    data.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        assert E.main(["--out", str(data), "--only", "mine"]) == 0
    if payload_override is not None:
        for f in data.glob("mine.*js"):
            f.unlink()
        WC.write_section(data, "mine", payload_override)
    return root


def _render(page: str, web: Path, nav: dict | None = None) -> dict:
    r = subprocess.run(
        [_node(), "-e", _stub(page), str(web), "full", json.dumps(nav or {})],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, f"page render ({page}) failed: {r.stderr[-1500:]}"
    return json.loads(r.stdout)


def test_pages_render_in_node():
    """The page My factors with rows, empty and with a broken ledger; the page of each test (a values
    file that has a decision, a formula test with a final gap and a card route, a held test, a
    withdrawn test), of a test with a broken ledger and of an unknown id. Needs node."""
    if _node() is None:
        print("        (skipped the node run: set NODE or FACTORLAB_NODE)")
        return
    p = built()
    web = _web()
    m = _render("mine", web)
    t = m["text"]
    for r in p["rows"]:
        # a withdrawn test has no verdict and no forward state on the pages
        word = "none (withdrawn)" if r["withdrawn"] else r["verdict"]["word"]
        assert r["id"] in t and r["name"] in t and word in t, r["id"]
    assert "withdrawn (1)" in t and "Every test (4)" in t
    assert "#/factor/FT-0001" in m["links"] or "#/factor/FT-0001" in str(m["links"]), m["links"]
    assert p["expected_false_useful"]["text"] in t and 'Expected false "useful" verdicts' in t
    assert "0.3" in t and "26 of 156" in t and "held at 2026-04-26" in t and "withdrawn" in t
    assert "1 alarm of the data" in t and p["alarms"][0]["text"] in t
    for line in rules.RULE_TEXT:
        want = line[2:] if line.startswith("- ") else line  # an item of the line before it
        assert want in t, line
    for s in p["specs"]:
        assert s["sha256"] in t and s["file"] in t
    assert p["commands"]["register_library"] in t and p["commands"]["where"] in t
    assert "No factor is registered." not in t
    empty = dict(p)
    for k in ("rows", "alarms"):
        empty[k] = []
    empty.pop("parts", None)
    empty["ledger"] = {**p["ledger"], "present": False}
    empty["counts"] = {**p["counts"], "registered": 0, "withdrawn": 0, "n": 0}
    e = _render("mine", _web(empty))
    assert "No factor is registered." in e["text"] and p["commands"]["check_formula"] in e["text"]
    assert "The rule" in e["text"] and "Book specifications" in e["text"]
    broken = dict(empty)
    broken["ledger"] = {**p["ledger"], "ok": False, "problems": ["weeks.jsonl line 4 does not hold the hash of line 3"]}
    b = _render("mine", _web(broken))
    assert "The trial ledger does not pass its checks" in b["text"] and "weeks.jsonl line 4" in b["text"]
    assert p["commands"]["verify"] in b["text"]
    # the page of a test with a broken ledger names the cause, not an unknown id
    bf = _render("factor", _web(broken), {"param": "FT-0002"})
    assert "The trial ledger does not pass its checks" in bf["text"] and "weeks.jsonl line 4" in bf["text"]
    assert p["commands"]["verify"] in bf["text"] and "#/mine" in bf["links"], bf["links"]
    assert "No registered test has this id." not in bf["text"] + str(bf["subtitle"]), bf["text"][:600]
    # the page of each test
    f1 = _render("factor", web, {"param": "FT-0001"})
    t = f1["text"]
    assert f1["title"] == "FT-0001" and f1["subtitle"] == "values-a", (f1["title"], f1["subtitle"])
    assert "Point in time: not proven. A forward value counts only from the first daily close after its log time." in t
    assert "a.csv" in t and "useful" in t and "by the forward test" in t
    assert "Week 26: mean " in t and "decided at week 26" in t and "BTC hold gives context only." in t
    assert "Result: " + p["rows"][0]["historical"]["word"] in t
    f2 = _render("factor", web, {"param": "FT-0002", "card": "forward"})
    assert "Point in time: proven at each Sunday cut." in f2["text"] and "ts_mean(clv, 7)" in f2["text"]
    # a card route (#/factor/FT-0002/forward): the page calls ctx.cardReady once its cards exist
    assert f2["ready"] == [{"card": "forward", "found": True}], f2["ready"]
    assert f1["ready"] == [{"card": None, "found": False}], f1["ready"]
    assert "update.py logs each new week. To log the weeks yourself:" in f2["text"]
    assert (
        p["commands"]["where_one"] == "Run this command from the repository root."
        and p["commands"]["where_one"] in f2["text"]
    )
    r2 = p["rows"][1]
    assert f"registrations.jsonl line {r2['line']}, {r2['registered_at']}" in f2["text"]
    assert "Week 3: Missing data in this week: ETH-USDT funding_rate from 2026-03-10. The week counts." in f2["text"]
    f3 = _render("factor", web, {"param": "ft-0003"})
    assert "Held at 2026-04-26: a value is missing; the week waits up to 7 days." in f3["text"]
    assert "Before the first OKX funding print on disk, funding_rate is the Binance rate." in f3["text"]
    assert "Note of the registration: a note" in f3["text"] and "FT-0002" in f3["text"]
    f4 = _render("factor", web, {"param": "FT-0004"})
    assert "Withdrawn before the first forward week." in f4["text"] and "a withdrawal" in f4["text"]
    assert "none (withdrawn)" in f4["text"] and "by the historical test" not in f4["text"]
    unknown = _render("factor", web, {"param": "FT-0099"})
    assert "No registered test has this id." in unknown["text"] and "#/mine" in unknown["links"]
    assert unknown["subtitle"] == "No registered test has this id.", unknown["subtitle"]
    lib = _render("factor", web, {"param": "CF00001"})
    assert "CF00001 is a library definition." in lib["text"] and "#/library/search" in lib["links"]
    assert lib["subtitle"] == "A library definition.", lib["subtitle"]
    # an error inside a render function is a render error, not a missing part file
    b = dict(M.build())
    parts = b.pop(WC.PARTS_KEY)
    bad = json.loads(WC.dumps(parts["ft-0002"]))
    bad["forward"]["path"][0] = None
    err = _render("factor", _web(WC.with_parts(b, {**parts, "ft-0002": bad})), {"param": "FT-0002"})
    et = err["text"]
    assert "This test page failed to render" in et and "Part file not available" not in et, et[:600]
    gone = _render(
        "factor", _web(WC.with_parts(b, {k: v for k, v in parts.items() if k != "ft-0002"})), {"param": "FT-0002"}
    )
    assert "The part file data/mine.ft-0002.js is not in this folder." in gone["text"], gone["text"][:600]
    # without the result files the historical card says how to write them again
    with ledger_variant() as root:
        (root / "results" / "ft-0002" / "historical.json").unlink()
        web2 = _web()
    f5 = _render("factor", web2, {"param": "FT-0002"})
    assert "The result files are absent. Run judge.py verify --write-results." in f5["text"]
    assert p["commands"]["write_results"] in f5["text"]


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Run it with no arguments. It prints one PASS or FAIL line for each test.",
    ).parse_args()
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} export_mine tests passed")
    raise SystemExit(1 if failed else 0)
