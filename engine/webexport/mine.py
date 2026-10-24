"""Section "mine" (the page My factors, #/mine, and the page of one test, #/factor/<id>): build() -> dict.

The builder judges nothing. It reads the trial ledger data/trials/ (engine/trials.py) read-only:
``trials.read()`` (each chain, checked link by link and against head.json), ``trials.problems``
(the line checks of judge.py verify), ``trials.status_data()`` (the view of judge.py status) and
the result files results/<id>/. It checks them before it shows a number:

- a broken ledger (a chain check or a line check fails) gives ``ledger.ok = false``, the
  problems and no rows;
- the historical numbers of each registration line (total return, Sharpe ratio, the 1-year
  windows and the share with a positive Sharpe ratio) must equal the numbers that
  ``rules.historical_test`` gives on results/<id>/historical_daily.parquet and
  historical_weekly.parquet, within the 6 significant digits of the ledger;
- the bounds and the state of each frozen week must equal ``rules.forward_test`` of the frozen
  weekly net excess returns and the registered sigma floor, and the verdict of each row must
  equal ``rules.verdict``;
- the net excess return of results/<id>/forward_weekly.parquet must equal the chain.
A difference raises DataMismatch: the exporter then fails the section and keeps its old file.
Two differences are not errors, because judge.py writes the result files again from the data on
disk while the ledger keeps the frozen numbers. (1) historical.json, historical_daily.parquet and
historical_weekly.parquet agree with each other but not with the registration line: judge.py
verify --write-results computed them again after a change of the data (an alarm
history_changed can record it). (2) forward_weekly.parquet differs from the chain only in weeks
that an alarm book_week_changed of the test names. The row and the part then get
"results_differ" (one entry with a text for each case). The verdict, the rows and the forward
path still come from the ledger; the historical charts and statistics come from the result files.

Payload (the page My factors):
    ledger      {present, ok, problems, lines {chain: n}, head_sha256 {chain: hex}}
    rule        rules.rule_record() and its text (rules.RULE_TEXT)
    specs       the two shipped book specifications: file, body (normalized), sha256, book_key
    counts      registered, withdrawn, n, the forward states, the verdicts, the inputs
    expected_false_useful   {alpha, n, value, text}: alpha x N over the tests that are not withdrawn
    rows        one row for each registered test (the row of judge.py status --json), with
                "part": the name of its part file and "results_differ" (above)
    alarms      the alarms of the data (an alarm without a test id, for example data_gap)
    code        {line, matches, changed}: the code fingerprint of the ledger against today's code
    words       the labels of the verdicts and of the forward states, with their badge status
    notes       the text of each note code of a formula (userfactor.NOTES)
    commands    the judge.py commands, with placeholders in angle brackets, and the texts "where"
                (for several commands) and "where_one" (for one command)

Part file ``mine.<id in lower case>`` (one for each registered test, also a withdrawn one; the
page of the test loads it): the registration line (without the history values of a values
file, which the ledger holds; their count is given), the withdrawal line, the historical test
(weekly marks as multiples of the start, the 1-year windows, the reasons, the statistics of the
book, the control book, BTC hold and the net excess, the factor numbers and the Sundays at which
the book has the control portfolio; null when the result files are absent), the forward test
(each frozen week with its bounds and state, the equity marks, the end of the next week, the
milestones of week 26 and week 156), the verdict, the alarms of the test and "results_differ".
The registration block is the line without seq and prev_sha256, with "line" (its line number in
registrations.jsonl) and "sha256" (the hash of the line).
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import book
import config
import ledger as LG
import rules
import trials
import userfactor
from webexport.common import DataMismatch, data_path, rnd, to_date, with_parts

SCHEMA = 1
SPEC_NAMES = ("long_short", "long_only")
RESULT_FILES = ("historical.json", "historical_daily.parquet", "historical_weekly.parquet")
FORWARD_FILE = "forward_weekly.parquet"
WEEK = timedelta(days=7)
# the numbers of a ledger line have 6 significant digits (ledger.num)
NUM_RTOL = 5e-6
WHERE = "Run these commands from the repository root."
WHERE_ONE = "Run this command from the repository root."
_JUDGE = "cd engine && .venv/bin/python -B judge.py"
COMMANDS = {
    "where": WHERE,
    "where_one": WHERE_ONE,
    "check_formula": f'{_JUDGE} check --formula="<formula>"',
    "register_formula": f'{_JUDGE} register --formula="<formula>" --name <name>',
    "check_library": f"{_JUDGE} check --library <CF id>",
    "register_library": f"{_JUDGE} register --library <CF id>",
    "check_values": f"{_JUDGE} check --values <file.csv>",
    "register_values": f"{_JUDGE} register --values <file.csv> --name <name>",
    "log_values": f"{_JUDGE} log-values <FT id> --values <file.csv>",
    "log_week": f"{_JUDGE} log-week",
    "status": f"{_JUDGE} status",
    "verify": f"{_JUDGE} verify",
    "write_results": f"{_JUDGE} verify --write-results",
    "spec": f"{_JUDGE} spec long_only",
}
# the badge status of each verdict word and forward state (the page shows a symbol and the word)
VERDICT_BADGES = {
    "useful": "good",
    "useless": "critical",
    "useless (historical)": "warning",
    "unproven (historical pass)": "neutral",
}
STATE_BADGES = {"waiting": "neutral", "running": "info", "useful": "good", "useless": "critical"}
EXPECTED_LABEL = 'Expected false "useful" verdicts'


# --------------------------------------------------------------------------- small helpers


def _close(a, b) -> bool:
    """a and b are equal within the 6 significant digits of a ledger number (None equals None)."""
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) <= NUM_RTOL * max(abs(a), abs(b)) + 1e-12


def _day(x) -> date:
    return LG.as_day(x)


def _present(root: Path) -> bool:
    """True when the ledger folder holds head.json or a chain file."""
    if not root.is_dir():
        return False
    return os.path.lexists(root / LG.HEAD) or any(os.path.lexists(root / rel) for rel in trials.CHAINS.values())


def _newest_bar() -> str | None:
    """The newest daily perp bar on disk (OKX, any coin of config.UNIVERSE), or None."""
    last = None
    for sym in config.UNIVERSE:
        p = data_path(f"raw/ohlcv/{config.sym_to_file(sym)}.parquet")
        if p.is_file():
            ts = pd.read_parquet(p, columns=["ts"])["ts"]
            if len(ts):
                d = to_date(ts.max())
                last = d if last is None or (d is not None and d > last) else last
    return last


def _specs() -> list[dict]:
    out = []
    for name in SPEC_NAMES:
        s = book.load_spec(name)
        out.append(
            {"file": f"engine/bookspecs/{name}.toml", "body": s.body, "sha256": s.sha256, "book_key": s.book_key}
        )
    return out


def _words() -> dict:
    return {
        "verdicts": [{"key": k, "label": k, "badge": v} for k, v in VERDICT_BADGES.items()],
        "states": [{"key": k, "label": k, "badge": STATE_BADGES[k]} for k in rules.FORWARD_STATES],
    }


def _ledger_block(present: bool, rec: dict | None, problems: list[str]) -> dict:
    lines = {k: len(rec[k]["records"]) for k in trials.CHAINS} if rec is not None else {}
    head = {k: (rec[k]["shas"][-1] if rec[k]["shas"] else LG.GENESIS) for k in trials.CHAINS} if rec else {}
    return {
        "present": present,
        "ok": not problems,
        "problems": list(problems),
        "folder": f"data/{trials.FOLDER}",
        "files": dict(trials.CHAINS),
        "lines": lines,
        "head_sha256": head,
    }


def _expected(line: dict) -> dict:
    """The expected number of false "useful" verdicts (trials.expected_line) with its tile label."""
    return {**line, "label": EXPECTED_LABEL}


def _empty_counts() -> dict:
    return {
        "registered": 0,
        "withdrawn": 0,
        "n": 0,
        "states": dict.fromkeys(rules.FORWARD_STATES, 0),
        "verdicts": {"useful": 0, "useless": 0, "unproven": 0},
        "inputs": {"formula": 0, "values_file": 0},
    }


# --------------------------------------------------------------------------- the checks of one test


def _week_lines(rec: dict, test_id: str) -> list[dict]:
    return [w for w in rec["weeks"]["records"] if w.get("kind") == "book_week" and w.get("test_id") == test_id]


def _results(test_id: str) -> dict | None:
    """The result files of a test, or None when one of RESULT_FILES is absent."""
    d = trials.root() / LG.RESULTS / test_id.lower()
    if not all((d / f).is_file() for f in RESULT_FILES):
        return None
    out = {
        "doc": json.loads((d / "historical.json").read_text(encoding="utf-8")),
        "daily": pd.read_parquet(d / "historical_daily.parquet"),
        "weekly": pd.read_parquet(d / "historical_weekly.parquet"),
        "forward": pd.read_parquet(d / FORWARD_FILE) if (d / FORWARD_FILE).is_file() else None,
    }
    return out


def _utc(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def _returns(equity: pd.Series) -> pd.Series:
    """Daily returns; a day after a mark of 0 or less (a bust) has the return 0.0 (book.py)."""
    r = equity.pct_change()
    return r.where(~(equity.shift(1) <= 0), 0.0)


def _daily_excess(daily: pd.DataFrame) -> pd.Series:
    """The daily net excess return of the book over its control (the control "zero": the book's own
    return), without the first row, from historical_daily.parquet."""
    idx = _utc(daily["date"])
    d = _returns(pd.Series(daily["book_equity"].to_numpy(dtype=np.float64), index=idx))
    ctl = daily["control_equity"].to_numpy(dtype=np.float64)
    if np.isfinite(ctl).any():
        d = d - _returns(pd.Series(ctl, index=idx))
    return d.iloc[1:]


def _hist_diff(h: dict, ref: dict) -> str | None:
    """The first key of the historical numbers in which h and ref differ (None: they agree)."""
    for k in ("total_return", "sharpe", "share_positive"):
        if not _close(h[k], ref.get(k)):
            return k
    for k in ("n_windows", "n_positive", "weeks", "held_weeks"):
        if ref.get(k) is None or int(h[k]) != int(ref[k]):
            return k
    if h["result"] != ref.get("result"):
        return "result"
    return None


def _alarm_note(alarms: list[dict], kind: str) -> str:
    return f" The alarm {kind} records the change." if any(a.get("type") == kind for a in alarms) else ""


def _historical_part(t: dict, res: dict, alarms: list[dict]) -> tuple[dict, dict | None]:
    """The historical block of a part, after the check against the registration line, and the
    "results_differ" entry when the result files were computed again on changed data."""
    tid = t["test_id"]
    hl = t["historical"]
    daily, weekly, doc = res["daily"], res["weekly"], res["doc"]
    d = _daily_excess(daily)
    wk_idx = _utc(weekly["sunday"])
    exc = pd.Series(weekly["excess"].to_numpy(dtype=np.float64), index=wk_idx)
    held = pd.Series(weekly["held"].to_numpy(dtype=bool), index=wk_idx)
    floor = float(t["book_spec"]["body"]["forward"]["sigma_floor_fallback"])
    h = rules.historical_test(d, exc, held=held, fallback_floor=floor)
    differ = None
    key = _hist_diff(h, hl)
    if key is not None:
        # result files that agree with each other were computed again on changed data (verify
        # --write-results); the ledger keeps the numbers of the registration
        if _hist_diff(h, doc.get("historical") or {}) is not None:
            raise DataMismatch(f"{tid}: the historical {key} of the result files is not the number of the ledger")
        differ = {
            "block": "historical",
            "alarm": "history_changed" if any(a.get("type") == "history_changed" for a in alarms) else None,
            "weeks": None,
            "text": (
                "The historical result files differ from the registration line. judge.py verify --write-results "
                "wrote them again after a change of the data." + _alarm_note(alarms, "history_changed") + " The "
                "verdict uses the numbers of the registration line. The charts and the statistics show the result files."
            ),
        }
    # the weekly marks: each Sunday S0 .. E as a multiple of the mark of S0
    s0, e = pd.Timestamp(hl["start"], tz="UTC"), pd.Timestamp(hl["end"], tz="UTC")
    sundays = pd.date_range(s0, e, freq="7D")
    didx = _utc(daily["date"])

    def marks(col: str) -> list | None:
        v = pd.Series(daily[col].to_numpy(dtype=np.float64), index=didx).reindex(sundays).to_numpy()
        if not np.isfinite(v).any():
            return None
        base = v[0]
        if not (np.isfinite(base) and base > 0):
            return [None] * len(v)
        return rnd(v / base, 6)

    x = (1.0 + d).cumprod()
    x = pd.concat([pd.Series([1.0], index=pd.DatetimeIndex([s0])), x])
    stats = doc.get("stats") or {}

    def pick(block):
        if not isinstance(block, dict):
            return None
        return {k: block.get(k) for k in ("total_return", "sharpe", "ann_vol", "max_dd")}

    block = {
        "start": hl["start"],
        "end": hl["end"],
        "weekly": {
            "sundays": [str(s.date()) for s in sundays],
            "book": marks("book_equity"),
            "control": marks("control_equity"),
            "btc": marks("btc_equity"),
            "excess": rnd(x.reindex(sundays).to_numpy(dtype=np.float64), 6),
        },
        "windows": [{"start": w["start"], "end": w["end"], "sharpe": rnd(w["sharpe"], 6)} for w in h["windows"]],
        "reasons": list(hl.get("reasons") or []),
        "stats": {
            "book": pick(stats.get("book")),
            "control": pick(stats.get("control")),
            "btc": pick(stats.get("btc")),
            "excess": pick(stats.get("excess")),
        },
        "factor_numbers": doc.get("factor_numbers"),
        "missed": list(doc.get("missed") or []),
    }
    return block, differ


def _forward_part(
    t: dict, weeks: list[dict], fwd: dict, res: dict | None, alarms: list[dict], next_week_end
) -> tuple[dict, dict | None]:
    """The forward block of a part, after the check of each frozen week against rules.forward_test,
    and the "results_differ" entry for the weeks that an alarm book_week_changed names."""
    tid = t["test_id"]
    for w, p in zip(weeks, fwd["path"]):
        cs = w.get("cs") or {}
        for k in ("mean", "sd", "lrsd", "sigma", "lower", "upper"):
            if not _close(cs.get(k), p.get(k)):
                raise DataMismatch(f"{tid}: week {w['week']}: {k} does not follow from the frozen weeks")
        if cs.get("state") != p["state"]:
            raise DataMismatch(f"{tid}: week {w['week']}: the state does not follow from the frozen weeks")
    differ = None
    fw = res.get("forward") if res else None
    if fw is not None and weeks:
        got = pd.Series(fw["excess"].to_numpy(dtype=np.float64), index=_utc(fw["sunday"]))
        want = np.array([float(w["excess"]) for w in weeks], dtype=np.float64)
        have = got.reindex(_utc([w["sunday"] for w in weeks])).to_numpy(dtype=np.float64)
        bad = [int(weeks[i]["week"]) for i in np.flatnonzero(~LG.same_values(have, want))]
        # judge.py writes forward_weekly.parquet from a new replay; the frozen weeks stay, and an
        # alarm book_week_changed names each week whose returns changed
        named = {
            int(x)
            for a in alarms
            if a.get("type") == "book_week_changed"
            for x in ((a.get("detail") or {}).get("weeks") or [])
        }
        if [x for x in bad if x not in named]:
            raise DataMismatch(f"{tid}: the net excess return of {FORWARD_FILE} is not the excess of the ledger")
        if bad:
            differ = {
                "block": "forward",
                "alarm": "book_week_changed",
                "weeks": bad,
                "text": (
                    f"{FORWARD_FILE} differs from the frozen week{'s' if len(bad) > 1 else ''} "
                    f"{', '.join(str(x) for x in bad)}. The alarm book_week_changed records the change. "
                    "The verdict uses the frozen weeks of the ledger, and the page shows them."
                ),
            }
    f0 = _day(t["first_forward_sunday"])
    control = t["book_spec"]["body"]["control"]["kind"] != "zero"
    path = []
    for w, p in zip(weeks, fwd["path"]):
        path.append(
            {
                "week": int(w["week"]),
                "sunday": w["sunday"],
                "end": w["end"],
                "book_ret": rnd(w["book_ret"], 6),
                "control_ret": rnd(w["control_ret"], 6),
                "excess": rnd(w["excess"], 6),
                "btc_ret": rnd(w.get("btc_ret"), 6),
                "mean": rnd(p["mean"], 6),
                "sigma": rnd(p["sigma"], 6),
                "lower": rnd(p["lower"], 6),
                "upper": rnd(p["upper"], 6),
                "state": p["state"],
                "held": bool(w.get("held", True)),
                "final_gaps": [list(g) for g in (w.get("final_gaps") or [])],
            }
        )

    def equity(key: str) -> list:
        """The marks of the frozen weeks as a multiple of the first forward Sunday; unknown (None)
        from the first week without a return."""
        out: list[float | None] = [1.0]
        v: float | None = 1.0
        for w in weeks:
            r = w.get(key)
            ok = v is not None and r is not None and bool(np.isfinite(float(r)))
            v = v * (1.0 + float(r)) if ok and v is not None and r is not None else None
            out.append(v)
        return rnd(out, 6)

    block = {
        "first_forward_sunday": str(f0),
        "path": path,
        "equity": {
            "sundays": [str(f0)] + [w["end"] for w in weeks],
            "book": equity("book_ret"),
            "control": equity("control_ret") if control else None,
            "btc": equity("btc_ret"),
        },
        "next_week_end": next_week_end,
        "milestones": {
            "week26_end": str(f0 + WEEK * rules.FIRST_DECISION_WEEK),
            "week156_end": str(f0 + WEEK * rules.MAX_WEEKS),
        },
    }
    return block, differ


def _registration_view(t: dict, shas: list[str]) -> dict:
    """The registration line as the page shows it: without seq and prev_sha256, and without the
    history values of a values file (the ledger holds them; the view gives their count), with
    "line" (the line number in registrations.jsonl) and "sha256" (the hash of the line)."""
    view = {k: v for k, v in t.items() if k not in ("seq", "prev_sha256", "withdrawn", "line")}
    f = dict(view.get("factor") or {})
    if "history" in f:
        f["history"] = {"sundays": len(f["history"]), "in_ledger": True}
    view["factor"] = f
    seq = int(t["seq"])
    return {**view, "line": seq, "sha256": shas[seq - 1]}


# --------------------------------------------------------------------------- build


def build() -> dict:
    root = trials.root()
    present = _present(root)
    payload: dict = {
        "schema": SCHEMA,
        "available": True,
        "as_of": None,
        "rule": {**rules.rule_record(), "text": list(rules.RULE_TEXT)},
        "specs": _specs(),
        "words": _words(),
        "commands": dict(COMMANDS),
        "rows": [],
        "alarms": [],
        "code": None,
        "notes": dict(userfactor.NOTES),
        "counts": _empty_counts(),
        "expected_false_useful": _expected(trials.expected_line(0)),
    }
    if not present:
        payload["ledger"] = _ledger_block(False, None, [])
        return payload
    try:
        rec = trials.read()
        problems = trials.problems(rec)
    except LG.RecordBroken as exc:
        rec, problems = None, [str(exc)]
    if problems or rec is None:
        payload["ledger"] = _ledger_block(True, rec, problems)
        payload["counts"] = None
        payload["expected_false_useful"] = None
        return payload
    payload["ledger"] = _ledger_block(True, rec, [])
    sd = trials.status_data()
    regs = {t["test_id"]: t for t in trials.registered(rec)}
    shas = rec["registrations"]["shas"]
    rows, parts = [], {}
    inputs = {"formula": 0, "values_file": 0}
    for row in sd["tests"]:
        tid = row["id"]
        t = regs[tid]
        if t["withdrawn"] is None:
            inputs[t["input"]] += 1
        weeks = _week_lines(rec, tid)
        fwd = rules.forward_test([float(w["excess"]) for w in weeks], float(t["sigma_floor"]))
        v = rules.verdict(t["historical"]["result"], fwd)
        if (v["word"], v["by"]) != (row["verdict"]["word"], row["verdict"]["by"]):
            raise DataMismatch(f"{tid}: the verdict of the status is not rules.verdict of the ledger")
        if fwd["state"] != row["forward"]["state"] or fwd["weeks"] != row["forward"]["weeks"]:
            raise DataMismatch(f"{tid}: the forward state of the status is not the state of the frozen weeks")
        res = _results(tid)
        part = tid.lower()
        own = [a for a in rec["alarms"]["records"] if a.get("test_id") == tid]
        hist, h_diff = _historical_part(t, res, own) if res is not None else (None, None)
        forward, f_diff = _forward_part(t, weeks, fwd, res, own, row["forward"].get("next_week_end"))
        differ = [x for x in (h_diff, f_diff) if x is not None]
        rows.append(
            {
                **row,
                "verdict": {**row["verdict"], "badge": VERDICT_BADGES.get(row["verdict"]["word"], "neutral")},
                "forward": {**row["forward"], "badge": STATE_BADGES.get(row["forward"]["state"], "neutral")},
                "results": res is not None,
                "results_differ": differ,
                "line": int(t["seq"]),
                "part": part,
            }
        )
        withdrawal = t["withdrawn"]
        parts[part] = {
            "id": tid,
            "registration": _registration_view(t, shas),
            "withdrawal": None if withdrawal is None else {k: x for k, x in withdrawal.items() if k != "prev_sha256"},
            "historical": hist,
            "forward": forward,
            "verdict": {"word": v["word"], "by": v["by"]},
            "alarms": [
                {k: a.get(k) for k in ("type", "test_id", "source", "detected_at", "data_through", "detail")}
                | {"text": trials.alarm_text(a)}
                for a in own
            ],
            "results_differ": differ,
        }
    tot = sd["totals"]
    payload.update(
        {
            "as_of": _newest_bar() if rows else None,
            "rows": rows,
            "counts": {**tot, "inputs": inputs},
            "expected_false_useful": _expected(sd["expected_false_useful"]),
            "rule": sd["rule"],
            "code": sd["code"],
            "alarms": [{**a, "text": trials.alarm_text(a)} for a in sd["alarms"] if not a.get("test_id")],
        }
    )
    return with_parts(payload, parts) if parts else payload
