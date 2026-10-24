"""The trial ledger of your factor tests (data/trials/): check, register, the weekly logs, the
forward test, verify, withdraw and code changes. engine/judge.py is the command line.

A TEST is one registration of one factor with one book specification (book.py): a formula of
the library expression language, a library definition or a values file (userfactor.py). Its id
is FT-0001, FT-0002, ...

THE LEDGER. data/trials/ holds append-only, hash-chained JSONL files (ledger.py):
    registrations.jsonl     one line for each registration, withdrawal and code change
    values/okx.jsonl        one line for each logged Sunday of the formula tests on OKX data
    values/binance.jsonl    the same for the formula tests on Binance data
    values/file.jsonl       one line for each log-values call of a values-file test
    weeks.jsonl             one line for each complete forward week of a test (frozen)
    alarms.jsonl            one line for each alarm (an alarm is recorded once)
    head.json               the line count and the last hash of each chain
    results/<id>/           result files (historical.json and three Parquet tables). They are
                            derived and rewritable; verify --write-results writes them again.

CHECK (read-only). The historical test of the factor with the book: the book runs from the first
Sunday with at least book.MIN_NAMES eligible coins with a value (S0) to the last Sunday with a
complete week (E). rules.historical_test gives "useless", "pass" (unproven) or "none" (no
complete week). The output also gives the numbers of the factor, the correlation with each
registered test, the related tests (|correlation| >= RELATED_CORR) and a duplicate.

REGISTER. One registration line with the historical result, the sigma floor of the forward test
and the first forward Sunday F: the first Sunday after the newest data and after the day of the
registration (one clock read). A history without a complete week, a formula that is not proven
point in time, a duplicate of a registered test and code that the ledger does not hold are
refused. A test that the historical test calls "useless" can be registered.

THE FORWARD TEST. Week w of a test has the signal Sunday S_w = F + 7 (w - 1) and ends at
S_w + 7. It is complete when the OKX data reaches S_w + 7 with no pending gap of close or
funding on the days S_w + 1 .. S_w + 7 (a gap is final after GRACE_DAYS days: then the week
counts and a data_gap alarm names it), and, for a formula test, when the value line of S_w
holds the test. log_week() logs the cut values of the formula tests (the PIT check of each
Sunday), replays each book from F and freezes each new complete week in weeks.jsonl with the
state of rules.forward_test. A later replay that gives another number is an alarm; the
decision uses the frozen weeks only. The first decision is final.

A VALUES FILE is not proven point in time. log_values() logs its new Sundays. A value counts
from the first daily close after its log time (counts_from), and only in its own week. At a
Sunday without a used value the book has the control portfolio, and the week counts.

CODE CHANGES. Each registration and code_change line holds the hashes of CODE_FILES and the
values of config.py that the tests read (JUDGE_CONFIG). register and log-week refuse code that differs from
the latest of these lines; record_code_change() appends a code_change line.
log_week(accept_release=True) records a new VERSION by itself. A change of a user switch in
config.py (for example ALPHAFORGE_ENABLED) is not a code change.

EXIT CODES. 0: done. 1: a refusal or a broken ledger. 3: a new alarm.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

import book
import config as C
import data_io
import labels
import ledger as LG
import libfactors as lf
import rules
import userfactor as uf
import version

FOLDER = "trials"
CHAINS = {
    "registrations": "registrations.jsonl",
    "values_okx": "values/okx.jsonl",
    "values_binance": "values/binance.jsonl",
    "values_file": "values/file.jsonl",
    "weeks": "weeks.jsonl",
    "alarms": "alarms.jsonl",
}
ID_PREFIX = "FT-"
EXIT_OK, EXIT_REFUSED, EXIT_ALARM = 0, 1, 3
CODE_FILES = (
    "engine/judge.py",
    "engine/trials.py",
    "engine/rules.py",
    "engine/book.py",
    "engine/userfactor.py",
    "engine/labels.py",
    "engine/ledger.py",
    "engine/backtest.py",
    "engine/libfactors.py",
    "engine/data_io.py",
    "library/factorlab/expr.py",
    "library/factorlab/operators.py",
    "library/factorlab/fields.py",
    "library/factorlab/data/store.py",
)
# engine/config.py is not in CODE_FILES: you edit it (for example ALPHAFORGE_ENABLED). The
# values of config.py that the tests read are recorded instead (config_record).
JUDGE_CONFIG = ("EXCHANGE_ID", "FUNDING_FALLBACK_EXCHANGE_ID", "UNIVERSE")
DUP_VALUE_CORR = 0.999  # a mean within-Sunday Spearman above this: the same signed values
RELATED_CORR = 0.85  # |mean within-Sunday Spearman| at or above this: a related test
NAME_RE = book.NAME_RE
WEEK = timedelta(days=7)
DAY = timedelta(days=1)
ABOUT = "the line count and the sha256 of the last line of each chain of the trial ledger (judge.py)"
PIT_FILE_RULE = "A values file is not proven point in time. A forward value counts only from the first daily close after its log time."
LOG_RULE = (
    "A Sunday is logged for a formula test when the data of the source reaches the Sunday for each live coin. "
    "Each field that the test reads must have a value on each of the 7 days up to the Sunday. "
    "A gap that is less than 7 days old holds the log. An older gap does not hold it, and it gives an alarm."
)
WEEK_RULE = (
    "A week counts when the OKX data reaches its end. "
    "Each live coin must have its bar and its funding value on each day of the week. "
    "A gap that is less than 7 days old holds the week. An older gap does not hold it, and it gives an alarm."
)
HASH_OF = "each field of the panels cut at the Sunday (userfactor.field_sha256)"
HELD_GAP = "a value is missing; the week waits up to 7 days"
STATES_DECIDED = ("useful", "useless")


class RecordBroken(LG.RecordBroken):
    """The trial ledger does not pass a check. The command stops with exit 1."""

    prefix = "judge"


def count_text(n: int, one: str, many: str | None = None) -> str:
    """'1 test', '2 tests': the number and the noun in the form that agrees with it."""
    return f"{n} {one if n == 1 else (many or one + 's')}"


def refused(text: str) -> SystemExit:
    """A refusal of a command: exit 1 with the text."""
    return SystemExit(f"judge: {text}")


Log = Callable[[str], Any]


# --------------------------------------------------------------------------- the ledger


def root() -> Path:
    return Path(C.TRIALS)


def ledger() -> LG.Ledger:
    """The trial ledger in data/trials."""
    return LG.Ledger(root(), CHAINS, about=ABOUT, error=RecordBroken)


def read() -> dict:
    """Every chain of the trial ledger, checked link by link and against head.json (read-only).
    An absent folder gives empty chains."""
    return ledger().read()


def _head(rec: dict, key: str = "registrations") -> dict:
    c = rec[key]
    return {"lines": len(c["records"]), "sha256": c["shas"][-1] if c["shas"] else LG.GENESIS}


def _now() -> datetime:
    """The one clock read of a command (ledger.CLOCK), in UTC."""
    t = LG.CLOCK()
    return t.astimezone(UTC) if t.tzinfo is not None else t.replace(tzinfo=UTC)


def _stamp(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _day(x) -> date:
    return LG.as_day(x)


def _ts(x) -> pd.Timestamp:
    return LG.ts(x)


def _num(x, digits: int = 6) -> float | None:
    return LG.num(x, digits)


def _finite(x) -> float | None:
    """x as a full-precision float, or None (unknown) when x is not a finite number."""
    x = float(x)
    return x if np.isfinite(x) else None


# --------------------------------------------------------------------------- code and config


def config_record() -> dict:
    """The values of config.py that the tests read: {values, sym_files, sha256}."""
    values = {k: getattr(C, k) for k in JUDGE_CONFIG}
    values["UNIVERSE"] = list(values["UNIVERSE"])
    sym_files = [C.sym_to_file(s) for s in C.UNIVERSE]
    return {
        "values": values,
        "sym_files": sym_files,
        "sha256": LG.sha_text(LG.canonical({"values": values, "sym_files": sym_files})),
    }


def code_record() -> dict:
    """{files: {rel: sha256} of CODE_FILES, config: config_record(), sha256 of both}."""
    files = lf.code_fingerprint(CODE_FILES)["files"]
    cfg = config_record()
    return {"files": files, "config": cfg, "sha256": LG.sha_text(LG.canonical({"files": files, "config": cfg}))}


def latest_code(rec: dict) -> dict | None:
    """{line, code, version} of the latest registration or code_change line, or None."""
    out = None
    for r in rec["registrations"]["records"]:
        if r.get("kind") in ("registration", "code_change"):
            out = {"line": r["seq"], "code": r["code"], "version": r.get("version")}
    return out


def code_drift(rec: dict, now: dict | None = None) -> dict | None:
    """None when the code has the fingerprint of the latest code line (or no line exists), else
    {line, recorded_sha256, now_sha256, changed {file: {recorded, now}}, config_changed}."""
    last = latest_code(rec)
    if last is None:
        return None
    now = code_record() if now is None else now
    if now["sha256"] == last["code"]["sha256"]:
        return None
    old = last["code"]["files"]
    changed = {
        f: {"recorded": old.get(f), "now": now["files"].get(f)}
        for f in sorted(set(old) | set(now["files"]))
        if old.get(f) != now["files"].get(f)
    }
    return {
        "line": last["line"],
        "recorded_sha256": last["code"]["sha256"],
        "now_sha256": now["sha256"],
        "changed": changed,
        "config_changed": last["code"]["config"]["sha256"] != now["config"]["sha256"],
        "version": last["version"],
    }


DRIFT_TEXT = "Record the code change first: judge.py record-code-change --reason TEXT"


# --------------------------------------------------------------------------- the registered tests


def registered(rec: dict) -> list[dict]:
    """The registration lines in order, each with "withdrawn" (the withdrawal line or None) and
    "line" (its line number)."""
    out, by_id = [], {}
    for r in rec["registrations"]["records"]:
        if r.get("kind") == "registration":
            t = {**r, "withdrawn": None, "line": r["seq"]}
            by_id[r["test_id"]] = t
            out.append(t)
        elif r.get("kind") == "withdrawal":
            if r.get("test_id") not in by_id:
                raise RecordBroken(f"registrations.jsonl line {r['seq']} withdraws an unknown test {r.get('test_id')}")
            by_id[r["test_id"]]["withdrawn"] = r
    return out


def active(rec: dict) -> list[dict]:
    return [t for t in registered(rec) if t["withdrawn"] is None]


def related(rec: dict) -> dict[str, list[str]]:
    """test id -> the related tests: the ids in its own registration line and the later
    registration lines that name it."""
    out: dict[str, list[str]] = {}
    for t in registered(rec):
        out.setdefault(t["test_id"], [])
        for other in t.get("related") or []:
            out[t["test_id"]].append(other)
            out.setdefault(other, []).append(t["test_id"])
    return {k: sorted(set(v)) for k, v in out.items()}


def _week_lines(rec: dict, test_id: str) -> list[dict]:
    return [w for w in rec["weeks"]["records"] if w.get("kind") == "book_week" and w.get("test_id") == test_id]


def decision(rec: dict, test_id: str) -> dict | None:
    """The deciding week line of a test, or None."""
    for w in _week_lines(rec, test_id):
        if (w.get("cs") or {}).get("state") in STATES_DECIDED:
            return w
    return None


def forward_of(rec: dict, t: dict) -> dict:
    """rules.forward_test of the frozen weekly excess returns of a test and its sigma floor."""
    xs = [float(w["excess"]) for w in _week_lines(rec, t["test_id"])]
    return rules.forward_test(xs, float(t["sigma_floor"]))


def verdict_of(rec: dict, t: dict) -> dict:
    fwd = forward_of(rec, t)
    return rules.verdict(t["historical"]["result"], fwd)


# --------------------------------------------------------------------------- values of the tests


def _frame(values: dict, symbols: list[str] | None = None) -> pd.DataFrame:
    """{sunday: [20 values]} -> a DataFrame (UTC Sunday index x the 20 symbols)."""
    cols = symbols or [C.sym_to_file(s) for s in C.UNIVERSE]
    if not values:
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC"), dtype=np.float64)
    days = sorted(values)
    arr = np.array([LG.from_list(values[d]) for d in days], dtype=np.float64)
    return pd.DataFrame(arr, index=pd.DatetimeIndex([_ts(d) for d in days]).as_unit("ns"), columns=cols)


def _sources() -> dict:
    """A cache of loaded panel sources for one command."""
    return {}


def _source(cache: dict, name: str) -> dict:
    if name not in cache:
        cache[name] = uf.load_source(name)
    return cache[name]


def file_logs(rec: dict, test_id: str) -> dict:
    """The first log of each Sunday of a values-file test: {sunday: {values, counts_from, used,
    seq}} in Sunday order."""
    out: dict[str, dict] = {}
    for r in rec["values_file"]["records"]:
        if r.get("kind") != "file_values" or r.get("test_id") != test_id:
            continue
        for s, v in r["values"].items():
            if s not in out:
                out[s] = {"values": v, "counts_from": r["counts_from"][s], "used": r["used"][s], "seq": r["seq"]}
    return dict(sorted(out.items()))


def formula_logs(rec: dict, t: dict) -> dict:
    """The logged cut values of a formula test: {sunday: {values, seq}} in Sunday order."""
    key = f"values_{t['factor']['source']}"
    out = {}
    for r in rec[key]["records"]:
        if r.get("kind") == "week_values" and t["test_id"] in r.get("values", {}):
            out[r["sunday"]] = {"values": r["values"][t["test_id"]], "seq": r["seq"]}
    return dict(sorted(out.items()))


def registered_values(rec: dict, *, daily: bool = True, through: date | None = None, cache: dict | None = None) -> dict:
    """test id -> the values (date x the 20 symbols) of each registered, not withdrawn test.

    A formula: computed on the full panels of its source (each day, or its Sundays with
    daily=False; cut at `through` when given). A values file: its history values and its logged
    forward values (Sundays only). A formula of a source whose data is not on disk is left out."""
    cache = _sources() if cache is None else cache
    out: dict[str, pd.DataFrame] = {}
    for t in active(rec):
        if t["input"] == "formula":
            src_name = t["factor"]["source"]
            try:
                src = _source(cache, src_name)
            except SystemExit:
                continue
            panels = src["panels"] if through is None else uf.cut(src["panels"], through)
            idx = panels["close"].index
            v = pd.DataFrame(uf.full_values(t["factor"]["expression"], panels), index=idx, columns=src["symbols"])
            out[t["test_id"]] = v if daily else v.iloc[lf.sunday_positions(idx)]
        else:
            vals = dict(t["factor"]["history"])
            for s, x in file_logs(rec, t["test_id"]).items():
                vals.setdefault(s, x["values"])
            f = _frame(vals)
            if through is not None:
                f = f[f.index <= _ts(through)]
            out[t["test_id"]] = f
    return out


def mean_spearman(a: pd.DataFrame, b: pd.DataFrame, min_names: int = book.MIN_NAMES) -> float | None:
    """The mean within-Sunday Spearman correlation of two value frames on their common Sundays
    with at least `min_names` coins with a value in both (ties share the mean rank). None
    without such a Sunday."""
    cols = [C.sym_to_file(s) for s in C.UNIVERSE]
    ai = pd.DatetimeIndex(a.index)
    bi = pd.DatetimeIndex(b.index)
    common = ai[ai.isin(bi)]
    common = common[common.dayofweek == 6]
    if not len(common):
        return None
    A = lf.snap_ties(a.reindex(index=common, columns=cols).to_numpy(dtype=np.float64))
    B = lf.snap_ties(b.reindex(index=common, columns=cols).to_numpy(dtype=np.float64))
    ok = np.isfinite(A) & np.isfinite(B)
    keep = ok.sum(axis=1) >= min_names
    if not keep.any():
        return None
    ra = pd.DataFrame(np.where(ok, A, np.nan)[keep]).rank(axis=1).to_numpy()
    rb = pd.DataFrame(np.where(ok, B, np.nan)[keep]).rank(axis=1).to_numpy()
    da = ra - np.nanmean(ra, axis=1, keepdims=True)
    db = rb - np.nanmean(rb, axis=1, keepdims=True)
    num = np.nansum(da * db, axis=1)
    den = np.sqrt(np.nansum(da * da, axis=1) * np.nansum(db * db, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    r = r[np.isfinite(r)]
    return float(r.mean()) if len(r) else None


def _signed(values: pd.DataFrame, sign: int) -> dict:
    """{sunday: [20 signed values]} (+0.0 so that -0.0 has one form)."""
    return {str(d.date()): LG.finite_list(sign * row + 0.0) for d, row in zip(values.index, values.to_numpy())}


def values_sha256(values: pd.DataFrame) -> str:
    """sha256 of the canonical JSON of the rows of a value frame (the history values)."""
    return LG.sha_text(LG.canonical([LG.finite_list(row) for row in values.to_numpy(dtype=np.float64)]))


# --------------------------------------------------------------------------- the history of a test


def market() -> dict:
    """The OKX closes, the funding and the eligible coins of the books (one read).

    The rows are every UTC day from the first bar to the last bar. A day on which no coin has
    a bar is a row of NaN closes (the engine books no return on it and on the next day), so
    each Sunday of a week has its equity mark."""
    prices = data_io.close_matrix()
    idx = pd.DatetimeIndex(prices.index)
    days = pd.date_range(idx.min(), idx.max(), freq="D", name=idx.name)
    if len(days) != len(idx):
        prices = prices.reindex(days)
        idx = pd.DatetimeIndex(prices.index)
    funding = data_io.funding_matrix(idx, prices.columns)
    return {"prices": prices, "funding": funding, "eligible": labels.eligible(idx)}


def history_span(values: pd.DataFrame, eligible: pd.DataFrame, okx_complete: date) -> tuple | None:
    """(S0, E) of the historical test, or None without a complete week (module docstring)."""
    v = values.copy()
    v.index = pd.DatetimeIndex(v.index)
    el = eligible.reindex(index=v.index, columns=v.columns, fill_value=False).to_numpy(dtype=bool)
    n = (np.isfinite(v.to_numpy(dtype=np.float64)) & el).sum(axis=1)
    ok = np.flatnonzero(n >= book.MIN_NAMES)
    if not len(ok):
        return None
    s0 = v.index[ok[0]]
    have = np.flatnonzero(np.isfinite(v.to_numpy(dtype=np.float64)).any(axis=1))
    s_last = v.index[have[-1]]
    last_sunday = _ts(okx_complete)
    last_sunday -= pd.Timedelta(days=(last_sunday.dayofweek + 1) % 7)
    e = min(last_sunday, s_last + pd.Timedelta(days=7))
    if e < s0 + pd.Timedelta(days=7):
        return None
    return s0, e


def _stats_of(res) -> dict:
    s = res.stats
    keys = ("total_return", "sharpe", "ann_vol", "max_dd", "fees_total", "funding_total", "n_rebalances", "n_trades")
    return {k: (float(s[k]) if isinstance(s.get(k), (int, float)) and np.isfinite(s[k]) else None) for k in keys}


def _btc_stats(eq: pd.Series) -> dict:
    """The BTC hold numbers of the daily BTC marks of a run. Each number is None (unknown) when a
    mark is not finite. The engine scales the BTC hold by its first close. Without the BTC close
    of the first day, each mark is NaN."""
    eq = eq.astype(np.float64)
    v = eq.to_numpy()
    if len(v) < 2 or not (np.isfinite(v).all() and (v > 0).all()):
        return {"total_return": None, "sharpe": None, "ann_vol": None, "max_dd": None}
    d = eq.pct_change().iloc[1:]
    return {
        "total_return": float(v[-1] / v[0] - 1.0),
        "sharpe": rules.sharpe(d.to_numpy()),
        "ann_vol": float(d.std() * np.sqrt(rules.ANNUAL_DAYS)) if len(d) > 1 else None,
        "max_dd": float((eq / eq.cummax() - 1.0).min()),
    }


def _btc_weekly(run, wk: pd.DataFrame) -> pd.DataFrame:
    """wk with btc_ret NaN (unknown) at each signal Sunday S where the BTC mark of S or S + 7 is
    not finite. book.weekly_returns gives 0.0 there, which is not a BTC return."""
    bench = run.book.equity["benchmark"].astype(np.float64)
    bench.index = pd.DatetimeIndex(bench.index)
    s = pd.DatetimeIndex(wk.index)
    a = bench.reindex(s).to_numpy(dtype=np.float64)
    b = bench.reindex(s + pd.Timedelta(days=7)).to_numpy(dtype=np.float64)
    known = np.isfinite(a) & np.isfinite(b) & (a > 0)
    if known.all():
        return wk
    out = wk.copy()
    out["btc_ret"] = np.where(known, out["btc_ret"].to_numpy(dtype=np.float64), np.nan)
    return out


def run_history(spec: book.BookSpec, values: pd.DataFrame, s0, e, mk: dict) -> dict:
    """The historical test: the book run S0 .. E, its weekly table and rules.historical_test."""
    # one time unit for check and verify, so the result tables of the two are equal
    s0, e = _ts(s0).as_unit("ns"), _ts(e).as_unit("ns")
    sundays = pd.date_range(s0, e - pd.Timedelta(days=7), freq="7D")
    v = values.copy()
    v.index = pd.DatetimeIndex(v.index)
    v = v.reindex(sundays)
    run = book.run_book(spec, v, start=s0, end=e, prices=mk["prices"], funding=mk["funding"], eligible=mk["eligible"])
    wk = _btc_weekly(run, book.weekly_table(run))
    hist = rules.historical_test(
        book.daily_excess(run), wk["excess"], held=wk["held"], fallback_floor=spec.forward.sigma_floor_fallback
    )
    btc = _btc_stats(run.book.equity["benchmark"])
    return {"run": run, "weekly": wk, "hist": hist, "values": v, "btc": btc}


def history_numbers(h: dict, okx: dict) -> dict:
    """{stats, factor_numbers} of a historical test h (run_history): the book, control, excess and
    BTC numbers, and the factor numbers on the history values (the Sundays S0 .. E - 7). check
    and verify --write-results both use it, so historical.json has the same numbers."""
    run = h["run"]
    excess = {k: h["hist"].get(k) for k in ("total_return", "sharpe", "max_dd")}
    # the SD of each daily excess return S0 + 1 .. E (as the Sharpe ratio of rules.historical_test)
    d = book.daily_excess(run).astype(np.float64)
    excess["ann_vol"] = float(d.std(ddof=1) * np.sqrt(rules.ANNUAL_DAYS)) if len(d) > 1 else None
    return {
        "stats": {
            "book": _stats_of(run.book),
            "control": _stats_of(run.control) if run.control is not None else None,
            "excess": excess,
            "btc": h["btc"],
        },
        "factor_numbers": uf.factor_numbers(h["values"], okx),
    }


def _hist_record(h: dict, span, vals_sha: str | None, btc: dict | None, missed: list) -> dict:
    keys_num = (
        "total_return",
        "sharpe",
        "share_positive",
        "weekly_mean",
        "weekly_sd",
        "weekly_lrsd",
        "weekly_ac1",
        "max_dd",
    )
    out = {
        "result": h["result"],
        "word": h["word"],
        "reasons": list(h["reasons"]),
        "start": str(span[0].date()) if span else None,
        "end": str(span[1].date()) if span else None,
        "weeks": int(h["weeks"]),
        "held_weeks": int(h["held_weeks"]),
        "n_windows": int(h["n_windows"]),
        "n_positive": int(h["n_positive"]),
        "values_sha256": vals_sha,
        "btc": {k: _num(btc.get(k)) for k in ("total_return", "sharpe", "max_dd")} if btc else None,
        "missed": list(missed),
    }
    for k in keys_num:
        out[k] = _num(h.get(k))
    return out


# --------------------------------------------------------------------------- check


def _input_kind(inp: dict) -> str:
    if "values_file" in inp:
        return "values_file"
    return "formula"


def _related_and_duplicate(
    rec: dict, kind: str, factor: dict, spec: book.BookSpec, hist_values: pd.DataFrame, signed_sha: str, cache: dict
) -> dict:
    """The correlation with each registered test, the related tests and the duplicate."""
    regs = {t["test_id"]: t for t in active(rec)}
    vals = registered_values(rec, daily=False, cache=cache)
    corr, rel, dup = [], [], None
    for tid, t in regs.items():
        u = vals.get(tid)
        c = mean_spearman(hist_values, u) if u is not None and len(hist_values) else None
        if c is not None:
            corr.append({"test_id": tid, "corr": c, "abs_corr": abs(c)})
            if abs(c) >= RELATED_CORR:
                rel.append(tid)
        if t["book_spec"]["book_key"] != spec.book_key or dup is not None:
            continue
        same_id = False
        if kind == "formula" and t["input"] == "formula":
            f = t["factor"]
            same_id = (
                f["expression_sha256"] == factor["expression_sha256"]
                and f["source"] == factor["source"]
                and t["book_spec"]["body"]["book"]["sign"] == spec.sign
            )
        elif kind == "values_file" and t["input"] == "values_file":
            same_id = t.get("signed_values_sha256") == signed_sha
        signed = None if c is None else c * spec.sign * int(t["book_spec"]["body"]["book"]["sign"])
        if same_id or (signed is not None and signed > DUP_VALUE_CORR):
            dup = tid
    corr.sort(key=lambda x: -x["abs_corr"])
    return {"correlations": corr, "related": sorted(rel), "duplicate": dup}


def check(inp: dict, spec: book.BookSpec, *, novelty: bool = False, now: datetime | None = None) -> dict:
    """The read-only check of a factor with a book (module docstring). inp is one of
    {"formula": text, "source": name | None}, {"library": id, "source": name | None} or
    {"values_file": path}. now: the one clock read of the command. Writes nothing."""
    now = _now() if now is None else now
    kind = _input_kind(inp)
    library_id = None
    f: dict = {}
    if kind == "formula":
        # the static check comes first: a refused formula reads no data
        expr = inp.get("formula")
        if "library" in inp:
            lib = uf.library_formula(inp["library"])
            library_id, expr = lib["factor_id"], lib["expression"]
        f = uf.validate_formula(cast(str, expr), inp.get("source"))
    rec = read()
    cache = _sources()
    okx = _source(cache, "okx")
    if kind == "formula":
        src = okx if f["source"] == "okx" else _source(cache, "binance")
        fv = uf.formula_values(f["canonical"], src)
        values = pd.DataFrame(fv["values"], index=fv["sundays"], columns=fv["symbols"])
        factor = {
            "expression": f["canonical"],
            "expression_sha256": f["expression_sha256"],
            "source": f["source"],
            "raw_fields": f["raw_fields"],
            "notes": f["notes"],
            "library_id": library_id,
        }
        pit = {k: fv["pit"][k] for k in ("status", "sundays", "cells", "finite_cells", "rule")}
        pit_detail = fv["pit"]
        through = {"okx": str(okx["data_through"]), "binance": str(src["data_through"]) if src is not okx else None}
        complete = {
            "okx": str(okx["complete_through"]),
            "binance": str(src["complete_through"]) if src is not okx else None,
        }
        src_through = src["data_through"]
    else:
        path = Path(inp["values_file"])
        values, report = uf.read_values_file(path, through=okx["data_through"])
        factor = {
            "file_name": report["file_name"],
            "file_sha256": report["file_sha256"],
            "rows": report["rows"],
            "sunday_rows": report["sunday_rows"],
            "first_date": report["first_date"],
            "last_date": report["last_date"],
        }
        pit = {
            "status": "not_proven",
            "sundays": int(len(values)),
            "cells": int(values.size),
            "finite_cells": int(np.isfinite(values.to_numpy()).sum()),
            "rule": PIT_FILE_RULE,
        }
        pit_detail = {**pit, "file_report": report}
        through = {"okx": str(okx["data_through"]), "binance": None}
        complete = {"okx": str(okx["complete_through"]), "binance": None}
        src_through = okx["data_through"]
    mk = market()
    span = history_span(values, mk["eligible"], okx["complete_through"])
    out: dict[str, Any] = {
        "input": kind,
        "factor": factor,
        "pit": pit,
        "pit_detail": pit_detail,
        "spec": {
            "name": spec.name,
            "version": spec.version,
            "body": spec.body,
            "sha256": spec.sha256,
            "book_key": spec.book_key,
            "source": spec.source,
        },
        "data_through": through,
        "complete_through": complete,
        "now": _stamp(now),
    }
    if span is None:
        hist = rules.historical_test(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            held=pd.Series(dtype=bool),
            fallback_floor=spec.forward.sigma_floor_fallback,
        )
        hv = values.iloc[0:0]
        out.update(
            {
                "historical": _hist_record(hist, None, None, None, []),
                "historical_full": hist,
                "_run": None,
                "stats": None,
                "factor_numbers": None,
            }
        )
    else:
        s0, e = span
        h = run_history(spec, values, s0, e, mk)
        hv = h["values"]
        run = h["run"]
        out.update(
            {
                "historical": _hist_record(h["hist"], span, values_sha256(hv), h["btc"], run.missed),
                "historical_full": h["hist"],
                "_run": run,
                "_weekly": h["weekly"],
                **history_numbers(h, okx),
            }
        )
    hist_full = out["historical_full"]
    out["sigma_floor"] = float(hist_full["sigma_floor"])
    out["sigma_floor_source"] = hist_full["sigma_floor_source"]
    out["_values"] = values
    out["_history_values"] = hv
    signed_rows = hv if len(hv) else values
    out["signed_values_sha256"] = LG.sha_text(LG.canonical(_signed(signed_rows, spec.sign)))
    rd = _related_and_duplicate(rec, kind, factor, spec, hv, out["signed_values_sha256"], cache)
    out.update(rd)
    out["largest"] = rd["correlations"][0] if rd["correlations"] else None
    out["library_corr"] = None
    if novelty:
        try:
            from factorfinder.incumbents import library_corr  # type: ignore[import-not-found,unused-ignore]
        except ImportError:
            out["library_corr"] = {"available": False}
        else:
            out["library_corr"] = library_corr(values)
    first = max(_day(okx["data_through"]), _day(src_through), now.date())
    out["first_forward_sunday"] = str(LG.next_sunday_after(first))
    return out


def public(result: dict) -> dict:
    """The JSON part of a check result (no frame, no run)."""
    return {k: v for k, v in result.items() if not k.startswith("_") and k not in ("historical_full", "pit_detail")}


# --------------------------------------------------------------------------- register


def _next_id(rec: dict) -> str:
    nums = [int(t["test_id"][len(ID_PREFIX) :]) for t in registered(rec)]
    return f"{ID_PREFIX}{max(nums, default=0) + 1:04d}"


def _default_name(inp: dict, result: dict) -> str:
    f = result["factor"]
    if f.get("library_id"):
        return str(f["library_id"])
    if result["input"] == "formula":
        return f"formula-{f['expression_sha256'][:8]}"
    return f"file-{f['file_sha256'][:8]}"


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _json_bytes(obj) -> bytes:
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (pd.Timestamp, date)):
            return str(x.date()) if isinstance(x, pd.Timestamp) else str(x)
        return x

    return (json.dumps(clean(obj), indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def write_history_results(test_id: str, result: dict) -> None:
    """results/<id>/historical.json, historical_daily.parquet and historical_weekly.parquet."""
    run = result.get("_run")
    if run is None:
        return
    lid = test_id.lower()
    led = ledger()
    hist = dict(result["historical_full"])
    doc = {
        "test_id": test_id,
        "historical": hist,
        "start": result["historical"]["start"],
        "end": result["historical"]["end"],
        "stats": result["stats"],
        "factor_numbers": result["factor_numbers"],
        "missed": list(run.missed),
    }
    led.write_file(f"{lid}/historical.json", _json_bytes(doc))
    eq = run.book.equity
    daily = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(eq.index),
            "book_equity": eq["equity"].to_numpy(dtype=np.float64),
            "control_equity": (
                run.control.equity["equity"].reindex(eq.index).to_numpy(dtype=np.float64)
                if run.control is not None
                else np.full(len(eq), np.nan)
            ),
            "btc_equity": eq["benchmark"].to_numpy(dtype=np.float64),
        }
    )
    led.write_file(f"{lid}/historical_daily.parquet", _parquet_bytes(daily))
    led.write_file(f"{lid}/historical_weekly.parquet", _parquet_bytes(result["_weekly"].reset_index(drop=True)))


def registration_line(
    test_id: str, name: str, note: str | None, result: dict, spec: book.BookSpec, now: datetime
) -> dict:
    factor = dict(result["factor"])
    if result["input"] == "values_file":
        factor["history"] = _signed(result["_history_values"], 1)
    return {
        "kind": "registration",
        "test_id": test_id,
        "name": name,
        "registered_at": _stamp(now),
        "note": note,
        "input": result["input"],
        "factor": factor,
        "pit": result["pit"],
        "book_spec": {"body": spec.body, "sha256": spec.sha256, "book_key": spec.book_key, "source": spec.source},
        "signed_values_sha256": result["signed_values_sha256"],
        "related": list(result["related"]),
        "rule": rules.rule_record(),
        "historical": result["historical"],
        "sigma_floor": _num(result["sigma_floor"]),
        "sigma_floor_source": result["sigma_floor_source"],
        "data_through": result["data_through"],
        "complete_through": result["complete_through"],
        "first_forward_sunday": result["first_forward_sunday"],
        "code": code_record(),
        "version": version.VERSION,
    }


def pit_why(p: dict) -> str:
    """Why a formula with the PIT status "fail" or "empty" is not proven point in time (UI text)."""
    if p["status"] == "fail":
        return f"the value at the Sunday cut {p.get('first_diff_sunday')} is not equal to the full computation"
    return "the formula has no value at any Sunday"


def register(
    inp: dict,
    spec: book.BookSpec,
    *,
    name: str | None,
    note: str | None,
    dry_run: bool = False,
    log: Log = print,
) -> int:
    """Register a test (module docstring). Returns 0; a refusal raises SystemExit (exit 1)."""
    now = _now()
    rec = read()
    if code_drift(rec) is not None:
        raise refused(f"not registered. The code differs from the ledger. {DRIFT_TEXT}")
    if name is not None and not NAME_RE.fullmatch(name):
        raise refused('the name must have 1 to 64 characters: letters, digits, "_", "." or "-".')
    if note is not None and not note.strip():
        raise refused("the note is empty.")
    result = check(inp, spec, now=now)
    if result["input"] == "formula" and result["pit"]["status"] != "proven":
        raise refused(f"not registered. The formula is not proven point in time: {pit_why(result['pit_detail'])}.")
    if result["historical"]["result"] == "none":
        raise refused(
            "Not registered. The history has no complete week of values. "
            "Give values for at least one Sunday before the registration week."
        )
    if result["duplicate"]:
        raise refused(f"Not registered. Test {result['duplicate']} has the same book and the same factor values.")
    test_id = _next_id(rec)
    line = registration_line(test_id, name or _default_name(inp, result), note, result, spec, now)
    if dry_run:
        log(json.dumps(line, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False))
        log("Dry run: nothing is written.")
        return EXIT_OK
    led = ledger()
    led.append("registrations", [line])
    write_history_results(test_id, result)
    log(f"Registered {test_id} ({line['name']}).")
    log(f"  Historical test: {result['historical']['word']}.")
    log(
        f"  First forward Sunday: {line['first_forward_sunday']}. Sigma floor {line['sigma_floor']:.6g} ({line['sigma_floor_source']})."
    )
    log(f"  Related registered tests: {', '.join(line['related']) or 'none'}.")
    log("Next steps (run these commands from the repository root):")
    log("  cd engine")
    if result["input"] == "values_file":
        log(f"  .venv/bin/python -B judge.py log-values {test_id} --values <file.csv>")
    log("  .venv/bin/python -B judge.py log-week")
    log(f"  .venv/bin/python -B judge.py status {test_id}")
    return EXIT_OK


# --------------------------------------------------------------------------- alarms


def _alarm(kind: str, test_id: str | None, source: str, detail: dict, now: datetime, data_through) -> dict:
    body = {"type": kind, "test_id": test_id, "source": source, "detail": detail}
    return {
        "kind": "alarm",
        **body,
        "key": LG.sha_text(LG.canonical(body)),
        "detected_at": _stamp(now),
        "data_through": str(data_through),
    }


def known_alarms(rec: dict) -> set:
    return {a["key"] for a in rec["alarms"]["records"]}


def alarm_text(a: dict) -> str:
    d = a["detail"]
    who = f" {a['test_id']}" if a.get("test_id") else ""
    if a["type"] == "data_gap":
        return (
            f"data_gap ({a['source']}): missing data: {d['coin']} {d['field']} from {d['first_day']}. The week counts."
        )
    if a["type"] == "inputs_changed":
        return (
            f"inputs_changed ({a['source']}): the data of {count_text(len(d['sundays']), 'logged Sunday')} changed "
            f"({d['sundays'][0]} to {d['sundays'][-1]}). Changed fields: {', '.join(d['changed_fields'])}."
        )
    return f"{a['type']}{who} ({a['source']}): {LG.canonical(d)[:300]}"


def _gap_alarms(src: dict, final: list, now: datetime) -> list[dict]:
    """One data_gap alarm for each run of missing days of the final gaps."""
    seen, out = set(), []
    for coin, field, day in final:
        first = uf.gap_start(src, coin, field, day)
        key = (coin, field, first)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            _alarm(
                "data_gap",
                None,
                src["source"],
                {"coin": coin, "field": field, "first_day": first},
                now,
                src["data_through"],
            )
        )
    return out


# --------------------------------------------------------------------------- the formula value logs


def _recheck(rec: dict, src: dict, fulls: dict, now: datetime) -> list[dict]:
    """Alarms for the logged Sundays of one source (read-only): data_shorter, inputs_changed and
    value_changed. fulls: {test id: full values of its formula on the source panels}."""
    source, panels = src["source"], src["panels"]
    idx = panels["close"].index
    shorter, changed, values = [], {}, {}
    for w in rec[f"values_{source}"]["records"]:
        if w.get("kind") != "week_values":
            continue
        s = _day(w["sunday"])
        if s > src["complete_through"] or _ts(s) not in idx:
            shorter.append(str(s))
            continue
        fields = list((w.get("inputs") or {}).get("fields_sha256") or {})
        c = uf.cut({f: panels[f] for f in fields if f in panels}, s)
        now_h = uf.field_sha256(c)
        old = w["inputs"]["fields_sha256"]
        diff = sorted(f for f in old if old.get(f) != now_h.get(f))
        if diff:
            changed[str(s)] = diff
        pos = int(idx.get_loc(_ts(s)))
        bad = [
            tid
            for tid, v in sorted(w["values"].items())
            if tid in fulls and not LG.same_values(LG.from_list(v), fulls[tid][pos]).all()
        ]
        if bad:
            values[str(s)] = bad
    out = []
    dt = src["data_through"]
    if shorter:
        out.append(_alarm("data_shorter", None, source, {"sundays": shorter}, now, dt))
    if changed:
        detail = {
            "sundays": sorted(changed),
            "changed_fields": sorted({f for v in changed.values() for f in v}),
            "by_sunday": changed,
        }
        out.append(_alarm("inputs_changed", None, source, detail, now, dt))
    if values:
        out.append(_alarm("value_changed", None, source, {"tests": values}, now, dt))
    return out


def _log_formula_values(
    rec: dict,
    src: dict,
    tests: list[dict],
    decided: dict,
    through: date | None,
    now: datetime,
    code_sha: str,
    log: Log,
) -> tuple[list[dict], dict]:
    """Log the cut values of the formula tests of one source (module docstring). Returns
    (alarms, held {test id: (sunday, reason)}). Appends the value lines."""
    source = src["source"]
    key = f"values_{source}"
    led = ledger()
    logged: dict[str, set] = {t["test_id"]: set(formula_logs(rec, t)) for t in tests}
    fulls: dict[str, np.ndarray] = {}
    by_expr: dict[str, np.ndarray] = {}
    for t in tests:
        e = t["factor"]["expression"]
        if e not in by_expr:
            by_expr[e] = uf.full_values(e, src["panels"])
        fulls[t["test_id"]] = by_expr[e]
    alarms = _recheck(rec, src, fulls, now)
    held: dict[str, tuple[str, str]] = {}
    live = [t for t in tests if t["withdrawn"] is None and decided.get(t["test_id"]) is None]
    if not live:
        return alarms, held
    first = min(_day(t["first_forward_sunday"]) for t in live)
    for s in uf.closed_sundays(first, src["complete_through"], through):
        ready, finals, fields = [], [], {"close"}
        for t in live:
            tid = t["test_id"]
            f0 = _day(t["first_forward_sunday"])
            if f0 > s or str(s) in logged[tid] or tid in held:
                continue
            need = uf.closed_sundays(f0, s - WEEK)
            if any(str(d) not in logged[tid] for d in need):
                continue
            expr = t["factor"]["expression"]
            g = uf.gaps(src, s - timedelta(days=6), s, uf.read_fields([expr]))
            finals += g["final"]
            if g["pending"]:
                c, fld, day = g["pending"][0]
                held[tid] = (str(s), f"a value is missing ({c} {fld} on {day}); the week waits up to 7 days")
                continue
            sv = uf.sunday_value(expr, src, s, fulls[tid])
            if not sv["equal"]:
                detail = {"sunday": str(s), "diff_cells": int((~LG.same_values(sv["values"], sv["full"])).sum())}
                alarms.append(_alarm("pit_mismatch", tid, source, detail, now, src["data_through"]))
                held[tid] = (str(s), "pit_mismatch")
                continue
            ready.append((t, sv))
            fields |= set(uf.read_fields([expr]))
        alarms += _gap_alarms(src, finals, now)
        if not ready:
            continue
        cut = uf.cut({f: src["panels"][f] for f in sorted(fields)}, s)
        rec_now = read()
        line: dict[str, Any] = {
            "kind": "week_values",
            "sunday": str(s),
            "source": source,
            "logged_at": _stamp(now),
            "data_through": str(src["data_through"]),
            "complete_through": str(src["complete_through"]),
            "closed_check": {
                "rule": LOG_RULE,
                "fields": sorted(fields),
                "final_gaps": [list(x) for x in sorted(set(finals))],
            },
            "symbols": list(src["symbols"]),
            "inputs": {"fields_sha256": uf.field_sha256(cut), "hash_of": HASH_OF},
            "code_sha256": code_sha,
            "registrations": _head(rec_now),
            "pit_check": {
                "rule": uf.PIT_RULE,
                "full_panels_through": str(src["data_through"]),
                "cells": sum(sv["cells"] for _, sv in ready),
                "finite_cells": sum(sv["finite_cells"] for _, sv in ready),
                "equal": True,
            },
            "values": {t["test_id"]: LG.finite_list(sv["values"]) for t, sv in ready},
        }
        if source == "binance":
            line["inputs"]["cleaning_freeze_sha256"] = lf.cleaning_freeze_record()["sha256"]
        led.append(key, [line])
        for t, _sv in ready:
            logged[t["test_id"]].add(str(s))
        log(f"{source}: logged {s} for {', '.join(t['test_id'] for t, _ in ready)}")
    for tid, (sunday, why) in held.items():
        log(f"{tid}: held at {sunday}: {why}")
    return alarms, held


# --------------------------------------------------------------------------- the weeks


def _week_complete(okx: dict, s: date, through: date | None) -> tuple[bool, list, str | None]:
    """(complete, final gaps, the reason when not complete) of the week with signal Sunday s."""
    end = s + WEEK
    if okx["data_through"] < end:
        return False, [], f"the week ends on {end}; the OKX data ends on {okx['data_through']}"
    if through is not None and end > through:
        return False, [], f"the week ends after {through}"
    g = uf.gaps(okx, s + DAY, end, ["close", "funding_rate"])
    if g["pending"]:
        c, f, day = g["pending"][0]
        return False, g["final"], f"a value is missing ({c} {f} on {day}); the week waits up to 7 days"
    return True, g["final"], None


def replay_inputs(rec: dict, t: dict) -> tuple[pd.DataFrame, dict | None, dict]:
    """(values, schedule, refs) of the forward replay of a test: the logged values that the book
    may use, the trade schedule (a values file) and {sunday: {chain, seq}} of the value lines."""
    f0 = _day(t["first_forward_sunday"])
    if t["input"] == "formula":
        logs = formula_logs(rec, t)
        chain = CHAINS[f"values_{t['factor']['source']}"]
        refs = {s: {"chain": chain, "seq": x["seq"]} for s, x in logs.items()}
        return _frame({s: x["values"] for s, x in logs.items()}), None, refs
    logs = file_logs(rec, t["test_id"])
    used = {s: x for s, x in logs.items() if x["used"] and _day(s) >= f0}
    refs = {s: {"chain": CHAINS["values_file"], "seq": x["seq"]} for s, x in used.items()}
    return _frame({s: x["values"] for s, x in used.items()}), {s: x["counts_from"] for s, x in used.items()}, refs


def _schedule(used_from: dict, f0: date, last_sunday: date) -> dict:
    """{trade date: signal Sunday} of a values-file test up to the signal Sunday last_sunday."""
    out = {}
    for s in uf.closed_sundays(f0, last_sunday):
        key = used_from.get(str(s))
        trade = _day(key) if key is not None else s + DAY
        out[_ts(trade)] = _ts(s)
    return out


def _targets(spec: book.BookSpec, values: pd.DataFrame, start, end, eligible: pd.DataFrame, schedule) -> dict:
    """{signal Sunday: target weights} of the book, in the order of the engine's calls."""
    s0, e0 = _ts(start), _ts(end)
    sundays = pd.date_range(s0, e0 - pd.Timedelta(days=7), freq="7D")
    v = values.copy()
    v.index = pd.DatetimeIndex(v.index)
    ff = book.factor_frame(v.reindex(sundays), eligible)
    ff = book._every_sunday(ff, sundays)
    by_date = {d: g.set_index("symbol") for d, g in ff.groupby("date")}
    bb = book.book_weights_fn(spec)
    out = {}
    trades = sorted(schedule.items()) if schedule else [(s + pd.Timedelta(days=1), s) for s in sundays]
    for t, sig in trades:
        if not (s0 < t <= e0):
            continue
        cs = by_date.get(sig)
        if cs is not None and not cs.empty:
            out[sig] = pd.Series(bb(cs), dtype=float)
    return out


def _matrix_sha(df: pd.DataFrame, name: str) -> str:
    import hashlib

    h = hashlib.sha256()
    uf._field_update(h, name, df)
    return h.hexdigest()


def replay(t: dict, rec: dict, mk: dict, end_sunday: date) -> dict:
    """The forward replay of a test from F to the end of the week with signal Sunday end_sunday:
    {run, weekly, values, schedule, refs}."""
    f0 = _day(t["first_forward_sunday"])
    spec = book.spec_from_body(t["book_spec"]["body"], source=t["book_spec"].get("source") or "ledger")
    values, used_from, refs = replay_inputs(rec, t)
    schedule = _schedule(used_from, f0, end_sunday) if used_from is not None else None
    end = end_sunday + WEEK
    run = book.run_book(
        spec,
        values,
        start=_ts(f0),
        end=_ts(end),
        prices=mk["prices"],
        funding=mk["funding"],
        eligible=mk["eligible"],
        schedule=schedule,
    )
    return {
        "run": run,
        "weekly": _btc_weekly(run, book.weekly_table(run)),
        "values": values,
        "schedule": schedule,
        "refs": refs,
        "spec": spec,
    }


def _week_detail(rp: dict, s: pd.Timestamp, targets: dict) -> tuple[dict, dict | None]:
    run = rp["run"]
    end = s + pd.Timedelta(days=7)
    daily = run.book.daily
    didx = pd.DatetimeIndex(daily.index)
    inside = (didx > s) & (didx <= end)
    trades = run.book.trades
    tdates = pd.DatetimeIndex(trades["date"]) if len(trades) else pd.DatetimeIndex([], tz="UTC")
    tin = sorted(set(tdates[(tdates > s) & (tdates <= end)]))
    trade_date = tin[0] if tin else None
    g_day = trade_date if trade_date is not None else s + pd.Timedelta(days=1)
    gross = float(daily["gross"].get(g_day, np.nan)) if g_day in daily.index else None
    w = targets.get(s)
    row = rp["weekly"].loc[s]
    b = {
        "cost": _num(float(daily["cost"].to_numpy()[inside].sum())),
        "funding_usd": _num(float(daily["funding_usd"].to_numpy()[inside].sum())),
        "turnover": _num(row["book_turnover"]),
        "gross": _num(gross),
        "n_long": int((w > 0).sum()) if w is not None else 0,
        "n_short": int((w < 0).sum()) if w is not None else 0,
        "trade_date": str(trade_date.date()) if trade_date is not None else None,
    }
    ctl = None
    if run.control is not None:
        cd = run.control.daily
        cidx = pd.DatetimeIndex(cd.index)
        cin = (cidx > s) & (cidx <= end)
        ctl = {
            "cost": _num(float(cd["cost"].to_numpy()[cin].sum())),
            "funding_usd": _num(float(cd["funding_usd"].to_numpy()[cin].sum())),
        }
    return b, ctl


def _cs_record(path_row: dict) -> dict:
    return {k: _num(path_row.get(k)) for k in ("mean", "sd", "lrsd", "sigma", "lower", "upper")} | {
        "state": path_row["state"]
    }


def _test_weeks(
    t: dict, rec: dict, okx: dict, mk: dict, through: date | None, now: datetime, code_sha: str
) -> tuple[list[dict], list[dict], str | None]:
    """(new book_week lines, alarms, the reason why the next week waits) of one test."""
    tid = t["test_id"]
    frozen = _week_lines(rec, tid)
    if decision(rec, tid) is not None:
        return [], [], None
    f0 = _day(t["first_forward_sunday"])
    k = len(frozen)
    logged = set(formula_logs(rec, t)) if t["input"] == "formula" else None
    weeks: list[tuple[int, date, list]] = []
    alarms: list[dict] = []
    wait = None
    w = k + 1
    while w <= rules.MAX_WEEKS:
        s = f0 + WEEK * (w - 1)
        ok, final, why = _week_complete(okx, s, through)
        if not ok:
            wait = why
            break
        if logged is not None and str(s) not in logged:
            wait = f"the value of {s} is not logged"
            break
        alarms += _gap_alarms(okx, final, now)
        weeks.append((w, s, final))
        w += 1
    if not weeks:
        return [], alarms, wait
    end_sunday = weeks[-1][1]
    rp = replay(t, rec, mk, end_sunday)
    wk = rp["weekly"]
    if frozen:
        idx = [_ts(x["sunday"]) for x in frozen]
        got_b = wk["book_ret"].reindex(idx).to_numpy(dtype=np.float64)
        got_c = wk["control_ret"].reindex(idx).to_numpy(dtype=np.float64)
        want_b = np.array([x["book_ret"] for x in frozen], dtype=np.float64)
        want_c = np.array([x["control_ret"] for x in frozen], dtype=np.float64)
        bad = ~(LG.same_values(got_b, want_b) & LG.same_values(got_c, want_c))
        if bad.any():
            detail = {"weeks": [int(frozen[i]["week"]) for i in np.flatnonzero(bad)]}
            alarms.append(_alarm("book_week_changed", tid, "okx", detail, now, okx["data_through"]))
    targets = _targets(rp["spec"], rp["values"], _ts(f0), _ts(end_sunday + WEEK), mk["eligible"], rp["schedule"])
    xs = [float(x["excess"]) for x in frozen]
    floor = float(t["sigma_floor"])
    rec_now = read()
    reg_head = _head(rec_now)
    lines = []
    prices, funding = mk["prices"], mk["funding"]
    pidx = pd.DatetimeIndex(prices.index)
    for w, s, final in weeks:
        st = _ts(s)
        row = wk.loc[st]
        x = float(row["excess"])
        xs.append(x)
        fwd = rules.forward_test(xs, floor)
        b, ctl = _week_detail(rp, st, targets)
        span = (pidx >= _ts(f0)) & (pidx <= st + pd.Timedelta(days=7))
        line = {
            "kind": "book_week",
            "test_id": tid,
            "week": w,
            "sunday": str(s),
            "end": str(s + WEEK),
            "logged_at": _stamp(now),
            "book_ret": float(row["book_ret"]),
            "control_ret": float(row["control_ret"]),
            "excess": x,
            "btc_ret": _finite(row["btc_ret"]),
            "book": b,
            "control": ctl,
            "values_ref": rp["refs"].get(str(s)),
            "held": bool(row["held"]),
            "final_gaps": [list(g) for g in final],
            "inputs": {
                "prices_sha256": _matrix_sha(prices[span], "close"),
                "funding_sha256": _matrix_sha(funding[span], "funding"),
            },
            "code_sha256": code_sha,
            "registrations": reg_head,
            "cs": _cs_record(fwd["path"][-1]),
        }
        lines.append(line)
        if fwd["state"] in STATES_DECIDED:
            break
    return lines, alarms, wait


def forward_table(t: dict, rec: dict, mk: dict) -> pd.DataFrame | None:
    """The weekly table of the forward replay up to the last frozen week (None without one)."""
    frozen = _week_lines(rec, t["test_id"])
    if not frozen:
        return None
    rp = replay(t, rec, mk, _day(frozen[-1]["sunday"]))
    return rp["weekly"].reset_index(drop=True)


def _write_forward(t: dict, rec: dict, mk: dict) -> None:
    table = forward_table(t, rec, mk)
    if table is not None:
        ledger().write_file(f"{t['test_id'].lower()}/forward_weekly.parquet", _parquet_bytes(table))


# --------------------------------------------------------------------------- log-week


def _record_release(rec: dict, now: datetime, log: Log) -> None:
    last = latest_code(rec)
    if last is None:
        return
    reason = f"Factorlab {version.VERSION}"
    _append_code_change(rec, reason, [], now)
    log(f"Recorded a code change: {reason}.")


def _append_alarms(new: list[dict], log: Log) -> int:
    if not new:
        return EXIT_OK
    ledger().append("alarms", new)
    for a in new:
        log(f"ALARM {alarm_text(a)}")
    return EXIT_ALARM


def log_week(*, through: date | None = None, accept_release: bool = False, log: Log = print) -> int:
    """Log the formula values of each closed Sunday and freeze each new complete forward week
    (module docstring). Returns 0, 3 (a new alarm) or raises SystemExit (exit 1)."""
    now = _now()
    rec = read()
    tests = active(rec)
    if not tests:
        log("No test is registered. Nothing to log.")
        return EXIT_OK
    drift = code_drift(rec)
    if drift is not None:
        if accept_release and drift["version"] != version.VERSION:
            _record_release(rec, now, log)
            rec = read()
        else:
            raise refused(f"log-week refused. The code differs from the ledger. {DRIFT_TEXT}")
    code_sha = code_record()["sha256"]
    known = known_alarms(rec)
    alarms: list[dict] = []
    decided = {t["test_id"]: (decision(rec, t["test_id"]) or {}).get("sunday") for t in tests}
    for source in uf.SOURCES:
        mine = [t for t in registered(rec) if t["input"] == "formula" and t["factor"]["source"] == source]
        if not any(t["withdrawn"] is None for t in mine):
            continue
        try:
            src = uf.load_source(source)
        except SystemExit as exc:
            log(f"{source}: the data is not available ({exc}). The tests of this source wait.")
            continue
        if source == "binance":
            lf.cleaning_freeze_record()
        found, _held = _log_formula_values(rec, src, mine, decided, through, now, code_sha, log)
        alarms += found
        rec = read()
    okx = uf.load_source("okx")
    mk = market()
    for t in active(rec):
        rec = read()
        lines, found, wait = _test_weeks(t, rec, okx, mk, through, now, code_sha)
        alarms += found
        if lines:
            ledger().append("weeks", lines)
            rec = read()
            _write_forward(t, rec, mk)
            last = lines[-1]
            log(
                f"{t['test_id']}: froze week{'s' if len(lines) > 1 else ''} "
                f"{lines[0]['week']}{'' if len(lines) == 1 else ' to ' + str(last['week'])}; state {last['cs']['state']}"
            )
        elif wait and decision(rec, t["test_id"]) is None:
            log(f"{t['test_id']}: the next week waits: {wait}.")
    new, seen = [], set(known)
    for a in alarms:
        if a["key"] not in seen:
            new.append(a)
            seen.add(a["key"])
    return _append_alarms(new, log)


# --------------------------------------------------------------------------- log-values


def counts_from(sunday: date, now: datetime) -> date:
    """The first daily close after the log time, and never before the Monday of the Sunday."""
    return max(sunday + DAY, now.date() + DAY)


def log_values(test_id: str, path: Path, log: Log = print) -> int:
    """Log the new Sundays of a values-file test (module docstring). Returns 0; a refusal raises
    SystemExit (exit 1). A changed value of a logged cell appends an alarm and is refused."""
    now = _now()
    rec = read()
    by_id = {t["test_id"]: t for t in registered(rec)}
    t = by_id.get(test_id)
    if t is None:
        raise refused(f"{test_id} is not a registered test.")
    if t["withdrawn"] is not None:
        raise refused(f"{test_id} is withdrawn.")
    if t["input"] != "values_file":
        raise refused(f"{test_id} is a formula test. Factorlab computes its values (judge.py log-week).")
    if decision(rec, test_id) is not None:
        raise refused(f"{test_id} has its forward decision. It takes no more values.")
    frame, report = uf.read_values_file(Path(path), through=now.date(), drop_no_bar=False)
    f0 = _day(t["first_forward_sunday"])
    logged = file_logs(rec, test_id)
    before = [str(d.date()) for d in frame.index if d.date() < f0]
    if before:
        log(
            f"Not used: {count_text(len(before), 'Sunday')} before the first forward Sunday {f0} ({before[0]} to {before[-1]})."
        )
    new_vals, new_from, new_used, changed, skipped, extra = {}, {}, {}, [], [], []
    for d, row in zip(frame.index, frame.to_numpy(dtype=np.float64)):
        s = d.date()
        if s < f0:
            continue
        key = str(s)
        if key in logged:
            old = LG.from_list(logged[key]["values"])
            have = np.isfinite(old)
            both = have & np.isfinite(row)
            if not LG.same_values(row[both], old[both]).all():
                changed.append(key)
            elif (np.isfinite(row) & ~have).any():
                extra.append(key)
            else:
                skipped.append(key)
            continue
        if not np.isfinite(row).any():
            continue
        cf = counts_from(s, now)
        new_vals[key] = LG.finite_list(row)
        new_from[key] = str(cf)
        new_used[key] = bool(cf <= s + WEEK)
    if changed:
        detail = {"sundays": changed, "file_sha256": report["file_sha256"]}
        a = _alarm("file_value_changed", test_id, "file", detail, now, now.date())
        if a["key"] not in known_alarms(rec):
            ledger().append("alarms", [a])
        raise refused(
            f"refused. The file gives another value for a logged cell of {', '.join(changed[:10])}. "
            "Only the first log of a Sunday counts. An alarm is recorded."
        )
    for key in skipped:
        log(f"{key}: already logged with the same values (skipped).")
    for key in extra:
        log(f"{key}: not used: only the first log of a Sunday counts.")
    if not new_vals:
        log("No new Sunday to log.")
        return EXIT_OK
    line = {
        "kind": "file_values",
        "test_id": test_id,
        "logged_at": _stamp(now),
        "file": {"name": report["file_name"], "sha256": report["file_sha256"], "rows": report["rows"]},
        "symbols": [C.sym_to_file(s) for s in C.UNIVERSE],
        "values": new_vals,
        "counts_from": new_from,
        "used": new_used,
        "code_sha256": code_record()["sha256"],
        "registrations": _head(rec),
    }
    ledger().append("values_file", [line])
    for key in new_vals:
        s = _day(key)
        cf = _day(new_from[key])
        if not new_used[key]:
            log(f"{key}: too late: not used.")
        elif cf == s + DAY:
            log(f"{key}: counts from Monday {cf}.")
        else:
            log(f"{key}: counts from {cf}.")
    return EXIT_OK


# --------------------------------------------------------------------------- withdraw and code changes


def withdraw(test_id: str, reason: str, log: Log = print) -> int:
    now = _now()
    rec = read()
    by_id = {t["test_id"]: t for t in registered(rec)}
    t = by_id.get(test_id)
    if t is None:
        raise refused(f"{test_id} is not a registered test.")
    if t["withdrawn"] is not None:
        raise refused(f"{test_id} is already withdrawn.")
    if not reason or not reason.strip():
        raise refused("a withdrawal needs a reason (--reason).")
    f0 = _day(t["first_forward_sunday"])
    try:
        okx_through: date | None = uf.load_source("okx")["data_through"]
    except SystemExit:
        okx_through = None
    latest = max(d for d in (okx_through, now.date()) if d is not None)
    if latest >= f0:
        raise refused(
            f"{test_id} cannot be withdrawn now: its first forward Sunday is {f0}, "
            f"and the date or the data has reached {latest}."
        )
    holds = any(test_id in r.get("values", {}) for k in ("values_okx", "values_binance") for r in rec[k]["records"])
    holds = holds or any(r.get("test_id") == test_id for r in rec["values_file"]["records"])
    holds = holds or bool(_week_lines(rec, test_id))
    if holds:
        raise refused(f"{test_id} has logged values. It cannot be withdrawn.")
    line = {
        "kind": "withdrawal",
        "test_id": test_id,
        "withdrawn_at": _stamp(now),
        "reason": reason.strip(),
        "data_through": {"okx": str(okx_through) if okx_through else None},
        "first_forward_sunday": str(f0),
    }
    ledger().append("registrations", [line])
    log(f"Withdrew {test_id} before its first forward Sunday {f0}.")
    return EXIT_OK


def _append_code_change(rec: dict, reason: str, notes: list[str], now: datetime) -> dict:
    last = latest_code(rec)
    if last is None:
        raise refused("the ledger holds no registration. There is no code fingerprint to change.")
    code = code_record()
    old = last["code"]["files"]
    changed = {
        f: {"recorded": old.get(f), "now": code["files"].get(f)}
        for f in sorted(set(old) | set(code["files"]))
        if old.get(f) != code["files"].get(f)
    }
    line: dict[str, Any] = {
        "kind": "code_change",
        "changed_at": _stamp(now),
        "reason": reason.strip(),
        "version": version.VERSION,
        "previous": {"line": last["line"], "code_sha256": last["code"]["sha256"]},
        "changed_files": changed,
        "unchanged_files": {f: h for f, h in code["files"].items() if f not in changed},
        "config_changed": last["code"]["config"]["sha256"] != code["config"]["sha256"],
        "code": code,
    }
    if notes:
        line["notes"] = notes
    ledger().append("registrations", [line])
    return line


def record_code_change(reason: str, notes: list[str] | None = None, log: Log = print) -> int:
    now = _now()
    rec = read()
    if not any(r.get("kind") == "registration" for r in rec["registrations"]["records"]):
        raise refused("the ledger holds no registration. There is no code fingerprint to change.")
    if not reason or not reason.strip():
        raise refused("a code change needs a reason (--reason).")
    notes = [n.strip() for n in (notes or [])]
    if any(not n for n in notes):
        raise refused("a note is empty (--note).")
    if code_drift(rec) is None:
        last = cast(dict, latest_code(rec))
        raise refused(f"the code has the fingerprint of registrations.jsonl line {last['line']}. Nothing to record.")
    line = _append_code_change(rec, reason, notes, now)
    changed = ", ".join(line["changed_files"]) or "none"
    log(
        f"Recorded a code change after line {line['previous']['line']}: files {changed}; "
        f"the test values of config.py changed: {'yes' if line['config_changed'] else 'no'}."
    )
    return EXIT_OK


# --------------------------------------------------------------------------- status


def expected_line(n: int) -> dict:
    value = rules.expected_false_useful(n)
    text = (
        'Expected number of false "useful" verdicts, for a false-pass rate of 10 % for each test: '
        f"0.1 x N = {value:.1f} (N = {n} registered {'test that is' if n == 1 else 'tests that are'} not withdrawn)."
    )
    return {"alpha": rules.ALPHA, "n": n, "value": value, "text": text}


def _held(t: dict, rec: dict, cache: dict) -> dict | None:
    """{sunday, reason} when a pending gap or a pit_mismatch holds a formula test, or when the data
    of its source ends before the OKX data reaches the next Sunday (for example a Binance panel
    that is not updated), else None."""
    if t["input"] != "formula" or t["withdrawn"] is not None or decision(rec, t["test_id"]) is not None:
        return None
    try:
        src = _source(cache, t["factor"]["source"])
    except SystemExit:
        return None
    f0 = _day(t["first_forward_sunday"])
    logs = formula_logs(rec, t)
    nxt = f0 + WEEK * len(logs)
    lagging = nxt > src["complete_through"]
    if lagging and src["data_through"] < nxt:
        if src["source"] == "okx":
            return None  # the next Sunday is not in the data yet: the test waits, it is not held
        try:
            okx_ct = _source(cache, "okx")["complete_through"]
        except SystemExit:
            return None
        if nxt > okx_ct:
            return None
        venue = "Binance" if src["source"] == "binance" else src["source"]
        return {
            "sunday": str(nxt),
            "reason": f"the {venue} data ends on {src['data_through']}; the test waits for its update",
        }
    # Here the data of the source reaches nxt. When complete_through is before nxt, a live coin
    # lags behind the others: its bar has a pending gap. Then the test is held.
    for a in rec["alarms"]["records"]:
        if (
            a.get("type") == "pit_mismatch"
            and a.get("test_id") == t["test_id"]
            and a["detail"].get("sunday") == str(nxt)
        ):
            return {"sunday": str(nxt), "reason": "pit_mismatch"}
    g = uf.gaps(src, nxt - timedelta(days=6), nxt, uf.read_fields([t["factor"]["expression"]]))
    if g["pending"] or lagging:
        return {"sunday": str(nxt), "reason": HELD_GAP}
    return None


def status_data(test_id: str | None = None, *, data: bool = True) -> dict:
    """The read-only view of the ledger (the source of judge.py status and of the dashboard).
    data=False leaves out the fields that read the market data (held)."""
    rec = read()
    regs = registered(rec)
    rel = related(rec)
    cache = _sources()
    rows = []
    states = dict.fromkeys(rules.FORWARD_STATES, 0)
    verdicts = {"useful": 0, "useless": 0, "unproven": 0}
    for t in regs:
        fwd = forward_of(rec, t)
        v = rules.verdict(t["historical"]["result"], fwd)
        tid = t["test_id"]
        f0 = _day(t["first_forward_sunday"])
        if t["withdrawn"] is None:
            states[fwd["state"]] += 1
            verdicts[
                "useless" if v["word"].startswith("useless") else "useful" if v["word"] == "useful" else "unproven"
            ] += 1
        if test_id is not None and tid != test_id:
            continue
        nxt = f0 + WEEK * fwd["weeks"]
        f = t["factor"]
        spec = t["book_spec"]["body"]
        rows.append(
            {
                "id": tid,
                "name": t["name"],
                "input": t["input"],
                "source": f.get("source"),
                "library_id": f.get("library_id"),
                "expression": f.get("expression"),
                "file": {"name": f["file_name"], "sha256": f["file_sha256"]} if t["input"] == "values_file" else None,
                "pit": t["pit"]["status"],
                "notes": list(f.get("notes") or []),
                "spec": {
                    "name": spec["name"],
                    "version": spec["version"],
                    "sha256": t["book_spec"]["sha256"],
                    "direction": spec["book"]["direction"],
                    "sign": spec["book"]["sign"],
                    "control": spec["control"]["kind"],
                    "fee_bps": spec["costs"]["fee_bps"],
                    "slippage_bps": spec["costs"]["slippage_bps"],
                },
                "registered_at": t["registered_at"],
                "first_forward_sunday": str(f0),
                "withdrawn": t["withdrawn"] is not None,
                "historical": {
                    k: t["historical"].get(k)
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
                        "weekly_sd",
                        "weekly_lrsd",
                        "weekly_ac1",
                    )
                }
                | {"sigma_floor": t["sigma_floor"], "sigma_floor_source": t["sigma_floor_source"]},
                "forward": {
                    "state": fwd["state"],
                    "weeks": fwd["weeks"],
                    "decided_week": fwd["decided_week"],
                    "mean": fwd["mean"],
                    "sigma": fwd["sigma"],
                    "lower": fwd["lower"],
                    "upper": fwd["upper"],
                    "next_week_end": None if fwd["state"] in STATES_DECIDED else str(nxt + WEEK),
                    "held": _held(t, rec, cache) if data else None,
                },
                "related": rel.get(tid, []),
                "verdict": {"word": v["word"], "by": v["by"]},
                "alarms": sum(1 for a in rec["alarms"]["records"] if a.get("test_id") == tid),
            }
        )
    n = sum(1 for t in regs if t["withdrawn"] is None)
    drift = code_drift(rec)
    last = latest_code(rec)
    return {
        "folder": f"data/{FOLDER}",
        "chains": {CHAINS[k]: len(rec[k]["records"]) for k in CHAINS},
        "tests": rows,
        "totals": {
            "registered": len(regs),
            "withdrawn": len(regs) - n,
            "n": n,
            "states": states,
            "verdicts": verdicts,
        },
        "expected_false_useful": expected_line(n),
        "rule": {**rules.rule_record(), "text": list(rules.RULE_TEXT)},
        "code": None
        if last is None
        else {"line": last["line"], "matches": drift is None, "changed": sorted(drift["changed"]) if drift else []},
        "alarms": [
            {k: a.get(k) for k in ("type", "test_id", "source", "detected_at", "detail")}
            for a in rec["alarms"]["records"]
        ],
    }


# --------------------------------------------------------------------------- verify


def _problems_registrations(rec: dict) -> list[str]:
    out: list[str] = []
    want_n = 0
    prev_code = None
    any_value: dict[str, bool] = {}
    for k in ("values_okx", "values_binance"):
        for r in rec[k]["records"]:
            for tid in r.get("values", {}):
                any_value.setdefault(tid, True)
    for r in rec["values_file"]["records"]:
        any_value.setdefault(r.get("test_id"), True)
    for r in rec["weeks"]["records"]:
        any_value.setdefault(r.get("test_id"), True)
    by_id: dict[str, dict] = {}
    for r in rec["registrations"]["records"]:
        n, kind = r["seq"], r.get("kind")
        where = f"registrations.jsonl line {n}"
        if kind == "registration":
            want_n += 1
            if r.get("test_id") != f"{ID_PREFIX}{want_n:04d}":
                out.append(f"{where}: the id {r.get('test_id')} is not {ID_PREFIX}{want_n:04d}")
            by_id[r["test_id"]] = r
            bs = r["book_spec"]
            try:
                spec = book.spec_from_body(bs["body"])
            except book.SpecError as exc:
                out.append(f"{where}: {exc}")
            else:
                if spec.sha256 != bs["sha256"] or spec.book_key != bs["book_key"]:
                    out.append(f"{where}: the book specification does not have its recorded hashes")
            if r["rule"].get("rule_version") != rules.RULE_VERSION:
                out.append(f"{where}: the rule version {r['rule'].get('rule_version')} is not {rules.RULE_VERSION}")
            elif r["rule"] != rules.rule_record():
                out.append(f"{where}: the rule record is not the rule of {rules.RULE_VERSION}")
            dt = r["data_through"]
            days = [_day(v) for v in dt.values() if v] + [_day(r["registered_at"])]
            if r["first_forward_sunday"] != str(LG.next_sunday_after(max(days))):
                out.append(f"{where}: the first forward Sunday does not follow from the data and the registration time")
            if r["historical"]["result"] not in ("useless", "pass"):
                out.append(f"{where}: the historical result {r['historical']['result']!r} cannot be registered")
            if r["input"] == "values_file":
                hist = _frame(r["factor"]["history"])
                if values_sha256(hist) != r["historical"]["values_sha256"]:
                    out.append(f"{where}: the history values do not have their recorded hash")
                sign = int(r["book_spec"]["body"]["book"]["sign"])
                if LG.sha_text(LG.canonical(_signed(hist, sign))) != r["signed_values_sha256"]:
                    out.append(f"{where}: the signed history values do not have their recorded hash")
            prev_code = r
        elif kind == "withdrawal":
            t = by_id.get(r.get("test_id"))
            if t is None:
                out.append(f"{where}: the withdrawal names an unknown test")
                continue
            f0 = _day(t["first_forward_sunday"])
            okx = (r.get("data_through") or {}).get("okx")
            if _day(r["withdrawn_at"]) >= f0 or (okx and _day(okx) >= f0):
                out.append(f"{where}: {r['test_id']} was withdrawn on or after its first forward Sunday")
            if any_value.get(r["test_id"]):
                out.append(f"{where}: {r['test_id']} was withdrawn, but a value or week line holds it")
        elif kind == "code_change":
            if prev_code is None:
                out.append(f"{where}: a code change without a registration before it")
                continue
            old, new = prev_code["code"], r["code"]
            diff = {
                f: {"recorded": old["files"].get(f), "now": new["files"].get(f)}
                for f in sorted(set(old["files"]) | set(new["files"]))
                if old["files"].get(f) != new["files"].get(f)
            }
            if r["previous"] != {"line": prev_code["seq"], "code_sha256": old["sha256"]}:
                out.append(f"{where}: the code change does not name the fingerprint of line {prev_code['seq']}")
            if r["changed_files"] != diff or r["unchanged_files"] != {
                f: h for f, h in new["files"].items() if f not in diff
            }:
                out.append(f"{where}: the changed and unchanged files do not agree with the two fingerprints")
            if r["config_changed"] != (old["config"]["sha256"] != new["config"]["sha256"]):
                out.append(f"{where}: config_changed does not agree with the two fingerprints")
            if not str(r.get("reason", "")).strip():
                out.append(f"{where}: the code change has no reason")
            prev_code = r
        else:
            out.append(f"{where}: unknown kind {kind!r}")
    return out


def _problems_values(rec: dict) -> list[str]:
    out: list[str] = []
    regs = {t["test_id"]: t for t in registered(rec)}
    reg_lines = rec["registrations"]["records"]
    symbols = [C.sym_to_file(s) for s in C.UNIVERSE]
    for source in uf.SOURCES:
        key = f"values_{source}"
        seen: dict[str, list[date]] = {}
        for r in rec[key]["records"]:
            where = f"{CHAINS[key]} line {r['seq']}"
            if r.get("kind") != "week_values" or r.get("source") != source:
                out.append(f"{where}: not a {source} value line")
                continue
            s = _day(r["sunday"])
            pit = r.get("pit_check") or {}
            if pit.get("equal") is not True or "cells" not in pit:
                out.append(f"{where}: the PIT check is not recorded")
            if r.get("symbols") != symbols or any(len(v) != len(symbols) for v in r["values"].values()):
                out.append(f"{where}: the coins are not the 20 symbols of config.UNIVERSE")
            k = r["registrations"]["lines"]
            if (
                k > len(reg_lines)
                or (rec["registrations"]["shas"][k - 1] if k else LG.GENESIS) != r["registrations"]["sha256"]
            ):
                out.append(f"{where}: the registration head that it records is not in registrations.jsonl")
            for tid in r["values"]:
                t = regs.get(tid)
                if t is None or t["input"] != "formula" or t["factor"]["source"] != source or t["line"] > k:
                    out.append(f"{where}: {tid} is not a registered formula test of {source} at that line")
                    continue
                if t["withdrawn"] is not None:
                    out.append(f"{where}: {tid} is withdrawn")
                days = seen.setdefault(tid, [])
                want = _day(t["first_forward_sunday"]) + WEEK * len(days)
                if s != want:
                    out.append(f"{where}: {tid} has the Sunday {s}, the ledger wants {want}")
                days.append(s)
            if (
                source == "binance"
                and (r.get("inputs") or {}).get("cleaning_freeze_sha256") != lf.CLEANING_FREEZE["sha256"]
            ):
                out.append(f"{where}: its cleaning rules are not the frozen set")
    # file_values: each Sunday of a test is logged once, and counts_from and used follow from
    # logged_at. A Sunday can come after a later Sunday in the chain: a Sunday without a value is
    # not written, and its later log counts from after its week (used is false).
    seen_f: dict[str, set] = {}
    for r in rec["values_file"]["records"]:
        where = f"{CHAINS['values_file']} line {r['seq']}"
        tid = r.get("test_id")
        t = regs.get(tid)
        if r.get("kind") != "file_values" or t is None or t["input"] != "values_file":
            out.append(f"{where}: not a value line of a registered values-file test")
            continue
        logged_at = datetime.strptime(r["logged_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        days = sorted(r["values"])
        if sorted(r["counts_from"]) != days or sorted(r["used"]) != days:
            out.append(f"{where}: counts_from and used do not have the Sundays of the values")
        for s in days:
            if s in seen_f.setdefault(tid, set()):
                out.append(f"{where}: {tid} logs {s} a second time")
            seen_f[tid].add(s)
            if _day(s) < _day(t["first_forward_sunday"]):
                out.append(f"{where}: {tid} logs {s}, before its first forward Sunday")
            cf = counts_from(_day(s), logged_at)
            if r["counts_from"].get(s) != str(cf) or r["used"].get(s) != (cf <= _day(s) + WEEK):
                out.append(f"{where}: counts_from or used of {s} does not follow from logged_at")
    return out


def _problems_weeks(rec: dict) -> list[str]:
    out: list[str] = []
    regs = {t["test_id"]: t for t in registered(rec)}
    by_test: dict[str, list[dict]] = {}
    for r in rec["weeks"]["records"]:
        where = f"weeks.jsonl line {r['seq']}"
        t = regs.get(r.get("test_id"))
        if r.get("kind") != "book_week" or t is None:
            out.append(f"{where}: not a week line of a registered test")
            continue
        lst = by_test.setdefault(t["test_id"], [])
        w = len(lst) + 1
        if r["week"] != w:
            out.append(f"{where}: {t['test_id']} has week {r['week']}, the ledger wants {w}")
        s = _day(t["first_forward_sunday"]) + WEEK * (w - 1)
        if r["sunday"] != str(s) or r["end"] != str(s + WEEK):
            out.append(f"{where}: the Sunday of week {w} of {t['test_id']} is not {s}")
        if lst and (lst[-1].get("cs") or {}).get("state") in STATES_DECIDED:
            out.append(f"{where}: {t['test_id']} has a week after its decision")
        lst.append(r)
        fwd = rules.forward_test([float(x["excess"]) for x in lst], float(t["sigma_floor"]))
        want = _cs_record(fwd["path"][-1])
        if r.get("cs") != want:
            out.append(f"{where}: the bounds or the state of week {w} do not follow from the frozen weeks")
    return out


def problems(rec: dict) -> list[str]:
    """The problems of the ledger lines (an empty list when there is none). Read-only."""
    try:
        registered(rec)
    except RecordBroken as exc:
        return [str(exc)]
    return _problems_registrations(rec) + _problems_values(rec) + _problems_weeks(rec)


def _history_values_of(t: dict, cache: dict) -> pd.DataFrame | None:
    """The history values of a registered test, computed again (a formula) or from its line."""
    h = t["historical"]
    if h["start"] is None:
        return None
    sundays = pd.date_range(_ts(h["start"]), _ts(h["end"]) - pd.Timedelta(days=7), freq="7D")
    if t["input"] == "values_file":
        return _frame(t["factor"]["history"]).reindex(sundays)
    src = _source(cache, t["factor"]["source"])
    fv = uf.formula_values(t["factor"]["expression"], src, last=_day(h["end"]))
    return pd.DataFrame(fv["values"], index=fv["sundays"], columns=fv["symbols"]).reindex(sundays)


def verify(*, write_results: bool = False, log: Log = print) -> int:
    """Check every line and compute the logged numbers again (module docstring). Returns 0, 3 (a
    new alarm, recorded in alarms.jsonl) or raises SystemExit (exit 1) for a problem."""
    now = _now()
    rec = read()
    found = problems(rec)
    drift = code_drift(rec)
    if drift is not None:
        found.append(
            f"the code differs from the fingerprint of registrations.jsonl line {drift['line']}: "
            f"{', '.join(drift['changed']) or 'config.py values'}. {DRIFT_TEXT}"
        )
    for p in found:
        log(f"PROBLEM {p}")
    if found:
        raise RecordBroken(f"{count_text(len(found), 'problem')} in the trial ledger")
    n = {CHAINS[k]: len(rec[k]["records"]) for k in CHAINS}
    chains = ", ".join(f"{k} {count_text(v, 'line')}" for k, v in n.items())
    log(f"Chains: {chains}. Each link and head.json agree.")
    tests = registered(rec)
    if not tests:
        log("No test is registered.")
        return EXIT_OK
    cache = _sources()
    alarms: list[dict] = []
    for source in uf.SOURCES:
        mine = [t for t in tests if t["input"] == "formula" and t["factor"]["source"] == source]
        if not mine or not rec[f"values_{source}"]["records"]:
            continue
        try:
            src = _source(cache, source)
        except SystemExit as exc:
            log(f"{source}: the data is not available ({exc}). Its logged Sundays are not computed again.")
            continue
        fulls = {t["test_id"]: uf.full_values(t["factor"]["expression"], src["panels"]) for t in mine}
        alarms += _recheck(rec, src, fulls, now)
    okx = _source(cache, "okx")
    mk = market()
    for t in tests:
        if t["withdrawn"] is not None:
            continue
        try:
            hv = _history_values_of(t, cache)
        except SystemExit as exc:
            log(f"{t['test_id']}: the history is not computed again ({exc}).")
            hv = None
        if hv is not None and values_sha256(hv) != t["historical"]["values_sha256"]:
            alarms.append(
                _alarm(
                    "history_changed",
                    t["test_id"],
                    t["factor"].get("source") or "file",
                    {"start": t["historical"]["start"], "end": t["historical"]["end"]},
                    now,
                    okx["data_through"],
                )
            )
        frozen = _week_lines(rec, t["test_id"])
        if frozen:
            rp = replay(t, rec, mk, _day(frozen[-1]["sunday"]))
            idx = [_ts(x["sunday"]) for x in frozen]
            got = rp["weekly"].reindex(idx)
            ok = LG.same_values(got["book_ret"].to_numpy(), [x["book_ret"] for x in frozen]) & LG.same_values(
                got["control_ret"].to_numpy(), [x["control_ret"] for x in frozen]
            )
            if not ok.all():
                detail = {"weeks": [int(frozen[i]["week"]) for i in np.flatnonzero(~ok)]}
                alarms.append(_alarm("book_week_changed", t["test_id"], "okx", detail, now, okx["data_through"]))
        if write_results:
            if hv is not None:
                spec = book.spec_from_body(t["book_spec"]["body"], source=t["book_spec"].get("source") or "ledger")
                h = run_history(spec, hv, _ts(t["historical"]["start"]), _ts(t["historical"]["end"]), mk)
                # the label weeks of the registration: the OKX data that it recorded
                okx_reg = {**okx, "complete_through": _day(t["complete_through"]["okx"])}
                result = {
                    "_run": h["run"],
                    "_weekly": h["weekly"],
                    "historical_full": h["hist"],
                    "historical": t["historical"],
                    **history_numbers(h, okx_reg),
                }
                write_history_results(t["test_id"], result)
            _write_forward(t, rec, mk)
    log(f"Computed again: {count_text(len(tests), 'test')}, data through {okx['data_through']}.")
    known = known_alarms(rec)
    new, seen = [], set(known)
    for a in alarms:
        if a["key"] not in seen:
            new.append(a)
            seen.add(a["key"])
    if not new:
        log("No new alarm.")
        return EXIT_OK
    try:
        return _append_alarms(new, log)
    except C.WriteRefused as exc:
        for a in new:
            log(f"ALARM {alarm_text(a)}")
        log(f"The alarms are not recorded: {exc}")
        return EXIT_ALARM
