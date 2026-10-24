"""System section: how the platform runs and how to trust it (web/pages/system.js).

Everything here is READ-ONLY. It reads:
- the steps of ``update.plan()`` (engine/update.py; the plan is computed, never run) and
  ``STEP_IO``, what each step reads and writes;
- the version in the file VERSION (``engine/version.py``, the one version source);
- file metadata and date columns of every file under ``config.DATA``
  (the data inventory: rows, date range, bytes) and of the checkout's own ``engine/logs``;
- the raw datasets of each coin (the data coverage: rows, first, last, steps between rows);
- the engine's own text: ``GLOSSARY.md``, the documents and the ``test_*.py`` files that
  ``run_tests.py`` runs (parsed with ``ast``; tests are never run);
- the build log ``update.log`` in the checkout's own ``engine/logs`` if it exists (never a
  FACTORLAB_LOGS override; see ``_build_log``), reduced to step names, results and
  timestamps (never error text or paths).

The product rules (INVARIANTS, RULES) and the programs (PROGRAMS) are constants in this module.
"""

from __future__ import annotations

import ast
import json
import os
import re
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

import config
import run_tests
import update
import version
from webexport.common import DATA, ENGINE_ROOT, demo, rnd, to_date

REPO = ENGINE_ROOT.parent

# --------------------------------------------------------------------------- helpers


def _read(rel: str) -> str:
    """Text of a repo file (path relative to the repo root)."""
    return (REPO / rel).read_text(encoding="utf-8")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- the update steps

RAW = [
    "data/raw/ohlcv/",
    "data/raw/spot_ohlcv/",
    "data/raw/funding/",
    "data/raw/funding_binanceusdm/",
    "data/raw/open_interest/",
]
PANEL = "library/data/binance_um_1d/"
PARTS = "library/data/binance_um_1d_parts/"
# What each step of update.py reads and writes, and when update.py runs it. Keyed by the step
# name of update.plan(); test_export_system.py runs plan() in each state and checks each entry.
STEP_IO: dict[str, dict] = {
    "demo data": {
        "what": "Write the synthetic demo data set: the daily bars and the funding of the 20 coins, and a Binance panel.",
        "reads": [],
        "writes": RAW + ["data/demo.json", PANEL],
        "when": "--demo only (no download)",
    },
    "OKX download": {
        "what": "Download the new closed daily bars, the funding and the open interest. The bar that is still open is never written.",
        "reads": [
            "OKX public API (perp and spot bars, funding, open interest)",
            "Binance USDT-M public funding archive (data only)",
        ],
        "writes": RAW,
        "network": True,
        "when": "each update, not with --offline or --demo",
    },
    "Binance download": {
        "what": "Download the closed days of the Binance panel that are not on disk, one command for each coin, in parallel.",
        "reads": ["Binance public USD-M API and data.binance.vision (data only)"],
        "writes": [PARTS],
        "network": True,
        "when": "not with --offline, --demo or --skip-binance",
    },
    "Binance merge": {
        "what": "Merge the downloaded parts into the panel (the first download).",
        "reads": [PARTS],
        "writes": [PANEL],
        "when": "after the first download",
    },
    "Binance append": {
        "what": "Add the new closed days to the panel. No old cell changes.",
        "reads": [PARTS, PANEL],
        "writes": [PANEL],
        "when": "after a download of new days",
    },
    "labels": {
        "what": "Build the label table: fwd_ret_1 and fwd_ret_7 of each coin and day, from the perpetual closes.",
        "reads": ["data/raw/ohlcv/"],
        "writes": ["data/labels/labels.parquet"],
        "when": "each update",
    },
    "factor tests, weekly log": {
        "what": (
            "judge.py log-week --accept-release: log the formula values of each closed Sunday and freeze each "
            "complete forward week. Exit status 3 (a new alarm) also counts as done."
        ),
        "reads": RAW + ["data/labels/labels.parquet", "data/trials/", PANEL],
        "writes": ["data/trials/"],
        "when": "each update, when data/trials/registrations.jsonl exists",
    },
    "library catalog": {
        "what": "Build the factor library catalog (python -m factorlab build).",
        "reads": ["library/factorlab/"],
        "writes": ["library/catalog/"],
        "when": "when the catalog is absent, not with --skip-library",
    },
    "library on OKX data, compute": {
        "what": "Compute each library definition that the OKX data can compute, and prove it point in time.",
        "reads": RAW + ["library/catalog/factors.db"],
        "writes": ["data/library_on_engine/"],
        "when": "each update, not with --skip-library",
    },
    "library on OKX data, screen": {
        "what": "The screen of the computed definitions on the Sundays of the zone G.",
        "reads": ["data/library_on_engine/", "data/labels/labels.parquet", "data/raw/ohlcv/"],
        "writes": ["data/library_on_engine/"],
        "when": "each update, not with --skip-library",
    },
    "library on Binance data, compute": {
        "what": "Compute each library definition on the Binance panel (Binance as data only).",
        "reads": [PANEL, "library/catalog/factors.db", "data/library_on_engine/"],
        "writes": ["data/library_on_engine/binance/"],
        "when": "each update with a Binance panel, not with --skip-library or --skip-binance",
    },
    "library on Binance data, screen": {
        "what": "The screen of the new family and the cross-check of the definitions that the OKX screen screened.",
        "reads": ["data/library_on_engine/", "data/labels/labels.parquet", "data/raw/ohlcv/"],
        "writes": ["data/library_on_engine/binance/"],
        "when": "each update with a Binance panel, not with --skip-library or --skip-binance",
    },
    "factor lab": {
        "what": "The factor lab: the network-diffusion specs and their numbers, appended to the registry.",
        "reads": ["data/raw/ohlcv/", "data/labels/labels.parquet", "data/trials/", "data/research/"],
        "writes": ["data/research/factor_registry.parquet"],
        "when": "the first update of each UTC day",
    },
    "AlphaForge cycle": {
        "what": "Score the forward weeks of each active AlphaForge formula, and mine when a place is free.",
        "reads": RAW + ["data/labels/labels.parquet", "data/library_on_engine/", "data/trials/", "data/research/"],
        "writes": [
            "data/research/alphaforge_library.json",
            "data/research/alphaforge_registry.parquet",
            "data/research/alphaforge_cycles.jsonl",
            "data/research/alphaforge_latest.txt",
            "data/research/alphaforge_daily.parquet",
        ],
        "when": "each update, when ALPHAFORGE_ENABLED is True in engine/config.py",
    },
    "dashboard": {
        "what": "Export the data files of the pages (export_web.py).",
        # the same list as web/README.md ("The exporter only reads these files")
        "reads": [
            "data/",
            "library/catalog/",
            "engine/logs/",
            "VERSION",
            "engine/config.py",
            "engine/ingest.py",
            "setup.sh",
            "the documents, the glossary and the test files (System page)",
        ],
        "writes": ["web/data/"],
        "when": "each update",
    },
}
# the date of an update after the newest bar when the data folder has no bar
FALLBACK_DAY = date(2026, 1, 5)


def _command(step) -> str:
    """The command of a step without a path of this computer (a path gives its last part)."""
    if not step.argv:
        return "python -m factorlab download (one command for each coin)"
    args = [Path(a).name if os.path.isabs(a) else a for a in step.argv[1:]]
    return "python " + " ".join(args)


def _update_steps(newest_bar: str | None) -> dict:
    """The steps of update.plan() for this data folder (computed, never run): the plan of
    ./setup.sh (market data) or of ./setup.sh --demo (the demo data), on the day after the newest
    bar, with STEP_IO for each step."""
    today = (date.fromisoformat(newest_bar) + timedelta(days=1)) if newest_bar else FALLBACK_DAY
    is_demo = demo()
    steps = []
    for n, st in enumerate(update.plan(today, demo=is_demo), start=1):
        io = STEP_IO.get(st.name)
        steps.append(
            {
                "n": n,
                "name": st.name,
                "command": _command(st),
                "folder": "library" if Path(st.cwd).name == "library" else "engine",
                "described": io is not None,
                "what": io["what"] if io else None,
                "reads": list(io["reads"]) if io else [],
                "writes": list(io["writes"]) if io else [],
                "when": io["when"] if io else None,
                "network": bool(st.online),
                "ok_codes": list(st.ok_codes),
            }
        )
    return {
        "entry": "engine/update.py",
        "mode": "demo" if is_demo else "market",
        "day": str(today),
        "steps": steps,
        "switches": {"ALPHAFORGE_ENABLED": bool(getattr(config, "ALPHAFORGE_ENABLED", False))},
    }


# How often update.py (run by ./setup.sh) runs a writer. update.py's plan() decides this;
# test_export_system.py runs plan() in each state and checks every value below.
EACH = "each update"
EACH_ONLINE = "each update (not with --offline)"
DAILY = "first update of each UTC day"
LIBRARY = "each update (not with --skip-library)"
FACTOR_TESTS = "each update (when a registration exists)"
ALPHAFORGE = "each update (when AlphaForge is on)"
DEMO_BUILD = "the demo build (--demo)"
BY_HAND = "by hand"
READ_ONLY = "never (it writes nothing)"
NO_WRITER = "no program"  # a file that no program writes or uses

# The programs of Factorlab: what each one writes, and when update.py runs it.
PROGRAMS = [
    {
        "script": "ingest.py",
        "writes": RAW,
        "cadence": EACH_ONLINE,
        "note": "The download of the OKX data (and the Binance funding archive, data only).",
    },
    {"script": "labels.py", "writes": ["data/labels/labels.parquet"], "cadence": EACH, "note": "The label table."},
    {
        "script": "libfactors.py",
        "writes": ["data/library_on_engine/"],
        "cadence": LIBRARY,
        "note": "The library definitions computed and screened on the OKX data; with --source binance, on the Binance panel.",
    },
    {
        "script": "finder.py",
        "writes": [],
        "cadence": READ_ONLY,
        "note": "--check and --generator-probe: read-only checks and the label-free generator probe.",
    },
    {
        "script": "factor_lab.py",
        "writes": ["data/research/factor_registry.parquet"],
        "cadence": DAILY,
        "note": "The factor lab. The registry is append-only.",
    },
    {
        "script": "alphaforge_bridge.py",
        "writes": ["data/research/"],
        "cadence": ALPHAFORGE,
        "note": "The AlphaForge miner: the formulas, their forward weeks and the trial registry.",
    },
    {
        "script": "judge.py",
        "writes": ["data/trials/"],
        "cadence": FACTOR_TESTS,
        "note": (
            "The test of your factor: check (writes nothing), register, log-values, log-week, verify, "
            "withdraw, record-code-change, status and spec. update.py runs log-week --accept-release."
        ),
    },
    {"script": "export_web.py", "writes": ["web/data/"], "cadence": EACH, "note": "The data files of the pages."},
    {
        "script": "update.py",
        "writes": ["engine/logs/update.log", "engine/logs/update.lock"],
        "cadence": "./setup.sh runs it",
        "note": "Runs the update steps in order (card Update steps).",
    },
    {
        "script": "run_tests.py",
        "writes": [],
        "cadence": "by hand (./setup.sh runs the quick set)",
        "note": "The test suites. A suite writes only into its own temporary folder.",
    },
    {
        "script": "demo_data.py",
        "writes": RAW + ["data/demo.json", PANEL],
        "cadence": DEMO_BUILD,
        "note": "The synthetic demo data set (--write, --remove).",
    },
    {"script": "db.py", "writes": [], "cadence": BY_HAND, "note": "SQL over the data folder with DuckDB (read-only)."},
    {"script": "version.py", "writes": [], "cadence": BY_HAND, "note": "Prints the version of Factorlab."},
]


LOGS_LABEL = "engine/logs"  # the checkout's own log folder (config.LOGS when FACTORLAB_LOGS is unset)


def default_logs_dir() -> Path:
    """Where this checkout keeps update.log: <repo>/engine/logs."""
    return ENGINE_ROOT / "logs"


# update.py writes each line as "<time>  <text>": the version line of a run, one step line
# "<time>  <step>  <result>  <secs>s" for each step, the output of a step that shows its output,
# and with --verbose "<time>  <step> | <line>" for each output line. Only the step lines count.
# The result is "ok", update.ALARM_STATUS ("done (new alarm)": the step did its work and recorded
# a new alarm, so it counts as ok with an alarm note) or "failed (exit N)".
BUILD_LOG = "update.log"
_BUILD_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z  ")
_BUILD_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)  ([A-Za-z0-9 ,.()+&'-]{1,80}?)  "
    r"(ok|" + re.escape(update.ALARM_STATUS) + r"|failed \(exit -?\d{1,4}\))  (\d{1,7})s$"
)
# A step that starts more than this many seconds after the step before it ends begins a new run.
RUN_GAP_S = 120


def _utc(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts.replace("Z", "+00:00"))


def _build_log() -> dict:
    """The runs of update.py, from this checkout's engine/logs/update.log (step names, results,
    times; no output text). A run is a series of steps without a gap; a failed step ends its run.
    It always reads the checkout's OWN log folder, engine/logs, whatever FACTORLAB_LOGS says in the
    exporting shell: test runs point FACTORLAB_LOGS at a scratch folder, and the published file must
    not depend on the shell that exported it."""
    p = default_logs_dir() / BUILD_LOG
    out = {
        "file": f"{LOGS_LABEL}/{BUILD_LOG}",
        "present": p.is_file(),
        "lines": 0,
        "skipped": 0,  # lines without the "<time>  " start (not written by update.py)
        "other": 0,  # lines of update.py that are not step lines (version, step output)
        "runs": 0,
        "last_run": None,
        "steps": [],
    }
    if not out["present"]:
        return out
    recs = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        m = _BUILD_RE.match(line.rstrip())
        if not m:
            out["other" if _BUILD_TIME_RE.match(line) else "skipped"] += 1
            continue
        at, name, result, secs = m.group(1), m.group(2).strip(), m.group(3), int(m.group(4))
        failed = result.startswith("failed")
        recs.append(
            {
                "at": at,
                "step": name,
                "ok": not failed,
                "alarm": result == update.ALARM_STATUS,
                "exit": int(re.search(r"-?\d+", result).group(0)) if failed else None,
                "secs": secs,
            }
        )
    out["lines"] = len(recs)
    if not recs:
        return out
    runs: list[list[dict]] = []
    for r in recs:
        start = _utc(r["at"]) - pd.Timedelta(seconds=r["secs"])
        prev = runs[-1][-1] if runs else None
        if prev is None or not prev["ok"] or (start - _utc(prev["at"])).total_seconds() > RUN_GAP_S:
            runs.append([r])
        else:
            runs[-1].append(r)
    out["runs"] = len(runs)

    def summary(run: list[dict]) -> dict:
        failed = next((r for r in run if not r["ok"]), None)
        first = run[0]
        return {
            "started": (_utc(first["at"]) - pd.Timedelta(seconds=first["secs"])).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished": run[-1]["at"],
            "n_steps": len(run),
            "n_ok": sum(r["ok"] for r in run),
            "alarm_steps": [r["step"] for r in run if r["alarm"]],
            "first_step": first["step"],
            "last_step": run[-1]["step"],
            "failed_step": failed["step"] if failed else None,
            "failed_exit": failed["exit"] if failed else None,
            "secs": sum(r["secs"] for r in run),
            "steps": run,
        }

    out["last_run"] = summary(runs[-1])
    # each step name: its last result and the last time it ended without a failure (a run can
    # start at a step with --from, so this says when each output was last built)
    last: dict[str, dict] = {}
    for r in recs:
        e = last.setdefault(
            r["step"], {"step": r["step"], "last_at": None, "last_ok": None, "last_alarm": None, "last_ok_at": None}
        )
        e["last_at"], e["last_ok"], e["last_alarm"] = r["at"], r["ok"], r["alarm"]
        if r["ok"]:
            e["last_ok_at"] = r["at"]
    out["steps"] = list(last.values())
    return out


# --------------------------------------------------------------------------- environment

ENV_DOCS = {
    "FACTORLAB_DATA": {
        "default": "engine/data",
        "meaning": "Data root (config.DATA). Point it at a scratch copy to run a program that rewrites data.",
        "guard": "config.py refuses a path inside another checkout of this engine; a scratch copy anywhere else is fine.",
    },
    "FACTORLAB_LOGS": {
        "default": "engine/logs",
        "meaning": "The folder of update.log and update.lock (config.LOGS). Git-ignored; created on demand, never at import.",
        "guard": "config.py refuses a log directory outside this repo, so two checkouts can never share a lock.",
    },
    "FACTORLAB_OFFLINE": {
        "default": "1",
        "meaning": (
            "1 (the default): ingest.py makes no network request (except --check, which never fetches). Exactly 0 "
            "lets ingest.py download data. Any other value stops the program (config.offline_from). update.py sets "
            "0 for its download steps only."
        ),
        "guard": "ingest.py is the engine's only exchange client, and the factor library's download command obeys the same switch; only public endpoints, no keys.",
    },
    "FACTORLAB_WRITE_PROTECTED": {
        "default": "0",
        "meaning": (
            "The protected folders are {protected}. "
            "With 0 (the default) and the marker file .read_only in the data folder, every engine writer "
            "refuses to write there (config.guard_write). Exactly 1 allows the write. Any other value stops "
            "the program."
        ),
        "guard": "Without the marker file every writer works.",
    },
}
ENV_SOURCES = ("engine/config.py", "engine/ingest.py")
_ENV_RE = re.compile(r'os\.environ\.get\(\s*"(FACTORLAB_[A-Z_]+)"\s*,\s*([^)\n]+?)\)')


def _environment() -> list[dict]:
    found: dict[str, dict] = {}
    for rel in ENV_SOURCES:
        for name, expr in _ENV_RE.findall(_read(rel)):
            e = found.setdefault(name, {"read_in": [], "default_expr": expr.strip()})
            e["read_in"].append(Path(rel).name)
    missing = set(ENV_DOCS) - set(found)
    if missing:
        raise ValueError(f"documented env vars that the engine does not read: {sorted(missing)}")
    out = []
    protected = ", ".join(getattr(config, "PROTECTED_DIRS", ()))
    for name in ENV_DOCS:
        doc = dict(ENV_DOCS[name])
        doc["meaning"] = doc["meaning"].replace("{protected}", protected)
        out.append(
            {
                "name": name,
                **doc,
                "default_expr": found[name]["default_expr"],
                "read_in": sorted(set(found[name]["read_in"])),
            }
        )
    if "${PYTHON:-}" in _read("setup.sh"):
        out.append(
            {
                "name": "PYTHON",
                "default": "python3.13, python3.12 or python3",
                "meaning": "The Python that setup.sh uses to make engine/.venv.",
                "guard": "setup.sh stops when it is not Python 3.12 or later.",
                "default_expr": None,
                "read_in": ["setup.sh"],
            }
        )
    node_in = sorted(
        p.name
        for p in ENGINE_ROOT.glob("test_export_*.py")
        if 'os.environ.get("FACTORLAB_NODE")' in p.read_text(encoding="utf-8")
    )
    if node_in:
        for name in ("FACTORLAB_NODE", "NODE"):
            out.append(
                {
                    "name": name,
                    "default": "node on the PATH",
                    "meaning": "The node program of the page-render tests. FACTORLAB_NODE comes first, then NODE.",
                    "guard": "Without node, the page-render tests are skipped. The other tests run.",
                    "default_expr": None,
                    "read_in": node_in,
                }
            )
    return out


# --------------------------------------------------------------------------- data inventory

GROUPS = [
    # (id, relative dir, label)
    ("raw/ohlcv", "raw/ohlcv", "Perp daily bars (OKX)"),
    ("raw/spot_ohlcv", "raw/spot_ohlcv", "Spot daily bars (OKX, same venue)"),
    ("raw/funding", "raw/funding", "Funding prints (OKX)"),
    ("raw/funding_binanceusdm", "raw/funding_binanceusdm", "Funding archive (Binance USDT-M, data only)"),
    ("raw/open_interest", "raw/open_interest", "Open interest (OKX)"),
    ("labels", "labels", "Label table"),
    ("library_on_engine", "library_on_engine", "Library definitions on the engine's data"),
    ("research", "research", "Factor-lab registry and AlphaForge files"),
    ("trials", "trials", "Trial ledger of your factor tests"),
]
# Groups that are absent until their writer runs (the raw groups always exist).
OPTIONAL_GROUPS = ("labels", "library_on_engine", "research", "trials")
# Files directly in the data folder that a program of Factorlab writes: (name, writer, cadence).
TOP_FILES = (("demo.json", "demo_data.py --write", DEMO_BUILD),)

GROUP_NOTES = {
    "raw/ohlcv": "One file per symbol. Bars are stamped with their UTC open; the still-forming bar is never written.",
    "raw/spot_ohlcv": "Spot bars of the same venue. The library reads them for its spot fields.",
    "raw/funding": "Market funding RATE (+ = longs pay shorts). OKX serves about 3 months.",
    "raw/funding_binanceusdm": "Spliced in only where OKX has no print (the deep past). Never a trading venue.",
    "raw/open_interest": "OKX serves about 6 months; local history accumulates.",
    "labels": (
        "labels.parquet (labels.py): fwd_ret_1 and fwd_ret_7 of each coin and day, from the perpetual closes "
        "in raw/ohlcv. Each update writes it again from these bars."
    ),
    "library_on_engine": (
        "The factor library's computable definitions on the engine's 20 perps (libfactors.py): "
        "status.parquet (every definition), screen_g.parquet (the generator-zone screen) and manifest.json. "
        "values/ holds the weekly values (float64, written by --compute only) and is git-ignored. binance/ holds "
        "the same files for the Binance panel (--source binance; data only). They are the status, the screen of "
        "the new family, the cross-check of the OKX-screened definitions and a manifest. binance/ has its own "
        "git-ignored values/."
    ),
    "research": (
        "The factor-lab registry (append-only). When AlphaForge is on, also its files: the formulas, the trial "
        "registry, the cycle log and the daily table."
    ),
    "trials": (
        "The trial ledger of judge.py. registrations.jsonl holds one line for each registration, withdrawal and "
        "code change. values/okx.jsonl, values/binance.jsonl and values/file.jsonl hold the logged values. "
        "weeks.jsonl holds the frozen forward weeks. alarms.jsonl holds the alarms. Each line holds the hash of "
        "the line before it. head.json holds the head of each chain, and each write replaces it. results/<id>/ "
        "holds the result files of each test. judge.py verify --write-results writes them again."
    ),
}

# Why a group's last date can sit after the data close (shown when it does).
AFTER_CLOSE = {
    "raw/funding": "funding prints are stamped with their settlement time; the 00:00 UTC print settles at the close",
    "raw/funding_binanceusdm": "funding prints are stamped with their settlement time; the 00:00 UTC print settles at the close",
    "research": "run dates are the wall-clock UTC date of each research run, which falls after the close it used",
}

_DATE_COLS = ("date", "ts", "run_date")


def _writer(rel: str) -> tuple[str, str]:
    """(writer, cadence) for a path relative to DATA (a path that starts with "logs/" is in the
    checkout's own engine/logs). The cadence says when update.py runs the writer."""
    for name, writer, cadence in TOP_FILES:
        if rel == name:
            return writer, cadence
    if rel.startswith("raw/"):
        return "ingest.py (demo_data.py for the demo data)", EACH_ONLINE
    if rel.startswith("labels/"):
        return "labels.py", EACH
    if rel == "research/factor_registry.parquet":
        return "factor_lab.py", DAILY
    if rel.startswith("research/alphaforge_"):
        return "alphaforge_bridge.py", ALPHAFORGE
    if rel == "library_on_engine/binance" or rel.startswith("library_on_engine/binance/"):
        return "libfactors.py --source binance --compute / --screen", LIBRARY
    if rel == "library_on_engine/screen_g.parquet":
        return "libfactors.py --screen", LIBRARY
    if rel.startswith("library_on_engine/values"):
        return "libfactors.py --compute", LIBRARY  # only --compute writes the value shards
    if rel.startswith("library_on_engine/"):
        return "libfactors.py --compute / --screen", LIBRARY
    if rel == "trials/registrations.jsonl":
        return "judge.py register / withdraw / record-code-change / log-week --accept-release", BY_HAND
    if rel == "trials/values/file.jsonl":
        return "judge.py log-values", BY_HAND
    if rel == "trials/values" or rel.startswith("trials/values/"):
        return "judge.py log-week / log-values", FACTOR_TESTS
    if rel == "trials/weeks.jsonl":
        return "judge.py log-week", FACTOR_TESTS
    if rel == "trials/alarms.jsonl":
        return "judge.py log-week / log-values / verify", FACTOR_TESTS
    if rel == "trials/head.json":
        return "judge.py (each write)", "each write of judge.py"
    if rel == "trials/results" or rel.startswith("trials/results/"):
        return "judge.py register / log-week / verify --write-results", FACTOR_TESTS
    if rel in ("logs/update.log", "logs/update.lock"):
        return "update.py", EACH
    return "unknown", "unknown"


def _file_stats(p: Path) -> dict:
    """rows / first / last / bytes for one file (read-only)."""
    out = {
        "rows": None,
        "first": None,
        "last": None,
        "lines": None,
        "bytes": p.stat().st_size,
        "kind": p.suffix.lstrip(".") or "file",
    }
    suf = p.suffix
    if suf == ".parquet":
        pf = pq.ParquetFile(p)
        out["rows"] = pf.metadata.num_rows
        names = pf.schema_arrow.names
        col = next((c for c in _DATE_COLS if c in names), None)
        if col and out["rows"]:
            df = pd.read_parquet(p, columns=[col])
            s = pd.to_datetime(df[col], utc=True, errors="coerce")
            out["first"], out["last"] = to_date(s.min()), to_date(s.max())
    elif suf == ".tsv":
        df = pd.read_csv(p, sep="\t")
        out["rows"] = len(df)
        if "date" in df.columns and len(df):
            s = pd.to_datetime(df["date"], utc=True, errors="coerce")
            out["first"], out["last"] = to_date(s.min()), to_date(s.max())
    elif suf == ".jsonl":
        recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        out["rows"] = len(recs)
        ds = sorted(r["date"] for r in recs if isinstance(r, dict) and isinstance(r.get("date"), str))
        if ds:
            out["first"], out["last"] = to_date(ds[0]), to_date(ds[-1])
    elif suf == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            out["rows"] = len(obj)
    elif suf in (".txt", ".html", ".md"):
        out["lines"] = p.read_text(encoding="utf-8", errors="replace").count("\n")
    return out


def _item(path: Path) -> dict:
    rel = path.relative_to(DATA).as_posix()
    writer, cadence = _writer(rel)
    if path.is_dir():
        files = sorted(f for f in path.rglob("*") if f.is_file())
        stats = [_file_stats(f) for f in files]
        firsts = [s["first"] for s in stats if s["first"]]
        lasts = [s["last"] for s in stats if s["last"]]
        return {
            "name": path.name + "/",
            "path": rel + "/",
            "kind": "folder",
            "files": len(files),
            "bytes": sum(s["bytes"] for s in stats),
            "rows": sum(s["rows"] or 0 for s in stats) or None,
            "first": min(firsts) if firsts else None,
            "last": max(lasts) if lasts else None,
            "writer": writer,
            "cadence": cadence,
        }
    s = _file_stats(path)
    return {
        "name": path.name,
        "path": rel,
        "kind": s["kind"],
        "files": 1,
        "bytes": s["bytes"],
        "rows": s["rows"],
        "lines": s["lines"],
        "first": s["first"],
        "last": s["last"],
        "writer": writer,
        "cadence": cadence,
    }


def _data_root_label(data: Path) -> str:
    """'engine/data' for the checkout's own data root; a neutral phrase for any other
    copy. A scratch copy (FACTORLAB_DATA) is a local run detail, and its path is
    never published (see _build_log)."""
    return "engine/data" if Path(data).resolve() == (ENGINE_ROOT / "data").resolve() else "FACTORLAB_DATA override"


def _logs_group() -> dict:
    """The files of the checkout's own engine/logs (never a FACTORLAB_LOGS override): name, bytes,
    lines of a text file, and the writer. Not part of the data folder's totals."""
    d = default_logs_dir()
    items = []
    if d.is_dir():
        for f in sorted(d.iterdir(), key=lambda p: p.name):
            if not f.is_file() or f.name.startswith("."):
                continue
            writer, cadence = _writer("logs/" + f.name)
            text = f.suffix in (".log", ".txt", ".err")
            items.append(
                {
                    "name": f.name,
                    "path": f"{LOGS_LABEL}/{f.name}",
                    "kind": f.suffix.lstrip(".") or "file",
                    "files": 1,
                    "bytes": f.stat().st_size,
                    "rows": None,
                    "lines": f.read_text(encoding="utf-8", errors="replace").count("\n") if text else None,
                    "first": None,
                    "last": None,
                    "writer": writer,
                    "cadence": cadence,
                }
            )
    writers: list[str] = []
    for i in items:
        if i["writer"] not in writers:
            writers.append(i["writer"])
    return {
        "id": "logs",
        "label": "The log of update.py",
        "path": LOGS_LABEL + "/",
        "present": d.is_dir(),
        "area": "logs",
        "outside_data": True,
        "files": len(items),
        "bytes": sum(i["bytes"] for i in items),
        "rows": None,
        "first": None,
        "last": None,
        "writers": writers,
        "note": (
            "update.log has one line for each step of each update (the card Last build reads it); "
            "update.lock stops a second update while one runs. The folder is not in the data folder."
        ),
        "after_close": None,
        "symbols": None,
        "items": items,
    }


def _inventory() -> dict:
    groups = []
    covered: set[Path] = set()
    for gid, rel, label in GROUPS:
        d = DATA / rel
        if not d.is_dir():
            groups.append(
                {
                    "id": gid,
                    "label": label,
                    "path": rel + "/",
                    "present": False,
                    "items": [],
                    "files": 0,
                    "bytes": 0,
                    "rows": None,
                    "first": None,
                    "last": None,
                    "note": GROUP_NOTES.get(gid),
                    "writers": [],
                }
            )
            continue
        children = [c for c in sorted(d.iterdir(), key=lambda p: p.name) if not c.name.startswith(".")]
        items = [_item(c) for c in children]
        covered.update(f for f in d.rglob("*") if f.is_file())
        firsts = [i["first"] for i in items if i["first"]]
        lasts = [i["last"] for i in items if i["last"]]
        rows = [i["rows"] for i in items if i["rows"] is not None]
        writers = []
        for i in items:
            if i["writer"] not in writers:
                writers.append(i["writer"])
        groups.append(
            {
                "id": gid,
                "label": label,
                "path": rel + "/",
                "present": True,
                "area": rel.split("/")[0],
                "files": sum(i["files"] for i in items),
                "bytes": sum(i["bytes"] for i in items),
                "rows": sum(rows) if rows else None,
                "first": min(firsts) if firsts else None,
                "last": max(lasts) if lasts else None,
                "writers": writers,
                "note": GROUP_NOTES.get(gid),
                "after_close": AFTER_CLOSE.get(gid),
                "symbols": len(items) if rel.startswith("raw/") else None,
                "items": items,
            }
        )
    top = []
    for name, writer, cadence in TOP_FILES:
        f = DATA / name
        if f.is_file():
            covered.add(f)
        top.append(
            {
                "path": name,
                "present": f.is_file(),
                "bytes": f.stat().st_size if f.is_file() else 0,
                "writer": writer,
                "cadence": cadence,
            }
        )
    all_files = [f for f in DATA.rglob("*") if f.is_file() and not f.name.startswith(".")]
    # Files outside every group: the folders and files that no program of Factorlab writes (for
    # example a folder that another tool keeps in the data folder). The page lists the names only.
    outside = [f.relative_to(DATA) for f in all_files if f not in covered]
    other = sorted({(f.parts[0] + "/") if len(f.parts) > 1 else f.as_posix() for f in outside})
    newest_bar = next((g["last"] for g in groups if g["id"] == "raw/ohlcv"), None)
    return {
        "root": _data_root_label(DATA),
        "totals": {
            "files": len(all_files),
            "bytes": sum(f.stat().st_size for f in all_files),
            "groups": len([g for g in groups if g["present"]]),
        },
        "newest_raw_bar": newest_bar,
        "groups": groups,
        "logs": _logs_group(),
        "top_files": top,
        "other": other,
        "other_files": len(outside),
        "other_bytes": sum((DATA / f).stat().st_size for f in outside),
    }


# --------------------------------------------------------------------------- data coverage

DATASETS = [
    # id, relative folder, label, note
    (
        "ohlcv",
        "raw/ohlcv",
        "Perp daily bars (OKX)",
        "OHLCV of the USDT-margined perpetual swap; bars stamped with their UTC open time.",
    ),
    ("spot_ohlcv", "raw/spot_ohlcv", "Spot daily bars (OKX)", "Same-venue spot bars."),
    (
        "funding_okx",
        "raw/funding",
        "Funding prints (OKX)",
        "8-hourly prints. OKX serves about 3 months, so this file grows with each data update; it is the primary source.",
    ),
    (
        "funding_binance",
        "raw/funding_binanceusdm",
        "Funding prints (Binance archive)",
        "8-hourly Binance USDT-M prints, used as data only for the 8-hour slots that have no OKX print.",
    ),
    (
        "open_interest",
        "raw/open_interest",
        "Open interest (OKX)",
        "Daily open-interest value snapshot (16:00 UTC).",
    ),
]


def _coverage() -> dict:
    """The raw datasets of each coin (rows, first, last, the usual step between rows and the steps
    of another length) and the label rows of each coin."""
    datasets = []
    for ds_id, rel, label, note in DATASETS:
        folder = DATA / rel
        per = []
        if folder.exists():
            for f in sorted(folder.glob("*.parquet")):
                ts = pd.read_parquet(f, columns=["ts"])["ts"].sort_values()
                if ts.empty:
                    per.append({"symbol": f.stem, "rows": 0, "first": None, "last": None})
                    continue
                step = ts.diff().dropna().dt.round("1h")
                modal = step.mode().iloc[0] if len(step) else pd.NaT
                off = step[step != modal] if len(step) else step
                # the step that ends at ts[i] is at diff index i, so ts.loc[off.index] is where it happened
                off_ts = ts.loc[off.index] if len(off) else ts.iloc[:0]
                per.append(
                    {
                        "symbol": f.stem,
                        "rows": int(len(ts)),
                        "first": to_date(ts.iloc[0]),
                        "last": to_date(ts.iloc[-1]),
                        "step_h": rnd(modal / pd.Timedelta(hours=1), 2) if modal is not pd.NaT else None,
                        "max_gap_h": rnd(step.max() / pd.Timedelta(hours=1), 2) if len(step) else None,
                        "off_step": int(len(off)),
                        "off_first": to_date(off_ts.min()) if len(off_ts) else None,
                        "off_last": to_date(off_ts.max()) if len(off_ts) else None,
                    }
                )
        firsts = [p["first"] for p in per if p.get("first")]
        lasts = [p["last"] for p in per if p.get("last")]
        datasets.append(
            {
                "id": ds_id,
                "label": label,
                "path": f"data/{rel}/<SYMBOL>.parquet",
                "note": note,
                "symbols": per,
                "n_symbols": len(per),
                "first": min(firsts) if firsts else None,
                "last": max(lasts) if lasts else None,
                "rows": int(sum(p["rows"] for p in per)),
            }
        )
    lab = DATA / "labels" / "labels.parquet"
    if lab.is_file():
        d = pd.read_parquet(lab, columns=["date", "symbol"])
        g = d.groupby("symbol")["date"]
        size, lo, hi = g.size(), g.min(), g.max()
        per = [{"symbol": s, "rows": int(size[s]), "first": to_date(lo[s]), "last": to_date(hi[s])} for s in size.index]
        datasets.append(
            {
                "id": "labels",
                "label": "Label rows (labels.parquet)",
                "path": "data/labels/labels.parquet",
                "note": "One row for each (date, symbol) once 91 daily bars exist; rebuilt from the bars on each update.",
                "symbols": per,
                "n_symbols": len(per),
                "first": min(p["first"] for p in per) if per else None,
                "last": max(p["last"] for p in per) if per else None,
                "rows": int(len(d)),
            }
        )
    return {"datasets": datasets}


# --------------------------------------------------------------------------- product rules

# The kept rules of the data, stated once here and shown on the System page.
INVARIANTS = [
    "UTC everywhere. A daily bar is stamped with its open time (00:00 UTC). The bar that is still open "
    "is never written to disk.",
    "Static universe: the symbol list is in config.py. Add symbols, but never delete history (survivorship bias).",
    "Funding sign: a positive funding rate means that longs pay shorts.",
    "Research only: nothing in this repository can place an order.",
    "No keys: the engine reads public market data only. It holds no API key with trade permissions.",
]

# The rules of the test of a factor: (id, title, why)
RULES = [
    (
        "point-in-time",
        "A formula value for a Sunday uses only the data up to that Sunday.",
        "judge.py computes the value at each Sunday cut. The value of the data up to that Sunday must be equal "
        "to the value of the full computation, else the formula is refused.",
    ),
    (
        "values-file",
        "A values file is not proven point in time. A forward value counts from the first daily close after its log time.",
        "Factorlab cannot see how another program made the values, so the log time sets when a value can count.",
    ),
    (
        "ledger",
        "The trial ledger is append-only and hash-chained.",
        "Each line holds the hash of the line before it, and head.json holds the last hash. A changed or removed "
        "line stops each command of judge.py.",
    ),
    (
        "costs",
        "Each book pays fees, slippage and funding.",
        "The factor book and its control book pay the costs of the book specification, so the net excess return "
        "is after costs.",
    ),
    ("btc", "BTC hold gives context only.", "No rule reads the numbers of BTC hold."),
]


def _rules() -> dict:
    return {
        "invariants": [{"text": t} for t in INVARIANTS],
        "rules": [{"id": i, "title": t, "why": w} for i, t, w in RULES],
    }


# --------------------------------------------------------------------------- test suites

SUITE_INFO = {
    # file: (component under test, data mode)
    "test_backtest.py": ("backtest.py", "synthetic"),
    "test_accounting.py": ("backtest.py", "synthetic"),
    "test_labels.py": ("labels.py, data_io.label_table", "synthetic"),
    "test_book.py": ("book.py, bookspecs/*.toml, the schedule of backtest.run_backtest", "synthetic"),
    "test_rules.py": ("rules.py", "synthetic"),
    "test_userfactor.py": ("userfactor.py: the formula check, the Sunday-cut values, values files", "synthetic"),
    "test_trials.py": ("trials.py, judge.py: the trial ledger, the logs and the forward test", "synthetic"),
    "test_diffusion.py": ("diffusion.py", "synthetic"),
    "test_update.py": ("update.py (the step plan)", "synthetic"),
    "test_demo_data.py": (
        "demo_data.py, update.py --demo and --offline, the demo banner of meta.js and the pages",
        "synthetic",
    ),
    "test_version.py": (
        "version.py and VERSION (the one version source), CHANGELOG.md, the README badges, the fixed copies "
        "of the version, the relative links of the root documents, LICENSE, CITATION.cff, REUSE.toml, the issue forms, "
        "the contact channels",
        "synthetic",
    ),
    "test_alphaforge_bridge.py": ("alphaforge_bridge.py", "reads data"),
    "test_finder_isolation.py": ("finder.py, factorfinder/ (reads data, writes nothing)", "reads data"),
    "test_engine_rules.py": (
        "data_io.py, config.py (offline parser, write protection), ingest.py, every engine writer, db.py, "
        "alphaforge/data/store.py, the import check",
        "synthetic",
    ),
    "test_libfactors.py": ("libfactors.py", "reads data"),
    "test_libfactors_binance.py": ("libfactors.py --source binance (the merged Binance panel)", "reads data"),
    "test_finder_fieldmap.py": ("factorfinder/fieldmap.py", "reads data"),
    "test_finder_libops.py": ("factorfinder/libops.py", "reads data"),
    "test_finder_novelty.py": ("factorfinder/novelty.py", "reads data"),
    "test_finder_probe.py": ("finder.py --generator-probe, factorfinder/probe.py", "reads data"),
    "test_incumbents.py": ("factorfinder/incumbents.py: the incumbents and the library number", "synthetic"),
}
# The dashboard part that each exporter suite checks (test_export_*.py).
EXPORT_INFO = {
    "test_export_web.py": "the exporter, meta.js and the page shell",
    "test_export_library.py": "Factor library page",
    "test_export_research.py": "Search tools page",
    "test_export_mine.py": "My factors page and the page of one test",
    "test_export_system.py": "System page",
}
MODE_NOTES = {
    "synthetic": "Builds a tiny known world in memory or a temp directory; never touches engine/data.",
    "reads data": "Reads engine/data read-only (and synthetic sandboxes where it writes). Skips when the data is absent.",
}
# The commands of the test runner (from engine/).
RUN_ALL = (
    "python -B run_tests.py                  # every suite, one PASS, SKIP or FAIL line each\n"
    "python -B run_tests.py --quick          # a short smoke set\n"
    "python -B run_tests.py --only NAME      # one suite (repeat --only for more)\n"
    "python -B run_tests.py --list           # the suite names"
)


def _suite(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc = ast.get_docstring(tree) or ""
    tests = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_"):
            d = (ast.get_docstring(n) or "").strip().split("\n")[0].strip()
            tests.append({"name": n.name, "doc": d or None})
    first = doc.strip().split("\n\n")[0] if doc else ""
    return {
        "file": path.name,
        "suite": path.stem,
        "summary": _norm(first) or None,
        "n_tests": len(tests),
        "tests": tests,
    }


def _tests() -> dict:
    files = run_tests.engine_suites()
    engine = [p for p in files if not p.name.startswith("test_export_")]
    exporter = [p for p in files if p.name.startswith("test_export_")]
    suites = []
    for p in engine:
        s = _suite(p)
        # a suite without an entry shows no component (test_export_system.py asks for an entry)
        s["component"], s["mode"] = SUITE_INFO.get(p.name, (None, None))
        s["scratch_copy"] = p.stem in run_tests.SCRATCH_SUITES
        s["command"] = f"python -B run_tests.py --only {p.stem}"
        suites.append(s)
    ex = [_suite(p) for p in exporter]
    for s in ex:
        s["command"] = f"python -B run_tests.py --only {s['suite']}"
        s["component"] = EXPORT_INFO.get(s["file"])
    af = sorted((ENGINE_ROOT / "alphaforge" / "tests").glob("test_*.py"))
    af_n = 0
    for p in af:
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
                af_n += 1
    return {
        "engine": suites,
        "engine_totals": {"suites": len(suites), "tests": sum(s["n_tests"] for s in suites)},
        "exporter": ex,
        "exporter_totals": {"suites": len(ex), "tests": sum(s["n_tests"] for s in ex)},
        "alphaforge": {
            "path": "engine/alphaforge/tests/",
            "files": len(af),
            "test_functions": af_n,
            "command": "python -B run_tests.py --only alphaforge",
            "needs": "pytest and matplotlib (requirements-dev.txt)",
        },
        "runner": "engine/run_tests.py",
        "suites": run_tests.suite_names(),
        "modes": MODE_NOTES,
        "run_all": RUN_ALL,
        "note": "Counts are parsed from the test files (module-level test_* functions). The exporter never runs them.",
    }


# --------------------------------------------------------------------------- glossary


def _split_blocks(lines: list[str]) -> list[str]:
    """Markdown body -> blocks (paragraph, fenced code, table, quote, heading), hr dropped."""
    blocks, cur, fence = [], [], False

    def flush():
        if cur:
            blocks.append("\n".join(cur).strip("\n"))
            cur.clear()

    for line in lines:
        if fence:
            cur.append(line)
            if line.strip().startswith("```"):
                fence = False
                flush()
            continue
        if line.strip().startswith("```"):
            flush()
            cur.append(line)
            fence = True
            continue
        if not line.strip():
            flush()
            continue
        if re.match(r"^\s*(---+|\*\*\*+)\s*$", line):
            flush()
            continue
        if line.startswith("#"):
            flush()
            blocks.append(line.strip())
            continue
        # a table / quote / list starts a new block when it follows a different kind of line
        kind = "t" if line.lstrip().startswith("|") else "q" if line.lstrip().startswith(">") else "p"
        if cur:
            prev = cur[-1].lstrip()
            pk = "t" if prev.startswith("|") else "q" if prev.startswith(">") else "p"
            if pk != kind:
                flush()
        cur.append(line)
    flush()
    return [b for b in blocks if b.strip()]


_TERM_RE = re.compile(r"^\*\*([^*]{1,80}?)\*\*")


def _entries(blocks: list[str]) -> list[dict]:
    entries: list[dict] = []
    cur = None
    for b in blocks:
        if b.startswith("### "):
            cur = {"term": b[4:].strip().strip("`"), "md": b}
            entries.append(cur)
            continue
        if cur is not None and cur["md"].startswith("### "):
            cur["md"] += "\n\n" + b
            continue
        m = _TERM_RE.match(b)
        term = m.group(1).strip().rstrip(".").strip().strip("`") if m else None
        cur = {"term": term, "md": b}
        entries.append(cur)
        cur = None
    return entries


def _glossary() -> dict:
    text = _read("engine/GLOSSARY.md")
    lines = text.splitlines()
    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), "Glossary")
    # the card is named "Glossary"; the heading's own "Glossary:" label is not repeated
    short = re.sub(r"^Glossary\s*[:\u2014-]\s*", "", plain_heading(title)).strip()
    short = short[:1].upper() + short[1:] if short else None
    sections, intro, cur = [], [], None
    for ln in lines:
        if ln.startswith("# "):
            continue
        if ln.startswith("## "):
            head = ln[3:].strip()
            m = re.match(r"^(\d+)\.\s+(.*)$", head)
            num, name = (m.group(1), m.group(2)) if m else (None, head)
            cur = {"num": num, "title": name, "lines": []}
            sections.append(cur)
            continue
        (cur["lines"] if cur else intro).append(ln)
    out = []
    for i, s in enumerate(sections, start=1):
        sid = "glossary-" + (s["num"] or str(i))
        out.append({"id": sid, "num": s["num"], "title": s["title"], "entries": _entries(_split_blocks(s["lines"]))})
    ex = re.search(r"DAILY P&L — (\d{4}-\d{2}-\d{2})", text)
    return {
        "source": "engine/GLOSSARY.md",
        "title": title,
        "subtitle": short if short and short.lower() != "glossary" else None,
        "intro": "\n\n".join(_split_blocks(intro)),
        "sections": out,
        "n_entries": sum(len(s["entries"]) for s in out),
        "examples_date": ex.group(1) if ex else None,
    }


# --------------------------------------------------------------------------- docs index

DOC_GLOBS = [
    "README.md",
    "CHANGELOG.md",
    "ROADMAP.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "library/README.md",
    "engine/README.md",
    "engine/README_AUTOMATION.md",
    "engine/README_ALPHAFORGE.md",
    "engine/README_FINDER.md",
    "engine/GLOSSARY.md",
    "web/README.md",
]
DOC_NOTES = {
    "README.md": "What Factorlab does, setup, update, the search tools, the test of your factor, the dashboard and the tests.",
    "CHANGELOG.md": "The changes of each version, the newest first. The GitHub release of each version has the same text.",
    "ROADMAP.md": "The roadmap of Factorlab.",
    "CONTRIBUTING.md": "How to report a bug or ask a question, how to run the tests, and who maintains Factorlab.",
    "SECURITY.md": "How to report a security problem, and the security model.",
    "CODE_OF_CONDUCT.md": "The rules of behavior, and how to report a problem of behavior.",
    "LICENSE": "The MIT License of the code.",
    "THIRD_PARTY_NOTICES.md": "The material from other sources, its terms, and the publications of the methods.",
    "library/README.md": "The factor library: definitions, commands, the Binance panel and the tests.",
    "engine/README.md": "Command reference of every engine program, grouped by function, with options and outputs.",
    "engine/README_AUTOMATION.md": "update.py and a scheduled run of it.",
    "engine/README_ALPHAFORGE.md": (
        "The optional AlphaForge miner: the switch, the commands, the forward window, the capacity and the outputs."
    ),
    "engine/README_FINDER.md": "The factor library screen on Factorlab's data and the label-free generator probe.",
    "engine/GLOSSARY.md": "The research terms, defined. Rendered on this page.",
    "web/README.md": "The dashboard: pages, the no-data state, data files and the page API.",
}
_CJK = re.compile(r"[\u3400-\u9fff]")


def _doc_group(rel: str) -> str:
    if "/" not in rel:
        return "Repository"
    for prefix, name in (("engine/", "Engine"), ("library/", "Library"), ("web/", "Dashboard")):
        if rel.startswith(prefix):
            return name
    return "Other"


def plain_heading(md: str) -> str:
    """A Markdown heading as plain text: no backticks, asterisks or underscores used as marks,
    and a link keeps only its text."""
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"(\*\*|__)(.+?)\1", r"\2", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def _doc_title(path: Path) -> str:
    if path.suffix == ".html":
        m = re.search(r"<title>([^<]+)</title>", path.read_text(encoding="utf-8", errors="replace"))
        return m.group(1).strip() if m else path.name
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("# "):
            return plain_heading(ln[2:])
    return path.name


def _docs() -> list[dict]:
    seen, out = set(), []
    for pattern in DOC_GLOBS:
        for p in sorted(REPO.glob(pattern)):
            rel = p.relative_to(REPO).as_posix()
            if rel in seen or not p.is_file():
                continue
            seen.add(rel)
            text = p.read_text(encoding="utf-8", errors="replace")
            if p.suffix == ".html":
                body = re.sub(r"<[^>]+>", " ", text)
                cjk = len(_CJK.findall(body))
                lines = None
            else:
                body = text
                cjk = len(_CJK.findall(body))
                lines = text.count("\n")
            letters = len(re.findall(r"[A-Za-z]", body))
            # by the body only, so a title in another script does not decide the language
            lang = "zh" if cjk > letters * 0.15 else "en"
            desc = DOC_NOTES.get(rel)
            # the date in the file name, else in its folder
            date = re.search(r"(\d{4}-\d{2}-\d{2})", p.name) or re.search(r"(\d{4}-\d{2}-\d{2})", rel)
            group = _doc_group(rel)
            out.append(
                {
                    "path": rel,
                    "href": "../" + urllib.parse.quote(rel),
                    "title": _doc_title(p),
                    "description": desc,
                    "group": group,
                    "lang": lang,
                    "date": date.group(1) if date else None,
                    "lines": lines,
                    "bytes": p.stat().st_size,
                }
            )
    return out


# --------------------------------------------------------------------------- build


def build() -> dict:
    inv = _inventory()
    newest = inv["newest_raw_bar"]
    build_log = _build_log()
    return {
        "schema": 1,
        "snapshot": {
            "version": version.VERSION,  # the file VERSION (engine/version.py)
            "data_through": newest,  # the newest perp bar of any coin (export_web.data_through)
            "newest_raw_bar": newest,
            "data_root": inv["root"],
            "logs": {"dir": LOGS_LABEL, "update_log": bool(build_log["present"])},
            "build": build_log,
        },
        "update": _update_steps(newest),
        "programs": PROGRAMS,
        "environment": _environment(),
        "inventory": inv,
        "coverage": _coverage(),
        "rules": _rules(),
        "tests": _tests(),
        "glossary": _glossary(),
        "docs": _docs(),
    }
