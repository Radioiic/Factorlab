"""Download new market data and build every output of the engine.

Usage (from the repository root):
    engine/.venv/bin/python -B engine/update.py              # download, then build
    engine/.venv/bin/python -B engine/update.py --offline    # build from the data on disk
    engine/.venv/bin/python -B engine/update.py --demo       # build from synthetic demo data
    engine/.venv/bin/python -B engine/update.py --plan       # show the steps; run nothing

Options:
    --demo           write the synthetic demo data set (demo_data.py), then build every output
                     from it. No download. The dashboard shows "Synthetic demo data. Not market data."
    --skip-binance   do not download or use the Binance panel
    --skip-library   do not compute the factor library on the data (the longest step)
    --jobs N         number of Binance downloads at the same time (default 8)
    --from STEP      start at this step: its name from --plan (for example "dashboard") or its number
    --verbose        show the output of every step, and add each output line to logs/update.log

setup.sh runs this command. You can run it again at any time: each step adds the new days
and builds its output again. The first run downloads about four years of data. It can take
more than an hour.

The steps, in order: the download (or the demo data), labels, factor tests, the library
catalog, the library on the OKX data and on the Binance data, the factor lab, the AlphaForge
cycle and the dashboard. --plan shows the steps of a run.

Rules:
- --offline builds from the data on disk. It downloads nothing.
- --demo writes only into an empty data folder or into a demo folder (the file demo.json),
  and its Binance panel only into an absent, empty or demo library/data/binance_um_1d.
  It refuses a folder that holds other files: that folder can hold market data. A build
  without --demo refuses a demo data folder and (without --skip-binance) a demo panel, so
  no market data mixes with demo data. demo_data.py --remove removes the demo data set.
- Only the download steps use the network. They get FACTORLAB_OFFLINE=0. All other steps
  get FACTORLAB_OFFLINE=1.
- The Binance panel ends on the last Sunday that closed at least one full day before
  today (UTC). Binance can publish a daily archive file some hours late, so the panel
  does not ask for a Sunday that closed today.
- The factor lab runs once for each UTC day. It adds rows to its registry for the run date,
  so a second run on the same day would add copies.
- The step "factor tests, weekly log" (judge.py log-week --accept-release) runs only when the
  trial ledger data/trials/ has a registration. It logs the new forward weeks of your tests.
  A new VERSION of Factorlab is recorded as a code change. Its exit status 3 (a new alarm)
  is not a failure: the log line of the step says "done (new alarm)".
- The step "AlphaForge cycle" runs only when config.ALPHAFORGE_ENABLED is True.
- The command refuses a protected data folder (a folder that has the file .read_only).
- It takes the lock logs/update.lock and writes to logs/update.log. Each line of the log has one
  format: "<UTC time>  <text>". The first line of a run gives the version. Each step adds the
  line "<UTC time>  <step>  <result>  <seconds>s". The step "demo data" adds its output on the
  next line. With --verbose, each output line of each step follows its step line.
- A step that runs for more than STEP_TIMEOUT_S seconds (12 hours) stops, and the step fails.
- It stops at the first step that fails and shows the end of the output of that step. The
  message gives the step name for --from.
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import logging
import os
import subprocess  # nosec B404  # runs the steps of the build: fixed argument lists, no shell
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import config as C
import demo_data
import version

REPO = C.ROOT.parent
LIBRARY = REPO / "library"
PANEL = LIBRARY / "data" / "binance_um_1d"
PARTS = LIBRARY / "data" / "binance_um_1d_parts"
CATALOG = LIBRARY / "catalog" / "factors.db"
BINANCE_START = "2022-01-01"
# The fields of one coin's USD-M download with the metrics archive. A part folder without all
# of them is not complete, and the next run downloads that coin again.
BINANCE_FIELDS = (
    "close",
    "funding_rate",
    "high",
    "long_short_ratio",
    "low",
    "open",
    "open_interest",
    "open_interest_value",
    "premium",
    "quote_volume",
    "taker_buy_quote_volume",
    "taker_buy_sell_ratio",
    "taker_buy_volume",
    "top_long_short_ratio",
    "trades",
    "volume",
)
LOCK_NAME = "update.lock"
LOG_NAME = "update.log"
TAIL_LINES = 40
# The one format of each line of logs/update.log: the UTC time, two spaces, the text.
LOG_FORMAT = "%(asctime)s  %(message)s"
LOG_TIME = "%Y-%m-%dT%H:%M:%SZ"
# The longest time that one step can run. The first download and the first library
# computation take some hours; a step that runs longer than this stops and fails.
STEP_TIMEOUT_S = 12 * 3600
TIMEOUT_EXIT = 124  # the exit status that the log gives to a step that timed out
LOG = logging.getLogger("factorlab.update")


@dataclass
class Step:
    name: str
    argv: tuple
    cwd: Path
    online: bool = False
    group: list = field(default_factory=list)  # parallel commands (the Binance downloads)
    detail: str = ""  # extra text for --plan
    show_output: bool = False  # print the output of the step and add it to the log
    env: dict = field(default_factory=dict)  # more environment variables for this step only
    ok_codes: tuple[int, ...] = (0,)  # the exit statuses of a step that did its work


def binance_symbols() -> list[str]:
    """The Binance USD-M symbols of the engine's coins, in the engine's order (BTCUSDT, ...)."""
    return [C.sym_to_file(s).replace("-", "") for s in C.UNIVERSE]


def last_closed_sunday(today: date) -> date:
    """The last Sunday whose daily bar is closed on `today` (UTC)."""
    yesterday = today - timedelta(days=1)
    return yesterday - timedelta(days=(yesterday.weekday() + 1) % 7)


def binance_end(today: date) -> date:
    """The last day of the Binance download: the last Sunday that closed at least one full
    day before `today` (UTC). On a Monday this is the Sunday of the week before."""
    return last_closed_sunday(today - timedelta(days=1))


def panel_last_bar(panel: Path = PANEL) -> date | None:
    """The date of the last bar in the merged Binance panel, or None when there is no panel."""
    close = panel / "close.csv.gz"
    if not close.is_file():
        return None
    import pandas as pd

    idx = pd.read_csv(close, usecols=[0], index_col=0).index
    return pd.Timestamp(idx[-1]).date() if len(idx) else None


def part_complete(folder: Path) -> bool:
    """True when a coin's part folder has every field of BINANCE_FIELDS."""
    return all((folder / f"{f}.csv.gz").is_file() for f in BINANCE_FIELDS)


def _py(*args: str) -> tuple:
    return (sys.executable, "-B", *args)


def _download(symbol: str, first: date, last: date, out: str) -> tuple:
    return _py(
        "-m",
        "factorlab",
        "download",
        "--symbols",
        symbol,
        "--start",
        str(first),
        "--end",
        f"{last} 23:59:59",
        "--interval",
        "1d",
        "--market",
        "um",
        "--metrics-archive",
        "--out",
        out,
    )


def binance_steps(today: date, panel: Path = PANEL, parts: Path = PARTS) -> list[Step]:
    """The Binance download and merge steps for today (an empty list when nothing is due)."""
    end = binance_end(today)
    last = panel_last_bar(panel)
    if last is None:
        first = date.fromisoformat(BINANCE_START)
        todo = [s for s in binance_symbols() if not part_complete(parts / s)]
        group = [_download(s, first, end, str(parts / s)) for s in todo]
        # a fixed step name, so that the name stays valid when fewer coins are left to download
        steps = (
            [
                Step(
                    "Binance download",
                    (),
                    LIBRARY,
                    online=True,
                    group=group,
                    detail=f"{len(todo)} of {len(binance_symbols())} coins, {first} to {end}",
                )
            ]
            if todo
            else []
        )
        steps.append(
            Step("Binance merge", _py("scripts/merge_panels.py", "--parts", str(parts), "--out", str(panel)), LIBRARY)
        )
        return steps
    if last >= end:
        return []
    first = last + timedelta(days=1)
    folder = parts / "updates" / f"{first}_{end}"
    group = [_download(s, first, end, str(folder / s)) for s in binance_symbols()]
    return [
        Step("Binance download", (), LIBRARY, online=True, group=group, detail=f"{first} to {end}"),
        Step("Binance append", _py("scripts/merge_panels.py", "--append", "--parts", str(folder)), LIBRARY),
    ]


def factor_lab_ran(today: date) -> bool:
    """True when the factor-lab registry has rows of the UTC date `today`."""
    reg = C.DATA / "research" / "factor_registry.parquet"
    if not reg.is_file():
        return False
    import pandas as pd

    try:
        dates = pd.read_parquet(reg, columns=["run_date"])["run_date"].astype(str)
    except (OSError, ValueError, KeyError, TypeError):
        # a registry that cannot be read counts as "no run today": the factor lab runs
        return False
    return bool((dates == str(today)).any())


def _e(name: str, *args: str) -> Step:
    return Step(name, _py(*args), C.ROOT)


DEMO_STEP = "demo data"
FACTOR_TEST_STEP = "factor tests, weekly log"
ALARM_EXIT = 3  # judge.py log-week: the run recorded a new alarm
# The result of a step that exits with one of its ok_codes other than 0 (the log line of the
# step). The System page reads it as a step that did its work (webexport/system.py).
ALARM_STATUS = "done (new alarm)"


def plan(
    today: date,
    offline: bool = False,
    skip_binance: bool = False,
    skip_library: bool = False,
    demo: bool = False,
) -> list[Step]:
    """The steps of one update, in order. demo=True: no download; the first step writes the
    synthetic demo data set (demo_data.py)."""
    steps: list[Step] = []
    if demo:
        steps.append(
            Step(
                DEMO_STEP,
                _py("demo_data.py", "--write", *(("--no-panel",) if skip_binance else ())),
                C.ROOT,
                show_output=True,
                detail=f"{demo_data.BANNER} {len(C.UNIVERSE)} coins, {demo_data.FIRST_DAY}, "
                f"{demo_data.N_DAYS} days" + ("" if skip_binance else ", with a Binance panel"),
            )
        )
    elif not offline:
        steps.append(Step("OKX download", _py("ingest.py"), C.ROOT, online=True))
        if not skip_binance:
            steps += binance_steps(today)
    steps.append(_e("labels", "labels.py"))
    if (C.DATA / "trials" / "registrations.jsonl").is_file():
        # exit 3: the step recorded a new alarm in the trial ledger; the log of the step names it
        steps.append(
            Step(
                FACTOR_TEST_STEP,
                _py("judge.py", "log-week", "--accept-release"),
                C.ROOT,
                ok_codes=(0, ALARM_EXIT),
                detail="the trial ledger data/trials/ has a registration",
            )
        )
    if not skip_library:
        if not CATALOG.is_file():
            steps.append(Step("library catalog", _py("-m", "factorlab", "build"), LIBRARY))
        steps += [
            _e("library on OKX data, compute", "libfactors.py", "--compute"),
            _e("library on OKX data, screen", "libfactors.py", "--screen"),
        ]
        if not skip_binance and (
            demo or panel_last_bar() is not None or any(s.name.startswith("Binance ") for s in steps)
        ):
            steps += [
                _e("library on Binance data, compute", "libfactors.py", "--source", "binance", "--compute"),
                _e("library on Binance data, screen", "libfactors.py", "--source", "binance", "--screen"),
            ]
    if not factor_lab_ran(today):
        steps.append(_e("factor lab", "factor_lab.py"))
    if getattr(C, "ALPHAFORGE_ENABLED", False):
        steps.append(_e("AlphaForge cycle", "alphaforge_bridge.py", "--cycle"))
    steps.append(_e("dashboard", "export_web.py"))
    return steps


def start_index(steps: list[Step], value: str) -> int | None:
    """The 1-based step number for --from: a number, or a step name (case does not matter)."""
    if value.isdigit():
        i = int(value)
        return i if 1 <= i <= len(steps) else None
    names = [s.name.lower() for s in steps]
    return names.index(value.lower()) + 1 if value.lower() in names else None


def _env(online: bool, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update(extra or {})
    env["FACTORLAB_OFFLINE"] = "0" if online else "1"
    tmp = C.ROOT / ".tmp" / "update"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp)
    env.setdefault("MPLCONFIGDIR", str(tmp / "mpl"))
    return env


def _run_one(argv: tuple, cwd: Path, env: dict, timeout: float = STEP_TIMEOUT_S) -> tuple[int, str]:
    """Run one command. Returns its exit status and its output (stdout and stderr together).
    A command that runs longer than `timeout` seconds stops: the status is TIMEOUT_EXIT."""
    try:
        p = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # nosec B603
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.output.decode(errors="replace") if isinstance(exc.output, bytes) else (exc.output or "")
        return TIMEOUT_EXIT, out + f"\nThe command did not end in {timeout:.0f} s and was stopped.\n"
    return p.returncode, p.stdout


class _UTCFormatter(logging.Formatter):
    """The format of logs/update.log, with the time in UTC."""

    converter = time.gmtime


@contextmanager
def build_log(verbose: bool = False):
    """Add the lines of LOG to logs/update.log while the block runs (LOG_FORMAT, UTC time).
    verbose=True also writes the DEBUG lines (the output of each step)."""
    handler = logging.FileHandler(C.LOGS / LOG_NAME, encoding="utf-8")
    handler.setFormatter(_UTCFormatter(LOG_FORMAT, LOG_TIME))
    LOG.addHandler(handler)
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOG.propagate = False
    try:
        yield LOG
    finally:
        LOG.removeHandler(handler)
        handler.close()


DATA_TERMS = "The downloads use public data of OKX and Binance. Their terms apply: see README.md, Data terms."


def failure_hint(step: Step) -> str:
    """The text that update.py prints after a step that failed."""
    if step.online:
        return "Run the same command again. A first setup keeps the complete coin downloads on disk."
    return f"Correct the cause, then run the same command again with --from \"{step.name}\" to start at this step."


def start_lines(steps: list[Step], start: int = 1) -> list[str]:
    """The first lines of a run: the version, and the data terms when a download step follows."""
    lines = [f"Factorlab {version.VERSION}"]
    if any(s.online for i, s in enumerate(steps, 1) if i >= start):
        lines.append(DATA_TERMS)
    return lines


def run(steps: list[Step], jobs: int, start: int = 1, verbose: bool = False) -> int:
    """Run the steps from step number `start`. Returns 0, or the exit status of the first step
    that fails. verbose=True shows the output of every step and adds it to the log."""
    with build_log(verbose):
        return _run_steps(steps, jobs, start, verbose)


def _run_steps(steps: list[Step], jobs: int, start: int, verbose: bool) -> int:
    for i, s in enumerate(steps, 1):
        if i < start:
            continue
        t0 = time.time()
        print(f"[{i}/{len(steps)}] {s.name} ...", flush=True)
        env = _env(s.online, s.env)
        if s.group:
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                results = list(pool.map(functools.partial(_run_one, cwd=s.cwd, env=env), s.group))
            bad = [(a, out) for a, (rc, out) in zip(s.group, results) if rc != 0]
            rc, out = (1, bad[0][1]) if bad else (0, "")
            if bad:
                syms = [a[a.index("--symbols") + 1] for a, _ in bad if "--symbols" in a]
                out = (
                    f"{len(bad)} of {len(s.group)} downloads failed: {', '.join(syms)}\n"
                    f"The output of the first failed download:\n{out}"
                )
        else:
            rc, out = _run_one(s.argv, s.cwd, env)
        secs = time.time() - t0
        ok = rc in s.ok_codes
        status = ("ok" if rc == 0 else ALARM_STATUS) if ok else f"failed (exit {rc})"
        LOG.info(f"{s.name}  {status}  {secs:.0f}s")
        if verbose:
            for line in out.splitlines():
                print(f"      {line}", flush=True)
                LOG.debug(f"{s.name} | {line}")
        else:
            shown = " ".join(out.split()) if s.show_output and rc == 0 else ""
            if shown:
                print(f"      {shown}", flush=True)
                LOG.info(shown)
            if ok and rc != 0:  # a new alarm: show its lines
                for line in out.splitlines():
                    if line.startswith("ALARM"):
                        print(f"      {line}", flush=True)
        print(f"      {status}, {secs:.0f} s", flush=True)
        if not ok:
            tail = "\n".join(out.splitlines()[-TAIL_LINES:])
            print(f"\nThe step '{s.name}' failed. The end of its output:\n{tail}", file=sys.stderr)
            print("\n" + failure_hint(s), file=sys.stderr)
            return rc
    return 0


REMOVE_DEMO = "engine/.venv/bin/python -B engine/demo_data.py --remove"


def demo_check(demo: bool, skip_binance: bool, data: Path | None = None, panel: Path | None = None) -> str | None:
    """Why this build must not start (demo data and market data must not mix), or None."""
    data = Path(C.DATA if data is None else data)
    panel = Path(demo_data.PANEL_DIR if panel is None else panel)
    if demo:
        why = demo_data.refusal(data, None if skip_binance else panel)
        return f"--demo: {why}." if why else None
    if demo_data.is_demo(data):
        return (
            "the data folder holds the synthetic demo data (demo.json). To build from market data, "
            f"remove the demo data first ({REMOVE_DEMO}), or use another folder (FACTORLAB_DATA=<folder>)."
        )
    if not skip_binance and demo_data.is_demo_panel(panel):
        return (
            "library/data/binance_um_1d holds the demo Binance panel. To build from market data, remove "
            f"the demo data first ({REMOVE_DEMO}), or use --skip-binance."
        )
    return None


def refusal(demo: bool, offline: bool, skip_binance: bool) -> str | None:
    """Why a run with these options must not start in this data folder, or None."""
    marker = C.DATA / C.PROTECT_MARKER
    if marker.exists():
        return (
            f"the data folder {C.DATA} is protected (it has {marker.name}). Remove that file to build into this folder."
        )
    why = demo_check(demo, skip_binance)
    if why:
        return why
    if offline and not demo and not any(C.RAW_OHLCV.glob("*.parquet")):
        return "no market data on disk. Run ./setup.sh without --offline."
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download new market data and build every output.")
    ap.add_argument("--offline", action="store_true", help="build from the market data on disk; download nothing")
    ap.add_argument(
        "--demo", action="store_true", help="build from synthetic demo data (not market data); download nothing"
    )
    ap.add_argument("--plan", action="store_true", help="show the steps; run nothing")
    ap.add_argument("--skip-binance", action="store_true", help="do not download or use the Binance panel")
    ap.add_argument("--skip-library", action="store_true", help="do not compute the factor library")
    ap.add_argument("--jobs", type=int, default=8, help="Binance downloads at the same time (default 8)")
    ap.add_argument(
        "--from", dest="start", default="1", metavar="STEP", help="start at this step (a name from --plan, or a number)"
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="show the output of every step and add each output line to logs/update.log",
    )
    a = ap.parse_args(argv)

    why = refusal(a.demo, a.offline, a.skip_binance)
    if why and not a.plan:
        print(f"update.py: {why}", file=sys.stderr)
        return 1
    today = datetime.now(UTC).date()
    steps = plan(today, a.offline, a.skip_binance, a.skip_library, a.demo)
    if a.plan:
        if why:
            # --plan runs nothing, so it gives the refusal of a run as a note and exits 0
            print(f"Note: a run with these options stops at once: {why}")
        for i, s in enumerate(steps, 1):
            net = "  (network)" if s.online else ""
            n = f"  ({len(s.group)} {'command' if len(s.group) == 1 else 'commands'})" if s.group else ""
            d = f"  [{s.detail}]" if s.detail else ""
            print(f"{i:2d}. {s.name}{net}{n}{d}")
        return 0
    C.LOGS.mkdir(parents=True, exist_ok=True)
    with open(C.LOGS / LOCK_NAME, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("update.py: another update is running (logs/update.lock).", file=sys.stderr)
            return 1
        start = start_index(steps, a.start)
        if start is None:
            print(f"update.py: no step '{a.start}' in this plan. Run with --plan to see the steps.", file=sys.stderr)
            return 2
        lines = start_lines(steps, start)
        for line in lines:
            print(line, flush=True)
        with build_log(a.verbose):
            LOG.info(lines[0])
        return run(steps, a.jobs, start, a.verbose)


if __name__ == "__main__":
    sys.exit(main())
