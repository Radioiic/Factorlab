"""Run the test suites of the repository and print one result line for each suite.

Usage (from the repository root):
    engine/.venv/bin/python -B engine/run_tests.py                  all suites
    engine/.venv/bin/python -B engine/run_tests.py --quick          a short smoke set
    engine/.venv/bin/python -B engine/run_tests.py --only test_labels --only library
    engine/.venv/bin/python -B engine/run_tests.py --list           print the suite names

The suites:
    test_*.py      every engine/test_*.py file, each in its own process (cwd engine/).
                   Files that git ignores are not run.
    library        python -m pytest tests -q, in library/
    alphaforge     python -m pytest alphaforge/tests -q, in engine/
                   Without pytest in the engine interpreter, these two suites are SKIP
                   (./setup.sh --test installs pytest).

Each suite runs with the engine interpreter (engine/.venv/bin/python when it exists).
Temporary files go to a new folder in the git-ignored engine/.tmp/, which the runner
removes at the end.

The result of a suite:
    FAIL   its exit status is not 0. The runner then prints the last lines of its output.
    SKIP   its exit status is 0 and it ran no test: every test that it reports is
           skipped, or it reports no test and prints a skip line. A test is skipped
           when a "SKIP  <test name>" line or a "(skipped: ...)" line comes before its
           PASS line. For pytest, the summary line gives the counts.
    PASS   its exit status is 0 and it ran at least one test.
A test that skips only a part of its checks counts as run. The last line gives the
number of passed, skipped and failed suites. When GITHUB_STEP_SUMMARY is set, the runner
also writes a table of the suites to that file. The exit status of the runner is 1 when
a suite fails, else 0. A skipped suite does not make the runner fail.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess  # nosec B404  # runs the suites and git: fixed argument lists, no shell
import sys
import tempfile
import time
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
REPO = ENGINE.parent
LIBRARY = REPO / "library"

# Suites that run against a copy of the data folder. No suite needs a copy (the dashboard reads this).
SCRATCH_SUITES: tuple[str, ...] = ()
# Suites outside engine/test_*.py.
EXTRA_SUITES = ("library", "alphaforge")
# The smoke set of --quick: fast suites that need no data folder.
QUICK = ("test_backtest", "test_accounting", "test_labels", "test_book", "test_rules", "test_export_web")
TAIL_LINES = 40

# Lines of a script suite (engine/test_*.py): "  PASS  name", "  FAIL  name: ...",
# "  SKIP  name: reason" and "        (skipped: reason)".
_RESULT_RE = re.compile(r"^\s*(PASS|FAIL)\s+(\S+?):?(?:\s|$)")
_SKIP_RE = re.compile(r"^\s*SKIP\s+(\S+?):?(?:\s|$)")
_SKIPPED_RE = re.compile(r"^\s*\(skipped:\s*(.*?)\)?\s*$")
# The last line of a pytest run, for example "68 passed, 2 skipped in 3.1s".
_PYTEST_RE = re.compile(r"\b(\d+) (passed|skipped|failed|errors?|xfailed|xpassed)\b")


def classify(output: str, pytest: bool = False) -> dict:
    """The test counts of one suite from its output: {"passed", "skipped", "failed", "reason"}.

    reason is the text of the first skip line (empty when there is none). A script suite
    counts a test as skipped when a "SKIP  <same name>" line or a "(skipped: ...)" line
    comes after the previous result line and before its PASS line. A "SKIP" line that names
    something else (a part of a test) does not make the test skipped, and neither does a
    "(part skipped: ...)" line: that test ran a part of its checks, so it counts as run."""
    out = {"passed": 0, "skipped": 0, "failed": 0, "reason": ""}
    lines = output.splitlines()
    if pytest:
        for line in reversed(lines):
            found = _PYTEST_RE.findall(line)
            if found and (" in " in line or line.strip().startswith("=")):
                for n, kind in found:
                    if kind in ("passed", "xpassed"):
                        out["passed"] += int(n)
                    elif kind in ("skipped", "xfailed"):
                        out["skipped"] += int(n)
                    else:
                        out["failed"] += int(n)
                break
        for line in lines:
            m = re.match(r"^SKIPPED \[\d+\] \S+: (.*)$", line.strip())
            if m:
                out["reason"] = m.group(1)
                break
        return out
    pending = None  # the name of a "SKIP  name" line, or "" for a "(skipped: ...)" line
    for line in lines:
        m = _RESULT_RE.match(line)
        if m:
            kind, name = m.groups()
            if kind == "FAIL":
                out["failed"] += 1
            elif pending is not None and pending in ("", name):
                out["skipped"] += 1
            else:
                out["passed"] += 1
            pending = None
            continue
        m = _SKIP_RE.match(line)
        if m:
            pending = m.group(1)
            out["reason"] = out["reason"] or line.split("SKIP", 1)[1].strip()
            continue
        m = _SKIPPED_RE.match(line)
        if m:
            pending = ""
            out["reason"] = out["reason"] or m.group(1)
    return out


def verdict(returncode: int, counts: dict) -> str:
    """PASS, SKIP or FAIL for one suite (see the module docstring)."""
    if returncode != 0:
        return "FAIL"
    if counts["passed"] == 0 and (counts["skipped"] or counts["reason"]):
        return "SKIP"
    return "PASS"


def _counts_text(counts: dict) -> str:
    total = counts["passed"] + counts["skipped"] + counts["failed"]
    if not total:
        return ""
    text = f", {total} test{'s' if total != 1 else ''}"
    return text + (f", {counts['skipped']} skipped" if counts["skipped"] else "")


def _step_summary(rows: list[tuple[str, str, str]], line: str) -> None:
    """Append a Markdown table of the suites to $GITHUB_STEP_SUMMARY (when it is set)."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    text = ["### Test suites", "", line, "", "| Suite | Result | Details |", "|---|---|---|"]
    text += [f"| {n} | {r} | {d.replace('|', '/')} |" for n, r, d in rows]
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(text) + "\n")
    except OSError as exc:  # the summary is extra: a write error does not change the result
        print(f"run_tests: no step summary ({type(exc).__name__}: {exc})", file=sys.stderr)


def interpreter() -> str:
    """The engine interpreter: engine/.venv when it exists, else this interpreter."""
    for rel in ("bin/python", "Scripts/python.exe"):
        p = ENGINE / ".venv" / rel
        if p.is_file():
            return str(p)
    return sys.executable


def _git_ignored(paths: list[Path]) -> set[Path]:
    """The paths that git ignores (an empty set without git or outside a work tree)."""
    git = shutil.which("git")
    if not paths or git is None:
        return set()
    try:
        r = subprocess.run(
            [git, "-C", str(REPO), "check-ignore", "--stdin"],  # nosec B603  # no shell
            input="\n".join(str(p) for p in paths),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if r.returncode not in (0, 1):
        return set()
    return {Path(line).resolve() for line in r.stdout.splitlines() if line.strip()}


def engine_suites() -> list[Path]:
    """engine/test_*.py in name order, without the files that git ignores."""
    files = sorted(ENGINE.glob("test_*.py"))
    ignored = _git_ignored(files)
    return [p for p in files if p.resolve() not in ignored]


def suite_names() -> list[str]:
    """Every suite name, in run order."""
    return [p.stem for p in engine_suites()] + list(EXTRA_SUITES)


NO_PYTEST = (
    "pytest is not installed: run ./setup.sh --test, or engine/.venv/bin/python -m pip install "
    "--require-hashes -r engine/requirements-dev.txt"
)


def has_pytest(py: str) -> bool:
    """True when the interpreter `py` can import pytest."""
    # Compare the paths without resolving symbolic links: a venv interpreter is a link to its
    # base interpreter, but it has other packages.
    if os.path.abspath(py) == os.path.abspath(sys.executable):
        return importlib.util.find_spec("pytest") is not None
    try:
        r = subprocess.run(
            [py, "-c", "import importlib.util, sys; sys.exit(importlib.util.find_spec('pytest') is None)"],
            capture_output=True,
            timeout=120,
        )  # nosec B603  # the engine interpreter, no shell
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def _command(name: str, py: str, tmp: Path) -> tuple[list[str], Path, dict] | str:
    """(argv, cwd, extra env) for one suite, or the reason to skip it."""
    if name in EXTRA_SUITES and not has_pytest(py):
        return NO_PYTEST
    if name == "library":
        return [py, "-B", "-m", "pytest", "tests", "-q"], LIBRARY, {}
    if name == "alphaforge":
        return (
            [
                py,
                "-B",
                "-m",
                "pytest",
                "alphaforge/tests",
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(tmp / "alphaforge-pytest"),
            ],
            ENGINE,
            {},
        )
    return [py, "-B", f"{name}.py"], ENGINE, {}


def run(names: list[str], verbose: bool = False) -> int:
    py = interpreter()
    (ENGINE / ".tmp").mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="run_tests_", dir=ENGINE / ".tmp"))
    failed: list[str] = []
    skipped: list[str] = []
    rows: list[tuple[str, str, str]] = []
    started = time.monotonic()
    try:
        for name in names:
            t0 = time.monotonic()
            cmd = _command(name, py, tmp)
            if isinstance(cmd, str):
                print(f"SKIP  {name}: {cmd}", flush=True)
                skipped.append(name)
                rows.append((name, "SKIP", cmd))
                continue
            argv, cwd, extra = cmd
            suite_tmp = tmp / f"{name}-tmp"
            suite_tmp.mkdir()
            env = {**os.environ, "TMPDIR": str(suite_tmp), "MPLCONFIGDIR": str(suite_tmp / "mpl"), **extra}
            r = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # nosec B603
                text=True,
                errors="replace",
            )
            dt = time.monotonic() - t0
            counts = classify(r.stdout, pytest=name in EXTRA_SUITES)
            result = verdict(r.returncode, counts)
            ok = result != "FAIL"
            detail = f"{dt:.1f} s{_counts_text(counts)}"
            if result == "SKIP":
                print(
                    f"SKIP  {name}  ({detail}): no test ran" + (f"; {counts['reason']}" if counts["reason"] else ""),
                    flush=True,
                )
            else:
                print(f"{result}  {name}  ({detail})", flush=True)
            rows.append(
                (name, result, detail + (f"; {counts['reason']}" if result == "SKIP" and counts["reason"] else ""))
            )
            if verbose or not ok:
                lines = r.stdout.rstrip("\n").splitlines()
                shown = lines if verbose else lines[-TAIL_LINES:]
                for line in shown:
                    print(f"      {line}")
            if not ok:
                failed.append(name)
            elif result == "SKIP":
                skipped.append(name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    total = time.monotonic() - started
    passed = len(names) - len(failed) - len(skipped)
    line = f"{passed} passed, {len(skipped)} skipped, {len(failed)} failed of {len(names)} suites ({total:.0f} s)"
    print("\n" + line)
    if skipped:
        print(f"skipped: {', '.join(skipped)}")
    if failed:
        print(f"failed: {', '.join(failed)}")
    _step_summary(rows, line)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the test suites; one PASS, SKIP or FAIL line for each suite.")
    ap.add_argument("--quick", action="store_true", help="run the short smoke set only")
    ap.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="NAME",
        help="run this suite only (repeat the option, or give a comma list)",
    )
    ap.add_argument("--list", action="store_true", help="print the suite names and exit")
    ap.add_argument("--verbose", action="store_true", help="print the full output of each suite")
    args = ap.parse_args(argv)
    names = suite_names()
    if args.list:
        print("\n".join(names))
        return 0
    if args.only:
        want = [n.strip().removesuffix(".py") for item in args.only for n in item.split(",") if n.strip()]
        unknown = [n for n in want if n not in names]
        if unknown:
            ap.error(f"unknown suite(s): {', '.join(unknown)} (see --list)")
        names = [n for n in names if n in want]
    elif args.quick:
        names = [n for n in names if n in QUICK]
    return run(names, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
