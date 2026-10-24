"""Tests of update.py's step plan. No network, no download, no build: the tests call the
pure functions with temporary folders.

    python test_update.py

Checks:
    1. the order of the steps of each plan (the download, labels, factor tests, the library,
       the factor lab, AlphaForge, the dashboard);
    2. last_closed_sunday and binance_end give the right Sunday for each weekday;
    3. the Binance download step has one fixed name, whatever the number of coins left;
    4. a pending Binance merge brings the library steps on Binance data;
    5. start_index accepts a step name (any case) or a number;
    6. the factor lab runs once for each UTC day.

Checks of the install files that setup.sh and the workflow use (no network):
    7. each requirement file pins each package with == and its sha256 hashes, and no
       package is in two files;
    8. setup.sh and the workflow install with --require-hashes, and setup.sh keeps its options;
    9. library/pyproject.toml has the version in VERSION, and requirements.txt pins its dependencies;
   10. ruff does not format engine/libfactors.py (CLEANING_FREEZE pins tokens of that file).

Checks of the run (no network):
   11. a step that runs longer than its timeout stops and fails with TIMEOUT_EXIT;
   12. update.py --help gives each option with a help text;
   13. the step "factor tests, weekly log" comes after "labels" only when the trial ledger has a
       registration. Its exit status 3 (a new alarm) is not a failure;
   14. --plan exits 0 and gives the refusal of a run (for example demo data without --demo) as a
       note before the steps.
"""

from __future__ import annotations

import gzip
import re
import tempfile
import tomllib
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import config as C
import update as U

MONDAY = date(2026, 10, 5)


class _Data:
    """Point config.DATA at an empty temporary folder for the duration of a block."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="update_test_")
        self.root = Path(self._tmp.name)
        self._saved = C.DATA
        C.DATA = self.root
        return self.root

    def __exit__(self, *exc):
        C.DATA = self._saved
        self._tmp.cleanup()


def _names(steps):
    return [s.name for s in steps]


# The step names of the full plan, in the order of update.plan() (each plan keeps this order).
ORDER = (
    "demo data",
    "OKX download",
    "Binance download",
    "Binance merge",
    "Binance append",
    "labels",
    "factor tests, weekly log",
    "library catalog",
    "library on OKX data, compute",
    "library on OKX data, screen",
    "library on Binance data, compute",
    "library on Binance data, screen",
    "factor lab",
    "AlphaForge cycle",
    "dashboard",
)


def test_plan_order():
    """Each plan has the steps of ORDER in that order: the factor lab comes after the library
    steps, and the dashboard is the last step."""
    saved = U.binance_steps, U.panel_last_bar, U.CATALOG, C.ALPHAFORGE_ENABLED
    try:
        U.panel_last_bar = lambda *a, **k: None
        U.binance_steps = lambda today: [
            U.Step("Binance download", (), U.LIBRARY, online=True),
            U.Step("Binance merge", ("merge",), U.LIBRARY),
        ]
        U.CATALOG = Path(tempfile.gettempdir()) / "factorlab-no-such-catalog.db"
        C.ALPHAFORGE_ENABLED = True
        with _Data() as root:
            (root / "trials").mkdir()
            (root / "trials" / "registrations.jsonl").write_text("")
            full = _names(U.plan(MONDAY))
            demo = _names(U.plan(MONDAY, demo=True))
            offline = _names(U.plan(MONDAY, offline=True, skip_library=True))
    finally:
        U.binance_steps, U.panel_last_bar, U.CATALOG, C.ALPHAFORGE_ENABLED = saved
    assert full == [n for n in ORDER if n not in ("demo data", "Binance append")], full
    assert demo == [
        n for n in ORDER if n not in ("OKX download", "Binance download", "Binance merge", "Binance append")
    ]
    assert offline == ["labels", "factor tests, weekly log", "factor lab", "AlphaForge cycle", "dashboard"], offline
    for names in (full, demo, offline):
        assert names == sorted(names, key=ORDER.index), names
        assert names[-1] == "dashboard"
    # the demo plan of a data folder without a registration: no factor-test step
    with _Data():
        demo = _names(U.plan(MONDAY, demo=True))
    assert U.FACTOR_TEST_STEP not in demo and demo[:2] == ["demo data", "labels"], demo


def test_last_closed_sunday_for_each_weekday():
    for k in range(7):
        today = MONDAY + timedelta(days=k)  # Monday .. Sunday
        sun = U.last_closed_sunday(today)
        assert sun.weekday() == 6, (today, sun)
        assert sun < today and (today - sun).days <= 7, (today, sun)
    assert U.last_closed_sunday(MONDAY) == date(2026, 10, 4)
    assert U.last_closed_sunday(date(2026, 10, 11)) == date(2026, 10, 4), "a Sunday is not closed on itself"
    # the download asks for a Sunday only one full day after its close
    assert U.binance_end(MONDAY) == date(2026, 9, 27)
    assert U.binance_end(MONDAY + timedelta(days=1)) == date(2026, 10, 4)


def _complete_part(folder: Path) -> None:
    folder.mkdir(parents=True)
    for f in U.BINANCE_FIELDS:
        (folder / f"{f}.csv.gz").write_bytes(gzip.compress(b"date\n"))


def _panel(folder: Path, last: date) -> None:
    folder.mkdir(parents=True)
    idx = pd.date_range(last - timedelta(days=2), last, freq="D", name="date")
    pd.DataFrame({"BTCUSDT": [1.0, 2.0, 3.0]}, index=idx).to_csv(folder / "close.csv.gz")


def test_binance_download_name_is_fixed():
    syms = U.binance_symbols()
    with tempfile.TemporaryDirectory(prefix="update_test_") as tmp:
        panel, parts = Path(tmp) / "panel", Path(tmp) / "parts"
        first = U.binance_steps(MONDAY, panel, parts)
        assert _names(first) == ["Binance download", "Binance merge"], _names(first)
        assert len(first[0].group) == len(syms) and first[0].online
        for s in syms[:-1]:
            _complete_part(parts / s)
        left = U.binance_steps(MONDAY, panel, parts)
        assert _names(left) == _names(first), "the step name changed with the number of coins"
        assert len(left[0].group) == 1 and left[0].group[0][left[0].group[0].index("--symbols") + 1] == syms[-1]
        _complete_part(parts / syms[-1])
        assert _names(U.binance_steps(MONDAY, panel, parts)) == ["Binance merge"]
        # an existing panel: nothing is due up to the download end, then one update
        _panel(panel, U.binance_end(MONDAY))
        assert U.panel_last_bar(panel) == U.binance_end(MONDAY)
        assert U.binance_steps(MONDAY, panel, parts) == []
        upd = U.binance_steps(MONDAY + timedelta(days=7), panel, parts)
        assert _names(upd) == ["Binance download", "Binance append"], _names(upd)
        assert len(upd[0].group) == len(syms)


def test_pending_merge_brings_the_binance_library_steps():
    saved = U.binance_steps, U.panel_last_bar, U.CATALOG
    try:
        U.panel_last_bar = lambda *a, **k: None
        U.binance_steps = lambda today: [U.Step("Binance merge", ("merge",), U.LIBRARY)]
        U.CATALOG = Path(__file__)  # an existing file: no catalog step
        with _Data():
            names = _names(U.plan(MONDAY))
            assert "library on Binance data, compute" in names, names
            assert "library on Binance data, screen" in names
            assert "library on Binance data, compute" not in _names(U.plan(MONDAY, skip_binance=True))
            U.binance_steps = lambda today: []
            assert "library on Binance data, compute" not in _names(U.plan(MONDAY))
    finally:
        U.binance_steps, U.panel_last_bar, U.CATALOG = saved


def test_start_index():
    steps = [U.Step("OKX download", (), C.ROOT), U.Step("factor lab", (), C.ROOT), U.Step("dashboard", (), C.ROOT)]
    assert U.start_index(steps, "1") == 1 and U.start_index(steps, "3") == 3
    assert U.start_index(steps, "Factor Lab") == 2 and U.start_index(steps, "dashboard") == 3
    assert U.start_index(steps, "0") is None and U.start_index(steps, "4") is None
    assert U.start_index(steps, "no such step") is None


def test_factor_lab_once_a_day():
    with _Data() as root:
        assert "factor lab" in _names(U.plan(MONDAY, offline=True, skip_library=True))
        (root / "research").mkdir()
        pd.DataFrame({"run_date": [str(MONDAY)], "factor": ["x"]}).to_parquet(
            root / "research" / "factor_registry.parquet", index=False
        )
        assert U.factor_lab_ran(MONDAY) and not U.factor_lab_ran(MONDAY + timedelta(days=1))
        assert "factor lab" not in _names(U.plan(MONDAY, offline=True, skip_library=True))
        assert "factor lab" in _names(U.plan(MONDAY + timedelta(days=1), offline=True, skip_library=True))


# ---- the install files (setup.sh, engine/requirements*.txt, library/pyproject.toml, the workflow)
REPO = Path(__file__).resolve().parent.parent
REQ_FILES = ("requirements.txt", "requirements-dev.txt", "requirements-ci.txt")
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)\s*(?:;\s*(.+?))?$")
_HASH = re.compile(r"^--hash=sha256:[0-9a-f]{64}$")


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements(name: str) -> tuple[list[str], dict[str, tuple[str, list[str]]]]:
    """(the -r includes, {canonical name: (version, hashes)}) of one engine requirement file."""
    lines, cur = [], ""
    for raw in (C.ROOT / name).read_text().splitlines():
        line = raw.split(" #", 1)[0].strip() if not raw.lstrip().startswith("#") else ""
        if not line:
            continue
        cur += line[:-1].strip() + " " if line.endswith("\\") else line
        if not line.endswith("\\"):
            lines.append(cur.strip())
            cur = ""
    assert not cur, f"{name}: the last entry ends with a line continuation"
    includes, pins = [], {}
    for entry in lines:
        if entry.startswith("-r "):
            includes.append(entry[3:].strip())
            continue
        words = entry.split(" --hash=")
        head, hashes = words[0].strip(), ["--hash=" + w.strip() for w in words[1:]]
        m = _PIN.match(head)
        assert m, f"{name}: not an exact pin: {head!r}"
        assert hashes and all(_HASH.match(h) for h in hashes), f"{name}: {head} has no sha256 hash"
        assert len(set(hashes)) == len(hashes), f"{name}: {head} has a hash two times"
        key = _canon(m.group(1))
        assert key not in pins, f"{name}: {key} is pinned two times"
        pins[key] = (m.group(2), hashes)
    return includes, pins


def test_requirement_files_are_hashed_pins():
    files = {n: _requirements(n) for n in REQ_FILES}
    assert files["requirements.txt"][0] == []
    assert files["requirements-dev.txt"][0] == ["requirements.txt"]
    assert files["requirements-ci.txt"][0] == ["requirements-dev.txt"]
    seen: dict[str, str] = {}
    for n, (_, pins) in files.items():
        for key in pins:
            assert key not in seen, f"{key} is in {seen.get(key)} and in {n}"
            seen[key] = n
    runtime = files["requirements.txt"][1]
    for key in ("ccxt", "numpy", "pandas", "pyarrow", "duckdb", "scikit-learn", "scipy", "setuptools"):
        assert key in runtime, f"requirements.txt does not pin {key}"
    assert {"pytest", "matplotlib"} <= set(files["requirements-dev.txt"][1])
    assert {"ruff", "bandit", "mypy", "pandas-stubs", "coverage", "pip-audit"} <= set(files["requirements-ci.txt"][1])


def test_setup_and_workflow_install_with_hashes():
    setup = (REPO / "setup.sh").read_text()
    installs = [line for line in setup.splitlines() if "pip install" in line and not line.lstrip().startswith("#")]
    req = [line for line in installs if " -r " in line]
    assert req and all("--require-hashes" in line for line in req), req
    assert any("--no-deps" in line and "-e library" in line for line in installs), installs
    assert "engine/requirements-dev.txt" in setup and "engine/requirements.txt" in setup
    for option in (
        "--offline",
        "--demo",
        "--skip-binance",
        "--skip-library",
        "--jobs",
        "--from",
        "--test",
        "--no-open",
    ):
        assert option in setup, f"setup.sh does not accept {option}"
    flow = (REPO / ".github" / "workflows" / "tests.yml").read_text()
    req = [line for line in flow.splitlines() if "pip install" in line and " -r " in line]
    assert req and all("--require-hashes" in line for line in req), req
    for uses in re.findall(r"uses:\s*(\S+)", flow):
        assert re.search(r"@[0-9a-f]{40}$", uses), f"the workflow action {uses} is not pinned by a commit"


def test_library_package_metadata():
    meta = tomllib.loads((REPO / "library" / "pyproject.toml").read_text())["project"]
    assert meta["version"] == (REPO / "VERSION").read_text().strip(), "library/pyproject.toml and VERSION differ"
    assert meta["name"] == "factorlab" and meta["requires-python"] == ">=3.12"
    assert meta["scripts"] == {"factorlab": "factorlab.cli:main"}
    runtime = _requirements("requirements.txt")[1]
    for dep in meta["dependencies"]:
        name = re.split(r"[<>=!~;\[ ]", dep, maxsplit=1)[0]
        assert _canon(name) in runtime, f"requirements.txt does not pin the library dependency {name}"


def test_ruff_does_not_format_libfactors():
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["ruff"]
    assert "engine/libfactors.py" in cfg["format"]["exclude"]
    assert cfg.get("force-exclude") is True


def test_step_timeout_stops_the_command():
    import sys

    rc, out = U._run_one(
        (sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"), C.ROOT, {}, timeout=1
    )
    assert rc == U.TIMEOUT_EXIT and "did not end in 1 s" in out, (rc, out)
    rc, out = U._run_one((sys.executable, "-c", "print('done')"), C.ROOT, {}, timeout=60)
    assert rc == 0 and out.strip() == "done", (rc, out)


def test_factor_test_step_only_with_a_ledger():
    with _Data() as root:
        assert U.FACTOR_TEST_STEP not in _names(U.plan(MONDAY, offline=True, skip_library=True))
        (root / "trials").mkdir()
        (root / "trials" / "registrations.jsonl").write_text("")
        steps = U.plan(MONDAY, offline=True, skip_library=True)
        names = _names(steps)
        assert names.index(U.FACTOR_TEST_STEP) == names.index("labels") + 1, names
        step = steps[names.index(U.FACTOR_TEST_STEP)]
        assert step.argv[-3:] == ("judge.py", "log-week", "--accept-release") and step.ok_codes == (0, 3)
        assert not step.online
        assert U.FACTOR_TEST_STEP in _names(U.plan(MONDAY, demo=True, skip_library=True))


def test_factor_test_step_accepts_exit_3():
    import contextlib
    import io
    import sys

    code = "import sys; print('ALARM data_gap (okx): a test alarm'); sys.exit(3)"
    alarm = U.Step("factor tests, weekly log", (sys.executable, "-c", code), C.ROOT, ok_codes=(0, 3))
    after = U.Step("dashboard", (sys.executable, "-c", "print('next')"), C.ROOT)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = U._run_steps([alarm, after], jobs=1, start=1, verbose=False)
    text = out.getvalue()
    assert rc == 0 and "done (new alarm)" in text and "ALARM data_gap (okx): a test alarm" in text, text
    assert "[2/2] dashboard" in text
    plain = U.Step("other", (sys.executable, "-c", "import sys; sys.exit(3)"), C.ROOT)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        assert U._run_steps([plain, after], jobs=1, start=1, verbose=False) == 3


def test_help_names_every_option():
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            U.main(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
    text = buf.getvalue()
    want = {
        "--help",
        "--offline",
        "--demo",
        "--plan",
        "--skip-binance",
        "--skip-library",
        "--jobs",
        "--from",
        "--verbose",
    }
    got = set(re.findall(r"--[a-z][a-z-]*", text))
    assert got == want, f"options in the help: only in the help {sorted(got - want)}; missing {sorted(want - got)}"


def test_plan_notes_the_refusal_of_a_run():
    """--plan runs nothing and exits 0. On a data folder that a run refuses (here the demo data
    without --demo), it gives the refusal as a note before the steps; the run itself exits 1."""
    import contextlib
    import io

    import demo_data

    with _Data() as root:
        (root / demo_data.MARKER).write_text("{}", encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = U.main(["--plan", "--skip-binance"])
        text = out.getvalue()
        assert rc == 0, rc
        assert text.startswith(
            "Note: a run with these options stops at once: the data folder holds the synthetic demo data"
        ), text
        assert re.search(r"^ 1\. ", text, re.M), text
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            assert U.main(["--skip-binance"]) == 1
        assert "update.py: the data folder holds the synthetic demo data" in err.getvalue(), err.getvalue()
    with _Data():
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            assert U.main(["--plan", "--skip-binance", "--demo"]) == 0
        assert "Note:" not in out.getvalue(), out.getvalue()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} update tests passed")
    raise SystemExit(1 if failed else 0)
