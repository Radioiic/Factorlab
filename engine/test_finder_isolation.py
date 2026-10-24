"""Isolation tests for Factor Finder (finder.py, factorfinder/): it reads data and writes nothing.

Usage:
    python -B test_finder_isolation.py
    TMPDIR=$PWD/.tmp python -B test_finder_isolation.py   # keep the scratch copy repo-local

`finder.py --check` runs as a subprocess against a small scratch data folder (FACTORLAB_DATA:
the perp bars only), and the folder is content-hashed before and after: a file that changed,
appeared or vanished fails the test. Content hashes, not mtimes. The other tests patch the
network and the AlphaForge bridge's writers to raise, check the data-through date and check
that an import creates nothing and loads no network module. The real data folder is never written; the scratch copy lives in a temp
directory removed at exit. Without the perp bars every test is skipped (exit 0).
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import io
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

import config as C
import factorfinder as F
import finder

ENGINE = Path(__file__).resolve().parent
_SCRATCH: dict = {}


# --------------------------------------------------------------------------- helpers


def scratch_data() -> Path:
    """One small scratch data folder per run (the perp bars only; removed at exit)."""
    if "data" not in _SCRATCH:
        tmp = tempfile.mkdtemp(prefix="finder_iso_")
        atexit.register(shutil.rmtree, tmp, True)
        dst = Path(tmp) / "data"
        shutil.copytree(C.RAW_OHLCV, dst / C.RAW_OHLCV.relative_to(C.DATA))
        _SCRATCH["data"] = dst
    return _SCRATCH["data"]


def tree_hash(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def run(args: list[str], *, data: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, FACTORLAB_DATA=str(data))
    return subprocess.run(
        [sys.executable, "-B", str(ENGINE / "finder.py"), *args],
        cwd=ENGINE,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def call(args: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = finder.main(args)
    return rc, out.getvalue()


# --------------------------------------------------------------------------- tests


def test_check_writes_nothing():
    """--check on a scratch FACTORLAB_DATA exits 0 and leaves the folder byte-identical; it
    names the probe inputs that the folder does not have, and prints no absolute path."""
    data = scratch_data()
    before = tree_hash(data)
    r = run(["--check"], data=data)
    assert r.returncode == 0, (r.returncode, r.stdout[-800:], r.stderr[-1500:])
    assert tree_hash(data) == before, "finder.py --check changed the data folder"
    assert "FAIL" not in r.stdout and "finder checks passed" in r.stdout, r.stdout
    assert "probe inputs" in r.stdout and "not on disk" in r.stdout, r.stdout
    for leak in ("/Users/", "/home/", str(Path.home())):
        assert leak not in r.stdout + r.stderr, "--check printed an absolute path"


def test_cli_options():
    """Exactly one of --check and --generator-probe; --n and --seed belong to the probe."""
    for bad in (
        [],
        ["--check", "--generator-probe"],
        ["--unknown"],
        ["--check", "--n", "5"],
        ["--check", "--seed", "1"],
    ):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                finder.main(bad)
                raise AssertionError(f"{bad} was accepted")
            except SystemExit as exc:
                assert exc.code == 2, bad
    with contextlib.redirect_stdout(io.StringIO()):
        assert finder.generator_probe(0, 1) == 2


def test_no_network():
    """With every way to open a connection patched to raise, --check still completes."""

    def boom(*_a, **_k):
        raise AssertionError("Finder attempted a network call")

    saved = (socket.create_connection, socket.socket.connect, socket.getaddrinfo)
    # patch connect/getaddrinfo, not the socket constructor: multiprocessing's socketpair needs it
    socket.create_connection, socket.socket.connect, socket.getaddrinfo = boom, boom, boom
    try:
        rc, _ = call(["--check"])
        assert rc == 0
    finally:
        socket.create_connection, socket.socket.connect, socket.getaddrinfo = saved
    code = (
        "import sys, contextlib, io, finder, factorfinder, factorfinder.config_io\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    finder.main(['--check'])\n"
        "print(','.join(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('ingest', 'ccxt', 'requests', 'aiohttp', 'alphaforge_bridge', 'alphaforge'))))"
    )
    r = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ENGINE,
        capture_output=True,
        text=True,
        timeout=300,
        env=dict(os.environ, FACTORLAB_DATA=str(scratch_data())),
    )
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.strip().splitlines()[-1:] in ([""], []), f"Finder imported {r.stdout.strip()}"


def test_bridge_writers_never_called():
    """Finder never calls the AlphaForge bridge's writers: they are patched to raise around --check."""
    import alphaforge_bridge as AB

    names = ("run_cycle", "mine", "refresh", "save_zoo", "record", "_atomic_text")
    saved = {n: getattr(AB, n) for n in names}

    def make(n):
        def _raise(*_a, **_k):
            raise AssertionError(f"alphaforge_bridge.{n} was called")

        return _raise

    for n in names:
        setattr(AB, n, make(n))
    try:
        rc, _ = call(["--check"])
        assert rc == 0
    finally:
        for n, f in saved.items():
            setattr(AB, n, f)


def test_data_through_is_the_last_bar_on_disk():
    """data_through() is the newest perp bar on disk, never after the UTC wall-clock date."""
    last = []
    for sym in C.UNIVERSE:
        p = C.RAW_OHLCV / f"{C.sym_to_file(sym)}.parquet"
        if p.exists():
            last.append(pd.read_parquet(p, columns=["ts"])["ts"].max())
    through = max(last).date()
    assert F.data_through() == through
    assert through <= F.utc_today()
    with tempfile.TemporaryDirectory() as tmp:
        code = (
            "import factorfinder as F\n"
            "try:\n    F.data_through()\n    print('ACCEPTED')\n"
            "except F.FinderError as exc:\n    print('REFUSED', exc)"
        )
        r = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ENGINE,
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(os.environ, FACTORLAB_DATA=str(Path(tmp) / "data")),
        )
        assert r.returncode == 0 and r.stdout.startswith("REFUSED") and "no daily bars" in r.stdout, r.stdout


def test_import_has_no_side_effects():
    """An import of the package creates no file (config.py makes its empty data folders).
    The import loads neither ingest.py nor the bridge."""
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        shutil.copytree(C.RAW_OHLCV, data / C.RAW_OHLCV.relative_to(C.DATA))
        before = tree_hash(data)
        code = (
            "import sys, factorfinder, factorfinder.config_io, finder\n"
            "print(sorted(m for m in ('ingest', 'ccxt', 'alphaforge_bridge') if m in sys.modules))"
        )
        r = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ENGINE,
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(os.environ, FACTORLAB_DATA=str(data)),
        )
        assert r.returncode == 0, r.stderr[-800:]
        assert r.stdout.strip() == "[]", r.stdout
        assert tree_hash(data) == before, "import created or changed a file"
    for src in (ENGINE / "finder.py", *sorted((ENGINE / "factorfinder").glob("*.py"))):
        text = src.read_text(encoding="utf-8")
        assert "/Users/" not in text and "/home/" not in text, f"{src.name} hard-codes an absolute path"


if __name__ == "__main__":
    if not any((C.RAW_OHLCV / f"{C.sym_to_file(s)}.parquet").exists() for s in C.UNIVERSE):
        print("  SKIP  every finder isolation test skipped: no perp bars in the data folder (run ./setup.sh)")
        raise SystemExit(0)
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} finder isolation tests passed")
    raise SystemExit(1 if failed else 0)
