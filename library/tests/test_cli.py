"""The CLI's browsing commands work offline against catalog/factors.db without
changing it. ``show`` prints the engine commands that check a definition and register
it for a test. ``build`` writes a missing catalog and refuses to replace an existing
one; the other writing commands refuse the catalog folder. Without a catalog, the
browsing commands print the setup hint."""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from factorlab import FactorDB, cli, paths
from factorlab import build as build_mod
from factorlab.paths import (
    CATALOG_DB,
    CATALOG_DIR,
    NO_CATALOG,
    CatalogWriteError,
    in_catalog,
    is_catalog_db,
    refuse_catalog,
)

needs_catalog = pytest.mark.skipif(not CATALOG_DB.is_file(), reason=NO_CATALOG)


def _digest(path=CATALOG_DB):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def before():
    if not CATALOG_DB.is_file():
        pytest.skip(NO_CATALOG)
    return _digest()


@pytest.fixture
def no_catalog(tmp_path, monkeypatch):
    """Point the package at an empty catalog folder inside tmp_path."""
    cat = tmp_path / "catalog"
    monkeypatch.setattr(paths, "CATALOG_DIR", cat)
    monkeypatch.setattr(paths, "CATALOG_DB", cat / "factors.db")
    return cat / "factors.db"


def _fake_build(calls, fail=False):
    """A stand-in for build_database: one definition, and the SUMMARY.md export."""

    def fake(db_path, panels=None, synthetic=None, workers=None, export_dir=None, log=print):
        calls.append(str(db_path))
        if fail:
            raise RuntimeError("interrupted")
        with FactorDB(db_path) as db:
            db.write_catalog()
            db.replace_factors(
                [
                    {
                        "factor_id": "CF00001",
                        "name": "f1",
                        "family": "x",
                        "category": "y",
                        "expression": "ret",
                        "raw_inputs": "close",
                    }
                ]
            )
        if export_dir:
            _fake_export(None, export_dir)
        return {"n_factors": 1}

    return fake


def _fake_export(db, export_dir):
    """A stand-in for export_catalog: only SUMMARY.md."""
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    (Path(export_dir) / "SUMMARY.md").write_text("# test\n")


@needs_catalog
def test_browse_commands_are_read_only(before, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the default database is found next to the package, from any directory
    with FactorDB(CATALOG_DB, readonly=True) as db:
        n = db.count()
        first_funding = db.factors(category="funding")["factor_id"].iloc[0]
    assert cli.main(["summary"]) == 0
    assert capsys.readouterr().out.startswith(f"{n} factors")
    assert cli.main(["list", "--category", "funding", "--limit", "5"]) == 0
    out = capsys.readouterr().out
    assert first_funding in out and "(5 factors)" in out
    assert cli.main(["list", "--category", "funding", "--limit", "1"]) == 0
    assert "(1 factor)" in capsys.readouterr().out, "a count of 1 has the singular noun"
    assert cli.main(["show", "CF00001"]) == 0
    out = capsys.readouterr().out
    assert "ts_mean(ret, 3)" in out and out.splitlines()[-3:] == cli.judge_lines("CF00001")
    assert cli.main(["show", "no.such.factor"]) == 1
    capsys.readouterr()
    assert _digest() == before
    assert not list(CATALOG_DIR.glob("*-journal")) and not list(CATALOG_DIR.glob("*-wal"))
    assert not list(tmp_path.iterdir()), "a browsing command created a file"


@needs_catalog
def test_readonly_connection_refuses_writes(before):
    with FactorDB(CATALOG_DB, readonly=True) as db:
        assert db.count() > 0
        with pytest.raises(sqlite3.OperationalError):
            db.conn.execute("CREATE TABLE scratch (x INTEGER)")
    assert _digest() == before


def test_readonly_does_not_create_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        FactorDB(tmp_path / "missing.db", readonly=True)
    assert not (tmp_path / "missing.db").exists()


def _small_db(path):
    """A database with two definitions, for the show tests (no catalog needed)."""
    with FactorDB(path) as db:
        db.write_catalog()
        db.replace_factors(
            [
                {
                    "factor_id": f"CF0000{i}",
                    "name": f"f{i}",
                    "family": "x",
                    "category": "y",
                    "expression": e,
                    "raw_inputs": "close",
                }
                for i, e in ((1, "ret"), (2, "ts_mean(ret, 5)"))
            ]
        )
    return path


def test_show_prints_the_judge_commands(tmp_path, capsys):
    """show prints the definition, then where to run the commands, then the check and the
    register command of the definition (by id, also when it is given by name)."""
    db = str(_small_db(tmp_path / "f.db"))
    assert cli.main(["--db", db, "show", "f2"]) == 0
    out = capsys.readouterr().out
    assert "ts_mean(ret, 5)" in out
    assert out.splitlines()[-4:] == [
        "",
        "Run these commands from the repository root:",
        "Check this formula: (cd engine && .venv/bin/python -B judge.py check --library CF00002)",
        "Register it for a test: (cd engine && .venv/bin/python -B judge.py register --library CF00002)",
    ]
    assert cli.main(["--db", db, "show", "CF00001"]) == 0
    assert capsys.readouterr().out.splitlines()[-2:] == [
        "Check this formula: (cd engine && .venv/bin/python -B judge.py check --library CF00001)",
        "Register it for a test: (cd engine && .venv/bin/python -B judge.py register --library CF00001)",
    ]
    # the list command has no stage filter
    with pytest.raises(SystemExit) as exc:
        cli.main(["--db", db, "list", "--stage", "x"])
    assert exc.value.code == 2
    capsys.readouterr()


def test_judge_commands_exist_in_the_engine(tmp_path):
    """Each printed command is a real command of engine/judge.py: its help names --library.
    The engine runs on an empty data folder in tmp_path, so it writes nothing in the repository."""
    engine = Path(__file__).resolve().parents[2] / "engine"
    env = {**os.environ, "FACTORLAB_DATA": str(tmp_path / "data")}
    for label, cmd in cli.JUDGE_COMMANDS:
        assert cmd.startswith("(cd engine && .venv/bin/python -B judge.py ") and cmd.endswith(")"), label
        sub = cmd.split()[6]
        r = subprocess.run(
            [sys.executable, "-B", "judge.py", sub, "--help"], cwd=engine, env=env, capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr
        assert "--library ID" in r.stdout, sub


@needs_catalog
def test_writing_commands_refuse_the_catalog(before, capsys, tmp_path):
    for argv in (
        ["build"],
        ["--db", str(tmp_path / "f.db"), "build", "--export", str(CATALOG_DIR)],
        ["evaluate", "--data", str(tmp_path)],
        ["compute", "--data", str(tmp_path)],
        ["download", "--symbols", "BTCUSDT", "--out", str(CATALOG_DIR / "x")],
    ):
        assert cli.main(argv) == 2, argv
        assert "refusing" in capsys.readouterr().err
    assert _digest() == before
    assert not (tmp_path / "f.db").exists()


def test_without_a_catalog_browse_commands_print_the_setup_hint(no_catalog, capsys, tmp_path):
    for argv in (["summary"], ["show", "CF00001"], ["list"]):
        assert cli.main(argv) == 2, argv
        assert capsys.readouterr().err.strip() == NO_CATALOG
    # another missing database gives the plain message
    assert cli.main(["--db", str(tmp_path / "other.db"), "summary"]) == 2
    assert "factor database not found" in capsys.readouterr().err
    assert not no_catalog.exists()


def test_build_writes_a_missing_catalog_and_refuses_to_replace_it(no_catalog, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(build_mod, "build_database", _fake_build(calls))
    monkeypatch.setattr(build_mod, "export_catalog", _fake_export)
    assert cli.main(["build"]) == 0
    assert no_catalog.is_file() and (no_catalog.parent / "SUMMARY.md").is_file()
    assert calls == [str(no_catalog) + ".partial"], "a new catalog is built into a temporary file first"
    assert not list(no_catalog.parent.glob("*.partial"))
    capsys.readouterr()
    assert cli.main(["summary"]) == 0
    assert capsys.readouterr().out.startswith("1 factor\n")
    digest = _digest(no_catalog)
    assert cli.main(["build"]) == 2
    assert "refusing" in capsys.readouterr().err
    assert _digest(no_catalog) == digest and len(calls) == 1
    assert cli.main(["build", "--overwrite-catalog"]) == 0
    assert calls[-1] == str(no_catalog)
    # compute and evaluate refuse the catalog folder, with or without a catalog
    for argv in (["compute", "--data", "x"], ["evaluate", "--data", "x"]):
        assert cli.main(argv) == 2
        assert "refusing" in capsys.readouterr().err


def test_an_interrupted_new_build_leaves_no_catalog(no_catalog, monkeypatch):
    calls = []
    monkeypatch.setattr(build_mod, "build_database", _fake_build(calls, fail=True))
    monkeypatch.setattr(build_mod, "export_catalog", _fake_export)
    with pytest.raises(RuntimeError, match="interrupted"):
        cli.main(["build"])
    assert calls and not no_catalog.exists()
    assert not list(no_catalog.parent.glob("*")), "no database, no .partial and no export files"


def test_without_a_catalog_build_writes_only_the_catalog(no_catalog, capsys, monkeypatch, tmp_path):
    """Without a catalog, build may write catalog/factors.db and its export, and nothing else in catalog/."""
    calls = []
    monkeypatch.setattr(build_mod, "build_database", _fake_build(calls))
    monkeypatch.setattr(build_mod, "export_catalog", _fake_export)
    cat = no_catalog.parent
    for argv in (
        ["--db", str(cat / "other.db"), "build"],
        ["--db", str(tmp_path / "x.db"), "build", "--export", str(cat)],
        ["build", "--export", str(cat / "sub")],
    ):
        assert cli.main(argv) == 2, argv
        assert "refusing" in capsys.readouterr().err
    assert calls == [] and not cat.exists()
    assert cli.main(["build", "--export", str(tmp_path / "export")]) == 0
    assert no_catalog.is_file() and (tmp_path / "export" / "SUMMARY.md").is_file()


def test_guard_helpers(tmp_path):
    assert in_catalog(CATALOG_DB) and in_catalog(CATALOG_DIR) and in_catalog(CATALOG_DIR / "sub" / "x.csv")
    assert not in_catalog(tmp_path) and not in_catalog(CATALOG_DIR.parent / "data" / "factors.db")
    assert is_catalog_db(CATALOG_DB) and not is_catalog_db(tmp_path / "factors.db")
    with pytest.raises(CatalogWriteError):
        refuse_catalog(tmp_path, CATALOG_DB)
    refuse_catalog(CATALOG_DB, allow=True)
    refuse_catalog(tmp_path, None)
    with pytest.raises(CatalogWriteError, match="--overwrite-catalog"):
        refuse_catalog(CATALOG_DIR / "x")
    with pytest.raises(CatalogWriteError) as exc:
        refuse_catalog(CATALOG_DIR / "x", option=False)
    assert "--overwrite-catalog" not in str(exc.value) and "refusing" in str(exc.value)


@needs_catalog
def test_a_copy_can_be_written(tmp_path, before):
    """The documented workflow: copy the catalog, then write into the copy."""
    copy = tmp_path / "factors.db"
    shutil.copy2(CATALOG_DB, copy)
    with FactorDB(copy) as db:
        db.write_stats([{"factor_id": "CF00001", "dataset": "t", "horizon": 1, "ic_mean": 0.1}])
        assert len(db.stats("t")) == 1
    assert _digest() == before
