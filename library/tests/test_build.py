"""The build step gives the same database bytes from the same inputs.

The tests build a small catalog (the first 200 generated specs on a small synthetic
panel) twice in the temporary folder and compare the files byte for byte. The meta table
holds only values that the inputs set, and an in-place rebuild replaces the whole meta
table.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing

import pytest

from factorlab import build
from factorlab.database import FactorDB
from factorlab.generator import generate_factors

SMALL = {"n_symbols": 12, "n_bars": 400, "seed": 7}


@pytest.fixture(scope="module")
def specs():
    return generate_factors()[:200]


def _build(path, specs, monkeypatch):
    monkeypatch.setattr(build, "generate_factors", lambda: list(specs))
    return build.build_database(path, synthetic=SMALL, workers=1, log=lambda _msg: None)


def _meta(path):
    with closing(sqlite3.connect(path)) as con:
        return {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM meta")}


def test_two_builds_give_the_same_bytes(tmp_path, specs, monkeypatch):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    info_a = _build(a, specs, monkeypatch)
    info_b = _build(b, specs, monkeypatch)
    assert info_a == info_b
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()
    assert set(info_a) == {"n_factors", "n_candidates", "rejected", "validation_data"}
    assert _meta(a) == info_a


def test_rebuild_in_place_replaces_the_meta_table(tmp_path, specs, monkeypatch):
    path = tmp_path / "c.db"
    with FactorDB(path) as db:
        db.set_meta(stale_key=1.5)
    info = _build(path, specs, monkeypatch)
    assert _meta(path) == info


def test_export_is_reproducible(tmp_path, specs, monkeypatch):
    path = tmp_path / "d.db"
    _build(path, specs, monkeypatch)
    out = []
    for name in ("e1", "e2"):
        with FactorDB(path, readonly=True) as db:
            build.export_catalog(db, tmp_path / name)
        out.append({f.name: f.read_bytes() for f in (tmp_path / name).iterdir()})
    assert out[0] == out[1]
    assert set(out[0]) == {"crypto_factors.csv.gz", "summary_by_family_category.csv", "SUMMARY.md"}
