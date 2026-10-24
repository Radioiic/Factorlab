"""Integrity checks of the built catalog (catalog/factors.db and its CSV/Markdown
exports). Every check opens the files read-only and recomputes the claim from the
stored rows instead of trusting a stored summary:

* the definitions: ids, names and expressions are unique; every expression parses
  and its stored raw inputs, look-back and derivatives flag match the parser; the
  catalog is exactly the generator's output minus the build-time rejects; the
  CSV/Markdown exports agree with the database. A build on the synthetic validation
  panel (the default, which ./setup.sh uses) gives fixed counts, and the tests check them;
* hygiene: no machine paths in any catalog text.

Without a catalog every test is skipped. Run from library/:  python -m pytest tests -q
"""

from __future__ import annotations

import json
import re
import sqlite3

import pandas as pd
import pytest

from factorlab.expr import max_lookback, parse, raw_dependencies
from factorlab.generator import generate_factors
from factorlab.paths import CATALOG_DB, CATALOG_DIR, NO_CATALOG

pytestmark = pytest.mark.skipif(not CATALOG_DB.is_file(), reason=NO_CATALOG)


@pytest.fixture(scope="module")
def con():
    c = sqlite3.connect(CATALOG_DB.resolve().as_uri() + "?mode=ro", uri=True)
    yield c
    c.close()


def _q(con, sql):
    return pd.read_sql_query(sql, con)


@pytest.fixture(scope="module")
def factors(con):
    return _q(con, "SELECT * FROM factors ORDER BY factor_id")


@pytest.fixture(scope="module")
def meta(con):
    return {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM meta")}


@pytest.fixture(scope="module")
def synthetic(meta):
    """True for a build on the default synthetic validation panel: its counts are fixed."""
    return meta.get("validation_data") == "synthetic"


# ----------------------------------------------------------------------------- definitions


def test_definitions_are_unique_and_numbered(factors, synthetic):
    if synthetic:
        assert len(factors) == 28_904
    for col in ("factor_id", "name", "expression", "expr_hash"):
        assert factors[col].is_unique, col
    # ids are assigned 1..N in generation order with no gaps
    assert factors["factor_id"].tolist() == [f"CF{i:05d}" for i in range(1, len(factors) + 1)]
    assert factors["family"].nunique() == 13 and factors["category"].nunique() == 15


def test_build_meta_matches_rows(meta, factors, synthetic):
    assert meta["n_factors"] == len(factors)
    assert meta["n_candidates"] - sum(meta["rejected"].values()) == len(factors)
    if synthetic:
        assert meta["rejected"] == {"rank_duplicate": 997, "no_dispersion": 1, "low_coverage": 3}


def test_every_expression_parses_and_metadata_matches_the_parser(factors, synthetic):
    derivs = {
        "funding_rate",
        "open_interest",
        "open_interest_value",
        "long_short_ratio",
        "top_long_short_ratio",
        "taker_buy_sell_ratio",
        "premium",
    }
    for r in factors.itertuples():
        parse(r.expression)
        deps = raw_dependencies(r.expression)
        assert r.raw_inputs == ",".join(sorted(deps)), r.factor_id
        assert r.lookback == max_lookback(r.expression), r.factor_id
        assert r.needs_derivs == int(bool(deps & derivs)), r.factor_id
    if synthetic:
        assert int((factors["needs_derivs"] == 0).sum()) == 17_295
        assert int(factors["needs_derivs"].sum()) == 11_609


def test_catalog_is_the_generator_output_minus_rejects(factors, meta):
    """The stored library is reproducible from the code: every stored definition is a
    generated candidate with the same family/category/expression, in generation
    order, and the candidates left out are exactly the build-time rejects."""
    specs = generate_factors()
    assert len(specs) == 29_905
    pos = {s.name: i for i, s in enumerate(specs)}
    by_name = {s.name: s for s in specs}
    order = [pos[n] for n in factors["name"]]
    assert order == sorted(order), "factor_id order must follow generation order"
    for r in factors.itertuples():
        s = by_name[r.name]
        assert (s.family, s.category, s.expression) == (r.family, r.category, r.expression), r.factor_id
    assert len(specs) - len(factors) == sum(meta["rejected"].values())


def test_exports_agree_with_the_database(factors):
    csv = pd.read_csv(CATALOG_DIR / "crypto_factors.csv.gz")
    assert csv["factor_id"].tolist() == factors["factor_id"].tolist()
    for col in ("name", "family", "category", "expression", "raw_inputs"):
        assert csv[col].tolist() == factors[col].tolist(), col
    summ = pd.read_csv(CATALOG_DIR / "summary_by_family_category.csv")
    counted = factors.groupby(["family", "category"]).size().rename("n_factors").reset_index()
    pd.testing.assert_frame_equal(summ, counted, check_dtype=False)
    md = (CATALOG_DIR / "SUMMARY.md").read_text(encoding="utf-8")
    assert f"Total factors: **{len(factors):,}**" in md
    for fam, n in factors["family"].value_counts().items():
        assert f"| {fam} | {n:,} |" in md, fam


# ----------------------------------------------------------------------------- hygiene


# machine paths; synthetic shapes only
_PATHY = re.compile(r"/tmp/|/private/|/Users/|/home/|[A-Za-z]:\\Users\\|file://")


def test_no_machine_paths_in_catalog_text(con):
    text_files = [p for p in CATALOG_DIR.rglob("*") if p.is_file() and p.suffix in (".md", ".csv", ".json")]
    assert len(text_files) >= 2  # SUMMARY.md and summary_by_family_category.csv at least
    for p in text_files:
        assert not _PATHY.search(p.read_text(encoding="utf-8")), p.name
    for p in CATALOG_DIR.rglob("*.csv.gz"):
        assert not _PATHY.search(pd.read_csv(p).astype(str).to_csv(index=False)), p.name
    for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
        for c in cols:
            for (v,) in con.execute(f'SELECT "{c}" FROM "{table}" WHERE typeof("{c}") = \'text\''):
                assert not _PATHY.search(v), (table, c)
