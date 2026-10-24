"""Factor library section (#/library): build() -> dict.

The factor library (``<repo>/library``) holds the rank-distinct formula definitions that
``python -m factorlab build`` writes. The page shows numbers and definitions only: the
counts, the families, the search with each formula and its two commands (check, register),
and the two screens on Factorlab's data. No block gives a verdict.

Read-only: ``library/catalog/factors.db`` is opened through SQLite's ``mode=ro`` URI (SQLite
itself refuses any write). Under the engine data directory only
``library_on_engine/{status.parquet, screen_g.parquet, manifest.json}`` (the ``engine``
block) and ``library_on_engine/binance/{status.parquet, screen_g.parquet, crosscheck.parquet,
manifest.json}`` (the ``binance`` block) are read; nothing there is written. If the catalog
is missing the section is ``{"available": false}`` and the page shows an empty state.

Payload (t-statistics 2 dp):
    totals        definitions, candidates, build rejects, families, categories, data needs
    requirements  the three data needs (REQUIREMENTS): close and dollar volume only, other kline
                  fields, perpetual-futures data
    families      per family: definitions and their data needs
    categories    per category: definitions and their data needs
    commands      {where, check, register}: the judge.py commands of a definition ({id} is
                  its id); the page shows both commands for each definition
    index_info    the counts of the search index; the index itself is the PART FILE
                  data/library.index.js (common.with_parts; FL.data["library.index"]), which
                  the page loads when the search card comes near the screen: every
                  definition, compactly: names front-coded in id order (ids are CF00001.. in
                  that order, so they are implicit), family runs, category, data-need and
                  source codes (the data of a test: 0 OKX, 1 Binance, userfactor.auto_source).
                  expr_parts names the parts data/library.expr-<k>.js: the canonical
                  expression of each definition in index order, expr_chunk definitions each;
                  the page loads the part of a definition when the search shows it
    engine        the library ON FACTORLAB'S DATA (engine/libfactors.py): the funnel from
                  definitions to computable, computed, point-in-time verified and screened;
                  what cannot be computed and why; the generator-zone (G) screen summary; the
                  definitions with a residual-IC q <= 0.10 (at most TOP_MAX rows by rank; "n"
                  and "n_q05" are the full counts, and size05/size10 and repid05/repid10 give
                  each row's signal size and representative over every such definition)
    binance       the same library on Binance's public data for the same 20 coins
                  (libfactors.py --source binance; Binance is a data source only): the
                  funnel, the point-in-time conventions and checks, what the new fields give,
                  the NEW FAMILY (the definitions the OKX screen did not screen, its own
                  Benjamini-Hochberg family) with its definitions at a residual-IC q <= 0.10,
                  and the CROSS-CHECK of the OKX-screened definitions (agreement numbers
                  only: no q-value on Binance; with the manifest's two notes, that the
                  agreement is largely mechanical and that on G both venues read the same
                  Binance funding archive). Labels and regressors are the OKX screen's.
                  Both blocks name the code files that changed since their computation
                  (meta.code_changed, from the manifest's code hashes).

Front coding: entry i is ``name[i] = name[i-1][:p[i]] + s[i]``. Consecutive ids
come from one template (``stat.mean.ret.w3``, ``stat.mean.ret.w5``), so this
halves the name list; the page decodes it once.

The engine block reads only the three small files of ``<config.DATA>/library_on_engine/``
(status.parquet, screen_g.parquet, manifest.json), never the git-ignored ``values/``
shards. Its counts are recomputed from the two parquet files and checked against the
manifest (a mismatch fails the export). If the folder is missing the block is
``{"available": false}`` and the page shows how to compute it. The binance block works the
same way on its four files (never its ``values/``); without the folder it is
``{"available": false}`` and the page shows the download, merge and compute commands.

The code of the section is in three modules: webexport/library_catalog.py (the catalog,
the families, the search index and the expression parts), library_engine.py (the engine
block) and library_binance.py (the binance block). This module holds build() and gives the
names of all three (``__all__``).
"""

from __future__ import annotations

import json
from pathlib import Path

from webexport.common import data_path, with_parts
from webexport.library_binance import (
    BINANCE_COMMANDS,
    BINANCE_FILES,
    BINANCE_NEW_FIELDS,
    BINANCE_NOTES,
    BINANCE_ORIGIN_LABELS,
    BINANCE_ORIGINS,
    BINANCE_PROVENANCE_KEYS,
    BINANCE_STATS,
    BINANCE_SUBDIR,
    BINANCE_TERMS,
    CLEANING_LABELS,
    REL_BINANCE_DIR,
    RHO_DP,
    _binance,
    _cleaned_where,
    input_class,
)
from webexport.library_catalog import (
    CATALOG,
    DERIV_FIELDS,
    EXPR_CHUNK,
    FAMILY_IDEAS,
    FREE_FIELDS,
    REL_DB,
    REL_README,
    REQUIREMENTS,
    SOURCES,
    T_DP,
    _categories,
    _connect,
    _families,
    _index,
    _q,
    _t,
    canonical,
    expr_parts,
    front_code,
    front_decode,
    requirement,
    sources,
)
from webexport.library_engine import (
    ENGINE_COMMANDS,
    ENGINE_DIR,
    ENGINE_FILES,
    ENGINE_NOTES,
    ENGINE_QS,
    FIELD_LABELS,
    MEAN_DP,
    Q_DP,
    REL_ENGINE_DIR,
    TOP_MAX,
    TOP_Q,
    _bh,
    _code_changed,
    _engine,
)

# The code of the section is in three modules (library_catalog, library_engine and
# library_binance). This module gives their names as one module: the tests read them here.
__all__ = [
    "BINANCE_COMMANDS",
    "BINANCE_FILES",
    "BINANCE_NEW_FIELDS",
    "BINANCE_NOTES",
    "BINANCE_ORIGINS",
    "BINANCE_ORIGIN_LABELS",
    "BINANCE_PROVENANCE_KEYS",
    "BINANCE_STATS",
    "BINANCE_SUBDIR",
    "BINANCE_TERMS",
    "CATALOG",
    "CLEANING_LABELS",
    "COMMANDS",
    "DERIV_FIELDS",
    "ENGINE_COMMANDS",
    "ENGINE_DIR",
    "ENGINE_FILES",
    "ENGINE_NOTES",
    "ENGINE_QS",
    "EXPR_CHUNK",
    "FAMILY_IDEAS",
    "FIELD_LABELS",
    "FREE_FIELDS",
    "MEAN_DP",
    "Q_DP",
    "REL_BINANCE_DIR",
    "REL_DB",
    "REL_ENGINE_DIR",
    "REL_README",
    "REQUIREMENTS",
    "RHO_DP",
    "SOURCES",
    "TOP_MAX",
    "TOP_Q",
    "T_DP",
    "_bh",
    "_binance",
    "_categories",
    "_cleaned_where",
    "_code_changed",
    "_connect",
    "_engine",
    "_families",
    "_index",
    "_q",
    "_t",
    "build",
    "canonical",
    "expr_parts",
    "front_code",
    "front_decode",
    "input_class",
    "requirement",
    "sources",
]

# the two commands of each definition: {id} is its id (the page puts it in with textContent)
COMMANDS = {
    "where": "Run these commands from the repository root.",
    "check": "cd engine && .venv/bin/python -B judge.py check --library {id}",
    "register": "cd engine && .venv/bin/python -B judge.py register --library {id}",
}


def build(
    catalog: Path | None = None,
    engine_dir: Path | None = None,
    binance_dir: Path | None = None,
) -> dict:
    cat = Path(catalog) if catalog is not None else CATALOG
    db = cat / "factors.db"
    if not db.is_file():
        return {
            "schema": 1,
            "available": False,
            "reason": f"{REL_DB} is not built",
            "source": {"db": REL_DB, "readme": REL_README},
        }
    con = _connect(db)
    try:
        fac = _q(
            con,
            "SELECT factor_id, name, family, category, expression, raw_inputs, needs_derivs FROM factors ORDER BY factor_id",
        )
        meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM meta")}
    finally:
        con.close()

    families = sorted(fac["family"].unique())
    categories = sorted(fac["category"].unique())
    req = fac["raw_inputs"].map(requirement)
    edir = Path(engine_dir) if engine_dir is not None else data_path(ENGINE_DIR)
    src = sources(fac)
    index = _index(fac, families, categories, src)
    exprs = expr_parts(fac)
    payload = {
        "schema": 1,
        "available": True,
        "source": {"db": REL_DB, "readme": REL_README},
        "totals": {
            "definitions": len(fac),
            "candidates": int(meta["n_candidates"]),
            "rejected": {k: int(v) for k, v in meta["rejected"].items()},
            "validation_data": meta.get("validation_data"),
            "families": len(families),
            "categories": len(categories),
            "requirements": [int((req == i).sum()) for i in range(len(REQUIREMENTS))],
            "needs_derivs": int(fac["needs_derivs"].sum()),
            "sources": [int(sum(1 for x in src if x == i)) for i in range(len(SOURCES))],
        },
        "requirements": REQUIREMENTS,
        "families": _families(fac),
        "categories": _categories(fac),
        "commands": dict(COMMANDS),
        # the search index is a part file (data/library.index.js), the expressions are the parts
        # data/library.expr-<k>.js: the page loads them when the search needs them, so the first
        # view does not wait for them
        "index_info": {
            "part": "index",
            "n": len(fac),
            "families": len(families),
            "categories": len(categories),
            "expr_parts": list(exprs),
            "expr_chunk": EXPR_CHUNK,
        },
        "engine": _engine(edir, fac, families, db),
        # the same library on Binance's public data for the same coins: its own block,
        # shown when <library_on_engine>/binance exists, an empty state otherwise
        "binance": _binance(
            Path(binance_dir) if binance_dir is not None else edir / BINANCE_SUBDIR, edir, fac, families, db
        ),
    }
    return with_parts(payload, {"index": index, **exprs})
