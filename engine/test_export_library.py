"""Tests for the factor-library section of the web export (webexport/library.py + pages/library.js).

Script-style like the other engine tests:  python -B test_export_library.py
Reads <repo>/library/catalog read-only (content hashes of the catalog and of the engine
data directory are compared around build(), and every SQLite connection the builder
opens must be a mode=ro URI). The page's numbers are recomputed here from the catalog's
CSV export and SUMMARY.md with plain pandas -- not through the builder, the library package
or the database -- and compared:

  totals    definitions, candidates and build rejects, families, categories, data needs, the
            data source of a test (OKX or Binance)
  families  per-family and per-category counts and data needs
  index     front-coded names, family runs, category, data-need and source codes, decoded and
            compared row by row
  exprs     the expression parts: the canonical text of each definition in index order, equal
            to the identity that judge.py records
  commands  the check and register command of each definition, equal to the commands of
            python -m factorlab show
  words     no verdict, tier, stage or gate-pass word in a key or a text of the payload and its parts
  engine    the library on Factorlab's data (engine/data/library_on_engine): the groups
            re-derived from the catalog's raw inputs, the funnel, what cannot be computed,
            the point-in-time cells, the G-zone screen (Benjamini-Hochberg q-values
            recomputed here, distinct signals, stability, families), the zones against
            factorfinder/eval_config.toml and the table of the definitions with a low rIC q,
            all against status.parquet, screen_g.parquet, manifest.json, the raw open-interest
            files and the catalog CSV; the builder never opens values/; a missing folder
            or a missing screen is an empty state
  binance   the same library on Binance's data (engine/data/library_on_engine/binance):
            roles, origins and new fields re-derived from the catalog and
            the OKX status, the funnel and PIT cells, the new family's Benjamini-Hochberg
            q-values, distinct signals, G_fit replication, overlap and table, the
            cross-check's agreement statistics (Spearman, signs, input classes with gap found
            in the formula, the agreement without the funding readers, largest
            disagreements) with its two notes (mechanical agreement; the
            same funding archive on G), all against
            status.parquet, screen_g.parquet, crosscheck.parquet and manifest.json; the panel
            checks against the manifest and, when the git-ignored panel is on disk
            (library/data/binance_um_1d), recomputed from it and the engine's raw files; a
            missing folder is a "not computed" state; each block says when the code changed
            after the computation of its outputs (the manifest's code hashes)
  file      size <= 1.5 MB for library.js and for each part file, ASCII wrapper, privacy
            guard, no machine paths, freshness
  page      file rules, nav and script wiring, and a node render with searches (sorted over
            every match), the formulas and commands of the rows, the table filters and the
            page order (needs node on PATH, NODE or FACTORLAB_NODE; skipped otherwise)

The data files are read from web/data, or from FACTORLAB_WEB_DATA (a temp export of
export_web.py --out); the page and asset files always come from web/.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from webexport import SECTIONS
from webexport import common as C
from webexport import library as X

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
WEB = REPO / "web"
# the exported data files: web/data, or a temp export (FACTORLAB_WEB_DATA)
WEB_DATA = Path(os.environ.get("FACTORLAB_WEB_DATA") or (WEB / "data"))
CAT = REPO / "library" / "catalog"
WRAP_RE = re.compile(r'^window\.FL=window\.FL\|\|\{\};FL\.data=FL\.data\|\|\{\};FL\.data\["library"\]=(.*);\n$', re.S)
WRAP_INDEX_RE = re.compile(
    r'^window\.FL=window\.FL\|\|\{\};FL\.data=FL\.data\|\|\{\};FL\.data\["library\.index"\]=(.*);\n$', re.S
)
WRAP_PART_RE = re.compile(
    r'^window\.FL=window\.FL\|\|\{\};FL\.data=FL\.data\|\|\{\};FL\.data\["library\.([a-z0-9-]+)"\]=(.*);\n$', re.S
)
# the words that the page and its data must not show (numbers only): verdicts, tiers, stages, gate-pass words
VERDICT_WORDS = re.compile(
    r"\b(useful|useless|unproven|effective|survivors?|promotable|novel|confirmed|retired|probation|"
    r"healthy|honesty|tiers?|stages?)\b"
    r"|\bpass(es|ed)?\s+(the\s+|all\s+|every\s+)?(gates?|screen)\b|\bgates?\s+pass(es|ed)?\b|forward-only",
    re.IGNORECASE,
)
# the two commands of a definition (judge.py)
CHECK = "cd engine && .venv/bin/python -B judge.py check --library {id}"
REGISTER = "cd engine && .venv/bin/python -B judge.py register --library {id}"
WHERE = "Run these commands from the repository root."
MACHINE_PATHS = re.compile(r"/tmp/|/private/|/Users/|/home/|file://")
FREE = {"close", "quote_volume"}
DERIVS = {
    "funding_rate",
    "open_interest",
    "open_interest_value",
    "long_short_ratio",
    "top_long_short_ratio",
    "taker_buy_sell_ratio",
    "premium",
}

# the engine's supplied raw fields (libfactors.SUPPLIED), written out here independently
SUPPLIED = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "funding_rate",
    "open_interest",
    "open_interest_value",
}
ENG = C.DATA / "library_on_engine"
BIN = ENG / "binance"
PANEL = REPO / "library" / "data" / "binance_um_1d"  # git-ignored; the panel checks recompute from it when present
# the library's raw fields Factorlab's OKX data lacks, in the library's own field order, and the open-interest fields
BIN_ONLY = [
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "long_short_ratio",
    "top_long_short_ratio",
    "taker_buy_sell_ratio",
    "premium",
]
OI = ["open_interest", "open_interest_value"]

_PAYLOAD = None
_PARTS = None
_SRC = None
_ESRC = None
_BSRC = None


def _split(built: dict) -> tuple[dict, dict]:
    """(the section payload as export_web writes it, {part: payload}) of a build() result."""
    built = dict(built)
    parts = built.pop(C.PARTS_KEY, None) or {}
    if parts:
        built["parts"] = sorted(parts)
    return json.loads(C.dumps(built)), {k: json.loads(C.dumps(v)) for k, v in parts.items()}


def payload() -> dict:
    global _PAYLOAD, _PARTS
    if _PAYLOAD is None:
        _PAYLOAD, _PARTS = _split(X.build())
    return _PAYLOAD


def part(name: str) -> dict:
    payload()
    return _PARTS[name]


def src() -> dict:
    """The catalog's own exports, loaded once with plain pandas."""
    global _SRC
    if _SRC is None:
        _SRC = {
            "fac": pd.read_csv(CAT / "crypto_factors.csv.gz"),
            "summary_md": (CAT / "SUMMARY.md").read_text(encoding="utf-8"),
        }
    return _SRC


def _texts(x) -> list:
    """Every string value and every key of a JSON-like object, at each depth."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [str(k) for k in x] + [t for v in x.values() for t in _texts(v)]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _texts(v)]
    return []


def _word_hits(texts) -> list:
    """The texts with a word of VERDICT_WORDS (an underscore counts as a space)."""
    return [t for t in texts if VERDICT_WORDS.search(str(t).replace("_", " "))]


def okx_source(fac: pd.DataFrame) -> list[bool]:
    """True where judge.py tests the definition on the OKX data: every raw field is an OKX field, no
    open interest, and no vwap or vwap_dev in the formula (written out here, not userfactor.auto_source)."""
    sets = fac["raw_inputs"].map(lambda x: set(x.split(",")))
    vwap = fac["expression"].str.contains(r"\bvwap(?:_dev)?\b", regex=True)
    return [f <= SUPPLIED - set(OI) and not v for f, v in zip(sets, vwap)]


def esrc() -> dict:
    """libfactors.py's three committed outputs, read with plain pandas and json."""
    global _ESRC
    if _ESRC is None:
        _ESRC = {
            "st": pd.read_parquet(ENG / "status.parquet"),
            "g": pd.read_parquet(ENG / "screen_g.parquet"),
            "man": json.loads((ENG / "manifest.json").read_text(encoding="ascii")),
        }
    return _ESRC


def bsrc() -> dict:
    """libfactors.py --source binance's four committed outputs, read with plain pandas and json."""
    global _BSRC
    if _BSRC is None:
        _BSRC = {
            "st": pd.read_parquet(BIN / "status.parquet"),
            "g": pd.read_parquet(BIN / "screen_g.parquet"),
            "cc": pd.read_parquet(BIN / "crosscheck.parquet"),
            "man": json.loads((BIN / "manifest.json").read_text(encoding="ascii")),
        }
    return _BSRC


def has_binance() -> bool:
    return all(
        (BIN / n).is_file() for n in ("status.parquet", "screen_g.parquet", "crosscheck.parquet", "manifest.json")
    )


def has_engine() -> bool:
    return all((ENG / n).is_file() for n in ("status.parquet", "screen_g.parquet", "manifest.json"))


def close(a, b, tol, what=""):
    assert a is not None and b is not None, f"{what}: None ({a!r} vs {b!r})"
    assert abs(float(a) - float(b)) <= tol, f"{what}: {a!r} != {b!r} (tol {tol})"


def content_hashes(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def bh_q(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values, written out independently of factorlab.stats."""
    n = len(p)
    order = np.argsort(p, kind="stable")
    q = np.empty(n)
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * n / rank)
        q[i] = running
    return np.minimum(q, 1.0)


# --------------------------------------------------------------------------- failure messages

SHOW = 20  # a failure message shows at most this many items


def _ne(a, b) -> bool:
    """a != b for one item of a message; a comparison that gives no single truth value counts as a difference."""
    try:
        return bool(a != b)
    except (TypeError, ValueError):
        return True


def num(x) -> str:
    """A number with thousands separators; any other value (None, a text) as repr. A message never
    fails on the value that it shows."""
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)):
        return f"{x:,}"
    return repr(x)


def total(*xs):
    """The sum of the values when each one is a number, else None (for a message: num(total(...)))."""
    nums = [
        x for x in xs if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))
    ]
    return sum(nums) if len(nums) == len(xs) else None


def _short(x, n: int = 200) -> str:
    """repr(x), cut to at most n characters."""
    r = repr(x)
    return r if len(r) <= n else r[: n - 3] + "..."


def shown(items) -> str:
    """The count and at most SHOW of the items: '3: [a, b, c]'."""
    xs = list(items)
    more = f", ... and {len(xs) - SHOW:,} more" if len(xs) > SHOW else ""
    return f"{len(xs):,}: [" + ", ".join(str(x) for x in xs[:SHOW]) + more + "]"


def neq(what: str, got, want) -> str:
    """The message of a value or count check: both sides."""
    return f"{what}: {_short(got, 600)} != {_short(want, 600)}"


def diff_sets(what: str, got, want) -> str:
    """The message of a set check: the two counts and the items that only one side has."""
    got, want = set(got), set(want)
    return (
        f"{what}: {len(got):,} against {len(want):,}; only on the left {shown(sorted(got - want, key=str))}; "
        f"only on the right {shown(sorted(want - got, key=str))}"
    )


def diff_lists(what: str, got, want, ids=None) -> str:
    """The message of a list check: the two lengths and the positions that differ, each named by
    `ids` (for example the factor ids) when given, else by its position."""
    got, want = list(got), list(want)
    names = [] if ids is None else list(ids)
    bad = [i for i in range(max(len(got), len(want))) if i >= len(got) or i >= len(want) or _ne(got[i], want[i])]
    rows = [
        f"{names[i] if i < len(names) else i}: {_short(got[i]) if i < len(got) else '(none)'} != "
        f"{_short(want[i]) if i < len(want) else '(none)'}"
        for i in bad
    ]
    return f"{what}: lengths {len(got):,} and {len(want):,}; positions that differ {shown(rows)}"


def diff_dicts(what: str, got, want) -> str:
    """The message of a dict check: the keys whose values differ (or that only one side has), with both values."""
    got, want = dict(got), dict(want)
    keys = sorted(set(got) | set(want), key=str)
    rows = [
        f"{k}: {_short(got[k]) if k in got else '(none)'} != {_short(want[k]) if k in want else '(none)'}"
        for k in keys
        if k not in got or k not in want or _ne(got[k], want[k])
    ]
    return f"{what}: keys that differ {shown(rows)}"


def failing(what: str, ok, ids) -> str:
    """The message of an element-wise check: the ids where `ok` is False (in order) and the counts."""
    flags = []
    for x in np.asarray(ok, dtype=object).ravel():
        try:
            flags.append(bool(x))
        except (TypeError, ValueError):  # a missing value (pd.NA) fails
            flags.append(False)
    names = list(ids)
    bad = [names[i] if i < len(names) else i for i, f in enumerate(flags) if not f]
    return f"{what}: {len(bad):,} of {len(flags):,} fail; {shown(bad)}"


def missing_text(needle, text) -> str:
    """The message of a text check: the text that is not there and the start of the text searched."""
    text = str(text)
    return f"{needle!r} is not in the text ({len(text):,} characters): {text[:300]!r}"


def text_check(what: str, text, present=(), absent=()) -> str:
    """The message of a check that some texts are in a text and others are not: the ones that fail."""
    text = str(text)
    return (
        f"{what}: not found {[w for w in present if w not in text]}; "
        f"found but not allowed {[w for w in absent if w in text]}"
    )


def positions(what: str, ids, names) -> str:
    """The message of an order check: the position of each name in ids (None when it is absent)."""
    ids = list(ids)
    return f"{what}: " + ", ".join(f"{x} {ids.index(x) if x in ids else None}" for x in names)


# --------------------------------------------------------------------------- tests


def test_build_is_read_only():
    global _PAYLOAD
    cat_before, data_before = content_hashes(CAT), content_hashes(C.DATA)
    uris = []
    real = sqlite3.connect

    def spy(database, *a, **kw):
        uris.append((str(database), bool(kw.get("uri"))))
        return real(database, *a, **kw)

    sqlite3.connect = spy  # the one sqlite3 module: library_catalog._connect calls it
    try:
        _PAYLOAD = None
        payload()
    finally:
        sqlite3.connect = real
    assert uris and all(u and s.startswith("file:") and s.endswith("?mode=ro") for s, u in uris), (
        f"the catalog connections (path, uri flag) are not all read-only file: URIs: {uris}"
    )
    assert content_hashes(CAT) == cat_before, "build() changed the library catalog"
    assert content_hashes(C.DATA) == data_before, "build() changed files under the engine data directory"
    assert not list(CAT.glob("*-journal")) and not list(CAT.glob("*-wal")), (
        f"SQLite journal files in the catalog folder: {shown(p.name for p in [*CAT.glob('*-journal'), *CAT.glob('*-wal')])}"
    )


def test_totals_recomputed():
    s, d = src(), payload()
    fac, t = s["fac"], d["totals"]
    assert d["available"] is True and d["schema"] == 1, f"available {d['available']!r}, schema {d.get('schema')!r}"
    assert t["definitions"] == len(fac) == 28_904, (
        f"definitions: payload {num(t['definitions'])}, catalog CSV {len(fac):,}, expected 28,904"
    )
    assert t["families"] == fac["family"].nunique() == 13 and t["categories"] == fac["category"].nunique() == 15, (
        f"families: payload {t['families']}, catalog CSV {fac['family'].nunique()}, expected 13; "
        f"categories: payload {t['categories']}, catalog CSV {fac['category'].nunique()}, expected 15"
    )
    m = re.search(r"Total factors: \*\*([\d,]+)\*\* \(from ([\d,]+) generated candidates\)", s["summary_md"])
    assert (
        int(m.group(1).replace(",", "")) == t["definitions"]
        and int(m.group(2).replace(",", "")) == t["candidates"] == 29_905
    ), (
        f"SUMMARY.md: {m.group(1)} definitions from {m.group(2)} candidates; payload: {num(t['definitions'])} "
        f"definitions from {num(t['candidates'])} candidates (expected 29,905)"
    )
    rej = ast.literal_eval(
        re.search(r"Rejected during validation: (\{.*\})", s["summary_md"]).group(1)
    )  # a dict literal written by build.py
    assert t["rejected"] == rej and t["candidates"] - sum(rej.values()) == t["definitions"], (
        diff_dicts("rejected (payload against SUMMARY.md)", t["rejected"], rej)
        + f"; candidates {num(t['candidates'])} - rejected {sum(rej.values()):,} = "
        f"{num(total(t['candidates'], -sum(rej.values())))}, definitions {num(t['definitions'])}"
    )
    sets = fac["raw_inputs"].map(lambda x: set(x.split(",")))
    need = [
        int(sets.map(lambda f: f <= FREE).sum()),
        int(sets.map(lambda f: not f <= FREE and not f & DERIVS).sum()),
        int(sets.map(lambda f: bool(f & DERIVS)).sum()),
    ]
    assert t["requirements"] == need and sum(need) == len(fac), (
        f"requirements (free, kline, derivs): payload {t['requirements']}, catalog CSV {need} "
        f"(sum {sum(need):,}, definitions {len(fac):,})"
    )
    assert t["needs_derivs"] == int(fac["needs_derivs"].sum()) == need[2] == 11_609, (
        f"needs_derivs: payload {num(t['needs_derivs'])}, catalog column {int(fac['needs_derivs'].sum()):,}, "
        f"raw inputs {need[2]:,}, expected 11,609"
    )
    assert need[0] + need[1] == 17_295, (
        f"definitions without a derivatives field: {need[0]:,} + {need[1]:,} = {need[0] + need[1]:,}, expected 17,295"
    )
    okx = okx_source(fac)
    assert t["sources"] == [sum(okx), len(okx) - sum(okx)], neq(
        "sources (OKX, Binance), payload against the catalog CSV", t["sources"], [sum(okx), len(okx) - sum(okx)]
    )
    assert [r["key"] for r in d["requirements"]] == ["free", "kline", "derivs"], neq(
        "requirement keys", [r["key"] for r in d["requirements"]], ["free", "kline", "derivs"]
    )
    # a catalog from `factorlab build` holds the definitions only: no other block
    assert set(d) == {
        "schema",
        "available",
        "source",
        "totals",
        "requirements",
        "families",
        "categories",
        "commands",
        "index_info",
        "engine",
        "binance",
        "parts",
    }, sorted(d)
    assert set(t) == {
        "definitions",
        "candidates",
        "rejected",
        "validation_data",
        "families",
        "categories",
        "requirements",
        "needs_derivs",
        "sources",
    }, sorted(t)


def test_families_and_categories_recomputed():
    s, d = src(), payload()
    fac = s["fac"]
    sets = fac["raw_inputs"].map(lambda x: set(x.split(",")))
    need = np.array([0 if f <= FREE else 2 if f & DERIVS else 1 for f in sets])
    rows = {r["family"]: r for r in d["families"]}
    assert set(rows) == set(fac["family"]) and [r["n"] for r in d["families"]] == sorted(
        (r["n"] for r in d["families"]), reverse=True
    ), (
        diff_sets("families (payload against the catalog CSV)", rows, fac["family"])
        + f"; counts in page order {[(r['family'], r['n']) for r in d['families']]}"
    )
    for fam, n in fac["family"].value_counts().items():
        r = rows[fam]
        m = (fac["family"] == fam).to_numpy()
        want = (n, [int((need[m] == k).sum()) for k in range(3)])
        assert (r["n"], r["requirements"]) == want, neq(
            f"family {fam}: (n, requirements) against the CSV", (r["n"], r["requirements"]), want
        )
        assert sum(r["requirements"]) == r["n"], neq(
            f"family {fam}: the sum of the requirements against n", sum(r["requirements"]), r["n"]
        )
        assert r["idea"], f"family {fam}: the idea text is empty ({r['idea']!r})"
        assert set(r) == {"family", "idea", "n", "requirements"}, diff_sets(
            f"family {fam}: keys", r, {"family", "idea", "n", "requirements"}
        )
    assert sum(r["n"] for r in d["families"]) == len(fac), neq(
        "the sum of the family counts against the definitions", sum(r["n"] for r in d["families"]), len(fac)
    )
    cats = {r["category"]: r for r in d["categories"]}
    for cat, n in fac["category"].value_counts().items():
        m = (fac["category"] == cat).to_numpy()
        assert cats[cat]["n"] == n, neq(f"category {cat}: n against the CSV", cats[cat]["n"], n)
        want = [int((need[m] == k).sum()) for k in range(3)]
        assert cats[cat]["requirements"] == want, neq(
            f"category {cat}: requirements against the CSV", cats[cat]["requirements"], want
        )
        assert set(cats[cat]) == {"category", "n", "requirements"}, diff_sets(
            f"category {cat}: keys", cats[cat], {"category", "n", "requirements"}
        )


def test_index_decodes_to_the_catalog():
    s, d = src(), payload()
    fac = s["fac"]
    # the index and the formulas are part files; the section payload names them and carries the counts
    n_parts = math.ceil(len(fac) / 8000)
    exprs = [f"expr-{k}" for k in range(n_parts)]
    assert "index" not in d and d["parts"] == sorted(["index", *exprs]), (
        f"'index' in the section payload: {'index' in d}; " + neq("parts", d.get("parts"), sorted(["index", *exprs]))
    )
    ix = part("index")
    want_info = {
        "part": "index",
        "n": len(fac),
        "families": fac["family"].nunique(),
        "categories": fac["category"].nunique(),
        "expr_parts": exprs,
        "expr_chunk": 8000,
    }
    assert d["index_info"] == want_info, diff_dicts("index_info", d["index_info"], want_info)
    a = ix["all"]
    names, prev = [], ""
    for k, suf in zip(a["p"], a["s"]):
        prev = prev[:k] + suf
        names.append(prev)
    assert names == fac["name"].tolist(), "front-coded names do not decode to the catalog"
    ids = [f"{a['id_prefix']}{i + a['id_start']:0{a['id_digits']}d}" for i in range(a["n"])]
    assert ids == fac["factor_id"].tolist(), diff_lists("decoded ids against the catalog", ids, fac["factor_id"])
    fam = [None] * a["n"]
    for code, start, count in a["fam_runs"]:
        fam[start : start + count] = [ix["families"][code]] * count
    assert fam == fac["family"].tolist(), diff_lists(
        "families of the family runs against the catalog", fam, fac["family"], fac["factor_id"]
    )
    assert [ix["categories"][c] for c in a["cat"]] == fac["category"].tolist(), diff_lists(
        "categories against the catalog", [ix["categories"][c] for c in a["cat"]], fac["category"], fac["factor_id"]
    )
    sets = fac["raw_inputs"].map(lambda x: set(x.split(",")))
    want_req = [0 if f <= FREE else 2 if f & DERIVS else 1 for f in sets]
    assert a["req"] == want_req, diff_lists("data-need codes against the catalog", a["req"], want_req, fac["factor_id"])
    assert ix["sources"] == ["okx", "binance"], neq("source names", ix["sources"], ["okx", "binance"])
    assert a["source"] == [0 if o else 1 for o in okx_source(fac)], diff_lists(
        "data-source codes against the catalog", a["source"], [0 if o else 1 for o in okx_source(fac)], fac["factor_id"]
    )
    keys = {"n", "id_prefix", "id_digits", "id_start", "p", "s", "fam_runs", "cat", "req", "source"}
    assert set(a) == keys, diff_sets("keys of the index rows", a, keys)
    assert set(ix) == {"families", "categories", "requirements", "sources", "all"}, sorted(ix)
    assert [r["key"] for r in ix["requirements"]] == ["free", "kline", "derivs"], neq(
        "requirement keys of the index", [r["key"] for r in ix["requirements"]], ["free", "kline", "derivs"]
    )


def test_expression_parts_decode_to_the_catalog():
    """The parts expr-0, expr-1, ... hold the canonical text of each definition in index order,
    8,000 each. The text is the identity that judge.py records for the formula
    (userfactor.validate_formula on a sample), and a parse of the text gives the catalog's tree."""
    import userfactor as uf

    fac = src()["fac"]
    info = payload()["index_info"]
    got = []
    for k, name in enumerate(info["expr_parts"]):
        x = part(name)
        got_x = (sorted(x), x.get("start"), x.get("n"))
        want_x = (["expr", "n", "start"], k * 8000, len(x.get("expr") or []))
        assert got_x == want_x, neq(
            f"part {name}: (keys, start, n) against (keys, k * 8000, the expressions)", got_x, want_x
        )
        assert x["n"] == min(8000, len(fac) - x["start"]), neq(
            f"part {name}: n against min(8000, definitions - start)", x["n"], min(8000, len(fac) - x["start"])
        )
        got += x["expr"]
    assert len(got) == len(fac), neq("expression texts in the parts against the definitions", len(got), len(fac))
    # a parse of each text and of the catalog's text gives the same tree (the catalog writes
    # 2.000000 where the canonical text writes 2.0)
    for g, e in zip(got, fac["expression"]):
        assert ast.dump(ast.parse(g, mode="eval")) == ast.dump(ast.parse(e, mode="eval")), (
            f"the tree of the part text {g!r} differs from the tree of the catalog text {e!r}"
        )
        assert g == ast.unparse(ast.parse(e, mode="eval")), neq(
            "the part text against the canonical catalog text", g, ast.unparse(ast.parse(e, mode="eval"))
        )
    differ = [i for i, (g, e) in enumerate(zip(got, fac["expression"])) if g != e]
    assert differ, "the canonical text differs from the catalog text for some definitions"
    sample = sorted(set(range(0, len(fac), 997)) | set(differ[:20]))
    for i in sample:
        canon = uf.validate_formula(fac["expression"].iloc[i])["canonical"]
        assert canon == got[i], neq(
            f"{fac['factor_id'].iloc[i]}: validate_formula's canonical text against the part", canon, got[i]
        )


def test_register_command_per_formula():
    """Each definition has a check command and a register command: the payload holds the two
    templates ({id} is the id); the page puts the id in with textContent; python -m factorlab show
    prints the same two commands, under the line that names the repository root."""
    import subprocess
    import sys

    d = payload()
    want = {"where": WHERE, "check": CHECK, "register": REGISTER}
    assert d["commands"] == want, diff_dicts("commands", d["commands"], want)
    fid = src()["fac"]["factor_id"].iloc[966]
    r = subprocess.run(
        [sys.executable, "-B", "-m", "factorlab", "show", fid],
        cwd=REPO / "library",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, f"python -m factorlab show {fid}: exit {r.returncode}; stderr {r.stderr[-800:]!r}"
    out = r.stdout
    assert "Run these commands from the repository root:" in out, missing_text(
        "Run these commands from the repository root:", out
    )
    assert CHECK.format(id=fid) in out and REGISTER.format(id=fid) in out, text_check(
        f"python -m factorlab show {fid}", out, present=(CHECK.format(id=fid), REGISTER.format(id=fid))
    )
    js = (WEB / "pages" / "library.js").read_text(encoding="utf-8")
    split_join = 'cmds[k].split("{id}").join(id)'
    assert split_join in js and "ui.code(" in js, (
        f"library.js lacks {[w for w in (split_join, 'ui.code(') if w not in js]}"
    )
    # the commands are in the search rows and in the two screen tables
    assert js.count("library_cmds(cmds, r.id)") == 3, neq(
        "the calls library_cmds(cmds, r.id) in library.js", js.count("library_cmds(cmds, r.id)"), 3
    )


def test_payload_has_no_verdict_words():
    """No verdict, tier, stage or gate-pass word in a key or a text of the payload and its parts,
    and in the text of the page."""
    d = payload()
    hits = _word_hits(_texts(d))
    assert not hits, f"texts and keys of the payload with a verdict word: {shown(hits)}"
    for name in d["parts"]:
        hits = _word_hits(_texts(part(name)))
        assert not hits, f"texts and keys of the part {name} with a verdict word: {shown(hits)}"
    js = (WEB / "pages" / "library.js").read_text(encoding="utf-8")
    code = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    code = re.sub(r"(?m)^\s*//.*$", "", code)
    lits = re.findall(r'"((?:[^"\\\n]|\\.)*)"', code)
    hits = _word_hits(lits)
    assert not hits, f"string literals of library.js with a verdict word: {shown(hits)}"
    # the word list has teeth
    teeth = ["the survivors", "stays unproven", "the top tier", "it passes the screen"]
    assert _word_hits(teeth) == teeth, neq("the word hits of four texts with a verdict word", _word_hits(teeth), teeth)


def test_file_size_privacy_and_freshness():
    d = payload()
    sizes = {}
    files = [("library.js", d, None)] + [(f"library.{n}.js", part(n), n) for n in d["parts"]]
    for name, obj, key in files:
        text = C.render_section("library", obj, part=key)  # raises on a private string
        raw = text.encode("ascii")
        sizes[name] = len(raw)
        assert len(raw) <= C.SECTION_MAX_BYTES, f"{name} would be {len(raw):,} bytes"
        assert not C.privacy_hits(text), C.privacy_hits(text)  # the exporter's own guard
        assert not MACHINE_PATHS.search(text), f"a machine path reached {name}"
        assert not re.search(r"(?i)generated (?:by|with)|co-autho[r]ed-by", text), (  # no credit lines
            f"a credit line in {name}: {re.search(r'(?i)generated (?:by|with)|co-autho[r]ed-by', text).group(0)!r}"
        )
    assert sum(sizes.values()) <= C.TOTAL_MAX_BYTES, (
        f"all library data files: {sum(sizes.values()):,} bytes, budget {C.TOTAL_MAX_BYTES:,}"
    )
    # the first view does not carry the search index: the section file stays small
    assert sizes["library.js"] < sizes["library.index.js"] or part("index")["all"]["n"] < 2000, (
        f"library.js {num(sizes['library.js'])} bytes is not smaller than library.index.js "
        f"{num(sizes['library.index.js'])} bytes; index rows {num(part('index')['all']['n'])}"
    )
    f = WEB_DATA / "library.js"
    if f.exists():  # the exported file is the current export
        m = WRAP_RE.match(f.read_text(encoding="ascii"))
        assert m, "library.js does not use the data-file wrapper"
        assert json.loads(m.group(1)) == d, f"{f} is stale: run python export_web.py --only library"
        for n in d["parts"]:
            fi = WEB_DATA / f"library.{n}.js"
            mi = WRAP_PART_RE.match(fi.read_text(encoding="ascii")) if fi.exists() else None
            assert mi and mi.group(1) == n and json.loads(mi.group(2)) == part(n), f"{fi} is missing or stale"
    print(
        f"        (library.js: {sizes['library.js']:,} bytes, library.index.js: {sizes['library.index.js']:,} bytes; "
        f"budget {C.SECTION_MAX_BYTES:,} each)"
    )


def test_missing_catalog_is_an_empty_state():
    with tempfile.TemporaryDirectory() as tmp:
        d = X.build(Path(tmp))
        assert d["available"] is False and "is not built" in d["reason"], (
            f"available {d['available']!r}, reason {d.get('reason')!r}"
        )
        C.render_section("library", d)
        assert not list(Path(tmp).iterdir()), "the builder created a file"


def test_front_code_roundtrip():
    words = ["stat.mean.ret.w3", "stat.mean.ret.w5", "stat.mean.ret.w10", "a", "", "ab", "stat"]
    p, s = X.front_code(words)
    assert X.front_decode(p, s) == words and p[1] == len("stat.mean.ret.w"), (
        diff_lists("decoded words", X.front_decode(p, s), words)
        + f"; prefix lengths {p} (the second must be {len('stat.mean.ret.w')})"
    )


# --------------------------------------------------------------------------- on Factorlab's data


def _skip_engine() -> bool:
    if not has_engine():
        print(
            "        (skipped: engine/data/library_on_engine has no outputs; run libfactors.py --compute and --screen)"
        )
        assert payload()["engine"]["available"] is False, (
            f"the engine block is available ({payload()['engine']['available']!r}) without the engine outputs"
        )
        return True
    return False


def test_engine_groups_funnel_and_coverage_recomputed():
    """Groups re-derived from the catalog's raw inputs; every funnel and coverage count from status.parquet."""
    if _skip_engine():
        return
    s, e = src(), payload()["engine"]
    st, man = esrc()["st"], esrc()["man"]
    fac = s["fac"]
    # the names of the definitions in the failure messages: the id and the formula
    who = (fac["factor_id"] + " " + fac["expression"]).tolist()
    assert e["available"] is True and st["factor_id"].tolist() == fac["factor_id"].tolist(), (
        f"engine block available {e['available']!r}; "
        + diff_lists("status.parquet ids against the catalog", st["factor_id"], fac["factor_id"])
    )
    # classification, written out from the catalog CSV: a definition is computable when every raw
    # input is a field the engine supplies; premium-only definitions are their own group
    inputs = fac["raw_inputs"].map(lambda x: set(x.split(",")))
    missing = inputs.map(lambda f: f - SUPPLIED)
    group = missing.map(lambda m: "computable" if not m else "needs_premium" if m == {"premium"} else "missing_fields")
    assert group.tolist() == st["group"].tolist(), diff_lists(
        "group (catalog raw inputs against status.parquet)", group, st["group"], who
    )
    comp = group == "computable"
    oi = comp & inputs.map(lambda f: bool(f & {"open_interest", "open_interest_value"}))
    vwap = comp & fac["expression"].str.contains(r"\bvwap(?:_dev)?\b")
    assert oi.tolist() == st["oi_short_history"].tolist() and vwap.tolist() == st["vwap_proxy"].tolist(), (
        diff_lists("oi_short_history (catalog against status.parquet)", oi, st["oi_short_history"], who)
        + "; "
        + diff_lists("vwap_proxy (catalog against status.parquet)", vwap, st["vwap_proxy"], who)
    )
    base = comp & inputs.map(lambda f: "open_interest" in f)
    assert base.tolist() == st["oi_base_proxy"].tolist(), diff_lists(
        "oi_base_proxy (catalog against status.parquet)", base, st["oi_base_proxy"], who
    )
    ok, empty = st["compute"] == "ok", (st["compute"] == "ok") & (st["finite_cells"] == 0)
    scr = st["screen"]
    pit = st["pit_check"] == "pass"
    n = e["counts"]
    want = {
        "definitions": len(fac),
        "computable": int(comp.sum()),
        "needs_premium": int((group == "needs_premium").sum()),
        "missing_fields": int((group == "missing_fields").sum()),
        "computed": int(ok.sum()),
        "compute_errors": int((st["compute"] == "error").sum()),
        "empty": int(empty.sum()),
        "empty_oi": int((empty & oi).sum()),
        "empty_vwap": int((empty & ~oi & vwap).sum()),
        "empty_other": int((empty & ~oi & ~vwap).sum()),
        "pit_pass": int(pit.sum()),
        "pit_fail": int((st["pit_check"] == "fail").sum()),
        "pit_no_finite": int((st["pit_check"] == "no_finite_cells").sum()),
        "oi": int(oi.sum()),
        "oi_pit_pass": int((oi & pit).sum()),
        "oi_no_g": int((oi & (scr == "no_g_values")).sum()),
        "oi_vwap": int((oi & vwap).sum()),
        "oi_empty": int((oi & empty).sum()),
        "oi_base_proxy": int(st["oi_base_proxy"].sum()),
        "vwap_proxy": int(vwap.sum()),
        "vwap_empty": int((vwap & empty).sum()),
        "vwap_verified": int((scr == "vwap_proxy").sum()),
        "no_g_values": int((scr == "no_g_values").sum()),
        "no_g_lookback": int((~oi & (scr == "no_g_values")).sum()),
        "too_few_weeks": int((scr == "too_few_weeks").sum()),
        "screened": int((scr == "screened").sum()),
    }
    assert n == want, diff_dicts("engine counts (payload against status.parquet)", n, want)
    # the empty factors without open interest are the skew and kurtosis of vwap_dev
    # (vwap equals close on OKX's data, so vwap_dev is constant)
    vwap_dev_moments = {f"ts_{k}(vwap_dev, {w})" for k in ("skew", "kurt") for w in (180, 240, 360)}
    assert set(fac.loc[(empty & ~oi).to_numpy(), "expression"]) <= vwap_dev_moments, (
        "empty definitions without open interest that are not a skew or kurtosis of vwap_dev "
        + shown(
            f"{i} {x}"
            for i, x in zip(
                fac.loc[(empty & ~oi).to_numpy(), "factor_id"], fac.loc[(empty & ~oi).to_numpy(), "expression"]
            )
            if x not in vwap_dev_moments
        )
        + f" (empty without open interest: {int((empty & ~oi).sum()):,}; compute: "
        + f"{dict(st.loc[(empty & ~oi).to_numpy(), 'compute'].value_counts())})"
    )
    # the values are weekly, so the OI history ends at the last Sunday, and an autocorrelation
    # adds its lag to the window
    through = pd.Timestamp(man["panels"]["data_through"])
    if e["meta"]["oi_from"] is None:  # no open-interest history: every OI factor is empty
        assert (empty | ~oi).all(), failing("OI factors with a value without an OI history", empty | ~oi, who)
    else:
        last_sunday = through - pd.Timedelta(days=(through.weekday() + 1) % 7)
        sunday_days = len(pd.date_range(e["meta"]["oi_from"], last_sunday, freq="D"))
        need = st["lookback"] + st["name"].str.extract(r"\.lag(\d+)\.")[0].fillna(0).astype(int)
        eo = (empty & oi).to_numpy()
        assert (need[eo] > sunday_days).all(), failing(
            f"empty OI factors whose window (lookback + lag) is inside the OI history of {sunday_days} days",
            need[eo] > sunday_days,
            [f"{i} window {w}" for i, w in zip(st.loc[eo, "factor_id"], need[eo])],
        )
    # a PIT pass needs a finite compared cell: every empty factor is 'no_finite_cells', never 'pass'
    assert (st.loc[empty.to_numpy(), "pit_check"] == "no_finite_cells").all(), failing(
        "empty factors whose pit_check is not no_finite_cells",
        st.loc[empty.to_numpy(), "pit_check"] == "no_finite_cells",
        [f"{i} {c}" for i, c in zip(st.loc[empty.to_numpy(), "factor_id"], st.loc[empty.to_numpy(), "pit_check"])],
    )
    # on newer data an open-interest factor can get its first value after the last cut: not empty,
    # but no cut compares a finite value, so it is unproven too
    late = ((st["pit_check"] == "no_finite_cells") & ~empty).to_numpy()
    assert n["pit_no_finite"] == n["empty"] + int(late.sum()), (
        f"pit_no_finite {num(n['pit_no_finite'])} != empty {num(n['empty'])} + not empty without a finite compared "
        f"cell {int(late.sum()):,}: {shown(st.loc[late, 'factor_id'])}"
    )
    last_cut = man["compute"]["pit"]["cuts"][-1]
    assert (st.loc[late, "first_value_sunday"] > last_cut).all() and st.loc[late, "oi_short_history"].all(), failing(
        f"not empty without a finite compared cell, but with a first value on or before the last cut {last_cut} "
        "or without open interest",
        (st.loc[late, "first_value_sunday"] > last_cut) & st.loc[late, "oi_short_history"],
        [
            f"{i} first {v} oi {o}"
            for i, v, o in zip(
                st.loc[late, "factor_id"], st.loc[late, "first_value_sunday"], st.loc[late, "oi_short_history"]
            )
        ],
    )
    # every computed definition is verified, unproven for lack of values, or failed; every verified one
    # is screened or excluded for one stated reason
    assert n["computed"] == n["computable"] == n["pit_pass"] + n["pit_no_finite"] + n["pit_fail"], (
        f"computed {num(n['computed'])}, computable {num(n['computable'])}, pit_pass {num(n['pit_pass'])} + pit_no_finite "
        f"{num(n['pit_no_finite'])} + pit_fail {num(n['pit_fail'])} = {num(total(n['pit_pass'], n['pit_no_finite'], n['pit_fail']))}"
    )
    assert n["screened"] + n["vwap_verified"] + n["no_g_values"] + n["too_few_weeks"] == n["pit_pass"], (
        f"screened {num(n['screened'])} + vwap_verified {num(n['vwap_verified'])} + no_g_values {num(n['no_g_values'])} + "
        f"too_few_weeks {num(n['too_few_weeks'])} = "
        f"{num(total(n['screened'], n['vwap_verified'], n['no_g_values'], n['too_few_weeks']))}, pit_pass {num(n['pit_pass'])}"
    )
    mc = man["compute"]["counts"]
    want_groups = {
        "computable": n["computable"],
        "missing_fields": n["missing_fields"],
        "needs_premium": n["needs_premium"],
    }
    assert mc["groups"] == want_groups, diff_dicts(
        "manifest groups against the payload counts", mc["groups"], want_groups
    )
    got_m = (mc["computed_ok"], mc["all_nan_on_sundays"], mc["computable_needing_oi"], mc["computable_vwap_proxy"])
    want_m = (n["computed"], n["empty"], n["oi"], n["vwap_proxy"])
    assert got_m == want_m, neq(
        "manifest (computed_ok, all_nan_on_sundays, computable_needing_oi, computable_vwap_proxy) against the "
        "payload (computed, empty, oi, vwap_proxy)",
        got_m,
        want_m,
    )
    want_split = {"open_interest": n["empty_oi"], "vwap_proxy": n["empty_vwap"], "other": n["empty_other"]}
    assert mc["all_nan_split"] == want_split, diff_dicts(
        "manifest all_nan_split against the payload counts", mc["all_nan_split"], want_split
    )
    # funnel: keys, counts, narrowing
    f = {x["key"]: x for x in e["funnel"]}
    keys = ["library", "computable", "computed", "pit", "screened", "ric_q10", "ric_q05"]
    assert [x["key"] for x in e["funnel"]] == keys, diff_lists("funnel keys", [x["key"] for x in e["funnel"]], keys)
    steps5 = ("library", "computable", "computed", "pit", "screened")
    want_n = [n["definitions"], n["computable"], n["computed"], n["pit_pass"], n["screened"]]
    assert [f[k]["n"] for k in steps5] == want_n, diff_lists(
        "funnel counts against the payload counts", [f[k]["n"] for k in steps5], want_n, steps5
    )
    g = esrc()["g"]
    for k, q in (("ric_q10", 0.10), ("ric_q05", 0.05)):
        assert f[k]["n"] == int((g["ric_q"] <= q).sum()) and f[k]["signals"] == int(
            g[f"ric_rep_{k[4:]}"][g["ric_q"] <= q].sum()
        ), (
            f"funnel {k}: factors {num(f[k]['n'])}, signals {num(f[k]['signals'])}; screen_g.parquet: factors "
            f"{int((g['ric_q'] <= q).sum()):,}, signals {num(int(g[f'ric_rep_{k[4:]}'][g['ric_q'] <= q].sum()))}"
        )
    steps = [x["n"] for x in e["funnel"]]
    assert steps == sorted(steps, reverse=True), diff_lists(
        "funnel counts in page order against the narrowing order", steps, sorted(steps, reverse=True), keys
    )
    sn = f["screened"]["note"]
    assert (w := "{} vwap proxies".format(f"{num(n['vwap_verified'])}")) in sn, missing_text(w, sn)
    assert (w := "{} need open interest".format(f"{num(n['oi_no_g'])}")) in sn, missing_text(w, sn)
    # the reason for the empty factors is stated per cause, not as "longer than the data"
    cn = f["computed"]["note"]
    assert (
        w := "{} open-interest factors whose window is longer than the OI history".format(f"{num(n['empty_oi'])}")
    ) in cn, missing_text(w, cn)
    assert (w := f"{n['empty_vwap']:d} skew or kurtosis of vwap_dev") in cn and "longer than the data" not in cn, (
        missing_text(w, cn) + f"; 'longer than the data' in the text: {'longer than the data' in cn}"
    )
    # coverage rows
    cov = {x["key"]: x for x in e["coverage"]}
    cov_keys = ["premium", "fields", "oi", "vwap"]
    assert [x["key"] for x in e["coverage"]] == cov_keys, diff_lists(
        "coverage keys", [x["key"] for x in e["coverage"]], cov_keys
    )
    got_c = (cov["premium"]["n"], cov["fields"]["n"], cov["oi"]["n"], cov["vwap"]["n"])
    want_c = (n["needs_premium"], n["missing_fields"], n["oi"], n["vwap_proxy"])
    assert got_c == want_c, neq(
        "coverage counts (premium, fields, oi, vwap) against the payload counts "
        "(needs_premium, missing_fields, oi, vwap_proxy)",
        got_c,
        want_c,
    )
    computed = {k: cov[k]["computed"] for k in cov_keys}
    assert cov["premium"]["computed"] is False and cov["fields"]["computed"] is False, neq(
        "coverage 'computed' flags (premium and fields must be False)", computed, {"premium": False, "fields": False}
    )
    assert cov["oi"]["computed"] is True and cov["vwap"]["computed"] is True, neq(
        "coverage 'computed' flags (oi and vwap must be True)", computed, {"oi": True, "vwap": True}
    )
    # fields the engine's OKX data does not have: definitions needing each, from the catalog's raw inputs
    fields = {x["field"]: x for x in e["fields"]}
    assert set(fields) == set().union(*missing) == set(X.FIELD_LABELS), (
        diff_sets("fields (payload against the catalog raw inputs)", fields, set().union(*missing))
        + "; "
        + diff_sets("fields (payload against FIELD_LABELS)", fields, X.FIELD_LABELS)
    )
    for fld, row in fields.items():
        n_fld = int(missing.map(lambda m, fld=fld: fld in m).sum())
        only_fld = int(missing.map(lambda m, fld=fld: m == {fld}).sum())
        assert row["n"] == n_fld, f"{fld}: definitions that need it {num(row['n'])}, catalog raw inputs {n_fld:,}"
        assert row["only"] == only_fld, f"{fld}: definitions that need only it {num(row['only'])}, catalog {only_fld:,}"
        assert row["label"], f"{fld}: no label ({row.get('label')!r})"
    field_n = [x["n"] for x in e["fields"]]
    assert field_n == sorted(field_n, reverse=True), diff_lists(
        "field counts in page order against the descending order",
        field_n,
        sorted(field_n, reverse=True),
        [x["field"] for x in e["fields"]],
    )
    # open interest really starts where the page says: the first stored OKX open-interest value
    firsts = [
        pd.read_parquet(p, columns=["ts", "oi_value"]).dropna(subset=["oi_value"])["ts"].min()
        for p in sorted((C.DATA / "raw" / "open_interest").glob("*.parquet"))
    ]
    firsts = [t for t in firsts if pd.notna(t)]
    want_from = min(firsts).strftime("%Y-%m-%d") if firsts else None
    assert e["meta"]["oi_from"] == want_from, neq(
        "oi_from against the first stored OKX open-interest value", e["meta"]["oi_from"], want_from
    )
    on = cov["oi"]["note"]
    assert (w := f"stored only from {e['meta']['oi_from'] or 'n/a'}") in on, missing_text(w, on)
    assert (
        w := "{} are verified point in time and {} are empty".format(
            f"{num(n['oi_pit_pass'])}", f"{num(n['oi_empty'])}"
        )
    ) in on, missing_text(w, on)
    assert "stamped 16:00 UTC" in on, missing_text("stamped 16:00 UTC", on)
    vn = cov["vwap"]["note"]
    assert (w := f"{num(n['vwap_empty'])} of them are empty on every Sunday") in vn, missing_text(w, vn)


def test_engine_pit_meta_and_zones():
    if _skip_engine():
        return
    e = payload()["engine"]
    st, man = esrc()["st"], esrc()["man"]
    p = e["pit"]
    cuts = range(1, 4)
    assert p["cells"] == [int(st[f"pit_cells_t{j}"].sum()) for j in cuts] == man["compute"]["pit"]["compared_cells"], (
        f"compared cells by cut: payload {p['cells']}, status.parquet {[int(st[f'pit_cells_t{j}'].sum()) for j in cuts]}, "
        f"manifest {man['compute']['pit']['compared_cells']}"
    )
    assert p["finite"] == [int(st[f"pit_finite_t{j}"].sum()) for j in cuts] == man["compute"]["pit"]["finite_cells"], (
        f"finite cells by cut: payload {p['finite']}, status.parquet {[int(st[f'pit_finite_t{j}'].sum()) for j in cuts]}, "
        f"manifest {man['compute']['pit']['finite_cells']}"
    )
    assert all(int(st[f"pit_diff_cells_t{j}"].sum()) == 0 for j in cuts), (
        "definitions with differing cells by cut: "
        + "; ".join(f"t{j} {shown(st.loc[st[f'pit_diff_cells_t{j}'] > 0, 'factor_id'])}" for j in cuts)
    )
    assert p["cuts"] == man["zones"]["pit_cuts"] and len(p["cuts"]) == 3, neq(
        "PIT cuts (payload against the manifest zones, 3 cuts)", p["cuts"], man["zones"]["pit_cuts"]
    )
    assert (p["rtol"], p["atol"]) == (1e-9, 1e-12), neq(
        "PIT tolerances (rtol, atol)", (p["rtol"], p["atol"]), (1e-9, 1e-12)
    )
    # the third cut is what reaches the open-interest factors whose values start after the second cut
    oi = st["oi_short_history"] & (st["pit_check"] == "pass")
    late = oi & (st["pit_finite_t1"] == 0) & (st["pit_finite_t2"] == 0) & (st["pit_finite_t3"] > 0)
    assert p["oi_from_last_cut_only"] == int(late.sum()), neq(
        "OI factors with finite cells from the last cut only (payload against status.parquet)",
        p["oi_from_last_cut_only"],
        int(late.sum()),
    )
    want_oi = [int((st["oi_short_history"] & (st[f"pit_finite_t{j}"] > 0)).sum()) for j in cuts]
    assert p["oi_finite"] == want_oi, neq(
        "OI factors with finite cells by cut (payload against status.parquet)", p["oi_finite"], want_oi
    )
    pit_note = next(x for x in e["funnel"] if x["key"] == "pit")["note"]
    assert (w := "%.1f million values compared" % (sum(p["cells"]) / 1e6)) in pit_note, missing_text(w, pit_note)
    assert (w := "%.1f million of them finite" % (sum(p["finite"]) / 1e6)) in pit_note, missing_text(w, pit_note)
    # "equal" is stated with its tolerance, never as identical
    assert (
        w := "equal up to floating-point rounding (rtol 1e-9, atol 1e-12)"
    ) in pit_note and "identical" not in pit_note, (
        missing_text(w, pit_note) + f"; 'identical' in it: {'identical' in pit_note}"
    )
    assert (w := "{} and {}".format(", ".join(p["cuts"][:-1]), p["cuts"][-1])) in pit_note, missing_text(w, pit_note)
    assert (w := "{} had no finite value to compare".format(f"{num(e['counts']['pit_no_finite'])}")) in pit_note, (
        missing_text(w, pit_note)
    )
    m = e["meta"]
    got_meta = (m["data_through"], m["first_bar"], m["symbols"], m["benchmark"])
    want_meta = (
        man["panels"]["data_through"],
        man["panels"]["first_bar"],
        len(man["panels"]["symbols"]),
        "BTC-USDT",
    )
    assert got_meta == want_meta, neq("meta (data_through, first_bar, symbols, benchmark)", got_meta, want_meta)
    assert m["symbols"] == 20 and len(m["fields"]) == len(SUPPLIED) and set(m["fields"]) == SUPPLIED, (
        f"symbols {m['symbols']} (expected 20); "
        + diff_sets("fields against SUPPLIED", m["fields"], SUPPLIED)
        + f"; {len(m['fields'])} fields listed"
    )
    db_sha = hashlib.sha256((CAT / "factors.db").read_bytes()).hexdigest()
    assert m["catalog_current"] == (db_sha == man["inputs"]["library/catalog/factors.db"]["sha256"]) is True, (
        f"catalog_current {m['catalog_current']!r}; catalog sha256 {db_sha[:12]}, "
        f"manifest {man['inputs']['library/catalog/factors.db']['sha256'][:12]} (both must be the same)"
    )
    # which code produced the outputs: the files whose hash differs from the manifest's record
    changed = [f for f, h in man["code"]["files"].items() if hashlib.sha256((REPO / f).read_bytes()).hexdigest() != h]
    assert m["code_changed"] == changed and m["code_current"] is (not changed), (
        neq("code_changed (payload against the files on disk)", m["code_changed"], changed)
        + f"; code_current {m['code_current']!r}"
    )
    # zones: the generator config's boundaries and counts, Sunday arithmetic done here
    with (ROOT / "factorfinder" / "eval_config.toml").open("rb") as fh:
        z = tomllib.load(fh)["zones"]
    rows = {r["key"]: r for r in e["zones"]["rows"]}
    zone_keys = ["g_fit", "g_gap", "g_eval", "embargo", "i"]
    assert [r["key"] for r in e["zones"]["rows"]] == zone_keys, neq(
        "zone keys", [r["key"] for r in e["zones"]["rows"]], zone_keys
    )
    for key, (cfg, cnt) in {
        "g_fit": ("g_fit_sundays", "g_fit"),
        "g_gap": ("g_internal_embargo_sundays", "g_internal_embargo"),
        "g_eval": ("g_eval_sundays", "g_eval"),
        "embargo": ("embargo_sundays", "embargo"),
        "i": ("i_sundays", "i"),
    }.items():
        r = rows[key]
        assert [r["start"], r["end"]] == z[cfg], neq(
            f"zone {key}: [start, end] against {cfg}", [r["start"], r["end"]], z[cfg]
        )
        days = pd.date_range(r["start"], r["end"], freq="7D")
        assert (days.dayofweek == 6).all(), f"zone {key}: a date that is not a Sunday from {r['start']} to {r['end']}"
        got_n, want_n = (r["sundays"], len(days)), (z["expected_counts"][cnt], z["expected_counts"][cnt])
        assert got_n == want_n, neq(
            f"zone {key}: (sundays, the Sundays from start to end) against the expected count", got_n, want_n
        )
    sealed = [r["key"] for r in e["zones"]["rows"] if r.get("sealed")]
    assert rows["i"]["sealed"] is True and sum(1 for r in e["zones"]["rows"] if r.get("sealed")) == 1, neq(
        "sealed zones", sealed, ["i"]
    )
    assert e["zones"]["label_cutoff_row"] == z["g_eval_sundays"][1] == "2024-06-09", (
        f"label_cutoff_row {e['zones']['label_cutoff_row']!r}, config G_eval end {z['g_eval_sundays'][1]!r}, "
        "expected '2024-06-09'"
    )
    assert e["zones"]["g_eval_label_close"] == z["g_eval_label_close"], neq(
        "g_eval_label_close (payload against the config)", e["zones"]["g_eval_label_close"], z["g_eval_label_close"]
    )
    # the notes name the zone G by its dates, and say how to test a definition in a book
    want_notes = [
        "These numbers come from the Sundays of the zone G (2022-04-03 to 2024-06-09). "
        "No screen reads a label after 2024-06-09.",
        "q is the Benjamini-Hochberg q-value over the screened definitions.",
        "A low q or a large t is a number. It is not a test of a book.",
        "To test a definition in a book, use its register command.",
    ]
    assert e["notes"] == want_notes, diff_lists("engine notes", e["notes"], want_notes)
    formatted = [x.format(g0="2022-04-03", g1="2024-06-09") for x in X.ENGINE_NOTES]
    assert e["notes"] == formatted, diff_lists("engine notes against ENGINE_NOTES", e["notes"], formatted)


def test_engine_screen_recomputed():
    """BH q-values recomputed here from the stored p-values; survivors, signals, the G_fit selection and
    its G_eval check, the sensitivity rows and families."""
    if _skip_engine():
        return
    e = payload()["engine"]
    st, g, man = esrc()["st"], esrc()["g"], esrc()["man"]
    sc = e["screen"]
    assert set(g["factor_id"]) == set(st.loc[st["screen"] == "screened", "factor_id"]), diff_sets(
        "screen_g.parquet ids against the screened ids of status.parquet",
        g["factor_id"],
        st.loc[st["screen"] == "screened", "factor_id"],
    )
    assert sc["n_tests_ic"] == sc["n_tests_ric"] == len(g), (
        f"n_tests_ic {num(sc['n_tests_ic'])}, n_tests_ric {num(sc['n_tests_ric'])}, screen_g.parquet rows {len(g):,}"
    )
    # t = mean / sd * sqrt(n), on the stored float32 statistics
    for stat in ("ic", "ric"):
        t = g[f"{stat}_mean"].astype(float) / g[f"{stat}_sd"].astype(float) * np.sqrt(g[f"{stat}_n"].astype(float))
        ok = np.isclose(t, g[f"{stat}_t"].astype(float), rtol=1e-4, atol=1e-4)
        assert ok.all(), failing(
            f"{stat}_t against {stat}_mean / {stat}_sd * sqrt({stat}_n)",
            ok,
            [f"{i} {a} != {b}" for i, a, b in zip(g["factor_id"], g[f"{stat}_t"], t)],
        )
    assert (g["ic_n"] >= sc["min_weeks"]).all() and sc["min_weeks"] == 20, (
        f"min_weeks {sc['min_weeks']} (expected 20); "
        + failing("screened factors with fewer weekly ICs than min_weeks", g["ic_n"] >= sc["min_weeks"], g["factor_id"])
    )
    few = st.loc[st["screen"] == "too_few_weeks"]
    assert (st.loc[st["screen"] == "too_few_weeks", "g_weeks"] < sc["min_weeks"]).all(), failing(
        f"too_few_weeks factors with at least {sc['min_weeks']} weekly ICs",
        few["g_weeks"] < sc["min_weeks"],
        [f"{i} {w}" for i, w in zip(few["factor_id"], few["g_weeks"])],
    )
    # g_weeks is the weekly-IC count of every factor with G values, the screened ones' ic_n included
    gw = st.set_index("factor_id").loc[g["factor_id"], "g_weeks"].to_numpy()
    assert (st.set_index("factor_id").loc[g["factor_id"], "g_weeks"].to_numpy() == g["ic_n"].to_numpy()).all(), failing(
        "screened factors whose g_weeks differs from ic_n", gw == g["ic_n"].to_numpy(), g["factor_id"]
    )
    rows = {(b["stat"], b["q"]): b for b in sc["bh"]}
    for stat in ("ic", "ric"):
        q = bh_q(g[f"{stat}_p"].to_numpy(dtype=float))
        assert np.allclose(q, g[f"{stat}_q"].to_numpy(dtype=float), rtol=1e-9, atol=1e-12), f"{stat} BH q differs"
        for lvl in (0.05, 0.10):
            tag = f"q{round(lvl * 100):02d}"
            surv = q <= lvl
            sig = g.loc[surv, f"{stat}_signal_{tag}"]
            rep = g.loc[surv, f"{stat}_rep_{tag}"]
            assert (sig >= 0).all() and (g.loc[~surv, f"{stat}_signal_{tag}"] == -1).all(), (
                failing(f"{stat} {tag} survivors without a signal", sig >= 0, g.loc[surv, "factor_id"])
                + "; "
                + failing(
                    f"{stat} {tag} non-survivors with a signal",
                    g.loc[~surv, f"{stat}_signal_{tag}"] == -1,
                    g.loc[~surv, "factor_id"],
                )
            )
            # every distinct signal has exactly one representative: its member with the largest |t|
            for sid, grp in g[surv].groupby(f"{stat}_signal_{tag}"):
                assert int(grp[f"{stat}_rep_{tag}"].sum()) == 1, (
                    f"{stat} {tag} signal {sid}: {int(grp[f'{stat}_rep_{tag}'].sum())} representatives "
                    f"{shown(grp.loc[grp[f'{stat}_rep_{tag}'].astype(bool), 'factor_id'])} of {shown(grp['factor_id'])}"
                )
                best = grp[f"{stat}_t"].abs().max()
                rep_t = float(grp.loc[grp[f"{stat}_rep_{tag}"], f"{stat}_t"].iloc[0])
                assert abs(abs(rep_t) - best) < 1e-6, (
                    f"{stat} {tag} signal {sid}: the representative "
                    f"{grp.loc[grp[f'{stat}_rep_{tag}'], 'factor_id'].iloc[0]} has |t| {abs(rep_t)}, "
                    f"the largest |t| is {best} ({grp.loc[grp[f'{stat}_t'].abs().idxmax(), 'factor_id']})"
                )
            b = rows[(stat, lvl)]
            got_b = (b["factors"], b["signals"], b["positive"])
            want_b = (int(surv.sum()), int(rep.sum()), int((surv & (g[f"{stat}_t"] > 0).to_numpy()).sum()))
            assert got_b == want_b, neq(
                f"{stat} q <= {lvl}: (factors, signals, positive), payload against screen_g.parquet", got_b, want_b
            )
            want_m = (
                man["screen"]["counts"][f"{stat}_count_{tag}"],
                man["screen"]["counts"][f"{stat}_distinct_signals_{tag}"],
            )
            assert (b["factors"], b["signals"]) == want_m, neq(
                f"{stat} {tag} (factors, signals), payload against the manifest", (b["factors"], b["signals"]), want_m
            )
    # replication as the zone design intends: BH over the G_fit p-values alone (recomputed here),
    # then the sign and t on G_eval, which that selection never saw
    # the keys of the screen block: the pooled-G G_eval agreement is in sample, so no key holds it
    keys = {"bh", "by_family", "by_family_note", "dedup", "evidence", "family_rule", "g_sundays", "generated_at"}
    keys |= {"label", "mask", "min_weeks", "n_tests_ic", "n_tests_ric", "names_max", "names_min", "pooled_note"}
    keys |= {"precision", "regressors", "replication", "sensitivity"}
    assert set(sc) == keys, diff_sets("keys of the screen block", sc, keys)
    rp = {(r["stat"], r["q"]): r for r in sc["replication"]["rows"]}
    for stat in ("ic", "ric"):
        fp = g[f"{stat}_fit_p"].to_numpy(dtype=float)
        fq = np.full(len(fp), np.nan)
        fq[np.isfinite(fp)] = bh_q(fp[np.isfinite(fp)])  # a factor without a G_fit t is not in the family
        stored = g[f"{stat}_fit_q"].to_numpy(dtype=float)
        assert np.allclose(fq, g[f"{stat}_fit_q"].to_numpy(dtype=float), rtol=1e-9, atol=1e-12, equal_nan=True), (
            failing(
                f"{stat} G_fit q (recomputed against stored)",
                np.isclose(fq, stored, rtol=1e-9, atol=1e-12, equal_nan=True),
                [f"{i} {a} != {b}" for i, a, b in zip(g["factor_id"], fq, stored)],
            )
        )
        fm, em = g[f"{stat}_fit_mean"].astype(float), g[f"{stat}_eval_mean"].astype(float)
        same = fm.notna() & em.notna() & (fm != 0) & (np.sign(fm) == np.sign(em))
        for lvl in (0.05, 0.10):
            sel = fq <= lvl
            r = rp[(stat, lvl)]
            got_p = (r["selected"], r["same_sign"], r["same_sign_t2"], r["base_pairs"], r["base_same_sign"])
            want_p = (
                int(sel.sum()),
                int((sel & same).sum()),
                int((sel & same & (g[f"{stat}_eval_t"].abs() >= 2)).sum()),
                int((fm.notna() & em.notna() & (fm != 0)).sum()),
                int(same.sum()),
            )
            assert got_p == want_p, neq(
                f"{stat} q <= {lvl} G_fit selection (selected, same_sign, same_sign_t2, base_pairs, base_same_sign), "
                "payload against screen_g.parquet",
                got_p,
                want_p,
            )
            m = man["screen"]["counts"]["fit_selection"][stat][f"q{round(lvl * 100):02d}"]
            want_r = (m["selected"], m["eval_same_sign"], m["eval_same_sign_t2"])
            got_r = (r["selected"], r["same_sign"], r["same_sign_t2"])
            assert got_r == want_r, neq(
                f"{stat} q{round(lvl * 100):02d} G_fit selection (selected, same_sign, same_sign_t2), payload against "
                "the manifest",
                got_r,
                want_r,
            )
    # the sensitivity rows are the manifest's, in its order; the registered row is the screen itself
    sens = {r["key"]: r for r in sc["sensitivity"]}
    ms = man["screen"]["sensitivity"]
    assert list(sens) == list(ms) and list(sens)[0] == "registered", neq(
        "sensitivity rows (payload against the manifest; the first is registered)", list(sens), list(ms)
    )
    for k, r in sens.items():
        got_s = (r["screened"], r["q05"], r["q10"], r["fit_selected"], r["same_sign"], r["same_sign_t2"])
        want_s = (
            ms[k]["screened"],
            ms[k]["ric_count_q05"],
            ms[k]["ric_count_q10"],
            ms[k]["ric_fit_selected_q05"],
            ms[k]["eval_same_sign"],
            ms[k]["eval_same_sign_t2"],
        )
        assert got_s == want_s, neq(
            f"sensitivity row {k} (screened, q05, q10, fit_selected, same_sign, same_sign_t2), payload against the "
            "manifest",
            got_s,
            want_s,
        )
    got_reg = (sens["registered"]["screened"], sens["registered"]["q05"], sens["registered"]["q10"])
    want_reg = (len(g), rows[("ric", 0.05)]["factors"], rows[("ric", 0.10)]["factors"])
    assert got_reg == want_reg, neq(
        "the registered sensitivity row (screened, q05, q10) against the screen", got_reg, want_reg
    )
    g_sundays = sum(r["sundays"] for r in e["zones"]["rows"] if r["key"] in ("g_fit", "g_eval"))
    assert sc["g_sundays"] == 114 == g_sundays, (
        f"g_sundays {sc['g_sundays']}, G_fit + G_eval Sundays of the zones {g_sundays}, expected 114"
    )
    got_names = (sc["names_min"], sc["names_max"])
    want_names = (
        man["screen"]["counts"]["names_per_sunday"]["min"],
        man["screen"]["counts"]["names_per_sunday"]["max"],
    )
    assert got_names == want_names, neq(
        "names per Sunday (min, max), payload against the manifest", got_names, want_names
    )
    fams = {r["family"]: r for r in sc["by_family"]}
    assert set(fams) == set(st["family"]), diff_sets(
        "families of the screen against status.parquet", fams, st["family"]
    )
    for fam, r in fams.items():
        fs, fg = st[st["family"] == fam], g[g["family"] == fam]
        rr = fg["ric_q"] <= 0.05
        got_f = (
            r["definitions"],
            r["computable"],
            r["screened"],
            r["ic_q05"],
            r["ric_q05"],
            r["ric_q05_signals"],
            r["ric_q05_touched"],
        )
        want_f = (
            len(fs),
            int((fs["group"] == "computable").sum()),
            len(fg),
            int((fg["ic_q"] <= 0.05).sum()),
            int(rr.sum()),
            int((rr & fg["ric_rep_q05"]).sum()),
            int(fg.loc[rr, "ric_signal_q05"].nunique()),
        )
        assert got_f == want_f, neq(
            f"family {fam} (definitions, computable, screened, ic_q05, ric_q05, ric_q05_signals, ric_q05_touched), "
            "payload against status.parquet and screen_g.parquet",
            got_f,
            want_f,
        )
        assert r["ric_q05_signals"] <= r["ric_q05_touched"] <= r["ric_q05"], (
            f"family {fam}: ric_q05_signals {num(r['ric_q05_signals'])} <= ric_q05_touched "
            f"{num(r['ric_q05_touched'])} <= ric_q05 {num(r['ric_q05'])} does not hold"
        )
    # a signal is counted in its representative's family, which the page states
    assert "family of its representative" in sc["by_family_note"], missing_text(
        "family of its representative", sc["by_family_note"]
    )
    fam_signals = sum(r["ric_q05_signals"] for r in sc["by_family"])
    assert fam_signals == rows[("ric", 0.05)]["signals"], neq(
        "the sum of the rIC q05 signals of the families against the screen", fam_signals, rows[("ric", 0.05)]["signals"]
    )
    # the screen ranks full-precision values and ties only floating-point noise
    assert (
        sc["precision"] == man["screen"]["precision"] and "float64" in sc["precision"] and "1e-12" in sc["precision"]
    ), neq(
        "precision (payload against the manifest; it names float64 and 1e-12)",
        sc["precision"],
        man["screen"]["precision"],
    )
    fam_q05 = sum(r["ric_q05"] for r in sc["by_family"])
    assert fam_q05 == rows[("ric", 0.05)]["factors"], neq(
        "the sum of the rIC q05 factors of the families against the screen", fam_q05, rows[("ric", 0.05)]["factors"]
    )
    fam_defs = sum(r["definitions"] for r in sc["by_family"])
    assert fam_defs == len(st), neq(
        "the sum of the definitions of the families against status.parquet", fam_defs, len(st)
    )
    q05_by_fam = [r["ric_q05"] for r in sc["by_family"]]
    assert q05_by_fam == sorted(q05_by_fam, reverse=True), diff_lists(
        "rIC q05 counts of the families in page order against the descending order",
        q05_by_fam,
        sorted(q05_by_fam, reverse=True),
        [r["family"] for r in sc["by_family"]],
    )
    # the regressors and the label the page prints are the manifest's own words
    assert sc["regressors"] == man["screen"]["ric_regressors"] and sc["label"] == man["screen"]["label"], (
        diff_dicts("regressors (payload against the manifest)", sc["regressors"], man["screen"]["ric_regressors"])
        + "; "
        + neq("label", sc["label"], man["screen"]["label"])
    )
    # the settings the page prints are the generator config's, and the deviations name what still differs
    with (ROOT / "factorfinder" / "eval_config.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    rc = sc["regressors"]["config"]
    got_rc = (rc["ridge_lambda"], rc["tilt_shrink_days"], rc["btc_beta_days"], rc["min_names_per_sunday"])
    want_rc = (
        cfg["regressors"]["ridge_lambda"],
        cfg["regressors"]["tilt_shrink_days"],
        cfg["regressors"]["btc_beta_days"],
        cfg["g0"]["min_names_per_sunday"],
    )
    assert got_rc == want_rc, neq(
        "regressor settings (ridge_lambda, tilt_shrink_days, btc_beta_days, min_names_per_sunday), payload against "
        "eval_config.toml",
        got_rc,
        want_rc,
    )
    assert rc["set"] == cfg["regressors"]["set"] and "zoo_pc1" not in rc["used"], (
        neq("regressor set (payload against eval_config.toml)", rc["set"], cfg["regressors"]["set"])
        + f"; used {rc['used']} (zoo_pc1 must not be in it)"
    )
    assert any("zoo_pc1" in d for d in sc["regressors"]["deviations"]), (
        f"no deviation names zoo_pc1: {shown(sc['regressors']['deviations'])}"
    )
    assert "n/(n+180)" in sc["regressors"]["tilt"] and "ridge" in sc["regressors"]["method"], (
        f"tilt {sc['regressors']['tilt']!r} (must name n/(n+180)); method {sc['regressors']['method']!r} "
        "(must name ridge)"
    )


def _check_signal_info(g: pd.DataFrame, want: pd.DataFrame, t: dict) -> None:
    """Each shown row's signal size and representative, counted here over every survivor."""
    for lvl in ("05", "10"):
        sv = g[g[f"ric_signal_q{lvl}"] >= 0]
        size = sv.groupby(f"ric_signal_q{lvl}").size().to_dict()
        rep = sv[sv[f"ric_rep_q{lvl}"].astype(bool)].set_index(f"ric_signal_q{lvl}")["factor_id"].to_dict()
        sig = want[f"ric_signal_q{lvl}"].astype(int).tolist()
        want_size = [None if x < 0 else size[x] for x in sig]
        assert t[f"size{lvl}"] == want_size, diff_lists(
            f"size{lvl} of the shown rows against the signal sizes of screen_g.parquet", t[f"size{lvl}"], want_size
        )
        want_rep = [None if x < 0 else rep[x] for x in sig]
        assert t[f"repid{lvl}"] == want_rep, diff_lists(
            f"repid{lvl} of the shown rows against the representatives of screen_g.parquet", t[f"repid{lvl}"], want_rep
        )


def test_engine_top_table_recomputed():
    if _skip_engine():
        return
    s, e = src(), payload()["engine"]
    g = esrc()["g"]
    t = e["top"]
    order = g.assign(a=g["ric_t"].abs()).sort_values(["a", "factor_id"], ascending=[False, True], kind="stable")
    rank = {fid: i + 1 for i, fid in enumerate(order["factor_id"])}
    every = order[order["ric_q"] <= 0.10]
    want = every.head(X.TOP_MAX)  # the rows are capped; the counts are the full ones
    assert t["q"] == 0.10 and t["n"] == len(every) and t["shown"] == len(want) and t["ranked_among"] == len(g), neq(
        "top table (q, n, shown, ranked_among)",
        (t["q"], t["n"], t["shown"], t["ranked_among"]),
        (0.10, len(every), len(want), len(g)),
    )
    assert t["max_rows"] == X.TOP_MAX and t["n_q05"] == int((g["ric_q"] <= 0.05).sum()), neq(
        "top table (max_rows, n_q05)", (t["max_rows"], t["n_q05"]), (X.TOP_MAX, int((g["ric_q"] <= 0.05).sum()))
    )
    want_rank = [rank[x] for x in want["factor_id"]]
    assert t["id"] == want["factor_id"].tolist() and t["rank"] == want_rank, (
        diff_lists("top table ids", t["id"], want["factor_id"])
        + "; "
        + diff_lists("ranks", t["rank"], want_rank, t["id"])
    )
    assert t["rank"] == sorted(t["rank"]), diff_lists(
        "ranks in table order against the sorted order", t["rank"], sorted(t["rank"]), t["id"]
    )
    _check_signal_info(g, want, t)
    fac = s["fac"].set_index("factor_id")
    fams = e["families"]
    for i, (_, x) in enumerate(want.iterrows()):
        fid = x["factor_id"]
        assert t["name"][i] == x["name"] == fac.at[fid, "name"] and fams[t["fam"][i]] == x["family"], (
            f"{fid}: name {t['name'][i]!r} / screen {x['name']!r} / catalog {fac.at[fid, 'name']!r}; "
            f"family {fams[t['fam'][i]]!r} / screen {x['family']!r}"
        )
        assert t["expr"][i] == fac.at[fid, "expression"], neq(f"{fid} formula", t["expr"][i], fac.at[fid, "expression"])
        for k, col, tol in (
            ("ic_mean", "ic_mean", 6e-5),
            ("ic_t", "ic_t", 0.006),
            ("ic_q", "ic_q", 6e-5),
            ("ric_mean", "ric_mean", 6e-5),
            ("ric_t", "ric_t", 0.006),
            ("ric_q", "ric_q", 6e-5),
            ("persistence", "persistence", 0.006),
            ("fit_t", "ric_fit_t", 0.006),
            ("fit_q", "ric_fit_q", 6e-5),
            ("eval_t", "ric_eval_t", 0.006),
        ):
            if pd.isna(x[col]):  # no value (for example no G_fit week): the export writes null
                assert t[k][i] is None, f"{fid} {k}: {t[k][i]!r} (the source has no value)"
            else:
                close(t[k][i], x[col], tol, f"{fid} {k}")
        for lvl in ("05", "10"):
            sig = int(x[f"ric_signal_q{lvl}"])
            assert t[f"sig{lvl}"][i] == (None if sig < 0 else sig) and t[f"rep{lvl}"][i] == bool(
                x[f"ric_rep_q{lvl}"]
            ), (
                f"{fid} q{lvl}: signal {t[f'sig{lvl}'][i]!r} (screen {sig}), representative {t[f'rep{lvl}'][i]!r} "
                f"(screen {bool(x[f'ric_rep_q{lvl}'])})"
            )
    # the q <= 0.05 subset and its distinct signals, as the page's filters show them
    q05 = [i for i in range(t["shown"]) if t["ric_q"][i] <= 0.05]
    counts = esrc()["man"]["screen"]["counts"]
    assert (
        len(q05) == counts["ric_count_q05"] and sum(t["rep05"][i] for i in q05) == counts["ric_distinct_signals_q05"]
    ), (
        f"rows with q <= 0.05: {len(q05):,} (manifest {num(counts['ric_count_q05'])}); their representatives "
        f"{num(sum(t['rep05'][i] for i in q05))} (manifest {num(counts['ric_distinct_signals_q05'])})"
    )
    assert sum(t["rep10"]) == counts["ric_distinct_signals_q10"], neq(
        "representatives at q <= 0.10 (table against the manifest)", sum(t["rep10"]), counts["ric_distinct_signals_q10"]
    )
    assert all(t["sig05"][i] is not None for i in q05), (
        f"rows with q <= 0.05 and no q05 signal: {shown(t['id'][i] for i in q05 if t['sig05'][i] is None)}"
    )
    assert "verdict" not in t, "the top table has the key verdict"


def test_engine_block_reads_no_values_shards():
    """The builder reads status.parquet and screen_g.parquet only, never the large values/
    shards, so the export works without them."""
    if _skip_engine():
        return
    seen = []
    real = pd.read_parquet  # the one pandas module: the builder modules call pd.read_parquet

    def spy(path, *a, **kw):
        seen.append(Path(path).resolve())
        return real(path, *a, **kw)

    pd.read_parquet = spy
    try:
        X.build()
    finally:
        pd.read_parquet = real
    got_e = sorted(p.name for p in seen if p.parent == ENG.resolve())
    assert got_e == ["screen_g.parquet", "status.parquet"], neq(
        "parquet files that the build read in library_on_engine/", got_e, ["screen_g.parquet", "status.parquet"]
    )
    got_b = sorted(p.name for p in seen if p.parent == BIN.resolve())
    want_b = ["crosscheck.parquet", "screen_g.parquet", "status.parquet"] if has_binance() else []
    assert got_b == want_b, neq("parquet files that the build read in library_on_engine/binance/", got_b, want_b)
    other = [str(p) for p in seen if p.parent not in (ENG.resolve(), BIN.resolve())]
    assert not other, f"parquet files that the build read in another folder: {shown(other)}"
    # and with values/ absent the payload is the same
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("status.parquet", "screen_g.parquet", "manifest.json"):
            shutil.copy2(ENG / name, Path(tmp) / name)
        if has_binance():
            (Path(tmp) / "binance").mkdir()
            for name in ("status.parquet", "screen_g.parquet", "crosscheck.parquet", "manifest.json"):
                shutil.copy2(BIN / name, Path(tmp) / "binance" / name)
        fresh, _ = _split(X.build(engine_dir=Path(tmp)))
        assert fresh["engine"] == payload()["engine"] and fresh["binance"] == payload()["binance"], (
            diff_dicts("engine block without values/ against the export", fresh["engine"], payload()["engine"])
            + "; "
            + diff_dicts("binance block without values/ against the export", fresh["binance"], payload()["binance"])
        )


def test_engine_missing_is_an_empty_state():
    with tempfile.TemporaryDirectory() as tmp:
        d = X.build(engine_dir=Path(tmp))
        e = d["engine"]
        assert e["available"] is False and "has no computation" in e["reason"], (
            f"available {e['available']!r}, reason {e.get('reason')!r}"
        )
        assert any("libfactors.py --compute" in c for c in e["commands"]) and any(
            "--screen" in c for c in e["commands"]
        ), f"the commands lack libfactors.py --compute or --screen: {e['commands']}"
        C.render_section("library", d)
        assert not list(Path(tmp).iterdir()), "the builder created a file"
        if not has_engine():
            return
        # a computation without the screen: the funnel stops at the screened step, the screen blocks are null
        shutil.copy2(ENG / "status.parquet", Path(tmp) / "status.parquet")
        man = dict(esrc()["man"])
        man.pop("screen")
        (Path(tmp) / "manifest.json").write_text(json.dumps(man), encoding="ascii")
        e = X.build(engine_dir=Path(tmp))["engine"]
        assert e["available"] is True and e["screen"] is None and e["top"] is None, (
            f"available {e['available']!r}, screen is None {e['screen'] is None}, top is None {e['top'] is None}"
        )
        assert "--screen" in e["screen_missing"] and [x["key"] for x in e["funnel"]][-1] == "screened", (
            f"screen_missing {e.get('screen_missing')!r}; funnel keys {[x['key'] for x in e['funnel']]}"
        )


# --------------------------------------------------------------------------- on Binance's data


def _skip_binance() -> bool:
    if not has_binance():
        print(
            "        (skipped: engine/data/library_on_engine/binance has no outputs; run libfactors.py --source binance)"
        )
        assert payload()["binance"]["available"] is False, (
            f"the binance block is available ({payload()['binance']['available']!r}) without the Binance outputs"
        )
        return True
    return False


def _bin_classes(fac: pd.DataFrame) -> dict:
    """The binance status columns re-derived from the catalog CSV: OI from the raw inputs, vwap
    from the formula, the fields OKX lacks, the new fields, the taker-ratio flag."""
    inputs = fac["raw_inputs"].map(lambda x: set(x.split(",")))
    vwap = fac["expression"].str.contains(r"\bvwap(?:_dev)?\b")
    only = inputs.map(lambda i: [f for f in BIN_ONLY if f in i])
    return {
        "uses_oi": inputs.map(lambda i: bool(i & set(OI))),
        "uses_vwap": vwap,
        "only": only.map(",".join),
        "new_fields": pd.Series(
            [",".join([f for f in BIN_ONLY + OI if f in i] + (["vwap"] if v else [])) for i, v in zip(inputs, vwap)],
            index=fac.index,
        ),
        "taker5m": inputs.map(lambda i: "taker_buy_sell_ratio" in i),
    }


def test_binance_funnel_and_counts_recomputed():
    """Roles, origins and flags re-derived from the catalog and the OKX status; every funnel count
    and the PIT cells from the binance status.parquet; all against the manifest."""
    if _skip_binance():
        return
    s, b = src(), payload()["binance"]
    st, man = bsrc()["st"], bsrc()["man"]
    fac = s["fac"]
    assert b["available"] is True and st["factor_id"].tolist() == fac["factor_id"].tolist(), (
        f"binance block available {b['available']!r}; "
        + diff_lists("binance status.parquet ids against the catalog", st["factor_id"], fac["factor_id"])
    )
    k = _bin_classes(fac)
    for col, want in (
        ("uses_oi", k["uses_oi"]),
        ("uses_vwap", k["uses_vwap"]),
        ("binance_only_fields", k["only"]),
        ("new_fields", k["new_fields"]),
        ("taker_ratio_last_5m", k["taker5m"]),
    ):
        assert st[col].tolist() == want.tolist(), diff_lists(
            f"binance status.parquet column {col} against the classes recomputed here",
            st[col].tolist(),
            want.tolist(),
            st["factor_id"],
        )
    assert (st["group"] == "computable").all() and not st["vwap_proxy"].any(), (
        "every definition computable, none degenerate"
    )
    # the role is the OKX screen's own status: cross-check what it screened, new family the rest
    okx = esrc()["st"]
    assert st["okx_screen"].tolist() == okx["screen"].tolist() and st["okx_group"].tolist() == okx["group"].tolist(), (
        diff_lists(
            "okx_screen (binance status against the OKX status)", st["okx_screen"], okx["screen"], st["factor_id"]
        )
        + "; "
        + diff_lists(
            "okx_group (binance status against the OKX status)", st["okx_group"], okx["group"], st["factor_id"]
        )
    )
    role = np.where(okx["screen"] == "screened", "crosscheck", "new_family")
    assert st["role"].tolist() == role.tolist(), diff_lists(
        "role (binance status against the OKX screen)", st["role"], role, st["factor_id"]
    )
    origin = np.select(
        [role == "crosscheck", (k["only"] != "").to_numpy(), k["uses_oi"].to_numpy(), k["uses_vwap"].to_numpy()],
        ["okx_screened", "binance_only_fields", "oi", "vwap"],
        default="other",
    )
    assert st["origin"].tolist() == origin.tolist(), diff_lists(
        "origin (binance status against the catalog)", st["origin"], origin, st["factor_id"]
    )
    new = st["role"] == "new_family"
    n = b["counts"]
    want = {
        "definitions": len(st),
        "computable": len(st),
        "computed": int((st["compute"] == "ok").sum()),
        "compute_errors": int((st["compute"] == "error").sum()),
        "empty": int(((st["compute"] == "ok") & (st["finite_cells"] == 0)).sum()),
        "pit_pass": int((st["pit_check"] == "pass").sum()),
        "pit_fail": int((st["pit_check"] == "fail").sum()),
        "pit_no_finite": int((st["pit_check"] == "no_finite_cells").sum()),
        "uses_oi": int(k["uses_oi"].sum()),
        "uses_vwap": int(k["uses_vwap"].sum()),
        "vwap_degenerate": 0,
        "taker_last_5m": int(k["taker5m"].sum()),
        "new_family": int(new.sum()),
        "crosscheck": int((~new).sum()),
        "new_screened": int((new & (st["screen"] == "screened")).sum()),
        "new_too_few_weeks": int((new & (st["screen"] == "too_few_weeks")).sum()),
        "new_no_g_values": int((new & (st["screen"] == "no_g_values")).sum()),
        "crosscheck_with_g": int((~new & (st["screen"] == "crosscheck")).sum()),
    }
    assert n == want, {x: (n.get(x), v) for x, v in want.items() if n.get(x) != v}
    # every definition computes on the Binance panel and passes the point-in-time proof
    assert n["computed"] == n["definitions"] and n["pit_fail"] == 0, (
        f"computed {num(n['computed'])} of {num(n['definitions'])} definitions; pit_fail {num(n['pit_fail'])}: "
        f"{shown(st.loc[st['pit_check'] == 'fail', 'factor_id'])}; not computed: "
        f"{shown(st.loc[st['compute'] != 'ok', 'factor_id'])}"
    )
    # the cross-check is exactly what the OKX screen screened; the new family is everything else
    okx_screened = int((okx["screen"] == "screened").sum())
    assert n["crosscheck"] == int((okx["screen"] == "screened").sum()) == n["crosscheck_with_g"], (
        f"crosscheck {num(n['crosscheck'])}, OKX screened {okx_screened:,}, crosscheck_with_g {num(n['crosscheck_with_g'])}"
    )
    assert n["new_screened"] + n["new_too_few_weeks"] + n["new_no_g_values"] == n["new_family"], (
        f"new_screened {num(n['new_screened'])} + new_too_few_weeks {num(n['new_too_few_weeks'])} + new_no_g_values "
        f"{num(n['new_no_g_values'])} = {num(total(n['new_screened'], n['new_too_few_weeks'], n['new_no_g_values']))}, "
        f"new_family {num(n['new_family'])}"
    )
    # the screen's weekly-IC filter: too_few_weeks below 20, screened at 20 or more
    few = st.loc[new & (st["screen"] == "too_few_weeks")]
    assert (st.loc[new & (st["screen"] == "too_few_weeks"), "g_weeks"] < 20).all(), failing(
        "new-family too_few_weeks factors with 20 or more weekly ICs",
        few["g_weeks"] < 20,
        [f"{i} {w}" for i, w in zip(few["factor_id"], few["g_weeks"])],
    )
    scr = st.loc[new & (st["screen"] == "screened")]
    assert (st.loc[new & (st["screen"] == "screened"), "g_weeks"] >= 20).all(), failing(
        "new-family screened factors with fewer than 20 weekly ICs",
        scr["g_weeks"] >= 20,
        [f"{i} {w}" for i, w in zip(scr["factor_id"], scr["g_weeks"])],
    )
    nog = st.loc[new & (st["screen"] == "no_g_values")]
    assert (st.loc[new & (st["screen"] == "no_g_values"), "g_weeks"] == 0).all(), failing(
        "new-family no_g_values factors with weekly ICs",
        nog["g_weeks"] == 0,
        [f"{i} {w}" for i, w in zip(nog["factor_id"], nog["g_weeks"])],
    )
    mc = man["compute"]["counts"]
    want_origin = {o: int((new & (st["origin"] == o)).sum()) for o in X.BINANCE_ORIGINS}
    assert mc["new_family_by_origin"] == want_origin, diff_dicts(
        "new_family_by_origin (manifest against the binance status)", mc["new_family_by_origin"], want_origin
    )
    # PIT cells
    p = b["pit"]
    cuts = range(1, 4)
    assert p["cells"] == [int(st[f"pit_cells_t{j}"].sum()) for j in cuts] == man["compute"]["pit"]["compared_cells"], (
        f"compared cells by cut: payload {p['cells']}, status.parquet {[int(st[f'pit_cells_t{j}'].sum()) for j in cuts]}, "
        f"manifest {man['compute']['pit']['compared_cells']}"
    )
    assert p["finite"] == [int(st[f"pit_finite_t{j}"].sum()) for j in cuts] == man["compute"]["pit"]["finite_cells"], (
        f"finite cells by cut: payload {p['finite']}, status.parquet {[int(st[f'pit_finite_t{j}'].sum()) for j in cuts]}, "
        f"manifest {man['compute']['pit']['finite_cells']}"
    )
    assert all(int(st[f"pit_diff_cells_t{j}"].sum()) == 0 for j in cuts), (
        "definitions with differing cells by cut: "
        + "; ".join(f"t{j} {shown(st.loc[st[f'pit_diff_cells_t{j}'] > 0, 'factor_id'])}" for j in cuts)
    )
    want_oi = [int((k["uses_oi"] & (st[f"pit_finite_t{j}"] > 0)).sum()) for j in cuts]
    assert p["oi_finite"] == want_oi, neq(
        "OI factors with finite cells by cut (payload against status.parquet)", p["oi_finite"], want_oi
    )
    assert p["cuts"] == man["zones"]["pit_cuts"] and (p["rtol"], p["atol"]) == (1e-9, 1e-12), (
        neq("PIT cuts (payload against the manifest zones)", p["cuts"], man["zones"]["pit_cuts"])
        + "; "
        + neq("tolerances (rtol, atol)", (p["rtol"], p["atol"]), (1e-9, 1e-12))
    )
    # funnel: keys, counts, narrowing outside the two branch rows
    f = {x["key"]: x for x in b["funnel"]}
    keys = ["library", "computed", "pit", "crosscheck", "new_family", "screened", "ric_q10", "ric_q05", "fit_q05"]
    assert [x["key"] for x in b["funnel"]] == keys, diff_lists("funnel keys", [x["key"] for x in b["funnel"]], keys)
    steps6 = ("library", "computed", "pit", "crosscheck", "new_family", "screened")
    want_n = [n["definitions"], n["computed"], n["pit_pass"], n["crosscheck"], n["new_family"], n["new_screened"]]
    assert [f[x]["n"] for x in steps6] == want_n, diff_lists(
        "funnel counts against the payload counts", [f[x]["n"] for x in steps6], want_n, steps6
    )
    g = bsrc()["g"]
    for key, q in (("ric_q10", 0.10), ("ric_q05", 0.05)):
        surv = g["ric_q"] <= q
        want_k = (int(surv.sum()), int(g.loc[surv, f"ric_rep_{key[4:]}"].sum()))
        assert (f[key]["n"], f[key]["signals"]) == want_k, neq(
            f"funnel {key} (factors, signals) against screen_g.parquet", (f[key]["n"], f[key]["signals"]), want_k
        )
    branch = [f[x].get("branch") is not None for x in f]
    want_branch = [False, False, False, True, False, False, False, False, True]
    assert [f[x].get("branch") is not None for x in f] == want_branch, diff_lists(
        "branch rows of the funnel", branch, want_branch, list(f)
    )
    steps = [x["n"] for x in b["funnel"] if not x.get("branch")]
    assert steps == sorted(steps, reverse=True), diff_lists(
        "funnel counts without the branch rows, against the same counts from the largest",
        steps,
        sorted(steps, reverse=True),
        [x["key"] for x in b["funnel"] if not x.get("branch")],
    )
    pit_note = f["pit"]["note"]
    assert (w := "%.1f million values compared" % (sum(p["cells"]) / 1e6)) in pit_note, missing_text(w, pit_note)
    assert (w := "%.1f million of them finite" % (sum(p["finite"]) / 1e6)) in pit_note, missing_text(w, pit_note)
    assert (
        w := "equal up to floating-point rounding (rtol 1e-9, atol 1e-12)"
    ) in pit_note and "identical" not in pit_note, (
        missing_text(w, pit_note) + f"; 'identical' in it: {'identical' in pit_note}"
    )
    assert (
        w := "all {} open-interest factors have finite values at every cut".format(f"{num(n['uses_oi'])}")
    ) in pit_note, missing_text(w, pit_note)
    byo = mc["new_family_by_origin"]
    assert (
        w := "{} read a field the engine's OKX data does not have, {} open interest, {} vwap, {} other".format(
            *tuple(f"{num(byo[o])}" for o in X.BINANCE_ORIGINS)
        )
    ) in f["new_family"]["note"], missing_text(w, f["new_family"]["note"])
    assert (w := f"({n['new_too_few_weeks']:d} have fewer, {n['new_no_g_values']:d} no value on G)") in f["screened"][
        "note"
    ], missing_text(w, f["screened"]["note"])
    fq = X._bh(g["ric_fit_p"].to_numpy(dtype=float))
    assert f["fit_q05"]["n"] == int((fq <= 0.05).sum()), neq(
        "funnel fit_q05 against the G_fit q-values recomputed here", f["fit_q05"]["n"], int((fq <= 0.05).sum())
    )
    # meta and notes
    m = b["meta"]
    pm = man["panels"]
    got_meta = (m["symbols"], m["first_bar"], m["data_through"])
    want_meta = (20, pm["first_bar"], pm["data_through"])
    assert got_meta == want_meta, neq(
        "meta (symbols, first_bar, data_through) against the manifest panels", got_meta, want_meta
    )
    want_bars = len(pd.date_range(pm["first_bar"], pm["data_through"], freq="D"))
    assert m["bars"] == want_bars, neq("bars against the days from first_bar to data_through", m["bars"], want_bars)
    want_fields = BIN_ONLY + OI + ["open", "high", "low", "close", "volume", "quote_volume", "funding_rate"]
    assert set(m["fields"]) == set(want_fields), diff_sets("panel fields", m["fields"], want_fields)
    assert m["catalog_current"] is True and m["okx_current"] is True, (
        f"catalog_current {m['catalog_current']!r}, okx_current {m['okx_current']!r} (both must be True)"
    )
    changed = [f for f, h in man["code"]["files"].items() if hashlib.sha256((REPO / f).read_bytes()).hexdigest() != h]
    assert m["code_changed"] == changed and m["code_current"] is (not changed), (
        neq("code_changed (payload against the files on disk)", m["code_changed"], changed)
        + f"; code_current {m['code_current']!r}"
    )
    okx_files = {
        k_: hashlib.sha256((ENG / k_).read_bytes()).hexdigest() == v for k_, v in man["inputs"]["okx_screen"].items()
    }
    assert m["okx_current"] == all(
        hashlib.sha256((ENG / k_).read_bytes()).hexdigest() == v for k_, v in man["inputs"]["okx_screen"].items()
    ), f"okx_current {m['okx_current']!r}; OKX screen files with the recorded hash: {okx_files}"
    want_map = {sym: sym.replace("-", "") for sym in pm["symbols"]}
    assert pm["symbol_map"] == want_map, diff_dicts("symbol_map", pm["symbol_map"], want_map)
    h = " ".join(b["notes"])
    for w in (
        "Binance is a data source only",
        "for the same 20 coins",
        "Nothing here trades.",
        "the 7-day forward return from the OKX label table",
        "Only the factor inputs come from Binance",
        "separate Benjamini-Hochberg family: the {} definitions".format(num(n["new_family"])),
        "({} reach the screen)".format(num(n["new_screened"])),
        "The cross-check of the {} definitions".format(num(n["crosscheck"])),
        "agreement numbers only",
        "These numbers come from the Sundays of the zone G (2022-04-03 to 2024-06-09).",
        "To test a definition in a book, use its register command.",
    ):
        assert w in h, missing_text(w, h)
    assert len(b["notes"]) == len(X.BINANCE_NOTES), neq(
        "the number of notes against BINANCE_NOTES", len(b["notes"]), len(X.BINANCE_NOTES)
    )
    assert b["zones"] == payload()["engine"]["zones"], "the Binance screen uses the OKX screen's zones"


def test_binance_screen_recomputed():
    """BH over the new family recomputed from its p-values; survivors, signals, replication,
    sensitivity, families, origins, fields and the overlap counts."""
    if _skip_binance():
        return
    b = payload()["binance"]
    st, g, man = bsrc()["st"], bsrc()["g"], bsrc()["man"]
    sc, mcnt = b["screen"], man["screen"]["counts"]
    new = st["role"] == "new_family"
    assert set(g["factor_id"]) == set(st.loc[new & (st["screen"] == "screened"), "factor_id"]), diff_sets(
        "screen_g.parquet ids against the screened new-family ids of status.parquet",
        g["factor_id"],
        st.loc[new & (st["screen"] == "screened"), "factor_id"],
    )
    assert not g["factor_id"].isin(st.loc[~new, "factor_id"]).any(), "an OKX-screened factor is in the new family"
    assert sc["n_tests_ic"] == sc["n_tests_ric"] == len(g) and sc["min_weeks"] == 20, (
        f"n_tests_ic {num(sc['n_tests_ic'])}, n_tests_ric {num(sc['n_tests_ric'])}, screen_g.parquet rows {len(g):,}; "
        f"min_weeks {sc['min_weeks']} (expected 20)"
    )
    assert (g["ic_n"] >= 20).all() and sc["names_min"] <= sc["names_max"] <= 20 and sc["g_sundays"] == 114, (
        failing("screened factors with fewer than 20 weekly ICs", g["ic_n"] >= 20, g["factor_id"])
        + f"; names per Sunday {sc['names_min']} to {sc['names_max']} (at most 20); g_sundays {sc['g_sundays']} "
        "(expected 114)"
    )
    for stat in ("ic", "ric"):
        t = g[f"{stat}_mean"].astype(float) / g[f"{stat}_sd"].astype(float) * np.sqrt(g[f"{stat}_n"].astype(float))
        ok = np.isclose(t, g[f"{stat}_t"].astype(float), rtol=1e-4, atol=1e-4)
        assert ok.all(), failing(
            f"Binance {stat}_t against {stat}_mean / {stat}_sd * sqrt({stat}_n)",
            ok,
            [f"{i} {a} != {b}" for i, a, b in zip(g["factor_id"], g[f"{stat}_t"], t)],
        )
    rows = {(x["stat"], x["q"]): x for x in sc["bh"]}
    for stat in ("ic", "ric"):
        q = bh_q(g[f"{stat}_p"].to_numpy(dtype=float))  # this family's tests only
        assert np.allclose(q, g[f"{stat}_q"].to_numpy(dtype=float), rtol=1e-9, atol=1e-12), f"{stat} BH q differs"
        for lvl in (0.05, 0.10):
            tag = f"q{round(lvl * 100):02d}"
            surv = q <= lvl
            assert (g.loc[surv, f"{stat}_signal_{tag}"] >= 0).all() and (
                g.loc[~surv, f"{stat}_signal_{tag}"] == -1
            ).all(), (
                failing(
                    f"{stat} {tag} survivors without a signal",
                    g.loc[surv, f"{stat}_signal_{tag}"] >= 0,
                    g.loc[surv, "factor_id"],
                )
                + "; "
                + failing(
                    f"{stat} {tag} non-survivors with a signal",
                    g.loc[~surv, f"{stat}_signal_{tag}"] == -1,
                    g.loc[~surv, "factor_id"],
                )
            )
            for sid, grp in g[surv].groupby(f"{stat}_signal_{tag}"):
                assert int(grp[f"{stat}_rep_{tag}"].sum()) == 1, (
                    f"{stat} {tag} signal {sid}: {int(grp[f'{stat}_rep_{tag}'].sum())} representatives "
                    f"{shown(grp.loc[grp[f'{stat}_rep_{tag}'].astype(bool), 'factor_id'])} of {shown(grp['factor_id'])}"
                )
                best = grp[f"{stat}_t"].abs().max()
                rep_t = float(grp.loc[grp[f"{stat}_rep_{tag}"], f"{stat}_t"].iloc[0])
                assert abs(abs(rep_t) - best) < 1e-6, (
                    f"{stat} {tag} signal {sid}: the representative "
                    f"{grp.loc[grp[f'{stat}_rep_{tag}'], 'factor_id'].iloc[0]} has |t| {abs(rep_t)}, "
                    f"the largest |t| is {best} ({grp.loc[grp[f'{stat}_t'].abs().idxmax(), 'factor_id']})"
                )
            r = rows[(stat, lvl)]
            got_r = (r["factors"], r["signals"], r["positive"])
            want_r = (
                int(surv.sum()),
                int(g.loc[surv, f"{stat}_rep_{tag}"].sum()),
                int((surv & (g[f"{stat}_t"] > 0).to_numpy()).sum()),
            )
            assert got_r == want_r, neq(
                f"{stat} {tag} (factors, signals, positive) against screen_g.parquet", got_r, want_r
            )
            want_m = (mcnt[f"{stat}_count_{tag}"], mcnt[f"{stat}_distinct_signals_{tag}"])
            assert (r["factors"], r["signals"]) == want_m, neq(
                f"{stat} {tag} (factors, signals), payload against the manifest", (r["factors"], r["signals"]), want_m
            )
    # replication: BH over the G_fit p-values alone, checked on G_eval
    rp = {(x["stat"], x["q"]): x for x in sc["replication"]["rows"]}
    for stat in ("ic", "ric"):
        fp = g[f"{stat}_fit_p"].to_numpy(dtype=float)
        fq = np.full(len(fp), np.nan)
        fq[np.isfinite(fp)] = bh_q(fp[np.isfinite(fp)])
        stored = g[f"{stat}_fit_q"].to_numpy(dtype=float)
        assert np.allclose(fq, g[f"{stat}_fit_q"].to_numpy(dtype=float), rtol=1e-9, atol=1e-12, equal_nan=True), (
            failing(
                f"{stat} G_fit q (recomputed against stored)",
                np.isclose(fq, stored, rtol=1e-9, atol=1e-12, equal_nan=True),
                [f"{i} {a} != {b}" for i, a, b in zip(g["factor_id"], fq, stored)],
            )
        )
        assert sc["replication"]["smallest_fit_q"][stat] == round(float(np.nanmin(fq)), 4), neq(
            f"{stat} smallest G_fit q (payload against the recomputed q)",
            sc["replication"]["smallest_fit_q"][stat],
            round(float(np.nanmin(fq)), 4),
        )
        fm, em = g[f"{stat}_fit_mean"].astype(float), g[f"{stat}_eval_mean"].astype(float)
        same = fm.notna() & em.notna() & (fm != 0) & (np.sign(fm) == np.sign(em))
        for lvl in (0.05, 0.10):
            sel = fq <= lvl
            r = rp[(stat, lvl)]
            got_p = (
                r["n_tests"],
                r["selected"],
                r["same_sign"],
                r["same_sign_t2"],
                r["base_pairs"],
                r["base_same_sign"],
            )
            want_p = (
                int(np.isfinite(fp).sum()),
                int(sel.sum()),
                int((sel & same).sum()),
                int((sel & same & (g[f"{stat}_eval_t"].abs() >= 2)).sum()),
                int((fm.notna() & em.notna() & (fm != 0)).sum()),
                int(same.sum()),
            )
            assert got_p == want_p, neq(
                f"Binance {stat} q <= {lvl} G_fit selection (n_tests, selected, same_sign, same_sign_t2, base_pairs, "
                "base_same_sign), payload against screen_g.parquet",
                got_p,
                want_p,
            )
    # sensitivity rows are the manifest's, in its order; the registered one is the screen
    sens = {x["key"]: x for x in sc["sensitivity"]}
    ms = man["screen"]["sensitivity"]
    assert list(sens) == list(ms) and list(sens)[0] == "registered", neq(
        "sensitivity rows (payload against the manifest; the first is registered)", list(sens), list(ms)
    )
    for key, x in sens.items():
        got_s = (x["screened"], x["q05"], x["q10"], x["fit_selected"], x["same_sign"], x["same_sign_t2"])
        want_s = (
            ms[key]["screened"],
            ms[key]["ric_count_q05"],
            ms[key]["ric_count_q10"],
            ms[key]["ric_fit_selected_q05"],
            ms[key]["eval_same_sign"],
            ms[key]["eval_same_sign_t2"],
        )
        assert got_s == want_s, neq(
            f"Binance sensitivity row {key} (screened, q05, q10, fit_selected, same_sign, same_sign_t2), payload "
            "against the manifest",
            got_s,
            want_s,
        )
    got_reg = (sens["registered"]["screened"], sens["registered"]["q05"], sens["registered"]["q10"])
    want_reg = (len(g), rows[("ric", 0.05)]["factors"], rows[("ric", 0.10)]["factors"])
    assert got_reg == want_reg, neq(
        "the registered sensitivity row (screened, q05, q10) against the screen", got_reg, want_reg
    )
    assert all(x["note"] == ms[k_]["note"] for k_, x in sens.items()), (
        f"sensitivity rows whose note differs from the manifest: {shown(k_ for k_, x in sens.items() if x['note'] != ms[k_]['note'])}"
    )
    # families
    fams = {x["family"]: x for x in sc["by_family"]}
    assert set(fams) == set(g["family"]), diff_sets(
        "families of the screen against screen_g.parquet", fams, g["family"]
    )
    for fam, x in fams.items():
        fg = g[g["family"] == fam]
        rr = fg["ric_q"] <= 0.05
        got_f = (x["candidates"], x["screened"], x["ric_q05"], x["ric_q05_signals"], x["ric_q10"], x["ic_q05"])
        want_f = (
            int((new & (st["family"] == fam)).sum()),
            len(fg),
            int(rr.sum()),
            int((rr & fg["ric_rep_q05"]).sum()),
            int((fg["ric_q"] <= 0.10).sum()),
            int((fg["ic_q"] <= 0.05).sum()),
        )
        assert got_f == want_f, neq(
            f"Binance family {fam} (candidates, screened, ric_q05, ric_q05_signals, ric_q10, ic_q05), payload "
            "against status.parquet and screen_g.parquet",
            got_f,
            want_f,
        )
    fam_signals = sum(x["ric_q05_signals"] for x in sc["by_family"])
    assert fam_signals == rows[("ric", 0.05)]["signals"], neq(
        "the sum of the rIC q05 signals of the families against the screen", fam_signals, rows[("ric", 0.05)]["signals"]
    )
    # overlap with the OKX screen at the same q: counts from the per-factor columns, OKX set sizes from the OKX screen
    og = esrc()["g"]
    for x in sc["overlap"]:
        key = f"{x['stat']}_q{round(x['q'] * 100):02d}"
        surv = g[f"{x['stat']}_q"] <= x["q"]
        assert x["okx_count"] == int((og[f"{x['stat']}_q"] <= x["q"]).sum()), neq(
            f"overlap {key}: okx_count against the OKX screen",
            x["okx_count"],
            int((og[f"{x['stat']}_q"] <= x["q"]).sum()),
        )
        want_ov = (int(surv.sum()), int((surv & (g[f"okx_{key}_max_rho"] >= 0.7)).sum()))
        assert (x["count"], x["overlapping"]) == want_ov, neq(
            f"overlap {key} (count, overlapping) against screen_g.parquet", (x["count"], x["overlapping"]), want_ov
        )
        want_okx_values = man["screen"]["counts"]["overlap_with_okx_q_sets"][key]["overlapping_with_okx_values"]
        assert x["overlapping_okx_values"] == want_okx_values, neq(
            f"overlap {key}: overlapping_okx_values against the manifest", x["overlapping_okx_values"], want_okx_values
        )
    assert sc["regressors"] == man["screen"]["ric_regressors"] and sc["label"] == man["screen"]["label"], (
        diff_dicts("regressors (payload against the manifest)", sc["regressors"], man["screen"]["ric_regressors"])
        + "; "
        + neq("label", sc["label"], man["screen"]["label"])
    )
    assert sc["regressors"]["config"] == payload()["engine"]["screen"]["regressors"]["config"], (
        "the OKX screen's rIC settings"
    )


def test_binance_fields_and_origins_recomputed():
    """What the new fields unlock: per field and per origin, from the new_fields and origin columns."""
    if _skip_binance():
        return
    b = payload()["binance"]
    st, g, man = bsrc()["st"], bsrc()["g"], bsrc()["man"]
    new = st["role"] == "new_family"
    sf = st["new_fields"].map(lambda x: set(f for f in x.split(",") if f))
    gf = g["new_fields"].map(lambda x: set(f for f in x.split(",") if f))
    sets = {
        "ric_q05": g["ric_q"] <= 0.05,
        "ric_q10": g["ric_q"] <= 0.10,
        "ic_q05": g["ic_q"] <= 0.05,
        "ic_q10": g["ic_q"] <= 0.10,
    }
    rows = {x["field"]: x for x in b["fields"]}
    assert set(rows) == set(BIN_ONLY + OI + ["vwap"]) == set(X.BINANCE_NEW_FIELDS), (
        diff_sets("new fields (payload against BIN_ONLY, OI and vwap)", rows, BIN_ONLY + OI + ["vwap"])
        + "; "
        + diff_sets("new fields (payload against BINANCE_NEW_FIELDS)", rows, X.BINANCE_NEW_FIELDS)
    )
    for fld, x in rows.items():
        has_s, has_g = sf.map(lambda v, f=fld: f in v), gf.map(lambda v, f=fld: f in v)
        got_c, want_c = (x["candidates"], x["screened"]), (int((new & has_s).sum()), int(has_g.sum()))
        assert got_c == want_c, neq(
            f"new field {fld} (candidates, screened) against the status and the screen", got_c, want_c
        )
        for key, m in sets.items():
            assert x[key] == int((m & has_g).sum()), neq(
                f"new field {fld}: {key} against screen_g.parquet", x[key], int((m & has_g).sum())
            )
            want = man["screen"]["counts"]["q_sets"][key]["by_new_field"].get(fld, 0)
            assert x[key] == want, neq(f"new field {fld}: {key} against the manifest", x[key], want)
        assert x["label"] and x["source"], f"{fld}: label {x.get('label')!r}, source {x.get('source')!r}"
    assert "5-minute snapshot" in rows["taker_buy_sell_ratio"]["source"], missing_text(
        "5-minute snapshot", rows["taker_buy_sell_ratio"]["source"]
    )
    q05_by_field = [x["ric_q05"] for x in b["fields"]]
    assert q05_by_field == sorted(q05_by_field, reverse=True), diff_lists(
        "rIC q05 counts of the fields in page order against the descending order",
        q05_by_field,
        sorted(q05_by_field, reverse=True),
        [x["field"] for x in b["fields"]],
    )
    org = {x["origin"]: x for x in b["by_origin"]}
    assert list(org) == list(X.BINANCE_ORIGINS), neq(
        "origins (payload against BINANCE_ORIGINS)", list(org), list(X.BINANCE_ORIGINS)
    )
    for o, x in org.items():
        mo, go = new & (st["origin"] == o), g["origin"] == o
        got_c, want_c = (x["candidates"], x["screened"]), (int(mo.sum()), int(go.sum()))
        assert got_c == want_c, neq(
            f"origin {o} (candidates, screened) against the status and the screen", got_c, want_c
        )
        want_st = {k_: int(v) for k_, v in st.loc[mo, "okx_screen"].value_counts().sort_index().items()}
        assert x["okx_status"] == want_st, diff_dicts(
            f"origin {o}: okx_status against status.parquet", x["okx_status"], want_st
        )
        for key, m in sets.items():
            assert x[key] == int((m & go).sum()), neq(
                f"origin {o}: {key} against screen_g.parquet", x[key], int((m & go).sum())
            )
    # the OKX status behind the first origin: OKX could not compute any of them
    assert set(org["binance_only_fields"]["okx_status"]) <= {"not_computable"}, (
        f"OKX states of the origin binance_only_fields other than not_computable: "
        f"{sorted(set(org['binance_only_fields']['okx_status']) - {'not_computable'})}"
    )


def test_binance_top_table_recomputed():
    if _skip_binance():
        return
    s, b = src(), payload()["binance"]
    g = bsrc()["g"]
    t = b["top"]
    order = g.assign(a=g["ric_t"].abs()).sort_values(["a", "factor_id"], ascending=[False, True], kind="stable")
    rank = {fid: i + 1 for i, fid in enumerate(order["factor_id"])}
    every = order[order["ric_q"] <= 0.10]
    want = every.head(X.TOP_MAX)  # the rows are capped; the counts are the full ones
    assert t["q"] == 0.10 and t["n"] == len(every) and t["shown"] == len(want) and t["ranked_among"] == len(g), neq(
        "top table (q, n, shown, ranked_among)",
        (t["q"], t["n"], t["shown"], t["ranked_among"]),
        (0.10, len(every), len(want), len(g)),
    )
    assert t["n_q05"] == int((g["ric_q"] <= 0.05).sum()), neq(
        "n_q05 against screen_g.parquet", t["n_q05"], int((g["ric_q"] <= 0.05).sum())
    )
    want_rank = [rank[x] for x in want["factor_id"]]
    assert t["id"] == want["factor_id"].tolist() and t["rank"] == [rank[x] for x in want["factor_id"]] == sorted(
        t["rank"]
    ), (
        diff_lists("top table ids", t["id"], want["factor_id"])
        + "; "
        + diff_lists("ranks", t["rank"], want_rank, t["id"])
        + "; "
        + diff_lists("ranks against the sorted order", t["rank"], sorted(t["rank"]), t["id"])
    )
    _check_signal_info(g, want, t)
    fac = s["fac"].set_index("factor_id")
    fams, origins = b["families"], b["origins"]
    for i, (_, x) in enumerate(want.iterrows()):
        fid = x["factor_id"]
        assert (
            t["name"][i] == x["name"] == fac.at[fid, "name"]
            and fams[t["fam"][i]] == x["family"] == fac.at[fid, "family"]
        ), (
            f"{fid}: name {t['name'][i]!r} / screen {x['name']!r} / catalog {fac.at[fid, 'name']!r}; family "
            f"{fams[t['fam'][i]]!r} / screen {x['family']!r} / catalog {fac.at[fid, 'family']!r}"
        )
        assert t["expr"][i] == fac.at[fid, "expression"] and t["fields"][i] == x["new_fields"], (
            neq(f"{fid} formula", t["expr"][i], fac.at[fid, "expression"])
            + "; "
            + neq("new fields", t["fields"][i], x["new_fields"])
        )
        assert origins[t["origin"][i]] == x["origin"] and t["taker5m"][i] == bool(x["taker_ratio_last_5m"]), (
            neq(f"{fid} origin", origins[t["origin"][i]], x["origin"])
            + "; "
            + neq("taker5m", t["taker5m"][i], bool(x["taker_ratio_last_5m"]))
        )
        for key, col, tol in (
            ("ic_mean", "ic_mean", 6e-5),
            ("ic_t", "ic_t", 0.006),
            ("ic_q", "ic_q", 6e-5),
            ("ric_mean", "ric_mean", 6e-5),
            ("ric_t", "ric_t", 0.006),
            ("ric_q", "ric_q", 6e-5),
            ("persistence", "persistence", 0.006),
            ("fit_t", "ric_fit_t", 0.006),
            ("fit_q", "ric_fit_q", 6e-5),
            ("eval_t", "ric_eval_t", 0.006),
            ("okx_rho05", "okx_ric_q05_max_rho", 6e-4),
            ("okx_rho10", "okx_ric_q10_max_rho", 6e-4),
        ):
            if pd.isna(x[col]):  # e.g. no G_fit week: the page prints a dash
                assert t[key][i] is None, f"{fid} {key}"
            else:
                close(t[key][i], x[col], tol, f"{fid} {key}")
        for lvl in ("05", "10"):
            sig = int(x[f"ric_signal_q{lvl}"])
            assert t[f"sig{lvl}"][i] == (None if sig < 0 else sig) and t[f"rep{lvl}"][i] == bool(
                x[f"ric_rep_q{lvl}"]
            ), (
                f"{fid} q{lvl}: signal {t[f'sig{lvl}'][i]!r} (screen {sig}), representative {t[f'rep{lvl}'][i]!r} "
                f"(screen {bool(x[f'ric_rep_q{lvl}'])})"
            )
    q05 = [i for i in range(t["shown"]) if t["sig05"][i] is not None]
    mc = bsrc()["man"]["screen"]["counts"]
    assert len(q05) == mc["ric_count_q05"] and sum(t["rep05"][i] for i in q05) == mc["ric_distinct_signals_q05"], (
        f"rows with a q05 signal: {len(q05):,} (manifest {num(mc['ric_count_q05'])}); their representatives "
        f"{num(sum(t['rep05'][i] for i in q05))} (manifest {num(mc['ric_distinct_signals_q05'])})"
    )
    assert sum(t["rep10"]) == mc["ric_distinct_signals_q10"], neq(
        "representatives at q <= 0.10 (table against the manifest)", sum(t["rep10"]), mc["ric_distinct_signals_q10"]
    )
    assert all(t["ric_q"][i] <= 0.05 for i in q05), "rows with a q05 signal and rIC q above 0.05: " + shown(
        f"{t['id'][i]} {t['ric_q'][i]}" for i in q05 if not t["ric_q"][i] <= 0.05
    )
    # the manifest's leading rows are the table's first rows, with its numbers
    lead = bsrc()["man"]["screen"]["top_by_abs_ric_t"]
    assert [r["factor_id"] for r in lead] == t["id"][: len(lead)], diff_lists(
        "the manifest's leading rows against the first rows of the table",
        [r["factor_id"] for r in lead],
        t["id"][: len(lead)],
    )
    for i, r in enumerate(lead):
        close(t["ric_t"][i], r["ric_t"], 0.006, r["factor_id"])
        close(t["persistence"][i], r["persistence"], 0.006, r["factor_id"])
    # survivors without a G_fit week (values start inside G_eval) cannot be reached by the G_fit selection
    want_nofit = (int(every["ric_fit_t"].isna().sum()), int(g["ric_fit_t"].isna().sum()))
    assert (t["no_fit"], t["no_fit_screened"]) == want_nofit, neq(
        "(no_fit, no_fit_screened) against screen_g.parquet", (t["no_fit"], t["no_fit_screened"]), want_nofit
    )
    n_fit_tests = bsrc()["man"]["screen"]["counts"]["fit_selection"]["ric"]["n_tests"]
    assert t["no_fit_screened"] == len(g) - n_fit_tests, (
        f"no_fit_screened {num(t['no_fit_screened'])} != screened {len(g):,} - G_fit tests {n_fit_tests:,}"
    )
    assert all((want["ric_fit_t"].isna() == (want["ric_fit_n"] == 0)).tolist()), failing(
        "rows whose missing G_fit t does not match a G_fit week count of 0",
        want["ric_fit_t"].isna() == (want["ric_fit_n"] == 0),
        [f"{i} t {a} n {c}" for i, a, c in zip(want["factor_id"], want["ric_fit_t"], want["ric_fit_n"])],
    )
    # the overlap columns count where the screen counted
    ov = {(x["stat"], x["q"]): x["overlapping"] for x in b["screen"]["overlap"]}
    n_rho05 = sum(1 for i in q05 if t["okx_rho05"][i] >= 0.7)
    assert n_rho05 == ov[("ric", 0.05)], neq(
        "rows with okx_rho05 >= 0.7 against the overlap count", n_rho05, ov[("ric", 0.05)]
    )
    n_rho10 = sum(1 for v in t["okx_rho10"] if v >= 0.7)
    assert n_rho10 == ov[("ric", 0.10)], neq(
        "rows with okx_rho10 >= 0.7 against the overlap count", n_rho10, ov[("ric", 0.10)]
    )


def test_binance_crosscheck_recomputed():
    """Agreement statistics recomputed from crosscheck.parquet and the catalog (input classes with gap
    found in the formula); the OKX columns are the OKX screen's own values."""
    if _skip_binance():
        return
    s, b = src(), payload()["binance"]
    cc, man, st = bsrc()["cc"], bsrc()["man"], bsrc()["st"]
    x = b["crosscheck"]
    assert set(cc["factor_id"]) == set(st.loc[st["role"] == "crosscheck", "factor_id"]), diff_sets(
        "crosscheck.parquet ids against the cross-check ids of status.parquet",
        cc["factor_id"],
        st.loc[st["role"] == "crosscheck", "factor_id"],
    )
    assert not [c for c in cc.columns if c.startswith("binance_") and c.endswith(("_q", "_p"))], (
        "a Binance q or p in the cross-check"
    )
    assert not [c for c in cc.columns if "signal" in c or "rep" in c], "a survivor column in the cross-check"
    og = esrc()["g"].set_index("factor_id").loc[cc["factor_id"]]
    for c in ("ic_t", "ric_t", "ric_q", "ric_fit_t", "ric_eval_t", "ic_n"):
        ok = np.isclose(cc[f"okx_{c}"].to_numpy(dtype=float), og[c].to_numpy(dtype=float), equal_nan=True)
        assert ok.all(), failing(
            f"crosscheck.parquet okx_{c} against the OKX screen_g.parquet {c}",
            ok,
            [f"{i} {a} != {b}" for i, a, b in zip(cc["factor_id"], cc[f"okx_{c}"], og[c])],
        )
    a, bb = cc["okx_ric_t"].astype(float), cc["binance_ric_t"].astype(float)
    both = a.notna() & bb.notna()

    def sp(u, v):
        m = u.notna() & v.notna()
        r1, r2 = u[m].rank().to_numpy(), v[m].rank().to_numpy()
        return round(float(np.corrcoef(r1, r2)[0, 1]), 4)

    fac = s["fac"].set_index("factor_id")
    cls = []
    for fid in cc["factor_id"]:
        e, i = fac.at[fid, "expression"], set(fac.at[fid, "raw_inputs"].split(","))
        vol, fund = bool(i & {"volume", "quote_volume"}), "funding_rate" in i
        cls.append(
            "gap"
            if re.search(r"\bgap\b", e)
            else "volume_and_funding"
            if vol and fund
            else "volume"
            if vol
            else "funding"
            if fund
            else "price_only"
            if i <= {"open", "high", "low", "close"}
            else "other"
        )
    cls = pd.Series(cls, index=cc.index)
    sm = x["summary"]
    rho = cc["value_rho"].astype(float)
    got_n = (sm["factors"], sm["compared_ric_t"])
    want_n = (len(cc), int(both.sum()))
    assert got_n == want_n, neq("(factors, compared_ric_t) against crosscheck.parquet", got_n, want_n)
    assert sm["ric_t_spearman"] == sp(a, bb), neq(
        "ric_t_spearman against the value recomputed here", sm["ric_t_spearman"], sp(a, bb)
    )
    ic_sp = sp(cc["okx_ic_t"].astype(float), cc["binance_ic_t"].astype(float))
    assert sm["ic_t_spearman"] == ic_sp, neq(
        "ic_t_spearman against the value recomputed here", sm["ic_t_spearman"], ic_sp
    )
    no_gap = sp(a[cls != "gap"], bb[cls != "gap"])
    assert sm["ric_t_spearman_without_gap"] == no_gap, neq(
        "ric_t_spearman_without_gap against the value recomputed here", sm["ric_t_spearman_without_gap"], no_gap
    )
    same_sign = round(float((np.sign(a) == np.sign(bb))[both].mean()), 4)
    assert sm["same_ric_sign"] == same_sign, neq(
        "same_ric_sign against the signs of the two t", sm["same_ric_sign"], same_sign
    )
    same_col = round(float(cc.loc[both, "same_ric_sign"].mean()), 4)
    assert sm["same_ric_sign"] == same_col, neq(
        "same_ric_sign against the column same_ric_sign of crosscheck.parquet", sm["same_ric_sign"], same_col
    )
    got_rho = (sm["rho_median"], sm["rho_p10"], sm["rho_at_least_0_9"], sm["rho_below_0_5"])
    want_rho = (
        round(float(rho.median()), 4),
        round(float(np.percentile(rho.dropna(), 10)), 4),
        round(float((rho >= 0.9).mean()), 4),
        round(float((rho < 0.5).mean()), 4),
    )
    assert got_rho == want_rho, neq(
        "value correlations (median, p10, share >= 0.9, share < 0.5) against crosscheck.parquet", got_rho, want_rho
    )
    for r in x["okx_levels"]:
        m = cc["okx_ric_q"] <= r["q"]
        same = m & both & (np.sign(a) == np.sign(bb))
        got_l = (r["okx_count"], r["same_sign"], r["same_sign_t2"])
        want_l = (int(m.sum()), int(same.sum()), int((same & (bb.abs() >= 2)).sum()))
        assert got_l == want_l, neq(
            f"OKX level q <= {r['q']} (okx_count, same_sign, same_sign_t2) against crosscheck.parquet", got_l, want_l
        )
        okx_n = int((esrc()["g"]["ric_q"] <= r["q"]).sum())
        assert r["okx_count"] == okx_n, neq(
            f"OKX level q <= {r['q']}: okx_count against the OKX screen", r["okx_count"], okx_n
        )
    by = {r["cls"]: r for r in x["by_class"]}
    assert set(by) == set(cls), diff_sets("input classes (payload against the classes recomputed here)", by, cls)
    for c, r in by.items():
        m = cls == c
        got_c, want_c = (r["factors"], r["ric_t_spearman"]), (int(m.sum()), sp(a[m], bb[m]))
        assert got_c == want_c, neq(
            f"input class {c} (factors, ric_t_spearman) against crosscheck.parquet", got_c, want_c
        )
        want_rho = round(float(rho[m].median()), 4)
        assert r["median_rho"] == want_rho, neq(
            f"input class {c}: median_rho against crosscheck.parquet", r["median_rho"], want_rho
        )
    # the definitions that read funding_rate read the same funding on both venues on G (the OKX funding there
    # is Binance's own archive): the agreement without them, and the caveats beside the numbers
    fund = pd.Series(["funding_rate" in fac.at[f_, "raw_inputs"].split(",") for f_ in cc["factor_id"]], index=cc.index)
    by_fund = sum(by[c]["factors"] for c in ("funding", "volume_and_funding") if c in by)
    assert (
        sm["funding_readers"]
        == int(fund.sum())
        == sum(by[c]["factors"] for c in ("funding", "volume_and_funding") if c in by)
    ), (
        f"funding_readers {num(sm['funding_readers'])}, definitions that read funding_rate {int(fund.sum()):,}, "
        f"the classes funding and volume_and_funding {by_fund:,}"
    )
    want_fund = {c: int((fund & (cls == c)).sum()) for c in by}
    got_fund = {c: r["funding_readers"] for c, r in by.items()}
    assert {c: r["funding_readers"] for c, r in by.items()} == want_fund, diff_dicts(
        "funding readers by input class (payload against the catalog)", got_fund, want_fund
    )
    no_fund = sp(a[~fund], bb[~fund])
    assert sm["ric_t_spearman_without_funding"] == no_fund, neq(
        "ric_t_spearman_without_funding against the value recomputed here",
        sm["ric_t_spearman_without_funding"],
        no_fund,
    )
    same_no_fund = round(float((np.sign(a) == np.sign(bb))[both & ~fund].mean()), 4)
    assert sm["same_ric_sign_without_funding"] == same_no_fund, neq(
        "same_ric_sign_without_funding against the value recomputed here",
        sm["same_ric_sign_without_funding"],
        same_no_fund,
    )
    # the two caveats are the manifest's own words, beside the numbers
    assert (
        x["agreement_note"] == man["crosscheck"]["agreement_note"]
        and "not an independent replication" in x["agreement_note"]
    ), neq(
        "agreement_note (payload against the manifest; it says 'not an independent replication')",
        x["agreement_note"],
        man["crosscheck"]["agreement_note"],
    )
    assert (
        x["funding_note"] == man["crosscheck"]["funding_note"]
        and f"{num(sm['funding_readers'])} definitions" in x["funding_note"]
    ), neq(
        f"funding_note (payload against the manifest; it names {num(sm['funding_readers'])} definitions)",
        x["funding_note"],
        man["crosscheck"]["funding_note"],
    )
    # the largest disagreements, in |OKX t - Binance t| order (ties by file order)
    diff = np.where(both, (a - bb).abs(), -1.0)
    order = np.argsort(-diff, kind="stable")
    want_ids = [cc["factor_id"].iloc[i] for i in order[:10]]
    assert [r["id"] for r in x["disagreements"]] == want_ids, diff_lists(
        "the largest disagreements", [r["id"] for r in x["disagreements"]], want_ids
    )
    for r in x["disagreements"]:
        y = cc.set_index("factor_id").loc[r["id"]]
        close(r["okx_ric_t"], y["okx_ric_t"], 0.006, r["id"])
        close(r["binance_ric_t"], y["binance_ric_t"], 0.006, r["id"])
        assert r["cls"] == cls[cc["factor_id"] == r["id"]].iloc[0] and r["cause"], (
            f"{r['id']}: class {r['cls']!r} (recomputed {cls[cc['factor_id'] == r['id']].iloc[0]!r}), cause {r.get('cause')!r}"
        )
    # the series table is the manifest's (it needs the panels; the panel test recomputes it when present)
    assert [r["series"] for r in x["series"]] == list(man["crosscheck"]["series_agreement"]), neq(
        "the series of the cross-venue table against the manifest",
        [r["series"] for r in x["series"]],
        list(man["crosscheck"]["series_agreement"]),
    )


def _panel_frames() -> dict:
    """The merged Binance panel as the engine's symbols, one row per UTC day from the first kline bar
    to the last, every field NaN where the coin has no kline bar."""
    raw = {f.name.split(".")[0]: pd.read_csv(f, index_col=0, parse_dates=True) for f in sorted(PANEL.glob("*.csv.gz"))}
    close_ = raw["close"]
    idx = pd.date_range(close_.dropna(how="all").index.min(), close_.dropna(how="all").index.max(), freq="D")
    assert str(idx.tz) == "UTC", "the panel's bars must be stamped in UTC"
    out = {}
    for f, df in raw.items():
        df = df.reindex(idx).rename(columns=lambda c: c[:-4] + "-USDT")
        out[f] = df.where(raw["close"].reindex(idx).rename(columns=lambda c: c[:-4] + "-USDT").notna())
    return out


def _cleaned_frames(p: dict) -> tuple[dict, dict]:
    """The panel after the cleaning rules (written out here): open interest or its value <= 0 clears
    both; an average price (quote / base volume) outside the bar's [low, high], relative tolerance
    1e-9, clears the base volume, for volume and for taker_buy_volume. Returns (cleaned frames,
    {rule: [(date, symbol), ...] in date then column order})."""
    lo, hi = p["low"].to_numpy(), p["high"].to_numpy()

    def outside(q, b):
        with np.errstate(all="ignore"):
            px = p[q].to_numpy() / p[b].to_numpy()
            return ~np.isnan(px) & ((px < lo * (1 - 1e-9)) | (px > hi * (1 + 1e-9)))

    oi = ((p["open_interest"] <= 0) | (p["open_interest_value"] <= 0)).to_numpy()
    rules = {
        "open_interest_nonpositive": (oi, ("open_interest", "open_interest_value")),
        "volume_vwap_outside_low_high": (outside("quote_volume", "volume"), ("volume",)),
        "taker_buy_volume_vwap_outside_low_high": (
            outside("taker_buy_quote_volume", "taker_buy_volume"),
            ("taker_buy_volume",),
        ),
    }
    out, cells = dict(p), {}
    for rule, (m, fields) in rules.items():
        for f in fields:
            out[f] = out[f].mask(m)
        r, c = np.nonzero(m)
        cells[rule] = [(str(p["close"].index[i].date()), p["close"].columns[j]) for i, j in sorted(zip(r, c))]
    return out, cells


def test_binance_panel_checks():
    """The panel checks shown on the page: the manifest's, and recomputed from the git-ignored panel
    and the engine's raw files when the panel is on disk."""
    if _skip_binance():
        return
    b, man = payload()["binance"], bsrc()["man"]
    ch, k = man["panels"]["checks"], b["checks"]
    want_funding = {
        "day_cells": ch["funding_vs_engine_binance_prints"]["day_cells"],
        "differing": ch["funding_vs_engine_binance_prints"]["differing_above_1e-12"],
        "max_abs_diff": ch["funding_vs_engine_binance_prints"]["max_abs_diff"],
    }
    assert k["funding"] == want_funding, diff_dicts(
        "funding check (payload against the manifest)", k["funding"], want_funding
    )
    assert k["funding"]["differing"] == 0 and k["funding"]["max_abs_diff"] < 1e-15, (
        f"funding cells that differ {k['funding']['differing']} (expected 0), max |difference| "
        f"{k['funding']['max_abs_diff']} (expected below 1e-15)"
    )
    assert k["vwap"]["collapses"] is False and 0.005 < k["vwap"]["median_abs"] < 0.05, (
        f"vwap collapses {k['vwap']['collapses']!r} (expected False), median |close / vwap - 1| "
        f"{k['vwap']['median_abs']} (expected between 0.005 and 0.05)"
    )
    assert k["metrics_timing"]["median_d"] < 0.001 < 0.01 < k["metrics_timing"]["median_d1"], (
        f"metrics timing: median_d {k['metrics_timing']['median_d']} (expected below 0.001), median_d1 "
        f"{k['metrics_timing']['median_d1']} (expected above 0.01)"
    )
    assert (
        0.999 < k["okx_close"]["ratio_min"] <= k["okx_close"]["ratio_max"] <= 1.0005
        and k["okx_close"]["corr_min"] > 0.998
    ), (
        f"OKX close check: ratio {k['okx_close']['ratio_min']} to {k['okx_close']['ratio_max']} (expected inside "
        f"(0.999, 1.0005]), corr_min {k['okx_close']['corr_min']} (expected above 0.998)"
    )
    assert [x["field"] for x in b["coverage"]] == list(ch["coverage"]) and len(b["coverage"]) == 16, diff_lists(
        "coverage fields (payload against the manifest, 16 fields)",
        [x["field"] for x in b["coverage"]],
        list(ch["coverage"]),
    )
    prov = man["panels"]["provenance"]
    assert "FACTORLAB_OFFLINE=0" in prov["download"] and "merge_panels.py" in prov["merge"], (
        f"provenance download {prov['download']!r} (must name FACTORLAB_OFFLINE=0), merge {prov['merge']!r} "
        "(must name merge_panels.py)"
    )
    # the page shows only the provenance fields it names
    want_prov = set(prov) & set(X.BINANCE_PROVENANCE_KEYS)
    assert set(b["meta"]["provenance"]) == set(prov) & set(X.BINANCE_PROVENANCE_KEYS), diff_sets(
        "provenance keys of the page against the manifest and BINANCE_PROVENANCE_KEYS",
        b["meta"]["provenance"],
        want_prov,
    )
    assert not MACHINE_PATHS.search(json.dumps(man)), "a machine path in the binance manifest"
    # the appends of new closed days: each starts on the day after the one before it, and the
    # last one ends on the panel's last bar; provenance.bars is the panel's whole span; the page
    # shows the appends on one line, and the first note names the panel's first and last bar
    apps = prov.get("appends") or []
    for prev, a in zip(apps, apps[1:]):
        nxt = str((pd.Timestamp(prev["last_bar"]) + pd.Timedelta(days=1)).date())
        assert a["first_bar"] == nxt, neq(
            "the first bar of an append against the day after the append before it", a["first_bar"], nxt
        )
    for a in apps:
        assert a["old_cells"] == "byte-equal" and not a["parts"].startswith("/"), (
            f"append {a.get('first_bar')} to {a.get('last_bar')}: old_cells {a.get('old_cells')!r} (must be "
            f"'byte-equal'), parts {a.get('parts')!r} (must be a relative path)"
        )
    if apps:
        assert apps[-1]["last_bar"] == prov["bars"][1], neq(
            "the last bar of the last append against the last bar of the panel", apps[-1]["last_bar"], prov["bars"][1]
        )
    want_bars = (man["panels"]["first_bar"], man["panels"]["data_through"])
    assert tuple(prov["bars"]) == want_bars, neq(
        "provenance bars against the panel span", tuple(prov["bars"]), want_bars
    )
    want = (
        "; ".join(
            f"{a['first_bar']} to {a['last_bar']} ({a['bars']} bar{'' if a['bars'] == 1 else 's'}, on "
            f"{a['appended_at'][:10]})"
            for a in apps
        )
        or None
    )
    assert b["meta"]["appended"] == want, neq("the appends line of the page", b["meta"]["appended"], want)
    lead = (
        f"Binance is a data source only: its public USD-M market data for the same 20 coins, "
        f"from {prov['bars'][0]} to {prov['bars'][1]}."
    )
    assert b["notes"][0].startswith(
        f"Binance is a data source only: its public USD-M market data for the same 20 coins, "
        f"from {prov['bars'][0]} to {prov['bars'][1]}."
    ), f"the first note does not start with {lead!r}: {b['notes'][0]!r}"
    if PANEL.is_dir() and (PANEL / "appends.jsonl").is_file():
        log = [json.loads(x) for x in (PANEL / "appends.jsonl").read_text().splitlines() if x.strip()]
        assert log == apps, "the manifest's appends differ from the panel folder's appends.jsonl"
    # the freeze of the cleaning rules
    fz = man["panels"].get("cleaning_freeze")
    if fz:
        want_fz = {"version": fz["version"], "sha256": fz["sha256"], "rules": 3}
        assert b["meta"]["cleaning_freeze"] == want_fz, diff_dicts(
            "cleaning_freeze of the page against the manifest", b["meta"]["cleaning_freeze"], want_fz
        )
        assert fz["matches_frozen_set"] is True and fz["rules"] == list(ch["cleaning"]), (
            f"matches_frozen_set {fz['matches_frozen_set']!r}; "
            + neq("frozen rules against the cleaning rules", fz["rules"], list(ch["cleaning"]))
        )
    # the archive errors set missing: the manifest's record, shown rule by rule
    cl = ch["cleaning"]
    assert [x["rule"] for x in k["cleaning"]] == list(cl) == list(X.CLEANING_LABELS), (
        f"cleaning rules: page {[x['rule'] for x in k['cleaning']]}, manifest {list(cl)}, "
        f"CLEANING_LABELS {list(X.CLEANING_LABELS)}"
    )
    for row in k["cleaning"]:
        rec = cl[row["rule"]]
        got_r = (row["cells"], row["fields"], row["label"])
        want_r = (rec["cells"], rec["fields_set_missing"], X.CLEANING_LABELS[row["rule"]])
        assert got_r == want_r, neq(
            f"cleaning rule {row['rule']} (cells, fields, label) against the manifest", got_r, want_r
        )
        want_w = X._cleaned_where(rec["where"])
        assert row["where"] == want_w, neq(
            f"cleaning rule {row['rule']}: where against the manifest", row["where"], want_w
        )
        assert rec["cells"] == len(rec["where"]), neq(
            f"cleaning rule {row['rule']}: the manifest's cells against its where list", rec["cells"], len(rec["where"])
        )
    firing = ch["cleaning_after"]["cells_firing_after_cleaning"]
    assert set(ch["cleaning_after"]["cells_firing_after_cleaning"].values()) == {0}, (
        f"cells that a rule still finds after the cleaning: {firing}"
    )
    assert k["vwap"]["inside_low_high"] == 1.0, "after cleaning every VWAP lies inside its bar's range"
    if not PANEL.is_dir():
        print("        (panel recomputation skipped: library/data/binance_um_1d is not on disk)")
        return
    p, cleared = _cleaned_frames(_panel_frames())
    assert set(p) == set(man["panels"]["fields"]) and list(p["close"].columns) == man["panels"]["symbols"], (
        diff_sets("panel fields (files against the manifest)", p, man["panels"]["fields"])
        + "; "
        + diff_lists("panel symbols (files against the manifest)", p["close"].columns, man["panels"]["symbols"])
    )
    assert cleared == {r: [(w["date"], w["symbol"]) for w in x["where"]] for r, x in cl.items()}, "cleaned cells differ"
    for x in b["coverage"]:
        firsts = p[x["field"]].apply(lambda s_: s_.first_valid_index()).dropna()
        got_x = (x["cells"], x["first_min"], x["first_max"])
        want_x = (int(p[x["field"]].notna().sum().sum()), str(min(firsts).date()), str(max(firsts).date()))
        assert got_x == want_x, neq(
            f"coverage of {x['field']} (cells, first_min, first_max) against the panel", got_x, want_x
        )
    close_ = p["close"]
    with np.errstate(all="ignore"):
        imp = p["open_interest_value"] / p["open_interest"]
        d0 = np.abs((imp / close_ - 1).to_numpy()).ravel()
        d1 = np.abs((imp / close_.shift(1) - 1).to_numpy()).ravel()
        rel = np.abs((close_ / (p["quote_volume"] / p["volume"]) - 1).to_numpy()).ravel()
    fin = np.isfinite(d0) & np.isfinite(d1)
    tm = k["metrics_timing"]
    assert tm["cells"] == int(fin.sum()), neq("metrics timing cells against the panel", tm["cells"], int(fin.sum()))
    assert np.isclose(tm["median_d"], np.median(d0[fin]), rtol=1e-9) and np.isclose(
        tm["median_d1"], np.median(d1[fin]), rtol=1e-9
    ), (
        f"metrics timing: median_d {tm['median_d']} (panel {np.median(d0[fin])}), median_d1 {tm['median_d1']} "
        f"(panel {np.median(d1[fin])})"
    )
    assert np.isclose(k["vwap"]["median_abs"], np.median(rel[np.isfinite(rel)]), rtol=1e-9), neq(
        "vwap median_abs against the panel", k["vwap"]["median_abs"], np.median(rel[np.isfinite(rel)])
    )
    assert k["vwap"]["cells"] == int(np.isfinite(rel).sum()), neq(
        "vwap cells against the panel", k["vwap"]["cells"], int(np.isfinite(rel).sum())
    )
    ratios, corrs, days, n, worst = [], [], 0, 0, 0.0
    for sym in close_.columns:
        okx = pd.read_parquet(C.DATA / "raw" / "ohlcv" / f"{sym}.parquet").set_index("ts")["close"]
        j = pd.concat([okx.rename("o"), close_[sym].rename("b")], axis=1, sort=True).dropna()
        r = np.log(j).diff().dropna()
        ratios.append(float((j["b"] / j["o"]).median()))
        corrs.append(float(r["o"].corr(r["b"])))
        days += len(j)
        fr = pd.read_parquet(C.DATA / "raw" / "funding_binanceusdm" / f"{sym}.parquet").set_index("ts")["funding_rate"]
        eng = fr.resample("1D", label="left", closed="left").sum(min_count=1).dropna()
        mine = p["funding_rate"][sym].dropna()
        common = mine.index.intersection(eng.index)
        n += len(common)
        worst = max(worst, float((mine.reindex(common) - eng.reindex(common)).abs().max()))
    ok = k["okx_close"]
    assert (ok["symbols"], ok["days"]) == (20, days), neq(
        "OKX close check (symbols, days) against the panel", (ok["symbols"], ok["days"]), (20, days)
    )
    got_close = [ok["ratio_min"], ok["ratio_max"], ok["corr_min"], ok["corr_max"]]
    want_close = [min(ratios), max(ratios), min(corrs), max(corrs)]
    assert np.allclose(
        [ok["ratio_min"], ok["ratio_max"], ok["corr_min"], ok["corr_max"]],
        [min(ratios), max(ratios), min(corrs), max(corrs)],
        rtol=1e-12,
    ), neq("OKX close check (ratio_min, ratio_max, corr_min, corr_max) against the panel", got_close, want_close)
    assert k["funding"]["day_cells"] == n and worst == k["funding"]["max_abs_diff"], (
        f"funding check: day_cells {num(k['funding']['day_cells'])} (panel {n:,}), max_abs_diff "
        f"{k['funding']['max_abs_diff']} (panel {worst})"
    )
    # the cross-venue series the page lists, cut at the last G label row, on the days both venues have
    cut = pd.Timestamp("2024-06-09", tz="UTC")
    okx = {
        f: pd.DataFrame(
            {
                sym: pd.read_parquet(C.DATA / "raw" / "ohlcv" / f"{sym}.parquet").set_index("ts")[f]
                for sym in close_.columns
            }
        )
        for f in ("open", "high", "low", "close")
    }
    oidx = pd.date_range(min(df.index.min() for df in okx.values()), cut, freq="D", tz="UTC")
    okx = {f: df.reindex(oidx).where(okx["close"].reindex(oidx).notna()) for f, df in okx.items()}
    bnc = {f: p[f].loc[:cut] for f in ("open", "high", "low", "close")}
    idx = oidx.intersection(bnc["close"].index)

    def series(d):
        return {
            "ret": d["close"] / d["close"].shift(1) - 1,
            "oc_ret": d["close"] / d["open"] - 1,
            "gap": d["open"] / d["close"].shift(1) - 1,
            "hl_range": (d["high"] - d["low"]) / d["close"],
        }

    so, sb = series(okx), series(bnc)
    rows = {r["series"]: r for r in b["crosscheck"]["series"]}
    for name in ("ret", "oc_ret", "gap", "hl_range"):
        xo, xb = so[name].reindex(idx).to_numpy().ravel(), sb[name].reindex(idx).to_numpy().ravel()
        m = np.isfinite(xo) & np.isfinite(xb)
        r = rows[name]
        assert r["cells"] == int(m.sum()), neq(
            f"series {name}: cells against the days both venues have", r["cells"], int(m.sum())
        )
        for key, x in (("median_abs_okx", xo), ("median_abs_binance", xb)):
            want = np.median(np.abs(x[m]))
            assert np.isclose(r[key], want, rtol=1e-9), neq(f"series {name}: {key} against the panels", r[key], want)
        want = round(float(np.corrcoef(xo[m], xb[m])[0, 1]), 4)
        assert r["corr"] == want, neq(f"series {name}: corr against the panels", r["corr"], want)
    assert rows["gap"]["corr"] < 0.1 and rows["ret"]["corr"] > 0.999, (
        f"cross-venue correlation: gap {rows['gap']['corr']} (expected below 0.1), ret {rows['ret']['corr']} "
        "(expected above 0.999)"
    )


def test_binance_missing_is_an_empty_state():
    with tempfile.TemporaryDirectory() as tmp:
        d = X.build(binance_dir=Path(tmp) / "binance")
        b = d["binance"]
        assert b["available"] is False and "has no computation" in b["reason"] and b["source"] == X.REL_BINANCE_DIR, (
            f"available {b['available']!r}, reason {b.get('reason')!r}, source {b.get('source')!r} "
            f"(expected {X.REL_BINANCE_DIR!r})"
        )
        cmds = "\n".join(b["commands"])
        for w in (
            "FACTORLAB_OFFLINE=0",
            "--market um --metrics-archive",
            "merge_panels.py",
            "--source binance --compute",
            "--source binance --screen",
            "export_web.py --only library",
        ):
            assert w in cmds, missing_text(w, cmds)
        assert d["engine"] == payload()["engine"], "the OKX block does not depend on the Binance folder"
        C.render_section("library", d)
        assert not list(Path(tmp).iterdir()), "the builder created a file"
        if not has_binance():
            return
        # a computation without the screen: the funnel stops at the new family, the screen blocks are null
        bd = Path(tmp) / "binance"
        bd.mkdir()
        shutil.copy2(BIN / "status.parquet", bd / "status.parquet")
        man = dict(bsrc()["man"])
        man.pop("screen")
        man.pop("crosscheck")
        (bd / "manifest.json").write_text(json.dumps(man), encoding="ascii")
        b = X.build(binance_dir=bd)["binance"]
        assert b["available"] is True and b["screen"] is None and b["top"] is None and b["crosscheck"] is None, (
            f"available {b['available']!r}, screen is None {b['screen'] is None}, top is None {b['top'] is None}, "
            f"crosscheck is None {b['crosscheck'] is None}"
        )
        assert "--screen" in b["screen_missing"] and [x["key"] for x in b["funnel"]][-1] == "new_family", (
            f"screen_missing {b.get('screen_missing')!r}; funnel keys {[x['key'] for x in b['funnel']]}"
        )


def test_page_rules_and_wiring():
    js = (WEB / "pages" / "library.js").read_text(encoding="utf-8")
    assert '"use strict"' in js and 'id: "library"' in js and 'title: "Factor library"' in js, text_check(
        "library.js", js, present=('"use strict"', 'id: "library"', 'title: "Factor library"')
    )
    for bad in ("?.", "??", "innerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert bad not in js, f"library.js uses {bad!r}"
    assert not re.search(r"(?m)^\s*(?:import|export)\s", js), "ES module syntax (blocked on file://)"
    assert 'badge("good"' not in js and "badge('good'" not in js, "no badge marks a definition as good"
    # the cards of the page, in the order of the code
    cards = re.findall(r'card\(\{ id: "([a-z-]+)"', js)
    want = [
        "families",
        "search",
        "use",
        "engine-funnel",
        "engine-coverage",
        "engine-screen",
        "engine-top",
        "binance-funnel",
        "binance-fields",
        "binance-screen",
        "binance-top",
        "binance-crosscheck",
        "binance-panel",
        "screen-summary",
    ]
    assert cards == want, diff_lists("the card ids of library.js", cards, want)
    # no link to the page of a registered test: a definition is not a test
    assert "factorHref" not in js, text_check("library.js", js, absent=("factorHref",))
    # the statement and the subtitle give numbers and the two commands
    statement = " formula definitions. Numbers from two screens on Factorlab's data. Each definition has a command that checks it and registers it for a test."
    assert (
        " formula definitions. Numbers from two screens on Factorlab's data. Each definition has a command that checks it and registers it for a test."
        in js
    ), text_check("library.js", js, present=(statement,))
    # the order of the page: the screen results and details, then the whole library
    order = [
        js.index(x)
        for x in (
            "      renderStatement(root, d, ctx.nav);",
            "      renderEngine(root, d);",
            '      library_defer(rest.body, "kpis"',
            '      library_defer(rest.body, "search"',
            '      library_defer(rest.body, "families"',
            '      library_defer(rest.body, "use"',
        )
    ]
    assert order == sorted(order), diff_lists(
        "the positions of the page parts in library.js, in page order", order, sorted(order)
    )
    # inside the screen section: the summary first, then the OKX cards, then the Binance block
    order = [
        js.index(x)
        for x in (
            "    renderScreenSummary(sec.body, d);",
            '      library_defer(sec.body, "engine-top"',
            "    renderBinance(sec.body, d);",
        )
    ]
    assert order == sorted(order), diff_lists(
        "the positions of the parts of the screen section in library.js, in page order", order, sorted(order)
    )
    # method notes are closed details in the neutral style; the callouts are the states of a computation only
    for summary in ("How to read this page",):
        assert f'summary: "{summary}"' in js, missing_text(f'summary: "{summary}"', js)
    callouts = sorted(set(re.findall(r'ui\.callout\("(\w+)", "([^"]+)"', js)))
    want_callouts = [
        ("info", "The code changed after the computation"),
        ("warning", "The OKX screen changed after this computation"),
        ("warning", "The catalog changed after this computation"),
    ]
    assert callouts == want_callouts, diff_lists("the callouts (kind, title) of library.js", callouts, want_callouts)
    assert '"Not computed on Binance\'s data"' in js, text_check(
        "library.js", js, present=('"Not computed on Binance\'s data"',)
    )
    assert "ui.badge(\"good\"" not in js, text_check("library.js", js, absent=('ui.badge("good"',))
    # page CSS: text never uses the --muted token, the controls use the shared bordered classes
    css = js[js.index("var CSS = [") : js.index('].join("\\n");')]
    assert not re.search(r"(?<![-\w])color:var\(--muted\)", css), "text in --muted"
    assert "border-strong);border-radius:8px" not in css, text_check(
        "the CSS of library.js", css, absent=("border-strong);border-radius:8px",)
    )
    assert '"class": "fl-input"' in js and '"class": "fl-select"' in js, text_check(
        "library.js", js, present=('"class": "fl-input"', '"class": "fl-select"')
    )
    # the scope filter names the data on which judge.py would test a definition (no test exists)
    scope = '["okx", "Test data: OKX"], ["binance", "Test data: Binance"]'
    assert '["okx", "Test data: OKX"], ["binance", "Test data: Binance"]' in js and "Tested on" not in js, text_check(
        "library.js", js, present=(scope,), absent=("Tested on",)
    )
    # the search index is the part file, loaded on demand; the sort covers every match; the formulas
    # are the expression parts, loaded for the shown rows (and all of them for a query)
    lazy = ('ctx.loadPart(info.part || "index")', "IntersectionObserver", "onSort:")
    assert 'ctx.loadPart(info.part || "index")' in js and "IntersectionObserver" in js and "onSort:" in js, text_check(
        "library.js", js, present=lazy
    )
    parts = ("ctx.loadPart(n)", "needParts(rows.map(function (r) { return r.i; }), terms.length > 0, library_run);")
    assert (
        "ctx.loadPart(n)" in js
        and "needParts(rows.map(function (r) { return r.i; }), terms.length > 0, library_run);" in js
    ), text_check("library.js", js, present=parts)
    assert SECTIONS[0] == "library", neq("the first section", SECTIONS[0], "library")
    app = (WEB / "assets" / "fl-app.js").read_text(encoding="utf-8")
    order = re.findall(r'\{ id: "(\w+)", title:', app)
    assert order[:2] == ["library", "research"], neq(
        "the first two routes of fl-app.js", order[:2], ["library", "research"]
    )
    assert "library:" in app and "navIcon(m.icon)" in app, text_check(
        "fl-app.js", app, present=("library:", "navIcon(m.icon)")
    )
    # index.html loads only the shell; the data and page files of a route load on demand (FL.load)
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert "assets/fl-core.js" in html and "data/meta.js" in html and "assets/fl-app.js" in html, text_check(
        "index.html", html, present=("assets/fl-core.js", "data/meta.js", "assets/fl-app.js")
    )
    assert "data/library.js" not in html and "pages/library.js" not in html, text_check(
        "index.html", html, absent=("data/library.js", "pages/library.js")
    )
    assert re.search(r'items: \[[^\]]*"library"', app), "the Factor library page is not in a nav group"


_DOM_STUB = r"""
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const WEB = process.argv[1], VARIANT = process.argv[2], SEARCHES = JSON.parse(process.argv[3] || "[]"), WDATA = process.argv[4];
class CL { constructor(n) { this.n = n; } g() { return (this.n.attrs["class"] || "").split(/\s+/).filter(Boolean); }
  s(a) { this.n.attrs["class"] = a.join(" "); } add(...c) { const a = this.g(); c.forEach(x => { if (!a.includes(x)) a.push(x); }); this.s(a); }
  remove(...c) { this.s(this.g().filter(x => !c.includes(x))); } contains(c) { return this.g().includes(c); }
  toggle(c, f) { const w = f === undefined ? !this.contains(c) : !!f; if (w) this.add(c); else this.remove(c); return w; } }
class N { constructor(t) { this.tagName = String(t).toUpperCase(); this.nodeName = this.tagName; this.children = []; this.childNodes = this.children;
    this.attrs = {}; this.style = { setProperty(k, v) { this[k] = v; }, removeProperty(k) { delete this[k]; } }; this.parentNode = null; this._t = ""; this.L = {}; this.nodeType = 1; this.classList = new CL(this); this.dataset = {}; }
  get className() { return this.attrs["class"] || ""; } set className(v) { this.attrs["class"] = v; }
  get id() { return this.attrs.id || ""; } set id(v) { this.attrs.id = v; }
  get firstChild() { return this.children[0] || null; } get lastChild() { return this.children[this.children.length - 1] || null; }
  get isConnected() { let n = this; while (n.parentNode) n = n.parentNode; return n === D.documentElement; }
  appendChild(c) { if (c.nodeType === 11) { c.children.slice().forEach(x => this.appendChild(x)); c.children.length = 0; return c; }
    if (c.parentNode) c.parentNode.removeChild(c); c.parentNode = this; this.children.push(c); return c; }
  insertBefore(c, r) { if (!r) return this.appendChild(c); if (c.parentNode) c.parentNode.removeChild(c);
    const i = this.children.indexOf(r); c.parentNode = this; this.children.splice(i < 0 ? this.children.length : i, 0, c); return c; }
  removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parentNode = null; return c; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  setAttribute(k, v) { this.attrs[k] = String(v); } getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; } hasAttribute(k) { return k in this.attrs; } setAttributeNS(_, k, v) { this.setAttribute(k, v); }
  addEventListener(t, f) { (this.L[t] = this.L[t] || []).push(f); } removeEventListener() {} dispatchEvent() { return true; }
  get textContent() { return this.nodeType === 3 ? this._t : this._t + this.children.map(c => c.textContent).join(""); }
  set textContent(v) { this.children.length = 0; this._t = v == null ? "" : String(v); }
  get innerHTML() { return ""; } set innerHTML(v) { if (v) throw new Error("innerHTML with content"); this.children.length = 0; this._t = ""; }
  get clientWidth() { return 0; } get clientHeight() { return 0; } get offsetWidth() { return 0; }
  getBoundingClientRect() { return { left: 0, top: 0, width: 0, height: 0, right: 0, bottom: 0 }; }
  m(s) { return s[0] === "#" ? this.attrs.id === s.slice(1) : s[0] === "." ? this.classList.contains(s.slice(1)) : this.tagName === s.toUpperCase(); }
  querySelectorAll(s) { const o = []; const w = n => n.children.forEach(c => { if (c.nodeType === 1) { if (c.m(s)) o.push(c); w(c); } }); w(this); return o; }
  querySelector(s) { return this.querySelectorAll(s)[0] || null; }
  closest(s) { let n = this; while (n) { if (n.nodeType === 1 && n.m(s)) return n; n = n.parentNode; } return null; }
  contains(o) { let n = o; while (n) { if (n === this) return true; n = n.parentNode; } return false; }
  focus() {} blur() {} scrollIntoView() {}
  getContext() { return { measureText: s => ({ width: String(s).length * 6.5 }), font: "" }; }
  getBBox() { return { x: 0, y: 0, width: 0, height: 0 }; } }
class T extends N { constructor(t) { super("#text"); this.nodeType = 3; this._t = String(t); } }
const D = { documentElement: new N("html"), head: new N("head"), body: new N("body"),
  createElement: t => new N(t), createElementNS: (_, t) => new N(t), createTextNode: t => new T(t),
  createDocumentFragment: () => { const f = new N("#frag"); f.nodeType = 11; return f; },
  createEvent: () => ({ initCustomEvent() {}, initEvent() {} }),
  querySelector: s => D.documentElement.querySelector(s), querySelectorAll: s => D.documentElement.querySelectorAll(s),
  getElementById: i => D.documentElement.querySelector("#" + i), addEventListener() {}, removeEventListener() {}, readyState: "complete" };
D.documentElement.appendChild(D.head); D.documentElement.appendChild(D.body);
const W = { document: D, console, setTimeout: () => 0, clearTimeout() {}, requestAnimationFrame: () => 0, cancelAnimationFrame() {},
  addEventListener() {}, removeEventListener() {}, dispatchEvent() { return true; },
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} }),
  getComputedStyle: () => ({ getPropertyValue: () => "#888888" }), localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  location: { hash: "#/library", search: "" }, history: { replaceState() {} }, navigator: { userAgent: "node" },
  ResizeObserver: class { observe() {} unobserve() {} disconnect() {} },
  CustomEvent: class { constructor(t, o) { this.type = t; this.detail = o && o.detail; } }, Event: class { constructor(t) { this.type = t; } },
  URLSearchParams };
W.window = W; W.self = W;
const C = vm.createContext(W);
// the expression parts load through ctx.loadPart (below); the other files load here
const PARTS = {};
for (const f of fs.readdirSync(WDATA).filter(x => /^library\.expr-\d+\.js$/.test(x))) PARTS[f.slice(8, -3)] = fs.readFileSync(path.join(WDATA, f), "utf8");
for (const f of ["assets/fl-core.js", "assets/fl-charts.js", "data/meta.js", "data/library.js", "data/library.index.js", "pages/library.js"]) {
  const full = f.startsWith("data/") ? path.join(WDATA, f.slice(5)) : path.join(WEB, f);
  if (f === "data/library.index.js" && !fs.existsSync(full)) continue;
  vm.runInContext(fs.readFileSync(full, "utf8"), C, { filename: f });
}
const loadPart = n => {
  if (VARIANT === "noexpr" || !PARTS[n]) return Promise.reject({ state: "missing" });
  vm.runInContext(PARTS[n], C, { filename: "data/library." + n + ".js" });
  return Promise.resolve(W.FL.data["library." + n]);
};
const FL = W.FL;
if (VARIANT === "empty") FL.data.library = {};
if (VARIANT === "unavailable") FL.data.library = { schema: 1, available: false, reason: "library/catalog/factors.db is not built" };
if (VARIANT === "noindex") delete FL.data["library.index"];
if (VARIANT === "partial") { const d = FL.data.library; delete FL.data["library.index"]; delete d.index_info; delete d.commands; delete d.families;
  const e = d.engine || {}; delete e.top; delete e.screen; e.funnel = []; e.coverage = []; delete e.fields; delete e.zones;
  const b = d.binance || {}; delete b.top; delete b.screen; delete b.crosscheck; delete b.fields; delete b.by_origin; b.funnel = [];
  delete b.coverage; delete b.checks; delete b.meta; delete b.zones; }
if (VARIANT === "nobinance") FL.data.library.binance = { available: false, reason: "engine/data/library_on_engine/binance has no computation",
  commands: ["cd library", "python -B scripts/merge_panels.py", "cd ../engine", "python -B libfactors.py --source binance --compute"] };
if (VARIANT === "binanceless") delete FL.data.library.binance;
if (VARIANT === "codechanged") { const e = FL.data.library.engine || {}, b = FL.data.library.binance || {};
  if (e.meta) { e.meta.code_current = false; e.meta.code_changed = ["engine/libfactors.py"]; }
  if (b.meta) { b.meta.code_current = false; b.meta.code_changed = ["engine/libfactors.py"]; b.meta.okx_current = false; } }
if (VARIANT === "binanceonly") FL.data.library.engine = { available: false, reason: "engine/data/library_on_engine has no computation",
  commands: ["cd engine", "python -B libfactors.py --compute"] };
if (VARIANT === "noengine") FL.data.library.engine = { available: false, reason: "engine/data/library_on_engine has no computation",
  commands: ["cd engine", "python -B libfactors.py --compute"] };
if (VARIANT === "engineless") delete FL.data.library.engine;
const root = new N("div"); D.body.appendChild(root);
FL.pages.get("library").render(root, { id: "library", data: FL.data, section: FL.data.library, meta: FL.data.meta, colors: FL.colors,
  fmt: FL.fmt, ui: FL.ui, charts: FL.charts, params: new URLSearchParams(""), loadPart,
  nav: { current: "library", card: null, href: (p, c) => "#/" + p + (c ? "/" + c : ""), go() {} } });
const flush = async () => { for (let k = 0; k < 6; k++) await new Promise(r => setImmediate(r)); };
(async () => {
await flush();
root.querySelectorAll(".fl-tab").forEach(b => (b.L.click || []).forEach(f => f({})));
root.querySelectorAll(".fl-chart-toggle").forEach(b => (b.L.click || []).forEach(f => f({})));
// open every details, also the ones that an opened details renders
for (let k = 0; k < 4; k++) root.querySelectorAll("details").forEach(d => { if (d.open) return; (d.L.toggle || []).forEach(f => { d.open = true; f({}); }); d.open = true; });
const counts = [];
const searchCard = root.querySelector("#search") || root;
const input = searchCard.querySelectorAll("input")[0], scope = searchCard.querySelectorAll("select")[0];
const countEl = searchCard.querySelectorAll(".library-count")[0];
const rowsNow = () => { const t = root.querySelectorAll("#search")[0]; return t ? t.querySelectorAll("tr").length - 1 : null; };
let sorted = null;
let firstRows = null;
if (input && scope) {
  counts.push([countEl.textContent, rowsNow()]);
  // the first rows with their formulas (the expression part of the shown rows) and commands
  const tb0 = searchCard.querySelectorAll("tbody")[0];
  firstRows = tb0 ? tb0.querySelectorAll("tr").slice(0, 2).map(tr => tr.textContent) : null;
  for (const [q, sc] of SEARCHES) {
    input.value = q; scope.value = sc; (scope.L.change || []).forEach(f => f({}));
    await flush();   // a query loads every expression part, then the search runs again
    counts.push([countEl.textContent, rowsNow()]);
  }
  // a header click sorts every match; the first row is the largest value of all of them
  input.value = ""; scope.value = "all"; (scope.L.change || []).forEach(f => f({}));
  await flush();
  const btn = searchCard.querySelectorAll("button").find(b => b.textContent === "screen on Factorlab's data");
  if (btn) {
    // the first click sorts this text column from the smallest value; the second from the largest
    (btn.L.click || []).forEach(f => f({}));
    (btn.L.click || []).forEach(f => f({}));
    await flush();
    const body = searchCard.querySelectorAll("tbody")[0], first = body ? body.querySelectorAll("tr")[0] : null;
    sorted = [countEl.textContent, first ? first.children.map(td => td.textContent) : null];
  }
}
// the filters of the two screen tables: the state at the start, then after each click on the radios in `order`
const filters = (id, order) => {
  const out = [], top = root.querySelector("#" + id);
  if (!top || !top.querySelectorAll(".library-count").length) return out;
  const radios = top.querySelectorAll("button").filter(b => b.attrs.role === "radio");
  const state = () => [top.querySelectorAll(".library-count")[0].textContent, top.querySelectorAll("tbody")[0].querySelectorAll("tr").length];
  out.push(["start"].concat(state()));
  for (const i of order) { if (!radios[i]) break; (radios[i].L.click || []).forEach(f => f({})); out.push([radios[i].textContent].concat(state())); }
  return out;
};
const engine = filters("engine-top", [3, 1, 0, 2]), binance = filters("binance-top", [3, 0, 1, 2]);
const ids = []; const walk = n => n.children.forEach(c => { if (c.nodeType === 1) { if (c.attrs.id) ids.push(c.attrs.id); walk(c); } }); walk(root);
process.stdout.write(JSON.stringify({ text: root.textContent, ids, counts, engine, binance, sorted, firstRows, styles: D.head.children.length }));
})().catch(e => { console.error(e && e.stack || e); process.exit(1); });
"""


def _count(n: int, one: str) -> str:
    """The count text of the page: '1 factor', '2,000 factors'."""
    return f"{n:,} {one if n == 1 else one + 's'}"


def _one_per_signal(n: int) -> str:
    """The count text of the view "one per distinct signal"."""
    return f"{n:,} distinct signal, one row" if n == 1 else f"{n:,} distinct signals, one row each"


def _node() -> str | None:
    import os
    import shutil

    n = os.environ.get("FACTORLAB_NODE") or os.environ.get("NODE") or shutil.which("node")
    return n if n and Path(n).exists() else None


def _render(variant: str, searches=()) -> dict:
    import subprocess

    r = subprocess.run(
        [_node(), "-e", _DOM_STUB, str(WEB), variant, json.dumps(list(searches)), str(WEB_DATA)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, f"page render ({variant}) failed: {r.stderr[-1200:]}"
    return json.loads(r.stdout)


def _canonical() -> list[str]:
    """The canonical text of each definition (ast.unparse of the parsed catalog text)."""
    return [ast.unparse(ast.parse(e, mode="eval")) for e in src()["fac"]["expression"]]


def _expected_matches(query: str, scope: str) -> int:
    """The page's search rule, recomputed from the catalog CSV: every space-separated term must
    occur in id + name + family + category + formula, case-insensitively; the scope is all, the
    definitions tested on OKX data or those tested on Binance data."""
    fac = src()["fac"]
    okx = okx_source(fac)
    exprs = _canonical()
    terms = query.lower().split()
    n = 0
    for r, o, e in zip(fac.itertuples(), okx, exprs):
        if (scope == "okx" and not o) or (scope == "binance" and o):
            continue
        hay = " ".join([r.factor_id, r.name, r.family, r.category, e]).lower()
        n += all(t in hay for t in terms)
    return n


PAGE = 25  # the page size of the long tables (ui.table pageSize)


def test_page_renders():
    """Run the page in node against the exported files: the full page with searches (the formulas
    come from the expression parts), a payload without the catalog, an empty one and a partial
    one. Needs node."""
    if _node() is None:
        print("        (skipped: node not found; set NODE or FACTORLAB_NODE)")
        return
    if not (WEB_DATA / "library.js").exists():
        print(f"        (skipped: {WEB_DATA}/library.js is not exported)")
        return
    d = payload()
    searches = [
        ("funding", "all"),
        ("cf00001", "all"),
        ("ts_std ret", "all"),
        ("decay_linear funding", "binance"),
        ("", "okx"),
        ("", "binance"),
        ("zzzz-no-such-thing", "all"),
    ]
    full = _render("full", searches)
    t = full["text"]
    tot = d["totals"]
    for card in ("statement", "kpis", "search", "families", "use", "browse"):
        assert card in full["ids"], f"the page has no element with the id {card!r}"
    assert num(tot["definitions"]) in t and "stat.std.ex_btc_ret.w60" in t, text_check(
        "the page", t, present=(num(tot["definitions"]), "stat.std.ex_btc_ret.w60")
    )
    lead = (
        f"{num(tot['definitions'])} formula definitions. Numbers from two screens on Factorlab's data. "
        "Each definition has a command that checks it and registers it for a test."
    )
    assert lead in t, missing_text(lead, t)
    assert "System page, Glossary" in t and "Computable on OKX data" in t and "Computable on Binance data" in t, (
        text_check(
            "the page", t, present=("System page, Glossary", "Computable on OKX data", "Computable on Binance data")
        )
    )
    assert "✓" not in t, "a 'good' badge was drawn"
    assert full["styles"] == 1, neq("style elements in the head", full["styles"], 1)
    # the first rows show their formula (the part expr-0) and their two commands
    fac = src()["fac"]
    exprs = _canonical()
    for i, row in enumerate(full["firstRows"]):
        fid = fac["factor_id"].iloc[i]
        assert row.startswith(fid) and exprs[i] in row, (
            f"row {i} of the search table does not start with {fid!r} or has not the formula {exprs[i]!r}: {row!r}"
        )
        assert CHECK.format(id=fid) in row and REGISTER.format(id=fid) in row, text_check(
            f"row {i} ({fid}) of the search table", row, present=(CHECK.format(id=fid), REGISTER.format(id=fid))
        )
    want = [_expected_matches("", "all")] + [_expected_matches(q, sc) for q, sc in searches]
    got = []
    for text, nrows in full["counts"]:
        m = re.match(r"([\d,]+) match", text)
        assert m, f"the count text of the search does not start with '<n> match': {text!r}"
        n = int(m.group(1).replace(",", ""))
        got.append(n)
        # an empty table shows one "no match" row
        assert nrows == min(n, PAGE) or (n == 0 and nrows == 1), (
            f"search count {text!r}: {nrows} table rows, expected {min(n, PAGE) if n else '1 (the no-match row)'}"
        )
    assert got == want, diff_lists("search match counts of the page against the catalog CSV", got, want)
    assert got[0] == tot["definitions"] and got[-1] == 0, (
        f"matches of the empty search {got[0]:,} (definitions {num(tot['definitions'])}); matches of the last search "
        f"{got[-1]:,} (expected 0)"
    )
    assert got[5] + got[6] == tot["definitions"] and got[5] == tot["sources"][0], (
        f"matches with the scopes OKX {got[5]:,} and Binance {got[6]:,} (sum {got[5] + got[6]:,}, definitions "
        f"{num(tot['definitions'])}); OKX source count {num(tot['sources'][0])}"
    )
    # a header sort orders every match before the cut: the first row has the largest |rIC t| of all
    scr = {}
    for blk in ("engine", "binance"):
        top = (d.get(blk) or {}).get("top")
        if top:
            for fid, tt in zip(top["id"], top["ric_t"]):
                scr.setdefault(fid, {})[blk] = tt
    if scr:

        def key(fid):
            x = scr[fid]
            return abs(x["engine"] if "engine" in x else x["binance"])

        # the largest |rIC t| first; a tie keeps the id order
        best = min(scr, key=lambda fid: (-key(fid), fid))
        cnt, first = full["sorted"]
        assert first and first[0] == best, neq(
            "the first row after the sort (its cells) against the id with the largest |rIC t|", first, best
        )
        assert cnt.endswith("by screen on Factorlab's data, largest first"), (
            f"the count text after the sort does not end with 'by screen on Factorlab's data, largest first': {cnt!r}"
        )
    # the order of the page: statement, contents, screen results and details, whole library
    ids = full["ids"]
    assert ids.index("statement") < ids.index("engine") < ids.index("browse") < ids.index("search"), positions(
        "page order", ids, ("statement", "engine", "browse", "search")
    )
    assert "screen-summary" in ids and ids.index("engine") < ids.index("screen-summary"), positions(
        "page order", ids, ("engine", "screen-summary")
    )
    assert "Forward test" not in t and "in the forward test" not in t, text_check(
        "the page", t, absent=("Forward test", "in the forward test")
    )
    for x in ("Screen results", "OKX screen", "Binance data", "Search"):
        assert x in t, missing_text(x, t)
    assert not _word_hits([t]), f"verdict words in the page text: {shown(VERDICT_WORDS.findall(t))}"
    # the library on Factorlab's data: its cards, its method notes and the table filters
    e = d["engine"]
    if has_engine():
        for card in (
            "engine",
            "engine-notes",
            "engine-funnel",
            "engine-coverage",
            "engine-screen",
            "engine-kpis",
            "engine-top",
            "browse",
        ):
            assert card in ids, f"the page has no element with the id {card!r}"
        assert (
            ids.index("screen-summary") < ids.index("engine-funnel") < ids.index("engine-top") < ids.index("browse")
        ), positions("page order", ids, ("screen-summary", "engine-funnel", "engine-top", "browse"))
        for line in e["notes"]:
            assert line in t, missing_text(line, t)
        n = e["counts"]
        for v in (
            n["computable"],
            n["pit_pass"],
            n["screened"],
            n["needs_premium"],
            n["missing_fields"],
            n["oi"],
            n["vwap_proxy"],
        ):
            assert num(v) in t, missing_text(num(v), t)
        for s_ in (
            "Premium index",
            "Trades, taker flow or long/short ratios",
            "Open interest: computable, short history",
            f"stored only from {e['meta']['oi_from']}",
            "computed, not screened",
            "not computed",
            "taker_buy_volume",
            "residual IC (rIC)",
            "no zoo_pc1 regressor",
            "rIC q ≤ 0.05 over all of G",
            "Selected on G_fit, same sign on G_eval",
            "G_eval is part of this selection",
            "How much the rIC settings move the counts",
            "rIC q G_fit",
            "n/(n+180)",
            "equal up to floating-point rounding (rtol 1e-9, atol 1e-12)",
            "{} and {}".format(", ".join(e["pit"]["cuts"][:-1]), e["pit"]["cuts"][-1]),
            "signals touched",
            "in the family of its representative",
            "stamped 16:00 UTC",
            "float64",
            "OKX open-interest statistics",
            "G_fit 69",
            "G_eval 45",
            "I sealed 117",
            "Definitions with rIC q ≤ 0.10, by |rIC t|",
            "To test a definition in a book, use its register command.",
        ):
            assert s_ in t, missing_text(s_, t)
        assert "rubik" not in t, text_check("the page", t, absent=("rubik",))
        # each row of the table has the two commands of its definition
        for fid in e["top"]["id"][:3]:
            assert CHECK.format(id=fid) in t and REGISTER.format(id=fid) in t, text_check(
                f"the OKX table row {fid}", t, present=(CHECK.format(id=fid), REGISTER.format(id=fid))
            )
        # when the code changed after the computation, the page says so (read-only; the numbers are the recorded ones)
        stale = not e["meta"]["code_current"] or (has_binance() and not d["binance"]["meta"]["code_current"])
        assert ("The code changed after the computation" in t) == stale, (
            f"the code changed after the computation: {stale}; the page says so: "
            f"{'The code changed after the computation' in t}"
        )
        if not e["meta"]["code_current"]:
            changed = "The manifest's code hash differs from today's " + ", ".join(e["meta"]["code_changed"])
            assert ("The manifest's code hash differs from today's " + ", ".join(e["meta"]["code_changed"])) in t, (
                missing_text(changed, t)
            )
        # the filters: start (q <= 0.05, one per signal), then one per signal, q <= 0.10, q <= 0.05, every factor;
        # the table shows the first PAGE rows of each view
        g = esrc()["g"]
        q05, q10 = g["ric_q"] <= 0.05, g["ric_q"] <= 0.10
        n05, s05, s10 = int(q05.sum()), int(g.loc[q05, "ric_rep_q05"].sum()), int(g.loc[q10, "ric_rep_q10"].sum())
        got = full["engine"]
        want_rows = [min(x, PAGE) for x in (s05, s05, s10, s05, n05)]
        assert [r[2] for r in got] == want_rows, diff_lists(
            "the table rows of each OKX filter view against screen_g.parquet", [r[2] for r in got], want_rows
        )
        want0 = f"{_one_per_signal(s05)} (the member with the largest |rIC t|) at q ≤ 0.05"
        assert got[0][1] == want0, neq("the count text of the first OKX view", got[0][1], want0)
        assert got[3][1].startswith(_one_per_signal(s05)) and got[3][1].endswith("q ≤ 0.05"), (
            f"the count text of the fourth OKX view {got[3][1]!r} does not start with "
            f"{_one_per_signal(s05)!r} and end with 'q ≤ 0.05'"
        )
        want4 = f"{_count(n05, 'factor')} in {_count(s05, 'distinct signal')} at q ≤ 0.05"
        assert got[4][1] == want4, neq("the count text of the last OKX view", got[4][1], want4)
        want_filters = [
            "one per distinct signal",
            f"q ≤ 0.10 ({int(q10.sum()):,})",
            f"q ≤ 0.05 ({n05:,})",
            "every factor",
        ]
        assert [r[0] for r in got[1:]] == [
            "one per distinct signal",
            f"q ≤ 0.10 ({int(q10.sum()):,})",
            f"q ≤ 0.05 ({n05:,})",
            "every factor",
        ], diff_lists("the filter labels of the OKX table", [r[0] for r in got[1:]], want_filters)
        miss = _render("noengine")
        assert "Not computed on Factorlab's data" in miss["text"] and "libfactors.py --compute" in miss["text"], (
            text_check(
                "the page without the engine block",
                miss["text"],
                present=("Not computed on Factorlab's data", "libfactors.py --compute"),
            )
        )
        assert "engine-top" not in miss["ids"] and "search" in miss["ids"] and "screen-summary" in miss["ids"], (
            f"the cards of the page without the engine block: {miss['ids']} (no engine-top; search and screen-summary)"
        )
    # the same library on Binance's data: a separate block at the end of the section
    if has_binance():
        b = d["binance"]
        for card in (
            "binance",
            "binance-notes",
            "binance-funnel",
            "binance-fields",
            "binance-screen",
            "binance-kpis",
            "binance-top",
            "binance-crosscheck",
            "binance-cross-kpis",
            "binance-panel",
            "binance-terms",
        ):
            assert card in ids, f"the page has no element with the id {card!r}"
        assert ids.index("engine-top") < ids.index("binance") < ids.index("binance-panel") < ids.index("browse"), (
            positions("page order", ids, ("engine-top", "binance", "binance-panel", "browse"))
        )
        assert "Binance data for the same 20 coins" in t, missing_text("Binance data for the same 20 coins", t)
        assert "Binance Vision archive (data.binance.vision), CC BY-NC-SA 4.0" in t, missing_text(
            "Binance Vision archive (data.binance.vision), CC BY-NC-SA 4.0", t
        )
        for line in b["notes"]:
            assert line in t, missing_text(line, t)
        n, bm, bc = b["counts"], b["meta"], b["checks"]["funding"]
        for v in (n["new_family"], n["new_screened"], n["crosscheck"], n["taker_last_5m"]):
            assert num(v) in t, missing_text(num(v), t)
        for s_ in (
            "Binance is a data source only",
            "From definitions to screened factors, on Binance's data",
            "Screened on OKX: cross-checked only",
            "New family: not screened on OKX",
            "What the new fields unlock",
            "Why the OKX screen could not screen them",
            "taker_buy_quote_volume",
            "the ratio of those 5 minutes, not of the whole day",
            "Fields the engine's OKX data does not have",
            "rIC q ≤ 0.05 over all of G",
            "Benjamini-Hochberg q-values (new family only)",
            "it adds no test to the OKX screen's family",
            "New-family definitions with rIC q ≤ 0.10, by |rIC t|",
            "Cross-venue agreement of the OKX-screened definitions",
            "Agreement numbers only: no q-value on Binance",
            "The largest disagreements",
            "is not an independent replication",
            "With the same labels and regressors and closely correlated values, agreement is expected",
            "The Binance panel: provenance, point in time, checks",
            f"{num(bm['bars'])} daily bars, {bm['first_bar']} to {bm['data_through']}",
            f"on {num(bc['day_cells'])} coin-days (0 differ by more than 1e-12",
            "Archive errors set missing",
            "a real VWAP",
            "the snapshot is the bar's last",
            ("New days" if bm["appended"] else "none: the panel holds the first download only"),
            "Binance new family",
            "Binance cross-check",
        ):
            assert s_ in t, missing_text(s_, t)
        for fid in b["top"]["id"][:3]:
            assert CHECK.format(id=fid) in t and REGISTER.format(id=fid) in t, text_check(
                f"the Binance table row {fid}", t, present=(CHECK.format(id=fid), REGISTER.format(id=fid))
            )
        if bm["appended"]:
            appended = bm["appended"] + ". merge_panels.py --append added them, and no old cell changed"
            assert bm["appended"] + ". merge_panels.py --append added them, and no old cell changed" in t, missing_text(
                appended, t
            )
        if bm["cleaning_freeze"]:
            freeze = (
                f"CLEANING_FREEZE pins the 3 rules, version {bm['cleaning_freeze']['version']} "
                f"(sha256 {bm['cleaning_freeze']['sha256'][:12]}"
            )
            assert (
                f"CLEANING_FREEZE pins the 3 rules, version {bm['cleaning_freeze']['version']} "
                f"(sha256 {bm['cleaning_freeze']['sha256'][:12]}"
            ) in t, missing_text(freeze, t)
        assert "Pre-2022" not in t, text_check("the page", t, absent=("Pre-2022",))
        # filters: the same opening view as the OKX table (q <= 0.05, one per signal), then one per
        # signal, q <= 0.05, q <= 0.10, every factor
        g = bsrc()["g"]
        q05, q10 = g["ric_q"] <= 0.05, g["ric_q"] <= 0.10
        n10, s05, s10 = int(q10.sum()), int(g.loc[q05, "ric_rep_q05"].sum()), int(g.loc[q10, "ric_rep_q10"].sum())
        got = full["binance"]
        want_rows = [min(x, PAGE) for x in (s05, s05, s05, s10, n10)]
        assert [r[2] for r in got] == want_rows, diff_lists(
            "the table rows of each Binance filter view against screen_g.parquet", [r[2] for r in got], want_rows
        )
        want0 = f"{_one_per_signal(s05)} (the member with the largest |rIC t|) at q ≤ 0.05"
        assert got[0][1] == want0, neq("the count text of the first Binance view", got[0][1], want0)
        want_filters = [
            "one per distinct signal",
            f"q ≤ 0.05 ({int(q05.sum()):,})",
            f"q ≤ 0.10 ({n10:,})",
            "every factor",
        ]
        assert [r[0] for r in got[1:]] == [
            "one per distinct signal",
            f"q ≤ 0.05 ({int(q05.sum()):,})",
            f"q ≤ 0.10 ({n10:,})",
            "every factor",
        ], diff_lists("the filter labels of the Binance table", [r[0] for r in got[1:]], want_filters)
        want4 = f"{_count(n10, 'factor')} in {_count(s10, 'distinct signal')} at q ≤ 0.10"
        assert got[4][1] == want4, neq("the count text of the last Binance view", got[4][1], want4)
        miss = _render("nobinance")
        assert "Not computed on Binance's data" in miss["text"] and "--source binance --compute" in miss["text"], (
            text_check(
                "the page without the Binance block",
                miss["text"],
                present=("Not computed on Binance's data", "--source binance --compute"),
            )
        )
        assert "binance" in miss["ids"] and "binance-top" not in miss["ids"] and "engine-top" in miss["ids"], (
            f"the cards of the page without the Binance block: {miss['ids']} (binance and engine-top; no binance-top)"
        )
        only = _render("binanceonly")
        assert "Not computed on Factorlab's data" in only["text"] and "binance-top" in only["ids"], (
            text_check(
                "the page with the Binance block only", only["text"], present=("Not computed on Factorlab's data",)
            )
            + f"; cards {only['ids']} (binance-top must be one)"
        )
    gone = _render("unavailable")
    assert "No data" in gone["text"] and "./setup.sh" in gone["text"], text_check(
        "the page without data", gone["text"], present=("No data", "./setup.sh")
    )
    noix = _render("noindex")
    assert "data/library.index.js is not in this export." in noix["text"] and "Search not available" in noix["text"], (
        text_check(
            "the page without the index part",
            noix["text"],
            present=("data/library.index.js is not in this export.", "Search not available"),
        )
    )
    # without the expression parts each row names the show command of the library CLI
    noex = _render("noexpr")
    no_expr = "formula: not loaded (or: cd library && ../engine/.venv/bin/python -m factorlab show CF00001)"
    assert (
        "formula: not loaded (or: cd library && ../engine/.venv/bin/python -m factorlab show CF00001)"
        in noex["firstRows"][0]
    ), missing_text(no_expr, noex["firstRows"][0])
    for variant in ("empty", "partial", "engineless", "binanceless"):
        _render(variant)  # must not throw


def test_page_says_when_the_code_or_the_okx_screen_changed():
    """The page with a computation that the code on disk did not make: each screen shows the callout
    "The code changed after the computation" with the file. The Binance screen also shows the
    callout of an OKX screen that changed after its computation. Needs node."""
    if _node() is None:
        print("        (skipped: node not found; set NODE or FACTORLAB_NODE)")
        return
    if not (WEB_DATA / "library.js").exists():
        print(f"        (skipped: {WEB_DATA}/library.js is not exported)")
        return
    d = payload()
    engine = bool((d.get("engine") or {}).get("meta"))
    binance = bool((d.get("binance") or {}).get("meta"))
    if not engine and not binance:
        print("        (skipped: no screen computation in the export)")
        return
    t = _render("codechanged")["text"]
    title = "The code changed after the computation"
    assert t.count(title) == engine + binance, neq(f"the count of {title!r}", t.count(title), engine + binance)
    texts = []
    if engine:
        texts.append(
            title + "The manifest's code hash differs from today's engine/libfactors.py. Run libfactors.py "
            "--compute and --screen to compute these numbers with the code on disk."
        )
    if binance:
        texts.append(
            title + "The manifest's code hash differs from today's engine/libfactors.py. Run libfactors.py "
            "--source binance --compute and --screen to compute these numbers with the code on disk."
        )
        texts.append(
            "The OKX screen changed after this computationThe new family and the cross-check use the OKX screen "
            "of that computation: run libfactors.py --source binance --compute and --screen again."
        )
    for text in texts:
        assert text in t, missing_text(text, t)
    full = _render("full")["text"]
    okx_stale = binance and d["binance"]["meta"].get("okx_current") is False
    assert ("The OKX screen changed after this computation" in full) == okx_stale, (
        f"okx_current is False: {okx_stale}; the page has the callout: "
        f"{'The OKX screen changed after this computation' in full}"
    )


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Run it with no arguments. It prints one PASS, FAIL or SKIP line for each test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    if not (CAT / "factors.db").is_file():
        print("SKIP  export_library: library/catalog/factors.db is not built (python -m factorlab build)")
        raise SystemExit(0)
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} export_library tests passed")
    raise SystemExit(1 if failed else 0)
