"""The factor library catalog for the Factor library page (webexport/library.py).

It reads ``library/catalog/factors.db`` read-only (SQLite ``mode=ro``) and gives the totals,
the families, the categories, the search index and the expression parts of the page.
webexport.library.build() calls these functions; the payload is described there. The page
shows numbers and definitions only.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pandas as pd

import userfactor
from factorlab.expr import parse
from webexport.common import ENGINE_ROOT, rnd

CATALOG = ENGINE_ROOT.parent / "library" / "catalog"
REL_DB = "library/catalog/factors.db"
REL_README = "library/README.md"
# the raw fields of plain daily price data: close and dollar volume
FREE_FIELDS = frozenset({"close", "quote_volume"})
DERIV_FIELDS = frozenset(
    {
        "funding_rate",
        "open_interest",
        "open_interest_value",
        "long_short_ratio",
        "top_long_short_ratio",
        "taker_buy_sell_ratio",
        "premium",
    }
)
REQUIREMENTS = [
    {"key": "free", "label": "Close and dollar volume only", "note": "the fields of plain daily price data"},
    {"key": "kline", "label": "Other kline fields", "note": "OHLC, trade counts or taker flow"},
    {
        "key": "derivs",
        "label": "Perpetual-futures data",
        "note": "funding, open interest, long/short ratios or premium",
    },
]
# the data source of each definition for a test (userfactor.auto_source): 0 OKX, 1 Binance
SOURCES = ("okx", "binance")
# the expression parts of the search: this many definitions in each part, in index order
EXPR_CHUNK = 8000
# one line per family (the library README has the long version)
FAMILY_IDEAS = {
    "stat_rolling": "rolling statistics (mean, std, z-score, rank, skew, slope, ...) of stationary series",
    "correlation": "rolling correlations between series, price-volume correlation, autocorrelation",
    "interaction": "products, spreads and sign-gates of z-scores from different information sets",
    "second_order": "z-score, rank or change of smoothed statistics",
    "lead_lag": "correlation of a lagged series with another series today",
    "ma_crossover": "short/long SMA and EMA crossovers",
    "level_transform": "momentum, MA distance, range position, trend and drawdown of levels",
    "conditional": "averages of a series on up/down, high/low-volume or funding-sign bars",
    "technical": "RSI, stochastics, ADX, range-based volatility, spread and microstructure estimators",
    "momentum": "skip-period, risk-adjusted, excess and volume-weighted momentum",
    "market_beta": "beta, up/down beta, idiosyncratic volatility, residual momentum",
    "crypto_derivatives": "funding carry, OI/price divergence, leverage, basis, positioning",
    "alpha101_style": "crypto adaptations of the Alpha101 formulas",
}
T_DP = 2


def _connect(db: Path) -> sqlite3.Connection:
    """Read-only connection: mode=ro makes SQLite refuse every write and never create the file."""
    return sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)


def _q(con: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, con)


def front_code(strings: list[str]) -> tuple[list[int], list[str]]:
    """(shared-prefix lengths, suffixes) with name[i] = name[i-1][:p[i]] + s[i]."""
    p, s, prev = [], [], ""
    for x in strings:
        k, m = 0, min(len(x), len(prev))
        while k < m and x[k] == prev[k]:
            k += 1
        p.append(k)
        s.append(x[k:])
        prev = x
    return p, s


def front_decode(p: list[int], s: list[str]) -> list[str]:
    out, prev = [], ""
    for k, suf in zip(p, s):
        prev = prev[:k] + suf
        out.append(prev)
    return out


def _runs(codes: list[int]) -> list[list[int]]:
    """[[code, start, count], ...] for consecutive equal codes."""
    out: list[list[int]] = []
    for i, c in enumerate(codes):
        if out and out[-1][0] == c:
            out[-1][2] += 1
        else:
            out.append([c, i, 1])
    return out


def requirement(raw_inputs: str) -> int:
    """0 = close + dollar volume only, 1 = other kline fields, 2 = needs perp derivatives data."""
    f = set(raw_inputs.split(","))
    if f <= FREE_FIELDS:
        return 0
    return 2 if f & DERIV_FIELDS else 1


def _t(x):
    return rnd(x, T_DP)


def _families(fac: pd.DataFrame) -> list[dict]:
    """Per family: the definitions and their data needs (REQUIREMENTS order)."""
    req = fac["raw_inputs"].map(requirement)
    rows = []
    for fam, n in fac["family"].value_counts().sort_index().items():
        r = req[fac["family"] == fam]
        rows.append(
            {
                "family": fam,
                "idea": FAMILY_IDEAS.get(fam, ""),
                "n": int(n),
                "requirements": [int((r == i).sum()) for i in range(len(REQUIREMENTS))],
            }
        )
    return sorted(rows, key=lambda x: (-x["n"], x["family"]))


def _categories(fac: pd.DataFrame) -> list[dict]:
    """Per category: the definitions and their data needs (REQUIREMENTS order)."""
    req = fac["raw_inputs"].map(requirement)
    rows = []
    for cat, n in fac["category"].value_counts().sort_index().items():
        r = req[fac["category"] == cat]
        rows.append(
            {"category": cat, "n": int(n), "requirements": [int((r == i).sum()) for i in range(len(REQUIREMENTS))]}
        )
    return sorted(rows, key=lambda x: (-x["n"], x["category"]))


def canonical(expr: str) -> str:
    """The canonical text of a library expression: ast.unparse of the parsed tree, the text that
    judge.py records for the formula (userfactor.validate_formula)."""
    return ast.unparse(parse(expr))


def sources(fac: pd.DataFrame) -> list[int]:
    """The data source of each definition for a test (userfactor.auto_source; SOURCES index)."""
    return [SOURCES.index(userfactor.auto_source(e)) for e in fac["expression"]]


def expr_parts(fac: pd.DataFrame) -> dict[str, dict]:
    """The expression parts: {"expr-0": {"start", "expr"}, ...}, EXPR_CHUNK definitions each, in
    index order; "start" is the index of the first definition of the part."""
    exprs = [canonical(e) for e in fac["expression"]]
    return {
        f"expr-{k}": {"start": i, "n": len(exprs[i : i + EXPR_CHUNK]), "expr": exprs[i : i + EXPR_CHUNK]}
        for k, i in enumerate(range(0, len(exprs), EXPR_CHUNK))
    }


def _index(fac: pd.DataFrame, families: list[str], categories: list[str], src: list[int]) -> dict:
    fcode = {f: i for i, f in enumerate(families)}
    ccode = {c: i for i, c in enumerate(categories)}
    p, s = front_code(fac["name"].tolist())
    return {
        "families": families,
        "categories": categories,
        "requirements": REQUIREMENTS,
        "sources": list(SOURCES),
        "all": {
            "n": len(fac),
            "id_prefix": "CF",
            "id_digits": 5,
            "id_start": 1,
            "p": p,
            "s": s,
            "fam_runs": _runs([fcode[f] for f in fac["family"]]),
            "cat": [ccode[c] for c in fac["category"]],
            "req": [requirement(x) for x in fac["raw_inputs"]],
            "source": src,  # SOURCES index of each definition (0 = okx, 1 = binance)
        },
    }
