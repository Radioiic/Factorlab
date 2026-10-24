"""Build the factor database: generate -> validate -> de-duplicate -> store.

Validation evaluates every candidate expression on a validation panel
(synthetic by default, or your own downloaded data) and rejects factors that

* raise an error,
* have too little coverage (mostly NaN),
* have no cross-sectional dispersion (identical for every coin), or
* are *rank-equivalent* to an earlier factor (identical cross-sectional
  ranks, or exactly reversed ranks, on every validation bar).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .database import FactorDB
from .expr import Evaluator, max_lookback, raw_dependencies
from .fields import RAW_FIELDS
from .generator import FactorSpec, generate_factors

DERIV_FIELDS = {f.name for f in RAW_FIELDS if f.group == "derivatives"}

MIN_COVERAGE = 0.15
MIN_DISPERSION_ROWS = 0.5
FINGERPRINT_ROWS = 250

_WORKER: dict = {}


def _init_worker(panel_source):
    """Make the evaluator of this process: on the given panels, or on a synthetic panel from its arguments."""
    if all(isinstance(v, pd.DataFrame) for v in panel_source.values()):
        panels = panel_source
    else:
        from .data.synthetic import make_synthetic_panel

        panels = make_synthetic_panel(**panel_source)
    _WORKER["ev"] = Evaluator(panels, cache_size=256)


def _fingerprint(ranks: np.ndarray) -> str:
    """Return a hash of the rank matrix (rounded to 6 decimals; NaN counts as -1)."""
    r = np.where(np.isnan(ranks), -1.0, np.round(ranks, 6))
    return hashlib.sha1(r.tobytes(), usedforsecurity=False).hexdigest()


def _check(expression: str) -> dict:
    """Evaluate one expression on the validation panel and return its checks.

    The result has ``ok`` False and a rejection ``kind`` and ``reason``, or ``ok`` True
    with the coverage, the dispersion, the rank autocorrelation and two rank fingerprints.
    """
    ev: Evaluator = _WORKER["ev"]
    try:
        with np.errstate(all="ignore"):
            v = ev(expression)
    except Exception as exc:  # noqa: BLE001 - every failure is a rejection reason
        return {"ok": False, "kind": "error", "reason": f"{type(exc).__name__}: {exc}"[:200]}
    arr = v.to_numpy(dtype=float)
    coverage = float(np.isfinite(arr).mean())
    if coverage < MIN_COVERAGE:
        return {"ok": False, "kind": "low_coverage", "reason": f"coverage {coverage:.3f}"}
    ranks = v.rank(axis=1, pct=True)
    rarr = ranks.to_numpy()
    valid_rows = np.isfinite(arr).sum(axis=1) >= 3
    disp = np.nanstd(rarr[valid_rows], axis=1) if valid_rows.any() else np.array([0.0])
    # rows where all names tie have zero rank dispersion
    std_raw = (
        np.nanstd(np.where(np.isfinite(arr), arr, np.nan)[valid_rows], axis=1) if valid_rows.any() else np.array([0.0])
    )
    dispersed = float((std_raw > 1e-12).mean())
    if dispersed < MIN_DISPERSION_ROWS:
        return {"ok": False, "kind": "no_dispersion", "reason": f"dispersed rows {dispersed:.2f}"}
    tail = rarr[-FINGERPRINT_ROWS:]
    neg = (-v).rank(axis=1, pct=True).to_numpy()[-FINGERPRINT_ROWS:]
    r_now, r_prev = rarr[1:], rarr[:-1]
    ok = np.isfinite(r_now) & np.isfinite(r_prev)
    ac = np.nan
    if ok.sum() > 50:
        a, b = r_now[ok], r_prev[ok]
        ac = float(np.corrcoef(a, b)[0, 1])
    return {
        "ok": True,
        "coverage": coverage,
        "cs_dispersion": float(np.nanmean(disp)),
        "rank_autocorr": ac,
        "fp": _fingerprint(tail),
        "fp_neg": _fingerprint(neg),
    }


def _check_many(expressions: list[str]) -> list[dict]:
    """Return the checks of each expression, in order."""
    return [_check(e) for e in expressions]


def validate(
    specs: list[FactorSpec], panel_source, workers: int | None = None, chunk: int = 200, log=print
) -> list[dict]:
    """Check each candidate on the validation panel and return one result dict for each spec, in order.

    ``panel_source`` is a dict of panels, or the keyword arguments of ``make_synthetic_panel``.
    ``workers`` is the number of processes (default: the number of CPUs); ``log`` receives
    progress lines (None for no output).
    """
    workers = workers or max(1, (os.cpu_count() or 2))
    exprs = [s.expression for s in specs]
    chunks = [exprs[i : i + chunk] for i in range(0, len(exprs), chunk)]
    results: list[dict] = []
    t0 = time.time()
    if workers == 1:
        _init_worker(panel_source)
        for i, c in enumerate(chunks):
            results.extend(_check_many(c))
            if log and (i + 1) % 20 == 0:
                log(f"  validated {len(results)}/{len(exprs)} ({time.time() - t0:.0f}s)")
        return results
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(panel_source,)) as pool:
        for i, res in enumerate(pool.map(_check_many, chunks)):
            results.extend(res)
            if log and (i + 1) % 20 == 0:
                log(f"  validated {len(results)}/{len(exprs)} ({time.time() - t0:.0f}s)")
    return results


def build_database(
    db_path: str | Path,
    panels: dict[str, pd.DataFrame] | None = None,
    synthetic: dict | None = None,
    workers: int | None = None,
    export_dir: str | Path | None = None,
    log=print,
) -> dict:
    """Generate, validate and store the factor library in the database ``db_path``.

    The validation panel is ``panels``, or else a synthetic panel (``synthetic`` gives its
    arguments; default 40 symbols, 1,600 bars, seed 7). The function replaces the factor
    table and the field and operator tables, writes the build information into the meta
    table and, with ``export_dir``, writes the CSV and Markdown export. The same inputs
    give the same database content. Return the build information: ``n_factors``,
    ``n_candidates``, ``rejected`` (count for each reason) and ``validation_data``.
    """
    t0 = time.time()
    specs = generate_factors()
    log(f"generated {len(specs)} candidate factors")
    source = panels if panels is not None else (synthetic or {"n_symbols": 40, "n_bars": 1600, "seed": 7})
    results = validate(specs, source, workers=workers, log=log)

    rejected: dict[str, int] = {}
    seen: set[str] = set()
    rows = []
    for spec, res in zip(specs, results):
        if not res["ok"]:
            rejected[res["kind"]] = rejected.get(res["kind"], 0) + 1
            continue
        if res["fp"] in seen or res["fp_neg"] in seen:
            rejected["rank_duplicate"] = rejected.get("rank_duplicate", 0) + 1
            continue
        seen.add(res["fp"])
        deps = raw_dependencies(spec.expression)
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "category": spec.category,
                "expression": spec.expression,
                "description": spec.description,
                "window": spec.window,
                "lookback": max_lookback(spec.expression),
                "raw_inputs": ",".join(sorted(deps)),
                "needs_derivs": int(bool(deps & DERIV_FIELDS)),
                "params": json.dumps(spec.params, sort_keys=True),
                "expr_hash": hashlib.sha1(spec.expression.encode(), usedforsecurity=False).hexdigest()[:16],
                "coverage": round(res["coverage"], 4),
                "cs_dispersion": round(res["cs_dispersion"], 4),
                "rank_autocorr": None if not np.isfinite(res["rank_autocorr"]) else round(res["rank_autocorr"], 4),
            }
        )
    for i, r in enumerate(rows, 1):
        r["factor_id"] = f"CF{i:05d}"

    with FactorDB(db_path) as db:
        db.write_catalog()
        n = db.replace_factors(rows)
        info = {
            "n_factors": n,
            "n_candidates": len(specs),
            "rejected": rejected,
            "validation_data": "synthetic" if panels is None else "user panels",
        }
        # the meta table holds only values that the inputs set (no clock time): the same inputs give the same content
        db.replace_meta(**info)
        if export_dir:
            export_catalog(db, export_dir)
    log(f"stored {n} factors in {db_path} (rejected: {rejected}) in {time.time() - t0:.1f}s")
    return info


def export_catalog(db: FactorDB, export_dir: str | Path) -> None:
    """Write the factor table of ``db`` into ``export_dir``.

    The files: ``crypto_factors.csv.gz`` (all factors), ``summary_by_family_category.csv``
    and ``SUMMARY.md`` (counts and examples).
    """
    d = Path(export_dir)
    d.mkdir(parents=True, exist_ok=True)
    df = db.factors()
    df.to_csv(d / "crypto_factors.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary = df.groupby(["family", "category"]).size().rename("n_factors").reset_index()
    summary.to_csv(d / "summary_by_family_category.csv", index=False)
    meta = db.meta()
    lines = [
        "# Crypto factor library",
        "",
        f"Total factors: **{len(df):,}** (from {meta.get('n_candidates', 0):,} generated candidates)",
        "",
        f"Rejected during validation: {meta.get('rejected')}",
        "",
        "## By family",
        "",
        "| family | factors |",
        "|---|---:|",
    ]
    for fam, n in df["family"].value_counts().items():
        lines.append(f"| {fam} | {n:,} |")
    lines += ["", "## By category", "", "| category | factors |", "|---|---:|"]
    for cat, n in df["category"].value_counts().items():
        lines.append(f"| {cat} | {n:,} |")
    lines += [
        "",
        "## Data requirements",
        "",
        f"* OHLCV / kline fields only: **{int((df['needs_derivs'] == 0).sum()):,}** factors",
        f"* also need perpetual-futures data (funding, OI, long/short, premium): **{int(df['needs_derivs'].sum()):,}** factors",
        "",
        "## Examples",
        "",
        "| id | name | expression |",
        "|---|---|---|",
    ]
    for _, r in df.groupby("family").head(2).iterrows():
        expr = r["expression"] if len(r["expression"]) < 120 else r["expression"][:117] + "..."
        lines.append(f"| {r['factor_id']} | `{r['name']}` | `{expr}` |")
    (d / "SUMMARY.md").write_text("\n".join(lines) + "\n")
