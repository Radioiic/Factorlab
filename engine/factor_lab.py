"""Daily factor search over a grid of diffusion specifications, with an append-only registry.

Usage::

    python factor_lab.py                 # score the grid, add each trial to the registry
    python factor_lab.py --report        # read the registry, run nothing
    python factor_lab.py --grid 36       # score all 36 specifications of the fixed grid

The factor lab scores a fixed grid of network-diffusion specifications (diffusion.py) on
the Sundays. It gives numbers only:

    1. each trial goes into the append-only registry data/research/factor_registry.parquet;
    2. the registry counts DISTINCT specifications: one more day of data for a spec that
       the registry has is the same specification, not a new trial;
    3. q_bh is the Benjamini-Hochberg q-value over the latest p-value of each distinct
       specification in the registry (the cumulative family); p_bonferroni is
       min(1, p x the number of distinct specifications);
    4. each spec gets the signed mean within-Sunday Spearman correlation with the incumbent
       of the largest absolute value (factorfinder/incumbents.py: the incumbents are your
       registered tests, the active AlphaForge formulas and the three risk proxies). The largest
       absolute correlation with the library definitions on the G Sundays is a separate number.

Nothing here writes into a book, a ledger or the trial ledger. A spec reaches the test
of a book only as a values file that you register with judge.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as student_t

import config as C
import data_io
from diffusion import DiffusionSpec, build
from factorfinder import incumbents as INC

RESEARCH = C.RESEARCH
REGISTRY = RESEARCH / "factor_registry.parquet"

MIN_WEEKS = 60  # a spec with fewer weekly ICs gets no row
LABEL = "fwd_ret_7"
#: the incumbent sources of the comparison (factorfinder/incumbents.py)
INCUMBENT_SOURCES = ("user", "zoo", "proxy")

REGISTRY_COLUMNS = [
    "run_date",
    "factor",
    "params",
    "weeks",
    "ic",
    "t",
    "p",
    "persistence",
    "turnover",
    "ic_per_turnover",
    "coverage",
    "first_seen",
    "data_through",
    "max_incumbent_corr",
    "closest_incumbent",
    "n_incumbents",
    "max_library_corr",
    "closest_library",
]


def default_grid(limit: int = 24) -> list[DiffusionSpec]:
    """Deterministic grid, ordered so a smaller --grid is a strict prefix."""
    out: list[DiffusionSpec] = []
    for mode in ("signed", "abs"):
        for kernel, knn in (("corr", 5), ("knn", 5), ("invdist", 5)):
            for threshold in (1.5, 2.0, 2.5):
                for decay, lags in ((0.7, 5), (0.9, 10)):
                    out.append(
                        DiffusionSpec(mode=mode, kernel=kernel, knn=knn, threshold=threshold, decay=decay, lags=lags)
                    )
    return out[:limit]


def evaluate(specs: list[DiffusionSpec], *, members: list | None = None) -> pd.DataFrame:
    """Score every spec: IC, t, p, persistence, coverage and the correlations with the incumbents
    and the library.

    `members` (tests) replaces the incumbents of factorfinder/incumbents.incumbents()."""
    prices = data_io.close_matrix()
    labels = data_io.label_table(columns=["date", "symbol", LABEL])
    cache: dict = {}
    diff = build(prices, specs, corr_cache=cache, progress=True)
    if diff.empty:
        return pd.DataFrame()

    sundays = sorted({d for d in diff.date.unique() if pd.Timestamp(d).weekday() == 6})
    wk = diff[diff.date.isin(sundays)].merge(labels, on=["date", "symbol"], how="left")
    wk["label"] = wk.groupby("date")[LABEL].rank(pct=True)
    if members is None:
        members = INC.incumbents(sources=INCUMBENT_SOURCES)
    frames = INC.rank_frames(members, pd.DatetimeIndex(sundays), columns=list(prices.columns))
    data_through = str(pd.Timestamp(prices.index.max()).date())

    n_universe = prices.shape[1]
    rows = []
    for s in specs:
        col = f"{s.name()}_rank"
        if col not in wk:
            continue
        d = wk.dropna(subset=[col, "label"])
        ics = (
            d.groupby("date")
            .apply(
                lambda g, col=col: spearmanr(g[col], g["label"]).statistic if len(g) > 4 else np.nan,
                include_groups=False,
            )
            .dropna()
        )
        if len(ics) < MIN_WEEKS:
            continue
        piv = wk.pivot_table(index="date", columns="symbol", values=col).reindex(columns=list(prices.columns))
        persistence = float(piv.corrwith(piv.shift(1), axis=1, method="spearman").dropna().mean())
        ic = float(ics.mean())
        tstat = float(ic / ics.std() * np.sqrt(len(ics))) if ics.std() > 0 else 0.0
        pval = float(2 * student_t.sf(abs(tstat), df=len(ics) - 1))
        near = INC.nearest(piv, frames)
        lib = INC.library_corr(piv)
        rows.append(
            {
                "factor": s.name(),
                "params": json.dumps(asdict(s), sort_keys=True),
                "weeks": len(ics),
                "ic": ic,
                "t": tstat,
                "p": pval,
                "persistence": persistence,
                "turnover": 1.0 - persistence,
                "ic_per_turnover": abs(ic) / max(1.0 - persistence, 0.01),
                "coverage": float(wk.groupby("date")[col].count().mean() / n_universe),
                "data_through": data_through,
                "max_incumbent_corr": near["corr"],
                "closest_incumbent": near["name"],
                "n_incumbents": int(near["n_incumbents"]),
                "max_library_corr": None if lib is None else lib["max_abs"],
                "closest_library": None if lib is None else lib["nearest"],
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["incumbents"] = INC.counts(members)
    return out


def load_registry() -> pd.DataFrame:
    if REGISTRY.exists():
        return pd.read_parquet(REGISTRY)
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def _registry_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Every registry column, in REGISTRY_COLUMNS order; an absent column is null."""
    out = rows.copy()
    for c in REGISTRY_COLUMNS:
        if c not in out.columns:
            out[c] = None
    for c in ("ic", "t", "p", "persistence", "turnover", "ic_per_turnover", "coverage"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    for c in ("max_incumbent_corr", "max_library_corr"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["n_incumbents"] = pd.to_numeric(out["n_incumbents"], errors="coerce").astype("Int64")
    for c in ("closest_incumbent", "closest_library", "data_through"):
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out[REGISTRY_COLUMNS]


def record(results: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """Append this run to the registry. The distinct-specification count is what matters."""
    reg = load_registry()
    seen = set(reg["factor"]) if len(reg) else set()
    results = results.copy()
    results["run_date"] = run_date
    results["first_seen"] = [f not in seen for f in results["factor"]]
    C.guard_write(REGISTRY)  # refused while the data folder is protected
    RESEARCH.mkdir(parents=True, exist_ok=True)
    new = _registry_frame(results)
    out = pd.concat([_registry_frame(reg), new], ignore_index=True) if len(reg) else new
    tmp = REGISTRY.with_name(f".{REGISTRY.name}.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(REGISTRY)
    return out


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """The Benjamini-Hochberg q-values of the p-values p, in the same order (the library's code)."""
    import libfactors

    p = np.asarray(p, dtype=float)
    return libfactors.bh_qvalues(p) if p.size else np.zeros(0)


def assess(results: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Numbers only: q_bh, p_bonferroni and n_distinct over the cumulative family.

    The family is the latest p-value of each distinct spec in the registry, with the p-values
    of this run in place of the registry's for its specs. Order: p ascending."""
    if results.empty:
        return results
    r = results.copy()
    family: dict[str, float] = {}
    if len(registry):
        reg = registry.sort_values("run_date", kind="stable")
        for f, p in zip(reg["factor"], pd.to_numeric(reg["p"], errors="coerce")):
            family[str(f)] = float(p)
    for f, p in zip(r["factor"], pd.to_numeric(r["p"], errors="coerce")):
        family[str(f)] = float(p)
    names = list(family)
    pv = np.array([family[n] for n in names], dtype=float)
    pv = np.where(np.isfinite(pv), pv, 1.0)
    q = dict(zip(names, bh_qvalues(pv)))
    n_distinct = len(names)
    r["q_bh"] = [float(q[str(f)]) for f in r["factor"]]
    r["p_bonferroni"] = np.minimum(1.0, pd.to_numeric(r["p"], errors="coerce").to_numpy(dtype=float) * n_distinct)
    r["n_distinct"] = n_distinct
    r.attrs["n_distinct"] = n_distinct
    return r.sort_values(["p", "factor"], kind="stable").reset_index(drop=True)


def _count(n, one: str, many: str | None = None) -> str:
    """'1 spec', '2 specs': the number and the noun in the form that agrees with it."""
    return f"{n} {one if n == 1 else (many or one + 's')}"


def _num(x, spec: str) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return format(v, spec) if np.isfinite(v) else "-"


def render(r: pd.DataFrame, registry: pd.DataFrame) -> str:
    """The printed table: numbers only."""
    if r.empty:
        return "Factor lab: no spec has enough weekly ICs to give a row."
    n_distinct = int(r.attrs.get("n_distinct", r["factor"].nunique()))
    runs = registry["run_date"].nunique() if len(registry) else 1
    L = [
        f"Factor lab: {_count(len(r), 'spec')}, {_count(n_distinct, 'distinct spec')} in the registry. Numbers only.",
        f"  registry run dates: {runs}",
    ]
    inc = r.attrs.get("incumbents")
    if inc is not None:
        L.append(
            f"  incumbents: {_count(inc.get('user', 0), 'registered test')}, "
            f"{_count(inc.get('zoo', 0), 'active zoo formula')}, {_count(inc.get('proxy', 0), 'risk proxy', 'risk proxies')}"
        )
    # a row without a count (null): the count is unknown, not 0
    n_inc = pd.to_numeric(r.get("n_incumbents", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(n_inc) and (n_inc == 0).all():
        L.append("  No incumbent: no registered test, no active zoo formula and no risk proxy.")
    L.append("")
    L.append(
        f"{'factor':<40}{'weeks':>6}{'ic':>9}{'t':>7}{'p':>10}{'q_bh':>8}{'p_bonf':>8}{'persist':>8}"
        f"{'turnover':>9}{'coverage':>9}{'incumbent corr':>16}{'library corr':>13}"
    )
    L.append("-" * 143)
    for row in r.itertuples():
        n_row = getattr(row, "n_incumbents", None)
        n_txt = "-" if n_row is None or pd.isna(n_row) else str(int(n_row))
        inc_corr = f"{_num(getattr(row, 'max_incumbent_corr', None), '+.2f')} ({n_txt})"
        L.append(
            f"{row.factor:<40}{int(row.weeks):>6}{_num(row.ic, '+9.4f')}{_num(row.t, '+7.2f')}{_num(row.p, '10.2e')}"
            f"{_num(row.q_bh, '8.3f')}{_num(row.p_bonferroni, '8.3f')}{_num(row.persistence, '8.3f')}"
            f"{_num(row.turnover, '9.3f')}{_num(row.coverage, '9.2f')}{inc_corr:>16}"
            f"{_num(getattr(row, 'max_library_corr', None), '13.2f')}"
        )
    L.append("")
    L.append(
        "  incumbent corr: the signed mean within-Sunday Spearman correlation with the incumbent of the "
        "largest absolute value (number of incumbents)."
    )
    L.append("  library corr: the largest absolute correlation with a library definition on the G Sundays.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Score the diffusion specs, add each trial to the registry and print the numbers."
    )
    ap.add_argument(
        "--grid", type=int, default=24, help="number of specs of the fixed grid to score (default 24, at most 36)"
    )
    ap.add_argument("--report", action="store_true", help="read registry only, run nothing")
    ap.add_argument("--no-write", action="store_true", help="score and print; add nothing to the registry")
    a = ap.parse_args(argv)

    if a.report:
        reg = load_registry()
        if reg.empty:
            print("The registry is empty. Run `python factor_lab.py` first.")
            return
        # a run date can be logged twice: the last rows of each spec count
        latest = reg[reg.run_date == reg.run_date.max()].drop_duplicates("factor", keep="last")
        print(render(assess(latest.copy(), reg), reg))
        return

    if not a.no_write:
        C.guard_write(REGISTRY)  # refuse a protected target before scoring the grid
    try:
        INC.check_zoo_file()  # a zoo file of another format stops the run here, before any computation
    except INC.ZooFormatError as exc:
        print(f"factor_lab: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    specs = default_grid(a.grid)
    print(f"factor_lab: scoring {_count(len(specs), 'diffusion spec')}")
    results = evaluate(specs)
    if results.empty:
        print("No spec has enough weekly ICs to give a row.")
        return
    run_date = str(pd.Timestamp.now(tz="UTC").date())
    reg = record(results, run_date) if not a.no_write else load_registry()
    print()
    out = assess(results, reg)
    out.attrs["incumbents"] = results.attrs.get("incumbents")
    print(render(out, reg))
    if not a.no_write:
        print(f"\nlogged {_count(len(results), 'trial')} -> data/research/{REGISTRY.name}")


if __name__ == "__main__":
    main()
