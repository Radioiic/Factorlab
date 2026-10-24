"""The label-free generator probe: how much room the existing factors leave for new formulas.

    .venv/bin/python -B finder.py --generator-probe [--n N] [--seed S]

READ-ONLY AND LABEL-FREE. The probe draws N random formulas from the generator's
terminals (21 library-named series and 4 Factorlab-only series, eval_config
[generator.terminals]) and the library operator registry, and measures, without looking
at a single return label, which of them are well-formed factors and their largest absolute
correlation with the members of the novelty index. It writes nothing (its report says
"writes: none"). It reads only what fieldmap.py and novelty.py read:
    - bars, funding and spot bars, cut at the 2024-06-09 row at the time of the read
      (no later value is used; no open-interest file is opened);
    - the library's value shards;
    - the date and symbol columns of the label table and the G rows of the AlphaForge table,
      through novelty.read_columns() (no fwd_* column, no row after the cutoff);
    - the trial ledger (data/trials/), and for a registered Binance formula the Binance
      panel cut at the cutoff.
It never opens screen_g.parquet or library/catalog. test_finder_probe.py watches every
open with an audit hook.

WHAT IT DOES, PER FORMULA
    draw       ramped half-and-half (libops.draw_trees: depth budgets 2..3, "grow" and
               "full" alternating) with alphaforge's ExpressionGenerator, unchanged, over
               FINDER_REGISTRY; a draw over the limits (depth 3, 20 nodes, nested lookback
               180 days, eval_config [generator.settings]) is redrawn; the tree is then simplified
               (the generator's own rewrites), and a repeated canonical string is a
               struct duplicate
    evaluate   on the terminals cut at the G cutoff (fieldmap.build_terminals), values on
               the 114 G Sundays
    G0         the label-free sanity checks of eval_config [g0], as the probe reads them:
                 a Sunday counts when >= 15 eligible coins have a finite value;
                 coverage = counted Sundays / 114 >= 0.80 (R_G0_COVERAGE);
                 finite share = finite / eligible cells on counted Sundays >= 0.95
                 (R_G0_FINITE);
                 no counted Sunday with one value for every coin (R_G0_CONSTANT);
                 >= 10 distinct values on every counted Sunday (R_G0_DISTINCT; values
                 tie-snapped first, as the screen does).
               A formula whose leaves are all market-wide (mkt_ret, btc_ret: one value
               for every coin on a date) is constant per date and fails R_G0_CONSTANT
               (or R_G0_COVERAGE / R_G0_FINITE, checked first); where a recursive
               operator (ema) splits the coins by listing date, it has only a few
               values on a Sunday (coins listed on the same date share one; 4 listing
               dates on G), below the 10 of R_G0_DISTINCT. The report counts these
               formulas ("market-wide leaves only"); test_finder_probe.py checks that
               none passes G0
    lint       future perturbation: every terminal value after the 2023-07-30 row
               (the G_fit / G_eval boundary) replaced by noise, missing cells kept
               missing; every value on or before it must stay bit-identical
               (R_LINT_FUTURE); and no value before a coin's first perp bar
               (R_LINT_LISTING). Bit-identity is guaranteed only under this VALUE
               mutation: when the later rows are cut off instead (the real-time case),
               the library's windowed dot products (decay_linear, ts_slope, ts_rsq) round
               differently by up to a few 1e-12, because the BLAS product sees another
               number of rows. test_finder_probe.py checks truncation separately: the
               finite mask is unchanged, values agree within the library screen's PIT
               tolerance (rtol 1e-9, atol 1e-12) and the tie-snapped ranks on every G
               Sunday are identical
    dedup      value duplicates within the probe: identical ranks on every cell, or their
               mirror image (a sign flip is the same signal)
    bands      novelty.max_corr against the index (every member measured exactly) gives
               the largest absolute correlation; its band: below_050 (below 0.50),
               from_050_to_085 (0.50 to 0.85), from_085_library (0.85 or more, nearest a
               library definition), from_085_other (0.85 or more, nearest a user test, a
               zoo formula or a risk proxy), no_overlap (no member with a correlation);
               above 0.999 it is also counted as value-identical. Thresholds: eval_config
               [generator.settings]
    null       novelty.null_max: the same exact maximum (every member measured) for
               NULL_PERMS copies of the formula with its coins permuted (one fixed
               permutation per copy); its median is the formula's chance floor for the
               maximum over the whole index
The report gives each count and share overall and by terminal group: formulas built only
from library-named terminals, and formulas using at least one Factorlab-only terminal.
"share valid" is the share of distinct formulas (after simplification) that pass G0 and
the lint; the band shares are shares of the valid formulas left after value dedup.
Shares, quartiles and null floors are kept at full precision in the report and rounded
once, when lines() prints them.
The report gives numbers only. No label is read, so it says nothing about whether a
formula predicts returns.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

from factorfinder import config_io as K
from factorfinder import fieldmap as FM
from factorfinder import libops as LO
from factorfinder import novelty as N

DEFAULT_N = 2000
DEFAULT_SEED = 1
NULL_PERMS = 4
LINT_CUT = "2023-07-30"  # the G_fit / G_eval boundary (the internal embargo Sunday)
GROUPS = ("all", "library_only", "uses_factorlab")
CLASSES = ("below_050", "from_050_to_085", "from_085_library", "from_085_other", "no_overlap")
#: the printed label of each band (lines())
CLASS_LABELS = {
    "below_050": "largest |corr| below 0.50",
    "from_050_to_085": "0.50 to 0.85",
    "from_085_library": "0.85 or more, nearest a library definition",
    "from_085_other": "0.85 or more, nearest a user test, a zoo formula or a risk proxy",
    "no_overlap": "no overlap",
}


def derived_seed(seed: int) -> int:
    """The generator's seed: sha256('finder-generator-probe:' || seed), so a probe seed
    never coincides with the AlphaForge bridge's seeds."""
    return int.from_bytes(hashlib.sha256(f"finder-generator-probe:{int(seed)}".encode()).digest()[:8], "big")


# --------------------------------------------------------------------------- checks


def g0_label_free(values: np.ndarray, el: np.ndarray, g0: dict) -> tuple[str | None, dict]:
    """(first failing reason or None, statistics) for (T, m) G-Sunday values."""
    import libfactors as LF

    v = LF.snap_ties(np.asarray(values, dtype=np.float64))
    fin = np.isfinite(v) & el
    n = fin.sum(axis=1)
    counted = n >= int(g0["min_names_per_sunday"])
    coverage = float(counted.mean())
    elig = el.sum(axis=1)
    finite_share = float(n[counted].sum() / max(elig[counted].sum(), 1)) if counted.any() else 0.0
    distinct = np.array([len(np.unique(v[t][fin[t]])) if counted[t] else 0 for t in range(len(v))])
    stats = {
        "coverage": round(coverage, 4),
        "finite_share": round(finite_share, 4),
        "min_distinct": int(distinct[counted].min()) if counted.any() else 0,
    }
    if coverage < float(g0["min_coverage"]):
        return "R_G0_COVERAGE", stats
    if finite_share < float(g0["min_finite_share"]):
        return "R_G0_FINITE", stats
    if (distinct[counted] <= 1).any():
        return "R_G0_CONSTANT", stats
    if (distinct[counted] < int(g0["min_distinct_values_per_date"])).any():
        return "R_G0_DISTINCT", stats
    return None, stats


def perturbed_terminals(terms: dict, cut: pd.Timestamp, seed: int) -> dict:
    """Every terminal value after `cut` replaced by noise of the terminal's own scale
    (missing cells stay missing); rows on or before `cut` untouched."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, f in terms.items():
        a = f.to_numpy(dtype=float).copy()
        late = f.index > cut
        x = a[late]
        sd = np.nanstd(a[~late]) if np.isfinite(a[~late]).any() else 1.0
        noise = rng.standard_normal(x.shape) * (sd if sd > 0 else 1.0)
        a[late] = np.where(np.isfinite(x), noise, np.nan)
        out[name] = pd.DataFrame(a, index=f.index, columns=f.columns)
    return out


def lint_reason(full: np.ndarray, perturbed: np.ndarray, k: int, first_bar: np.ndarray) -> str | None:
    """R_LINT_FUTURE when a value on one of the first `k` rows changed although only later
    inputs were perturbed (bit-identity, NaN equal to NaN); R_LINT_LISTING when a value
    exists before a coin's first perp bar; else None."""
    if not np.array_equal(full[:k], perturbed[:k], equal_nan=True):
        return "R_LINT_FUTURE"
    if (np.isfinite(full) & ~first_bar).any():
        return "R_LINT_LISTING"
    return None


def value_key(c: N.Candidate) -> str:
    """Identity of a formula's G-Sunday ranks, the same for a sign flip."""

    def h(r):
        return hashlib.sha256(np.packbits(c.mask).tobytes() + (2 * r).astype(np.int16).tobytes()).hexdigest()

    return min(h(c.ranks), h(-c.ranks))


def novelty_class(r: dict, th: dict) -> str:
    """The band of the largest absolute correlation (CLASSES)."""
    x = r["max_abs"]
    if not np.isfinite(x):
        return "no_overlap"
    if x < th["max_abs_corr"]:
        return "below_050"
    if x < th["duplicate"]:
        return "from_050_to_085"
    return "from_085_library" if r["source"] == "library" else "from_085_other"


# --------------------------------------------------------------------------- run


def _thresholds(cfg: dict) -> dict:
    d = cfg["generator"]["settings"]
    return {
        "max_abs_corr": float(d["novelty_max_abs_corr"]),
        "duplicate": float(d["novelty_duplicate_abs_corr"]),
        "same": float(d["novelty_same_hypothesis_abs_corr"]),
    }


def run(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    *,
    null_perms: int = NULL_PERMS,
    log=None,
    index: N.NoveltyIndex | None = None,
) -> dict:
    """The probe. Returns {"report": deterministic dict, "timings": {...}, "journal": lines}.
    The probe writes nothing, so "journal" is always ["writes: none"]."""
    say = log or (lambda *_: None)
    t0 = time.time()
    cfg = K.load()
    ver = K.versions(cfg)
    g0 = cfg["g0"]
    lim = LO.defaults(cfg)
    th = _thresholds(cfg)
    cut = FM.g_cutoff(cfg)
    lint_cut = pd.Timestamp(LINT_CUT, tz="UTC")
    inputs = FM.load_inputs(cut)
    terms = FM.build_terminals(inputs)
    names = FM.terminal_names(cfg)
    ix = N.build_index(inputs=inputs) if index is None else index
    timings = {"inputs_and_index_s": round(time.time() - t0, 1)}
    say(
        f"index: {ix.counts['total']:,} members ({ix.counts['library']:,} library), pool {ix.pool_hash[:12]} "
        f"({timings['inputs_and_index_s']}s)"
    )
    perp = FM.perp_panels(inputs)
    first_bar = perp["close"].notna().cummax().to_numpy()  # True from a coin's first bar on
    pert = perturbed_terminals(terms, lint_cut, derived_seed(seed) % (2**32))
    ev, ev_p = LO.evaluator(terms), LO.evaluator(pert)
    k_lint = int(terms[names[0]].index.searchsorted(lint_cut, side="right"))

    t1 = time.time()
    gen = LO.generator(names, seed=derived_seed(seed), max_depth=lim["max_depth"])
    trees, draw = LO.draw_trees(gen, n, lim)
    timings["draw_s"] = round(time.time() - t1, 1)

    rows = []
    seen_struct: set = set()
    seen_value: set = set()
    t2 = time.time()
    for i, raw_tree in enumerate(trees):
        tree = gen.simplify(raw_tree)
        st = LO.tree_stats(tree)
        text = LO.to_string(tree)
        grp = "uses_factorlab" if set(st["fields"]) & set(FM.FACTORLAB_TERMINALS) else "library_only"
        row = {
            "i": i,
            "expr": text,
            "group": grp,
            "depth": st["depth"],
            "nodes": st["nodes"],
            "lookback": st["lookback"],
            "market_only": FM.market_only(st["fields"]),
            "status": None,
        }
        rows.append(row)
        if (i + 1) % 250 == 0:
            say(f"  {i + 1:,}/{len(trees):,} formulas ({time.time() - t2:.0f}s)")
        if text in seen_struct:
            row["status"] = "R_DUP_STRUCT"
            continue
        seen_struct.add(text)
        try:
            full = LO.evaluate(tree, ev=ev)
        except Exception as exc:  # noqa: BLE001 - an operator failure is a probe outcome, recorded
            row["status"], row["error"] = "R_EVAL_ERROR", type(exc).__name__
            continue
        vals = full.reindex(index=ix.sundays, columns=ix.symbols).to_numpy(dtype=float)
        reason, g0s = g0_label_free(vals, ix.el, g0)
        row.update(g0s)
        if reason:
            row["status"] = reason
            continue
        lint = lint_reason(
            full.to_numpy(dtype=float), LO.evaluate(tree, ev=ev_p).to_numpy(dtype=float), k_lint, first_bar
        )
        if lint:
            row["status"] = lint
            continue
        c = N.candidate(ix, vals)
        key = value_key(c)
        if key in seen_value:
            row["status"] = "R_DUP_VALUE"
            continue
        seen_value.add(key)
        r = N.max_corr(ix, vals, c=c)
        nulls = N.null_max(ix, vals, null_perms, [derived_seed(seed) % (2**32), i])
        row.update(
            status="valid",
            max_abs=r["max_abs"],
            nearest=r["nearest"],
            nearest_source=r["source"],
            nearest_family=r["family"],
            cls=novelty_class(r, th),
            identical=bool(np.isfinite(r["max_abs"]) and r["max_abs"] > th["same"]),
            null_median=float(np.nanmedian(nulls)) if np.isfinite(nulls).any() else float("nan"),
            n_rechecked=r["n_rechecked"],
        )
    timings["formulas_s"] = round(time.time() - t2, 1)
    timings["total_s"] = round(time.time() - t0, 1)
    label_reads = [p for p, cols in N.READS if any(str(x).startswith(N.LABEL_PREFIX) for x in cols)]
    report = {
        "probe": {
            "n": int(n),
            "seed": int(seed),
            "generator_seed": derived_seed(seed),
            "null_perms": int(null_perms),
            "lint_cut": LINT_CUT,
            "data_through_used": str(cut.date()),
        },
        "config": {
            "eval_config_version": ver["eval_config_version"],
            "generator_config_version": ver["generator_config_version"],
            "status": "terminals: [generator.terminals] (21 library-named, 4 Factorlab-only); operators, "
            "windows, limits and novelty thresholds: [generator.settings]",
            "limits": {k: lim[k] for k in ("max_depth", "max_nodes", "max_nested_lookback")},
            "thresholds": th,
            "g0": {
                k: g0[k]
                for k in ("min_coverage", "min_names_per_sunday", "min_finite_share", "min_distinct_values_per_date")
            },
        },
        "terminals": {"library": list(FM.terminal_config(cfg)["library"]), "factorlab": list(FM.FACTORLAB_TERMINALS)},
        "operators": {
            "mineable": sorted(k for k, s in LO.registry().items() if s.mineable),
            "not_mineable": sorted(k for k, s in LO.registry().items() if not s.mineable),
        },
        "index": N.describe(ix),
        "draw": draw,
        "summary": summarize(rows),
        "label_columns_read": len(label_reads),
        "tables_read": sorted({p for p, _ in N.READS}),
    }
    return {"report": report, "rows": rows, "timings": timings, "journal": ["writes: none"]}


def _share(a: int, b: int) -> float | None:
    """a / b at full precision (lines() rounds once, when it prints), None when b is 0."""
    return a / b if b else None


def summarize(rows: list[dict]) -> dict:
    out = {}
    for g in GROUPS:
        rs = [r for r in rows if g == "all" or r["group"] == g]
        drawn = len(rs)
        st = pd.Series([r["status"] for r in rs], dtype=object)
        cnt = {k: int(v) for k, v in st.value_counts().sort_index().items()}
        lint_ok = [r for r in rs if r["status"] in ("valid", "R_DUP_VALUE")]
        valid = [r for r in rs if r["status"] == "valid"]
        k = len(valid)
        cls = pd.Series([r["cls"] for r in valid], dtype=object).value_counts()
        mx = np.array([r["max_abs"] for r in valid], dtype=float)
        nl = np.array([r["null_median"] for r in valid], dtype=float)
        fin_n = nl[np.isfinite(nl)]
        fin_m = mx[np.isfinite(mx)]
        novel_below_null = sum(
            1
            for r in valid
            if np.isfinite(r["max_abs"]) and np.isfinite(r["null_median"]) and r["max_abs"] <= r["null_median"]
        )
        distinct = drawn - cnt.get("R_DUP_STRUCT", 0)
        out[g] = {
            "drawn": drawn,
            "distinct": distinct,
            "status_counts": cnt,
            "passed_g0_and_lint": len(lint_ok),
            "share_valid": _share(len(lint_ok), distinct),
            "value_unique_valid": k,
            "share_value_unique_valid": _share(k, distinct),
            "novelty_counts": {c: int(cls.get(c, 0)) for c in CLASSES},
            **{f"share_{c}": _share(int(cls.get(c, 0)), k) for c in CLASSES},
            "share_from_085": _share(int(cls.get("from_085_library", 0)) + int(cls.get("from_085_other", 0)), k),
            "value_identical": sum(1 for r in valid if r["identical"]),
            "share_value_identical": _share(sum(1 for r in valid if r["identical"]), k),
            "max_abs_corr_quartiles": [float(q) for q in np.quantile(fin_m, [0.25, 0.5, 0.75])] if fin_m.size else None,
            "null_floor_median": float(np.median(fin_n)) if fin_n.size else None,
            "null_floor_p90": float(np.quantile(fin_n, 0.9)) if fin_n.size else None,
            "null_floor_at_or_above_0_5": int((fin_n >= 0.5).sum()),
            "share_null_floor_at_or_above_0_5": _share(int((fin_n >= 0.5).sum()), int(fin_n.size)),
            "share_max_at_or_below_own_null": _share(novel_below_null, k),
            "nearest_sources": {s: sum(1 for r in valid if r["nearest_source"] == s) for s in N.SOURCES},
            # formulas whose leaves are all market-wide (constant per date): distinct ones,
            # and how many of them passed G0 and the lint (none should)
            "market_only_distinct": sum(1 for r in rs if r["market_only"] and r["status"] != "R_DUP_STRUCT"),
            "market_only_passed": sum(1 for r in lint_ok if r["market_only"]),
        }
    return out


def lines(result: dict) -> list[str]:
    """The printed report."""
    rep, tm = result["report"], result["timings"]
    p, ix = rep["probe"], rep["index"]
    L = [
        "=== FACTOR FINDER -- label-free generator probe (read-only; no label is read) ===",
        f"  formulas       {p['n']:,} drawn, seed {p['seed']} (generator seed {p['generator_seed']}), "
        f"{p['null_perms']} permutations each for the null floor",
        f"  config         eval {rep['config']['eval_config_version'][:12]}, generator "
        f"{rep['config']['generator_config_version'][:12]} ({rep['config']['status']})",
        f"  data           G zone only: values on {ix['sundays'][2]} Sundays {ix['sundays'][0]} -> {ix['sundays'][1]}; "
        f"inputs cut at {p['data_through_used']}",
        f"  Index: {ix['counts']['library']:,} library, {ix['counts']['user']} user, "
        f"{ix['counts']['zoo']} zoo, {ix['counts']['proxy']} proxy members.",
        f"  pool_hash      {ix['pool_hash'][:16]}",
        f"  draws          {rep['draw']['draws']:,} drawn for {p['n']:,} within the limits; redrawn: "
        + (", ".join(f"{k} {v}" for k, v in sorted(rep['draw']['rejected'].items())) or "none"),
        f"  labels read    {rep['label_columns_read']} label column(s)",
        "",
    ]
    hdr = f"  {'':<66}{'all':>16}{'library-only':>17}{'uses Factorlab':>18}"
    L.append(hdr)
    s = rep["summary"]

    def get(g, key):
        v = s[g]
        for part in key.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        return v

    def row(label, key, fmt="{:,}", count=None):
        # shares are stored at full precision and rounded here, once; `count` prints the
        # count the share is made of next to it
        vals = []
        for g in GROUPS:
            v = get(g, key)
            if v is None:
                vals.append("-")
            elif fmt == "%":
                vals.append(f"{v:.1%}" if count is None else f"{get(g, count):,} ({v:.1%})")
            else:
                vals.append(fmt.format(v))
        L.append(f"  {label:<66}{vals[0]:>16}{vals[1]:>17}{vals[2]:>18}")

    row("drawn", "drawn")
    row("distinct after simplification", "distinct")
    row("passed G0 and the lint", "passed_g0_and_lint")
    row("  share valid (of distinct)", "share_valid", "%")
    row("value-unique valid (scored)", "value_unique_valid")
    for c in CLASSES:
        row("  " + CLASS_LABELS[c], f"share_{c}", "%", f"novelty_counts.{c}")
    row("  value-identical, above 0.999", "share_value_identical", "%", "value_identical")
    row("null floor, median", "null_floor_median", "{:.3f}")
    row("null floor, 90th percentile", "null_floor_p90", "{:.3f}")
    row("  share with null floor >= 0.5", "share_null_floor_at_or_above_0_5", "%", "null_floor_at_or_above_0_5")
    row("market-wide leaves only (distinct)", "market_only_distinct")
    row("  of them passed G0 and the lint", "market_only_passed")
    L.append("")
    L.append(
        "  why formulas were set aside (all): "
        + ", ".join(f"{k} {v:,}" for k, v in s["all"]["status_counts"].items() if k != "valid")
    )
    L.append("  These are numbers. No label was read, so they do not show if a formula predicts returns.")
    L.append(
        f"  time: index {tm['inputs_and_index_s']}s, draw {tm['draw_s']}s, formulas {tm['formulas_s']}s, "
        f"total {tm['total_s']}s"
    )
    L += [f"  {x}" for x in result["journal"]]
    return L
