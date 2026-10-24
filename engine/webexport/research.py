"""Search tools section: the factor lab registry, the AlphaForge zoo and the generator probe.

Numbers only. The builder is read-only over ``config.DATA``:

- factor lab: ``research/factor_registry.parquet`` with ``factor_lab.assess`` (q_bh,
  p_bonferroni over the cumulative family of distinct specs). The registry is deduplicated
  on (run_date, factor) first because a run can be logged twice with identical rows.
- AlphaForge (only when ``config.ALPHAFORGE_ENABLED`` is True):
  ``research/alphaforge_library.json`` (the zoo, format 3; a file of another format fails the
  section with ZooFormatError), ``research/alphaforge_cycles.jsonl`` (one line per
  cycle) and ``research/alphaforge_registry.parquet`` (the mining runs). Host details
  (``cohorts[*].environment``) are never exported. With AlphaForge off the payload is
  ``{"enabled": false}``.
- the generator probe: its command. The probe writes no file, so the page shows no result.
"""

from __future__ import annotations

import json
import math

import pandas as pd

import config
import factor_lab
from factorfinder import incumbents as INC
from webexport.common import data_path, exists, read_json, rnd

PROBE = {
    "where": "Run this command from the repository root.",
    "command": "cd engine && .venv/bin/python -B finder.py --generator-probe --n 2000 --seed 1",
    "text": "It prints the counts of each band of the largest |corr|. It writes no file.",
}
AF_SETTING_KEYS = (
    "forward_window_weeks",
    "max_factors",
    "mine_every_days",
    "population",
    "generations",
    "max_depth",
    "min_valid_icir_ann",
)
GATE_KEYS = ("valid_icir", "valid_t", "persistence", "coverage", "max_incumbent_corr", "max_library_corr")


# --------------------------------------------------------------------------- small helpers


def sig(x, digits: int = 4):
    """Round to significant digits (p-values, q-values); NaN/inf -> None."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    if f == 0:
        return 0.0
    return float(f"{f:.{digits}g}")


def _fin(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _bool(x):
    return None if x is None else bool(x)


def _str(x):
    return x if isinstance(x, str) and x else None


def _int(x):
    f = _fin(x)
    return int(f) if f is not None else None


def _date(x):
    """A YYYY-MM-DD text or None."""
    if not isinstance(x, str) or len(x) < 10:
        return None
    try:
        return str(pd.Timestamp(x[:10]).date())
    except ValueError:
        return None


# --------------------------------------------------------------------------- factor lab


def _dedup_registry(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.drop_duplicates(["run_date", "factor"], keep="last").reset_index(drop=True)


def _params(s: str) -> dict:
    try:
        p = json.loads(s) if isinstance(s, str) else {}
    except ValueError:
        p = {}
    keep = ("mode", "kernel", "knn", "threshold", "decay", "lags", "corr_window", "z_window")
    return {k: p.get(k) for k in keep if k in p}


def _col(r, name):
    return getattr(r, name, None)


def build_factor_lab() -> dict | None:
    if not exists("research/factor_registry.parquet"):
        return None
    raw = pd.read_parquet(factor_lab.REGISTRY)
    if raw.empty:
        return {
            "meta": {"n_rows_raw": 0, "n_runs": 0, "n_distinct": 0, "last_run": None, "data_through": None},
            "latest": [],
            "runs": [],
            "history": {"run_dates": [], "factors": {}},
        }
    reg = _dedup_registry(raw)
    n_distinct = int(reg["factor"].nunique())

    # how many times each run_date was logged (a run appends one row per spec)
    per_date = raw.groupby("run_date").agg(rows=("factor", "size"), specs=("factor", "nunique"))
    copies = (per_date["rows"] / per_date["specs"]).round().astype(int)
    duplicates = [{"run_date": str(d), "copies": int(c)} for d, c in copies.items() if c > 1]

    last_run = str(reg["run_date"].max())
    latest = reg[reg["run_date"] == last_run].copy()
    a = factor_lab.assess(latest, reg)  # p ascending
    rows = []
    for r in a.itertuples():
        rows.append(
            {
                "factor": r.factor,
                "params": _params(r.params),
                "weeks": int(r.weeks),
                "ic": rnd(r.ic, "ic"),
                "t": rnd(r.t, "t"),
                "p": sig(r.p),
                "q_bh": sig(r.q_bh),
                "p_bonferroni": sig(r.p_bonferroni),
                "persistence": rnd(r.persistence, "num"),
                "turnover": rnd(r.turnover, "num"),
                "coverage": rnd(r.coverage, "num"),
                "max_incumbent_corr": rnd(_col(r, "max_incumbent_corr"), "num"),
                "closest_incumbent": _str(_col(r, "closest_incumbent")),
                "n_incumbents": _int(_col(r, "n_incumbents")),
                "max_library_corr": rnd(_col(r, "max_library_corr"), "num"),
                "closest_library": _str(_col(r, "closest_library")),
            }
        )

    # the numbers of each run, each scored against the registry as it stood then
    runs = []
    run_dates = sorted(reg["run_date"].unique().tolist())
    for rd in run_dates:
        g = reg[reg["run_date"] == rd].copy()
        upto = reg[reg["run_date"] <= rd]
        s = factor_lab.assess(g, upto)
        runs.append(
            {
                "run_date": str(rd),
                "specs": int(len(s)),
                "copies": int(copies.get(rd, 1)),
                "n_distinct": int(s.attrs.get("n_distinct", upto["factor"].nunique())),
                "min_p": sig(s["p"].min()),
                "min_q_bh": sig(s["q_bh"].min()),
                "max_abs_t": rnd(s["t"].abs().max(), "t"),
                "max_weeks": int(s["weeks"].max()),
            }
        )

    # per-factor history across runs, columnar and aligned with run_dates (null = not scored that day)
    hist = {}
    order = [r["factor"] for r in rows] + sorted(set(reg["factor"]) - {r["factor"] for r in rows})
    for f in order:
        g = reg[reg["factor"] == f].set_index("run_date")
        hist[f] = {
            "ic": [rnd(g["ic"].get(d), "ic") if d in g.index else None for d in run_dates],
            "t": [rnd(g["t"].get(d), "t") if d in g.index else None for d in run_dates],
            "persistence": [rnd(g["persistence"].get(d), "num") if d in g.index else None for d in run_dates],
        }

    through = None
    if "data_through" in latest.columns:
        vals = [_date(x) for x in latest["data_through"].dropna().astype(str)]
        vals = [v for v in vals if v]
        through = max(vals) if vals else None
    n_inc = [r["n_incumbents"] for r in rows if r["n_incumbents"] is not None]
    return {
        "meta": {
            "n_rows_raw": int(len(raw)),
            "n_rows": int(len(reg)),
            "n_runs": len(run_dates),  # distinct run dates (what factor_lab --report counts)
            "n_logged_runs": int(copies.sum()),
            "duplicates": duplicates,
            "first_run": str(run_dates[0]),
            "last_run": last_run,
            "data_through": through,
            "n_distinct": n_distinct,
            "n_specs_latest": int(len(rows)),
            "label": factor_lab.LABEL,
            "min_weeks": factor_lab.MIN_WEEKS,
            "max_n_incumbents": max(n_inc) if n_inc else None,
            "max_persistence_ever": rnd(raw["persistence"].max(), "num"),
        },
        "latest": rows,
        "runs": runs,
        "history": {"run_dates": [str(d) for d in run_dates], "factors": hist},
    }


# --------------------------------------------------------------------------- AlphaForge zoo


def _settings() -> dict:
    """The AlphaForge settings of config.py (the bridge's defaults under them)."""
    import alphaforge_bridge as AB

    s = AB.settings()
    return {k: s.get(k) for k in AF_SETTING_KEYS}


def _cycles() -> list[dict]:
    """The cycle log, numbers only. A line without the key "mined" is not a cycle line and is
    not exported."""
    if not exists("research/alphaforge_cycles.jsonl"):
        return []
    out = []
    with open(data_path("research/alphaforge_cycles.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except ValueError:
                continue
            if not isinstance(c, dict) or "mined" not in c:
                continue
            out.append(
                {
                    "date": c.get("date"),
                    "data_through": c.get("data_through"),
                    "mined": _bool(c.get("mined")),
                    "added": list(c.get("added") or []),
                    "left": list(c.get("left") or []),
                    "active": _int(c.get("active")),
                    "room": _int(c.get("room")),
                    "n_evaluated": _int(c.get("n_evaluated")),
                    "seed": _int(c.get("seed")),
                }
            )
    return out


def _mining_runs() -> dict:
    if not exists("research/alphaforge_registry.parquet"):
        return {"runs": [], "n_rows": 0, "n_distinct_hypotheses": None}
    reg = pd.read_parquet(data_path("research/alphaforge_registry.parquet"))
    runs = []
    labelled = reg[reg["run_id"].notna()]
    for rid, g in labelled.groupby("run_id", sort=True):
        sel = g["selected"].fillna(False).astype(bool)
        runs.append(
            {
                "run_id": str(rid),
                "run_date": str(g["run_date"].iloc[0]),
                "evaluated": _int(g["n_evaluated_in_run"].iloc[0]),
                "hall_of_fame": int(len(g)),
                "valid_rule": int(g["valid_passed"].fillna(False).astype(bool).sum()),
                "selected": int(sel.sum()),
                "selected_names": sorted(str(x) for x in g.loc[sel, "library_name"] if isinstance(x, str) and x),
            }
        )
    return {
        "runs": runs,
        "n_rows": int(len(reg)),
        "n_distinct_hypotheses": int(reg["hypothesis"].nunique()) if "hypothesis" in reg else None,
    }


def _forward(fw: dict) -> dict:
    ser = [(str(d), _fin(v)) for d, v in (fw.get("series") or [])]
    return {
        "weeks": int(fw.get("weeks") or 0),
        "ic": rnd(fw.get("ic"), "ic"),
        "t": rnd(fw.get("t"), "t"),
        "p": sig(fw.get("p")),
        "dates": [d for d, _ in ser],
        "ic_series": [rnd(v, "ic") for _, v in ser],
    }


def build_alphaforge() -> dict:
    if not bool(getattr(config, "ALPHAFORGE_ENABLED", False)):
        return {"enabled": False}
    import alphaforge_bridge as AB

    settings = _settings()
    zoo = (
        INC.check_zoo_format(read_json("research/alphaforge_library.json"))
        if exists("research/alphaforge_library.json")
        else AB.empty_zoo()
    )
    members = []
    for m in zoo.get("factors") or []:
        gates = m.get("gates") or {}
        win = m.get("window") or {}
        lib = m.get("library") or {}
        members.append(
            {
                "name": m.get("name"),
                "expr": m.get("expr"),
                "selected_on": m.get("selected_on"),
                "cohort": _int(m.get("cohort")),
                "active": bool(m.get("active")),
                "left_on": m.get("left_on"),
                "window": {
                    "weeks": _int(win.get("weeks")),
                    "first_sunday": win.get("first_sunday"),
                    "last_sunday": win.get("last_sunday"),
                },
                "forward": _forward(m.get("forward") or {}),
                "gates": {k: rnd(gates.get(k), "num") for k in GATE_KEYS},
                "closest_library": _str(lib.get("nearest")),
            }
        )
    cohorts = []
    for c in zoo.get("cohorts") or []:
        st = c.get("settings") or {}
        hof = c.get("hall_of_fame")
        cohorts.append(
            {
                "cohort": _int(c.get("cohort")),
                "selected_on": c.get("selected_on"),
                "run_id": c.get("run_id"),
                "data_through": c.get("data_through"),
                "added": list(c.get("added") or []),
                "n_evaluations": _int(c.get("n_evaluations")),
                "hall_of_fame": hof if isinstance(hof, int) else (len(hof) if isinstance(hof, list) else None),
                "n_valid_hof": _int(c.get("n_valid_hof")),
                "n_valid_rule": _int(c.get("n_valid_rule")),
                "n_low_corr": _int(c.get("n_low_corr")),
                "n_uncorrelated": _int(c.get("n_uncorrelated")),
                "n_selected": _int(c.get("n_selected")),
                "incumbent_counts": {k: _int(v) for k, v in (c.get("incumbent_counts") or {}).items()},
                "seed": _int(st.get("seed")),
                "max_corr": _fin(st.get("max_corr")),
                "min_valid_t": _fin(st.get("min_valid_t")),
            }
        )
    cycles = _cycles()
    active = sum(1 for m in members if m["active"])
    max_factors = int(settings.get("max_factors") or 6)
    last_cycle = cycles[-1] if cycles else None
    through = _date(last_cycle.get("data_through")) if last_cycle else None
    return {
        "enabled": True,
        "meta": {
            "data_through": through or _date(zoo.get("data_through")),
            "last_cycle": last_cycle.get("date") if last_cycle else None,
            "last_mining": max((c["selected_on"] for c in cohorts if c.get("selected_on")), default=None),
            "n_cohorts": len(cohorts),
            "fields": list(zoo.get("fields") or []),
            "label": zoo.get("label"),
        },
        "settings": settings,
        "counts": {"members": len(members), "active": active, "room": max(max_factors - active, 0)},
        "members": members,
        "cohorts": cohorts,
        "cycles": cycles,
        "registry": {
            **_mining_runs(),
            "n_expressions_searched": _int((zoo.get("registry") or {}).get("n_expressions_searched")),
        },
    }


# --------------------------------------------------------------------------- section


def build() -> dict:
    lab = build_factor_lab()
    af = build_alphaforge()
    dates = [
        d for d in ((lab or {}).get("meta", {}).get("data_through"), (af.get("meta") or {}).get("data_through")) if d
    ]
    return {
        "schema": 1,
        "as_of": max(dates) if dates else None,
        "factor_lab": lab,
        "alphaforge": af,
        "probe": dict(PROBE),
    }
