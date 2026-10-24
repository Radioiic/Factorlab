"""The ``engine`` block of the Factor library page (webexport/library.py): the library
computed and screened on Factorlab's OKX data (engine/libfactors.py).

It reads only ``<config.DATA>/library_on_engine/{status.parquet, screen_g.parquet,
manifest.json}``, never the ``values/`` shards. It computes the counts again from the two
parquet files and compares them with the manifest; a difference raises DataMismatch.
The Binance block (webexport/library_binance.py) uses the shared helpers of this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from webexport.common import ENGINE_ROOT, DataMismatch, rnd
from webexport.library_catalog import T_DP

# ---- the library on Factorlab's data (libfactors.py outputs; read-only)
ENGINE_DIR = "library_on_engine"  # under config.DATA (honours FACTORLAB_DATA)
REL_ENGINE_DIR = "engine/data/library_on_engine"  # the label the page prints, never a real path
ENGINE_FILES = ("status.parquet", "screen_g.parquet", "manifest.json")
ENGINE_COMMANDS = [
    "cd engine",
    ".venv/bin/python -B libfactors.py --compute    # classify, compute, point-in-time proof",
    ".venv/bin/python -B libfactors.py --screen     # the G-zone screen",
    ".venv/bin/python -B export_web.py --only library",
]
ENGINE_QS = (0.05, 0.10)  # the BH levels libfactors.py reports
TOP_Q = 0.10  # the table lists the rIC survivors at this level
TOP_MAX = 500  # at most this many rows per survivor table (by rank); the
# counts stay the full ones, so the file size has a bound
Q_DP, MEAN_DP = 4, 4
# raw fields the library defines and the engine's OKX data does not have (libfactors: not supplied)
FIELD_LABELS = {
    "premium": "premium index (perpetual price against its index)",
    "trades": "number of trades per bar",
    "taker_buy_volume": "taker buy volume, base units",
    "taker_buy_quote_volume": "taker buy volume, quote units",
    "taker_buy_sell_ratio": "taker buy/sell volume ratio",
    "long_short_ratio": "long/short account ratio",
    "top_long_short_ratio": "top-trader long/short ratio",
}
# the notes the page prints over the block (also pinned by test_export_library.py)
ENGINE_NOTES = [
    "These numbers come from the Sundays of the zone G ({g0} to {g1}). No screen reads a label after {g1}.",
    "q is the Benjamini-Hochberg q-value over the screened definitions.",
    "A low q or a large t is a number. It is not a test of a book.",
    "To test a definition in a book, use its register command.",
]


# --------------------------------------------------------------------------- the library on Factorlab's data


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_changed(man: dict) -> list[str]:
    """The code files the computation recorded (repository-relative names in the manifest's
    ``code.files``) whose content differs today; [] when today's code produced the outputs.
    Read-only: the numbers are still the recorded ones, the page only says which code made them."""
    rec = (man.get("code") or {}).get("files") or {}
    repo = ENGINE_ROOT.parent
    out = []
    for rel, h in rec.items():
        p = repo / rel
        if rel.startswith("/") or ".." in Path(rel).parts or not p.is_file() or _sha256(p) != h:
            out.append(rel)
    return out


def _plain_oi(text):
    """'OKX rubik ...' -> 'OKX open-interest statistics ...' in a manifest text the page shows."""
    if not isinstance(text, str):
        return text
    return re.sub(r"\b(?:OKX )?rubik\b", "OKX open-interest statistics", text)


def _sundays(a: str, b: str) -> int:
    """Sundays from a to b inclusive (both Sundays)."""
    return (date.fromisoformat(b) - date.fromisoformat(a)).days // 7 + 1


def _no_finite_why(n: dict, last_cut: str) -> str:
    """Why the PIT check compared no finite value. Most of these factors are empty. An
    open-interest factor can also get its first value after the last cut: it is not empty,
    but no cut reaches it."""
    late = n["pit_no_finite"] - n["empty"]
    if late <= 0:
        return "the empty ones above"
    return "the {} empty ones above, and {} whose first value comes after the last cut, {}".format(
        f"{n['empty']:,}",
        f"{late:,}",
        last_cut,
    )


def _sci(x: float) -> str:
    """1e-09 -> '1e-9' (a tolerance as the docs write it)."""
    m, e = f"{x:.0e}".split("e")
    return f"{m}e{int(e)}"


def _qtag(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up q-values over the finite entries of p (NaN elsewhere),
    written out here so the export recomputes libfactors.py's G_fit selection independently."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    fin = np.flatnonzero(np.isfinite(p))
    if fin.size:
        order = fin[np.argsort(p[fin], kind="stable")]
        m = fin.size
        raw = p[order] * m / np.arange(1, m + 1)
        q[order] = np.minimum(np.minimum.accumulate(raw[::-1])[::-1], 1.0)
    return q


def _engine_missing(reason: str) -> dict:
    return {"available": False, "reason": reason, "source": REL_ENGINE_DIR, "commands": ENGINE_COMMANDS}


def _engine_zones(z: dict) -> dict:
    emb = z["embargo_sundays"]
    rows = [
        {
            "key": "g_fit",
            "label": "G_fit",
            "role": "screened (first part of G)",
            "start": z["g_fit_sundays"][0],
            "end": z["g_fit_sundays"][1],
        },
        {
            "key": "g_gap",
            "label": "G embargo",
            "role": "no label used",
            "start": z["g_internal_embargo_sunday"],
            "end": z["g_internal_embargo_sunday"],
        },
        {
            "key": "g_eval",
            "label": "G_eval",
            "role": "screened (second part of G)",
            "start": z["g_eval_sundays"][0],
            "end": z["g_eval_sundays"][1],
        },
        {"key": "embargo", "label": "embargo", "role": "no label used", "start": emb[0], "end": emb[1]},
        {
            "key": "i",
            "label": "I",
            "role": "not read by the screen or the probe",
            "sealed": True,
            "start": z["i_sundays"][0],
            "end": z["i_sundays"][1],
        },
    ]
    for r in rows:
        r["sundays"] = _sundays(r["start"], r["end"])
    return {
        "version": z["version"],
        "rows": rows,
        "label_cutoff_row": z["label_cutoff_row"],
        "g_eval_label_close": z["g_eval_label_close"],
        "pit_cuts": list(z["pit_cuts"]),
    }


def _engine(edir: Path, fac: pd.DataFrame, families: list[str], db: Path) -> dict:
    """The library on Factorlab's data, from status.parquet, screen_g.parquet and manifest.json.

    Every count is recomputed from the parquet files and asserted equal to the manifest's own
    count, so a stale or hand-edited file fails the export instead of reaching the page."""
    files = {n: edir / n for n in ENGINE_FILES}
    if not files["status.parquet"].is_file() or not files["manifest.json"].is_file():
        return _engine_missing(f"{REL_ENGINE_DIR} has no computation")
    man = json.loads(files["manifest.json"].read_text(encoding="ascii"))
    if "compute" not in man:
        return _engine_missing(f"{REL_ENGINE_DIR}/manifest.json records no computation")
    st = pd.read_parquet(files["status.parquet"])
    mc = man["compute"]["counts"]
    if not (len(st) == len(fac) and st["factor_id"].tolist() == fac["factor_id"].tolist()):
        raise DataMismatch("status.parquet does not list the catalog's definitions in id order")

    group = st["group"]
    ok, pit = st["compute"] == "ok", st["pit_check"] == "pass"
    oi, vw = st["oi_short_history"].astype(bool), st["vwap_proxy"].astype(bool)
    empty = ok & (st["finite_cells"] == 0)
    scr = st["screen"]
    n = {
        "definitions": len(st),
        "computable": int((group == "computable").sum()),
        "needs_premium": int((group == "needs_premium").sum()),
        "missing_fields": int((group == "missing_fields").sum()),
        "computed": int(ok.sum()),
        "compute_errors": int((st["compute"] == "error").sum()),
        "empty": int(empty.sum()),
        # why they are empty: open-interest windows longer than the OI history, or statistics of
        # vwap_dev, a series that is zero up to rounding on this data
        "empty_oi": int((empty & oi).sum()),
        "empty_vwap": int((empty & ~oi & vw).sum()),
        "empty_other": int((empty & ~oi & ~vw).sum()),
        # pass = every compared cell equal AND at least one finite cell compared
        "pit_pass": int(pit.sum()),
        "pit_fail": int((st["pit_check"] == "fail").sum()),
        "pit_no_finite": int((st["pit_check"] == "no_finite_cells").sum()),
        "oi": int(oi.sum()),
        "oi_pit_pass": int((oi & pit).sum()),
        "oi_no_g": int((oi & (scr == "no_g_values")).sum()),
        "oi_vwap": int((oi & vw).sum()),
        "oi_empty": int((oi & empty).sum()),
        "oi_base_proxy": int(st["oi_base_proxy"].astype(bool).sum()),
        "vwap_proxy": int(vw.sum()),
        "vwap_empty": int((vw & empty).sum()),
        # the screen-status partition of the verified factors (vwap proxies among them only)
        "vwap_verified": int((scr == "vwap_proxy").sum()),
        "no_g_values": int((scr == "no_g_values").sum()),
        "no_g_lookback": int((~oi & (scr == "no_g_values")).sum()),
        "too_few_weeks": int((scr == "too_few_weeks").sum()),
        "screened": int((scr == "screened").sum()),
    }
    # the manifest's own counts must agree with the files (a mismatch fails the export)
    if not (
        mc["definitions"] == n["definitions"]
        and mc["groups"] == {k: n[k] for k in ("computable", "missing_fields", "needs_premium")}
    ):
        raise DataMismatch("status groups disagree with the manifest")
    if (mc["computed_ok"], sum(mc["compute_errors"].values()), mc["all_nan_on_sundays"]) != (
        n["computed"],
        n["compute_errors"],
        n["empty"],
    ):
        raise DataMismatch("compute counts disagree with the manifest")
    if mc["all_nan_split"] != {
        "open_interest": n["empty_oi"],
        "vwap_proxy": n["empty_vwap"],
        "other": n["empty_other"],
    }:
        raise DataMismatch("empty-factor reasons disagree with the manifest")
    if (mc["computable_needing_oi"], mc["computable_vwap_proxy"], mc["computable_oi_base_proxy"]) != (
        n["oi"],
        n["vwap_proxy"],
        n["oi_base_proxy"],
    ):
        raise DataMismatch("flag counts disagree with the manifest")
    mp = man["compute"]["pit"]
    if (mp["pass"], mp["fail"], mp["no_finite_cells"]) != (n["pit_pass"], n["pit_fail"], n["pit_no_finite"]):
        raise DataMismatch("PIT counts disagree with the manifest")
    n_cuts = len(mp["cuts"])
    pit_cells = [int(st[f"pit_cells_t{k}"].sum()) for k in range(1, n_cuts + 1)]
    pit_finite = [int(st[f"pit_finite_t{k}"].sum()) for k in range(1, n_cuts + 1)]
    if mp["compared_cells"] != pit_cells:
        raise DataMismatch("PIT cells disagree")
    if mp["finite_cells"] != pit_finite:
        raise DataMismatch("PIT finite cells disagree")
    if not (int(st[[f"pit_diff_cells_t{k}" for k in range(1, n_cuts + 1)]].to_numpy().sum()) == 0 or n["pit_fail"] > 0):
        raise DataMismatch("a PIT difference cell is set, but no factor failed the PIT check")
    fin_any = st[[f"pit_finite_t{k}" for k in range(1, n_cuts + 1)]].sum(axis=1) > 0
    if not ((fin_any[pit]).all() and not fin_any[st["pit_check"] == "no_finite_cells"].any()):
        raise DataMismatch("a PIT pass without a finite compared cell")
    # every computed factor is accounted for once, in funnel order
    if n["pit_pass"] != n["screened"] + n["vwap_verified"] + n["no_g_values"] + n["too_few_weeks"]:
        raise DataMismatch("the PIT passes do not add up to the funnel rows")

    panels = man["panels"]
    m_oi = re.search(r"from (\d{4}-\d{2}-\d{2})", panels["derivations"]["open_interest_value"])
    not_supplied = [x.strip() for x in panels["derivations"]["not_supplied"].split(",")]
    # definitions that need each field the engine's OKX data does not have (a definition can need several)
    lists = [x.split(",") for x in st.loc[st["missing_fields"] != "", "missing_fields"]]
    need, only = Counter(f for fs in lists for f in fs), Counter(fs[0] for fs in lists if len(fs) == 1)
    if not (set(need) == set(not_supplied) == set(FIELD_LABELS)):
        raise DataMismatch("missing fields disagree with the manifest")
    fields = [
        {"field": f, "label": FIELD_LABELS[f], "n": int(need[f]), "only": int(only.get(f, 0))}
        for f in sorted(need, key=lambda f: (-need[f], f))
    ]

    # the first bar with open interest: the manifest field, else the date in the manifest's derivation text
    oi_from = panels.get("oi_first_bar") or (m_oi.group(1) if m_oi else None)
    cells, finite = sum(pit_cells) / 1e6, sum(pit_finite) / 1e6
    cuts = mp["cuts"]
    empty_why = (
        [f"{n['empty_oi']:,} open-interest factors whose window is longer than the OI history"] if n["empty_oi"] else []
    )
    if n["empty_vwap"]:
        empty_why.append(f"{n['empty_vwap']:,} skew or kurtosis of vwap_dev, a series that is zero up to rounding here")
    if n["empty_other"]:
        empty_why.append(f"{n['empty_other']:,} other")
    funnel = [
        {
            "key": "library",
            "label": "Definitions in the library",
            "n": n["definitions"],
            "signals": None,
            "note": "each one classified by the raw fields its formula reads",
        },
        {
            "key": "computable",
            "label": "Computable on Factorlab's data",
            "n": n["computable"],
            "signals": None,
            "note": "every input is a field Factorlab stores for its 20 OKX perpetuals; the other {} need a field "
            "it does not have (card below)".format(f"{n['definitions'] - n['computable']:,}"),
        },
        {
            "key": "computed",
            "label": "Computed",
            "n": n["computed"],
            "signals": None,
            "note": f"with the library's own expression evaluator, {n['compute_errors']:d} errors; {n['empty']:,} are "
            f"empty on every Sunday{(': ' + ' and '.join(empty_why)) if empty_why else ''}",
        },
        {
            "key": "pit",
            "label": "Point-in-time verified",
            "n": n["pit_pass"],
            "signals": None,
            "note": f"recomputed on data cut at {', '.join(cuts[:-1])} and {cuts[-1]}: every value up to each cut is "
            f"equal up to floating-point rounding (rtol {_sci(mp['rtol'])}, atol {_sci(mp['atol'])}), with at least "
            f"one finite value compared ({cells:.1f} million values compared, {finite:.1f} million of them finite); "
            f"{n['pit_no_finite']:,} had no finite value to compare ({_no_finite_why(n, cuts[-1])}), so nothing "
            f"is proven for them and they are not counted here; {n['pit_fail']:d} failed",
        },
        {
            "key": "screened",
            "label": "Screened on the generator zone G",
            "n": n["screened"],
            "signals": None,
            "note": f"of the verified, not screened: {n['vwap_verified']:,} vwap proxies, {n['no_g_values']:,} with no "
            f"value on G ({n['oi_no_g']:,} need open interest, stored only from {oi_from or 'n/a'}; "
            f"{n['no_g_lookback']:d} have lookbacks too long to reach G) and {n['too_few_weeks']:d} with too few weekly ICs",
        },
    ]
    coverage = [
        {
            "key": "premium",
            "label": "Premium index",
            "n": n["needs_premium"],
            "computed": False,
            "note": "the only missing input is the perpetual's premium index. The engine's OKX data does not have it, and a "
            "perp-versus-spot basis would be a proxy that changes the definition, so none is computed.",
        },
        {
            "key": "fields",
            "label": "Trades, taker flow or long/short ratios",
            "n": n["missing_fields"],
            "computed": False,
            "note": "trade counts, taker buy volume and the long/short account ratios are not in the engine's OKX data; "
            "the fields are listed below (a definition can need several).",
        },
        {
            "key": "oi",
            "label": "Open interest: computable, short history",
            "n": n["oi"],
            "computed": True,
            "note": f"OKX open interest is stored only from {oi_from or 'n/a'}. All {n['oi']:,} are computed; "
            f"{n['oi_pit_pass']:,} are verified point in time and {n['oi_empty']:,} are empty on every Sunday "
            "(their window is longer than the OI history), so nothing could be compared for them. None has a value "
            f"on the generator zone, so none is screened ({n['oi_vwap']:d} are also vwap proxies). Open interest in "
            f"coins is derived as the USD value / close ({n['oi_base_proxy']:,} definitions). The OKX open-interest "
            "statistics are stamped 16:00 UTC and booked on that UTC day, point in time if each is the snapshot at "
            "its stamp.",
        },
        {
            "key": "vwap",
            "label": "VWAP: computable, a proxy here",
            "n": n["vwap_proxy"],
            "computed": True,
            "note": "quote volume is close x volume here (OKX's quote volume is not stored), so vwap equals close "
            "and vwap_dev is zero up to rounding. Computed, but kept out of the screen"
            + (f"; {n['vwap_empty']:,} of them are empty on every Sunday" if n["vwap_empty"] else "")
            + ".",
        },
    ]
    meta = {
        "about": man.get("about"),
        "computed_at": man["compute"]["generated_at"],
        "screened_at": man["screen"]["generated_at"] if "screen" in man else None,
        "data_through": panels["data_through"],
        "first_bar": panels["first_bar"],
        "symbols": len(panels["symbols"]),
        "benchmark": panels["benchmark"],
        "fields": panels["fields"],
        # the page names OKX's endpoint by what it is (display text only; the manifest keeps its words)
        "derivations": {k: _plain_oi(v) for k, v in panels["derivations"].items() if k != "not_supplied"},
        "oi_from": oi_from,
        # the computation recorded the catalog's fingerprint; False means the catalog changed since
        "catalog_current": man["inputs"]["library/catalog/factors.db"]["sha256"] == _sha256(db),
        # the code files whose content changed after the computation of the outputs
        "code_changed": _code_changed(man),
    }
    meta["code_current"] = not meta["code_changed"]
    out = {
        "available": True,
        "source": REL_ENGINE_DIR,
        "commands": ENGINE_COMMANDS,
        "files": [f"{REL_ENGINE_DIR}/{x}" for x in ENGINE_FILES],
        "families": families,
        "meta": meta,
        "counts": n,
        "funnel": funnel,
        "coverage": coverage,
        "fields": fields,
        "pit": {
            "cuts": list(mp["cuts"]),
            "rtol": mp["rtol"],
            "atol": mp["atol"],
            "rule": mp["rule"],
            "cells": pit_cells,
            "finite": pit_finite,
            "oi_finite": mp["oi_factors_with_finite_cells"],
            "oi_from_last_cut_only": mp["oi_pass_from_last_cut_only"],
        },
        "zones": _engine_zones(man["zones"]),
        "screen": None,
        "top": None,
    }
    z = out["zones"]["rows"]
    out["notes"] = [s.format(g0=z[0]["start"], g1=z[2]["end"]) for s in ENGINE_NOTES]
    if "screen" not in man or not files["screen_g.parquet"].is_file():
        out["screen_missing"] = (
            "the G-zone screen has not been run: cd engine && .venv/bin/python -B libfactors.py --screen"
        )
        return out
    out.update(_engine_screen(man, st, pd.read_parquet(files["screen_g.parquet"]), fac, families, n))
    for q in sorted(ENGINE_QS, reverse=True):  # q <= 0.10, then the narrower q <= 0.05
        b = next(x for x in out["screen"]["bh"] if x["stat"] == "ric" and x["q"] == q)
        out["funnel"].append(
            {
                "key": "ric_" + _qtag(q),
                "label": f"Residual IC with Benjamini-Hochberg q ≤ {q:.2f}",
                "n": b["factors"],
                "signals": b["signals"],
                "note": "q over all {} screened definitions, on the Sundays of the zone G".format(
                    f"{out['screen']['n_tests_ric']:,}"
                ),
            }
        )
    return out


def _signal_info(g: pd.DataFrame, t: pd.DataFrame) -> dict:
    """For each shown survivor row, per BH level: the member count of its distinct signal and the id
    of the signal's representative, from every survivor of the screen (not only the shown rows)."""
    out = {}
    for lvl in ("05", "10"):
        col, rep = f"ric_signal_q{lvl}", f"ric_rep_q{lvl}"
        sv = g[g[col] >= 0]
        size = sv.groupby(col).size()
        rep_id = sv[sv[rep].astype(bool)].set_index(col)["factor_id"]
        out[f"size{lvl}"] = [None if int(v) < 0 else int(size.get(int(v), 0)) for v in t[col]]
        out[f"repid{lvl}"] = [None if int(v) < 0 else rep_id.get(int(v)) for v in t[col]]
    return out


def _engine_screen(
    man: dict, st: pd.DataFrame, g: pd.DataFrame, fac: pd.DataFrame, families: list[str], n: dict
) -> dict:
    ms = man["screen"]
    sc = ms["counts"]
    screened_ids = st.loc[st["screen"] == "screened", "factor_id"]
    if not (len(g) == n["screened"] == sc["screened"] and set(g["factor_id"]) == set(screened_ids)):
        raise DataMismatch("screen_g.parquet does not hold exactly the screened factors")
    if sc["screen_status"] != {k: int(v) for k, v in st["screen"].value_counts().sort_index().items()}:
        raise DataMismatch("the screen status counts disagree with the manifest")
    m_weeks = re.search(r">= (\d+) weekly ICs", ms["family"])
    min_weeks = int(m_weeks.group(1)) if m_weeks else None
    bh = []
    for stat in ("ic", "ric"):
        if sc[f"n_tests_{stat}"] != len(g):
            raise DataMismatch("the number of screen tests disagrees with screen_g.parquet")
        for q in ENGINE_QS:
            tag = _qtag(q)
            surv = g[f"{stat}_q"] <= q
            reps = surv & g[f"{stat}_rep_{tag}"].astype(bool)
            row = {
                "stat": stat,
                "q": q,
                "factors": int(surv.sum()),
                "signals": int(reps.sum()),
                "positive": int((surv & (g[f"{stat}_t"] > 0)).sum()),
            }
            if (row["factors"], row["signals"]) != (
                sc[f"{stat}_count_{tag}"],
                sc[f"{stat}_distinct_signals_{tag}"],
            ):
                raise DataMismatch(f"{stat} {tag} survivors disagree with the manifest")
            if int(g.loc[surv, f"{stat}_signal_{tag}"].nunique()) != row["signals"]:
                raise DataMismatch("the distinct signals of the survivors disagree with the manifest")
            bh.append(row)
    # replication the way the zone design intends: BH on the G_fit p-values alone, checked on
    # G_eval; recomputed here from screen_g.parquet and asserted equal to the manifest
    fs = sc["fit_selection"]
    replication = {"rule": fs["rule"], "rows": []}
    for stat in ("ic", "ric"):
        fit_q = _bh(g[f"{stat}_fit_p"].to_numpy(dtype=float))
        if not np.allclose(fit_q, g[f"{stat}_fit_q"].to_numpy(dtype=float), equal_nan=True):
            raise DataMismatch(f"{stat} G_fit q differs")
        fm, em = g[f"{stat}_fit_mean"].to_numpy(dtype=float), g[f"{stat}_eval_mean"].to_numpy(dtype=float)
        both = np.isfinite(fm) & np.isfinite(em) & (fm != 0)
        same = both & (np.sign(em) == np.sign(fm))
        t2 = np.abs(g[f"{stat}_eval_t"].to_numpy(dtype=float)) >= 2
        if (int(both.sum()), int(same.sum())) != (fs[stat]["base_pairs"], fs[stat]["base_same_sign"]):
            raise DataMismatch(stat)
        for q in ENGINE_QS:
            sel = fit_q <= q
            row = {
                "stat": stat,
                "q": q,
                "selected": int(sel.sum()),
                "same_sign": int((sel & same).sum()),
                "same_sign_t2": int((sel & same & t2).sum()),
                "base_pairs": int(both.sum()),
                "base_same_sign": int(same.sum()),
            }
            m = fs[stat][_qtag(q)]
            if (row["selected"], row["same_sign"], row["same_sign_t2"]) != (
                m["selected"],
                m["eval_same_sign"],
                m["eval_same_sign_t2"],
            ):
                raise DataMismatch(f"{stat} {q} G_fit selection disagrees")
            replication["rows"].append(row)
    sens = ms["sensitivity"]  # the manifest's variants, in its own order
    reg = sens["registered"]
    if (reg["screened"], reg["ric_count_q05"], reg["ric_count_q10"]) != (
        len(g),
        sc["ric_count_q05"],
        sc["ric_count_q10"],
    ):
        raise DataMismatch("the registered variant is not the screen")
    sensitivity = [
        {
            "key": k,
            "note": v["note"],
            "screened": v["screened"],
            "q05": v["ric_count_q05"],
            "q10": v["ric_count_q10"],
            "fit_selected": v["ric_fit_selected_q05"],
            "same_sign": v["eval_same_sign"],
            "same_sign_t2": v["eval_same_sign_t2"],
        }
        for k, v in sens.items()
    ]
    by_family = []
    for fam in families:
        f_st, f_g = st[st["family"] == fam], g[g["family"] == fam]
        rr = f_g["ric_q"] <= 0.05
        by_family.append(
            {
                "family": fam,
                "definitions": len(f_st),
                "computable": int((f_st["group"] == "computable").sum()),
                "screened": len(f_g),
                "ic_q05": int((f_g["ic_q"] <= 0.05).sum()),
                "ric_q05": int(rr.sum()),
                # a distinct signal counts in the family of its representative (its strongest member)...
                "ric_q05_signals": int((rr & f_g["ric_rep_q05"].astype(bool)).sum()),
                # ...so a family can hold survivors and 0 signals; this counts every signal it touches
                "ric_q05_touched": int(f_g.loc[rr, "ric_signal_q05"].nunique()),
            }
        )
    by_family.sort(key=lambda r: (-r["ric_q05"], -r["screened"], r["family"]))
    screen = {
        "generated_at": ms["generated_at"],
        "evidence": ms["evidence"],
        "label": ms["label"],
        "mask": ms["mask"],
        "family_rule": ms["family"],
        "min_weeks": min_weeks,
        "regressors": ms["ric_regressors"],
        "dedup": ms["dedup"],
        "n_tests_ic": sc["n_tests_ic"],
        "n_tests_ric": sc["n_tests_ric"],
        "g_sundays": sc["g_sundays"],
        "names_min": sc["names_per_sunday"]["min"],
        "names_max": sc["names_per_sunday"]["max"],
        "bh": bh,
        "replication": replication,
        "sensitivity": sensitivity,
        "by_family": by_family,
        "by_family_note": (
            "A distinct signal is counted once, in the family of its representative (the member "
            "with the largest |rIC t|). So a family can have definitions at q ≤ 0.05 and no signal of its own. "
            "'signals touched' counts every distinct signal with a definition at q ≤ 0.05 in the family."
        ),
        "precision": ms.get("precision"),
        "pooled_note": ms["pooled_g_note"],
    }

    # every rIC survivor at TOP_Q, ranked by |rIC t| among all screened factors (ties by id)
    order = g.assign(_a=g["ric_t"].abs()).sort_values(["_a", "factor_id"], ascending=[False, True], kind="stable")
    rank = pd.Series(np.arange(1, len(order) + 1), index=order["factor_id"].to_numpy())
    top_all = order[order["ric_q"] <= TOP_Q]
    t = top_all.head(TOP_MAX)
    expr = fac.set_index("factor_id")["expression"]
    fcode = {f: i for i, f in enumerate(families)}

    def sig(col):
        return [None if int(v) < 0 else int(v) for v in t[col]]

    top = {
        "q": TOP_Q,
        "n": len(top_all),
        "shown": len(t),
        "max_rows": TOP_MAX,
        "ranked_among": len(g),
        "n_q05": int((g["ric_q"] <= 0.05).sum()),
        **_signal_info(g, t),
        "rank": [int(rank[x]) for x in t["factor_id"]],
        "id": t["factor_id"].tolist(),
        "name": t["name"].tolist(),
        "fam": [fcode[x] for x in t["family"]],
        "expr": [expr[x] for x in t["factor_id"]],
        "ic_mean": rnd(t["ic_mean"], MEAN_DP),
        "ic_t": rnd(t["ic_t"], T_DP),
        "ic_q": rnd(t["ic_q"], Q_DP),
        "ric_mean": rnd(t["ric_mean"], MEAN_DP),
        "ric_t": rnd(t["ric_t"], T_DP),
        "ric_q": rnd(t["ric_q"], Q_DP),
        "persistence": rnd(t["persistence"], 2),
        "fit_t": rnd(t["ric_fit_t"], T_DP),
        "fit_q": rnd(t["ric_fit_q"], Q_DP),
        "eval_t": rnd(t["ric_eval_t"], T_DP),
        "sig05": sig("ric_signal_q05"),
        "rep05": [bool(x) for x in t["ric_rep_q05"]],
        "sig10": sig("ric_signal_q10"),
        "rep10": [bool(x) for x in t["ric_rep_q10"]],
    }
    return {"screen": screen, "top": top}
