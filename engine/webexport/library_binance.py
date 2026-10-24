"""The ``binance`` block of the Factor library page (webexport/library.py): the library
computed and screened on Binance's public data for the same 20 coins
(engine/libfactors.py --source binance).

It reads only ``<config.DATA>/library_on_engine/binance/{status.parquet, screen_g.parquet,
crosscheck.parquet, manifest.json}``, never the ``values/`` shards. It computes the counts
again from the files and compares them with the manifest; a difference raises DataMismatch.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from webexport.common import DataMismatch, rnd
from webexport.library_catalog import T_DP, _t
from webexport.library_engine import (
    ENGINE_QS,
    MEAN_DP,
    Q_DP,
    REL_ENGINE_DIR,
    TOP_MAX,
    TOP_Q,
    _bh,
    _code_changed,
    _engine_zones,
    _qtag,
    _sci,
    _sha256,
    _signal_info,
)

# ---- the same library on Binance's public data for the same 20 coins (libfactors.py --source
# binance). Binance is a data source only; read-only like the engine block (library_engine.py).
BINANCE_SUBDIR = "binance"  # <library_on_engine>/binance
REL_BINANCE_DIR = REL_ENGINE_DIR + "/" + BINANCE_SUBDIR
BINANCE_FILES = ("status.parquet", "screen_g.parquet", "crosscheck.parquet", "manifest.json")
BINANCE_COMMANDS = [
    "cd library",
    "# network for this command only, one coin per call (the loop over all 20 is in scripts/merge_panels.py)",
    'FACTORLAB_OFFLINE=0 ../engine/.venv/bin/python -B -m factorlab download --symbols BTCUSDT --start 2022-01-01 '
    '--end "<last closed day> 23:59:59" --interval 1d --market um --metrics-archive --out data/binance_um_1d_parts/BTCUSDT',
    "../engine/.venv/bin/python -B scripts/merge_panels.py      # offline: data/binance_um_1d_parts/ -> data/binance_um_1d/",
    "cd ../engine",
    ".venv/bin/python -B libfactors.py --source binance --compute",
    ".venv/bin/python -B libfactors.py --source binance --screen",
    ".venv/bin/python -B export_web.py --only library",
]
# the data terms of the Binance block (README.md, Data terms): the daily metrics come from the archive
BINANCE_TERMS = (
    "Source: Binance public USD-M market data. The daily metrics (open interest, long/short ratios, "
    "taker ratio) come from the Binance Vision archive (data.binance.vision), CC BY-NC-SA 4.0; credit: "
    "Binance Vision."
)
# why a definition is in the new family (libfactors.ORIGINS, first match), with the reason OKX could not screen it
BINANCE_ORIGINS = ("binance_only_fields", "oi", "vwap", "other")
BINANCE_ORIGIN_LABELS = {
    "binance_only_fields": (
        "Fields the engine's OKX data does not have",
        "premium index, trade counts, taker flow or long/short ratios: not computable on OKX's data",
    ),
    "oi": (
        "Open interest",
        "OKX's open-interest history starts after the generator zone, so none reached the OKX screen",
    ),
    "vwap": (
        "VWAP",
        "vwap equals close on OKX's data (its quote volume is close x volume), so these were kept out there",
    ),
    "other": ("Other", "computable on OKX but not screened there (too few weekly ICs or no value on G)"),
}
# what a new-family factor gains on Binance (libfactors.NEW_FIELDS): a field OKX lacks, open interest
# over G, a real VWAP; (label, how the Binance panel builds it)
BINANCE_NEW_FIELDS = {
    "premium": ("premium index (perpetual price against its index)", "close of the daily premium-index kline"),
    "trades": ("number of trades per bar", "daily kline"),
    "taker_buy_volume": ("taker buy volume, base units", "daily kline"),
    "taker_buy_quote_volume": ("taker buy volume, USDT", "daily kline"),
    "taker_buy_sell_ratio": (
        "taker buy/sell volume ratio",
        "last available 5-minute snapshot of the bar: the ratio of those 5 minutes, not of the day",
    ),
    "long_short_ratio": ("long/short account ratio", "last available 5-minute snapshot of the bar"),
    "top_long_short_ratio": ("top-trader long/short position ratio", "last available 5-minute snapshot of the bar"),
    "open_interest": ("open interest, coins", "last available 5-minute snapshot of the bar"),
    "open_interest_value": ("open interest, USDT", "last available 5-minute snapshot of the bar"),
    "vwap": ("VWAP = quote volume / volume", "daily kline; a real VWAP here, equal to close on OKX's data"),
}
# the label-free cleaning rules of the Binance adapter (libfactors.CLEANING_RULES, in its order)
CLEANING_LABELS = {
    "open_interest_nonpositive": "open interest or its value ≤ 0 (both set missing)",
    "volume_vwap_outside_low_high": "volume whose average price (quote / base volume) leaves the bar's low-high range",
    "taker_buy_volume_vwap_outside_low_high": "taker buy volume whose average price leaves the bar's low-high range",
}
BINANCE_STATS = (("ric", 0.05), ("ric", 0.10), ("ic", 0.05), ("ic", 0.10))
# the manifest's panels.provenance fields the page shows (other fields stay in the manifest)
BINANCE_PROVENANCE_KEYS = ("use", "sources", "bars", "download", "merge", "update", "library_version", "appends")
RHO_DP = 3
# the notes the page prints over the Binance block (also pinned by test_export_library.py)
BINANCE_NOTES = [
    "Binance is a data source only: its public USD-M market data for the same {symbols} coins, from {first} to "
    "{last}. Each update adds the new closed days. Nothing here trades.",
    "The labels are Factorlab's own: the 7-day forward return from the OKX label table, with the OKX screen's "
    "residual-IC regressors. Only the factor inputs come from Binance.",
    "The new family is a separate Benjamini-Hochberg family: the {new:,} definitions the OKX screen did not screen, "
    "tested only against each other ({screened:,} reach the screen).",
    "The cross-check of the {cc:,} definitions the OKX screen did screen gives agreement numbers only: no "
    "q-value on Binance. With the same labels and regressors and closely correlated values, agreement is expected. "
    "It is not an independent replication.",
    "These numbers come from the Sundays of the zone G ({g0} to {g1}). No screen reads a label after {g1}.",
    "A low q or a large t is a number. It is not a test of a book. To test a definition in a book, use its "
    "register command.",
]


# --------------------------------------------------------------------------- the same library on Binance's data


def _binance_missing(reason: str) -> dict:
    return {"available": False, "reason": reason, "source": REL_BINANCE_DIR, "commands": BINANCE_COMMANDS}


def _appended_text(appends: list) -> str | None:
    """The appends of new closed days as one line ("<first> to <last> (<n> bars, on <date>)"),
    or None when the panel holds the first download only."""
    if not appends:
        return None
    return "; ".join(
        f"{a['first_bar']} to {a['last_bar']} ({a['bars']:,} bar{'' if a['bars'] == 1 else 's'}, on "
        f"{a['appended_at'][:10]})"
        for a in appends
    )


def _cleaned_where(cells: list) -> str:
    """The cleaned cells of one rule as a short text, in date order: a date with more than
    three coins as 'date (n coins)', otherwise 'date COIN' (with the average price against the
    bar's range for the volume rules)."""
    parts, by_day = [], {}
    for w in cells:
        by_day.setdefault(w["date"], []).append(w)
    for day, ws in by_day.items():
        if len(ws) > 3:
            parts.append(f"{day} ({len(ws)} coins)")
            continue
        for w in ws:
            coin = w["symbol"].split("-")[0]
            if "average_price" in w:
                parts.append(f"{day} {coin} (average price {w['average_price']:g} against {w['low']:g}-{w['high']:g})")
            else:
                parts.append(f"{day} {coin}")
    return ", ".join(parts)


def _fields_of(s: pd.Series) -> pd.Series:
    """'a,b' -> {'a', 'b'}; '' -> empty set (status and screen_g store new_fields comma-joined)."""
    return s.fillna("").map(lambda x: frozenset(f for f in x.split(",") if f))


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    """Spearman correlation over the pairs both have: Pearson of average ranks (no scipy)."""
    m = a.notna() & b.notna()
    return None if m.sum() < 3 else float(a[m].rank().corr(b[m].rank()))


def _r4(x) -> float | None:
    return None if x is None or not np.isfinite(float(x)) else round(float(x), 4)


def input_class(raw_inputs: str, expression: str) -> str:
    """The cross-check's exclusive input class (libfactors.input_class), written out here: a
    definition that reads gap (open / previous close - 1; the library's only derived field built
    on it is gap itself) is 'gap'; otherwise by its raw inputs."""
    if re.search(r"\bgap\b", expression):
        return "gap"
    i = set(raw_inputs.split(","))
    vol, fund = bool(i & {"volume", "quote_volume"}), "funding_rate" in i
    if vol and fund:
        return "volume_and_funding"
    if vol:
        return "volume"
    if fund:
        return "funding"
    return "price_only" if i <= {"open", "high", "low", "close"} else "other"


def _binance(bdir: Path, edir: Path, fac: pd.DataFrame, families: list[str], db: Path) -> dict:
    """The library on Binance's public data for the same 20 coins (libfactors.py --source binance).

    Reads status.parquet, screen_g.parquet, crosscheck.parquet and manifest.json only (never
    values/). Every count is recomputed from the parquet files and asserted equal to the
    manifest's own, as in _engine()."""
    files = {n: bdir / n for n in BINANCE_FILES}
    if not files["status.parquet"].is_file() or not files["manifest.json"].is_file():
        return _binance_missing(f"{REL_BINANCE_DIR} has no computation")
    man = json.loads(files["manifest.json"].read_text(encoding="ascii"))
    if "compute" not in man:
        return _binance_missing(f"{REL_BINANCE_DIR}/manifest.json records no computation")
    st = pd.read_parquet(files["status.parquet"])
    if not (len(st) == len(fac) and st["factor_id"].tolist() == fac["factor_id"].tolist()):
        raise DataMismatch("binance status.parquet does not list the catalog's definitions in id order")
    mc, mp = man["compute"]["counts"], man["compute"]["pit"]
    ok, pit = st["compute"] == "ok", st["pit_check"] == "pass"
    new, cc = st["role"] == "new_family", st["role"] == "crosscheck"
    scr, oi = st["screen"], st["uses_oi"].astype(bool)
    n_cuts = len(mp["cuts"])
    pit_cells = [int(st[f"pit_cells_t{k}"].sum()) for k in range(1, n_cuts + 1)]
    pit_finite = [int(st[f"pit_finite_t{k}"].sum()) for k in range(1, n_cuts + 1)]
    n = {
        "definitions": len(st),
        "computable": int((st["group"] == "computable").sum()),
        "computed": int(ok.sum()),
        "compute_errors": int((st["compute"] == "error").sum()),
        "empty": int((ok & (st["finite_cells"] == 0)).sum()),
        "pit_pass": int(pit.sum()),
        "pit_fail": int((st["pit_check"] == "fail").sum()),
        "pit_no_finite": int((st["pit_check"] == "no_finite_cells").sum()),
        "uses_oi": int(oi.sum()),
        "uses_vwap": int(st["uses_vwap"].sum()),
        "vwap_degenerate": int(st["vwap_proxy"].sum()),
        "taker_last_5m": int(st["taker_ratio_last_5m"].sum()),
        "new_family": int(new.sum()),
        "crosscheck": int(cc.sum()),
        "new_screened": int((new & (scr == "screened")).sum()),
        "new_too_few_weeks": int((new & (scr == "too_few_weeks")).sum()),
        "new_no_g_values": int((new & (scr == "no_g_values")).sum()),
        "crosscheck_with_g": int((cc & (scr == "crosscheck")).sum()),
    }
    # the manifest's own counts must agree with the file (a mismatch fails the export)
    if (mc["definitions"], mc["groups"], mc["roles"]) != (
        n["definitions"],
        {"computable": n["computable"]},
        {"crosscheck": n["crosscheck"], "new_family": n["new_family"]},
    ):
        raise DataMismatch("binance groups or roles disagree with the manifest")
    if (mc["computed_ok"], sum(mc["compute_errors"].values()), mc["all_nan_on_sundays"]) != (
        n["computed"],
        n["compute_errors"],
        n["empty"],
    ):
        raise DataMismatch("binance compute counts disagree with the manifest")
    if (mc["uses_oi"], mc["uses_vwap"], mc["vwap_proxy_degenerate"], mc["taker_ratio_last_5m"]) != (
        n["uses_oi"],
        n["uses_vwap"],
        n["vwap_degenerate"],
        n["taker_last_5m"],
    ):
        raise DataMismatch("binance flags disagree with the manifest")
    if mc["new_family_by_origin"] != {o: int((new & (st["origin"] == o)).sum()) for o in BINANCE_ORIGINS}:
        raise DataMismatch("the new-family counts by origin disagree with the manifest")
    if mc["new_family_by_okx_screen"] != {
        k: int(v) for k, v in st.loc[new, "okx_screen"].value_counts().sort_index().items()
    }:
        raise DataMismatch("the new-family counts by OKX screen status disagree with the manifest")
    if (mp["pass"], mp["fail"], mp["no_finite_cells"]) != (n["pit_pass"], n["pit_fail"], n["pit_no_finite"]):
        raise DataMismatch("binance PIT counts disagree with the manifest")
    if (mp["compared_cells"], mp["finite_cells"]) != (pit_cells, pit_finite):
        raise DataMismatch("binance PIT cells disagree")
    oi_finite = [int((oi & (st[f"pit_finite_t{k}"] > 0)).sum()) for k in range(1, n_cuts + 1)]
    if mp["oi_factors_with_finite_cells"] != oi_finite:
        raise DataMismatch("the open-interest factors with finite PIT cells disagree with the manifest")
    if not (int(st[[f"pit_diff_cells_t{k}" for k in range(1, n_cuts + 1)]].to_numpy().sum()) == 0 or n["pit_fail"] > 0):
        raise DataMismatch("a binance PIT difference cell is set, but no factor failed the PIT check")
    # every definition is in exactly one role; the new family is what the OKX screen did not screen
    if n["new_family"] + n["crosscheck"] != n["definitions"]:
        raise DataMismatch("the new family and the cross-check do not add up to the definitions")
    if not ((st.loc[cc, "okx_screen"] == "screened").all() and not (st.loc[new, "okx_screen"] == "screened").any()):
        raise DataMismatch("the OKX screen status of the new family or the cross-check is wrong")

    panels = man["panels"]
    zones = _engine_zones(man["zones"])
    z = zones["rows"]
    okx_in = man["inputs"].get("okx_screen", {})
    okx_cur = {k: (edir / k).is_file() and _sha256(edir / k) == v for k, v in okx_in.items()}
    first, last = panels["first_bar"], panels["data_through"]
    chk = panels["checks"]
    meta = {
        "about": man.get("about"),
        "venue": panels["venue"],
        "computed_at": man["compute"]["generated_at"],
        "screened_at": man["screen"]["generated_at"] if "screen" in man else None,
        "symbols": len(panels["symbols"]),
        "benchmark": panels["benchmark"],
        "fields": panels["fields"],
        "first_bar": first,
        "data_through": last,
        "bars": (date.fromisoformat(last) - date.fromisoformat(first)).days + 1,
        "conventions": panels["conventions"],
        "provenance": {k: v for k, v in panels["provenance"].items() if k in BINANCE_PROVENANCE_KEYS},
        "catalog_current": man["inputs"]["library/catalog/factors.db"]["sha256"] == _sha256(db),
        # the OKX screen the new family and the cross-check were defined against is still the one on disk
        "okx_current": bool(okx_cur) and all(okx_cur.values()),
        "code_changed": _code_changed(man),
    }
    meta["code_current"] = not meta["code_changed"]
    # the appends of new closed days (merge_panels.py --append) and the freeze of the cleaning rules
    meta["appended"] = _appended_text(panels["provenance"].get("appends") or [])
    fz = panels.get("cleaning_freeze")
    meta["cleaning_freeze"] = (
        None if not fz else {"version": fz["version"], "sha256": fz["sha256"], "rules": len(fz["rules"])}
    )
    fund, tim, vw, okx = (
        chk["funding_vs_engine_binance_prints"],
        chk["metrics_snapshot_timing"],
        chk["vwap"],
        chk["okx_close_agreement"],
    )
    checks = {
        "funding": {
            "day_cells": fund["day_cells"],
            "differing": fund["differing_above_1e-12"],
            "max_abs_diff": fund["max_abs_diff"],
        },
        "metrics_timing": {
            "cells": tim["cells"],
            "median_d": tim["median_abs_implied_price_vs_close_D"],
            "median_d1": tim["median_abs_implied_price_vs_close_D_minus_1"],
            "p99_d": tim["p99_abs_implied_price_vs_close_D"],
        },
        "vwap": {
            "collapses": vw["collapses_onto_close"],
            "cells": vw["cells"],
            "median_abs": vw["median_abs_close_over_vwap_minus_1"],
            "inside_low_high": vw["share_inside_low_high"],
        },
        "okx_close": {
            "symbols": okx["symbols"],
            "days": okx["days"],
            "ratio_min": okx["median_price_ratio_min_max"][0],
            "ratio_max": okx["median_price_ratio_min_max"][1],
            "corr_min": okx["daily_return_corr_min_max"][0],
            "corr_max": okx["daily_return_corr_min_max"][1],
        },
        # the archive errors libfactors.py set missing before computing (panels.checks.cleaning)
        "cleaning": [
            {
                "rule": r,
                "label": CLEANING_LABELS[r],
                "fields": x["fields_set_missing"],
                "cells": x["cells"],
                "where": _cleaned_where(x["where"]),
            }
            for r, x in chk["cleaning"].items()
        ],
    }
    if [x["rule"] for x in checks["cleaning"]] != list(CLEANING_LABELS):
        raise DataMismatch("the manifest's cleaning rules changed")
    if set(chk["cleaning_after"]["cells_firing_after_cleaning"].values()) != {0}:
        raise DataMismatch("a cleaning rule still fires")
    coverage = [
        {
            "field": f,
            "cells": c["cells"],
            "first_min": (c["first_valid_min_max"] or [None, None])[0],
            "first_max": (c["first_valid_min_max"] or [None, None])[1],
        }
        for f, c in chk["coverage"].items()
    ]
    cuts = mp["cuts"]
    nf = st.loc[new, "origin"].value_counts()
    funnel = [
        {
            "key": "library",
            "label": "Definitions in the library",
            "n": n["definitions"],
            "signals": None,
            "note": f"every one computable here: the Binance panel supplies all {len(panels['fields']):d} raw fields the "
            "library defines",
        },
        {
            "key": "computed",
            "label": "Computed on the Binance panel",
            "n": n["computed"],
            "signals": None,
            "note": f"with the library's own expression evaluator, {n['compute_errors']:d} errors; {n['empty']:d} empty "
            "on every Sunday",
        },
        {
            "key": "pit",
            "label": "Point-in-time verified",
            "n": n["pit_pass"],
            "signals": None,
            "note": f"recomputed on data cut at {', '.join(cuts[:-1])} and {cuts[-1]}: every value up to each cut is "
            f"equal up to floating-point rounding (rtol {_sci(mp['rtol'])}, atol {_sci(mp['atol'])}), with at least "
            f"one finite value compared ({sum(pit_cells) / 1e6:.1f} million values compared, "
            f"{sum(pit_finite) / 1e6:.1f} million of them finite); {n['pit_fail']:d} failed, "
            f"{n['pit_no_finite']:d} had no finite value to compare; all {n['uses_oi']:,} "
            "open-interest factors have finite values at every cut",
        },
        {
            "key": "crosscheck",
            "label": "Screened on OKX: cross-checked only",
            "n": n["crosscheck"],
            "signals": None,
            "branch": "agreement only",
            "note": "the definitions the OKX screen already screened; their Binance statistics are compared with "
            "OKX's, with no new selection (last card of this block)",
        },
        {
            "key": "new_family",
            "label": "New family: not screened on OKX",
            "n": n["new_family"],
            "signals": None,
            "note": "{} read a field the engine's OKX data does not have, {} open interest, {} vwap, {} other".format(
                *tuple(f"{int(nf.get(o, 0)):,}" for o in BINANCE_ORIGINS)
            ),
        },
    ]
    out = {
        "available": True,
        "source": REL_BINANCE_DIR,
        "commands": BINANCE_COMMANDS,
        "terms": BINANCE_TERMS,
        "files": [f"{REL_BINANCE_DIR}/{x}" for x in BINANCE_FILES],
        "families": families,
        "origins": list(BINANCE_ORIGINS),
        "origin_labels": [BINANCE_ORIGIN_LABELS[o][0] for o in BINANCE_ORIGINS],
        "meta": meta,
        "checks": checks,
        "coverage": coverage,
        "counts": n,
        "funnel": funnel,
        "pit": {
            "cuts": list(cuts),
            "rtol": mp["rtol"],
            "atol": mp["atol"],
            "rule": mp["rule"],
            "cells": pit_cells,
            "finite": pit_finite,
            "oi_finite": oi_finite,
        },
        "zones": zones,
        "screen": None,
        "fields": None,
        "by_origin": None,
        "top": None,
        "crosscheck": None,
    }
    bars = panels["provenance"]["bars"]
    out["notes"] = [
        s.format(
            symbols=meta["symbols"],
            first=bars[0],
            last=bars[1],
            new=n["new_family"],
            screened=n["new_screened"],
            cc=n["crosscheck"],
            g0=z[0]["start"],
            g1=z[2]["end"],
        )
        for s in BINANCE_NOTES
    ]
    if "screen" not in man or not files["screen_g.parquet"].is_file():
        out["screen_missing"] = (
            "the Binance screen has not been run: cd engine && .venv/bin/python -B libfactors.py --source binance --screen"
        )
        return out
    g = pd.read_parquet(files["screen_g.parquet"])
    out.update(_binance_screen(man, st, g, fac, families, n))
    s = out["screen"]
    out["funnel"].append(
        {
            "key": "screened",
            "label": "Screened in the new family",
            "n": n["new_screened"],
            "signals": None,
            "note": f"with at least {s['min_weeks']:d} weekly ICs on G ({n['new_too_few_weeks']:d} have fewer, "
            f"{n['new_no_g_values']:d} no value on G); Benjamini-Hochberg over these {s['n_tests_ric']:,} "
            "alone, separately for IC and rIC",
        }
    )
    for q in sorted(ENGINE_QS, reverse=True):  # q <= 0.10, then the narrower q <= 0.05
        b = next(x for x in s["bh"] if x["stat"] == "ric" and x["q"] == q)
        out["funnel"].append(
            {
                "key": "ric_" + _qtag(q),
                "label": f"Residual IC survives Benjamini-Hochberg at q ≤ {q:.2f}",
                "n": b["factors"],
                "signals": b["signals"],
                "note": f"over the new family's {s['n_tests_ric']:,} screened factors; screening evidence on G, "
                "in sample on G_eval",
            }
        )
    rep = {(r["stat"], r["q"]): r for r in s["replication"]["rows"]}
    r05, r10 = rep[("ric", 0.05)], rep[("ric", 0.10)]
    out["funnel"].append(
        {
            "key": "fit_q05",
            "label": "Replication: selected on G_fit alone at rIC q ≤ 0.05",
            "n": r05["selected"],
            "signals": None,
            "branch": "the replication check",
            "note": "the smallest G_fit q is {:.4f}; at q ≤ 0.10, {} are selected, {} keep their sign on G_eval and {} reach "
            "|rIC t| ≥ 2 there (with no selection, {} of all screened factors keep their sign)".format(
                s["replication"]["smallest_fit_q"]["ric"],
                f"{r10['selected']:,}",
                f"{r10['same_sign']:,}",
                f"{r10['same_sign_t2']:,}",
                f"{100 * r10['base_same_sign'] / r10['base_pairs']:.0f}%",
            ),
        }
    )
    if "crosscheck" in man and files["crosscheck.parquet"].is_file():
        out["crosscheck"] = _binance_cross(man, st, pd.read_parquet(files["crosscheck.parquet"]), fac)
    else:
        out["crosscheck_missing"] = (
            "crosscheck.parquet is missing: cd engine && .venv/bin/python -B libfactors.py --source binance --screen"
        )
    return out


def _binance_screen(
    man: dict, st: pd.DataFrame, g: pd.DataFrame, fac: pd.DataFrame, families: list[str], n: dict
) -> dict:
    ms = man["screen"]
    sc = ms["counts"]
    new = st["role"] == "new_family"
    if not (
        len(g) == n["new_screened"] == sc["screened"]
        and set(g["factor_id"]) == set(st.loc[new & (st["screen"] == "screened"), "factor_id"])
    ):
        raise DataMismatch("binance screen_g.parquet is not the screened new family")
    if sc["screen_status"] != {k: int(v) for k, v in st.loc[new, "screen"].value_counts().sort_index().items()}:
        raise DataMismatch("the new-family screen status counts disagree with the manifest")
    if (sc["candidates"], sc["no_g_values"]) != (n["new_family"], n["new_no_g_values"]):
        raise DataMismatch("the new-family candidate counts disagree with the manifest")
    if not (g["okx_screen"] != "screened").all():
        raise DataMismatch("an OKX-screened factor is in the new family")
    m_weeks = re.search(r">= (\d+) weekly ICs", ms["family"])
    bh = []
    for stat in ("ic", "ric"):
        if sc[f"n_tests_{stat}"] != len(g):
            raise DataMismatch("the number of new-family screen tests disagrees with screen_g.parquet")
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
                raise DataMismatch(f"binance {stat} {tag} survivors disagree with the manifest")
            if int(g.loc[surv, f"{stat}_signal_{tag}"].nunique()) != row["signals"]:
                raise DataMismatch("the distinct signals of the binance survivors disagree with the manifest")
            bh.append(row)
    fs = sc["fit_selection"]
    replication = {"rule": fs["rule"], "rows": [], "smallest_fit_q": {}}
    for stat in ("ic", "ric"):
        fp = g[f"{stat}_fit_p"].to_numpy(dtype=float)
        fit_q = _bh(fp)
        if not np.allclose(fit_q, g[f"{stat}_fit_q"].to_numpy(dtype=float), equal_nan=True):
            raise DataMismatch(f"binance {stat} G_fit q differs")
        fm, em = g[f"{stat}_fit_mean"].to_numpy(dtype=float), g[f"{stat}_eval_mean"].to_numpy(dtype=float)
        both = np.isfinite(fm) & np.isfinite(em) & (fm != 0)
        same = both & (np.sign(em) == np.sign(fm))
        t2 = np.abs(g[f"{stat}_eval_t"].to_numpy(dtype=float)) >= 2
        if (int(np.isfinite(fp).sum()), int(both.sum()), int(same.sum())) != (
            fs[stat]["n_tests"],
            fs[stat]["base_pairs"],
            fs[stat]["base_same_sign"],
        ):
            raise DataMismatch(stat)
        replication["smallest_fit_q"][stat] = _r4(np.nanmin(fit_q))
        for q in ENGINE_QS:
            sel = fit_q <= q
            row = {
                "stat": stat,
                "q": q,
                "n_tests": int(np.isfinite(fp).sum()),
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
                raise DataMismatch(f"binance {stat} {q} G_fit selection disagrees")
            replication["rows"].append(row)
    sens = ms["sensitivity"]  # the manifest's variants, in its own order
    reg = sens["registered"]
    if (reg["screened"], reg["ric_count_q05"], reg["ric_count_q10"], reg["ric_fit_selected_q05"]) != (
        len(g),
        sc["ric_count_q05"],
        sc["ric_count_q10"],
        fs["ric"]["q05"]["selected"],
    ):
        raise DataMismatch("the registered variant is not the Binance screen")
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

    # survivor sets by origin, by new field and by family, checked against the manifest's breakdowns
    gf = _fields_of(g["new_fields"])
    sf = _fields_of(st["new_fields"])
    surv = {f"{s}_{_qtag(q)}": (g[f"{s}_q"] <= q) for s, q in BINANCE_STATS}
    for key, m in surv.items():
        b = sc["q_sets"][key]
        s_ = g[m]
        if b["by_origin"] != {k: int(v) for k, v in s_["origin"].value_counts().sort_index().items()}:
            raise DataMismatch(key)
        if b["by_family"] != {k: int(v) for k, v in s_["family"].value_counts().sort_index().items()}:
            raise DataMismatch(key)
        got = {f: int(gf[m].map(lambda x, f=f: f in x).sum()) for f in BINANCE_NEW_FIELDS}
        if b["by_new_field"] != {f: k for f, k in got.items() if k}:
            raise DataMismatch(key)
        if b["taker_ratio_last_5m"] != int(s_["taker_ratio_last_5m"].sum()):
            raise DataMismatch(key)
    if not (set().union(*gf) <= set(BINANCE_NEW_FIELDS) and set().union(*sf) <= set(BINANCE_NEW_FIELDS)):
        raise DataMismatch("a factor reads a new field that is not a Binance field")
    if sf[st["role"] == "crosscheck"].map(bool).any():
        raise DataMismatch("a cross-checked definition reads a new field")
    by_origin = []
    for o in BINANCE_ORIGINS:
        mo, go = new & (st["origin"] == o), g["origin"] == o
        row = {
            "origin": o,
            "label": BINANCE_ORIGIN_LABELS[o][0],
            "why": BINANCE_ORIGIN_LABELS[o][1],
            "candidates": int(mo.sum()),
            "screened": int(go.sum()),
            "okx_status": {k: int(v) for k, v in st.loc[mo, "okx_screen"].value_counts().sort_index().items()},
        }
        row.update({key: int((m & go).sum()) for key, m in surv.items()})
        by_origin.append(row)
    if {r["origin"]: r["candidates"] for r in by_origin} != sc["candidates_by_origin"]:
        raise DataMismatch("the candidates by origin disagree with the manifest")
    if {r["origin"]: r["screened"] for r in by_origin} != sc["screened_by_origin"]:
        raise DataMismatch("the screened factors by origin disagree with the manifest")
    fields = []
    for f, (label, src) in BINANCE_NEW_FIELDS.items():
        row = {
            "field": f,
            "label": label,
            "source": src,
            "candidates": int((new & sf.map(lambda x, f=f: f in x)).sum()),
            "screened": int(gf.map(lambda x, f=f: f in x).sum()),
        }
        row.update({key: int((m & gf.map(lambda x, f=f: f in x)).sum()) for key, m in surv.items()})
        fields.append(row)
    fields.sort(key=lambda r: (-r["ric_q05"], -r["ric_q10"], -r["screened"], r["field"]))
    by_family = []
    for fam in families:
        fg = g[g["family"] == fam]
        if not len(fg):
            continue
        rr = fg["ric_q"] <= 0.05
        by_family.append(
            {
                "family": fam,
                "candidates": int((new & (st["family"] == fam)).sum()),
                "screened": len(fg),
                "ric_q05": int(rr.sum()),
                "ric_q05_signals": int((rr & fg["ric_rep_q05"].astype(bool)).sum()),
                "ric_q10": int((fg["ric_q"] <= 0.10).sum()),
                "ic_q05": int((fg["ic_q"] <= 0.05).sum()),
            }
        )
    by_family.sort(key=lambda r: (-r["ric_q05"], -r["ric_q10"], -r["screened"], r["family"]))
    # overlap of each survivor set with the OKX screen's survivors of the same statistic and level:
    # the largest |pooled rank correlation| on Binance values, stored per factor in screen_g
    overlap = []
    for s, q in BINANCE_STATS:
        key = f"{s}_{_qtag(q)}"
        m = sc["overlap_with_okx_q_sets"][key]
        row = {
            "stat": s,
            "q": q,
            "okx_count": m["okx_count"],
            "count": int(surv[key].sum()),
            "overlapping": int((surv[key] & (g[f"okx_{key}_max_rho"] >= 0.7)).sum()),
            "overlapping_okx_values": m["overlapping_with_okx_values"],
        }
        if (row["count"], row["overlapping"]) != (m["new_family_count"], m["overlapping"]):
            raise DataMismatch(key)
        overlap.append(row)
    screen = {
        "generated_at": ms["generated_at"],
        "evidence": ms["evidence"],
        "label": ms["label"],
        "mask": ms["mask"],
        "family_rule": ms["family"],
        "min_weeks": int(m_weeks.group(1)) if m_weeks else None,
        "regressors": ms["ric_regressors"],
        "dedup": ms["dedup"],
        "precision": ms.get("precision"),
        "pooled_note": ms["pooled_g_note"],
        "n_tests_ic": sc["n_tests_ic"],
        "n_tests_ric": sc["n_tests_ric"],
        "g_sundays": sc["g_sundays"],
        "names_min": sc["names_per_sunday"]["min"],
        "names_max": sc["names_per_sunday"]["max"],
        "bh": bh,
        "replication": replication,
        "sensitivity": sensitivity,
        "by_family": by_family,
        "overlap": overlap,
    }

    # every rIC survivor at TOP_Q, ranked by |rIC t| among the new family's screened factors (ties by id)
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
        "fields": t["new_fields"].tolist(),
        "origin": [BINANCE_ORIGINS.index(x) for x in t["origin"]],
        "taker5m": [bool(x) for x in t["taker_ratio_last_5m"]],
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
        # largest |pooled rank correlation| with any OKX rIC survivor at the same level (Binance values)
        "okx_rho05": rnd(t["okx_ric_q05_max_rho"], RHO_DP),
        "okx_rho10": rnd(t["okx_ric_q10_max_rho"], RHO_DP),
        # survivors whose values start inside G_eval: no G_fit week, so no G_fit selection can reach them
        "no_fit": int(top_all["ric_fit_t"].isna().sum()),
        "no_fit_screened": int(g["ric_fit_t"].isna().sum()),
    }
    # the leading survivors as libfactors.py lists them, against the table's own order
    if [r["factor_id"] for r in ms["top_by_abs_ric_t"]] != order["factor_id"].head(
        len(ms["top_by_abs_ric_t"])
    ).tolist():
        raise DataMismatch("the top factors by |rIC t| disagree with the manifest")
    for key, rows in (("ric_q05", sc["q_sets"]["ric_q05"]), ("ric_q10", sc["q_sets"]["ric_q10"])):
        if rows["factors"] != int(surv[key].sum()):
            raise DataMismatch("a survivor count disagrees with the manifest")
    return {"screen": screen, "fields": fields, "by_origin": by_origin, "top": top}


def _binance_cross(man: dict, st: pd.DataFrame, cc: pd.DataFrame, fac: pd.DataFrame) -> dict:
    """Cross-venue agreement of the definitions the OKX screen screened: recomputed from
    crosscheck.parquet and the catalog, asserted equal to the manifest. Agreement only: no
    q-value and no survivor list on Binance."""
    mx = man["crosscheck"]
    if set(cc["factor_id"]) != set(st.loc[st["screen"] == "crosscheck", "factor_id"]):
        raise DataMismatch("crosscheck.parquet does not hold the cross-checked definitions")
    if not (len(cc) == mx["factors"] == mx["counts"]["with_g_values"]):
        raise DataMismatch("the cross-check factor counts disagree with the manifest")
    if any(c.endswith("_q") and c.startswith("binance") for c in cc.columns):
        raise DataMismatch("a Binance q-value in the cross-check")
    a, b = cc["okx_ric_t"].astype(float), cc["binance_ric_t"].astype(float)
    ai, bi = cc["okx_ic_t"].astype(float), cc["binance_ic_t"].astype(float)
    both, both_ic = a.notna() & b.notna(), ai.notna() & bi.notna()
    rho = cc["value_rho"].astype(float)
    fr = rho.notna()
    cat = fac.set_index("factor_id")
    cls = pd.Series(
        [input_class(cat.at[f, "raw_inputs"], cat.at[f, "expression"]) for f in cc["factor_id"]], index=cc.index
    )
    # definitions that read funding_rate: before the first OKX print the engine's funding is
    # Binance's own archive (data_io's splice), so on G that input is the same on both venues
    fund = pd.Series(["funding_rate" in cat.at[f, "raw_inputs"].split(",") for f in cc["factor_id"]], index=cc.index)
    s = {
        "factors": len(cc),
        "compared_ric_t": int(both.sum()),
        "ric_t_spearman": _r4(_spearman(a, b)),
        "ic_t_spearman": _r4(_spearman(ai, bi)),
        "ric_t_spearman_without_gap": _r4(_spearman(a[cls != "gap"], b[cls != "gap"])),
        "same_ric_sign": _r4((np.sign(a) == np.sign(b))[both].mean()),
        "same_ic_sign": _r4((np.sign(ai) == np.sign(bi))[both_ic].mean()),
        "rho_median": _r4(rho[fr].median()),
        "rho_p10": _r4(np.percentile(rho[fr], 10)),
        "rho_at_least_0_9": _r4((rho[fr] >= 0.9).mean()),
        "rho_below_0_5": _r4((rho[fr] < 0.5).mean()),
        "funding_readers": int(fund.sum()),
        "ric_t_spearman_without_funding": _r4(_spearman(a[~fund], b[~fund])),
        "same_ric_sign_without_funding": _r4((np.sign(a) == np.sign(b))[both & ~fund].mean()),
    }
    if (
        s["factors"],
        s["compared_ric_t"],
        s["ric_t_spearman"],
        s["ic_t_spearman"],
        s["ric_t_spearman_without_gap"],
        s["same_ric_sign"],
        s["same_ic_sign"],
    ) != (
        mx["factors"],
        mx["compared_ric_t"],
        mx["ric_t_spearman"],
        mx["ic_t_spearman"],
        mx["ric_t_spearman_without_gap"],
        mx["same_ric_sign_share"],
        mx["same_ic_sign_share"],
    ):
        raise DataMismatch("cross-check agreement disagrees with the manifest")
    if (s["funding_readers"], s["ric_t_spearman_without_funding"], s["same_ric_sign_without_funding"]) != (
        mx["funding_readers"],
        mx["ric_t_spearman_without_funding"],
        mx["same_ric_sign_share_without_funding"],
    ):
        raise DataMismatch("cross-check agreement without the funding readers disagrees with the manifest")
    # the two caveats libfactors.py writes beside these numbers, checked against the recomputed ones
    agree, fnote = mx["agreement_note"], mx["funding_note"]
    if not (
        "not an independent replication" in agree and "median {:.2f} across venues".format(s["rho_median"]) in agree
    ):
        raise DataMismatch("the agreement note of the manifest changed")
    f_ser = mx["series_agreement"]["funding"]
    if not (
        "not cross-venue evidence" in fnote
        and f"the {s['funding_readers']:,} definitions that read funding_rate" in fnote
    ):
        raise DataMismatch("the funding note of the manifest changed")
    if not ("{:.3f}".format(s["ric_t_spearman_without_funding"]) in fnote and f"{f_ser['cells']:,} coin-days" in fnote):
        raise DataMismatch("the funding note does not give the recorded figures")
    if (s["rho_median"], s["rho_p10"], s["rho_at_least_0_9"], s["rho_below_0_5"]) != (
        mx["value_rho"]["median"],
        mx["value_rho"]["p10"],
        mx["value_rho"]["share_at_least_0.9"],
        mx["value_rho"]["share_below_0.5"],
    ):
        raise DataMismatch("cross-check value correlations disagree with the manifest")
    okx_surv = []
    for q in ENGINE_QS:
        m = cc["okx_ric_q"] <= q
        same = m & both & (np.sign(a) == np.sign(b))
        row = {
            "q": q,
            "okx_count": int(m.sum()),
            "same_sign": int(same.sum()),
            "same_sign_t2": int((same & (b.abs() >= 2)).sum()),
            "median_rho": _r4(rho[m & fr].median()),
        }
        mm = mx[f"okx_ric_{_qtag(q)}"]
        if (row["okx_count"], row["same_sign"], row["same_sign_t2"], row["median_rho"]) != (
            mm["okx_count"],
            mm["binance_same_sign"],
            mm["binance_same_sign_abs_t_ge_2"],
            mm["median_value_rho"],
        ):
            raise DataMismatch(q)
        okx_surv.append(row)
    by_class = []
    for c in sorted(set(cls)):
        m = cls == c
        row = {
            "cls": c,
            "factors": int(m.sum()),
            "ric_t_spearman": _r4(_spearman(a[m], b[m])),
            "same_ric_sign": _r4((np.sign(a) == np.sign(b))[m & both].mean()),
            "median_rho": _r4(rho[m & fr].median()),
            "funding_readers": int((m & fund).sum()),
        }
        mm = mx["by_input_class"][c]
        if (
            row["factors"],
            row["ric_t_spearman"],
            row["same_ric_sign"],
            row["median_rho"],
            row["funding_readers"],
        ) != (
            mm["factors"],
            mm["ric_t_spearman"],
            mm["same_ric_sign_share"],
            mm["median_value_rho"],
            mm["funding_readers"],
        ):
            raise DataMismatch(c)
        by_class.append(row)
    if set(mx["by_input_class"]) != set(cls):
        raise DataMismatch("the input classes disagree with the manifest")
    by_class.sort(key=lambda r: -r["factors"])
    # the largest |OKX rIC t - Binance rIC t| gaps, in libfactors.py's order; the cause text is its own
    diff = np.where(both, (a - b).abs(), -1.0)
    order = np.argsort(-diff, kind="stable")
    rows = mx["largest_disagreements"]
    if [r["factor_id"] for r in rows] != [cc["factor_id"].iloc[i] for i in order[: len(rows)]]:
        raise DataMismatch("the largest disagreements are not in order")
    head = order[:50][diff[order[:50]] >= 0]
    if mx["largest_50_disagreements_by_class"] != {
        k: int(v) for k, v in cls.iloc[head].value_counts().sort_index().items()
    }:
        raise DataMismatch("the largest disagreements by input class disagree with the manifest")
    look = cc.set_index("factor_id")
    disagreements = []
    for r in rows:
        x = look.loc[r["factor_id"]]
        if cls[cc["factor_id"] == r["factor_id"]].iloc[0] != r["input_class"]:
            raise DataMismatch("the input class of a disagreement row is wrong")
        disagreements.append(
            {
                "id": r["factor_id"],
                "name": r["name"],
                "family": r["family"],
                "cls": r["input_class"],
                "okx_ric_t": _t(x["okx_ric_t"]),
                "binance_ric_t": _t(x["binance_ric_t"]),
                "rho": rnd(x["value_rho"], RHO_DP),
                "okx_weeks": int(x["okx_ic_n"]),
                "binance_weeks": int(x["binance_ic_n"]),
                "cause": r["cause"],
            }
        )
    series = [
        {
            "series": k,
            "cells": v["cells"],
            "median_abs_okx": v["median_abs_okx"],
            "median_abs_binance": v["median_abs_binance"],
            "corr": v["daily_corr"],
            "spearman": v["daily_spearman"],
            "identical": v.get("identical_share"),
        }
        for k, v in mx.get("series_agreement", {}).items()
    ]
    return {
        "rule": mx["rule"],
        "set": mx["set"],
        "agreement_note": agree,
        "funding_note": fnote,
        "summary": s,
        "okx_levels": okx_surv,
        "by_class": by_class,
        "series": series,
        "disagreements": disagreements,
    }
