"""Tests for the search tools section of the web export (webexport/research.py).

Script-style like the other engine tests:  python -B test_export_research.py

Every key number is recomputed here INDEPENDENTLY from the source files with plain
pandas/json (no factor_lab import) and compared with build(): the factor-lab q-values and
p_bonferroni (own Benjamini-Hochberg), the numbers of each run, and the AlphaForge counts,
formulas and cycle lines of a fixture zoo (built in a temporary data folder, with the switch
on in a child process). Also checks the numbers-only rule, the size budget, privacy, the page
in node and that build() leaves engine/data byte-identical. Reads engine data read-only;
writes nothing there. Without the factor-lab registry the suite prints SKIP and exits 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from webexport import common as C
from webexport import research as R

DATA = C.DATA
ENGINE = Path(__file__).resolve().parent
WEB = ENGINE.parent / "web"
# generic patterns only
PERSONAL = ("/Users/", "/home/", "Mach-O", "macOS-")
# the inputs the whole suite needs; without one of them the suite prints SKIP
REQUIRED = ("research/factor_registry.parquet",)
KEYS = {"schema", "as_of", "factor_lab", "alphaforge", "probe"}
# the words that a search tool must not write (numbers only): verdicts, tiers, stages, gate-pass words
VERDICT_WORDS = re.compile(
    r"\b(useful|useless|unproven|effective|survivors?|promotable|novel|confirmed|retired|probation|"
    r"healthy|honesty|tiers?|stages?)\b"
    r"|\bpass(es|ed)?\s+(the\s+|all\s+|every\s+)?(gates?|screen)\b|\bgates?\s+pass(es|ed)?\b|forward-only",
    re.IGNORECASE,
)


def missing_inputs() -> list[str]:
    return [r for r in REQUIRED if not (DATA / r).is_file()]


_CACHE: dict = {}


def payload() -> dict:
    if "p" not in _CACHE:
        _CACHE["p"] = json.loads(C.dumps(R.build()))
    return _CACHE["p"]


def close(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def rel(a, b, r=1e-3):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= r * max(abs(float(b)), 1e-12)


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
    return [t for t in texts if VERDICT_WORDS.search(str(t).replace("_", " "))]


# --------------------------------------------------------------------------- independent re-implementations


def bh_q(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values, written from the definition: q_(k) = min over j >= k of p_(j) n / j."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p, kind="stable")
    q_sorted = np.empty(n)
    run = 1.0
    for k in range(n - 1, -1, -1):
        run = min(run, p[order[k]] * n / (k + 1))
        q_sorted[k] = run
    out = np.empty(n)
    out[order] = q_sorted
    return out


def family_q(reg: pd.DataFrame, upto: str) -> tuple[dict, int]:
    """{factor: q} over the latest p of each distinct spec in the registry up to the run `upto`."""
    r = reg[reg["run_date"] <= upto].sort_values("run_date", kind="stable")
    latest = r.groupby("factor")["p"].last()
    q = bh_q(latest.to_numpy(dtype=float))
    return dict(zip(latest.index, q)), int(len(latest))


def tree_digest(*rels: str) -> dict:
    out = {}
    for rel_ in rels:
        root = DATA / rel_
        for f in sorted(root.rglob("*")):
            if f.is_file():
                st = f.stat()
                out[str(f.relative_to(DATA))] = (st.st_size, st.st_mtime_ns, hashlib.sha256(f.read_bytes()).hexdigest())
    return out


# --------------------------------------------------------------------------- an AlphaForge fixture

_ZOO = {
    "format": 3,
    "run_id": "2026-07-05T00:00:00Z",
    "data_through": "2026-07-04",
    "fields": ["returns", "range"],
    "label": "fwd_ret_7 = close(D+7)/close(D) - 1 stamped on D (the label table)",
    "settings": {"max_factors": 6, "mine_every_days": 7, "forward_window_weeks": 52},
    "factors": [
        {
            "name": "af_01",
            "expr": "neg(ts_std(returns, 20))",
            "selected_on": "2025-06-01",
            "cohort": 1,
            "active": False,
            "left_on": "2026-06-07",
            "window": {"weeks": 52, "first_sunday": "2025-06-08", "last_sunday": "2026-05-31"},
            "forward": {"weeks": 52, "ic": 0.0123456, "t": 1.234567, "p": 0.2201, "series": [["2025-06-08", 0.1]]},
            "gates": {"valid_icir": 0.71, "valid_t": 1.9, "persistence": 0.81, "coverage": 0.97},
            "library": {"nearest": "CF00006", "corr": -0.31, "max_abs": 0.31},
        },
        {
            "name": "af_02",
            "expr": "ts_mean(range, 10)",
            "selected_on": "2026-07-05",
            "cohort": 2,
            "active": True,
            "left_on": None,
            "window": {"weeks": 52, "first_sunday": "2026-07-12", "last_sunday": "2027-07-04"},
            "forward": {"weeks": 0, "ic": None, "t": None, "p": None, "series": []},
            "gates": {
                "valid_icir": 0.6,
                "valid_t": 1.6,
                "persistence": 0.9,
                "coverage": 0.95,
                "max_incumbent_corr": 0.12,
                "max_library_corr": 0.44,
            },
        },
    ],
    "cohorts": [
        {"cohort": 1, "selected_on": "2025-06-01", "run_id": "r1", "added": ["af_01"], "n_evaluations": 500},
        {
            "cohort": 2,
            "selected_on": "2026-07-05",
            "run_id": "r2",
            "added": ["af_02"],
            "n_evaluations": 600,
            "hall_of_fame": 40,
            "n_valid_hof": 30,
            "n_valid_rule": 9,
            "n_low_corr": 5,
            "n_uncorrelated": 2,
            "n_selected": 1,
            "incumbent_counts": {"user": 1, "zoo": 0, "proxy": 3},
            "settings": {"seed": 7, "max_corr": 0.5, "min_valid_t": 1.5},
            "environment": {"platform": "should not be exported"},
        },
    ],
    "registry": {"n_distinct_hypotheses": 70, "n_expressions_searched": 1100},
}
_CYCLES = [
    # two cycle lines and, between them, a line that is not a cycle line (no "mined")
    {
        "date": "2025-06-01",
        "data_through": "2025-05-31",
        "mined": True,
        "added": ["af_01"],
        "left": [],
        "active": 1,
        "room": 5,
        "n_evaluated": 500,
        "seed": 3,
    },  # fmt: skip
    {"date": "2025-06-02"},
    {
        "date": "2026-07-05",
        "data_through": "2026-07-04",
        "mined": True,
        "added": ["af_02"],
        "left": ["af_01"],
        "active": 1,
        "room": 5,
        "n_evaluated": 600,
        "seed": 7,
    },  # fmt: skip
]

_CHILD = r"""
import json, sys
import config
config.ALPHAFORGE_ENABLED = True
from webexport import research as R, common as C
print(json.dumps(json.loads(C.dumps(R.build()))))
"""


def af_fixture() -> dict:
    """The research payload of a temporary data folder with the fixture zoo (switch on)."""
    if "af" in _CACHE:
        return _CACHE["af"]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "data" / "research"
        d.mkdir(parents=True)
        (d / "alphaforge_library.json").write_text(json.dumps(_ZOO), encoding="utf-8")
        (d / "alphaforge_cycles.jsonl").write_text("".join(json.dumps(c) + "\n" for c in _CYCLES), encoding="utf-8")
        pd.DataFrame(
            {
                "run_date": ["2026-07-05"] * 3 + ["2025-06-01"],
                "run_id": ["r2", "r2", "r2", None],
                "n_evaluated_in_run": [600, 600, 600, 500],
                "hypothesis": ["a", "b", "c", "d"],
                "valid_passed": [True, True, False, True],
                "selected": [True, False, False, False],
                "library_name": ["af_02", "(|corr| 0.62 with proxy:size_proxy)", "", ""],
            }
        ).to_parquet(d / "alphaforge_registry.parquet", index=False)
        r = subprocess.run(
            [sys.executable, "-B", "-c", _CHILD],
            cwd=ENGINE,
            env=dict(os.environ, FACTORLAB_DATA=str(Path(tmp) / "data")),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        _CACHE["af"] = json.loads(r.stdout.strip().splitlines()[-1])
    return _CACHE["af"]


# --------------------------------------------------------------------------- tests


def test_build_is_read_only():
    before = tree_digest("research")
    R.build()
    after = tree_digest("research")
    assert before == after, "build() changed files under engine/data"


def test_shape_size_and_privacy():
    p = payload()
    assert set(p) == KEYS and p["schema"] == 1, sorted(p)
    assert isinstance(p["factor_lab"], dict)
    import config

    if not config.ALPHAFORGE_ENABLED:
        assert p["alphaforge"] == {"enabled": False}, p["alphaforge"]
    text = C.render_section("research", R.build())  # raises PrivacyError on a leak
    n = len(text.encode("ascii"))
    assert n < C.SECTION_MAX_BYTES, f"research.js {n:,} bytes over the section budget"
    assert n < 250_000, f"research.js unexpectedly large: {n:,} bytes"
    hits = [s for s in PERSONAL if C.term_pattern(s).search(text)]
    assert not hits, f"personal/host strings in payload: {hits}"
    assert '"environment"' not in text and '"platform"' not in text, "host environment leaked from the zoo cohorts"
    assert "nan" not in re.sub(r'"[^"]*"', "", text.split("=", 3)[-1]).lower(), "bare NaN in JSON"
    af = C.render_section("research", af_fixture())
    assert '"environment"' not in af and "should not be exported" not in af and "free text" not in af


def test_deterministic():
    a = C.dumps(R.build())
    b = C.dumps(R.build())
    assert a == b, "build() is not deterministic"


def test_factor_lab_counts():
    raw = pd.read_parquet(DATA / "research" / "factor_registry.parquet")
    reg = raw.drop_duplicates(["run_date", "factor"], keep="last")
    m = payload()["factor_lab"]["meta"]
    assert m["n_rows_raw"] == len(raw)
    assert m["n_rows"] == len(reg)
    assert m["n_runs"] == raw["run_date"].nunique()
    assert m["n_distinct"] == raw["factor"].nunique()
    assert m["last_run"] == str(raw["run_date"].max()) and m["first_run"] == str(raw["run_date"].min())
    per = raw.groupby("run_date").size() / raw.groupby("run_date")["factor"].nunique()
    assert m["n_logged_runs"] == int(per.round().sum())
    dups = {d: int(c) for d, c in per.round().items() if c > 1}
    assert {d["run_date"]: d["copies"] for d in m["duplicates"]} == dups
    assert close(m["max_persistence_ever"], raw["persistence"].max(), 1e-4)
    keys = {"data_through", "duplicates", "first_run", "label", "last_run", "max_n_incumbents", "max_persistence_ever"}
    keys |= {"min_weeks", "n_distinct", "n_logged_runs", "n_rows", "n_rows_raw", "n_runs", "n_specs_latest"}
    assert set(m) == keys, (
        f"factor-lab meta keys: only in the payload {sorted(set(m) - keys)}; missing {sorted(keys - set(m))}"
    )
    lat = reg[reg["run_date"] == reg["run_date"].max()]
    want = None
    if "data_through" in lat.columns and lat["data_through"].notna().any():
        want = max(str(x)[:10] for x in lat["data_through"].dropna())
    assert m["data_through"] == want
    dates = [d for d in (want, (payload()["alphaforge"].get("meta") or {}).get("data_through")) if d]
    assert payload()["as_of"] == (max(dates) if dates else None)


def test_factor_lab_latest_numbers():
    """q_bh over the latest p of each distinct spec, p_bonferroni = min(1, p x distinct specs), the
    incumbent and library numbers of the registry, and the order by p."""
    raw = pd.read_parquet(DATA / "research" / "factor_registry.parquet")
    reg = raw.drop_duplicates(["run_date", "factor"], keep="last")
    last = str(reg["run_date"].max())
    lat = reg[reg["run_date"] == last].set_index("factor")
    q, nd = family_q(reg, last)
    rows = payload()["factor_lab"]["latest"]
    assert len(rows) == len(lat) == len({x["factor"] for x in rows}), "latest run must have one row per spec"
    for x in rows:
        r = lat.loc[x["factor"]]
        assert rel(x["q_bh"], q[x["factor"]]), (x["factor"], x["q_bh"], q[x["factor"]])
        assert rel(x["p_bonferroni"], min(1.0, float(r["p"]) * nd))
        assert (
            close(x["ic"], r["ic"], 1e-6)
            and close(x["t"], r["t"], 1e-4)
            and close(x["persistence"], r["persistence"], 1e-4)
        )
        assert rel(x["p"], r["p"]) and close(x["coverage"], r["coverage"], 1e-4)
        for k in ("max_incumbent_corr", "max_library_corr"):
            v = r.get(k) if k in lat.columns else None
            v = None if v is None or not np.isfinite(float(v)) else float(v)
            assert close(x[k], v, 1e-4), (k, x[k], v)
        for k in ("closest_incumbent", "closest_library"):
            v = r.get(k) if k in lat.columns else None
            assert x[k] == (v if isinstance(v, str) and v else None), (k, x[k], v)
        if "n_incumbents" in lat.columns and pd.notna(r["n_incumbents"]):
            assert x["n_incumbents"] == int(r["n_incumbents"])
        keys = {"factor", "params", "weeks", "ic", "t", "p", "q_bh", "p_bonferroni", "persistence", "turnover"}
        keys |= {"coverage", "n_incumbents", "max_incumbent_corr", "closest_incumbent", "max_library_corr"}
        keys |= {"closest_library"}
        assert set(x) == keys, (
            f"latest row keys: only in the row {sorted(set(x) - keys)}; missing {sorted(keys - set(x))}"
        )
    ps = [x["p"] for x in rows]
    assert ps == sorted(ps), "latest rows are not in p order"
    assert set(payload()["factor_lab"]) == {"meta", "latest", "runs", "history"}, sorted(payload()["factor_lab"])


def test_factor_lab_runs_and_history():
    raw = pd.read_parquet(DATA / "research" / "factor_registry.parquet")
    reg = raw.drop_duplicates(["run_date", "factor"], keep="last")
    fl = payload()["factor_lab"]
    dates = sorted(reg["run_date"].unique())
    assert fl["history"]["run_dates"] == [str(d) for d in dates]
    assert [r["run_date"] for r in fl["runs"]] == [str(d) for d in dates]
    for r in fl["runs"]:
        g = reg[reg["run_date"] == r["run_date"]]
        q, nd = family_q(reg, r["run_date"])
        assert r["specs"] == len(g) and r["n_distinct"] == nd
        assert rel(r["min_p"], g["p"].min()) and rel(r["min_q_bh"], min(q[f] for f in g["factor"]))
        assert close(r["max_abs_t"], g["t"].abs().max(), 1e-4) and r["max_weeks"] == int(g["weeks"].max())
        keys = {"run_date", "specs", "n_distinct", "copies", "min_p", "min_q_bh", "max_abs_t", "max_weeks"}
        assert set(r) == keys, f"run row keys: only in the row {sorted(set(r) - keys)}; missing {sorted(keys - set(r))}"
    hist = fl["history"]["factors"]
    assert set(hist) == set(reg["factor"])
    for f, h in hist.items():
        for key in ("ic", "t", "persistence"):
            assert len(h[key]) == len(dates), f"{f}.{key} not aligned with run_dates"
        g = reg[reg["factor"] == f].set_index("run_date")
        for i, d in enumerate(dates):
            if d in g.index:
                assert close(h["t"][i], g.at[d, "t"], 1e-4) and close(h["ic"][i], g.at[d, "ic"], 1e-6)
            else:
                assert h["t"][i] is None


def test_alphaforge_numbers_on_a_fixture():
    """With the switch on: the counts, the settings, each formula's numbers, the cycle lines (a line
    without "mined" is left out) and the mining runs, recomputed from the fixture."""
    A = af_fixture()["alphaforge"]
    assert A["enabled"] is True
    assert set(A) == {"enabled", "meta", "settings", "counts", "members", "cohorts", "cycles", "registry"}
    import config

    assert A["settings"]["forward_window_weeks"] == 52 and A["settings"]["max_factors"] == 6
    assert set(A["settings"]) == set(R.AF_SETTING_KEYS)
    for k in R.AF_SETTING_KEYS:
        assert A["settings"][k] == config.ALPHAFORGE.get(k, A["settings"][k]), k
    assert A["counts"] == {"members": 2, "active": 1, "room": 5}
    assert A["meta"]["data_through"] == "2026-07-04" and A["meta"]["last_cycle"] == "2026-07-05"
    assert A["meta"]["last_mining"] == "2026-07-05" and A["meta"]["n_cohorts"] == 2
    m1, m2 = A["members"]
    assert (m1["name"], m1["active"], m1["left_on"], m1["cohort"]) == ("af_01", False, "2026-06-07", 1)
    assert m1["window"] == {"weeks": 52, "first_sunday": "2025-06-08", "last_sunday": "2026-05-31"}
    assert (
        m1["forward"]["weeks"] == 52
        and close(m1["forward"]["ic"], 0.0123456)
        and close(m1["forward"]["t"], 1.234567, 1e-3)
    )
    assert m1["forward"]["dates"] == ["2025-06-08"] and m1["closest_library"] == "CF00006"
    assert m1["gates"]["max_incumbent_corr"] is None and close(m1["gates"]["valid_icir"], 0.71)
    assert m2["active"] is True and m2["forward"]["ic"] is None and m2["forward"]["weeks"] == 0
    assert close(m2["gates"]["max_library_corr"], 0.44)
    c2 = A["cohorts"][1]
    assert (c2["n_valid_rule"], c2["n_low_corr"], c2["n_selected"], c2["seed"]) == (9, 5, 1, 7)
    assert c2["incumbent_counts"] == {"user": 1, "zoo": 0, "proxy": 3}
    cyc = A["cycles"]
    assert [c["date"] for c in cyc] == ["2025-06-01", "2026-07-05"], "a line without mined is not exported"
    keys = {"date", "data_through", "mined", "added", "left", "active", "room", "n_evaluated", "seed"}
    assert all(set(c) == keys for c in cyc), cyc
    assert cyc[0]["active"] == 1 and cyc[0]["left"] == [] and cyc[0]["room"] == 5 and cyc[0]["seed"] == 3
    assert cyc[1]["left"] == ["af_01"] and cyc[1]["active"] == 1 and cyc[1]["room"] == 5 and cyc[1]["seed"] == 7
    assert set(A["meta"]) == {"data_through", "last_cycle", "last_mining", "n_cohorts", "fields", "label"}, A["meta"]
    runs = A["registry"]["runs"]
    assert len(runs) == 1 and runs[0]["run_id"] == "r2" and runs[0]["hall_of_fame"] == 3
    assert runs[0]["valid_rule"] == 2 and runs[0]["selected"] == 1 and runs[0]["selected_names"] == ["af_02"]
    assert A["registry"]["n_rows"] == 4 and A["registry"]["n_expressions_searched"] == 1100
    assert af_fixture()["factor_lab"] is None and af_fixture()["as_of"] == "2026-07-04"


def test_zoo_of_another_format_fails_the_section():
    """The export does not read a zoo file of another format than 3, or a file with no format.
    build() stops with ZooFormatError. The message names the file and the two formats."""
    for fmt in (2, 4, None):
        bad = {k: v for k, v in _ZOO.items() if k != "format"}
        if fmt is not None:
            bad["format"] = fmt
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "data" / "research"
            d.mkdir(parents=True)
            (d / "alphaforge_library.json").write_text(json.dumps(bad), encoding="utf-8")
            r = subprocess.run(
                [sys.executable, "-B", "-c", _CHILD],
                cwd=ENGINE,
                env=dict(os.environ, FACTORLAB_DATA=str(Path(tmp) / "data")),
                capture_output=True,
                text=True,
                timeout=300,
            )
        assert r.returncode != 0, f"a zoo file with the format {fmt!r} was exported"
        assert "ZooFormatError" in r.stderr and f"format {fmt!r}" in r.stderr, r.stderr[-1500:]
        assert "alphaforge_library.json" in r.stderr and "reads only format 3" in r.stderr, r.stderr[-1500:]


def test_alphaforge_off_is_one_key():
    """With the switch off the payload is {"enabled": false}, whatever files the folder holds."""
    import config

    if config.ALPHAFORGE_ENABLED:
        print("        (skipped: AlphaForge is on in this checkout)")
        return
    assert payload()["alphaforge"] == {"enabled": False}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "data" / "research"
        d.mkdir(parents=True)
        (d / "alphaforge_library.json").write_text(json.dumps(_ZOO), encoding="utf-8")
        code = "import json; from webexport import research as R; print(json.dumps(R.build()['alphaforge']))"
        r = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ENGINE,
            env=dict(os.environ, FACTORLAB_DATA=str(Path(tmp) / "data")),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, r.stderr[-1500:]
        assert json.loads(r.stdout.strip().splitlines()[-1]) == {"enabled": False}


def test_probe_card():
    """The probe block: where to run the command, the command and what it prints; the page shows it."""
    p = payload()["probe"]
    assert p == {
        "where": "Run this command from the repository root.",
        "command": "cd engine && .venv/bin/python -B finder.py --generator-probe --n 2000 --seed 1",
        "text": "It prints the counts of each band of the largest |corr|. It writes no file.",
    }
    js = (WEB / "pages" / "research.js").read_text(encoding="utf-8")
    assert 'id: "probe"' in js and "renderProbe(h, d.probe)" in js and "ui.code(P.command" in js


def test_payloads_are_numbers_only():
    """The payload of this data folder and of the AlphaForge fixture: no verdict, tier, stage or
    gate-pass word in a key or a text."""
    for p in (payload(), af_fixture()):
        hits = _word_hits(_texts(p))
        assert not hits, hits[:10]


def test_degrades_without_research_files():
    """A data folder without research/ still builds: factor_lab is None, alphaforge one key (off)
    or empty (on), the probe block is there."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "data"
        d.mkdir()
        code = (
            "import json; from webexport import research as R, common as C; "
            "p = json.loads(C.dumps(R.build())); print(json.dumps(p))"
        )
        env = dict(os.environ, FACTORLAB_DATA=str(d))
        out = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ENGINE,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        r = json.loads(out.stdout.strip().splitlines()[-1])
        assert set(r) == KEYS and r["factor_lab"] is None and r["as_of"] is None
        assert r["alphaforge"]["enabled"] in (False, True) and r["probe"]["command"]


def test_page_file_static_checks():
    js = (WEB / "pages" / "research.js").read_text(encoding="utf-8")
    assert 'id: "research"' in js and "FL.pages.register" in js
    assert re.search(r'id: "research",\s*title: "Search tools"', js)
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", js, flags=re.S)
    for bad in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "?.",
        "??",
        "import ",
        "export ",
        "fetch(",
    ):
        assert bad not in code, f"page uses forbidden construct {bad!r}"
    assert not re.search(r"https?://", code), "page references an external URL"
    # every node is built with FL.ui.el / FL.ui.svg (textContent only); the one raw element is the static <style>
    assert "createElement(" not in code.replace('document.createElement("style")', "")
    # numbers only: no verdict, status or gate wording, no pass/fail marks
    hits = _word_hits(re.findall(r'"([^"\n]*)"', code))
    assert not hits, hits
    for bad in ("✓", "✕"):
        assert bad not in code, f"page has the mark {bad!r}"
    # the empty states and the line for an empty incumbent set
    assert (
        '"The factor lab has not run."' in code
        and 'LAB_COMMAND = "cd engine && .venv/bin/python -B factor_lab.py"' in code
    )
    assert '"AlphaForge is off (ALPHAFORGE_ENABLED in engine/config.py)."' in code
    assert '"No incumbent: no registered test, no active zoo formula and no risk proxy."' in code
    assert "research_status(root, d)" in code and "ui.toc(root" in code
    assert "showTrend" in code and 'c.key !== "_trend"' in code and 'id: "lab-heatmap"' in code
    assert 'ui.callout("critical"' not in code and 'ui.badge("critical"' not in code
    assert not re.search(r"(?<![-\w])color:var\(--muted\)", js)
    low = js.lower()
    for word in ("go-live", "go live", "live trading", "never live"):
        assert word not in low, f"page says {word!r}"


def _node() -> str | None:
    n = os.environ.get("FACTORLAB_NODE") or os.environ.get("NODE") or shutil.which("node")
    return n if n and Path(n).exists() else None


def _render(p: dict) -> dict:
    import test_export_web as TW

    with tempfile.TemporaryDirectory() as tmp:
        web = Path(tmp) / "web"
        shutil.copytree(WEB / "assets", web / "assets")
        shutil.copytree(WEB / "pages", web / "pages")
        (web / "data").mkdir()
        C.write_section(web / "data", "research", p)
        return TW.render_page("research", web=web)


def test_page_renders_in_node():
    """The page in node: this data folder, the AlphaForge fixture, and an empty payload. Needs node."""
    if _node() is None:
        print("        (skipped the node run: set NODE or FACTORLAB_NODE)")
        return
    full = _render(payload())
    t = full["text"]
    assert "Factor lab" in t and "Latest run" in t and "q_bh" in t and "Generator probe" in t
    assert payload()["probe"]["command"] in t and "This part of the page failed" not in t
    assert {"factor-lab", "alphaforge", "probe"} <= set(full["ids"]), full["ids"]
    lat = payload()["factor_lab"]["latest"]
    assert lat[0]["factor"].replace("dif_", "").replace("_w90", "") in t
    if payload()["alphaforge"] == {"enabled": False}:
        assert "AlphaForge is off (ALPHAFORGE_ENABLED in engine/config.py)." in t
    af = _render(af_fixture())
    ta = af["text"]
    assert "af_01" in ta and "af_02" in ta and "neg(ts_std(returns, 20))" in ta and "52 / 52" in ta
    assert "1 / 6" in ta and "no (left " in ta
    assert "This part of the page failed" not in ta
    empty = _render({"schema": 1, "as_of": None, "factor_lab": None, "alphaforge": {"enabled": False},
                     "probe": payload()["probe"]})  # fmt: skip
    te = empty["text"]
    assert "The factor lab has not run." in te and "cd engine && .venv/bin/python -B factor_lab.py" in te
    assert "AlphaForge is off" in te and "This part of the page failed" not in te
    for out in (full, af, empty):
        hits = _word_hits([out["text"]])
        assert not hits, hits[0][:400] if hits else None


def test_written_file_if_present():
    f = WEB / "data" / "research.js"
    if not f.exists():
        return
    text = f.read_text(encoding="ascii")
    m = re.match(
        r'^window\.FL=window\.FL\|\|\{\};FL\.data=FL\.data\|\|\{\};FL\.data\["research"\]=(.*);\n$', text, re.S
    )
    assert m, "web/data/research.js does not use the data-file wrapper"
    data = json.loads(m.group(1))
    assert set(data) == set(payload()), "web/data/research.js has different top-level keys than build(); re-export"
    if data.get("factor_lab") and payload().get("factor_lab"):
        assert set(data["factor_lab"]["meta"]) == set(payload()["factor_lab"]["meta"]), (
            "web/data/research.js factor_lab.meta is stale; re-run export_web.py --only research"
        )
    assert not C.privacy_hits(text)


def headline() -> list[str]:
    p = payload()
    fl, A = p["factor_lab"], p["alphaforge"]
    lat = fl["latest"]
    out = [
        f"factor lab: {fl['meta']['n_distinct']} distinct specs, {fl['meta']['n_runs']} run dates "
        f"({fl['meta']['n_logged_runs']} logged), last {fl['meta']['last_run']}, {len(lat)} specs, "
        f"smallest q_bh {min((r['q_bh'] for r in lat if r['q_bh'] is not None), default=None)}"
    ]
    if A.get("enabled") is not True:
        return out + ["AlphaForge is off"]
    return out + [
        f"alphaforge: {A['counts']['active']}/{A['settings']['max_factors']} active, {A['counts']['members']} formulas"
    ]


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Run it with no arguments. It prints one PASS, FAIL or SKIP line for each test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    if missing_inputs():
        print(f"SKIP  export_research: the data folder has no {', '.join(missing_inputs())}")
        raise SystemExit(0)
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    if not failed:
        for line in headline():
            print(f"  INFO  {line}")
    print(f"\n{len(tests) - failed}/{len(tests)} export_research tests passed")
    raise SystemExit(1 if failed else 0)
