"""Tests for the System section of the web dashboard (webexport/system.py).

Script-style like the other engine tests:  python -B test_export_system.py
Recomputes the key numbers independently from the source files (os.walk, pandas,
plain text parsing, update.plan()) and checks them against build(). Reads engine data
read-only (it hashes engine/data before and after build() to prove it) and writes nothing.
Without the perp bars the suite prints SKIP and exits 0.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import config
import update as U
from webexport import common as C
from webexport import system as S

ENGINE = Path(__file__).resolve().parent
REPO = ENGINE.parent
DATA = Path(config.DATA)
WEB_SYSTEM = REPO / "web" / "data" / "system.js"
PAGE = REPO / "web" / "pages" / "system.js"
WRAP_RE = re.compile(r'^window\.FL=window\.FL\|\|\{\};FL\.data=FL\.data\|\|\{\};FL\.data\["system"\]=(.*);\n$', re.S)
# generic patterns only
PERSONAL = ("/Users/", "/home/")
# the inputs the whole suite needs; without one of them the suite prints SKIP
REQUIRED = ("raw/ohlcv",)
# outputs that only some data folders have: the research files, the trial ledger, the library screen
OPTIONAL_PATHS = (
    "data/research/",
    "data/trials/",
    "data/library_on_engine/",
    "data/labels/",
    "data/demo.json",
)
# the step names of the update plan of a finished build (the final step list of update.py)
FINAL_STEPS = (
    "demo data",
    "OKX download",
    "Binance download",
    "Binance merge",
    "Binance append",
    "labels",
    "factor tests, weekly log",
    "library catalog",
    "library on OKX data, compute",
    "library on OKX data, screen",
    "library on Binance data, compute",
    "library on Binance data, screen",
    "factor lab",
    "AlphaForge cycle",
    "dashboard",
)


def missing_inputs() -> list[str]:
    # config.py makes the empty folder raw/ohlcv on import, so look for its bar files
    return [r for r in REQUIRED if not any((DATA / r).glob("*.parquet"))]


_CACHE: dict = {}


def built() -> dict:
    if "d" not in _CACHE:
        _CACHE["d"] = json.loads(C.dumps(S.build()))
    return _CACHE["d"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def _newest_bar() -> str | None:
    bars = [pd.read_parquet(p, columns=["ts"])["ts"] for p in (DATA / "raw" / "ohlcv").glob("*.parquet")]
    bars = [b for b in bars if len(b)]
    return str(pd.to_datetime(max(b.max() for b in bars), utc=True).date()) if bars else None


# --------------------------------------------------------------------------- tests


def test_build_is_read_only_and_deterministic():
    def snapshot():
        out = {}
        for root, _, files in os.walk(DATA):
            for f in files:
                p = Path(root) / f
                st = p.stat()
                out[str(p.relative_to(DATA))] = (st.st_size, st.st_mtime_ns, hashlib.sha1(p.read_bytes()).hexdigest())
        return out

    before = snapshot()
    a = json.loads(C.dumps(S.build()))
    b = json.loads(C.dumps(S.build()))
    after = snapshot()
    assert before == after, "build() changed something under engine/data"
    assert a == b, "build() is not deterministic"
    _CACHE["d"] = a


def test_snapshot_date_is_the_newest_bar():
    d = built()["snapshot"]
    # the version row: the text of the file VERSION, the one version source
    assert d["version"] == (REPO / "VERSION").read_text(encoding="ascii").strip()
    # the data-through date is the newest perp bar of any coin (as meta.data_through)
    assert d["data_through"] == d["newest_raw_bar"] == _newest_bar()
    # always the checkout's own engine/logs, never the exporting shell's FACTORLAB_LOGS
    assert d["logs"] == {"dir": "engine/logs", "update_log": (REPO / "engine" / "logs" / "update.log").is_file()}
    assert set(d) == {"version", "data_through", "newest_raw_bar", "data_root", "logs", "build"}, sorted(d)


def test_published_labels_never_name_a_local_override():
    """Test runs point FACTORLAB_LOGS (and often FACTORLAB_DATA) at git-ignored scratch folders.
    That must not change what is published: the logs label is always engine/logs, the build log
    and the logs group come from the checkout's own engine/logs (an override is neither read nor
    named), the data root of any other copy is a neutral phrase, and no scratch path reaches the page."""
    import tempfile

    own = REPO / "engine" / "logs"
    saved = config.LOGS
    with tempfile.TemporaryDirectory(prefix="sys_logs_") as tmp:
        fake = Path(tmp)
        (fake / "update.log").write_text("2026-10-02T00:20:00Z  labels  ok  1s\n")
        (fake / "other.log").write_text("x\n")
        config.LOGS = fake
        try:
            b = S._build_log()
            g = S._logs_group()
        finally:
            config.LOGS = saved
    assert S.LOGS_LABEL == "engine/logs" and S.default_logs_dir() == own
    assert b["file"] == "engine/logs/update.log" and b["present"] == (own / "update.log").is_file()
    assert g["path"] == "engine/logs/" and "other.log" not in [i["name"] for i in g["items"]], g["items"]
    assert S._data_root_label(ENGINE / "data") == "engine/data"
    scratch = REPO / "engine" / ".tmp" / "x" / "data"
    assert S._data_root_label(scratch) == "FACTORLAB_DATA override"
    assert not [s for s in _strings(built()) if ".tmp/" in s or str(REPO) in s], "a scratch path is in the build"
    if WEB_SYSTEM.exists():
        assert ".tmp/" not in WEB_SYSTEM.read_text(encoding="ascii"), "web/data/system.js names a scratch path"


def _plan_names(**state) -> list[str]:
    """The step names of update.plan() in one state of a temporary data folder (nothing runs)."""
    import tempfile
    from unittest import mock

    with tempfile.TemporaryDirectory(prefix="sys_plan_") as tmp:
        data, lib = Path(tmp) / "data", Path(tmp) / "lib"
        data.mkdir()
        if state.get("registered"):
            (data / "trials").mkdir()
            (data / "trials" / "registrations.jsonl").write_text("")
        panel, parts = lib / "panel", lib / "parts"
        if state.get("panel"):
            panel.mkdir(parents=True)
            (panel / "close.csv.gz").write_bytes(b"")
        real = U.binance_steps
        with (
            mock.patch.object(config, "DATA", data),
            mock.patch.object(config, "ALPHAFORGE_ENABLED", state.get("alphaforge", False)),
            mock.patch.object(U, "factor_lab_ran", return_value=state.get("lab_ran", False)),
            mock.patch.object(U, "panel_last_bar", return_value=date(2026, 1, 1) if state.get("panel") else None),
            mock.patch.object(U, "binance_steps", side_effect=lambda today: real(today, panel=panel, parts=parts)),
            mock.patch.object(U, "CATALOG", lib / ("catalog.db" if state.get("catalog") else "absent.db")),
        ):
            if state.get("catalog"):
                lib.mkdir(exist_ok=True)
                (lib / "catalog.db").write_bytes(b"")
            steps = U.plan(
                date(2026, 1, 12),
                offline=state.get("offline", False),
                skip_binance=state.get("skip_binance", False),
                skip_library=state.get("skip_library", False),
                demo=state.get("demo", False),
            )
    return [s.name for s in steps]


def test_update_steps_are_the_update_plan():
    """The card lists the steps of update.plan() for this data folder, in order (the plan of
    ./setup.sh, or of ./setup.sh --demo for the demo data, on the day after the newest bar).
    STEP_IO describes each step name of the final step list, and only those; each name of
    STEP_IO is a step of plan() in some state, and the plans of all states keep one order."""
    u = built()["update"]
    newest = _newest_bar()
    day = date.fromisoformat(newest) + timedelta(days=1) if newest else S.FALLBACK_DAY
    plan = U.plan(day, demo=C.demo())
    assert u["mode"] == ("demo" if C.demo() else "market") and u["day"] == str(day)
    assert [s["name"] for s in u["steps"]] == [st.name for st in plan]
    assert [s["n"] for s in u["steps"]] == list(range(1, len(plan) + 1))
    for s, st in zip(u["steps"], plan):
        io = S.STEP_IO.get(st.name)
        assert s["described"] is (io is not None), st.name
        assert s["network"] is bool(st.online) and s["ok_codes"] == list(st.ok_codes), st.name
        # the command names no path of this computer: an absolute path gives its last part
        assert not re.search(r"(^|\s)/", s["command"]) and str(REPO) not in s["command"], s["command"]
        if st.argv:
            script = next((a for a in st.argv[1:] if not a.startswith("-")), None)
            assert script is None or Path(script).name in s["command"], (s["command"], st.argv)
        if io:
            assert (s["what"], s["reads"], s["writes"], s["when"]) == (
                io["what"],
                io["reads"],
                io["writes"],
                io["when"],
            )
    assert set(S.STEP_IO) == set(FINAL_STEPS), sorted(set(S.STEP_IO) ^ set(FINAL_STEPS))
    # each described name is a step of plan() in some state, and the plans keep the order of FINAL_STEPS
    states = [
        {},
        {"demo": True},
        {"offline": True},
        {"panel": True},
        {"registered": True, "alphaforge": True, "lab_ran": False},
        {"catalog": True, "lab_ran": True},
        {"skip_library": True, "skip_binance": True},
    ]
    seen, plans = set(), []
    for st in states:
        names = _plan_names(**st)
        plans.append(names)
        seen |= set(names)
    assert set(S.STEP_IO) <= seen, sorted(set(S.STEP_IO) - seen)
    # one order: two steps that are in two plans have the same order in both
    for a in plans:
        for b in plans:
            common = [n for n in a if n in b]
            assert common == [n for n in b if n in a], (a, b)
    # the conditions of the "when" texts
    assert "factor tests, weekly log" in _plan_names(registered=True)
    assert "factor tests, weekly log" not in _plan_names()
    assert "AlphaForge cycle" in _plan_names(alphaforge=True) and "AlphaForge cycle" not in _plan_names()
    assert "factor lab" in _plan_names(lab_ran=False) and "factor lab" not in _plan_names(lab_ran=True)
    assert "OKX download" not in _plan_names(offline=True) and "OKX download" not in _plan_names(demo=True)
    assert "demo data" in _plan_names(demo=True) and "demo data" not in _plan_names()
    assert "library catalog" not in _plan_names(catalog=True) and "library catalog" in _plan_names()
    assert not [n for n in _plan_names(skip_library=True) if n.startswith("library")]
    st = next(s for s in U.plan(day, demo=C.demo()) if s.name == "dashboard")
    assert st.ok_codes == (0,)
    fac = next((s for s in u["steps"] if s["name"] == "factor tests, weekly log"), None)
    if fac:
        assert fac["ok_codes"] == [0, 3]
    sw = u["switches"]
    assert sw == {"ALPHAFORGE_ENABLED": bool(config.ALPHAFORGE_ENABLED)}


def _data_io_reads() -> dict[str, set[str]]:
    """data_io function -> the data/ paths it reads, derived from data_io.py with ast
    (C.<NAME> / "file" -> that file, a bare C.<NAME> -> that folder), closed over calls
    between data_io functions."""
    tree = ast.parse((ENGINE / "data_io.py").read_text())
    funcs = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}

    def rel(p: Path, is_dir: bool) -> str:
        return "data/" + Path(p).resolve().relative_to(DATA.resolve()).as_posix() + ("/" if is_dir else "")

    direct, calls = {}, {}
    for name, f in funcs.items():
        paths, in_binop = set(), set()
        for n in ast.walk(f):
            if (
                isinstance(n, ast.BinOp)
                and isinstance(n.op, ast.Div)
                and isinstance(n.left, ast.Attribute)
                and isinstance(n.left.value, ast.Name)
                and n.left.value.id == "C"
                and isinstance(n.right, ast.Constant)
                and isinstance(n.right.value, str)
            ):
                paths.add(rel(Path(getattr(config, n.left.attr)) / n.right.value, False))
                in_binop.add(id(n.left))
        for n in ast.walk(f):
            if (
                isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name)
                and n.value.id == "C"
                and id(n) not in in_binop
                and isinstance(getattr(config, n.attr, None), Path)
            ):
                paths.add(rel(getattr(config, n.attr), True))
        direct[name] = paths
        calls[name] = {
            n.func.id
            for n in ast.walk(f)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs
        }
    out = {}
    for name in funcs:
        seen, todo, acc = set(), [name], set()
        while todo:
            x = todo.pop()
            if x in seen:
                continue
            seen.add(x)
            acc |= direct[x]
            todo += list(calls[x])
        out[name] = acc
    return out


def _calls_data_io(src: str) -> set[str]:
    return set(re.findall(r"\bdata_io\.([a-z_]+)\(", src))


# the source files of the steps that run an engine script: the data_io loaders they call are the
# reads of these steps (the two steps of libfactors.py share one file, so they share the check)
STEP_SOURCES = {
    ("labels",): ("labels.py",),
    ("factor tests, weekly log",): ("trials.py", "userfactor.py", "book.py"),
    ("library on OKX data, compute", "library on OKX data, screen"): ("libfactors.py",),
    ("factor lab",): ("factor_lab.py",),
    ("AlphaForge cycle",): ("alphaforge_bridge.py",),
}


def test_step_reads_match_the_data_io_loaders_each_script_calls():
    """Each raw-data or label read that a step makes through data_io is listed in its reads (STEP_IO)."""
    loaders = _data_io_reads()
    assert loaders["daily_funding"] >= {"data/raw/funding/", "data/raw/funding_binanceusdm/"}
    # label_table reads labels.path() (data/labels/labels.parquet), which the ast scan does not see
    import labels

    assert labels.path() == Path(config.LABELS) / "labels.parquet"
    loaders["label_table"] = loaders.get("label_table", set()) | {"data/labels/labels.parquet"}
    for names, files in STEP_SOURCES.items():
        fns = set().union(*(_calls_data_io((ENGINE / f).read_text()) for f in files))
        derived = set().union(*(loaders[f] for f in fns if f in loaders)) if fns else set()
        listed = set().union(*(S.STEP_IO[n]["reads"] for n in names))
        name = " / ".join(names)
        # a folder in the list covers the files in it
        missing = {r for r in derived if not any(r == x or (x.endswith("/") and r.startswith(x)) for x in listed)}
        assert not missing, f"{name}: reads via data_io not listed: {sorted(missing)}"


def test_every_written_path_exists_in_the_snapshot():
    """Each data path that a step of this data folder or a program names exists (or is optional)."""
    steps = [s for s in built()["update"]["steps"] if s["described"]]
    paths = [w for s in steps for w in s["writes"] + s["reads"] if w.startswith("data/")]
    paths += [w for p in built()["programs"] for w in p["writes"] if w.startswith("data/")]
    assert paths
    for w in paths:
        if w.startswith(OPTIONAL_PATHS):
            continue  # not every data folder has these outputs
        target = DATA / w[len("data/") :]
        if w == "data/":
            continue
        assert target.exists(), f"described path does not exist in the snapshot: {w}"
        assert target.is_dir() == w.endswith("/"), f"file/folder mismatch for {w}"


def test_programs_are_the_engine_programs():
    """The programs of the card: each one is a file in engine/ with a command line, and the list
    names judge.py (data/trials/), labels.py and no program that is not there."""
    progs = built()["programs"]
    names = [p["script"] for p in progs]
    assert names == [
        "ingest.py",
        "labels.py",
        "libfactors.py",
        "finder.py",
        "factor_lab.py",
        "alphaforge_bridge.py",
        "judge.py",
        "export_web.py",
        "update.py",
        "run_tests.py",
        "demo_data.py",
        "db.py",
        "version.py",
    ], names
    for p in progs:
        assert (ENGINE / p["script"]).is_file(), p["script"]
        assert p["cadence"] and p["note"], p["script"]
    by = {p["script"]: p for p in progs}
    assert by["judge.py"]["writes"] == ["data/trials/"] and by["labels.py"]["writes"] == ["data/labels/labels.parquet"]
    assert by["finder.py"]["writes"] == [] and by["db.py"]["writes"] == []


def test_environment_defaults_come_from_the_code():
    env = {e["name"]: e for e in built()["environment"]}
    assert set(env) == {
        "FACTORLAB_DATA",
        "FACTORLAB_LOGS",
        "FACTORLAB_OFFLINE",
        "FACTORLAB_WRITE_PROTECTED",
        "PYTHON",
        "FACTORLAB_NODE",
        "NODE",
    }
    # each variable of the card is in the table of README.md
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    listed = {n for pair in re.findall(r"^\| `([A-Z_]+)`(?:, `([A-Z_]+)`)? \|", readme, re.M) for n in pair if n}
    assert set(env) <= listed, sorted(set(env) - listed)
    assert "${PYTHON:-}" in (REPO / "setup.sh").read_text() and env["PYTHON"]["read_in"] == ["setup.sh"]
    node_in = sorted(
        p.name for p in ENGINE.glob("test_export_*.py") if 'os.environ.get("FACTORLAB_NODE")' in p.read_text()
    )
    assert env["FACTORLAB_NODE"]["read_in"] == env["NODE"]["read_in"] == node_in
    cfg = (ENGINE / "config.py").read_text()
    assert (
        'os.environ.get("FACTORLAB_DATA", ROOT / "data")' in cfg and env["FACTORLAB_DATA"]["default"] == "engine/data"
    )
    assert (
        'os.environ.get("FACTORLAB_LOGS", ROOT / "logs")' in cfg and env["FACTORLAB_LOGS"]["default"] == "engine/logs"
    )
    assert 'os.environ.get("FACTORLAB_OFFLINE", "1")' in (ENGINE / "ingest.py").read_text()
    assert env["FACTORLAB_OFFLINE"]["default"] == "1" and env["FACTORLAB_OFFLINE"]["read_in"] == ["ingest.py"]
    assert 'os.environ.get("FACTORLAB_WRITE_PROTECTED", "0")' in cfg
    assert env["FACTORLAB_WRITE_PROTECTED"]["default"] == "0"
    assert env["FACTORLAB_WRITE_PROTECTED"]["read_in"] == ["config.py"]
    # the protected folders come from config.PROTECTED_DIRS
    assert (
        f"The protected folders are {', '.join(config.PROTECTED_DIRS)}." in env["FACTORLAB_WRITE_PROTECTED"]["meaning"]
    )
    assert not [e for e in env.values() if "{" in e["meaning"]]


def test_inventory_matches_an_independent_walk():
    inv = built()["inventory"]
    files = []
    for root, dirs, names in os.walk(DATA):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        files += [Path(root) / n for n in names if not n.startswith(".")]
    assert inv["totals"]["files"] == len(files)
    assert inv["totals"]["bytes"] == sum(f.stat().st_size for f in files)
    groups = {g["id"]: g for g in inv["groups"]}
    assert list(groups) == [g for g, _, _ in S.GROUPS]
    assert set(groups) >= {"labels", "library_on_engine", "research", "trials"}
    # a file outside every group lies in a folder (or is a file) that no group covers
    tops = {rel.split("/")[0] for _, rel, _ in S.GROUPS}
    assert all(o.rstrip("/") not in tops for o in inv["other"]), inv["other"]
    # the files of the data folder itself that a program writes (demo.json) are in no group and not in "other"
    top = {f["path"]: f for f in inv["top_files"]}
    assert set(top) == {"demo.json"}
    assert top["demo.json"]["present"] == (DATA / "demo.json").is_file() and "demo.json" not in inv["other"]
    n_top = sum(int(f["present"]) for f in inv["top_files"])
    assert sum(g["files"] for g in groups.values()) + inv["other_files"] + n_top == len(files)
    assert (
        sum(g["bytes"] for g in groups.values()) + inv["other_bytes"] + sum(f["bytes"] for f in inv["top_files"])
        == inv["totals"]["bytes"]
    )
    # a group is present when its folder exists; the optional groups appear once written
    assert inv["totals"]["groups"] == sum((DATA / rel).is_dir() for _, rel, _ in S.GROUPS)

    def date_col(df):
        for c in ("date", "ts", "run_date"):
            if c in df.columns:
                return c
        return None

    for gid, g in groups.items():
        gdir = DATA / gid
        gfiles = [f for f in files if gdir in f.parents]
        assert g["files"] == len(gfiles), gid
        assert g["bytes"] == sum(f.stat().st_size for f in gfiles), gid
        rows, firsts, lasts = 0, [], []
        for it in g["items"]:
            p = DATA / it["path"].rstrip("/")
            sub = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
            irows, ifirst, ilast, ns = None, [], [], []
            for f in sub:
                n = None
                if f.suffix == ".parquet":
                    df = pd.read_parquet(f)
                    n = len(df)
                    c = date_col(df)
                    if c and n:
                        s = pd.to_datetime(df[c], utc=True)
                        ifirst.append(str(s.min().date()))
                        ilast.append(str(s.max().date()))
                elif f.suffix == ".tsv":
                    lines = [x for x in f.read_text().splitlines() if x.strip()]
                    n = len(lines) - 1
                elif f.suffix == ".jsonl":
                    recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
                    n = len(recs)
                    ds = [r["date"] for r in recs if "date" in r]
                    if ds:
                        ifirst.append(min(ds))
                        ilast.append(max(ds))
                elif f.suffix == ".json":
                    o = json.loads(f.read_text())
                    n = len(o) if isinstance(o, list) else None
                ns.append(n)
                if p.is_file():
                    irows = n
            # a folder: the rows of all its files (parquet, tsv, jsonl records, json lists)
            if p.is_dir():
                irows = sum(x or 0 for x in ns) or None
            assert it["rows"] == irows, f"{it['path']}: rows {it['rows']} != {irows}"
            assert it["first"] == (min(ifirst) if ifirst else None), it["path"]
            assert it["last"] == (max(ilast) if ilast else None), it["path"]
            # each file of a group has its writer
            assert it["writer"] != "unknown", it["path"]
            if irows is not None:
                rows += irows
            firsts += ifirst
            lasts += ilast
        assert g["rows"] == (rows or None), f"{gid}: rows"
        assert g["first"] == (min(firsts) if firsts else None) and g["last"] == (max(lasts) if lasts else None), gid
    assert groups["raw/ohlcv"]["symbols"] == 20 and groups["raw/ohlcv"]["first"] >= config.BACKFILL_START[:10]
    assert groups["raw/ohlcv"]["last"] == built()["snapshot"]["newest_raw_bar"]
    if groups["labels"]["present"]:
        assert {i["path"]: i["cadence"] for i in groups["labels"]["items"]}["labels/labels.parquet"] == S.EACH
    # the checkout's own engine/logs: its files, outside the data folder and its totals
    lg = inv["logs"]
    own = REPO / "engine" / "logs"
    want = sorted(f.name for f in own.iterdir() if f.is_file() and not f.name.startswith(".")) if own.is_dir() else []
    assert [i["name"] for i in lg["items"]] == want and lg["path"] == "engine/logs/" and lg["outside_data"] is True
    assert lg["files"] == len(want) and lg["present"] == own.is_dir()
    for i in lg["items"]:
        assert i["writer"] == ("update.py" if i["name"] in ("update.log", "update.lock") else "unknown"), i
    # any last date after the close carries an explanation. The page compares with
    # meta.data_through: the newest perp bar of any coin (export_web.data_through()).
    through = built()["snapshot"]["newest_raw_bar"]
    for g in groups.values():
        if g["last"] and g["last"] > through:
            assert g.get("after_close"), f"{g['id']} ends after the close without an explanation"


def test_inventory_lists_a_folder_that_no_program_writes():
    """A folder of the data folder that no program of Factorlab writes is in no group: the
    inventory names it under "other" (the inventory is built on a synthetic data tree)."""
    import tempfile

    saved = S.DATA
    with tempfile.TemporaryDirectory(prefix="sys_inv_") as tmp:
        data = Path(tmp) / "data"
        for g in ("raw/ohlcv", "research", "trials"):
            (data / g).mkdir(parents=True)
        (data / "daily").mkdir()
        for day in ("2026-09-30", "2026-10-01"):
            (data / "daily" / f"{day}.json").write_text(json.dumps({"date": day}))
        (data / "old.json").write_text("{}")
        S.DATA = data
        try:
            inv = S._inventory()
        finally:
            S.DATA = saved
    assert inv["other"] == ["daily/", "old.json"] and inv["other_files"] == 3, inv["other"]
    groups = {g["id"]: g for g in inv["groups"]}
    assert "daily" not in groups and not groups["labels"]["present"] and groups["trials"]["present"]
    assert not any(g.startswith("finder") for g in groups), "no group for a writer that does not exist"
    assert all(S.GROUP_NOTES.get(g) for g in S.OPTIONAL_GROUPS)
    assert S._writer("daily/2026-10-01.json") == ("unknown", "unknown")


def test_inventory_names_the_library_writers_exactly():
    """data/library_on_engine/ (libfactors.py): only --compute writes the value shards, only --screen
    writes screen_g.parquet, and both rewrite status.parquet and manifest.json; binance/ is written by
    --source binance only. No other program writes the folder."""
    groups = {g["id"]: g for g in built()["inventory"]["groups"]}
    assert "library_on_engine" in S.OPTIONAL_GROUPS and S.GROUP_NOTES["library_on_engine"]
    writers = [p["script"] for p in built()["programs"] if any("library_on_engine" in w for w in p["writes"])]
    assert writers == ["libfactors.py"], writers
    g = groups["library_on_engine"]
    if not (DATA / "library_on_engine").is_dir():
        assert not g["present"]
        return
    want = {
        "library_on_engine/values/": "libfactors.py --compute",
        "library_on_engine/screen_g.parquet": "libfactors.py --screen",
        "library_on_engine/status.parquet": "libfactors.py --compute / --screen",
        "library_on_engine/manifest.json": "libfactors.py --compute / --screen",
    }
    want = {k: v for k, v in want.items() if (DATA / k).exists()}  # each written file, when it exists
    # the Binance panel's outputs, one folder item, when they exist
    binance = {"library_on_engine/binance/": "libfactors.py --source binance --compute / --screen"}
    items = {i["path"]: i["writer"] for i in g["items"]}
    assert {k: items.get(k) for k in want} == want, items
    assert set(items) <= set(want) | set(binance), sorted(set(items) - set(want) - set(binance))
    if (DATA / "library_on_engine" / "binance").is_dir():
        assert {k: items.get(k) for k in binance} == binance, items
    assert S._writer("library_on_engine/binance/values/values_000.parquet")[0] == binance["library_on_engine/binance/"]
    assert "binance/" in S.GROUP_NOTES["library_on_engine"]
    assert "values/" in S.GROUP_NOTES["library_on_engine"] and "--compute only" in S.GROUP_NOTES["library_on_engine"]


def test_inventory_names_the_trial_ledger_writers():
    """data/trials/ (judge.py): each chain file, head.json, the values folder and the result files
    have their judge.py command and their cadence; the group note names each chain of trials.CHAINS."""
    import trials

    w = S._writer
    assert w("trials/registrations.jsonl") == (
        "judge.py register / withdraw / record-code-change / log-week --accept-release",
        S.BY_HAND,
    )
    assert (
        w("trials/values/okx.jsonl")
        == w("trials/values/binance.jsonl")
        == ("judge.py log-week / log-values", S.FACTOR_TESTS)
    )
    assert w("trials/values") == ("judge.py log-week / log-values", S.FACTOR_TESTS)
    assert w("trials/values/file.jsonl") == ("judge.py log-values", S.BY_HAND)
    assert w("trials/weeks.jsonl") == ("judge.py log-week", S.FACTOR_TESTS)
    assert w("trials/alarms.jsonl")[1] == S.FACTOR_TESTS and w("trials/head.json")[0] == "judge.py (each write)"
    assert w("trials/results/ft-0001/historical.json")[0] == "judge.py register / log-week / verify --write-results"
    assert w("labels/labels.parquet") == ("labels.py", S.EACH) and w("demo.json") == (
        "demo_data.py --write",
        S.DEMO_BUILD,
    )
    note = S.GROUP_NOTES["trials"]
    for rel in trials.CHAINS.values():
        assert rel in note, rel
    assert "head.json" in note and "results/<id>/" in note and "append-only" not in note.split(".")[0]
    g = {x["id"]: x for x in built()["inventory"]["groups"]}["trials"]
    for it in g["items"]:
        assert it["writer"].startswith("judge.py"), it


def test_data_coverage_recomputed():
    """The coverage card: each raw dataset of each coin (rows, first, last, the usual step and the
    other steps), and the label rows of each coin, read here with plain pandas."""
    cov = built()["coverage"]
    ds = {x["id"]: x for x in cov["datasets"]}
    want_ids = ["ohlcv", "spot_ohlcv", "funding_okx", "funding_binance", "open_interest"]
    assert list(ds)[:5] == want_ids
    for (did, rel, _, _), x in zip(S.DATASETS, cov["datasets"]):
        folder = DATA / rel
        files = sorted(folder.glob("*.parquet")) if folder.exists() else []
        assert [p["symbol"] for p in x["symbols"]] == [f.stem for f in files], did
        total = 0
        for f, p in zip(files, x["symbols"]):
            ts = pd.read_parquet(f, columns=["ts"])["ts"].sort_values()
            total += len(ts)
            assert p["rows"] == len(ts), (did, f.stem)
            if len(ts):
                assert (p["first"], p["last"]) == (str(ts.iloc[0].date()), str(ts.iloc[-1].date())), (did, f.stem)
                steps = ts.diff().dropna().dt.round("1h")
                if len(steps):
                    mode = steps.mode().iloc[0]
                    assert p["step_h"] == round(mode / pd.Timedelta(hours=1), 2), (did, f.stem)
                    assert p["off_step"] == int((steps != mode).sum()), (did, f.stem)
        assert x["rows"] == total and x["n_symbols"] == len(files), did
    lab = DATA / "labels" / "labels.parquet"
    if lab.is_file():
        d = pd.read_parquet(lab, columns=["date", "symbol"])
        x = ds["labels"]
        assert x["rows"] == len(d) and x["n_symbols"] == d["symbol"].nunique()
        for p in x["symbols"]:
            sub = d[d["symbol"] == p["symbol"]]
            assert p["rows"] == len(sub), p["symbol"]
    else:
        assert "labels" not in ds
    src = PAGE.read_text()
    assert 'id: "coverage"' in src and '["coverage", "Data coverage"]' in src


def test_rules_are_the_builder_constants():
    r = built()["rules"]
    assert set(r) == {"invariants", "rules"}
    assert [x["text"] for x in r["invariants"]] == S.INVARIANTS
    assert [(x["id"], x["title"], x["why"]) for x in r["rules"]] == [tuple(x) for x in S.RULES]
    text = " ".join(S.INVARIANTS).lower()
    for word in ("utc", "open time", "static universe", "funding sign", "longs pay shorts", "research only", "api key"):
        assert word in text, word
    titles = [x["title"] for x in r["rules"]]
    assert titles == [
        "A formula value for a Sunday uses only the data up to that Sunday.",
        "A values file is not proven point in time. A forward value counts from the first daily close after its log time.",
        "The trial ledger is append-only and hash-chained.",
        "Each book pays fees, slippage and funding.",
        "BTC hold gives context only.",
    ]


def test_test_suite_counts_are_parsed_correctly():
    import run_tests

    t = built()["tests"]
    files = [p for p in run_tests.engine_suites() if not p.name.startswith("test_export_")]
    want = {p.name: len(re.findall(r"^def test_\w+\(", p.read_text(), re.M)) for p in files}
    got = {s["file"]: s["n_tests"] for s in t["engine"]}
    assert got == want, (got, want)
    assert t["engine_totals"] == {"suites": len(want), "tests": sum(want.values())}
    # the suite list comes from the test runner, not from a document
    assert t["runner"] == "engine/run_tests.py" and t["suites"] == run_tests.suite_names()
    assert {"library", "alphaforge"} <= set(t["suites"])
    assert {p.stem for p in files} <= set(t["suites"])
    # every engine suite has a component and a data mode in SUITE_INFO
    assert {p.name for p in files} <= set(S.SUITE_INFO), sorted({p.name for p in files} - set(S.SUITE_INFO))
    assert "python -B run_tests.py" in t["run_all"] and "--quick" in t["run_all"] and "--only" in t["run_all"]
    for s in t["engine"]:
        assert s["command"] == f"python -B run_tests.py --only {s['suite']}"
        assert s["scratch_copy"] is (s["suite"] in run_tests.SCRATCH_SUITES)
    # each data mode of a suite has its text in the legend
    assert {s["mode"] for s in t["engine"] if s["mode"]} <= set(t["modes"]), t["modes"]
    # other sections' exporter tests may be edited while this runs: compare a fresh parse with the files
    fresh = json.loads(C.dumps(S._tests()))["exporter"]
    ex = {
        p.name: len(re.findall(r"^def test_\w+\(", p.read_text(), re.M))
        for p in run_tests.engine_suites()
        if p.name.startswith("test_export_")
    }
    assert {s["file"]: s["n_tests"] for s in fresh} == ex
    assert set(ex) <= set(S.EXPORT_INFO), sorted(set(ex) - set(S.EXPORT_INFO))
    af_files = sorted((ENGINE / "alphaforge" / "tests").glob("test_*.py"))
    af = sum(len(re.findall(r"^\s*(?:async\s+)?def test\w*\(", p.read_text(), re.M)) for p in af_files)
    assert af_files and t["alphaforge"]["test_functions"] == af and t["alphaforge"]["files"] == len(af_files)
    assert t["alphaforge"]["command"] == "python -B run_tests.py --only alphaforge"


def test_glossary_keeps_every_line():
    g = built()["glossary"]
    text = (ENGINE / "GLOSSARY.md").read_text()
    heads = re.findall(r"^## (.+)$", text, re.M)
    assert len(g["sections"]) == len(heads) and heads
    assert [((s["num"] + ". ") if s["num"] else "") + s["title"] for s in g["sections"]] == heads
    assert g["title"] == re.search(r"^# (.+)$", text, re.M).group(1).strip()
    ex = re.search(r"DAILY P&L — (\d{4}-\d{2}-\d{2})", text)
    assert g["examples_date"] == (ex.group(1) if ex else None)
    # every non-blank, non-rule line of each section survives into its entries, in order
    parts = re.split(r"^## .+$", text, flags=re.M)[1:]
    for sec, body in zip(g["sections"], parts):
        src = [ln.rstrip() for ln in body.splitlines() if ln.strip() and not re.match(r"^\s*-{3,}\s*$", ln)]
        out = [ln.rstrip() for e in sec["entries"] for ln in e["md"].splitlines() if ln.strip()]
        assert src == out, f"glossary section {sec['num']} lost or reordered lines"
    assert g["n_entries"] == sum(len(s["entries"]) for s in g["sections"]) > 0
    terms = {e["term"] for s in g["sections"] for e in s["entries"] if e["term"]}
    assert {"Sharpe ratio", "Turnover"} <= terms


def test_docs_index_links_resolve():
    docs = built()["docs"]
    paths = [d["path"] for d in docs]
    want = [p for p in S.DOC_GLOBS if (REPO / p).is_file()]
    assert paths == want, (paths, want)
    assert "README.md" in paths and "engine/GLOSSARY.md" in paths
    # the community and legal files of the repository root (LICENSE is the MIT License)
    assert {
        "CHANGELOG.md",
        "ROADMAP.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    } <= set(paths)
    assert "LICENSE" + ".md" not in S.DOC_GLOBS
    assert (REPO / "LICENSE").read_text(encoding="utf-8").startswith("MIT License\n")
    assert not [p for p in paths if p.endswith(".html")], "no HTML document is indexed"
    for d in docs:
        assert d["href"].startswith("../") and urllib.parse.unquote(d["href"][3:]) == d["path"]
        assert re.match(r"^\.\./[A-Za-z0-9%._~\-/]+$", d["href"]), d["href"]
        p = REPO / d["path"]
        assert p.is_file() and d["bytes"] == p.stat().st_size
        if p.suffix == ".md":
            first_h1 = next(ln[2:].strip() for ln in p.read_text().splitlines() if ln.startswith("# "))
            assert d["title"] == S.plain_heading(first_h1)
            assert "`" not in d["title"] and "**" not in d["title"], d["title"]
        assert d["description"], f"{d['path']} has no description"
        assert d["lang"] in ("en", "zh")
        keys = {"path", "href", "title", "description", "group", "lang", "bytes", "lines", "date"}
        assert set(d) == keys, (
            f"{d['path']}: keys only in the entry {sorted(set(d) - keys)}; missing {sorted(keys - set(d))}"
        )
    groups = {d["path"]: d["group"] for d in docs}
    assert groups["CHANGELOG.md"] == groups["ROADMAP.md"] == "Repository"
    if "library/README.md" in groups and "web/README.md" in groups:
        assert (groups["library/README.md"], groups["web/README.md"], groups["README.md"]) == (
            "Library",
            "Dashboard",
            "Repository",
        )
    # every indexed document is written in neutral wording: no first-person voice
    # "My factors" is the name of a page, not a first-person voice
    first_person = re.compile(
        r"\b[Mm]y\b(?! factors\b)|\bI'(m|ve|d|ll)\b|\bI (chose|ran|reran|re-ran|reproduced|checked|found|think|recommend)\b|\u6211"
    )  # \u6211: first person in Chinese
    for d in docs:
        if d["path"].endswith(".md"):
            hit = first_person.search((REPO / d["path"]).read_text(encoding="utf-8"))
            assert not hit, (d["path"], hit.group(0))


def test_size_budget_and_privacy():
    text = C.render_section("system", S.build())  # raises PrivacyError on a personal string
    n = len(text.encode("ascii"))
    assert n < C.SECTION_MAX_BYTES and n < 250_000, f"system.js is {n:,} bytes"
    low = text.lower()
    assert not [p for p in PERSONAL if p.lower() in low]
    assert not [s for s in PERSONAL if C.term_pattern(s).search(text)]
    for s in _strings(built()):
        assert not re.match(r"^(/|~/|[A-Za-z]:\\)", s), f"absolute path exported: {s[:60]}"
        assert "/private/" not in s and "/tmp/" not in s and "/home/" not in s, s[:60]


def test_web_data_file_matches_build():
    if not WEB_SYSTEM.exists():
        return
    m = WRAP_RE.match(WEB_SYSTEM.read_text(encoding="ascii"))
    assert m, "web/data/system.js does not use the data-file wrapper"
    on_disk = json.loads(m.group(1))
    now = built()
    # exporter test files can appear between an export and this test; compare the rest exactly
    for k in now:
        if k == "tests":
            a, b = dict(on_disk[k]), dict(now[k])
            for x in ("exporter", "exporter_totals"):
                a.pop(x, None), b.pop(x, None)
            assert a == b, "tests block differs from a fresh build; re-run export_web.py --only system"
        elif k == "snapshot":
            # update.py writes the line of its dashboard step after that step: the build log can
            # have more lines than the export saw, never fewer
            a, b = dict(on_disk[k]), dict(now[k])
            ba, bb = a.pop("build", None) or {}, b.pop("build", None) or {}
            assert a == b, "snapshot differs from a fresh build; re-run export_web.py --only system"
            assert ba.get("lines", 0) <= bb.get("lines", 0) and ba.get("file") == bb.get("file")
        elif k == "inventory":
            # the same for engine/logs/update.log: it grows after the export
            a, b = dict(on_disk[k]), dict(now[k])
            la, lb = a.pop("logs", None) or {}, b.pop("logs", None) or {}
            assert a == b, "inventory differs from a fresh build; re-run export_web.py --only system"
            assert [i["name"] for i in la.get("items", [])] == [i["name"] for i in lb.get("items", [])]
        else:
            assert on_disk[k] == now[k], f"{k} differs from a fresh build; re-run export_web.py --only system"


def test_page_never_uses_html_strings():
    src = PAGE.read_text()
    code = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    code = re.sub(r"(?m)^\s*//.*$", "", code)
    for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert bad not in code, f"pages/system.js uses {bad}"
    assert "?." not in re.sub(r"\"[^\"]*\"|'[^']*'|/[^/\n]+/[gimsuy]*", "", code), "optional chaining is not ES2019"
    assert "??" not in code
    assert "FL.pages.register" in src and 'id: "system"' in src
    low = code.lower()
    for word in ("go-live", "go live", "live trading"):
        assert word not in low, f"pages/system.js says {word!r}"
    cards = re.findall(r'ui\.card\(\{ id: "([a-z-]+)"', src)
    want = [
        "build",
        "update",
        "programs",
        "environment",
        "data-age",
        "inventory",
        "coverage",
        "invariants",
        "rules",
        "tests",
        "exporter-tests",
        "alphaforge-tests",
        "glossary",
        "docs",
        "export",
    ]
    assert cards == want, f"card ids of pages/system.js: {cards}; want {want}"


def test_page_layout_contract():
    """Layout rules: short glossary, capped docs table, phone cards, nowrap dates, shared jump links,
    one listing of the tests and of the inventory files, no column with one value."""
    src = PAGE.read_text()
    # every date column keeps "YYYY-MM-DD" on one line
    assert src.count("render: dateCell") == src.count('render: dateCell, className: "num nowrap') >= 5
    # glossary sections are collapsed <details>; the filter opens matches and restores the reader's state
    assert 'className: "system-gsec"' in src and "ui.details(secWrap" in src
    assert "sn.d.root.open = q ? hits > 0 : sn.userOpen" in src
    assert "nav.card === s.id" in src, "a #/system/glossary-N link must open its section"
    # the page's window listener goes with the page (ctx.listen); without the shell it removes itself
    assert 'ctx.listen(window, "hashchange", system_onHash)' in src
    assert 'window.removeEventListener("hashchange", system_onHash)' in src
    # the documents table scrolls inside its frame
    assert re.search(r"rows: docs, compact: true, maxHeight: \d+", src)
    # the programs: a table on wide screens, cards under 600px
    assert 'className: "system-programs-table"' in src and '"class": "system-ocards"' in src
    assert re.search(
        r"@media \(max-width:599\.98px\)\{[^\"]*\"\s*\+\s*\n\s*\"\.system-programs-table\{display:none\}\.system-ocards\{display:flex\}\}",
        src,
    )
    # every other table shows its rows as cards on a phone (the shell's stack option)
    n_tables = src.count("ui.table(")
    assert src.count("stack: true") == n_tables - 1, (src.count("stack: true"), n_tables)
    # jump links: the shared ui.toc, no page-local style
    assert "ui.toc(root, tocItems" in src and ".system-toc" not in src
    # paths break only after "/", command options stay on one line
    assert 'createElement("wbr")' in src and ".system-nb{white-space:nowrap}" in src
    # one listing each: the build blocks, the coins of the data age and of the coverage, the test
    # names and the glossary sections are closed blocks; one group's files at a time
    assert src.count("ui.details(") == 6, (
        "the two build blocks, the coins (age, coverage), the test names, the glossary"
    )
    assert '"class": "fl-select"' in src and "drawGroup(sel.value)" in src
    # columns with one value are not drawn
    assert 'c.key !== "date" || docs.some' in src and 'c.key !== "present" || secRows.some' in src
    # wording: group rows are a size measure, not a count of every table's rows
    assert "Rows count every table" not in src
    # text never uses the non-text token --muted; controls use the control border
    assert not re.search(r"(?<![-\w])color:var\(--muted\)", src)
    assert "border:1px solid var(--control-border)" in src
    # cross-references use the navigation names
    for text in ("The Paper page", "the Backtests page", "Factors page"):
        assert text not in src, text


def test_build_log_is_read_from_update_log():
    """The Last build card reads engine/logs/update.log: runs split at a pause or a failed step,
    the last run, the last complete run (ends with the dashboard step, no failure) and each step's
    last result. Bad lines are counted, never published."""
    import tempfile
    from unittest import mock

    lines = [
        "2026-10-09T08:16:12Z  factor lab  failed (exit 1)  8s",
        "2026-10-09T08:16:55Z  labels  ok  1s",
        "2026-10-09T08:17:56Z  library on OKX data, compute  ok  60s",
        "2026-10-09T08:18:00Z  dashboard  ok  4s",
        "a line with /Users/x/secret text",
        "2026-10-10T08:59:59Z  Factorlab 0.2.2",
        "2026-10-10T09:00:10Z  labels  ok  10s",
        "2026-10-10T09:00:10Z  labels | 20 symbols",
        "2026-10-10T09:00:20Z  library on OKX data, screen  failed (exit 2)  10s",
    ]
    with tempfile.TemporaryDirectory(prefix="sys_build_") as tmp:
        (Path(tmp) / "update.log").write_text("\n".join(lines) + "\n")
        with mock.patch.object(S, "default_logs_dir", return_value=Path(tmp)):
            b = S._build_log()
    assert b["present"] and b["lines"] == 6 and b["skipped"] == 1 and b["runs"] == 3
    assert b["other"] == 2  # the version line and a --verbose output line: not steps, not bad lines
    assert b["file"] == "engine/logs/update.log"
    r = b["last_run"]
    assert (r["started"], r["finished"], r["n_steps"], r["n_ok"]) == (
        "2026-10-10T09:00:00Z",
        "2026-10-10T09:00:20Z",
        2,
        1,
    )
    assert (r["failed_step"], r["failed_exit"], r["first_step"], r["last_step"]) == (
        "library on OKX data, screen",
        2,
        "labels",
        "library on OKX data, screen",
    )
    assert r["alarm_steps"] == [] and not any(x["alarm"] for x in r["steps"])
    assert b["last_run"]["steps"][0]["step"] == "labels"
    steps = {e["step"]: e for e in b["steps"]}
    assert steps["labels"]["last_ok_at"] == "2026-10-10T09:00:10Z"
    assert steps["factor lab"]["last_ok"] is False and steps["factor lab"]["last_ok_at"] is None
    assert steps["dashboard"]["last_ok_at"] == "2026-10-09T08:18:00Z"
    assert steps["library on OKX data, screen"]["last_ok"] is False
    assert "secret" not in json.dumps(b)
    with (
        tempfile.TemporaryDirectory(prefix="sys_build_") as tmp,
        mock.patch.object(S, "default_logs_dir", return_value=Path(tmp)),
    ):
        empty = S._build_log()
    assert empty["present"] is False and empty["last_run"] is None
    # the published block is the checkout's own log
    assert built()["snapshot"]["build"] == json.loads(C.dumps(S._build_log()))
    src = PAGE.read_text()
    assert 'id: "build"' in src and "snap.build" in src and 'label: "Last build (update.py)"' in src


def test_build_log_reads_a_new_alarm_as_done():
    """A step that exits with an ok code other than 0 (judge.py log-week: exit 3, a new alarm) has
    the result update.ALARM_STATUS in update.log. The reader counts it as ok with an alarm note: the
    run does not end there, the Last build card counts it in n_ok and names it in alarm_steps, and
    the step table keeps it with last_alarm. update.py itself writes the log of this test (a
    scratch FACTORLAB_LOGS folder), so the reader and the writer cannot use two texts."""
    import contextlib
    import io
    import sys
    import tempfile
    from unittest import mock

    code = "import sys; print('ALARM data_gap (okx): a test alarm'); sys.exit(3)"
    steps = [
        U.Step("labels", (sys.executable, "-c", "pass"), config.ROOT),
        U.Step(U.FACTOR_TEST_STEP, (sys.executable, "-c", code), config.ROOT, ok_codes=(0, U.ALARM_EXIT)),
        U.Step("dashboard", (sys.executable, "-c", "pass"), config.ROOT),
    ]
    with tempfile.TemporaryDirectory(prefix="sys_alarm_") as tmp:
        with (
            mock.patch.object(config, "LOGS", Path(tmp)),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            assert U.run(steps, jobs=1) == 0
        text = (Path(tmp) / "update.log").read_text(encoding="utf-8")
        assert f"  {U.FACTOR_TEST_STEP}  {U.ALARM_STATUS}  " in text, text
        with mock.patch.object(S, "default_logs_dir", return_value=Path(tmp)):
            b = S._build_log()
    assert b["present"] and b["lines"] == 3 and b["skipped"] == 0 and b["runs"] == 1, b
    r = b["last_run"]
    assert (r["n_steps"], r["n_ok"], r["failed_step"], r["failed_exit"]) == (3, 3, None, None), r
    assert (r["first_step"], r["last_step"], r["alarm_steps"]) == ("labels", "dashboard", [U.FACTOR_TEST_STEP])
    by_step = {x["step"]: x for x in r["steps"]}
    assert by_step[U.FACTOR_TEST_STEP]["ok"] is True and by_step[U.FACTOR_TEST_STEP]["alarm"] is True
    assert by_step[U.FACTOR_TEST_STEP]["exit"] is None
    assert by_step["labels"]["alarm"] is False and by_step["dashboard"]["alarm"] is False
    last = {e["step"]: e for e in b["steps"]}
    assert last[U.FACTOR_TEST_STEP]["last_ok"] is True and last[U.FACTOR_TEST_STEP]["last_alarm"] is True
    assert last[U.FACTOR_TEST_STEP]["last_ok_at"] == by_step[U.FACTOR_TEST_STEP]["at"]
    assert "ALARM" not in json.dumps(b)  # the output of a step is never published
    # a later run without an alarm clears last_alarm; a failed run after it ends the run there
    lines = [
        "2026-10-12T09:00:05Z  labels  ok  5s",
        f"2026-10-12T09:00:08Z  {U.FACTOR_TEST_STEP}  {U.ALARM_STATUS}  3s",
        "2026-10-12T09:00:09Z  factor lab  failed (exit 1)  1s",
        "2026-10-13T09:00:05Z  labels  ok  5s",
        f"2026-10-13T09:00:08Z  {U.FACTOR_TEST_STEP}  ok  3s",
    ]
    with tempfile.TemporaryDirectory(prefix="sys_alarm_") as tmp:
        (Path(tmp) / "update.log").write_text("\n".join(lines) + "\n")
        with mock.patch.object(S, "default_logs_dir", return_value=Path(tmp)):
            b2 = S._build_log()
    assert (b2["lines"], b2["runs"]) == (5, 2), b2
    assert b2["last_run"]["alarm_steps"] == [] and b2["last_run"]["n_ok"] == 2
    assert {e["step"]: e["last_alarm"] for e in b2["steps"]}[U.FACTOR_TEST_STEP] is False
    src = PAGE.read_text()
    assert "lastRun.alarm_steps" in src and "r.last_alarm" in src and "r.alarm" in src
    assert f'ui.badge("advisory", "{U.ALARM_STATUS}")' in src
    if _node() is None or not (REPO / "web" / "data" / "meta.js").exists() or not WEB_SYSTEM.exists():
        print("        (skipped the node render: node or the exported files are missing)")
        return
    page = _render_system("alarm", build=b)["text"]
    assert "3 steps, all ok; 1 step recorded a new alarm (factor tests, weekly log)" in page, page[:600]
    assert "A new alarm is not a failure. Read it with judge.py status." in page
    assert f"{U.FACTOR_TEST_STEP}i{U.ALARM_STATUS}" in page  # the advisory badge (icon "i")
    assert "1 new alarm, from engine/logs/update.log" in page
    assert "stopped at" not in page


def test_cadence_matches_the_update_plan():
    """The cadence words say when update.py runs each writer. update.plan() is run in each state
    (factor lab already run today or not, a registration on disk, AlphaForge on or off,
    --skip-library, --offline) and the scripts of its steps are compared with the words."""
    import tempfile
    from unittest import mock

    def scripts(**state):
        with tempfile.TemporaryDirectory(prefix="sys_plan_") as tmp:
            data = Path(tmp)
            if state.get("registered"):
                (data / "trials").mkdir()
                (data / "trials" / "registrations.jsonl").write_text("")
            with (
                mock.patch.object(config, "DATA", data),
                mock.patch.object(config, "ALPHAFORGE_ENABLED", state.get("alphaforge", False)),
                mock.patch.object(U, "factor_lab_ran", return_value=state.get("lab_ran", True)),
                mock.patch.object(U, "panel_last_bar", return_value=None),
                mock.patch.object(U, "binance_steps", return_value=[]),
            ):
                steps = U.plan(
                    date(2026, 1, 5),
                    offline=state.get("offline", True),
                    skip_binance=True,
                    skip_library=state.get("skip_library", False),
                )
        out = set()
        for st in steps:
            argv = [a for a in st.argv if not a.startswith("-") or a.startswith("--")]
            name = next((a for a in argv if a.endswith(".py")), None)
            if name:
                out.add(" ".join([Path(name).name] + [a for a in argv[argv.index(name) + 1 :] if a.startswith("--")]))
        return out

    later = scripts()
    assert "factor_lab.py" in scripts(lab_ran=False) and "factor_lab.py" not in later  # DAILY
    assert "judge.py --accept-release" in scripts(registered=True) and not any(x.startswith("judge.py") for x in later)
    assert "alphaforge_bridge.py --cycle" in scripts(alphaforge=True) and "alphaforge_bridge.py --cycle" not in later
    assert "libfactors.py --compute" in later and "libfactors.py --compute" not in scripts(skip_library=True)
    assert "ingest.py" in scripts(offline=False) and "ingest.py" not in later  # EACH_ONLINE
    assert {"labels.py", "export_web.py"} <= later  # EACH
    want = {
        "labels/labels.parquet": S.EACH,
        "research/factor_registry.parquet": S.DAILY,
        "library_on_engine/status.parquet": S.LIBRARY,
        "trials/weeks.jsonl": S.FACTOR_TESTS,
        "trials/values/okx.jsonl": S.FACTOR_TESTS,
        "trials/registrations.jsonl": S.BY_HAND,
        "research/alphaforge_library.json": S.ALPHAFORGE,
        "raw/ohlcv/BTC-USDT.parquet": S.EACH_ONLINE,
    }
    for rel, cad in want.items():
        assert S._writer(rel)[1] == cad, (rel, S._writer(rel))
    out = {p["script"]: p["cadence"] for p in built()["programs"]}
    assert out["factor_lab.py"] == S.DAILY and out["libfactors.py"] == S.LIBRARY
    assert out["judge.py"] == S.FACTOR_TESTS and out["ingest.py"] == S.EACH_ONLINE
    assert out["labels.py"] == S.EACH and out["export_web.py"] == S.EACH
    assert out["alphaforge_bridge.py"] == S.ALPHAFORGE and out["demo_data.py"] == S.DEMO_BUILD


META_RE = re.compile(r'^window\.FL=window\.FL\|\|\{\};FL\.data=FL\.data\|\|\{\};FL\.data\["meta"\]=(.*);\n$', re.S)


def _node() -> str | None:
    import shutil

    n = os.environ.get("FACTORLAB_NODE") or os.environ.get("NODE") or shutil.which("node")
    return n if n and Path(n).exists() else None


def _render_system(variant: str, build: dict | None = None) -> dict:
    """Run pages/system.js in node on the exported files. The DOM stub is the one of
    test_export_web.py (read with ast, not imported), set to this page. The variant
    "late" moves the fourth coin's last bar back to 8 days before the newest bar. A `build`
    block (the form of _build_log()) replaces the exported one."""
    import subprocess

    tree = ast.parse((ENGINE / "test_export_web.py").read_text(encoding="utf-8"))
    stub = next(
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "_DOM_STUB" for x in n.targets)
    )
    anchor = 'const root = new N("div");'
    assert stub.count(anchor) == 1 and stub.count('"pages/PAGE.js"') == 1, "the DOM stub of test_export_web.py changed"
    stub = stub.replace("PAGE", "system").replace(
        anchor,
        (
            'if (VARIANT === "late") { const a = FL.data.meta.data_age, c = a.coins[3];'
            ' c.last_bar = new Date(Date.parse(a.newest_bar) - 8 * 864e5).toISOString().slice(0, 10);'
            ' c.days_behind = 8; a.behind_newest = [c.symbol]; }\n'
            + (f"FL.data.system.snapshot.build = {json.dumps(build)};\n" if build is not None else "")
            + anchor
        ),
    )
    r = subprocess.run([_node(), "-e", stub, str(REPO / "web"), variant], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"system page render ({variant}) failed: {r.stderr[-800:]}"
    return json.loads(r.stdout)


def _age_card(text: str) -> str:
    i = text.index("The age of the data at the exporter run.")
    return text[i : text.index("Data inventory", i)]


def test_data_age_card():
    """The Data age card shows meta.data_age (export_web.py). It shows the data-through date, the
    days behind the export date and the last bar of each coin. Thus a stall of one coin shows.
    The test always checks the source contract. When it finds node (PATH, NODE or
    FACTORLAB_NODE), it also renders the exported files and a variant with one stalled coin."""
    src = PAGE.read_text()
    assert 'id: "data-age"' in src and '["data-age", "Data age"]' in src and "meta.data_age" in src
    # one data-through date on the page: meta.data_through (the newest perp bar of any coin)
    assert "var through = meta.data_through || snap.data_through;" in src
    meta_f = REPO / "web" / "data" / "meta.js"
    if _node() is None or not meta_f.exists() or not WEB_SYSTEM.exists():
        print("        (skipped the node render: node or the exported files are missing)")
        return
    age = json.loads(META_RE.match(meta_f.read_text(encoding="ascii")).group(1))["data_age"]
    full = _render_system("full")
    assert full["ids"].index("data-age") < full["ids"].index("inventory"), full["ids"]
    card = _age_card(full["text"])
    lag = age["days_behind"]
    assert f"Exporter run{age['export_date']} (UTC date)" in card
    assert f"Data through{age['data_through']} UTC close (the newest daily perp bar of any coin)" in card
    assert f"Days behind{lag:,}{' (current)' if lag <= 0 else ''}Newest coin bar{age['newest_bar']}" in card
    assert f"Oldest coin bar{age['oldest_bar']}" in card
    if not age["behind_newest"]:
        assert f"All {age['symbols']} coins have a last bar on {age['newest_bar']}." in card
        assert "stops before the newest bar" not in card
        # the coin table is closed when all coins share the last bar
        assert f"Last bar of each coin{age['symbols']} coins" in card
    late = _age_card(_render_system("late")["text"])
    sym = age["coins"][3]["symbol"]
    stale = str((pd.Timestamp(age["newest_bar"]) - pd.Timedelta(days=8)).date())
    assert "1 coin stops before the newest bar" in late and f"{sym} (last bar {stale})." in late
    assert f"{sym}{stale}8!behind the newest bar" in late
    for c in age["coins"]:
        if c["symbol"] != sym:
            assert f"{c['symbol']}{c['last_bar']}{c['days_behind']:,}\u2022same as the newest coin" in late, c["symbol"]
    # the status never shows a green check: it compares coins, not the age of the data
    assert "\u2713newest bar" not in late


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Run it with no arguments. It prints one PASS, FAIL or SKIP line for each test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    if missing_inputs():
        print(f"SKIP  export_system: the data folder has no {', '.join(missing_inputs())}")
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
    print(f"\n{len(tests) - failed}/{len(tests)} export_system tests passed")
    raise SystemExit(1 if failed else 0)
