"""Tests for the web exporter foundation (export_web.py + webexport/common.py).

Script-style like the other engine tests:  python -B test_export_web.py
Writes only into temporary directories; reads engine data read-only. The tests
that compare with engine data skip that part when the data folder does not have it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import export_web as E
import version as V
from webexport import SECTIONS
from webexport import common as C

WEB_DATA = Path(__file__).resolve().parent.parent / "web" / "data"
WRAP_RE = re.compile(
    r'^window\.FL=window\.FL\|\|\{\};FL\.data=FL\.data\|\|\{\};FL\.data\["([a-z][a-z0-9_]*(?:\.[a-z0-9][a-z0-9_-]*)?)"\]=(.*);\n$',
    re.S,
)
# generic path patterns only
PERSONAL = ("/Users/", "/home/")

# Route registry (fl-app.js SECTIONS) and page titles. Route ids are stable. The data sections
# are the routes except "factor", the page of one test, which reads the section "mine".
NAV = [
    ("library", "Factor library"),
    ("research", "Search tools"),
    ("mine", "My factors"),
    ("system", "System"),
]
ROUTES = NAV + [("factor", "Factor test")]
# The nav: the find tools, the tests of your factors, then the system.
NAV_GROUPS = [
    ("find", "Find", ["library", "research"]),
    ("test", "Test", ["mine"]),
    ("system", "System", ["system"]),
]


def _payload(text: str) -> tuple[str, object]:
    m = WRAP_RE.match(text)
    assert m, f"wrapper format wrong: {text[:120]!r}"
    return m.group(1), json.loads(m.group(2))


def test_sections_and_order():
    # export order = the route registry order (the nav order is NAV_GROUPS)
    assert tuple(i for i, _ in NAV) == SECTIONS
    assert E.parse_only(None) == list(SECTIONS)
    assert E.parse_only("system, library") == ["library", "system"], "--only must keep nav order"
    try:
        E.parse_only("library,nope")
    except SystemExit:
        pass
    else:
        raise AssertionError("unknown section accepted")


def test_nav_order_labels_and_landing_page():
    """fl-app.js's SECTIONS (route registry + fallback titles) equals webexport.SECTIONS plus the
    factor route, NAV_GROUPS groups every nav route once, every page registers the same title,
    the landing route is My factors, and each page file is a route."""
    web = WEB_DATA.parent
    app = (web / "assets" / "fl-app.js").read_text(encoding="utf-8")
    block = app[app.index("var SECTIONS = [") : app.index("];", app.index("var SECTIONS = ["))]
    nav = re.findall(r'\{ id: "([a-z]+)", title: "([^"]+)", icon: "([a-z]+)"([^}]*)\}', block)
    assert [(i, t) for i, t, _, _ in nav] == ROUTES, nav
    extra = {i: rest for i, _, _, rest in nav}
    assert "param: true" in extra["factor"] and "nav: false" in extra["factor"], (
        "the factor route carries an id and is not in the nav"
    )
    assert 'data: ["mine"]' in extra["factor"], "the factor route loads the section mine"
    assert all(not rest.strip() for i, rest in extra.items() if i != "factor")
    gblock = app[app.index("var NAV_GROUPS = [") : app.index("];", app.index("var NAV_GROUPS = ["))]
    groups = [
        (g, label, re.findall(r'"([a-z]+)"', items))
        for g, label, items in re.findall(r'\{ id: "([a-z]+)", label: "([^"]+)", items: \[([^\]]*)\] \}', gblock)
    ]
    assert groups == NAV_GROUPS, groups
    in_nav = [i for _, _, items in NAV_GROUPS for i in items]
    assert sorted(in_nav) == sorted(i for i, _ in NAV) and len(in_nav) == len(set(in_nav)), "every nav route once"
    assert re.search(r'var DEFAULT = "mine";', app), "My factors is the landing page"
    # the nav icon of My factors: a clipboard with a tick
    assert 'mine: ["M9 3h6v3H9z", "M7 4.5H5V21h14V4.5h-2", "M8.5 13.5l2.5 2.5 4.5-5"]' in app
    assert 'el("p", { text: "Find a factor. Test it in a book." })' in app
    files = sorted(f.stem for f in (web / "pages").glob("*.js"))
    assert files == sorted(i for i, _ in ROUTES), f"page files {files}; routes {[i for i, _ in ROUTES]}"
    assert "PARAM_RE = /^[A-Za-z0-9_.:-]{1,128}$/" in app, "the factor id: letters, digits, _ - . :"
    for sid, title in ROUTES:
        js = (web / "pages" / f"{sid}.js").read_text(encoding="utf-8")
        reg = js[js.index("FL.pages.register") :]
        assert re.search(rf'id: "{sid}",\s*title: "{re.escape(title)}"', reg), f"pages/{sid}.js title is not {title!r}"
    # no page, shell or exporter text points toward live trading
    for f in [web / "assets" / "fl-app.js", web / "index.html"] + sorted((web / "pages").glob("*.js")):
        low = f.read_text(encoding="utf-8").lower()
        for word in ("go-live", "go live", "live trading", "goes live"):
            assert word not in low, f"{f.name} says {word!r}"


def test_wrapper_format_and_nan_cleaning():
    fake = {
        "dates": ["2025-01-05", "2025-01-06"],
        "series": {"a": [1.5, float("nan")], "b": [float("inf"), -float("inf")]},
        "np": [np.float64(0.25), np.int64(3), np.bool_(True), np.float32("nan")],
        "ts": pd.Timestamp("2026-10-01", tz="UTC"),
        "tsm": pd.Timestamp("2026-10-02T05:53:07", tz="UTC"),
        "tuple": (1, 2),
        "text": "café — ok",
    }
    with tempfile.TemporaryDirectory() as tmp:
        n = C.write_section(tmp, "fake", fake)
        f = Path(tmp) / "fake.js"
        raw = f.read_bytes()
        assert n == len(raw)
        raw.decode("ascii")  # ensure_ascii: works on file:// with any charset
        text = raw.decode("ascii")
        assert "NaN" not in text and "Infinity" not in text
        name, data = _payload(text)
        assert name == "fake"
        assert data["series"]["a"] == [1.5, None]
        assert data["series"]["b"] == [None, None]
        assert data["np"] == [0.25, 3, True, None]
        assert data["ts"] == "2026-10-01"
        assert data["tsm"] == "2026-10-02T05:53:07Z"
        assert data["tuple"] == [1, 2]
        assert data["text"] == "café — ok"
        assert not [p for p in Path(tmp).iterdir() if p.name.startswith(".")], "temp file left behind"


def test_rounding_helpers():
    assert C.rnd(1234.5678, "usd") == 1234.57
    assert C.rnd(0.0075694321, "ret") == 0.007569
    assert C.rnd(0.123456789, "rank") == 0.1235
    assert C.rnd(float("nan")) is None and C.rnd(None) is None and C.rnd(float("inf")) is None
    assert C.rnd(-0.0000001, "usd") == 0.0 and math.copysign(1, C.rnd(-0.0000001, "usd")) == 1.0
    assert C.rnd(pd.Series([1.234567, np.nan]), 2) == [1.23, None]
    try:
        C.rnd(1.0, "nonsense")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown kind accepted")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-05", "2025-01-06"], utc=True),
            "equity": [10000.0, 10195.385847],
            "ret": [0.0, np.nan],
        }
    )
    blk = C.series_cols(df, {"equity": "usd", "ret": "ret"})
    assert blk == {"dates": ["2025-01-05", "2025-01-06"], "series": {"equity": [10000.0, 10195.39], "ret": [0.0, None]}}
    recs = C.records(df, {"equity": "usd"})
    assert recs[1] == {"date": "2025-01-06", "equity": 10195.39, "ret": None}
    assert C.to_dates(pd.Series(pd.to_datetime(["2026-10-01 00:00"], utc=True))) == ["2026-10-01"]
    assert C.to_date(None) is None and C.to_date("2026-10-01") == "2026-10-01"


def test_clean_json_rejects_paths_and_frames():
    for bad in (Path("x"), pd.DataFrame({"a": [1]})):
        try:
            C.clean_json({"x": bad})
        except TypeError:
            pass
        else:
            raise AssertionError(f"clean_json accepted {type(bad).__name__}")


def test_size_guard():
    with tempfile.TemporaryDirectory() as tmp:
        big = {"x": "a" * 2000}
        try:
            C.write_section(tmp, "big", big, max_bytes=1000)
        except C.SizeError:
            pass
        else:
            raise AssertionError("oversize section was written")
        assert not (Path(tmp) / "big.js").exists(), "oversize section must not be written"
        assert C.write_section(tmp, "small", {"x": 1}, max_bytes=1000) < 1000
    assert C.check_sizes({"paper": 1_400_000, "risk": 10}) == []
    assert C.check_sizes({"paper": C.SECTION_MAX_BYTES + 1}), "per-section budget not enforced"
    five = {s: 1_400_000 for s in ("a", "b", "c", "d", "e")}
    assert any("total" in p for p in C.check_sizes(five)), "total budget not enforced"


def test_privacy_guard():
    with tempfile.TemporaryDirectory() as tmp:
        # synthetic fixtures, joined at run time so the source holds no literal path or address
        at, users, home = "@", "/" + "Users/", "/" + "home/"
        for leak in (
            users + "someone/x",
            home + "someone/x",
            "first.last" + at + "mail.example.net",
            "me" + at + "example.org",
        ):
            try:
                C.write_section(tmp, "leak", {"note": leak})
            except C.PrivacyError:
                pass
            else:
                raise AssertionError(f"private string not caught: {leak!r}")
        assert not (Path(tmp) / "leak.js").exists()
        C.write_section(tmp, "fine", {"note": "axqztb UTC close, 1.00x advisory"})


def test_term_patterns():
    # synthetic placeholder terms
    pats = C.term_patterns(["Zebraville", "XQZT"])

    def hit(s):
        return [n for n, rx in pats if rx.search(s)]

    assert hit("lives in zebraville") and hit("XQZT time") and not hit("xqzt lower") and not hit("aXQZTb")
    assert [n for n, _ in pats] == ["term #1", "term #2"], "hit names must not show the term"


# A synthetic term in a non-Latin script, written as \u escapes (Chinese for "placeholder
# privacy word"; it is not a real private value).
NON_ASCII_TERM = "\u5360\u4f4d\u9690\u79c1\u8bcd"


def _with_terms(terms, fn):
    """Run fn() with PRIVATE_PATTERNS extended by synthetic terms, then restore it.
    PRIVATE_PATTERNS is read at call time, so this runs the real exporter path."""
    saved = C.PRIVATE_PATTERNS
    C.PRIVATE_PATTERNS = C._GENERIC_PATTERNS + C.term_patterns(terms)
    try:
        return fn()
    finally:
        C.PRIVATE_PATTERNS = saved


def test_non_ascii_private_term_blocked_end_to_end():
    """The check runs on every original key and nested value before the ASCII escaping,
    then on the escaped text decoded again. In the escaped JSON a non-ASCII term is a run
    of \\uXXXX escapes that the pattern does not match (and the browser shows the term)."""
    term = NON_ASCII_TERM
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        leaks = {
            "value": {"note": term},
            "inside a value": {"note": "near " + term + " today"},
            "key": {term: 1},
            "nested value": {"a": [{"b": {"c": ["ok", "x" + term]}}]},
            "nested key": {"a": [{"b": {term + "_k": [1, 2]}}]},
            "tuple member": {"t": ("ok", term)},
            "full-width look-alike": {
                "note": "\uff3a\uff45\uff42\uff52\uff41\uff56\uff49\uff4c\uff4c\uff45"
            },  # NFKC -> Zebraville
            # invisible format characters (category Cf) splitting a term: NFKC keeps them,
            # the browser does not show them, so the guard must look through them
            "zero-width space": {"note": "Zebra\u200bville"},
            "soft hyphen": {"note": "Zebra\u00adville"},
            "zero-width joiner in a key": {"Zebra\u200dville": 1},
            "zero-width space in a non-ASCII term": {"note": term[:2] + "\u200b" + term[2:]},
        }

        def run():
            for what, payload in leaks.items():
                # the escaped serialisation really does hide the term from a plain text search
                assert term not in C.dumps(payload)
                assert C.payload_privacy_hits(payload), f"{what}: original strings not checked"
                try:
                    C.write_section(out, "leak", payload)
                except C.PrivacyError as exc:
                    assert term not in str(exc) and "Zebraville" not in str(exc), "the error must not echo the term"
                    assert "term #" in str(exc)
                else:
                    raise AssertionError(f"non-ASCII term not caught ({what})")
                assert not (out / "leak.js").exists(), f"{what}: a blocked section was written"
            # the second line of defence: escaped text is decoded before matching (callers that
            # check finished text, e.g. the trial ledger in ledger.py, are covered too)
            assert C.privacy_hits(C.dumps({"note": term})) == ["term #1"]
            assert C.privacy_hits(json.dumps({"note": term})) == ["term #1"]
            assert C.privacy_hits(C.dumps({"note": "Zebra\u200bville"})) == ["term #2"]
            # other non-ASCII text still passes, and round-trips unchanged
            C.write_section(out, "fine", {"note": "café — \u5360\u4f4d ok", "—": 1})
            _, data = _payload((out / "fine.js").read_text(encoding="ascii"))
            assert data == {"note": "café — \u5360\u4f4d ok", "—": 1}

        _with_terms([term, "Zebraville"], run)
        assert out.exists() and not [p for p in out.iterdir() if p.name.startswith(".")], "temp file left behind"
    # without the synthetic terms the same payload is fine (the term is not a real one)
    assert not C.payload_privacy_hits({term: term})


def test_refuses_engine_data_dir():
    for bad in (C.DATA, C.DATA / "web"):
        try:
            C.check_out_dir(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"out dir inside engine data accepted: {bad.name}")


def test_meta_and_cli_on_temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        C.write_section(tmp, "library", {"ok": True})
        E.write_meta(Path(tmp))
        name, meta = _payload((Path(tmp) / "meta.js").read_text(encoding="ascii"))
        assert name == "meta"
        assert set(meta["sections"]) == set(SECTIONS)
        assert meta["sections"]["library"]["present"] and meta["sections"]["library"]["bytes"] > 0
        assert meta["sections"]["library"]["state"] == "kept" and meta["sections"]["library"]["as_of"] is None
        assert meta["sections"]["research"] == {
            "present": False,
            "bytes": None,
            "as_of": None,
            "available": None,
            "state": "kept",
            "written_at": None,
            "parts": {},
        }
        keys = {"schema", "version", "demo", "generated_at", "data_through", "data_close_utc", "timezone"}
        keys |= {"bar_stamp", "sections", "total_bytes", "budget", "data_age"}
        assert set(meta) == keys, (
            f"meta keys: only in meta {sorted(set(meta) - keys)}; missing {sorted(keys - set(meta))}"
        )
        assert meta["version"] == V.VERSION and V.read(V.VERSION_FILE) == V.VERSION  # the file VERSION
        assert meta["demo"] is C.demo()
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", meta["generated_at"])
        # data_through: the newest perp bar of any coin (an independent read of data/raw/ohlcv)
        bars = [pd.read_parquet(f, columns=["ts"])["ts"] for f in sorted((C.DATA / "raw" / "ohlcv").glob("*.parquet"))]
        bars = [b for b in bars if len(b)]
        if bars:
            assert meta["data_through"] == str(pd.to_datetime(max(b.max() for b in bars), utc=True).date())
            assert meta["data_close_utc"].startswith(
                str((pd.Timestamp(meta["data_through"]) + pd.Timedelta(days=1)).date())
            )
        else:
            assert meta["data_through"] is None and meta["data_close_utc"] is None
        assert meta["data_through"] == E.data_through() == meta["data_age"]["newest_bar"]
        # the CLI on a section without a builder only (re)writes meta.js
        missing = [s for s in SECTIONS if E.load_builder(s) is None]
        if missing:
            assert E.main(["--out", tmp, "--only", missing[0]]) == 0
            assert (Path(tmp) / "meta.js").exists() and not (Path(tmp) / f"{missing[0]}.js").exists()


WEB = WEB_DATA.parent


def _css_block(css: str, selector_re: str) -> dict[str, str]:
    """Custom properties declared in the first block whose selector matches selector_re."""
    m = re.search(selector_re + r"\s*\{([^{}]*)\}", css)
    assert m, f"block not found: {selector_re}"
    return dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", m.group(1)))


def test_css_color_tokens_have_dark_values():
    css = (WEB / "assets" / "app.css").read_text(encoding="utf-8")
    light = _css_block(css, r"(?m)^:root")
    os_dark = _css_block(css, r':root:not\(\[data-theme="light"\]\)')
    stamp_dark = _css_block(css, r'(?m)^:root\[data-theme="dark"\]')
    # status colors (--st-*) are one set for both themes by spec; every other color token needs a dark value
    colors = {k for k, v in light.items() if re.match(r"\s*(#|rgba?\()", v) and not k.startswith("--st-")}
    assert {"--bg", "--surface", "--ink", "--ink-2", "--ink-3", "--muted", "--c1", "--c8"} <= colors
    for name, block in (("OS-dark", os_dark), ("data-theme=dark", stamp_dark)):
        missing = sorted(colors - set(block))
        assert not missing, f"{name} block lacks dark values for {missing}"
    assert os_dark == stamp_dark, "the two dark blocks must declare identical values"

    # contrast rule: --ink-3 is the muted TEXT token and must clear WCAG AA (4.5:1) on every surface
    def lum(h):
        c = [int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    def ratio(a, b):
        la, lb = lum(a), lum(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)

    for block in (light, os_dark):
        for surf in ("--bg", "--surface", "--surface-2"):
            r = ratio(block["--ink-3"].strip(), block[surf].strip())
            assert r >= 4.5, f"--ink-3 on {surf}: {r:.2f}:1 < 4.5:1"


def test_index_html_load_order():
    """index.html loads only the core assets, meta.js and the shell; the shell loads each data
    and page file on demand (classic <script> tags, which work on file://)."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'type="module"' not in html, "ES modules are blocked on file://"
    srcs = re.findall(r'<script src="([^"]+)"', html)
    expect = ["assets/fl-core.js", "assets/fl-charts.js", "data/meta.js", "assets/fl-app.js"]
    assert srcs == expect, f"script order drifted: {srcs}"
    assert re.search(r'<script src="data/meta.js" onerror="FL_MISS\(this\)"></script>', html), (
        "meta.js must tolerate a missing file"
    )
    # a fetched file that does not run is recorded (FL_BROKEN) for the empty states
    assert "window.FL_BROKEN = [];" in html and 'window.addEventListener("error"' in html
    core = (WEB / "assets" / "fl-core.js").read_text(encoding="utf-8")
    assert '"data/" + section + ".js"' in core and '"pages/" + id + ".js"' in core
    assert '"?v=" + encodeURIComponent(String(v))' in core, "loaded files carry the export version"
    for sid in SECTIONS + ("factor",):
        assert (WEB / "pages" / f"{sid}.js").is_file(), f"pages/{sid}.js is missing"


def test_footer_license_and_version_text():
    """The footer gives the version and the licenses only: the code license (MIT) and the data
    terms, with the version from meta.js (the file VERSION)."""
    app = (WEB / "assets" / "fl-app.js").read_text(encoding="utf-8")
    assert '"Code: MIT License. Market data: OKX and Binance, under their own terms.' in app
    assert "CC BY-NC-SA 4.0 (credit: Binance Vision)" in app
    assert 'typeof meta.version === "string"' in app and '"Factorlab " + meta.version' in app
    foot = app[app.index("  function footer() {") :]
    foot = foot[: foot.index("\n  }\n") + 4]
    assert foot.count("f.appendChild(") == 1, "the footer has one paragraph: the version and the licenses"
    for text in ("static snapshot", "FL_MISSING", "FL_BROKEN", "sections exported"):
        assert text not in foot, f"the footer has the text {text!r}"


def test_web_data_has_no_personal_strings():
    if not WEB_DATA.exists():
        return
    for f in sorted(WEB_DATA.glob("*.js")):
        text = f.read_text(encoding="utf-8", errors="replace")
        # short all-caps terms match case-sensitively on word boundaries (C.term_pattern)
        hits = [p for p in PERSONAL if C.term_pattern(p).search(text)]
        assert not hits, f"{f.name} contains personal strings: {hits}"
        assert WRAP_RE.match(text), f"{f.name} does not use the data-file wrapper"
        # the files are ASCII-escaped: check the DECODED strings too (what the browser shows)
        _, data = _payload(text)
        assert not C.payload_privacy_hits(data), f"{f.name}: {C.payload_privacy_hits(data)}"
        strings = set(C._strings(data))
        hits = [p for p in PERSONAL if any(C.term_pattern(p).search(s) for s in strings)]
        assert not hits, f"{f.name} contains personal strings after decoding: {len(hits)} term(s)"


# --------------------------------------------------------------------------- chart kit (node)

# Just enough DOM to run fl-core.js + fl-charts.js in node. Unlike the page stubs, the chart
# root reports a real width (640 px), so the line chart actually draws its SVG.
_CHART_STUB = r"""
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const WEB = process.argv[1];
class CL { constructor(n) { this.n = n; } g() { return (this.n.attrs["class"] || "").split(/\s+/).filter(Boolean); }
  s(a) { this.n.attrs["class"] = a.join(" "); } add(...c) { const a = this.g(); c.forEach(x => { if (!a.includes(x)) a.push(x); }); this.s(a); }
  remove(...c) { this.s(this.g().filter(x => !c.includes(x))); } contains(c) { return this.g().includes(c); }
  toggle(c, f) { const w = f === undefined ? !this.contains(c) : !!f; if (w) this.add(c); else this.remove(c); return w; } }
class N { constructor(t) { this.tagName = String(t).toUpperCase(); this.children = []; this.childNodes = this.children; this.attrs = {};
    this.style = {}; this.parentNode = null; this._t = ""; this.L = {}; this.nodeType = 1; this.classList = new CL(this); this.dataset = {}; }
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
  get innerHTML() { return ""; } set innerHTML(v) { this.children.length = 0; this._t = ""; }
  get clientWidth() { return 640; } get clientHeight() { return 240; } get offsetWidth() { return 640; }
  getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 240, right: 640, bottom: 240 }; }
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
  location: { hash: "", search: "" }, history: { replaceState() {} }, navigator: { userAgent: "node" },
  CustomEvent: class { constructor(t, o) { this.type = t; this.detail = o && o.detail; } }, Event: class { constructor(t) { this.type = t; } },
  URLSearchParams };
W.window = W; W.self = W;
const C = vm.createContext(W);
for (const f of ["assets/fl-core.js", "assets/fl-charts.js"]) vm.runInContext(fs.readFileSync(path.join(WEB, f), "utf8"), C, { filename: f });
const FL = W.FL;
const x = ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"];
function chart(series, extra) {
  const box = new N("div"); D.body.appendChild(box);
  const c = FL.charts.line(box, Object.assign({ x: x, series: series, y: { format: "num" } }, extra || {}));
  const g = c.root.querySelectorAll("g").filter(n => n.classList.contains("lines"))[0];
  return { prepared: c._inst.series.map(s => s.width),
           drawn: g ? g.children.filter(n => n.tagName === "PATH").map(n => n.getAttribute("stroke-width")) : null };
}
const out = {
  perSeries: chart([{ id: "a", color: "#111", width: 1.5, values: [1, 2, 3, 4] }, { id: "b", color: "#222", values: [2, 3, 4, 5] },
                    { id: "c", color: "#333", width: "3", values: [3, 4, 5, 6] }, { id: "d", color: "#444", width: 0, values: [4, 5, 6, 7] }]),
  chartWide: chart([{ id: "a", color: "#111", values: [1, 2, 3, 4] }, { id: "b", color: "#222", width: 1.5, values: [2, 3, 4, 5] }], { lineWidth: 2.5 })
};
process.stdout.write(JSON.stringify(out));
"""


def _node() -> str | None:
    import os
    import shutil

    n = os.environ.get("FACTORLAB_NODE") or os.environ.get("NODE") or shutil.which("node")
    return n if n and Path(n).exists() else None


# A small DOM stand-in, enough to run fl-core.js + fl-charts.js + one page in node. Charts
# measure 0 px here, so SVGs are not drawn; everything else renders. "PAGE" stands for the page
# id: a suite replaces it with its page (test_export_system.py inserts its variants before the
# line 'const root = new N("div");'). Arguments: the web folder, a variant name and a JSON object
# for ctx.nav (for example {"param": "CF00001"}). It prints {text, ids, links, title}.
_DOM_STUB = r"""
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const WEB = process.argv[1], VARIANT = process.argv[2], NAV = JSON.parse(process.argv[3] || "{}");
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
  location: { hash: "#/PAGE", search: "" }, history: { replaceState() {} }, navigator: { userAgent: "node" },
  ResizeObserver: class { observe() {} unobserve() {} disconnect() {} },
  CustomEvent: class { constructor(t, o) { this.type = t; this.detail = o && o.detail; } }, Event: class { constructor(t) { this.type = t; } },
  URLSearchParams };
W.window = W; W.self = W;
const C = vm.createContext(W);
for (const f of ["assets/fl-core.js", "assets/fl-charts.js", "data/meta.js", "data/PAGE.js", "pages/PAGE.js"]) {
  if (f.startsWith("data/") && !fs.existsSync(path.join(WEB, f))) continue;  // a page without a data file
  vm.runInContext(fs.readFileSync(path.join(WEB, f), "utf8"), C, { filename: f });
}
const FL = W.FL;
const TITLE = [], SUBTITLE = [];
const root = new N("div"); D.body.appendChild(root);
FL.pages.get("PAGE").render(root, { id: "PAGE", data: FL.data, section: FL.data.PAGE, meta: FL.data.meta, colors: FL.colors,
  fmt: FL.fmt, ui: FL.ui, charts: FL.charts, params: new URLSearchParams(""), setTitle(t, s) { TITLE.push(String(t)); if (s !== undefined) SUBTITLE.push(s === null ? "" : String(s)); },
  nav: Object.assign({ current: "PAGE", card: null, param: null, badParam: false, href: (p, c) => "#/" + p + (c ? "/" + c : ""), go() {} }, NAV) });
root.querySelectorAll(".fl-chart-toggle").forEach(b => (b.L.click || []).forEach(f => f({})));
root.querySelectorAll("details").forEach(d => (d.L.toggle || []).forEach(f => { d.open = true; f({}); }));
process.stdout.write(JSON.stringify({ text: root.textContent, ids: root.querySelectorAll("section").map(s => s.id),
  links: root.querySelectorAll("a").map(a => a.getAttribute("href")), title: TITLE.length ? TITLE[TITLE.length - 1] : null,
  subtitle: SUBTITLE.length ? SUBTITLE[SUBTITLE.length - 1] : null }));
"""


def render_page(page: str, variant: str = "full", nav: dict | None = None, web: Path | None = None) -> dict:
    """Run pages/<page>.js once in node with _DOM_STUB on the files of ``web`` (default web/)."""
    import subprocess

    stub = _DOM_STUB.replace("PAGE", page)
    args = [_node(), "-e", stub, str(web or WEB_DATA.parent), variant, json.dumps(nav or {})]
    r = subprocess.run(args, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"page render ({page}, {variant}) failed: {r.stderr[-800:]}"
    return json.loads(r.stdout)


def test_factor_page_without_data():
    """The factor page reads the section mine (route data ["mine"]) and needs no data for a library
    id: a library id links to the library search; a bad id and no id give a short line. The page of
    a registered test is checked in test_export_mine.py. Needs node for the render."""
    js = (WEB_DATA.parent / "pages" / "factor.js").read_text(encoding="utf-8")
    reg = js[js.index("FL.pages.register") :]
    assert 'section: "mine"' in reg and 'data: ["mine"]' in reg and "requiresData: false" in reg
    if _node() is None:
        print("        (skipped the node run: set NODE or FACTORLAB_NODE)")
        return
    lib = render_page("factor", nav={"param": "CF00001"})
    assert lib["title"] == "CF00001", lib["title"]
    assert "CF00001 is a library definition." in lib["text"], lib["text"]
    assert lib["links"] == ["#/library/search"] and "Find CF00001 in the library search" in lib["text"]
    bad = render_page("factor", nav={"badParam": True})
    assert "The id in the address is not valid." in bad["text"] and bad["title"] is None
    assert bad["links"] == ["#/mine"]
    none = render_page("factor")
    assert "The address names no test." in none["text"] and none["links"] == ["#/mine"]


def test_line_width_survives_preparation():
    """prepareLine keeps series.width, so drawLine (which reads s.width) draws each series at
    its own width (the Risk page's 1.5px lines) and the 2px default only applies without one.
    Runs the real chart kit in node. Needs node (FACTORLAB_NODE, NODE or PATH)."""
    js = (WEB / "assets" / "fl-charts.js").read_text(encoding="utf-8")
    prep = js[js.index("function prepareLine") : js.index("function bucketFn")]
    assert re.search(r"\bwidth:\s*sw\b", prep), "prepareLine must carry the per-series width"
    node = _node()
    if node is None:
        print("        (skipped the node render: set NODE or FACTORLAB_NODE)")
        return
    import subprocess

    r = subprocess.run([node, "-e", _CHART_STUB, str(WEB)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"chart probe failed: {r.stderr[-800:]}"
    out = json.loads(r.stdout)
    # requested widths survive preparation (a numeric string is accepted; 0 / absent -> default)
    assert out["perSeries"]["prepared"] == [1.5, None, 3, None], out
    # ... and reach the drawn path; unspecified widths keep the 2px default
    assert out["perSeries"]["drawn"] == ["1.5", "2", "3", "2"], out
    # a chart-wide lineWidth still applies to series without their own width
    assert out["chartWide"]["drawn"] == ["2.5", "1.5"], out


def test_data_age_block():
    """The data-age block of meta.js shows a stalled copy of the data and a stalled coin. The
    data-through date is the newest bar of any coin, so it does not show a stalled coin. The
    test checks the functions on a synthetic case. Then it checks the exported meta.js against
    an independent read of data/raw/ohlcv."""
    a = E.data_age("2026-10-08", "2026-10-01", {"AAA-USDT": "2026-10-01", "BBB-USDT": "2026-09-29", "CCC-USDT": None})
    assert a["export_date"] == "2026-10-08" and a["data_through"] == "2026-10-01"
    assert a["days_behind"] == 6, "the closes of 2026-10-03 to 2026-10-08 (00:00 UTC) are not in the data"
    assert (a["newest_bar"], a["oldest_bar"], a["symbols"]) == ("2026-10-01", "2026-09-29", 3)
    assert a["behind_newest"] == ["BBB-USDT", "CCC-USDT"], "a stalled coin and a coin with no bar"
    assert [c["days_behind"] for c in a["coins"]] == [6, 8, None]
    assert [c["symbol"] for c in a["coins"]] == ["AAA-USDT", "BBB-USDT", "CCC-USDT"], "input order is kept"
    assert E.days_behind("2026-10-02", "2026-10-01") == 0, "on the day after the bar, the data is current"
    assert E.days_behind("2026-10-02", None) is None
    cur = E.data_age("2026-10-02", "2026-10-01", {"AAA-USDT": "2026-10-01", "BBB-USDT": "2026-10-01"})
    assert cur["days_behind"] == 0 and cur["behind_newest"] == []
    empty = E.data_age("2026-10-02", None, {"AAA-USDT": None})
    assert empty["days_behind"] is None and empty["newest_bar"] is None and empty["behind_newest"] == ["AAA-USDT"]

    f = WEB_DATA / "meta.js"
    if not f.exists() or not C.data_path("raw/ohlcv").is_dir():
        print("        (skipped the exported block: web/data/meta.js or data/raw/ohlcv is not there)")
        return
    import config

    _, meta = _payload(f.read_text(encoding="ascii"))
    age = meta["data_age"]
    files = {p.stem: p for p in sorted((C.DATA / "raw" / "ohlcv").glob("*.parquet"))}
    want = {}
    for sym in config.UNIVERSE:
        name = sym.split("/")[0] + "-" + sym.split("/")[1].split(":")[0]
        p = files.get(name)
        ts = pd.to_datetime(pd.read_parquet(p)["ts"], utc=True) if p is not None else pd.Series([], dtype=object)
        want[name] = str(ts.max().date()) if len(ts) else None
    assert [c["symbol"] for c in age["coins"]] == list(want), "one row per coin, in universe order"
    assert {c["symbol"]: c["last_bar"] for c in age["coins"]} == want
    assert age["symbols"] == len(want) == 20
    known = [d for d in want.values() if d]
    through = max(known) if known else None
    assert age["data_through"] == meta["data_through"] == through
    assert age["export_date"] == meta["generated_at"][:10]
    export = pd.Timestamp(age["export_date"])
    assert age["days_behind"] == (None if through is None else (export - pd.Timestamp(through)).days - 1)
    for c in age["coins"]:
        assert c["days_behind"] == (None if c["last_bar"] is None else (export - pd.Timestamp(c["last_bar"])).days - 1)
    assert (age["newest_bar"], age["oldest_bar"]) == (max(known), min(known))
    assert age["behind_newest"] == [s for s, d in want.items() if d is None or d < max(known)]


def test_no_data_payload():
    """A section without its input files exports {"available": false, "reason": ...}: the reason
    names the missing file relative to the data folder, and never carries an absolute path."""
    p = C.no_data_payload(FileNotFoundError(2, "No such file", str(C.DATA / "research" / "x.parquet")))
    assert p == {"schema": 1, "available": False, "reason": "research/x.parquet is not in the data folder"}
    p = C.no_data_payload(FileNotFoundError(2, "No such file", "/elsewhere/y.parquet"))
    assert p["reason"] == "y.parquet is not in the data folder"
    assert (
        C.no_data_payload(SystemExit("No label table. Run labels.py first."))["reason"]
        == "No label table. Run labels.py first"
    )
    for exc in (SystemExit("missing /Users/someone/data"), SystemExit(1), C.NoData("")):
        assert C.no_data_payload(exc)["reason"] == "the input data is not in the data folder", exc
    assert (
        C.no_data_payload(C.NoData("the trial ledger is not in the data folder"))["reason"]
        == "the trial ledger is not in the data folder"
    )
    # the app shell shows the reason in place of the page
    app = (WEB_DATA.parent / "assets" / "fl-app.js").read_text(encoding="utf-8")
    assert 'section.available === false && typeof section.reason === "string"' in app


def test_only_absent_data_is_no_data():
    """Only absent input data gives the no-data payload. A missing repository file or
    another SystemExit is a defect: the section fails and the export exits 1."""
    assert C.is_no_data(C.NoData("x"))
    assert C.is_no_data(FileNotFoundError(2, "No such file", str(C.DATA / "paper" / "x.parquet")))
    assert not C.is_no_data(FileNotFoundError(2, "No such file", str(E.ROOT / "GLOSSARY.md")))
    assert not C.is_no_data(FileNotFoundError("no file name"))
    assert C.is_no_data(SystemExit("No label table. Run labels.py first."))
    assert not C.is_no_data(SystemExit("No factor table. Run factors.py first."))
    assert C.is_no_data(SystemExit("No raw OHLCV data. Run ingest.py first."))
    assert not C.is_no_data(SystemExit("refusing to write"))
    assert not C.is_no_data(SystemExit(1))
    old_loader = E.load_builder
    with tempfile.TemporaryDirectory() as tmp:

        def fake(section):
            def build():
                if section == "library":
                    raise FileNotFoundError(2, "No such file", str(E.ROOT / "GLOSSARY.md"))
                raise SystemExit("some other stop")

            return build

        E.load_builder = fake
        try:
            rc = E.main(["--out", tmp, "--only", "library,research"])
        finally:
            E.load_builder = old_loader
        assert rc == 1
        assert not (Path(tmp) / "library.js").exists() and not (Path(tmp) / "research.js").exists()


def test_export_on_an_empty_data_folder():
    """export_web.py on an empty data folder (FACTORLAB_DATA) exits 0. The export writes every
    section file, meta.js has no data date, and no file goes into the data folder."""
    import os
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as tmp:
        data, out = Path(tmp) / "data", Path(tmp) / "out"
        data.mkdir()
        env = dict(os.environ, FACTORLAB_DATA=str(data), FACTORLAB_OFFLINE="1")
        r = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(E.__file__).resolve()),
                "--out",
                str(out),
            ],
            cwd=Path(E.__file__).resolve().parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        _, meta = _payload((out / "meta.js").read_text(encoding="ascii"))
        assert meta["data_through"] is None and meta["data_age"]["newest_bar"] is None
        for s in SECTIONS:
            _, d = _payload((out / f"{s}.js").read_text(encoding="ascii"))
            assert isinstance(d, dict) and meta["sections"][s]["state"] in ("written", "no_data"), meta["sections"][s]
            if meta["sections"][s]["state"] == "no_data":
                assert d["reason"] and not C.privacy_hits(d["reason"]), d["reason"]
        # engine modules may create empty folders at import; no file may appear
        assert not [p for p in data.rglob("*") if p.is_file()], "the export wrote a file into the data folder"


def test_part_files():
    """A builder's with_parts payload writes <section>.<part>.js next to <section>.js: the section
    lists the part names, each part has its own wrapper key, a failing part writes nothing, and a
    later export removes the part files that it does not write."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        payload = C.with_parts(
            {"as_of": "2026-10-01", "days": 2}, {"2026-10-01": {"x": 1}, "index": {"names": ["a", "b"]}}
        )
        n, parts = C.write_section_parts(out, "demo", payload)
        assert sorted(parts) == ["2026-10-01", "index"] and n == (out / "demo.js").stat().st_size
        _, main = _payload((out / "demo.js").read_text(encoding="ascii"))
        assert main == {"as_of": "2026-10-01", "days": 2, "parts": ["2026-10-01", "index"]}, (
            "the parts key is replaced by the names"
        )
        key, part = _payload((out / "demo.2026-10-01.js").read_text(encoding="ascii"))
        assert (key, part) == ("demo.2026-10-01", {"x": 1})
        assert sorted(C.part_files(out, "demo")) == ["2026-10-01", "index"]
        # meta.js lists the parts with their sizes and counts them in the total
        E.write_meta(out, {"demo": "written"})
        _, meta = _payload((out / "meta.js").read_text(encoding="ascii"))
        assert "demo" not in meta["sections"], "only webexport.SECTIONS are listed"
        # a later export without one part removes its file; a plain write_section removes all
        C.write_section(out, "demo", C.with_parts({"days": 1}, {"index": {"names": []}}))
        assert sorted(C.part_files(out, "demo")) == ["index"]
        C.write_section(out, "demo", {"days": 0})
        assert C.part_files(out, "demo") == {}
        # a part over the budget, a private string in a part or a bad name: nothing is written
        for bad in (
            C.with_parts({"a": 1}, {"big": {"x": "a" * 3000}}),
            C.with_parts({"a": 1}, {"p": {"note": "/" + "Users/someone/x"}}),
        ):
            try:
                C.write_section(out, "demo2", bad, max_bytes=2000)
            except (C.SizeError, C.PrivacyError):
                pass
            else:
                raise AssertionError("a bad part was written")
            assert not list(out.glob("demo2*")), "a failing part leaves no file"
        for name in ("Bad", "a.b", "", "-x", "x" * 65):
            try:
                C.with_parts({"a": 1}, {name: {}})
            except ValueError:
                pass
            else:
                raise AssertionError(f"bad part name accepted: {name!r}")
        try:
            C.with_parts({"parts": 1}, {"a": {}})
        except ValueError:
            pass
        else:
            raise AssertionError("a payload with its own 'parts' key was accepted")
        assert not [p for p in out.iterdir() if p.name.startswith(".")], "temp file left behind"
        # section files never count as parts of another section
        C.write_section(out, "paper", {"ok": 1})
        assert C.part_files(out, "pape") == {} and C.part_files(out, "paper") == {}


def test_meta_section_states_as_of_and_parts():
    """meta.sections[s] carries the section's as_of, its state in this run, its part files, and
    keeps "failed" for a section that a later run did not export (its file is from an earlier run)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        C.write_section(out, "library", C.with_parts({"as_of": "2026-10-05", "x": 1}, {"2026-10-04": {"d": 1}}))
        C.write_section(out, "research", {"as_of": "not a date"})
        C.write_section(out, "system", C.no_data_payload(C.NoData("no input")))
        E.write_meta(out, {"library": "written", "research": "failed", "system": "no_data"})
        _, meta = _payload((out / "meta.js").read_text(encoding="ascii"))
        lib = meta["sections"]["library"]
        assert lib["as_of"] == "2026-10-05" and lib["state"] == "written" and lib["available"] is None
        assert lib["parts"] == {"2026-10-04": (out / "library.2026-10-04.js").stat().st_size}
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", lib["written_at"])
        assert meta["sections"]["research"]["as_of"] is None and meta["sections"]["research"]["state"] == "failed"
        assert meta["sections"]["system"]["available"] is False and meta["sections"]["system"]["state"] == "no_data"
        files = sum(f.stat().st_size for f in out.glob("*.js") if f.name != "meta.js")
        assert meta["total_bytes"] == files, "the total counts the section and part files"
        # a later run that exports only the library keeps research "failed"
        E.write_meta(out, {"library": "written"})
        _, meta2 = _payload((out / "meta.js").read_text(encoding="ascii"))
        assert meta2["sections"]["research"]["state"] == "failed" and meta2["sections"]["system"]["state"] == "kept"
        # a damaged file: present, not available
        (out / "system.js").write_text("window.FL=window.FL||{};FL.data[\"system\"]={\"a\":", encoding="ascii")
        _, meta3 = _payload(E.C.render_section("meta", E.build_meta(out)).replace("\n", "") + "\n")
        assert meta3["sections"]["system"]["present"] is True and meta3["sections"]["system"]["available"] is False


def test_export_main_records_states_and_parts():
    """export_web.main() writes the parts of a builder and records each section's state in meta.js.
    A later run that exports one section keeps the state of the others ("kept", or "failed")."""
    old_loader = E.load_builder
    with tempfile.TemporaryDirectory() as tmp:

        def fake(section):
            def build():
                if section == "library":
                    return C.with_parts({"as_of": "2026-10-02"}, {"2026-10-01": {"a": 1}, "2026-10-02": {"a": 2}})
                if section == "research":
                    raise C.NoData("no research input")
                raise ValueError("broken builder")

            return build

        E.load_builder = fake
        try:
            rc = E.main(["--out", tmp, "--only", "library,research,system"])
            _, meta = _payload((Path(tmp) / "meta.js").read_text(encoding="ascii"))
            rc2 = E.main(["--out", tmp, "--only", "research"])
            _, meta2 = _payload((Path(tmp) / "meta.js").read_text(encoding="ascii"))
        finally:
            E.load_builder = old_loader
        assert rc == 1, "a failing section fails the export"
        st = {k: v["state"] for k, v in meta["sections"].items()}
        assert (st["library"], st["research"], st["system"]) == ("written", "no_data", "failed")
        assert sorted(meta["sections"]["library"]["parts"]) == ["2026-10-01", "2026-10-02"]
        assert (Path(tmp) / "library.2026-10-02.js").exists() and not (Path(tmp) / "system.js").exists()
        assert rc2 == 0
        st2 = {k: v["state"] for k, v in meta2["sections"].items()}
        assert (st2["library"], st2["research"], st2["system"]) == ("kept", "no_data", "failed"), st2


def test_export_removes_the_files_of_other_sections():
    """export_web.main() removes a data file whose section is not in SECTIONS, with its part
    files, and prints each name. A data file of a section in SECTIONS, a file without the
    wrapper and a file whose wrapper key is not its name stay."""
    import contextlib
    import io

    def wrap(key: str) -> str:
        return 'window.FL=window.FL||{};FL.data=FL.data||{};FL.data["' + key + '"]={"rows":[]};\n'

    old_loader = E.load_builder
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        files = {
            "extra.js": wrap("extra"),
            "board.js": wrap("board"),
            "board.2026-10-01.js": wrap("board.2026-10-01"),
            "system.js": wrap("system"),
            "notes.js": "// not a data file\n",
            "copy.js": wrap("extra"),
        }
        for name, text in files.items():
            (out / name).write_text(text, encoding="ascii")
        E.load_builder = lambda section: lambda: {"ok": True}
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = E.main(["--out", tmp, "--only", "library"])
        finally:
            E.load_builder = old_loader
        left = sorted(f.name for f in out.iterdir())
        assert rc == 0
        assert left == ["copy.js", "library.js", "meta.js", "notes.js", "system.js"], left
        for name in ("board.2026-10-01.js", "board.js", "extra.js"):
            assert f"[export_web] removed {name}:" in buf.getvalue(), buf.getvalue()
        assert E.remove_other_sections(out) == []


# --------------------------------------------------------------------------- app shell (node)

# A DOM stand-in that runs fl-core.js, fl-charts.js, meta.js and fl-app.js together. A <script>
# appended to <head> runs the file from the web folder (missing file: onerror; any file: onload,
# as a browser does), so the on-demand loader, the routes and the page life run as in a browser.
_SHELL_STUB = r"""
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const WEB = process.argv[1];
class CL { constructor(n) { this.n = n; } g() { return (this.n.attrs["class"] || "").split(/\s+/).filter(Boolean); }
  s(a) { this.n.attrs["class"] = a.join(" "); } add(...c) { const a = this.g(); c.forEach(x => { if (!a.includes(x)) a.push(x); }); this.s(a); }
  remove(...c) { this.s(this.g().filter(x => !c.includes(x))); } contains(c) { return this.g().includes(c); }
  toggle(c, f) { const w = f === undefined ? !this.contains(c) : !!f; if (w) this.add(c); else this.remove(c); return w; } }
class N { constructor(t) { this.tagName = String(t).toUpperCase(); this.children = []; this.childNodes = this.children; this.attrs = {};
    this.style = {}; this.parentNode = null; this._t = ""; this.L = {}; this.nodeType = 1; this.classList = new CL(this); this.dataset = {}; this.hidden = false; }
  get className() { return this.attrs["class"] || ""; } set className(v) { this.attrs["class"] = v; }
  get id() { return this.attrs.id || ""; } set id(v) { this.attrs.id = v; }
  get firstChild() { return this.children[0] || null; } get lastChild() { return this.children[this.children.length - 1] || null; }
  get isConnected() { let n = this; while (n.parentNode) n = n.parentNode; return n === D.documentElement; }
  appendChild(c) { if (c.nodeType === 11) { c.children.slice().forEach(x => this.appendChild(x)); c.children.length = 0; return c; }
    if (c.parentNode) c.parentNode.removeChild(c); c.parentNode = this; this.children.push(c); return c; }
  insertBefore(c, r) { if (!r) return this.appendChild(c); if (c.nodeType === 11) { c.children.slice().forEach(x => this.insertBefore(x, r)); c.children.length = 0; return c; }
    if (c.parentNode) c.parentNode.removeChild(c); const i = this.children.indexOf(r); c.parentNode = this; this.children.splice(i < 0 ? this.children.length : i, 0, c); return c; }
  removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parentNode = null; return c; }
  setAttribute(k, v) { this.attrs[k] = String(v); } getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; } hasAttribute(k) { return k in this.attrs; } setAttributeNS(_, k, v) { this.setAttribute(k, v); }
  addEventListener(t, f) { (this.L[t] = this.L[t] || []).push(f); } removeEventListener(t, f) { const a = this.L[t] || []; const i = a.indexOf(f); if (i >= 0) a.splice(i, 1); } dispatchEvent() { return true; }
  get textContent() { return this.nodeType === 3 ? this._t : this._t + this.children.map(c => c.textContent).join(""); }
  set textContent(v) { this.children.length = 0; this._t = v == null ? "" : String(v); }
  get clientWidth() { return 640; } get clientHeight() { return 240; } get offsetWidth() { return 640; }
  getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 240, right: 640, bottom: 240 }; }
  m(s) { const a = /^\[([\w-]+)(?:=([^\]]+))?\]$/.exec(s); if (a) return a[2] === undefined ? a[1] in this.attrs : this.attrs[a[1]] === a[2].replace(/"/g, "");
    return s[0] === "#" ? this.attrs.id === s.slice(1) : s[0] === "." ? this.classList.contains(s.slice(1)) : this.tagName === s.toUpperCase(); }
  querySelectorAll(s) { const o = []; const w = n => n.children.forEach(c => { if (c.nodeType === 1) { if (c.m(s)) o.push(c); w(c); } }); w(this); return o; }
  querySelector(s) { return this.querySelectorAll(s)[0] || null; }
  closest(s) { let n = this; while (n) { if (n.nodeType === 1 && n.m(s)) return n; n = n.parentNode; } return null; }
  contains(o) { let n = o; while (n) { if (n === this) return true; n = n.parentNode; } return false; }
  focus() {} blur() {} scrollIntoView() {}
  getContext() { return { measureText: s => ({ width: String(s).length * 6.5 }), font: "" }; }
  getBBox() { return { x: 0, y: 0, width: 0, height: 0 }; } }
class T extends N { constructor(t) { super("#text"); this.nodeType = 3; this._t = String(t); } }
const loaded = [];
const D = { documentElement: new N("html"), head: new N("head"), body: new N("body"),
  createElement: t => new N(t), createElementNS: (_, t) => new N(t), createTextNode: t => new T(t),
  createDocumentFragment: () => { const f = new N("#frag"); f.nodeType = 11; return f; },
  createEvent: () => ({ initCustomEvent() {}, initEvent() {} }),
  querySelector: s => D.documentElement.querySelector(s), querySelectorAll: s => D.documentElement.querySelectorAll(s),
  getElementById: i => D.documentElement.querySelector("#" + i), addEventListener() {}, removeEventListener() {}, readyState: "complete" };
D.documentElement.appendChild(D.head); D.documentElement.appendChild(D.body);
const app = new N("div"); app.id = "fl-app"; D.body.appendChild(app);
// window listeners, counted by type; {signal} removes a listener on abort, as a browser does
const WL = {};
function addL(t, f, o) { (WL[t] = WL[t] || []).push(f); if (o && o.signal) o.signal.addEventListener("abort", () => remL(t, f)); }
function remL(t, f) { const a = WL[t] || []; const i = a.indexOf(f); if (i >= 0) a.splice(i, 1); }
function fire(t, ev) { (WL[t] || []).slice().forEach(f => f(ev || { type: t })); }
const loc = { hash: "", search: "", pathname: "/index.html", protocol: "file:" };
const W = { document: D, console, setTimeout, clearTimeout, setInterval: () => 0, clearInterval() {}, requestAnimationFrame: () => 0, cancelAnimationFrame() {},
  addEventListener: addL, removeEventListener: remL, dispatchEvent(ev) { fire(ev.type, ev); return true; },
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} }),
  getComputedStyle: () => ({ getPropertyValue: () => "#888888", scrollMarginTop: "0" }), localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  location: loc, history: { replaceState(_, __, u) { const i = String(u).indexOf("#"); loc.hash = i >= 0 ? String(u).slice(i) : ""; } },
  navigator: { userAgent: "node" }, innerHeight: 800, pageYOffset: 0, scrollTo() {}, performance, AbortController,
  CustomEvent: class { constructor(t, o) { this.type = t; this.detail = o && o.detail; } }, Event: class { constructor(t) { this.type = t; } },
  URLSearchParams, Promise };
W.window = W; W.self = W;
W.FL_MISSING = []; W.FL_BROKEN = []; W.FL_MISS = s => W.FL_MISSING.push(s.getAttribute("src"));
const C = vm.createContext(W);
function run(rel) { vm.runInContext(fs.readFileSync(path.join(WEB, rel), "utf8"), C, { filename: rel }); }
// <script> in <head>: run the file, then onload (a file that throws still loads, as in a browser)
const headAppend = D.head.appendChild.bind(D.head);
D.head.appendChild = function (s) {
  headAppend(s);
  if (s.tagName === "SCRIPT" && s.src) {
    const rel = String(s.src).split("?")[0];
    loaded.push(String(s.src));
    setImmediate(() => {
      const f = path.join(WEB, rel);
      if (!fs.existsSync(f)) { s.onerror && s.onerror({}); return; }
      try { run(rel); } catch (e) { fire("error", { filename: "file:///x/" + rel, message: String(e) }); }
      s.onload && s.onload({});
    });
  }
  return s;
};
const tick = (n = 25) => new Promise(r => { let k = 0; const step = () => (++k >= n ? setTimeout(r, 30) : setImmediate(step)); step(); });
async function go(hash) { loc.hash = hash; fire("hashchange"); await tick(); }
const main = () => D.getElementById("fl-main");
const text = () => main().textContent;
(async () => {
  for (const f of ["assets/fl-core.js", "assets/fl-charts.js", "data/meta.js", "assets/fl-app.js"]) run(f);
  const FL = W.FL;
  await tick();
  const out = { landing: { hash: loc.hash, h1: main().querySelector("h1").textContent }, scripts: loaded.slice(0, 2) };
  out.nav = D.documentElement.querySelectorAll(".fl-nav-group").map(g => [g.querySelector(".fl-nav-label").textContent,
    g.querySelectorAll("a").map(a => a.textContent)]);
  out.current = D.documentElement.querySelector("[aria-current=page]").textContent;
  // the page life: five rounds over every route keep the window listener counts constant
  const routes = ["library", "research", "mine", "system"];
  const counts = [];
  for (let r = 0; r < 5; r++) {
    for (const p of routes) await go("#/" + p);
    counts.push({ theme: (WL["fl:themechange"] || []).length, resize: (WL["resize"] || []).length, hash: (WL["hashchange"] || []).length });
  }
  out.counts = counts;
  out.titles = {};
  for (const p of routes) { await go("#/" + p); out.titles[p] = main().querySelector("h1").textContent; }
  // a theme change redraws the live charts; a sparkline of an old page is gone
  out.liveAfter = FL.charts.prune ? (FL.charts.refresh(), true) : false;
  // the factor route passes its id; a bad id is flagged; a card after the id
  await go("#/factor/CF00001");
  out.factor = JSON.parse(W.__lastCtx || "null");
  out.factorHead = main().querySelector("h1").textContent;
  await go("#/factor/A.b_c-1:2/notes");
  out.factor2 = JSON.parse(W.__lastCtx || "null");
  await go("#/factor/bad%20id");
  out.factor3 = JSON.parse(W.__lastCtx || "null");
  // from the route without an id to a bad id: the page renders again with the flag
  await go("#/factor");
  await go("#/factor/bad%20id");
  out.factor4 = JSON.parse(W.__lastCtx || "null");
  out.factorHref = FL.app.factorHref("x:1.a", "card");
  // a card that the page draws after a part file loads: ctx.cardReady() scrolls to it (true); a card
  // that is not on the page gives false; a scroll of the reader before the part loads gives false
  W.__cardReady = null; await go("#/factor/late/late-part");
  out.cardReady = W.__cardReady;
  W.__cardReady = null; await go("#/factor/late2/no-card");
  out.cardReadyMissing = W.__cardReady;
  W.__cardReady = null; loc.hash = "#/factor/late3/late-part"; fire("hashchange"); fire("wheel"); await tick();
  out.cardReadyScrolled = W.__cardReady;
  // a part file on demand, and a part that is not there
  await go("#/library");
  await tick();
  out.part = W.__part || null;
  out.partMissing = W.__partMissing || null;
  // the deferred card renders after the first paint; a deep link renders it at once
  await go("#/system");
  await go("#/library/late-card");
  out.deepLink = !!D.getElementById("late-card") && D.getElementById("late-card").className;
  out.deferLeft = main().querySelectorAll(".fl-defer").length;
  // a table with pageSize: 2 rows first, then all after "Show all"
  const tb = main().querySelector(".fl-table--stack");
  out.tableRows = tb ? tb.querySelector("tbody").children.length : null;
  out.rowHeader = tb ? tb.querySelector("tbody").children[0].children[0].tagName + ":" + tb.querySelector("tbody").children[0].children[0].getAttribute("scope") : null;
  out.dataLabel = tb ? tb.querySelector("tbody").children[0].children[1].getAttribute("data-label") : null;
  const more = main().querySelector(".fl-table-more");
  if (more) { (more.children[2].L.click || []).forEach(f => f({})); out.tableRowsAll = tb.querySelector("tbody").children.length; out.moreText = more.children[0].textContent; }
  out.cardLevel = main().querySelector("#lvl") ? main().querySelector("#lvl").querySelector(".fl-card-title").tagName : null;
  out.toc = main().querySelector(".fl-toc") ? main().querySelector(".fl-toc").querySelectorAll("a").map(a => a.getAttribute("href")) : null;
  // a missing data file and a file that does not parse: the empty states say which, with both remedies
  await go("#/system");
  out.missingText = text();
  await go("#/research");
  out.brokenText = text();
  out.stale = {};
  out.stale.research = main().querySelectorAll(".fl-badge").map(b => b.textContent);
  await go("#/library");
  out.stale.library = main().querySelectorAll(".fl-badge").map(b => b.textContent);
  out.footer = main().querySelector(".fl-footer").textContent;
  out.footerParas = main().querySelector(".fl-footer").children.length;
  out.snapshot = D.documentElement.querySelector(".fl-snapshot").textContent;
  out.navMissing = D.documentElement.querySelectorAll(".fl-nav-missing").map(sp => sp.parentNode.children[1].textContent);
  // the stale helpers
  out.age = FL.stale.data({ data_through: "2026-10-07" }, Date.UTC(2026, 9, 12, 9));
  out.ageCurrent = FL.stale.data({ data_through: "2026-10-07" }, Date.UTC(2026, 9, 8, 1)).stale;
  // an unknown route and an unknown route with a parameter -> My factors
  await go("#/nope");
  out.unknown = loc.hash;
  await go("#/nope/x");
  out.unknownParam = loc.hash;
  process.stdout.write(JSON.stringify(out));
})().catch(e => { console.error(e && e.stack || e); process.exit(1); });
"""

# Synthetic page files: each registers its route's title; "library" also exercises the shell API.
_PAGE_JS = r"""
(function () {
  var FL = window.FL;
  FL.pages.register({ id: "%(id)s", title: "%(title)s", icon: "overview"%(extra)s,
    render: function (root, ctx) {
      var ui = FL.ui;
      window.__lastCtx = JSON.stringify({ id: ctx.id, param: ctx.nav.param, badParam: ctx.nav.badParam, card: ctx.nav.card, section: ctx.sectionId });
      // a sparkline, a chart and page listeners: all must end with the page
      FL.charts.sparkline(root, { values: [1, 2, 3] });
      FL.charts.line(root, { x: ["2025-01-06", "2025-01-07"], series: [{ id: "a", values: [1, 2] }] });
      ctx.onTheme(function () {});
      ctx.listen(window, "resize", function () {});
      window.addEventListener("fl:themechange", function () {}, { signal: ctx.signal });
      %(body)s
    } });
})();
"""
_SHELL_BODY = r"""
      ui.toc(root, [["lvl", "Level card"], ["late-card", "Late card"]], { page: "library" });
      ui.card({ id: "lvl", title: "Level 2", level: 2 }, root);
      ui.table(root, { columns: [{ key: "n", label: "Name", rowHeader: true }, { key: "v", label: "Value", format: "num" }],
        rows: [{ n: "a", v: 1 }, { n: "b", v: 2 }, { n: "c", v: 3 }, { n: "d", v: 4 }, { n: "e", v: 5 }], pageSize: 2, stack: true });
      ui.defer(root, { id: "late-card" }, function (host) { ui.card({ id: "late-card", title: "Late" }, host); });
"""
_PARTS_BODY = r"""
      ctx.loadPart("2026-10-01").then(function (p) { window.__part = p; });
      ctx.loadPart("2026-01-01").then(null, function (e) { window.__partMissing = e.state; });
"""
_LATE_CARD_BODY = r"""
      if (/^late/.test(ctx.nav.param || "")) ctx.loadPart("2026-10-01").then(function () {
        ui.card({ id: "late-part", title: "Late part" }, root);
        window.__cardReady = ctx.cardReady();
      });
"""


def _shell_fixture(root: Path) -> None:
    """A web folder with the real assets, synthetic data and synthetic pages."""
    import shutil

    shutil.copytree(WEB / "assets", root / "assets")
    (root / "pages").mkdir()
    data = root / "data"
    data.mkdir()
    for sid, title in ROUTES:
        extra = ', section: "library", data: ["library"]' if sid == "factor" else ""
        body = _SHELL_BODY + _PARTS_BODY if sid == "library" else _LATE_CARD_BODY if sid == "factor" else ""
        (root / "pages" / f"{sid}.js").write_text(
            _PAGE_JS % {"id": sid, "title": title, "extra": extra, "body": body}, encoding="utf-8"
        )
    # library: a section with a part file and a data date before the header date
    C.write_section(data, "library", C.with_parts({"as_of": "2026-10-06"}, {"2026-10-01": {"day": "2026-10-01"}}))
    # mine: a section without a data date (the landing page)
    C.write_section(data, "mine", {"schema": 1, "available": True, "as_of": None, "rows": []})
    # system: a missing data file; research: a file that the browser fetches but that does not run,
    # from an export whose last run of the section failed
    (data / "research.js").write_text('window.FL=window.FL||{};FL.data["research"]={"a":', encoding="ascii")
    meta = E.build_meta(data, {"research": "failed"})
    meta["data_through"] = "2026-10-07"
    meta["version"] = V.VERSION  # the footer shows the version of meta.js
    meta["demo"] = False  # the shell shows the data-age badge (a demo data folder sets True)
    C.write_section(data, "meta", meta)


def test_app_shell_in_node():
    """The shell in node: the landing route, the nav groups, the on-demand loader (versioned URLs),
    constant window listeners over five rounds of every route, the factor route and its id, part
    files, deferred cards and deep links (also a card that a page draws after a part file loads),
    the table, toc and card options, the empty states of a missing and a damaged data file, the
    stale badges and the data-age helper. Needs node."""
    node = _node()
    if node is None:
        print("        (skipped the node run: set NODE or FACTORLAB_NODE)")
        return
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "web"
        root.mkdir()
        _shell_fixture(root)
        r = subprocess.run([node, "-e", _SHELL_STUB, str(root)], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"shell run failed: {r.stderr[-1500:]}"
        out = json.loads(r.stdout)
    assert out["landing"] == {"hash": "#/mine", "h1": "My factors"}, out["landing"]
    gen = out["scripts"][0].split("?v=")[1]
    assert sorted(s.split("?")[0] for s in out["scripts"]) == ["data/mine.js", "pages/mine.js"], out["scripts"]
    assert all(s.endswith("?v=" + gen) for s in out["scripts"]) and gen, "data and page URLs carry the export version"
    titles = dict(ROUTES)
    assert out["nav"] == [
        [label, [titles[i] + (" (not available)" if i in ("system", "research") else "") for i in items]]
        for _, label, items in NAV_GROUPS
    ], out["nav"]
    assert out["current"] == "My factors"
    assert len({json.dumps(c, sort_keys=True) for c in out["counts"]}) == 1, (
        f"window listeners grow with route changes: {out['counts']}"
    )
    assert out["titles"] == {i: t for i, t in NAV}, out["titles"]
    assert out["factor"] == {
        "id": "factor",
        "param": "CF00001",
        "badParam": False,
        "card": None,
        "section": "library",
    }, out["factor"]
    assert out["factorHead"] == "Factor test"
    assert (out["factor2"]["param"], out["factor2"]["card"]) == ("A.b_c-1:2", "notes")
    assert out["factor3"]["param"] is None and out["factor3"]["badParam"] is True
    assert out["factor4"]["param"] is None and out["factor4"]["badParam"] is True, out["factor4"]
    assert out["factorHref"] == "#/factor/x:1.a/card"
    assert out["cardReady"] is True, "a card drawn after a part file loads: the route scrolls to it"
    assert out["cardReadyMissing"] is False and out["cardReadyScrolled"] is False, (
        out["cardReadyMissing"],
        out["cardReadyScrolled"],
    )
    assert out["part"] == {"day": "2026-10-01"} and out["partMissing"] == "missing", (out["part"], out["partMissing"])
    assert out["deepLink"] == "fl-card" and out["deferLeft"] == 0, "a deep link renders its deferred card"
    assert out["tableRows"] == 2 and out["tableRowsAll"] == 5 and out["moreText"] == "All 5 rows.", out
    assert out["rowHeader"] == "TH:row" and out["dataLabel"] == "Value"
    assert out["cardLevel"] == "H2"
    assert out["toc"] == ["#/library/lvl", "#/library/late-card"]
    assert (
        "data/system.js is not in this folder." in out["missingText"]
        and "./setup.sh" in out["missingText"]
        and "cd engine && .venv/bin/python -B export_web.py" in out["missingText"]
    ), out["missingText"][:400]
    assert "data/research.js does not parse" in out["brokenText"], out["brokenText"][:400]
    assert any("data through 6 Oct 2026" in b for b in out["stale"]["library"]), out["stale"]
    assert any("The last export of this section failed" in b for b in out["stale"]["research"]), out["stale"]
    # the footer: the version and the licenses only (the pages name a missing or broken file)
    assert out["footer"].startswith("Factorlab " + V.VERSION + ". Code: MIT License."), out["footer"][-400:]
    assert "CC BY-NC-SA 4.0" in out["footer"] and "Binance Vision" in out["footer"]
    assert out["footerParas"] == 1 and "./setup.sh" not in out["footer"], out["footer"]
    assert "data/research.js" not in out["footer"] and "sections exported" not in out["footer"]
    # the data-age badge comes first in the header (its icon "!" then its text)
    assert out["snapshot"].startswith("!Data is ") and "To add them: ./setup.sh" in out["snapshot"], out["snapshot"]
    assert set(out["navMissing"]) >= {"System", "Search tools"}, out["navMissing"]
    assert out["age"]["ageDays"] == 5 and out["age"]["missingDays"] == 4 and out["age"]["stale"]
    assert out["age"]["text"] == "Data is 4 days behind: last bar 7 Oct 2026; 4 closed days are missing."
    assert out["ageCurrent"] is False, "on the day after the bar the data is current"
    assert out["unknown"] == out["unknownParam"] == "#/mine", (out["unknown"], out["unknownParam"])


def test_shared_css_rules():
    """The shared CSS: no text uses --muted, the control token exists in every theme, the light
    slots c3-c5 clear 3:1 on the surface, and the forced-colors block is there."""
    css = (WEB / "assets" / "app.css").read_text(encoding="utf-8")
    # --muted colours marks only: an icon (svg, currentColor) may use it, a text rule may not
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if re.search(r"(?<![-\w])(?:color|fill)\s*:\s*var\(--muted\)", body):
            assert all(x.strip().endswith("svg") for x in sel.split(",")), f"text uses --muted: {sel.strip()}"
    light = _css_block(css, r"(?m)^:root")
    assert "--control-border" in light

    def lum(h):
        c = [int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    def ratio(a, b):
        la, lb = lum(a.strip()), lum(b.strip())
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)

    dark = _css_block(css, r'(?m)^:root\[data-theme="dark"\]')
    for block in (light, dark):
        for surf in ("--bg", "--surface", "--surface-2"):
            assert ratio(block["--control-border"], block[surf]) >= 3, f"--control-border on {surf}"
        for tok in (
            "--c1",
            "--c2",
            "--c3",
            "--c4",
            "--c5",
            "--c7",
            "--c8",
        ):
            assert ratio(block[tok], block["--surface"]) >= 3, (
                f"{tok} on --surface: {ratio(block[tok], block['--surface']):.2f}"
            )
    # colour-blind separation of the light c2 (orange) and c4: a chart without entity colors draws its
    # series in slot order (fl-charts.js), so c2 and c4 can share a chart.
    # OKLab distance x 100 after a full protan or deutan simulation (the 2009 physiologically based
    # simulation model, severity 1.0, linear RGB)
    sims = {
        "protan": ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
        "deutan": ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
    }

    def lin(h):
        c = [int(h.strip()[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]

    def oklab(rgb):
        r, g, b = [min(1.0, max(0.0, x)) for x in rgb]
        lms = [
            (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3),
            (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3),
            (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3),
        ]
        ll, m, q = lms
        return (
            0.2104542553 * ll + 0.7936177850 * m - 0.0040720468 * q,
            1.9779984951 * ll - 2.4285922050 * m + 0.4505937099 * q,
            0.0259040371 * ll + 0.7827717662 * m - 0.8086757660 * q,
        )

    for kind, mat in sims.items():
        a2, a4 = (
            [sum(mat[i][j] * c[j] for j in range(3)) for i in range(3)]
            for c in (lin(light["--c2"]), lin(light["--c4"]))
        )
        d = 100 * math.dist(oklab(a2), oklab(a4))
        assert d >= 8, f"light c2/c4 separation under {kind} simulation: {d:.1f} < 8"
    assert "@media (forced-colors: active)" in css
    core = (WEB / "assets" / "fl-core.js").read_text(encoding="utf-8")
    for tok in ("c3", "c4", "c5"):
        assert re.search(r'{}: "{}"'.format(tok, light["--" + tok].strip()), core), f"STATIC.light {tok} equals app.css"


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Run it with no arguments. It prints one PASS, FAIL or SKIP line for each test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_args()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} export_web tests passed")
    raise SystemExit(1 if failed else 0)
