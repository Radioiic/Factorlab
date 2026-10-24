"""Tests of the one version source (the file VERSION, engine/version.py) and of the license files.

    python test_version.py

Checks:
    1. VERSION holds one Semantic Version (MAJOR.MINOR.PATCH) and version.VERSION is that text;
    2. version.read() refuses a text that is not a Semantic Version;
    3. `python version.py` prints the version;
    4. the factor library (factorlab.__version__) and CITATION.cff give the same version;
    5. LICENSE is the MIT License with the copyright line of Factorlab, and CITATION.cff and
       REUSE.toml give the SPDX identifier MIT;
    6. the license files are LICENSE, library/LICENSE (the same text), LICENSES/ISC.txt and
       LICENSES/MIT.txt, and REUSE.toml names only MIT and ISC;
    7. the issue forms ask for the version and have the two privacy check boxes;
    8. CHANGELOG.md: its form (Keep a Changelog), its top released section is VERSION, the
       date of that section is the date-released of CITATION.cff, and `version.py --changelog`
       prints the text of a section;
    9. the README badges: the version badge is VERSION, the Python badge gives the versions of
       the test workflow and of library/pyproject.toml, and no badge reads the repository from
       another site;
   10. each fixed copy of the version in the documents and the issue forms is VERSION, and a
       version tag v* (a tag run of the workflow) is "v" + VERSION;
   11. each relative link of the root documents opens a file of the repository, and each
       #anchor is a heading of that file;
   12. no text file of the repository holds an e-mail address, other than a GitHub no-reply
       address and an example domain, and SECURITY.md and CODE_OF_CONDUCT.md name the private
       reporting form of GitHub.
The tests need no market data and write only into temporary folders.
"""

from __future__ import annotations

import datetime as dt
import html
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

import version as V

ENGINE = Path(__file__).resolve().parent
REPO = ENGINE.parent
PY = sys.executable
COPYRIGHT = "Copyright (c) 2026 Radioiic"
# folders that hold no file of the repository: environments and caches (any depth), and the
# data and output folders that setup.sh makes (at these paths only)
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}
_SKIP_PATHS = (
    "engine/data/",
    "engine/logs/",
    "engine/.tmp/",
    "engine/outputs/",
    "web/data/",
    "library/catalog/",
    "library/data/",
    "build/",
)
_TEXT = {".py", ".js", ".md", ".sh", ".toml", ".yml", ".yaml", ".cff", ".txt", ".html", ".css", ".ini", ".template", ""}


def _repo_files() -> list[Path]:
    """The files of the repository: git's list (tracked and not ignored) in a work tree, else a
    walk without the environment and data folders."""
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-co", "--exclude-standard"], capture_output=True, text=True
        )
        if r.returncode == 0 and r.stdout.strip():
            return [REPO / ln for ln in r.stdout.splitlines() if ln.strip() and (REPO / ln).is_file()]
    except OSError:
        pass
    out = []
    for p in REPO.rglob("*"):
        rel = p.relative_to(REPO)
        if p.is_file() and not (set(rel.parts[:-1]) & _SKIP_DIRS) and not rel.as_posix().startswith(_SKIP_PATHS):
            out.append(p)
    return out


def test_version_file_is_one_semantic_version():
    text = (REPO / "VERSION").read_text(encoding="ascii")
    assert text.endswith("\n") and text.count("\n") == 1, "VERSION holds one line"
    assert V.SEMVER.match(text.strip()), text
    assert V.VERSION == text.strip() == V.read()
    for good in ("0.1.0", "1.20.3", "1.0.0-rc.1", "2.0.0+build.5"):
        assert V.SEMVER.match(good), good


def test_read_refuses_a_bad_version():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "VERSION"
        for bad in ("", "1.0", "v1.0.0", "01.0.0", "1.0.0-", "1.0.x", "1.0.0 x"):
            p.write_text(bad + "\n", encoding="ascii")
            try:
                V.read(p)
            except ValueError as exc:
                assert "MAJOR.MINOR.PATCH" in str(exc)
            else:
                raise AssertionError(f"read() accepted {bad!r}")
        p.write_text("3.4.5\n", encoding="ascii")
        assert V.read(p) == "3.4.5"


def test_version_cli_prints_the_version():
    r = subprocess.run([PY, "-B", str(ENGINE / "version.py")], capture_output=True, text=True, cwd=ENGINE)
    assert r.returncode == 0 and r.stdout == V.VERSION + "\n", (r.returncode, r.stdout, r.stderr)


def test_library_and_citation_use_the_same_version():
    r = subprocess.run(
        [PY, "-B", "-c", "import factorlab; print(factorlab.__version__)"],
        capture_output=True,
        text=True,
        cwd=REPO / "library",
    )
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout.strip() == V.VERSION, r.stdout
    cff = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"^version: " + re.escape(V.VERSION) + r"$", cff, re.M), "CITATION.cff version"
    assert re.search(r"^cff-version: 1\.2\.0$", cff, re.M)
    assert re.search(r"^title: Factorlab$", cff, re.M)
    assert re.search(r"^  - alias: Radioiic$", cff, re.M)
    # a package file gives no second, fixed version (it can read VERSION, or give none)
    for rel in ("pyproject.toml", "library/pyproject.toml"):
        p = REPO / rel
        if p.is_file():
            fixed = re.findall(r'^version\s*=\s*"([^"]*)"', p.read_text(encoding="utf-8"), re.M)
            assert all(v == V.VERSION for v in fixed), (rel, fixed)


def test_license_is_mit():
    text = (REPO / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License\n\n" + COPYRIGHT + "\n\n"), text[:120]
    assert "Permission is hereby granted, free of charge" in text
    assert 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND' in text
    cff = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"^license: MIT$", cff, re.M)
    reuse = (REPO / "REUSE.toml").read_text(encoding="utf-8")
    assert re.search(
        r'^path = "\*\*"\nprecedence = "aggregate"\nSPDX-FileCopyrightText = "2026 Radioiic"\n'
        r'SPDX-License-Identifier = "MIT"$',
        reuse,
        re.M,
    ), "REUSE.toml: MIT for every file"
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    lic = readme.split("\n## License\n", 1)[1].split("\n## ", 1)[0]
    assert COPYRIGHT in lic and "[MIT License](LICENSE)" in lic, lic[:300]
    notices = (REPO / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "Factorlab is under the MIT License" in notices
    # the Binance Vision values in a test are facts with a credit line; no share-alike claim on the file
    binance = notices.split("## Values from the Binance Vision archive", 1)[1].split("\n## ", 1)[0]
    assert "Credit: Binance Vision." in binance and "CC BY-NC-SA" not in binance, binance


def test_license_files_are_mit_and_isc():
    lic = sorted(
        p.relative_to(REPO).as_posix()
        for p in _repo_files()
        if p.name.upper().startswith("LICENSE") or p.parent.name == "LICENSES"
    )
    assert lic == ["LICENSE", "LICENSES/ISC.txt", "LICENSES/MIT.txt", "library/LICENSE"], lic
    # the package of the factor library carries the same MIT text
    assert (REPO / "library" / "LICENSE").read_bytes() == (REPO / "LICENSE").read_bytes()
    assert re.search(
        r'^license-files = \["LICENSE"\]$', (REPO / "library" / "pyproject.toml").read_text(encoding="utf-8"), re.M
    )
    reuse = (REPO / "REUSE.toml").read_text(encoding="utf-8")
    ids = set(re.findall(r'^SPDX-License-Identifier = "([^"]+)"$', reuse, re.M))
    assert ids and ids <= {"MIT", "MIT AND ISC"}, ids


def test_issue_forms_ask_for_the_version_and_the_privacy_checks():
    forms = REPO / ".github" / "ISSUE_TEMPLATE"
    bug = (forms / "bug_report.yml").read_text(encoding="utf-8")
    for label in ("Version or commit", "System", "Python version", "Command", "Message", "Expected result"):
        assert f"label: {label}\n" in bug, label
    assert (
        "I attached no market data" in bug and "My messages and logs contain no user name and no private path." in bug
    )
    assert (forms / "question.yml").is_file()
    config = (forms / "config.yml").read_text(encoding="utf-8")
    assert "blank_issues_enabled: false" in config and "SECURITY.md" in config
    pr = (REPO / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    assert pr.startswith("Factorlab does not accept pull requests.")


# --------------------------------------------------------------------------- CHANGELOG.md

REPO_URL = "https://github.com/Radioiic/Factorlab"
CHANGE_KINDS = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")


def _semver_key(v: str) -> tuple:
    """The precedence of a version (Semantic Versioning 2.0.0, section 11): MAJOR.MINOR.PATCH as
    numbers; a pre-release comes before its release; the pre-release identifiers compare one by
    one, a numeric identifier as a number and before a text identifier; build metadata has no
    effect."""
    core = re.match(r"(\d+)\.(\d+)\.(\d+)", v)
    assert core, v
    pre = v.split("+", 1)[0][core.end() :].lstrip("-")
    ids = tuple((0, int(x), "") if x.isdigit() else (1, 0, x) for x in pre.split(".")) if pre else ()
    return (int(core.group(1)), int(core.group(2)), int(core.group(3)), 0 if pre else 1, ids)


def _changelog() -> tuple[list[str], list[tuple[str, str, int]]]:
    """The lines of CHANGELOG.md and its released sections as (version, date, line index)."""
    lines = (REPO / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    releases = [(m.group(1), m.group(2), i) for i, ln in enumerate(lines) if (m := V.RELEASE_HEADING.match(ln))]
    return lines, releases


def test_changelog_top_release_is_version():
    lines, releases = _changelog()
    assert lines[0] == "# Changelog", lines[0]
    h2 = [ln for ln in lines if ln.startswith("## ")]
    assert h2 and h2[0] == "## [Unreleased]", h2[:2]
    assert all(ln == "## [Unreleased]" or V.RELEASE_HEADING.match(ln) for ln in h2), h2
    assert releases, "CHANGELOG.md has no released section"
    version, date, _ = releases[0]
    assert version == V.VERSION, f"the top released section is {version}, VERSION is {V.VERSION}"
    day = dt.date.fromisoformat(date)
    assert day <= dt.datetime.now(dt.UTC).date(), f"the date {date} is after today (UTC)"


def test_changelog_form():
    lines, releases = _changelog()
    versions = [v for v, _, _ in releases]
    assert len(set(versions)) == len(versions), versions
    assert all(V.SEMVER.match(v) for v in versions), versions
    assert versions == sorted(versions, key=_semver_key, reverse=True), "the newest version is not first"
    dates = [d for _, d, _ in releases]
    today = dt.datetime.now(dt.UTC).date()
    for d in dates:
        assert dt.date.fromisoformat(d) <= today, f"the date {d} is not a date before or on today (UTC)"
    assert dates == sorted(dates, reverse=True), "the dates do not go down"
    # each subsection is one of the Keep a Changelog kinds, and each released section has items
    for ln in lines:
        if ln.startswith("### "):
            assert ln[4:] in CHANGE_KINDS, ln
    starts = [i for i, ln in enumerate(lines) if ln.startswith("## ")] + [len(lines)]
    for version, _, i in releases:
        end = min(j for j in starts if j > i)
        assert any(ln.lstrip().startswith("- ") for ln in lines[i:end]), f"the section {version} has no item"
    # each item is on one line: a GitHub release shows each line break of its text
    body = lines[starts[0] :]
    for k, ln in enumerate(body[1:], start=1):
        prev = body[k - 1]
        if ln.strip() and prev.strip() and not re.match(r"^(#{2,3} |\s*- |\[[^\]]+\]: )", ln):
            raise AssertionError(f"CHANGELOG.md: an item continues on the next line: {ln.strip()!r}")
    # the link definitions: one for Unreleased (from the current version) and one for each version.
    # Markdown uses the first definition of a label, so each label occurs one time only.
    pairs = re.findall(r"^\[([^\]]+)\]: (\S+)$", "\n".join(lines), re.M)
    labels = [label.lower() for label, _ in pairs]
    assert len(set(labels)) == len(labels), f"a link label has two definitions: {labels}"
    links = dict(pairs)
    assert links.get("Unreleased") == f"{REPO_URL}/compare/v{V.VERSION}...HEAD", links.get("Unreleased")
    for version in versions:
        assert links.get(version) == f"{REPO_URL}/releases/tag/v{version}", (version, links.get(version))
    assert set(links) == {"Unreleased", *versions}, sorted(links)


def test_citation_date_is_the_changelog_date():
    _, releases = _changelog()
    cff = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    found = re.search(r"^date-released: (\d{4}-\d{2}-\d{2})$", cff, re.M)
    assert found, "CITATION.cff has no date-released"
    assert (V.VERSION, found.group(1)) == releases[0][:2], (found.group(1), releases[0][:2])


def test_changelog_section_command():
    text = V.changelog_section()
    assert text.strip() and text.endswith("\n")
    assert not [ln for ln in text.splitlines() if ln.startswith("## ") or V.LINK_DEFINITION.match(ln)], text
    r = subprocess.run([PY, "-B", str(ENGINE / "version.py"), "--changelog"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == text, r.stderr[-400:]
    r = subprocess.run([PY, "-B", str(ENGINE / "version.py"), "--changelog", "99.0.0"], capture_output=True, text=True)
    assert r.returncode != 0 and "99.0.0" in r.stderr, (r.returncode, r.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "CHANGELOG.md"
        f.write_text(
            "# Changelog\n\n## [Unreleased]\n\n## [1.1.0] - 2026-01-02\n\n### Fixed\n\n- b\n\n"
            "## [1.0.0] - 2026-01-01\n\n### Added\n\n- a\n\n[1.0.0]: https://example.org/1.0.0\n",
            encoding="utf-8",
        )
        assert V.changelog_section("1.1.0", f) == "### Fixed\n\n- b\n"
        assert V.changelog_section("1.0.0", f) == "### Added\n\n- a\n"
        for absent in ("Unreleased", "1.2.0", "1.0"):
            try:
                V.changelog_section(absent, f)
            except ValueError:
                continue
            raise AssertionError(f"changelog_section accepted {absent!r}")


# --------------------------------------------------------------------------- badges and fixed versions


def _badges() -> list[tuple[str, str, str]]:
    """The badges between the title and the first paragraph of README.md as (alt, image, link)."""
    lines = (REPO / "README.md").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# Factorlab" and lines[1] == "", lines[:2]
    block = []
    for ln in lines[2:]:
        if not ln.strip():
            break
        block.append(ln)
    badge = re.compile(r"^\[!\[([^\]]+)\]\((\S+)\)\]\((\S+)\)$")
    out = []
    for ln in block:
        m = badge.match(ln)
        assert m, f"README.md: a line between the title and the first paragraph is not a badge: {ln}"
        out.append(m.groups())
    return out


def _shields_text(text: str) -> str:
    """The badge text of a static shields.io badge: "--" is "-", "__" is "_", "_" is a space."""
    text = urllib.parse.unquote(text)
    return text.replace("--", "\0").replace("__", "\1").replace("_", " ").replace("\0", "-").replace("\1", "_")


def test_readme_badges():
    found = _badges()
    assert len(found) == 4, f"README.md has {len(found)} badges, not 4: {found}"
    for _alt, img, _link in found:
        # a static badge or the badge of the workflow: no badge reads the repository from another site
        assert img.startswith(("https://img.shields.io/badge/", f"{REPO_URL}/actions/workflows/")), img
    badges = {alt.split(" ")[0].rstrip(":"): (alt, img, link) for alt, img, link in found}
    assert set(badges) == {"Test", "License", "Python", "Version"}, sorted(badges)
    alt, img, link = badges["Test"]
    m = re.match(re.escape(f"{REPO_URL}/actions/workflows/") + r"([^/]+)/badge\.svg$", img)
    assert m and (REPO / ".github" / "workflows" / m.group(1)).is_file(), img
    assert link == f"{REPO_URL}/actions/workflows/{m.group(1)}", link
    alt, img, link = badges["License"]
    assert (alt, img, link) == ("License: MIT", "https://img.shields.io/badge/license-MIT-blue", "LICENSE")
    alt, img, link = badges["Version"]
    m = re.match(r"^https://img\.shields\.io/badge/version-(.+)-blue$", img)
    assert m and _shields_text(m.group(1)) == V.VERSION, img
    assert (alt, link) == (f"Version {V.VERSION}", "CHANGELOG.md"), (alt, link)
    # the Python badge: the versions of the test workflow and of the package of the library
    alt, img, link = badges["Python"]
    m = re.match(r"^https://img\.shields\.io/badge/python-(.+)-blue$", img)
    assert m, img
    shown = [v.strip() for v in _shields_text(m.group(1)).split("|")]
    flow = (REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    tested = sorted(set(re.findall(r'python-version: "(3\.\d+)"', flow)), key=lambda v: int(v.split(".")[1]))
    lib = (REPO / "library" / "pyproject.toml").read_text(encoding="utf-8")
    classified = re.findall(r'"Programming Language :: Python :: (3\.\d+)"', lib)
    floor = re.search(r'^requires-python = ">=(3\.\d+)"$', lib, re.M)
    assert shown == tested == classified, (shown, tested, classified)
    assert floor and floor.group(1) == shown[0], floor
    assert alt == "Python " + " and ".join(shown) and link == "#requirements", (alt, link)


def test_fixed_copies_of_the_version():
    """The documents and the issue forms give the version in a few fixed places; each one is VERSION.
    CHANGELOG.md has a section for each version and is not checked here."""
    patterns = (r"`Factorlab (\d+\.\d+\.\d+[^`]*)`", r'placeholder: "(\d+\.\d+\.\d+[^"]*)"')
    found = []
    for p in _repo_files():
        rel = p.relative_to(REPO).as_posix()
        if p.suffix not in {".md", ".yml", ".yaml", ".toml", ".cff"} or rel == "CHANGELOG.md":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for pat in patterns:
            found += [(rel, v) for v in re.findall(pat, text)]
    assert found, "no fixed copy of the version found: the patterns of this test are out of date"
    stale = [(rel, v) for rel, v in found if v != V.VERSION]
    assert not stale, f"VERSION is {V.VERSION}: {stale}"


def test_release_tag_is_the_version():
    """Only on a tag run of the workflow, or when HEAD has a version tag v*."""
    tag = os.environ.get("GITHUB_REF_NAME", "") if os.environ.get("GITHUB_REF_TYPE") == "tag" else ""
    if not tag:
        r = subprocess.run(
            ["git", "-C", str(REPO), "describe", "--exact-match", "--tags", "--match", "v*", "HEAD"],
            capture_output=True,
            text=True,
        )
        tag = r.stdout.strip() if r.returncode == 0 else ""
        if tag:
            # the tag belongs to the commit: compare it with the VERSION of the commit, so that the
            # files of the next release can change before their commit
            r = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:VERSION"], capture_output=True, text=True)
            committed = r.stdout.strip() if r.returncode == 0 else ""
            if committed != V.VERSION:
                print(f"        (VERSION changed after the commit of {tag}: the tag check does not apply)")
                return
    if not tag:
        print("        (no version tag on this commit: the tag check does not apply)")
        return
    assert tag == "v" + V.VERSION, f"the tag {tag} is not v{V.VERSION}"
    V.changelog_section(V.VERSION)  # raises when CHANGELOG.md has no released section for VERSION


# --------------------------------------------------------------------------- contact channels

# the same e-mail pattern as the export check of the dashboard data (webexport/common.py)
_MAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
# the commit address of a GitHub account (ID+name), and the reserved example domains (RFC 2606)
_NOREPLY = re.compile(r"\d+\+[A-Za-z0-9-]+@users\.noreply\.github\.com")
_EXAMPLE_DOMAINS = ("example.org", "example.com", "example.invalid")
# the steps of the private reporting form of GitHub in SECURITY.md and CODE_OF_CONDUCT.md
REPORT_FORM = (
    "private reporting form",
    "the tab **Security and quality**",
    "Click **Report a vulnerability**.",
    "Click **Submit report**.",
)


def _mail_allowed(addr: str) -> bool:
    domain = addr.rsplit("@", 1)[1].lower()
    return bool(_NOREPLY.fullmatch(addr)) or any(domain == d or domain.endswith("." + d) for d in _EXAMPLE_DOMAINS)


def test_mail_check_allows_only_the_noreply_form_and_the_example_domains():
    for good in ("1+name@users.noreply.github.com", "a@example.org", "b@docs.example.com", "c@example.invalid"):
        assert _mail_allowed(good), good
    # the addresses that the check refuses are made from two parts: this file holds no such address
    bad = [
        "@".join(parts)
        for parts in (
            ("name", "users.noreply.github.com"),  # no account ID
            ("1+name", "noreply.github.com"),
            ("a", "example.net.io"),
            ("a", "notexample.org"),
            ("first.last", "mail.provider.net"),
        )
    ]
    for addr in bad:
        assert not _mail_allowed(addr), addr
    found = _MAIL.findall(f"Write to <{bad[-1]}>; the commits use 1+x@users.noreply.github.com.")
    assert found == [bad[-1], "1+x@users.noreply.github.com"], found


def test_no_mail_address_in_the_repository():
    """The contact channels are GitHub only: no file of the repository holds an e-mail address,
    other than a GitHub no-reply address and an example domain. Binary files and the hash lines
    of engine/requirements*.txt are not read."""
    bad = []
    n = 0
    for p in _repo_files():
        rel = p.relative_to(REPO).as_posix()
        data = p.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        n += 1
        hashes = re.fullmatch(r"engine/requirements[^/]*\.txt", rel) is not None
        for i, ln in enumerate(text.splitlines(), start=1):
            if hashes and "--hash=" in ln:
                continue
            bad += [(rel, i, addr) for addr in _MAIL.findall(ln) if not _mail_allowed(addr)]
    assert n > 100, f"only {n} text files read: the file list of this test is out of date"
    assert not bad, f"an e-mail address in the repository: {bad}"


def test_reports_use_the_private_reporting_form():
    """SECURITY.md and CODE_OF_CONDUCT.md give the steps of the private reporting form of GitHub."""
    for rel in ("SECURITY.md", "CODE_OF_CONDUCT.md"):
        text = (REPO / rel).read_text(encoding="utf-8")
        missing = [s for s in REPORT_FORM if s not in text]
        assert not missing, (rel, missing)
    conduct = (REPO / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
    assert 'Write "Code of conduct" in the title.' in conduct
    config = (REPO / ".github" / "ISSUE_TEMPLATE" / "config.yml").read_text(encoding="utf-8")
    assert "private reporting form" in config


# --------------------------------------------------------------------------- links of the root documents

ROOT_DOCS = (
    "README.md",
    "CHANGELOG.md",
    "ROADMAP.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "THIRD_PARTY_NOTICES.md",
)


# The link check reads this part of GitHub Markdown: ATX headings (with or without closing #),
# setext headings, fenced code blocks (``` and ~~~), code spans, inline links and images,
# reference definitions on their own line, and <...> destinations. It does not read indented
# code blocks or HTML blocks: write code in fenced blocks in the root documents.
_FENCE = re.compile(r"^ {0,3}(```|~~~)")


def _heading_lines(text: str) -> list[str]:
    """The text of each heading of a Markdown text, outside the fenced code blocks."""
    out = []
    fence = ""
    prev = ""
    for ln in text.splitlines():
        m = _FENCE.match(ln)
        if m:
            fence = "" if fence == m.group(1) else (fence or m.group(1))
            prev = ""
            continue
        if fence:
            continue
        atx = re.match(r"^ {0,3}#{1,6}[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$", ln)
        if atx:
            out.append(atx.group(1))
        elif re.match(r"^ {0,3}(=+|-+)[ \t]*$", ln) and prev.strip() and not re.match(r"^\s*([-*+]|\d+\.) ", prev):
            out.append(prev.strip())  # a setext heading: the line above === or ---
        prev = ln
    return out


def _slugs(path: Path) -> set[str]:
    """The anchors of the headings of a Markdown file, as GitHub makes them: the text in lower
    case, without punctuation and emphasis marks, with "-" for each space; "-1", "-2" for a
    heading text that occurs again."""
    seen: dict[str, int] = {}
    out = set()
    for heading in _heading_lines(path.read_text(encoding="utf-8")):
        text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # a link or an image gives its text
        text = html.unescape(text.replace("`", ""))
        text = re.sub(r"(?<![\w])[*_]+|[*_]+(?![\w])", "", text)  # emphasis marks around words
        slug = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        out.add(slug if n == 0 else f"{slug}-{n}")
    return out


def _links(text: str) -> list[str]:
    """The link targets of a Markdown text, without the fenced code blocks and the code spans."""
    kept = []
    fence = ""
    for ln in text.splitlines():
        m = _FENCE.match(ln)
        if m:
            fence = "" if fence == m.group(1) else (fence or m.group(1))
            continue
        if not fence:
            kept.append(ln)
    text = re.sub(r"`[^`\n]*`", "", "\n".join(kept))
    inline = re.findall(r"\]\(\s*(<[^>]*>|[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)", text)
    # a reference definition is a whole line; a footnote ([^1]: text) is not a link
    defined = re.findall(r"^ {0,3}\[(?!\^)[^\]]+\]:\s+(<[^>]*>|\S+)(?:\s+\"[^\"]*\")?\s*$", text, re.M)
    return [t[1:-1] if t.startswith("<") and t.endswith(">") else t for t in inline + defined]


def test_relative_links_of_the_root_documents():
    files = {p.relative_to(REPO).as_posix() for p in _repo_files()}
    folders = {f.rsplit("/", 1)[0] for f in files if "/" in f}
    bad = []
    n = 0
    for doc in ROOT_DOCS:
        src = REPO / doc
        if not src.is_file():
            continue
        for target in _links(src.read_text(encoding="utf-8")):
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):  # https:, mailto: and other schemes
                continue
            n += 1
            path, _, anchor = target.partition("#")
            path = urllib.parse.unquote(path.split("?", 1)[0])
            anchor = urllib.parse.unquote(anchor)
            base = REPO if path.startswith("/") else src.parent  # "/x" is the root of the repository
            rel = (base / path.lstrip("/")).resolve() if path else src.resolve()
            try:
                key = rel.relative_to(REPO.resolve()).as_posix()
            except ValueError:
                bad.append((doc, target, "outside the repository"))
                continue
            if key not in files and key.rstrip("/") not in folders:
                bad.append((doc, target, "no such file"))
            elif not anchor or (not key.endswith(".md") and re.fullmatch(r"L\d+(-L\d+)?", anchor)):
                continue  # no anchor, or a line anchor of a code file
            elif key not in files or not key.endswith(".md") or anchor not in _slugs(REPO / key):
                bad.append((doc, target, "no such heading"))
    assert n > 20, f"only {n} relative links found: the link pattern of this test is out of date"
    assert not bad, bad


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} version and license tests passed")
    raise SystemExit(1 if failed else 0)
