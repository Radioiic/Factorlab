"""The version of Factorlab. The one source is the file VERSION at the root of the repository.

    python version.py                    # print the version
    python version.py --changelog        # print the CHANGELOG.md section of the version
    python version.py --changelog 0.2.1  # print the CHANGELOG.md section of 0.2.1

The factor library (factorlab.__version__), the dashboard (meta.js, the footer and the
System page) and the log of update.py all read this file. The version has the Semantic
Versioning 2.0.0 form MAJOR.MINOR.PATCH. CHANGELOG.md at the root of the repository has one
section for each released version, with the heading "## [X.Y.Z] - YYYY-MM-DD" (a UTC date).
The GitHub release of a version has the text of its section.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "VERSION"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"

# MAJOR.MINOR.PATCH, an optional pre-release (-...) and optional build metadata (+...)
SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def read(path: Path | str = VERSION_FILE) -> str:
    """The version in ``path``. Raises ValueError when the text is not a Semantic Version."""
    text = Path(path).read_text(encoding="ascii").strip()
    if not SEMVER.match(text):
        raise ValueError(f"{Path(path).name} does not hold a version of the form MAJOR.MINOR.PATCH: {text!r}")
    return text


VERSION = read()

# "## [0.2.1] - 2026-10-24": the heading of a released version in CHANGELOG.md
RELEASE_HEADING = re.compile(r"^## \[([^\]]+)\] - (\d{4}-\d{2}-\d{2})(?: \[YANKED\])?$")
# "[0.2.1]: https://...": a link definition at the end of CHANGELOG.md
LINK_DEFINITION = re.compile(r"^\[[^\]]+\]: \S+$")


def changelog_section(version: str = VERSION, path: Path | str = CHANGELOG_FILE) -> str:
    """The text of the section of ``version`` in CHANGELOG.md, without its heading.

    The section ends at the next "## " heading or at the link definitions. Raises ValueError
    when CHANGELOG.md has no released section for ``version``."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if (m := RELEASE_HEADING.match(ln)) and m.group(1) == version),
        None,
    )
    if start is None:
        raise ValueError(f"{Path(path).name} has no section '## [{version}] - YYYY-MM-DD'")
    body = []
    for ln in lines[start + 1 :]:
        if ln.startswith("## ") or LINK_DEFINITION.match(ln):
            break
        body.append(ln)
    text = "\n".join(body).strip()
    if not text:
        raise ValueError(f"the section of {version} in {Path(path).name} is empty")
    return text + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Print the version of Factorlab (the file VERSION).")
    ap.add_argument(
        "--changelog",
        nargs="?",
        const=VERSION,
        metavar="VERSION",
        help="print the CHANGELOG.md section of VERSION (default: the version in the file VERSION)",
    )
    a = ap.parse_args()
    if a.changelog is None:
        print(VERSION)
    else:
        try:
            print(changelog_section(a.changelog), end="")
        except (OSError, ValueError) as exc:
            raise SystemExit(f"version.py: {exc}") from None
