"""Export the engine data snapshot to the static web dashboard (web/data/*.js).

    cd engine
    python export_web.py                  # rebuild every section, then meta.js
    python export_web.py --only library,system
    python export_web.py --out <dir>      # write the data files to <dir> instead

Read-only over config.DATA (honours FACTORLAB_DATA); the "library" section
reads <repo>/library/catalog, also read-only, and the small files of
data/library_on_engine/ (never its git-ignored values/ shards); the "mine" section (the
first page, My factors) reads the trial ledger data/trials/ and its result files.
Writes ONLY into the output directory (default: <repo>/web/data) and refuses
a path inside the engine data directory. Sections and their order:
webexport.SECTIONS. Each section comes from webexport/<section>.py
(build() -> dict). A section without a module is skipped, and its file in
the output directory, if any, stays. A data file of a section that SECTIONS
does not have and its part files are removed from the output directory
(remove_other_sections()). When the data folder does not have
the files that a section needs, the section file holds {"available": false, "reason": ...} and the
page shows a short line. Output is deterministic except meta.generated_at and
meta.data_age (the age of the data on the export date).
Exit code 1 if a section fails, leaks a private string, or
the size budget is exceeded (section or part file > 1.5 MB, total > 6 MB). The part files of
the section "mine" (one for each registered test; the page of a test loads one) each have
their own limit of 200 KB and are not in the total (common.PART_MAX_BYTES).

meta.version is the version of Factorlab (the file VERSION). meta.demo is true when the data
folder holds the synthetic demo data (engine/demo_data.py): each page then shows the banner
"Synthetic demo data. Not market data."

meta.js describes each section (meta.sections[s]): present, bytes, as_of (the section's
own data date, when it has one), available (false for the no-data payload), state
("written", "no_data", "failed", "no_builder" or "kept" for a section that this run did
not export), written_at (the file time) and parts ({part: bytes}, the part files of
common.with_parts). A failed section keeps its old file; its state stays "failed" in the next
runs that do not export it, so the page can show that the file is older than the header.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

import version
from webexport import SECTIONS
from webexport import common as C

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT.parent / "web" / "data"


def _part_files(n: int) -> str:
    return f"{n} part file" if n == 1 else f"{n} part files"


def _rel(p: Path) -> str:
    """Repo-relative label for log lines (never an absolute path)."""
    try:
        return str(Path(p).resolve().relative_to(ROOT.parent))
    except ValueError:
        return Path(p).name


def data_through() -> str | None:
    """The newest daily perp bar (UTC open date) over the coins of config.UNIVERSE, or None
    when no coin has a bar (coin_last_bars())."""
    known = [d for d in coin_last_bars().values() if d]
    return max(known) if known else None


def coin_last_bars() -> dict[str, str | None]:
    """The last daily perp bar (UTC open date) of each coin in config.UNIVERSE.

    It reads data/raw/ohlcv/<coin>.parquet, in universe order. A coin with no
    file or no rows gets None. The data-through date is the newest bar of any
    coin, so a stall of one coin does not show there. This list shows it."""
    import config

    out: dict[str, str | None] = {}
    for sym in config.UNIVERSE:
        name = config.sym_to_file(sym)
        p = C.data_path(f"raw/ohlcv/{name}.parquet")
        last = None
        if p.exists():
            ts = pd.read_parquet(p, columns=["ts"])["ts"]
            if len(ts):
                last = C.to_date(ts.max())
        out[name] = last
    return out


def days_behind(export_date: str, day: str | None) -> int | None:
    """The number of daily closes (00:00 UTC) after the close of the bar ``day``, up to ``export_date``.

    The bar of day D closes at 00:00 UTC on D + 1. On that date the value is 0:
    the data is current. A value of 6 means that six closes are not in the data."""
    if day is None:
        return None
    return (pd.Timestamp(export_date) - pd.Timestamp(day)).days - 1


def data_age(export_date: str, through: str | None, last_bars: dict[str, str | None]) -> dict:
    """The data-age block of meta.js. It shows a stalled copy of the data and a stalled coin.

    ``through`` is the data-through date (the newest perp bar of any coin).
    ``last_bars`` maps each coin to its last perp bar (coin_last_bars())."""
    known = [d for d in last_bars.values() if d]
    newest = max(known) if known else None
    coins = [{"symbol": s, "last_bar": d, "days_behind": days_behind(export_date, d)} for s, d in last_bars.items()]
    behind = [c["symbol"] for c in coins if c["last_bar"] is None or c["last_bar"] < newest]
    return {
        "export_date": export_date,
        "data_through": through,
        "days_behind": days_behind(export_date, through),
        "symbols": len(coins),
        "newest_bar": newest,
        "oldest_bar": min(known) if known else None,
        "behind_newest": behind,
        "coins": coins,
    }


def next_day(date: str | None) -> str | None:
    if not date:
        return None
    return (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")


def load_builder(section: str):
    """Return webexport.<section>.build, or None if that module does not exist."""
    name = f"webexport.{section}"
    try:
        mod = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            return None
        raise
    build = getattr(mod, "build", None)
    if not callable(build):
        raise AttributeError(f"{name} has no build() function")
    return build


_WRAP_RE = re.compile(r'^window\.FL=window\.FL\|\|\{\};FL\.data=FL\.data\|\|\{\};FL\.data\["([^"]+)"\]=(.*);\n?$', re.S)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STATES = ("written", "no_data", "failed", "no_builder", "kept")


def read_data_file(f: Path) -> object | None:
    """The payload of a data file (the JSON inside the wrapper), or None when it does not parse."""
    try:
        m = _WRAP_RE.match(f.read_text(encoding="ascii"))
        return json.loads(m.group(2)) if m else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None


_KEY_RE = re.compile(r'^window\.FL=window\.FL\|\|\{\};FL\.data=FL\.data\|\|\{\};FL\.data\["([^"]+)"\]=')


def remove_other_sections(out: Path) -> list[str]:
    """Remove the data files in ``out`` whose section is not meta and not in SECTIONS.

    A file counts only when it has the data-file wrapper and its key is its file name
    without ".js" ("<section>.js", or "<section>.<part>.js" for a part file). Any other
    file stays. Returns the names of the removed files, in name order."""
    keep = {"meta", *SECTIONS}
    removed = []
    for f in sorted(out.glob("*.js")):
        if not f.is_file() or f.is_symlink():
            continue
        try:
            with f.open("rb") as fh:
                head = fh.read(256).decode("ascii")
        except (OSError, UnicodeDecodeError):
            continue
        m = _KEY_RE.match(head)
        if not m or m.group(1) + ".js" != f.name or m.group(1).split(".", 1)[0] in keep:
            continue
        f.unlink()
        removed.append(f.name)
    return removed


def old_states(out: Path) -> dict[str, str]:
    """The section states of the meta.js already in ``out`` (empty when there is none)."""
    old = read_data_file(out / "meta.js")
    secs = old.get("sections") if isinstance(old, dict) else None
    if not isinstance(secs, dict):
        return {}
    return {k: v.get("state") for k, v in secs.items() if isinstance(v, dict) and v.get("state") in STATES}


def section_info(out: Path, s: str, state: str | None) -> dict:
    """meta.sections[s] (see the module docstring)."""
    f = out / f"{s}.js"
    parts = {name: p.stat().st_size for name, p in C.part_files(out, s).items()}
    if not f.exists():
        return {
            "present": False,
            "bytes": None,
            "as_of": None,
            "available": None,
            "state": state or "kept",
            "written_at": None,
            "parts": parts,
        }
    payload = read_data_file(f)
    as_of = payload.get("as_of") if isinstance(payload, dict) else None
    available = payload.get("available") if isinstance(payload, dict) else None
    mtime = _dt.datetime.fromtimestamp(f.stat().st_mtime, tz=_dt.UTC)
    return {
        "present": True,
        "bytes": f.stat().st_size,
        "as_of": as_of if isinstance(as_of, str) and _DATE_RE.match(as_of) else None,
        "available": available if isinstance(available, bool) else (None if payload is not None else False),
        "state": state or "kept",
        "written_at": C.iso_utc(mtime),
        "parts": parts,
    }


def build_meta(out: Path, states: dict[str, str] | None = None) -> dict:
    """The meta.js payload. ``states``: the section states of this run (main()); a section
    that this run did not export keeps a "failed" state of the earlier meta.js."""
    states = dict(states or {})
    carried = old_states(out)
    sections = {}
    total = 0
    for s in SECTIONS:
        st = states.get(s) or ("failed" if carried.get(s) == "failed" else "kept")
        info = section_info(out, s, st)
        if info["present"]:
            total += info["bytes"]
        total += sum(info["parts"].values())
        sections[s] = info
    last_bars = coin_last_bars()
    known = [d for d in last_bars.values() if d]
    dt = max(known) if known else None  # data_through(): the newest perp bar of any coin
    generated = C.iso_utc()
    return {
        "schema": 1,
        "version": version.VERSION,
        "demo": C.demo(),
        "generated_at": generated,
        "data_through": dt,
        "data_close_utc": next_day(dt),
        "timezone": "UTC",
        "bar_stamp": "open",
        "sections": sections,
        "total_bytes": total,
        "budget": {"section_max_bytes": C.SECTION_MAX_BYTES, "total_max_bytes": C.TOTAL_MAX_BYTES},
        "data_age": data_age(generated[:10], dt, last_bars),
    }


def write_meta(out: Path, states: dict[str, str] | None = None) -> int:
    """(Re)write meta.js from the section files that exist in ``out``."""
    return C.write_section(out, "meta", build_meta(out, states))


def parse_only(value: str | None) -> list[str]:
    if not value:
        return list(SECTIONS)
    wanted = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown section(s): {', '.join(unknown)} (choose from {', '.join(SECTIONS)})")
    return [s for s in SECTIONS if s in wanted]


def size_report(out: Path, status: dict[str, str]) -> tuple[dict[str, int | None], list[str]]:
    sizes: dict[str, int | None] = {}
    print(f"\n{'file':<16}{'bytes':>12}{'KB':>10}  status")
    print("-" * 52)
    for s in ("meta",) + tuple(SECTIONS):
        f = out / f"{s}.js"
        n = f.stat().st_size if f.exists() else None
        sizes[s] = n
        st = status.get(s, "kept" if n is not None else "missing")
        print(
            f"{s + '.js':<16}{(f'{n:,}' if n is not None else '-'):>12}{(f'{n / 1024:.1f}' if n is not None else '-'):>10}  {st}"
        )
        parts = C.part_files(out, s)
        if parts:
            for name, p in parts.items():
                sizes[f"{s}.{name}"] = p.stat().st_size
            pn = sum(p.stat().st_size for p in parts.values())
            label = f"{s}.*.js"
            print(f"{label:<16}{pn:>12,}{pn / 1024:>10.1f}  {_part_files(len(parts))}")
    # the parts that load one at a time (common.PART_MAX_BYTES) are not in the budget total
    total = sum(
        n for name, n in sizes.items() if n is not None and not ("." in name and name.split(".")[0] in C.PART_MAX_BYTES)
    )
    print("-" * 52)
    print(
        f"{'total':<16}{total:>12,}{total / 1024:>10.1f}  budget {C.SECTION_MAX_BYTES:,} per section, {C.TOTAL_MAX_BYTES:,} total"
    )
    return sizes, C.check_sizes(sizes)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default: <repo>/web/data)")
    ap.add_argument("--only", default=None, help="comma list of sections to rebuild (default: all)")
    args = ap.parse_args(argv)

    out = C.check_out_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = parse_only(args.only)
    status: dict[str, str] = {}
    states: dict[str, str] = {}
    failed = False

    for s in todo:
        t0 = time.perf_counter()
        try:
            build = load_builder(s)
            if build is None:
                status[s] = f"no builder (webexport/{s}.py)"
                states[s] = "no_builder"
                continue
            try:
                payload = build()
            except (C.NoData, FileNotFoundError, SystemExit) as exc:
                if isinstance(exc, SystemExit) and not C.is_no_data(exc):
                    # a SystemExit is not an Exception: report it like any other failure
                    raise RuntimeError(f"SystemExit: {exc.code}") from exc
                if not C.is_no_data(exc):
                    raise
                # The data folder does not have the input files of this section:
                # the page shows a short line in place of the section.
                payload = C.no_data_payload(exc)
                status[s] = "no data: " + payload["reason"]
                C.write_section(out, s, payload)
                states[s] = "no_data"
                continue
            if not isinstance(payload, dict):
                raise TypeError(f"{s}.build() must return a dict, got {type(payload).__name__}")
            n, parts = C.write_section_parts(out, s, payload)
            status[s] = f"written in {time.perf_counter() - t0:.1f}s" + (
                f" ({_part_files(len(parts))})" if parts else ""
            )
            states[s] = "written"
        except Exception as exc:  # report every section, then fail at the end
            failed = True
            states[s] = "failed"
            status[s] = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"[export_web] section {s!r} failed:", file=sys.stderr)
            traceback.print_exc()

    for name in remove_other_sections(out):
        print(f"[export_web] removed {name}: the dashboard has no section {name.split('.', 1)[0]!r}")
    write_meta(out, states)
    status["meta"] = "written"
    print(f"[export_web] read {_rel(C.DATA)}/ -> wrote {_rel(out)}/ (data through {data_through()} UTC close)")
    _, problems = size_report(out, status)
    for p in problems:
        print(f"[export_web] SIZE BUDGET EXCEEDED: {p}", file=sys.stderr)
    if problems or failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
