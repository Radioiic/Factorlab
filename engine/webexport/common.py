"""Shared helpers for the web exporter (read-only over the engine data).

Every section builder (``webexport/<section>.py``) uses these so the JS data
files share one shape and one rounding policy:

    from webexport.common import DATA, read_parquet, to_dates, rnd, series_cols, records

    def build() -> dict:
        d = read_parquet("labels/labels.parquet")
        return {"labels": series_cols(d, {"fwd_ret_7": "ret"})}

Conventions:
- Dates are "YYYY-MM-DD" strings (UTC). A daily bar is stamped with its OPEN time.
- Floats are rounded by kind: prices/equity/usd 2 dp; returns/weights/IC 6 dp;
  ranks/percentiles 4 dp. NaN/inf become None (JSON null).
- Prefer columnar layouts: {"dates": [...], "series": {"book": [...], ...}}.
- Nothing here writes into ``DATA``. ``write_section`` writes only into the
  output directory it is given (``web/data`` by default) and refuses ``DATA``.
- Part files: a builder can split data that a page needs only on request (one day of a
  history, a search index) into part files. It returns ``with_parts(payload, parts)``;
  ``write_section`` then writes ``<section>.<part>.js`` for each part (wrapper key
  ``FL.data["<section>.<part>"]``), puts the sorted part names in ``payload["parts"]`` and
  removes the part files of an earlier export that the new one does not have. The page
  loads a part with ``ctx.loadPart(part)`` or ``FL.load.part(section, part)``.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import math
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config

DATA: Path = config.DATA
ENGINE_ROOT: Path = config.ROOT

# Size budget for web/data (bytes, decimal units).
SECTION_MAX_BYTES = 1_500_000
TOTAL_MAX_BYTES = 6_000_000
# The part files of these sections load one at a time (the page of one test loads its own part):
# each part has its own limit, and the parts are not in the total.
PART_MAX_BYTES = {"mine": 200_000}
MINE_PART_MAX_BYTES = PART_MAX_BYTES["mine"]

# Rounding policy by kind (decimal places).
DP: dict[str, int] = {
    "usd": 2,
    "price": 2,
    "equity": 2,
    "ret": 6,
    "pct": 6,
    "weight": 6,
    "ic": 6,
    "rate": 6,
    "decimal": 6,
    "rank": 4,
    "percentile": 4,
    "t": 4,
    "score": 4,
    "x": 4,
    "idx": 4,
    "num": 4,
}

# Strings that must never reach web/data. The patterns are generic: absolute home
# paths, share links (by URL shape), hidden config folders in a home folder, and
# e-mail addresses.
_GENERIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("home path", re.compile(r"/(?:Users|home)/[^/\s\"']+")),
    (
        "share link",
        re.compile(
            r"(?i)\b(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s/\"'<>()]+)*?"
            r"/(?:share|shares|shared|chat|c|g|conversations?|artifacts?)/[A-Za-z0-9_-]{8,}"
        ),
    ),
    ("home config folder", re.compile(r"(?:~|\$HOME|\$\{HOME\})/\.[A-Za-z0-9_]")),
    ("email address", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")),
)


def term_pattern(term: str) -> re.Pattern[str]:
    """Short ALL-CAPS terms match case-sensitively on word boundaries (so an
    abbreviation does not match ordinary words). All other terms match as a
    case-insensitive substring."""
    if term.isascii() and term.isalnum() and term.isupper() and len(term) <= 5:
        return re.compile(r"\b" + re.escape(term) + r"\b")
    return re.compile(re.escape(term), re.IGNORECASE)


def term_patterns(terms: Iterable[str]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Named patterns for a list of terms. The name ("term #N") does not show the term."""
    return tuple((f"term #{i + 1:d}", term_pattern(t)) for i, t in enumerate(terms))


PRIVATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = _GENERIC_PATTERNS

_SECTION_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# A part name: lower-case letters, digits, "_" and "-" (a date such as 2026-10-01 is valid).
PART_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
# The payload key that carries the parts from a builder to write_section (never written).
PARTS_KEY = "__parts__"


class PrivacyError(ValueError):
    """A payload contained a personal string (path, account, location)."""


class SizeError(ValueError):
    """A payload exceeded the size budget."""


class NoData(LookupError):
    """The data folder does not have the files that a section needs."""


class DataMismatch(RuntimeError):
    """The input files of a section do not agree with each other (for example, a manifest
    count and the table that it counts). The export of that section stops."""


def demo() -> bool:
    """True when DATA holds the synthetic demo data (the marker file of demo_data.py).
    meta.js carries it as meta.demo, and every page then shows the demo banner."""
    import demo_data

    return demo_data.is_demo(DATA)


# the messages of the data_io loaders that find no data (data_io.close_matrix, data_io.label_table)
LOADER_NO_DATA = ("No raw OHLCV data", "No label table")


def is_no_data(exc: BaseException) -> bool:
    """True when ``exc`` reports absent input data, not a defect.

    A NoData is absent data. A FileNotFoundError is absent data only when the missing
    file is in the data folder. A SystemExit is absent data only when a data_io loader
    raised it with one of the LOADER_NO_DATA messages. A missing repository file (a
    document, for example) is a defect, and the export fails."""
    if isinstance(exc, NoData):
        return True
    if isinstance(exc, FileNotFoundError):
        if not exc.filename:
            return False
        try:
            Path(str(exc.filename)).resolve().relative_to(DATA.resolve())
        except (ValueError, OSError):
            return False
        return True
    if isinstance(exc, SystemExit):
        return isinstance(exc.code, str) and exc.code.startswith(LOADER_NO_DATA)
    return False


def no_data_payload(exc: BaseException) -> dict:
    """The section payload for absent input data: {"schema", "available": False, "reason"}.

    ``exc`` is a NoData, a FileNotFoundError or the SystemExit of an engine loader that
    finds no data. The reason names the missing file relative to the data folder; a
    reason with a private string (a path, for example) becomes a generic text."""
    if isinstance(exc, FileNotFoundError) and exc.filename:
        p = Path(str(exc.filename))
        try:
            name = p.resolve().relative_to(DATA.resolve()).as_posix()
        except (ValueError, OSError):
            name = p.name
        reason = f"{name} is not in the data folder"
    elif isinstance(exc, FileNotFoundError):
        reason = "an input file is not in the data folder"
    elif isinstance(exc, SystemExit):
        reason = exc.code if isinstance(exc.code, str) else ""
    else:
        reason = str(exc)
    reason = reason.strip().rstrip(".")
    if not reason or privacy_hits(reason):
        reason = "the input data is not in the data folder"
    return {"schema": 1, "available": False, "reason": reason}


# --------------------------------------------------------------------------- reading


def data_path(rel: str | os.PathLike) -> Path:
    """Resolve a path relative to ``config.DATA`` (absolute paths pass through)."""
    p = Path(rel)
    return p if p.is_absolute() else DATA / p


def read_parquet(rel: str | os.PathLike, columns: list[str] | None = None, sort: str | None = "date") -> pd.DataFrame:
    """Read a parquet file under ``DATA`` (read-only). Sorted by ``sort`` if present."""
    df = pd.read_parquet(data_path(rel), columns=columns)
    if sort and sort in df.columns:
        df = df.sort_values(sort, kind="stable").reset_index(drop=True)
    return df


def read_json(rel: str | os.PathLike) -> Any:
    with open(data_path(rel), encoding="utf-8") as fh:
        return json.load(fh)


def exists(rel: str | os.PathLike) -> bool:
    return data_path(rel).exists()


# --------------------------------------------------------------------------- values


def to_date(x: Any) -> str | None:
    """One timestamp/date/str -> 'YYYY-MM-DD' (UTC calendar date) or None."""
    if x is None:
        return None
    if isinstance(x, str):
        if not x:
            return None
        x = pd.Timestamp(x)
    if isinstance(x, (np.datetime64,)):
        x = pd.Timestamp(x)
    if isinstance(x, pd.Timestamp):
        if pd.isna(x):
            return None
        if x.tzinfo is not None:
            x = x.tz_convert("UTC")
        return x.strftime("%Y-%m-%d")
    if isinstance(x, _dt.datetime):
        if x.tzinfo is not None:
            x = x.astimezone(_dt.UTC)
        return x.strftime("%Y-%m-%d")
    if isinstance(x, _dt.date):
        return x.isoformat()
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def to_dates(values: Iterable[Any] | pd.Series | pd.Index) -> list[str | None]:
    """Series/Index/list of timestamps -> list of 'YYYY-MM-DD' strings (UTC); NaT -> None."""
    if isinstance(values, (pd.Series, pd.Index)):
        s = pd.to_datetime(pd.Series(values), utc=True)
        return [None if pd.isna(v) else v.strftime("%Y-%m-%d") for v in s]
    return [to_date(v) for v in values]


def iso_utc(ts: _dt.datetime | pd.Timestamp | None = None) -> str:
    """UTC ISO timestamp 'YYYY-MM-DDTHH:MM:SSZ' (now if ts is None)."""
    if ts is None:
        ts = _dt.datetime.now(_dt.UTC)
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _dp(kind: int | str | None) -> int | None:
    if kind is None:
        return None
    if isinstance(kind, (int, np.integer)):
        return int(kind)
    if kind not in DP:
        raise KeyError(f"unknown rounding kind {kind!r}; use an int or one of {sorted(DP)}")
    return DP[kind]


def _rnd1(x: Any, dp: int | None) -> float | int | None:
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    if dp is None:
        return f
    r = round(f, dp)
    return 0.0 if r == 0 else r  # no "-0.0"


def rnd(x: Any, kind: int | str | None = 6):
    """Round a scalar or a sequence by kind ('usd', 'ret', 'rank', ... or an int dp).

    Scalars -> float | None; Series/ndarray/list -> list. NaN/inf/None -> None.
    """
    dp = _dp(kind)
    if isinstance(x, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        return [_rnd1(v, dp) for v in (x.tolist() if hasattr(x, "tolist") else x)]
    return _rnd1(x, dp)


def series_cols(
    df: pd.DataFrame,
    cols: Mapping[str, int | str] | Iterable[str],
    date_col: str | None = "date",
    kind: int | str = 6,
    rename: Mapping[str, str] | None = None,
) -> dict:
    """Columnar block: {"dates": [...], "series": {name: [rounded values...]}}.

    cols: {column: kind} or an iterable of columns (all rounded by ``kind``).
    ``date_col`` may be a column name or None to use the index.
    """
    if not isinstance(cols, Mapping):
        cols = {c: kind for c in cols}
    dates = to_dates(df.index if date_col is None or date_col not in df.columns else df[date_col])
    rename = rename or {}
    return {
        "dates": dates,
        "series": {rename.get(c, c): rnd(df[c], k) for c, k in cols.items()},
    }


def records(
    df: pd.DataFrame,
    kinds: Mapping[str, int | str] | None = None,
    dates: Iterable[str] = ("date",),
    columns: Iterable[str] | None = None,
) -> list[dict]:
    """DataFrame -> list of dicts with date columns as 'YYYY-MM-DD' and floats rounded.

    kinds: {column: kind}; float columns not listed use 6 dp; ints stay ints.
    """
    kinds = dict(kinds or {})
    cols = list(columns) if columns is not None else list(df.columns)
    date_cols = set(dates)
    out_cols: dict[str, list] = {}
    for c in cols:
        s = df[c]
        if c in date_cols or pd.api.types.is_datetime64_any_dtype(s):
            out_cols[c] = to_dates(s)
        elif c in kinds:
            out_cols[c] = rnd(s, kinds[c])
        elif pd.api.types.is_bool_dtype(s):
            out_cols[c] = [None if pd.isna(v) else bool(v) for v in s]
        elif pd.api.types.is_integer_dtype(s):
            out_cols[c] = [None if pd.isna(v) else int(v) for v in s]
        elif pd.api.types.is_float_dtype(s):
            out_cols[c] = rnd(s, 6)
        else:
            out_cols[c] = [None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else v for v in s]
    n = len(df)
    return [{c: out_cols[c][i] for c in cols} for i in range(n)]


# --------------------------------------------------------------------------- JSON


def clean_json(obj: Any) -> Any:
    """Recursively make ``obj`` JSON-safe: NaN/inf -> None, numpy/pandas scalars -> Python,
    Timestamps -> 'YYYY-MM-DD' (midnight) or UTC ISO, arrays/tuples/sets -> lists.
    Paths are rejected (they leak machine layout): convert them explicitly.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int,)) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Mapping):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [clean_json(v) for v in sorted(obj, key=str)]
    if isinstance(obj, np.ndarray):
        return [clean_json(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [clean_json(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        raise TypeError("clean_json: convert DataFrames with series_cols() or records() first")
    if isinstance(obj, (pd.Timestamp, _dt.datetime, np.datetime64)):
        ts = pd.Timestamp(obj)
        if pd.isna(ts):
            return None
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0:
            return ts.strftime("%Y-%m-%d")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(obj, _dt.date):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    if isinstance(obj, os.PathLike):
        raise TypeError("clean_json: refusing a filesystem path; export a name or a relative label instead")
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"clean_json: unsupported type {type(obj).__name__}")


def _dumps_clean(cleaned: Any) -> str:
    return json.dumps(cleaned, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def dumps(obj: Any) -> str:
    """Compact, ASCII-only, NaN-free JSON (works on file:// regardless of charset)."""
    return _dumps_clean(clean_json(obj))


def wrap(section: str, payload_json: str) -> str:
    """The exact data-file wrapper the page expects."""
    return f'window.FL=window.FL||{{}};FL.data=FL.data||{{}};FL.data["{section}"]={payload_json};\n'


# A JSON \uXXXX escape, surrogate pairs first.
_U_ESC = re.compile(r"\\u([dD][89abAB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})|\\u([0-9a-fA-F]{4})")


def _unescape_u(text: str) -> str:
    def rep(m: re.Match[str]) -> str:
        if m.group(3):
            return chr(int(m.group(3), 16))
        hi, lo = int(m.group(1), 16), int(m.group(2), 16)
        return chr(0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00))

    return _U_ESC.sub(rep, text)


def _views(s: str) -> tuple[str, ...]:
    """The forms of one string a privacy pattern must see.

    ``dumps`` writes ASCII only, so a non-ASCII character reaches the file as a
    ``\\uXXXX`` escape that the browser turns back into the character. Matching
    only the escaped text would let a non-ASCII deny term (a place name in its
    own script, say) through the guard while the page still shows it. So every
    check also looks at the escape-decoded form and at its NFKC normalisation
    (full-width or composed look-alikes of a term or an '@').

    Invisible format characters (Unicode category Cf: zero-width space U+200B,
    soft hyphen U+00AD, joiners, bidi marks) are removed in one more view
    before NFKC, which keeps them: a term split by one would otherwise pass the
    guard and still render as the plain term in the browser.
    """
    out = [s]
    if "\\u" in s:
        out.append(_unescape_u(s))
    for v in list(out):
        if not v.isascii():
            bare = "".join(ch for ch in v if unicodedata.category(ch) != "Cf")
            if bare not in out:
                out.append(bare)
    for v in list(out):
        n = unicodedata.normalize("NFKC", v)
        if n not in out:
            out.append(n)
    return tuple(out)


def privacy_hits(text: str) -> list[str]:
    """Names of PRIVATE_PATTERNS found in ``text`` (empty list = clean).

    ``text`` may be a raw string or an ASCII-escaped JSON serialisation: the
    decoded and normalised views are checked too (see ``_views``)."""
    views = _views(text)
    return [name for name, rx in PRIVATE_PATTERNS if any(rx.search(v) for v in views)]


def _strings(obj: Any):
    """Every string in a ``clean_json`` result: dict keys and values at any depth."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def payload_privacy_hits(obj: Any) -> list[str]:
    """Names of PRIVATE_PATTERNS found in any ORIGINAL string of a payload.

    Checked one decoded string at a time (keys and nested values), before any
    JSON or ASCII escaping, so what is tested is exactly what the page
    displays. ``obj`` is cleaned with ``clean_json`` first, so dates and numpy
    keys are checked in the string form they are written in."""
    strings = [v for s in set(_strings(clean_json(obj))) for v in _views(s)]
    return [name for name, rx in PRIVATE_PATTERNS if any(rx.search(s) for s in strings)]


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def with_parts(payload: Mapping[str, Any], parts: Mapping[str, Any]) -> dict:
    """Return ``payload`` with part files attached (see the module docstring).

    ``parts`` maps a part name (PART_RE) to a JSON-able payload. Each part file has the same
    size budget and privacy check as a section file. The written section payload gets
    ``"parts": [sorted names]``, so ``payload`` must not have a "parts" key."""
    if not isinstance(payload, Mapping):
        raise TypeError("with_parts: the payload must be a dict")
    if "parts" in payload:
        raise ValueError("with_parts: the payload already has a 'parts' key")
    bad = [str(k) for k in parts if not isinstance(k, str) or not PART_RE.match(k)]
    if bad:
        raise ValueError(f"with_parts: bad part name(s): {', '.join(bad[:5])}")
    out = dict(payload)
    out[PARTS_KEY] = dict(parts)
    return out


def part_file_name(section: str, part: str) -> str:
    """The file name of a part: '<section>.<part>.js'."""
    return f"{section}.{part}.js"


def part_files(out_dir: str | os.PathLike, section: str) -> dict[str, Path]:
    """The part files of ``section`` that exist in ``out_dir``: {part name: path}."""
    out = Path(out_dir)
    found: dict[str, Path] = {}
    if not out.is_dir():
        return found
    prefix = section + "."
    for f in out.glob(f"{section}.*.js"):
        name = f.name[len(prefix) : -3]
        if PART_RE.match(name):
            found[name] = f
    return dict(sorted(found.items()))


def check_out_dir(out_dir: str | os.PathLike) -> Path:
    """Refuse output locations inside the engine data snapshot."""
    out = Path(out_dir).expanduser().resolve()
    if _is_inside(out, DATA) or _is_inside(out, ENGINE_ROOT / "data"):
        raise ValueError(f"refusing to write web data inside the engine data directory: {out}")
    return out


def render_section(section: str, obj: Any, part: str | None = None) -> str:
    """Clean + validate + serialize + wrap; raises PrivacyError on personal strings.

    With ``part``, the wrapper key is "<section>.<part>" (a part file).

    The privacy check runs on the ORIGINAL strings (every key and nested value
    of the cleaned payload) before the ASCII-escaping serialisation, and then
    once more on the final text (escape-decoded too) as a second line of
    defence. A check of the escaped output alone misses a non-ASCII term:
    ``ensure_ascii`` turns it into ``\\uXXXX`` escapes, and the browser
    restores it on load.
    """
    if not _SECTION_RE.match(section):
        raise ValueError(f"bad section name {section!r}")
    if part is not None and not PART_RE.match(str(part)):
        raise ValueError(f"bad part name {part!r}")
    key = section if part is None else f"{section}.{part}"
    cleaned = clean_json(obj)
    hits = payload_privacy_hits(cleaned)
    text = wrap(key, _dumps_clean(cleaned))
    hits += [h for h in privacy_hits(text) if h not in hits]
    if hits:
        # pattern names only ("term #3"), never the matched text
        raise PrivacyError(
            f"{'part' if part is not None else 'section'} {key!r} contains private strings: {', '.join(hits)}"
        )
    return text


def _atomic_write(out: Path, name: str, raw: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=out)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; data files are meant to be served
        os.replace(tmp, out / name)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_section_parts(
    out_dir: str | os.PathLike, section: str, obj: Any, max_bytes: int = SECTION_MAX_BYTES
) -> tuple[int, dict[str, int]]:
    """Write ``<section>.js`` and its part files; returns (section bytes, {part: bytes}).

    Every file is rendered and checked first (privacy, size), so a failure writes nothing.
    Then the parts are written, then the section file (so it never names a part that is not
    on disk), then the part files of an earlier export that this one does not have are removed.
    """
    out = check_out_dir(out_dir)
    payload = obj
    parts: Mapping[str, Any] = {}
    if isinstance(obj, Mapping) and PARTS_KEY in obj:
        payload = dict(obj)
        parts = payload.pop(PARTS_KEY) or {}
        if "parts" in payload:
            raise ValueError(f"section {section!r}: the payload already has a 'parts' key")
        bad = [str(k) for k in parts if not isinstance(k, str) or not PART_RE.match(k)]
        if bad:
            raise ValueError(f"section {section!r}: bad part name(s): {', '.join(bad[:5])}")
        payload["parts"] = sorted(parts)
    raw = render_section(section, payload).encode("ascii")
    if len(raw) > max_bytes:
        raise SizeError(f"section {section!r} is {len(raw):,} bytes > budget {max_bytes:,}")
    part_raw: dict[str, bytes] = {}
    part_max = min(max_bytes, PART_MAX_BYTES.get(section, max_bytes))
    for name in sorted(parts):
        b = render_section(section, parts[name], part=name).encode("ascii")
        if len(b) > part_max:
            raise SizeError(f"part {section}.{name} is {len(b):,} bytes > budget {part_max:,}")
        part_raw[name] = b
    out.mkdir(parents=True, exist_ok=True)
    for name, b in part_raw.items():
        _atomic_write(out, part_file_name(section, name), b)
    _atomic_write(out, f"{section}.js", raw)
    for name, f in part_files(out, section).items():
        if name not in part_raw:
            f.unlink()
    return len(raw), {name: len(b) for name, b in part_raw.items()}


def write_section(out_dir: str | os.PathLike, section: str, obj: Any, max_bytes: int = SECTION_MAX_BYTES) -> int:
    """Write ``<out_dir>/<section>.js`` atomically; returns its size in bytes.

    A payload from ``with_parts`` also writes its part files (``write_section_parts``).
    Raises SizeError (nothing written) if the payload or a part exceeds ``max_bytes``,
    PrivacyError if it contains personal strings, ValueError for an out_dir
    inside the engine data directory.
    """
    return write_section_parts(out_dir, section, obj, max_bytes)[0]


# --------------------------------------------------------------------------- shared checks


def check_sizes(
    sizes: Mapping[str, int | None], section_max: int = SECTION_MAX_BYTES, total_max: int = TOTAL_MAX_BYTES
) -> list[str]:
    """Budget problems for {name: bytes}; empty list = within budget.

    A name is a section ("library") or a part ("library.index"). A part of a section in
    PART_MAX_BYTES ("mine.ft-0001") has that limit and is not in the total: the page loads one
    such part at a time."""
    problems = []
    total = 0
    for name, n in sizes.items():
        if n is None:
            continue
        own = PART_MAX_BYTES.get(name.split(".", 1)[0]) if "." in name else None
        if own is not None:
            if n > own:
                problems.append(f"{name}.js is {n:,} bytes > {own:,}")
            continue
        total += n
        if name != "meta" and n > section_max:
            problems.append(f"{name}.js is {n:,} bytes > {section_max:,}")
    if total > total_max:
        problems.append(f"web/data total is {total:,} bytes > {total_max:,}")
    return problems
