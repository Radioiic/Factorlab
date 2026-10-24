"""An append-only, hash-chained record: chain files of canonical JSON lines and head.json.

A ledger is one folder directly in the data folder (config.DATA). It holds:

    <chain>.jsonl            one or more chain files (in the folder or in its values/ folder)
    head.json                the line count and the sha256 of the last line of each chain.
                             It is not a chain: each write replaces it.
    results/<id>/<file>      result files (rewritable, not chained; write_file)

THE HASH CHAIN. Each line is canonical JSON (sorted keys, no spaces, ASCII, no NaN). Each line
holds "seq", its line number, and "prev_sha256", the sha256 of the line before it (64 zeros on
line 1). read() checks every chain link by link and against head.json (a JSON object; each chain
has a line count, a whole number of 0 or more, and a sha256). An edited line breaks the
chain at the next line. head.json holds the line count and the last hash of each chain, so the
check also finds an edited or removed last line. A chain can hold more lines than head.json
records: a write that stops between the chain file and head.json leaves such a line, and the
next write updates head.json. A broken record stops the command (RecordBroken, exit 1).

THE WRITE BOUNDARY. Each write calls config.guard_write first. The ledger folder must resolve
directly inside the data folder. The folder and each sub-folder must be real folders (no
symlink). A target must be a plain file with one link, directly in the folder, in a sub-folder
of a chain, or in results/<id> (an id in lower case, for example ft-0001). Each write goes to a
temporary file first, and a rename then replaces the target. append() reads the whole record
first, so it never extends a broken record. It refuses a line that holds "seq" or "prev_sha256",
a key that is not a string (a read would give the line back in another form) and a line with a
private string (webexport.common.privacy_hits). A refused append writes nothing.

THE CLOCK. CLOCK gives the current time (UTC). Only a test replaces it, in its own process.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

GENESIS = "0" * 64
HEAD = "head.json"
RESULTS = "results"
# a sub-folder of results/: a test id in lower case (ft-0001, ...)
RESULT_ID = re.compile(r"^[a-z]+-[0-9]{4,}$")
# the point-in-time comparison of the library screen (libfactors.PIT_RTOL, PIT_ATOL)
PIT_RTOL, PIT_ATOL = 1e-9, 1e-12


class RecordBroken(SystemExit):
    """A chain file or head.json does not pass a check. The command stops with exit 1.

    A subclass sets `prefix`, the name at the start of each message."""

    prefix = "ledger"

    def __init__(self, message: str):
        super().__init__(f"{self.prefix}: {message}")


# --------------------------------------------------------------------------- small helpers


def _lines(n: int) -> str:
    """'1 line', '2 lines': the number and the noun in the form that agrees with it."""
    return f"{n} {'line' if n == 1 else 'lines'}"


def canonical(obj) -> str:
    """The one serialisation of a record line: sorted keys, no spaces, ASCII, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _system_clock() -> datetime:
    return datetime.now(UTC)


CLOCK: Callable[[], datetime] = _system_clock


def _now() -> datetime:
    t = CLOCK()
    return t.astimezone(UTC) if t.tzinfo is not None else t.replace(tzinfo=UTC)


def now_utc() -> str:
    """CLOCK() as YYYY-MM-DDTHH:MM:SSZ."""
    return _now().strftime("%Y-%m-%dT%H:%M:%SZ")


def next_sunday_after(d: date) -> date:
    """The first Sunday after d (d itself never counts)."""
    return d + timedelta(days=(6 - d.weekday()) % 7 or 7)


def as_day(x) -> date:
    return x if isinstance(x, date) and not isinstance(x, datetime) else pd.Timestamp(x).date()


def ts(d) -> pd.Timestamp:
    return pd.Timestamp(str(as_day(d)), tz="UTC")


def finite_list(row) -> list:
    """Floats for JSON: NaN and +-inf become null."""
    return [float(v) if np.isfinite(v) else None for v in np.asarray(row, dtype=np.float64)]


def from_list(values) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def same_values(a, b) -> np.ndarray:
    """The point-in-time comparison: np.isclose, rtol 1e-9, atol 1e-12, NaN equal to NaN."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = np.where(np.isfinite(a), a, np.nan)
    b = np.where(np.isfinite(b), b, np.nan)
    return np.isclose(a, b, rtol=PIT_RTOL, atol=PIT_ATOL, equal_nan=True)


def num(x, digits: int = 6) -> float | None:
    """x with `digits` significant digits, or None when x is not a finite number."""
    if x is None:
        return None
    x = float(x)
    return float(f"{x:.{digits}g}") if np.isfinite(x) else None


def _check_keys(obj, where: str = "the record") -> None:
    """Refuse a key of a JSON object that is not a string, at each depth. JSON keys are strings:
    a read gives an integer key back as a string, in another sort order. So the line that read()
    gets back is then not the line that append() wrote, and the chain is broken."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"{where} has the key {k!r}: a record line must use string keys only")
            _check_keys(v, f"{where}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_keys(v, f"{where}[{i}]")


# --------------------------------------------------------------------------- the ledger


class Ledger:
    """One hash-chained record in the folder `root` (directly in config.DATA).

    chains maps a key to the path of its chain file, relative to root ("name.jsonl" or
    "values/name.jsonl"). about is the text of the "about" field of head.json. error is the
    RecordBroken class of the messages (a subclass sets their prefix)."""

    def __init__(
        self,
        root: Path,
        chains: Mapping[str, str],
        about: str,
        *,
        error: type[RecordBroken] = RecordBroken,
    ) -> None:
        self.root = Path(root)
        self.chains = dict(chains)
        self.about = about
        self.error = error
        subs = set()
        for key, rel in self.chains.items():
            p = Path(rel)
            if p.is_absolute() or ".." in p.parts or len(p.parts) > 2 or not p.name.endswith(".jsonl"):
                raise ValueError(f"chain {key!r}: {rel!r} is not a .jsonl file in the folder or in one sub-folder")
            if len(p.parts) == 2:
                subs.add(p.parts[0])
        if RESULTS in subs:
            raise ValueError(f"a chain cannot be in the folder {RESULTS}/")
        self.subdirs = tuple(sorted(subs))

    # ---- read

    def _name(self) -> str:
        return f"data/{self.root.name}"

    def _plain_file(self, path: Path) -> None:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
            raise self.error(f"{path.name} is not a plain file with one link")

    def read_chain(self, path: Path) -> tuple[list[dict], list[str]]:
        """The records of a chain file and the sha256 of each line. A broken link stops the read."""
        if not os.path.lexists(path):
            return [], []
        self._plain_file(path)
        try:
            text = path.read_bytes().decode("ascii")
        except UnicodeDecodeError:
            raise self.error(f"{path.name} holds a byte that is not ASCII") from None
        if text and not text.endswith("\n"):
            raise self.error(f"{path.name}: the last line has no line end (an incomplete write)")
        lines = text.split("\n")[:-1] if text else []
        recs, shas, prev = [], [], GENESIS
        for i, line in enumerate(lines, start=1):
            try:
                obj = json.loads(line)
            except ValueError:
                raise self.error(f"{path.name} line {i} is not JSON") from None
            if not isinstance(obj, dict) or canonical(obj) != line:
                raise self.error(f"{path.name} line {i} is not a canonical JSON object")
            if obj.get("seq") != i:
                raise self.error(f"{path.name} line {i} has seq {obj.get('seq')!r}: a line was removed or moved")
            if obj.get("prev_sha256") != prev:
                raise self.error(
                    f"{path.name} line {i} does not hold the hash of line {i - 1}: "
                    f"line {i - 1} was changed, or a line was removed"
                )
            prev = sha_text(line)
            recs.append(obj)
            shas.append(prev)
        return recs, shas

    def read_head(self) -> dict | None:
        p = self.root / HEAD
        if not os.path.lexists(p):
            return None
        self._plain_file(p)
        try:
            return json.loads(p.read_text(encoding="ascii"))
        except ValueError:
            raise self.error(f"{HEAD} is not JSON") from None

    def read(self) -> dict:
        """Every chain, checked link by link and against head.json. Read-only.

        Returns {key: {"records": [...], "shas": [...], "path": rel}}, "head" and "root". A chain
        can hold more lines than the head records (a run that stopped between two writes). It
        cannot hold fewer. Its line at the head position must have the hash that the head records."""
        r = self.root
        for d in (r, *(r / s for s in self.subdirs)):
            if d.is_symlink():
                raise self.error(f"{self._name()}/{d.relative_to(r).as_posix() or '.'} is a symlink")
        head = self.read_head()
        if head is not None and not isinstance(head, dict):
            raise self.error(f"{HEAD} is not a JSON object: {HEAD} was changed")
        out: dict = {"head": head, "root": r}
        for key, rel in self.chains.items():
            recs, shas = self.read_chain(r / rel)
            h = (head or {}).get(key)
            if h is None:
                if recs:
                    raise self.error(
                        f"{rel} has {_lines(len(recs))}, but {HEAD} records none: {HEAD} is absent or was changed"
                    )
            else:
                n = h.get("lines") if isinstance(h, dict) else None
                if type(n) is not int or n < 0:
                    raise self.error(
                        f"{HEAD} does not record a line count of {rel} (a whole number, 0 or more): {HEAD} was changed"
                    )
                if len(recs) < n:
                    raise self.error(f"{rel} has {_lines(len(recs))}, but {HEAD} records {n}: a line was removed")
                want = GENESIS if n == 0 else shas[n - 1]
                if not isinstance(h.get("sha256"), str):
                    raise self.error(f"{HEAD} does not record a sha256 of {rel}: {HEAD} was changed")
                if h["sha256"] != want:
                    raise self.error(
                        f"line {n} of {rel} does not have the hash that {HEAD} records: the line was changed"
                    )
            out[key] = {"records": recs, "shas": shas, "path": rel}
        return out

    # ---- the write boundary

    def _refuse(self, text: str) -> SystemExit:
        return SystemExit(f"{self.error.prefix}: write refused{text}")

    def _guard(self, path) -> Path:
        """Refuse any write that is not a plain file in the folder, in a sub-folder of a chain or
        in results/<id>. Returns the target as a Path."""
        C.guard_write(path)
        r = self.root
        if r.resolve().parent != Path(C.DATA).resolve():
            raise self._refuse(f": {self._name()} does not resolve inside the data folder")
        p = Path(path)
        folders = [r, *(r / s for s in self.subdirs)]
        allowed = list(folders)
        if p.parent.parent == r / RESULTS and RESULT_ID.fullmatch(p.parent.name):
            folders += [r / RESULTS, p.parent]
            allowed.append(p.parent)
        for d in folders:
            if d.is_symlink() or (d.exists() and not d.is_dir()):
                raise self._refuse(f": {self._name()}/{d.relative_to(r).as_posix() or '.'} is not a real folder")
        if p.parent not in allowed or p.name in ("", ".", ".."):
            raise self._refuse(f" outside {self._name()}: {p.name}")
        if os.path.lexists(p):
            st = os.lstat(p)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
                raise self._refuse(f" over {p.name}: not a plain file with one link")
        return p

    def _write_atomic(self, path: Path, data: bytes) -> None:
        p = self._guard(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._guard(p)  # check again: created by someone else in between?
        tmp = self._guard(p.with_name(f".{p.name}.tmp"))
        if os.path.lexists(tmp):
            os.unlink(tmp)  # a temporary file of a run that stopped
        with open(tmp, "xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)

    def _privacy(self, text: str, what: str) -> None:
        from webexport.common import privacy_hits

        hits = privacy_hits(text)
        if hits:
            raise self._refuse(f" for {what}: private strings ({', '.join(hits)})")

    def _write_head(self) -> None:
        head: dict[str, object] = {"about": self.about, "updated_at": now_utc()}
        for key, rel in self.chains.items():
            recs, shas = self.read_chain(self.root / rel)
            head[key] = {"lines": len(recs), "sha256": shas[-1] if shas else GENESIS}
        text = json.dumps(head, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
        self._write_atomic(self.root / HEAD, text.encode("ascii"))

    # ---- write

    def append(self, key: str, objs: list[dict]) -> list[str]:
        """Append records to one chain in one atomic write, then update head.json.

        The method reads and checks every chain first, so an append never extends a broken
        record. It returns the sha256 of each new line. An empty list writes nothing."""
        rel = self.chains[key]
        self.read()
        if not objs:
            return []
        path = self.root / rel
        recs, shas = self.read_chain(path)
        prev, seq = (shas[-1] if shas else GENESIS), len(recs)
        lines, out = [], []
        for obj in objs:
            if "seq" in obj or "prev_sha256" in obj:
                raise ValueError("seq and prev_sha256 belong to the chain, not to the caller")
            _check_keys(obj)
            seq += 1
            line = canonical({**obj, "seq": seq, "prev_sha256": prev})
            # read() accepts only a line that json.loads and canonical give back unchanged
            if canonical(json.loads(line)) != line:
                raise ValueError("a record line is not the same after a read (it is not canonical JSON)")
            prev = sha_text(line)
            lines.append(line)
            out.append(prev)
        text = "\n".join(lines) + "\n"
        self._privacy(text, rel)
        old = path.read_bytes() if os.path.lexists(path) else b""
        self._write_atomic(path, old + text.encode("ascii"))
        self._write_head()
        return out

    def write_file(self, rel: str, data: bytes) -> None:
        """Write a result file results/<id>/<name> (rel = "<id>/<name>"). A result file is derived
        and rewritable: it is not a chain, and head.json does not record it."""
        p = Path(rel)
        if len(p.parts) != 2 or not RESULT_ID.fullmatch(p.parts[0]):
            raise self._refuse(f": {rel!r} is not <id>/<file name> (an id in lower case)")
        if p.suffix in (".json", ".jsonl", ".txt", ".csv"):
            try:
                self._privacy(data.decode("utf-8"), rel)
            except UnicodeDecodeError:
                raise self._refuse(f" for {rel}: the text is not UTF-8") from None
        self._write_atomic(self.root / RESULTS / p, data)
