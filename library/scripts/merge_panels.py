"""Merge per-symbol panel downloads into one panel set (offline).

    python scripts/merge_panels.py --symbols BTCUSDT,ETHUSDT,...      # the defaults below
    python scripts/merge_panels.py --parts data/binance_um_1d_parts --out data/binance_um_1d

`python -m factorlab download` writes one folder of panels per call. Downloading one
symbol per call (so a failed symbol can be retried alone) leaves
``<parts>/<SYMBOL>/<field>.csv.gz`` with a single column each. This script joins them
into ``<out>/<field>.csv.gz`` with one column per symbol, in the order given by
``--symbols``, on the union of the symbols' ``close`` index (the kline bars), and
prints each field's coverage. It reads and writes local files only: no network, and
it refuses the ``catalog/`` folder as an output.

These commands make a Binance panel for the engine's 20 coins. The panel uses Binance
USD-M public data as DATA ONLY. The 20 symbols are the engine's 20 OKX perpetuals, in
the engine's order. Replace <LAST> with the last closed day (UTC):

    cd library
    for s in BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT DOTUSDT \\
             LTCUSDT BCHUSDT UNIUSDT ATOMUSDT ETCUSDT FILUSDT NEARUSDT APTUSDT OPUSDT \\
             ARBUSDT AAVEUSDT; do
      FACTORLAB_OFFLINE=0 python -B -m factorlab download --symbols "$s" --start 2022-01-01 \\
          --end "<LAST> 23:59:59" --interval 1d --market um --metrics-archive \\
          --out "data/binance_um_1d_parts/$s"
    done
    python -B scripts/merge_panels.py          # offline: parts -> data/binance_um_1d/

The loop takes about 9 minutes per coin. You can run the calls in parallel. Each call
writes its own part folder. The merge puts the columns in the ``--symbols`` order, so the
result does not depend on the order of the calls.

Both folders are under the git-ignored ``data/``. The merged files are gzip-compressed
CSV written by ``factorlab.data.save_panels`` with the gzip header's modification time
fixed at 0 (``GZIP_MTIME``), so a new merge of the same parts gives byte-identical
files. The engine's libfactors.py manifest hashes the decompressed content, which does
not depend on the header.

Incremental mode (``--append``). It adds new closed bars to the merged panel and
changes no cell that is already there. Download the new days one coin per call into a
new folder below the parts folder. Replace <FIRST> with the day after the last merged
bar and <LAST> with the last closed day. Then append them:

    cd library
    for s in BTCUSDT ETHUSDT ...; do                    # the 20 symbols above, in that order
      FACTORLAB_OFFLINE=0 python -B -m factorlab download --symbols "$s" --start <FIRST> \
          --end "<LAST> 23:59:59" --interval 1d --market um --metrics-archive \
          --out "data/binance_um_1d_parts/updates/<FIRST>_<LAST>/$s"
    done
    python -B scripts/merge_panels.py --append --parts data/binance_um_1d_parts/updates/<FIRST>_<LAST>

The append obeys these rules, and it stops (exit 2) and writes nothing when one fails:

* The new bars must follow the last merged bar day by day, with no gap.
* Each new bar must be closed: its open plus one day is not later than the clock (UTC).
* A coin with a kline bar on the last merged bar must have a kline bar (close) on each
  new day.
* A coin with a kline bar on a new day must have a value in each field that it had on
  the last merged bar. A missing value means that the source was not complete at download
  time. An example is a metrics archive file that Binance had not published at download
  time: the downloader skips a missing archive file (HTTP 404) without an error. Download
  that day again after one day. As an alternative, use ``--through`` to stop before it. The panel is
  append-only, so a missing cell that the append accepts stays missing.
  If Binance never publishes the file, the append stops on every run until you use
  ``--through``.
* A downloaded bar that the panel already has must agree with it cell by cell, in the
  CSV text. The merged cells always stay as they are.

The append writes each field file to a temporary folder first. Each new file must start
with the old file's decompressed content, byte for byte, so every old cell stays the
same. Then the append replaces the merged files. The gzip header time stays fixed at 0,
so the result is the file that a full merge of the same rows gives. Each append adds one
line to ``appends.jsonl`` in the merged folder: the new bars, the clock and the parts
folder. A full merge removes that file, because it rebuilds the panel from the parts. To
rebuild the panel from the parts, run the full merge first. Then run each append in date
order.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

LIBRARY = Path(__file__).resolve().parents[1]
if str(LIBRARY) not in sys.path:
    sys.path.insert(0, str(LIBRARY))

from factorlab.data import load_panels, save_panels  # noqa: E402
from factorlab.paths import CatalogWriteError, refuse_catalog  # noqa: E402

# the engine's 20 OKX perpetuals (engine/config.py UNIVERSE), as Binance USD-M symbols
SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "FILUSDT",
    "NEARUSDT",
    "APTUSDT",
    "OPUSDT",
    "ARBUSDT",
    "AAVEUSDT",
)
PARTS = LIBRARY / "data" / "binance_um_1d_parts"
OUT = LIBRARY / "data" / "binance_um_1d"
# gzip header modification time of every merged file: fixed, so a re-merge is byte-identical
GZIP_MTIME = 0
# one JSON line per append, in the merged folder (engine/libfactors.py copies it into its manifest)
APPEND_LOG = "appends.jsonl"


def merge(parts: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """{field: DataFrame(time x symbol)} from parts/<symbol>/<field>.csv.gz.

    Every symbol must have a part folder with the same set of fields; each field is
    reindexed to the union of the symbols' close index, columns in `symbols` order."""
    fields: dict[str, dict[str, pd.Series]] = {}
    names = None
    for s in symbols:
        d = Path(parts) / s
        if not d.is_dir():
            raise FileNotFoundError(f"no part folder for {s}")
        p = load_panels(d)
        if names is None:
            names = set(p)
        elif set(p) != names:
            raise ValueError(f"{s} has fields {sorted(p)}, expected {sorted(names)}")
        for f, df in p.items():
            if list(df.columns) != [s]:
                raise ValueError(f"{d.name}/{f}.csv.gz holds {list(df.columns)}, expected [{s}]")
            fields.setdefault(f, {})[s] = df[s]
    if not fields or "close" not in fields:
        raise ValueError("no close panel among the parts")
    panels = {f: pd.DataFrame(d).reindex(columns=list(symbols)).sort_index() for f, d in fields.items()}
    index = panels["close"].index
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError("the merged close index is not sorted and unique")
    return {f: p.reindex(index=index) for f, p in panels.items()}


def coverage(panels: dict[str, pd.DataFrame]) -> list[str]:
    """One line per field: filled cells, and the earliest and latest first valid date over the symbols."""
    index = panels["close"].index
    lines = [f"index {index[0]} -> {index[-1]}, {len(index)} rows, {len(panels['close'].columns)} symbols"]
    for f, p in sorted(panels.items()):
        firsts = p.apply(lambda c: c.first_valid_index())
        missing = [c for c in p.columns if pd.isna(firsts[c])]
        have = firsts.dropna()
        span = f"{min(have).date()} / {max(have).date()}" if len(have) else "none"
        lines.append(
            f"{f:24s} {int(p.notna().sum().sum()):7d} cells, first valid (min / max) {span}"
            + (f", no data: {missing}" if missing else "")
        )
    return lines


class AppendRefused(ValueError):
    """The append stopped before it wrote a file (see the module docstring for the rules)."""


def _csv_text(df: pd.DataFrame) -> str:
    """The CSV text that save_panels writes for a frame (same float format, no compression)."""
    buf = io.StringIO()
    df.to_csv(buf, float_format="%.10g")
    return buf.getvalue()


def _shown(path: Path) -> str:
    """A path for a log or a message: relative to library/ when inside it, else the last name only."""
    path = Path(path).resolve()
    return path.relative_to(LIBRARY).as_posix() if LIBRARY in path.parents else path.name


def _read_merged(out: Path) -> dict[str, tuple[bytes, pd.DataFrame]]:
    """{field: (decompressed bytes, frame)} of every merged file, columns in file order."""
    files = sorted(Path(out).glob("*.csv.gz"))
    if not files:
        raise AppendRefused(f"no merged panel in {Path(out).name}: run the full merge first")
    got = {}
    for f in files:
        raw = gzip.decompress(f.read_bytes())
        got[f.name[: -len(".csv.gz")]] = (
            raw,
            pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True).astype(float),
        )
    return got


def append(parts: Path, out: Path, symbols: list[str], *, now: datetime | None = None, through=None) -> dict:
    """Add the new closed bars in `parts` to the merged panel in `out` (module docstring).

    Returns a report: the bars added, the last bar, and the filled cells of each field on
    the new bars. Raises AppendRefused, and writes nothing, when a rule fails."""
    out = Path(out)
    now = pd.Timestamp(datetime.now(UTC) if now is None else now)
    now = now.tz_localize("UTC") if now.tz is None else now.tz_convert("UTC")
    old = _read_merged(out)
    new = merge(Path(parts), symbols)
    if set(new) != set(old):
        raise AppendRefused(f"the parts have fields {sorted(new)}, the merged panel {sorted(old)}")
    frames = {f: df for f, (_, df) in old.items()}
    index = frames["close"].index
    for f, df in frames.items():
        if list(df.columns) != list(symbols):
            raise AppendRefused(f"{f}.csv.gz has columns {list(df.columns)}, --symbols gives {list(symbols)}")
        if not df.index.equals(index):
            raise AppendRefused(f"{f}.csv.gz has another index than close.csv.gz")
    last = index[-1]
    fresh = new["close"].index
    if through is not None:
        fresh = fresh[fresh <= pd.Timestamp(through, tz="UTC")]
    # downloaded bars that the panel already has: they must agree in the CSV text, and they stay as they are
    overlap = fresh[fresh <= last]
    differ = []
    for f in sorted(frames):
        a = frames[f].reindex(overlap)
        b = new[f].reindex(index=overlap, columns=symbols)
        a_txt, b_txt = _csv_text(a).splitlines()[1:], _csv_text(b).splitlines()[1:]
        differ += [f"{f} {x.split(',')[0]}" for x, y in zip(a_txt, b_txt) if x != y]
    if differ:
        raise AppendRefused(
            "downloaded bars differ from the merged cells: "
            + "; ".join(differ[:10])
            + (f" (and {len(differ) - 10} more)" if len(differ) > 10 else "")
        )
    added = fresh[fresh > last]
    if len(added) == 0:
        return {
            "added": [],
            "last_bar": str(last.date()),
            "rows": len(index),
            "overlap_checked": len(overlap),
            "filled": {},
        }
    want = pd.date_range(last + pd.Timedelta(days=1), periods=len(added), freq="D")
    if not added.equals(want):
        raise AppendRefused(
            f"the new bars do not follow the last merged bar ({last.date()}) day by day: "
            f"{[str(d.date()) for d in added]}"
        )
    forming = [str(d.date()) for d in added if d + pd.Timedelta(days=1) > pd.Timestamp(now)]
    if forming:
        raise AppendRefused(f"bar(s) {forming} are not closed at {pd.Timestamp(now).isoformat()}")
    # completeness for each coin. A coin with a kline bar on the last merged bar needs a kline
    # bar on each new day: close is checked without the bar condition. A coin with a kline bar on
    # a new day needs a value in each other field that it had on the last merged bar (a skipped
    # archive file leaves one coin empty). The panel is append-only, so an accepted hole stays.
    bars = new["close"].reindex(index=added, columns=symbols).notna()
    late = []
    for f, df in sorted(new.items(), key=lambda kv: (kv[0] != "close", kv[0])):  # close first
        had = frames[f].iloc[-1].notna()
        gap = df.reindex(index=added, columns=symbols).isna() & had
        if f != "close":
            gap &= bars
        late += [f"{f} {s} {d.date()}" for d, row in gap.iterrows() for s in row.index[row.to_numpy()]]
    if late:
        no_bar = sum(x.startswith("close ") for x in late)
        raise AppendRefused(
            f"{len(late)} value(s) are missing, although each coin had a value in that field on the last "
            f"merged bar ({last.date()}): "
            + "; ".join(late[:10])
            + (f" (and {len(late) - 10} more)" if len(late) > 10 else "")
            + (f". In {no_bar} of them, the coin has no kline bar (close)" if no_bar else "")
            + ". The source was not complete at download time. For example, Binance had not published "
            "an archive file at download time. Download those days again after one day. As an alternative, use --through "
            "to stop before them"
        )
    combined = {
        f: pd.concat([frames[f], new[f].reindex(index=added, columns=symbols)]).rename_axis(index.name) for f in frames
    }
    tmp = out.parent / f".{out.name}.append-tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        save_panels(combined, tmp, gzip_mtime=GZIP_MTIME)
        for f, (raw, _) in old.items():
            blob = (tmp / f"{f}.csv.gz").read_bytes()
            if int.from_bytes(blob[4:8], "little") != GZIP_MTIME:
                raise AppendRefused(f"{f}.csv.gz: the gzip header time is not {GZIP_MTIME}")
            if not gzip.decompress(blob).startswith(raw):
                raise AppendRefused(f"{f}.csv.gz: an old cell would change, so the append wrote nothing")
        for f in old:
            os.replace(tmp / f"{f}.csv.gz", out / f"{f}.csv.gz")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for f, (raw, _) in old.items():  # the published files, checked again
        if not gzip.decompress((out / f"{f}.csv.gz").read_bytes()).startswith(raw):
            raise AssertionError(f"{f}.csv.gz: an old cell changed after the replace")
    entry = {
        "appended_at": pd.Timestamp(now).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "first_bar": str(added[0].date()),
        "last_bar": str(added[-1].date()),
        "bars": len(added),
        "rows": len(index) + len(added),
        "parts": _shown(parts),
        "old_cells": "byte-equal",
    }
    with open(out / APPEND_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return {
        "added": [str(d.date()) for d in added],
        "last_bar": str(added[-1].date()),
        "rows": len(index) + len(added),
        "overlap_checked": len(overlap),
        "filled": {f: int(combined[f].loc[added].notna().sum().sum()) for f in sorted(combined)},
    }


def main(argv=None) -> int:
    """Run the full merge, or the append with ``--append``. Return 0 on success, 2 for a refused output or append."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", default=str(PARTS), help="folder of per-symbol part folders")
    ap.add_argument("--out", default=str(OUT), help="output folder (one <field>.csv.gz per field)")
    ap.add_argument("--symbols", default=",".join(SYMBOLS), help="comma separated, in output column order")
    ap.add_argument(
        "--append",
        action="store_true",
        help="add the new closed bars in --parts to the merged panel in --out; no old cell changes",
    )
    ap.add_argument("--through", help="with --append: add no bar after this UTC day (YYYY-MM-DD)")
    ap.add_argument("--now", help="with --append: the clock for the closed-bar rule (UTC; for tests)")
    args = ap.parse_args(argv)
    try:
        refuse_catalog(args.out, what="merge panels", option=False)
    except CatalogWriteError as exc:
        print(f"merge_panels: {exc}", file=sys.stderr)
        return 2
    symbols = [s for s in args.symbols.split(",") if s]
    if args.append:
        now = None
        if args.now is not None:
            t = pd.Timestamp(args.now)
            now = (t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")).to_pydatetime()
        try:
            rep = append(Path(args.parts), Path(args.out), symbols, now=now, through=args.through)
        except AppendRefused as exc:
            print(f"merge_panels: append refused: {exc}", file=sys.stderr)
            return 2
        if not rep["added"]:
            print(
                f"nothing to append: the panel ends on {rep['last_bar']} ({rep['overlap_checked']} downloaded "
                "bar(s) already in it agree with it)"
            )
            return 0
        print(
            f"appended {len(rep['added'])} bar(s) {rep['added'][0]} -> {rep['added'][-1]}; the panel now has "
            f"{rep['rows']} rows; every old cell is byte-equal"
        )
        for f, n in rep["filled"].items():
            print(f"{f:24s} {n:5d} cells on the new bars")
        return 0
    panels = merge(Path(args.parts), symbols)
    for line in coverage(panels):
        print(line)
    save_panels(panels, args.out, gzip_mtime=GZIP_MTIME)
    log = Path(args.out) / APPEND_LOG
    if log.is_file():
        log.unlink()
        print(f"removed {APPEND_LOG}: the full merge rebuilds the panel from the parts only")
    print(f"saved {len(panels)} fields to {_shown(Path(args.out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
