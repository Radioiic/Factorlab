"""The input of a factor test: a formula or a values file, its values at each Sunday and its numbers.

THE FORMULA. A formula is an expression of the library expression language
(library/factorlab/expr.py), for example -ts_mean(ex_ret, 20). validate_formula() checks it
before any computation:
    - the parser accepts it (numbers, field names, the library operators, + - * / ** and
      unary minus only);
    - each window or lag argument is a constant whole number from 1 to MAX_WINDOW, so no
      operator can read a later row (delay(close, -1) is refused);
    - the first argument of a ts_*, cs_*, clip or bench call reads data (it is not a constant);
    - the panel source has each raw field that the formula reads. On the OKX data a formula
      that reads open interest is refused: the time of the stored OKX open-interest value is
      not proven.
The canonical text (ast.unparse of the parsed tree) is the identity of the formula. Spacing
and extra parentheses do not change it.

THE VALUES AT EACH SUNDAY CUT (formula_values). For each Sunday S up to the date of the
complete data of the source, the value is computed on the panels cut at S (no row after S).
The same value from the full panels must be equal (np.isclose, rtol 1e-9, atol 1e-12, NaN
equal to NaN). Then the formula is proven point in time. The official values are the cut
values.

THE VALUES FILE (read_values_file). A CSV file (UTF-8, a header row date,symbol,value) or a
Parquet file (the columns date, symbol and value). A date is a UTC day. A symbol is a coin of
config.UNIVERSE (BTC-USDT, BTCUSDT or BTC/USDT:USDT). A value is a finite number; an empty
value is missing. Factorlab cannot prove a values file point in time.

THE DATA DATES (load_source). data_through is the newest last bar over the 20 coins. A coin
is live when its last bar is at most GRACE_DAYS days before data_through. complete_through is
the oldest last bar over the live coins. A missing cell (coin, field, day) is a gap. A gap is
final when data_through is GRACE_DAYS days or more after its day, else it is pending (gaps).

THE NUMBERS (factor_numbers). The weekly IC and rIC of the label fwd_ret_7, the persistence and
the coverage, on the Sundays with a complete label. They are numbers only: no rule reads them.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

import config as C
import data_io
import libfactors as lf
from factorlab.expr import ExpressionError, _constant_value, max_lookback, parse, raw_dependencies

SOURCES = ("okx", "binance")
MAX_WINDOW = 1000  # the largest window or lag of a formula
MAX_FILE_BYTES = 100 * 2**20
MAX_FILE_ROWS = 2_000_000
PIT_RTOL, PIT_ATOL = 1e-9, 1e-12  # = libfactors.PIT_RTOL, PIT_ATOL
GRACE_DAYS = 7  # a missing cell is final after this many days
OKX_REFUSED_FIELDS = ("open_interest", "open_interest_value")
VALUE_COLUMNS = ("date", "symbol", "value")
LIST_MAX = 10  # a refusal names at most this many items
PIT_RULE = (
    "Each Sunday value is computed on the data up to that Sunday and is equal to the value of the full computation."
)
NOTES = {
    "vwap_close": "On the OKX data, vwap is equal to the close.",
    "funding_splice": "Before the first OKX funding print on disk, funding_rate is the Binance rate.",
    "taker_last_5m": "taker_buy_sell_ratio is the last 5-minute value of the day.",
}
# The exceptions of an operator that make a formula impossible to compute
COMPUTE_ERRORS = (ValueError, TypeError, AttributeError, IndexError, ZeroDivisionError)


class InputError(ValueError):
    """A formula or a values file that Factorlab does not accept. Its text is a UI text."""


def _symbols() -> list[str]:
    return [C.sym_to_file(s) for s in C.UNIVERSE]


def _day(x) -> date:
    """A date (a date, a Timestamp or a text YYYY-MM-DD)."""
    return x if isinstance(x, date) and not isinstance(x, datetime) else pd.Timestamp(x).date()


def _ts(x) -> pd.Timestamp:
    """The UTC midnight Timestamp of a day."""
    return pd.Timestamp(str(_day(x)), tz="UTC")


def same_values(a, b) -> np.ndarray:
    """The point-in-time comparison of two value arrays: np.isclose with PIT_RTOL and PIT_ATOL,
    NaN equal to NaN (+-inf counts as NaN)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = np.where(np.isfinite(a), a, np.nan)
    b = np.where(np.isfinite(b), b, np.nan)
    return np.isclose(a, b, rtol=PIT_RTOL, atol=PIT_ATOL, equal_nan=True)


# --------------------------------------------------------------------------- the formula


def _signatures() -> dict[str, tuple[tuple[str, ...], ...]]:
    """{operator: the accepted argument kinds} ("p" data, "w" window, "l" lag, "f" number)."""
    from factorfinder.libops import _SIGNATURES

    out: dict[str, tuple[tuple[str, ...], ...]] = {}
    for name, (kinds, _category, _comm) in _SIGNATURES.items():
        if name in ("add", "sub", "mul", "neg"):
            continue  # the arithmetic is written with + - * and unary minus
        out[name] = (kinds,)
    out["ts_autocorr"] = (("p", "w"), ("p", "w", "l"))
    out["bench"] = (("p",),)
    return out


def _invalid(text: str) -> InputError:
    return InputError(f"The formula is not valid: {text}")


def _check_arguments(tree: ast.AST) -> None:
    """The static argument rules (module docstring). InputError for the first rule that fails."""
    sigs = _signatures()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        forms = sigs.get(name)
        if forms is None:
            raise _invalid(f"{name} is not an operator of the library.")
        kinds = next((f for f in forms if len(f) == len(node.args)), None)
        if kinds is None:
            counts = " or ".join(str(len(f)) for f in forms)
            noun = "argument" if counts == "1" else "arguments"
            raise _invalid(f"{name} takes {counts} {noun}, not {len(node.args)}.")
        for k, (kind, arg) in enumerate(zip(kinds, node.args), start=1):
            if kind in ("w", "l"):
                v = _constant_value(arg)
                if v is None or not np.isfinite(v) or v != int(v) or not 1 <= v <= MAX_WINDOW:
                    raise _invalid(f"{name} argument {k} must be an integer from 1 to {MAX_WINDOW}.")
            elif (
                kind == "p"
                and k == 1
                and (name.startswith(("ts_", "cs_")) or name in ("clip", "bench"))
                and _constant_value(arg) is not None
            ):
                raise _invalid(f"{name} argument 1 must read data. It cannot be a constant.")


def _source_problem(raw: set[str], source: str) -> str | None:
    """The refusal text of a formula that reads a field that the source cannot give, or None."""
    if source == "okx":
        missing = sorted(raw - set(lf.SUPPLIED))
        if missing:
            return f"The formula reads {missing[0]}. The OKX data does not have this field. Use --source binance."
        oi = sorted(raw & set(OKX_REFUSED_FIELDS))
        if oi:
            return (
                f"The formula reads {oi[0]}. On the OKX data, the time of the open-interest value is not proven. "
                "Use --source binance."
            )
        return None
    missing = sorted(raw - set(lf.BINANCE_FIELDS))
    if missing:
        return f"The formula reads {missing[0]}. The Binance data does not have this field."
    return None


def _parse(expr: str) -> ast.AST:
    if not isinstance(expr, str):
        raise _invalid("a formula must be a text.")
    try:
        return parse(expr)
    except ExpressionError as exc:
        raise _invalid(str(exc)) from None


def auto_source(expr: str) -> str:
    """The source "okx" when the OKX data has every raw field of the formula (no vwap, no open interest),
    else "binance"."""
    raw = set(raw_dependencies(expr))
    names = lf.expr_atoms(expr)[1]
    if raw <= set(lf.SUPPLIED) and not names & set(lf.VWAP_FIELDS) and not raw & set(OKX_REFUSED_FIELDS):
        return "okx"
    return "binance"


def validate_formula(expr: str, source: str | None = None) -> dict:
    """The static check of a formula (module docstring). Returns {canonical, expression_sha256,
    source, raw_fields, names, notes, max_lookback}. InputError when the formula is refused."""
    tree = _parse(expr.strip() if isinstance(expr, str) else expr)
    canonical = ast.unparse(tree)
    if not canonical.isascii():
        raise _invalid("the formula must be ASCII text.")
    if canonical != expr.strip():
        tree = _parse(canonical)
    _check_arguments(tree)
    raw = set(raw_dependencies(canonical))
    names = set(lf.expr_atoms(canonical)[1])
    if source is None:
        source = auto_source(canonical)
    if source not in SOURCES:
        raise InputError(f"The source must be {' or '.join(SOURCES)}, not {source!r}.")
    problem = _source_problem(raw, source)
    if problem:
        raise InputError(problem)
    notes = []
    if source == "okx":
        if names & set(lf.VWAP_FIELDS):
            notes.append("vwap_close")
        if "funding_rate" in raw:
            notes.append("funding_splice")
    elif "taker_buy_sell_ratio" in raw:
        notes.append("taker_last_5m")
    return {
        "canonical": canonical,
        "expression_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        "source": source,
        "raw_fields": sorted(raw),
        "names": sorted(names),
        "notes": notes,
        "max_lookback": int(max_lookback(canonical)),
    }


def library_formula(factor_id: str, *, catalog: Path | None = None) -> dict:
    """The library definition with this id: {factor_id, name, family, expression}. Read-only."""
    path = Path(lf.CATALOG if catalog is None else catalog)
    if not path.is_file():
        raise InputError("The library catalog is not on disk. Run ./setup.sh first.")
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT factor_id, name, family, expression FROM factors WHERE factor_id = ?", (str(factor_id),)
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise InputError(f"No library definition has the id {factor_id}.")
    return dict(zip(("factor_id", "name", "family", "expression"), row))


# --------------------------------------------------------------------------- the data of a source


def load_source(source: str, raw: dict | None = None) -> dict:
    """The full panels of a panel source and its dates (module docstring).

    Returns {source, panels, presence, symbols, last_bars {coin: date}, data_through,
    complete_through, live}. presence holds the panels that the gap rule reads: the OKX panels,
    or the Binance panels before the cleaning rules (a cleaned cell is not a gap)."""
    if source not in SOURCES:
        raise ValueError(f"unknown panel source {source!r}")
    if source == "okx":
        raw = lf.load_raw() if raw is None else raw
        panels = lf.build_panels(raw)
        presence = panels
    else:
        raw = lf.load_raw_binance() if raw is None else raw
        presence = lf.build_panels_binance(raw, clean=False)
        panels = lf.clean_binance(presence)
    symbols = list(panels["close"].columns)
    if symbols != _symbols():
        raise SystemExit(f"judge: the {source} data does not hold the 20 coins of config.UNIVERSE in their order")
    close = panels["close"]
    last_bars = {}
    for s in symbols:
        has = close.index[close[s].notna().to_numpy()]
        if len(has):
            last_bars[s] = has.max().date()
    if not last_bars:
        raise SystemExit(f"judge: the {source} data has no daily bar")
    through = max(last_bars.values())
    live = [s for s, d in last_bars.items() if d >= through - timedelta(days=GRACE_DAYS)]
    return {
        "source": source,
        "panels": panels,
        "presence": presence,
        "symbols": symbols,
        "last_bars": last_bars,
        "data_through": through,
        "complete_through": min(last_bars[s] for s in live),
        "live": live,
    }


def live_coins(src: dict) -> list[str]:
    """The coins whose last bar is at most GRACE_DAYS days before data_through."""
    return list(src["live"])


def complete_through(src: dict) -> date:
    """The oldest last bar over the live coins."""
    return cast(date, src["complete_through"])


def cut(panels: dict, sunday) -> dict:
    """The panels cut at the row of the Sunday (no row after it)."""
    t = _ts(sunday)
    return {k: v.loc[:t] for k, v in panels.items()}


def read_fields(expressions: list[str]) -> list[str]:
    """The raw fields that the formulas read, and close (each coin's bar), sorted."""
    out = {"close"}
    for e in expressions:
        out |= set(raw_dependencies(e))
    return sorted(out)


def gaps(src: dict, first, last, fields: list[str]) -> dict:
    """The missing cells of the days first .. last: {"pending": [...], "final": [...]}, each a
    list of (coin, field, day). For each field and each coin with a value in that field on at
    least one day of first - GRACE_DAYS .. last, each day of first .. last without a value is a
    gap (also before the first value of the span: a coin or a field that starts in the week). A
    day without the bar of the coin is one gap of "close" (each field of the panels is missing
    without a bar). A gap is final when data_through >= day + GRACE_DAYS, else pending."""
    pres = src["presence"]
    f0, l0 = _day(first), _day(last)
    days = pd.date_range(_ts(f0 - timedelta(days=GRACE_DAYS)), _ts(l0), freq="D")
    start = GRACE_DAYS  # the row of `first`
    through = src["data_through"]
    bar = np.isfinite(pres["close"].reindex(index=days, columns=src["symbols"]).to_numpy(dtype=np.float64))
    out: dict[str, list[tuple[str, str, str]]] = {"pending": [], "final": []}
    for f in fields:
        a = pres[f].reindex(index=days, columns=src["symbols"]).to_numpy(dtype=np.float64)
        has = np.isfinite(a)
        gap = ~has & has.any(axis=0)  # a coin with no value in the whole span has no gap
        if f != "close" and "close" in fields:
            gap &= bar  # a day without a bar is the gap of close
        for i, j in zip(*np.nonzero(gap[start:])):
            day = days[start + i].date()
            kind = "final" if through >= day + timedelta(days=GRACE_DAYS) else "pending"
            out[kind].append((src["symbols"][j], f, str(day)))
    for v in out.values():
        v.sort(key=lambda x: (x[2], x[0], x[1]))
    return out


def gap_start(src: dict, coin: str, field: str, day) -> str:
    """The first day of the run of missing days of (coin, field) that holds `day`. Without an
    earlier value the run starts on the first day of the data."""
    s = src["presence"][field][coin]
    t = _ts(day)
    known = s.index[(s.index < t) & s.notna().to_numpy()]
    first = known.max() + pd.Timedelta(days=1) if len(known) else s.index.min()
    return str(min(first, t).date())


def closed_sundays(first, complete_through, through=None) -> list[date]:
    """The Sundays from `first` to `complete_through` (and not after `through`)."""
    last = _day(complete_through)
    if through is not None:
        last = min(last, _day(through))
    out, d = [], _day(first)
    while d <= last:
        out.append(d)
        d += timedelta(days=7)
    return out


def _field_update(h, f: str, df: pd.DataFrame) -> None:
    """Add one field to a hash: the name, the coins, the dates (ns) and the values. NaN has one
    bit pattern and -0.0 counts as 0.0, so the hash follows the numbers only."""
    a = df.to_numpy(dtype=np.float64)
    a = np.where(np.isnan(a), np.nan, a) + 0.0
    h.update(f.encode("ascii") + b"\0" + "\t".join(map(str, df.columns)).encode("ascii") + b"\0")
    h.update(np.ascontiguousarray(cast(Any, df.index).as_unit("ns").asi8, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(a, dtype="<f8").tobytes())


def field_sha256(panels: dict) -> dict:
    """{field: sha256 of that field alone}."""
    out = {}
    for f in sorted(panels):
        h = hashlib.sha256()
        _field_update(h, f, panels[f])
        out[f] = h.hexdigest()
    return out


# --------------------------------------------------------------------------- values of a formula


def full_values(expr: str, panels: dict) -> np.ndarray:
    """(days, coins) float64 values of a formula on the panels (+-inf as NaN)."""
    try:
        v = lf._values(lf._evaluator(panels), expr)
    except COMPUTE_ERRORS as exc:
        raise InputError(f"The formula cannot be computed: {type(exc).__name__}: {exc}") from None
    return np.where(np.isfinite(v), v, np.nan)


def _cut_row(expr: str, panels: dict, pos: int) -> np.ndarray:
    """The last row of the values on the panels cut at row `pos`."""
    part = {k: v.iloc[: pos + 1] for k, v in panels.items()}
    return full_values(expr, part)[-1]


def formula_values(expr: str, src: dict, *, last: date | None = None) -> dict:
    """The values of a formula at each Sunday cut (module docstring) up to complete_through (and
    `last`). Returns {sundays, symbols, values (T, 20): the cut values, full (T, 20), pit}."""
    if src["source"] == "binance":
        lf.cleaning_freeze_record()  # SystemExit when the cleaning rules are not the frozen set
    panels = src["panels"]
    idx = panels["close"].index
    stop = src["complete_through"] if last is None else min(src["complete_through"], last)
    pos = [int(p) for p in lf.sunday_positions(idx) if idx[p].date() <= stop]
    full_all = full_values(expr, panels)
    full = full_all[pos] if pos else np.zeros((0, len(src["symbols"])))
    cuts = np.array([_cut_row(expr, panels, p) for p in pos]).reshape(len(pos), len(src["symbols"]))
    equal = same_values(cuts, full)
    finite = int(np.isfinite(cuts).sum())
    sundays = pd.DatetimeIndex([idx[p] for p in pos])
    diff_rows = np.flatnonzero(~equal.all(axis=1)) if len(pos) else np.array([], dtype=int)
    if len(diff_rows):
        status = "fail"
    elif finite == 0:
        status = "empty"
    else:
        status = "proven"
    pit = {
        "status": status,
        "sundays": len(pos),
        "cells": int(cuts.size),
        "finite_cells": finite,
        "diff_cells": int((~equal).sum()),
        "first_diff_sunday": str(sundays[diff_rows[0]].date()) if len(diff_rows) else None,
        "rule": PIT_RULE,
    }
    return {"sundays": sundays, "symbols": list(src["symbols"]), "values": cuts, "full": full, "pit": pit}


def sunday_value(expr: str, src: dict, sunday, full: np.ndarray | None = None) -> dict:
    """The cut value of one Sunday and its check against the full value: {values (20,), full,
    equal, cells, finite_cells}. full: the full values of the formula on the source panels."""
    if src["source"] == "binance":
        lf.cleaning_freeze_record()
    panels = src["panels"]
    idx = panels["close"].index
    t = _ts(sunday)
    if t not in idx:
        raise SystemExit(f"judge: the {src['source']} data has no row for {_day(sunday)}")
    pos = int(idx.get_loc(t))
    if full is None:
        full = full_values(expr, panels)
    v = _cut_row(expr, panels, pos)
    same = same_values(v, full[pos])
    return {
        "values": v,
        "full": full[pos],
        "equal": bool(same.all()),
        "cells": int(same.size),
        "finite_cells": int(np.isfinite(v).sum()),
    }


# --------------------------------------------------------------------------- the values file

_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})?)?$")
_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_UTC_NAMES = ("UTC", "Etc/UTC", "Etc/GMT", "GMT", "Z", "+00:00")


def _aliases() -> dict[str, str]:
    """{accepted symbol name: engine symbol} for the 20 coins."""
    out = {}
    for s in C.UNIVERSE:
        eng = C.sym_to_file(s)
        out[eng] = eng
        out[s] = eng
        out[lf.binance_symbol(eng)] = eng
    return out


def _refuse(text: str) -> InputError:
    return InputError(f"The values file is not valid: {text}")


def _some(items: list) -> str:
    shown = ", ".join(str(x) for x in items[:LIST_MAX])
    return shown + (f" (and {len(items) - LIST_MAX} more)" if len(items) > LIST_MAX else "")


def _csv_frame(data: bytes) -> pd.DataFrame:
    """date, symbol and value as text, one row for each data row of a CSV file."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise _refuse("the CSV file is not UTF-8 text.") from None
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or rows[0] != list(VALUE_COLUMNS):
        raise _refuse('the first line of a CSV file must be "date,symbol,value".')
    body = [r for r in rows[1:] if r]
    bad = [n for n, r in enumerate(rows[1:], start=2) if r and len(r) != 3]
    if bad:
        raise _refuse(f"these lines do not have 3 fields: {_some(bad)}.")
    if len(body) > MAX_FILE_ROWS:
        raise _refuse(f"the file has more than {MAX_FILE_ROWS:,} rows.")
    lines = [n for n, r in enumerate(rows[1:], start=2) if r]
    df = pd.DataFrame(body, columns=list(VALUE_COLUMNS), dtype=object)
    df.index = pd.Index(lines, name="line")
    d = df["date"].astype(str).str.strip()
    m = d.str.extract(_DATE_RE)
    bad_date = m[0].isna()
    if bad_date.any():
        raise _refuse(f"these lines do not have a date YYYY-MM-DD: {_some(list(df.index[bad_date]))}.")
    has_time = m[1].notna()
    frac_zero = m[4].fillna("0").str.fullmatch(r"0+")
    midnight = ~has_time | ((m[1] == "00") & (m[2] == "00") & (m[3] == "00") & frac_zero)
    if not midnight.all():
        raise _refuse(f"a time must be 00:00:00. These lines have another time: {_some(list(df.index[~midnight]))}.")
    offset_ok = m[5].isna() | m[5].isin(["Z", "+00:00"])
    if not offset_ok.all():
        raise _refuse(f"a time must be UTC. These lines have another offset: {_some(list(df.index[~offset_ok]))}.")
    try:
        days = pd.to_datetime(m[0], format="%Y-%m-%d", utc=True)
    except (ValueError, TypeError):
        raise _refuse("a date is not a valid calendar date.") from None
    v = df["value"].astype(str).str.strip()
    missing = (v == "") | (v.str.lower() == "nan")
    number = v.str.fullmatch(_NUM_RE)
    infinite = v.str.lower().str.lstrip("+-").isin(["inf", "infinity"])
    if infinite.any():
        raise _refuse(f"a value must be a finite number. These lines have inf: {_some(list(df.index[infinite]))}.")
    not_number = ~missing & ~number
    if not_number.any():
        raise _refuse(f"these lines have a value that is not a number: {_some(list(df.index[not_number]))}.")
    values = pd.to_numeric(v.where(~missing, "nan"), errors="coerce").astype(np.float64)
    if not np.isfinite(values[~missing].to_numpy()).all():
        raise _refuse("a value must be a finite number.")
    return pd.DataFrame({"date": days.to_numpy(), "symbol": df["symbol"].astype(str).to_numpy(), "value": values})


def _parquet_frame(path: Path) -> pd.DataFrame:
    """date, symbol and value of a Parquet file, with the type rules of read_values_file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(path)
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise _refuse(f"the Parquet file cannot be read: {type(exc).__name__}.") from None
    if sorted(table.column_names) != sorted(VALUE_COLUMNS):
        raise _refuse("a Parquet file must have exactly the columns date, symbol and value.")
    if table.num_rows > MAX_FILE_ROWS:
        raise _refuse(f"the file has more than {MAX_FILE_ROWS:,} rows.")
    dt = table.schema.field("date").type
    if pa.types.is_date(dt):
        days = pd.to_datetime(pd.Series(table.column("date").to_pylist()), utc=True)
    elif pa.types.is_timestamp(dt):
        tz = getattr(dt, "tz", None)
        if tz is not None and str(tz) not in _UTC_NAMES:
            raise _refuse("a timestamp of the column date must be UTC.")
        days = pd.to_datetime(table.column("date").to_pandas(), utc=True)
    else:
        raise _refuse("the column date must be a date or a timestamp.")
    if days.isna().any():
        raise _refuse("the column date has an empty cell.")
    if not (days == days.dt.normalize()).all():
        raise _refuse("a time must be 00:00:00.")
    st = table.schema.field("symbol").type
    if not (pa.types.is_string(st) or pa.types.is_large_string(st)):
        raise _refuse("the column symbol must be text.")
    vt = table.schema.field("value").type
    if not (pa.types.is_floating(vt) or pa.types.is_integer(vt)):
        raise _refuse("the column value must be a number column (not a true/false or text column).")
    values = table.column("value").to_pandas().astype(np.float64)
    if np.isinf(values.to_numpy()).any():
        raise _refuse("a value must be a finite number.")
    symbols = pd.Series(table.column("symbol").to_pylist(), dtype=object).fillna("")
    return pd.DataFrame({"date": days.to_numpy(), "symbol": symbols.astype(str).to_numpy(), "value": values.to_numpy()})


def read_values_file(path: Path, *, through: date, drop_no_bar: bool = True) -> tuple[pd.DataFrame, dict]:
    """(frame, report) of a values file (module docstring).

    frame: one row for each Sunday of the file (UTC midnight index), the 20 engine symbols in
    config.UNIVERSE order, NaN where the file has no value. Rows on a day that is not a Sunday
    are dropped and counted. drop_no_bar: also drop (and count) the rows of a coin without an
    OKX bar on that date. A date after `through` is refused."""
    p = Path(path)
    if not p.is_file():
        raise InputError(f"The values file {p.name} is not a file.")
    suffix = p.suffix
    if suffix not in (".csv", ".parquet"):
        raise _refuse("the file name must end with .csv or .parquet.")
    if p.stat().st_size > MAX_FILE_BYTES:
        raise _refuse(f"the file has more than {MAX_FILE_BYTES:,} bytes.")
    data = p.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise _refuse(f"the file has more than {MAX_FILE_BYTES:,} bytes.")
    df = _csv_frame(data) if suffix == ".csv" else _parquet_frame(p)
    if not len(df):
        raise _refuse("the file has no data row.")
    alias = _aliases()
    unknown = sorted(set(df["symbol"]) - set(alias))
    if unknown:
        raise _refuse(f"these symbols are not coins of Factorlab: {_some(unknown)}.")
    df["symbol"] = df["symbol"].map(alias)
    dup = df.duplicated(["date", "symbol"], keep=False)
    if dup.any():
        pairs = sorted({f"{d.date()} {s}" for d, s in zip(df["date"][dup], df["symbol"][dup])})
        raise _refuse(f"these date and symbol pairs are in the file more than one time: {_some(pairs)}.")
    late = df["date"].dt.date > _day(through)
    if late.any():
        raise _refuse(f"the file has a date after {_day(through)}: {df['date'][late].min().date()}.")
    rows = len(df)
    sunday = df["date"].dt.dayofweek == 6
    non_sunday = int((~sunday).sum())
    df = df[sunday]
    no_bar = 0
    if drop_no_bar and len(df):
        close = data_io.close_matrix().reindex(columns=_symbols())
        has = close.notna().reindex(pd.DatetimeIndex(df["date"]), fill_value=False).to_numpy(dtype=bool)
        col = {s: j for j, s in enumerate(_symbols())}
        bar = has[np.arange(len(df)), [col[s] for s in df["symbol"]]]
        no_bar = int((~bar).sum())
        df = df[bar]
    frame = df.pivot(index="date", columns="symbol", values="value").reindex(columns=_symbols()).sort_index()
    frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC").as_unit("ns")
    frame.index.name = None
    frame.columns.name = None
    frame = frame.astype(np.float64)
    report = {
        "file_name": p.name,
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "rows": rows,
        "sunday_rows": len(df),
        "used_cells": int(np.isfinite(frame.to_numpy()).sum()),
        "non_sunday_rows": non_sunday,
        "no_bar_rows": no_bar,
        "first_date": str(frame.index[0].date()) if len(frame) else None,
        "last_date": str(frame.index[-1].date()) if len(frame) else None,
    }
    return frame, report


# --------------------------------------------------------------------------- risk proxies and numbers


def risk_proxies(panels: dict, symbols: list, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """The three registered rIC regressors on each date, from the OKX bars up to that date:
    pit_symbol_tilt ("tilt"), btc_beta_90d ("beta") and size_proxy ("size"), (T, m) arrays.

    The formulas of libfactors.regressors. Each one reads only rows up to its date, so the
    values from the full panels equal the values from panels cut at the date."""
    close = panels["close"].reindex(columns=symbols)
    tilt = lf.pit_tilt(close, lf.TILT_SHRINK_DAYS)
    r = close / close.shift(1) - 1.0
    rb = r[lf.BENCHMARK]

    def m(x):
        return x.rolling(lf.BETA_DAYS, min_periods=lf.BETA_DAYS).mean()

    cov = m(r.mul(rb, axis=0)).sub(m(r).mul(m(rb), axis=0))
    var = m(rb * rb) - m(rb) ** 2
    beta = cov.div(var.where(var > 0), axis=0)
    qv = panels["quote_volume"].reindex(columns=symbols)
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.log(qv.rolling(lf.SIZE_DAYS, min_periods=lf.SIZE_DAYS).mean())

    def pick(x):
        return x.reindex(dates).to_numpy(dtype=float)

    return {"tilt": pick(tilt), "beta": pick(beta), "size": pick(size)}


def factor_numbers(values: pd.DataFrame, okx: dict) -> dict:
    """The numbers of a factor (no rule reads them) on its Sundays with a complete label
    (S + 7 <= the OKX complete_through): weeks, ic_mean, ic_t, ric_mean, ric_t (the residual IC
    on the three risk proxies), persistence and coverage. Without the label table:
    {"available": False, "reason"}."""
    symbols = _symbols()
    v = values.copy()
    v.index = pd.DatetimeIndex(v.index)
    v.index = v.index.tz_localize("UTC") if v.index.tz is None else v.index.tz_convert("UTC")
    last = _ts(okx["complete_through"]) - pd.Timedelta(days=7)
    v = v[(v.index <= last) & (v.index.dayofweek == 6)].reindex(columns=symbols)
    out: dict[str, Any] = {
        "available": True,
        "weeks": 0,
        "ic_mean": None,
        "ic_t": None,
        "ric_mean": None,
        "ric_t": None,
        "persistence": None,
        "coverage": None,
    }
    if not len(v):
        return out
    try:
        lab = data_io.label_table(columns=["date", "symbol", lf.LABEL], dates=v.index)
    except SystemExit as exc:
        return {"available": False, "reason": str(exc)}
    sundays = pd.DatetimeIndex(v.index)
    label, elig = lf.label_grid(lab, sundays, symbols)
    vals = lf.snap_ties(v.to_numpy(dtype=np.float64)[None])
    regs = risk_proxies(okx["panels"], symbols, sundays)
    ic, ric, _n, _base = lf.weekly_ic(vals, label, elig, cast(dict, regs))
    st_ic, st_ric = lf.tstats(ic), lf.tstats(ric)
    pers, _cnt = lf.persistence(vals, elig, sundays)
    from factorfinder import probe

    _reason, g0 = probe.g0_label_free(v.to_numpy(dtype=np.float64), elig, lf.eval_config()["g0"])

    def f(x) -> float | None:
        x = float(x)
        return x if np.isfinite(x) else None

    out.update(
        {
            "weeks": int(len(sundays)),
            "ic_mean": f(st_ic["mean"][0]),
            "ic_t": f(st_ic["t"][0]),
            "ric_mean": f(st_ric["mean"][0]),
            "ric_t": f(st_ric["t"][0]),
            "persistence": f(pers[0]),
            "coverage": f(g0["coverage"]),
        }
    )
    return out
