"""Load, validate and version-hash the evaluation config (eval_config.toml).

The module turns eval_config.toml into two version hashes and a list of structural
problems. It writes nothing.

CANONICAL JSON -- the one serialisation of every hash here
    json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False,
               ensure_ascii=True), with every float first quantized to 12
    significant digits. Quantizing removes the last-bit noise that differs
    between platforms and summation orders, so the same value always hashes
    the same; allow_nan=False makes a NaN or Inf a hard error instead of a
    silently unhashable "NaN" token; ensure_ascii keeps the bytes identical
    whatever the locale. Anything that is not a JSON type (a TOML date, a
    numpy scalar, a set) raises TypeError rather than being coerced, because a
    coercion rule would itself become part of the hash.

TWO VERSIONS
    eval_config_version      sha256 over every table except [generator]
    generator_config_version sha256 over [generator] alone
    finder.py --check and the generator probe print both.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from datetime import date, timedelta
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("eval_config.toml")
SIG_DIGITS = 12

REQUIRED_TABLES = ("meta", "label", "zones", "regressors", "g0", "generator")
#: zone name -> key of its [first, last] Sunday pair, in calendar order
ZONE_SUNDAYS = (
    ("g_fit", "g_fit_sundays"),
    ("g_internal_embargo", "g_internal_embargo_sundays"),
    ("g_eval", "g_eval_sundays"),
    ("embargo", "embargo_sundays"),
    ("i", "i_sundays"),
)


# --------------------------------------------------------------------------- canonical JSON


def _quantize(obj, where: str = "$"):
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite number at {where}")
        q = float(format(obj, f".{SIG_DIGITS}g"))
        return 0.0 if q == 0 else q  # -0.0 and 0.0 hash the same
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"non-string key {k!r} at {where}")
            out[k] = _quantize(v, f"{where}.{k}")
        return out
    if isinstance(obj, (list, tuple)):
        return [_quantize(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    raise TypeError(f"{type(obj).__name__} at {where} is not a canonical JSON type")


def canonical_json(obj) -> str:
    return json.dumps(_quantize(obj), sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)


def sha256_hex(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()


# --------------------------------------------------------------------------- load / versions


def load(path: str | Path = CONFIG_PATH) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def split(cfg: dict) -> tuple[dict, dict]:
    """(eval part, generator part)."""
    return {k: v for k, v in cfg.items() if k != "generator"}, dict(cfg.get("generator") or {})


def versions(cfg: dict) -> dict:
    ev, gen = split(cfg)
    return {"eval_config_version": sha256_hex(ev), "generator_config_version": sha256_hex(gen)}


# --------------------------------------------------------------------------- validation


def _sunday(s: str) -> date:
    d = date.fromisoformat(s)
    if d.weekday() != 6:
        raise ValueError(f"{s} is not a Sunday")
    return d


def _n_sundays(a: date, b: date) -> int:
    return (b - a).days // 7 + 1


def validate(cfg: dict) -> list[str]:
    """Structural problems, empty when the sheet is coherent. No statistics here:
    only types, ranges and calendar arithmetic."""
    out: list[str] = []
    try:
        canonical_json(cfg)
    except (TypeError, ValueError) as exc:
        return [f"not canonical-JSON safe: {exc}"]
    if cfg.get("schema") != 1:
        out.append("schema must be 1")
    missing = [t for t in REQUIRED_TABLES if not isinstance(cfg.get(t), dict)]
    if missing:
        return out + [f"missing tables: {', '.join(missing)}"]

    lab = cfg["label"]
    if (lab.get("name"), lab.get("horizon_days"), lab.get("lag_days")) != ("fwd_ret_7", 7, 0):
        out.append("label must be fwd_ret_7 with horizon 7 and lag 0")

    # zones: Sundays, ordered, contiguous at the weekly grid, counts as declared
    z = cfg["zones"]
    prev_last = None
    counts = z.get("expected_counts") or {}
    for name, key in ZONE_SUNDAYS:
        try:
            first, last = (_sunday(x) for x in z[key])
        except (KeyError, ValueError, TypeError) as exc:
            out.append(f"zones.{key}: {exc}")
            continue
        if last < first:
            out.append(f"zones.{key}: ends before it starts")
        if prev_last is not None and first != prev_last + timedelta(days=7):
            out.append(f"zones.{key}: does not start the Sunday after the previous zone")
        if counts.get(name) != _n_sundays(first, last):
            out.append(f"zones.{key}: {_n_sundays(first, last)} Sundays, expected_counts says {counts.get(name)}")
        prev_last = last
    if sum(int(v) for v in counts.values()) != 234:
        out.append("zones.expected_counts must sum to 234 Sundays")

    for k, lo, hi in (("g0.min_coverage", 0, 1), ("g0.min_finite_share", 0, 1)):
        v = cfg
        for part in k.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo < v <= hi):
            out.append(f"{k} must be a number in ({lo}, {hi}]")
    gen = cfg["generator"]
    for t in ("terminals", "settings"):
        if not isinstance(gen.get(t), dict):
            out.append(f"generator must define the table {t}")
    return out
