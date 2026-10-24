"""Factor library - the de-duplicated "book of alphas".

A :class:`FactorLibrary` is an in-memory, optionally disk-backed registry of
named factors: the expression that produced each one, its value panel, its
evaluation :class:`~alphaforge.evaluation.report.FactorReport` and free-form
tags.  The one rule it enforces is **novelty**: a candidate whose mean
cross-sectional Spearman correlation with any incumbent exceeds ``max_corr`` is
refused unless ``force=True``.

Each new factor is compared with the incumbents by cross-sectional rank
correlation, so a redundant factor does not go into the library.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.evaluation.ic import rowwise_corr
from alphaforge.evaluation.report import FactorReport

logger = logging.getLogger("alphaforge.library")

__all__ = ["FactorLibrary", "mean_spearman_corr"]

_LIBRARY_JSON = "library.json"
_FORMAT_VERSION = 1
_DATE_NAME = "date"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _check_name(name: Any) -> str:
    """Validate a factor name (it doubles as the parquet file stem)."""
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"invalid factor name {name!r} (non-empty str without path separators)")
    return name


def _as_panel(values: Any, name: str) -> pd.DataFrame:
    """Defensive ``float64`` copy of a value panel with a sorted index."""
    if not isinstance(values, pd.DataFrame):
        raise TypeError(f"values for {name!r} must be a pandas DataFrame, got {type(values).__name__}")
    df = values.astype(np.float64, copy=True)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if isinstance(df.index, pd.DatetimeIndex):
        # freq-less, named "date" - same convention as DataStore, so an
        # in-memory panel and its parquet round-trip compare equal.
        df.index = pd.DatetimeIndex(np.asarray(df.index), name=_DATE_NAME)
    if not all(isinstance(c, str) for c in df.columns):
        df.columns = pd.Index([str(c) for c in df.columns], dtype=object)
    return df


def _as_tags(tags: Any) -> tuple[str, ...]:
    if tags is None:
        return ()
    if isinstance(tags, str):
        return (tags,)
    return tuple(str(t) for t in tags)


def mean_spearman_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Mean over dates of the row-wise Spearman correlation of two panels.

    The panels are aligned on the *intersection* of their dates and assets
    (a train-window factor can be compared with a full-history one); dates on
    which the correlation is undefined (fewer than five joint pairs, or a
    constant row) are ignored.

    Returns:
        The mean correlation, or ``NaN`` when there is no overlap at all.
    """
    dates = a.index.intersection(b.index)
    assets = a.columns.intersection(b.columns)
    if len(dates) == 0 or len(assets) == 0:
        return float("nan")
    a2 = a.reindex(index=dates, columns=assets)
    b2 = b.reindex(index=dates, columns=assets)
    corr = rowwise_corr(a2, b2, method="spearman").to_numpy(dtype=float)
    corr = corr[np.isfinite(corr)]
    return float(corr.mean()) if corr.size else float("nan")


def _json_safe(obj: Any) -> Any:
    """Recursively convert to plain JSON types (``NaN``/``inf`` -> ``None``,
    ``Timestamp`` -> ISO string, numpy scalars -> Python scalars)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset, np.ndarray)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _report_to_dict(report: Any) -> dict[str, Any] | None:
    """Scalar dict of a report (``FactorReport`` or an already-loaded dict)."""
    if report is None:
        return None
    if hasattr(report, "to_dict"):
        return dict(report.to_dict())
    if isinstance(report, dict):
        return dict(report)
    raise TypeError(f"report must be a FactorReport, a dict or None, got {type(report).__name__}")


_REPORT_INT_FIELDS = ("horizon", "lag", "n_quantiles")
_REPORT_TS_FIELDS = ("start", "end")
_REPORT_STR_FIELDS = ("name", "expr")


def _report_from_dict(d: dict[str, Any] | None, name: str) -> FactorReport | dict[str, Any] | None:
    """Rebuild a :class:`FactorReport` from its JSON scalar dict.

    Only the scalar fields are persisted, so the series / frame
    fields of the rebuilt report are *empty*.  If the dict cannot populate the
    dataclass (e.g. written by a different version) the plain dict is returned
    instead, with a warning.
    """
    if d is None:
        return None
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(FactorReport):
        if f.name in FactorReport._SERIES_FIELDS:
            continue
        v = d.get(f.name)
        if f.name in _REPORT_STR_FIELDS:
            kwargs[f.name] = name if (f.name == "name" and v is None) else v
        elif f.name in _REPORT_TS_FIELDS:
            kwargs[f.name] = pd.Timestamp(v) if v is not None else pd.NaT
        elif f.name in _REPORT_INT_FIELDS:
            kwargs[f.name] = int(v) if v is not None else 0
        else:
            kwargs[f.name] = float(v) if v is not None else float("nan")
    try:
        return FactorReport(
            ic_series=pd.Series(dtype=float, name="ic"),
            rank_ic_series=pd.Series(dtype=float, name="ic"),
            quantile_returns=pd.DataFrame(),
            ls_series=pd.Series(dtype=float, name="long_short"),
            ic_decay=pd.Series(dtype=float, index=pd.Index([], name="horizon"), name="ic"),
            **kwargs,
        )
    except TypeError as exc:  # pragma: no cover - defensive against schema drift
        logger.warning("could not rebuild FactorReport for %r (%s); keeping the raw dict", name, exc)
        return dict(d)


# ----------------------------------------------------------------------------
# FactorLibrary
# ----------------------------------------------------------------------------
class FactorLibrary:
    """Registry of named factors with correlation-based de-duplication.

    Args:
        root: Optional directory used by :meth:`save` when no explicit root is
            given.  Nothing is read from it at construction - use
            :meth:`load` for that.
        max_corr: Rejection threshold: :meth:`add` refuses a factor whose
            ``|mean row-wise Spearman corr|`` with any incumbent is *strictly
            greater* than this (unless ``force``).  Stored as
            :attr:`max_corr_threshold` because :meth:`max_corr` is a method.

    Entries are kept in insertion order; ``len(lib)``, ``name in lib`` and
    iteration over names are supported.

    By construction, each factor in the library is novel relative to the
    factors that the library held before it.
    """

    def __init__(self, root: str | Path | None = None, max_corr: float = 0.7) -> None:
        max_corr = float(max_corr)
        if not (0.0 <= max_corr <= 1.0):
            raise ValueError(f"max_corr must be in [0, 1], got {max_corr}")
        self.root: Path | None = Path(root) if root is not None else None
        self.max_corr_threshold: float = max_corr
        self._entries: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------ dunder
    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._entries))

    def __repr__(self) -> str:
        return (
            f"FactorLibrary(n={len(self._entries)}, max_corr={self.max_corr_threshold}, "
            f"root={str(self.root) if self.root else None}, names={self.names()})"
        )

    # ------------------------------------------------------------ mutation
    def add(
        self,
        name: str,
        expr: str,
        values: pd.DataFrame,
        report: FactorReport | None = None,
        tags: tuple[str, ...] = (),
        force: bool = False,
    ) -> bool:
        """Register a factor unless it duplicates an incumbent.

        Args:
            name: Unique factor name (also the parquet file stem on save).
            expr: Expression string that produced the factor (free text).
            values: Factor value panel (copied as ``float64``).
            report: Optional :class:`FactorReport` (or a scalar dict, e.g. one
                that came back from :meth:`load`).
            tags: Free-form labels, e.g. ``("seed", "reversal")``.
            force: Skip the correlation screen, and allow replacing an existing
                entry of the same name.

        Returns:
            ``True`` if the factor was stored; ``False`` if it was rejected
            because ``|corr|`` with some incumbent exceeds
            :attr:`max_corr_threshold` (nothing is changed in that case).

        Raises:
            ValueError: on an invalid name, or a duplicate name without ``force``.
            TypeError: if ``values`` is not a DataFrame.
        """
        _check_name(name)
        vals = _as_panel(values, name)
        if name in self._entries and not force:
            raise ValueError(f"factor {name!r} already in library (pass force=True to replace it)")
        if report is not None:
            _report_to_dict(report)  # type check only
        other, corr = self.max_corr(vals, exclude=(name,))
        if not force and other is not None and abs(corr) > self.max_corr_threshold:
            logger.info(
                "library: rejected %r - |corr| %.3f with %r exceeds max_corr %.2f",
                name,
                corr,
                other,
                self.max_corr_threshold,
            )
            return False
        self._entries[name] = {
            "expr": str(expr) if expr is not None else "",
            "values": vals,
            "report": report,
            "tags": _as_tags(tags),
        }
        logger.info(
            "library: added %r (%d x %d, max |corr| %.3f with %s, force=%s) -> %d factors",
            name,
            vals.shape[0],
            vals.shape[1],
            abs(corr) if other is not None else 0.0,
            other,
            force,
            len(self._entries),
        )
        return True

    def remove(self, name: str) -> None:
        """Delete a factor (``KeyError`` if unknown)."""
        if name not in self._entries:
            raise KeyError(f"unknown factor {name!r}; library has {self.names()}")
        del self._entries[name]
        logger.info("library: removed %r -> %d factors", name, len(self._entries))

    # ------------------------------------------------------------ queries
    def max_corr(self, values: pd.DataFrame, *, exclude: tuple[str, ...] = ()) -> tuple[str | None, float]:
        """Most-correlated incumbent of a candidate panel.

        Correlation is the mean over overlapping dates of the row-wise Spearman
        correlation (:func:`~alphaforge.evaluation.ic.rowwise_corr`), assets and
        dates aligned on their intersection.

        Args:
            values: Candidate panel.
            exclude: Names to leave out of the comparison (used when replacing).

        Returns:
            ``(name, corr)`` of the incumbent with the largest ``|corr|`` -
            ``corr`` keeps its sign.  ``(None, 0.0)`` for an empty library;
            ``(None, nan)`` if no incumbent overlaps the candidate at all.
        """
        if not isinstance(values, pd.DataFrame):
            raise TypeError("values must be a pandas DataFrame")
        best_name: str | None = None
        best_corr = float("nan")
        compared = 0
        for other, entry in self._entries.items():
            if other in exclude:
                continue
            compared += 1
            c = mean_spearman_corr(values, entry["values"])
            if not math.isfinite(c):
                continue
            if best_name is None or abs(c) > abs(best_corr):
                best_name, best_corr = other, c
        if compared == 0:
            return None, 0.0
        if best_name is None:
            logger.warning("library: candidate shares no valid dates/assets with any of %d incumbents", compared)
        return best_name, best_corr

    def names(self) -> list[str]:
        """Factor names in insertion order."""
        return list(self._entries)

    def get(self, name: str) -> dict[str, Any]:
        """``{"expr", "values", "report", "tags"}`` for a factor (``KeyError`` if unknown).

        ``values`` is the library's own panel (not a copy) - treat it as read-only.
        """
        try:
            e = self._entries[name]
        except KeyError:
            raise KeyError(f"unknown factor {name!r}; library has {self.names()}") from None
        return {"expr": e["expr"], "values": e["values"], "report": e["report"], "tags": e["tags"]}

    def values(self, name: str) -> pd.DataFrame:
        """Value panel of a factor (read-only view)."""
        return self.get(name)["values"]

    def all_values(self) -> dict[str, pd.DataFrame]:
        """``{name: values}`` for every factor, in insertion order."""
        return {name: e["values"] for name, e in self._entries.items()}

    def to_frame(self) -> pd.DataFrame:
        """One row per factor (index ``name``): ``expr``, ``tags`` and the report scalars.

        ``tags`` is a comma-joined string so the frame is CSV-friendly.  Report
        scalar columns follow :meth:`FactorReport.scalar_fields` order; factors
        without a report have ``NaN`` there.
        """
        rows: list[dict[str, Any]] = []
        for name, e in self._entries.items():
            row: dict[str, Any] = {"name": name, "expr": e["expr"], "tags": ",".join(e["tags"])}
            d = _report_to_dict(e["report"])
            if d:
                d.pop("name", None)  # the library's own name / expr are authoritative
                d.pop("expr", None)
                row.update(d)
            rows.append(row)
        base_cols = ["expr", "tags"]
        if not rows:
            return pd.DataFrame(columns=base_cols, index=pd.Index([], name="name"))
        df = pd.DataFrame(rows).set_index("name")
        report_cols = [c for c in FactorReport.scalar_fields() if c not in ("name", *base_cols) and c in df.columns]
        rest = [c for c in df.columns if c not in base_cols and c not in report_cols]
        return df[base_cols + report_cols + rest]

    def correlation_matrix(self) -> pd.DataFrame:
        """Symmetric ``names x names`` matrix of mean row-wise Spearman correlations (diag 1).

        Pairs without any overlapping valid dates are ``NaN``.

        A cluster in this matrix is a group of factors that act as one signal.
        """
        names = self.names()
        n = len(names)
        mat = np.eye(n, dtype=np.float64)
        for i in range(n):
            a = self._entries[names[i]]["values"]
            for j in range(i + 1, n):
                c = mean_spearman_corr(a, self._entries[names[j]]["values"])
                mat[i, j] = mat[j, i] = c
        return pd.DataFrame(mat, index=pd.Index(names, name="name"), columns=pd.Index(names, name="name"))

    # ------------------------------------------------------------ persistence
    def save(self, root: str | Path | None = None) -> None:
        """Write ``<root>/<name>.parquet`` per factor and ``<root>/library.json``.

        ``library.json`` holds, per factor, ``expr``, ``tags`` and the report
        scalars (``FactorReport.to_dict()``, made JSON-safe: ``NaN`` -> ``null``,
        timestamps -> ISO strings) plus the library's ``max_corr``.  Series
        fields of the reports are **not** persisted.  ``root``
        defaults to the constructor's root and becomes the library's root
        afterwards.  Parquet files of factors removed since an earlier save are
        left in place but are not referenced by the manifest.
        """
        root = Path(root) if root is not None else self.root
        if root is None:
            raise ValueError("no root given: pass save(root) or construct FactorLibrary(root=...)")
        root.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {}
        for name, e in self._entries.items():
            rel = f"{name}.parquet"
            vals = e["values"]
            if vals.index.name != _DATE_NAME and isinstance(vals.index, pd.DatetimeIndex):
                vals = vals.rename_axis(index=_DATE_NAME)
            vals.to_parquet(root / rel)
            manifest[name] = {
                "expr": e["expr"],
                "tags": list(e["tags"]),
                "report": _json_safe(_report_to_dict(e["report"])),
                "file": rel,
            }
        payload = {
            "format_version": _FORMAT_VERSION,
            "max_corr": self.max_corr_threshold,
            "n_factors": len(manifest),
            "factors": manifest,
        }
        (root / _LIBRARY_JSON).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        self.root = root
        logger.info("library: saved %d factors to %s", len(manifest), root)

    @classmethod
    def load(cls, root: str | Path) -> FactorLibrary:
        """Load a library written by :meth:`save`.

        Reports come back as :class:`FactorReport` objects whose scalar fields
        are restored and whose series / frame fields are empty.
        """
        root = Path(root)
        path = root / _LIBRARY_JSON
        if not path.exists():
            raise FileNotFoundError(f"no {_LIBRARY_JSON} under {root}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = int(payload.get("format_version", 0))
        if version > _FORMAT_VERSION:
            logger.warning("library format version %d is newer than supported %d", version, _FORMAT_VERSION)
        lib = cls(root=root, max_corr=float(payload.get("max_corr", 0.7)))
        for name, meta in payload.get("factors", {}).items():
            vals = pd.read_parquet(root / meta.get("file", f"{name}.parquet"))
            if not isinstance(vals.index, pd.DatetimeIndex):
                with contextlib.suppress(TypeError, ValueError):
                    vals.index = pd.DatetimeIndex(vals.index)
            vals = _as_panel(vals, name)
            if isinstance(vals.index, pd.DatetimeIndex):
                vals.index.name = _DATE_NAME
            lib._entries[name] = {
                "expr": str(meta.get("expr", "")),
                "values": vals,
                "report": _report_from_dict(meta.get("report"), name),
                "tags": _as_tags(meta.get("tags", ())),
            }
        logger.info("library: loaded %d factors from %s", len(lib), root)
        return lib
