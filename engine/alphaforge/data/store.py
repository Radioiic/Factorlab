"""Point-in-time panel store for AlphaForge.

A :class:`DataStore` holds a set of *aligned* panels (``pd.DataFrame`` with a
``DatetimeIndex`` named ``"date"`` as rows and asset ids as columns),
produces forward-return labels, slices date ranges, and round-trips to disk.

All fields, factors and labels share one ``dates x assets`` grid, and a panel that does
not align fails at ``add`` time.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("alphaforge.data.store")

__all__ = ["DataStore"]

_FORMAT_VERSION = 1
_META_JSON = "meta.json"
_FIELDS_DIR = "fields"
_META_DIR = "meta"
_DATE_NAME = "date"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _normalise_dates(index: Any) -> pd.DatetimeIndex:
    """Coerce ``index`` to a sorted, unique, freq-less ``DatetimeIndex`` named ``"date"``."""
    idx = pd.DatetimeIndex(index)
    if idx.has_duplicates:
        raise ValueError("DataStore dates must be unique")
    if not idx.is_monotonic_increasing:
        idx = idx.sort_values()
    # Drop any inferred freq so freshly simulated and re-loaded stores compare equal.
    return pd.DatetimeIndex(np.asarray(idx), name=_DATE_NAME)


def _normalise_assets(columns: Any) -> pd.Index:
    """Coerce ``columns`` to a unique ``Index`` of ``str`` asset ids (order preserved)."""
    idx = pd.Index([str(c) for c in columns], dtype=object)
    if idx.has_duplicates:
        raise ValueError("DataStore assets must be unique")
    return idx


def _align_panel(panel: pd.DataFrame, dates: pd.DatetimeIndex, assets: pd.Index) -> pd.DataFrame:
    """Return ``panel`` reindexed to ``dates x assets`` as float64 without mutating the input."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"panel must be a pandas DataFrame, got {type(panel).__name__}")
    df = panel.copy(deep=False)  # new container, shared data: index/columns edits stay local
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    if df.index.has_duplicates or df.columns.has_duplicates:
        raise ValueError("panel index/columns must be unique")
    if not all(isinstance(c, str) for c in df.columns):
        df.columns = pd.Index([str(c) for c in df.columns], dtype=object)
    if not (df.index.equals(dates) and df.columns.equals(assets)):
        df = df.reindex(index=dates, columns=assets)
    if not (df.dtypes == np.dtype("float64")).all():
        df = df.astype("float64")
    # Share the store's exact index/columns objects (cheap, and guarantees identity).
    df.index = dates
    df.columns = assets
    return df


def _json_default(o: Any) -> Any:
    """``json.dumps`` fallback for numpy / pandas / dataclass scalars."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, pd.Timedelta)):
        return str(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return list(o)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def _is_panel_like(v: Any) -> bool:
    return isinstance(v, (pd.DataFrame, pd.Series))


def _axis_kind(idx: pd.Index) -> str:
    """Manifest label for an axis dtype: ``"object"`` or pandas' own dtype string."""
    return "object" if idx.dtype == object else str(idx.dtype)


def _restore_axis(idx: pd.Index, kind: str | None) -> pd.Index:
    """Undo parquet's string-axis conversion for an axis saved as ``object``.

    Why: parquet stores an axis of Python ``str`` as a string column, and pandas >= 3
    reads that back as its ``str`` dtype, not ``object``.  The store keeps asset ids as
    ``object`` (see :func:`_normalise_assets`), so without this a save/load roundtrip
    silently changes the axis dtype of asset-indexed metadata (``beta``, ``shares``,
    the asset columns of ``true_alphas`` ...) while every value stays identical.
    ``kind`` is the dtype recorded at save time; ``None`` means a manifest without
    recorded dtypes; then a string axis is taken to be an asset axis and restored to
    ``object``, the store's convention.  An axis saved with the ``str``
    dtype is recorded as such and left alone, so the roundtrip is strict both ways.
    """
    if isinstance(idx, pd.MultiIndex) or not isinstance(idx.dtype, pd.StringDtype):
        return idx
    if kind is None or kind == "object":
        return pd.Index(idx.to_numpy(dtype=object), dtype=object, name=idx.name)
    return idx


def _write_panel(root: Path, rel: str, obj: pd.DataFrame | pd.Series) -> dict:
    """Write a DataFrame/Series to ``root/rel`` (parquet) and return its manifest entry.

    The entry records each axis dtype (``axes``) so :func:`_read_panel` can restore it;
    an entry without ``axes`` also loads (see :func:`_restore_axis`).
    """
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.Series):
        obj.to_frame(name="value").to_parquet(path)
        return {"kind": "series", "path": rel, "name": obj.name, "axes": {"index": _axis_kind(obj.index)}}
    df = obj
    if not all(isinstance(c, str) for c in df.columns):
        df = df.copy(deep=False)
        df.columns = pd.Index([str(c) for c in df.columns], dtype=object)
    df.to_parquet(path)
    return {"kind": "frame", "path": rel, "axes": {"index": _axis_kind(df.index), "columns": _axis_kind(df.columns)}}


def _read_panel(root: Path, entry: Mapping[str, Any]) -> pd.DataFrame | pd.Series:
    df = pd.read_parquet(root / entry["path"])
    axes = entry.get("axes") or {}
    df.index = _restore_axis(df.index, axes.get("index"))
    if entry["kind"] == "series":
        s = df.iloc[:, 0]
        s.name = entry.get("name")
        return s
    df.columns = _restore_axis(df.columns, axes.get("columns"))
    return df


# --------------------------------------------------------------------------------------
# DataStore
# --------------------------------------------------------------------------------------
class DataStore:
    """Container of aligned ``dates x assets`` panels plus free-form metadata.

    Args:
        fields: mapping ``name -> panel``.  Every panel is reindexed to the store grid.
            Unless ``dates``/``assets`` are given, the grid is taken from the *first*
            panel in the mapping (so later panels are aligned to it, exactly like
            :meth:`add`).
        meta: optional metadata dict.  Values may be JSON-serialisable scalars/dicts,
            ``DataFrame``/``Series`` objects, or dicts of those (all survive
            :meth:`save` / :meth:`load`).
        dates: optional explicit date grid (keyword-only).
        assets: optional explicit asset grid (keyword-only).

    Notes:
        :meth:`get` returns the store's *internal* frame (no copy) for speed; callers
        must treat it as read-only, consistent with the rule that operators never mutate
        their inputs.
    """

    def __init__(
        self,
        fields: Mapping[str, pd.DataFrame],
        meta: dict | None = None,
        *,
        dates: Any | None = None,
        assets: Any | None = None,
    ) -> None:
        self._fields: dict[str, pd.DataFrame] = {}
        self._fwd_cache: dict[tuple[int, int], pd.DataFrame] = {}
        self.meta: dict = dict(meta) if meta else {}

        if dates is None or assets is None:
            if fields:
                first = next(iter(fields.values()))
                if not isinstance(first, pd.DataFrame):
                    raise TypeError("fields values must be pandas DataFrames")
                ref_dates, ref_assets = first.index, first.columns
            else:
                ref_dates, ref_assets = pd.DatetimeIndex([]), pd.Index([], dtype=object)
            dates = ref_dates if dates is None else dates
            assets = ref_assets if assets is None else assets
        self._dates = _normalise_dates(dates)
        self._assets = _normalise_assets(assets)

        for name, panel in fields.items():
            self.add(name, panel)

    # ---------------------------------------------------------------- properties
    @property
    def dates(self) -> pd.DatetimeIndex:
        """Sorted, unique ``DatetimeIndex`` named ``"date"`` shared by every panel."""
        return self._dates

    @property
    def assets(self) -> pd.Index:
        """Asset ids (``str``) shared by every panel."""
        return self._assets

    @property
    def fields(self) -> list[str]:
        """Sorted list of field names."""
        return sorted(self._fields)

    # ---------------------------------------------------------------- access
    def get(self, name: str) -> pd.DataFrame:
        """Return the aligned panel for ``name``.

        Raises:
            KeyError: if the field does not exist.
        """
        try:
            return self._fields[name]
        except KeyError:
            raise KeyError(f"Unknown field {name!r}; available fields: {self.fields}") from None

    def has(self, name: str) -> bool:
        """Whether a field named ``name`` exists."""
        return name in self._fields

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._fields

    def add(self, name: str, panel: pd.DataFrame, overwrite: bool = False) -> None:
        """Add (or replace) a field, reindexing it to ``dates x assets`` as float64.

        Args:
            name: field name (non-empty, no path separators).
            panel: any DataFrame; rows/columns missing from the grid become NaN,
                extra rows/columns are dropped.
            overwrite: allow replacing an existing field.

        Raises:
            ValueError: if the field exists and ``overwrite`` is False.
        """
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise ValueError(f"invalid field name {name!r}")
        if name in self._fields and not overwrite:
            raise ValueError(f"field {name!r} already exists (pass overwrite=True to replace)")
        self._fields[name] = _align_panel(panel, self._dates, self._assets)
        if name == "close":
            self._fwd_cache.clear()  # labels depend on close
        logger.debug("added field %r (%d x %d)", name, len(self._dates), len(self._assets))

    # ---------------------------------------------------------------- labels
    def fwd_return(self, horizon: int = 5, lag: int = 1) -> pd.DataFrame:
        """Forward simple return label, cached per ``(horizon, lag)``.

        ``fwd = close.shift(-(lag + horizon)) / close.shift(-lag) - 1``: a signal computed at
        the close of ``t`` is traded at the close of ``t + lag`` and held ``horizon`` days.
        The last ``lag + horizon`` rows are NaN (unknown future).

        With the default ``lag=1``, the label starts at the close after the signal close, so
        no fill occurs at the signal close.

        Args:
            horizon: holding period in trading days (>= 1).
            lag: entry delay in trading days (>= 0).

        Returns:
            Panel aligned to the store (internal cached object; treat as read-only).
        """
        horizon = int(horizon)
        lag = int(lag)
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if lag < 0:
            raise ValueError("lag must be >= 0")
        key = (horizon, lag)
        cached = self._fwd_cache.get(key)
        if cached is None:
            close = self.get("close")
            with np.errstate(divide="ignore", invalid="ignore"):
                cached = close.shift(-(lag + horizon)) / close.shift(-lag) - 1.0
            cached = cached.replace([np.inf, -np.inf], np.nan)
            self._fwd_cache[key] = cached
        return cached

    # ---------------------------------------------------------------- slicing
    def slice(self, start: Any = None, end: Any = None) -> DataStore:
        """Return a new store restricted to dates in ``[start, end]`` (labels inclusive).

        ``start``/``end`` accept anything ``DataFrame.loc`` accepts on a ``DatetimeIndex``
        (``Timestamp``, ``"2019-06-30"``, partial strings such as ``"2019"``, or ``None``).
        Assets are unchanged.  ``meta`` is shallow-copied; metadata panels (or dicts of
        panels) whose index equals the store's dates are sliced to the same range so they
        stay aligned.  The forward-return cache is *not* carried over: labels on a sliced
        store are recomputed from the sliced ``close`` and are therefore NaN at the tail,
        which is the strict "the future is unknown" reading required for train/test splits.
        """
        indexer = self._dates.slice_indexer(start, end)
        new_dates = self._dates[indexer]
        new_fields = {name: df.iloc[indexer] for name, df in self._fields.items()}

        def _slice_value(v: Any) -> Any:
            if _is_panel_like(v) and isinstance(v.index, pd.DatetimeIndex) and v.index.equals(self._dates):
                return v.iloc[indexer]
            if isinstance(v, dict):
                return {k: _slice_value(x) for k, x in v.items()}
            return v

        new_meta = {k: _slice_value(v) for k, v in self.meta.items()}
        return DataStore(new_fields, new_meta, dates=new_dates, assets=self._assets)

    # ---------------------------------------------------------------- persistence
    def save(self, root: str | Path) -> None:
        """Persist the store: ``fields/<name>.parquet``, ``meta.json`` and ``meta/*.parquet``.

        JSON-serialisable metadata goes into ``meta.json``; ``DataFrame``/``Series`` values
        (and dicts of them) are written as parquet under ``meta/`` and re-assembled by
        :meth:`load`.  Anything else is stored as its ``repr`` with a warning.
        """
        root = Path(root)
        fields_dir = root / _FIELDS_DIR
        fields_dir.mkdir(parents=True, exist_ok=True)
        for name, df in self._fields.items():
            df.to_parquet(fields_dir / f"{name}.parquet")

        json_meta: dict[str, Any] = {}
        manifest: dict[str, Any] = {}
        for key, val in self.meta.items():
            key = str(key)
            if _is_panel_like(val):
                manifest[key] = _write_panel(root, f"{_META_DIR}/{key}.parquet", val)
            elif isinstance(val, dict) and any(_is_panel_like(v) for v in val.values()):
                panels: dict[str, Any] = {}
                plain: dict[str, Any] = {}
                for sub, v in val.items():
                    sub = str(sub)
                    if _is_panel_like(v):
                        panels[sub] = _write_panel(root, f"{_META_DIR}/{key}/{sub}.parquet", v)
                    else:
                        plain[sub] = self._jsonable(f"{key}.{sub}", v)
                manifest[key] = {"kind": "dict", "panels": panels, "json": plain}
            else:
                json_meta[key] = self._jsonable(key, val)

        payload = {
            "format_version": _FORMAT_VERSION,
            "fields": self.fields,
            "n_dates": int(len(self._dates)),
            "n_assets": int(len(self._assets)),
            "start": str(self._dates[0]) if len(self._dates) else None,
            "end": str(self._dates[-1]) if len(self._dates) else None,
            "meta": json_meta,
            "meta_panels": manifest,
        }
        (root / _META_JSON).write_text(json.dumps(payload, indent=2, default=_json_default))
        logger.info("saved DataStore (%d fields) to %s", len(self._fields), root)

    @staticmethod
    def _jsonable(key: str, val: Any) -> Any:
        """Return ``val`` if JSON-serialisable (via ``_json_default``), else its repr."""
        try:
            json.dumps(val, default=_json_default)
            return val
        except (TypeError, ValueError):
            logger.warning("meta[%r] of type %s is not JSON-serialisable; storing repr()", key, type(val).__name__)
            return repr(val)

    @classmethod
    def load(cls, root: str | Path) -> DataStore:
        """Load a store written by :meth:`save`."""
        root = Path(root)
        payload = json.loads((root / _META_JSON).read_text())
        version = payload.get("format_version", 0)
        if version > _FORMAT_VERSION:
            logger.warning("DataStore format version %s is newer than supported %s", version, _FORMAT_VERSION)
        fields = {name: pd.read_parquet(root / _FIELDS_DIR / f"{name}.parquet") for name in payload["fields"]}
        meta: dict[str, Any] = dict(payload.get("meta", {}))
        for key, entry in payload.get("meta_panels", {}).items():
            if entry["kind"] == "dict":
                d: dict[str, Any] = {sub: _read_panel(root, e) for sub, e in entry["panels"].items()}
                d.update(entry.get("json", {}))
                meta[key] = d
            else:
                meta[key] = _read_panel(root, entry)
        store = cls(fields, meta)
        logger.info("loaded DataStore (%d fields) from %s", len(fields), root)
        return store

    # ---------------------------------------------------------------- repr
    def __repr__(self) -> str:
        span = f"{self._dates[0].date()}..{self._dates[-1].date()}" if len(self._dates) else "empty"
        return (
            f"DataStore(dates={len(self._dates)} [{span}], assets={len(self._assets)}, "
            f"fields={self.fields}, meta_keys={sorted(map(str, self.meta))})"
        )
