"""The novelty index: every existing factor's G-zone values, and a candidate's largest
|correlation| with any of them.

LABEL-FREE. Nothing here reads a label, a screen result or a library test result:
    * tables are read only through read_columns(), which takes an explicit column list,
      refuses any fwd_* (label) column with LabelColumnError, filters rows to dates <= the
      G cutoff (the 2024-06-09 Sunday row) and raises ZoneLeak if a later row comes back;
    * the library's values come from the screen's value shards (data/library_on_engine/
      values/), read the same way (the shard rows are Sundays), and the definitions are
      chosen from status.parquet's label-free columns only, read with an explicit
      projection (STATUS_COLUMNS): never screen or g_weeks, which the screen derives
      from labels, and never a field of the later PIT cuts (pit_check and every *_t2 or
      *_t3 column), which the screen derives from values dated after the cutoff;
    * every other member is computed in memory from the perp panels cut at the G cutoff
      (fieldmap.load_inputs(): raw bars and funding are cut as soon as they are read, so no
      later value is used), or read with read_columns() (rows up to the cutoff only).
The index is BUILT IN MEMORY and never written. Its content hash, pool_hash, identifies it.

WHICH LIBRARY DEFINITIONS JOIN (select_library): compute "ok" (the definition evaluated
without an error and its values were stored; the other states are "not_run", a field the
engine's data lacks, and "error", which no definition has), not a vwap proxy (decided by
the field names), and PROVEN POINT IN TIME AT THE FIRST CUT, 2024-06-09, the G cutoff
itself: pit_error_t1 empty, pit_diff_cells_t1 == 0 and pit_finite_t1 > 0. The screen's T1
check recomputes each definition on bars cut at 2024-06-09 and compares it with the stored
values on every Sunday up to that row, so a member's stored G values are equal (rtol 1e-9,
atol 1e-12) to values computed from data through the cutoff alone. No field of a later
cut is read: the screen's overall pit_check also needs the cuts T2 and T3
(libfactors.PIT_CUTS), so a definition equal at T1 and failing only at T2 or T3 would
still join (test_finder_novelty.py proves that mutating the T2/T3 fields and pit_check
leaves the membership unchanged). Definitions without a T1 proof are listed in
`dropped`: "no_g_values" when pit_finite_t1 is 0 (no finite value on any Sunday through
the cutoff, which covers the OKX open-interest definitions: OKX open interest starts
after G), "pit_fail_t1" otherwise.

WHAT IS IN IT (values on the 114 G Sundays, G_fit then G_eval, 20 perps):
    library   every library definition select_library() admits that has a finite value
              on a G Sunday
    user      every registered test of the trial ledger (data/trials/) that is not
              withdrawn, as "user:FT-0001": a formula is evaluated on the panels of its
              source cut at the G cutoff (the OKX perp panels above, or the Binance panel
              built with through = the cutoff); a values file gives the history values of
              its registration line on the G Sundays. A formula of a source whose data is
              not on disk is dropped ("no_panel")
    zoo       every AlphaForge formula, active or not (af_01, ...), read-only from
              data/research/alphaforge_daily.parquet (G rows), when the file exists.
              test_finder_novelty.py proves the table's G rows bit-identical to
              factor_values on bars cut at the G cutoff, so they are point in time like
              the library members
    proxy     the registered rIC regressors as factors: pit_symbol_tilt, btc_beta_90d
              and size_proxy (libfactors.regressors, the screen's own code). A tilt
              learned on G_fit is not a member: it is a per-coin mean of returns after
              each Sunday, a label statistic in all but name
A member's cells are its finite values on rows present in the label table (the eligible
rows; only the date and symbol columns are read). A Sunday with fewer than 15
such names ([g0] min_names_per_sunday) is empty for that member; a member with fewer than
20 non-empty Sundays, or with no variation on any of them, is dropped (listed in
`dropped`, with the reason). Values are tie-snapped first (libfactors.snap_ties), so
floating-point noise never orders names.

THE STATISTIC: pooled within-Sunday rank correlation on common cells. For a candidate c
and a member j, each Sunday t with at least 15 names finite for both (the common set S)
contributes the centred average ranks x (of c) and y (of j) taken over S; the correlation
is sum x*y / sqrt(sum x^2 * sum y^2) over all such Sundays, defined when there are at
least 20 of them (otherwise the pair has no value and cannot make a candidate a
duplicate). Novelty is max |corr| over the members (a sign flip is the same signal).

EXACT FOR EVERY MEMBER (max_corr, the default): exact() measures the correlation with
every member of the index. A member's ranks over the common set S are its own ranks minus
a correction from the few names it has and the candidate lacks (at most m - 15 = 5 on a
counted Sunday): centred rank over S of name i = own centred rank - 0.5 * sum over those
names k of sign(v_i - v_k). Ranks are multiples of 0.5, so the work is done on doubled
ranks in small integers, and every sum, in units of 1/4, is an integer below 2^24 (exact
in float32; float64 when the index is larger); N, Dx and Dy are therefore exact and the
result does not depend on the order of summation (test_finder_novelty.py checks it
against brute_force() on fixtures).
THE OPTIONAL PREFILTER (max_corr(exhaustive=False), bounds() and scan()), kept for callers
that only need the values that can reach a threshold:
    1. prefilter (float32, one batched matrix product per mask group): for every member an
       UPPER BOUND B >= |corr|. On a Sunday where the member has no name the candidate
       lacks, the member's own ranks restricted to S are its ranks over S, and the term is
       exact. Otherwise the member's ranks over S differ from its own ranks by at most e/2
       per name (e = names it has and the candidate lacks), which bounds both the
       cross-product and the member's spread; the candidate's ranks over S are always
       exact (they depend only on the masks). B = (|N| + slack) / sqrt(Dx * Dy_low).
    2. exact recheck: every member with B >= RECALL_FLOOR (0.45), plus the TOP_K (32)
       largest B.
    So every member with |corr| >= 0.45 is found and measured exactly (test_finder_
    novelty.py checks recall against brute force on fixtures), but below 0.45 that path's
    maximum is only a LOWER BOUND of the true maximum: when the masks differ (a permuted
    copy, a late listing) the bounds are loose and the 32 largest often miss the true
    maximum. The probe never uses it.
THE PERMUTATION NULL (null_max): the exact max |corr| (every member measured) for a copy of
the candidate whose values are moved by one fixed random permutation of the coins
eligible on every G Sunday (the same permutation on every Sunday, so the candidate's
persistence and its distribution are kept; the coins listed during G are left empty). It
is label-free: how large a maximum over the whole index is by chance for a factor like
this one. test_finder_novelty.py compares it with brute force on fixtures whose masks
differ.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from factorfinder import config_io as K
from factorfinder import fieldmap as FM

LABEL_PREFIX = "fwd_"
#: status.parquet columns the screen derives from labels; never read here
LABEL_DERIVED_STATUS = ("screen", "g_weeks")
#: status.parquet columns the screen derives from values dated after the G cutoff (the
#: later PIT cuts, and the overall verdict that combines them); never read here
LATER_CUT_STATUS = ("pit_check",)
LATER_CUT_SUFFIXES = ("_t2", "_t3")
STATUS_COLUMNS = (
    "factor_id",
    "family",
    "compute",
    "vwap_proxy",
    "pit_finite_t1",
    "pit_diff_cells_t1",
    "pit_error_t1",
    "values_shard",
)
MIN_SUNDAYS = 20  # the library screen's own minimum (libfactors.MIN_WEEKS)
RECALL_FLOOR = 0.45  # prefilter path: every |corr| >= this is found and measured exactly
TOP_K = 32  # prefilter path: members always rechecked exactly (largest bound)
BOUND_EPS = 1e-3  # float32 slack added to the stage-1 bound
SOURCES = ("library", "user", "zoo", "proxy")
ZOO_TABLE = "alphaforge_daily.parquet"  # in data/research/

#: every table read_columns() served in this process: (data-relative path, columns)
READS: list[tuple[str, tuple[str, ...]]] = []


class LabelColumnError(ValueError):
    """A fwd_* (label) column, a label-derived column or an unprojected read was requested."""


class ZoneLeak(ValueError):
    """A value dated after the G cutoff reached the index."""


# --------------------------------------------------------------------------- reading


def _label(path: Path) -> str:
    try:
        return "data/" + Path(path).resolve().relative_to(Path(C.DATA).resolve()).as_posix()
    except ValueError:
        return Path(path).name


def read_columns(path, columns, *, through: pd.Timestamp, date_col: str = "date") -> pd.DataFrame:
    """Rows dated <= `through` of the listed columns, and nothing else.

    Refuses (LabelColumnError) a missing column list and any fwd_* column, so a label can
    never be loaded by accident; raises ZoneLeak if the filtered table still holds a later
    date."""
    import pyarrow.parquet as pq

    if columns is None or isinstance(columns, str):
        raise LabelColumnError("read_columns needs an explicit column list")
    cols = [date_col] + [c for c in columns if c != date_col]
    bad = [c for c in cols if str(c).startswith(LABEL_PREFIX)]
    if bad:
        raise LabelColumnError(f"refusing to read label column(s) {bad} from {_label(path)}")
    t = pq.read_table(path, columns=cols, filters=[(date_col, "<=", through)])
    df = t.to_pandas()
    if len(df) and pd.Timestamp(df[date_col].max()) > through:
        raise ZoneLeak(f"{_label(path)}: a row after {through.date()} came back")
    READS.append((_label(path), tuple(cols)))
    return df


def _schema_columns(path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def g_sundays() -> pd.DatetimeIndex:
    """The 114 G Sundays (G_fit then G_eval), libfactors' own zone table."""
    import libfactors as LF

    days, _ = LF.g_sundays()
    return days


def eligibility(sundays: pd.DatetimeIndex, symbols: list, through: pd.Timestamp) -> np.ndarray:
    """(T, m) bool: the (Sunday, perp) row exists in the label table (only its date and symbol
    columns are read). FileNotFoundError without the label table."""
    import labels

    path = labels.path()
    if not path.is_file():
        raise FileNotFoundError("the novelty index needs the label table: run labels.py first")
    ft = read_columns(path, ["date", "symbol"], through=through)
    pres = ft.assign(_x=1.0).pivot(index="date", columns="symbol", values="_x")
    return pres.reindex(index=sundays, columns=symbols).notna().to_numpy()


def _wide(long: pd.DataFrame, col: str, sundays, symbols, date_col: str = "date") -> np.ndarray:
    w = long.pivot(index=date_col, columns="symbol", values=col)
    return w.reindex(index=sundays, columns=symbols).to_numpy(dtype=float)


# --------------------------------------------------------------------------- members


def _library_members(sundays, symbols, through, only=None) -> tuple[list, list, np.ndarray, list]:
    """(ids, families, values (k, T, m), dropped) of the screenable library definitions
    (only those in `only`, when given)."""
    import libfactors as LF

    root = LF.out_dir()
    if not (root / "status.parquet").is_file():
        raise FileNotFoundError("the novelty index needs the library's values: run libfactors.py --compute first")
    st = _status(root)
    if only is not None:
        st = st[st["factor_id"].isin(set(only))]
    keep, pre_dropped = select_library(st)
    ids, fams, vals = [], [], []
    for shard, g in keep.groupby("values_shard", sort=True):
        if not (root / LF.VALUES_NAME / shard).is_file():
            raise FileNotFoundError(f"value shard {shard} is missing: run libfactors.py --compute first")
        df = read_columns(
            root / LF.VALUES_NAME / shard, ["symbol"] + list(g["factor_id"]), through=through, date_col="sunday"
        )
        # shard rows are (sunday, symbol), Sunday-major; pick the G Sundays' blocks and
        # check that every block lists the perps in the index's order
        day = pd.DatetimeIndex(df["sunday"]).as_unit("ns")
        pos = {d: i for i, d in enumerate(day) if i == 0 or day[i - 1] != d}
        if any(d not in pos for d in sundays):
            raise KeyError(f"{shard}: a G Sunday is not stored")
        rows = np.concatenate([np.arange(pos[d], pos[d] + len(symbols)) for d in sundays])
        if df["symbol"].to_numpy()[rows].tolist() != list(symbols) * len(sundays):
            raise ValueError(f"{shard}: rows are not in (Sunday, perp) order")
        fid = list(g["factor_id"])
        bad = [f for f in fid if df[f].dtype != np.float64]
        if bad:
            raise ValueError(f"{bad[0]}: stored values are {df[bad[0]].dtype}, not float64")
        block = df[fid].to_numpy(dtype=np.float64)[rows].reshape(len(sundays), len(symbols), len(fid))
        ids += fid
        fams += [str(f) for f in g["family"]]
        vals.append(block.transpose(2, 0, 1))
    v = np.concatenate(vals) if vals else np.zeros((0, len(sundays), len(symbols)))
    has = np.isfinite(v).any(axis=(1, 2))
    dropped = pre_dropped + [(i, "no_g_values") for i, h in zip(ids, has) if not h]
    return [i for i, h in zip(ids, has) if h], [f for f, h in zip(fams, has) if h], v[has], dropped


def select_library(st: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """(rows that join the index, [(factor_id, reason)] of the computed, non-vwap-proxy rows
    that do not) from status.parquet's label-free, cutoff-bounded columns (the module
    docstring, "WHICH LIBRARY DEFINITIONS JOIN"). Reads no later-cut field."""
    base = st[(st["compute"] == "ok") & ~st["vwap_proxy"].astype(bool)]
    t1 = (
        (base["pit_error_t1"].fillna("").astype(str) == "")
        & (base["pit_diff_cells_t1"] == 0)
        & (base["pit_finite_t1"] > 0)
    )
    out = base[~t1]
    reason = np.where(
        (out["pit_error_t1"].fillna("").astype(str) == "")
        & (out["pit_diff_cells_t1"] == 0)
        & (out["pit_finite_t1"] == 0),
        "no_g_values",
        "pit_fail_t1",
    )
    return base[t1], list(zip(out["factor_id"].tolist(), reason.tolist()))


def _status(root: Path) -> pd.DataFrame:
    """status.parquet, the STATUS_COLUMNS projection only: no label-derived column (screen,
    g_weeks) and no field of a later PIT cut (pit_check, *_t2, *_t3) is ever read."""
    import pyarrow.parquet as pq

    cols = [c for c in STATUS_COLUMNS]
    if set(cols) & set(LABEL_DERIVED_STATUS):
        raise LabelColumnError("a label-derived status column was requested")
    later = [c for c in cols if c in LATER_CUT_STATUS or c.endswith(LATER_CUT_SUFFIXES)]
    if later:
        raise ZoneLeak(f"status columns {later} are derived from values dated after the G cutoff")
    df = pq.read_table(root / "status.parquet", columns=cols).to_pandas()
    READS.append((_label(root / "status.parquet"), tuple(cols)))
    return df


def _user_members(panels, sundays, symbols, through) -> tuple[list, np.ndarray, list]:
    """(ids, values (k, T, m), dropped) of the registered tests that are not withdrawn
    (trials.active). A formula: evaluated on the panels of its source cut at the G cutoff. A
    values file: the history values of its registration line."""
    import libfactors as LF
    import trials
    import userfactor as uf

    ids, vals, dropped = [], [], []
    binance: dict = {}
    for t in trials.active(trials.read()):
        tid = f"user:{t['test_id']}"
        if t["input"] == "formula":
            src = t["factor"]["source"]
            if src == "okx":
                p = panels
            else:
                if "p" not in binance:
                    try:
                        binance["p"] = LF.build_panels_binance(LF.load_raw_binance(), through=through)
                    except SystemExit:
                        binance["p"] = None
                p = binance["p"]
            if p is None:
                dropped.append((tid, "no_panel"))
                continue
            try:
                full = uf.full_values(t["factor"]["expression"], p)
            except uf.InputError:
                dropped.append((tid, "compute_error"))
                continue
            idx = pd.DatetimeIndex(p["close"].index)
            frame = pd.DataFrame(full, index=idx, columns=list(p["close"].columns))
        else:
            hist = t["factor"].get("history") or {}
            days = sorted(hist)
            frame = pd.DataFrame(
                [[np.nan if x is None else float(x) for x in hist[d]] for d in days],
                index=pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in days]),
                columns=symbols,
            )
        frame.index = frame.index.tz_localize("UTC") if frame.index.tz is None else frame.index.tz_convert("UTC")
        frame = frame[frame.index <= through]
        ids.append(tid)
        vals.append(frame.reindex(index=sundays, columns=symbols).to_numpy(dtype=np.float64))
    return ids, (np.stack(vals) if vals else np.zeros((0, len(sundays), len(symbols)))), dropped


def _zoo_members(sundays, symbols, through) -> tuple[list, np.ndarray]:
    """Every AlphaForge formula of data/research/alphaforge_daily.parquet (active or not); none
    without the table."""
    path = C.DATA / "research" / ZOO_TABLE
    if not path.is_file():
        return [], np.zeros((0, len(sundays), len(symbols)))
    cols = [c for c in _schema_columns(path) if c.startswith("af_") and c[3:].isdigit()]
    if not cols:
        return [], np.zeros((0, len(sundays), len(symbols)))
    df = read_columns(path, ["symbol"] + cols, through=through)
    return cols, np.stack([_wide(df, c, sundays, symbols) for c in cols])


def _proxy_members(panels, sundays, symbols) -> tuple[list, np.ndarray]:
    import libfactors as LF

    no_labels = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns, UTC]")})
    regs = LF.regressors(no_labels, panels, symbols, sundays, tilt="pit")
    names = {"tilt": "pit_symbol_tilt", "beta": "btc_beta_90d", "size": "size_proxy"}
    return [names[k] for k in LF.REG_KEYS], np.stack([regs[k] for k in LF.REG_KEYS])


# --------------------------------------------------------------------------- ranks


def centred_ranks(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Centred average ranks over the masked cells of each row (last axis); 0 elsewhere.

    rank - (n + 1) / 2, ties averaged; exact multiples of 0.5."""
    from scipy.stats import rankdata

    v = np.where(mask, values, np.inf)  # unmasked cells rank last, never among the masked
    r = rankdata(v, axis=-1, method="average")
    n = mask.sum(axis=-1, keepdims=True)
    return np.where(mask, r - (n + 1) / 2.0, 0.0)


def prepare(values: np.ndarray, el: np.ndarray, min_names: int) -> tuple[np.ndarray, np.ndarray]:
    """(mask, centred ranks) of (k, T, m) or (T, m) values: tie-snapped, finite and eligible,
    and Sundays with fewer than `min_names` such names emptied."""
    import libfactors as LF

    v = LF.snap_ties(np.asarray(values, dtype=np.float64))
    m = np.isfinite(v) & el
    m &= m.sum(axis=-1, keepdims=True) >= min_names
    return m, centred_ranks(np.where(m, v, 0.0), m)


@dataclass
class NoveltyIndex:
    """The index, in memory. Arrays are laid out (T, K, m) for batched products."""

    sundays: pd.DatetimeIndex
    symbols: list
    el: np.ndarray  # (T, m) eligible cells
    ids: list
    sources: list
    families: list
    ranks: np.ndarray  # (T, K, m) float32 centred own ranks, 0 outside the mask
    ranks2: np.ndarray  # ranks ** 2
    group: np.ndarray  # (K,) mask-group id; members of a group are contiguous
    gmask: np.ndarray  # (G, T, m) bool
    bounds: np.ndarray  # (G + 1,) member offsets of the groups
    min_names: int
    pool_hash: str
    dropped: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    through: pd.Timestamp | None = None

    @property
    def size(self) -> int:
        return len(self.ids)

    @cached_property
    def ranks_x2(self) -> np.ndarray:
        """(T, K, m) doubled centred ranks as small integers (|2 x rank| <= m - 1, so a
        difference of two is at most 2(m - 1)): int8 up to 64 coins, else int16. Derived
        from `ranks`, so not hashed."""
        m = len(self.symbols)
        dt = np.int8 if 2 * (m - 1) <= np.iinfo(np.int8).max else np.int16
        return np.ascontiguousarray(np.rint(2.0 * self.ranks).astype(dt))


def _hash(header: dict, *arrays: np.ndarray) -> str:
    h = hashlib.sha256(K.canonical_json(header).encode("ascii"))
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def build_index(
    *,
    library: bool = True,
    library_ids: list | None = None,
    inputs: dict | None = None,
    sources: tuple[str, ...] = SOURCES,
) -> NoveltyIndex:
    """Build the index in memory from the current config.DATA (read-only).

    `sources` names the parts (default: every source); `library_ids` limits the library part
    to those definitions (tests); `library=False` leaves it out."""
    bad = [s for s in sources if s not in SOURCES]
    if bad:
        raise ValueError(f"unknown novelty source(s) {bad}")
    srcs = [s for s in SOURCES if s in sources and (library or s != "library")]
    cfg = K.load()
    through = FM.g_cutoff(cfg)
    min_names = int(cfg["g0"]["min_names_per_sunday"])
    sundays = g_sundays()
    if sundays.max() > through:
        raise ZoneLeak("a G Sunday after the cutoff")
    symbols = FM.symbols()
    panels = None
    if {"user", "proxy"} & set(srcs):
        inputs = FM.load_inputs(through) if inputs is None else inputs
        if inputs["through"] > through:
            raise ZoneLeak("the panels run past the G cutoff")
        panels = FM.perp_panels(inputs)
    el = eligibility(sundays, symbols, through)

    parts: list[tuple[str, list, list, np.ndarray]] = []
    dropped: list = []
    for src in srcs:
        if src == "library":
            lid, lfam, lval, ldrop = _library_members(sundays, symbols, through, only=library_ids)
            parts.append(("library", lid, lfam, lval))
            dropped += ldrop
            continue
        if src == "user":
            ids, vals, udrop = _user_members(panels, sundays, symbols, through)
            dropped += udrop
        elif src == "zoo":
            ids, vals = _zoo_members(sundays, symbols, through)
        else:
            ids, vals = _proxy_members(panels, sundays, symbols)
        parts.append((src, ids, [""] * len(ids), vals))

    ids = [i for _, x, _, _ in parts for i in x]
    sources_ = [s for s, x, _, _ in parts for _ in x]
    fams = [f for _, _, x, _ in parts for f in x]
    stacks = [v for *_, v in parts if len(v)]
    vals = np.concatenate(stacks, axis=0) if stacks else np.zeros((0, len(sundays), len(symbols)))
    return assemble(
        vals,
        el,
        ids,
        sources_,
        fams,
        sundays=sundays,
        symbols=symbols,
        min_names=min_names,
        through=through,
        dropped=dropped,
    )


def assemble(
    values: np.ndarray,
    el: np.ndarray,
    ids: list,
    sources: list,
    families: list | None = None,
    *,
    sundays: pd.DatetimeIndex,
    symbols: list,
    min_names: int,
    through: pd.Timestamp,
    dropped: list | None = None,
) -> NoveltyIndex:
    """An index from member values (k, T, m) on `sundays` x `symbols` (build_index's last
    step; the tests build fixtures with it)."""
    if sundays.max() > through:
        raise ZoneLeak("a Sunday after the cutoff")
    families = [""] * len(ids) if families is None else list(families)
    dropped = list(dropped or [])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate member ids")
    mask, ranks = prepare(values, el[None], min_names)
    nsun = mask.any(axis=2).sum(axis=1)
    var = (ranks**2).sum(axis=(1, 2))
    keep = (nsun >= MIN_SUNDAYS) & (var > 0)
    for i in np.flatnonzero(~keep):
        dropped.append((ids[i], "too_few_sundays" if nsun[i] < MIN_SUNDAYS else "constant_per_date"))
    ids = [x for x, k in zip(ids, keep) if k]
    sources = [x for x, k in zip(sources, keep) if k]
    fams = [x for x, k in zip(families, keep) if k]
    mask, ranks = mask[keep], ranks[keep]

    # mask groups, members of a group contiguous (stable within a group: input order)
    flat = np.ascontiguousarray(np.packbits(mask.reshape(len(mask), -1), axis=1))
    rows = flat.view(np.dtype((np.void, flat.shape[1]))).reshape(-1)
    _, first, inv = np.unique(rows, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    uniq = mask.reshape(len(mask), -1)[first]
    order = np.argsort(inv, kind="stable")
    ids = [ids[i] for i in order]
    sources = [sources[i] for i in order]
    fams = [fams[i] for i in order]
    group = inv[order].astype(np.int32)
    bounds_ = np.searchsorted(group, np.arange(len(uniq) + 1)).astype(np.int64)
    gmask = uniq.reshape(len(uniq), len(sundays), len(symbols))
    r32 = np.ascontiguousarray(ranks[order].transpose(1, 0, 2)).astype(np.float32)
    header = {
        "kind": "finder_novelty_index",
        "through": str(through.date()),
        "sundays": [str(d.date()) for d in sundays],
        "symbols": list(symbols),
        "min_names": min_names,
        "min_sundays": MIN_SUNDAYS,
        "ties": "libfactors.snap_ties",
        "ids": ids,
        "sources": sources,
    }
    pool_hash = _hash(header, r32, gmask, group, el)
    counts = {s: sources.count(s) for s in SOURCES}
    counts["total"] = len(ids)
    counts["mask_groups"] = int(len(uniq))
    counts["dropped"] = {r: sum(1 for _, x in dropped if x == r) for r in sorted({x for _, x in dropped})}
    return NoveltyIndex(
        sundays=sundays,
        symbols=list(symbols),
        el=el,
        ids=ids,
        sources=sources,
        families=fams,
        ranks=r32,
        ranks2=r32 * r32,
        group=group,
        gmask=gmask,
        bounds=bounds_,
        min_names=min_names,
        pool_hash=pool_hash,
        dropped=sorted(dropped),
        counts=counts,
        through=through,
    )


# --------------------------------------------------------------------------- the statistic


@dataclass
class Candidate:
    mask: np.ndarray  # (T, m)
    ranks: np.ndarray  # (T, m) centred own ranks
    sgn: np.ndarray  # (T, m, m) sign(rank_i - rank_k) on masked pairs, 0 elsewhere


def candidate(ix: NoveltyIndex, values: np.ndarray) -> Candidate:
    """A candidate's (T, m) values on the index's Sundays and coins, prepared like a member."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != ix.el.shape:
        raise ValueError(f"candidate values must be {ix.el.shape}, got {values.shape}")
    m, r = prepare(values, ix.el, ix.min_names)
    s = np.sign(r[:, :, None] - r[:, None, :]) * (m[:, :, None] & m[:, None, :])
    return Candidate(mask=m, ranks=r, sgn=s)


def _subset_ranks(sgn: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Centred average ranks over the cells `sub` (..., m) from a sign matrix (..., m, m):
    rank_i - (n + 1) / 2 = 0.5 * sum over k in sub of sign(v_i - v_k)."""
    return 0.5 * sub * np.einsum("...ik,...k->...i", sgn, sub.astype(np.float64))


def bounds(ix: NoveltyIndex, c: Candidate) -> tuple[np.ndarray, np.ndarray]:
    """The optional prefilter's stage 1: (upper bound >= |corr|, estimate of corr) for
    every member (float32 work)."""
    K_ = ix.size
    B = np.zeros(K_)
    est = np.zeros(K_)
    Mc = c.mask
    for g in range(len(ix.bounds) - 1):
        a, b = int(ix.bounds[g]), int(ix.bounds[g + 1])
        if a == b:
            continue
        Mg = ix.gmask[g]
        S = Mg & Mc
        n = S.sum(axis=1)
        valid = n >= ix.min_names
        if valid.sum() < MIN_SUNDAYS:
            continue  # corr undefined for every member of g
        e = (Mg & ~Mc).sum(axis=1).astype(np.float64)
        X = _subset_ranks(c.sgn, S)  # (T, m) exact candidate ranks over S
        dx = (X * X).sum(axis=1)
        l1 = np.abs(X).sum(axis=1)
        XS = np.stack([X, S.astype(np.float64)], axis=-1).astype(np.float32)  # (T, m, 2)
        cs = np.matmul(ix.ranks[:, a:b, :], XS).astype(np.float64)  # (T, n_g, 2)
        sy2 = np.matmul(ix.ranks2[:, a:b, :], XS[:, :, 1:]).astype(np.float64)[..., 0]
        cprod, sy = cs[..., 0], cs[..., 1]
        nn = np.maximum(n, 1)[:, None]
        py2 = np.maximum(sy2 - sy * sy / nn, 0.0)
        ey = (e / 2.0 * np.sqrt(n))[:, None]
        dylo = np.maximum(np.sqrt(py2) - ey, 0.0) ** 2
        v = valid[:, None]
        N = np.where(v, cprod, 0.0).sum(axis=0)
        slack = float(np.where(valid, e / 2.0 * l1, 0.0).sum())
        Dx = float(np.where(valid, dx, 0.0).sum())
        Dylo = np.where(v, dylo, 0.0).sum(axis=0)
        Dy = np.where(v, py2, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            bb = (np.abs(N) + slack) / np.sqrt(Dx * Dylo)
            ee = N / np.sqrt(Dx * Dy)
        B[a:b] = np.where(Dx * Dylo > 0, bb, np.where(Dx > 0, np.inf, 0.0))
        est[a:b] = np.where(np.isfinite(ee), ee, 0.0)
    return B, est


EXACT_CHUNK = 2048  # members per vectorized block in exact()


def exact(ix: NoveltyIndex, c: Candidate, members: np.ndarray | None = None) -> np.ndarray:
    """The exact corr for the given member positions (every member when None); NaN where
    it is undefined. Vectorized per mask group: on a Sunday, the group's common set S with
    the candidate is the same for every member of the group, the candidate's ranks over S
    are computed once, and a member's ranks over S are its own ranks corrected for the
    names in E = (member's names) - (candidate's names):
        2 * rank_S(i) = 2 * own(i) - sum over k in E of sign(own(i) - own(k)).
    Everything is an integer in units of 1/2 (ranks) or 1/4 (products), so N, Dx and Dy
    are exact in any summation order (module docstring)."""
    members = np.arange(ix.size, dtype=np.int64) if members is None else np.asarray(members, dtype=np.int64)
    out = np.full(len(members), np.nan)
    if not len(members):
        return out
    T, m = ix.el.shape
    acc = np.float32 if T * m * (m - 1) ** 2 < 2**24 else np.float64  # exact integer sums
    R2 = ix.ranks_x2
    grp = ix.group[members]
    for g in np.unique(grp):
        sel = np.flatnonzero(grp == g)
        Mg = ix.gmask[g]  # (T, m) the group's cells
        S = Mg & c.mask
        ok = S.sum(axis=1) >= ix.min_names  # counted Sundays, the same for the group
        if ok.sum() < MIN_SUNDAYS:
            continue  # corr undefined for every member of g
        S &= ok[:, None]
        E = Mg & ~c.mask & ok[:, None]
        X = _subset_ranks(c.sgn, S)  # (T, m) candidate ranks over S, exact
        dx = float((X * X).sum())
        X2 = np.rint(2.0 * X).astype(acc)
        e_max = int(E.sum(axis=1).max())
        if e_max:
            order = np.argsort(~E, axis=1, kind="stable")[:, :e_max]  # E's names first
            valid = np.take_along_axis(E, order, axis=1)
        S_int = S.astype(R2.dtype)
        for a in range(0, len(sel), EXACT_CHUNK):
            part = sel[a : a + EXACT_CHUNK]
            R = R2[:, members[part], :]  # (T, s, m) doubled own ranks
            Y = R.copy()
            for q in range(e_max):
                rows = np.flatnonzero(valid[:, q])
                if rows.size:
                    Rq = np.take_along_axis(R[rows], order[rows, q][:, None, None], axis=2)  # (r, s, 1)
                    Y[rows] -= np.sign(R[rows] - Rq).astype(R2.dtype)
            Y *= S_int[:, None, :]  # doubled ranks over S, 0 elsewhere
            Yf = Y.astype(acc)
            n4 = np.einsum("tsm,tm->s", Yf, X2).astype(np.float64)
            dy4 = np.einsum("tsm,tsm->s", Yf, Yf).astype(np.float64)
            N, Dy = n4 / 4.0, dy4 / 4.0
            with np.errstate(invalid="ignore", divide="ignore"):
                rho = N / np.sqrt(dx * Dy)
            out[part] = np.where(dx * Dy > 0, rho, np.nan)
    return out


def scan(
    ix: NoveltyIndex, c: Candidate, *, floor: float = RECALL_FLOOR, top_k: int = TOP_K
) -> tuple[np.ndarray, np.ndarray]:
    """The optional prefilter path: (member positions rechecked, their exact corr) for every
    member whose stage-1 bound reaches `floor` (so every member with |corr| >= floor), plus
    the `top_k` largest bounds. Below `floor` its maximum is a lower bound; max_corr()
    measures every member unless asked not to."""
    B, _ = bounds(ix, c)
    short = set(np.flatnonzero(floor - BOUND_EPS <= B).tolist())
    if top_k:
        short |= set(np.argsort(-B, kind="stable")[:top_k].tolist())
    short = np.array(sorted(i for i in short if B[i] > 0), dtype=np.int64)
    return short, exact(ix, c, short)


def max_corr(
    ix: NoveltyIndex,
    values: np.ndarray,
    *,
    exhaustive: bool = True,
    floor: float = RECALL_FLOOR,
    top_k: int = TOP_K,
    c: Candidate | None = None,
) -> dict:
    """Largest |corr| of a candidate with any member.

    exhaustive (the default): every member measured exactly, so max_abs is the exact
    maximum. exhaustive=False: the prefilter path (scan()), exact when max_abs >= `floor`
    and a lower bound of the maximum below it.

    Returns max_abs (NaN if no member has a defined corr), corr (signed), the nearest
    member's id, source and family, n_rechecked (members measured exactly), n_at_floor
    (members with |corr| >= floor) and exact (True when max_abs is the exact maximum)."""
    c = candidate(ix, values) if c is None else c
    if exhaustive:
        short = np.arange(ix.size, dtype=np.int64)
        rho = exact(ix, c, short)
    else:
        short, rho = scan(ix, c, floor=floor, top_k=top_k)
    a = np.abs(rho)
    if not np.isfinite(a).any():
        return {
            "max_abs": float("nan"),
            "corr": float("nan"),
            "nearest": None,
            "source": None,
            "family": None,
            "n_rechecked": int(len(short)),
            "n_at_floor": 0,
            "exact": bool(exhaustive),
        }
    j = int(np.nanargmax(a))
    k = int(short[j])
    return {
        "max_abs": float(a[j]),
        "corr": float(rho[j]),
        "nearest": ix.ids[k],
        "source": ix.sources[k],
        "family": ix.families[k] or None,
        "n_rechecked": int(len(short)),
        "n_at_floor": int(np.nansum(a >= floor)),
        "exact": bool(exhaustive or a[j] >= floor),
    }


def brute_force(ix: NoveltyIndex, values: np.ndarray) -> np.ndarray:
    """The corr with every member by the definition, one pair and one Sunday at a time
    (scipy ranks on the common set). Slow; the reference for the tests."""
    from scipy.stats import rankdata

    v = np.asarray(values, dtype=np.float64)
    m, _ = prepare(v, ix.el, ix.min_names)
    import libfactors as LF

    vs = LF.snap_ties(v)
    out = np.full(ix.size, np.nan)
    for k in range(ix.size):
        Mj = ix.gmask[ix.group[k]]
        N = Dx = Dy = 0.0
        ns = 0
        for t in range(len(ix.sundays)):
            S = Mj[t] & m[t]
            if S.sum() < ix.min_names:
                continue
            x = rankdata(vs[t, S]) - (S.sum() + 1) / 2.0
            y = rankdata(ix.ranks[t, k, S].astype(np.float64)) - (S.sum() + 1) / 2.0
            N += float(x @ y)
            Dx += float(x @ x)
            Dy += float(y @ y)
            ns += 1
        if ns >= MIN_SUNDAYS and Dx * Dy > 0:
            out[k] = N / np.sqrt(Dx * Dy)
    return out


def always_eligible(ix: NoveltyIndex) -> np.ndarray:
    """Column positions of the coins eligible on every G Sunday."""
    return np.flatnonzero(ix.el.all(axis=0))


def permuted(ix: NoveltyIndex, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The candidate's values moved by one random permutation of the always-eligible coins
    (the same on every Sunday); the coins listed during G are left empty."""
    cols = always_eligible(ix)
    perm = rng.permutation(cols)
    out = np.full_like(np.asarray(values, dtype=np.float64), np.nan)
    out[:, cols] = np.asarray(values, dtype=np.float64)[:, perm]
    return out


def null_max(ix: NoveltyIndex, values: np.ndarray, n_perm: int, seed) -> np.ndarray:
    """The exact max |corr| (every member measured) of `n_perm` permuted copies
    (permuted()); label-free."""
    rng = np.random.default_rng(seed)
    return np.array([max_corr(ix, permuted(ix, values, rng), exhaustive=True)["max_abs"] for _ in range(int(n_perm))])


def describe(ix: NoveltyIndex) -> dict:
    """A JSON-safe summary (no arrays)."""
    return {
        "pool_hash": ix.pool_hash,
        "through": str(ix.through.date()) if ix.through is not None else None,
        "sundays": [str(ix.sundays[0].date()), str(ix.sundays[-1].date()), len(ix.sundays)],
        "symbols": len(ix.symbols),
        "counts": ix.counts,
    }


def snapshot(ix: NoveltyIndex) -> str:
    """JSON of the index's identity (for byte-identity checks in tests)."""
    return json.dumps(
        {"pool_hash": ix.pool_hash, "ids": ix.ids, "sources": ix.sources, "dropped": ix.dropped}, sort_keys=True
    )
