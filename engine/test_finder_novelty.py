"""Tests for Factor Finder's novelty index (factorfinder/novelty.py).

Usage (from engine/):
    python -B test_finder_novelty.py
    TMPDIR=$PWD/.tmp python -B test_finder_novelty.py   # keep the scratch copies repo-local

Reads the data folder read-only. The byte-identity test and the user-member test build the
index in child processes on scratch copies of the files they read (FACTORLAB_DATA, under
TMPDIR, removed at exit). Without the perp bars, the label table or the library values
(data/library_on_engine), every test is skipped (exit 0).

  exact               on fixtures with listings, warm-ups, holes, ties and flat Sundays:
                      exact() equals brute force for every member and max_corr (every
                      member measured) equals the brute-force maximum, at every level;
                      on the optional prefilter path every member whose brute-force
                      |corr| is >= 0.45 is rechecked and measured exactly, and the
                      stage-1 bound is never below the true |corr|
  permutation null    null_max equals the brute-force maximum of the same permuted
                      copies, whose masks differ from the members' (the coins listed
                      during G are left empty), where the prefilter path without top-k
                      padding falls short; a fixed permutation of the always-eligible
                      coins; reproducible for a seed
  G zone only         the index holds the 114 G Sundays and nothing later; a reader that
                      returns a later row raises ZoneLeak
  label columns       reading a fwd_* column (or an unprojected read) raises; building the
                      index requests no fwd_* column and no label-derived status column
  byte-identical      every value dated after the 2024-06-09 row (bars, funding, the zoo
                      table, the library's value shards), every label,
                      and every status.parquet field the screen derives from labels or
                      from the later PIT cuts (screen, g_weeks, pit_check, *_t2, *_t3)
                      overwritten: the same pool_hash, members and drops; one value
                      changed before the cutoff: a different pool_hash
  membership          select_library() reads only fields that end at the cutoff: mutating
                      the T2/T3 fields and pit_check leaves it unchanged, a T1 difference
                      removes the definition, and a later-cut status column is refused
  contents            library, user, zoo and proxy members as documented; drops recorded
                      with their reason; the build is deterministic; a registered formula
                      and a values file of a fixture ledger are user members, with their
                      values on the panels cut at the G cutoff; a withdrawn test is not
  zoo rows            the zoo members' G rows, read from the zoo table, are bit-identical
                      to their expressions (from the zoo library file) evaluated with the
                      bridge's build_store and factor_values on bars cut at the 2024-06-09
                      row. A member without an expression fails the test.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import labels
from factorfinder import novelty as N

ENGINE = Path(__file__).resolve().parent
CUT = pd.Timestamp("2024-06-09", tz="UTC")
ZOO_TABLE = C.DATA / "research" / "alphaforge_daily.parquet"


def zoo_columns() -> list:
    """The af_NN columns of the zoo table (empty without the table)."""
    import pyarrow.parquet as pq

    path = ZOO_TABLE
    if not path.exists():
        return []
    return [c for c in pq.read_schema(path).names if c.startswith("af_") and c[3:].isdigit()]


_CACHE: dict = {}


def _index():
    if "ix" not in _CACHE:
        _CACHE["ix"] = N.build_index()
    return _CACHE["ix"]


# --------------------------------------------------------------------------- fixtures


def _fixture(seed: int = 0, k: int = 160, T: int = 60, m: int = 20):
    """A synthetic index: members built from a few latent factors (so correlations span
    0 to 1), with late listings, leading warm-ups, random holes, coarse ties and flat
    Sundays; and candidates of every kind."""
    rng = np.random.default_rng(seed)
    sundays = pd.date_range("2022-04-03", periods=T, freq="7D", tz="UTC")
    el = np.ones((T, m), bool)
    el[:20, 17] = False
    el[:35, 18] = False
    el[:50, 19] = False
    latent = rng.normal(size=(6, T, m))
    latent += 0.8 * np.cumsum(rng.normal(size=(6, 1, m)) * 0.3, axis=1)  # persistent coin effects
    vals = []
    for i in range(k):
        w = rng.normal(size=6) * (rng.random(6) < 0.5)
        if not w.any():
            w[rng.integers(6)] = 1.0
        v = np.tensordot(w, latent, axes=1) + rng.normal(size=(T, m)) * rng.uniform(0.05, 2.0)
        kind = i % 6
        if kind == 1:
            v[: rng.integers(5, 25)] = np.nan  # warm-up
        elif kind == 2:
            v[rng.random(v.shape) < 0.04] = np.nan  # holes
        elif kind == 3:
            v = np.round(v)  # coarse ties
        elif kind == 4:
            v[rng.integers(0, T, 3)] = 1.0  # flat Sundays
        elif kind == 5:
            v[:, rng.integers(0, m)] = np.nan  # a coin never present
        vals.append(v)
    vals = np.stack(vals)
    ix = N.assemble(
        vals,
        el,
        [f"m{i:03d}" for i in range(k)],
        ["library"] * (k - 10) + ["proxy"] * 10,
        sundays=sundays,
        symbols=[f"S{j}" for j in range(m)],
        min_names=15,
        through=CUT,
    )
    cands = []
    for j in rng.choice(k, 12, replace=False):
        v = vals[j].copy()
        cands.append(v)  # a member itself
        cands.append(-v + rng.normal(size=v.shape) * rng.uniform(0.2, 1.5))  # sign flip plus noise
        x = v.copy()
        x[rng.random(x.shape) < 0.15] = np.nan  # other holes
        cands.append(x)
        y = v.copy()
        y[: rng.integers(8, 30), :] = np.nan  # a longer warm-up
        y[:, rng.integers(0, m)] = np.nan
        cands.append(y)
    for _ in range(12):
        w = rng.normal(size=6)
        cands.append(np.tensordot(w, latent, axes=1) + rng.normal(size=(T, m)) * rng.uniform(0.1, 3.0))
    cands.append(rng.normal(size=(T, m)))
    cands.append(np.round(rng.normal(size=(T, m)) * 2))
    for v in list(cands[:8]):
        cands.append(N.permuted(ix, v, rng))
    return ix, cands


# --------------------------------------------------------------------------- tests


def test_exact_and_recall_against_brute_force_on_fixtures():
    """exact() and max_corr equal brute force for every member and every maximum; on the
    prefilter path every |corr| >= 0.45 is found and measured exactly and the bound
    always holds."""
    pairs, checked, tight = 0, 0, 0
    for seed in (0, 1):
        ix, cands = _fixture(seed)
        assert ix.counts["mask_groups"] > 20, "the fixture must exercise many mask patterns"
        for v in cands:
            bf = N.brute_force(ix, v)
            c = N.candidate(ix, v)
            B, _ = N.bounds(ix, c)
            fin = np.isfinite(bf)
            assert (np.abs(bf[fin]) <= B[fin] + 1e-9).all(), "the stage-1 bound is below a true |corr|"
            short, rho = N.scan(ix, c)
            hi = np.flatnonzero(fin & (np.abs(np.nan_to_num(bf)) >= N.RECALL_FLOOR))
            assert set(hi) <= set(short.tolist()), "a member with |corr| >= 0.45 was not rechecked"
            pos = {int(s): i for i, s in enumerate(short)}
            for h in hi:
                assert abs(rho[pos[int(h)]] - bf[h]) < 1e-12, (rho[pos[int(h)]], bf[h])
            ex = N.exact(ix, c, np.arange(ix.size))
            assert np.array_equal(np.isfinite(ex), fin) and np.allclose(ex[fin], bf[fin], rtol=0, atol=1e-12)
            r = N.max_corr(ix, v, c=c)
            assert r["exact"] and r["n_rechecked"] == ix.size
            if fin.any():
                assert abs(r["max_abs"] - np.nanmax(np.abs(bf))) < 1e-12, (r["max_abs"], np.nanmax(np.abs(bf)))
            else:
                assert np.isnan(r["max_abs"])
            assert r["n_at_floor"] == len(hi)
            r2 = N.max_corr(ix, v, c=c, exhaustive=False)
            if len(hi):
                assert abs(r2["max_abs"] - np.nanmax(np.abs(bf))) < 1e-12 and r2["exact"]
            pairs += len(hi)
            checked += 1
            tight += int(len(short) < ix.size)
    assert checked >= 100 and pairs >= 300, (checked, pairs)
    assert tight >= 0.5 * checked, "the prefilter never narrowed the recheck"


def test_index_holds_g_sundays_only():
    """114 G Sundays, the last 2024-06-09; a read that returns a later row raises."""
    ix = _index()
    assert len(ix.sundays) == 114 and ix.sundays[0] == pd.Timestamp("2022-04-03", tz="UTC")
    assert ix.sundays.max() == CUT == ix.through
    assert pd.Timestamp("2023-07-30", tz="UTC") not in ix.sundays, "the internal embargo Sunday is not a G Sunday"
    import pyarrow.parquet as pq

    saved = pq.read_table
    pq.read_table = lambda path, columns=None, filters=None, **k: saved(path, columns=columns)  # ignores the filter
    try:
        N.read_columns(labels.path(), ["date", "symbol"], through=CUT)
        raise AssertionError("a later row came back without ZoneLeak")
    except N.ZoneLeak:
        pass
    finally:
        pq.read_table = saved
    try:
        N.assemble(
            np.zeros((1, 2, 20)),
            np.ones((2, 20), bool),
            ["x"],
            ["library"],
            sundays=pd.DatetimeIndex([CUT, CUT + pd.Timedelta(days=7)]),
            symbols=list(range(20)),
            min_names=15,
            through=CUT,
        )
        raise AssertionError("an index with a Sunday after the cutoff was assembled")
    except N.ZoneLeak:
        pass


def test_reading_a_label_column_raises():
    """read_columns refuses fwd_* and unprojected reads; the build requests neither, and
    reads no label-derived status column."""
    for cols in (["symbol", "fwd_ret_7"], ["fwd_ret_1"], None):
        try:
            N.read_columns(labels.path(), cols, through=CUT)
            raise AssertionError(f"read_columns accepted {cols}")
        except N.LabelColumnError:
            pass
    import pyarrow.parquet as pq

    saved_rt, saved_status = pq.read_table, N.STATUS_COLUMNS
    requests = []

    def watch(path, columns=None, **k):
        where = path if isinstance(path, (str, os.PathLike)) else getattr(path, "name", None)
        if where is None:
            raise AssertionError("an unnamed parquet read")
        names = pq.read_schema(where).names
        requests.append((Path(where).name, columns))
        if columns is None and any(n.startswith("fwd_") for n in names):
            raise AssertionError(f"an unprojected read of {Path(path).name}, which holds labels")
        if columns is not None and any(str(c).startswith("fwd_") for c in columns):
            raise AssertionError(f"a label column was requested from {Path(path).name}")
        return saved_rt(path, columns=columns, **k)

    pq.read_table = watch
    try:
        N.build_index(library_ids=["CF00006", "CF00061"])
    finally:
        pq.read_table = saved_rt
    assert any(n == "labels.parquet" for n, _ in requests)
    status = [c for n, c in requests if n == "status.parquet"]
    assert status and all(not set(c) & {"screen", "g_weeks"} for c in status)
    N.STATUS_COLUMNS = saved_status + ("screen",)
    try:
        N.build_index(library_ids=["CF00006"])
        raise AssertionError("a label-derived status column was read")
    except N.LabelColumnError:
        pass
    finally:
        N.STATUS_COLUMNS = saved_status


def _scratch_copy(shards: list[str]) -> Path:
    tmp = tempfile.mkdtemp(prefix="finder_novelty_")
    atexit.register(shutil.rmtree, tmp, True)
    dst = Path(tmp) / "data"
    for rel in ("raw", "labels"):
        shutil.copytree(C.DATA / rel, dst / rel)
    (dst / "research").mkdir()
    if ZOO_TABLE.is_file():  # the zoo table is optional input
        shutil.copy2(ZOO_TABLE, dst / "research")
    lib = dst / "library_on_engine" / "values"
    lib.mkdir(parents=True)
    shutil.copy2(C.DATA / "library_on_engine" / "status.parquet", dst / "library_on_engine")
    for s in shards:
        shutil.copy2(C.DATA / "library_on_engine" / "values" / s, lib)
    return dst


#: status.parquet fields the index must never depend on: derived from labels (screen,
#: g_weeks) or from values after the cutoff (pit_check and the T2/T3 PIT cuts)
LATER_STATUS = (
    "screen",
    "g_weeks",
    "pit_check",
    "pit_cells_t2",
    "pit_finite_t2",
    "pit_diff_cells_t2",
    "pit_error_t2",
    "pit_cells_t3",
    "pit_finite_t3",
    "pit_diff_cells_t3",
    "pit_error_t3",
)


def _mutate_later_status(st: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The LATER_STATUS columns overwritten: every T2/T3 cut failed with differences or an
    error, pit_check failed, the screen columns scrambled."""
    rng = np.random.default_rng(seed)
    st = st.copy()
    n = len(st)
    st["pit_check"] = rng.choice(["fail", "no_finite_cells", "pass"], n)
    st["screen"] = rng.choice(["pit_fail", "screened", "too_few_weeks"], n)
    st["g_weeks"] = rng.integers(0, 100, n).astype(st["g_weeks"].dtype)
    for j in (2, 3):
        st[f"pit_cells_t{j}"] = rng.integers(0, 5000, n).astype(np.int32)
        st[f"pit_finite_t{j}"] = rng.integers(0, 5000, n).astype(np.int32)
        st[f"pit_diff_cells_t{j}"] = rng.integers(1, 50, n).astype(np.int32)
        st[f"pit_error_t{j}"] = rng.choice(["", "ValueError", "KeyError"], n)
    return st


def _mutate_later_and_labels(root: Path, seed: int) -> None:
    """Every value dated after the cutoff, every fwd_* label on every row, and every
    status.parquet field derived from labels or from the later PIT cuts, overwritten."""
    rng = np.random.default_rng(seed)
    for p in sorted(root.rglob("*.parquet")):
        if p.name == "status.parquet":
            _mutate_later_status(pd.read_parquet(p), seed).to_parquet(p, index=False)
            continue
        df = pd.read_parquet(p)
        dcol = next(c for c in ("date", "ts", "sunday") if c in df)
        late = pd.DatetimeIndex(df[dcol]) > CUT
        for c in df.columns:
            if c in (dcol, "symbol") or not pd.api.types.is_float_dtype(df[c]):
                continue
            if c.startswith("fwd_"):
                df[c] = rng.normal(size=len(df))
            elif late.any():
                df.loc[late, c] = rng.normal(size=int(late.sum())) * 1e3
        df.to_parquet(p, index=False)


def _build_in_child(data: Path, ids: list[str]) -> dict:
    code = (
        "import json, hashlib, numpy as np\nfrom factorfinder import novelty as N\n"
        f"ix = N.build_index(library_ids={ids!r})\n"
        "h = hashlib.sha256(ix.ranks.tobytes() + ix.gmask.tobytes() + ix.el.tobytes()).hexdigest()\n"
        "print(json.dumps({'pool_hash': ix.pool_hash, 'arrays': h, 'ids': ix.ids, 'dropped': ix.dropped,"
        " 'counts': ix.counts}))"
    )
    env = dict(os.environ, FACTORLAB_DATA=str(data))
    r = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ENGINE, env=env, capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_mutating_later_values_and_labels_leaves_the_index_byte_identical():
    """Same pool_hash, arrays and members after every later value and every label changed;
    a different pool_hash after one value before the cutoff changed."""
    st = pd.read_parquet(C.DATA / "library_on_engine" / "status.parquet", columns=["factor_id", "values_shard"])
    shards = ["values_000.parquet", "values_013.parquet"]
    ids = list(st.loc[st["values_shard"].isin(shards), "factor_id"])[::5]
    a_root = _scratch_copy(shards)
    b_root = _scratch_copy(shards)
    _mutate_later_and_labels(b_root, 7)
    before = hashlib.sha256((a_root / "labels" / "labels.parquet").read_bytes()).hexdigest()
    a, b = _build_in_child(a_root, ids), _build_in_child(b_root, ids)
    assert hashlib.sha256((a_root / "labels" / "labels.parquet").read_bytes()).hexdigest() == before
    assert a["counts"]["library"] > 100 and a["counts"]["proxy"] == 3
    assert a == b, "a value after the cutoff or a label reached the index"
    # without the zoo table the index has no zoo member and the same other members
    if ZOO_TABLE.is_file():
        z_root = _scratch_copy(shards)
        (z_root / "research" / ZOO_TABLE.name).unlink()
        nz = _build_in_child(z_root, ids)
        assert nz["counts"]["zoo"] == 0 and nz["counts"]["library"] == a["counts"]["library"]
        assert nz["ids"] == [i for i in a["ids"] if not (i.startswith("af_") and i[3:].isdigit())]
    else:
        assert a["counts"]["zoo"] == 0
        print("        (no zoo table in this data folder: the index has no zoo member, zoo part skipped)")
    # the test can see a change: one library value on a G Sunday
    p = b_root / "library_on_engine" / "values" / "values_000.parquet"
    df = pd.read_parquet(p)
    fid = next(i for i in a["ids"] if i in df.columns)
    on = (
        (pd.DatetimeIndex(df["sunday"]) == pd.Timestamp("2023-01-01", tz="UTC"))
        & ~df["symbol"].isin(["APT-USDT", "OP-USDT", "ARB-USDT"]).to_numpy()
        & df[fid].notna().to_numpy()
    )
    rows = np.flatnonzero(on)
    row = int(rows[np.argmin(df[fid].to_numpy()[rows])])  # the smallest value becomes the largest
    df.loc[row, fid] = float(np.nanmax(np.abs(df[fid].to_numpy()))) * 10 + 1
    df.to_parquet(p, index=False)
    c = _build_in_child(b_root, ids)
    assert c["pool_hash"] != a["pool_hash"]


def test_library_membership_uses_no_later_cut_field():
    """select_library() admits the same definitions, with the same drops, when every
    T2/T3 PIT field and pit_check are mutated; a T1 difference removes a definition; and
    reading a later-cut or label-derived status column is refused."""
    root = C.DATA / "library_on_engine"
    cols = list(N.STATUS_COLUMNS) + [c for c in LATER_STATUS if c not in ("screen", "g_weeks")]
    st = pd.read_parquet(root / "status.parquet", columns=cols).assign(screen="", g_weeks=np.int16(0))
    keep, drop = N.select_library(st)
    assert len(keep) > 12000 and len(drop) > 2000
    for seed in (1, 2, 3):
        k2, d2 = N.select_library(_mutate_later_status(st, seed))
        assert k2["factor_id"].tolist() == keep["factor_id"].tolist() and d2 == drop, seed
    # the rule can see a T1 failure
    victim = keep["factor_id"].iloc[100]
    bad = st.copy()
    bad.loc[bad["factor_id"] == victim, "pit_diff_cells_t1"] = 3
    k3, d3 = N.select_library(bad)
    assert victim not in set(k3["factor_id"]) and (victim, "pit_fail_t1") in d3
    # the projection holds no later-cut or label-derived column, and adding one is refused
    assert not set(N.STATUS_COLUMNS) & set(LATER_STATUS)
    saved = N.STATUS_COLUMNS
    for extra, err in (("pit_check", N.ZoneLeak), ("pit_finite_t3", N.ZoneLeak), ("g_weeks", N.LabelColumnError)):
        N.STATUS_COLUMNS = saved + (extra,)
        try:
            N._status(root)
            raise AssertionError(f"status column {extra} was read")
        except err:
            pass
        finally:
            N.STATUS_COLUMNS = saved


def test_index_contents_and_determinism():
    """Library, user, zoo and proxy members; drops recorded; the same build twice gives the same
    pool_hash; the user members of a fixture ledger (_check_fixture_ledger)."""
    import trials

    assert N.SOURCES == ("library", "user", "zoo", "proxy")
    ix = _index()
    c = ix.counts
    assert set(c) >= set(N.SOURCES) and not {"engine", "candidate"} & set(c)
    assert c["library"] > 12000 and c["proxy"] == 3
    assert c["zoo"] == len(zoo_columns())
    users = {f"user:{t['test_id']}" for t in trials.active(trials.read())}
    assert {i for i, s in zip(ix.ids, ix.sources) if s == "user"} <= users
    assert c["total"] == ix.size == len(set(ix.ids))
    assert {"pit_symbol_tilt", "btc_beta_90d", "size_proxy", *zoo_columns()} <= set(ix.ids)
    assert not any(i.startswith("fwd_") or i.endswith("_rank") for i in ix.ids)
    reasons = {r for _, r in ix.dropped}
    assert reasons <= {"no_g_values", "too_few_sundays", "constant_per_date", "no_panel", "compute_error"}
    lib_dropped = sum(1 for i, r in ix.dropped if r == "no_g_values")
    assert lib_dropped > 2000, "the open-interest definitions have no G values"
    assert (ix.ranks.shape == (114, ix.size, 20)) and ix.ranks.dtype == np.float32
    assert np.all((ix.ranks * 2) == np.round(ix.ranks * 2)), "centred ranks are multiples of 0.5"
    again = N.build_index()
    assert again.pool_hash == ix.pool_hash and again.ids == ix.ids
    _check_fixture_ledger()


_USER_CHILD = r"""
import json, sys
import numpy as np, pandas as pd
import book, trials, userfactor as uf
from factorfinder import fieldmap as FM, novelty as N
spec = book.load_spec()
sundays = N.g_sundays()
rng = np.random.default_rng(3)
syms = FM.symbols()
lines = ["date,symbol,value"]
for d in pd.date_range("2022-01-02", "2026-09-20", freq="7D"):
    for s, v in zip(syms, rng.normal(size=len(syms))):
        lines.append(f"{d.date()},{s},{float(v)!r}")
open(sys.argv[1], "w").write(chr(10).join(lines) + chr(10))
quiet = lambda *a, **k: None
assert trials.register({"formula": "ts_mean(clv, 7)", "source": None}, spec, name=None, note=None, log=quiet) == 0
assert trials.register({"values_file": sys.argv[1]}, spec, name="vf", note=None, log=quiet) == 0
assert trials.register({"formula": "ts_mean(ret, 9)", "source": None}, spec, name=None, note=None, log=quiet) == 0
assert trials.withdraw("FT-0003", "not needed", log=quiet) == 0
ix = N.build_index(sources=("user", "proxy"))
cut = FM.g_cutoff()
inputs = FM.load_inputs(cut)
p = FM.perp_panels(inputs)
full = pd.DataFrame(uf.full_values("ts_mean(clv, 7)", p), index=p["close"].index, columns=list(p["close"].columns))
want = full.reindex(index=sundays, columns=syms).to_numpy()
k = ix.ids.index("user:FT-0001")
r_ix = ix.ranks[:, k, :]
m_ix = ix.gmask[ix.group[k]]
m_want, r_want = N.prepare(want, ix.el, ix.min_names)
hist = trials.registered(trials.read())[1]["factor"]["history"]
vf = np.array([[np.nan if x is None else x for x in hist.get(str(d.date()), [None] * len(syms))] for d in sundays])
k2 = ix.ids.index("user:FT-0002")
m2, r2 = N.prepare(vf, ix.el, ix.min_names)
print(json.dumps({"ids": ix.ids, "sources": ix.sources, "counts": ix.counts,
                  "formula_equal": bool(np.array_equal(m_ix, m_want) and np.array_equal(r_ix, r_want.astype(np.float32))),
                  "file_equal": bool(np.array_equal(ix.gmask[ix.group[k2]], m2) and np.array_equal(ix.ranks[:, k2, :], r2.astype(np.float32))),
                  "through": str(inputs["through"].date())}))
"""


def _check_fixture_ledger():
    """A scratch copy (bars and label table) with a fixture ledger: a registered formula and a values
    file are user members of the index; the formula's G ranks are those of its values on the panels
    cut at the G cutoff, the file's those of its registered history; a withdrawn test is not a
    member."""
    tmp = tempfile.mkdtemp(prefix="finder_novelty_user_")
    atexit.register(shutil.rmtree, tmp, True)
    dst = Path(tmp) / "data"
    for rel in ("raw", "labels"):
        shutil.copytree(C.DATA / rel, dst / rel)
    r = subprocess.run(
        [sys.executable, "-B", "-c", _USER_CHILD, str(Path(tmp) / "values.csv")],
        cwd=ENGINE,
        env=dict(os.environ, FACTORLAB_DATA=str(dst)),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    users = [i for i, s in zip(out["ids"], out["sources"]) if s == "user"]
    assert users == ["user:FT-0001", "user:FT-0002"], users
    assert out["counts"]["user"] == 2 and out["counts"]["proxy"] == 3 and out["counts"]["library"] == 0
    assert out["formula_equal"] and out["file_equal"], out
    assert out["through"] == "2024-06-09"


def zoo_expressions() -> dict:
    """{member name: expression} from the zoo library file (data/research/
    alphaforge_library.json); empty without the file."""
    path = C.DATA / "research" / "alphaforge_library.json"
    if not path.exists():
        return {}
    lib = json.loads(path.read_text())
    return {m["name"]: m["expr"] for m in lib.get("factors", [])}


def test_zoo_rows_equal_their_expressions_on_the_store_cut_at_the_g_cutoff():
    """The index reads the zoo members' G rows from data/research/alphaforge_daily.parquet. Those
    rows are point in time. The test evaluates each member's expression (from the zoo library
    file) on bars cut at the 2024-06-09 row, with the bridge's own build_store and
    factor_values. On every G Sunday, the values are bit-identical and have the same finite
    mask. Each member of the table needs an expression, so no member enters the index without
    this proof. The index holds exactly the table's members."""
    path = ZOO_TABLE
    if not path.exists():
        print("        (no zoo table in this data folder: the index has no zoo member, nothing to compare)")
        return
    import alphaforge_bridge as AB
    import data_io

    close = data_io.close_matrix()
    store = AB.build_store(close.loc[close.index <= CUT])
    assert store.get("close").index.max() == CUT.tz_convert(None), "the store runs past the G cutoff"
    universe = AB._universe(store)
    ev = AB.Evaluator(store, cache_size=64)
    sundays, symbols = N.g_sundays(), list(close.columns)
    cols = [c for c in N._schema_columns(path) if c.startswith("af_") and c[3:].isdigit()]
    assert cols == zoo_columns(), f"the zoo table holds {cols}"
    pinned = zoo_expressions()
    unpinned = [c for c in cols if c not in pinned]
    assert not unpinned, (
        f"{', '.join(unpinned)} has no expression in the zoo library file, so the index would "
        "hold it without a point-in-time proof"
    )
    table = N.read_columns(path, ["symbol"] + cols, through=CUT)
    for name, expr in ((k, v) for k, v in pinned.items() if k in cols):
        v = AB.factor_values(expr, ev, store, universe)
        v.index = pd.DatetimeIndex(v.index).tz_localize("UTC")
        got = v.reindex(index=sundays, columns=symbols).to_numpy(dtype=float)
        want = N._wide(table, name, sundays, symbols)
        assert np.isfinite(want).sum() > 2000, name
        assert np.array_equal(got, want, equal_nan=True), f"{name}: the table's G rows differ from the cut-store values"
    ix = _index()
    zoo = sorted(i for i, s in zip(ix.ids, ix.sources) if s == "zoo")
    assert zoo == sorted(cols), f"the index holds zoo members {zoo}"


def test_null_max_equals_brute_force_where_masks_differ():
    """Each permuted copy leaves the coins listed during G empty, so its mask differs from
    the members that have them; null_max (every member measured) equals the brute-force
    maximum of the same copies, and the prefilter path with no top-k padding falls short
    on some of them (so the comparison can see a lower bound)."""
    copies, short = 0, 0
    for seed in (0, 1):
        ix, cands = _fixture(seed)
        late = np.setdiff1d(np.arange(ix.el.shape[1]), N.always_eligible(ix))
        assert len(late) == 3
        with_late = ix.gmask[ix.group][:, :, late].any(axis=(1, 2))  # members with a late coin
        assert with_late.mean() > 0.5, "most members must have cells a permuted copy lacks"
        for j, v in enumerate(cands[::4]):
            got = N.null_max(ix, v, 2, [seed, j])
            rng = np.random.default_rng([seed, j])  # the same copies, in order
            for g in got:
                p = N.permuted(ix, v, rng)
                c = N.candidate(ix, p)
                assert not c.mask[:, late].any(), "a permuted copy has a value on a coin listed during G"
                bf = N.brute_force(ix, p)
                if not np.isfinite(bf).any():
                    assert np.isnan(g)
                    continue
                want = float(np.nanmax(np.abs(bf)))
                assert abs(g - want) < 1e-12, (seed, j, g, want)
                lower = N.max_corr(ix, p, c=c, exhaustive=False, top_k=0)["max_abs"]
                short += int(not np.isfinite(lower) or lower < want - 1e-9)
                copies += 1
    assert copies >= 40, copies
    assert short >= 1, "the prefilter path never fell short: the fixture cannot tell a lower bound from the maximum"


def test_permutation_null_is_label_free_and_reproducible():
    """permuted() moves the always-eligible coins by one fixed permutation and leaves the
    coins listed during G empty; null_max is reproducible for a seed and, for a copy of a
    member, far below its observed maximum of 1."""
    ix = _index()
    cols = N.always_eligible(ix)
    assert len(cols) == 17, "17 perps are eligible on every G Sunday"
    k = ix.ids.index("CF00006")
    v = ix.ranks[:, k, :].astype(float)
    v[~ix.gmask[ix.group[k]]] = np.nan
    p = N.permuted(ix, v, np.random.default_rng(4))
    late = np.setdiff1d(np.arange(20), cols)
    assert np.isnan(p[:, late]).all()
    # one permutation for every Sunday: each target column is one source column throughout,
    # and the sources are the always-eligible columns, each used once
    src = []
    for j in cols:
        hit = [c for c in cols if np.array_equal(p[:, j], v[:, c], equal_nan=True)]
        assert len(hit) == 1, "a column is not a copy of exactly one source column"
        src.append(hit[0])
    assert sorted(src) == sorted(cols.tolist()) and src != cols.tolist()
    assert N.max_corr(ix, v)["max_abs"] > 0.999
    a = N.null_max(ix, v, 3, [11, 0])
    b = N.null_max(ix, v, 3, [11, 0])
    assert np.array_equal(a, b) and (a < 0.9).all(), a


def missing_data() -> str | None:
    """Why the suite cannot run on this data folder, or None."""
    import libfactors as LF

    if not any((C.RAW_OHLCV / f"{C.sym_to_file(s)}.parquet").exists() for s in C.UNIVERSE):
        return "no perp bars in the data folder (run ./setup.sh)"
    if not labels.path().is_file():
        return "no label table in the data folder (run ./setup.sh)"
    if not ((LF.out_dir() / "status.parquet").is_file() and LF.has_values()):
        return "no library values in data/library_on_engine (run libfactors.py --compute)"
    return None


if __name__ == "__main__":
    why = missing_data()
    if why:
        print(f"  SKIP  every finder novelty-index test skipped: {why}")
        raise SystemExit(0)
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} finder novelty-index tests passed")
    raise SystemExit(1 if failed else 0)
