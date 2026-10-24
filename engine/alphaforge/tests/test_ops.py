"""Tests for ``alphaforge.ops``.

Self-contained: every panel is built from ``numpy.random.default_rng(0)``.
Coverage:

* every ``ts_*`` op: no-look-ahead, ``min_periods`` NaN prefix, NaN-in-window;
* ``ts_mean/std/sum/min/max/corr/cov/skew/kurt`` against pandas rolling,
  ``ts_rank`` / ``ts_argmax`` / ``ts_argmin`` / ``ts_product`` against slow
  ``rolling.apply`` references;
* every ``cs_*`` / ``group_*`` op: NaN pattern preserved, all-NaN row stays NaN,
  plus the statistical property each op promises;
* registry completeness / arity / flags / idempotency;
* performance on a 1000 x 300 panel with ``d = 60``.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

import alphaforge.ops as ops
from alphaforge.ops import REGISTRY, OpSpec, get_op, list_ops, register_all_builtin

# ----------------------------------------------------------------------------
# panel builders
# ----------------------------------------------------------------------------

N_DATES, N_ASSETS, D = 60, 20, 5


def make_panel(
    rng: np.random.Generator, n: int = N_DATES, m: int = N_ASSETS, nan_frac: float = 0.05, ties: bool = False
) -> pd.DataFrame:
    a = rng.standard_normal((n, m))
    if ties:
        a = np.round(a, 1)
    if nan_frac > 0:
        a[rng.random((n, m)) < nan_frac] = np.nan
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame(a, index=idx, columns=[f"S{i:04d}" for i in range(m)])


def make_groups(rng: np.random.Generator, like: pd.DataFrame, n_groups: int = 4) -> pd.DataFrame:
    return pd.DataFrame(rng.integers(0, n_groups, like.shape).astype(float), index=like.index, columns=like.columns)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def x(rng) -> pd.DataFrame:
    return make_panel(rng)


@pytest.fixture
def y(rng) -> pd.DataFrame:
    rng.standard_normal(10)  # decorrelate from x
    return make_panel(rng)


@pytest.fixture
def g(rng, x) -> pd.DataFrame:
    return make_groups(rng, x)


@pytest.fixture
def clean(rng) -> pd.DataFrame:
    return make_panel(rng, nan_frac=0.0)


TS_UNARY = [
    "ts_delay",
    "ts_delta",
    "ts_pct_change",
    "ts_mean",
    "ts_sum",
    "ts_std",
    "ts_min",
    "ts_max",
    "ts_argmax",
    "ts_argmin",
    "ts_rank",
    "ts_zscore",
    "ts_decay_linear",
    "ts_skew",
    "ts_kurt",
    "ts_product",
]
TS_BINARY = ["ts_corr", "ts_cov"]
TS_ALL = TS_UNARY + TS_BINARY
TS_LAGGED = ["ts_delay", "ts_delta", "ts_pct_change"]  # depend on x[t] and x[t-d] only
TS_WINDOWED = [n for n in TS_ALL if n not in TS_LAGGED]  # depend on the whole window [t-d+1, t]

CS_ALL = [
    "cs_rank",
    "cs_zscore",
    "cs_demean",
    "cs_scale",
    "cs_winsorize",
    "cs_quantile_clip",
    "group_neutralize",
    "group_rank",
    "group_zscore",
    "cs_regress_out",
    "cs_neutralize",
]
CS_UNARY = ["cs_rank", "cs_zscore", "cs_demean", "cs_scale", "cs_winsorize", "cs_quantile_clip"]
GROUP_OPS = ["group_neutralize", "group_rank", "group_zscore"]


def call_ts(name: str, x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    fn = getattr(ops, name)
    return fn(x, y, d) if name in TS_BINARY else fn(x, d)


def call_cs(name: str, x: pd.DataFrame, g: pd.DataFrame, size: pd.DataFrame) -> pd.DataFrame:
    fn = getattr(ops, name)
    if name in GROUP_OPS:
        return fn(x, g)
    if name == "cs_regress_out":
        return fn(x, size)
    if name == "cs_neutralize":
        return fn(x, g, size)
    return fn(x)


def default_args(spec: OpSpec, x: pd.DataFrame, y: pd.DataFrame, g: pd.DataFrame) -> list:
    """Build a plausible positional argument list from ``arg_types``."""
    args: list = []
    panels = iter([x, y, y])
    for t in spec.arg_types:
        if t == "panel":
            args.append(next(panels))
        elif t == "group":
            args.append(g)
        elif t == "int":
            args.append(D)
        elif t == "float":
            args.append(spec.float_choices[0])
    if spec.name == "clip":
        args = [x, -1.0, 1.0]
    if spec.name == "cs_quantile_clip":
        args = [x, 0.05, 0.95]
    return args


# ============================================================================
# generic operator contract (every registered op)
# ============================================================================


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_operator_contract(name, x, y, g):
    """float64, same index/columns (name kept), inputs untouched, no ±inf."""
    spec = get_op(name)
    args = default_args(spec, x, y, g)
    snapshots = [a.copy() for a in args if isinstance(a, pd.DataFrame)]
    out = spec(*args)
    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(x.index) and out.columns.equals(x.columns)
    assert out.index.name == "date"
    assert (out.dtypes == np.float64).all()
    vals = out.to_numpy()
    assert not np.isinf(vals).any()
    assert out is not x
    for a, s in zip([a for a in args if isinstance(a, pd.DataFrame)], snapshots):
        pd.testing.assert_frame_equal(a, s)


# ============================================================================
# time-series operators
# ============================================================================


@pytest.mark.parametrize("name", TS_ALL)
def test_ts_no_lookahead(name, x, y, rng):
    """Perturbing rows > t (values *and* NaN pattern) leaves output rows <= t unchanged."""
    t = 30
    base = call_ts(name, x, y, D)
    x2, y2 = x.copy(), y.copy()
    x2.iloc[t + 1 :] = make_panel(rng, nan_frac=0.2).iloc[t + 1 :].to_numpy() * 7.0
    y2.iloc[t + 1 :] = make_panel(rng, nan_frac=0.2).iloc[t + 1 :].to_numpy() * 3.0
    x2.iloc[t + 1] = np.nan  # the very next row all-missing must not leak either
    pert = call_ts(name, x2, y2, D)
    pd.testing.assert_frame_equal(base.iloc[: t + 1], pert.iloc[: t + 1])
    # and the future did actually change (sanity: the test is not vacuous)
    assert not base.iloc[t + 1 :].equals(pert.iloc[t + 1 :])


@pytest.mark.parametrize("name", TS_ALL)
def test_ts_min_periods_nan_prefix(name, clean, rng):
    """First rows are NaN until a full window exists; the first full window is finite."""
    y2 = make_panel(rng, nan_frac=0.0)
    out = call_ts(name, clean, y2, D)
    n_prefix = D if name in TS_LAGGED else D - 1
    assert out.iloc[:n_prefix].isna().all().all(), f"{name}: expected {n_prefix} NaN prefix rows"
    assert out.iloc[n_prefix].notna().all(), f"{name}: first complete window should be finite"


@pytest.mark.parametrize("name", TS_WINDOWED)
def test_ts_nan_in_window_propagates(name, clean, rng):
    """A NaN at (i, j) blanks rows i..i+d-1 of column j only (min_periods=window)."""
    y2 = make_panel(rng, nan_frac=0.0)
    i, j = 20, 3
    base = call_ts(name, clean, y2, D)
    xn = clean.copy()
    xn.iloc[i, j] = np.nan
    out = call_ts(name, xn, y2, D)
    col = out.iloc[:, j]
    assert col.iloc[i : i + D].isna().all()
    assert np.isfinite(col.iloc[i + D])
    assert np.isfinite(col.iloc[i - 1])
    others = [c for c in range(out.shape[1]) if c != j]
    pd.testing.assert_frame_equal(out.iloc[:, others], base.iloc[:, others])


@pytest.mark.parametrize("name", TS_BINARY)
def test_ts_binary_nan_in_other_propagates(name, clean, rng):
    y2 = make_panel(rng, nan_frac=0.0)
    y2.iloc[20, 3] = np.nan
    out = call_ts(name, clean, y2, D)
    assert out.iloc[20:25, 3].isna().all() and np.isfinite(out.iloc[25, 3]) and np.isfinite(out.iloc[19, 3])


def test_ts_mean_std_sum_min_max_vs_pandas(x):
    r = x.rolling(D, min_periods=D)
    pd.testing.assert_frame_equal(ops.ts_mean(x, D), r.mean())
    pd.testing.assert_frame_equal(ops.ts_std(x, D), r.std())
    pd.testing.assert_frame_equal(ops.ts_sum(x, D), r.sum())
    pd.testing.assert_frame_equal(ops.ts_min(x, D), r.min())
    pd.testing.assert_frame_equal(ops.ts_max(x, D), r.max())


def test_ts_rank_vs_slow_reference(rng):
    x = make_panel(rng, n=40, m=6, nan_frac=0.05, ties=True)  # ties matter for the 0.5 convention
    d = 7
    ref = x.rolling(d, min_periods=d).apply(lambda w: rankdata(w)[-1] / len(w), raw=True)
    pd.testing.assert_frame_equal(ops.ts_rank(x, d), ref)


def test_ts_rank_bounds_and_degenerate_window(clean):
    out = ops.ts_rank(clean, 10)
    v = out.to_numpy()[~np.isnan(out.to_numpy())]
    assert v.min() >= 1 / 10 - 1e-12 and v.max() <= 1.0 + 1e-12
    # strictly increasing column -> today is always the max -> 1.0
    mono = clean.copy()
    mono.iloc[:, 0] = np.arange(len(mono), dtype=float)
    assert np.allclose(ops.ts_rank(mono, 10).iloc[9:, 0], 1.0)
    # d = 1: every valid cell is 1.0
    one = ops.ts_rank(clean, 1)
    assert np.allclose(one.to_numpy(), 1.0)
    assert ops.ts_rank(clean, 5.0).equals(ops.ts_rank(clean, 5))  # integral float accepted


def test_ts_argmax_argmin_vs_slow_reference(rng):
    x = make_panel(rng, n=50, m=6, nan_frac=0.05, ties=True)
    d = 6
    ref_max = x.rolling(d, min_periods=d).apply(lambda w: len(w) - 1 - np.argmax(w), raw=True)
    ref_min = x.rolling(d, min_periods=d).apply(lambda w: len(w) - 1 - np.argmin(w), raw=True)
    pd.testing.assert_frame_equal(ops.ts_argmax(x, d), ref_max)
    pd.testing.assert_frame_equal(ops.ts_argmin(x, d), ref_min)


def test_ts_argmax_semantics(clean):
    mono = clean.copy()
    mono.iloc[:, 0] = np.arange(len(mono), dtype=float)  # increasing: max is today
    assert (ops.ts_argmax(mono, 5).iloc[4:, 0] == 0).all()
    assert (ops.ts_argmin(mono, 5).iloc[4:, 0] == 4).all()  # min is the oldest row of the window
    assert ops.ts_argmax(clean, 100).isna().all().all()  # window longer than history
    assert (ops.ts_argmax(clean, 1) == 0).all().all()


def test_ts_corr_cov_vs_pandas(x, y):
    pd.testing.assert_frame_equal(ops.ts_corr(x, y, 10), x.rolling(10, min_periods=10).corr(y))
    pd.testing.assert_frame_equal(ops.ts_cov(x, y, 10), x.rolling(10, min_periods=10).cov(y))
    c = ops.ts_corr(x, y, 10).to_numpy()
    assert np.nanmin(c) >= -1.0 and np.nanmax(c) <= 1.0
    self_corr = ops.ts_corr(x, x, 10)
    assert np.allclose(self_corr.to_numpy()[~np.isnan(self_corr.to_numpy())], 1.0)
    const = x.copy()
    const.iloc[:, 0] = 2.0
    assert ops.ts_corr(x, const, 10).iloc[:, 0].isna().all()  # zero variance -> NaN, never inf


def test_ts_corr_unsorted_columns_preserved(x, y):
    cols = list(x.columns)[::-1]
    xr, yr = x[cols], y[cols]
    out = ops.ts_corr(xr, yr, 10)
    assert list(out.columns) == cols
    pd.testing.assert_frame_equal(out, ops.ts_corr(x, y, 10)[cols])


def test_ts_decay_linear_weights(clean):
    d = 4
    out = ops.ts_decay_linear(clean, d)
    w = np.array([4, 3, 2, 1], dtype=float) / 10.0  # newest first
    t = 10
    expected = sum(w[k] * clean.iloc[t - k, 0] for k in range(d))
    assert out.iloc[t, 0] == pytest.approx(expected)
    assert out.iloc[: d - 1].isna().all().all()
    pd.testing.assert_frame_equal(ops.ts_decay_linear(clean, 1), clean)


def test_ts_zscore_definition(x):
    r = x.rolling(D, min_periods=D)
    expected = (x - r.mean()) / r.std()
    pd.testing.assert_frame_equal(ops.ts_zscore(x, D), expected)
    const = x.copy()
    const.iloc[:, 0] = 1.0
    assert ops.ts_zscore(const, D).iloc[:, 0].isna().all()  # zero std -> NaN


def test_ts_product_vs_rolling_apply(x):
    ref = x.rolling(D, min_periods=D).apply(np.prod, raw=True)
    pd.testing.assert_frame_equal(ops.ts_product(x, D), ref)


def test_ts_skew_kurt_vs_pandas(x):
    pd.testing.assert_frame_equal(ops.ts_skew(x, 10), x.rolling(10, min_periods=10).skew())
    pd.testing.assert_frame_equal(ops.ts_kurt(x, 10), x.rolling(10, min_periods=10).kurt())
    assert ops.ts_skew(x, 2).isna().all().all() and ops.ts_kurt(x, 3).isna().all().all()


def test_ts_lag_semantics(x):
    pd.testing.assert_frame_equal(ops.ts_delay(x, 1), x.shift(1))
    pd.testing.assert_frame_equal(ops.ts_delay(x, 3), x.shift(3))
    pd.testing.assert_frame_equal(ops.ts_delta(x, 3), x - x.shift(3))
    pd.testing.assert_frame_equal(ops.ts_pct_change(x, 3), x / x.shift(3) - 1)
    z = x.copy()
    z.iloc[5, 0] = 0.0
    assert np.isnan(ops.ts_pct_change(z, 1).iloc[6, 0])  # division by a zero base -> NaN


@pytest.mark.parametrize("name", TS_ALL)
def test_ts_window_validation(name, x, y):
    with pytest.raises(ValueError):
        call_ts(name, x, y, 0)
    with pytest.raises(TypeError):
        call_ts(name, x, y, 2.5)


# ============================================================================
# element-wise operators
# ============================================================================


def test_elementwise_division_by_zero_is_nan(x):
    z = x.copy()
    z.iloc[0, :] = 0.0
    out = ops.div(x, z)
    assert out.iloc[0].isna().all()
    assert not np.isinf(out.to_numpy()).any()
    assert ops.inverse(z).iloc[0].isna().all()
    assert np.isnan(ops.div(z, z).iloc[0, 0])  # 0/0


def test_elementwise_domains(x):
    neg = ops.neg(ops.abs_(x))  # all <= 0 (or NaN)
    assert ops.log(neg).notna().sum().sum() == 0
    assert ops.sqrt(neg.where(neg < 0)).isna().all().all()
    pos = ops.abs_(x) + 1.0
    pd.testing.assert_frame_equal(ops.log(pos), np.log(pos))
    pd.testing.assert_frame_equal(ops.sqrt(pos), np.sqrt(pos))
    zero = x * 0.0
    assert ops.log(zero).isna().all().all()  # log(0) = -inf -> NaN


def test_signed_transforms(x):
    pd.testing.assert_frame_equal(ops.signed_log(x), np.sign(x) * np.log1p(x.abs()))
    pd.testing.assert_frame_equal(ops.signed_sqrt(x), np.sign(x) * np.sqrt(x.abs()))
    assert (ops.sign(x).abs().fillna(1.0) <= 1.0).all().all()


def test_power_and_arithmetic(x, y):
    pd.testing.assert_frame_equal(ops.power(x, 2.0), x**2)
    assert ops.power(x.where(x < 0), 0.5).isna().all().all()  # negative base, fractional exponent
    pd.testing.assert_frame_equal(ops.add(x, y), x + y)
    pd.testing.assert_frame_equal(ops.sub(x, y), x - y)
    pd.testing.assert_frame_equal(ops.mul(x, y), x * y)
    pd.testing.assert_frame_equal(ops.add(x, 1.0), x + 1.0)  # scalar broadcast
    pd.testing.assert_frame_equal(ops.sub(1.0, x), 1.0 - x)


def test_max_min_nan_propagation(x, y):
    out_max, out_min = ops.max_(x, y), ops.min_(x, y)
    either_nan = x.isna() | y.isna()
    assert out_max.isna().equals(either_nan) and out_min.isna().equals(either_nan)
    pd.testing.assert_frame_equal(out_max, pd.DataFrame(np.maximum(x, y), index=x.index, columns=x.columns))
    assert (out_max >= out_min).where(~either_nan, True).all().all()


def test_clip(x):
    out = ops.clip(x, -0.5, 0.5)
    assert out.min().min() >= -0.5 and out.max().max() <= 0.5
    assert out.isna().equals(x.isna())
    with pytest.raises(ValueError):
        ops.clip(x, 1.0, -1.0)


def test_binary_op_realigns_other_operand(x, y):
    shuffled = y.iloc[::-1, ::-1]  # same labels, different order
    pd.testing.assert_frame_equal(ops.add(x, shuffled), x + y)


# ============================================================================
# cross-sectional operators
# ============================================================================


@pytest.mark.parametrize("name", CS_UNARY + ["group_neutralize", "group_rank"])
def test_cs_nan_pattern_preserved(name, x, g, y):
    out = call_cs(name, x, g, y)
    assert out.isna().equals(x.isna()), f"{name} changed the NaN pattern"


def test_group_zscore_nan_pattern_is_input_plus_singletons(x, g):
    """group_zscore's only extra NaNs are zero-dispersion (single-member) groups."""
    out = ops.group_zscore(x, g)
    singleton = pd.DataFrame(False, index=x.index, columns=x.columns)
    for i in range(len(x)):
        counts = x.iloc[i].dropna().groupby(g.iloc[i]).size()
        singleton.iloc[i] = g.iloc[i].map(counts).eq(1).to_numpy() & x.iloc[i].notna().to_numpy()
    assert singleton.any().any()  # the fixture does contain singleton groups
    assert out.isna().equals(x.isna() | singleton)


@pytest.mark.parametrize("name", CS_ALL)
def test_cs_all_nan_row_stays_nan(name, x, g, y):
    xn = x.copy()
    xn.iloc[7] = np.nan
    out = call_cs(name, xn, g, y)
    assert out.iloc[7].isna().all()
    assert out.iloc[8].notna().any()  # neighbouring dates unaffected


def test_cs_rank_range_and_ties(x):
    out = ops.cs_rank(x)
    v = out.to_numpy()
    assert np.nanmax(v) == 1.0 and np.nanmin(v) > 0.0
    assert (out.max(axis=1) == 1.0).all()
    pd.testing.assert_frame_equal(out, x.rank(axis=1, pct=True))
    tied = x.copy()
    tied.iloc[0, :] = 1.0
    assert np.allclose(
        ops.cs_rank(tied).iloc[0].dropna(), (1 + tied.iloc[0].notna().sum()) / 2 / tied.iloc[0].notna().sum()
    )


def test_cs_zscore_moments(x):
    out = ops.cs_zscore(x)
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(out.std(axis=1, ddof=0), 1.0, atol=1e-12)
    ref = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1, ddof=0), axis=0)
    pd.testing.assert_frame_equal(out, ref)
    const = x.copy()
    const.iloc[0] = 0.1  # sum-of-0.1's is not exactly representable: still detected as constant
    assert ops.cs_zscore(const).iloc[0].isna().all()
    single = x.copy()
    single.iloc[1] = np.nan
    single.iloc[1, 0] = 3.0
    assert ops.cs_zscore(single).iloc[1].isna().all()


def test_cs_demean(x):
    out = ops.cs_demean(x)
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-12)
    pd.testing.assert_frame_equal(out, x.sub(x.mean(axis=1), axis=0))


def test_cs_scale_sum_abs_one(x):
    out = ops.cs_scale(x)
    assert np.allclose(out.abs().sum(axis=1), 1.0)
    zeros = x.copy()
    zeros.iloc[0] = 0.0
    assert ops.cs_scale(zeros).iloc[0].isna().all()  # documented degenerate row


def test_cs_winsorize_bounds(x):
    xo = x.copy()
    xo.iloc[0, 0] = 50.0  # gross outlier
    k = 3.0
    out = ops.cs_winsorize(xo, k)
    med = np.nanmedian(xo.to_numpy(), axis=1)
    mad = np.nanmedian(np.abs(xo.to_numpy() - med[:, None]), axis=1) * 1.4826
    lo, hi = med - k * mad, med + k * mad
    v = out.to_numpy()
    ok = np.isnan(v) | ((v >= lo[:, None] - 1e-12) & (v <= hi[:, None] + 1e-12))
    assert ok.all()
    assert out.iloc[0, 0] == pytest.approx(hi[0])
    inside = (xo.to_numpy() >= lo[:, None]) & (xo.to_numpy() <= hi[:, None])
    assert np.array_equal(v[inside], xo.to_numpy()[inside])  # interior values untouched
    flat = x.copy()
    flat.iloc[1, :15] = 2.0  # MAD == 0 -> row unchanged (including its outliers)
    pd.testing.assert_frame_equal(ops.cs_winsorize(flat).iloc[[1]], flat.iloc[[1]])
    with pytest.raises(ValueError):
        ops.cs_winsorize(x, 0.0)


def test_cs_quantile_clip_vs_numpy(x):
    out = ops.cs_quantile_clip(x, 0.1, 0.9)
    q = np.nanquantile(x.to_numpy(), [0.1, 0.9], axis=1)
    ref = np.clip(x.to_numpy(), q[0][:, None], q[1][:, None])
    np.testing.assert_allclose(out.to_numpy(), ref, equal_nan=True)
    assert out.isna().equals(x.isna())
    with pytest.raises(ValueError):
        ops.cs_quantile_clip(x, 0.9, 0.1)


# ---- group ops ---------------------------------------------------------------


def slow_group_transform(x: pd.DataFrame, g: pd.DataFrame, how: str) -> pd.DataFrame:
    rows = []
    for i in range(len(x)):
        s, gi = x.iloc[i], g.iloc[i]
        grp = s.groupby(gi)
        if how == "neutralize":
            rows.append(s - grp.transform("mean"))
        elif how == "zscore":
            rows.append((s - grp.transform("mean")) / grp.transform("std", ddof=0))
        else:
            rows.append(grp.rank(pct=True))
    out = pd.DataFrame(rows, index=x.index, columns=x.columns).astype(float)
    return out.where(g.notna())


def test_group_neutralize_group_means_zero(x, g):
    out = ops.group_neutralize(x, g)
    for i in range(len(x)):
        means = out.iloc[i].groupby(g.iloc[i]).mean()
        assert np.allclose(means, 0.0, atol=1e-12)
    pd.testing.assert_frame_equal(out, slow_group_transform(x, g, "neutralize"))


def test_group_zscore_moments(x, g):
    out = ops.group_zscore(x, g)
    ref = slow_group_transform(x, g, "zscore")
    ref = ref.replace([np.inf, -np.inf], np.nan)
    pd.testing.assert_frame_equal(out, ref)
    for i in range(len(x)):
        grp = out.iloc[i].groupby(g.iloc[i])
        assert np.allclose(grp.mean().dropna(), 0.0, atol=1e-12)
        assert np.allclose(grp.std(ddof=0).dropna(), 1.0, atol=1e-12)


def test_group_rank_within_group(x, g):
    out = ops.group_rank(x, g)
    pd.testing.assert_frame_equal(out, slow_group_transform(x, g, "rank"))
    for i in range(len(x)):
        mx = out.iloc[i].groupby(g.iloc[i]).max()
        assert np.allclose(mx.dropna(), 1.0)
    v = out.to_numpy()
    assert np.nanmin(v) > 0 and np.nanmax(v) <= 1.0


@pytest.mark.parametrize("name", GROUP_OPS)
def test_group_nan_group_gives_nan(name, x, g):
    gn = g.copy()
    gn.iloc[:, 2] = np.nan
    gn.iloc[3, 5] = np.nan
    out = getattr(ops, name)(x, gn)
    assert out.iloc[:, 2].isna().all() and np.isnan(out.iloc[3, 5])
    expected_nan = x.isna() | gn.isna()
    if name == "group_zscore":  # may add NaNs for single-member groups (zero dispersion)
        assert (out.isna() | ~expected_nan).all().all()
    else:
        assert out.isna().equals(expected_nan)


def test_group_ops_ignore_group_membership_elsewhere(x, g):
    """Group ids on other dates must not affect a date (keys are (date, group))."""
    out = ops.group_neutralize(x, g)
    g2 = g.copy()
    g2.iloc[10:] = (g2.iloc[10:] + 1) % 4
    out2 = ops.group_neutralize(x, g2)
    pd.testing.assert_frame_equal(out.iloc[:10], out2.iloc[:10])


# ---- regression-based ----------------------------------------------------------


def test_cs_regress_out_residual_orthogonal(x, y):
    resid = ops.cs_regress_out(x, y)
    joint = x.notna() & y.notna()
    assert resid.isna().equals(~joint)
    r, s = resid.to_numpy(), y.to_numpy()
    for i in range(len(x)):
        m = joint.iloc[i].to_numpy()
        assert abs(np.sum(r[i, m] * s[i, m])) < 1e-9  # orthogonal to x
        assert abs(np.sum(r[i, m])) < 1e-9  # orthogonal to the intercept
    # matches numpy least squares on one date
    i, m = 5, joint.iloc[5].to_numpy()
    A = np.column_stack([np.ones(m.sum()), s[i, m]])
    beta = np.linalg.lstsq(A, x.to_numpy()[i, m], rcond=None)[0]
    np.testing.assert_allclose(r[i, m], x.to_numpy()[i, m] - A @ beta, atol=1e-10)


def test_cs_regress_out_degenerate_rows(x, y):
    few = x.copy()
    few.iloc[0] = np.nan
    few.iloc[0, :2] = 1.0  # only two valid pairs
    assert ops.cs_regress_out(few, y).iloc[0].isna().all()
    const = y.copy()
    const.iloc[1] = 3.0
    out = ops.cs_regress_out(x, const)
    pd.testing.assert_frame_equal(out.iloc[[1]], ops.cs_demean(x).iloc[[1]])


def test_cs_neutralize_orthogonal_to_industry_and_size(x, g, y):
    size = y
    out = ops.cs_neutralize(x, g, size)
    joint = x.notna() & g.notna() & size.notna()
    assert out.isna().equals(~joint)
    for i in range(len(x)):
        m = joint.iloc[i].to_numpy()
        row = out.iloc[i][m]
        assert np.allclose(row.groupby(g.iloc[i][m]).mean(), 0.0, atol=1e-9)
        assert abs(np.dot(row.to_numpy(), size.iloc[i][m].to_numpy())) < 1e-9
    # equals joint OLS on [industry dummies, size] for one date
    i = 12
    m = joint.iloc[i].to_numpy()
    dummies = pd.get_dummies(g.iloc[i][m].astype(int)).to_numpy(dtype=float)
    A = np.column_stack([dummies, size.iloc[i][m].to_numpy()])
    yy = x.iloc[i][m].to_numpy()
    beta = np.linalg.lstsq(A, yy, rcond=None)[0]
    np.testing.assert_allclose(out.iloc[i][m].to_numpy(), yy - A @ beta, atol=1e-9)


# ============================================================================
# registry
# ============================================================================

EXPECTED_ARITY = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "neg": 1,
    "abs": 1,
    "sign": 1,
    "log": 1,
    "signed_log": 1,
    "sqrt": 1,
    "signed_sqrt": 1,
    "power": 2,
    "inverse": 1,
    "max": 2,
    "min": 2,
    "clip": 3,
    "ts_delay": 2,
    "ts_delta": 2,
    "ts_pct_change": 2,
    "ts_mean": 2,
    "ts_sum": 2,
    "ts_std": 2,
    "ts_min": 2,
    "ts_max": 2,
    "ts_argmax": 2,
    "ts_argmin": 2,
    "ts_rank": 2,
    "ts_zscore": 2,
    "ts_corr": 3,
    "ts_cov": 3,
    "ts_decay_linear": 2,
    "ts_skew": 2,
    "ts_kurt": 2,
    "ts_product": 2,
    "cs_rank": 1,
    "cs_zscore": 1,
    "cs_demean": 1,
    "cs_scale": 1,
    "cs_winsorize": 2,
    "cs_quantile_clip": 3,
    "group_neutralize": 2,
    "group_rank": 2,
    "group_zscore": 2,
    "cs_regress_out": 2,
    "cs_neutralize": 3,
}


def test_registry_contains_all_ops_with_arity():
    assert set(REGISTRY) == set(EXPECTED_ARITY)
    assert len(REGISTRY) == 45
    for name, arity in EXPECTED_ARITY.items():
        spec = get_op(name)
        assert spec.name == name and len(spec.arg_types) == arity and spec.arity == arity
        assert set(spec.arg_types) <= {"panel", "int", "float", "group"}
        assert spec.category in {"elementwise", "ts", "cs", "group"}
        assert spec.arg_types[0] == "panel"
        assert spec.int_choices == (3, 5, 10, 20, 40, 60) or name in {"ts_delay", "ts_delta", "ts_pct_change"}


def test_registry_underscore_names_and_types():
    assert get_op("abs").func is ops.abs_ and get_op("max").func is ops.max_ and get_op("min").func is ops.min_
    assert not any(n.endswith("_") for n in REGISTRY)
    assert get_op("ts_corr").arg_types == ("panel", "panel", "int")
    assert get_op("ts_cov").arg_types == ("panel", "panel", "int")
    assert get_op("power").arg_types == ("panel", "float")
    assert get_op("clip").arg_types == ("panel", "float", "float")
    assert get_op("cs_winsorize").arg_types == ("panel", "float")
    assert get_op("cs_quantile_clip").arg_types == ("panel", "float", "float")
    assert get_op("cs_regress_out").arg_types == ("panel", "panel")
    assert get_op("cs_neutralize").arg_types == ("panel", "group", "panel")
    for name in GROUP_OPS:
        assert get_op(name).arg_types == ("panel", "group") and get_op(name).category == "group"
    for name in TS_ALL:
        assert get_op(name).category == "ts" and get_op(name).arg_types[-1] == "int"
    for name in ["add", "sub", "mul", "div", "neg", "abs", "clip"]:
        assert get_op(name).category == "elementwise"


def test_registry_flags():
    assert {n for n, s in REGISTRY.items() if s.commutative} == {"add", "mul", "max", "min"}
    assert {n for n, s in REGISTRY.items() if not s.mineable} == {
        "clip",
        "cs_quantile_clip",
        "cs_neutralize",
        "cs_regress_out",
    }
    mineable = list_ops(mineable_only=True)
    assert len(mineable) == 41 and all(s.mineable for s in mineable)
    assert [s.name for s in list_ops("ts")] == sorted(TS_ALL)
    assert len(list_ops("group")) == 3
    assert [s.name for s in list_ops()] == sorted(REGISTRY)
    with pytest.raises(ValueError):
        list_ops("nope")


def test_registry_lookup_and_idempotency():
    with pytest.raises(KeyError):
        get_op("does_not_exist")
    with pytest.raises(KeyError):
        get_op("abs_")
    before = dict(REGISTRY)
    register_all_builtin()
    register_all_builtin(force=True)
    assert before == REGISTRY and len(REGISTRY) == 45


def test_opspec_is_frozen_and_validated(x):
    spec = get_op("ts_mean")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]
    pd.testing.assert_frame_equal(spec(x, D), ops.ts_mean(x, D))
    with pytest.raises(ValueError):
        OpSpec("bad", ops.ts_mean, ("panel", "string"), "ts")
    with pytest.raises(ValueError):
        OpSpec("bad", ops.ts_mean, ("panel", "int"), "weird")
    with pytest.raises(ValueError):
        OpSpec("bad", ops.ts_mean, ("panel", "int"), "ts", commutative=True)
    with pytest.raises(ValueError):
        ops.register(get_op("add"), overwrite=False)


# ============================================================================
# performance (target 0.5 s; 1.0 s threshold to avoid flaky failures)
# ============================================================================


@pytest.fixture(scope="module")
def big():
    rng = np.random.default_rng(0)
    x = make_panel(rng, n=1000, m=300, nan_frac=0.02)
    y = make_panel(rng, n=1000, m=300, nan_frac=0.02)
    g = make_groups(rng, x, n_groups=15)
    return x, y, g


PERF_CASES = [name for name in TS_ALL] + GROUP_OPS + ["cs_rank", "cs_zscore", "cs_winsorize", "cs_neutralize"]


@pytest.mark.parametrize("name", PERF_CASES)
def test_performance_1000x300(name, big):
    x, y, g = big
    d = 60

    def fn():
        return call_ts(name, x, y, d) if name in TS_ALL else call_cs(name, x, g, y)

    fn()  # warm-up (no JIT anywhere, but excludes one-off import/alloc effects)
    t0 = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - t0
    assert out.shape == x.shape
    assert elapsed < 1.0, f"{name} took {elapsed:.3f}s on 1000x300 (target < 0.5 s)"
