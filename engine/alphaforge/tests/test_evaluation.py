"""Tests for alphaforge.evaluation.

Self-contained: a duck-typed ``FakeStore`` built from a synthetic close panel
stands in for ``alphaforge.data.DataStore`` (which is developed concurrently).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from alphaforge.evaluation import (
    FactorReport,
    assign_quantiles,
    evaluate_factor,
    ic_decay,
    ic_series,
    ic_stats,
    long_short_returns,
    monotonicity,
    quantile_returns,
    quantile_turnover,
    rank_autocorr,
    rowwise_corr,
)

N_DATES, N_ASSETS = 60, 20


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
class FakeStore:
    """Minimal stand-in for DataStore: .dates, .assets, .fwd_return, .get."""

    def __init__(self, close: pd.DataFrame):
        self._close = close
        self.meta: dict = {}

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._close.index

    @property
    def assets(self) -> pd.Index:
        return self._close.columns

    def fwd_return(self, horizon: int = 5, lag: int = 1) -> pd.DataFrame:
        c = self._close
        return c.shift(-(lag + horizon)) / c.shift(-lag) - 1  # the DataStore label convention

    def get(self, name: str) -> pd.DataFrame:
        if name == "close":
            return self._close
        raise KeyError(name)


def _dates(n: int = N_DATES) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n, name="date")


def _assets(n: int = N_ASSETS) -> list[str]:
    return [f"S{i:04d}" for i in range(n)]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def store(rng) -> FakeStore:
    rets = rng.normal(0.0, 0.02, size=(N_DATES, N_ASSETS))
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return FakeStore(pd.DataFrame(close, index=_dates(), columns=_assets()))


@pytest.fixture
def noisy_pair(rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two random panels with ~10% NaN each, plus a few degenerate rows."""
    a = rng.standard_normal((N_DATES, N_ASSETS))
    b = 0.5 * a + rng.standard_normal((N_DATES, N_ASSETS))
    a[rng.random(a.shape) < 0.10] = np.nan
    b[rng.random(b.shape) < 0.10] = np.nan
    a[3, :] = np.nan  # all-NaN row
    a[4, 4:] = np.nan  # only 4 valid pairs -> NaN
    a[5, :] = 1.5  # constant row -> NaN
    A = pd.DataFrame(a, index=_dates(), columns=_assets())
    B = pd.DataFrame(b, index=_dates(), columns=_assets())
    return A, B


# --------------------------------------------------------------------------- #
# rowwise_corr / ic_series
# --------------------------------------------------------------------------- #
def _reference_rowwise(A: pd.DataFrame, B: pd.DataFrame, method: str) -> np.ndarray:
    out = np.full(len(A), np.nan)
    for i in range(len(A)):
        x, y = A.iloc[i].to_numpy(), B.iloc[i].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 5 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
            continue
        if method == "pearson":
            out[i] = stats.pearsonr(x[m], y[m])[0]
        else:
            out[i] = stats.spearmanr(x[m], y[m])[0]
    return out


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_rowwise_corr_matches_scipy_per_row(noisy_pair, method):
    A, B = noisy_pair
    if method == "spearman":  # introduce ties to exercise average ranking
        A = A.round(1)
    got = rowwise_corr(A, B, method=method)
    ref = _reference_rowwise(A, B, method)
    assert isinstance(got, pd.Series) and got.index.equals(A.index)
    np.testing.assert_allclose(got.to_numpy(), ref, rtol=1e-10, atol=1e-12, equal_nan=True)
    # degenerate rows
    assert np.isnan(got.iloc[3]) and np.isnan(got.iloc[4]) and np.isnan(got.iloc[5])
    assert got.dropna().between(-1, 1).all()


def test_rowwise_corr_edge_cases(noisy_pair):
    A, B = noisy_pair
    A_copy, B_copy = A.copy(), B.copy()
    # inf is treated as missing
    A2 = A.copy()
    A2.iloc[10, 0] = np.inf
    A3 = A.copy()
    A3.iloc[10, 0] = np.nan
    assert rowwise_corr(A2, B).iloc[10] == pytest.approx(rowwise_corr(A3, B).iloc[10])
    # inputs are not mutated
    pd.testing.assert_frame_equal(A, A_copy)
    pd.testing.assert_frame_equal(B, B_copy)
    # misaligned inputs are inner-aligned
    B_sub = B.iloc[5:40, 2:]
    got = rowwise_corr(A, B_sub)
    assert got.index.equals(B_sub.index)
    ref = rowwise_corr(A.loc[B_sub.index, B_sub.columns], B_sub)
    pd.testing.assert_series_equal(got, ref)
    # min_periods is honoured
    assert rowwise_corr(A, B, min_periods=17).iloc[4:].isna().sum() > got.isna().sum()
    with pytest.raises(ValueError):
        rowwise_corr(A, B, method="kendall")


def test_ic_series_perfect_factor(store):
    h, lag = 5, 1
    fwd = store.fwd_return(h, lag)
    for method in ("spearman", "pearson"):
        ic_pos = ic_series(fwd, fwd, method=method)
        ic_neg = ic_series(-fwd, fwd, method=method)
        assert ic_pos.name == "ic" and ic_pos.index.equals(store.dates)
        # label unknown on the last lag + h dates -> NaN, never dropped
        assert ic_pos.iloc[-(lag + h) :].isna().all()
        assert ic_pos.iloc[: -(lag + h)].notna().all()
        np.testing.assert_allclose(ic_pos.dropna().to_numpy(), 1.0, atol=1e-12)
        np.testing.assert_allclose(ic_neg.dropna().to_numpy(), -1.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# ic_stats / ic_decay
# --------------------------------------------------------------------------- #
def test_ic_stats_matches_scipy_ttest(rng):
    ic = pd.Series(rng.normal(0.03, 0.1, size=200), index=_dates(200))
    ic.iloc[[5, 17, 99]] = np.nan
    st = ic_stats(ic, periods_per_year=252)
    x = ic.dropna()
    ref = stats.ttest_1samp(x, 0.0)
    assert set(st) == {"mean", "std", "ir", "ir_ann", "t_stat", "p_value", "positive_ratio", "n", "abs_mean"}
    assert st["n"] == 197
    assert st["mean"] == pytest.approx(x.mean())
    assert st["std"] == pytest.approx(x.std(ddof=1))
    assert st["ir"] == pytest.approx(x.mean() / x.std(ddof=1))
    assert st["ir_ann"] == pytest.approx(st["ir"] * math.sqrt(252))
    assert st["t_stat"] == pytest.approx(ref.statistic, rel=1e-12)
    assert st["p_value"] == pytest.approx(ref.pvalue, rel=1e-10)
    assert st["positive_ratio"] == pytest.approx((x > 0).mean())
    assert st["abs_mean"] == pytest.approx(x.abs().mean())


def test_ic_stats_degenerate_inputs():
    empty = ic_stats(pd.Series([np.nan, np.nan], dtype=float))
    assert empty["n"] == 0 and math.isnan(empty["mean"]) and math.isnan(empty["p_value"])
    two = ic_stats(pd.Series([0.1, 0.2]))
    assert two["n"] == 2 and math.isfinite(two["t_stat"]) and math.isnan(two["p_value"])
    const = ic_stats(pd.Series([0.1] * 10))
    assert const["mean"] == pytest.approx(0.1) and math.isnan(const["ir"]) and math.isnan(const["t_stat"])


def test_ic_decay_peaks_at_true_horizon(store):
    factor = store.fwd_return(3, 1)
    decay = ic_decay(factor, store, horizons=(1, 2, 3, 5), lag=1)
    assert isinstance(decay, pd.Series) and decay.index.name == "horizon"
    assert list(decay.index) == [1, 2, 3, 5]
    assert decay.loc[3] == pytest.approx(1.0, abs=1e-12)
    assert (decay.drop(3) < 1.0).all()
    assert decay.loc[3] == decay.max()


# --------------------------------------------------------------------------- #
# quantiles
# --------------------------------------------------------------------------- #
def test_assign_quantiles_labels_and_nan_pattern(rng):
    f = rng.standard_normal((N_DATES, N_ASSETS))
    f[rng.random(f.shape) < 0.15] = np.nan
    f[7, :] = np.nan
    F = pd.DataFrame(f, index=_dates(), columns=_assets())
    q = assign_quantiles(F, 10)
    assert q.shape == F.shape and q.index.equals(F.index) and q.columns.equals(F.columns)
    pd.testing.assert_frame_equal(q.isna(), F.isna())
    # pandas >= 3 keeps NaN cells in stack(); the NaN pattern is asserted just above,
    # so the label check runs on the non-missing labels only.
    vals = q.stack().dropna().unique()
    assert set(vals) <= set(np.arange(1.0, 11.0))
    # highest factor -> n, lowest -> 1, per date
    valid = F.notna().sum(axis=1) > 0
    Fv, qv = F.loc[valid], q.loc[valid]
    rows = np.arange(len(Fv))
    top_col = Fv.columns.get_indexer(Fv.idxmax(axis=1))
    bot_col = Fv.columns.get_indexer(Fv.idxmin(axis=1))
    assert (qv.to_numpy()[rows, top_col] == 10.0).all()
    assert (qv.to_numpy()[rows, bot_col] == 1.0).all()
    assert (qv.max(axis=1) == 10.0).all() and (qv.min(axis=1) == 1.0).all()
    # bucket never decreases with the factor value on a date
    for t in F.index[valid][:10]:
        row = F.loc[t].dropna().sort_values()
        assert q.loc[t, row.index].is_monotonic_increasing


def test_assign_quantiles_balanced_and_exact():
    # 20 distinct values, 10 buckets -> exactly two names per bucket every date
    F = pd.DataFrame(np.tile(np.arange(20.0), (5, 1)), index=_dates(5), columns=_assets())
    q = assign_quantiles(F, 10)
    counts = q.apply(lambda r: r.value_counts().sort_index(), axis=1)
    assert (counts == 2).all().all()
    # rank r of 10 with 10 buckets must give exactly r (guards ceil(0.7*10) == 8)
    F10 = pd.DataFrame(np.arange(1.0, 11.0)[None, :], index=_dates(1), columns=_assets(10))
    np.testing.assert_array_equal(assign_quantiles(F10, 10).to_numpy()[0], np.arange(1.0, 11.0))
    # ties share a bucket
    Ft = pd.DataFrame([[1.0, 1.0, 1.0, 5.0]], index=_dates(1), columns=_assets(4))
    np.testing.assert_array_equal(assign_quantiles(Ft, 2).to_numpy()[0], [1.0, 1.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        assign_quantiles(F, 1)


def test_assign_quantiles_is_cross_sectional_only(rng):
    F = pd.DataFrame(rng.standard_normal((N_DATES, N_ASSETS)), index=_dates(), columns=_assets())
    base = assign_quantiles(F, 5)
    G = F.copy()
    G.iloc[30:] = rng.standard_normal((N_DATES - 30, N_ASSETS))
    pd.testing.assert_frame_equal(assign_quantiles(G, 5).iloc[:30], base.iloc[:30])


def test_quantile_returns_matches_manual_groupby(store, rng):
    fwd = store.fwd_return(5, 1)
    f = rng.standard_normal(fwd.shape)
    f[rng.random(f.shape) < 0.1] = np.nan
    F = pd.DataFrame(f, index=fwd.index, columns=fwd.columns)
    n = 5
    qret = quantile_returns(F, fwd, n)
    assert list(qret.columns) == list(range(1, n + 1)) and qret.index.equals(fwd.index)
    q = assign_quantiles(F, n)
    for t in fwd.index:
        manual = fwd.loc[t].groupby(q.loc[t]).mean().reindex(range(1, n + 1))
        np.testing.assert_allclose(qret.loc[t].to_numpy(), manual.to_numpy(), equal_nan=True, atol=1e-14)
    # last lag + h dates have no label
    assert qret.iloc[-6:].isna().all().all()
    ls = long_short_returns(F, fwd, n)
    assert ls.name == "long_short"
    pd.testing.assert_series_equal(ls, (qret[n] - qret[1]).rename("long_short"))


def test_perfect_factor_is_monotone(store):
    fwd = store.fwd_return(5, 1)
    qret = quantile_returns(fwd, fwd, 10)
    valid = qret.dropna()
    assert len(valid) > 0
    assert (valid.diff(axis=1).iloc[:, 1:] > 0).all().all()
    assert monotonicity(qret) == pytest.approx(1.0)
    assert monotonicity(quantile_returns(-fwd, fwd, 10)) == pytest.approx(-1.0)
    assert (long_short_returns(fwd, fwd, 10).dropna() > 0).all()


def test_monotonicity_degenerate():
    idx = _dates(3)
    assert math.isnan(monotonicity(pd.DataFrame({1: [0.1, 0.2, 0.3]}, index=idx)))
    assert math.isnan(monotonicity(pd.DataFrame({1: [0.1] * 3, 2: [0.1] * 3}, index=idx)))
    hump = pd.DataFrame({1: [0.0] * 3, 2: [1.0] * 3, 3: [0.0] * 3}, index=idx)
    assert -1.0 < monotonicity(hump) < 1.0


# --------------------------------------------------------------------------- #
# turnover
# --------------------------------------------------------------------------- #
def test_rank_autocorr_constant_over_time(rng):
    row = rng.standard_normal(N_ASSETS)
    F = pd.DataFrame(np.tile(row, (N_DATES, 1)), index=_dates(), columns=_assets())
    for lag in (1, 3):
        ac = rank_autocorr(F, lag)
        assert ac.name == "rank_autocorr" and ac.index.equals(F.index)
        assert ac.iloc[:lag].isna().all()
        np.testing.assert_allclose(ac.iloc[lag:].to_numpy(), 1.0, atol=1e-12)
    # a fully re-shuffled ranking every day has |autocorr| well below 1
    R = pd.DataFrame(rng.standard_normal((N_DATES, N_ASSETS)), index=_dates(), columns=_assets())
    ac = rank_autocorr(R)
    assert ac.iloc[1:].between(-1, 1).all() and abs(ac.mean()) < 0.5
    with pytest.raises(ValueError):
        rank_autocorr(F, 0)


def test_quantile_turnover_bounds_and_manual(rng):
    f = rng.standard_normal((N_DATES, N_ASSETS))
    f[rng.random(f.shape) < 0.1] = np.nan
    F = pd.DataFrame(f, index=_dates(), columns=_assets())
    n = 5
    to = quantile_turnover(F, n)
    assert to.name == f"turnover_q{n}" and to.index.equals(F.index)
    assert np.isnan(to.iloc[0])
    assert to.dropna().between(0.0, 1.0).all()
    q = assign_quantiles(F, n)
    for i in range(1, N_DATES):
        cur = set(q.columns[q.iloc[i].to_numpy() == n])
        prev = set(q.columns[q.iloc[i - 1].to_numpy() == n])
        assert to.iloc[i] == pytest.approx(len(cur - prev) / len(cur))
    # bottom bucket selectable
    to1 = quantile_turnover(F, n, quantile=1)
    assert to1.name == "turnover_q1" and to1.dropna().between(0, 1).all()
    with pytest.raises(ValueError):
        quantile_turnover(F, n, quantile=n + 1)


def test_quantile_turnover_extremes(rng):
    row = rng.standard_normal(N_ASSETS)
    F = pd.DataFrame(np.tile(row, (N_DATES, 1)), index=_dates(), columns=_assets())
    assert (quantile_turnover(F, 4).iloc[1:] == 0.0).all()
    # alternate between two disjoint top halves -> turnover 1 every day
    a = np.r_[np.zeros(10), np.ones(10)]
    alt = np.vstack([a if i % 2 == 0 else 1 - a for i in range(N_DATES)])
    F2 = pd.DataFrame(alt + rng.random(alt.shape) * 0.1, index=_dates(), columns=_assets())
    assert (quantile_turnover(F2, 2).iloc[1:] == 1.0).all()
    # warm-up (all-NaN rows) -> NaN, not a spurious 1.0
    F3 = F.copy()
    F3.iloc[:5] = np.nan
    to = quantile_turnover(F3, 4)
    assert to.iloc[:6].isna().all() and (to.iloc[6:] == 0.0).all()


# --------------------------------------------------------------------------- #
# evaluate_factor / FactorReport
# --------------------------------------------------------------------------- #
def _noisy_factor(store: FakeStore, rng: np.random.Generator, h: int = 5, lag: int = 1) -> pd.DataFrame:
    fwd = store.fwd_return(h, lag)
    noise = rng.standard_normal(fwd.shape) * fwd.stack().std()
    f = fwd.fillna(0.0) + noise  # defined on every date, including the label-less tail
    f[rng.random(f.shape) < 0.05] = np.nan
    return f


def test_evaluate_factor_scalars_finite_and_consistent(store, rng):
    h, n = 5, 5
    F = _noisy_factor(store, rng, h)
    rep = evaluate_factor(F, store, name="noisy", expr="fwd + noise", horizon=h, lag=1, n_quantiles=n)
    assert isinstance(rep, FactorReport)
    d = rep.to_dict()
    assert set(d) == set(FactorReport.scalar_fields())
    assert not any(isinstance(v, (pd.Series, pd.DataFrame)) for v in d.values())
    float_keys = [k for k, v in d.items() if isinstance(v, float)]
    assert all(math.isfinite(d[k]) for k in float_keys), {k: d[k] for k in float_keys if not math.isfinite(d[k])}
    assert rep.start == store.dates[0] and rep.end == store.dates[-1]
    assert rep.ic_mean > 0.3 and rep.rank_ic_mean > 0.3 and rep.ic_p < 1e-6
    # annualisation
    ls = rep.ls_series.dropna()
    assert rep.ls_ann_return == pytest.approx(ls.mean() * 252 / h)
    assert rep.ls_ann_vol == pytest.approx(ls.std(ddof=1) * math.sqrt(252 / h))
    assert rep.ls_sharpe == pytest.approx(rep.ls_ann_return / rep.ls_ann_vol)
    assert rep.top_q_ann_return == pytest.approx(rep.quantile_returns[n].mean() * 252 / h)
    assert rep.bottom_q_ann_return == pytest.approx(rep.quantile_returns[1].mean() * 252 / h)
    pd.testing.assert_series_equal(
        rep.ls_series, (rep.quantile_returns[n] - rep.quantile_returns[1]).rename("long_short")
    )
    equity = (ls / h).cumsum()
    assert rep.ls_max_dd == pytest.approx((equity - equity.cummax().clip(lower=0)).min())
    assert rep.ls_max_dd <= 0.0
    # icir / stats consistency
    assert rep.icir == pytest.approx(rep.ic_mean / rep.ic_std)
    assert rep.icir_ann == pytest.approx(rep.icir * math.sqrt(252))
    assert rep.ic_t == pytest.approx(stats.ttest_1samp(rep.ic_series.dropna(), 0).statistic)
    assert rep.rank_icir == pytest.approx(ic_stats(rep.rank_ic_series)["ir"])
    # coverage
    fwd = store.fwd_return(h, 1)
    ratio = (F.notna().sum(1) / fwd.notna().sum(1)).replace([np.inf], np.nan)
    assert rep.coverage == pytest.approx(ratio.mean()) and 0.9 < rep.coverage <= 1.0
    assert rep.turnover_mean == pytest.approx(quantile_turnover(F, n).mean())
    assert rep.autocorr_mean == pytest.approx(rank_autocorr(F).mean())
    assert list(rep.ic_decay.index) == [1, 2, 3, 5, 10, 20]
    # summary / table / json
    s = rep.summary()
    assert isinstance(s, str) and "noisy" in s and s.count("\n") >= 5
    tbl = FactorReport.table([rep, rep])
    assert list(tbl.index) == ["noisy", "noisy"] and "icir_ann" in tbl.columns and "ic_series" not in tbl.columns
    assert FactorReport.table([]).empty
    json.dumps(rep.to_dict(json_safe=True))


def test_evaluate_factor_window_slicing(store, rng):
    F = _noisy_factor(store, rng)
    start, end = "2020-02-03", "2020-03-06"
    rep = evaluate_factor(F, store, horizon=5, start=start, end=end, n_quantiles=4)
    assert rep.start == pd.Timestamp(start) and rep.end == pd.Timestamp(end)
    expected = store.dates[(store.dates >= start) & (store.dates <= end)]
    assert rep.ic_series.index.equals(expected)
    assert rep.rank_ic_series.index.equals(expected)
    assert rep.quantile_returns.index.equals(expected)
    assert rep.ls_series.index.equals(expected)
    # the slice equals the evaluation of the pre-sliced factor
    rep2 = evaluate_factor(F.loc[start:end], store, horizon=5, n_quantiles=4)
    assert rep2.to_dict() == rep.to_dict() or all(
        (a == b) or (isinstance(a, float) and isinstance(b, float) and math.isclose(a, b, rel_tol=1e-12))
        for a, b in zip(rep.to_dict().values(), rep2.to_dict().values())
    )
    # inclusive bounds on a date not in the index still work
    rep3 = evaluate_factor(F, store, start="2020-02-01", end="2020-03-08")
    assert rep3.start == pd.Timestamp(start) and rep3.end == pd.Timestamp(end)
    with pytest.raises(ValueError):
        evaluate_factor(F, store, start="2030-01-01")
    with pytest.raises(ValueError):
        evaluate_factor(F, store, horizon=0)


def test_evaluate_factor_perfect_factor(store):
    fwd = store.fwd_return(5, 1)
    rep = evaluate_factor(fwd, store, name="oracle", horizon=5, lag=1, n_quantiles=10)
    assert rep.ic_mean == pytest.approx(1.0) and rep.rank_ic_mean == pytest.approx(1.0)
    assert rep.monotonicity == pytest.approx(1.0)
    assert rep.ls_ann_return > 0 and rep.top_q_ann_return > rep.bottom_q_ann_return
    assert rep.ic_decay.loc[5] == pytest.approx(1.0)
    # IC undefined once the label is unknown -> ICIR still finite on the known part
    assert rep.ic_series.iloc[-6:].isna().all()
    assert math.isfinite(rep.turnover_mean) and 0 <= rep.turnover_mean <= 1


def test_evaluate_factor_ic_uses_only_row_t(store, rng):
    """Row t of every per-date series depends only on row t of the factor."""
    F = _noisy_factor(store, rng)
    rep = evaluate_factor(F, store, horizon=3)
    G = F.copy()
    G.iloc[40:] = rng.standard_normal(G.iloc[40:].shape)
    rep2 = evaluate_factor(G, store, horizon=3)
    pd.testing.assert_series_equal(rep.ic_series.iloc[:40], rep2.ic_series.iloc[:40])
    pd.testing.assert_frame_equal(rep.quantile_returns.iloc[:40], rep2.quantile_returns.iloc[:40])
    pd.testing.assert_series_equal(rep.rank_ic_series.iloc[:40], rep2.rank_ic_series.iloc[:40])
