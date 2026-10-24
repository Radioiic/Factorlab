"""Tests for alphaforge.portfolio (construct / backtest / optimizer).

Self-contained: a tiny duck-typed ``FakeStore`` stands in for ``DataStore`` so the tests
do not depend on ``alphaforge.data`` (built concurrently).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from alphaforge.portfolio import (
    BacktestResult,
    backtest,
    cap_and_redistribute,
    hold_weights,
    optimize_weights,
    rebalance_schedule,
    rolling_risk,
    weights_from_scores,
)

N_DATES = 80
N_ASSETS = 20


class FakeStore:
    """Minimal stand-in for ``alphaforge.data.DataStore`` (duck-typed)."""

    def __init__(self, fields: dict[str, pd.DataFrame], meta: dict | None = None):
        self._fields = dict(fields)
        self.meta = meta or {}

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._fields["close"].index

    @property
    def assets(self) -> pd.Index:
        return self._fields["close"].columns

    def get(self, name: str) -> pd.DataFrame:
        return self._fields[name]

    def has(self, name: str) -> bool:
        return name in self._fields


def _panel(values: np.ndarray, dates: pd.DatetimeIndex, assets: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(values, index=dates, columns=assets, dtype=float)


def make_store(n_dates: int = N_DATES, n_assets: int = N_ASSETS, seed: int = 0, halts: bool = False) -> FakeStore:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B", name="date")
    assets = pd.Index([f"S{i:04d}" for i in range(n_assets)])
    log_ret = rng.normal(0.0, 0.02, size=(n_dates, n_assets))
    close = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    tradable = np.ones((n_dates, n_assets))
    if halts:
        tradable[rng.random((n_dates, n_assets)) < 0.05] = 0.0
    close_p = _panel(close, dates, assets)
    returns = close_p.pct_change()
    fields = {
        "close": close_p,
        "tradable": _panel(tradable, dates, assets),
        "returns": returns,
        "industry": _panel(np.tile((np.arange(n_assets) // 5).astype(float), (n_dates, 1)), dates, assets),
    }
    return FakeStore(fields)


def fwd_1d(store: FakeStore, lag: int = 1) -> pd.DataFrame:
    close = store.get("close")
    return close.shift(-(lag + 1)) / close.shift(-lag) - 1.0


@pytest.fixture(scope="module")
def store() -> FakeStore:
    return make_store()


@pytest.fixture(scope="module")
def scores(store: FakeStore) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return _panel(rng.normal(size=(N_DATES, N_ASSETS)), store.dates, store.assets)


# ======================================================================================
# construct.weights_from_scores
# ======================================================================================
def test_legs_sum_to_plus_minus_one(scores):
    w = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    assert w.shape == scores.shape
    assert w.index.equals(scores.index) and w.columns.equals(scores.columns)
    assert not w.isna().any().any()
    long_sum = w.clip(lower=0).sum(axis=1)
    short_sum = w.clip(upper=0).sum(axis=1)
    assert np.allclose(long_sum, 1.0)
    assert np.allclose(short_sum, -1.0)
    assert np.allclose(w.abs().sum(axis=1), 2.0)
    # ceil(0.2 * 20) = 4 names per leg, equal weight 0.25
    assert ((w > 0).sum(axis=1) == 4).all()
    assert ((w < 0).sum(axis=1) == 4).all()
    # w[w > 0] masks every other cell to NaN, and pandas >= 3 keeps NaN in stack():
    # drop them so the check covers exactly the selected long names.
    assert np.allclose(w[w > 0].stack().dropna().to_numpy(), 0.25)
    # long names have higher scores than short names on every date
    top = scores.where(w > 0).min(axis=1)
    bottom = scores.where(w < 0).max(axis=1)
    assert (top > bottom).all()


def test_default_bottom_frac_equals_top_frac_and_asymmetric_legs(scores):
    w = weights_from_scores(scores, top_frac=0.1, max_weight=1.0)
    assert ((w > 0).sum(axis=1) == 2).all() and ((w < 0).sum(axis=1) == 2).all()
    w2 = weights_from_scores(scores, top_frac=0.1, bottom_frac=0.3, max_weight=1.0)
    assert ((w2 > 0).sum(axis=1) == 2).all() and ((w2 < 0).sum(axis=1) == 6).all()
    assert np.allclose(w2.clip(lower=0).sum(axis=1), 1.0)
    assert np.allclose(w2.clip(upper=0).sum(axis=1), -1.0)
    w3 = weights_from_scores(scores, top_frac=0.2, bottom_frac=0.0, max_weight=1.0)
    assert (w3 >= 0).all().all() and np.allclose(w3.sum(axis=1), 1.0)


def test_cap_respected_and_legs_still_sum_to_one_when_feasible(scores):
    # 10 names per leg, cap 0.1 -> exactly feasible
    w = weights_from_scores(scores, top_frac=0.5, max_weight=0.1)
    assert (w.abs() <= 0.1 + 1e-12).all().all()
    assert np.allclose(w.clip(lower=0).sum(axis=1), 1.0)
    assert np.allclose(w.clip(upper=0).sum(axis=1), -1.0)


def test_cap_is_hard_when_infeasible(scores, caplog):
    # 2 names per leg, cap 0.3 -> legs can only reach 0.6
    with caplog.at_level(logging.WARNING, logger="alphaforge.portfolio.construct"):
        w = weights_from_scores(scores, top_frac=0.1, max_weight=0.3)
    assert (w.abs() <= 0.3 + 1e-12).all().all()
    assert np.allclose(w.clip(lower=0).sum(axis=1), 0.6)
    assert np.allclose(w.clip(upper=0).sum(axis=1), -0.6)
    assert any("under-invested" in rec.getMessage() for rec in caplog.records)
    # max_weight=None disables the cap
    w_none = weights_from_scores(scores, top_frac=0.1, max_weight=None)
    assert np.allclose(w_none.clip(lower=0).sum(axis=1), 1.0)


def test_cap_and_redistribute_is_exact_water_filling():
    w = np.array([[0.5, 0.3, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25], [0.0, 0.0, 0.0, 0.0]])
    out = cap_and_redistribute(w, 0.3)
    assert np.allclose(out[0], [0.3, 0.3, 0.2, 0.2])
    assert np.allclose(out[1], 0.25)
    assert np.allclose(out[2], 0.0)
    # rescale to a target, then cap; infeasible row is capped hard
    out2 = cap_and_redistribute(np.array([[1.0, 3.0, 1.0], [1.0, 1.0, 1.0]]), 0.2, target=np.array([1.0, 1.0]))
    assert np.allclose(out2[0], [0.2, 0.2, 0.2]) and np.allclose(out2[1], 0.2)
    out3 = cap_and_redistribute(np.array([[0.7, 0.1, 0.1, 0.1]]), 0.25, target=np.array([1.0]))
    assert np.allclose(out3, 0.25) and np.isclose(out3.sum(), 1.0)
    with pytest.raises(ValueError):
        cap_and_redistribute(np.array([[-0.1, 0.5]]), 0.3)
    # NaN entries are treated as zero, input is not mutated
    src = np.array([[np.nan, 0.5, 0.5]])
    out4 = cap_and_redistribute(src, 1.0)
    assert np.isnan(src[0, 0]) and np.allclose(out4, [[0.0, 0.5, 0.5]])


def test_industry_neutral_selection_picks_from_each_industry(store):
    # scores increase with the column index -> the naive top-4 all sit in industry 3
    s = _panel(np.tile(np.arange(N_ASSETS, dtype=float), (N_DATES, 1)), store.dates, store.assets)
    industry = store.get("industry")  # 4 industries x 5 names
    w_naive = weights_from_scores(s, top_frac=0.2, max_weight=1.0)
    assert set(industry.iloc[0][w_naive.iloc[0] > 0]) == {3.0}
    w_neutral = weights_from_scores(s, top_frac=0.2, industry=industry, max_weight=1.0)
    for d in (0, N_DATES // 2, N_DATES - 1):
        row = w_neutral.iloc[d]
        ind = industry.iloc[d]
        assert sorted(ind[row > 0]) == [0.0, 1.0, 2.0, 3.0]
        assert sorted(ind[row < 0]) == [0.0, 1.0, 2.0, 3.0]
        # net industry exposure is zero by construction
        assert np.allclose(row.groupby(ind).sum(), 0.0)
    assert np.allclose(w_neutral.clip(lower=0).sum(axis=1), 1.0)
    assert np.allclose(w_neutral.clip(upper=0).sum(axis=1), -1.0)
    # within each industry the long name has the top score and the short name the bottom
    top_in_ind = s.iloc[0].groupby(industry.iloc[0]).idxmax()
    assert set(w_neutral.iloc[0][w_neutral.iloc[0] > 0].index) == set(top_in_ind)


def test_industry_nan_codes_are_excluded(scores, store):
    industry = store.get("industry").copy()
    industry["S0019"] = np.nan
    scores2 = scores.copy()
    scores2["S0019"] = 100.0  # would be the top pick if it were eligible
    w = weights_from_scores(scores2, top_frac=0.2, industry=industry, max_weight=1.0)
    assert (w["S0019"] == 0.0).all()


def test_long_only(scores):
    w = weights_from_scores(scores, top_frac=0.3, max_weight=1.0, long_only=True)
    assert (w >= 0).all().all()
    assert np.allclose(w.sum(axis=1), 1.0)
    assert ((w > 0).sum(axis=1) == 6).all()


def test_nan_score_handling(scores):
    s = scores.copy()
    s.iloc[5] = np.nan  # no scores at all
    s.iloc[6, :19] = np.nan  # single valid score
    s.iloc[7, 3] = np.nan  # one missing name on an otherwise normal date
    w = weights_from_scores(s, top_frac=0.2, max_weight=1.0)
    assert w.iloc[5].isna().all()
    assert w.iloc[6].notna().all()
    assert np.isclose(w.iloc[6].sum(), 1.0) and (w.iloc[6] >= 0).all()  # short leg cannot overlap
    assert w.iloc[7, 3] == 0.0 and w.iloc[7].notna().all()
    assert np.isclose(w.iloc[7].clip(lower=0).sum(), 1.0)
    assert w.dtypes.eq(float).all()


def test_ties_are_broken_deterministically_without_overlap(store):
    s = _panel(np.ones((N_DATES, N_ASSETS)), store.dates, store.assets)
    w = weights_from_scores(s, top_frac=0.3, max_weight=1.0)
    assert ((w > 0).sum(axis=1) == 6).all() and ((w < 0).sum(axis=1) == 6).all()
    assert not ((w > 0) & (w < 0)).any().any()
    assert np.allclose(w.sum(axis=1), 0.0)
    w2 = weights_from_scores(s, top_frac=0.5, max_weight=1.0)  # 10 + 10 = all names, still no overlap
    assert ((w2 != 0).sum(axis=1) == N_ASSETS).all()


def test_weights_are_row_wise_no_lookahead(scores):
    w_a = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    s2 = scores.copy()
    s2.iloc[40:] = np.random.default_rng(7).normal(size=s2.iloc[40:].shape)
    w_b = weights_from_scores(s2, top_frac=0.2, max_weight=1.0)
    pd.testing.assert_frame_equal(w_a.iloc[:40], w_b.iloc[:40])


def test_weights_from_scores_validation_and_immutability(scores):
    original = scores.copy()
    weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    pd.testing.assert_frame_equal(scores, original)
    with pytest.raises(ValueError):
        weights_from_scores(scores, top_frac=0.0)
    with pytest.raises(ValueError):
        weights_from_scores(scores, top_frac=0.6, bottom_frac=0.6)
    with pytest.raises(ValueError):
        weights_from_scores(scores, top_frac=0.2, max_weight=0.0)


# ======================================================================================
# construct.rebalance_schedule / hold_weights
# ======================================================================================
def test_rebalance_schedule(store):
    sched = rebalance_schedule(store.dates, every=5)
    assert isinstance(sched, pd.DatetimeIndex)
    assert sched.equals(store.dates[::5])
    assert sched[0] == store.dates[0]
    assert rebalance_schedule(store.dates, every=1).equals(store.dates)
    with pytest.raises(ValueError):
        rebalance_schedule(store.dates, every=0)


def test_hold_weights_forward_fills(scores):
    w = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    sched = scores.index[3::5]
    held = hold_weights(w, sched)
    assert held.shape == w.shape and held.index.equals(w.index)
    assert (held.iloc[:3] == 0.0).all().all()  # before first rebalance -> flat
    for i in range(3, N_DATES):
        last = 3 + ((i - 3) // 5) * 5
        pd.testing.assert_series_equal(held.iloc[i], w.iloc[last], check_names=False)
    # a schedule date without scores keeps the previous book
    w2 = w.copy()
    w2.iloc[8] = np.nan
    held2 = hold_weights(w2, sched)
    pd.testing.assert_series_equal(held2.iloc[8], w.iloc[3], check_names=False)
    pd.testing.assert_series_equal(held2.iloc[12], w.iloc[3], check_names=False)
    pd.testing.assert_series_equal(held2.iloc[13], w.iloc[13], check_names=False)
    assert not held2.isna().any().any()
    # schedule dates outside the panel are ignored
    held3 = hold_weights(w, sched.append(pd.DatetimeIndex(["2030-01-01"])))
    pd.testing.assert_frame_equal(held3, held)


# ======================================================================================
# backtest
# ======================================================================================
def test_perfect_foresight_backtest_is_positive(store):
    fwd = fwd_1d(store, lag=1)
    w = weights_from_scores(fwd, top_frac=0.25, max_weight=1.0)
    bt = backtest(w, store, cost_bps=0.0, lag=1)
    assert isinstance(bt, BacktestResult)
    assert (bt.gross_returns > 0).all()
    assert bt.ann_return > 0 and bt.sharpe > 0 and bt.hit_rate == 1.0
    assert bt.equity.is_monotonic_increasing
    assert bt.max_dd == 0.0 and np.isnan(bt.calmar)  # no drawdown -> Calmar undefined


def test_zero_cost_means_net_equals_gross_and_cost_formula(scores, store):
    w = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    bt0 = backtest(w, store, cost_bps=0.0)
    pd.testing.assert_series_equal(bt0.returns, bt0.gross_returns, check_names=False)
    assert bt0.total_cost == 0.0
    bt = backtest(w, store, cost_bps=10.0)
    expected_net = bt.gross_returns - 10.0 / 1e4 * bt.turnover
    pd.testing.assert_series_equal(bt.returns, expected_net, check_names=False)
    assert np.isclose(bt.total_cost, (10.0 / 1e4 * bt.turnover).sum())
    pd.testing.assert_series_equal(bt.gross_returns, bt0.gross_returns, check_names=False)
    assert bt.ann_return < bt0.ann_return
    np.testing.assert_allclose(bt.equity.to_numpy(), np.cumprod(1.0 + bt.returns.to_numpy()))


def test_turnover_of_constant_weights_is_zero_after_first_day(scores, store):
    w0 = weights_from_scores(scores, top_frac=0.2, max_weight=1.0).iloc[0]
    w = pd.DataFrame(np.tile(w0.to_numpy(), (N_DATES, 1)), index=store.dates, columns=store.assets)
    bt = backtest(w, store, cost_bps=10.0)
    assert np.isclose(bt.turnover.iloc[0], w0.abs().sum()) and np.isclose(bt.turnover.iloc[0], 2.0)
    assert np.allclose(bt.turnover.iloc[1:], 0.0)
    assert np.isclose(bt.total_cost, 10.0 / 1e4 * 2.0)
    assert np.isclose(bt.avg_turnover, 2.0 / bt.n_days)


def test_return_alignment_and_index_are_decision_dates(scores, store):
    w = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    close = store.get("close")
    for lag in (0, 1, 2):
        bt = backtest(w, store, cost_bps=0.0, lag=lag)
        # trailing lag+1 decision dates have no forward return and are dropped
        assert bt.returns.index.equals(store.dates[: N_DATES - lag - 1])
        assert bt.n_days == N_DATES - lag - 1 and bt.start == store.dates[0]
        t = 10
        r = close.iloc[t + lag + 1] / close.iloc[t + lag] - 1.0
        assert np.isclose(bt.gross_returns.iloc[t], (w.iloc[t] * r).sum())
    assert bt.returns.index.name == "date"


def test_backtest_has_no_lookahead_in_weights_usage(scores, store):
    """Return at t depends only on w_t (and prices after t); changing later weights leaves it unchanged."""
    w = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    bt_a = backtest(w, store, cost_bps=0.0)
    w2 = w.copy()
    w2.iloc[50:] = -w2.iloc[50:]
    bt_b = backtest(w2, store, cost_bps=0.0)
    pd.testing.assert_series_equal(bt_a.gross_returns.iloc[:50], bt_b.gross_returns.iloc[:50])
    pd.testing.assert_series_equal(bt_a.turnover.iloc[:50], bt_b.turnover.iloc[:50])


def test_untradable_names_get_zero_weight_and_legs_are_renormalised(scores):
    st = make_store()
    tradable = st.get("tradable").copy()
    tradable["S0000"] = 0.0
    tradable.iloc[20, 5] = 0.0
    close = st.get("close").copy()
    close["S0000"] = close["S0000"] * np.exp(np.linspace(0, 20, N_DATES))  # absurd returns on the halted name
    rigged = FakeStore({"close": close, "tradable": tradable, "returns": close.pct_change()})

    # weights that hold S0000 long on every date
    w = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    w["S0000"] = 0.25
    long_before = w.clip(lower=0).sum(axis=1)
    short_before = w.clip(upper=0).sum(axis=1)

    bt = backtest(w, rigged, cost_bps=0.0, lag=1)
    assert (bt.positions["S0000"] == 0.0).all()
    # tradable is checked at t+lag: untradable at row 20 blocks the decision made at row 19
    assert bt.positions.iloc[19]["S0005"] == 0.0
    # gross per leg preserved (renormalised), and the crazy return never enters the P&L
    idx = bt.positions.index
    assert np.allclose(bt.positions.clip(lower=0).sum(axis=1), long_before.loc[idx])
    assert np.allclose(bt.positions.clip(upper=0).sum(axis=1), short_before.loc[idx])
    assert bt.gross_returns.abs().max() < 0.5
    # manual replication: zero blocked names (tradable at t+lag == 0), rescale each leg, dot with r_{t+lag}
    blocked = tradable.shift(-1) == 0.0
    w_eff = w.mask(blocked, 0.0)
    long_after = w_eff.clip(lower=0).sum(axis=1)
    short_after = w_eff.clip(upper=0).sum(axis=1)
    w_eff = w_eff.clip(lower=0).mul(long_before / long_after, axis=0) + w_eff.clip(upper=0).mul(
        short_before / short_after, axis=0
    )
    r = close.shift(-2) / close.shift(-1) - 1.0
    expected = (w_eff * r).sum(axis=1).loc[idx]
    pd.testing.assert_series_equal(bt.gross_returns, expected, check_names=False)
    pd.testing.assert_frame_equal(bt.positions, w_eff.loc[idx], check_names=False)


def test_backtest_works_without_tradable_field(scores):
    st = make_store()
    st_no_trad = FakeStore({"close": st.get("close")})
    w = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    bt = backtest(w, st_no_trad, cost_bps=5.0)
    assert bt.n_days == N_DATES - 2


def test_leading_flat_rows_dropped_and_interior_nan_rows_flat(scores, store):
    w = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    w.iloc[:3] = np.nan  # warm-up
    w.iloc[3] = 0.0  # what hold_weights emits before the first rebalance
    w.iloc[10] = np.nan
    bt = backtest(w, store, cost_bps=10.0)
    assert bt.start == store.dates[4]
    assert bt.returns.index[0] == store.dates[4]
    assert bt.n_days == N_DATES - 4 - 2
    assert bt.gross_returns.loc[store.dates[10]] == 0.0
    assert np.isclose(bt.turnover.loc[store.dates[10]], w.iloc[9].abs().sum())  # unwind
    assert np.isclose(bt.turnover.loc[store.dates[11]], w.iloc[11].abs().sum())  # rebuild
    assert np.isclose(bt.turnover.iloc[0], 2.0)  # first live day pays for the full book
    with pytest.raises(ValueError):
        backtest(w.iloc[:4], store)
    # warm-up NaN scores -> hold_weights zeros -> backtest starts at the first rebalance with a book
    s = scores.copy()
    s.iloc[:12] = np.nan
    held = hold_weights(weights_from_scores(s, top_frac=0.2, max_weight=1.0), rebalance_schedule(store.dates, 5))
    assert (held.iloc[:15] == 0.0).all().all()
    assert backtest(held, store).start == store.dates[15]


def test_start_end_slice_uses_prices_beyond_end(scores, store):
    w = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    bt = backtest(w, store, start=store.dates[10], end=store.dates[30])
    assert bt.returns.index.equals(store.dates[10:31])
    bt_full = backtest(w, store)
    pd.testing.assert_series_equal(bt.gross_returns, bt_full.gross_returns.loc[store.dates[10] : store.dates[30]])
    # turnover restarts from cash on the first date of the window
    assert np.isclose(bt.turnover.iloc[0], w.iloc[10].abs().sum())
    # string bounds work as well
    bt_s = backtest(w, store, start=str(store.dates[10].date()), end=str(store.dates[30].date()))
    assert bt_s.n_days == 21


def test_hold_weights_then_backtest_only_trades_on_rebalance_days(scores, store):
    w = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    held = hold_weights(w, rebalance_schedule(store.dates, every=5))
    bt = backtest(held, store, cost_bps=10.0)
    on_sched = bt.turnover.index.isin(store.dates[::5])
    assert np.allclose(bt.turnover[~on_sched], 0.0)
    assert (bt.turnover[on_sched] > 0).all()


def test_backtest_result_summary_and_to_dict(scores, store):
    w = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    bt = backtest(w, store, cost_bps=10.0)
    text = bt.summary()
    assert isinstance(text, str) and "Sharpe" in text and "max drawdown" in text and "\n" in text
    d = bt.to_dict()
    assert isinstance(d, dict)
    for key in ("ann_return", "ann_vol", "sharpe", "max_dd", "calmar", "avg_turnover", "total_cost", "hit_rate"):
        assert key in d
        assert isinstance(d[key], float) and np.isfinite(d[key]), key
    assert d["n_days"] == bt.n_days == N_DATES - 2
    assert d["max_dd"] <= 0.0 and 0.0 <= d["hit_rate"] <= 1.0
    assert np.isclose(d["ann_return"], bt.returns.mean() * 252)
    assert np.isclose(d["ann_vol"], bt.returns.std(ddof=1) * np.sqrt(252))
    assert np.isclose(d["sharpe"], d["ann_return"] / d["ann_vol"])
    assert np.isclose(d["max_dd"], (bt.equity / bt.equity.cummax() - 1.0).min())
    assert np.isclose(d["calmar"], d["ann_return"] / abs(d["max_dd"]))
    assert np.isclose(d["total_return"], bt.equity.iloc[-1] - 1.0)
    assert "returns" not in d  # scalars only
    # dataclass repr must not dump whole series
    assert "Series" not in repr(bt) and "sharpe" in repr(bt)


def test_backtest_validation(scores, store):
    w = weights_from_scores(scores, top_frac=0.2, max_weight=1.0)
    with pytest.raises(ValueError):
        backtest(w, store, lag=-1)
    with pytest.raises(ValueError):
        backtest(w, store, cost_bps=-1.0)
    with pytest.raises(TypeError):
        backtest(w.to_numpy(), store)


# ======================================================================================
# optimizer
# ======================================================================================
@pytest.fixture(scope="module")
def risk(store) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return _panel(rng.uniform(0.01, 0.05, size=(N_DATES, N_ASSETS)), store.dates, store.assets)


def test_optimizer_gross_exposure_and_dollar_neutral(scores, risk):
    w = optimize_weights(scores, risk=risk, max_weight=None, gross=2.0)
    assert w.shape == scores.shape and not w.isna().any().any()
    assert np.allclose(w.abs().sum(axis=1), 2.0, atol=1e-9)
    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-9)
    # shape check: w is proportional to the demeaned score / risk^2
    raw = scores / risk**2
    raw = raw.sub(raw.mean(axis=1), axis=0)
    ratio = (w / raw).to_numpy()
    assert np.allclose(ratio, ratio[:, [0]], atol=1e-9)
    assert (ratio > 0).all()
    # risk_aversion does not change the shape when scaling to gross
    w2 = optimize_weights(scores, risk=risk, max_weight=None, gross=2.0, risk_aversion=1.0)
    pd.testing.assert_frame_equal(w, w2)
    # custom gross
    w3 = optimize_weights(scores, risk=risk, max_weight=None, gross=1.5)
    assert np.allclose(w3.abs().sum(axis=1), 1.5)


def test_optimizer_default_unit_risk_is_score_proportional(scores):
    w = optimize_weights(scores, max_weight=None)
    raw = scores.sub(scores.mean(axis=1), axis=0)
    ratio = (w / raw).to_numpy()
    assert np.allclose(ratio, ratio[:, [0]], atol=1e-9)


def test_optimizer_cap_feasible_keeps_gross_and_neutrality():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=30, freq="B", name="date")
    assets = pd.Index([f"S{i:04d}" for i in range(300)])
    s = _panel(rng.standard_t(3, size=(30, 300)), dates, assets)  # heavy tails so the cap binds
    r = _panel(rng.uniform(0.005, 0.05, size=(30, 300)), dates, assets)
    w = optimize_weights(s, risk=r, max_weight=0.02, gross=2.0)
    assert (w.abs() <= 0.02 + 1e-12).all().all()
    assert (w.abs() == 0.02).any().any()  # cap actually binds somewhere
    assert np.allclose(w.abs().sum(axis=1), 2.0, atol=1e-9)
    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-9)
    # NaN scores -> 0.0
    s2 = s.copy()
    s2.iloc[:, :10] = np.nan
    w2 = optimize_weights(s2, risk=r, max_weight=0.02, gross=2.0)
    assert (w2.iloc[:, :10] == 0.0).all().all()
    assert np.allclose(w2.abs().sum(axis=1), 2.0, atol=1e-9)


def test_optimizer_cap_infeasible_is_hard_and_stays_neutral(scores, risk, caplog):
    # 20 names x 0.02 = 0.4 < gross 2 -> cannot reach the target
    with caplog.at_level(logging.WARNING, logger="alphaforge.portfolio.optimizer"):
        w = optimize_weights(scores, risk=risk, max_weight=0.02, gross=2.0)
    assert (w.abs() <= 0.02 + 1e-12).all().all()
    assert (w.abs().sum(axis=1) < 2.0).all()
    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-9)
    assert any("under-invested" in rec.getMessage() for rec in caplog.records)


def test_optimizer_industry_neutral(scores, risk, store):
    industry = store.get("industry")
    w = optimize_weights(scores, risk=risk, industry=industry, max_weight=None)
    for d in (0, 17, N_DATES - 1):
        assert np.allclose(w.iloc[d].groupby(industry.iloc[d]).sum(), 0.0, atol=1e-9)
    assert np.allclose(w.abs().sum(axis=1), 2.0)
    industry2 = industry.copy()
    industry2["S0007"] = np.nan
    w2 = optimize_weights(scores, risk=risk, industry=industry2, max_weight=None)
    assert (w2["S0007"] == 0.0).all()


def test_optimizer_nan_rows_and_risk_handling(scores, risk):
    s = scores.copy()
    s.iloc[4] = np.nan
    r = risk.copy()
    r.iloc[6, 2] = np.nan
    r.iloc[6, 3] = 0.0
    w = optimize_weights(s, risk=r, max_weight=None)
    assert w.iloc[4].isna().all()
    assert w.iloc[6, 2] == 0.0 and w.iloc[6, 3] == 0.0
    assert np.isclose(w.iloc[6].abs().sum(), 2.0) and np.isclose(w.iloc[6].sum(), 0.0)
    assert w.dtypes.eq(float).all()
    # inputs untouched
    assert np.isnan(r.iloc[6, 2]) and r.iloc[6, 3] == 0.0


def test_optimizer_risk_from_store_has_no_lookahead(scores, store):
    w = optimize_weights(scores, store=store, risk_window=20, max_weight=None)
    # returns[0] is NaN, so a 20-day rolling std is first defined at row 20 -> rows 0..19 have no risk
    assert w.iloc[:20].isna().all().all()
    assert not w.iloc[20:].isna().any().any()
    rr = rolling_risk(store, 20)
    pd.testing.assert_frame_equal(rr, store.get("returns").rolling(20, min_periods=20).std())
    # perturb future returns: weights up to that point are unchanged
    close = store.get("close").copy()
    close.iloc[50:] *= np.exp(np.random.default_rng(9).normal(0, 0.1, size=close.iloc[50:].shape))
    st2 = FakeStore({"close": close, "returns": close.pct_change(), "tradable": store.get("tradable")})
    w2 = optimize_weights(scores, store=st2, risk_window=20, max_weight=None)
    pd.testing.assert_frame_equal(w.iloc[:50], w2.iloc[:50])
    assert not np.allclose(w.iloc[60:].to_numpy(), w2.iloc[60:].to_numpy())


def test_optimizer_not_dollar_neutral_and_natural_size(scores, risk):
    s_pos = scores.abs() + 0.1
    w = optimize_weights(s_pos, risk=risk, max_weight=None, dollar_neutral=False)
    assert (w >= 0).all().all()
    assert np.allclose(w.sum(axis=1), 2.0)
    # natural MV size is capped at gross rather than scaled to it
    w_nat = optimize_weights(scores, risk=risk, max_weight=None, scale_to_gross=False, risk_aversion=1e9)
    assert (w_nat.abs().sum(axis=1) < 2.0).all() and (w_nat.abs().sum(axis=1) > 0).all()
    w_big = optimize_weights(scores, risk=risk, max_weight=None, scale_to_gross=False, risk_aversion=1e-3)
    assert np.allclose(w_big.abs().sum(axis=1), 2.0)


def test_optimizer_validation(scores):
    with pytest.raises(ValueError):
        optimize_weights(scores, gross=0.0)
    with pytest.raises(ValueError):
        optimize_weights(scores, risk_aversion=0.0)
    with pytest.raises(ValueError):
        optimize_weights(scores, max_weight=-0.1)
    with pytest.raises(TypeError):
        optimize_weights(scores.to_numpy())


def test_optimizer_weights_feed_backtest(scores, risk, store):
    w = optimize_weights(scores, risk=risk, max_weight=0.2)
    bt = backtest(w, store, cost_bps=10.0)
    assert bt.n_days == N_DATES - 2
    assert np.isfinite(bt.sharpe)
