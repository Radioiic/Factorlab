"""Integrity tests for the event-driven diffusion factor layer.

Synthetic data only, no network, no disk. The two that matter are
``test_future_prices_do_not_change_past_diffusion`` (the point-in-time
guarantee, proved by mutating the future and demanding the past is bit-identical)
and ``test_no_self_diffusion`` (an asset must never re-receive its own shock,
which would turn the factor into a disguised own-momentum term).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from diffusion import DiffusionSpec, _shrinkage_correlation, build, graph_weights, shock_series


def _prices(n_days=260, n_assets=6, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    common = rng.normal(0, 0.02, n_days)
    cols = {}
    for a in range(n_assets):
        r = 0.6 * common + rng.normal(0, 0.02, n_days)
        cols[f"A{a}-USDT"] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=idx)


def test_no_self_diffusion():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 5))
    corr = np.corrcoef(x.T)
    for kernel in ("corr", "invdist", "knn"):
        w = graph_weights(corr, kernel, 3)
        assert np.allclose(np.diag(w), 0.0), f"{kernel}: self-weight is not zero"
        s = w.sum(axis=1)
        assert np.allclose(s[s > 0], 1.0), f"{kernel}: rows are not stochastic"


def test_market_wide_move_creates_no_events():
    """A day when everything moves together is a market move, not N shocks."""
    idx = pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC")
    rng = np.random.default_rng(5)
    base = pd.DataFrame(rng.normal(0, 0.005, (80, 4)), index=idx, columns=[f"A{i}" for i in range(4)])
    base.iloc[60] = 0.25  # identical 25% move in every asset
    sh = shock_series(base, z_window=30, threshold=2.0, mode="signed")
    assert np.allclose(sh.iloc[60].to_numpy(), 0.0), "a perfectly common move was recorded as idiosyncratic events"


def test_threshold_gates_contribution():
    idx = pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    r = pd.DataFrame(rng.normal(0, 0.01, (80, 4)), index=idx, columns=[f"A{i}" for i in range(4)])
    loose = shock_series(r, 30, 0.5, "abs").to_numpy().sum()
    tight = shock_series(r, 30, 4.0, "abs").to_numpy().sum()
    assert loose > tight, "raising the threshold did not reduce recorded events"


def test_future_prices_do_not_change_past_diffusion():
    px = _prices()
    cut = px.index[-30]
    a = build(px[px.index <= cut], [DiffusionSpec()])
    tampered = px.copy()
    tampered.loc[tampered.index > cut] *= 3.0  # violently rewrite the future
    b = build(tampered, [DiffusionSpec()])
    b = b[b.date <= cut]
    key = ["date", "symbol"]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    assert len(m) == len(a) and len(m) > 0, "row sets diverged"
    col = DiffusionSpec().name()
    delta = float((m[f"{col}_a"] - m[f"{col}_b"]).abs().max())
    assert delta < 1e-12, f"future data leaked into past rows (max delta {delta:.2e})"


def test_deterministic_and_ranked():
    px = _prices(n_days=200, n_assets=5)
    specs = [DiffusionSpec(), DiffusionSpec(mode="abs")]
    a, b = build(px, specs), build(px, specs)
    pd.testing.assert_frame_equal(a, b)
    for s in specs:
        rk = a[f"{s.name()}_rank"].dropna()
        assert rk.between(0, 1).all(), "ranks escaped [0, 1]"
    assert not a.duplicated(["date", "symbol"]).any()


def test_spec_validation_rejects_nonsense():
    for bad in (
        DiffusionSpec(kernel="nope"),
        DiffusionSpec(mode="nope"),
        DiffusionSpec(decay=0.0),
        DiffusionSpec(lags=0),
        DiffusionSpec(threshold=-1.0),
    ):
        try:
            bad.validate()
        except ValueError:
            continue
        raise AssertionError(f"validate() accepted {bad}")


def _synthetic_prices(n_days=190, n_assets=6, seed=17):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-07", periods=n_days, freq="D", tz="UTC")
    market = rng.normal(0.0004, 0.018, n_days)
    sector_a = rng.normal(0.0, 0.010, n_days)
    sector_b = rng.normal(0.0, 0.012, n_days)
    returns = []
    for i in range(n_assets):
        sector = sector_a if i < n_assets // 2 else sector_b
        idio = rng.normal(0.0, 0.015 + i * 0.001, n_days)
        returns.append(0.65 * market + 0.25 * sector + idio)
    values = 100 * np.exp(np.cumsum(np.column_stack(returns), axis=0))
    return pd.DataFrame(values, index=dates, columns=[f"ASSET-{i}" for i in range(n_assets)])


def test_shrinkage_correlation_health():
    """The shrinkage correlation of a return window is finite, symmetric, has a unit diagonal and is
    positive semidefinite; a window that is too small or holds a missing value is refused."""
    prices = _synthetic_prices(n_days=100)
    returns = prices.pct_change(fill_method=None).dropna()
    corr = _shrinkage_correlation(returns.tail(60))
    assert np.isfinite(corr).all()
    assert np.allclose(corr, corr.T)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.linalg.eigvalsh(corr).min() >= -1e-10
    holed = returns.tail(60).copy()
    holed.iloc[3, 2] = np.nan
    for bad in (returns.tail(60).iloc[:, :2], holed, returns.tail(1)):
        try:
            _shrinkage_correlation(bad)
        except ValueError:
            continue
        raise AssertionError("a bad return window was accepted")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} diffusion tests passed")
    raise SystemExit(1 if failed else 0)
