"""Tests for alphaforge.data (simulator, DataStore, universe).

Self-contained: only numpy/pandas and alphaforge.data are used.  A single 300 x 1000
simulation (the default size) is shared across the module; smaller sims cover the rest.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from alphaforge.data import (
    TRUE_ALPHA_KEYS,
    DataStore,
    SimConfig,
    apply_universe,
    build_universe,
    simulate_market,
)

REQUIRED_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "returns",
    "market_cap",
    "bp",
    "industry",
    "tradable",
}


# ----------------------------------------------------------------------------- helpers
def rowwise_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Per-date Spearman correlation: rank across assets, then row-wise Pearson."""
    both = a.notna() & b.notna()
    a = a.where(both)
    b = b.where(both)
    ra = a.rank(axis=1)
    rb = b.rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    ic = num / den.replace(0.0, np.nan)
    ic[both.sum(axis=1) < 5] = np.nan
    return ic


def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    """Row-wise z-score with population std (matches the simulator)."""
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1, ddof=0), axis=0)


def make_store(n_dates: int = 60, n_assets: int = 20, seed: int = 0) -> DataStore:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    assets = [f"S{i + 1:04d}" for i in range(n_assets)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_dates, n_assets)), axis=0)), dates, assets)
    volume = pd.DataFrame(np.exp(rng.normal(12, 0.5, (n_dates, n_assets))), dates, assets)
    tradable = pd.DataFrame(1.0, dates, assets)
    return DataStore({"close": close, "volume": volume, "amount": close * volume, "tradable": tradable})


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def big() -> dict:
    cfg = SimConfig()  # 300 assets x 1000 days, seed 42
    t0 = time.perf_counter()
    store = simulate_market(cfg)
    return {"cfg": cfg, "store": store, "elapsed": time.perf_counter() - t0}


@pytest.fixture(scope="module")
def small_cfg() -> SimConfig:
    return SimConfig(n_assets=40, n_days=180, n_industries=5, seed=7, suspension_prob=0.02)


@pytest.fixture(scope="module")
def small(small_cfg) -> DataStore:
    return simulate_market(small_cfg)


# ----------------------------------------------------------------------------- simulator
def test_simulation_speed(big):
    assert big["elapsed"] < 10.0, f"300x1000 simulation took {big['elapsed']:.1f}s"


def test_determinism_given_seed(small_cfg, small):
    again = simulate_market(small_cfg)
    for name in small.fields:
        pd.testing.assert_frame_equal(small.get(name), again.get(name))
    for k in TRUE_ALPHA_KEYS:
        pd.testing.assert_frame_equal(small.meta["true_alphas"][k], again.meta["true_alphas"][k])
    other = simulate_market(SimConfig(**{**asdict(small_cfg), "seed": small_cfg.seed + 1}))
    assert not np.allclose(small.get("close").to_numpy(), other.get("close").to_numpy())


def test_shapes_and_alignment(big):
    store, cfg = big["store"], big["cfg"]
    assert len(store.dates) == cfg.n_days and len(store.assets) == cfg.n_assets
    assert store.dates.name == "date"
    assert store.dates.is_monotonic_increasing and store.dates.is_unique
    assert store.fields == sorted(store.fields)
    assert set(store.fields) >= REQUIRED_FIELDS
    assert list(store.assets[:2]) == ["S0001", "S0002"]
    for name in store.fields:
        df = store.get(name)
        assert df.index.equals(store.dates) and df.index.name == "date"
        assert df.columns.equals(store.assets)
        assert (df.dtypes == np.dtype("float64")).all()
        assert store.has(name) and name in store
    for k in TRUE_ALPHA_KEYS:
        ta = store.meta["true_alphas"][k]
        assert ta.index.equals(store.dates) and ta.columns.equals(store.assets)
    assert store.meta["sim_config"] == asdict(cfg)
    assert store.meta["beta"].index.equals(store.assets)
    assert store.meta["idio_vol"].index.equals(store.assets)


def test_field_sanity(big):
    store = big["store"]
    close, open_, high, low, vwap = (store.get(n) for n in ("close", "open", "high", "low", "vwap"))
    assert (close > 0).all().all()
    assert (high >= np.maximum(open_, close) - 1e-12).all().all()
    assert (low <= np.minimum(open_, close) + 1e-12).all().all()
    assert ((vwap >= low - 1e-12) & (vwap <= high + 1e-12)).all().all()
    ret = store.get("returns")
    assert ret.iloc[0].isna().all()
    manual = close / close.shift(1) - 1
    pd.testing.assert_frame_equal(ret.iloc[1:], manual.iloc[1:])
    vol, amt = store.get("volume"), store.get("amount")
    pd.testing.assert_frame_equal(amt, vwap * vol)
    mcap = store.get("market_cap")
    pd.testing.assert_frame_equal(mcap, close * store.meta["shares"], check_names=False)
    ind = store.get("industry")
    assert ind.nunique().max() == 1  # constant per asset
    assert set(np.unique(ind.to_numpy())) <= set(float(i) for i in range(big["cfg"].n_industries))
    assert set(np.unique(store.get("tradable").to_numpy())) == {0.0, 1.0}
    assert (store.get("bp") > 0).all().all()


def test_fwd_return_formula_no_lookahead(small):
    close = small.get("close")
    for h, lag in [(1, 0), (5, 1), (10, 2)]:
        fwd = small.fwd_return(horizon=h, lag=lag)
        manual = close.shift(-(lag + h)) / close.shift(-lag) - 1
        pd.testing.assert_frame_equal(fwd, manual)
        assert fwd.iloc[-(lag + h) :].isna().all().all()
        assert small.fwd_return(horizon=h, lag=lag) is fwd  # cached
    with pytest.raises(ValueError):
        small.fwd_return(horizon=0)
    with pytest.raises(ValueError):
        small.fwd_return(horizon=1, lag=-1)


def test_true_alpha_ic_sign_and_magnitude(big):
    store, cfg = big["store"], big["cfg"]
    fwd = store.fwd_return(horizon=1, lag=0)
    coefs = {
        "reversal": cfg.alpha_reversal,
        "momentum": cfg.alpha_momentum,
        "volume": cfg.alpha_volume,
        "value": cfg.alpha_value,
    }
    assert set(store.meta["true_alphas"]) == set(TRUE_ALPHA_KEYS) == set(coefs)
    assert store.meta["true_alpha_coefs"] == coefs
    for k, coef in coefs.items():
        ic = rowwise_spearman(store.meta["true_alphas"][k], fwd).mean()
        assert np.sign(ic) == np.sign(coef), f"{k}: IC {ic:.4f} vs coef {coef}"
        assert abs(ic) > 0.02, f"{k}: |IC| = {abs(ic):.4f} <= 0.02"


def test_true_alphas_are_history_only_signals(big):
    """Each planted alpha equals a documented function of fields at dates <= t."""
    store = big["store"]
    ta = store.meta["true_alphas"]
    close, volume, bp = store.get("close"), store.get("volume"), store.get("bp")
    expected = {
        "reversal": cs_zscore(close / close.shift(5) - 1),
        "momentum": cs_zscore(close.shift(20) / close.shift(120) - 1),
        "volume": cs_zscore(np.log(volume / volume.rolling(20, min_periods=1).mean())),
        "value": cs_zscore(np.log(bp)),
    }
    first_valid = {"reversal": 5, "momentum": 120, "volume": 19, "value": 0}
    for k, exp in expected.items():
        got = ta[k]
        assert got.iloc[: first_valid[k]].isna().all().all()
        g, e = got.iloc[first_valid[k] :], exp.iloc[first_valid[k] :]
        assert g.isna().equals(e.isna())
        assert np.allclose(g.to_numpy(), e.to_numpy(), equal_nan=True, atol=1e-9)
    # z-scores are cross-sectional: mean 0, std 1 on every defined row
    row = ta["reversal"].iloc[100]
    assert abs(row.mean()) < 1e-9 and abs(row.std(ddof=0) - 1) < 1e-9


def test_point_in_time_bp_does_not_change_before_lag(big):
    store, cfg = big["store"], big["cfg"]
    lag = pd.Timedelta(days=cfg.fundamental_lag_days)
    bq: pd.DataFrame = store.meta["bvps_quarterly"]
    pit = store.get("bp") * store.get("close")  # recovers bvps_pit exactly (up to fp)
    dates = store.dates
    n_updates = n_embargo = 0
    for q in bq.index:
        if not (dates[0] < q + lag <= dates[-1]):
            continue  # visibility date outside the sample
        n_updates += 1
        first_visible = pit.loc[dates >= q + lag].iloc[0]
        assert np.allclose(first_visible.to_numpy(), bq.loc[q].to_numpy(), rtol=1e-9)
        if q <= dates[0]:
            continue
        before = pit.loc[dates < q].iloc[-1]
        embargo = pit.loc[(dates >= q) & (dates < q + lag)]
        assert len(embargo) > 20
        # nothing changes between quarter end and quarter end + lag ...
        assert np.allclose(embargo.to_numpy(), before.to_numpy()[None, :], rtol=1e-9)
        # ... and the number published after the lag is genuinely new
        assert not np.allclose(first_visible.to_numpy(), before.to_numpy(), rtol=1e-6)
        n_embargo += 1
    assert n_embargo >= 10
    # daily bvps_pit is piecewise constant: it changes only on visibility dates
    changes = (pit.diff().abs() > 1e-9 * pit.abs()).any(axis=1)
    assert changes.sum() == n_updates


def test_suspension_days(big):
    store, cfg = big["store"], big["cfg"]
    tradable, volume, amount = store.get("tradable"), store.get("volume"), store.get("amount")
    h = (tradable == 0.0).to_numpy()
    assert abs(h.mean() - cfg.suspension_prob) < 0.002 and h.sum() > 100
    v, a = volume.to_numpy(), amount.to_numpy()
    assert np.isnan(v[h]).all() and np.isfinite(v[~h]).all()
    assert np.isnan(a[h]).all() and np.isfinite(a[~h]).all()
    r = store.get("returns").to_numpy()
    assert (r[1:][h[1:]] == 0.0).all()
    close = store.get("close")
    d = close.diff().to_numpy()
    assert (d[1:][h[1:]] == 0.0).all()  # price flat
    c = close.to_numpy()
    for name in ("open", "high", "low", "vwap"):
        assert (store.get(name).to_numpy()[h] == c[h]).all()
    # the volume alpha is undefined on a halted day (no volume), so it contributes nothing
    assert np.isnan(store.meta["true_alphas"]["volume"].to_numpy()[h]).all()


# ----------------------------------------------------------------------------- DataStore
def test_store_add_get_has_alignment():
    store = make_store()
    with pytest.raises(KeyError):
        store.get("nope")
    assert not store.has("nope")
    # misaligned panel: shuffled subset of dates/assets, int columns -> reindexed & cast
    rng = np.random.default_rng(1)
    sub_dates = store.dates[::2][::-1]
    sub_assets = list(store.assets[:5])[::-1]
    raw = pd.DataFrame(rng.integers(0, 5, (len(sub_dates), 5)), index=sub_dates, columns=sub_assets)
    store.add("x", raw)
    x = store.get("x")
    assert x.index.equals(store.dates) and x.columns.equals(store.assets)
    assert (x.dtypes == np.dtype("float64")).all()
    assert x.loc[sub_dates, sub_assets].to_numpy().tolist() == raw.astype(float).to_numpy().tolist()
    assert x.loc[store.dates[1::2]].isna().all().all()
    assert x[store.assets[5:]].isna().all().all()
    assert raw.dtypes.iloc[0] != np.dtype("float64")  # input untouched
    with pytest.raises(ValueError):
        store.add("x", raw)
    store.add("x", raw * 2, overwrite=True)
    assert store.get("x").loc[sub_dates[0], sub_assets[0]] == 2 * raw.iloc[0, 0]
    assert "x" in store.fields and store.fields == sorted(store.fields)
    # overwriting close invalidates the label cache
    fwd_before = store.fwd_return(5, 1)
    store.add("close", store.get("close") * 1.01, overwrite=True)
    assert store.fwd_return(5, 1) is not fwd_before
    assert "DataStore(" in repr(store)


def test_store_constructor_uses_first_panel_grid():
    dates = pd.bdate_range("2021-01-01", periods=10)
    a = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
    b = pd.DataFrame(2.0, index=dates[2:], columns=["B", "C"])
    store = DataStore({"a": a, "b": b})
    assert list(store.assets) == ["A", "B"] and len(store.dates) == 10
    assert store.get("b")["A"].isna().all()
    assert store.get("b").iloc[:2].isna().all().all()
    assert store.get("b").loc[dates[2:], "B"].eq(2.0).all()
    with pytest.raises(ValueError):
        DataStore({"a": pd.concat([a, a])})  # duplicate dates


def test_save_load_roundtrip(tmp_path, small):
    root = tmp_path / "store"
    small.save(root)
    assert (root / "meta.json").exists()
    loaded = DataStore.load(root)
    assert loaded.fields == small.fields
    assert loaded.dates.equals(small.dates) and loaded.dates.name == "date"
    assert loaded.assets.equals(small.assets)
    for name in small.fields:
        pd.testing.assert_frame_equal(loaded.get(name), small.get(name))
    assert loaded.meta["sim_config"] == small.meta["sim_config"]
    assert loaded.meta["true_alpha_coefs"] == small.meta["true_alpha_coefs"]
    for k in TRUE_ALPHA_KEYS:
        pd.testing.assert_frame_equal(loaded.meta["true_alphas"][k], small.meta["true_alphas"][k])
    for k in ("beta", "idio_vol", "shares"):
        pd.testing.assert_series_equal(loaded.meta[k], small.meta[k])
    pd.testing.assert_frame_equal(loaded.meta["bvps_quarterly"], small.meta["bvps_quarterly"])
    pd.testing.assert_frame_equal(loaded.fwd_return(5, 1), small.fwd_return(5, 1))


def test_slice(small):
    start, end = "2018-03-01", "2018-06-30"
    sub = small.slice(start, end)
    assert sub.dates[0] >= pd.Timestamp(start) and sub.dates[-1] <= pd.Timestamp(end)
    assert sub.dates.equals(small.dates[(small.dates >= start) & (small.dates <= end)])
    assert sub.assets.equals(small.assets) and sub.fields == small.fields
    for name in small.fields:
        pd.testing.assert_frame_equal(sub.get(name), small.get(name).loc[start:end])
    # date-indexed meta panels follow the slice; asset-indexed meta is untouched
    for k in TRUE_ALPHA_KEYS:
        pd.testing.assert_frame_equal(sub.meta["true_alphas"][k], small.meta["true_alphas"][k].loc[start:end])
    pd.testing.assert_series_equal(sub.meta["beta"], small.meta["beta"])
    assert sub.meta["sim_config"] == small.meta["sim_config"]
    # labels are recomputed on the sliced close: tail is NaN, head matches the full store
    fwd_sub, fwd_full = sub.fwd_return(5, 1), small.fwd_return(5, 1)
    assert fwd_sub.iloc[-6:].isna().all().all()
    pd.testing.assert_frame_equal(fwd_sub.iloc[:-6], fwd_full.loc[sub.dates].iloc[:-6])
    # inclusive labels, Timestamp bounds and open ends
    ts_sub = small.slice(small.dates[10], small.dates[20])
    assert len(ts_sub.dates) == 11
    assert small.slice().dates.equals(small.dates)
    assert small.slice(end=small.dates[4]).dates.equals(small.dates[:5])
    assert small.slice(start=small.dates[-3]).dates.equals(small.dates[-3:])


# ----------------------------------------------------------------------------- universe
def test_build_universe_rules(big):
    store = big["store"]
    uni = build_universe(store, min_price=1.0, min_adv_quantile=0.1, adv_window=20)
    assert uni.index.equals(store.dates) and uni.columns.equals(store.assets)
    vals = uni.to_numpy()
    assert set(np.unique(vals[~np.isnan(vals)])) == {1.0}
    h = (store.get("tradable") == 0.0).to_numpy()
    assert np.isnan(vals[h]).all()
    in_frac = uni.notna().mean(axis=1)
    assert 0.85 <= in_frac.mean() <= 0.91  # ~10% ADV cut + halts
    assert (in_frac <= 0.905).all()
    # the ADV rule: excluded, non-halted names have ADV below the row quantile (or undefined)
    adv = store.get("amount").rolling(20, min_periods=1).mean().to_numpy()
    thr = np.nanquantile(adv, 0.1, axis=1)[:, None]
    excluded = np.isnan(vals) & ~h
    with np.errstate(invalid="ignore"):
        assert ((adv < thr) | np.isnan(adv))[excluded].all()
        assert (adv >= thr)[~np.isnan(vals)].all()
    # price rule
    strict = build_universe(store, min_price=25.0, min_adv_quantile=0.0).to_numpy()
    c = store.get("close").to_numpy()
    assert np.isnan(strict[c < 25.0]).all()
    assert (strict[(c >= 25.0) & ~h] == 1.0).all()
    # no filters -> everything but halts
    loose = build_universe(store, min_price=0.0, min_adv_quantile=0.0).to_numpy()
    assert np.isnan(loose).sum() == h.sum() and np.isnan(loose[h]).all()
    everything = build_universe(store, min_price=0.0, min_adv_quantile=0.0, require_tradable=False)
    assert everything.notna().all().all()


def test_universe_no_lookahead_and_apply():
    store = make_store(n_dates=80, n_assets=30, seed=3)
    uni = build_universe(store, min_price=0.0, min_adv_quantile=0.2, adv_window=10)
    t = 50
    amount = store.get("amount").copy()
    amount.iloc[t + 1 :] *= np.random.default_rng(9).uniform(0.01, 100, amount.iloc[t + 1 :].shape)
    tampered = DataStore({"close": store.get("close"), "amount": amount, "tradable": store.get("tradable")})
    uni2 = build_universe(tampered, min_price=0.0, min_adv_quantile=0.2, adv_window=10)
    pd.testing.assert_frame_equal(uni.iloc[: t + 1], uni2.iloc[: t + 1])
    assert not uni.iloc[t + 1 :].equals(uni2.iloc[t + 1 :])
    # apply_universe masks with NaN and aligns
    close = store.get("close")
    masked = apply_universe(close, uni)
    assert masked.isna().equals(uni.isna())
    keep = uni.notna().to_numpy()
    assert (masked.to_numpy()[keep] == close.to_numpy()[keep]).all()
    partial = apply_universe(close, uni.iloc[:, :10])  # missing columns -> not in universe
    assert partial.iloc[:, 10:].isna().all().all()
    assert close.notna().all().all()  # input untouched
