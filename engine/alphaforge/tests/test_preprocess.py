"""Tests for alphaforge.preprocess.

Self-contained: a small synthetic ``DataStore`` (close, market_cap, industry)
built with numpy; no mining / research modules involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.data import DataStore
from alphaforge.evaluation.ic import rowwise_corr
from alphaforge.ops.cs import cs_winsorize, cs_zscore
from alphaforge.preprocess import log_market_cap, preprocess_factor

N_DATES, N_ASSETS, N_IND = 60, 20, 4
NO_COVERAGE_ROW = 7  # raw factor is all-NaN here
EMPTY_UNIVERSE_ROW = 3  # universe excludes every name here
OUTLIER = (10, 0)  # (row, col) of an injected outlier


def _dates(n: int = N_DATES) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n, name="date")


def _assets(n: int = N_ASSETS) -> list[str]:
    return [f"S{i:04d}" for i in range(n)]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def store(rng) -> DataStore:
    idx, cols = _dates(), _assets()
    rets = rng.normal(0.0, 0.02, size=(N_DATES, N_ASSETS))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    shares = np.exp(rng.normal(3.0, 1.0, size=N_ASSETS))  # log-normal size dispersion
    mcap = close * shares
    industry = pd.DataFrame(np.tile(np.arange(N_ASSETS) % N_IND, (N_DATES, 1)).astype(float), index=idx, columns=cols)
    beta = pd.DataFrame(rng.normal(1.0, 0.3, size=(N_DATES, N_ASSETS)), index=idx, columns=cols)
    return DataStore({"close": close, "market_cap": mcap, "industry": industry, "beta": beta})


@pytest.fixture
def universe(rng) -> pd.DataFrame:
    u = np.where(rng.random((N_DATES, N_ASSETS)) < 0.85, 1.0, np.nan)
    u[:, N_ASSETS - 1] = np.nan  # last name never investable
    u[EMPTY_UNIVERSE_ROW, :] = np.nan
    return pd.DataFrame(u, index=_dates(), columns=_assets())


@pytest.fixture
def raw(rng, store) -> pd.DataFrame:
    """Raw factor with size + industry exposure, ~5% NaN, an outlier and an empty row."""
    size = np.log(store.get("market_cap").to_numpy())
    ind = store.get("industry").to_numpy()
    x = rng.standard_normal((N_DATES, N_ASSETS)) + 0.5 * size + 0.8 * ind
    x[rng.random(x.shape) < 0.05] = np.nan
    x[OUTLIER] = 500.0
    x[NO_COVERAGE_ROW, :] = np.nan
    return pd.DataFrame(x, index=_dates(), columns=_assets())


def _valid_rows(out: pd.DataFrame, min_valid: int = 3) -> np.ndarray:
    return (out.notna().sum(axis=1) >= min_valid).to_numpy()


# --------------------------------------------------------------------------- #
# full pipeline
# --------------------------------------------------------------------------- #
def test_output_is_row_standardised(raw, store, universe):
    out = preprocess_factor(raw, store, universe=universe)
    rows = _valid_rows(out)
    assert rows.sum() > 50
    np.testing.assert_allclose(out[rows].mean(axis=1).to_numpy(), 0.0, atol=1e-10)
    np.testing.assert_allclose(out[rows].std(axis=1, ddof=0).to_numpy(), 1.0, atol=1e-10)
    assert out.index.equals(raw.index) and out.columns.equals(raw.columns)
    assert (out.dtypes == np.float64).all()


def test_industry_means_are_zero(raw, store, universe):
    out = preprocess_factor(raw, store, universe=universe)
    ind = store.get("industry")
    rows = _valid_rows(out)
    for g in range(N_IND):
        grp = out.where(ind == float(g))
        means = grp[rows].mean(axis=1).dropna()
        assert len(means) > 0
        np.testing.assert_allclose(means.to_numpy(), 0.0, atol=1e-9)


def test_size_exposure_is_regressed_out(raw, store, universe):
    out = preprocess_factor(raw, store, universe=universe)
    size = log_market_cap(store, out)
    corr = rowwise_corr(out, size, method="pearson").dropna()
    assert len(corr) > 50
    np.testing.assert_allclose(corr.to_numpy(), 0.0, atol=1e-8)
    # and the raw factor really was size-loaded, so the test has teeth
    raw_corr = rowwise_corr(raw.where(universe.notna()), size, method="pearson").dropna()
    assert raw_corr.mean() > 0.3


def test_nan_pattern_respects_universe(raw, store, universe):
    out = preprocess_factor(raw, store, universe=universe)
    in_u = universe.notna()
    # nothing outside the universe survives
    assert out[~in_u].isna().all().all()
    # inside the universe every cell is filled on rows with coverage (incl. raw-NaN cells)
    rows = [i for i in range(N_DATES) if i not in (NO_COVERAGE_ROW, EMPTY_UNIVERSE_ROW)]
    assert out.iloc[rows][in_u.iloc[rows]].notna().sum().sum() == in_u.iloc[rows].sum().sum()
    raw_missing_in_u = raw.isna() & in_u
    raw_missing_in_u.iloc[NO_COVERAGE_ROW] = False
    assert raw_missing_in_u.sum().sum() > 0
    assert out[raw_missing_in_u].notna().sum().sum() == raw_missing_in_u.sum().sum()
    # rows without coverage / without universe stay all-NaN
    assert out.iloc[NO_COVERAGE_ROW].isna().all()
    assert out.iloc[EMPTY_UNIVERSE_ROW].isna().all()


def test_no_coverage_row_stays_nan_without_universe(raw, store):
    out = preprocess_factor(raw, store)
    assert out.iloc[NO_COVERAGE_ROW].isna().all()
    other = out.drop(out.index[NO_COVERAGE_ROW])
    assert other.notna().all().all()  # close known everywhere -> everything filled


def test_winsorize_clips_outlier(raw, store):
    x = raw.where(raw.notna())
    mad = preprocess_factor(x, store, winsor="mad", fill=None, neutralize=(), standardize=False)
    ref = cs_winsorize(x, k=3.0)
    pd.testing.assert_frame_equal(mad, ref)
    r, c = OUTLIER
    assert mad.iat[r, c] < raw.iat[r, c]
    assert mad.iat[r, c] == ref.iat[r, c]

    none = preprocess_factor(x, store, winsor=None, fill=None, neutralize=(), standardize=False)
    pd.testing.assert_frame_equal(none, x)

    q = preprocess_factor(x, store, winsor=(0.05, 0.95), fill=None, neutralize=(), standardize=False)
    hi = x.quantile(0.95, axis=1)
    lo = x.quantile(0.05, axis=1)
    assert (q.le(hi + 1e-12, axis=0) | q.isna()).all().all()
    assert (q.ge(lo - 1e-12, axis=0) | q.isna()).all().all()
    assert q.isna().equals(x.isna())

    k1 = preprocess_factor(x, store, winsor="mad", winsor_k=1.0, fill=None, neutralize=(), standardize=False)
    pd.testing.assert_frame_equal(k1, cs_winsorize(x, k=1.0))


def test_fill_methods(raw, store, universe):
    masked = raw.where(universe.notna())
    zero = preprocess_factor(raw, store, universe=universe, winsor=None, fill="zero", neutralize=(), standardize=False)
    med = preprocess_factor(
        raw, store, universe=universe, winsor=None, fill="cs_median", neutralize=(), standardize=False
    )
    target = masked.isna() & universe.notna()
    target.iloc[NO_COVERAGE_ROW] = False
    assert target.sum().sum() > 0
    assert (zero[target] == 0.0).sum().sum() == target.sum().sum()
    row_median = masked.median(axis=1)
    expected = masked.where(~target, row_median, axis=0)
    pd.testing.assert_frame_equal(med, expected)
    # untouched cells are exactly the masked input
    pd.testing.assert_frame_equal(zero[~target], masked[~target])
    # no-coverage row is not filled by either method
    assert zero.iloc[NO_COVERAGE_ROW].isna().all() and med.iloc[NO_COVERAGE_ROW].isna().all()


def test_all_steps_off_is_identity(raw, store):
    out = preprocess_factor(raw, store, winsor=None, fill=None, neutralize=(), standardize=False)
    pd.testing.assert_frame_equal(out, raw)
    assert out is not raw


def test_standardize_only_matches_cs_zscore(raw, store):
    out = preprocess_factor(raw, store, winsor=None, fill=None, neutralize=(), standardize=True)
    pd.testing.assert_frame_equal(out, cs_zscore(raw))


def test_industry_only_and_size_only(raw, store):
    x = raw.where(raw.notna())
    ind_only = preprocess_factor(x, store, winsor=None, fill=None, neutralize=("industry",), standardize=False)
    industry = store.get("industry")
    for g in range(N_IND):
        means = ind_only.where(industry == float(g)).mean(axis=1).dropna()
        np.testing.assert_allclose(means.to_numpy(), 0.0, atol=1e-9)
    size_only = preprocess_factor(x, store, winsor=None, fill=None, neutralize=("size",), standardize=False)
    corr = rowwise_corr(size_only, log_market_cap(store, x), method="pearson").dropna()
    np.testing.assert_allclose(corr.to_numpy(), 0.0, atol=1e-8)


def test_extra_store_field_is_regressed_out(raw, store, universe):
    out = preprocess_factor(raw, store, universe=universe, neutralize=("industry", "size", "beta"))
    corr = rowwise_corr(out, store.get("beta"), method="pearson").dropna()
    assert len(corr) > 50
    np.testing.assert_allclose(corr.to_numpy(), 0.0, atol=1e-8)
    with pytest.raises(ValueError, match="neutralize item"):
        preprocess_factor(raw, store, neutralize=("industry", "no_such_field"))


def test_missing_exposure_gives_nan(raw, store, universe):
    """A name with unknown industry / market cap cannot be neutralised -> NaN."""
    mcap = store.get("market_cap").copy()
    mcap.iloc[:, 0] = np.nan
    ind = store.get("industry").copy()
    ind.iloc[:, 1] = np.nan
    store.add("market_cap", mcap, overwrite=True)
    store.add("industry", ind, overwrite=True)
    out = preprocess_factor(raw, store, universe=universe)
    assert out.iloc[:, 0].isna().all() and out.iloc[:, 1].isna().all()
    rows = _valid_rows(out)
    np.testing.assert_allclose(out[rows].mean(axis=1).to_numpy(), 0.0, atol=1e-10)


def test_sub_range_factor_is_supported(raw, store, universe):
    sub = raw.iloc[10:40, :15]
    out = preprocess_factor(sub, store, universe=universe)
    assert out.index.equals(sub.index) and out.columns.equals(sub.columns)
    full = preprocess_factor(raw.iloc[:, :15], store, universe=universe)
    pd.testing.assert_frame_equal(out, full.iloc[10:40])


def test_no_look_ahead_and_no_mutation(raw, store, universe):
    before = raw.copy()
    out = preprocess_factor(raw, store, universe=universe)
    pd.testing.assert_frame_equal(raw, before)
    t = 30
    perturbed = raw.copy()
    perturbed.iloc[t + 1 :] = np.random.default_rng(1).standard_normal(perturbed.iloc[t + 1 :].shape)
    out2 = preprocess_factor(perturbed, store, universe=universe)
    pd.testing.assert_frame_equal(out.iloc[: t + 1], out2.iloc[: t + 1])


def test_invalid_options_raise(raw, store):
    with pytest.raises(ValueError):
        preprocess_factor(raw, store, winsor="sigma")
    with pytest.raises(ValueError):
        preprocess_factor(raw, store, winsor=(0.9, 0.1))
    with pytest.raises(ValueError):
        preprocess_factor(raw, store, fill="mean")
    with pytest.raises(TypeError):
        preprocess_factor(raw.to_numpy(), store)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        preprocess_factor(raw, DataStore({"close": store.get("close")}))  # no industry / market_cap
