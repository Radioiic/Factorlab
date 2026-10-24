"""Tests for alphaforge.combine: combine_linear / orient / MLCombiner."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from alphaforge.combine import MLCombiner, combine_linear, orient, resolve_weights
from alphaforge.evaluation.ic import rowwise_corr
from alphaforge.ops.cs import cs_rank, cs_zscore

N_DATES, N_ASSETS = 60, 20


def _dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, name="date")


def _assets(n: int) -> list[str]:
    return [f"S{i:04d}" for i in range(n)]


def _panel(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=_dates(values.shape[0]), columns=_assets(values.shape[1]))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def two(rng) -> dict[str, pd.DataFrame]:
    a = rng.standard_normal((N_DATES, N_ASSETS))
    b = rng.standard_normal((N_DATES, N_ASSETS))
    a[rng.random(a.shape) < 0.05] = np.nan
    b[rng.random(b.shape) < 0.05] = np.nan
    return {"a": _panel(a), "b": _panel(b)}


# --------------------------------------------------------------------------- #
# orient / combine_linear
# --------------------------------------------------------------------------- #
def test_orient(two):
    f = two["a"]
    pd.testing.assert_frame_equal(orient(f, -0.02), -f)
    pd.testing.assert_frame_equal(orient(f, 0.02), f)
    pd.testing.assert_frame_equal(orient(f, float("nan")), f)
    pd.testing.assert_frame_equal(orient(f, 0.0), f)
    assert orient(f, 0.02) is not f


def test_equal_weights_of_self_is_zscore(two):
    f = two["a"]
    out = combine_linear({"x": f, "y": f})
    pd.testing.assert_frame_equal(out, cs_zscore(f))
    pd.testing.assert_frame_equal(combine_linear({"x": f}), cs_zscore(f))


def test_orientation_flips_negative_ic(two):
    f = two["a"]
    table = pd.DataFrame({"ic_mean": [0.03, -0.03], "icir": [0.5, 0.5]}, index=pd.Index(["pos", "neg"], name="name"))
    pd.testing.assert_frame_equal(combine_linear({"neg": f}, ic_table=table), -cs_zscore(f))
    pd.testing.assert_frame_equal(combine_linear({"pos": f}, ic_table=table), cs_zscore(f))
    both = combine_linear({"pos": f, "neg": f}, ic_table=table)  # z and -z cancel
    np.testing.assert_allclose(both.to_numpy()[np.isfinite(both.to_numpy())], 0.0, atol=1e-12)
    # a factor missing from the table keeps its sign
    pd.testing.assert_frame_equal(combine_linear({"unknown": f}, ic_table=table), cs_zscore(f))
    # a "name" column instead of an index is accepted
    pd.testing.assert_frame_equal(combine_linear({"neg": f}, ic_table=table.reset_index()), -cs_zscore(f))


@pytest.mark.parametrize("scheme,col", [("icir", "icir"), ("ic", "ic_mean")])
def test_stat_weights_are_proportional(two, scheme, col):
    a, b = two["a"], two["b"]
    table = pd.DataFrame({"ic_mean": [0.04, -0.02], "icir": [2.0, -1.0]}, index=pd.Index(["a", "b"], name="name"))
    w = resolve_weights(["a", "b"], scheme, table)
    expect = np.abs(table[col].to_numpy())
    np.testing.assert_allclose(w.to_numpy(), expect / expect.sum())
    assert w.sum() == pytest.approx(1.0)
    out = combine_linear({"a": a, "b": b}, weights=scheme, ic_table=table)
    za, zb = cs_zscore(a), -cs_zscore(b)  # b has negative IC -> flipped
    wa, wb = w["a"], w["b"]
    both = za.notna() & zb.notna()
    expected = (wa * za + wb * zb) / (wa + wb)
    pd.testing.assert_frame_equal(out[both], expected[both])
    only_a = za.notna() & zb.isna()
    pd.testing.assert_frame_equal(out[only_a], za[only_a])  # re-normalised over available factors
    only_b = za.isna() & zb.notna()
    pd.testing.assert_frame_equal(out[only_b], zb[only_b])
    assert out[za.isna() & zb.isna()].isna().all().all()


def test_explicit_weights_and_validation(two):
    a, b = two["a"], two["b"]
    w = resolve_weights(["a", "b"], {"a": 3.0, "b": 1.0})
    np.testing.assert_allclose(w.to_numpy(), [0.75, 0.25])
    out = combine_linear({"a": a, "b": b}, weights={"a": 3.0, "b": 1.0})
    za, zb = cs_zscore(a), cs_zscore(b)
    both = za.notna() & zb.notna()
    pd.testing.assert_frame_equal(out[both], (0.75 * za + 0.25 * zb)[both])
    # names missing from the mapping get weight 0
    pd.testing.assert_frame_equal(combine_linear({"a": a, "b": b}, weights={"a": 1.0}), za)
    np.testing.assert_allclose(resolve_weights(["a", "b"], "equal").to_numpy(), [0.5, 0.5])
    with pytest.raises(ValueError):
        resolve_weights(["a", "b"], {"a": -1.0, "b": 1.0})
    with pytest.raises(ValueError):
        resolve_weights(["a", "b"], {"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError):
        resolve_weights(["a", "b"], "sharpe")
    with pytest.raises(ValueError):
        combine_linear({"a": a}, weights="icir")  # no ic_table
    table = pd.DataFrame({"ic_mean": [0.04], "icir": [2.0]}, index=pd.Index(["a"], name="name"))
    with pytest.raises(ValueError, match="no row"):
        combine_linear({"a": a, "b": b}, weights="icir", ic_table=table)
    with pytest.raises(ValueError):
        combine_linear({})
    with pytest.raises(TypeError):
        combine_linear({"a": a.to_numpy()})  # type: ignore[dict-item]


def test_union_alignment_and_no_mutation(two):
    a, b = two["a"], two["b"]
    a_before = a.copy()
    sub = b.iloc[10:30, :10]
    out = combine_linear({"a": a, "b": sub})
    assert out.index.equals(a.index) and out.columns.equals(a.columns)
    za = cs_zscore(a)
    zb = cs_zscore(sub).reindex(index=a.index, columns=a.columns)
    both = za.notna() & zb.notna()
    pd.testing.assert_frame_equal(out[both], (0.5 * za + 0.5 * zb)[both])
    pd.testing.assert_frame_equal(out[zb.isna()], za[zb.isna()])
    pd.testing.assert_frame_equal(a, a_before)


# --------------------------------------------------------------------------- #
# MLCombiner
# --------------------------------------------------------------------------- #
ML_DATES, ML_ASSETS, HORIZON, LAG = 160, 30, 5, 1
TRAIN_WINDOW, REFIT_EVERY, MIN_TRAIN = 60, 10, 30


@pytest.fixture
def ml_data(rng) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Three random factors; label = cs_rank(f1) + noise, NaN on the unknown tail."""
    f1 = _panel(rng.standard_normal((ML_DATES, ML_ASSETS)))
    f2 = _panel(rng.standard_normal((ML_DATES, ML_ASSETS)))
    f3 = _panel(rng.standard_normal((ML_DATES, ML_ASSETS)))
    label = cs_rank(f1) + 0.3 * rng.standard_normal((ML_DATES, ML_ASSETS))
    label.iloc[-(LAG + HORIZON) :] = np.nan
    return {"f1": f1, "f2": f2, "f3": f3}, label


def _combiner(model: str, **kw) -> MLCombiner:
    return MLCombiner(
        train_window=TRAIN_WINDOW, refit_every=REFIT_EVERY, min_train=MIN_TRAIN, model=model, seed=0, **kw
    )


@pytest.mark.parametrize("model", ["ridge", "hgb"])
def test_predictions_start_after_first_refit(ml_data, model, caplog):
    factors, label = ml_data
    m = _combiner(model)
    with caplog.at_level(logging.INFO, logger="alphaforge.combine.ml"):
        pred = m.fit_predict(factors, label, horizon=HORIZON, lag=LAG)
    assert pred.shape == label.shape and pred.index.equals(label.index) and pred.columns.equals(label.columns)
    first = MIN_TRAIN - 1 + LAG + HORIZON  # first date with >= min_train realised labels
    assert m.refit_dates_[0] == label.index[first]
    assert pred.iloc[: first + 1].isna().all().all()  # NaN up to and including the refit date
    assert pred.iloc[first + 1 :].notna().all().all()  # every later date is predicted
    expected_refits = list(range(first, ML_DATES, REFIT_EVERY))
    assert [label.index.get_loc(d) for d in m.refit_dates_] == expected_refits
    assert len(m.refit_history_) == len(expected_refits)
    assert all(h["n_rows"] <= TRAIN_WINDOW * ML_ASSETS for h in m.refit_history_)
    assert sum("refit at" in r.getMessage() for r in caplog.records) == len(expected_refits)


@pytest.mark.parametrize("model", ["ridge", "hgb"])
def test_no_look_ahead(ml_data, model):
    """Perturbing labels/features that were not realised by t leaves predictions <= t unchanged."""
    factors, label = ml_data
    base = _combiner(model).fit_predict(factors, label, horizon=HORIZON, lag=LAG)
    t = 80
    prng = np.random.default_rng(123)
    # label rows >= t - lag - horizon are realised only after t: they must not matter for preds <= t
    bad_label = label.copy()
    cut = t - LAG - HORIZON
    bad_label.iloc[cut:] = prng.standard_normal(bad_label.iloc[cut:].shape)
    bad_factors = {n: f.copy() for n, f in factors.items()}
    for f in bad_factors.values():  # features strictly after t are unknown at t
        f.iloc[t + 1 :] = prng.standard_normal(f.iloc[t + 1 :].shape)
    other = _combiner(model).fit_predict(bad_factors, bad_label, horizon=HORIZON, lag=LAG)
    pd.testing.assert_frame_equal(base.iloc[: t + 1], other.iloc[: t + 1])
    assert not np.allclose(base.iloc[t + 1 :].to_numpy(), other.iloc[t + 1 :].to_numpy(), equal_nan=True)


@pytest.mark.parametrize("model", ["hgb", "ridge"])
def test_learns_the_signal(ml_data, model):
    factors, label = ml_data
    m = _combiner(model)
    pred = m.fit_predict(factors, label, horizon=HORIZON, lag=LAG)
    ic = rowwise_corr(pred, label, method="spearman").dropna()
    assert len(ic) > 100
    assert ic.mean() > 0.4  # clearly positive: label ~ rank(f1) + noise
    assert (ic > 0).mean() > 0.9
    imp = m.feature_importances_
    assert isinstance(imp, pd.Series) and list(imp.index) == ["f1", "f2", "f3"]
    assert imp.idxmax() == "f1" and imp["f1"] > 0
    assert m.last_model_ is not None


def test_determinism_and_model_kwargs(ml_data):
    factors, label = ml_data
    p1 = _combiner("hgb").fit_predict(factors, label, horizon=HORIZON, lag=LAG)
    p2 = _combiner("hgb").fit_predict(factors, label, horizon=HORIZON, lag=LAG)
    pd.testing.assert_frame_equal(p1, p2)
    m = _combiner("hgb", max_iter=5, max_depth=2)
    assert m.model_kwargs == {"max_iter": 5, "max_depth": 2}
    p3 = m.fit_predict(factors, label, horizon=HORIZON, lag=LAG)
    assert m.last_model_.max_iter == 5 and m.last_model_.max_depth == 2
    assert not p3.equals(p1)
    r = _combiner("ridge", alpha=10.0)
    r.fit_predict(factors, label, horizon=HORIZON, lag=LAG)
    assert r.last_model_.alpha == 10.0


def test_missing_features_and_labels(ml_data):
    factors, label = ml_data
    factors = {n: f.copy() for n, f in factors.items()}
    factors["f2"].iloc[:, 0] = np.nan  # one feature always missing for asset 0
    for f in factors.values():
        f.iloc[100, 5] = np.nan  # every feature missing for one cell
    label = label.copy()
    label.iloc[10:15] = np.nan  # five early dates without any label (inside the first min_train window)
    for model in ("ridge", "hgb"):
        m = _combiner(model)
        pred = m.fit_predict(factors, label, horizon=HORIZON, lag=LAG)
        first = m.refit_dates_[0]
        assert pred.loc[first:].iloc[1:, 0].notna().all()  # partial features still predicted
        assert np.isnan(pred.iloc[100, 5])  # no features at all -> NaN
        assert np.isfinite(pred.iloc[100, 6])
        # missing label dates delay the first refit by the number of unusable dates
        assert label.index.get_loc(first) == MIN_TRAIN - 1 + 5 + LAG + HORIZON


def test_never_enough_training_data(ml_data, caplog):
    factors, label = ml_data
    m = MLCombiner(train_window=50, refit_every=5, min_train=1000, model="ridge")
    with caplog.at_level(logging.WARNING, logger="alphaforge.combine.ml"):
        pred = m.fit_predict(factors, label, horizon=HORIZON, lag=LAG)
    assert pred.isna().all().all() and m.refit_dates_ == [] and m.feature_importances_ is None
    assert any("min_train" in r.getMessage() for r in caplog.records)


def test_invalid_arguments(ml_data):
    factors, label = ml_data
    with pytest.raises(ValueError):
        MLCombiner(model="xgboost")
    with pytest.raises(ValueError):
        MLCombiner(refit_every=0)
    with pytest.raises(ValueError):
        MLCombiner(train_window=0)
    m = MLCombiner(model="ridge")
    with pytest.raises(ValueError):
        m.fit_predict(factors, label, horizon=0, lag=1)
    with pytest.raises(ValueError):
        m.fit_predict(factors, label, horizon=5, lag=-1)
    with pytest.raises(ValueError):
        m.fit_predict({}, label, horizon=5, lag=1)
    with pytest.raises(TypeError):
        m.fit_predict(factors, label.to_numpy(), horizon=5, lag=1)  # type: ignore[arg-type]
