"""Tests for alphaforge.library.

Self-contained: synthetic panels only; a ``FactorReport`` is produced with
``evaluate_factor`` against a duck-typed store (evaluation is an earlier stage).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from alphaforge.evaluation import FactorReport, evaluate_factor
from alphaforge.library import FactorLibrary, mean_spearman_corr

N_DATES, N_ASSETS = 60, 20


class FakeStore:
    def __init__(self, close: pd.DataFrame):
        self._close = close

    def fwd_return(self, horizon: int = 5, lag: int = 1) -> pd.DataFrame:
        c = self._close
        return c.shift(-(lag + horizon)) / c.shift(-lag) - 1


def _dates(n: int = N_DATES) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n, name="date")


def _assets(n: int = N_ASSETS) -> list[str]:
    return [f"S{i:04d}" for i in range(n)]


def _panel(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=_dates(values.shape[0]), columns=_assets(values.shape[1]))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def factors(rng) -> dict[str, pd.DataFrame]:
    f1 = rng.standard_normal((N_DATES, N_ASSETS))
    f1[rng.random(f1.shape) < 0.05] = np.nan
    near = f1 + 1e-3 * rng.standard_normal(f1.shape)  # rank-corr ~ 1
    f3 = rng.standard_normal((N_DATES, N_ASSETS))  # independent
    return {"f1": _panel(f1), "near": _panel(near), "f3": _panel(f3), "neg": _panel(-f1)}


@pytest.fixture
def report(rng, factors) -> FactorReport:
    rets = rng.normal(0.0, 0.02, size=(N_DATES, N_ASSETS))
    close = _panel(100.0 * np.exp(np.cumsum(rets, axis=0)))
    return evaluate_factor(factors["f1"], FakeStore(close), name="f1", expr="cs_rank(x)", horizon=5, lag=1)


# --------------------------------------------------------------------------- #
# add / max_corr
# --------------------------------------------------------------------------- #
def test_add_accepts_independent_and_rejects_duplicates(factors):
    lib = FactorLibrary(max_corr=0.7)
    assert lib.max_corr(factors["f1"]) == (None, 0.0)
    assert lib.add("f1", "x", factors["f1"], tags=("seed",)) is True
    assert lib.add("near", "x+eps", factors["near"]) is False  # corr ~ 1
    assert lib.add("neg", "-x", factors["neg"]) is False  # corr ~ -1: |corr| counts
    assert lib.names() == ["f1"]
    assert lib.add("f3", "y", factors["f3"]) is True
    assert lib.names() == ["f1", "f3"] and len(lib) == 2 and "f3" in lib
    assert lib.add("neg", "-x", factors["neg"], force=True) is True
    assert lib.names() == ["f1", "f3", "neg"]


def test_max_corr_returns_signed_best_match(factors):
    lib = FactorLibrary()
    lib.add("f1", "x", factors["f1"])
    lib.add("f3", "y", factors["f3"])
    name, corr = lib.max_corr(factors["near"])
    assert name == "f1" and corr > 0.99
    name, corr = lib.max_corr(factors["neg"])
    assert name == "f1" and corr < -0.99
    # reference value: mean of per-date spearman over the overlap
    assert corr == pytest.approx(mean_spearman_corr(factors["neg"], factors["f1"]))
    _, c_ind = lib.max_corr(_panel(np.random.default_rng(5).standard_normal((N_DATES, N_ASSETS))))
    assert abs(c_ind) < 0.3


def test_threshold_is_strict_and_configurable(factors):
    assert FactorLibrary(max_corr=1.0).add("a", "x", factors["f1"])
    lib = FactorLibrary(max_corr=1.0)
    lib.add("f1", "x", factors["f1"])
    assert lib.add("near", "x", factors["near"]) is True  # |corr| <= 1.0 accepted
    with pytest.raises(ValueError):
        FactorLibrary(max_corr=1.5)


def test_duplicate_name_requires_force(factors):
    lib = FactorLibrary()
    lib.add("f1", "x", factors["f1"])
    with pytest.raises(ValueError, match="already"):
        lib.add("f1", "x2", factors["f3"])
    assert lib.add("f1", "x2", factors["f3"], force=True) is True
    assert lib.get("f1")["expr"] == "x2"
    pd.testing.assert_frame_equal(lib.values("f1"), factors["f3"], check_freq=False)
    with pytest.raises(ValueError):
        lib.add("bad/name", "x", factors["f3"])
    with pytest.raises(TypeError):
        lib.add("arr", "x", factors["f3"].to_numpy())  # type: ignore[arg-type]


def test_overlap_alignment(factors):
    lib = FactorLibrary()
    lib.add("f1", "x", factors["f1"])
    sub = factors["near"].iloc[20:40, :12]  # sub-range -> compared on the overlap
    name, corr = lib.max_corr(sub)
    assert name == "f1" and corr > 0.99
    assert lib.add("sub", "x", sub) is False
    disjoint = factors["f3"].copy()
    disjoint.index = pd.bdate_range("2030-01-01", periods=N_DATES, name="date")
    name, corr = lib.max_corr(disjoint)
    assert name is None and math.isnan(corr)
    assert lib.add("disjoint", "y", disjoint) is True  # cannot be assessed -> accepted


# --------------------------------------------------------------------------- #
# accessors / frames
# --------------------------------------------------------------------------- #
def test_get_values_all_values_remove(factors, report):
    lib = FactorLibrary()
    lib.add("f1", "cs_rank(x)", factors["f1"], report=report, tags=("seed", "reversal"))
    lib.add("f3", "y", factors["f3"])
    e = lib.get("f1")
    assert set(e) == {"expr", "values", "report", "tags"}
    assert e["expr"] == "cs_rank(x)" and e["tags"] == ("seed", "reversal") and e["report"] is report
    pd.testing.assert_frame_equal(e["values"], factors["f1"], check_freq=False)
    assert factors["f1"] is not e["values"]  # stored as a copy
    assert e["values"].index.freq is None and e["values"].index.name == "date"  # store convention
    assert list(lib.all_values()) == ["f1", "f3"]
    assert list(lib) == ["f1", "f3"]
    lib.remove("f1")
    assert lib.names() == ["f3"]
    with pytest.raises(KeyError):
        lib.get("f1")
    with pytest.raises(KeyError):
        lib.remove("f1")


def test_to_frame_columns(factors, report):
    lib = FactorLibrary()
    assert lib.to_frame().empty and lib.to_frame().index.name == "name"
    lib.add("f1", "cs_rank(x)", factors["f1"], report=report, tags=("seed", "reversal"))
    lib.add("f3", "y", factors["f3"], tags=("mined",))
    df = lib.to_frame()
    assert list(df.index) == ["f1", "f3"] and df.index.name == "name"
    assert list(df.columns[:2]) == ["expr", "tags"]
    for col in (
        "ic_mean",
        "icir",
        "icir_ann",
        "rank_ic_mean",
        "ls_sharpe",
        "turnover_mean",
        "coverage",
        "start",
        "end",
    ):
        assert col in df.columns
    assert "name" not in df.columns
    assert df.loc["f1", "expr"] == "cs_rank(x)" and df.loc["f1", "tags"] == "seed,reversal"
    assert df.loc["f1", "ic_mean"] == report.ic_mean
    assert math.isnan(df.loc["f3", "ic_mean"]) and df.loc["f3", "tags"] == "mined"


def test_correlation_matrix(factors):
    lib = FactorLibrary(max_corr=1.0)
    for n in ("f1", "near", "f3", "neg"):
        lib.add(n, n, factors[n])
    cm = lib.correlation_matrix()
    assert cm.shape == (4, 4) and list(cm.index) == list(cm.columns) == ["f1", "near", "f3", "neg"]
    np.testing.assert_allclose(np.diag(cm.to_numpy()), 1.0)
    np.testing.assert_allclose(cm.to_numpy(), cm.to_numpy().T)
    assert cm.loc["f1", "near"] > 0.99 and cm.loc["f1", "neg"] < -0.99
    assert abs(cm.loc["f1", "f3"]) < 0.3
    assert cm.loc["f1", "f3"] == pytest.approx(mean_spearman_corr(factors["f1"], factors["f3"]))
    assert FactorLibrary().correlation_matrix().shape == (0, 0)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_save_load_round_trip(tmp_path, factors, report):
    report.ls_sharpe = float("nan")  # JSON must survive NaN
    lib = FactorLibrary(root=tmp_path / "lib", max_corr=0.6)
    lib.add("f1", "cs_rank(x)", factors["f1"], report=report, tags=("seed", "reversal"))
    lib.add("f3", "ts_mean(y, 5)", factors["f3"])
    lib.save()
    root = tmp_path / "lib"
    assert (root / "f1.parquet").exists() and (root / "f3.parquet").exists()
    payload = json.loads(
        (root / "library.json").read_text(), parse_constant=lambda c: pytest.fail(f"bad JSON token {c}")
    )
    assert payload["max_corr"] == 0.6
    assert set(payload["factors"]) == {"f1", "f3"}
    assert payload["factors"]["f1"]["expr"] == "cs_rank(x)"
    assert payload["factors"]["f1"]["tags"] == ["seed", "reversal"]
    assert payload["factors"]["f1"]["report"]["ls_sharpe"] is None
    assert payload["factors"]["f1"]["report"]["ic_mean"] == pytest.approx(report.ic_mean)
    assert isinstance(payload["factors"]["f1"]["report"]["start"], str)
    assert payload["factors"]["f3"]["report"] is None

    loaded = FactorLibrary.load(root)
    assert loaded.names() == ["f1", "f3"]
    assert loaded.max_corr_threshold == 0.6 and loaded.root == root
    for n in ("f1", "f3"):
        pd.testing.assert_frame_equal(loaded.values(n), lib.values(n))
        assert loaded.get(n)["expr"] == lib.get(n)["expr"]
        assert loaded.get(n)["tags"] == lib.get(n)["tags"]
    rep = loaded.get("f1")["report"]
    assert isinstance(rep, FactorReport)
    assert rep.ic_mean == pytest.approx(report.ic_mean) and rep.icir_ann == pytest.approx(report.icir_ann)
    assert math.isnan(rep.ls_sharpe) and rep.start == report.start and rep.end == report.end
    assert rep.horizon == 5 and rep.lag == 1 and rep.expr == "cs_rank(x)"
    assert loaded.get("f3")["report"] is None
    pd.testing.assert_frame_equal(
        loaded.to_frame().drop(columns=["ls_sharpe"]), lib.to_frame().drop(columns=["ls_sharpe"])
    )
    assert isinstance(rep.summary(), str)
    # dedup still works on the loaded library
    assert loaded.add("near", "x", factors["near"]) is False

    # explicit root on save / no root at all
    lib.save(tmp_path / "other")
    assert (tmp_path / "other" / "library.json").exists()
    with pytest.raises(ValueError):
        FactorLibrary().save()
    with pytest.raises(FileNotFoundError):
        FactorLibrary.load(tmp_path / "nowhere")
