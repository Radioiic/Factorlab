"""Tests for ``alphaforge.research``: one tiny end-to-end run, many assertions.

The pipeline is run once per test session (module-scoped fixture) on an 80-asset,
320-day market with a 20 x 2 GP budget, and every test inspects the returned dict and
the files it wrote.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaforge.config import ResearchConfig
from alphaforge.data import DataStore, SimConfig
from alphaforge.evaluation import FactorReport
from alphaforge.expr import parse, to_string, validate
from alphaforge.library import FactorLibrary
from alphaforge.mining import MinedFactor
from alphaforge.portfolio import BacktestResult
from alphaforge.research import SEED_FACTORS, run_research, split_dates

SPEC_KEYS = {
    "store",
    "library",
    "reports_train",
    "reports_valid",
    "reports_test",
    "mined",
    "backtests",
    "paths",
    "timings",
}


@pytest.fixture(scope="module")
def tiny_cfg(tmp_path_factory: pytest.TempPathFactory) -> ResearchConfig:
    out = tmp_path_factory.mktemp("alphaforge_run")
    return ResearchConfig(
        sim=SimConfig(n_assets=80, n_days=320, seed=7),
        population_size=20,
        n_generations=2,
        hall_of_fame=20,
        out_dir=str(out),
        seed=3,
    )


@pytest.fixture(scope="module")
def result(tiny_cfg: ResearchConfig) -> dict:
    messages: list[str] = []
    res = run_research(tiny_cfg, log_fn=messages.append)
    res["_messages"] = messages
    return res


# ----------------------------------------------------------------------------- seeds
def test_seed_factors_are_valid_expressions() -> None:
    assert len(SEED_FACTORS) >= 8
    fields = {"open", "high", "low", "close", "volume", "amount", "vwap", "returns", "market_cap", "bp", "industry"}
    for name, expr in SEED_FACTORS.items():
        node = parse(expr)
        validate(node, fields=fields)
        assert to_string(node) == expr, f"{name} is not in canonical prefix form"


def test_split_dates_is_contiguous_and_ordered() -> None:
    cfg = ResearchConfig(train_frac=0.6, valid_frac=0.2)
    dates = pd.bdate_range("2020-01-01", periods=100, name="date")
    s = split_dates(cfg, dates)
    assert list(s) == ["train", "valid", "test"]
    assert len(s["train"]) + len(s["valid"]) + len(s["test"]) == 100
    assert s["train"][-1] < s["valid"][0] < s["valid"][-1] < s["test"][0]
    assert len(s["train"]) == 60 and len(s["valid"]) == 20 and len(s["test"]) == 20


# ----------------------------------------------------------------------------- returned dict
def test_result_keys_and_types(result: dict) -> None:
    assert set(result) >= SPEC_KEYS
    assert isinstance(result["store"], DataStore)
    assert isinstance(result["library"], FactorLibrary)
    for key in ("reports_train", "reports_valid", "reports_test"):
        assert isinstance(result[key], dict) and result[key]
        assert all(isinstance(r, FactorReport) for r in result[key].values())
    assert all(isinstance(m, MinedFactor) for m in result["mined"])
    assert isinstance(result["backtests"], dict)
    assert all(isinstance(b, BacktestResult) for b in result["backtests"].values())
    assert isinstance(result["timings"], dict) and result["timings"]["total"] > 0
    assert all(v >= 0 for v in result["timings"].values())


def test_log_fn_receives_progress(result: dict) -> None:
    msgs = result["_messages"]
    assert any("[step 1]" in m for m in msgs)
    assert any("done in" in m for m in msgs)


def test_library_has_seeds_and_reports_cover_every_factor(result: dict) -> None:
    lib: FactorLibrary = result["library"]
    names = lib.names()
    assert len(names) >= 8
    assert set(SEED_FACTORS) <= set(names)
    for key in ("reports_train", "reports_valid", "reports_test"):
        assert set(result[key]) == set(names), key
    # every library factor is oriented so its TRAIN IC is non-negative
    for n in names:
        ic = result["reports_train"][n].ic_mean
        assert not (math.isfinite(ic) and ic < -1e-12), f"{n} has negative train IC {ic}"
    for n in names:
        tags = lib.get(n)["tags"]
        assert ("seed" in tags) != ("mined" in tags)
    assert len([n for n in names if "mined" in lib.get(n)["tags"]]) == len(result["mined"])
    assert len(result["mined"]) <= result["config"].max_factors


def test_windows_do_not_overlap(result: dict) -> None:
    split = result["split"]
    r_tr, r_va, r_te = (result[k][next(iter(result[k]))] for k in ("reports_train", "reports_valid", "reports_test"))
    assert r_tr.start == split["train"][0] and r_tr.end == split["train"][-1]
    assert r_va.start == split["valid"][0] and r_va.end == split["valid"][-1]
    assert r_te.start == split["test"][0] and r_te.end == split["test"][-1]
    assert r_tr.end < r_va.start and r_va.end < r_te.start


def test_mined_factors_pass_validation_threshold(result: dict) -> None:
    cfg: ResearchConfig = result["config"]
    lib: FactorLibrary = result["library"]
    for n in lib.names():
        if "mined" in lib.get(n)["tags"]:
            assert result["reports_valid"][n].icir_ann >= cfg.min_valid_icir_ann
            assert parse(lib.get(n)["expr"])  # stored expression is parseable


# ----------------------------------------------------------------------------- backtests
def test_three_backtests_with_finite_sharpe(result: dict) -> None:
    bts = result["backtests"]
    assert set(bts) == {"best_single", "linear_icir", "ml_hgb"}
    table = result["tables"]["backtest_comparison"]
    assert list(table.index) == ["best_single", "linear_icir", "ml_hgb"]
    assert not table["sharpe"].isna().any()
    assert np.isfinite(table["sharpe"].to_numpy(dtype=float)).all()
    test_dates = result["split"]["test"]
    for bt in bts.values():
        assert bt.n_days > 0
        assert bt.returns.index[0] >= test_dates[0]
        assert bt.returns.index[-1] <= test_dates[-1]


def test_composites_cover_test_window(result: dict) -> None:
    test = result["split"]["test"]
    for label, panel in result["composites"].items():
        cov = panel.reindex(index=test).notna().any(axis=1).mean()
        assert cov > 0.9, f"{label} covers only {cov:.0%} of test dates"


# ----------------------------------------------------------------------------- files
def test_output_files_exist(result: dict) -> None:
    paths = result["paths"]
    out = Path(paths["out_dir"])
    assert (out / "report.html").is_file() and paths["report_html"] == out / "report.html"
    assert (out / "run.json").is_file()
    assert (out / "data" / "meta.json").is_file()
    assert (out / "library" / "library.json").is_file()
    for key in (
        "report_train",
        "report_valid",
        "report_test",
        "mining_history",
        "backtest_comparison",
        "rediscovery",
        "composite_ic",
        "hall_of_fame",
    ):
        assert (out / "tables" / f"{key}.csv").is_file(), key
    pngs = list((out / "figures").glob("*.png"))
    assert any(p.name == "corr_heatmap.png" for p in pngs)
    assert any(p.name == "mining_history.png" for p in pngs)
    assert sum(p.name.startswith("equity_") for p in pngs) == 3
    assert sum(p.name.startswith("top1_") for p in pngs) == 3  # ic, quantiles, ic_decay
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "charset" in html.lower() and "Out-of-sample" in html


def test_run_json_roundtrips_config(result: dict) -> None:
    data = json.loads(Path(result["paths"]["run_json"]).read_text(encoding="utf-8"))
    for key in ("config", "split", "timings", "headline", "backtests", "mined", "paths"):
        assert key in data
    cfg = ResearchConfig.from_dict(data["config"])
    assert cfg == result["config"]
    assert data["headline"]["n_mined"] == len(result["mined"])
    assert data["headline"]["best_oos_rank_icir"] is not None
    assert set(data["backtests"]) == {"best_single", "linear_icir", "ml_hgb"}


def test_saved_store_and_library_reload(result: dict) -> None:
    paths = result["paths"]
    store = DataStore.load(paths["data"])
    assert store.dates.equals(result["store"].dates) and store.assets.equals(result["store"].assets)
    assert "true_alphas" in store.meta
    lib = FactorLibrary.load(paths["library"])
    assert set(lib.names()) == set(result["library"].names())


def test_rediscovery_finds_planted_reversal(result: dict) -> None:
    red = result["tables"]["rediscovery"]
    assert {"corr_reversal", "corr_momentum", "corr_volume", "corr_value", "best_match"} <= set(red.columns)
    # the 5-day reversal seed is (up to sign) the planted reversal alpha
    assert red.loc["reversal_5d", "best_match"] == "reversal"
    assert abs(red.loc["reversal_5d", "corr_reversal"]) > 0.8
