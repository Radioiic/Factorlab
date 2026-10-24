import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from factorlab import operators as op
from factorlab.data.binance import parse_funding, parse_klines, parse_metrics_zip, to_bar
from factorlab.data.synthetic import make_synthetic_panel
from factorlab.database import FactorDB
from factorlab.evaluate import evaluate_factor, rank_ic
from factorlab.expr import Evaluator, ExpressionError, max_lookback, parse, raw_dependencies
from factorlab.generator import generate_factors


@pytest.fixture(scope="module")
def panels():
    return make_synthetic_panel(n_symbols=12, n_bars=500, seed=3)


@pytest.fixture(scope="module")
def specs():
    return generate_factors()


def test_generator_size_and_uniqueness(specs):
    assert 20_000 <= len(specs) <= 50_000
    names = [s.name for s in specs]
    assert len(set(names)) == len(names)
    assert len({s.expression for s in specs}) == len(specs)


def test_all_expressions_parse(specs):
    for s in specs:
        parse(s.expression)
        assert raw_dependencies(s.expression)


def test_parser_rejects_unsafe_code():
    for bad in ["__import__('os')", "close.__class__", "open_file(close)", "lambda: 1", "close if 1 else 2", "'a'"]:
        with pytest.raises(ExpressionError):
            parse(bad)


def test_raw_dependencies_expand_derived():
    assert raw_dependencies("ts_mean(ret, 5)") == {"close"}
    assert raw_dependencies("taker_imb * oi_chg") == {"taker_buy_volume", "volume", "open_interest_value"}
    assert max_lookback("ts_mean(delay(close, 5), 20)") == 25


def test_ts_rank_and_minmax():
    x = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 2]})
    r = op.ts_rank(x, 5)
    assert np.isnan(r.iloc[3, 0])
    assert r.iloc[4, 0] == 1.0
    assert r.iloc[5, 0] == pytest.approx(0.125)  # window [2,3,4,5,2]: one tie -> half credit
    assert op.ts_minmax(x, 5).iloc[4, 0] == 1.0


def test_no_lookahead(panels):
    """Changing future data must not change factor values at time t."""
    ev = Evaluator(panels)
    cut = 400
    future_shocked = {k: v.copy() for k, v in panels.items()}
    for v in future_shocked.values():
        v.iloc[cut:] = v.iloc[cut:] * 1.5
    ev2 = Evaluator(future_shocked)
    for expr in [
        "ts_rank(ret, 20)",
        "ts_corr(ret, dvol_chg, 30)",
        "decay_linear(taker_imb, 10)",
        "ts_beta(ret, mkt_ret, 60)",
        "ts_maxdd(close, 30)",
    ]:
        a, b = ev(expr).iloc[:cut], ev2(expr).iloc[:cut]
        pd.testing.assert_frame_equal(a, b)


def test_sample_of_factors_evaluate(panels, specs):
    ev = Evaluator(panels)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(specs), 300, replace=False):
        v = ev(specs[i].expression)
        assert v.shape == panels["close"].shape
        assert not np.isinf(v.to_numpy()).any()


def test_benchmark_fallback(panels):
    no_btc = {k: v.drop(columns=["BTCUSDT"]) for k, v in panels.items()}
    v = Evaluator(no_btc)("btc_ret")
    pd.testing.assert_series_equal(v.iloc[:, 0], v.iloc[:, 1], check_names=False)


def test_rank_ic_perfect():
    idx = pd.date_range("2024-01-01", periods=30)
    close = pd.DataFrame(np.cumprod(1 + np.random.default_rng(1).normal(0, 0.02, (30, 8)), axis=0), index=idx)
    fwd = close.shift(-1) / close - 1
    ic = rank_ic(fwd, fwd).dropna()
    assert np.allclose(ic, 1.0)
    stats = evaluate_factor(fwd, close, 1)
    assert stats["ic_mean"] == pytest.approx(1.0)


def test_database_roundtrip(tmp_path, panels):
    db = FactorDB(tmp_path / "f.db")
    db.write_catalog()
    db.replace_factors(
        [
            {
                "factor_id": "CF00001",
                "name": "x",
                "family": "f",
                "category": "c",
                "expression": "ts_mean(ret, 5)",
                "raw_inputs": "close",
            }
        ]
    )
    assert db.count() == 1
    assert db.get("x")["factor_id"] == "CF00001"
    assert len(db.factors(available_fields=["open"])) == 0
    vals = Evaluator(panels)("ts_mean(ret, 5)")
    n = db.write_values("CF00001", vals)
    assert n == int(vals.notna().sum().sum())
    back = db.read_values("CF00001")
    assert back.shape[1] == vals.shape[1]
    db.write_stats([{"factor_id": "CF00001", "dataset": "t", "horizon": 1, "ic_mean": np.float64(0.1), "icir": np.nan}])
    assert db.stats("t")["ic_mean"].iloc[0] == pytest.approx(0.1)
    db.close()


def test_binance_parsers():
    rows = [[1704067200000, "1", "2", "0.5", "1.5", "10", 1704153599999, "15", 7, "6", "9", "0"]]
    k = parse_klines(rows)
    assert k.loc[k.index[0], "quote_volume"] == 15.0
    f = parse_funding(
        [
            {"fundingTime": 1704067200000, "fundingRate": "0.0001"},
            {"fundingTime": 1704096000000, "fundingRate": "0.0002"},
        ]
    )
    assert to_bar(f, "1d", "sum").iloc[0] == pytest.approx(0.0003)
    csv = (
        "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
        "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2024-01-01 00:05:00,BTCUSDT,100,4000000,1.1,1.2,1.3,0.9\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BTCUSDT-metrics-2024-01-01.csv", csv)
    m = parse_metrics_zip(buf.getvalue())
    assert m["open_interest_value"].iloc[0] == 4_000_000
    assert m["long_short_ratio"].iloc[0] == 1.3
