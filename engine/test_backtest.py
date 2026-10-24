"""Integrity tests for the backtest engine — synthetic data, no network, <1s.

Usage:
    python test_backtest.py         # or: pytest test_backtest.py

Same method as the label-table tests: each test constructs a tiny
world where the correct answer is known in closed form, then asserts the
engine reproduces it EXACTLY (1e-9), not approximately. If you touch
backtest.py, run this. If you add a feature to the engine, add a test here
that would have caught it being wrong.

The six properties enforced:
    1. point-in-time: a Monday's trade cannot see that Monday's own bar
    2. funding sign: positive funding drains longs, pays shorts
    3. fees: charged once on turnover, to the dollar, and never re-charged
       while the portfolio just sits there
    4. flat: a zero-weight strategy produces a constant equity curve
    5. scale invariance: $1k and $1M produce proportional equity curves
    6. drift: turnover is measured against drifted weights, so rebalancing
       back to unchanged targets still trades the drift

These run under the engine's accounting (reconciled-v2: holdings are
target_w x post-fee equity, fees are charged on the dollars traded). Only test
3's expected numbers depend on the convention; the comment there explains it.
The trade-log <-> equity reconciliation itself lives in test_accounting.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import run_backtest

SYMS = ["AAA", "BBB", "CCC"]


# --- fixtures -----------------------------------------------------------------


def make_index(n=28, start="2023-01-01"):
    """Daily UTC index starting on a Sunday, so day 1 is a Monday rebalance."""
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


def flat_prices(index, syms=SYMS, px=100.0):
    return pd.DataFrame(px, index=index, columns=list(syms))


def factor_table(index, syms, sig_fn):
    """Long-format factor table with a single 'sig' column driven by sig_fn."""
    return pd.DataFrame([{"date": d, "symbol": s, "sig": float(sig_fn(d, s))} for d in index for s in syms])


def sig_weights(cs: pd.DataFrame) -> pd.Series:
    """Weights proportional to sig, normalized to gross 1 (0 if all flat)."""
    w = cs["sig"]
    gross = w.abs().sum()
    return w / gross if gross > 0 else w * 0.0


def no_funding(index, syms=SYMS):
    return pd.DataFrame(0.0, index=index, columns=list(syms))


def run(prices, factors, funding=None, **kw):
    kw.setdefault("initial_equity", 10_000.0)
    kw.setdefault("fee_bps", 0.0)
    kw.setdefault("slippage_bps", 0.0)
    return run_backtest(
        sig_weights,
        prices=prices,
        factors=factors,
        funding=no_funding(prices.index) if funding is None else funding,
        benchmark=None,
        **kw,
    )


# --- 1. point-in-time ---------------------------------------------------------


def test_no_lookahead():
    idx = make_index()
    jump_day = pd.Timestamp("2023-01-16", tz="UTC")  # a Monday
    px = flat_prices(idx)
    px.loc[idx >= jump_day, "AAA"] = 200.0  # AAA doubles ON the Monday bar

    # Signal fires only on the jump day itself. The Monday decision reads
    # SUNDAY's row, so this signal is unusable and the engine must stay flat —
    # an engine that peeks at Monday's own row would double its money here.
    f_leak = factor_table(idx, SYMS, lambda d, s: s == "AAA" and d == jump_day)
    r = run(px, f_leak, fee_bps=5.0, slippage_bps=5.0)
    assert len(r.trades) == 0, "engine traded on same-day information"
    assert abs(r.equity["equity"].iloc[-1] - 10_000.0) < 1e-9, "equity moved despite no legally usable signal"

    # Control: same jump, signal on Sunday -> legally captured, exactly 2x.
    sunday = jump_day - pd.Timedelta(days=1)
    f_legal = factor_table(idx, SYMS, lambda d, s: s == "AAA" and d == sunday)
    r = run(px, f_legal)
    assert abs(r.equity["equity"].iloc[-1] - 20_000.0) < 1e-9
    assert list(r.trades["side"]) == ["buy", "sell"]  # enter 01-16, exit 01-23


# --- 2. funding sign ----------------------------------------------------------


def test_funding_sign():
    # Funding is paid on NOTIONAL, and a perp position's notional does not
    # change just because your margin drains — so between rebalances the
    # dollar cost per day is CONSTANT (weights drift above 1 as equity falls),
    # and each Monday the rebalance resets notional to current equity.
    # Closed form: equity multiplies by (1 - 0.01 * bars) per holding stretch.
    idx = make_index()  # in from Monday (day 1)
    px = flat_prices(idx)
    fund = no_funding(idx)
    fund["AAA"] = 0.01  # longs pay 1%/day
    stretches = (7, 7, 7, 6)  # Mon..Sun x3, Mon..Sat

    f_long = factor_table(idx, SYMS, lambda d, s: s == "AAA")
    r = run(px, f_long, funding=fund)
    expect = 10_000.0
    for bars in stretches:
        expect *= 1 - 0.01 * bars
    assert abs(r.equity["equity"].iloc[-1] - expect) < 1e-9, "long should PAY funding"
    assert r.stats["funding_total"] < 0

    f_short = factor_table(idx, SYMS, lambda d, s: -(s == "AAA"))
    r = run(px, f_short, funding=fund)
    expect = 10_000.0
    for bars in stretches:
        expect *= 1 + 0.01 * bars
    assert abs(r.equity["equity"].iloc[-1] - expect) < 1e-9, "short should RECEIVE funding"
    assert r.stats["funding_total"] > 0


# --- 3. fees ------------------------------------------------------------------


def test_fees_charged_once_on_turnover():
    idx = make_index()
    px = flat_prices(idx)
    f = factor_table(idx, SYMS, lambda d, s: s == "AAA")  # hold AAA forever

    r = run(px, f, fee_bps=10.0)
    # One entry trade of gross 1.0 at 10bps; flat prices + no funding = no
    # drift, so the Mondays after it have zero turnover and must charge NOTHING.
    #
    # Expected values under reconciled-v2: the position is 100% of POST-fee
    # equity and the fee is 10bps of the dollars actually bought, so
    # fee = 0.001 * E_post and E_post = 10_000 - fee:
    #   E_post = 10_000 / 1.001 = 9_990.00999...,  fee = 9.99000999...
    # test_accounting.py reconciles the trade log with the equity.
    assert len(r.trades) == 1
    e_post = 10_000.0 / 1.001
    assert abs(r.stats["fees_total"] - (10_000.0 - e_post)) < 1e-9  # 10bps of E_post
    assert abs(r.trades["notional"].iloc[0] - e_post) < 1e-9  # bought exactly E_post
    assert abs(r.equity["equity"].iloc[-1] - e_post) < 1e-9

    r0 = run(px, f, fee_bps=0.0)
    assert r0.equity["equity"].iloc[-1] > r.equity["equity"].iloc[-1]


# --- 4. flat ------------------------------------------------------------------


def test_flat_when_no_positions():
    idx = make_index()
    px = flat_prices(idx)
    px["BBB"] = np.linspace(100, 300, len(idx))  # moving prices, no position
    f = factor_table(idx, SYMS, lambda d, s: 0.0)
    r = run(px, f, fee_bps=5.0, slippage_bps=5.0)
    assert len(r.trades) == 0
    assert (r.equity["equity"] == 10_000.0).all()
    assert r.stats["fees_total"] == 0.0


# --- 5. scale invariance ------------------------------------------------------


def test_scale_invariance():
    rng = np.random.default_rng(7)
    idx = make_index(n=60)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (len(idx), 3)), axis=0)), index=idx, columns=SYMS)
    fund = pd.DataFrame(rng.normal(0, 2e-4, (len(idx), 3)), index=idx, columns=SYMS)
    sig = {(d, s): rng.choice([-1.0, 0.0, 1.0]) for d in idx for s in SYMS}
    f = factor_table(idx, SYMS, lambda d, s: sig[(d, s)])

    r_small = run(px, f, funding=fund, fee_bps=5.0, slippage_bps=5.0, initial_equity=1_000.0)
    r_big = run(px, f, funding=fund, fee_bps=5.0, slippage_bps=5.0, initial_equity=1_000_000.0)
    ratio_small = r_small.equity["equity"] / 1_000.0
    ratio_big = r_big.equity["equity"] / 1_000_000.0
    assert np.allclose(ratio_small, ratio_big, rtol=1e-12), "same code must behave identically at $1k and $1M"
    # trades identical in weight terms too
    assert np.allclose(r_small.trades["delta_w"], r_big.trades["delta_w"], rtol=1e-12)


# --- 6. drift turnover --------------------------------------------------------


def test_drift_is_traded():
    idx = make_index()
    px = flat_prices(idx, syms=["AAA", "BBB"])
    px.loc[idx >= pd.Timestamp("2023-01-10", tz="UTC"), "AAA"] = 110.0  # +10% Tue bar
    f = factor_table(idx, ["AAA", "BBB"], lambda d, s: 1.0)  # always 50/50 target

    r = run(px, f)
    # After the Tuesday move: wA = .5*1.1/1.05, wB = .5/1.05. Rebalancing back
    # to 50/50 the following Monday must trade exactly that drift.
    drift_a = 0.5 * 1.1 / 1.05 - 0.5  # +0.0238095...
    monday2 = pd.Timestamp("2023-01-16", tz="UTC")
    t2 = r.trades[r.trades["date"] == monday2].set_index("symbol")
    assert abs(t2.loc["AAA", "delta_w"] + drift_a) < 1e-9  # sell the winner
    assert abs(t2.loc["BBB", "delta_w"] - drift_a) < 1e-9  # buy the laggard
    assert abs(r.equity["equity"].iloc[-1] - 10_000.0 * 1.05) < 1e-9


# --- runner -------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001 - report every test, then fail at the end
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} backtester integrity tests passed")
    raise SystemExit(1 if failed else 0)
