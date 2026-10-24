"""Accounting reconciliation tests for the backtest engine — synthetic, no disk.

Usage:
    python test_accounting.py         # or: pytest test_accounting.py

The accounting convention of the engine ("reconciled-v2", documented in
backtest.py's docstring) uses ONE set of dollars: fees leave equity first, every traded leg holds
target_w x E_post, and each leg's fee is rate x the dollars it actually
traded. These tests do not trust the engine's own bookkeeping. `audit()`
rebuilds the whole equity curve from the TRADE LOG and the market data alone:

    contracts += side x notional / ref_price        (each logged leg)
    equity    -= sum(logged cost)                   (at the open)
    equity    += sum(contracts x price change) - sum(contracts x open price x funding)

and requires it to match daily.parquet's equity, cost, gross and funding to
1e-9, and every logged leg to satisfy the convention (prev holding =
prev_w x E_pre, new holding = target_w x E_post, cost = rate x notional).
A log whose notionals and fees use the PRE-fee equity is run through the same
audit to prove the audit has teeth: it must FAIL, on the two-price scenario.

The properties enforced:
    1. the two-price scenario reconciles (quantities, fee, equity)
    2. a second rebalance (sell into a 50/50 book) reconciles in closed form
    3. a multi-asset long/short book with funding and fees reconciles, at $1k
       and $1M identically
    4. an unchanged target still trades the fee sliver (and logs it)
    5. a dust leg is neither traded, charged nor logged: its holdings stay put
    6. the fee fixed point is solved exactly (closed forms + residual)
    7. the audit finds a trade log that differs from the P&L
    8. the stats carry the accounting tag; other modes and fee rates that would eat the
       whole notional are refused
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as B
from backtest import run_backtest

TOL = 1e-9  # relative to the equity scale, as in test_backtest.py


# --- fixtures -----------------------------------------------------------------


def make_index(n, start="2023-01-01"):
    """Daily UTC index starting on a Sunday (2023-01-01), so day 1 is a Monday."""
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


def factor_table(index, syms, sig_fn):
    return pd.DataFrame([{"date": d, "symbol": s, "sig": float(sig_fn(d, s))} for d in index for s in syms])


def sig_weights(cs: pd.DataFrame) -> pd.Series:
    """Weights proportional to sig, normalized to gross 1 (0 if all flat)."""
    w = cs["sig"]
    gross = w.abs().sum()
    return w / gross if gross > 0 else w * 0.0


def run(prices, factors, funding=None, **kw):
    kw.setdefault("initial_equity", 10_000.0)
    kw.setdefault("fee_bps", 0.0)
    kw.setdefault("slippage_bps", 0.0)
    if funding is None:
        funding = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    return run_backtest(sig_weights, prices=prices, factors=factors, funding=funding, benchmark=None, **kw)


def day(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


# --- the independent audit ----------------------------------------------------


def audit(res, prices, funding, rate, initial_equity) -> list[str]:
    """Rebuild equity from the trade log + market data; return every mismatch.

    Uses only: trades (side, notional, cost, ref_price, prev_w, target_w),
    prices, funding and the starting equity. Holdings are tracked in
    CONTRACTS, the way an exchange would report them, so a log that describes
    a different position from the one the P&L earned cannot pass.
    """
    scale = float(initial_equity)
    bad: list[str] = []

    def check(label, got, want):
        if not abs(float(got) - float(want)) <= TOL * max(scale, abs(float(want))):
            bad.append(f"{label}: ledger {float(got):.12g} vs audit {float(want):.12g}")

    daily = res.daily
    if funding is None:
        funding = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    by_date = {d: g for d, g in res.trades.groupby("date")}
    q = pd.Series(0.0, index=prices.columns)  # contracts held
    eq = scale
    check(f"{prices.index[0].date()} equity", daily["equity"].iloc[0], eq)
    for i in range(1, len(prices.index)):
        t = prices.index[i]
        px0, px1 = prices.iloc[i - 1], prices.iloc[i]
        g = by_date.get(t)
        fees = 0.0
        if g is not None:
            e_pre = eq
            h_prev = q * px0
            for r in g.itertuples(index=False):
                if r.ref_price != px0[r.symbol]:
                    bad.append(f"{t.date()} {r.symbol}: ref_price is not the t-1 close")
                check(f"{t.date()} {r.symbol} cost = rate x notional", r.cost, rate * r.notional)
                check(f"{t.date()} {r.symbol} prev holding", h_prev[r.symbol], r.prev_w * e_pre)
                q[r.symbol] += (1.0 if r.side == "buy" else -1.0) * r.notional / r.ref_price
                fees += r.cost
            eq = e_pre - fees
            h_new = q * px0
            for r in g.itertuples(index=False):
                check(f"{t.date()} {r.symbol} new holding = target_w x E_post", h_new[r.symbol], r.target_w * eq)
            legs = list(g["symbol"])
            check(f"{t.date()} fee fixed point", fees, rate * (h_new[legs] - h_prev[legs]).abs().sum())
        check(f"{t.date()} cost", daily["cost"].loc[t], fees)
        hold = q * px0  # dollars held through bar t
        if eq > 0:
            check(f"{t.date()} gross", daily["gross"].loc[t] * scale, hold.abs().sum() / eq * scale)
        fund = -float((hold * funding.iloc[i]).sum())  # positive funding: longs pay
        check(f"{t.date()} funding_usd", daily["funding_usd"].loc[t], fund)
        eq = eq + float((q * (px1 - px0)).sum()) + fund
        check(f"{t.date()} equity", daily["equity"].loc[t], eq)
    return bad


def assert_reconciles(res, prices, funding, rate, initial_equity):
    bad = audit(res, prices, funding, rate, initial_equity)
    assert not bad, f"{len(bad)} mismatches, first: {bad[:3]}"


# --- 1. the two-price scenario ------------------------------------------------


def two_price_world():
    """$10,000; Monday 100% buy of AAA at $100 (Sunday close); Tuesday close $200."""
    idx = make_index(3)  # Sun, Mon, Tue
    px = pd.DataFrame({"AAA": [100.0, 100.0, 200.0]}, index=idx)
    f = factor_table(idx, ["AAA"], lambda d, s: d == day("2023-01-01"))
    return px, f


def test_two_price_scenario_reconciles():
    """$10k, 100% buy at $100, 10 bps, next bar $200: the run reconciles."""
    px, f = two_price_world()
    rate = 10 / 1e4
    r = run(px, f, fee_bps=10.0)
    # Closed form: E_post = E_pre - fee and fee = rate * E_post (the whole
    # post-fee equity is bought), so E_post = 10_000 / 1.001.
    e_post = 10_000.0 / 1.001
    fee = 10_000.0 - e_post  # 9.99000999...
    contracts = e_post / 100.0  # 99.9000999...
    assert len(r.trades) == 1
    leg = r.trades.iloc[0]
    assert leg["side"] == "buy" and leg["ref_price"] == 100.0
    assert abs(leg["notional"] - e_post) < 1e-9
    assert abs(leg["cost"] - fee) < 1e-9
    assert abs(leg["notional"] / leg["ref_price"] - contracts) < 1e-12
    assert abs(r.daily["cost"].loc[day("2023-01-02")] - fee) < 1e-9
    # the log's own arithmetic: contracts x $200, nothing left over
    assert abs(r.equity["equity"].iloc[-1] - contracts * 200.0) < 1e-9
    assert abs(r.equity["equity"].iloc[-1] - 2 * e_post) < 1e-9  # 19_980.01998...
    assert_reconciles(r, px, None, rate, 10_000.0)


# --- 2. a second rebalance ----------------------------------------------------


def test_second_rebalance_reconciles():
    """A second rebalance (sell into a 50/50 book) reconciles in closed form."""
    # Week 1: 100% AAA (as above). AAA doubles on Tuesday. Week 2 (Mon 01-09):
    # move to 50/50 AAA/BBB. BBB then rises 100 -> 150 on Wednesday 01-11.
    idx = make_index(12)  # 2023-01-01 .. 01-12
    px = pd.DataFrame({"AAA": 100.0, "BBB": 100.0}, index=idx)
    px.loc[idx >= day("2023-01-03"), "AAA"] = 200.0
    px.loc[idx >= day("2023-01-11"), "BBB"] = 150.0
    f = factor_table(idx, ["AAA", "BBB"], lambda d, s: s == "AAA" or d >= day("2023-01-08"))
    rate = 10 / 1e4
    r = run(px, f, fee_bps=10.0)

    e1 = 10_000.0 / 1.001  # post-fee equity, week 1
    e_pre2 = 2 * e1  # AAA doubled, held 100%
    # Week 2 fixed point: x = 1 - rate*(|0.5x - 1| + |0.5x|) = 1 - rate, exactly,
    # because the sold and bought halves' fee shrinkage cancels.
    e_post2 = (1 - rate) * e_pre2
    t2 = r.trades[r.trades["date"] == day("2023-01-09")].set_index("symbol")
    assert list(t2.index) == ["AAA", "BBB"]
    assert t2.loc["AAA", "side"] == "sell" and t2.loc["BBB", "side"] == "buy"
    assert abs(t2.loc["AAA", "notional"] - (e_pre2 - 0.5 * e_post2)) < 1e-9
    assert abs(t2.loc["BBB", "notional"] - 0.5 * e_post2) < 1e-9
    assert abs(t2["cost"].sum() - rate * e_pre2) < 1e-9
    # contracts after week 2: AAA e1/100 -> 0.5*e_post2/200, BBB 0 -> 0.5*e_post2/100
    q_aaa = e1 / 100.0 - t2.loc["AAA", "notional"] / 200.0
    q_bbb = t2.loc["BBB", "notional"] / 100.0
    assert abs(q_aaa - 0.5 * e_post2 / 200.0) < 1e-12
    assert abs(q_bbb - 0.5 * e_post2 / 100.0) < 1e-12
    final = q_aaa * 200.0 + q_bbb * 150.0  # no cash: fees already left equity
    assert abs(r.equity["equity"].iloc[-1] - final) < 1e-9
    assert abs(final - 1.25 * e_post2) < 1e-9
    assert_reconciles(r, px, None, rate, 10_000.0)


# --- 3. multi-asset long/short with funding -----------------------------------


def random_world(seed=11, n=84, syms=("AAA", "BBB", "CCC", "DDD", "EEE")):
    rng = np.random.default_rng(seed)
    idx = make_index(n)
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.04, (n, len(syms))), axis=0)), index=idx, columns=list(syms)
    )
    fund = pd.DataFrame(rng.normal(1e-4, 4e-4, (n, len(syms))), index=idx, columns=list(syms))
    sig = {(d, s): rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for d in idx for s in syms}
    f = factor_table(idx, list(syms), lambda d, s: sig[(d, s)])
    return px, fund, f


def test_multi_asset_long_short_reconciles():
    """A five-name long/short book with funding and fees reconciles at $1k, $10k and $1M."""
    px, fund, f = random_world()
    rate = (5 + 5) / 1e4
    r = run(px, f, funding=fund, fee_bps=5.0, slippage_bps=5.0)
    assert (r.trades["target_w"] < 0).any() and (r.trades["target_w"] > 0).any(), "fixture must hold shorts and longs"
    assert r.trades["date"].nunique() >= 10, "fixture must rebalance repeatedly"
    assert_reconciles(r, px, fund, rate, 10_000.0)
    # sum of leg costs is the daily fee, and fees_total is their sum
    by_day = r.trades.groupby("date")["cost"].sum()
    assert np.allclose(by_day.to_numpy(), r.daily["cost"].loc[by_day.index].to_numpy(), rtol=0, atol=1e-9)
    assert abs(r.stats["fees_total"] - r.trades["cost"].sum()) < 1e-9

    # % of equity sizing: $1k and $1M are the same book, dollar for dollar x1000
    small = run(px, f, funding=fund, fee_bps=5.0, slippage_bps=5.0, initial_equity=1_000.0)
    big = run(px, f, funding=fund, fee_bps=5.0, slippage_bps=5.0, initial_equity=1_000_000.0)
    assert np.allclose(small.equity["equity"] * 1000.0, big.equity["equity"], rtol=1e-12, atol=0)
    assert np.allclose(small.trades["notional"] * 1000.0, big.trades["notional"], rtol=1e-12, atol=0)
    assert list(small.trades["side"]) == list(big.trades["side"])
    assert_reconciles(small, px, fund, rate, 1_000.0)
    assert_reconciles(big, px, fund, rate, 1_000_000.0)


def test_first_rebalance_closed_form_long_short():
    """Opening from flat: E_post = E_pre / (1 + rate * gross) for any long/short split."""
    # From flat, every leg is a pure open: x = 1 - rate * gross * x, so
    # E_post = E_pre / (1 + rate * gross) whatever the long/short split.
    idx = make_index(3)
    syms = ["AAA", "BBB", "CCC", "DDD"]
    px = pd.DataFrame(100.0, index=idx, columns=syms)
    sign = {"AAA": 1.0, "BBB": 1.0, "CCC": -1.0, "DDD": -1.0}
    f = factor_table(idx, syms, lambda d, s: sign[s])
    rate = 10 / 1e4
    r = run(px, f, fee_bps=5.0, slippage_bps=5.0)
    e_post = 10_000.0 / (1 + rate)
    assert abs(r.daily["equity"].iloc[-1] - e_post) < 1e-9
    assert np.allclose(r.trades["notional"], 0.25 * e_post, rtol=0, atol=1e-9)
    assert list(r.trades["side"]) == ["buy", "buy", "sell", "sell"]
    assert_reconciles(r, px, None, rate, 10_000.0)


# --- 4. the fee sliver on an unchanged target ----------------------------------


def test_unchanged_target_trades_the_fee_sliver():
    """An unchanged target weight still sells target_w x fees, and the leg is logged."""
    # Week 1: 50/50 AAA/BBB. Week 2: 50/50 AAA/CCC. AAA's WEIGHT is unchanged,
    # but fees shrink equity, so holding 0.5 of the post-fee equity means
    # selling 0.5 * fee of AAA. x = 1 - rate*(|.5x-.5| + .5 + .5x) = 1 - rate.
    idx = make_index(12)
    syms = ["AAA", "BBB", "CCC"]
    px = pd.DataFrame(100.0, index=idx, columns=syms)  # flat: no drift at all
    f = factor_table(
        idx,
        syms,
        lambda d, s: s == "AAA" or (s == "BBB" and d < day("2023-01-08")) or (s == "CCC" and d >= day("2023-01-08")),
    )
    rate = 10 / 1e4
    r = run(px, f, fee_bps=10.0)
    e_pre2 = 10_000.0 / (1 + rate)
    t2 = r.trades[r.trades["date"] == day("2023-01-09")].set_index("symbol")
    assert set(t2.index) == {"AAA", "BBB", "CCC"}
    assert abs(t2.loc["AAA", "delta_w"]) < 1e-15  # weight unchanged ...
    assert t2.loc["AAA", "side"] == "sell"  # ... dollars are not
    assert abs(t2.loc["AAA", "notional"] - 0.5 * rate * e_pre2) < 1e-9
    assert abs(r.daily["gross"].loc[day("2023-01-09")] - 1.0) < 1e-12
    assert_reconciles(r, px, None, rate, 10_000.0)


# --- 5. dust -------------------------------------------------------------------


def test_dust_leg_keeps_its_holdings():
    """A dust leg is neither traded, charged nor logged; its holdings stay as drifted."""
    # Week 1: 50/50 AAA/BBB. AAA ticks up 1e-6 on Tuesday, so its drifted
    # weight is ~0.5 + 2.5e-7 — within TRADE_EPS of the 0.5 target. Week 2
    # also splits BBB into BBB/CCC, so the rebalance happens but AAA's leg is
    # dust: not traded, not charged, not logged, holdings exactly as drifted.
    # Then AAA rises 10%, so a position silently snapped to 0.5 would show up
    # as an equity mismatch in the audit.
    idx = make_index(12)
    syms = ["AAA", "BBB", "CCC"]
    px = pd.DataFrame(100.0, index=idx, columns=syms)
    px.loc[idx >= day("2023-01-03"), "AAA"] = 100.0001
    px.loc[idx >= day("2023-01-10"), "AAA"] = 110.0
    w2 = {"AAA": 0.5, "BBB": 0.25, "CCC": 0.25}
    f = factor_table(idx, syms, lambda d, s: (s != "CCC") * 0.5 if d < day("2023-01-08") else w2[s])
    r = run(px, f)  # zero fees: x == 1 exactly
    t2 = r.trades[r.trades["date"] == day("2023-01-09")]
    assert set(t2["symbol"]) == {"BBB", "CCC"}, "the dust leg must not be logged"
    held_aaa = 0.5 * 10_000.0 * 100.0001 / 100.0  # dollars, untouched
    e_mon2 = r.daily["equity"].loc[day("2023-01-08")]
    drifted = held_aaa / e_mon2
    assert 0 < drifted - 0.5 < B.TRADE_EPS
    assert abs(r.daily["gross"].loc[day("2023-01-09")] - (drifted + 0.25 + 0.25)) < 1e-12
    assert_reconciles(r, px, None, 0.0, 10_000.0)


# --- 6. the fixed point --------------------------------------------------------


def test_fee_fixed_point_is_exact():
    """The fee fixed point matches its closed forms and leaves a ~1e-16 residual."""
    rate = 10 / 1e4
    x, live = B._solve_fill(np.array([1.0]), np.array([0.0]), rate)
    assert abs(x - 1 / (1 + rate)) < 1e-15 and live.all()
    x, _ = B._solve_fill(np.array([0.5, 0.5]), np.array([1.0, 0.0]), rate)
    assert abs(x - (1 - rate)) < 1e-15
    x, live = B._solve_fill(np.zeros(3), np.zeros(3), rate)
    assert x == 1.0 and not live.any()
    rng = np.random.default_rng(3)
    for _ in range(200):
        t = rng.normal(0, 1, 8)
        t /= np.abs(t).sum() / rng.uniform(0.2, 3.0)  # gross 0.2 .. 3
        p = rng.normal(0, 0.3, 8)
        rr = rng.uniform(0, 0.02)  # up to 200 bps
        x, live = B._solve_fill(t, p, rr)
        resid = x - (1 - rr * np.abs(t[live] * x - p[live]).sum())
        assert abs(resid) < 1e-14, resid
        # the dust split is consistent with the solved x
        assert np.array_equal(live, np.abs(t * x - p) > B.TRADE_EPS)


# --- 7. the audit has teeth ----------------------------------------------------


def test_audit_finds_a_log_that_differs_from_the_pnl():
    """A log whose notional and fee use the pre-fee equity ($10,000 and $10: 100 contracts,
    implying $19,990) describes holdings that the P&L does not hold ($19,980.02): the audit
    finds the equity and the new holding."""
    px, f = two_price_world()
    rate = 10 / 1e4
    res = run(px, f, fee_bps=10.0)
    assert_reconciles(res, px, None, rate, 10_000.0)
    trades = res.trades.copy()
    trades["notional"] = trades["delta_w"].abs() * 10_000.0  # |delta_w| x E_pre
    trades["cost"] = trades["notional"] * rate
    bad = audit(B.Result(res.equity, trades, res.daily, res.stats), px, None, rate, 10_000.0)
    assert any("equity" in b for b in bad), bad
    assert any("new holding" in b for b in bad), bad


# --- 8. tags and guards ------------------------------------------------------------


def test_accounting_tag_and_guards():
    """Each run carries the accounting tag reconciled-v2; other modes and bad inputs are refused."""
    px, f = two_price_world()
    assert run(px, f).stats.get("accounting") == B.ACCOUNTING_RECONCILED == "reconciled-v2"
    assert B.ACCOUNTING_MODES == (B.ACCOUNTING_RECONCILED,)
    for bad_mode in ("v2", "", None, "other"):
        try:
            run(px, f, accounting=bad_mode)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accounting={bad_mode!r} was accepted")
    try:  # 100% fee on a gross-1 book
        run(px, f, fee_bps=10_000.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a fee rate that eats the whole notional was accepted")


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
    print(f"\n{len(tests) - failed}/{len(tests)} accounting reconciliation tests passed")
    raise SystemExit(1 if failed else 0)
