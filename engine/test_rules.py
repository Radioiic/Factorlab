"""Tests of rules.py: the historical test, the forward confidence sequence, the verdict and the
expected number of false "useful" verdicts.

Usage:
    python -B test_rules.py

The suite reads no data folder of the repository. At the start it sets FACTORLAB_DATA to a new
folder under the git-ignored engine/.tmp/ and removes that folder at the end. The book runs use
synthetic prices in memory.

The properties:
    1. the mixture parameter r at week 26, and the bounds and the long-run SD against plain loops
    2. the states: no decision before week 26; "useful" when the lower bound is above 0;
       "useless" when the upper bound is below 0, or at week 156; the first decision is final
    3. the false "useful" rate under a zero mean (normal, Student t3, GARCH, AR(1) up to 0.3, a
       volatility increase) is at most 10 %, and the crash-skew limit is the measured number
    4. the power for an annual excess Sharpe ratio of 2.0
    5. the rolling windows, the boundaries of the historical test and the order of its reasons
    6. for the control "zero" the historical numbers are the book's own statistics; for the
       control "equal_weight" the test reads the net excess return
    7. the sigma floor comes from the weeks with a factor position, or from the fallback
    8. BTC hold enters no rule
    9. the verdict table, the expected number of false "useful" verdicts, the rule record and
       the rule text
"""

from __future__ import annotations

import atexit
import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
(ENGINE / ".tmp").mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="test_rules_", dir=ENGINE / ".tmp"))
atexit.register(shutil.rmtree, TMP, True)
os.environ["FACTORLAB_DATA"] = str(TMP / "data")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import book  # noqa: E402
import config as C  # noqa: E402
import rules  # noqa: E402

SYMS = [C.sym_to_file(s) for s in C.UNIVERSE]
EPS = 2.0**-10  # an exact binary step for the made daily returns


# --------------------------------------------------------------------------- helpers


def plain_lrsd(x: list[float], lags: int = 4) -> list[float | None]:
    """The running Bartlett long-run SD with a double loop (no cumulative sums)."""
    out: list[float | None] = []
    for t in range(1, len(x) + 1):
        if t == 1:
            out.append(None)
            continue
        m = sum(x[:t]) / t
        d = [v - m for v in x[:t]]
        total = sum(v * v for v in d) / (t - 1)
        for lag in range(1, min(lags, t - 1) + 1):
            g = sum(d[i] * d[i - lag] for i in range(lag, t)) / (t - 1)
            total += 2 * (1 - lag / (lags + 1)) * g
        out.append(math.sqrt(max(total, 0.0)))
    return out


def plain_band(x: list[float], floor: float, r: float, alpha: float = 0.10) -> list[tuple]:
    """(mean, sigma, lower, upper) after each week, with plain Python sums."""
    lr = plain_lrsd(x)
    out = []
    for t in range(1, len(x) + 1):
        m = sum(x[:t]) / t
        sd = math.sqrt(sum((v - m) ** 2 for v in x[:t]) / (t - 1)) if t > 1 else None
        sig = floor if sd is None else max(sd, lr[t - 1] or 0.0, floor)
        h = sig * math.sqrt((t + r) * math.log((t + r) / (r * alpha**2))) / t
        out.append((m, sig, m - h, m + h))
    return out


def world(seed: int = 1, n_days: int = 600, drift=None, vol: float = 0.03):
    """Synthetic daily closes of the 20 engine symbols from Sunday 2023-01-01, small random
    funding, and every coin eligible."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")
    mu = np.zeros(len(SYMS)) if drift is None else np.asarray(drift, dtype=np.float64)
    steps = rng.normal(0.0, vol, (n_days, len(SYMS))) + mu
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=SYMS)
    funding = pd.DataFrame(rng.normal(1e-4, 2e-4, (n_days, len(SYMS))), index=idx, columns=SYMS)
    eligible = pd.DataFrame(True, index=idx, columns=SYMS)
    return prices, funding, eligible


def sunday_values(prices: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sundays = prices.index[prices.index.weekday == 6]
    return pd.DataFrame(rng.normal(0.0, 1.0, (len(sundays), len(SYMS))), index=sundays, columns=SYMS)


def run(spec, prices, funding, eligible, values):
    sundays = prices.index[prices.index.weekday == 6]
    return book.run_book(
        spec, values, start=sundays[0], end=sundays[-1], prices=prices, funding=funding, eligible=eligible
    )


def hist_of(r) -> dict:
    wk = book.weekly_table(r)
    return rules.historical_test(book.daily_excess(r), wk["excess"], held=wk["held"], fallback_floor=0.03)


def daily(values, start: str = "2023-01-01") -> tuple[pd.Series, pd.Series]:
    """(daily excess series from day 1, weekly series of the complete weeks) of made returns."""
    n = len(values) + 1
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    d = pd.Series(np.asarray(values, dtype=np.float64), index=idx[1:])
    weeks = (n - 1) // 7
    wk = pd.Series(np.zeros(weeks), index=idx[0 : 7 * weeks : 7])
    return d, wk


def held_all(wk: pd.Series) -> pd.Series:
    return pd.Series(True, index=wk.index)


# --- process generators of the simulation (units of the true weekly SD) -----------------


def gen(kind: str, n: int, weeks: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "normal":
        return rng.standard_normal((n, weeks))
    if kind == "t3":
        return rng.standard_t(3, (n, weeks)) / math.sqrt(3.0)
    if kind.startswith("ar"):
        phi = float(kind[2:])
        e = rng.standard_normal((n, weeks + 60)) * math.sqrt(1 - phi * phi)
        x = np.empty_like(e)
        x[:, 0] = rng.standard_normal(n)
        for i in range(1, e.shape[1]):
            x[:, i] = phi * x[:, i - 1] + e[:, i]
        return x[:, 60:]
    if kind == "garch":
        a, b, df = 0.10, 0.85, 5
        z = rng.standard_t(df, (n, weeks + 300)) / math.sqrt(df / (df - 2))
        h = np.ones(n)
        out = np.empty((n, weeks + 300))
        for i in range(weeks + 300):
            out[:, i] = np.sqrt(h) * z[:, i]
            h = (1 - a - b) + a * out[:, i] ** 2 + b * h
        return out[:, 300:]
    if kind.startswith("skew"):
        # steady small gains and a loss with probability p that holds 60 % of the variance
        p, vj = float(kind[4:]), 0.6
        c = math.sqrt(vj / (p * (1 - p)))
        hit = rng.random((n, weeks)) < p
        return math.sqrt(1 - vj) * rng.standard_normal((n, weeks)) + c * p - c * hit
    raise ValueError(kind)


def useful_rate(kind: str, *, n: int, seed: int, k: float = 1.0, floor: float | None = None, sharpe: float = 0.0):
    """The share of paths with the first decision "useful". The sigma floor comes from 52 history
    weeks of the same process (the rule of the historical test), or is the fixed `floor`."""
    rng = np.random.default_rng(seed)
    if floor is None:
        hist = gen(kind, n, rules.MIN_FLOOR_WEEKS, rng)
        floors = np.maximum(np.std(hist, axis=1, ddof=1), rules.long_run_sd(hist)[:, -1])
    else:
        floors = np.full(n, float(floor))
    x = k * (gen(kind, n, rules.MAX_WEEKS, rng) + sharpe / math.sqrt(52.0))
    useful = np.zeros(n, dtype=bool)
    b = rules.cs_bounds(x, sigma_floor=floors)  # one floor for each path
    code, first = rules.forward_states(b["lower"], b["upper"])
    decided = first >= 0
    useful[decided] = code[decided, first[decided]] == 2
    return float(useful.mean())


# --------------------------------------------------------------------------- 1. the boundary


def test_cs_mixture_r_at_week_26():
    from scipy.optimize import brentq

    r = rules.cs_mixture(rules.ALPHA, rules.TUNE_WEEK)
    assert abs(r - 3.9166) < 1e-4, r
    a, t = rules.ALPHA, rules.TUNE_WEEK
    root = brentq(lambda q: math.log((t + q) / (q * a * a)) - t / q, 0.01, 1000.0, xtol=1e-14)
    assert abs(r - root) < 1e-9, (r, root)
    # r makes the half-width smallest at week 26: another r gives a wider band there
    w = lambda q: math.sqrt((t + q) * math.log((t + q) / (q * a * a)))  # noqa: E731
    assert all(w(r) < w(r * f) for f in (0.5, 0.9, 0.99, 1.01, 1.1, 2.0))
    assert rules.forward_test([], 0.03)["r"] == r
    for bad in ((0.0, 26), (1.0, 26), (0.1, 0)):
        try:
            rules.cs_mixture(*bad)
        except ValueError:
            continue
        raise AssertionError(f"cs_mixture{bad} was accepted")


def test_cs_bounds_equal_a_plain_loop():
    rng = np.random.default_rng(7)
    r = rules.cs_mixture(rules.ALPHA, rules.TUNE_WEEK)
    for floor, x in ((0.03, list(rng.normal(0.004, 0.035, 90))), (0.5, list(rng.normal(0.0, 0.1, 40)))):
        b = rules.cs_bounds(np.array(x), sigma_floor=floor)
        for i, (m, sig, lo, hi) in enumerate(plain_band(x, floor, r)):
            assert abs(b["mean"][i] - m) < 1e-12 and abs(b["sigma"][i] - sig) < 1e-12, i
            assert abs(b["lower"][i] - lo) < 1e-12 and abs(b["upper"][i] - hi) < 1e-12, i
        assert b["sigma"][0] == floor and np.isnan(b["sd"][0]) and np.isnan(b["lrsd"][0])
        # 2-D paths give the numbers of one path at a time
        two = rules.cs_bounds(np.array([x, x[::-1]]), sigma_floor=floor)
        one = rules.cs_bounds(np.array(x[::-1]), sigma_floor=floor)
        assert np.allclose(two["lower"][0], b["lower"], rtol=0, atol=1e-12)
        assert np.allclose(two["upper"][1], one["upper"], rtol=0, atol=1e-12)
    for bad in ([0.1, float("nan")], [0.1, float("inf")]):
        try:
            rules.cs_bounds(np.array(bad), sigma_floor=0.03)
        except ValueError:
            continue
        raise AssertionError("a value that is not finite was accepted")


def test_long_run_sd_equals_a_plain_loop():
    rng = np.random.default_rng(3)
    for x in (
        list(rng.normal(0.01, 0.04, 60)),
        list(100.0 + rng.normal(0.0, 0.01, 30)),
        [0.5, -0.5] * 10,
        [0.2, 0.1],
    ):
        got = rules.long_run_sd(np.array(x))
        want = plain_lrsd(x)
        assert np.isnan(got[0]) and want[0] is None
        for i in range(1, len(x)):
            w = want[i]
            assert w is not None and abs(got[i] - w) <= 1e-9 * max(1.0, w) + 1e-12, (i, got[i], w)
    # the lags are min(4, t - 1): at t = 2 and t = 3 the sum has one and two lags
    x = [0.3, -0.1, 0.2, 0.05, -0.2, 0.1]
    assert np.allclose(rules.long_run_sd(np.array(x))[1:], plain_lrsd(x)[1:], rtol=1e-12, atol=1e-15)
    assert rules.long_run_sd(np.array([])).shape == (0,)


# --------------------------------------------------------------------------- 2. the states


def strong(n: int, level: float = 0.05) -> list[float]:
    return [level + 0.002 * (-1) ** k for k in range(n)]


def test_no_decision_before_week_26():
    out = rules.forward_test(strong(40, 0.5), 0.03)
    assert out["path"][5]["lower"] > 0, "the fixture must have a lower bound above 0 early"
    assert all(p["state"] == "running" for p in out["path"][:25])
    assert out["path"][25]["state"] == "useful" and out["decided_week"] == 26
    assert rules.forward_test(strong(25, 0.5), 0.03)["state"] == "running"
    empty = rules.forward_test([], 0.03)
    assert empty["state"] == "waiting" and empty["weeks"] == 0 and empty["path"] == [] and empty["decided_week"] is None


def test_useful_from_week_26_when_lower_above_zero():
    out = rules.forward_test(strong(60), 0.03)
    assert out["state"] == "useful" and out["decided_week"] == 26
    # a later first week with the lower bound above 0
    x = [0.0] * 30 + strong(100, 0.02)
    out = rules.forward_test(x, 0.01)
    k = out["decided_week"]
    assert out["state"] == "useful" and k > 26, k
    lows = [p["lower"] for p in out["path"]]
    assert lows[k - 1] > 0 and all(lo <= 0 for lo in lows[25 : k - 1])
    assert all(p["state"] == "running" for p in out["path"][: k - 1])


def test_useless_from_week_26_when_upper_below_zero():
    out = rules.forward_test([-v for v in strong(60)], 0.03)
    assert out["state"] == "useless" and out["decided_week"] == 26
    assert out["path"][24]["upper"] < 0 and out["path"][24]["state"] == "running"


def test_useless_at_week_156_when_not_useful():
    x = [0.0005 + 0.05 * (-1) ** k for k in range(170)]
    out = rules.forward_test(x, 0.03)
    assert out["decided_week"] == 156 and out["state"] == "useless" and out["weeks"] == 170
    assert all(p["lower"] <= 0 <= p["upper"] for p in out["path"][:155])
    assert out["path"][154]["state"] == "running" and out["path"][155]["state"] == "useless"


def test_the_first_decision_is_final():
    x = strong(40, 0.05) + [-0.3] * 120  # useful at week 26, then far below 0
    out = rules.forward_test(x, 0.03)
    assert out["decided_week"] == 26 and out["state"] == "useful"
    assert all(p["state"] == "useful" for p in out["path"][25:])
    assert out["path"][-1]["upper"] < 0, "the fixture must have an upper bound below 0 later"
    code, first = rules.forward_states(np.array([[1.0] * 30, [-1.0] * 30]), np.array([[2.0] * 30, [-0.5] * 30]))
    assert list(first) == [25, 25] and (code[0, 25:] == 2).all() and (code[1, 25:] == 3).all()


# --------------------------------------------------------------------------- 3. and 4. simulation


def test_false_useful_rate_under_zero_mean():
    """4,000 paths of 156 weeks, a fixed seed, the floor from 52 history weeks of the same process
    (the shortest history that gives a floor from the history)."""
    t0 = time.time()
    rates = {}
    for i, (kind, k) in enumerate(
        (("normal", 1.0), ("t3", 1.0), ("garch", 1.0), ("ar0.1", 1.0), ("ar0.2", 1.0), ("ar0.3", 1.0), ("ar0.3", 1.5))
    ):
        rates[(kind, k)] = useful_rate(kind, n=4000, seed=100 + i, k=k)
    for key, rate in rates.items():
        assert rate <= 0.10, (key, rate)
    assert rates[("normal", 1.0)] <= 0.03 and rates[("t3", 1.0)] <= 0.03, rates
    # the documented limit: crash skew p 0.02 with a fixed floor of 0.5 SD (measured 10.2 %)
    skew = useful_rate("skew0.02", n=4000, seed=200, floor=0.5)
    assert abs(skew - 0.102) <= 0.02, skew
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"the simulation took {elapsed:.1f} s"
    print("        " + ", ".join(f"{a} k {b}: {r:.2%}" for (a, b), r in rates.items()) + f", crash skew: {skew:.2%}")


def test_power_for_a_modest_effect():
    rate = useful_rate("normal", n=2000, seed=300, sharpe=2.0)
    assert rate >= 0.65, rate


# --------------------------------------------------------------------------- 5. the historical test


def test_window_counts():
    for n, want in ((365, 0), (366, 1), (457, 2), (1650, 15)):
        idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
        w = list(rules.windows(idx))
        assert len(w) == want, (n, len(w))
        if w:
            assert w[0][0] == idx[0] and all(e - s == pd.Timedelta(days=365) for s, e in w)
            assert all(b[0] - a[0] == pd.Timedelta(days=91) for a, b in zip(w, w[1:]))
            eq = pd.Series(np.arange(1.0, n + 1.0), index=idx)
            s, e = w[-1]
            assert len(eq[(eq.index >= s) & (eq.index <= e)]) == 366
    assert list(rules.windows(pd.DatetimeIndex([]))) == []
    eq = pd.Series(np.linspace(1.0, 2.0, 400), index=pd.date_range("2023-01-01", periods=400, freq="D", tz="UTC"))
    s, e = next(rules.windows(eq.index))
    assert rules.window_sharpe(eq, s, e) == rules.sharpe(eq[(eq.index >= s) & (eq.index <= e)].pct_change().dropna())


def test_historical_boundaries():
    reasons = rules.REASONS
    # a total return of exactly 0: useless, with the total-return reason only
    d, wk = daily([1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    h = rules.historical_test(d, wk, held=held_all(wk), fallback_floor=0.03)
    assert h["total_return"] == 0.0 and h["sharpe"] > 0 and h["result"] == "useless"
    assert h["reasons"] == [reasons[0]] and h["n_windows"] == 0 and h["word"] == "useless"
    # a flat book: Sharpe 0 is useless
    d, wk = daily([0.0] * 14)
    h = rules.historical_test(d, wk, held=held_all(wk), fallback_floor=0.03)
    assert h["sharpe"] == 0.0 and h["result"] == "useless" and reasons[1] in h["reasons"]
    # no window and a negative total return (a positive arithmetic mean): the total-return reason only
    d, wk = daily([0.5, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    h = rules.historical_test(d, wk, held=held_all(wk), fallback_floor=0.03)
    assert h["total_return"] < 0 < h["sharpe"] and h["reasons"] == [reasons[0]] and h["n_windows"] == 0
    # no window, a positive total return and Sharpe: pass, with no 1-year window
    d, wk = daily([EPS, -EPS / 2] * 10)
    h = rules.historical_test(d, wk, held=held_all(wk), fallback_floor=0.03)
    assert h["result"] == "pass" and h["word"] == "unproven (historical pass)" and h["reasons"] == []
    assert h["n_windows"] == 0 and h["share_positive"] is None and h["windows"] == []
    # exactly half of the windows positive: pass (two windows: one positive, one negative)
    days = np.arange(1, 457)
    c = np.where(days <= 90, 2 * EPS, np.where(days > 365, -EPS, np.where(days % 2 == 0, EPS, -EPS)))
    d, wk = daily(c)
    h = rules.historical_test(d, wk, held=held_all(wk), fallback_floor=0.03)
    assert (h["n_windows"], h["n_positive"], h["share_positive"]) == (2, 1, 0.5), h["windows"]
    assert h["result"] == "pass" and h["total_return"] > 0 and h["sharpe"] > 0
    # 7 of 15 windows positive: useless, with the window reason only (blocks of 91 days + - - +,
    # and the last days, which are in no window, carry the total return above 0)
    days = np.arange(1, 1650)
    c = np.array([(1, -1, -1, 1)[((t - 1) // 91) % 4] for t in days], dtype=np.float64) * EPS
    c[days >= 1640] = 0.003
    d, wk = daily(c)
    h = rules.historical_test(d, wk, held=held_all(wk), fallback_floor=0.03)
    assert (h["n_windows"], h["n_positive"]) == (15, 7) and h["total_return"] > 0 and h["sharpe"] > 0
    assert h["result"] == "useless" and h["reasons"] == [reasons[2]], h["reasons"]
    assert len(h["windows"]) == 15 and h["windows"][0]["start"] == "2023-01-01"
    # all three conditions: the reasons in their order
    c[days >= 1640] = c[days == 1639][0]
    d, wk = daily(c)
    h = rules.historical_test(d, wk, held=held_all(wk), fallback_floor=0.03)
    assert h["reasons"] == list(reasons), h["reasons"]
    # no complete week: no historical test
    for d, wk in (daily([0.01] * 6), (pd.Series(dtype=float), pd.Series(dtype=float))):
        h = rules.historical_test(d, wk, held=pd.Series(dtype=bool), fallback_floor=0.04)
        assert h["result"] == "none" and h["word"] == "no historical test" and h["weeks"] == 0
        assert h["sigma_floor"] == 0.04 and h["sigma_floor_source"] == "fallback" and h["reasons"] == []
    d, wk = daily([0.01] * 7)
    assert set(rules.historical_test(d, wk, held=held_all(wk), fallback_floor=0.03)) == set(rules._none_result(0.03))


def test_zero_control_veto_equals_book_stats():
    prices, funding, eligible = world(seed=4, n_days=500)
    values = sunday_values(prices, seed=5)
    r = run(book.load_spec("long_short"), prices, funding, eligible, values)
    assert r.control is None
    h = hist_of(r)
    s = r.book.stats
    assert h["sharpe"] == s["sharpe"], (h["sharpe"], s["sharpe"])
    assert math.isclose(h["total_return"], s["total_return"], rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(h["max_dd"], s["max_dd"], rel_tol=1e-9, abs_tol=1e-12)
    assert h["n_windows"] == len(list(rules.windows(r.book.equity.index)))


def test_excess_series_for_the_equal_weight_control():
    """A long-only book that gains, but less than the equal-weight control: useless."""
    drift = np.linspace(0.0005, 0.004, len(SYMS))
    prices, funding, eligible = world(seed=6, n_days=420, drift=drift, vol=0.004)
    funding[:] = 0.0
    sundays = prices.index[prices.index.weekday == 6]
    values = pd.DataFrame(np.repeat(-drift[None, :], len(sundays), axis=0), index=sundays, columns=SYMS)
    r = run(book.load_spec("long_only"), prices, funding, eligible, values)
    assert r.control is not None and r.book.stats["total_return"] > 0, r.book.stats["total_return"]
    h = hist_of(r)
    assert h["total_return"] < 0 and h["result"] == "useless", h
    assert h["reasons"][0] == rules.REASONS[0]
    # the excess series is the arithmetic difference of the daily returns
    d = book.daily_excess(r)
    want = r.book.equity["equity"].pct_change() - r.control.equity["equity"].pct_change()
    assert np.allclose(d.to_numpy(), want.iloc[1:].to_numpy(), rtol=0, atol=1e-15)


def test_sigma_floor_from_factor_position_weeks_or_fallback():
    rng = np.random.default_rng(9)
    d, _ = daily(list(rng.normal(0.001, 0.01, 7 * 60)))
    sundays = pd.date_range("2023-01-01", periods=60, freq="7D", tz="UTC")
    x = rng.normal(0.0, 0.04, 60)
    held = np.ones(60, dtype=bool)
    held[[3, 9, 15, 21, 30, 40, 50, 55]] = False  # weeks in which the book has the control portfolio
    x[~held] = 1e-5
    wk, hv = pd.Series(x, index=sundays), pd.Series(held, index=sundays)
    h = rules.historical_test(d, wk, held=hv, fallback_floor=0.03)
    xh = list(x[held])
    want = max(float(np.std(xh, ddof=1)), plain_lrsd(xh)[-1] or 0.0)
    assert h["held_weeks"] == 52 and h["weeks"] == 60 and h["sigma_floor_source"] == "history"
    assert abs(h["sigma_floor"] - want) < 1e-12, (h["sigma_floor"], want)
    assert (
        abs(h["weekly_sd"] - float(np.std(xh, ddof=1))) < 1e-15 and abs(h["weekly_mean"] - float(np.mean(xh))) < 1e-15
    )
    dev = np.array(xh) - np.mean(xh)
    assert abs(h["weekly_ac1"] - float(np.sum(dev[1:] * dev[:-1]) / np.sum(dev * dev))) < 1e-12
    # all 60 weeks would give another floor: the missed weeks are left out
    all_weeks = rules.historical_test(d, wk, held=pd.Series(True, index=sundays), fallback_floor=0.03)
    assert all_weeks["sigma_floor"] != h["sigma_floor"]
    # 51 weeks with a factor position: the fallback
    held[0] = False
    h = rules.historical_test(d, wk, held=pd.Series(held, index=sundays), fallback_floor=0.03)
    assert h["held_weeks"] == 51 and h["sigma_floor_source"] == "fallback" and h["sigma_floor"] == 0.03
    # fewer than 3 weeks with a factor position: no autocorrelation
    few = rules.historical_test(
        *daily([0.01] * 14), held=pd.Series([True, False], index=sundays[:2]), fallback_floor=0.03
    )
    assert few["weekly_ac1"] is None and few["weekly_sd"] is None and few["held_weeks"] == 1


def test_btc_never_enters_a_rule():
    prices, funding, eligible = world(seed=11, n_days=460)
    eligible["BTC-USDT"] = False  # the books do not hold BTC, so only the context can change
    values = sunday_values(prices, seed=12)
    other = prices.copy()
    other["BTC-USDT"] = other["BTC-USDT"] * np.linspace(1.0, 3.0, len(other))
    for name in ("long_short", "long_only"):
        spec = book.load_spec(name)
        a = run(spec, prices, funding, eligible, values)
        b = run(spec, other, funding, eligible, values)
        wa, wb = book.weekly_table(a), book.weekly_table(b)
        assert not np.allclose(wa["btc_ret"], wb["btc_ret"]), "the fixture must change BTC hold"
        cols = [c for c in book.WEEKLY_COLUMNS if not c.startswith("btc")]
        pd.testing.assert_frame_equal(wa[cols], wb[cols])
        ha, hb = hist_of(a), hist_of(b)
        assert ha == hb, name
        fa = rules.forward_test(wa["excess"].to_numpy(), ha["sigma_floor"])
        fb = rules.forward_test(wb["excess"].to_numpy(), hb["sigma_floor"])
        assert fa == fb
    # no rule of the module takes a BTC input
    import inspect

    for fn in (rules.historical_test, rules.cs_bounds, rules.forward_test, rules.forward_states, rules.verdict):
        assert not any("btc" in p.lower() or "bench" in p.lower() for p in inspect.signature(fn).parameters), fn


# --------------------------------------------------------------------------- 9. verdict and record


def test_verdict_composition():
    table = [
        ({"state": "useful"}, "pass", "useful", "forward"),
        ({"state": "useful"}, "useless", "useful", "forward"),
        ({"state": "useless"}, "pass", "useless", "forward"),
        ({"state": "useless"}, "useless", "useless", "forward"),
        (None, "pass", "unproven (historical pass)", "historical"),
        ({"state": "waiting"}, "pass", "unproven (historical pass)", "historical"),
        ({"state": "running"}, "pass", "unproven (historical pass)", "historical"),
        (None, "useless", "useless (historical)", "historical"),
        ({"state": "waiting"}, "useless", "useless (historical)", "historical"),
        ({"state": "running"}, "useless", "useless (historical)", "historical"),
        (None, "none", "no historical test", "historical"),
    ]
    for fwd, hist, word, by in table:
        v = rules.verdict(hist, fwd)
        assert (v["word"], v["by"]) == (word, by), (fwd, hist, v)
        assert v["final"] is (by == "forward")
    # the forward test output itself is a valid input
    assert rules.verdict("useless", rules.forward_test(strong(30), 0.03))["word"] == "useful"
    assert rules.verdict("pass", rules.forward_test([0.01, 0.02], 0.03))["word"] == "unproven (historical pass)"
    for bad in (("maybe", None), ("pass", {"state": "confirm"})):
        try:
            rules.verdict(*bad)
        except ValueError:
            continue
        raise AssertionError(f"verdict{bad} was accepted")


def test_expected_false_useful():
    tests = [
        {"state": "waiting", "withdrawn": False},
        {"state": "running", "withdrawn": False},
        {"state": "useful", "withdrawn": False},
        {"state": "useless", "withdrawn": False},
        {"state": "waiting", "withdrawn": True},
    ]
    n = sum(1 for t in tests if not t["withdrawn"])  # every state counts; a withdrawal does not
    assert n == 4 and rules.expected_false_useful(n) == 0.1 * 4
    assert rules.expected_false_useful(0) == 0.0 and math.isclose(rules.expected_false_useful(7), 0.7)
    assert rules.expected_false_useful(10, alpha=0.05) == 0.05 * 10
    for bad in (-1, 1.5, True):
        try:
            rules.expected_false_useful(bad)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"expected_false_useful({bad!r}) was accepted")


def test_rule_record_and_text():
    rec = rules.rule_record()
    assert set(rec) == {
        "rule_version",
        "alpha",
        "sides",
        "tune_week",
        "r",
        "first_decision_week",
        "max_weeks",
        "sd_rule",
        "lr_lags",
        "min_floor_weeks",
        "window_days",
        "step_days",
        "min_share_positive",
        "statistic",
    }, sorted(rec)
    assert rec["rule_version"] == "judge-1" and rec["alpha"] == 0.10 and rec["sides"] == "two-sided"
    assert rec["tune_week"] == 26 and rec["first_decision_week"] == 26 and rec["max_weeks"] == 156
    assert (
        rec["sd_rule"] == "max(sample SD, long-run SD, floor)" and rec["lr_lags"] == 4 and rec["min_floor_weeks"] == 52
    )
    assert rec["window_days"] == 365 and rec["step_days"] == 91 and rec["min_share_positive"] == 0.5
    assert rec["r"] == float(f"{rules.cs_mixture(0.10, 26):.6g}") and abs(rec["r"] - 3.9166) < 1e-4
    import json

    assert json.loads(json.dumps(rec, allow_nan=False)) == rec, "the record must be plain JSON"
    assert len(rules.RULE_TEXT) == 14
    banned = ("tier", "stage", "promotable", "survivor", "novel", "confirmed", "retired", "effective")
    for line in rules.RULE_TEXT:
        assert line.isascii(), line
        bare = re.sub(r'"[^"]*"', "", line).lower()
        words = set(re.findall(r"[a-z]+", bare))
        assert not words & set(banned), (line, words & set(banned))
        assert not words & {"useful", "useless", "unproven"}, f"a verdict word outside quotes: {line}"
        for sentence in re.split(r"(?<=[.:])\s+", line.lstrip("- ")):
            n = len(sentence.split())
            assert 0 < n <= 25, (n, sentence)
    text = " ".join(rules.RULE_TEXT)
    for need in ("week 26", "week 156", "10 %", "BTC hold gives context only", "1-year window"):
        assert need in text, need
    # the text names the verdict word that a registered test shows before a forward decision
    assert f'Its verdict is "{rules.WORD_BEFORE_DECISION["useless"]}" until the forward test decides.' in text


# --- runner -------------------------------------------------------------------------------

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
    print(f"\n{len(tests) - failed}/{len(tests)} rule tests passed")
    raise SystemExit(1 if failed else 0)
