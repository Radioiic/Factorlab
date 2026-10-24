"""The rules of a factor test: the historical test, the forward confidence sequence, the
verdict and the expected number of false "useful" verdicts.

This module holds only statistics. It reads no file and writes no file. The input of each
rule is the net excess return of a book over its control book, after costs (book.py).

THE HISTORICAL TEST (historical_test). The daily net excess return is
d(t) = R_book(t) - R_control(t), and the excess equity X starts at 1 on the seed Sunday and
grows by (1 + d(t)) each day. The result is "useless" when one of these conditions is true:
    the total return X(end) - 1 is 0 or less;
    the Sharpe ratio of d is 0 or less;
    the history has at least one rolling 1-year window, and fewer than half of the windows
        have a positive Sharpe ratio.
Else the result is "pass" ("unproven (historical pass)"). A history without a complete week
gives the result "none" (no historical test). The historical test never gives "useful".
Sharpe ratio: mean / SD (ddof 1) x sqrt(365) of the daily returns, 0.0 for fewer than two
returns or an SD of 0. Windows: 365 days, a step of 91 days, from the first row.

THE FORWARD CONFIDENCE SEQUENCE (cs_bounds, forward_test). x_1 .. x_t are the weekly net excess
returns of the complete forward weeks. After week t:
    mean_t  = (x_1 + ... + x_t) / t
    s_t     = the sample SD (ddof 1) of x_1 .. x_t
    l_t     = the Bartlett long-run SD with LR_LAGS lags (divisor t - 1)
    sigma_t = the largest of s_t, l_t and the sigma floor (the floor at t = 1)
    half_t  = sigma_t * sqrt((t + r) ln((t + r) / (r alpha^2))) / t
    bounds  = mean_t - half_t and mean_t + half_t
r is the mixture parameter that makes half_t smallest at week TUNE_WEEK (cs_mixture). This is
the two-sided normal-mixture boundary of the anytime-valid confidence-sequence literature,
with alpha ALPHA. The first decision is final:
    "useful"   the first week t >= FIRST_DECISION_WEEK with lower_t > 0
    "useless"  the first week t >= FIRST_DECISION_WEEK with upper_t < 0, or week MAX_WEEKS
The design value of the false "useful" rate of one test is ALPHA or less. It is not a
guarantee for every return process (strong serial correlation, a volatility increase after the
registration or a crash-prone book can make it larger).

THE VERDICT (verdict). A forward decision is the verdict. Before it, the historical result
gives the word: "unproven (historical pass)" or "useless (historical)".

THE EXPECTED NUMBER OF FALSE "useful" VERDICTS (expected_false_useful): ALPHA x N for N tests.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

RULE_VERSION = "judge-1"
ALPHA = 0.10
SIDES = "two-sided"
TUNE_WEEK = 26  # the mixture is tuned for the first decision week
FIRST_DECISION_WEEK = 26
MAX_WEEKS = 156
LR_LAGS = 4  # Bartlett (Newey-West) long-run SD
MIN_FLOOR_WEEKS = 52  # history weeks with a factor position for a sigma floor from the history
WINDOW_DAYS, STEP_DAYS, MIN_SHARE_POSITIVE = 365, 91, 0.5
ANNUAL_DAYS = 365
FORWARD_STATES = ("waiting", "running", "useful", "useless")
HISTORICAL_RESULTS = ("useless", "pass", "none")  # "none": no complete history week (check only)
HISTORICAL_WORDS = dict(zip(HISTORICAL_RESULTS, ("useless", "unproven (historical pass)", "no historical test")))
# the verdict word of each historical result while the forward test has no decision
WORD_BEFORE_DECISION = dict(
    zip(HISTORICAL_RESULTS, ("useless (historical)", "unproven (historical pass)", "no historical test"))
)
SD_RULE = "max(sample SD, long-run SD, floor)"
STATISTIC = "the weekly net excess return of the book over the control book, after costs"
REASONS = (
    "The total return after costs is 0 or less.",
    "The Sharpe ratio after costs is 0 or less.",
    "Fewer than half of the 1-year windows have a positive Sharpe ratio.",
)
RULE_TEXT: tuple[str, ...] = (
    'The historical test can give only "useless" or "unproven".',
    'The historical test gives "useless" when one of these conditions is true after costs:',
    "- The total return is 0 or less.",
    "- The Sharpe ratio is 0 or less.",
    "- Fewer than half of the rolling 1-year windows have a positive Sharpe ratio. "
    "This condition applies only when the history has a 1-year window.",
    'You can register a factor that the historical test calls "useless". '
    'Its verdict is "useless (historical)" until the forward test decides.',
    'Only the forward test can give "useful". It uses only the weeks after the registration.',
    "A forward decision replaces the historical result. "
    "A factor without a complete week of history cannot be registered.",
    "Each week, the test compares the net return of the book with the net return of the control book.",
    "A confidence sequence decides. Its design value for the false-pass rate of each test is 10 % or less.",
    "Strong serial correlation, or a large increase of volatility after the registration, "
    "can make the false-pass rate larger.",
    'From week 26: "useful" when the lower bound is more than 0, "useless" when the upper bound is less than 0.',
    'At week 156, a test that is not "useful" is "useless".',
    "BTC hold gives context only.",
)


def _sig(x: float, digits: int = 6) -> float:
    return float(f"{x:.{digits}g}")


def _f(x) -> float | None:
    """A finite float, or None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# --------------------------------------------------------------------------- Sharpe ratio and windows


def sharpe(returns: pd.Series | np.ndarray) -> float:
    """mean / SD (ddof 1) x sqrt(365) of daily returns; 0.0 for fewer than 2 returns or SD 0."""
    r = pd.Series(np.asarray(returns, dtype=np.float64))
    if len(r) > 1 and r.std() > 0:
        return float(r.mean() / r.std() * np.sqrt(ANNUAL_DAYS))
    return 0.0


def windows(
    index: pd.DatetimeIndex, length_days: int = WINDOW_DAYS, step_days: int = STEP_DAYS
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield (start, end) rolling windows over a date index, from its first date."""
    if len(index) == 0:
        return
    start, last = index[0], index[-1]
    length, step = pd.Timedelta(days=length_days), pd.Timedelta(days=step_days)
    while start + length <= last:
        yield start, start + length
        start += step


def window_sharpe(equity: pd.Series, start, end) -> float:
    """The Sharpe ratio of the daily returns of equity on start <= date <= end."""
    eq = equity[(equity.index >= start) & (equity.index <= end)]
    dr = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return sharpe(dr)


# --------------------------------------------------------------------------- running SDs


def _running(x: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """(s_t, l_t): the running sample SD and the running Bartlett long-run SD of x along its last
    axis. Both are NaN at t = 1. The sums use cumulative sums (O(t x lags)); x is shifted by its
    first value first, which does not change a deviation from the mean."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    if n == 0:
        return x.copy(), x.copy()
    y = x - x[..., :1]
    t = np.arange(1, n + 1, dtype=np.float64)
    s1 = np.cumsum(y, axis=-1)
    m = s1 / t
    den = np.maximum(t - 1.0, 1.0)
    g0 = np.maximum(np.cumsum(y * y, axis=-1) - t * m * m, 0.0) / den
    lr = g0.copy()
    zero = np.zeros((*y.shape[:-1], 1))
    s1p = np.concatenate([zero, s1], axis=-1)  # s1p[..., j] = the sum of the first j values
    tt = np.arange(1, n + 1)
    for lag in range(1, lags + 1):
        if lag >= n:
            break
        prod = np.zeros_like(y)
        prod[..., lag:] = y[..., lag:] * y[..., :-lag]
        p = np.cumsum(prod, axis=-1)
        a = s1 - s1p[..., np.minimum(lag, tt)]  # sum of y_i, i = lag+1 .. t
        b = s1p[..., np.maximum(tt - lag, 0)]  # sum of y_i, i = 1 .. t-lag
        g = (p - m * (a + b) + np.maximum(t - lag, 0.0) * m * m) / den
        lr += 2.0 * (1.0 - lag / (lags + 1.0)) * np.where(tt > lag, g, 0.0)
    s = np.where(t > 1, np.sqrt(g0), np.nan)
    lrsd = np.where(t > 1, np.sqrt(np.maximum(lr, 0.0)), np.nan)
    return s, lrsd


def long_run_sd(x, lags: int = LR_LAGS) -> np.ndarray:
    """The running Bartlett long-run SD l_t (module docstring), the shape of x. NaN at t = 1."""
    return _running(np.asarray(x, dtype=np.float64), lags)[1]


# --------------------------------------------------------------------------- the historical test


def _none_result(fallback_floor: float) -> dict:
    return {
        "result": "none",
        "word": HISTORICAL_WORDS["none"],
        "reasons": [],
        "total_return": None,
        "sharpe": None,
        "n_windows": 0,
        "n_positive": 0,
        "share_positive": None,
        "windows": [],
        "weeks": 0,
        "held_weeks": 0,
        "weekly_mean": None,
        "weekly_sd": None,
        "weekly_lrsd": None,
        "weekly_ac1": None,
        "max_dd": None,
        "sigma_floor": float(fallback_floor),
        "sigma_floor_source": "fallback",
    }


def historical_test(
    daily_excess: pd.Series, weekly_excess: pd.Series, *, held: pd.Series, fallback_floor: float
) -> dict:
    """The historical test (module docstring) of a book run from the seed Sunday S0 to the end E.

    daily_excess: the daily net excess returns of the rows S0 + 1 .. E (book.daily_excess).
    weekly_excess: the weekly net excess returns, one for each signal Sunday S0 .. E - 7.
    held: True for each signal Sunday at which the book has its factor position.
    fallback_floor: the sigma floor when the weeks with a factor position are fewer than MIN_FLOOR_WEEKS.
    The numbers are full floats; the caller rounds them when it writes them."""
    floor_fb = float(fallback_floor)
    d = pd.Series(daily_excess, dtype=np.float64)
    wk = pd.Series(weekly_excess, dtype=np.float64)
    if len(wk) == 0 or len(d) == 0:
        return _none_result(floor_fb)
    hv = pd.Series(held).reindex(wk.index)
    is_held = np.array([False if pd.isna(v) else bool(v) for v in hv], dtype=bool)
    seed = pd.DatetimeIndex([d.index[0] - pd.Timedelta(days=1)])
    x = pd.concat([pd.Series([1.0], index=seed), (1.0 + d).cumprod()])
    total = float(x.iloc[-1] - 1.0)
    sr = sharpe(d.to_numpy())
    wins: list[dict[str, Any]] = []
    for s, e in windows(pd.DatetimeIndex(x.index)):
        wins.append({"start": s.strftime("%Y-%m-%d"), "end": e.strftime("%Y-%m-%d"), "sharpe": window_sharpe(x, s, e)})
    n_windows = len(wins)
    n_positive = sum(1 for w in wins if w["sharpe"] > 0)
    share = n_positive / n_windows if n_windows else None
    reasons = []
    if total <= 0:
        reasons.append(REASONS[0])
    if sr <= 0:
        reasons.append(REASONS[1])
    if share is not None and share < MIN_SHARE_POSITIVE:
        reasons.append(REASONS[2])
    result = "useless" if reasons else "pass"
    xh = wk.to_numpy()[is_held]
    nh = len(xh)
    sd = float(np.std(xh, ddof=1)) if nh >= 2 else None
    lrsd = float(long_run_sd(xh)[-1]) if nh >= 2 else None
    ac1 = None
    if nh >= 3:
        dev = xh - xh.mean()
        g0 = float(np.sum(dev * dev))
        ac1 = float(np.sum(dev[1:] * dev[:-1]) / g0) if g0 > 0 else None
    floor, source = floor_fb, "fallback"
    if nh >= MIN_FLOOR_WEEKS and sd is not None and lrsd is not None:
        floor_h = max(sd, lrsd)
        if math.isfinite(floor_h) and floor_h > 0:
            floor, source = floor_h, "history"
    return {
        "result": result,
        "word": HISTORICAL_WORDS[result],
        "reasons": reasons,
        "total_return": total,
        "sharpe": sr,
        "n_windows": n_windows,
        "n_positive": n_positive,
        "share_positive": share,
        "windows": wins,
        "weeks": len(wk),
        "held_weeks": nh,
        "weekly_mean": float(xh.mean()) if nh else None,
        "weekly_sd": sd,
        "weekly_lrsd": lrsd,
        "weekly_ac1": ac1,
        "max_dd": float((x / x.cummax() - 1.0).min()),
        "sigma_floor": floor,
        "sigma_floor_source": source,
    }


# --------------------------------------------------------------------------- the forward test


def cs_mixture(alpha: float, t_star: int) -> float:
    """The mixture parameter r that makes the half-width smallest at week t_star.

    It minimises (t* + r) ln((t* + r) / (r alpha^2)); the derivative is zero where
    ln((t* + r) / (r alpha^2)) = t* / r. The left side minus the right side increases with r,
    so a bisection on ln r finds the root."""
    if not (0 < alpha < 1) or t_star < 1:
        raise ValueError("the confidence sequence needs 0 < alpha < 1 and a tune week >= 1")

    def g(r: float) -> float:
        return math.log((t_star + r) / (r * alpha * alpha)) - t_star / r

    lo, hi = 1e-9 * t_star, 1e9 * t_star
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    return math.sqrt(lo * hi)


def cs_bounds(x, *, alpha: float = ALPHA, sigma_floor: float, tune_week: int = TUNE_WEEK, lags: int = LR_LAGS) -> dict:
    """The bounds after each week (module docstring). x: the weekly values, shape (weeks,) or
    (paths, weeks), finite. sigma_floor: a number, or one number for each path. Returns arrays of
    the shape of x: mean, sd and lrsd (NaN at week 1), sigma, half (the half-width), lower, upper;
    and r."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError("x must have the shape (weeks,) or (paths, weeks)")
    if not np.isfinite(x).all():
        raise ValueError("the weekly values must be finite")
    floor = np.asarray(sigma_floor, dtype=np.float64)
    if not (np.isfinite(floor).all() and (floor >= 0).all()):
        raise ValueError("the sigma floor must be a finite number of 0 or more")
    if floor.ndim == 1 and x.ndim == 2 and len(floor) == x.shape[0]:
        floor = floor[:, None]
    elif floor.ndim != 0:
        raise ValueError("the sigma floor must be a number, or one number for each path")
    r = cs_mixture(float(alpha), int(tune_week))
    t = np.arange(1, x.shape[-1] + 1, dtype=np.float64)
    mean = np.cumsum(x, axis=-1) / t
    sd, lrsd = _running(x, lags)
    sigma = np.fmax(np.fmax(sd, lrsd), floor)
    half = sigma * np.sqrt((t + r) * np.log((t + r) / (r * alpha * alpha))) / t
    return {
        "mean": mean,
        "sd": sd,
        "lrsd": lrsd,
        "sigma": sigma,
        "half": half,
        "lower": mean - half,
        "upper": mean + half,
        "r": r,
    }


def forward_states(lower, upper) -> tuple[np.ndarray, np.ndarray]:
    """(the state code after each week, the index of the deciding week or -1). Codes: 1 running,
    2 useful, 3 useless (the positions in FORWARD_STATES). The arrays are (weeks,) or
    (paths, weeks). After the deciding week the code stays the decision."""
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    t = np.arange(1, lower.shape[-1] + 1)
    after = t >= FIRST_DECISION_WEEK
    useful = after & (lower > 0)
    useless = after & ~useful & ((upper < 0) | (t >= MAX_WEEKS))
    code = np.where(useful, 2, np.where(useless, 3, 1))
    decided = code > 1
    first = np.where(decided.any(axis=-1), decided.argmax(axis=-1), -1)
    if code.shape[-1]:
        two = code.reshape(-1, code.shape[-1])
        f = np.atleast_1d(first)
        dec = two[np.arange(len(two)), np.maximum(f, 0)]
        hold = (f[:, None] >= 0) & (np.arange(two.shape[1])[None, :] >= f[:, None])
        code = np.where(hold, dec[:, None], two).reshape(code.shape)
    return code, first


def forward_test(x: Sequence[float], sigma_floor: float) -> dict:
    """The forward test of one registered test from its frozen weekly net excess returns (in
    week order) and its registered sigma floor. Read-only; the module docstring has the rule."""
    xs = np.asarray(list(x), dtype=np.float64)
    out: dict = {
        "state": "waiting",
        "weeks": 0,
        "decided_week": None,
        "mean": None,
        "sd": None,
        "lrsd": None,
        "sigma": None,
        "lower": None,
        "upper": None,
        "r": cs_mixture(ALPHA, TUNE_WEEK),
        "path": [],
    }
    if len(xs) == 0:
        return out
    b = cs_bounds(xs, sigma_floor=sigma_floor)
    code, first = forward_states(b["lower"], b["upper"])
    n = len(xs)
    out.update(
        {
            "state": FORWARD_STATES[int(code[-1])],
            "weeks": n,
            "decided_week": int(first) + 1 if int(first) >= 0 else None,
            "mean": _f(b["mean"][-1]),
            "sd": _f(b["sd"][-1]),
            "lrsd": _f(b["lrsd"][-1]),
            "sigma": _f(b["sigma"][-1]),
            "lower": _f(b["lower"][-1]),
            "upper": _f(b["upper"][-1]),
        }
    )
    out["path"] = [
        {
            "week": i + 1,
            "excess": float(xs[i]),
            "mean": _f(b["mean"][i]),
            "sd": _f(b["sd"][i]),
            "lrsd": _f(b["lrsd"][i]),
            "sigma": _f(b["sigma"][i]),
            "lower": _f(b["lower"][i]),
            "upper": _f(b["upper"][i]),
            "state": FORWARD_STATES[int(code[i])],
        }
        for i in range(n)
    ]
    return out


# --------------------------------------------------------------------------- verdict and counts


def verdict(historical_result: str, forward: dict | None) -> dict:
    """The verdict of a test: {"word", "by", "final"}.

    A forward decision ("useful" or "useless") is the verdict (by "forward", final). Before it,
    the historical result gives the word: "unproven (historical pass)", "useless (historical)",
    or "no historical test" (a check without history; not a verdict word)."""
    if historical_result not in HISTORICAL_RESULTS:
        raise ValueError(f"unknown historical result {historical_result!r}")
    state = (forward or {}).get("state")
    if state is not None and state not in FORWARD_STATES:
        raise ValueError(f"unknown forward state {state!r}")
    if state in ("useful", "useless"):
        return {"word": state, "by": "forward", "final": True}
    return {"word": WORD_BEFORE_DECISION[historical_result], "by": "historical", "final": False}


def expected_false_useful(n: int, alpha: float = ALPHA) -> float:
    """The expected number of false "useful" verdicts of n tests: alpha x n. It holds when each
    test meets the design value of its false-pass rate (a sum of expectations)."""
    if isinstance(n, bool) or int(n) != n or n < 0:
        raise ValueError("n must be a whole number of 0 or more")
    return alpha * int(n)


def rule_record() -> dict:
    """The rule of this version, as the registration of a test records it."""
    return {
        "rule_version": RULE_VERSION,
        "alpha": ALPHA,
        "sides": SIDES,
        "tune_week": TUNE_WEEK,
        "r": _sig(cs_mixture(ALPHA, TUNE_WEEK)),
        "first_decision_week": FIRST_DECISION_WEEK,
        "max_weeks": MAX_WEEKS,
        "sd_rule": SD_RULE,
        "lr_lags": LR_LAGS,
        "min_floor_weeks": MIN_FLOOR_WEEKS,
        "window_days": WINDOW_DAYS,
        "step_days": STEP_DAYS,
        "min_share_positive": MIN_SHARE_POSITIVE,
        "statistic": STATISTIC,
    }
