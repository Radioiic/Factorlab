"""Synthetic market generator for AlphaForge.

The simulator produces a complete daily equity market (prices, volumes, fundamentals,
industries, trading halts) whose returns contain a small number of *known* cross-sectional
alphas.  Because the alphas are planted on purpose, every downstream module -- operators,
factor evaluation, the genetic miner, the combiner -- can be validated against ground truth,
which is impossible with real market data.

The simulator makes markets with planted signals: the miner must find them again, and a
look-ahead bug shows as an IC that is too large.

Return model (daily log returns)::

    r[i, t] = beta[i] * f_mkt[t] + f_ind[g(i), t] + m[t] * sum_k a_k * z_k[i, t-1] + eps[i, t]

* ``f_mkt`` / ``f_ind`` are i.i.d. normal market / industry factor returns.
* ``z_k`` are cross-sectional z-scores of quantities computable from history up to ``t-1``,
  so the alphas are *legitimately* discoverable.  With ``s`` the observation date
  (``s = t-1`` for the return on ``t``):

    ==========  =====================================================  ================
    key         observable quantity at the close of day ``s``          coefficient
    ==========  =====================================================  ================
    reversal    ``close[s] / close[s-5] - 1``                          ``alpha_reversal``
    momentum    ``close[s-20] / close[s-120] - 1``                     ``alpha_momentum``
    volume      ``log(volume[s] / mean(volume[s-19..s]))`` (NaN-skip)  ``alpha_volume``
    value       ``log(bvps_pit[s] / close[s])`` (point-in-time B/P)     ``alpha_value``
    ==========  =====================================================  ================

  Each z-score is cross-sectional (mean 0, std 1 across assets on day ``s``).  Where the
  quantity is undefined (insufficient history, halted volume) the z-score is NaN and the
  contribution to the next day's return is 0.  The value alpha is z-scored on ``log(bp)``
  rather than raw ``bp``: book-to-price is log-normal across names, so a raw z-score is
  dominated by a handful of deep-value outliers; the log is rank-equivalent (``cs_rank(bp)``
  rediscovers it identically) and gives every name a proportionate weight.
* ``m[t] = 1 + alpha_regime_amp * sin(2*pi*t/500)`` is a slow regime multiplier so alpha
  strength drifts over time (as in real markets) without flipping sign (``amp < 1``).
* ``eps`` is idiosyncratic noise with a per-asset log-normally dispersed volatility.

Prices: ``close[t] = close[t-1] * exp(r[t])``.  Open/high/low are drawn around close with
ranges scaled by each asset's idiosyncratic vol; ``vwap`` lies inside ``[low, high]``.
Volume follows a log-AR(1) with a per-asset level and a shock proportional to
``|r[t]| / idio_vol[i]``.  Book value per share follows a quarterly random walk and is
*point-in-time*: the value dated quarter-end ``q`` becomes visible on the first date
``>= q + fundamental_lag_days`` (calendar days) and is forward-filled until the next one.
Suspension (trading halt): return 0, open/high/low/vwap flat at close, volume/amount NaN,
``tradable = 0``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .store import DataStore

logger = logging.getLogger("alphaforge.data.simulator")

__all__ = ["SimConfig", "simulate_market", "TRUE_ALPHA_KEYS"]

#: Keys of ``store.meta["true_alphas"]`` in the order they enter the return generator.
TRUE_ALPHA_KEYS: tuple[str, ...] = ("reversal", "momentum", "volume", "value")

# ---- fixed structural constants (not fields of SimConfig) ----------------------------
REVERSAL_WINDOW = 5  #: past-return window for the reversal alpha
MOMENTUM_SKIP = 20  #: most recent days skipped by the momentum alpha
MOMENTUM_WINDOW = 120  #: look-back of the momentum alpha
VOLUME_WINDOW = 20  #: window of the volume-shock denominator
REGIME_PERIOD = 500.0  #: period (days) of the slow alpha-strength sine
_BETA_STD = 0.3  # per-asset market beta ~ N(1, 0.3)
_IDIO_LOG_SIGMA = 0.3  # per-asset idio vol = idio_vol * exp(N(0, 0.3))
_SHARES_LOG_MEAN = np.log(2e8)
_SHARES_LOG_SIGMA = 0.8
_PRICE0_LOG_MEAN = np.log(30.0)
_PRICE0_LOG_SIGMA = 0.7
_TURNOVER = 0.004  # typical daily volume / shares outstanding
_VOL_LEVEL_SIGMA = 0.3  # dispersion of per-asset log-volume level
_VOL_AR_PHI = 0.6  # log-volume AR(1) persistence
_VOL_AR_SIGMA = 0.35  # log-volume innovation std
_VOL_SHOCK_KAPPA = 0.25  # log-volume response per unit |r| / idio_vol
_OPEN_GAP_FRAC = 0.3  # overnight gap std as a fraction of idio vol
_RANGE_FRAC = 0.5  # intraday range std as a fraction of idio vol
_BP0_LOG_MEAN = np.log(0.6)  # initial book-to-price ~ 0.6
_BP0_LOG_SIGMA = 1.0  # cross-sectional dispersion of log(B/P) (realistic: ~0.8-1.0)
_BVPS_Q_DRIFT = 0.01  # quarterly log-drift of book value per share
_BVPS_Q_SIGMA = 0.10  # quarterly log-innovation std of book value per share
_QUARTER_FREQ = "QE-DEC"


@dataclass
class SimConfig:
    """Parameters of the synthetic market (see module docstring for the model).

    Attributes:
        n_assets: number of assets (``"S0001"`` ...).
        n_days: number of business days.
        n_industries: number of industry groups (uniform random assignment).
        start: first business day (``pd.bdate_range`` start).
        seed: numpy ``default_rng`` seed; output is fully deterministic given the seed.
        market_vol: daily std of the market factor return.
        industry_vol: daily std of each industry factor return.
        idio_vol: *median* idiosyncratic daily vol; per-asset dispersion is log-normal.
        alpha_reversal: return per unit z-score of the 5-day past return (negative).
        alpha_momentum: return per unit z-score of the t-120..t-20 return (positive).
        alpha_volume: return per unit z-score of ``log(vol / 20d mean vol)`` (negative).
        alpha_value: return per unit z-score of point-in-time book-to-price (positive).
        alpha_regime_amp: alphas are multiplied by ``1 + amp * sin(2*pi*t/500)``.
        suspension_prob: per asset-day probability of a trading halt.
        fundamental_lag_days: calendar days after quarter end before book value is visible.
    """

    n_assets: int = 300
    n_days: int = 1000  # business days
    n_industries: int = 15
    start: str = "2018-01-01"
    seed: int = 42
    # latent return structure (daily, decimal)
    market_vol: float = 0.010
    industry_vol: float = 0.006
    idio_vol: float = 0.020  # median idiosyncratic vol; per-asset dispersion applied
    # embedded true alphas (coefficient on cross-sectional z-score, in return units per day)
    alpha_reversal: float = -0.0015  # z(5-day past return)      -> negative
    alpha_momentum: float = 0.0010  # z(return t-120..t-20)     -> positive
    alpha_volume: float = -0.0010  # z(log(vol / 20d mean vol)) -> negative
    alpha_value: float = 0.0006  # z(book-to-price)          -> positive
    alpha_regime_amp: float = 0.5  # alphas scaled by (1 + amp * slow sine) so strength drifts
    suspension_prob: float = 0.004  # per asset-day prob of trading halt (volume NaN, price flat)
    fundamental_lag_days: int = 45  # point-in-time: quarterly book value visible only after lag


def _validate(cfg: SimConfig) -> None:
    if cfg.n_assets < 2:
        raise ValueError("n_assets must be >= 2 (cross-sectional z-scores need at least two names)")
    if cfg.n_days < 1:
        raise ValueError("n_days must be >= 1")
    if cfg.n_industries < 1:
        raise ValueError("n_industries must be >= 1")
    if not 0.0 <= cfg.suspension_prob < 1.0:
        raise ValueError("suspension_prob must be in [0, 1)")
    if min(cfg.market_vol, cfg.industry_vol, cfg.idio_vol) < 0:
        raise ValueError("volatilities must be non-negative")
    if cfg.idio_vol == 0:
        raise ValueError("idio_vol must be > 0 (it scales the volume shock and intraday ranges)")
    if cfg.fundamental_lag_days < 0:
        raise ValueError("fundamental_lag_days must be >= 0")


def _cs_zscore(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score of a 1-D vector (NaN-aware).

    Returns all-NaN when fewer than two finite values exist or the dispersion is zero;
    NaN inputs stay NaN.  Uses population std (``ddof=0``).
    """
    out = np.full(x.shape, np.nan)
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return out
    v = x[valid]
    sd = v.std()
    if not sd > 0.0:
        return out
    out[valid] = (v - v.mean()) / sd
    return out


def _nan_window_mean(win: np.ndarray) -> np.ndarray:
    """Column-wise NaN-skipping mean of a 2-D window; NaN where no finite value exists."""
    cnt = np.isfinite(win).sum(axis=0)
    tot = np.nansum(win, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return mean


def _quarterly_bvps(
    rng: np.random.Generator, dates: pd.DatetimeIndex, p0: np.ndarray, lag_days: int
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Quarterly book value per share and its point-in-time view on every date.

    Returns:
        ``(quarter_ends, bvps_q, bvps_pit)`` where ``bvps_q`` is ``(n_quarters, n_assets)``
        and ``bvps_pit`` is ``(n_days, n_assets)``: on date ``t`` the visible value is that of
        the latest quarter end ``q`` with ``q + lag_days <= t``.
    """
    lag = pd.Timedelta(days=lag_days)
    # Start early enough that at least one quarter end is already visible on day 0.
    q_ends = pd.date_range(dates[0] - lag - pd.Timedelta(days=200), dates[-1], freq=_QUARTER_FREQ)
    n_q, n = len(q_ends), len(p0)
    bvps0 = p0 * np.exp(rng.normal(_BP0_LOG_MEAN, _BP0_LOG_SIGMA, size=n))
    steps = rng.normal(_BVPS_Q_DRIFT, _BVPS_Q_SIGMA, size=(n_q, n))
    steps[0] = 0.0
    bvps_q = bvps0 * np.exp(np.cumsum(steps, axis=0))
    visible_from = (q_ends + lag).values
    pit_idx = np.searchsorted(visible_from, dates.values, side="right") - 1
    if pit_idx.min() < 0:  # pragma: no cover - guaranteed by the 200-day head start
        raise RuntimeError("no fundamental visible on the first simulation date")
    q_index = pd.DatetimeIndex(np.asarray(q_ends), name="quarter_end")  # freq-less (round-trips)
    return q_index, bvps_q, bvps_q[pit_idx]


def simulate_market(cfg: SimConfig) -> DataStore:
    """Generate a full synthetic market and wrap it in a DataStore.

    Fields produced (all panels): open, high, low, close, volume, amount, vwap, returns
    (close-to-close simple return, NaN on day 0), market_cap, bp (book-to-price, point-in-time),
    industry (float codes), tradable (1.0 / 0.0 — 0 on suspension days).
    Also stores ``store.meta["true_alphas"]`` = dict of the four embedded alpha panels
    (z-scored, as used in the return generator) so tests/reports can verify the miner
    rediscovers them.  Must be deterministic given cfg.seed.

    Metadata written to ``store.meta``:
        * ``"true_alphas"``: ``{"reversal", "momentum", "volume", "value"} -> panel`` of the
          cross-sectional z-scores **indexed at the date they were observable** (the value at
          ``s`` drives the return on ``s+1``), NaN where undefined.
        * ``"true_alpha_coefs"``: ``{key: coefficient}`` (sign tells the expected IC sign).
        * ``"sim_config"``: ``asdict(cfg)``.
        * ``"beta"``, ``"idio_vol"``, ``"shares"``: per-asset ``Series``.
        * ``"bvps_quarterly"``: ``DataFrame`` (index ``quarter_end``, columns assets) of the
          book value per share published for each quarter (before the visibility lag).

    The planted alphas use history only, so they are a known truth: a result much stronger
    than the planted strength shows a look-ahead bug.

    Args:
        cfg: simulation parameters.

    Returns:
        A :class:`~alphaforge.data.store.DataStore` with the fields listed above.
    """
    t_start = time.perf_counter()
    _validate(cfg)
    rng = np.random.default_rng(cfg.seed)
    T, N, K = int(cfg.n_days), int(cfg.n_assets), int(cfg.n_industries)

    # freq-less index: identical to what DataStore.load() rebuilds, so meta panels round-trip
    dates = pd.DatetimeIndex(np.asarray(pd.bdate_range(cfg.start, periods=T)), name="date")
    assets = pd.Index([f"S{i + 1:04d}" for i in range(N)], dtype=object)

    # ---------------------------------------------------------------- static asset traits
    industry = rng.integers(0, K, size=N)
    beta = rng.normal(1.0, _BETA_STD, size=N)
    idio = cfg.idio_vol * np.exp(rng.normal(0.0, _IDIO_LOG_SIGMA, size=N))
    shares = np.exp(rng.normal(_SHARES_LOG_MEAN, _SHARES_LOG_SIGMA, size=N))
    p0 = np.exp(rng.normal(_PRICE0_LOG_MEAN, _PRICE0_LOG_SIGMA, size=N))
    log_vol_level = np.log(shares * _TURNOVER) + rng.normal(0.0, _VOL_LEVEL_SIGMA, size=N)

    # ---------------------------------------------------------------- pre-drawn shocks
    f_mkt = rng.normal(0.0, cfg.market_vol, size=T)
    f_ind = rng.normal(0.0, cfg.industry_vol, size=(T, K))
    eps = rng.standard_normal((T, N)) * idio
    suspended = rng.random((T, N)) < cfg.suspension_prob
    n_open = rng.standard_normal((T, N))
    n_high = np.abs(rng.standard_normal((T, N)))
    n_low = np.abs(rng.standard_normal((T, N)))
    u_vwap = rng.random((T, N))
    n_vol = rng.standard_normal((T, N))
    q_ends, bvps_q, bvps_pit = _quarterly_bvps(rng, dates, p0, cfg.fundamental_lag_days)

    coef = {
        "reversal": float(cfg.alpha_reversal),
        "momentum": float(cfg.alpha_momentum),
        "volume": float(cfg.alpha_volume),
        "value": float(cfg.alpha_value),
    }
    regime = 1.0 + cfg.alpha_regime_amp * np.sin(2.0 * np.pi * np.arange(T) / REGIME_PERIOD)

    # ---------------------------------------------------------------- day loop
    # Each day's alpha z-scores are computed from the growing history (rows <= s) and used
    # only for the *next* day's return, so nothing computed here can see the future.
    logp = np.empty((T, N))
    logret = np.zeros((T, N))
    volume = np.full((T, N), np.nan)
    z = {k: np.full((T, N), np.nan) for k in TRUE_ALPHA_KEYS}
    lv_state = log_vol_level.copy()
    logp[0] = np.log(p0)

    for t in range(T):
        if t > 0:
            alpha = np.zeros(N)
            for k in TRUE_ALPHA_KEYS:
                alpha += coef[k] * np.nan_to_num(z[k][t - 1], nan=0.0)
            r = beta * f_mkt[t] + f_ind[t, industry] + regime[t] * alpha + eps[t]
            r[suspended[t]] = 0.0
            logret[t] = r
            logp[t] = logp[t - 1] + r

        # volume: log-AR(1) around the asset's level with an |r|/vol shock
        shock = np.abs(logret[t]) / idio
        lv_state = (
            log_vol_level
            + _VOL_AR_PHI * (lv_state - log_vol_level)
            + _VOL_SHOCK_KAPPA * shock
            + _VOL_AR_SIGMA * n_vol[t]
        )
        v = np.exp(lv_state)
        v[suspended[t]] = np.nan
        volume[t] = v

        # signals observable at the close of day t (history rows <= t only)
        if t >= REVERSAL_WINDOW:
            z["reversal"][t] = _cs_zscore(np.expm1(logp[t] - logp[t - REVERSAL_WINDOW]))
        if t >= MOMENTUM_WINDOW:
            z["momentum"][t] = _cs_zscore(np.expm1(logp[t - MOMENTUM_SKIP] - logp[t - MOMENTUM_WINDOW]))
        if t >= VOLUME_WINDOW - 1:
            mean_vol = _nan_window_mean(volume[t - VOLUME_WINDOW + 1 : t + 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                z["volume"][t] = _cs_zscore(np.log(volume[t] / mean_vol))
        z["value"][t] = _cs_zscore(np.log(bvps_pit[t]) - logp[t])  # z(log book-to-price)

    # ---------------------------------------------------------------- derived price fields
    close = np.exp(logp)
    prev_close = np.vstack([close[:1], close[:-1]])
    open_ = prev_close * np.exp(_OPEN_GAP_FRAC * idio * n_open)
    high = np.maximum(open_, close) * np.exp(_RANGE_FRAC * idio * n_high)
    low = np.minimum(open_, close) * np.exp(-_RANGE_FRAC * idio * n_low)
    vwap = np.clip(low + u_vwap * (high - low), low, high)
    for arr in (open_, high, low, vwap):  # halted names print flat at the last close
        arr[suspended] = close[suspended]
    amount = vwap * volume
    market_cap = close * shares
    returns = np.expm1(logret)
    returns[0] = np.nan
    tradable = (~suspended).astype(np.float64)
    bp = bvps_pit / close
    industry_panel = np.tile(industry.astype(np.float64), (T, 1))

    def panel(arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=dates, columns=assets)

    fields = {
        "open": panel(open_),
        "high": panel(high),
        "low": panel(low),
        "close": panel(close),
        "volume": panel(volume),
        "amount": panel(amount),
        "vwap": panel(vwap),
        "returns": panel(returns),
        "market_cap": panel(market_cap),
        "bp": panel(bp),
        "industry": panel(industry_panel),
        "tradable": panel(tradable),
    }
    meta = {
        "sim_config": asdict(cfg),
        "true_alphas": {k: panel(z[k]) for k in TRUE_ALPHA_KEYS},
        "true_alpha_coefs": coef,
        "beta": pd.Series(beta, index=assets, name="beta"),
        "idio_vol": pd.Series(idio, index=assets, name="idio_vol"),
        "shares": pd.Series(shares, index=assets, name="shares"),
        "bvps_quarterly": pd.DataFrame(bvps_q, index=q_ends, columns=assets),
    }
    store = DataStore(fields, meta)
    logger.info(
        "simulated market: %d assets x %d days, %d industries, %.3f%% halted asset-days, seed=%d (%.2fs)",
        N,
        T,
        K,
        100.0 * suspended.mean(),
        cfg.seed,
        time.perf_counter() - t_start,
    )
    return store
