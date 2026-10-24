"""Synthetic crypto market panel used for testing and factor validation.

The simulator is deliberately rich enough that every raw field has its own
dynamics: a common market factor with BTC as the benchmark, stochastic
volatility, volume that responds to absolute returns, mean-reverting funding
and basis, open interest following price and leverage, and staggered listings
(late coins have NaN history before listing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_panel(
    n_symbols: int = 30, n_bars: int = 900, seed: int = 7, freq: str = "1D"
) -> dict[str, pd.DataFrame]:
    """Return a synthetic panel set: every raw field as a frame (time x symbol) of ``n_bars`` bars.

    The symbols are BTCUSDT, ETHUSDT and ``n_symbols - 2`` coins C000USDT, ...; the bars
    start on 2022-01-01 (UTC) with the pandas frequency ``freq``. The same ``seed`` gives
    the same values.
    """
    rng = np.random.default_rng(seed)
    symbols = ["BTCUSDT", "ETHUSDT"] + [f"C{i:03d}USDT" for i in range(n_symbols - 2)]
    index = pd.date_range("2022-01-01", periods=n_bars, freq=freq, tz="UTC")
    T, N = n_bars, n_symbols

    # market factor with GARCH-like volatility
    mkt_vol = np.empty(T)
    mkt = np.empty(T)
    v = 0.03
    for t in range(T):
        v = np.sqrt(0.00002 + 0.08 * (mkt[t - 1] if t else 0) ** 2 + 0.9 * v**2)
        mkt_vol[t] = v
        mkt[t] = v * rng.standard_t(5) / np.sqrt(5 / 3)

    beta = np.concatenate([[1.0, 1.1], rng.uniform(0.6, 1.8, N - 2)])
    idio_scale = np.concatenate([[0.004, 0.01], rng.uniform(0.01, 0.05, N - 2)])
    # idiosyncratic stochastic vol
    log_iv = np.zeros((T, N))
    for t in range(1, T):
        log_iv[t] = 0.95 * log_iv[t - 1] + 0.25 * rng.standard_normal(N)
    idio = idio_scale * np.exp(0.5 * log_iv) * rng.standard_normal((T, N))
    # small persistent drift + weak short-term reversal so factors are not pure noise
    drift = rng.normal(0, 0.0008, N)
    ret = beta * mkt[:, None] + idio + drift
    ret[1:] -= 0.05 * idio[:-1]
    ret[:, 0] = mkt  # BTC is the market
    ret = np.clip(ret, -0.6, 1.5)

    close = 100 * np.exp(rng.uniform(-3, 6, N)) * np.cumprod(1 + ret, axis=0)
    prev = np.vstack([close[0] / (1 + ret[0]), close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.0015, (T, N)))
    bar_vol = np.abs(ret) + idio_scale * np.exp(0.5 * log_iv) * 0.8
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 1, (T, N))) * bar_vol * 0.6)
    low = np.minimum(open_, close) * (1 - np.clip(np.abs(rng.normal(0, 1, (T, N))) * bar_vol * 0.6, 0, 0.9))

    base_dvol = np.exp(rng.uniform(13, 21, N))
    log_turn = np.zeros((T, N))
    for t in range(1, T):
        log_turn[t] = 0.9 * log_turn[t - 1] + 0.2 * rng.standard_normal(N)
    quote_volume = base_dvol * np.exp(log_turn + 8 * np.abs(ret) + 0.3 * rng.standard_normal((T, N)))
    vwap = (open_ + high + low + close) / 4 * (1 + rng.normal(0, 0.002, (T, N)))
    vwap = np.clip(vwap, low, high)
    volume = quote_volume / vwap
    avg_size = np.exp(rng.uniform(4, 9, N)) * np.exp(0.2 * rng.standard_normal((T, N)))
    trades = np.maximum(np.round(quote_volume / avg_size), 1)
    taker_share = np.clip(0.5 + 2.0 * ret + rng.normal(0, 0.04, (T, N)), 0.05, 0.95)
    taker_buy_volume = volume * taker_share
    taker_buy_quote_volume = quote_volume * taker_share * (1 + rng.normal(0, 0.001, (T, N)))

    # derivatives
    funding = np.zeros((T, N))
    premium = np.zeros((T, N))
    oi_coin = np.zeros((T, N))
    oi_coin[0] = quote_volume[0] / close[0] * rng.uniform(0.3, 2, N)
    ls = np.zeros((T, N))
    top = np.zeros((T, N))
    for t in range(1, T):
        mom = np.log(close[t] / close[max(t - 7, 0)])
        funding[t] = 0.85 * funding[t - 1] + 0.15 * (0.0001 + 0.002 * mom) + rng.normal(0, 0.00008, N)
        premium[t] = 0.7 * premium[t - 1] + 0.3 * funding[t] * 0.8 + rng.normal(0, 0.0004, N)
        oi_coin[t] = oi_coin[t - 1] * np.exp(
            0.3 * ret[t] + rng.normal(0, 0.03, N) - 0.002 * np.log(oi_coin[t - 1] * close[t] / base_dvol)
        )
        ls[t] = 0.9 * ls[t - 1] - 0.5 * ret[t] + rng.normal(0, 0.05, N)
        top[t] = 0.92 * top[t - 1] + 0.3 * ret[t] + rng.normal(0, 0.05, N)
    taker_bs = np.exp(np.log(taker_share / (1 - taker_share)) * 0.8 + rng.normal(0, 0.05, (T, N)))

    raw = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": quote_volume,
        "trades": trades,
        "taker_buy_volume": taker_buy_volume,
        "taker_buy_quote_volume": taker_buy_quote_volume,
        "funding_rate": funding * 3,  # daily sum of three 8h fundings
        "open_interest": oi_coin,
        "open_interest_value": oi_coin * close,
        "long_short_ratio": np.exp(ls + 0.3),
        "top_long_short_ratio": np.exp(top + 0.1),
        "taker_buy_sell_ratio": taker_bs,
        "premium": premium,
    }

    # staggered listings: a third of the alt coins start later
    listing = np.zeros(N, dtype=int)
    late = rng.choice(np.arange(2, N), size=max((N - 2) // 3, 0), replace=False)
    listing[late] = rng.integers(T // 10, T // 2, len(late))
    mask = np.arange(T)[:, None] < listing[None, :]

    panels = {}
    for k, arr in raw.items():
        arr = arr.astype(float).copy()
        arr[mask] = np.nan
        panels[k] = pd.DataFrame(arr, index=index, columns=symbols)
    return panels
