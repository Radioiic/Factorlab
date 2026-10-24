"""Catalog of raw market-data fields and derived (intermediate) series.

Raw fields are what a data loader must provide as panels (time x symbol).
Derived series are named sub-expressions built from raw fields; factor
expressions may reference them by name exactly like raw fields.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawField:
    """One raw field that a data loader supplies as a panel: its name, group, data source and description."""

    name: str
    group: str  # "kline" | "derivatives"
    source: str
    description: str


@dataclass(frozen=True)
class DerivedField:
    """One named sub-expression of raw fields. ``stationary`` is False for price and volume levels."""

    name: str
    expression: str
    group: str
    description: str
    stationary: bool = True  # False for price/volume *levels*


RAW_FIELDS: list[RawField] = [
    RawField("open", "kline", "Binance spot/USD-M klines", "Bar open price"),
    RawField("high", "kline", "Binance spot/USD-M klines", "Bar high price"),
    RawField("low", "kline", "Binance spot/USD-M klines", "Bar low price"),
    RawField("close", "kline", "Binance spot/USD-M klines", "Bar close price"),
    RawField("volume", "kline", "Binance spot/USD-M klines", "Base-asset volume"),
    RawField("quote_volume", "kline", "Binance spot/USD-M klines", "Quote-asset (USDT) volume"),
    RawField("trades", "kline", "Binance spot/USD-M klines", "Number of trades"),
    RawField("taker_buy_volume", "kline", "Binance spot/USD-M klines", "Taker buy base-asset volume"),
    RawField("taker_buy_quote_volume", "kline", "Binance spot/USD-M klines", "Taker buy quote-asset volume"),
    RawField(
        "funding_rate",
        "derivatives",
        "Binance USD-M /fapi/v1/fundingRate",
        "Perpetual funding rate (per 8h, summed per bar)",
    ),
    RawField(
        "open_interest",
        "derivatives",
        "Binance USD-M metrics / openInterestHist",
        "Open interest in contracts (base asset)",
    ),
    RawField(
        "open_interest_value",
        "derivatives",
        "Binance USD-M metrics / openInterestHist",
        "Open interest notional in USDT",
    ),
    RawField(
        "long_short_ratio",
        "derivatives",
        "Binance USD-M globalLongShortAccountRatio",
        "Global long/short account ratio",
    ),
    RawField(
        "top_long_short_ratio",
        "derivatives",
        "Binance USD-M topLongShortPositionRatio",
        "Top-trader long/short position ratio",
    ),
    RawField(
        "taker_buy_sell_ratio",
        "derivatives",
        "Binance USD-M takerlongshortRatio",
        "Futures taker buy/sell volume ratio",
    ),
    RawField(
        "premium", "derivatives", "Binance USD-M premiumIndexKlines", "Perp premium index (perp vs index, close of bar)"
    ),
]

RAW_FIELD_NAMES = {f.name for f in RAW_FIELDS}

_LN2 = 0.6931471805599453

DERIVED_FIELDS: list[DerivedField] = [
    # ---- returns -------------------------------------------------------
    DerivedField("ret", "close / delay(close, 1) - 1", "return", "Simple close-to-close return"),
    DerivedField("logret", "log(close / delay(close, 1))", "return", "Log close-to-close return"),
    DerivedField("oc_ret", "close / open - 1", "return", "Intrabar open-to-close return"),
    DerivedField("gap", "open / delay(close, 1) - 1", "return", "Open vs previous close"),
    DerivedField("mkt_ret", "cs_mean(ret)", "return", "Equal-weight market return"),
    DerivedField("mkt_ret_vw", "cs_wmean(ret, delay(quote_volume, 1))", "return", "Liquidity-weighted market return"),
    DerivedField("btc_ret", "bench(ret)", "return", "Benchmark (BTC) return"),
    DerivedField("ex_ret", "ret - mkt_ret", "return", "Return in excess of equal-weight market"),
    DerivedField("ex_btc_ret", "ret - btc_ret", "return", "Return in excess of BTC"),
    DerivedField("abs_ret", "abs(ret)", "return", "Absolute return"),
    DerivedField("sq_ret", "ret * ret", "return", "Squared return"),
    DerivedField("up_ret", "emax(ret, 0)", "return", "Positive part of return"),
    DerivedField("down_ret", "emin(ret, 0)", "return", "Negative part of return"),
    # ---- bar shape -----------------------------------------------------
    DerivedField("hl_range", "(high - low) / close", "shape", "High-low range relative to close"),
    DerivedField("log_hl", "log(high / low)", "shape", "Log high/low range"),
    DerivedField("body", "div(abs(close - open), high - low)", "shape", "Candle body as fraction of range"),
    DerivedField("upper_shadow", "(high - emax(open, close)) / close", "shape", "Upper wick relative to close"),
    DerivedField("lower_shadow", "(emin(open, close) - low) / close", "shape", "Lower wick relative to close"),
    DerivedField("close_loc", "div(close - low, high - low)", "shape", "Close location in bar range, 0..1"),
    DerivedField("clv", "div((close - low) - (high - close), high - low)", "shape", "Close location value, -1..1"),
    DerivedField(
        "tr_pct",
        "(emax(high, delay(close, 1)) - emin(low, delay(close, 1))) / delay(close, 1)",
        "shape",
        "True range relative to previous close",
    ),
    # ---- volatility estimators (per bar) --------------------------------
    DerivedField(
        "parkinson_var", f"log(high / low) * log(high / low) / {4 * _LN2:.10f}", "volatility", "Parkinson variance"
    ),
    DerivedField(
        "gk_var",
        f"0.5 * square(log(high / low)) - {2 * _LN2 - 1:.10f} * square(log(close / open))",
        "volatility",
        "Garman-Klass variance",
    ),
    DerivedField(
        "rs_var",
        "log(high / close) * log(high / open) + log(low / close) * log(low / open)",
        "volatility",
        "Rogers-Satchell variance",
    ),
    # ---- price levels ---------------------------------------------------
    DerivedField("vwap", "div(quote_volume, volume)", "price", "Bar VWAP", stationary=False),
    DerivedField("typical", "(high + low + close) / 3", "price", "Typical price", stationary=False),
    DerivedField("vwap_dev", "div(close, vwap) - 1", "price", "Close vs bar VWAP"),
    # ---- volume / liquidity --------------------------------------------
    DerivedField("log_dvol", "log(quote_volume)", "liquidity", "Log dollar volume", stationary=False),
    DerivedField("log_trades", "log(trades)", "liquidity", "Log trade count", stationary=False),
    DerivedField(
        "trade_size", "div(quote_volume, trades)", "liquidity", "Average trade size in USDT", stationary=False
    ),
    DerivedField(
        "dvol_chg", "log(div(quote_volume, delay(quote_volume, 1)))", "liquidity", "Log change of dollar volume"
    ),
    DerivedField("amihud", "div(abs(ret), quote_volume) * 1e9", "liquidity", "Amihud illiquidity (per bar)"),
    DerivedField("signed_dvol", "slog(sign(ret) * quote_volume)", "liquidity", "Signed log dollar volume"),
    # ---- order flow -------------------------------------------------------
    DerivedField("taker_ratio", "div(taker_buy_volume, volume)", "flow", "Taker-buy share of volume"),
    DerivedField("taker_imb", "2 * div(taker_buy_volume, volume) - 1", "flow", "Taker buy/sell imbalance, -1..1"),
    DerivedField(
        "net_taker_dvol", "slog(2 * taker_buy_quote_volume - quote_volume)", "flow", "Signed log net taker dollar flow"
    ),
    # ---- derivatives -----------------------------------------------------
    DerivedField("funding", "funding_rate", "derivatives", "Funding rate"),
    DerivedField("log_oi", "log(open_interest_value)", "derivatives", "Log open-interest notional", stationary=False),
    DerivedField(
        "oi_chg",
        "log(div(open_interest_value, delay(open_interest_value, 1)))",
        "derivatives",
        "Log change of OI notional",
    ),
    DerivedField(
        "oi_coin_chg",
        "log(div(open_interest, delay(open_interest, 1)))",
        "derivatives",
        "Log change of OI in contracts",
    ),
    DerivedField("oi_turnover", "log(div(quote_volume, open_interest_value))", "derivatives", "Log volume / OI"),
    DerivedField("ls_ratio", "log(long_short_ratio)", "derivatives", "Log global long/short account ratio"),
    DerivedField(
        "top_ls_ratio", "log(top_long_short_ratio)", "derivatives", "Log top-trader long/short position ratio"
    ),
    DerivedField(
        "ls_divergence",
        "log(top_long_short_ratio) - log(long_short_ratio)",
        "derivatives",
        "Top traders vs crowd positioning",
    ),
    DerivedField("taker_bs", "log(taker_buy_sell_ratio)", "derivatives", "Log futures taker buy/sell ratio"),
    DerivedField("basis", "premium", "derivatives", "Perp premium / basis"),
    DerivedField("funding_premium_gap", "funding_rate - premium", "derivatives", "Funding minus premium"),
]

DERIVED_BY_NAME = {d.name: d for d in DERIVED_FIELDS}
