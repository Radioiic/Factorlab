"""Download the raw panels from Binance public endpoints (no API key needed).

* klines (spot or USD-M perpetual)   -> open/high/low/close/volume/...
* USD-M funding rate history         -> funding_rate (summed per bar)
* USD-M premium index klines         -> premium
* USD-M futures "data" endpoints     -> open interest, long/short ratios,
  taker buy/sell ratio.  NOTE: Binance only serves the last ~30 days on
  these REST endpoints; for full history use ``download_metrics_archive``
  which reads the daily ``metrics`` files on https://data.binance.vision.

Every request first checks ``factorlab.network.require_network()``: the library
is offline unless ``FACTORLAB_OFFLINE=0``, the same switch as Factorlab's engine.

Each data.binance.vision zip file is compared with the SHA-256 in its ``.CHECKSUM``
file before it is parsed. A response body larger than its size limit is an error.

Usage::

    from factorlab.data.binance import download_panels
    panels = download_panels(["BTCUSDT", "ETHUSDT"], start="2021-01-01", interval="1d")
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime, timedelta

import pandas as pd

from ..network import require_network

SPOT = "https://api.binance.com"
FAPI = "https://fapi.binance.com"
VISION = "https://data.binance.vision"

KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}
_PANDAS_FREQ = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
}

# Size limits in bytes. A REST response of 1,000 rows and a daily metrics file are much smaller.
MAX_JSON_BYTES = 16 * 2**20  # one REST response
MAX_ARCHIVE_BYTES = 50 * 2**20  # one data.binance.vision zip file
MAX_CHECKSUM_BYTES = 4 * 2**10  # one .CHECKSUM file
MAX_CSV_BYTES = 200 * 2**20  # the CSV file in a metrics zip, after decompression


class ResponseTooLarge(RuntimeError):
    """A response body is larger than its size limit."""


class ArchiveError(ValueError):
    """A data.binance.vision file is not correct: it has no checksum file, its checksum does not
    agree, or its content is not a daily metrics file."""


@functools.cache
def _ssl_context() -> ssl.SSLContext | None:
    """The SSL context of the downloads: the CA certificates of this Python (also SSL_CERT_FILE
    and SSL_CERT_DIR) and the CA certificates of certifi. Without certifi, None (the default
    context of urllib). setup.sh installs certifi.

    A Python from python.org on macOS has no CA certificates of its own until its "Install
    Certificates.command" runs. With that Python and without certifi, each https request fails
    with CERTIFICATE_VERIFY_FAILED."""
    try:
        import certifi
    except ImportError:
        return None
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(cafile=certifi.where())
    return ctx


def _urlopen(url: str):
    """urllib.request.urlopen for an https URL only (a file: or other scheme is an error)."""
    if not url.startswith("https://"):
        raise ValueError(f"only https URLs are allowed: {url!r}")
    return urllib.request.urlopen(url, timeout=30, context=_ssl_context())  # nosec B310  # the check above allows https URLs only


def _read_limited(r, limit: int, url: str) -> bytes:
    """The body of the response ``r``. A body larger than ``limit`` bytes is an error."""
    body = r.read(limit + 1)
    if len(body) > limit:
        raise ResponseTooLarge(f"the response from {url} is larger than {limit} bytes")
    return body


def _ms(ts) -> int:
    """Return ``ts`` as Unix time in milliseconds (an int stays as it is; a text or timestamp is read as UTC)."""
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000) if not isinstance(ts, int) else ts


def _retry_after(e: urllib.error.HTTPError) -> float | None:
    """The Retry-After header of ``e`` in seconds, or None."""
    value = e.headers.get("Retry-After") if e.headers else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _retry_wait(e: urllib.error.HTTPError, attempt: int) -> float:
    """The wait before the next try after HTTP 429 or 5xx: the larger of 2**attempt seconds
    and Retry-After. HTTP 418 means that Binance has banned the IP address for a time, and a
    retry makes the ban longer: it is an error at once (Binance REST API rules)."""
    after = _retry_after(e)
    if e.code == 418:
        raise RuntimeError(
            "Binance banned this IP address after too many requests (HTTP 418"
            + (f", Retry-After {after:.0f} s" if after is not None else "")
            + "): wait, then run again with fewer parallel downloads (--jobs)"
        ) from e
    return max(2**attempt, after or 0)


def _get(url: str, params: dict, retries: int = 5):
    """Return the decoded JSON of a GET request to ``url`` with ``params``, with retries (see _retry_wait)."""
    q = url + "?" + urllib.parse.urlencode(params)
    require_network("a Binance download")  # offline unless FACTORLAB_OFFLINE=0
    last: urllib.error.URLError | None = None
    for attempt in range(retries):
        try:
            with _urlopen(q) as r:
                return json.loads(_read_limited(r, MAX_JSON_BYTES, q))
        except urllib.error.HTTPError as e:
            if e.code in (418, 429) or e.code >= 500:
                last = e
                time.sleep(_retry_wait(e, attempt))
                continue
            raise
        except urllib.error.URLError as e:
            last = e
            time.sleep(2**attempt)
    # the reason of the last error (for example CERTIFICATE_VERIFY_FAILED) goes into the message
    raise RuntimeError(f"failed to fetch {q}" + (f": {last.reason}" if last is not None else "")) from last


def _get_bytes(url: str, retries: int = 6, limit: int = MAX_ARCHIVE_BYTES) -> bytes | None:
    """The body of ``url``, or None when the file does not exist (HTTP 404).

    A rate limit (HTTP 429), a server error (HTTP 5xx), a network error or a timeout causes a
    retry after 1, 2, 4, 8 and 16 seconds (on HTTP 429, at least Retry-After). After the last
    retry the error is raised. An IP ban (HTTP 418) is an error at once (see _retry_wait).
    A body larger than ``limit`` bytes is an error at once (ResponseTooLarge)."""
    require_network("a Binance archive download")  # offline unless FACTORLAB_OFFLINE=0
    for attempt in range(retries):
        try:
            with _urlopen(url) as r:
                return _read_limited(r, limit, url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 418 or ((e.code == 429 or e.code >= 500) and attempt < retries - 1):
                time.sleep(_retry_wait(e, attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            raise
    return None


def checksum_sha256(text: bytes, file_name: str) -> str:
    """The SHA-256 (64 lowercase hex digits) in a data.binance.vision .CHECKSUM file.

    The file has one line: the hex digest, white space and the zip file name. A file in
    another form, or for another file name, gives ArchiveError."""
    try:
        parts = text.decode("ascii").split()
    except UnicodeDecodeError:
        parts = []
    digest = parts[0].lower() if parts else ""
    if (
        len(parts) not in (1, 2)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or (len(parts) == 2 and parts[1].lstrip("*") != file_name)
    ):
        raise ArchiveError(f"the checksum file of {file_name} is not in the form '<sha256>  {file_name}'")
    return digest


def _get_archive(url: str) -> bytes | None:
    """The data.binance.vision file at ``url`` after a SHA-256 check, or None when the file
    does not exist (HTTP 404).

    The digest comes from ``url + ".CHECKSUM"``. When the file exists but its checksum file
    does not, or when the digests do not agree, the result is ArchiveError."""
    blob = _get_bytes(url, limit=MAX_ARCHIVE_BYTES)
    if blob is None:
        return None
    name = url.rsplit("/", 1)[-1]
    text = _get_bytes(url + ".CHECKSUM", limit=MAX_CHECKSUM_BYTES)
    if text is None:
        raise ArchiveError(f"{url} has no checksum file ({name}.CHECKSUM): the file cannot be verified")
    want = checksum_sha256(text, name)
    got = hashlib.sha256(blob).hexdigest()
    if got != want:
        raise ArchiveError(f"the SHA-256 of {name} is {got}, but its checksum file gives {want}: download it again")
    return blob


def fetch_klines(symbol: str, interval: str = "1d", start=None, end=None, market: str = "spot") -> pd.DataFrame:
    """Download the klines of ``symbol`` from ``start`` to ``end`` (UTC) and return them as a frame (see parse_klines).

    ``market`` is "spot" or "um" (USD-M perpetuals). The default start is 2017-01-01 and the default end is now.
    """
    base, path = (SPOT, "/api/v3/klines") if market == "spot" else (FAPI, "/fapi/v1/klines")
    start_ms = _ms(start or "2017-01-01")
    end_ms = _ms(end) if end else int(time.time() * 1000)
    rows = []
    while start_ms < end_ms:
        batch = _get(
            base + path,
            {"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": 1000},
        )
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + INTERVAL_MS[interval]
        if len(batch) < 1000:
            break
    return parse_klines(rows)


def parse_klines(rows: list) -> pd.DataFrame:
    """Return Binance kline rows as a frame of floats indexed by the UTC open time (no repeated bars)."""
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    if df.empty:
        return df
    df.index = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df = df.drop(columns=["open_time", "close_time", "ignore"]).astype(float)
    return df[~df.index.duplicated()]


def fetch_funding(symbol: str, start=None, end=None) -> pd.Series:
    """Download the funding settlements of the USD-M perpetual ``symbol`` and return them (see parse_funding)."""
    start_ms = _ms(start or "2019-09-01")
    end_ms = _ms(end) if end else int(time.time() * 1000)
    rows = []
    while start_ms < end_ms:
        batch = _get(
            FAPI + "/fapi/v1/fundingRate", {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
        )
        if not batch:
            break
        rows.extend(batch)
        start_ms = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
    return parse_funding(rows)


def parse_funding(rows: list) -> pd.Series:
    """Return funding rows as the series ``funding_rate`` indexed by the UTC settlement time (no repeated times)."""
    if not rows:
        return pd.Series(dtype=float, name="funding_rate")
    s = pd.Series(
        [float(r["fundingRate"]) for r in rows],
        index=pd.to_datetime([int(r["fundingTime"]) for r in rows], unit="ms", utc=True),
    )
    return s[~s.index.duplicated()].rename("funding_rate")


def fetch_premium(symbol: str, interval: str = "1d", start=None, end=None) -> pd.Series:
    """Download the premium-index klines of ``symbol`` and return the close of each bar as the series ``premium``."""
    start_ms = _ms(start or "2019-09-01")
    end_ms = _ms(end) if end else int(time.time() * 1000)
    rows = []
    while start_ms < end_ms:
        batch = _get(
            FAPI + "/fapi/v1/premiumIndexKlines",
            {"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": 1000},
        )
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + INTERVAL_MS[interval]
        if len(batch) < 1000:
            break
    if not rows:
        return pd.Series(dtype=float, name="premium")
    s = pd.Series([float(r[4]) for r in rows], index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True))
    return s[~s.index.duplicated()].rename("premium")


_FUT_DATA = {
    # endpoint -> {json key: field}
    "openInterestHist": {"sumOpenInterest": "open_interest", "sumOpenInterestValue": "open_interest_value"},
    "globalLongShortAccountRatio": {"longShortRatio": "long_short_ratio"},
    "topLongShortPositionRatio": {"longShortRatio": "top_long_short_ratio"},
    "takerlongshortRatio": {"buySellRatio": "taker_buy_sell_ratio"},
}


def fetch_futures_data(symbol: str, period: str = "1d") -> pd.DataFrame:
    """Recent (~30 days) OI / long-short / taker ratios from /futures/data."""
    frames = []
    for endpoint, mapping in _FUT_DATA.items():
        rows = _get(FAPI + f"/futures/data/{endpoint}", {"symbol": symbol, "period": period, "limit": 500})
        if not rows:
            continue
        idx = pd.to_datetime([int(r["timestamp"]) for r in rows], unit="ms", utc=True)
        frames.append(pd.DataFrame({f: [float(r[k]) for r in rows] for k, f in mapping.items()}, index=idx))
    return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()


def download_metrics_archive(symbol: str, start, end=None, interval: str = "1d") -> pd.DataFrame:
    """Full-history OI and ratio metrics from data.binance.vision daily files.

    Each daily zip holds 5-minute snapshots; we keep the last snapshot of each
    bar (these are state variables, not flows). A day without a zip file is
    skipped. Each zip file must agree with its .CHECKSUM file (_get_archive).
    """
    day = pd.Timestamp(start, tz="UTC").normalize()
    last = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp(datetime.now(UTC) - timedelta(days=1))
    frames = []
    require_network("a Binance archive download")  # offline unless FACTORLAB_OFFLINE=0
    while day <= last:
        url = f"{VISION}/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day:%Y-%m-%d}.zip"
        blob = _get_archive(url)
        if blob is not None:
            frames.append(parse_metrics_zip(blob))
        day += pd.Timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    return df.resample(_PANDAS_FREQ[interval], label="left", closed="left").last()


def parse_metrics_zip(blob: bytes) -> pd.DataFrame:
    """The 5-minute snapshots of one daily metrics zip file (the first file in the zip).

    Content that is not a daily metrics file gives ArchiveError. The CSV file is read
    in memory (no extraction), and at most MAX_CSV_BYTES of it."""
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names = z.namelist()
            if not names:
                raise ArchiveError("the metrics zip file contains no file")
            if z.getinfo(names[0]).file_size > MAX_CSV_BYTES:
                raise ArchiveError(f"the file in the metrics zip is larger than {MAX_CSV_BYTES} bytes")
            with z.open(names[0]) as f:
                data = f.read(MAX_CSV_BYTES + 1)
        if len(data) > MAX_CSV_BYTES:
            raise ArchiveError(f"the file in the metrics zip is larger than {MAX_CSV_BYTES} bytes")
        raw = pd.read_csv(io.BytesIO(data))
        idx = pd.to_datetime(raw["create_time"], utc=True)
        return pd.DataFrame(
            {
                "open_interest": raw["sum_open_interest"].astype(float).to_numpy(),
                "open_interest_value": raw["sum_open_interest_value"].astype(float).to_numpy(),
                "top_long_short_ratio": raw["sum_toptrader_long_short_ratio"].astype(float).to_numpy(),
                "long_short_ratio": raw["count_long_short_ratio"].astype(float).to_numpy(),
                "taker_buy_sell_ratio": raw["sum_taker_long_short_vol_ratio"].astype(float).to_numpy(),
            },
            index=idx,
        )
    except ArchiveError:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure means that the content is not a metrics file
        raise ArchiveError(f"not a daily metrics zip file: {type(exc).__name__}: {exc}") from exc


def to_bar(series: pd.Series | pd.DataFrame, interval: str, how: str) -> pd.Series | pd.DataFrame:
    """Return ``series`` resampled to bars of ``interval`` (UTC open time): the sum (``how`` "sum") or the last value."""
    r = series.resample(_PANDAS_FREQ[interval], label="left", closed="left")
    return r.sum(min_count=1) if how == "sum" else r.last()


def download_panels(
    symbols: list[str],
    start="2020-01-01",
    end=None,
    interval: str = "1d",
    market: str = "um",
    derivatives: bool = True,
    metrics_archive: bool = False,
    log=print,
) -> dict[str, pd.DataFrame]:
    """Download every raw field for ``symbols`` and return panels (time x symbol)."""
    per_field: dict[str, dict[str, pd.Series]] = {}

    def put(field, sym, s):
        per_field.setdefault(field, {})[sym] = s

    for sym in symbols:
        if log:
            log(f"downloading {sym}")
        k = fetch_klines(sym, interval, start, end, market=market)
        for col in k.columns:
            put(col, sym, k[col])
        if not derivatives:
            continue
        try:
            put("funding_rate", sym, to_bar(fetch_funding(sym, start, end), interval, "sum"))
            put("premium", sym, fetch_premium(sym, interval, start, end))
            extra = (
                download_metrics_archive(sym, start, end, interval)
                if metrics_archive
                else to_bar(fetch_futures_data(sym, interval), interval, "last")
            )
            for col in extra.columns:
                put(col, sym, extra[col])
        except Exception as exc:  # noqa: BLE001 - the message names the exception type
            # A spot coin can have no perpetual data. For a USD-M perpetual a failure is an
            # error: a panel without the derivatives fields is not complete.
            if market == "um":
                raise RuntimeError(f"derivatives download failed for {sym}: {type(exc).__name__}: {exc}") from exc
            if log:
                log(f"  no derivatives data for {sym}: {type(exc).__name__}: {exc}")
    panels = {f: pd.DataFrame(d).sort_index() for f, d in per_field.items()}
    index = panels["close"].index
    return {f: p.reindex(index=index, columns=symbols) for f, p in panels.items()}
