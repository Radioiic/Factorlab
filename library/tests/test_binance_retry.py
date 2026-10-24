"""A temporary server error must not leave a coin without its derivatives fields.

urlopen is replaced by a stub and time.sleep by a no-op, so these tests never reach the
network and never wait.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
import zipfile

import pandas as pd
import pytest

from factorlab import network
from factorlab.data import binance


def _metrics_zip(day: str) -> bytes:
    csv = (
        "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
        "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        f"{day} 23:55:00,BTCUSDT,1,2,1,1.1,1.2,0.9\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("m.csv", csv)
    return buf.getvalue()


class _Resp:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        return self.body if n is None or n < 0 else self.body[:n]


class _Stub:
    """Gives the listed outcomes in order: an int is an HTTP error code, a tuple is an HTTP
    error code and its headers, bytes is a body."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, url, timeout=None):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, int):
            out = (out, {})
        if isinstance(out, tuple):
            raise urllib.error.HTTPError(url, out[0], "error", out[1], None)
        return _Resp(out)


@pytest.fixture(autouse=True)
def _online_no_sleep(monkeypatch):
    monkeypatch.setenv(network.ENV, "0")
    monkeypatch.setattr(binance.time, "sleep", lambda s: None)


def test_server_error_is_retried(monkeypatch):
    stub = _Stub([503, 503, b"body"])
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    assert binance._get_bytes("https://example.invalid/f.zip") == b"body"
    assert stub.calls == 3


def test_missing_file_gives_none(monkeypatch):
    stub = _Stub([404])
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    assert binance._get_bytes("https://example.invalid/f.zip") is None
    assert stub.calls == 1


def test_error_after_the_last_retry_is_raised(monkeypatch):
    stub = _Stub([503] * 6)
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    with pytest.raises(urllib.error.HTTPError):
        binance._get_bytes("https://example.invalid/f.zip")
    assert stub.calls == 6


def test_ip_ban_is_not_retried(monkeypatch):
    """HTTP 418 is an IP ban: a retry makes it longer, so both getters stop at once."""
    stub = _Stub([(418, {"Retry-After": "120"})] * 2)
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    with pytest.raises(RuntimeError, match=r"HTTP 418, Retry-After 120 s"):
        binance._get_bytes("https://example.invalid/f.zip")
    assert stub.calls == 1
    with pytest.raises(RuntimeError, match=r"HTTP 418"):
        binance._get("https://example.invalid/api", {})
    assert stub.calls == 2


def test_rate_limit_waits_at_least_retry_after(monkeypatch):
    waits = []
    monkeypatch.setattr(binance.time, "sleep", waits.append)
    stub = _Stub([(429, {"Retry-After": "7"}), 429, b"body"])
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    assert binance._get_bytes("https://example.invalid/f.zip") == b"body"
    assert waits == [7, 2]  # max(2**0, 7), then max(2**1, no header)


def test_archive_survives_a_temporary_error(monkeypatch):
    blob = _metrics_zip("2024-01-01")
    checksum = f"{hashlib.sha256(blob).hexdigest()}  BTCUSDT-metrics-2024-01-01.zip\n".encode()
    stub = _Stub([503, blob, checksum])  # the zip after one retry, then its .CHECKSUM file
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    df = binance.download_metrics_archive("BTCUSDT", "2024-01-01", "2024-01-01")
    assert list(df.index) == [pd.Timestamp("2024-01-01", tz="UTC")]
    assert df["open_interest"].iloc[0] == 1.0


def test_perpetual_without_derivatives_is_an_error(monkeypatch):
    kl = pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp("2024-01-01", tz="UTC")])
    monkeypatch.setattr(binance, "fetch_klines", lambda *a, **k: kl)

    def fail(*a, **k):
        raise urllib.error.HTTPError("u", 503, "error", {}, None)

    monkeypatch.setattr(binance, "fetch_funding", fail)
    with pytest.raises(RuntimeError, match="derivatives download failed for BTCUSDT"):
        binance.download_panels(["BTCUSDT"], start="2024-01-01", end="2024-01-01", market="um", log=None)
    panels = binance.download_panels(["BTCUSDT"], start="2024-01-01", end="2024-01-01", market="spot", log=None)
    assert "close" in panels and "funding_rate" not in panels
