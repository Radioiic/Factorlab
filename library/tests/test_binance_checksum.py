"""data.binance.vision downloads: the SHA-256 check, the size limits and the zip parser.

urlopen is replaced by a stub that serves a dict of URLs, and time.sleep by a no-op, so
these tests never reach the network and never wait.
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

DAY = "2024-01-01"
ZIP_URL = f"{binance.VISION}/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{DAY}.zip"
ZIP_NAME = ZIP_URL.rsplit("/", 1)[-1]
HEADER = (
    "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
    "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
)


def _zip(csv: str, name: str = "m.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, csv)
    return buf.getvalue()


METRICS = _zip(HEADER + f"{DAY} 00:05:00,BTCUSDT,1,2,1,1.1,1.2,0.9\n{DAY} 23:55:00,BTCUSDT,3,4,1,1.3,1.4,0.8\n")


def _checksum(blob: bytes, name: str = ZIP_NAME) -> bytes:
    return f"{hashlib.sha256(blob).hexdigest()}  {name}\n".encode()


class _Resp:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int = -1) -> bytes:
        return self.body if n is None or n < 0 else self.body[:n]


class _Server:
    """Serves bytes by URL; a URL that is not in the dict gives HTTP 404."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.calls: list[str] = []

    def __call__(self, url, timeout=None):
        self.calls.append(url)
        if url not in self.files:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        return _Resp(self.files[url])


@pytest.fixture(autouse=True)
def _online_no_sleep(monkeypatch):
    monkeypatch.setenv(network.ENV, "0")
    monkeypatch.setattr(binance.time, "sleep", lambda s: None)


def _serve(monkeypatch, files):
    server = _Server(files)
    monkeypatch.setattr(urllib.request, "urlopen", server)
    return server


def test_archive_with_a_correct_checksum_is_parsed(monkeypatch):
    server = _serve(monkeypatch, {ZIP_URL: METRICS, ZIP_URL + ".CHECKSUM": _checksum(METRICS)})
    df = binance.download_metrics_archive("BTCUSDT", DAY, DAY)
    assert server.calls == [ZIP_URL, ZIP_URL + ".CHECKSUM"]
    assert list(df.index) == [pd.Timestamp(DAY, tz="UTC")]
    assert df.loc[pd.Timestamp(DAY, tz="UTC"), "open_interest"] == 3.0  # the last snapshot of the bar


def test_wrong_checksum_is_an_error(monkeypatch):
    other = _checksum(METRICS + b"x")
    _serve(monkeypatch, {ZIP_URL: METRICS, ZIP_URL + ".CHECKSUM": other})
    with pytest.raises(binance.ArchiveError, match="SHA-256"):
        binance.download_metrics_archive("BTCUSDT", DAY, DAY)


def test_corrupt_download_is_an_error(monkeypatch):
    bad = bytearray(METRICS)
    bad[len(bad) // 2] ^= 0xFF
    _serve(monkeypatch, {ZIP_URL: bytes(bad), ZIP_URL + ".CHECKSUM": _checksum(METRICS)})
    with pytest.raises(binance.ArchiveError, match="SHA-256"):
        binance.download_metrics_archive("BTCUSDT", DAY, DAY)


def test_zip_without_a_checksum_file_is_an_error(monkeypatch):
    _serve(monkeypatch, {ZIP_URL: METRICS})
    with pytest.raises(binance.ArchiveError, match="no checksum file"):
        binance.download_metrics_archive("BTCUSDT", DAY, DAY)


def test_day_without_a_zip_is_skipped_without_a_checksum_request(monkeypatch):
    server = _serve(monkeypatch, {})
    assert binance.download_metrics_archive("BTCUSDT", DAY, DAY).empty
    assert server.calls == [ZIP_URL]


@pytest.mark.parametrize(
    "text",
    [
        b"",
        b"not a checksum\n",
        b"abc123  " + ZIP_NAME.encode(),
        _checksum(METRICS, "ETHUSDT-metrics-2024-01-01.zip"),
        _checksum(METRICS) + b"extra words\n",
        b"\xff\xfe" * 40,
    ],
)
def test_checksum_file_in_another_form_is_an_error(text):
    with pytest.raises(binance.ArchiveError, match="checksum file"):
        binance.checksum_sha256(text, ZIP_NAME)


def test_checksum_file_forms_that_are_accepted():
    digest = hashlib.sha256(METRICS).hexdigest()
    assert binance.checksum_sha256(_checksum(METRICS), ZIP_NAME) == digest
    assert binance.checksum_sha256(f"{digest.upper()} *{ZIP_NAME}".encode(), ZIP_NAME) == digest
    assert binance.checksum_sha256(digest.encode(), ZIP_NAME) == digest


def test_body_above_the_limit_is_an_error_without_retry(monkeypatch):
    server = _serve(monkeypatch, {"https://example.invalid/f.zip": b"x" * 11})
    assert binance._get_bytes("https://example.invalid/f.zip", limit=11) == b"x" * 11
    with pytest.raises(binance.ResponseTooLarge, match="larger than 10 bytes"):
        binance._get_bytes("https://example.invalid/f.zip", limit=10)
    assert len(server.calls) == 2


def test_archive_and_checksum_limits(monkeypatch):
    _serve(monkeypatch, {ZIP_URL: METRICS, ZIP_URL + ".CHECKSUM": _checksum(METRICS)})
    monkeypatch.setattr(binance, "MAX_ARCHIVE_BYTES", len(METRICS) - 1)
    with pytest.raises(binance.ResponseTooLarge):
        binance.download_metrics_archive("BTCUSDT", DAY, DAY)
    monkeypatch.setattr(binance, "MAX_ARCHIVE_BYTES", len(METRICS))
    monkeypatch.setattr(binance, "MAX_CHECKSUM_BYTES", 10)
    with pytest.raises(binance.ResponseTooLarge):
        binance.download_metrics_archive("BTCUSDT", DAY, DAY)


def test_json_limit(monkeypatch):
    _serve(monkeypatch, {"https://example.invalid/api?a=1": b"[1, 2, 3]"})
    assert binance._get("https://example.invalid/api", {"a": 1}) == [1, 2, 3]
    monkeypatch.setattr(binance, "MAX_JSON_BYTES", 5)
    with pytest.raises(binance.ResponseTooLarge):
        binance._get("https://example.invalid/api", {"a": 1})


def test_only_https_urls_are_opened(monkeypatch):
    server = _serve(monkeypatch, {"file:///etc/hosts": b"x", "http://example.invalid/f": b"x"})
    for url in ("file:///etc/hosts", "http://example.invalid/f"):
        with pytest.raises(ValueError, match="only https"):
            binance._get_bytes(url)
    assert server.calls == []


def test_parser_gives_the_same_frame_as_a_direct_read():
    with zipfile.ZipFile(io.BytesIO(METRICS)) as z, z.open(z.namelist()[0]) as f:
        raw = pd.read_csv(f)
    df = binance.parse_metrics_zip(METRICS)
    assert list(df.index) == list(pd.to_datetime(raw["create_time"], utc=True))
    assert df["open_interest"].tolist() == raw["sum_open_interest"].astype(float).tolist()
    assert df["taker_buy_sell_ratio"].tolist() == raw["sum_taker_long_short_vol_ratio"].astype(float).tolist()


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"PK\x03\x04 not a zip",
        _zip("", "empty.csv"),
        _zip("a,b\n1,2\n"),  # no metrics columns
        _zip(HEADER + "not a date,BTCUSDT,x,y,1,1,1,1\n"),
        METRICS[: len(METRICS) // 2],
    ],
)
def test_bad_zip_content_is_an_archive_error(blob):
    with pytest.raises(binance.ArchiveError):
        binance.parse_metrics_zip(blob)


def test_zip_without_files_is_an_archive_error():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w"):
        pass
    with pytest.raises(binance.ArchiveError, match="no file"):
        binance.parse_metrics_zip(buf.getvalue())


def test_large_csv_in_the_zip_is_refused(monkeypatch):
    monkeypatch.setattr(binance, "MAX_CSV_BYTES", 100)
    big = _zip(HEADER + f"{DAY} 00:05:00,BTCUSDT,1,2,1,1.1,1.2,0.9\n" * 50)
    with pytest.raises(binance.ArchiveError, match="larger than 100 bytes"):
        binance.parse_metrics_zip(big)


@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::RuntimeWarning")  # pandas dates from random text
def test_random_bytes_give_an_archive_error():
    """Property test (Hypothesis): random bytes, and zip files with random content, give a
    frame or ArchiveError, never another exception."""
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st

    cfg = hypothesis.settings(max_examples=300, deadline=None, derandomize=True, database=None)

    def check(blob):
        try:
            out = binance.parse_metrics_zip(blob)
        except binance.ArchiveError:
            return
        assert isinstance(out, pd.DataFrame)

    @cfg
    @hypothesis.given(st.binary(max_size=2000))
    def random_bytes(blob):
        check(blob)

    @cfg
    @hypothesis.given(st.binary(max_size=200))
    def zip_prefix(tail):
        check(b"PK\x03\x04" + tail)

    @cfg
    @hypothesis.given(st.text(max_size=400))
    def random_csv(text):
        check(_zip(HEADER + text))
        check(_zip(text))

    @cfg
    @hypothesis.given(st.integers(min_value=0, max_value=len(METRICS) - 1), st.integers(min_value=0, max_value=255))
    def flipped_byte(pos, value):
        blob = bytearray(METRICS)
        blob[pos] = value
        check(bytes(blob))

    random_bytes()
    zip_prefix()
    random_csv()
    flipped_byte()
