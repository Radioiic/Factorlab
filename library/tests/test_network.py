"""`download` is the library's only network path, and it is offline by default.

Only the exact value FACTORLAB_OFFLINE=0 (the engine's switch) lets a request
out; unset or "1" refuses, and any other value is an error. urlopen is replaced
by a recorder, so these tests can never reach the network themselves.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from factorlab import cli, network
from factorlab.data import binance

REFUSED = (None, "1", "", "true", "yes", "false", " 0", "00", "0\n", "TRUE")
REPO = Path(__file__).resolve().parents[2]


class _Recorder:
    """Stands in for urllib.request.urlopen and records every call."""

    def __init__(self, payload=b"[]"):
        self.calls = []
        self.payload = payload

    def __call__(self, url, timeout=None, context=None):
        self.calls.append(url)
        body = self.payload

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, n=-1):
                return body if n is None or n < 0 else body[:n]

        return _Resp()


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", rec)
    return rec


def _set(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(network.ENV, raising=False)
    else:
        monkeypatch.setenv(network.ENV, value)


def test_parser_matches_the_engine_rule():
    assert network.offline_from(None) is True
    assert network.offline_from("1") is True
    assert network.offline_from("0") is False
    for value in REFUSED[2:]:
        with pytest.raises(network.NetworkDisabled, match="not a valid setting"):
            network.offline_from(value)


@pytest.mark.parametrize("value", REFUSED)
def test_download_refuses_before_any_request(value, recorder, monkeypatch, capsys, tmp_path):
    _set(monkeypatch, value)
    out = tmp_path / "panels"
    assert cli.main(["download", "--symbols", "BTCUSDT", "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert network.ENV in err
    assert recorder.calls == [], "a request left before the offline check"
    assert not out.exists(), "a refused download created its output folder"
    # tests/conftest.py: pytest's temp folders stay inside the repository, never the system temp dir
    assert REPO in tmp_path.resolve().parents, tmp_path


@pytest.mark.parametrize("value", REFUSED)
def test_python_api_is_gated_too(value, recorder, monkeypatch):
    _set(monkeypatch, value)
    with pytest.raises(network.NetworkDisabled):
        binance._get(binance.SPOT + "/api/v3/klines", {"symbol": "BTCUSDT"})
    with pytest.raises(network.NetworkDisabled):
        binance.download_metrics_archive("BTCUSDT", "2024-01-01", "2024-01-02")
    with pytest.raises(network.NetworkDisabled):
        binance._get_bytes(binance.VISION + "/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-01-01.zip")
    assert recorder.calls == []


def test_exact_zero_allows_the_request(recorder, monkeypatch):
    _set(monkeypatch, "0")
    network.require_network("test")  # no exception
    recorder.payload = json.dumps([1, 2]).encode()
    assert binance._get(binance.SPOT + "/api/v3/ping", {}) == [1, 2]
    assert len(recorder.calls) == 1
