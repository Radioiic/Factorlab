"""scripts/merge_panels.py joins per-symbol downloads into one panel set, offline.

Synthetic part folders only (written under the git-ignored data/pytest-tmp/); no network.
The --append tests check that new closed bars change no old byte. The append must stop on a
bar that is not closed and on a gap. It must also stop on an incomplete field, on a coin without
its kline bar, on a changed old cell and on other columns. A field is incomplete when one coin
has no value that it had on the last merged bar.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factorlab.data import load_panels, save_panels
from factorlab.paths import CATALOG_DIR

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_panels.py"


def _module():
    spec = importlib.util.spec_from_file_location("merge_panels", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parts(root: Path) -> None:
    """Two symbols with different first bars, two fields each (one column per part file)."""
    idx = pd.date_range("2022-01-01", periods=6, freq="D", tz="UTC", name="open_time")
    for sym, start in (("AAAUSDT", 0), ("BBBUSDT", 2)):
        ix = idx[start:]
        close = pd.DataFrame({sym: np.arange(len(ix), dtype=float) + 10 * (start + 1)}, index=ix)
        fund = pd.DataFrame({sym: np.full(len(ix) - 1, 1e-4)}, index=ix[:-1])  # the last bar has no settlement
        save_panels({"close": close, "funding_rate": fund}, root / sym)


def test_merge_orders_columns_and_uses_the_close_index(tmp_path):
    mod = _module()
    _parts(tmp_path / "parts")
    out = tmp_path / "merged"
    assert mod.main(["--parts", str(tmp_path / "parts"), "--out", str(out), "--symbols", "BBBUSDT,AAAUSDT"]) == 0
    with (out / "close.csv.gz").open("rb") as fh:
        import gzip

        header = gzip.decompress(fh.read()).decode().splitlines()[0]
    assert header.split(",")[1:] == ["BBBUSDT", "AAAUSDT"], "columns follow --symbols"
    p = load_panels(out)
    assert set(p) == {"close", "funding_rate"} and len(p["close"]) == 6
    assert p["close"]["BBBUSDT"].isna().sum() == 2 and p["close"]["AAAUSDT"].notna().all()
    assert p["funding_rate"]["AAAUSDT"].iloc[-1] != p["funding_rate"]["AAAUSDT"].iloc[-1]  # NaN kept
    assert p["close"].loc["2022-01-03", "BBBUSDT"] == 30.0


def test_remerge_is_byte_identical(tmp_path):
    """The merged files carry a fixed gzip modification time, so merging the same parts
    twice (into different folders, at different times) gives the same bytes."""
    mod = _module()
    _parts(tmp_path / "parts")
    outs = [tmp_path / "one", tmp_path / "two"]
    syms = ["AAAUSDT", "BBBUSDT"]
    for out in outs:
        assert mod.main(["--parts", str(tmp_path / "parts"), "--out", str(out), "--symbols", ",".join(syms)]) == 0
    names = sorted(p.name for p in outs[0].glob("*.csv.gz"))
    assert names == ["close.csv.gz", "funding_rate.csv.gz"]
    for name in names:
        a, b = (outs[0] / name).read_bytes(), (outs[1] / name).read_bytes()
        assert a == b, name
        assert a[:2] == b"\x1f\x8b" and int.from_bytes(a[4:8], "little") == mod.GZIP_MTIME == 0, name
    # without the option the header carries the write time; the CSV content is the same either way
    import gzip

    save_panels(mod.merge(tmp_path / "parts", syms), tmp_path / "plain")
    for name in names:
        assert gzip.decompress((tmp_path / "plain" / name).read_bytes()) == gzip.decompress(
            (outs[0] / name).read_bytes()
        )


def test_merge_refuses_inconsistent_parts(tmp_path):
    mod = _module()
    _parts(tmp_path / "parts")
    with pytest.raises(FileNotFoundError):
        mod.merge(tmp_path / "parts", ["AAAUSDT", "CCCUSDT"])
    extra = pd.DataFrame({"AAAUSDT": [1.0]}, index=pd.DatetimeIndex(["2022-01-01"], tz="UTC"))
    save_panels({"premium": extra}, tmp_path / "parts" / "AAAUSDT")  # a field BBBUSDT lacks
    with pytest.raises(ValueError):
        mod.merge(tmp_path / "parts", ["AAAUSDT", "BBBUSDT"])


def test_merge_refuses_the_catalog_folder(tmp_path, capsys):
    mod = _module()
    _parts(tmp_path / "parts")
    target = CATALOG_DIR / "merged_panels_should_not_exist"
    assert mod.main(["--parts", str(tmp_path / "parts"), "--out", str(target)]) == 2
    assert not target.exists()
    assert "refusing" in capsys.readouterr().err


# --------------------------------------------------------------------------- --append


def _days(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", tz="UTC", name="open_time")


def _write_parts(root: Path, idx: pd.DatetimeIndex, *, offset: float = 0.0, funding_last: bool = True) -> None:
    """Parts for AAAUSDT and BBBUSDT on `idx`: close = 100 + day number + offset, funding 1e-4
    on every bar (on none of the last bar when funding_last is False)."""
    base = (idx - pd.Timestamp("2022-01-01", tz="UTC")).days.to_numpy(dtype=float)
    for k, sym in enumerate(("AAAUSDT", "BBBUSDT")):
        close = pd.DataFrame({sym: 100.0 * (k + 1) + base + offset + 0.123456789}, index=idx)
        fund = pd.Series(1e-4 * (k + 1), index=idx)
        if not funding_last:
            fund.iloc[-1] = np.nan
        save_panels({"close": close, "funding_rate": fund.to_frame(sym)}, root / sym)


def _merged(tmp_path: Path, mod) -> Path:
    _write_parts(tmp_path / "base", _days("2022-01-01", 6))
    out = tmp_path / "merged"
    assert mod.main(["--parts", str(tmp_path / "base"), "--out", str(out), "--symbols", "AAAUSDT,BBBUSDT"]) == 0
    return out


def _snapshot(out: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(out.glob("*.csv.gz"))}


def _append(mod, parts: Path, out: Path, now: str = "2022-01-20", *extra: str) -> int:
    return mod.main(
        ["--append", "--parts", str(parts), "--out", str(out), "--symbols", "AAAUSDT,BBBUSDT", "--now", now, *extra]
    )


def test_append_adds_closed_bars_and_keeps_every_old_byte(tmp_path):
    """The appended files start with the old decompressed content, byte for byte. They keep the
    gzip header time 0. They equal a full merge of the same rows. A second append of the same
    parts changes nothing."""
    import gzip

    mod = _module()
    out = _merged(tmp_path, mod)
    before = _snapshot(out)
    _write_parts(tmp_path / "upd", _days("2022-01-07", 3))
    assert _append(mod, tmp_path / "upd", out) == 0
    after = _snapshot(out)
    assert sorted(after) == sorted(before) == ["close.csv.gz", "funding_rate.csv.gz"]
    for name, blob in after.items():
        assert gzip.decompress(blob).startswith(gzip.decompress(before[name])), name
        assert int.from_bytes(blob[4:8], "little") == mod.GZIP_MTIME == 0, name
    p = load_panels(out)
    assert len(p["close"]) == 9 and str(p["close"].index[-1].date()) == "2022-01-09"
    # the same rows in one full merge give the same bytes
    _write_parts(tmp_path / "full", _days("2022-01-01", 9))
    assert (
        mod.main(["--parts", str(tmp_path / "full"), "--out", str(tmp_path / "one"), "--symbols", "AAAUSDT,BBBUSDT"])
        == 0
    )
    assert _snapshot(tmp_path / "one") == after
    # one log line per append, with relative names only
    import json

    log = [json.loads(x) for x in (out / mod.APPEND_LOG).read_text().splitlines()]
    parts = log[0].pop("parts")
    assert log == [
        {
            "appended_at": "2022-01-20T00:00:00Z",
            "first_bar": "2022-01-07",
            "last_bar": "2022-01-09",
            "bars": 3,
            "rows": 9,
            "old_cells": "byte-equal",
        }
    ]
    assert parts.endswith("upd") and not parts.startswith("/") and str(tmp_path.parent) not in parts
    # a repeated append finds the bars already there, checks that they agree and writes nothing
    assert _append(mod, tmp_path / "upd", out) == 0
    assert _snapshot(out) == after and len((out / mod.APPEND_LOG).read_text().splitlines()) == 1
    # a full merge rebuilds the panel from the parts only, so it removes the log
    assert mod.main(["--parts", str(tmp_path / "base"), "--out", str(out), "--symbols", "AAAUSDT,BBBUSDT"]) == 0
    assert not (out / mod.APPEND_LOG).exists() and _snapshot(out) == before


def test_append_refuses_a_bar_that_is_not_closed(tmp_path, capsys):
    mod = _module()
    out = _merged(tmp_path, mod)
    before = _snapshot(out)
    _write_parts(tmp_path / "upd", _days("2022-01-07", 3))
    assert _append(mod, tmp_path / "upd", out, "2022-01-09T12:00:00Z") == 2  # 2022-01-09 closes at 01-10 00:00
    assert "not closed" in capsys.readouterr().err
    assert _snapshot(out) == before
    assert _append(mod, tmp_path / "upd", out, "2022-01-09T12:00:00Z", "--through", "2022-01-08") == 0
    assert str(load_panels(out)["close"].index[-1].date()) == "2022-01-08"
    assert _append(mod, tmp_path / "upd", out, "2022-01-10T00:00:00Z") == 0  # closed now: the last bar follows
    assert str(load_panels(out)["close"].index[-1].date()) == "2022-01-09"


def test_append_refuses_a_change_to_an_old_cell(tmp_path, capsys):
    """Downloaded bars that the panel already has must agree with it; on a difference the
    append stops and the merged files keep every byte."""
    mod = _module()
    out = _merged(tmp_path, mod)
    before = _snapshot(out)
    _write_parts(tmp_path / "same", _days("2022-01-05", 4))  # 2 old bars, 2 new bars
    _write_parts(tmp_path / "moved", _days("2022-01-05", 4), offset=1e-3)
    assert _append(mod, tmp_path / "moved", out) == 2
    assert "differ from the merged cells" in capsys.readouterr().err
    assert _snapshot(out) == before
    assert _append(mod, tmp_path / "same", out) == 0
    assert len(load_panels(out)["close"]) == 8


def test_append_refuses_a_gap_and_an_incomplete_field(tmp_path, capsys):
    mod = _module()
    out = _merged(tmp_path, mod)
    before = _snapshot(out)
    _write_parts(tmp_path / "gap", _days("2022-01-08", 2))  # 2022-01-07 is missing
    assert _append(mod, tmp_path / "gap", out) == 2
    assert "day by day" in capsys.readouterr().err
    _write_parts(tmp_path / "short", _days("2022-01-07", 3), funding_last=False)  # no funding for any coin on 01-09
    assert _append(mod, tmp_path / "short", out) == 2
    err = capsys.readouterr().err
    assert "2 value(s) are missing" in err and "funding_rate AAAUSDT 2022-01-09; funding_rate BBBUSDT 2022-01-09" in err
    assert "Download those days again after one day" in err
    assert _snapshot(out) == before
    assert _append(mod, tmp_path / "short", out, "2022-01-20", "--through", "2022-01-08") == 0
    assert str(load_panels(out)["close"].index[-1].date()) == "2022-01-08"


def test_append_refuses_one_coin_without_its_archive(tmp_path, capsys):
    """One coin's metrics archive is missing on the last new day (the downloader skips an HTTP
    404 without an error): the other coin has its value, but the append still stops, writes
    nothing and names the coin. --through stops before the day."""
    mod = _module()
    out = _merged(tmp_path, mod)
    before = _snapshot(out)
    _write_parts(tmp_path / "upd", _days("2022-01-07", 3))
    one = tmp_path / "upd" / "BBBUSDT"
    fund = load_panels(one)["funding_rate"]
    fund.iloc[-1] = np.nan  # BBBUSDT only, on 2022-01-09
    save_panels({"close": load_panels(one)["close"], "funding_rate": fund}, one)
    assert _append(mod, tmp_path / "upd", out) == 2
    err = capsys.readouterr().err
    assert "1 value(s) are missing" in err and "funding_rate BBBUSDT 2022-01-09" in err and "AAAUSDT" not in err
    assert _snapshot(out) == before and not (out / mod.APPEND_LOG).exists()
    assert _append(mod, tmp_path / "upd", out, "2022-01-20", "--through", "2022-01-08") == 0
    p = load_panels(out)
    assert str(p["close"].index[-1].date()) == "2022-01-08" and p["funding_rate"].iloc[-1].notna().all()
    # a coin without a value on the last merged bar needs none on the new bars
    out2 = tmp_path / "merged2"
    _write_parts(tmp_path / "base2", _days("2022-01-01", 6), funding_last=False)
    assert mod.main(["--parts", str(tmp_path / "base2"), "--out", str(out2), "--symbols", "AAAUSDT,BBBUSDT"]) == 0
    _write_parts(tmp_path / "upd2", _days("2022-01-07", 2), funding_last=False)
    assert _append(mod, tmp_path / "upd2", out2) == 0


@pytest.mark.parametrize("day", ["2022-01-08", "2022-01-09"])
def test_append_refuses_one_coin_without_its_kline_bar(tmp_path, capsys, day):
    """BBBUSDT has no kline bar on a middle new day or on the last new day. AAAUSDT has its
    bar, so the day stays in the index. The append stops, writes nothing and names the coin
    and the day. --through stops before the day."""
    mod = _module()
    out = _merged(tmp_path, mod)
    before = _snapshot(out)
    _write_parts(tmp_path / "upd", _days("2022-01-07", 3))
    one = tmp_path / "upd" / "BBBUSDT"
    p = load_panels(one)
    keep = p["close"].index != pd.Timestamp(day, tz="UTC")
    save_panels({f: df[keep] for f, df in p.items()}, one)  # no bar, no field on that day
    assert _append(mod, tmp_path / "upd", out) == 2
    err = capsys.readouterr().err
    assert f"close BBBUSDT {day}" in err and f"funding_rate BBBUSDT {day}" not in err and "AAAUSDT" not in err, err
    assert "In 1 of them, the coin has no kline bar (close)" in err and "--through" in err
    assert _snapshot(out) == before and not (out / mod.APPEND_LOG).exists()
    stop = str((pd.Timestamp(day) - pd.Timedelta(days=1)).date())
    assert _append(mod, tmp_path / "upd", out, "2022-01-20", "--through", stop) == 0
    p = load_panels(out)
    assert str(p["close"].index[-1].date()) == stop and p["close"].notna().all().all()


def test_append_refuses_other_fields_or_columns(tmp_path, capsys):
    mod = _module()
    out = _merged(tmp_path, mod)
    before = _snapshot(out)
    _write_parts(tmp_path / "upd", _days("2022-01-07", 2))
    assert (
        mod.main(
            [
                "--append",
                "--parts",
                str(tmp_path / "upd"),
                "--out",
                str(out),
                "--symbols",
                "BBBUSDT,AAAUSDT",
                "--now",
                "2022-01-20",
            ]
        )
        == 2
    )
    assert "--symbols gives" in capsys.readouterr().err
    extra = pd.DataFrame({"AAAUSDT": [1.0, 1.0]}, index=_days("2022-01-07", 2))
    for sym in ("AAAUSDT", "BBBUSDT"):
        save_panels({"premium": extra.rename(columns={"AAAUSDT": sym})}, tmp_path / "upd" / sym)
    assert _append(mod, tmp_path / "upd", out) == 2
    assert "the parts have fields" in capsys.readouterr().err
    assert _snapshot(out) == before
    assert not any(p.name.endswith(".append-tmp") for p in tmp_path.iterdir()), "a temporary folder was left"
