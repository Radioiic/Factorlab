"""Integrity tests for engine rules.

    SPLICE  data_io funding splice: a partial primary day keeps the fallback
            settlements of the slots the primary lacks (settlement-slot policy)
    OFFLINE FACTORLAB_OFFLINE: one strict parser (config.offline_from) for ingest.py;
            only exactly "0" reaches exchange code, other values are rejected
    STORE   AlphaForge DataStore save/load keeps metadata axis dtypes
    PROTECT every engine writer refuses to write into a protected folder while
            the data folder holds the marker file (config.guard_write);
            ingest.py refuses before any exchange client exists
    CLI     each engine program has a description and a help text for each option,
            and --help writes no file; db.py runs SQL on a data folder whose path has a
            quote, and it keeps SQL words that start with "-"; run_tests.py probes an
            interpreter that is a link to its own
    IMPORTS each import in engine/ and library/ names a module that exists (the standard
            library, an installed package or a module of Factorlab)
    CHECKOUT config.py refuses a data folder inside another checkout: a folder with its own
            update.py and config.py

The newly eligible symbol test (no re-ranking of history) is with the other
lookahead proofs in test_labels.py.

Usage:
    python test_engine_rules.py          # or: pytest test_engine_rules.py

The tests marked "guard" pin what must NOT change. Nothing here touches the
network or the real data/ tree: raw-data tests redirect config paths into a
temp dir, and the exchange client is a fake that only records whether it was
reached. The write protection's end-to-end test builds its own data folder from
the synthetic demo market in a temp dir and runs every writer there.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import config as C
import data_io

SYM = "AAA-USDT"
SLOT = pd.Timedelta(hours=8)


# --- shared helpers ------------------------------------------------------------


@contextlib.contextmanager
def raw_sandbox():
    """Point the raw-data config paths at a temp dir; restore on exit."""
    names = ("RAW_OHLCV", "RAW_FUNDING", "RAW_FUNDING_FB", "RAW_OI")
    saved = {n: getattr(C, n) for n in names}
    with tempfile.TemporaryDirectory() as td:
        for n in names:
            p = Path(td) / n.lower()
            p.mkdir(parents=True)
            setattr(C, n, p)
        try:
            yield Path(td)
        finally:
            for n, v in saved.items():
                setattr(C, n, v)


@contextlib.contextmanager
def env(name: str, value: str | None):
    """Set (or, with None, unset) one environment variable; restore on exit."""
    had, old = name in os.environ, os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if had:
            os.environ[name] = old
        else:
            os.environ.pop(name, None)


def write_funding(folder: Path, ts, rates) -> None:
    pd.DataFrame({"ts": pd.DatetimeIndex(ts), "funding_rate": list(rates)}).to_parquet(
        folder / f"{SYM}.parquet", index=False
    )


def eight_hourly(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="8h")


def day(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


# --- funding splice -------------------------------------------------------------


def test_partial_primary_day_keeps_fallback_settlements():
    """The primary's first settlement is 16:00 and the fallback has 00:00 and
    08:00 that day. The day must be fb00 + fb08 + p16; a day-level splice
    returns p16 alone."""
    with raw_sandbox():
        fb_ts = eight_hourly("2026-04-28 00:00", "2026-05-02 16:00")
        fb = pd.Series(np.arange(1, len(fb_ts) + 1) * 1e-5, index=fb_ts)
        # the fallback venue stamps a few ms after the hour, as the real files do
        write_funding(C.RAW_FUNDING_FB, fb_ts + pd.Timedelta(milliseconds=2), fb.to_numpy())
        p_ts = eight_hourly("2026-04-30 16:00", "2026-05-02 16:00")
        p = pd.Series(np.arange(1, len(p_ts) + 1) * -1e-4, index=p_ts)
        write_funding(C.RAW_FUNDING, p_ts, p.to_numpy())

        got = data_io.daily_funding(SYM)
        boundary = fb[day("2026-04-30 00:00")] + fb[day("2026-04-30 08:00")] + p[day("2026-04-30 16:00")]
        assert np.isclose(got[day("2026-04-30")], boundary, rtol=0, atol=1e-15), (
            f"boundary day {got[day('2026-04-30')]} != fb00 + fb08 + p16 = {boundary}"
        )
        # days fully covered by one venue: exactly that venue, nothing added
        assert np.isclose(got[day("2026-04-29")], fb["2026-04-29"].sum(), rtol=0, atol=1e-15)
        assert np.isclose(got[day("2026-05-01")], p["2026-05-01"].sum(), rtol=0, atol=1e-15)

        detail = data_io.daily_funding_detail(SYM)
        b = detail.loc[day("2026-04-30")]
        assert (b["n_primary"], b["n_fallback"], b["source"]) == (1, 2, "mixed"), b.to_dict()
        assert detail.loc[day("2026-04-29"), "source"] == "fallback"
        assert detail.loc[day("2026-05-01"), "source"] == "primary"
        pd.testing.assert_series_equal(detail["funding_rate"], got)

        # no double counting: every 8h slot is paid by exactly one venue
        s = data_io.funding_settlements(SYM)
        assert s.groupby("slot")["venue"].nunique().max() == 1
        assert set(s["venue"]) == {"primary", "fallback"}
        # the backtester's funding matrix carries the same spliced number
        fm = data_io.funding_matrix(pd.DatetimeIndex([day("2026-04-30")]), [SYM])
        assert np.isclose(fm.loc[day("2026-04-30"), SYM], boundary, rtol=0, atol=1e-15)


def test_primary_gap_is_filled_for_that_slot_only():
    """A primary that misses one settlement mid-history: the fallback fills
    that slot and no other (a day-level splice leaves the day short)."""
    with raw_sandbox():
        fb_ts = eight_hourly("2026-05-01 00:00", "2026-05-03 16:00")
        fb = pd.Series(np.full(len(fb_ts), 7e-5), index=fb_ts)
        write_funding(C.RAW_FUNDING_FB, fb_ts, fb.to_numpy())
        p_ts = eight_hourly("2026-05-01 00:00", "2026-05-03 16:00").drop(day("2026-05-02 08:00"))
        p = pd.Series(np.full(len(p_ts), 1e-4), index=p_ts)
        write_funding(C.RAW_FUNDING, p_ts, p.to_numpy())

        got = data_io.daily_funding(SYM)
        assert np.isclose(got[day("2026-05-02")], 1e-4 + 7e-5 + 1e-4, rtol=0, atol=1e-15), got[day("2026-05-02")]
        assert np.isclose(got[day("2026-05-01")], 3e-4, rtol=0, atol=1e-15)
        assert np.isclose(got[day("2026-05-03")], 3e-4, rtol=0, atol=1e-15)
        detail = data_io.daily_funding_detail(SYM)
        assert detail.loc[day("2026-05-02"), "source"] == "mixed"


def test_jittered_fallback_stamp_counts_on_its_settlement_day():
    """A fallback stamp a few ms BEFORE midnight is the next day's 00:00
    settlement: it is slotted and summed as 00:00, not as the previous day."""
    with raw_sandbox():
        fb_ts = pd.DatetimeIndex([day("2026-04-29 16:00"), day("2026-04-30 00:00"), day("2026-04-30 08:00")])
        stored = fb_ts + pd.to_timedelta([3, -3, -2], unit="ms")  # 23:59:59.997 for 00:00
        write_funding(C.RAW_FUNDING_FB, stored, [1e-5, 2e-5, 4e-5])
        write_funding(C.RAW_FUNDING, [day("2026-04-30 16:00")], [8e-5])
        got = data_io.daily_funding(SYM)
        assert np.isclose(got[day("2026-04-29")], 1e-5, rtol=0, atol=1e-15), got[day("2026-04-29")]
        assert np.isclose(got[day("2026-04-30")], 2e-5 + 4e-5 + 8e-5, rtol=0, atol=1e-15), got[day("2026-04-30")]


def test_guard_shorter_fallback_interval_is_never_double_counted():
    """guard: the fallback settled SOL every 2h in Nov 2022. Where the primary
    owns a slot, none of those 2h prints may be added on top; where it owns
    none, all of them count."""
    with raw_sandbox():
        fb_ts = pd.date_range(day("2026-04-29 00:00"), day("2026-05-01 22:00"), freq="2h")
        write_funding(C.RAW_FUNDING_FB, fb_ts, np.full(len(fb_ts), 1e-5))
        p_ts = eight_hourly("2026-04-30 16:00", "2026-05-02 00:00")
        write_funding(C.RAW_FUNDING, p_ts, np.full(len(p_ts), 1e-4))
        got = data_io.daily_funding(SYM)
        # before the primary exists: all twelve 2h fallback prints
        assert np.isclose(got[day("2026-04-29")], 12 * 1e-5, rtol=0, atol=1e-15)
        # 04-30: the fallback keeps the slots closing 00:00 (its 00:00 print)
        # and 08:00 (02..08); the slots closing 16:00 (10..16) and 05-01 00:00
        # (18..22) belong to the primary, whose 16:00 print alone is added
        s = data_io.funding_settlements(SYM)
        assert s.groupby("slot")["venue"].nunique().max() == 1
        on_430 = s[s["settle"].dt.floor("D") == day("2026-04-30")]
        assert set(on_430.loc[on_430["venue"] == "primary", "settle"]) == {day("2026-04-30 16:00")}
        assert sorted(on_430.loc[on_430["venue"] == "fallback", "settle"].dt.hour) == [0, 2, 4, 6, 8]
        assert np.isclose(got[day("2026-04-30")], 5 * 1e-5 + 1e-4, rtol=0, atol=1e-15), got[day("2026-04-30")]
        # 05-01: every slot primary-owned -> exactly the three primary prints
        assert np.isclose(got[day("2026-05-01")], 3 * 1e-4, rtol=0, atol=1e-15), got[day("2026-05-01")]


def test_guard_single_venue_days_are_unchanged():
    """guard: the slot policy is narrow. With one venue only, or on days one
    venue covers completely, the daily series equals a day-level splice."""
    with raw_sandbox():
        rng = np.random.default_rng(3)
        fb_ts = eight_hourly("2026-01-01 00:00", "2026-03-31 16:00")
        fb_stored = fb_ts + pd.to_timedelta(rng.integers(0, 30, len(fb_ts)), unit="ms")
        write_funding(C.RAW_FUNDING_FB, fb_stored, rng.normal(1e-4, 2e-4, len(fb_ts)))
        only_fb = data_io._daily_funding_from(C.RAW_FUNDING_FB / f"{SYM}.parquet")
        pd.testing.assert_series_equal(data_io.daily_funding(SYM), only_fb, check_freq=False, check_names=False)
        p_ts = eight_hourly("2026-03-01 00:00", "2026-04-15 16:00")
        write_funding(C.RAW_FUNDING, p_ts, rng.normal(-1e-4, 2e-4, len(p_ts)))
        old = data_io._daily_funding_from(C.RAW_FUNDING / f"{SYM}.parquet").combine_first(
            data_io._daily_funding_from(C.RAW_FUNDING_FB / f"{SYM}.parquet")
        )
        new = data_io.daily_funding(SYM).reindex(old.index)
        pd.testing.assert_series_equal(new, old, check_freq=False, check_names=False)


def test_guard_midnight_slot_is_booked_on_the_primary_day():
    """guard (documented boundary, data_io docstring): a fallback print at 20:00
    on day D lies in the slot (D 16:00, D+1 00:00], which the primary's D+1
    00:00 print owns once it is on disk. That print pays the whole period on
    D+1, so the 20:00 print is dropped from D -- and D is REVISED when the
    00:00 print arrives after D was first computed. Both states are pinned."""
    with raw_sandbox():
        fb_ts = pd.date_range(day("2026-04-29 00:00"), day("2026-05-01 20:00"), freq="4h")
        write_funding(C.RAW_FUNDING_FB, fb_ts, np.full(len(fb_ts), 1e-5))
        # before the primary's first midnight print is ingested: it starts at 05-01 08:00
        write_funding(C.RAW_FUNDING, eight_hourly("2026-05-01 08:00", "2026-05-02 00:00"), [1e-4] * 3)
        early = data_io.daily_funding(SYM)
        assert np.isclose(early[day("2026-04-30")], 6 * 1e-5, rtol=0, atol=1e-15), early[day("2026-04-30")]
        # 05-01: the fallback still owns the slot closing 05-01 00:00 (its 00:00 print),
        # the primary owns 08:00 and 16:00 (its 04:00 / 12:00 / 20:00 prints are dropped)
        assert np.isclose(early[day("2026-05-01")], 1e-5 + 2e-4, rtol=0, atol=1e-15)

        # the primary's 05-01 00:00 print arrives: day 04-30 loses its 20:00 print
        write_funding(C.RAW_FUNDING, eight_hourly("2026-05-01 00:00", "2026-05-02 00:00"), [1e-4] * 4)
        late = data_io.daily_funding(SYM)
        assert np.isclose(late[day("2026-04-30")], 5 * 1e-5, rtol=0, atol=1e-15), late[day("2026-04-30")]
        assert np.isclose(late[day("2026-05-01")], 3 * 1e-4, rtol=0, atol=1e-15)
        s = data_io.funding_settlements(SYM)
        dropped = set(fb_ts) - set(s.loc[s["venue"] == "fallback", "settle"])
        assert day("2026-04-30 20:00") in dropped
        assert s.groupby("slot")["venue"].nunique().max() == 1  # still never paid twice
        # days before the boundary are untouched by the late print
        assert np.isclose(late[day("2026-04-29")], early[day("2026-04-29")], rtol=0, atol=0)


# --- strict offline parser --------------------------------------------------------

ACCEPTED = {None: True, "1": True, "0": False}
REJECTED = ["", "true", "yes", "false", "no", " 0", "0 ", "00", "1 ", "on", "off", "TRUE"]


class _Reached(Exception):
    """Raised by the fake exchange: proves exchange code was reached."""


class _FakeExchange:
    constructed = 0

    def __init__(self, *args, **kwargs):
        type(self).constructed += 1

    def load_markets(self):
        raise _Reached()


def test_offline_parser_accepts_only_unset_1_and_0():
    for value, offline in ACCEPTED.items():
        assert C.offline_from(value) is offline, value
    for value in REJECTED:
        try:
            C.offline_from(value)
        except SystemExit as exc:
            msg = str(exc.code)
            assert "FACTORLAB_OFFLINE" in msg and "exactly 0" in msg, msg
        else:
            raise AssertionError(f"FACTORLAB_OFFLINE={value!r} was accepted")


RAW_DIRS = ("RAW_OHLCV", "RAW_SPOT_OHLCV", "RAW_FUNDING", "RAW_FUNDING_FB", "RAW_OI")


@contextlib.contextmanager
def raw_in_fake_protected():
    """config's raw folders inside a fake_protected() data folder; restored on
    exit. Nothing is written into them."""
    with fake_protected() as data:
        saved = {n: getattr(C, n) for n in RAW_DIRS}
        for n in RAW_DIRS:
            setattr(C, n, data / "raw" / saved[n].name)
        try:
            yield data
        finally:
            for n, v in saved.items():
                setattr(C, n, v)


def _ingest_outcome(value, *argv, protected: bool = False) -> str:
    """Run ingest.main() with a fake exchange; classify what happened.

    protected=False runs where nothing is protected (fake_protected without the
    marker file), so the outcome depends on FACTORLAB_OFFLINE alone, whichever data
    root the suite runs against. protected=True puts the raw folders inside a
    protected data folder. Every SystemExit also asserts that no exchange object
    was built."""
    import ingest

    saved_ccxt, saved_argv, saved_check = ingest.ccxt, sys.argv, ingest.check
    fake = SimpleNamespace(BaseError=Exception)
    setattr(fake, C.EXCHANGE_ID, _FakeExchange)
    setattr(fake, C.FUNDING_FALLBACK_EXCHANGE_ID, _FakeExchange)
    ingest.ccxt, sys.argv, ingest.check = fake, ["ingest.py", *argv], (lambda: None)
    _FakeExchange.constructed = 0
    world = raw_in_fake_protected() if protected else fake_protected(marker=False)
    try:
        with world, env("FACTORLAB_OFFLINE", value), contextlib.redirect_stdout(io.StringIO()):
            ingest.main()
        return "check" if argv == ("--check",) else "returned"
    except _Reached:
        return "reached"
    except SystemExit as exc:
        msg = str(exc.code)
        assert _FakeExchange.constructed == 0, "exchange constructed before the guard"
        if "network ingest disabled" in msg:
            return "offline"
        if "not a valid setting" in msg:
            return "rejected"
        if isinstance(exc, C.WriteRefused) and "refusing to write" in msg:
            return "protected"
        raise
    finally:
        ingest.ccxt, sys.argv, ingest.check = saved_ccxt, saved_argv, saved_check


def test_ingest_reaches_exchange_only_for_exact_zero():
    """Only exactly "0" reaches load_markets; "true", "yes" and "" do not."""
    assert _ingest_outcome(None) == "offline"
    assert _ingest_outcome("1") == "offline"
    assert _ingest_outcome("0") == "reached"
    for value in REJECTED:
        got = _ingest_outcome(value)
        assert got == "rejected", f"FACTORLAB_OFFLINE={value!r}: {got}"
    # --check never fetches, but still refuses a value it cannot parse
    assert _ingest_outcome("1", "--check") == "check"
    assert _ingest_outcome("true", "--check") == "rejected"


def test_ingest_goes_on_when_the_funding_fallback_is_unreachable():
    """The fallback venue is optional: a load_markets error there (e.g. a region
    block) is reported and the run continues with the primary venue only."""
    import ingest

    class Blocked(Exception):
        pass

    class Primary:
        markets = {}

        def __init__(self, *args, **kwargs):
            pass

        def load_markets(self):
            pass

    class Fallback(Primary):
        def load_markets(self):
            raise Blocked("451 restricted location")

    fake = SimpleNamespace(BaseError=Blocked)
    setattr(fake, C.EXCHANGE_ID, Primary)
    setattr(fake, C.FUNDING_FALLBACK_EXCHANGE_ID, Fallback)
    saved_ccxt, saved_argv = ingest.ccxt, sys.argv
    ingest.ccxt, sys.argv = fake, ["ingest.py"]
    err = io.StringIO()
    try:
        with (
            fake_protected(marker=False),
            env("FACTORLAB_OFFLINE", "0"),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(err),
        ):
            ingest.main()
    finally:
        ingest.ccxt, sys.argv = saved_ccxt, saved_argv
    assert "not reachable (Blocked): primary funding only" in err.getvalue(), err.getvalue()


def test_ingest_refuses_a_protected_folder_before_any_exchange_exists():
    """With network allowed (FACTORLAB_OFFLINE=0) but a protected data folder, ingest
    stops at the write protection: the fake exchange is never constructed (asserted
    in _ingest_outcome), so no client is built and no request is sent, in both
    modes. --check only reads and still runs. The explicit override and a data
    folder without the marker file reach the exchange."""
    for argv in ((), ("--spot-only",)):
        assert _ingest_outcome("0", *argv, protected=True) == "protected", argv
        assert _FakeExchange.constructed == 0
    assert _ingest_outcome("0", "--check", protected=True) == "check"
    assert _ingest_outcome("1", protected=True) == "offline"
    with env(C.PROTECT_OVERRIDE_ENV, "1"):
        assert _ingest_outcome("0", protected=True) == "reached"
    assert _ingest_outcome("0", "--spot-only") == "reached"


# --- DataStore metadata axis dtypes ---------------------------------------------------


def _store():
    from alphaforge.data.store import DataStore

    # freq-less, like every grid the store builds (parquet cannot keep a freq)
    dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=6, freq="D").to_numpy(), name="date")
    assets = pd.Index(["A1", "A2", "A3"], dtype=object)
    close = pd.DataFrame(np.arange(18.0).reshape(6, 3) + 1.0, index=dates, columns=assets)
    meta = {
        "beta": pd.Series([0.9, 1.0, 1.1], index=assets, name="beta"),
        "panels": {"alpha": pd.DataFrame(np.ones((6, 3)), index=dates, columns=assets)},
        "note": "plain json",
    }
    return DataStore({"close": close}, meta)


def test_store_roundtrip_keeps_metadata_axis_dtypes():
    """A save and a load keep each metadata axis dtype. pandas >= 3 reads a
    parquet string axis as `str`, but the store's asset axes are `object`."""
    from alphaforge.data.store import DataStore

    store = _store()
    with tempfile.TemporaryDirectory() as td:
        store.save(td)
        loaded = DataStore.load(td)
    pd.testing.assert_series_equal(loaded.meta["beta"], store.meta["beta"])
    pd.testing.assert_frame_equal(loaded.meta["panels"]["alpha"], store.meta["panels"]["alpha"])
    assert loaded.meta["note"] == "plain json"
    assert loaded.assets.dtype == object


def test_store_loads_a_manifest_without_axis_dtypes_and_keeps_str_axes():
    """guard: a manifest with no recorded axis dtypes still loads
    (string axes -> object, the store's asset convention), and an axis saved
    with pandas' str dtype comes back as str (strict both ways)."""
    from alphaforge.data.store import DataStore

    store = _store()
    with tempfile.TemporaryDirectory() as td:
        store.save(td)
        manifest = Path(td) / "meta.json"
        payload = json.loads(manifest.read_text())
        payload["meta_panels"]["beta"].pop("axes")
        for entry in payload["meta_panels"]["panels"]["panels"].values():
            entry.pop("axes")
        manifest.write_text(json.dumps(payload))
        bare = DataStore.load(td)
    pd.testing.assert_series_equal(bare.meta["beta"], store.meta["beta"])
    pd.testing.assert_frame_equal(bare.meta["panels"]["alpha"], store.meta["panels"]["alpha"])

    store.meta["labels"] = pd.Series([1.0, 2.0], index=pd.Index(["x", "y"], dtype="str"))
    with tempfile.TemporaryDirectory() as td:
        store.save(td)
        loaded = DataStore.load(td)
    pd.testing.assert_series_equal(loaded.meta["labels"], store.meta["labels"])


# --- write protection (config.guard_write) -----------------------------------------


@contextlib.contextmanager
def fake_protected(marker: bool = True):
    """A temp data folder with the protected folders, other folders and (optionally)
    the marker file. config's DATA and SHIPPED_DATA point at it; restored on exit."""
    saved = (C.DATA, C.SHIPPED_DATA)
    with tempfile.TemporaryDirectory() as td:
        data = Path(td) / "data"
        for name in (*C.PROTECTED_DIRS, "library_on_engine", "other"):
            (data / name).mkdir(parents=True)
        if marker:
            (data / C.PROTECT_MARKER).write_text("protected\n")
        C.DATA, C.SHIPPED_DATA = data, data
        try:
            yield data
        finally:
            C.DATA, C.SHIPPED_DATA = saved


def _refused(path) -> str | None:
    """The refusal message for a write to `path`, or None if it is allowed."""
    try:
        C.guard_write(path)
    except C.WriteRefused as exc:
        return str(exc.code)
    return None


def test_write_protection_refuses_only_the_protected_folders():
    with fake_protected() as data, env(C.PROTECT_OVERRIDE_ENV, None):
        for name in C.PROTECTED_DIRS:
            msg = _refused(data / name / "x.parquet")
            assert msg and f"data/{name}/x.parquet" in msg, (name, msg)
            assert str(data) not in msg, "the message must not print an absolute path"
        assert _refused(data / "labels")  # the folder itself
        assert _refused(data / "trials" / "results" / "FT-0001" / "history.json")  # not created
        (data / "trials").rmdir()
        assert _refused(data / "trials" / "x"), "a missing protected folder is still protected"
        link = data.parent / "link"
        link.symlink_to(data / "research")
        assert _refused(link / "factor_registry.parquet"), "a symlink must not get past"
        for ok in (
            data / "other" / "x.json",
            data / "finder_x" / "y",
            data / "library_on_engine" / "x",
            data / "demo.json",
            data.parent / "elsewhere" / "z",
        ):
            assert _refused(ok) is None, ok
        # no marker file: nothing is refused
        (data / C.PROTECT_MARKER).unlink()
        assert _refused(data / "labels" / "labels.parquet") is None


def test_write_protection_override_is_explicit_and_strict():
    with fake_protected() as data:
        target = data / "labels" / "labels.parquet"
        with env(C.PROTECT_OVERRIDE_ENV, "1"):
            assert _refused(target) is None
        with env(C.PROTECT_OVERRIDE_ENV, "0"):
            assert _refused(target)
        for bad in ("", "true", "yes", " 1", "01"):
            with env(C.PROTECT_OVERRIDE_ENV, bad):
                try:
                    C.guard_write(target)
                except C.WriteRefused:
                    raise AssertionError(f"{bad!r} must be rejected as a setting, not read as 0") from None
                except SystemExit as exc:
                    assert "not a valid setting" in str(exc.code), exc.code
                else:
                    raise AssertionError(f"{C.PROTECT_OVERRIDE_ENV}={bad!r} was accepted")


def test_writers_default_targets_are_protected_folders():
    """The writers' default targets lie in PROTECTED_DIRS. The folders are compared by
    NAME relative to the data folder, so the test means the same thing with
    FACTORLAB_DATA set (config's paths are then absolute paths inside the copy)."""
    assert C.PROTECTED_DIRS == ("raw", "labels", "research", "trials")
    import factor_lab
    import labels
    import trials

    defaults = {
        "RAW_OHLCV": "raw",
        "RAW_SPOT_OHLCV": "raw",
        "RAW_FUNDING": "raw",
        "RAW_FUNDING_FB": "raw",
        "RAW_OI": "raw",
        "LABELS": "labels",
        "TRIALS": "trials",
        "RESEARCH": "research",
    }
    targets = {name: getattr(C, name) for name in defaults} | {
        "labels.path()": labels.path(),
        "trials.root()": trials.root(),
        "factor_lab.REGISTRY": factor_lab.REGISTRY,
    }
    defaults |= {"labels.path()": "labels", "trials.root()": "trials", "factor_lab.REGISTRY": "research"}
    for name, path in targets.items():
        rel = Path(path).relative_to(C.DATA)
        assert rel.parts[0] == defaults[name] and rel.parts[0] in C.PROTECTED_DIRS, (name, rel)
    with fake_protected() as data:
        for name, folder in defaults.items():
            hit = C.protected_folder(data / folder / "x.parquet")
            assert hit is not None and hit[1] == folder, name
        assert C.protected_folder(data / "library_on_engine" / "x") is None


# Every engine entry point that writes into a protected folder, as (label, kind, argv, clock):
# kind "main" calls <module>.main() with sys.argv = argv; "argv" calls <module>.main(argv[1:]);
# "code_change" runs judge.py record-code-change with a changed code fingerprint; "ingest_upsert"
# and "bridge_text" call the write function directly. clock: the time of ledger.CLOCK (UTC).
# Read-only modes are listed separately and must still run.
_LATER = "2026-10-01T10:00:00"  # the day after the last bar of the demo market
_WRITERS = [
    ("labels.py", "main", ["labels.py"], None),
    ("factor_lab.py", "main", ["factor_lab.py"], None),
    ("judge.py register", "argv", ["judge.py", "register", "--formula=-ts_mean(ret, 5)"], _LATER),
    ("judge.py log-week", "argv", ["judge.py", "log-week"], _LATER),
    ("judge.py log-values", "argv", ["judge.py", "log-values", "FT-0002", "--values", "{forward_values}"], _LATER),
    ("judge.py withdraw", "argv", ["judge.py", "withdraw", "FT-0003", "--reason", "a test"], _LATER),
    ("judge.py record-code-change", "code_change", ["judge.py", "record-code-change", "--reason", "x"], _LATER),
    ("ingest upsert", "ingest_upsert", None, None),
    ("alphaforge_bridge writer", "bridge_text", None, None),
]
_READERS = [
    ("factor_lab.py --report", "main", ["factor_lab.py", "--report"], None),
    ("judge.py status", "argv", ["judge.py", "status"], _LATER),
    ("judge.py check", "argv", ["judge.py", "check", "--formula=ts_mean(clv, 7)"], _LATER),
    ("judge.py verify", "argv", ["judge.py", "verify"], _LATER),
    ("labels.py --status", "argv", ["labels.py", "--status"], None),
]

# The data folder of the writer test (FACTORLAB_DATA): the demo market up to Friday 2026-07-31,
# two tests registered on Saturday 2026-08-01 (a formula and a values file), then the bars up to
# 2026-09-30 and one more formula test registered on 2026-10-01 (its first forward Sunday is after
# the data, so it can be withdrawn). The tests are fixtures of this check, not results.
_SETUP = r"""
import contextlib, io, json, sys
from datetime import UTC, datetime
from pathlib import Path
import numpy as np, pandas as pd
import config as C, demo_data, judge, labels, ledger as LG
data, work = Path(sys.argv[1]), Path(sys.argv[2])
master = work / "master"
demo_data.write_raw(master, demo_data.market(demo_data.SEED))
def cut(day):
    lim = pd.Timestamp(day, tz="UTC")
    for p in sorted((master / "raw").rglob("*.parquet")):
        df = pd.read_parquet(p)
        keep = (pd.to_datetime(df["ts"], utc=True).dt.floor("D") <= lim).to_numpy()
        out = data / p.relative_to(master)
        out.parent.mkdir(parents=True, exist_ok=True)
        df[keep].to_parquet(out, index=False)
    labels.build()
def at(text):
    t = datetime.fromisoformat(text).replace(tzinfo=UTC)
    LG.CLOCK = lambda: t
def values(first, last, path):
    sundays = pd.date_range(first, last, freq="W-SUN")
    syms = [C.sym_to_file(s) for s in C.UNIVERSE]
    rng = np.random.default_rng(7)
    rows = [(d.date().isoformat(), s, float(rng.normal())) for d in sundays for s in syms]
    pd.DataFrame(rows, columns=["date", "symbol", "value"]).to_csv(path, index=False)
    return str(path)
out = {}
with contextlib.redirect_stdout(io.StringIO()):
    cut("2026-07-31")
    at("2026-08-01T10:00:00")
    out["FT-0001"] = judge.main(["register", "--formula=ts_mean(clv, 7)", "--name", "fixture-a"])
    hist = values("2023-01-01", "2026-07-26", work / "history.csv")
    out["FT-0002"] = judge.main(["register", "--values", hist, "--name", "fixture-b"])
    cut("2026-09-30")
    at("2026-10-01T10:00:00")
    out["FT-0003"] = judge.main(["register", "--formula=-ts_std(ret, 30)", "--name", "fixture-c"])
out["forward_values"] = values("2026-08-02", "2026-09-27", work / "forward.csv")
print(json.dumps(out))
"""

_CHILD = r"""
import contextlib, importlib, io, json, sys
from datetime import UTC, datetime
from pathlib import Path
import config as C
import pandas as pd
data = Path(sys.argv[1])
out = {}
for label, kind, argv, clock in json.loads(sys.argv[2]):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import ledger as LG
            if clock:
                t = datetime.fromisoformat(clock).replace(tzinfo=UTC)
                LG.CLOCK = lambda t=t: t
            rc = None
            if kind == "ingest_upsert":
                import ingest
                ingest._upsert(C.RAW_OHLCV / "BTC-USDT.parquet",
                               pd.DataFrame({"ts": [pd.Timestamp("2030-01-01", tz="UTC")], "close": [1.0]}))
            elif kind == "bridge_text":
                import alphaforge_bridge
                alphaforge_bridge._atomic_text(alphaforge_bridge.LATEST, "x")
            elif kind == "code_change":
                import judge, trials
                real = trials.code_record
                def changed():
                    r = json.loads(json.dumps(real()))
                    r["files"][sorted(r["files"])[0]] = "0" * 64
                    r["sha256"] = "0" * 64
                    return r
                trials.code_record = changed
                try:
                    rc = judge.main(argv[1:])
                finally:
                    trials.code_record = real
            elif kind == "argv":
                rc = importlib.import_module(argv[0][:-3]).main(argv[1:])
            else:
                sys.argv = argv
                rc = importlib.import_module(argv[0][:-3]).main()
        out[label] = "returned" if rc in (None, 0) else f"exit {rc}"
    except C.WriteRefused as exc:
        out[label] = "refused" if "refusing to write" in str(exc.code) else "refused?"
    except SystemExit as exc:
        out[label] = f"exit {exc.code}"
    except Exception as exc:
        out[label] = f"error {type(exc).__name__}: {exc}"
print(json.dumps(out))
"""


def _tree_hash(root: Path) -> dict:
    import hashlib

    return {
        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(root.rglob("*"))
        if f.is_file()
    }


def test_every_writer_refuses_a_protected_folder_and_writes_nothing():
    """Behavioural, end to end: a data folder built in a temp dir (the synthetic demo market, a
    label table and a trial ledger with three fixture tests) gets the marker file, and every
    writer's real entry point runs against it. Each must stop with the refusal and change no byte;
    the read-only modes must still run. A writer without its guard would write into the temp
    folder only, and this test would name it."""
    import subprocess

    with tempfile.TemporaryDirectory(prefix="protect_") as td:
        data, work = Path(td) / "data", Path(td) / "work"
        work.mkdir()
        env_ = {k: v for k, v in os.environ.items() if k != C.PROTECT_OVERRIDE_ENV}
        env_.update(FACTORLAB_DATA=str(data), FACTORLAB_OFFLINE="1")
        setup = subprocess.run(
            [sys.executable, "-B", "-c", _SETUP, str(data), str(work)],
            cwd=C.ROOT,
            env=env_,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert setup.returncode == 0, setup.stderr[-2000:]
        made = json.loads(setup.stdout.strip().splitlines()[-1])
        assert [made[k] for k in ("FT-0001", "FT-0002", "FT-0003")] == [0, 0, 0], made
        assert (data / "trials" / "registrations.jsonl").is_file()
        entries = [
            (label, kind, None if argv is None else [a.format(**made) for a in argv], clock)
            for label, kind, argv, clock in _WRITERS + _READERS
        ]
        (data / C.PROTECT_MARKER).write_text("protected\n")
        before = _tree_hash(data)
        proc = subprocess.run(
            [sys.executable, "-B", "-c", _CHILD, str(data), json.dumps(entries)],
            cwd=C.ROOT,
            env=env_,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        got = json.loads(proc.stdout.strip().splitlines()[-1])
        assert _tree_hash(data) == before, "a writer changed the protected folder"
    assert set(got) == {w[0] for w in _WRITERS + _READERS}, got
    wrong = {k: v for k, v in got.items() if k in {w[0] for w in _WRITERS} and v != "refused"}
    assert not wrong, f"writers that did not refuse: {wrong}"
    blocked = {k: v for k, v in got.items() if k in {r[0] for r in _READERS} and v != "returned"}
    assert not blocked, f"read-only modes that do not run: {blocked}"


# --- command-line programs ----------------------------------------------------------


def _programs() -> list[str]:
    """The engine programs: engine/*.py (no test file, no file that git ignores) with a __main__ block."""
    import run_tests

    files = sorted(Path(C.ROOT).glob("*.py"))
    ignored = run_tests._git_ignored(files)
    out = []
    for p in files:
        if p.name.startswith("test_") or p.resolve() in ignored:
            continue
        if 'if __name__ == "__main__":' in p.read_text(encoding="utf-8"):
            out.append(p.name)
    return out


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*"))}


def test_every_program_has_help():
    """Each engine program has a description and a help text for each option. --help
    exits 0 and writes no file into the data folder."""
    import ast

    programs = _programs()
    assert {"update.py", "db.py", "judge.py", "labels.py", "version.py"} <= set(programs), programs
    for name in programs:
        tree = ast.parse((Path(C.ROOT) / name).read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        parsers = [n for n in calls if n.func.attr == "ArgumentParser"]
        assert parsers, f"{name} has no argument parser"
        assert all(any(k.arg == "description" for k in n.keywords) for n in parsers), f"{name}: no description"
        for n in calls:
            if n.func.attr == "add_argument":
                assert any(k.arg == "help" for k in n.keywords), f"{name}: {ast.unparse(n)[:80]} has no help"
    with tempfile.TemporaryDirectory(prefix="cli_help_") as tmp:
        data = Path(tmp) / "data"
        data.mkdir()
        e = {
            **os.environ,
            "FACTORLAB_DATA": str(data),
            "FACTORLAB_OFFLINE": "1",
            "TMPDIR": tmp,
            "MPLCONFIGDIR": str(Path(tmp) / "mpl"),
        }
        import subprocess

        for name in programs:
            before = _tree(data)
            r = subprocess.run(
                [sys.executable, "-B", name, "--help"], cwd=C.ROOT, env=e, capture_output=True, text=True, timeout=300
            )
            assert r.returncode == 0 and "usage:" in r.stdout, (name, r.stdout[-500:], r.stderr[-1000:])
            # importing config makes the standard folders; --help adds no other file
            new = {k for k in _tree(data) if k not in before and not (data / k).is_dir()}
            assert not new, (name, sorted(new))


def test_db_reads_a_folder_with_a_quote_in_its_path():
    """db.py quotes the configuration paths in its SQL text (a quote in the path is doubled). The
    view labels reads the label table."""
    import subprocess

    with tempfile.TemporaryDirectory(prefix="db_") as tmp:
        data = Path(tmp) / "it's data"
        for sub in ("labels", "raw/ohlcv", "raw/funding"):
            (data / sub).mkdir(parents=True)
        day = pd.Timestamp("2025-01-01", tz="UTC")
        pd.DataFrame(
            {"date": [day, day], "symbol": ["BTC-USDT", "ETH-USDT"], "fwd_ret_1": [0.0, 0.0], "fwd_ret_7": [0.1, 0.2]}
        ).to_parquet(data / "labels" / "labels.parquet", index=False)
        pd.DataFrame({"ts": [day], "close": [1.0]}).to_parquet(data / "raw" / "ohlcv" / "BTC-USDT.parquet", index=False)
        pd.DataFrame({"ts": [day], "funding_rate": [1e-4]}).to_parquet(
            data / "raw" / "funding" / "BTC-USDT.parquet", index=False
        )
        e = {**os.environ, "FACTORLAB_DATA": str(data), "TMPDIR": tmp}
        r = subprocess.run(
            [sys.executable, "-B", "db.py", "SELECT count(*) AS n, sum(fwd_ret_7) AS s FROM labels"],
            cwd=C.ROOT,
            env=e,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, r.stderr[-1000:]
        assert r.stdout.split()[:2] == ["n", "s"] and r.stdout.split()[2:4] == ["2", "0.3"], r.stdout
        r = subprocess.run(
            [sys.executable, "-B", "db.py", "SELECT symbol FROM ohlcv"],
            cwd=C.ROOT,
            env=e,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0 and "BTC-USDT" in r.stdout, r.stdout + r.stderr[-1000:]
        # the words of an unquoted statement can start with "-"
        r = subprocess.run(
            [sys.executable, "-B", "db.py", "SELECT", "-fwd_ret_7", "AS", "m", "FROM", "labels", "ORDER", "BY", "m"],
            cwd=C.ROOT,
            env=e,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0 and r.stdout.split() == ["m", "-0.2", "-0.1"], r.stdout + r.stderr[-1000:]


def test_db_works_without_the_label_table():
    """Without data/labels/labels.parquet (after ingest.py, before labels.py), db.py makes the
    other views: a query of ohlcv works, and a query of labels names the absent view."""
    import subprocess

    with tempfile.TemporaryDirectory(prefix="db_") as tmp:
        data = Path(tmp) / "data"
        for sub in ("raw/ohlcv", "raw/funding"):
            (data / sub).mkdir(parents=True)
        day = pd.Timestamp("2025-01-01", tz="UTC")
        pd.DataFrame({"ts": [day], "close": [1.0]}).to_parquet(data / "raw" / "ohlcv" / "BTC-USDT.parquet", index=False)
        pd.DataFrame({"ts": [day], "funding_rate": [1e-4]}).to_parquet(
            data / "raw" / "funding" / "BTC-USDT.parquet", index=False
        )
        e = {**os.environ, "FACTORLAB_DATA": str(data), "TMPDIR": tmp}

        def run(sql):
            return subprocess.run(
                [sys.executable, "-B", "db.py", sql], cwd=C.ROOT, env=e, capture_output=True, text=True, timeout=300
            )

        r = run("SELECT count(*) AS n FROM ohlcv")
        assert r.returncode == 0 and r.stdout.split() == ["n", "1"], r.stdout + r.stderr[-1000:]
        r = run("SELECT count(*) AS n FROM funding")
        assert r.returncode == 0 and r.stdout.split() == ["n", "1"], r.stdout + r.stderr[-1000:]
        r = run("SELECT count(*) AS n FROM labels")
        assert r.returncode != 0 and "labels" in r.stderr and "does not exist" in r.stderr, r.stderr[-1000:]


def test_run_tests_probes_a_linked_interpreter():
    """run_tests.has_pytest runs the probe in an interpreter that is a link to this one (a venv
    interpreter is such a link, with other packages); it does not use this process's packages."""
    import subprocess
    from unittest import mock

    import run_tests

    with tempfile.TemporaryDirectory(prefix="probe_") as tmp:
        link = Path(tmp) / "python"
        link.symlink_to(sys.executable)
        done = subprocess.CompletedProcess([], 0)
        with mock.patch.object(run_tests.subprocess, "run", return_value=done) as run:
            assert run_tests.has_pytest(str(link)) is True
        assert run.call_count == 1 and run.call_args[0][0][0] == str(link), run.call_args
        with mock.patch.object(run_tests.subprocess, "run") as run:
            run_tests.has_pytest(sys.executable)
        assert run.call_count == 0, "this interpreter needs no probe"


# --- the imports of the Python files ------------------------------------------------------

REPO = Path(C.ROOT).parent


# Without git: the folders that the walk of the repository leaves out (data, outputs, environments,
# and each hidden folder except .github).
_WALK_SKIP_PARTS = {"__pycache__", "node_modules"}
_WALK_SKIP_PREFIX = ("engine/data/", "engine/logs/", "web/data/", "library/data/", "library/catalog/")


def _walk_skips(rel: Path) -> bool:
    folders = rel.parts[:-1]
    hidden = [f for f in folders if f.startswith(".") and f != ".github"]
    return bool(hidden or set(folders) & _WALK_SKIP_PARTS or rel.as_posix().startswith(_WALK_SKIP_PREFIX))


def _repo_text_files() -> list[str]:
    """The text files of the repository, relative paths: git's list (tracked and not ignored) in a
    work tree, else a walk without the data, output and environment folders."""
    import subprocess

    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-co", "--exclude-standard"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        listed = r.stdout.splitlines() if r.returncode == 0 and r.stdout.strip() else None
    except OSError:
        listed = None
    if listed is None:
        listed = [p.relative_to(REPO).as_posix() for p in REPO.rglob("*") if p.is_file()]
        listed = [rel for rel in listed if not _walk_skips(Path(rel))]
    out = []
    for rel in listed:
        p = REPO / rel
        if not rel.strip() or not p.is_file():
            continue
        try:
            p.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue  # an image or another binary file
        out.append(rel)
    return out


def test_each_import_names_a_module_that_exists():
    """Each import of a Python file of engine/ or library/ names a module that exists. The module
    is in the standard library, in an installed package or in Factorlab. The check finds the
    module and does not run it."""
    import ast
    import importlib.util

    # git's list of the repository files (or the walk without the data, output and environment
    # folders): a package folder named "data" (engine/alphaforge/data/, library/factorlab/data/)
    # is scanned too
    files = [rel for rel in _repo_text_files() if rel.endswith(".py") and rel.startswith(("engine/", "library/"))]
    assert {"engine/alphaforge/data/store.py", "library/factorlab/data/store.py"} <= set(files), "package data folders"
    assert len(files) > 50, len(files)
    names: dict[str, set[str]] = {}
    for rel in files:
        p = REPO / rel
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"), filename=str(p))):
            if isinstance(node, ast.Import):
                found = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found = [node.module.split(".")[0]]
            else:
                continue
            for name in found:
                names.setdefault(name, set()).add(rel)
    assert {"config", "factorlab", "alphaforge", "pandas"} <= set(names), sorted(names)
    # engine/ and library/ are the two roots of Factorlab's own modules (library/ holds factorlab)
    roots = [str(REPO / "engine"), str(REPO / "library")]
    saved = list(sys.path)
    sys.path[:0] = [r for r in roots if r not in sys.path]
    try:
        missing = {name: sorted(where)[:3] for name, where in names.items() if importlib.util.find_spec(name) is None}
    finally:
        sys.path[:] = saved
    assert not missing, f"imports of a module that does not exist (module: files): {missing}"


def test_foreign_checkout_sentinel_is_update_py():
    """config.foreign_checkout finds a folder with its own update.py and config.py; this
    checkout itself and a folder without update.py are no other checkout."""
    assert C.foreign_checkout(Path(C.ROOT) / "data" / "x") is None
    with tempfile.TemporaryDirectory(prefix="checkout_") as tmp:
        other = Path(tmp) / "engine"
        (other / "data").mkdir(parents=True)
        (other / "config.py").write_text("")
        assert C.foreign_checkout(other / "data") is None, "config.py alone is no checkout"
        (other / "run.py").write_text("")
        assert C.foreign_checkout(other / "data") is None, "the sentinel is update.py"
        (other / "update.py").write_text("")
        assert C.foreign_checkout(other / "data") is not None
        assert C.foreign_checkout(other / "data").resolve() == other.resolve()
    src = (Path(C.ROOT) / "config.py").read_text(encoding="utf-8")
    assert '(p / "update.py").is_file() and (p / "config.py").is_file()' in src


# --- runner -------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} engine integrity tests passed")
    raise SystemExit(1 if failed else 0)
