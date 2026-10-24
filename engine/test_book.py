"""Tests of book.py (the book specification, BandedBook, the book run and the weekly series) and
of the schedule argument of backtest.run_backtest.

Usage:
    python -B test_book.py

The suite reads no data folder of the repository. At the start it sets FACTORLAB_DATA to a new
folder under the git-ignored engine/.tmp/ and removes that folder at the end. The tests that need
market data write the synthetic demo market (demo_data.market, demo_data.write_raw) into that
folder; the other tests use synthetic prices in memory.

The properties:
    1. the shipped specifications have their pinned sha256; comments and layout do not change it;
       each refusal names the key and the rule; each change of a field gives another sha256
    2. book_key does not change with name, version, sign, the name of the cost scenario and the
       table [forward]
    3. BandedBook gives the weights of BandedQuintile (an independent reference construction in
       this file) at the default fractions (random worlds with ties and NaN, and four demo
       formulas)
    4. the fractions, the exit band and the sign do what the specification says
    5. at a Sunday with fewer than MIN_NAMES values the book has the control portfolio; the
       Sunday is reported
    6. only eligible coins get a weight; the control's universe does not depend on the factor
    7. the weekly series, the zero control and the equal-weight control
    8. schedule=None and a Monday schedule are the weekly rule of the engine (also pinned numbers
       for three rebalance weekdays); another schedule trades on its days only; the trade log of
       a book run reconciles; the cost scenarios
"""

from __future__ import annotations

import atexit
import hashlib
import math
import os
import shutil
import tempfile
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
(ENGINE / ".tmp").mkdir(exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="test_book_", dir=ENGINE / ".tmp"))
atexit.register(shutil.rmtree, TMP, True)
os.environ["FACTORLAB_DATA"] = str(TMP / "data")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import backtest as B  # noqa: E402
import book  # noqa: E402
import config as C  # noqa: E402
import data_io  # noqa: E402
import ledger  # noqa: E402

SYMS = [C.sym_to_file(s) for s in C.UNIVERSE]
LONG_SHORT_SHA = "cee7d7cd6fae97ae58e420b3a3b15152d34e410500db1de7469ac3e164461f51"
LONG_ONLY_SHA = "03549ed515d48d2ccda54b576a014ff99c58cecc8ff68ddd383e9513662cbf72"
DEFAULT_BODY = (
    '{"book":{"bottom_fraction":0.2,"direction":"long_short","exit_fraction":0.4,"rebalance":"weekly-monday",'
    '"sign":1,"top_fraction":0.2},"control":{"kind":"zero"},"costs":{"fee_bps":5.0,"scenario":"taker",'
    '"slippage_bps":5.0},"format":1,"forward":{"sigma_floor_fallback":0.03},"name":"long-short","version":1}'
)
FORMULAS = ("-ts_mean(ex_ret, 20)", "ts_mean(clv, 7)", "ts_rank(close, 365)", "-ts_std(ret, 30)")
_DEMO: dict = {}


# --------------------------------------------------------------------------- helpers


def spec_file(text: str, name: str = "spec.toml") -> Path:
    d = Path(tempfile.mkdtemp(prefix="spec_", dir=TMP))
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p


def spec_of(text: str) -> book.BookSpec:
    return book.load_spec(spec_file(text))


MINIMAL = 'format = 1\nname = "x"\nversion = 1\n'


def world(seed: int = 1, n_days: int = 300, vol: float = 0.03):
    """Synthetic daily closes of the 20 engine symbols from Sunday 2023-01-01, small random
    funding, and every coin eligible."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, vol, (n_days, len(SYMS))), axis=0)), index=idx, columns=SYMS
    )
    funding = pd.DataFrame(rng.normal(1e-4, 2e-4, (n_days, len(SYMS))), index=idx, columns=SYMS)
    return prices, funding, pd.DataFrame(True, index=idx, columns=SYMS)


def sundays_of(prices: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(prices.index[prices.index.weekday == 6])


def random_values(prices: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = sundays_of(prices)
    return pd.DataFrame(rng.normal(0.0, 1.0, (len(s), len(SYMS))), index=s, columns=SYMS)


def run(spec, prices, funding, eligible, values, **kw) -> book.BookRun:
    s = sundays_of(prices)
    return book.run_book(spec, values, start=s[0], end=s[-1], prices=prices, funding=funding, eligible=eligible, **kw)


def same_result(a: B.Result, b: B.Result) -> None:
    pd.testing.assert_frame_equal(a.equity, b.equity, check_exact=True)
    pd.testing.assert_frame_equal(a.trades, b.trades, check_exact=True)
    pd.testing.assert_frame_equal(a.daily, b.daily, check_exact=True)
    assert a.stats == b.stats


def demo() -> dict:
    """The demo market in this suite's data folder (written once), its library panels and
    the eligible matrix."""
    if not _DEMO:
        import demo_data
        import labels
        import libfactors as lf

        demo_data.write_raw(C.DATA, demo_data.market(demo_data.SEED))
        panels = lf.build_panels(lf.load_raw())
        _DEMO.update(panels=panels, ev=lf._evaluator(panels), eligible=labels.eligible(), lf=lf)
    return _DEMO


def demo_values(expr: str) -> pd.DataFrame:
    d = demo()
    idx = d["panels"]["close"].index
    vals = pd.DataFrame(d["lf"]._values(d["ev"], expr), index=idx, columns=d["panels"]["close"].columns)
    return vals.iloc[d["lf"].sunday_positions(idx)]


class BandedQuintile:
    """The reference banded quintile construction of BandedBook (a separate implementation).

    It enters a side inside the top (bottom) `entry` fraction and keeps a name until it leaves
    the top (bottom) `exit` fraction. Each side is equal-weight. Fewer than 5 values: no position."""

    MIN_NAMES = 5

    def __init__(self, score_fn, long_only=False, entry=0.20, exit=0.40):
        self.score_fn = score_fn
        self.long_only, self.entry, self.exit = long_only, entry, exit
        self.longs: set = set()
        self.shorts: set = set()

    def __call__(self, cs: pd.DataFrame) -> pd.Series:
        s = self.score_fn(cs).dropna()
        if len(s) < self.MIN_NAMES:
            self.longs, self.shorts = set(), set()
            return pd.Series(dtype=float)
        r = s.rank(pct=True)  # 1.0 = strongest long
        n = max(2, len(s) // 5)
        top_n = set(s.nlargest(n).index)
        self.longs = {x for x in self.longs if x in r.index and r[x] >= 1 - self.exit} | top_n
        w = pd.Series(0.0, index=s.index)
        if self.long_only:
            w[list(self.longs)] = 1.0 / len(self.longs)
        else:
            bot_n = set(s.nsmallest(n).index)
            self.shorts = {x for x in self.shorts if x in r.index and r[x] <= self.exit} | bot_n
            self.shorts -= self.longs  # a name is not on both sides
            self.longs -= bot_n
            w[list(self.longs)] = 0.5 / len(self.longs)
            w[list(self.shorts)] = -0.5 / len(self.shorts)
        return w


def banded_quintile():
    return BandedQuintile


# --------------------------------------------------------------------------- 1. the specification


def test_shipped_spec_hashes():
    ls = book.load_spec()
    assert ls.sha256 == LONG_SHORT_SHA and ledger.canonical(ls.body) == DEFAULT_BODY, ledger.canonical(ls.body)
    assert ls.source == "long_short.toml" and book.load_spec("long_short").sha256 == LONG_SHORT_SHA
    lo = book.load_spec("long_only")
    assert lo.sha256 == LONG_ONLY_SHA, lo.sha256
    assert lo.body["book"]["bottom_fraction"] is None and lo.control == "equal_weight" and lo.long_only
    assert book.load_spec(book.SPEC_DIR / "long_only.toml").sha256 == LONG_ONLY_SHA
    assert hashlib.sha256(DEFAULT_BODY.encode("ascii")).hexdigest() == LONG_SHORT_SHA
    # the properties of the default
    assert (ls.name, ls.version, ls.sign, ls.top, ls.bottom, ls.exit) == ("long-short", 1, 1, 0.2, 0.2, 0.4)
    assert (ls.fee_bps, ls.slippage_bps, ls.control, ls.forward.sigma_floor_fallback) == (5.0, 5.0, "zero", 0.03)
    # a body from a record gives the same specification; a body that is not normalized is refused
    again = book.spec_from_body(ls.body)
    assert again.sha256 == ls.sha256 and again.source == "ledger"
    raw = {**ls.body, "costs": {"scenario": "taker"}}
    try:
        book.spec_from_body(raw)
    except book.SpecError as exc:
        assert "normalized" in str(exc)
    else:
        raise AssertionError("a body that is not normalized was accepted")


def test_comments_and_layout_do_not_change_the_hash():
    text = (
        "# another layout\n"
        'version = 1\nname = "long-short"\nformat = 1\n'
        "[forward]\nsigma_floor_fallback = 0.030\n"
        '[control]\nkind = "zero"   # no position\n'
        '[costs]\nscenario = "taker"\nfee_bps = 5\nslippage_bps = 5.0\n'
        '[book]\nexit_fraction = 0.4\nbottom_fraction = 0.2\ntop_fraction = 0.2\nsign = 1\ndirection = "long_short"\n'
    )
    assert spec_of(text).sha256 == LONG_SHORT_SHA
    # every key except format, name and version has a default: the default file in four lines
    assert spec_of('format = 1\nname = "long-short"\nversion = 1\n').sha256 == LONG_SHORT_SHA
    inline = 'format = 1\nname = "long-only"\nversion = 1\nbook = { direction = "long_only" }\n'
    assert spec_of(inline).sha256 == LONG_ONLY_SHA


def test_spec_refusals():
    table = [
        (MINIMAL + "colour = 1\n", "colour"),
        (MINIMAL + "[other]\nx = 1\n", "other"),
        (MINIMAL + "[book]\nwidth = 1\n", "book.width"),
        (MINIMAL.replace("format = 1", "format = 2"), "format"),
        (MINIMAL.replace('"x"', '"a b"'), "name"),
        (MINIMAL.replace('"x"', '"' + "n" * 65 + '"'), "name"),
        (MINIMAL.replace("version = 1", "version = 0"), "version"),
        (MINIMAL.replace("version = 1", "version = 1.0"), "version"),
        ('format = 1\nname = "x"\n', "version"),
        (MINIMAL + "[book]\nsign = 0\n", "book.sign"),
        (MINIMAL + "[book]\nsign = true\n", "book.sign"),
        (MINIMAL + "[book]\ntop_fraction = 0\n", "book.top_fraction"),
        (MINIMAL + "[book]\ntop_fraction = 0.6\n", "book.top_fraction"),
        (MINIMAL + "[book]\ntop_fraction = true\n", "book.top_fraction"),
        (MINIMAL + "[book]\ntop_fraction = 0.3\nexit_fraction = 0.25\n", "book.exit_fraction"),
        (MINIMAL + "[book]\nexit_fraction = 0.6\n", "book.exit_fraction"),
        (MINIMAL + '[book]\ndirection = "long_only"\nbottom_fraction = 0.2\n', "book.bottom_fraction"),
        (MINIMAL + '[book]\ndirection = "short_only"\n', "book.direction"),
        (MINIMAL + '[book]\nrebalance = "daily"\n', "book.rebalance"),
        (MINIMAL + '[costs]\nscenario = "taker"\nfee_bps = 3\n', "costs.fee_bps"),
        (MINIMAL + '[costs]\nscenario = "custom"\nslippage_bps = 1\n', "costs.fee_bps"),
        (MINIMAL + '[costs]\nscenario = "custom"\nfee_bps = 101\nslippage_bps = 0\n', "costs.fee_bps"),
        (MINIMAL + '[costs]\nscenario = "custom"\nfee_bps = 1\nslippage_bps = -1\n', "costs.slippage_bps"),
        (MINIMAL + '[costs]\nscenario = "vip"\n', "costs.scenario"),
        (MINIMAL + '[control]\nkind = "btc"\n', "control.kind"),
        (MINIMAL + "[forward]\nsigma_floor_fallback = 0\n", "forward.sigma_floor_fallback"),
        (MINIMAL + "[forward]\nsigma_floor_fallback = 1.5\n", "forward.sigma_floor_fallback"),
        (MINIMAL + "book = 3\n", "[book]"),
        (MINIMAL.replace('"x"', '"x\\n"'), "name"),
        (MINIMAL + "[book]\ntop_fraction = 1" + "0" * 400 + "\n", "book.top_fraction"),
        (MINIMAL + "[book]\nexit_fraction = 1" + "0" * 400 + "\n", "book.exit_fraction"),
        (MINIMAL + '[costs]\nscenario = "taker"\nfee_bps = 1' + "0" * 400 + "\n", "costs.fee_bps"),
        (
            MINIMAL + '[costs]\nscenario = "custom"\nfee_bps = 1\nslippage_bps = 1' + "0" * 400 + "\n",
            "costs.slippage_bps",
        ),
        (MINIMAL + "[forward]\nsigma_floor_fallback = 1" + "0" * 400 + "\n", "forward.sigma_floor_fallback"),
    ]
    for text, word in table:
        try:
            spec_of(text)
        except book.SpecError as exc:
            assert word in str(exc), (word, str(exc))
            assert "spec.toml" in str(exc), str(exc)
        else:
            raise AssertionError(f"accepted: {text!r}")
    # a file of more than 64 KiB, a TOML syntax error (with its line) and a file that does not exist
    for text, word in ((MINIMAL + "# x\n" * 20_000, "65,536 bytes"), (MINIMAL + "name = \n", "line 4")):
        try:
            spec_of(text)
        except book.SpecError as exc:
            assert word in str(exc), (word, str(exc))
        else:
            raise AssertionError(f"accepted: {word}")
    try:
        book.load_spec(TMP / "absent.toml")
    except book.SpecError as exc:
        assert "absent.toml" in str(exc)
    else:
        raise AssertionError("a file that does not exist was accepted")
    # the custom costs need both values; the maker values are accepted when they are written
    custom = spec_of(MINIMAL + '[costs]\nscenario = "custom"\nfee_bps = 1.5\nslippage_bps = 0\n')
    assert (custom.fee_bps, custom.slippage_bps) == (1.5, 0.0)
    maker = spec_of(MINIMAL + '[costs]\nscenario = "maker"\nfee_bps = 2\nslippage_bps = 2.0\n')
    assert (maker.fee_bps, maker.slippage_bps) == (2.0, 2.0)


def test_user_changes_are_honoured():
    base = book.load_spec()
    prices, funding, eligible = world(seed=3, n_days=200)
    values = random_values(prices, seed=4)
    ref = book.weekly_table(run(base, prices, funding, eligible, values))
    changes = {
        "direction": ('[book]\ndirection = "long_only"\n', True),
        "sign": ("[book]\nsign = -1\n", True),
        "top": ("[book]\ntop_fraction = 0.3\n", True),
        "bottom": ("[book]\nbottom_fraction = 0.1\n", True),
        "exit": ("[book]\nexit_fraction = 0.5\n", True),
        "maker": ('[costs]\nscenario = "maker"\n', True),
        "custom": ('[costs]\nscenario = "custom"\nfee_bps = 1\nslippage_bps = 0\n', True),
        "control": ('[control]\nkind = "equal_weight"\n', True),
        "floor": ("[forward]\nsigma_floor_fallback = 0.05\n", False),
    }
    shas = {base.sha256}
    for key, (extra, other_run) in changes.items():
        s = spec_of('format = 1\nname = "long-short"\nversion = 1\n' + extra)
        assert s.sha256 not in shas, key
        shas.add(s.sha256)
        wk = book.weekly_table(run(s, prices, funding, eligible, values))
        differs = not np.allclose(wk["excess"], ref["excess"], rtol=0, atol=1e-15)
        assert differs is other_run, key
    floor = spec_of('format = 1\nname = "long-short"\nversion = 1\n[forward]\nsigma_floor_fallback = 0.05\n')
    assert floor.forward.sigma_floor_fallback == 0.05
    for extra in ('format = 1\nname = "other"\nversion = 1\n', 'format = 1\nname = "long-short"\nversion = 2\n'):
        s = spec_of(extra)
        assert s.sha256 not in shas
        shas.add(s.sha256)


def test_book_key_ignores_name_version_and_sign():
    """book_key is the same without name, version, book.sign, costs.scenario (the book uses only
    the two cost numbers) and the table [forward] (a setting of the forward test, not of the book)."""
    base = book.load_spec()
    head = 'format = 1\nname = "long-short"\nversion = 1\n'
    custom = head + '[costs]\nscenario = "custom"\nfee_bps = {}\nslippage_bps = {}\n'
    same = [
        spec_of('format = 1\nname = "renamed"\nversion = 7\n'),
        spec_of(head + "[book]\nsign = -1\n"),
        spec_of(head + "[forward]\nsigma_floor_fallback = 0.05\n"),
        spec_of(custom.format(5, "5.0")),  # the taker numbers under the name "custom"
    ]
    for s in same:
        assert s.book_key == base.book_key and s.sha256 != base.sha256
    maker = spec_of(head + '[costs]\nscenario = "maker"\n')
    custom_maker = spec_of(custom.format("2.0", 2))  # the maker numbers under the name "custom"
    assert custom_maker.book_key == maker.book_key != base.book_key and custom_maker.sha256 != maker.sha256
    # one form of zero: -0.0 and 0.0 give the same body
    assert spec_of(custom.format("-0.0", 0)).body == spec_of(custom.format(0, "0.0")).body
    others = [
        spec_of(head + "[book]\ntop_fraction = 0.3\n"),
        spec_of(custom.format(5, 4)),
        spec_of(custom.format(4, 5)),
        book.load_spec("long_only"),
    ]
    keys = {base.book_key, maker.book_key, *(o.book_key for o in others)}
    assert len(keys) == 2 + len(others), "two books with other numbers have the same book_key"
    body = {k: v for k, v in base.body.items() if k not in ("name", "version", "forward")}
    body["book"] = {k: v for k, v in base.body["book"].items() if k != "sign"}
    body["costs"] = {"fee_bps": 5.0, "slippage_bps": 5.0}
    assert base.book_key == hashlib.sha256(ledger.canonical(body).encode("ascii")).hexdigest()


# --------------------------------------------------------------------------- 3. BandedBook


def test_banded_book_equals_banded_quintile():
    bq = banded_quintile()
    rng = np.random.default_rng(5)
    for long_only in (False, True):
        for flat in ("cash", "equal_weight"):
            b = book.BandedBook(lambda cs: cs["value"], long_only=long_only, top=0.2, bottom=0.2, exit=0.4, flat=flat)
            a = bq(lambda cs: cs["value"], long_only=long_only)
            compared = below = 0
            for _ in range(300):
                n = int(rng.integers(5, 21))
                names = list(rng.choice(SYMS, size=n, replace=False))
                v = np.round(rng.normal(0.0, 1.0, n), 1)  # ties
                v[rng.random(n) < rng.uniform(0.0, 0.5)] = np.nan
                cs = pd.DataFrame({"value": v}, index=pd.Index(names, name="symbol"))
                wa, wb = a(cs), b(cs)
                if np.isfinite(v).sum() < book.MIN_NAMES:
                    below += 1
                    if flat == "equal_weight":
                        assert np.allclose(wb.to_numpy(), 1.0 / n) and len(wb) == n
                        continue
                assert wa.sort_index().equals(wb.sort_index()), (wa, wb)
                compared += 1
            assert compared > 200 and below > 5, (compared, below)
    # four demo formulas: every Sunday of the demo market, long-short and long-only
    el = demo()["eligible"]
    for expr in FORMULAS:
        ff = book.factor_frame(demo_values(expr), el)
        for long_only in (False, True):
            a = bq(lambda cs: cs["value"], long_only=long_only)
            b = book.BandedBook(lambda cs: cs["value"], long_only=long_only, top=0.2, bottom=0.2, exit=0.4)
            n = 0
            for _, g in ff.groupby("date"):
                cs = g.set_index("symbol")
                assert a(cs).sort_index().equals(b(cs).sort_index()), (expr, long_only)
                n += 1
            assert n > 200, n
    # the counts of the two constructions agree for every size of the cross-section
    for n in range(5, 21):
        assert max(2, n // 5) == max(book.MIN_PER_SIDE, math.floor(n * 0.2 + 1e-9)), n


def cs_of(values: dict) -> pd.DataFrame:
    return pd.DataFrame({"value": list(values.values())}, index=pd.Index(list(values), name="symbol"))


def test_fractions_are_honoured():
    v = {s: float(i) for i, s in enumerate(SYMS)}  # 20 names, no ties
    order = sorted(v, key=v.get)
    for top, bottom, n_long, n_short in (
        (0.10, 0.10, 2, 2),
        (0.30, 0.30, 6, 6),
        (0.30, 0.10, 6, 2),
        (0.5, 0.25, 10, 5),
    ):
        b = book.BandedBook(lambda cs: cs["value"], long_only=False, top=top, bottom=bottom, exit=0.5)
        w = b(cs_of(v))
        assert set(w[w > 0].index) == set(order[-n_long:]) and set(w[w < 0].index) == set(order[:n_short]), top
        assert np.isclose(w[w > 0].sum(), 0.5) and np.isclose(w[w < 0].sum(), -0.5)
    lo = book.BandedBook(lambda cs: cs["value"], long_only=True, top=0.30, bottom=None, exit=0.4)
    w = lo(cs_of(v))
    assert set(w[w > 0].index) == set(order[-6:]) and np.isclose(w.sum(), 1.0) and (w >= 0).all()
    # a specification with other fractions gives the same book as a BandedBook with them
    s = spec_of(MINIMAL + "[book]\ntop_fraction = 0.3\nbottom_fraction = 0.1\nexit_fraction = 0.45\n")
    f = book.book_weights_fn(s)
    assert (f.top, f.bottom, f.exit, f.long_only, f.flat) == (0.3, 0.1, 0.45, False, "cash")
    w = f(cs_of(v))
    assert (w > 0).sum() == 6 and (w < 0).sum() == 2


def test_exit_band_keeps_a_held_name():
    b = book.BandedBook(lambda cs: cs["value"], long_only=True, top=0.2, bottom=None, exit=0.4)
    v = {s: float(i) for i, s in enumerate(SYMS[:10])}  # 10 names: top 2 long, exit band top 4
    w1 = b(cs_of(v))
    held = set(w1[w1 > 0].index)
    assert held == set(SYMS[8:10])
    # SYMS[9] falls to rank 7 of 10 and SYMS[8] to rank 8 (inside the top 40 %): both stay, next to
    # the new top 2
    v2 = dict(v)
    v2[SYMS[9]], v2[SYMS[6]], v2[SYMS[7]] = 6.5, 9.5, 9.0
    w2 = b(cs_of(v2))
    assert set(w2[w2 > 0].index) == {SYMS[6], SYMS[7], SYMS[8], SYMS[9]}, set(w2[w2 > 0].index)
    # SYMS[9] falls below the band: it leaves
    v3 = dict(v2)
    v3[SYMS[9]] = -1.0
    w3 = b(cs_of(v3))
    assert SYMS[9] not in set(w3[w3 > 0].index)
    # a new instance has no memory (book_weights_fn gives a new one for each run)
    f1, f2 = book.book_weights_fn(book.load_spec()), book.book_weights_fn(book.load_spec())
    assert f1 is not f2 and not f1.longs and not f2.longs


def test_sign_minus_one_mirrors_the_book():
    prices, funding, eligible = world(seed=7, n_days=200)
    values = random_values(prices, seed=8)
    neg = spec_of('format = 1\nname = "long-short"\nversion = 1\n[book]\nsign = -1\n')
    a = run(neg, prices, funding, eligible, values)
    b = run(book.load_spec(), prices, funding, eligible, -values)
    same_result(a.book, b.book)
    # at the first rebalance (no held names yet) the long and short sides swap
    w_pos = book.book_weights_fn(book.load_spec())(cs_of({s: float(i) for i, s in enumerate(SYMS)}))
    w_neg = book.book_weights_fn(neg)(cs_of({s: float(i) for i, s in enumerate(SYMS)}))
    assert set(w_pos[w_pos > 0].index) == set(w_neg[w_neg < 0].index)
    assert set(w_pos[w_pos < 0].index) == set(w_neg[w_neg > 0].index)


# --------------------------------------------------------------------------- 5. and 6. the control


def test_fewer_than_min_names_holds_the_control():
    prices, funding, eligible = world(seed=9, n_days=150)
    values = random_values(prices, seed=10)
    s = sundays_of(prices)
    thin = {s[3]: 0, s[5]: 1, s[7]: 4, s[9]: 2}  # number of values kept at these Sundays
    values = values.drop(index=s[3])  # no row at all for s[3]
    for d, k in thin.items():
        if d in values.index:
            values.loc[d, SYMS[k:]] = np.nan
    # no eligible coin at s[11] (its values and prices are there): the engine gets one NO_COIN row
    eligible = eligible.copy()
    eligible.loc[s[11]] = False
    none = s[11]
    assert none in values.index and values.loc[none].notna().all()
    want_missed = sorted(d.strftime("%Y-%m-%d") for d in [*thin, none])
    # control "zero": the book holds no position in those weeks
    r = run(book.load_spec(), prices, funding, eligible, values)
    assert r.missed == want_missed, r.missed
    wk = book.weekly_table(r)
    gross = r.book.daily["gross"]
    for d in [*thin, none]:
        week = gross[(gross.index > d) & (gross.index <= d + pd.Timedelta(days=7))]
        assert (week == 0.0).all(), d
        cost = wk.loc[d, "book_cost"] / r.book.equity["equity"][d]
        assert abs(wk.loc[d, "excess"] + cost) < 1e-12, "the week holds cash: its return is the cost of the switch"
    assert list(wk.index[~wk["held"]].strftime("%Y-%m-%d")) == want_missed
    # control "equal_weight": the book holds the equal weights of the eligible coins in those weeks
    lo = book.load_spec("long_only")
    r = run(lo, prices, funding, eligible, values)
    wk = book.weekly_table(r)
    assert r.missed == want_missed
    for d in thin:
        monday = d + pd.Timedelta(days=1)
        t = r.book.trades[r.book.trades["date"] == monday]
        assert np.allclose(t["target_w"], 1.0 / len(SYMS)) and len(t) > 0, d
        e = r.book.equity["equity"][d]
        ec = r.control.equity["equity"][d]
        c_book = wk.loc[d, "book_cost"] / e
        c_ctrl = r.control.daily["cost"][monday] / ec
        # the excess of such a week is the cost difference of the switch only (no market return)
        assert abs(wk.loc[d, "excess"]) <= 2 * (c_book + c_ctrl) + 1e-12, (d, wk.loc[d, "excess"])
    # no eligible coin: the book and the control both go to no position for the week
    monday = none + pd.Timedelta(days=1)
    for res in (r.book, r.control):
        g = res.daily["gross"]
        assert (g[(g.index > none) & (g.index <= none + pd.Timedelta(days=7))] == 0.0).all()
        t = res.trades[res.trades["date"] == monday]
        assert len(t) > 0 and (t["target_w"] == 0.0).all() and book.NO_COIN not in set(t["symbol"])
    c_book = wk.loc[none, "book_cost"] / r.book.equity["equity"][none]
    c_ctrl = r.control.daily["cost"][monday] / r.control.equity["equity"][none]
    assert abs(wk.loc[none, "excess"] - (c_ctrl - c_book)) < 1e-12, wk.loc[none, "excess"]
    assert not wk.loc[none, "held"] and wk.loc[s[12], "held"]
    # BandedBook alone: 0 values and 1 to 4 values
    for k in range(5):
        cs = cs_of({sym: (float(i) if i < k else np.nan) for i, sym in enumerate(SYMS[:8])})
        cash = book.BandedBook(lambda c: c["value"], long_only=False, top=0.2, bottom=0.2, exit=0.4)
        ew = book.BandedBook(lambda c: c["value"], long_only=True, top=0.2, bottom=None, exit=0.4, flat="equal_weight")
        assert cash(cs).empty, k
        w = ew(cs)
        assert list(w.index) == SYMS[:8] and np.allclose(w, 1 / 8), k
        assert w.equals(book.equal_weight(cs))
    # only the row NO_COIN: no position with each control
    cs = cs_of({book.NO_COIN: np.nan})
    assert book.equal_weight(cs).empty and book.book_weights_fn(lo)(cs).empty
    assert book.book_weights_fn(book.load_spec())(cs).empty


def test_only_eligible_names_get_weight():
    import labels

    demo()
    el = labels.eligible()
    late = [C.sym_to_file(s) for s in ("OP/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT")]
    prices = data_io.close_matrix()
    s = sundays_of(prices)
    s = s[(s >= pd.Timestamp("2022-04-03", tz="UTC")) & (s <= pd.Timestamp("2023-12-31", tz="UTC"))]
    values = pd.DataFrame(np.random.default_rng(11).normal(0, 1, (len(s), len(SYMS))), index=s, columns=SYMS)
    values[late] = 100.0  # the late coins have the largest values: a book that ignores eligibility buys them
    for spec in (book.load_spec(), book.load_spec("long_only")):
        r = book.run_book(spec, values, start=s[0], end=s[-1])
        bought = r.book.trades[r.book.trades["side"] == "buy"]
        for _, t in bought.iterrows():
            assert bool(el.loc[t["date"] - pd.Timedelta(days=1), t["symbol"]]), (t["date"], t["symbol"])
        first = {sym: el.index[el[sym]].min() for sym in late}
        for sym in late:
            assert (bought[bought["symbol"] == sym]["date"] > first[sym]).all(), sym
        assert any((bought["symbol"] == sym).any() for sym in late), "the late coins must enter once eligible"


def test_control_universe_does_not_depend_on_factor_coverage():
    prices, funding, eligible = world(seed=12, n_days=200)
    full = random_values(prices, seed=13)
    sparse = full.copy()
    sparse.iloc[:, 5:] = np.nan  # the factor covers 5 coins only
    sparse.iloc[4, :] = np.nan  # and no coin at one Sunday
    lo = book.load_spec("long_only")
    a = run(lo, prices, funding, eligible, full)
    b = run(lo, prices, funding, eligible, sparse)
    same_result(a.control, b.control)
    assert not a.book.equity["equity"].equals(b.book.equity["equity"])
    uf = book.universe_frame(eligible, sundays_of(prices)[:-1])
    assert len(uf) == len(SYMS) * (len(sundays_of(prices)) - 1) and list(uf.columns) == ["date", "symbol"]


def test_missed_rebalance_is_reported():
    prices, funding, eligible = world(seed=14, n_days=120)
    values = random_values(prices, seed=15)
    s = sundays_of(prices)
    values.loc[s[2]] = np.nan
    values.loc[s[6], SYMS[3:]] = np.inf  # not finite: missing
    r = run(book.load_spec(), prices, funding, eligible, values)
    assert r.missed == [s[2].strftime("%Y-%m-%d"), s[6].strftime("%Y-%m-%d")], r.missed
    wk = book.weekly_table(r)
    assert list(wk["held"]) == [d not in (s[2], s[6]) for d in wk.index]
    assert wk["rebalanced"].all(), "the weekly book trades every Monday (also into cash)"
    ff = book.factor_frame(values, eligible)
    assert len(ff) == len(values) * len(SYMS) and ff["value"].isna().sum() == len(SYMS) + len(SYMS) - 3


# --------------------------------------------------------------------------- 7. weekly series


def test_weekly_returns_compound_the_daily_marks():
    prices, funding, eligible = world(seed=16, n_days=120)
    r = run(book.load_spec(), prices, funding, eligible, random_values(prices, seed=17))
    eq = r.book.equity["equity"]
    wr = book.weekly_returns(eq, r.start, r.end)
    dr = eq.pct_change()
    for sun, x in wr.items():
        days = dr[(dr.index > sun) & (dr.index <= sun + pd.Timedelta(days=7))]
        assert len(days) == 7 and abs((1 + days).prod() - 1 - x) < 1e-12, sun
        assert x == eq[sun + pd.Timedelta(days=7)] / eq[sun] - 1
    assert list(wr.index) == list(pd.date_range(r.start, r.end - pd.Timedelta(days=7), freq="7D"))
    try:
        book.weekly_returns(eq.drop(index=r.start + pd.Timedelta(days=14)), r.start, r.end)
    except ValueError:
        pass
    else:
        raise AssertionError("an absent Sunday row was accepted")
    bust = pd.Series([100.0, 50.0, -1.0, 0.0, 0.0], index=pd.date_range("2023-01-01", periods=5, freq="7D", tz="UTC"))
    assert list(book.weekly_returns(bust, bust.index[0], bust.index[-1])) == [-0.5, -1.02, 0.0, 0.0]
    # a bust: a short coin goes up 100 times; the run gets equity 0.0 up to the end, and the weeks after
    # the bust have the return 0.0
    flat = pd.DataFrame(100.0, index=prices.index[:60], columns=SYMS)
    flat.loc[flat.index >= pd.Timestamp("2023-01-11", tz="UTC"), SYMS[0]] = 10_000.0
    v = pd.DataFrame([[float(i) for i in range(len(SYMS))]] * 9, index=sundays_of(flat)[:9], columns=SYMS)
    import contextlib
    import io
    import warnings

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", RuntimeWarning)  # the CAGR of a negative equity
        br = book.run_book(
            book.load_spec(), v, start=v.index[0], end=v.index[-1], prices=flat, funding=0.0 * flat, eligible=eligible
        )
    assert br.book.equity.index.equals(flat.index[(flat.index >= br.start) & (flat.index <= br.end)])
    e = br.book.equity["equity"]
    assert e.min() < 0 or (e == 0.0).any(), "the fixture must bust"
    after = e[e.index > e.index[(e <= 0).to_numpy().argmax()]]
    assert (after == 0.0).all() and len(after) > 7
    wk = book.weekly_table(br)
    assert len(wk) == 8 and wk["book_ret"].iloc[1] == -1.0 and (wk["book_ret"].iloc[2:] == 0.0).all()
    for bad in (("2023-01-02", "2023-01-15"), ("2023-01-01", "2023-01-07")):
        try:
            book.run_book(book.load_spec(), pd.DataFrame(), start=bad[0], end=bad[1], prices=prices)
        except ValueError:
            continue
        raise AssertionError(f"run_book{bad} was accepted")


def test_zero_control_excess_is_the_book_return():
    prices, funding, eligible = world(seed=18, n_days=160)
    r = run(book.load_spec(), prices, funding, eligible, random_values(prices, seed=19))
    wk = book.weekly_table(r)
    assert r.control is None and (wk["control_ret"] == 0.0).all() and wk["control_equity"].isna().all()
    assert wk["excess"].equals(wk["book_ret"])
    assert list(wk.columns) == list(book.WEEKLY_COLUMNS)
    d = book.daily_excess(r)
    assert d.equals(r.book.equity["equity"].pct_change().iloc[1:])
    s = wk.index[3]
    assert wk.loc[s, "book_equity"] == r.book.equity["equity"][s + pd.Timedelta(days=7)]
    bench = r.book.equity["benchmark"]
    assert wk.loc[s, "btc_ret"] == bench[s + pd.Timedelta(days=7)] / bench[s] - 1
    t = r.book.trades
    monday = s + pd.Timedelta(days=1)
    assert np.isclose(wk.loc[s, "book_turnover"], t[t["date"] == monday]["notional"].sum() / r.book.equity["equity"][s])
    week = r.book.daily[(r.book.daily.index > s) & (r.book.daily.index <= s + pd.Timedelta(days=7))]
    assert np.isclose(wk.loc[s, "book_cost"], week["cost"].sum()) and np.isclose(
        wk.loc[s, "book_funding_usd"], week["funding_usd"].sum()
    )


def test_equal_weight_control_is_the_demo_market_book():
    prices, funding, eligible = world(seed=20, n_days=160)
    eligible.iloc[:40, :4] = False  # four coins enter later
    r = run(book.load_spec("long_only"), prices, funding, eligible, random_values(prices, seed=21))
    s = sundays_of(prices)
    direct = B.run_backtest(
        B.demo_market,
        prices=prices,
        funding=funding,
        factors=book.universe_frame(eligible, s[:-1]),
        start=str(s[0].date()),
        end=str(s[-1].date()),
        initial_equity=book.INITIAL_EQUITY,
        fee_bps=5.0,
        slippage_bps=5.0,
        benchmark="BTC-USDT",
        accounting=B.ACCOUNTING_RECONCILED,
    )
    same_result(r.control, direct)
    wk = book.weekly_table(r)
    want = book.weekly_returns(direct.equity["equity"], s[0], s[-1])
    assert wk["control_ret"].equals(want) and np.allclose(wk["excess"], wk["book_ret"] - want, rtol=0, atol=0)
    first = r.control.trades[r.control.trades["date"] == s[0] + pd.Timedelta(days=1)]
    assert len(first) == len(SYMS) - 4 and np.allclose(first["target_w"], 1 / (len(SYMS) - 4))


# --------------------------------------------------------------------------- 8. the schedule


def mondays(prices: pd.DataFrame) -> dict:
    return {t: t - pd.Timedelta(days=1) for t in prices.index if t.weekday() == 0}


# Pinned results of the weekly rule (schedule=None) on two worlds of test_accounting.py
# (random_world with the seeds 3 and 5, fee and slippage 5 bps each, the random funding), for
# the rebalance weekdays 0, 2 and 6 and the accounting convention: the first 16 digits of the
# sha256 of the trade lines "date|symbol|side", the number of trade lines, the end equity and
# the fees.
SCHEDULE_NONE_PINS = {
    (3, 0, "reconciled-v2"): ("9bfc9069021ac37d", 54, 8454.930684328008, 144.13100660506984),
    (3, 2, "reconciled-v2"): ("1fc9ab735660b9f6", 59, 10194.423726660312, 158.14011293282127),
    (3, 6, "reconciled-v2"): ("14d5244612f7f7cf", 51, 8026.107109373469, 143.2255573033148),
    (5, 0, "reconciled-v2"): ("6f8b5050a6dd45a4", 57, 5975.605941499881, 119.46740682287484),
    (5, 2, "reconciled-v2"): ("3ddb97cd605222e4", 56, 13050.472819949631, 161.11362295727486),
    (5, 6, "reconciled-v2"): ("93658fb73f5cf1ac", 53, 10559.178743896735, 140.40803758263738),
}


def trade_lines_sha(res: B.Result) -> str:
    t = res.trades
    text = "\n".join(f"{d.date()}|{s}|{side}" for d, s, side in zip(t["date"], t["symbol"], t["side"]))
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:16]


def test_schedule_none_is_bit_identical():
    """schedule=None runs the weekly rule of the engine: the tests of test_backtest.py and
    test_accounting.py pass with it, and each of their worlds gives the same result. For the
    rebalance weekdays 0, 2 and 6, it trades only on that weekday, it gives the pinned numbers
    of SCHEDULE_NONE_PINS, and the schedule {t: t - 1 day} of each such day t gives the same
    result."""
    import test_accounting as TA
    import test_backtest as TB

    def with_schedule(weights_fn, **kw):
        return B.run_backtest(weights_fn, schedule=None, **kw)

    for mod in (TB, TA):
        saved = mod.run_backtest
        mod.run_backtest = with_schedule
        try:
            for name, fn in sorted(vars(mod).items()):
                if name.startswith("test_") and callable(fn):
                    fn()
        finally:
            mod.run_backtest = saved
    px, fund, f = TA.random_world(seed=3)
    for acc in B.ACCOUNTING_MODES:
        same_result(
            TA.run(px, f, funding=fund, accounting=acc), TA.run(px, f, funding=fund, accounting=acc, schedule=None)
        )
    assert {k[2] for k in SCHEDULE_NONE_PINS} == set(B.ACCOUNTING_MODES)
    for (seed, wd, acc), (sha, n, equity, fees) in SCHEDULE_NONE_PINS.items():
        px, fund, f = TA.random_world(seed=seed)
        kw = {"funding": fund, "fee_bps": 5.0, "slippage_bps": 5.0, "accounting": acc, "rebalance_weekday": wd}
        res = TA.run(px, f, schedule=None, **kw)
        assert len(res.trades) and all(d.weekday() == wd for d in res.trades["date"]), (seed, wd)
        assert (trade_lines_sha(res), len(res.trades)) == (sha, n), (seed, wd, acc, trade_lines_sha(res))
        assert math.isclose(float(res.equity["equity"].iloc[-1]), equity, rel_tol=1e-9, abs_tol=0.0), (seed, wd, acc)
        assert math.isclose(float(res.stats["fees_total"]), fees, rel_tol=1e-9, abs_tol=0.0), (seed, wd, acc)
        days = {t: t - pd.Timedelta(days=1) for t in px.index if t.weekday() == wd}
        same_result(res, TA.run(px, f, schedule=days, **kw))


def test_schedule_monday_mapping_is_bit_identical():
    """A schedule {Monday: Sunday} of every Monday gives the result of the weekly rule."""
    import test_accounting as TA
    import test_backtest as TB

    def with_mondays(weights_fn, *, prices, **kw):
        return B.run_backtest(weights_fn, prices=prices, schedule=mondays(prices), **kw)

    for mod in (TB, TA):
        saved = mod.run_backtest
        mod.run_backtest = with_mondays
        try:
            for name, fn in sorted(vars(mod).items()):
                if name.startswith("test_") and callable(fn):
                    fn()
        finally:
            mod.run_backtest = saved
    for seed in (3, 5, 11):
        px, fund, f = TA.random_world(seed=seed)
        for acc in B.ACCOUNTING_MODES:
            same_result(
                TA.run(px, f, funding=fund, fee_bps=5.0, slippage_bps=5.0, accounting=acc),
                TA.run(px, f, funding=fund, fee_bps=5.0, slippage_bps=5.0, accounting=acc, schedule=mondays(px)),
            )
    idx = TB.make_index(n=60)
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.03, (60, 3)), axis=0)), index=idx, columns=TB.SYMS
    )
    f = TB.factor_table(idx, TB.SYMS, lambda d, s: float((d.day * 7 + ord(s[0])) % 3 - 1))
    same_result(TB.run(px, f), TB.run(px, f, schedule=mondays(px)))
    # a book run with a Monday schedule is the book run without one
    prices, funding, eligible = world(seed=22, n_days=120)
    values = random_values(prices, seed=23)
    a = run(book.load_spec(), prices, funding, eligible, values)
    b = run(book.load_spec(), prices, funding, eligible, values, schedule=mondays(prices))
    same_result(a.book, b.book)


def test_schedule_trades_on_the_given_day():
    import test_backtest as TB

    idx = TB.make_index(n=29)  # Sunday 2023-01-01 .. Sunday 2023-01-29
    px = TB.flat_prices(idx)
    px.loc[idx >= pd.Timestamp("2023-01-03", tz="UTC"), "AAA"] = 110.0  # AAA moves on Tuesday 01-03
    sunday = pd.Timestamp("2023-01-01", tz="UTC")
    monday = sunday + pd.Timedelta(days=1)
    # the Sunday rows say AAA, then BBB; every other row says CCC (a trade on another row buys CCC)
    second = sunday + pd.Timedelta(days=7)

    def sig(d, s):
        if d.weekday() == 6:
            return s == ("AAA" if d < second else "BBB")
        return s == "CCC"

    f = TB.factor_table(idx, TB.SYMS, sig)
    tuesday = sunday + pd.Timedelta(days=2)
    sched = {tuesday: sunday, tuesday + pd.Timedelta(days=7): sunday + pd.Timedelta(days=7)}
    r = TB.run(px, f, schedule=sched)
    assert set(r.trades["date"]) == set(sched), sorted(set(r.trades["date"]))
    first = r.trades[r.trades["date"] == tuesday]
    assert list(first["symbol"]) == ["AAA"] and first["ref_price"].iloc[0] == px.loc[monday, "AAA"]
    assert "CCC" not in set(r.trades["symbol"]) and set(r.trades["symbol"]) == {"AAA", "BBB"}
    # no position during the Monday bar; the trade at Tuesday 00:00 (at the Monday close) earns the Tuesday bar
    assert r.equity["equity"][monday] == 10_000.0 and abs(r.equity["equity"][tuesday] - 11_000.0) < 1e-9
    # without a schedule the book trades on the Mondays only
    plain = TB.run(px, f)
    assert all(d.weekday() == 0 for d in plain.trades["date"])
    for bad in (
        {pd.Timestamp("2023-01-03"): pd.Timestamp("2023-01-01")},  # no time zone
        {tuesday: sunday + pd.Timedelta(hours=1)},  # not midnight
        {tuesday: tuesday},  # the signal is not before the trade
        {tuesday.tz_convert("Asia/Tokyo"): sunday},  # another time zone
        {tuesday: "2023-01-01"},
    ):
        try:
            TB.run(px, f, schedule=bad)
        except ValueError:
            continue
        raise AssertionError(f"schedule {bad} was accepted")


def test_trade_log_reconciles():
    import test_accounting as TA

    prices, funding, eligible = world(seed=24, n_days=200)
    values = random_values(prices, seed=25)
    for name in ("long_short", "long_only"):
        spec = book.load_spec(name)
        r = run(spec, prices, funding, eligible, values)
        window = prices[(prices.index >= r.start) & (prices.index <= r.end)]
        rate = (spec.fee_bps + spec.slippage_bps) / 1e4
        for res in [r.book] + ([r.control] if r.control is not None else []):
            assert res.stats.get("accounting") == B.ACCOUNTING_RECONCILED
            TA.assert_reconciles(res, window, funding.loc[window.index], rate, book.INITIAL_EQUITY)


def test_cost_scenarios():
    assert book.COST_SCENARIOS == {"taker": (5.0, 5.0), "maker": (2.0, 2.0)}
    prices, funding, eligible = world(seed=26, n_days=160)
    values = random_values(prices, seed=27)
    fees = {}
    for key, extra, want in (
        ("taker", '[costs]\nscenario = "taker"\n', 10.0),
        ("maker", '[costs]\nscenario = "maker"\n', 4.0),
        ("custom", '[costs]\nscenario = "custom"\nfee_bps = 1.5\nslippage_bps = 0.25\n', 1.75),
        ("free", '[costs]\nscenario = "custom"\nfee_bps = 0\nslippage_bps = 0\n', 0.0),
    ):
        s = spec_of(MINIMAL + extra)
        assert s.fee_bps + s.slippage_bps == want, key
        r = run(s, prices, funding, eligible, values)
        t = r.book.trades
        assert np.allclose(t["cost"], t["notional"] * want / 1e4, rtol=1e-12, atol=1e-12), key
        fees[key] = r.book.stats["fees_total"]
    assert fees["taker"] > fees["maker"] > fees["custom"] > fees["free"] == 0.0, fees


# --- runner -------------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001 - report every test, then fail at the end
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} book tests passed")
    raise SystemExit(1 if failed else 0)
