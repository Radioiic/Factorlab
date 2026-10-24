"""Integrity tests for libfactors.py (library factors on Factorlab's data).

Script-style like the other engine suites:  python -B test_libfactors.py

Reads the data folder read-only (raw bars, funding, open interest, the label table and
the library catalog) and checks the outputs in data/library_on_engine/ when they
exist. The only writes go to temporary folders under TMPDIR (point it inside the
git-ignored engine/.tmp/). Without the raw bars, the label table or the catalog, every
test is skipped (exit 0); a test that needs the git-ignored values/ is skipped without
them.

  panel units          stored perp volume is in base-asset units (the perp/spot ratio
                       band, no power-of-ten jumps); quote_volume = close x volume;
                       vwap therefore collapses onto close, which is why vwap factors
                       are flagged and kept out of the screen
  panel layout         daily UTC open-time index to the last closed bar; every field
                       NaN before a perp's first bar; funding and OI are data_io's
                       series; no premium, trades, taker or long/short field
  classification       28,904 rows reconcile: 17,758 computable (3,311 need OI),
                       1,544 need only the premium index, 9,602 need fields the
                       engine's OKX data does not have
  open-interest stamp  every stored OKX open-interest print is stamped 16:00 UTC and daily_oi books
                       it on that UTC day (the point-in-time assumption libfactors
                       documents); a change in OKX's stamping fails here
  future mutation      every bar, funding and OI value after a cut overwritten with
                       noise: values on dates <= cut unchanged, for a sample covering
                       every library operator and every supplied field, at the three
                       PIT cuts
  PIT status           'pass' needs at least one finite compared cell; an all-NaN
                       factor is 'no_finite_cells', a lookahead factor 'fail'; an OI
                       factor whose values start between two cuts is compared at the
                       later cut (cuts taken from the open-interest history on disk)
  sealed I zone        a label or bar after the 2024-06-09 row raises LabelLeak; with
                       every label and bar after the cutoff mutated, the whole screen
                       output is byte-identical (and a change before it is not)
  statistics           weekly IC and rIC equal an independent per-Sunday reference
                       (scipy Spearman, a closed-form ridge, the registered regressors
                       rebuilt from the raw files); t, p and BH against scipy and a
                       hand-computed vector; dedup merges a factor with its sign flip
  registered settings  the tilt, beta, size, ridge and name floor are asserted equal to
                       factorfinder/eval_config.toml (a changed value stops the screen);
                       the tilt is point in time and shrunk n / (n + 180)
  G_fit selection      the G_fit q-values never change when every G_eval label does
                       (G_eval is the check, never the selection)
  full precision       the shards are float64 and the screen ranks them: a float32 copy
                       changes a factor's statistics, a few-ulp perturbation does not
                       (snap_ties), and a float32 shard is refused
  determinism          one worker and three workers give identical values
  writer boundary      only data/library_on_engine; a symlinked folder or a hard-linked
                       target is refused; a protected target is refused first
  outputs              status / screen_g / manifest reconcile with each other, carry
                       no path or private string and no verdict column, the stored
                       values equal a fresh computation, values/ is git-ignored;
                       g_weeks is every G factor's weekly-IC count (too_few_weeks too)
  numbers only         the outputs, the manifest texts and --status have no verdict, tier,
                       stage or gate word; --status prints counts and the states of the
                       code and the inputs only
  code state           --status names each code file whose hash differs from the manifest's
  missing catalog      the catalog reader stops with the setup instruction
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import data_io
import libfactors as L
from labels import path as label_path

_CACHE: dict = {}


def _base():
    """Catalog, classification, raw reads and full panels, loaded once."""
    if not _CACHE:
        cat = L.load_catalog()
        _CACHE["cat"] = cat
        _CACHE["status"] = L.classify(cat)
        _CACHE["raw"] = L.load_raw()
        _CACHE["panels"] = L.build_panels(_CACHE["raw"])
    return _CACHE["cat"], _CACHE["status"], _CACHE["raw"], _CACHE["panels"]


def _computable():
    cat, st, _, _ = _base()
    ok = st["group"].eq("computable").to_numpy()
    return cat[ok].reset_index(drop=True), st[ok].reset_index(drop=True)


def _oi_rows(panels: dict) -> int:
    """The number of panel rows from the first day with an open-interest value to the last
    bar (0 without open interest)."""
    have = panels["open_interest_value"].notna().any(axis=1).to_numpy()
    return int(len(have) - np.argmax(have)) if have.any() else 0


def _late_cut(panels: dict, days: int = 21) -> pd.Timestamp:
    """The last Sunday at least `days` days before the last bar."""
    idx = panels["close"].index
    sun = idx[L.sunday_positions(idx)]
    return sun[sun <= idx[-1] - pd.Timedelta(days=days)][-1]


def _eval_all(panels: dict, exprs: list) -> list:
    ev = L._evaluator(panels)
    return [L._values(ev, e) for e in exprs]


def _mutate_raw(raw: dict, cut: pd.Timestamp, seed: int) -> dict:
    """A copy of the raw reads with every bar, funding and OI value after `cut` replaced
    by positive noise (dates and calendar unchanged)."""
    rng = np.random.default_rng(seed)
    out = {}
    for sym, r in raw.items():
        px = r["ohlcv"].copy()
        late = px.index > cut
        px.loc[late] = rng.uniform(0.5, 2.0, size=(int(late.sum()), px.shape[1])) * px.loc[late].to_numpy()[::-1]
        f = None if r["funding"] is None else r["funding"].copy()
        if f is not None:
            f[f.index > cut] = rng.normal(0, 1e-3, size=int((f.index > cut).sum()))
        oi = None if r["oi"] is None else r["oi"].copy()
        if oi is not None:
            oi[oi.index > cut] = rng.uniform(1e8, 1e10, size=int((oi.index > cut).sum()))
        out[sym] = {"ohlcv": px, "funding": f, "oi": oi}
    return out


# Operators that no computable library factor uses: covered by these expressions, which
# exist only for the mutation test (they are not library factors).
EXTRA_EXPRESSIONS = (
    "clip(pct_change(close, 5), -0.2, 0.2)",
    "cs_demean(ret) ** 2 - ts_mean(funding_rate, 3)",
    "-ts_mean(div(open_interest, delay(open_interest, 1)), 5)",
)


def _operators():
    """The names of the library operators (libfactors puts library/ on sys.path)."""
    from factorlab.operators import OPERATORS

    return OPERATORS


def _required_atoms() -> set:
    ops = set(_operators()) | {"bench", "binop_Add", "binop_Sub", "binop_Mult", "binop_Div", "binop_Pow", "unary_USub"}
    return ops | set(L.SUPPLIED)


def _mutation_sample() -> list:
    """(id, expression) pairs: a greedy cover of every operator and supplied field by
    computable library factors (factor_id order), the EXTRA_EXPRESSIONS for what no
    library factor uses, and 40 more library factors drawn with a fixed seed."""
    comp, _ = _computable()
    need = _required_atoms()
    chosen, have = [], set()
    for fid, e in zip(comp["factor_id"], comp["expression"]):
        ops, names = L.expr_atoms(e)
        got = (ops | names) & need
        if got - have:
            chosen.append((fid, e))
            have |= got
    for i, e in enumerate(EXTRA_EXPRESSIONS):
        ops, names = L.expr_atoms(e)
        have |= (ops | names) & need
        chosen.append((f"extra_{i}", e))
    extra = comp.sample(40, random_state=11)
    chosen += [(f, e) for f, e in zip(extra["factor_id"], extra["expression"]) if f not in dict(chosen)]
    _CACHE["sample_cover"] = have
    return chosen


# --------------------------------------------------------------------------- panels


def test_volume_is_base_units_and_quote_volume_is_close_times_volume():
    """Perp volume / same-venue spot volume stays in a 1-50 band for every symbol and
    year: contract counts would put symbols whose OKX contract sizes differ by 10^5
    (0.01 BTC, 1000 DOGE) orders of magnitude apart."""
    _, _, raw, p = _base()
    ratios = {}
    for sym, r in raw.items():
        spot = data_io.spot_ohlcv(sym)
        assert spot is not None, sym
        j = r["ohlcv"]["volume"] / spot["volume"].reindex(r["ohlcv"].index)
        by_year = j.groupby(j.index.year).median()
        ratios[sym] = by_year
        assert ((by_year > 1.0) & (by_year < 50.0)).all(), (sym, by_year.round(2).to_dict())
    med = pd.concat(ratios).groupby(level=0).median()
    assert med.max() / med.min() < 20, "a power-of-ten unit break between symbols"
    btc_usd = (p["close"]["BTC-USDT"] * p["volume"]["BTC-USDT"]).median()
    assert 1e9 < btc_usd < 1e11, btc_usd  # billions of USDT a day, not millions
    assert np.allclose(p["quote_volume"], p["close"] * p["volume"], equal_nan=True, rtol=0, atol=0)


def test_vwap_collapses_onto_close_so_vwap_factors_are_flagged():
    _, st, _, p = _base()
    ev = L._evaluator(p)
    vwap, dev, close = L._values(ev, "vwap"), L._values(ev, "vwap_dev"), p["close"].to_numpy()
    fin = np.isfinite(close) & (p["volume"].to_numpy() > 0)
    assert np.allclose(vwap[fin], close[fin], rtol=1e-12, atol=0)
    assert np.nanmax(np.abs(dev)) < 1e-12, "vwap_dev is rounding noise only"
    flagged = st[st["vwap_proxy"]]
    assert len(flagged) == 1811 and (flagged["group"] == "computable").all()
    assert all(
        L.expr_atoms(e)[1] & set(L.VWAP_FIELDS)
        for e in _base()[0].set_index("factor_id").loc[flagged["factor_id"], "expression"]
    )


def test_panels_are_point_in_time_daily_and_nan_before_listing():
    _, _, raw, p = _base()
    idx = p["close"].index
    last_bar = max(r["ohlcv"].index.max() for r in raw.values())
    assert str(idx.tz) == "UTC" and (idx == idx.normalize()).all() and idx[-1] == last_bar
    assert idx.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all(), "one row per UTC day, no gaps"
    assert list(p) == list(L.SUPPLIED)
    assert not (
        {
            "premium",
            "trades",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "long_short_ratio",
            "top_long_short_ratio",
            "taker_buy_sell_ratio",
        }
        & set(p)
    )
    assert L.BENCHMARK in p["close"].columns and L.BENCHMARK == "BTC-USDT"
    for sym, r in raw.items():
        first = r["ohlcv"].index.min()
        for k, v in p.items():
            assert v.loc[v.index < first, sym].isna().all(), (sym, k)
        bars = r["ohlcv"].index
        f = data_io.daily_funding(sym)
        assert np.allclose(p["funding_rate"].loc[bars, sym], f.reindex(bars), equal_nan=True, rtol=0, atol=0), sym
        oi = data_io.daily_oi(sym).reindex(bars)
        assert np.allclose(p["open_interest_value"].loc[bars, sym], oi, equal_nan=True, rtol=0, atol=0), sym
        assert np.allclose(p["open_interest"].loc[bars, sym], oi / r["ohlcv"]["close"], equal_nan=True), sym
    # the library's benchmark-relative fields point at the engine's BTC perp
    ev = L._evaluator(p)
    btc = L._values(ev, "btc_ret")
    ret = p["close"] / p["close"].shift(1) - 1
    j = list(p["close"].columns).index("BTC-USDT")
    assert np.allclose(btc[:, j], ret["BTC-USDT"], equal_nan=True)
    assert np.allclose(btc[:, 3][np.isfinite(btc[:, 3])], ret["BTC-USDT"].to_numpy()[np.isfinite(btc[:, 3])])


# --------------------------------------------------------------------------- classification


def test_classification_reconciles_with_the_catalog():
    _, st, _, _ = _base()
    assert len(st) == 28904 and st["factor_id"].is_unique
    assert st["group"].value_counts().to_dict() == {"computable": 17758, "missing_fields": 9602, "needs_premium": 1544}
    assert int(st["oi_short_history"].sum()) == 3311
    comp = st[st["group"] == "computable"]
    assert (comp["missing_fields"] == "").all()
    assert (st.loc[st["group"] == "needs_premium", "missing_fields"] == "premium").all()
    miss = st.loc[st["group"] == "missing_fields", "missing_fields"].str.split(",")
    assert all(set(m) and not set(m) & set(L.SUPPLIED) and set(m) != {"premium"} for m in miss)
    assert not st.loc[st["group"] != "computable", ["oi_short_history", "vwap_proxy", "oi_base_proxy"]].any().any()


# --------------------------------------------------------------------------- point in time


def test_mutation_sample_covers_every_operator_and_field():
    sample = _mutation_sample()
    missing = _required_atoms() - _CACHE["sample_cover"]
    assert not missing, f"operators or fields no sample expression uses: {sorted(missing)}"
    assert len(sample) >= 40


def test_future_mutation_leaves_every_earlier_value_unchanged():
    """Through the whole adapter: every bar, funding and OI value after the cut is noise,
    and every factor value on a date <= cut is unchanged (all days, not only Sundays).
    The cuts: the three PIT cuts, and the last Sunday three weeks before the last bar.
    That last cut lies inside the open-interest history on disk, so OI factors have
    finite values to compare. The mutation must also reach the factors (values after the
    cut change), or the test proves nothing."""
    _, _, raw, panels = _base()
    sample = _mutation_sample()
    exprs = [e for _, e in sample]
    full = _eval_all(panels, exprs)
    late = _late_cut(panels)
    cuts = [pd.Timestamp(c, tz="UTC") for c in L.PIT_CUTS] + [late]
    for seed, cut in enumerate(cuts, start=1):
        mut = _eval_all(L.build_panels(_mutate_raw(raw, cut, seed)), exprs)
        k = int(panels["close"].index.searchsorted(cut, side="right"))
        changed, finite_cells, live = 0, 0, 0
        for (fid, e), a, b in zip(sample, full, mut):
            same = np.isclose(a[:k], b[:k], rtol=L.PIT_RTOL, atol=L.PIT_ATOL, equal_nan=True)
            assert same.all(), f"{fid} {e}: {int((~same).sum())} cells on or before {cut.date()} changed"
            finite_cells += int(np.isfinite(a[:k]).sum())
            changed += not np.array_equal(a[k:], b[k:], equal_nan=True)
            live += bool(np.isfinite(a[k:]).any() or np.isfinite(b[k:]).any())
        # a factor with no value after the cut (an OI factor without open interest) cannot change
        assert finite_cells > 0 and live >= 0.8 * len(sample), (cut, finite_cells, live)
        assert changed >= 0.9 * live, f"the mutation reached only {changed}/{live} factors"
    oi = [i for i, (_, e) in enumerate(sample) if set(L.OI_FIELDS) & L.expr_atoms(e)[1]]
    if not _oi_rows(panels):
        print("    (no open-interest value on disk: the OI factors are not compared)")
        return
    k2 = int(panels["close"].index.searchsorted(late, side="right"))
    assert oi and any(np.isfinite(full[i][:k2]).any() for i in oi), "no OI factor was compared"


def test_truncated_panels_reproduce_the_full_computation():
    """The --compute PIT proof itself, on the sample: compute_values marks every factor
    with finite values pass, with the expected number of compared cells; a factor with a
    deliberate lookahead (delay(close, -1)) fails; a factor that is empty on every Sunday
    is 'no_finite_cells', never 'pass' (nothing was compared). An open-interest factor
    whose values start after the second cut is compared at the third cut. The test takes
    the second and third cuts and the two OI windows from the open-interest history on
    disk, so the check works for any length of that history."""
    _, _, _, panels = _base()
    idx = panels["close"].index
    sun = idx[L.sunday_positions(idx)]
    n_oi = _oi_rows(panels)
    late_expr = f"ts_mean(open_interest_value, {max(5, n_oi // 2)})"
    sample = _mutation_sample()[:20] + [
        ("ahead", "delay(close, -1) / close - 1"),
        ("empty", f"ts_mean(open_interest_value, {n_oi + 30})"),
    ]
    cuts = [L.PIT_CUTS[0], L.PIT_CUTS[-1]]
    if n_oi:
        lv = _eval_all(panels, [late_expr])[0][L.sunday_positions(idx)]
        start = sun[np.isfinite(lv).any(axis=1)][0]  # the first Sunday with a late_oi value
        cuts = [L.PIT_CUTS[0], str((start - pd.Timedelta(days=7)).date()), str(sun[-1].date())]
        assert pd.Timestamp(cuts[0], tz="UTC") < start - pd.Timedelta(days=7), (cuts, start)
        sample.append(("late_oi", late_expr))
    else:
        print("    (no open-interest value on disk: the late open-interest factor is not checked)")
    df = pd.DataFrame(sample, columns=["factor_id", "expression"])
    res, values, _ = L.compute_values(panels, df, workers=1, cuts=tuple(cuts), log=None)
    assert values.dtype == np.float64
    res = res.set_index("factor_id")
    assert not res.loc["ahead", "pit_ok"] and res.loc["ahead", "pit_check"] == "fail", "a lookahead factor passed"
    assert res.loc["empty", "pit_ok"] and res.loc["empty", "finite_cells"] == 0
    assert res.loc["empty", "pit_check"] == "no_finite_cells", "an all-NaN factor was counted as proven"
    if n_oi:
        late = res.loc["late_oi"]
        assert (late["pit_finite_t1"], late["pit_finite_t2"]) == (0, 0) and late["pit_finite_t3"] > 0, late
        assert late["pit_check"] == "pass"
    rest = res.drop(index=[f for f in ("ahead", "empty", "late_oi") if f in res.index])
    assert rest["pit_ok"].all()
    # a sample factor with no finite value (an OI window longer than the OI history) proves nothing
    want = np.where(rest["finite_cells"] > 0, "pass", "no_finite_cells")
    assert (rest["pit_check"].to_numpy() == want).all()
    assert (rest["finite_cells"] > 0).sum() >= 15
    assert (L.pit_status(True, 1), L.pit_status(True, 0), L.pit_status(False, 5)) == ("pass", "no_finite_cells", "fail")
    for j, cut in enumerate(cuts, start=1):
        n_sun = int((idx[L.sunday_positions(idx)] <= pd.Timestamp(cut, tz="UTC")).sum())
        assert (res[f"pit_cells_t{j}"] == n_sun * len(panels["close"].columns)).all()


def test_open_interest_prints_are_stamped_16_utc():
    """libfactors' OI timing note rests on this: every stored OKX open-interest print is stamped
    16:00 UTC (00:00 UTC+8), and data_io.daily_oi books the print stamped D 16:00 on bar D
    (the snapshot-at-the-stamp assumption). A different stamp must fail here first."""
    _, _, raw, _ = _base()
    seen = 0
    for sym in raw:
        f = C.RAW_OI / f"{sym}.parquet"
        if not f.is_file():
            continue
        ts = pd.read_parquet(f, columns=["ts"])["ts"]
        assert str(ts.dt.tz) == "UTC", sym
        assert (ts.dt.hour == 16).all() and (ts.dt.minute == 0).all() and (ts.dt.second == 0).all(), (
            sym,
            ts.dt.hour.value_counts().to_dict(),
        )
        oi = data_io.daily_oi(sym).dropna()
        assert (oi.index == oi.index.normalize()).all()
        assert set(oi.index) == set(ts.dt.normalize()), sym  # the D 16:00 print lands on day D
        seen += 1
    assert seen == len(raw), "every perp has an open-interest file"


# --------------------------------------------------------------------------- the sealed I zone


def _screen_inputs(raw: dict, ft: pd.DataFrame, exprs: list):
    panels = L.build_panels(raw)
    sundays, _ = L.g_sundays()
    pos = panels["close"].index.get_indexer(sundays)
    vals = np.stack([v[pos] for v in _eval_all(panels, exprs)])  # float64, as stored
    return vals, list(panels["close"].columns), L.g_labels(ft), L.build_panels(raw, through=L.LABEL_CUTOFF)


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf)
    return buf.getvalue()


def _screen_sample() -> list:
    comp, st = _computable()
    pool = comp[~st["vwap_proxy"].to_numpy() & ~st["oi_short_history"].to_numpy() & (st["lookback"] < 400).to_numpy()]
    ids = list(pool.sample(50, random_state=5)["factor_id"])
    expr = comp.set_index("factor_id")["expression"]
    return [(f, expr[f]) for f in ids]


def test_screen_refuses_any_label_or_bar_after_the_cutoff():
    ft = data_io.label_table()
    lab = L.g_labels(ft)
    assert lab["date"].max() == L.LABEL_CUTOFF and (lab["date"].dt.dayofweek == 6).all()
    late = pd.concat(
        [lab, pd.DataFrame({"date": [L.LABEL_CUTOFF + pd.Timedelta(days=7)], "symbol": ["BTC-USDT"], L.LABEL: [0.0]})],
        ignore_index=True,
    )
    _, _, raw, _ = _base()
    sundays, _ = L.g_sundays()
    vals = np.zeros((1, len(sundays), len(raw)))
    cut = L.build_panels(raw, through=L.LABEL_CUTOFF)
    for bad_labels, bad_panels, what in ((late, cut, "label"), (lab, L.build_panels(raw), "bar")):
        try:
            L.screen_frame(vals, ["x"], list(raw), bad_labels, bad_panels)
        except L.LabelLeak:
            continue
        raise AssertionError(f"a {what} after the cutoff was accepted")
    assert issubclass(L.LabelLeak, AssertionError)


def test_screen_output_is_byte_identical_when_everything_after_the_cutoff_changes():
    sample = _screen_sample()
    exprs = [e for _, e in sample]
    ids = [f for f, _ in sample]
    _, _, raw, _ = _base()
    ft = data_io.label_table()
    df0, s0, w0 = L.screen_frame(*_insert_ids(_screen_inputs(raw, ft, exprs), ids))
    # every label (fwd_ret_1 and fwd_ret_7 of the label table) and every bar after the cutoff
    rng = np.random.default_rng(7)
    ft_m = ft.copy()
    late = ft_m["date"] > L.LABEL_CUTOFF
    num = ft_m.select_dtypes("number").columns
    ft_m.loc[late, num] = rng.normal(0, 0.2, size=(int(late.sum()), len(num)))
    df1, s1, w1 = L.screen_frame(*_insert_ids(_screen_inputs(_mutate_raw(raw, L.LABEL_CUTOFF, 8), ft_m, exprs), ids))
    assert len(df0) >= 40 and s0 == s1 and np.array_equal(w0, w1)
    assert _parquet_bytes(df0) == _parquet_bytes(df1), "the screen read something after the cutoff"
    # teeth: one label on the last G Sunday changes the output
    ft_e = ft.copy()
    hit = (ft_e["date"] == L.LABEL_CUTOFF) & (ft_e["symbol"] == "ETH-USDT")
    ft_e.loc[hit, L.LABEL] = ft_e.loc[hit, L.LABEL] + 0.5
    df2, _, _ = L.screen_frame(*_insert_ids(_screen_inputs(raw, ft_e, exprs), ids))
    assert _parquet_bytes(df0) != _parquet_bytes(df2)


def _insert_ids(inputs, ids):
    vals, symbols, labels, panels_cut = inputs
    return vals, ids, symbols, labels, panels_cut


# --------------------------------------------------------------------------- statistics


def test_weekly_ic_and_ric_match_an_independent_reference():
    """Per Sunday and factor, rebuilt by hand: names = label-table rows with finite
    factor, label, tilt, beta and size (at least 15 of them); Spearman IC with scipy;
    rIC = scipy Spearman of the closed-form ridge residual (lambda 1e-3, z-scored
    regressors, unpenalised intercept) of the factor's ranks, with the registered tilt
    (shrunk expanding mean of daily excess return), beta and size recomputed from the raw
    bar files symbol by symbol. Ties as the screen defines them: each Sunday's values
    rounded to 12 decimals of that Sunday's largest |value| (written out here)."""
    from scipy.stats import rankdata, spearmanr

    sample = _screen_sample()[:6]
    _, _, raw, _ = _base()
    ft = data_io.label_table()
    vals, symbols, labels, panels_cut = _screen_inputs(raw, ft, [e for _, e in sample])
    sundays, _ = L.g_sundays()
    regs = L.regressors(labels, panels_cut, symbols, sundays)
    lw = labels.pivot(index="date", columns="symbol", values=L.LABEL).reindex(index=sundays, columns=symbols)
    elig = (
        lw.notna().to_numpy()
        | labels.assign(x=1)
        .pivot(index="date", columns="symbol", values="x")
        .reindex(index=sundays, columns=symbols)
        .notna()
        .to_numpy()
    )
    ic, ric, _, _ = L.weekly_ic(L.snap_ties(vals), lw.to_numpy(), elig, regs)
    sun_lab = ft[(ft["date"].dt.dayofweek == 6) & (ft["date"] <= L.LABEL_CUTOFF)]
    # daily perp returns from the raw files, one symbol at a time; the equal-weight mean
    # over the perps with a return that day
    rets = {s: (lambda px: px["close"] / px["close"].shift(1) - 1)(data_io.ohlcv(s)) for s in symbols}
    rets = {s: r[r.index <= L.LABEL_CUTOFF] for s, r in rets.items()}
    days = sorted(set().union(*[set(r.dropna().index) for r in rets.values()]))
    ew = {d: np.mean([rets[s][d] for s in symbols if d in rets[s].index and np.isfinite(rets[s][d])]) for d in days}
    checked = 0
    for t in range(0, len(sundays), 9):  # every 9th Sunday, both zones
        d = sundays[t]
        tilt, beta, size = {}, {}, {}
        rb_px = data_io.ohlcv("BTC-USDT").loc[:d]
        rb = (rb_px["close"] / rb_px["close"].shift(1) - 1).iloc[-90:]
        for s in symbols:
            px = data_io.ohlcv(s)
            past = px.loc[:d]
            r = (past["close"] / past["close"].shift(1) - 1).iloc[-90:]
            ok = r.notna().all() and len(r) == 90
            beta[s] = np.cov(r, rb, ddof=0)[0, 1] / np.var(rb) if ok else np.nan
            dv = (past["close"] * past["volume"]).iloc[-30:]
            size[s] = np.log(dv.mean()) if len(dv) == 30 else np.nan
            ex = [rets[s][x] - ew[x] for x in rets[s].index if x <= d and np.isfinite(rets[s][x])]
            tilt[s] = sum(ex) / (len(ex) + 180) if ex else np.nan
        row = sun_lab[sun_lab["date"] == d].set_index("symbol")[L.LABEL]
        for f in range(len(sample)):
            raw_x = vals[f, t].astype(float)
            scale = np.nanmax(np.abs(raw_x)) if np.isfinite(raw_x).any() else 1.0
            x = pd.Series(np.round(raw_x / (scale if scale > 0 else 1.0) * 1e12), index=symbols)
            names = [
                s
                for s in symbols
                if s in row.index
                and np.isfinite(row[s])
                and np.isfinite(x[s])
                and np.isfinite(tilt[s])
                and np.isfinite(beta[s])
                and np.isfinite(size[s])
            ]
            if len(names) < 15:
                assert np.isnan(ic[f, t]) and np.isnan(ric[f, t])
                continue
            y = row[names].to_numpy()
            ref_ic = spearmanr(x[names], y).statistic
            rk = rankdata(x[names].to_numpy())
            X = np.column_stack([[tilt[s] for s in names], [beta[s] for s in names], [size[s] for s in names]])
            Z = (X - X.mean(axis=0)) / X.std(axis=0)
            yc = rk - rk.mean()
            b = np.linalg.solve(Z.T @ Z + 1e-3 * np.eye(3), Z.T @ yc)
            ref_ric = spearmanr(yc - Z @ b, y).statistic
            assert np.isclose(ic[f, t], ref_ic, atol=1e-9, equal_nan=True), (sample[f][0], str(d.date()))
            assert np.isclose(ric[f, t], ref_ric, atol=1e-7, equal_nan=True), (
                sample[f][0],
                str(d.date()),
                ric[f, t],
                ref_ric,
            )
            checked += 1
    assert checked >= 50, checked


def test_registered_settings_are_asserted_against_the_finder_config():
    """Every rIC setting equals factorfinder/eval_config.toml; a changed value in the
    config stops the screen (SystemExit) instead of running on settings nobody reads."""
    import tomllib

    with (C.ROOT / "factorfinder" / "eval_config.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    r = L.ric_settings()
    assert (
        (r["ridge_lambda"], r["tilt_shrink_days"], r["btc_beta_days"], r["min_names_per_sunday"])
        == (
            cfg["regressors"]["ridge_lambda"],
            cfg["regressors"]["tilt_shrink_days"],
            cfg["regressors"]["btc_beta_days"],
            cfg["g0"]["min_names_per_sunday"],
        )
        == (1e-3, 180, 90, 15)
    )
    assert r["set"] == cfg["regressors"]["set"] and r["used"] == ["pit_symbol_tilt", "btc_beta_90d", "size_proxy"]
    v = L.VARIANTS[L.PRIMARY]
    assert (v["tilt"], v["shrink_days"], v["ridge"], v["min_names"]) == ("pit", 180, 1e-3, 15)
    real = L.eval_config
    for table, key, value in (
        ("regressors", "tilt_shrink_days", 90),
        ("regressors", "ridge_lambda", 0.0),
        ("regressors", "btc_beta_days", 60),
        ("g0", "min_names_per_sunday", 5),
        ("regressors", "set", ["pit_symbol_tilt"]),
    ):
        changed = {k: dict(x) if isinstance(x, dict) else x for k, x in cfg.items()}
        changed[table][key] = value
        L.eval_config = lambda changed=changed: changed
        try:
            L.ric_settings()
        except SystemExit:
            continue
        finally:
            L.eval_config = real
        raise AssertionError(f"a changed [{table}] {key} was accepted")
    # the manifest names every remaining deviation from the registered statistic
    dev = " ".join(L.ric_description()["deviations"])
    for word in ("zoo_pc1", "two-sided", f">= {L.MIN_WEEKS}-week", "G0 checks", "label_mean"):
        assert word in dev, word


def test_pit_tilt_is_point_in_time_and_shrunk():
    """tilt(D) = sum of daily excess returns through D / (n + 180), unchanged when every
    close after D changes; a newly listed perp starts near zero (shrunk), not at its mean."""
    _, _, raw, panels = _base()
    close = panels["close"]
    tilt = L.pit_tilt(close)
    d = pd.Timestamp("2023-03-05", tz="UTC")
    r = close / close.shift(1) - 1
    ex = r.sub(r.mean(axis=1), axis=0).loc[:d]
    for s in ("BTC-USDT", "ETH-USDT", close.columns[-1]):
        e = ex[s].dropna()
        assert np.isclose(tilt.at[d, s], e.sum() / (len(e) + 180), rtol=1e-12, atol=0), s
        assert np.isclose(L.pit_tilt(close, 0).at[d, s], e.mean(), rtol=1e-12), s
    mut = L.build_panels(_mutate_raw(raw, d, 4))["close"]
    t2 = L.pit_tilt(mut)
    assert np.allclose(t2.loc[:d], tilt.loc[:d], equal_nan=True, rtol=0, atol=0), "the tilt read a close after D"
    assert not np.allclose(t2.loc[d:].iloc[1:], tilt.loc[d:].iloc[1:], equal_nan=True)
    # shrinkage: on the last-listed perp's 10th excess return the tilt is 10 / 190 of its mean
    late = r.notna().idxmax().idxmax()
    e = r.sub(r.mean(axis=1), axis=0)[late].dropna()
    tenth = e.index[9]
    assert np.isclose(tilt.at[tenth, late], e.iloc[:10].mean() * 10 / 190, rtol=1e-12)
    assert abs(tilt.at[tenth, late]) < abs(e.iloc[:10].mean())


def test_fit_selection_never_reads_g_eval():
    """The G_fit q-values (the zone design's selection) are identical when every G_eval
    label changes; the G_eval statistics (the check) do change."""
    sample = _screen_sample()[:25]
    _, _, raw, _ = _base()
    ft = data_io.label_table()
    ids = [f for f, _ in sample]
    exprs = [e for _, e in sample]
    df0, _, _ = L.screen_frame(*_insert_ids(_screen_inputs(raw, ft, exprs), ids))
    ft_m = ft.copy()
    ev = (ft_m["date"] >= pd.Timestamp(L.G_EVAL[0], tz="UTC")) & (ft_m["date"] <= L.LABEL_CUTOFF)
    ft_m.loc[ev, L.LABEL] = np.random.default_rng(12).normal(0, 0.2, size=int(ev.sum()))
    df1, _, _ = L.screen_frame(*_insert_ids(_screen_inputs(raw, ft_m, exprs), ids))
    for name in ("ic", "ric"):
        for col in (f"{name}_fit_n", f"{name}_fit_mean", f"{name}_fit_t", f"{name}_fit_p"):
            assert np.array_equal(df0[col].to_numpy(), df1[col].to_numpy(), equal_nan=True), col
        assert not np.allclose(df0[f"{name}_eval_mean"], df1[f"{name}_eval_mean"], equal_nan=True)
    # BH over the G_fit p-values of the batch; the selection count follows from it
    q = L.bh(df0["ric_fit_p"])
    assert np.allclose(df0["ric_fit_q"], q, equal_nan=True)
    fs = L.fit_selection(df0)["ric"]
    assert fs["q05"]["selected"] == int((q <= 0.05).sum()) and fs["n_tests"] == int(np.isfinite(q).sum())


def test_tstats_bh_and_dedup():
    from scipy.stats import ttest_1samp

    rng = np.random.default_rng(3)
    x = rng.normal(0.02, 0.2, size=(4, 30))
    x[1, :5] = np.nan
    x[2, :] = np.nan
    x[2, :2] = [0.1, 0.2]  # n = 2: below the minimum, t undefined
    s = L.tstats(x)
    for i in (0, 1, 3):
        v = x[i][np.isfinite(x[i])]
        ref = ttest_1samp(v, 0.0)
        assert np.isclose(s["t"][i], ref.statistic) and np.isclose(s["p"][i], ref.pvalue)
        assert s["n"][i] == v.size
    assert np.isnan(s["t"][2]) and np.isnan(s["p"][2])
    # BH on a hand-computed vector (Benjamini and Hochberg 1995, step-up)
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5])
    assert np.allclose(L.bh_qvalues(p), [0.025, 0.05, 0.05, 0.025, 0.5])
    # a factor, its sign flip and a near copy are one signal; an independent one is another
    a, b = rng.normal(size=500), rng.normal(size=500)
    V = np.stack([a, -a, a + 0.1 * rng.normal(size=500), b])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    cl, rep = L.dedup(V, np.array([3.0, 2.0, 1.0, 2.5]))
    assert cl[0] == cl[1] == cl[2] != cl[3] and rep.tolist() == [True, False, False, True]


def test_zones_match_the_finder_config():
    z = L.zones()
    sundays, fit = L.g_sundays()
    assert len(sundays) == 114 and fit.sum() == 69 and (~fit).sum() == 45
    assert pd.Timestamp("2023-07-30", tz="UTC") not in sundays
    assert sundays[-1] == L.LABEL_CUTOFF and z["label_cutoff_row"] == "2024-06-09"
    assert z["i_sundays"] == ["2024-06-30", "2026-09-20"]
    assert z["pit_cuts"] == list(L.PIT_CUTS) == ["2024-06-09", "2026-06-28", "2026-09-13"]


# --------------------------------------------------------------------------- full precision

# a rolling correlation of two rank series sits near 1 (real differences of about 1e-6 that
# float32 turns into ties), and an up-volume share that is exactly +-1 in exact arithmetic
# (floating-point noise that float64 alone would order)
PRECISION_IDS = ("CF18768", "CF19722")


def test_screen_uses_full_precision_and_ties_only_noise():
    """The screen's statistics come from the float64 values: a float32 copy of the same
    values changes them (CF18768), while a perturbation of a few ulp does not (CF19722,
    whose exact +-1 values carry rounding noise) because snap_ties ties only differences
    below 1e-12 of the Sunday's largest |value|. The stored screen rows equal factor_stats
    on a fresh float64 computation, and the stored shards are float64."""
    _, _, raw, _ = _base()
    ft = data_io.label_table()
    cat = _base()[0].set_index("factor_id")
    exprs = [cat.at[f, "expression"] for f in PRECISION_IDS]
    vals, symbols, labels, panels_cut = _screen_inputs(raw, ft, exprs)
    assert vals.dtype == np.float64
    s64, _ = L.factor_stats(vals, symbols, labels, panels_cut)
    s32, _ = L.factor_stats(vals.astype(np.float32).astype(np.float64), symbols, labels, panels_cut)
    cols = ["ic_t", "ric_t", "ic_fit_t", "ric_fit_t", "persistence"]
    assert not np.allclose(s64.loc[0, cols].astype(float), s32.loc[0, cols].astype(float), rtol=0, atol=1e-6), (
        "float32 storage would not change CF18768: the precision test has no teeth"
    )
    # a few ulp of noise on every value: the snapped statistics do not move, raw ranks would
    rng = np.random.default_rng(21)
    noisy = vals * (1.0 + rng.integers(-4, 5, size=vals.shape) * np.finfo(np.float64).eps)
    sn, _ = L.factor_stats(noisy, symbols, labels, panels_cut)
    pd.testing.assert_frame_equal(sn, s64)
    sundays = L.g_sundays()[0]
    lab, elig = L.label_grid(labels, sundays, symbols)
    regs = L.regressors(labels, panels_cut, symbols, sundays)
    raw_ic = L.weekly_ic(vals[1:], lab, elig, regs)[0]
    raw_ic_noisy = L.weekly_ic(noisy[1:], lab, elig, regs)[0]
    assert not np.allclose(raw_ic, raw_ic_noisy, equal_nan=True, rtol=0, atol=1e-12), (
        "CF19722's rounding noise does not order names: the tie test has no teeth"
    )
    # snap_ties: idempotent, order-preserving above 1e-12 of the row's scale, NaN kept
    x = np.array([[[1.0, 1.0 - 2e-16, 0.5, 0.5 + 1e-9, np.nan, -3e-17, 0.0]]])
    q = L.snap_ties(x)
    assert q[0, 0, 0] == q[0, 0, 1] and q[0, 0, 5] == q[0, 0, 6] and q[0, 0, 3] > q[0, 0, 2] and np.isnan(q[0, 0, 4])
    assert np.array_equal(L.snap_ties(q), q, equal_nan=True)
    # the stored screen rows are these float64 statistics
    out = L.out_dir()
    if (out / "screen_g.parquet").is_file() and L.has_values():
        g = pd.read_parquet(out / "screen_g.parquet").set_index("factor_id")
        for i, f in enumerate(PRECISION_IDS):
            if f in g.index:
                for c in cols + ["ic_p", "ric_p", "ric_fit_p"]:
                    assert np.isclose(float(g.at[f, c]), float(s64.at[i, c]), rtol=1e-6, atol=1e-7), (f, c)
        stored, _, _ = L.read_values(list(PRECISION_IDS), sundays)
        assert stored.dtype == np.float64 and np.array_equal(stored, vals, equal_nan=True), (
            "shards are not full precision"
        )


def test_a_float32_shard_is_refused():
    """read_values refuses a shard column that is not float64 (for example a float32 copy)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    saved = C.DATA
    with tempfile.TemporaryDirectory() as td:
        data = Path(td) / "data"
        vdir = data / L.OUT_NAME / L.VALUES_NAME
        vdir.mkdir(parents=True)
        C.DATA = data
        try:
            pd.DataFrame({"factor_id": ["CFX", "CFY"], "values_shard": ["values_000.parquet"] * 2}).to_parquet(
                data / L.OUT_NAME / "status.parquet"
            )
            days = pd.date_range("2022-04-03", periods=2, freq="7D", tz="UTC")
            t = pa.table(
                {
                    "sunday": pa.array(np.repeat(days.tz_localize(None).to_numpy(), 2), type=pa.timestamp("ms")),
                    "symbol": ["A", "B", "A", "B"],
                    "CFX": pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64()),
                    "CFY": pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float32()),
                }
            )
            pq.write_table(t, vdir / "values_000.parquet")
            v, _, _ = L.read_values(["CFX"])
            assert v.dtype == np.float64 and v[0].tolist() == [[1.0, 2.0], [3.0, 4.0]]
            try:
                L.read_values(["CFY"])
            except ValueError:
                pass
            else:
                raise AssertionError("a float32 shard was read")
        finally:
            C.DATA = saved


# --------------------------------------------------------------------------- determinism and writer


def test_compute_is_deterministic_across_worker_counts():
    comp, _ = _computable()
    df = comp.sample(150, random_state=4)[["factor_id", "expression"]].reset_index(drop=True)
    _, _, _, panels = _base()
    r1, v1, _ = L.compute_values(panels, df, workers=1, log=None)
    r3, v3, s3 = L.compute_values(panels, df, workers=3, log=None)
    assert s3["workers"] == 3
    assert np.array_equal(v1, v3, equal_nan=True) and r1.equals(r3)
    assert (L.cache_order(df["expression"]) == L.cache_order(df["expression"])).all()


def test_writer_refuses_everything_but_its_own_folder():
    assert C.protected_folder(C.SHIPPED_DATA / L.OUT_NAME / "status.parquet") is None
    saved = (C.DATA, C.SHIPPED_DATA)
    with tempfile.TemporaryDirectory() as td:
        data = Path(td) / "data"
        for name in C.PROTECTED_DIRS:
            (data / name).mkdir(parents=True)
        (data / C.PROTECT_MARKER).write_text("protected\n")
        C.DATA, C.SHIPPED_DATA = data, data
        try:

            def refused(fn) -> bool:
                try:
                    fn()
                except SystemExit:
                    return True
                return False

            out = L.out_dir()
            assert not refused(lambda: L._guard(out / "status.parquet"))
            assert refused(lambda: L._guard(data / "research" / "x.parquet")), "a protected target"
            assert refused(lambda: L._guard(data / "x.parquet")), "outside its folder"
            assert refused(lambda: L._guard(out / "sub" / "x.parquet")), "a subfolder other than values/"
            out.mkdir()
            L._write_json({"a": 1}, out / "manifest.json")
            os.link(out / "manifest.json", Path(td) / "other")
            assert refused(lambda: L._write_json({"a": 2}, out / "manifest.json")), "a hard-linked target"
            assert (out / "manifest.json").read_text().strip().endswith("}") and '"a": 1' in (
                out / "manifest.json"
            ).read_text()
            os.unlink(Path(td) / "other")
            (out / "manifest.json").unlink()
            out.rmdir()
            out.symlink_to(data / "research")
            assert refused(lambda: L._guard(out / "status.parquet")), "a symlinked data/library_on_engine"
            assert not any((data / "research").iterdir())
        finally:
            C.DATA, C.SHIPPED_DATA = saved


# --------------------------------------------------------------------------- the written outputs


def test_written_outputs_reconcile_and_carry_no_path():
    out = L.out_dir()
    if not (out / "manifest.json").is_file() or not L.has_values():
        print("    (no data/library_on_engine outputs with values/: skipped)")
        return
    import json

    from webexport.common import privacy_hits

    man = json.loads((out / "manifest.json").read_text())
    st = pd.read_parquet(out / "status.parquet")
    text = (out / "manifest.json").read_text()
    for s in (text, "\n".join(st.select_dtypes(exclude="number").astype(str).to_numpy().ravel())):
        assert not privacy_hits(s)
        assert str(C.ROOT) not in s and str(Path.home()) not in s and "/Users/" not in s
    assert len(st) == 28904 and "verdict" not in st.columns
    c = man["compute"]["counts"]
    assert c["groups"] == st["group"].value_counts().sort_index().to_dict()
    assert c["computed_ok"] == int((st["compute"] == "ok").sum())
    mp = man["compute"]["pit"]
    assert mp["pass"] == int((st["pit_check"] == "pass").sum())
    assert mp["fail"] == int((st["pit_check"] == "fail").sum())
    assert mp["no_finite_cells"] == int((st["pit_check"] == "no_finite_cells").sum())
    assert mp["pass"] + mp["no_finite_cells"] + mp["fail"] == c["computed_ok"]
    assert mp["cuts"] == list(L.PIT_CUTS) and (mp["rtol"], mp["atol"]) == (L.PIT_RTOL, L.PIT_ATOL)
    cut_cols = range(1, len(L.PIT_CUTS) + 1)
    assert mp["compared_cells"] == [int(st[f"pit_cells_t{j}"].sum()) for j in cut_cols]
    assert mp["finite_cells"] == [int(st[f"pit_finite_t{j}"].sum()) for j in cut_cols]
    # pass means at least one finite compared cell; no_finite_cells means none
    fin = st[[f"pit_finite_t{j}" for j in cut_cols]].sum(axis=1)
    assert (fin[st["pit_check"] == "pass"] > 0).all() and (fin[st["pit_check"] == "no_finite_cells"] == 0).all()
    assert (st.loc[st["pit_check"] == "no_finite_cells", "screen"] == "pit_no_finite_cells").all()
    empty = (st["compute"] == "ok") & (st["finite_cells"] == 0)
    oi = st["oi_short_history"]
    assert c["all_nan_on_sundays"] == int(empty.sum()) and c["all_nan_split"] == {
        "open_interest": int((empty & oi).sum()),
        "vwap_proxy": int((empty & ~oi & st["vwap_proxy"]).sum()),
        "other": int((empty & ~oi & ~st["vwap_proxy"]).sum()),
    }
    assert "16:00 UTC" in man["panels"]["derivations"]["open_interest_value"]
    assert man["compute"]["values"]["dtype"] == "float64"
    assert (st.loc[st["compute"] == "ok", "values_shard"] != "").all()
    # stored values equal a fresh computation, bit for bit (float64)
    cat, _, _, panels = _base()
    ids = list(st.loc[(st["compute"] == "ok") & (st["finite_cells"] > 0), "factor_id"].sample(25, random_state=9))
    vals, symbols, days = L.read_values(ids)
    assert symbols == list(panels["close"].columns)
    pos = panels["close"].index.get_indexer(days)
    fresh = _eval_all(panels, list(cat.set_index("factor_id").loc[ids, "expression"]))
    for i, f in enumerate(fresh):
        assert vals.dtype == np.float64 and np.array_equal(vals[i], f[pos], equal_nan=True), ids[i]
    # values/ is git-ignored (in a git checkout)
    repo = C.ROOT.parent
    rel = out / L.VALUES_NAME / "values_000.parquet"
    if C.DATA.resolve() == (C.ROOT / "data").resolve() and (repo / ".git").exists():
        chk = subprocess.run(["git", "check-ignore", "-q", str(rel.relative_to(repo))], cwd=repo)
        assert chk.returncode == 0, "values/ must be git-ignored"
    if "screen" not in man:
        return
    sg = pd.read_parquet(out / "screen_g.parquet")
    k = man["screen"]["counts"]
    assert len(sg) == k["screened"] == int((st["screen"] == "screened").sum())
    assert set(sg["factor_id"]) == set(st.loc[st["screen"] == "screened", "factor_id"])
    assert (sg["ic_n"] >= L.MIN_WEEKS).all() and "verdict" not in sg.columns
    assert not st.loc[st["screen"] == "screened", ["vwap_proxy"]].any().any()
    assert (st.loc[st["screen"] == "screened", "pit_check"] == "pass").all()
    for name in ("ic", "ric"):
        p = sg[f"{name}_p"].to_numpy()
        fin = np.isfinite(p)
        assert k[f"n_tests_{name}"] == int(fin.sum())
        assert np.allclose(sg[f"{name}_q"].to_numpy()[fin], L.bh_qvalues(p[fin]))
        for lev in L.Q_LEVELS:
            tag = f"q{int(round(lev * 100)):02d}"
            surv = sg[f"{name}_q"] <= lev
            assert k[f"{name}_count_{tag}"] == int(surv.sum())
            assert k[f"{name}_distinct_signals_{tag}"] == int(sg[f"{name}_rep_{tag}"].sum())
            assert (sg.loc[~surv, f"{name}_signal_{tag}"] == -1).all() and (
                sg.loc[surv, f"{name}_signal_{tag}"] >= 0
            ).all()
    # the G_fit selection: BH over the stored G_fit p-values, and the counts it reports
    for name in ("ic", "ric"):
        assert np.allclose(sg[f"{name}_fit_q"], L.bh(sg[f"{name}_fit_p"]), equal_nan=True)
    assert k["fit_selection"] == L.fit_selection(sg)
    # the sensitivity report's registered row is the screen itself
    sens = man["screen"]["sensitivity"]
    reg = sens[L.PRIMARY]
    assert (reg["screened"], reg["ric_count_q05"], reg["ric_count_q10"]) == (
        k["screened"],
        k["ric_count_q05"],
        k["ric_count_q10"],
    )
    assert reg["ric_fit_selected_q05"] == k["fit_selection"]["ric"]["q05"]["selected"]
    # the variant names and the texts are the module's own (when the same code wrote the manifest)
    if L.changed_code_files(man):
        print("    (the code changed after the computation of the manifest: its texts are not compared)")
    else:
        assert list(sens) == list(L.VARIANTS)
        assert man["screen"]["ric_regressors"] == json.loads(json.dumps(L.ric_description()))
    # g_weeks: the weekly-IC count on G of every factor with G values, screened or not
    gw = st.set_index("factor_id")["g_weeks"]
    assert (gw[sg["factor_id"]].to_numpy() == sg["ic_n"].to_numpy()).all()
    few = st[st["screen"] == "too_few_weeks"]
    assert len(few) == k["dropped_too_few_weeks"] and (few["g_weeks"] < L.MIN_WEEKS).all()
    assert (st.loc[~st["screen"].isin(["screened", "too_few_weeks"]), "g_weeks"] == 0).all()
    if len(few):
        sundays, _ = L.g_sundays()
        ids = list(few["factor_id"])
        v, symbols, _ = L.read_values(ids, sundays)
        lab = L.g_labels()
        regs = L.regressors(lab, L.build_panels(L.load_raw(), through=L.LABEL_CUTOFF), symbols, sundays)
        label, elig = L.label_grid(lab, sundays, symbols)
        ic, _, _, _ = L.weekly_ic(L.snap_ties(v), label, elig, regs)
        assert (np.isfinite(ic).sum(axis=1) == few["g_weeks"].to_numpy()).all(), "g_weeks of too_few_weeks rows"
        assert (few["g_weeks"] > 0).any()


# the words that a find tool must not write (numbers only): verdicts, tiers, stages, gate-pass words
VERDICT_WORDS = re.compile(
    r"\b(useful|useless|unproven|effective|survivors?|promotable|novel|confirmed|retired|tiers?|stages?)\b"
    r"|\bpass(es|ed)?\s+(the\s+|all\s+|every\s+)?(gates?|screen)\b|\bgates?\s+pass(es|ed)?\b|forward-only",
    re.IGNORECASE,
)


def _texts(x) -> list:
    """Every string value in a JSON-like object."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [t for v in x.values() for t in _texts(v)]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _texts(v)]
    return []


def _names(x) -> list:
    """Every key of a JSON-like object, at each depth."""
    if isinstance(x, dict):
        return [str(k) for k in x] + [t for v in x.values() for t in _names(v)]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _names(v)]
    return []


def _word_hits(texts) -> list:
    """The texts that hold a word of VERDICT_WORDS. An underscore counts as a space, so the
    check also finds a word inside a key or a column name (ric_xyz_q05 reads as ric xyz q05)."""
    return [t for t in texts if VERDICT_WORDS.search(str(t).replace("_", " "))]


def test_outputs_are_numbers_only():
    """The screen outputs and --status give numbers only: no verdict column and no verdict, tier or stage word.

    The check reads the manifest texts and keys, the column names and the string cells of the
    outputs, and the --status lines. --status prints the counts and the states of the code and
    the inputs. It prints no ranked table."""
    import contextlib
    import json

    assert not hasattr(L, "VERDICT")
    # the texts of the module: the manifest's about, evidence and notes, the rIC description
    own = [L.ABOUT, L.EVIDENCE, L.POOLED_G_NOTE, *L.VARIANT_NOTES.values()]
    assert not _word_hits(own + _texts(L.ric_description())), _word_hits(own + _texts(L.ric_description()))
    assert L.ABOUT.endswith("The values are numbers. No rule here gives a verdict.")
    # a fresh screen of a sample: no verdict column, no such word in its string cells
    sample = _screen_sample()[:6]
    _, _, raw, _ = _base()
    df, summary, _ = L.screen_frame(
        *_insert_ids(_screen_inputs(raw, data_io.label_table(), [e for _, e in sample]), [f for f, _ in sample])
    )
    assert "verdict" not in df.columns
    assert not _word_hits(list(df.columns) + _texts(summary) + _names(summary)), _word_hits(_names(summary))
    # the word list has teeth: also inside a key, and for a gate pass, but not for "not applied as gates"
    teeth = ["every factor stays unproven", "the top tier", "the survivors", "it passes the gates"]
    assert _word_hits(teeth) == teeth, (_word_hits(teeth), teeth)
    assert _word_hits(_names({"screen": {"counts": {"useful_q05": 1}}})) == ["useful_q05"]
    assert not _word_hits(["novelty", "the other G0 checks are not applied as gates", "ric_count_q05"])
    out = L.out_dir()
    if not (out / "manifest.json").is_file():
        print("    (part skipped: no data/library_on_engine outputs)")
        return
    man = json.loads((out / "manifest.json").read_text())
    assert not _word_hits(_texts(man) + _names(man)), _word_hits(_texts(man) + _names(man))
    if not L.changed_code_files(man):
        assert man["about"] == L.ABOUT
        if "screen" in man:
            assert man["screen"]["evidence"] == L.EVIDENCE and man["screen"]["pooled_g_note"] == L.POOLED_G_NOTE
    for name in ("status.parquet", "screen_g.parquet"):
        if (out / name).is_file():
            t = pd.read_parquet(out / name)
            cells = t.select_dtypes(exclude=["number", "bool"]).astype(str).to_numpy().ravel().tolist()
            assert "verdict" not in t.columns and not _word_hits(list(t.columns) + sorted(set(cells))), name
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert L.show_status() == 0
    text = buf.getvalue()
    assert not _word_hits(text.splitlines()), _word_hits(text.splitlines())
    assert "\n  code: " in text and "\n  inputs: " in text and " top " not in text and "factor_id" not in text


def test_missing_catalog_stops():
    """Without the library catalog the reader stops with the setup instruction."""
    saved = L.CATALOG
    with tempfile.TemporaryDirectory() as td:
        L.CATALOG = Path(td) / "absent.db"
        try:
            L.load_catalog()
            raise AssertionError("a missing catalog was accepted")
        except SystemExit as exc:
            assert "Run ./setup.sh first" in str(exc)
        finally:
            L.CATALOG = saved


def test_status_without_the_catalog_says_not_checked():
    """--status without the library catalog prints "inputs: not checked" and does not stop."""
    import contextlib

    saved = L.CATALOG
    with tempfile.TemporaryDirectory() as td:
        L.CATALOG = Path(td) / "absent.db"
        try:
            assert L.changed_inputs({"inputs": {}}, "okx") is None
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                L._print_states({"inputs": {}}, "okx", "--compute and --screen")
                if L._read_manifest():
                    assert L.show_status() == 0
        finally:
            L.CATALOG = saved
    text = buf.getvalue()
    assert "  inputs: not checked (an input is not on disk)" in text and "inputs: changed" not in text, text


def test_status_names_a_changed_code_file():
    """--status gives the line "code: changed after the computation" with each code file whose hash
    differs from the manifest's. With the same hashes it gives "code: the same as at the computation"."""
    import contextlib
    from unittest import mock

    now = L.code_fingerprint()["files"]
    changed = (
        "  code: changed after the computation: engine/libfactors.py (the manifest's code hash differs from "
        "today's files); --compute and --screen refresh the outputs\n"
    )
    for files, line in (
        ({**now, "engine/libfactors.py": "0" * 64}, changed),
        (now, "  code: the same as at the computation\n"),
    ):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            L._print_states({"code": {"files": files}, "inputs": {}}, "okx", "--compute and --screen")
        assert buf.getvalue().startswith(line), (buf.getvalue(), line)
    # the full --status of each source, with one recorded code hash changed in its manifest
    for source, show, refresh in (
        ("okx", L.show_status, "--compute and --screen"),
        ("binance", L.show_status_binance, "--source binance --compute and --screen"),
    ):
        man = L._read_manifest(source)
        if not man:
            print(f"    (part skipped: no {source} outputs)")
            continue
        man = {**man, "code": {**man.get("code", {}), "files": {**now, "engine/libfactors.py": "0" * 64}}}
        buf = io.StringIO()
        with mock.patch.object(L, "_read_manifest", return_value=man), contextlib.redirect_stdout(buf):
            assert show() == 0
        lines = buf.getvalue().splitlines()
        code = [x for x in lines if x.startswith("  code: ")]
        line = (
            "  code: changed after the computation: engine/libfactors.py (the manifest's code hash differs from "
            f"today's files); {refresh} refresh the outputs"
        )
        assert code == [line], (source, code, line)


def missing_data() -> str | None:
    """Why the suite cannot run on this data folder, or None."""
    if not L.CATALOG.is_file():
        return "no library catalog (run ./setup.sh)"
    if not any((C.RAW_OHLCV / f"{C.sym_to_file(s)}.parquet").exists() for s in C.UNIVERSE):
        return "no perp bars in the data folder (run ./setup.sh)"
    if not label_path().is_file():
        return "no label table in the data folder (run ./setup.sh)"
    return None


if __name__ == "__main__":
    why = missing_data()
    if why:
        print(f"  SKIP  every libfactors test skipped: {why}")
        raise SystemExit(0)
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"  PASS  {n}")
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            failed += 1
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} libfactors tests passed")
    raise SystemExit(1 if failed else 0)
