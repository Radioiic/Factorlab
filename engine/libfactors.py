"""Library factors on Factorlab's data: classify, compute, prove point in time, screen on G.

Usage (from engine/):
    python -B libfactors.py --status     # read-only summary of data/library_on_engine/
    python -B libfactors.py --compute    # classify all 28,904 definitions, compute every
                                         # computable one on the engine's panels, PIT proof
    python -B libfactors.py --screen     # the G-zone screen (needs --compute first)
    python -B libfactors.py --source binance --compute   # all 28,904 on the Binance panel
    python -B libfactors.py --source binance --screen    # its new family and cross-check
    python -B libfactors.py --source binance --status    # read-only
--source defaults to okx, the engine's own panels (everything below up to "THE BINANCE
SOURCE" describes it); the binance source is described at the end of this docstring.

This module puts every definition of the factor library (../library/) that the engine's
data can compute onto the engine's own point-in-time panels (the 20 OKX USDT perpetuals of
config.UNIVERSE), proves each computed factor point in time, and screens them on the
generator zone G (factorfinder/eval_config.toml [zones]). The inference zone I stays
SEALED: no label after the 2024-06-09 row is ever read.

NUMBERS ONLY. Every output is a number or a count on the generator zone G. No rule here
gives a verdict, and there is no book here. To test a definition in a book, use
engine/judge.py: judge.py check --library ID, then judge.py register --library ID.

Outputs, all in data/library_on_engine/ (not a protected folder; every write goes through
config.guard_write first):
    status.parquet     one row per library definition (28,904): group, missing
                       fields, flags, compute result, PIT check, screen status
    screen_g.parquet   one row per screened factor: IC and residual IC (rIC) on G,
                       G_fit and G_eval separately, BH q-values (pooled G, and G_fit
                       alone), persistence, coverage, distinct-signal clusters
    manifest.json      zones, n_tests, counts, input fingerprints, code hash,
                       timings, generated_at (UTC); paths are relative, never absolute
    values/            weekly (Sunday) float64 values of every computed factor,
                       sharded by factor_id; git-ignored (about 0.7 GB uncompressed).
                       Full precision on purpose: the screen ranks these values, and
                       a float32 copy turns real differences into ties (values near
                       0.99999 of a rolling correlation) and changes the statistics
status.parquet, screen_g.parquet and manifest.json are small.

THE PANEL ADAPTER (engine data -> the library's raw panels), point in time:
    index     one row per UTC day, stamped with the bar's OPEN time, from the first
              perp bar to the data-through date (the last closed bar on disk; the
              still-forming bar is never on disk, ingest.py drops it)
    columns   the engine's symbol files ("BTC-USDT", ...). The benchmark argument is
              "BTC-USDT", so the library's bench() and hence btc_ret / ex_btc_ret
              refer to the engine's BTC perp; mkt_ret is the equal-weight mean over
              the perps that have a bar that day.
    open, high, low, close, volume
              the OKX perp bars as stored (data_io.ohlcv). VOLUME IS IN BASE-ASSET
              UNITS (coins), not contracts: ccxt (pinned 4.5.71) maps OKX's swap
              candle field 6, volCcy ("number of base currency" for derivatives), to
              the volume column, not field 5 (contracts). Magnitude check
              (test_libfactors.py): the perp/spot base-volume ratio stays in a 1-50 band
              for every symbol and year, with no power-of-ten jump between symbols whose
              OKX contract sizes differ by 10^5 (0.01 BTC against 1000 DOGE), which
              contract counts would produce.
    quote_volume
              close x volume (USDT). This is the engine's own dollar-volume
              convention (the size_proxy regressor below, alphaforge_bridge
              "amount"). OKX's candle quote volume (volCcyQuote,
              VWAP x volume) is not stored, so the true turnover is unavailable.
              CONSEQUENCE: the library's derived field vwap = div(quote_volume,
              volume) equals close exactly, and vwap_dev = close / vwap - 1 is 0.
              Factors that use vwap or vwap_dev are computed but flagged
              vwap_proxy and kept out of the screen: on this data they are not the
              definitions the library wrote: vwap_dev is 0 up to rounding (about
              1e-16), whose cross-sectional ranks are noise, and vwap-level factors
              are copies of close-based ones. Not flagged: the 14 factors of the form
              close / (ts_sum(quote_volume, w) / ts_sum(volume, w)) - 1, whose
              denominator becomes the volume-weighted mean close over w days, the
              usual daily-bar stand-in for a w-day VWAP.
    funding_rate
              the daily sum of funding settlements, data_io.daily_funding: the
              per-8h-slot OKX/Binance splice, summed per UTC day with
              label='left', closed='left', so funding[D] is known at D's close. The
              library defines the field as per-8h rates summed per bar. Days without
              a settlement stay NaN (data_io.funding_matrix's zero fill is a P&L
              accrual convention, not an observation). Before the first OKX
              settlement on disk the rates are Binance USDT-M prints, a proxy for OKX's.
    open_interest_value
              oi_value (USD notional, treated as USDT) via data_io.daily_oi: OKX's
              rubik statistic, aggregated over all of the coin's contracts. OKX
              serves only recent months of it, so the history starts after G. TIMING: every stored rubik print is stamped 16:00 UTC
              (00:00 UTC+8, OKX's day boundary), not at a bar's 00:00 UTC open, and
              daily_oi (resample "1D", label and closed left, last print) books the
              print stamped D 16:00 on bar D. That is point in time on one
              ASSUMPTION: the value is the open-interest snapshot taken at its stamp,
              8 hours before D's close. If OKX's daily value instead described the
              UTC+8 day that starts at the stamp, bar D would carry up to 16 hours of
              bar D + 1. test_libfactors.py pins the 16:00 stamp, so a change in
              OKX's stamping fails before any OI factor is screened. No OI factor
              reaches the G screen (OI starts after G ends), so no screen statistic
              depends on the assumption.
    open_interest
              base-asset units = open_interest_value / close. The stored
              oi_contracts column is empty (rubik serves USD only), so the coin count
              is derived; factors using it are flagged oi_base_proxy.
    every field is NaN where the perp has no bar (before listing). premium, trades,
    taker flow and long/short ratios are NOT supplied: a perp-vs-spot basis would be a
    proxy that changes the premium factors' definition, and Factorlab collects none
    of the others.

POINT-IN-TIME PROOF. Every computed factor is recomputed on panels truncated at three
fixed cuts: T1 = 2024-06-09 (the last G Sunday), T2 = 2026-06-28 and T3 = 2026-09-13.
On every Sunday <= T, the truncated values must equal the full computation within
rtol 1e-9 and atol 1e-12 (np.isclose; NaN equals NaN). "Equal" thus means equal up to
floating-point rounding, not bit-identical. An OI factor has finite cells to compare
only at a cut inside its open-interest history. pit_check is
    pass             every compared cell equal, and at least one FINITE cell compared
                     over the three cuts
    no_finite_cells  every compared cell equal, but none was finite: nothing was
                     proven (the factor is empty on every Sunday up to T3); it is
                     counted apart and never screened
    fail             a compared cell differs, or a truncated computation raised;
                     never screened
test_libfactors.py adds a future-mutation test through the panel adapter.

THE G-ZONE SCREEN (zones, rIC regressors and the per-Sunday name floor from
factorfinder/eval_config.toml [zones], [regressors] and [g0], asserted equal here):
    Sundays   G_fit 2022-04-03 -> 2023-07-23 (69), G_eval 2023-08-06 -> 2024-06-09 (45);
              the internal embargo Sunday 2023-07-30 is excluded
    label     fwd_ret_7 from the label table (labels.py, data/labels/labels.parquet),
              Sunday rows through 2024-06-09 only (its label completes at the 2024-06-16
              close); a later label raises LabelLeak
    mask      rows present in the label table (the engine's eligible rows) with a
              finite factor value, label and regressors; a Sunday with fewer than 15
              such names gives no IC ([g0] min_names_per_sunday)
    ties      the screen reads the float64 values and, per factor and Sunday, treats
              values within 1e-12 of that Sunday's largest |value| as ties
              (snap_ties): floating-point noise between values that are equal in
              exact arithmetic (an up-volume share of exactly 1, a z-score of a
              flat funding window) never orders names, and real differences far
              above 1e-12 always do. Every statistic below uses ranks only.
    IC        weekly Spearman IC: mean, sd, t = mean / sd * sqrt(n), two-sided
              Student-t p (n - 1 df); the same on G_fit and G_eval separately
    rIC       each Sunday the factor's cross-sectional ranks are ridge-residualised
              (lambda 1e-3 on z-scored regressors, unpenalised intercept) on the
              registered point-in-time regressors, and the Spearman IC of the
              residual with the label is taken:
                pit_symbol_tilt  the shrunk expanding mean of the symbol's daily
                                 excess return (its perp return minus the equal-weight
                                 mean of the perps with a return that day) through D:
                                 sum / (n + 180) = n / (n + 180) x mean
                btc_beta_90d     OLS beta of its daily returns on the BTC perp's over
                                 the 90 returns up to D
                size_proxy       log of its 30-day mean USDT volume (close x volume)
              Raw IC rewards factors that merely rank symbols by their past mean
              return; rIC removes that part.
    persistence  mean week-over-week cross-sectional Spearman of the factor
    BH        Benjamini-Hochberg q-values over every screened factor, separately for
              IC and rIC, on the pooled G p-value; the factors at q <= 0.05 and at
              q <= 0.10 are grouped into distinct signals by pooled within-Sunday rank
              correlation (|rho| >= 0.7). The pooled-G q-values are IN SAMPLE on G_eval
              (G_eval's Sundays are part of the pooled p), so they say nothing about
              replication.
    G_fit selection, G_eval check  the zone design's own split: BH over the G_fit
              p-values alone (ic_fit_q, ric_fit_q); the selected factors' sign and t on
              G_eval, which the selection never saw, are the replication check
              (manifest screen.counts.fit_selection).
    DEVIATIONS FROM THE REGISTERED STATISTIC (manifest screen.ric_regressors.deviations):
    no zoo PC1 regressor (no zoo member has a value on a G Sunday); a two-sided p with
    BH over the screened family, not a one-sided test in a fixed sign; the screen's own
    >= 20-week family filter; the other G0 checks are not applied. The counts at each q
    depend on these choices. manifest screen.sensitivity gives the counts for each
    single change of the settings and for the label_mean variant. That variant uses the
    mean weekly fwd_ret_7 label over Sundays <= D - 7 as the tilt (0 below 8 labels,
    unshrunk), OLS and a 5-name floor.

THE BINANCE SOURCE. Binance is DATA ONLY here, as everywhere in Factorlab. The library's
own downloader (library/factorlab/data/binance.py) gets Binance's public USD-M data for
the same 20 coins, one symbol at a time. library/scripts/merge_panels.py merges the parts
offline into library/data/binance_um_1d/ (git-ignored). That script's docstring gives
the commands, and ./setup.sh runs them. The panel starts on
2022-01-01. merge_panels.py --append adds new closed days and changes no old cell, and
the manifest's panels.provenance.appends lists each append. `--source binance` computes
all 28,904 definitions on that panel and writes only data/library_on_engine/binance/
(status.parquet, screen_g.parquet, crosscheck.parquet, manifest.json; values/
git-ignored); the OKX outputs one folder up are never written by it.
    panel     EVERY input of every factor comes from the Binance panel: no formula mixes
              venues. Binance columns are mapped to the engine's symbol names and order
              (BTCUSDT -> BTC-USDT, ...), so the benchmark "BTC-USDT" (btc_ret,
              ex_btc_ret, bench()) is Binance's BTCUSDT. The panel has one row for each
              UTC day, stamped with the bar's open. The rows go from the first kline bar
              (2022-01-01) to the last closed one. Every field is NaN where the coin has no
              kline bar. All 16 raw fields of the library are supplied, so every
              definition is computable.
    fields, each known by bar D's close (D + 1 00:00 UTC):
      open, high, low, close, volume, quote_volume, trades, taker_buy_volume,
      taker_buy_quote_volume
              the USD-M 1d kline with open time D (/fapi/v1/klines): prices and flows of
              [D 00:00, D + 1 00:00). quote_volume is the true USDT turnover, so the
              library's vwap = quote_volume / volume is the bar's VWAP and vwap / vwap_dev
              are real series here (vwap_collapses() is False; the manifest records the
              measured |close / vwap - 1|), not degenerate as on OKX.
      funding_rate
              the sum of the settlements whose fundingTime lies in [D 00:00, D + 1 00:00)
              (/fapi/v1/fundingRate; the downloader's to_bar: resample 1D, label and
              closed left, sum with min_count 1, so NaN on a day without a settlement).
              A settlement stamped D + 1 00:00 (plus a few ms) belongs to D + 1. The
              compute checks it cell by cell against the same daily sum of the Binance
              prints the engine stores (raw/funding_binanceusdm, which data_io splices
              into OKX funding before the first OKX settlement on disk).
      premium
              the close (field 4) of the 1d premium-index kline with open time D
              (/fapi/v1/premiumIndexKlines): the premium index at the end of D.
      open_interest, open_interest_value, long_short_ratio, top_long_short_ratio,
      taker_buy_sell_ratio
              the data.binance.vision daily metrics files: 5-minute snapshots, of which
              the bar keeps the last one whose create_time (UTC) lies in [D 00:00,
              D + 1 00:00) (resample 1D, label and closed left, last). Whether a
              create_time marks the start or the end of its 5 minutes, the kept snapshot
              describes data up to D + 1 00:00 at the latest. The compute checks the
              timing: the implied mark price open_interest_value / open_interest must sit
              far closer to D's close than to D - 1's (manifest panels.checks).
              open_interest is Binance's coin count (sum_open_interest), not a derived
              proxy. The kept snapshot is the last NON-MISSING one in the bar (resample .last()),
              so on a day whose archive file is partial it is an earlier snapshot.
              taker_buy_sell_ratio is the ratio of the kept snapshot's 5 minutes
              (normally the bar's last; sum_taker_long_short_vol_ratio), not of the whole day as the library's
              REST definition with a 1d period would be: a noisier stand-in, flagged
              taker_ratio_last_5m and still screened. Coverage: open interest covers
              all of G (unlike OKX's recent months); the long/short, top-trader and taker
              ratios start a few weeks after the panel, with gaps.
    cleaning  archive errors are set missing before any factor reads the panel
              (CLEANING_RULES). Each rule reads one coin's fields on one bar, so it is
              point in time and reads no label, and no value is carried from another day:
              open_interest or open_interest_value <= 0 sets both missing; quote_volume /
              volume outside the bar's [low, high] sets volume missing; the same test on
              taker_buy_quote_volume / taker_buy_volume sets taker_buy_volume missing.
              manifest panels.checks.cleaning lists every cell by date and coin, and
              test_libfactors_binance.py pins the cells of the downloaded panel.
              CLEANING_FREEZE pins the rules with a hash, so a rule edit
              cannot change the values of the weekly log of the formula tests
              (judge.py log-week). The compute
              stops when the rules in the code differ from the pinned set, and manifest
              panels.cleaning_freeze records the hash.
    labels, regressors  exactly the OKX screen's: fwd_ret_7 from the label table,
              and pit_symbol_tilt, btc_beta_90d and size_proxy from the engine's OKX bars
              cut at the 2024-06-09 row (the same LabelLeak seal), so the two screens
              differ only in the factor inputs.
    NEW FAMILY  the definitions the OKX screen did not screen (OKX status != screened)
              that on Binance are PIT pass, not degenerate and have >= 20 weekly ICs on G:
              the same G screen (G_fit + G_eval, embargo excluded, registered rIC), with
              Benjamini-Hochberg over THIS family only (its own n_tests), its own
              distinct signals and its own G_fit -> G_eval replication check. origin, first
              match: binance_only_fields (they read premium, trades, taker flow or a
              long/short ratio: not computable on OKX), oi (open interest: OKX's history
              starts after G), vwap (degenerate on OKX), other. The manifest also gives
              the overlap of each set of factors at a q level with the OKX screen's set
              at the same level (|pooled rank correlation| >= 0.7).
    CROSS-CHECK  the definitions the OKX screen screened: their G statistics on
              the Binance panel, for cross-venue AGREEMENT only (crosscheck.parquet):
              no Benjamini-Hochberg, no q-value, no list of factors at a q level. For them
              the OKX screen gives the only q-values. What agreement can show is limited, and the
              manifest says so (crosscheck.agreement_note, funding_note): the two screens
              share the labels, regressors, Sundays, names and eligibility mask, and the
              same definition's values correlate closely across venues, so sign agreement
              is expected and is not an independent replication; and on G the engine's
              OKX funding is Binance's own settlement archive (data_io's splice before
              the first OKX settlement on disk), so for the definitions that read funding_rate that input is the
              same on both venues (series_agreement's identical_share; the agreement is
              also reported without them).
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
import sqlite3
import stat
import sys
import time
import tomllib
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import data_io
from labels import path as label_path

# The library is a sibling package (repo/library/factorlab), not an installed one.
# Appended, not prepended, so nothing in library/ can shadow an engine module; the
# assert below makes sure the import really is the repository's library.
LIBRARY = C.ROOT.parent / "library"
if str(LIBRARY) not in sys.path:
    sys.path.append(str(LIBRARY))
import factorlab  # noqa: E402
from factorlab.expr import Evaluator, parse, raw_dependencies  # noqa: E402
from factorlab.fields import DERIVED_BY_NAME, RAW_FIELD_NAMES, RAW_FIELDS  # noqa: E402
from factorlab.stats import bh_qvalues  # noqa: E402

if Path(factorlab.__file__).resolve().parent != (LIBRARY / "factorlab").resolve():
    raise SystemExit("libfactors: the imported factorlab package is not this repository's library/")

# --------------------------------------------------------------------------- constants

CATALOG = LIBRARY / "catalog" / "factors.db"
OUT_NAME = "library_on_engine"
VALUES_NAME = "values"

SUPPLIED = ("open", "high", "low", "close", "volume", "quote_volume", "funding_rate",
            "open_interest", "open_interest_value")
PREMIUM = "premium"
OI_FIELDS = ("open_interest", "open_interest_value")
VWAP_FIELDS = ("vwap", "vwap_dev")          # derived fields that collapse onto close here
BENCHMARK = C.sym_to_file("BTC/USDT:USDT")   # "BTC-USDT": the engine's BTC perp column

MAX_WORKERS = 8          # upper limit of worker processes
CHUNK = 60               # factors per pool job: contiguous runs of the cache-friendly order
CACHE_SIZE = 512         # sub-expression frames kept per Evaluator (about 0.28 MB each)
SHARD_FACTORS = 500      # factor columns per values shard

# Zones (the same boundaries as factorfinder/eval_config.toml [zones], which zones()
# asserts). The I zone (2024-06-30 -> 2026-09-20) is sealed: nothing here reads
# a label in it, and the only I-zone quantities written are factor VALUES (inputs).
G_FIT = ("2022-04-03", "2023-07-23")
G_INTERNAL_EMBARGO = "2023-07-30"
G_EVAL = ("2023-08-06", "2024-06-09")
I_SUNDAYS = ("2024-06-30", "2026-09-20")
LABEL_CUTOFF = pd.Timestamp("2024-06-09", tz="UTC")   # the last Sunday row whose label may be read
N_FIT, N_EVAL = 69, 45

# T1 is the last G Sunday; T2 and T3 are two later fixed cuts. An OI factor has finite
# cells to compare only at a cut inside its open-interest history.
PIT_CUTS = ("2024-06-09", "2026-06-28", "2026-09-13")
PIT_RTOL, PIT_ATOL = 1e-9, 1e-12
# snap_ties(): values within 10^-TIE_DIGITS of a Sunday's largest |value| are ties
TIE_DIGITS = 12

LABEL = "fwd_ret_7"
# The registered rIC settings (eval_config.toml [regressors] and [g0]); ric_settings()
# asserts every one equal to the config, the way zones() asserts the zones.
REGISTERED_SET = ("pit_symbol_tilt", "btc_beta_90d", "size_proxy", "zoo_pc1")
TILT_SHRINK_DAYS = 180   # pit_symbol_tilt = n / (n + 180) x expanding mean of daily excess return
BETA_DAYS = 90           # btc_beta_90d
SIZE_DAYS = 30           # size_proxy: "30d log dollar volume"
RIDGE_LAMBDA = 1e-3      # ridge residualisation, on z-scored regressors
MIN_NAMES = 15           # [g0] min_names_per_sunday: a Sunday with fewer valid names gives no IC
REG_KEYS = ("tilt", "beta", "size")
# Screen family. A factor enters the BH family only with >= MIN_WEEKS weekly ICs on G.
# The rule depends on sample size alone, never on the statistic (independent
# filtering, which keeps BH valid): below 20 weekly cross-sections the SE of a mean IC
# exceeds about 0.05 (the sd of a Spearman correlation of 20 unrelated names is
# 1/sqrt(19), about 0.23), larger than any plausible mean IC, so such a test has no
# power and would only dilute the family.
# Not registered: it is the screen's own rule (listed in the manifest's deviations).
MIN_WEEKS = 20
DEDUP_RHO = 0.7
Q_LEVELS = (0.05, 0.10)
# the manifest texts: numbers only, no rule here gives a verdict
ABOUT = ("The library definitions computed on Factorlab's data and screened on the Sundays of the zone G. "
         "The values are numbers. No rule here gives a verdict.")
BINANCE_ABOUT = ("The library definitions computed on the Binance USD-M panel (Binance is data only) and screened "
                 "on the Sundays of the zone G. The values are numbers. No rule here gives a verdict.")
EVIDENCE = ("numbers from the Sundays of the zone G only (2022-04-03 to 2024-06-09); no label after the "
            "2024-06-09 row is read")
CROSSCHECK_EVIDENCE = ("Agreement of the two venues on the same labels and regressors. It is not an independent "
                       "replication.")
POOLED_G_NOTE = ("the pooled-G q-values (ic_q, ric_q) use all 114 G Sundays, G_eval included, so they are in "
                 "sample on G_eval; the replication check is counts.fit_selection")
# the input key of the label table in the manifests (input_fingerprints)
LABEL_INPUT = "labels/labels.parquet"

# The rIC variants. "registered" is the screen's statistic (screen_g.parquet); the others
# are computed only for the manifest's sensitivity report, so a reader can see how far
# each unregistered choice moves the counts at each q. "label_mean" uses the mean weekly
# fwd_ret_7 label over Sundays <= D - 7 (0 below TILT_MIN_LABELS such labels, unshrunk)
# as the tilt, OLS and 5 names.
PRIMARY = "registered"
TILT_MIN_LABELS = 8
VARIANTS = {
    "registered": {"tilt": "pit", "shrink_days": TILT_SHRINK_DAYS, "ridge": RIDGE_LAMBDA, "min_names": MIN_NAMES},
    "names_5": {"tilt": "pit", "shrink_days": TILT_SHRINK_DAYS, "ridge": RIDGE_LAMBDA, "min_names": 5},
    "ols": {"tilt": "pit", "shrink_days": TILT_SHRINK_DAYS, "ridge": 0.0, "min_names": MIN_NAMES},
    "tilt_unshrunk": {"tilt": "pit", "shrink_days": 0, "ridge": RIDGE_LAMBDA, "min_names": MIN_NAMES},
    "no_tilt": {"tilt": "none", "shrink_days": None, "ridge": RIDGE_LAMBDA, "min_names": MIN_NAMES},
    "label_mean": {"tilt": "label_mean", "shrink_days": None, "ridge": 0.0, "min_names": 5},
}
VARIANT_NOTES = {
    "registered": "the screen's statistic: the registered tilt, beta and size, ridge 1e-3, 15 names",
    "names_5": "as registered, with a 5-name floor per Sunday",
    "ols": "as registered, with OLS instead of ridge",
    "tilt_unshrunk": "as registered, with the tilt's expanding mean unshrunk",
    "no_tilt": "as registered, without the tilt regressor (beta and size only)",
    "label_mean": ("tilt = mean weekly fwd_ret_7 label over Sundays <= D - 7, 0 below 8 labels, unshrunk; "
                   "OLS; 5 names"),
}
BINANCE_VARIANT_NOTES = {
    **VARIANT_NOTES,
    "label_mean": ("tilt = mean weekly fwd_ret_7 label over Sundays <= D - 7, 0 below 8 labels, unshrunk; "
                   "OLS; 5 names; applied to the new family"),
}

# files whose content is the code behind the outputs (manifest "code")
CODE_FILES = ("engine/libfactors.py", "engine/data_io.py", "engine/config.py",
              "library/factorlab/expr.py", "library/factorlab/operators.py",
              "library/factorlab/fields.py", "library/factorlab/stats.py")

# ---- panel sources
SOURCES = ("okx", "binance")
DEFAULT_SOURCE = "okx"
BINANCE_SUBDIR = "binance"                              # data/library_on_engine/binance/
BINANCE_PANEL_DIR = LIBRARY / "data" / "binance_um_1d"  # git-ignored (library/.gitignore /data/)
BINANCE_PANEL_SET = "binance_um_1d"                     # the manifest names the set, never a path
# every raw field the library defines, in its own order: the Binance panel supplies all 16
BINANCE_FIELDS = tuple(f.name for f in RAW_FIELDS)
# the fields OKX cannot supply at all (premium, trades, taker flow, long/short ratios)
BINANCE_ONLY = tuple(f for f in BINANCE_FIELDS if f not in SUPPLIED)
# what a new-family factor gains on Binance: a field OKX lacks, open interest over G, a true VWAP
NEW_FIELDS = BINANCE_ONLY + OI_FIELDS + ("vwap",)
TAKER_LAST_5M = "taker_buy_sell_ratio"                  # the bar's last 5-minute ratio (see the docstring)
ORIGINS = ("binance_only_fields", "oi", "vwap", "other")
BINANCE_CODE_FILES = CODE_FILES + ("library/factorlab/data/binance.py", "library/factorlab/data/store.py",
                                   "library/scripts/merge_panels.py")
BINANCE_FIRST_BAR = "2022-01-01"                         # the first bar of the merged panel
BINANCE_PROVENANCE = {
    "use": "Binance public data as data only",
    "sources": ["Binance USD-M public REST /fapi/v1/klines, interval 1d",
                "Binance USD-M public REST /fapi/v1/fundingRate",
                "Binance USD-M public REST /fapi/v1/premiumIndexKlines, interval 1d",
                "data.binance.vision daily metrics archive (futures/um/daily/metrics)"],
    "download": ("the library CLI, one symbol per call: download --symbols <SYMBOL> --start 2022-01-01 "
                 "--end '<last closed day> 23:59:59' --interval 1d --market um --metrics-archive, with "
                 "FACTORLAB_OFFLINE=0 for that command only (library/scripts/merge_panels.py docstring)"),
    "merge": "library/scripts/merge_panels.py (offline)",
    "update": ("each update uses the same call with --start and --end around the new closed days, into "
               "data/binance_um_1d_parts/updates/<first>_<last>/<SYMBOL>/. library/scripts/merge_panels.py --append "
               "then adds those days and changes no old cell. appends lists each append (from appends.jsonl in the "
               "panel folder)."),
    "pre_2022": f"not downloaded: the panel starts on {BINANCE_FIRST_BAR}",
}
BINANCE_APPEND_LOG = "appends.jsonl"      # written by library/scripts/merge_panels.py --append


class LabelLeak(AssertionError):
    """A label (or a bar used for a regressor) after the 2024-06-09 row reached the screen.

    An explicit raise, not a bare assert, so it still refuses under python -O."""


def _check_source(source: str) -> str:
    if source not in SOURCES:
        raise ValueError(f"unknown panel source {source!r}; one of {SOURCES}")
    return source


def out_dir(source: str = DEFAULT_SOURCE) -> Path:
    """data/library_on_engine (okx) or data/library_on_engine/binance (binance)."""
    base = Path(C.DATA) / OUT_NAME
    return base if _check_source(source) == "okx" else base / BINANCE_SUBDIR


# --------------------------------------------------------------------------- zones

EVAL_CONFIG = "factorfinder/eval_config.toml"


def eval_config() -> dict:
    """factorfinder/eval_config.toml, parsed (read-only).

    Read with tomllib directly, so this module does not import the factorfinder package."""
    with (C.ROOT / EVAL_CONFIG).open("rb") as fh:
        return tomllib.load(fh)


def ric_settings() -> dict:
    """The rIC settings, asserted equal to eval_config.toml [regressors] and [g0].

    A changed value stops the program (like zones()), so the screen can never silently
    run on settings other than the ones this module documents."""
    cfg = eval_config()
    r, g0 = cfg["regressors"], cfg["g0"]
    want = {"set": list(REGISTERED_SET), "ridge_lambda": RIDGE_LAMBDA, "tilt_shrink_days": TILT_SHRINK_DAYS,
            "btc_beta_days": BETA_DAYS, "size_proxy": f"{SIZE_DAYS}d log dollar volume"}
    for key, value in want.items():
        if r.get(key) != value:
            raise SystemExit(f"libfactors: [regressors] {key} is {r.get(key)!r} in eval_config.toml, expected {value!r}")
    if g0.get("min_names_per_sunday") != MIN_NAMES:
        raise SystemExit(f"libfactors: [g0] min_names_per_sunday is {g0.get('min_names_per_sunday')!r} in "
                         f"eval_config.toml, expected {MIN_NAMES}")
    return {"source": f"{EVAL_CONFIG} [regressors] and [g0]", **want, "min_names_per_sunday": MIN_NAMES,
            "used": [x for x in REGISTERED_SET if x != "zoo_pc1"]}


def zones() -> dict:
    """The zone boundaries, asserted equal to factorfinder/eval_config.toml [zones]."""
    z = eval_config()["zones"]
    want = {"g_fit_sundays": list(G_FIT), "g_internal_embargo_sundays": [G_INTERNAL_EMBARGO] * 2,
            "g_eval_sundays": list(G_EVAL), "i_sundays": list(I_SUNDAYS)}
    for key, value in want.items():
        if z.get(key) != value:
            raise SystemExit(f"libfactors: zone {key} is {z.get(key)} in eval_config.toml, expected {value}")
    if z["expected_counts"]["g_fit"] != N_FIT or z["expected_counts"]["g_eval"] != N_EVAL:
        raise SystemExit("libfactors: eval_config.toml zone counts changed")
    return {"version": z["version"], "g_fit_sundays": list(G_FIT), "g_internal_embargo_sunday": G_INTERNAL_EMBARGO,
            "g_eval_sundays": list(G_EVAL), "label_cutoff_row": str(LABEL_CUTOFF.date()),
            "g_eval_label_close": z["g_eval_label_close"], "embargo_sundays": z["embargo_sundays"],
            "i_sundays": list(I_SUNDAYS), "i_zone": "sealed: no label or metric in it is read or written",
            "pit_cuts": list(PIT_CUTS)}


def g_sundays() -> tuple[pd.DatetimeIndex, np.ndarray]:
    """The 114 screen Sundays (G_fit then G_eval; the embargo Sunday is not among them)
    and a boolean array marking the G_fit ones."""
    fit = pd.date_range(G_FIT[0], G_FIT[1], freq="7D", tz="UTC")
    ev = pd.date_range(G_EVAL[0], G_EVAL[1], freq="7D", tz="UTC")
    days = fit.append(ev)
    if len(fit) != N_FIT or len(ev) != N_EVAL or (days.dayofweek != 6).any():
        raise SystemExit("libfactors: the G zone is not 69 + 45 Sundays")
    if pd.Timestamp(G_INTERNAL_EMBARGO, tz="UTC") in days:
        raise SystemExit("libfactors: the internal embargo Sunday must not be screened")
    return days, np.r_[np.ones(N_FIT, bool), np.zeros(N_EVAL, bool)]


# --------------------------------------------------------------------------- panel adapter

def load_raw() -> dict:
    """The engine's own reads, per symbol, through data_io only (read-only).

    {sym: {"ohlcv": DataFrame(open, high, low, close, volume), "funding": daily Series or
    None, "oi": daily Series or None}}. Kept separate from build_panels() so a test can
    overwrite every value after a cut date and rebuild the panels."""
    raw = {}
    for symbol in C.UNIVERSE:
        sym = C.sym_to_file(symbol)
        px = data_io.ohlcv(sym)
        if px is None:
            continue
        raw[sym] = {"ohlcv": px[["open", "high", "low", "close", "volume"]].astype(float),
                    "funding": data_io.daily_funding(sym),
                    "oi": data_io.daily_oi(sym)}
    if not raw:
        raise SystemExit("libfactors: no perp bars on disk")
    return raw


def build_panels(raw: dict, through=None) -> dict[str, pd.DataFrame]:
    """Engine data -> the library's raw panels (see the module docstring for each field).

    `through` truncates every panel at that day (a PIT cut); by default the panels end at
    the data-through date, the last closed bar on disk."""
    for sym, r in raw.items():
        ix = r["ohlcv"].index
        if ix.tz is None or (ix != ix.normalize()).any():
            raise ValueError(f"{sym}: bars must be stamped with their UTC open at 00:00")
    first = min(r["ohlcv"].index.min() for r in raw.values())
    last = max(r["ohlcv"].index.max() for r in raw.values())
    if through is not None:
        last = min(last, pd.Timestamp(through, tz="UTC") if pd.Timestamp(through).tz is None
                   else pd.Timestamp(through))
    idx = pd.date_range(first, last, freq="D", name="ts")
    cols = list(raw)

    def wide(get) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=idx, columns=cols)
        for s in cols:
            v = get(raw[s])
            if v is not None:
                out[s] = v.reindex(idx).astype(float)
        return out

    p = {k: wide(lambda r, k=k: r["ohlcv"][k]) for k in ("open", "high", "low", "close", "volume")}
    bar = p["close"].notna()
    p = {k: v.where(bar) for k, v in p.items()}
    p["quote_volume"] = p["close"] * p["volume"]
    p["funding_rate"] = wide(lambda r: r["funding"]).where(bar)
    p["open_interest_value"] = wide(lambda r: r["oi"]).where(bar)
    p["open_interest"] = p["open_interest_value"] / p["close"]
    if set(p) != set(SUPPLIED):
        raise AssertionError("panel fields drifted from SUPPLIED")
    return {k: p[k] for k in SUPPLIED}


def sunday_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """Row positions of every Sunday in a daily panel index, first Sunday onwards."""
    return np.flatnonzero(index.dayofweek == 6)


# --------------------------------------------------------------------------- binance panel adapter

def binance_symbol(sym: str) -> str:
    """The engine's symbol name ("BTC-USDT", config.sym_to_file) -> the Binance USD-M symbol ("BTCUSDT")."""
    base, quote = sym.split("-")
    return base + quote


def binance_symbol_map() -> dict[str, str]:
    """{Binance symbol: engine symbol} for config.UNIVERSE, in the engine's order."""
    return {binance_symbol(C.sym_to_file(s)): C.sym_to_file(s) for s in C.UNIVERSE}


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def load_raw_binance(panel_dir: Path | None = None) -> dict:
    """The merged Binance panel per engine symbol, read-only through the library's own reader.

    {sym: DataFrame(index = UTC bar open, columns = BINANCE_FIELDS)}, keyed by the engine's
    symbol names in config.UNIVERSE order. The panel must hold exactly the 16 raw fields
    and exactly the universe's 20 coins (a missing or an extra coin stops the program).
    Reads nothing from engine/data. Kept separate from build_panels_binance() so a test
    can overwrite every value after a cut and rebuild the panels."""
    from factorlab.data import load_panels
    d = BINANCE_PANEL_DIR if panel_dir is None else Path(panel_dir)
    if not d.is_dir() or not any(d.glob("*.csv.gz")):
        raise SystemExit(f"libfactors: the Binance panel ({BINANCE_PANEL_SET}) is missing: download it and run "
                         "library/scripts/merge_panels.py first (commands in that script's docstring)")
    p = load_panels(d)
    if set(p) != set(BINANCE_FIELDS):
        raise SystemExit(f"libfactors: the Binance panel's fields {sorted(set(p) ^ set(BINANCE_FIELDS))} differ "
                         "from the library's 16 raw fields")
    want = binance_symbol_map()
    cols = set(p["close"].columns)
    if cols != set(want):
        raise SystemExit(f"libfactors: the Binance panel's coins differ from the universe: {sorted(cols ^ set(want))}")
    return {sym: pd.DataFrame({f: p[f][b] for f in BINANCE_FIELDS}).astype(float) for b, sym in want.items()}


def binance_panel_appends(panel_dir: Path | None = None) -> list[dict]:
    """The appends that library/scripts/merge_panels.py --append made to the panel, oldest first.

    It reads appends.jsonl in the panel folder ([] when the file is absent, for example after
    a full merge). Each append must start on the day after the end of the append before it.
    Read-only."""
    d = BINANCE_PANEL_DIR if panel_dir is None else Path(panel_dir)
    f = d / BINANCE_APPEND_LOG
    if not f.is_file():
        return []
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    for prev, r in zip(rows, rows[1:]):
        end = prev["last_bar"]
        if r["first_bar"] != str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()):
            raise SystemExit(f"libfactors: {BINANCE_APPEND_LOG} has an append that starts on {r['first_bar']}, "
                             f"but the panel before it ends on {end}")
    return rows


def build_panels_binance(raw: dict, through=None, clean: bool = True) -> dict[str, pd.DataFrame]:
    """Binance frames -> the library's 16 raw panels (module docstring, "THE BINANCE SOURCE").

    The layout of build_panels(): one row per UTC day stamped with the bar's open, from the
    first kline bar to the last one (or to `through`, a PIT cut), the engine's symbol
    columns in `raw` order, every field NaN where the coin has no kline bar. Reads nothing
    but `raw`, so no other venue's data can reach a Binance factor. With `clean` (the
    default, and what every factor reads) the CLEANING_RULES set the archive errors they
    find to NaN (clean_binance()); clean=False keeps the cells as downloaded, for the
    manifest's record of what was cleaned and for the tests."""
    for sym, df in raw.items():
        ix = df.index
        if ix.tz is None or str(ix.tz) != "UTC" or (ix != ix.normalize()).any() or not ix.is_unique:
            raise ValueError(f"{sym}: Binance bars must be stamped with their UTC open at 00:00, once each")
        if list(df.columns) != list(BINANCE_FIELDS):
            raise ValueError(f"{sym}: Binance frame columns must be the library's 16 raw fields")
    bars = [df.index[df["close"].notna()] for df in raw.values()]
    bars = [b for b in bars if len(b)]
    if not bars:
        raise SystemExit("libfactors: the Binance panel has no kline bar")
    first, last = min(b.min() for b in bars), max(b.max() for b in bars)
    if through is not None:
        last = min(last, _utc(through))
    idx = pd.date_range(first, last, freq="D", name="ts")
    cols = list(raw)
    p = {f: pd.DataFrame({s: raw[s][f].reindex(idx) for s in cols}, index=idx, columns=cols).astype(float)
         for f in BINANCE_FIELDS}
    bar = p["close"].notna()
    p = {f: p[f].where(bar) for f in BINANCE_FIELDS}
    return clean_binance(p) if clean else p


# ---- label-free cleaning of archive errors in the Binance panel (BINANCE_CONVENTIONS["cleaning"]).
# Each rule reads only one coin's fields on one bar, all known at that bar's close, so it is
# point in time and reads no label; the cells it fires on become NaN (missing, like a bar the
# coin does not have), never a value carried from another day (manifest
# panels.checks.cleaning lists every cell they clear).
CLEAN_TOL = 1e-9         # relative tolerance on the bar's [low, high] range
CLEANING_RULES = {
    # rule: (the fields set missing on the cells it fires on, what it tests)
    "open_interest_nonpositive": (
        ("open_interest", "open_interest_value"),
        "open_interest or open_interest_value <= 0: a listed perpetual always has open interest, so such a snapshot "
        "is an archive error; both fields of that coin and bar are set missing"),
    "volume_vwap_outside_low_high": (
        ("volume",),
        "quote_volume / volume (the bar's VWAP) outside the bar's [low, high] (relative tolerance 1e-9): an average "
        "price cannot leave the bar's range, so one of the two volumes is wrong; volume, the base-unit figure, is set "
        "missing (on the cell found, the quote-unit figures agree with the rest of the bar and the base one does not)"),
    "taker_buy_volume_vwap_outside_low_high": (
        ("taker_buy_volume",),
        "taker_buy_quote_volume / taker_buy_volume outside the bar's [low, high] (relative tolerance 1e-9): the same "
        "test for the taker-buy pair; taker_buy_volume, the base-unit figure, is set missing (on the cell found, the "
        "taker-buy share of quote volume matches the days around it and the base-unit share does not)"),
}


def binance_glitches(p: dict) -> dict[str, pd.DataFrame]:
    """{rule: boolean frame (bar x coin)}: the cells each CLEANING_RULES rule fires on.

    Evaluated on uncleaned panels (build_panels_binance(raw, clean=False)); every rule is
    tested on the same uncleaned cells, so the rules do not depend on each other. A NaN in
    any field a rule reads never fires it; a zero base volume with a positive quote volume
    does (an infinite average price)."""
    lo, hi = p["low"], p["high"]

    def outside(quote: str, base: str) -> pd.DataFrame:
        with np.errstate(all="ignore"):
            px = p[quote] / p[base]
        return (px.notna() & ((px < lo * (1 - CLEAN_TOL)) | (px > hi * (1 + CLEAN_TOL)))).astype(bool)

    return {
        "open_interest_nonpositive": ((p["open_interest"] <= 0) | (p["open_interest_value"] <= 0)).astype(bool),
        "volume_vwap_outside_low_high": outside("quote_volume", "volume"),
        "taker_buy_volume_vwap_outside_low_high": outside("taker_buy_quote_volume", "taker_buy_volume"),
    }


def clean_binance(p: dict) -> dict[str, pd.DataFrame]:
    """The panels with every CLEANING_RULES cell set to NaN in the rule's fields (a new dict;
    the input frames are not changed)."""
    out = dict(p)
    for rule, mask in binance_glitches(p).items():
        for f in CLEANING_RULES[rule][0]:
            out[f] = out[f].mask(mask)
    return out


def binance_cleaning_record(p: dict) -> dict:
    """The manifest's record of the cleaning (panels.checks.cleaning), from UNCLEANED panels:
    per rule its text, the fields set missing and every cell it fires on, by date and coin,
    with the values that fired it. Read-only."""
    masks = binance_glitches(p)
    out = {}
    with np.errstate(all="ignore"):
        for rule, mask in masks.items():
            fields, text = CLEANING_RULES[rule]
            hit = mask.stack()
            where = []
            for day, sym in hit[hit].index:
                cell = {"date": str(day.date()), "symbol": sym}
                if rule == "open_interest_nonpositive":
                    cell.update({f: float(p[f].at[day, sym]) for f in OI_FIELDS})
                else:
                    quote, base = ("quote_volume", "volume") if fields == ("volume",) else \
                        ("taker_buy_quote_volume", "taker_buy_volume")
                    cell.update({"average_price": round(float(p[quote].at[day, sym] / p[base].at[day, sym]), 6),
                                 "low": float(p["low"].at[day, sym]), "high": float(p["high"].at[day, sym])})
                where.append(cell)
            out[rule] = {"rule": text, "fields_set_missing": list(fields), "cells": len(where), "where": where}
    return out


# ---- the freeze of the cleaning rules. The rules are fixed, so a rule edit cannot change the
# values of the weekly log of the formula tests (judge.py log-week). The hash covers the rule names, the fields that each rule sets
# missing, the rule texts and CLEAN_TOL. It also covers the tokens of binance_glitches and
# clean_binance without their docstrings and comments, so comments, blank lines and the indent
# width do not count. The token list is the same on each Python version. run_compute_binance
# refuses rules whose hash differs from CLEANING_FREEZE["sha256"], and test_libfactors_binance.py
# pins the same hash.
CLEANING_FREEZE = {
    "version": "1",
    "sha256": "ec8b739ac9182fbb4b1d3d69faf9d017f9981b32637ac64f7573c22591462880",
    "form": "tokens without docstrings and comments (tokenize)",
}
CLEANING_CODE = ("binance_glitches", "clean_binance")


def _source_tokens(src: str) -> list[str]:
    """The tokens of Python source, without docstrings and comments, as a list of strings.

    The list does not depend on the Python version. Comments, blank lines, line breaks inside
    brackets and the width of the indentation do not change it. The block structure does:
    INDENT, DEDENT and NEWLINE become the markers <INDENT>, <DEDENT> and <NEWLINE>. A docstring
    is a statement of strings only, directly after a def or class line. An f-string is one item
    with its source text (Python 3.12 and later split it into parts)."""
    import io
    import tokenize
    starts, n = [], 0
    for line in src.splitlines(keepends=True):
        starts.append(n)
        n += len(line)

    def offset(pos: tuple) -> int:
        return starts[pos[0] - 1] + pos[1] if pos[0] <= len(starts) else n

    marks = {tokenize.INDENT: "<INDENT>", tokenize.DEDENT: "<DEDENT>", tokenize.NEWLINE: "<NEWLINE>"}
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
    units, depth, begin = [], 0, None             # (kind, text); kind "mark", "str" or "tok"
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        name = tokenize.tok_name[t.type]
        if name.endswith("STRING_START"):         # FSTRING_START (3.12), TSTRING_START (3.14)
            depth += 1
            begin = t.start if depth == 1 else begin
            continue
        if depth:
            if name.endswith("STRING_END"):
                depth -= 1
                if depth == 0:
                    units.append(("str", src[offset(begin):offset(t.end)]))
            continue
        if t.type in skip:
            continue
        if t.type in marks:
            units.append(("mark", marks[t.type]))
        else:
            units.append(("str" if t.type == tokenize.STRING else "tok", t.string))
    out, line_start, header, doc_slot, k = [], True, False, False, 0
    while k < len(units):
        kind, text = units[k]
        if text in ("<INDENT>", "<DEDENT>") and kind == "mark":
            out.append(text)
            k += 1
            continue
        if doc_slot and line_start and kind == "str":
            j = k
            while j < len(units) and units[j][0] == "str":
                j += 1
            if j < len(units) and units[j] == ("mark", "<NEWLINE>"):
                k, doc_slot = j + 1, False            # a docstring: its strings and its NEWLINE go
                continue
        doc_slot = False
        if kind == "mark":                            # <NEWLINE>: the end of a logical line
            out.append(text)
            doc_slot, header, line_start = header, False, True
        else:
            if line_start:
                header = kind == "tok" and text in ("def", "class", "async")
            out.append(text)
            line_start = False
        k += 1
    return out


def _code_tokens(fn) -> list[str]:
    """_source_tokens of a function's source. A change to the comments, the layout or the
    docstring does not change the result. A change to the code or to its block structure does."""
    import inspect
    import textwrap
    return _source_tokens(textwrap.dedent(inspect.getsource(fn)))


def cleaning_rules_fingerprint() -> dict:
    """The fingerprint of the cleaning rules as this code defines them (read-only)."""
    content = {
        "rules": {r: {"fields_set_missing": list(f), "text": t} for r, (f, t) in CLEANING_RULES.items()},
        "clean_tol": repr(CLEAN_TOL),
        "code": {name: _code_tokens(globals()[name]) for name in CLEANING_CODE},
    }
    text = json.dumps(content, sort_keys=True, ensure_ascii=True)
    return {"rules": list(CLEANING_RULES), "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "hash_of": ("the rule names, the fields that each rule sets missing, the rule texts, CLEAN_TOL, and the "
                        "tokens of " + " and ".join(CLEANING_CODE) + " without their docstrings and comments "
                        "(the same on each Python version)")}


def cleaning_freeze_record() -> dict:
    """The manifest's record of the freeze (panels.cleaning_freeze). It refuses rules that differ
    from the frozen set."""
    now = cleaning_rules_fingerprint()
    if now["sha256"] != CLEANING_FREEZE["sha256"]:
        raise SystemExit(f"libfactors: the Binance cleaning rules differ from the set that CLEANING_FREEZE pins "
                         f"(version {CLEANING_FREEZE['version']}, sha256 {CLEANING_FREEZE['sha256'][:12]}). "
                         "The weekly log of the formula tests (judge.py log-week) reads the pinned set only.")
    return {**CLEANING_FREEZE, "rules": now["rules"], "hash_of": now["hash_of"], "matches_frozen_set": True}


def source_panels(raw: dict, through=None, source: str = DEFAULT_SOURCE) -> dict[str, pd.DataFrame]:
    """The source's panels: build_panels() (okx) or build_panels_binance()."""
    return build_panels(raw, through) if _check_source(source) == "okx" else build_panels_binance(raw, through)


def vwap_collapses(panels: dict) -> bool:
    """True when the library's vwap (quote_volume / volume, its own definition, evaluated
    by the library's Evaluator) equals close within 1e-12 on every finite cell: then vwap
    copies close and vwap_dev is zero up to rounding, and factors reading them are
    degenerate (vwap_proxy). True on OKX, where quote_volume is close x volume; False on
    Binance, whose quote_volume is the true USDT turnover."""
    ev = _evaluator(panels)
    vw, c = _values(ev, "vwap"), panels["close"].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        dev = np.abs(vw / c - 1.0)
    fin = np.isfinite(dev)
    return bool(fin.any() and dev[fin].max() < 1e-12)


def binance_checks(panels: dict, uncleaned: dict) -> dict:
    """Label-free checks of the Binance panel's conventions (manifest panels.checks), on the
    cleaned `panels` the factors read; `uncleaned` (build_panels_binance(raw, clean=False))
    gives the cleaning record.

    cleaning  every cell each CLEANING_RULES rule set missing, by date and coin, and the
              number of cells each rule still fires on after cleaning (0)
    funding   the panel's funding_rate against the daily (label and closed left) sum of
              the Binance settlement prints the engine stores (raw/funding_binanceusdm,
              read through data_io), on every day-cell both have
    metrics   the implied mark price open_interest_value / open_interest against D's
              close and D - 1's close: a snapshot from the bar's last minutes sits next
              to D's close (a UTC+8 stamp would put it 8 hours earlier)
    vwap      |close / vwap - 1| and the share of vwap inside [low, high]
    okx       Binance against OKX closes: median price ratio and daily-return correlation
    coverage  filled cells and first valid date of every field
    Read-only; none of these values enters a factor."""
    close = panels["close"]
    out: dict = {"cleaning": binance_cleaning_record(uncleaned)}
    after = {rule: int(m.to_numpy().sum()) for rule, m in binance_glitches(panels).items()}
    if any(after.values()):
        raise AssertionError(f"a cleaning rule still fires on the cleaned panel: {after}")
    out["cleaning_after"] = {"cells_firing_after_cleaning": after,
                             "nonpositive_open_interest_cells": int((panels["open_interest"] <= 0).to_numpy().sum()
                                                                    + (panels["open_interest_value"] <= 0).to_numpy().sum())}
    n, diff, worst = 0, 0, 0.0
    for sym in close.columns:
        engine = data_io._daily_funding_from(C.RAW_FUNDING_FB / f"{sym}.parquet")
        if engine is None:
            continue
        mine = panels["funding_rate"][sym].dropna()
        common = mine.index.intersection(engine.dropna().index)
        d = (mine.reindex(common) - engine.reindex(common)).abs()
        n, diff, worst = n + len(common), diff + int((d > 1e-12).sum()), max(worst, float(d.max()) if len(d) else 0.0)
    out["funding_vs_engine_binance_prints"] = {
        "day_cells": n, "differing_above_1e-12": diff, "max_abs_diff": worst,
        "rule": "daily sum of settlements in [D 00:00, D + 1 00:00) UTC (resample 1D, label and closed left)"}
    with np.errstate(all="ignore"):
        imp = panels["open_interest_value"] / panels["open_interest"]
        dev = lambda x: np.abs(x.to_numpy(dtype=float)).ravel()   # noqa: E731
        d0, d1 = dev(imp / close - 1), dev(imp / close.shift(1) - 1)
    fin = np.isfinite(d0) & np.isfinite(d1)
    out["metrics_snapshot_timing"] = {
        "cells": int(fin.sum()),
        "median_abs_implied_price_vs_close_D": float(np.median(d0[fin])),
        "median_abs_implied_price_vs_close_D_minus_1": float(np.median(d1[fin])),
        "p99_abs_implied_price_vs_close_D": float(np.percentile(d0[fin], 99)),
        "rule": "the last non-missing 5-minute snapshot with create_time (UTC) in [D 00:00, D + 1 00:00), an "
                "earlier one when that day's archive file is partial: resample 1D, label and closed left, last"}
    ev = _evaluator(panels)
    vw = _values(ev, "vwap")
    lo, hi = panels["low"].to_numpy(dtype=float), panels["high"].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        rel = np.abs(close.to_numpy(dtype=float) / vw - 1.0)
    fin = np.isfinite(rel)
    out["vwap"] = {"collapses_onto_close": vwap_collapses(panels), "cells": int(fin.sum()),
                   "median_abs_close_over_vwap_minus_1": float(np.median(rel[fin])),
                   "share_inside_low_high": float(((vw >= lo * (1 - 1e-9)) & (vw <= hi * (1 + 1e-9)))[fin].mean())}
    ratios, corrs, days = [], [], 0
    for sym in close.columns:
        okx = data_io.ohlcv(sym)
        if okx is None:
            continue
        j = pd.concat([okx["close"].rename("o"), close[sym].rename("b")], axis=1, sort=True).dropna()
        r = np.log(j).diff().dropna()
        ratios.append(float((j["b"] / j["o"]).median()))
        corrs.append(float(r["o"].corr(r["b"])))
        days += len(j)
    out["okx_close_agreement"] = {"symbols": len(ratios), "days": days,
                                  "median_price_ratio_min_max": [min(ratios), max(ratios)],
                                  "daily_return_corr_min_max": [min(corrs), max(corrs)]}
    cov = {}
    for f in BINANCE_FIELDS:
        firsts = panels[f].apply(lambda s: s.first_valid_index()).dropna()
        cov[f] = {"cells": int(panels[f].notna().sum().sum()),
                  "first_valid_min_max": [str(min(firsts).date()), str(max(firsts).date())] if len(firsts) else None}
    out["coverage"] = cov
    return out


# --------------------------------------------------------------------------- catalog and classification

def _catalog_uri() -> str:
    """The read-only SQLite URI of the library catalog. Stops when the catalog is absent."""
    if not CATALOG.is_file():
        raise SystemExit("libfactors: no catalog (library/catalog/factors.db). Run ./setup.sh first.")
    return CATALOG.resolve().as_uri() + "?mode=ro"


def load_catalog() -> pd.DataFrame:
    """The 28,904 library definitions, read-only (SQLite mode=ro)."""
    con = sqlite3.connect(_catalog_uri(), uri=True)
    try:
        return pd.read_sql_query(
            "SELECT factor_id, name, family, category, expression, raw_inputs, needs_derivs, lookback "
            "FROM factors ORDER BY factor_id", con)
    finally:
        con.close()


@cache
def expr_atoms(expression: str) -> tuple[frozenset, frozenset]:
    """(operators, field names) an expression uses, derived fields expanded recursively.

    Operators include 'bench', the binary operators as 'binop_Add' etc. and unary minus;
    names include raw and derived fields. Used for the vwap flag and for choosing a
    future-mutation sample that covers every operator and field."""
    ops: set[str] = set()
    names: set[str] = set()
    tree = parse(expression)
    funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            ops.add(n.func.id)
        elif isinstance(n, ast.BinOp):
            ops.add("binop_" + type(n.op).__name__)
        elif isinstance(n, ast.UnaryOp):
            ops.add("unary_" + type(n.op).__name__)
        elif isinstance(n, ast.Name) and id(n) not in funcs:
            names.add(n.id)
            if n.id in DERIVED_BY_NAME:
                o, nm = expr_atoms(DERIVED_BY_NAME[n.id].expression)
                ops |= o
                names |= nm
    return frozenset(ops), frozenset(names)


def classify(cat: pd.DataFrame, supplied_fields: tuple = SUPPLIED) -> pd.DataFrame:
    """One row per definition: group, missing fields and the proxy flags.

    computable     every raw input is a supplied field
    needs_premium  only the premium index is missing (a basis proxy would change them)
    missing_fields anything else (trades, taker flow, long/short ratios, ...)
    The catalog's raw_inputs column is cross-checked against the parser's own
    dependency walk, so a stale catalog column cannot misroute a factor.
    `supplied_fields` is the OKX panel's SUPPLIED by default (the flags describe that
    panel); the binance source passes BINANCE_FIELDS and replaces the flags
    (binance_status())."""
    supplied = set(supplied_fields)
    rows = []
    for r in cat.itertuples(index=False):
        inputs = set(r.raw_inputs.split(","))
        deps = set(raw_dependencies(r.expression))
        if deps != inputs:
            raise SystemExit(f"libfactors: {r.factor_id} raw_inputs {sorted(inputs)} != parsed {sorted(deps)}")
        if not inputs <= RAW_FIELD_NAMES:
            raise SystemExit(f"libfactors: {r.factor_id} names an unknown raw field")
        missing = sorted(inputs - supplied)
        group = "computable" if not missing else ("needs_premium" if missing == [PREMIUM] else "missing_fields")
        names = expr_atoms(r.expression)[1]
        comp = group == "computable"
        rows.append({
            "factor_id": r.factor_id, "name": r.name, "family": r.family, "category": r.category,
            "group": group, "missing_fields": ",".join(missing),
            "oi_short_history": comp and bool(inputs & set(OI_FIELDS)),
            "vwap_proxy": comp and bool(names & set(VWAP_FIELDS)),
            "oi_base_proxy": comp and "open_interest" in inputs,
            "lookback": int(r.lookback),
        })
    return pd.DataFrame(rows)


def cache_order(expressions: pd.Series) -> np.ndarray:
    """A deterministic order that keeps factors sharing input series adjacent, so the
    Evaluator's LRU cache of sub-expressions hits more often than in a random order. The
    compute prints the hit rate of its own run."""
    def key(e: str) -> tuple:
        tree = parse(e)
        return tuple(sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})), e
    keys = [key(e) for e in expressions]
    return np.array(sorted(range(len(keys)), key=lambda i: keys[i]), dtype=np.int64)


# --------------------------------------------------------------------------- compute (worker side)

class _CountingCache(OrderedDict):
    """The Evaluator's LRU cache, counting hits (get() returning a frame) and misses."""
    hits = 0
    misses = 0

    def get(self, key, default=None):
        v = super().get(key, default)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v


_W: dict = {}


def _evaluator(panels: dict) -> Evaluator:
    ev = Evaluator(panels, benchmark=BENCHMARK, cache_size=CACHE_SIZE)
    ev._cache = _CountingCache()      # same OrderedDict API; only counts on top
    return ev


def _init_worker(panels: dict, cuts: tuple) -> None:
    _W.clear()
    _W["ev"] = _evaluator(panels)
    idx = next(iter(panels.values())).index
    _W["sun"] = sunday_positions(idx)
    _W["cuts"] = []
    for cut in cuts:
        tp = {k: v.loc[:pd.Timestamp(cut, tz="UTC")] for k, v in panels.items()}
        tix = next(iter(tp.values())).index
        _W["cuts"].append((cut, _evaluator(tp), sunday_positions(tix)))


def _values(ev: Evaluator, expression: str) -> np.ndarray:
    # NaN windows, logs of 0 and 0/0 are expected (the library maps them to NaN); their
    # RuntimeWarnings are silenced here only, never process-wide
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        out = ev(expression)
    return out.reindex(index=ev.index, columns=ev.columns).to_numpy(dtype=float)


def pit_status(equal: bool, finite_compared: int) -> str:
    """The PIT check of one factor: 'pass' only when every compared cell is equal AND at
    least one finite cell was compared; equal with nothing finite compared proves nothing
    ('no_finite_cells'); any difference or error is 'fail'."""
    if not equal:
        return "fail"
    return "pass" if finite_compared > 0 else "no_finite_cells"


def _one(fid: str, expression: str) -> dict:
    """Full computation, float64 Sunday values, and the PIT proof at every cut."""
    res = {"factor_id": fid}
    try:
        full = _values(_W["ev"], expression)[_W["sun"]]
    except Exception as exc:  # noqa: BLE001 - recorded per factor, never dropped
        res.update(ok=False, error_class=type(exc).__name__)
        return res
    finite = np.isfinite(full)
    # float64, exactly as computed (+-inf stored as NaN): the screen ranks these values
    res.update(ok=True, error_class="", values=np.where(finite, full, np.nan), finite_cells=int(finite.sum()))
    first = np.flatnonzero(finite.any(axis=1))
    res["first_sunday_pos"] = int(first[0]) if first.size else -1
    pit_ok, finite_compared = True, 0
    for j, (_cut, ev, pos) in enumerate(_W["cuts"], start=1):
        ref = full[: len(pos)]                     # Sundays <= cut are the leading Sundays
        try:
            trunc = _values(ev, expression)[pos]
        except Exception as exc:  # noqa: BLE001
            pit_ok = False
            res[f"pit_error_t{j}"] = type(exc).__name__
            res[f"pit_cells_t{j}"] = 0
            res[f"pit_finite_t{j}"] = 0
            continue
        same = np.isclose(trunc, ref, rtol=PIT_RTOL, atol=PIT_ATOL, equal_nan=True)
        pit_ok &= bool(same.all())
        res[f"pit_cells_t{j}"] = int(ref.size)
        res[f"pit_finite_t{j}"] = int(np.isfinite(ref).sum())
        res[f"pit_diff_cells_t{j}"] = int((~same).sum())
        finite_compared += res[f"pit_finite_t{j}"]
    res["pit_ok"] = pit_ok
    res["pit_check"] = pit_status(pit_ok, finite_compared)
    return res


def _chunk(items: list) -> tuple[list, tuple]:
    out = [_one(fid, expr) for fid, expr in items]
    caches = [_W["ev"]._cache] + [ev._cache for _, ev, _ in _W["cuts"]]
    return out, (os.getpid(), sum(c.hits for c in caches), sum(c.misses for c in caches))


def compute_values(panels: dict, factors: pd.DataFrame, workers: int = MAX_WORKERS,
                   cuts: tuple = PIT_CUTS, log=print) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Evaluate `factors` (factor_id, expression) on `panels`, with the PIT proof at `cuts`.

    Returns (per-factor results in the input order, float64 values (k, n_sundays, n_symbols)
    with NaN for failed factors, cache statistics). Deterministic: every factor's result
    is independent of cache state and of which worker computed it."""
    order = cache_order(factors["expression"])
    items = [(factors["factor_id"].iat[i], factors["expression"].iat[i]) for i in order]
    chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
    idx = next(iter(panels.values())).index
    n_sun, n_sym = len(sunday_positions(idx)), len(next(iter(panels.values())).columns)
    pos_of = {fid: i for i, fid in enumerate(factors["factor_id"])}
    values = np.full((len(factors), n_sun, n_sym), np.nan, dtype=np.float64)
    rows: list = [None] * len(factors)
    cache: dict = {}
    t0, done = time.time(), 0
    workers = max(1, min(int(workers), MAX_WORKERS, len(chunks) or 1))
    if workers == 1:
        _init_worker(panels, cuts)
        results = map(_chunk, chunks)
        pool = None
    else:
        pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(panels, cuts))
        results = pool.map(_chunk, chunks)
    try:
        for out, (pid, hits, misses) in results:
            cache[pid] = (hits, misses)           # cumulative per worker: keep the latest
            for r in out:
                i = pos_of[r["factor_id"]]
                v = r.pop("values", None)
                if v is not None:
                    values[i] = v
                rows[i] = r
            done += len(out)
            if log and (done == len(items) or done % 3000 < CHUNK):
                log(f"  computed {done:,}/{len(items):,} ({time.time() - t0:.0f}s)")
    finally:
        if pool is not None:
            pool.shutdown()
    hits = sum(h for h, _ in cache.values())
    misses = sum(m for _, m in cache.values())
    stats = {"workers": workers, "chunks": len(chunks), "cache_hits": hits, "cache_misses": misses,
             "cache_hit_rate": round(hits / max(hits + misses, 1), 4)}
    return pd.DataFrame(rows), values, stats


# --------------------------------------------------------------------------- write boundary

def _guard(path: Path, source: str = DEFAULT_SOURCE) -> Path:
    """Refuse any write that is not a plain file inside the source's real output folder.

    config.guard_write first (protected folders are read-only); then the literal
    folders must not be symlinks (a symlink to data/trials would otherwise turn a write
    here into a write on the trial ledger): data/library_on_engine, and for the binance source
    also its binance/ subfolder; the target must sit directly in the source's folder or in
    its values/ subfolder (so an okx write can never land in binance/, nor the reverse),
    and an existing target must be a regular file with one link."""
    C.guard_write(path)
    root = out_dir(source)
    base = out_dir(DEFAULT_SOURCE)
    p = Path(path)
    for d in dict.fromkeys((base, root, root / VALUES_NAME)):
        if d.is_symlink() or (d.exists() and not d.is_dir()):
            raise SystemExit(f"libfactors: refusing to write: data/{OUT_NAME}/{d.relative_to(base).as_posix()} "
                             "is not a real directory")
    if p.parent not in (root, root / VALUES_NAME) or p.name in ("", ".", ".."):
        raise SystemExit(f"libfactors: refusing to write outside data/{root.relative_to(Path(C.DATA)).as_posix()}: "
                         f"{p.name}")
    if os.path.lexists(p):
        st = os.lstat(p)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
            raise SystemExit(f"libfactors: refusing to write over {p.name}: not a plain single-link file")
    return p


def _make_dirs(source: str = DEFAULT_SOURCE) -> None:
    root = out_dir(source)
    _guard(root / "manifest.json", source)
    root.mkdir(parents=True, exist_ok=True)
    (root / VALUES_NAME).mkdir(exist_ok=True)
    _guard(root / VALUES_NAME / "x", source)          # re-check: created by someone else in between?


def _privacy(text: str, what: str) -> None:
    from webexport.common import privacy_hits
    hits = privacy_hits(text)
    if hits:
        raise SystemExit(f"libfactors: refusing to write {what}: private strings ({', '.join(hits)})")


def _write_table(table, path: Path, compression: str = "snappy", source: str = DEFAULT_SOURCE) -> None:
    """Snappy by default for the small files; the value shards use zstd."""
    import pyarrow.parquet as pq
    p = _guard(path, source)
    tmp = _guard(p.with_name(f".{p.name}.tmp"), source)
    pq.write_table(table, tmp, compression=compression)
    os.replace(tmp, p)


def _write_frame(df: pd.DataFrame, path: Path, source: str = DEFAULT_SOURCE) -> None:
    import pyarrow as pa
    strings = df.select_dtypes(include=["object", "string"])
    _privacy("\n".join(strings.astype(str).to_numpy().ravel().tolist() + [str(c) for c in df.columns]), path.name)
    _write_table(pa.Table.from_pandas(df, preserve_index=False), path, source=source)


def _write_json(obj: dict, path: Path, source: str = DEFAULT_SOURCE) -> None:
    text = json.dumps(obj, indent=1, sort_keys=False, ensure_ascii=True, allow_nan=False) + "\n"
    _privacy(text, path.name)
    p = _guard(path, source)
    tmp = _guard(p.with_name(f".{p.name}.tmp"), source)
    tmp.write_text(text, encoding="ascii")
    os.replace(tmp, p)


def _remove(path: Path, source: str = DEFAULT_SOURCE) -> None:
    if os.path.lexists(path):
        os.unlink(_guard(path, source))


# --------------------------------------------------------------------------- fingerprints

def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_fingerprints() -> dict:
    """sha256 of every input file, per folder (relative names only, never a path)."""
    out = {}
    syms = [C.sym_to_file(s) for s in C.UNIVERSE]
    for rel, folder in (("raw/ohlcv", C.RAW_OHLCV), ("raw/funding", C.RAW_FUNDING),
                        (f"raw/funding_{C.FUNDING_FALLBACK_EXCHANGE_ID}", C.RAW_FUNDING_FB),
                        ("raw/open_interest", C.RAW_OI)):
        files = [folder / f"{s}.parquet" for s in syms if (folder / f"{s}.parquet").is_file()]
        lines = [f"{f.name} {_sha_file(f)}" for f in files]
        out[rel] = {"files": len(files), "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}
    lt = label_path()
    out[LABEL_INPUT] = {"files": 1, "sha256": _sha_file(lt)} if lt.is_file() else None
    out["library/catalog/factors.db"] = {"files": 1, "sha256": _sha_file(CATALOG)}
    return out


def code_fingerprint(code_files: tuple = CODE_FILES) -> dict:
    repo = C.ROOT.parent
    files = {rel: _sha_file(repo / rel) for rel in code_files}
    return {"files": files, "sha256": hashlib.sha256(
        "\n".join(f"{k} {v}" for k, v in files.items()).encode()).hexdigest()}


def changed_code_files(man: dict) -> list:
    """The code files whose content differs from the hash that the manifest recorded, or that do
    not exist. Read-only; [] when the code on disk computed the outputs. --status prints them:
    the code changed after the computation of the outputs, also where the numbers still hold."""
    rec = (man.get("code") or {}).get("files") or {}
    repo = C.ROOT.parent
    return [f for f, h in rec.items() if not (repo / f).is_file() or _sha_file(repo / f) != h]


def changed_inputs(man: dict, source: str = DEFAULT_SOURCE) -> list | None:
    """The inputs whose fingerprint differs from the one that the manifest recorded at the
    computation, with the same rule as --screen (read-only). None when an input cannot be read
    (for example, the library catalog or the Binance panel is not on disk)."""
    rec = man.get("inputs") or {}
    okx = _check_source(source) == "okx"
    # the same "missing" rule as load_raw_binance: no folder, or no merged file in it
    if not okx and (not BINANCE_PANEL_DIR.is_dir() or not any(BINANCE_PANEL_DIR.glob("*.csv.gz"))):
        return None
    try:
        now = input_fingerprints() if okx else binance_inputs()
    except (OSError, ValueError, EOFError):
        return None
    if okx:
        return [k for k, v in rec.items() if k != LABEL_INPUT and now.get(k) != v]
    return [k for k in ("binance_panel", "library/catalog/factors.db", "okx_screen") if now.get(k) != rec.get(k)]


def _print_states(man: dict, source: str, refresh: str) -> None:
    """--status: the state of the code and of the inputs against the manifest."""
    changed = changed_code_files(man)
    if changed:
        print(f"  code: changed after the computation: {', '.join(changed)} (the manifest's code hash differs from "
              f"today's files); {refresh} refresh the outputs")
    else:
        print("  code: the same as at the computation")
    inp = changed_inputs(man, source)
    if inp is None:
        print("  inputs: not checked (an input is not on disk)")
    elif inp:
        print(f"  inputs: changed since the computation: {', '.join(inp)}; {refresh} refresh the outputs")
    else:
        print("  inputs: the same as at the computation")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- values shards

def _shard_name(k: int) -> str:
    return f"values_{k:03d}.parquet"


def write_values(ids: list, values: np.ndarray, sundays: pd.DatetimeIndex, symbols: list,
                 source: str = DEFAULT_SOURCE) -> dict:
    """Weekly float64 values, SHARD_FACTORS factor columns per file (factor_id order).

    Rows are (sunday, symbol), Sunday-major. The key column is "sunday", not "date". A
    check that changes each dated row of a data copy uses the date/ts columns, so it does
    not change these shards. Old shards not rewritten are removed, so the folder never
    mixes two runs."""
    import pyarrow as pa
    _make_dirs(source)
    order = np.argsort(np.array(ids, dtype=object))
    n_sun, n_sym = len(sundays), len(symbols)
    sunday_col = pa.array(np.repeat(sundays.tz_convert("UTC").tz_localize(None).to_numpy(), n_sym),
                        type=pa.timestamp("ms")).cast(pa.timestamp("ms", tz="UTC"))
    sym_col = pa.array(np.tile(np.array(symbols, dtype=object), n_sun), type=pa.string())
    shard_of, written = {}, []
    for k in range(0, len(order), SHARD_FACTORS):
        part = order[k:k + SHARD_FACTORS]
        name = _shard_name(k // SHARD_FACTORS)
        cols = {"sunday": sunday_col, "symbol": sym_col}
        for i in part:
            cols[ids[i]] = pa.array(np.asarray(values[i], dtype=np.float64).reshape(-1), type=pa.float64())
            shard_of[ids[i]] = name
        _write_table(pa.table(cols), out_dir(source) / VALUES_NAME / name, compression="zstd", source=source)
        written.append(name)
    for f in (out_dir(source) / VALUES_NAME).glob("values_*.parquet"):
        if f.name not in written:
            _remove(f, source)
    return shard_of


def has_values(source: str = DEFAULT_SOURCE) -> bool:
    """True when the source's git-ignored values/ folder holds at least one shard (the status
    and manifest files can exist without values/). Read-only."""
    d = out_dir(source) / VALUES_NAME
    return d.is_dir() and any(d.glob("values_*.parquet"))


def read_values(ids: list, sundays: pd.DatetimeIndex | None = None,
                source: str = DEFAULT_SOURCE) -> tuple[np.ndarray, list, pd.DatetimeIndex]:
    """(values (k, n_sundays, n_symbols) float64, symbols, sundays) for factor ids, read
    from the source's shards; `sundays` selects rows (all stored Sundays by default)."""
    import pyarrow.parquet as pq
    status = pd.read_parquet(out_dir(source) / "status.parquet", columns=["factor_id", "values_shard"])
    shard_of = dict(zip(status["factor_id"], status["values_shard"]))
    by_shard: dict = {}
    for i, fid in enumerate(ids):
        if not shard_of.get(fid):
            raise KeyError(f"{fid} has no stored values")
        by_shard.setdefault(shard_of[fid], []).append(i)
    out = None
    symbols: list = []
    days = None
    for name, pos in sorted(by_shard.items()):
        t = pq.read_table(out_dir(source) / VALUES_NAME / name, columns=["sunday", "symbol"] + [ids[i] for i in pos])
        d = pd.DatetimeIndex(t.column("sunday").to_pandas())
        syms = t.column("symbol").to_pylist()
        n_sym = len(dict.fromkeys(syms))
        all_days = d[::n_sym]
        symbols = syms[:n_sym]
        sel = np.arange(len(all_days)) if sundays is None else all_days.get_indexer(sundays)
        if (sel < 0).any():
            raise KeyError("a requested Sunday is not stored")
        days = all_days[sel]
        if out is None:
            out = np.full((len(ids), len(sel), n_sym), np.nan, dtype=np.float64)
        for i in pos:
            col = t.column(ids[i])
            # full precision only: a float32 shard would rank rounded values
            if str(col.type) != "double":
                raise ValueError(f"{ids[i]}: stored values are {col.type}, not float64; run --compute again")
            out[i] = col.to_numpy(zero_copy_only=False).reshape(len(all_days), n_sym)[sel]
    return out, symbols, days


# --------------------------------------------------------------------------- screen (pure)

def _refuse_late(dates, what: str) -> None:
    if len(dates) and pd.Timestamp(max(dates)) > LABEL_CUTOFF:
        raise LabelLeak(f"{what} after the {LABEL_CUTOFF.date()} row reached the G screen; the I zone is sealed")


def g_labels(ft: pd.DataFrame | None = None) -> pd.DataFrame:
    """Sunday rows of the label table through the 2024-06-09 row: date, symbol, fwd_ret_7.

    ft: the label table (data_io.label_table() when None). Rows present = the engine's
    eligible rows (labels.py has no row in a symbol's warm-up). Every later row is cut
    here, before anything else looks at it."""
    ft = data_io.label_table(columns=["date", "symbol", LABEL]) if ft is None else ft
    lab = ft.loc[ft["date"] <= LABEL_CUTOFF, ["date", "symbol", LABEL]]
    lab = lab[lab["date"].dt.dayofweek == 6].reset_index(drop=True)
    _refuse_late(lab["date"], "a label")
    return lab


def pit_tilt(close: pd.DataFrame, shrink_days: float = TILT_SHRINK_DAYS) -> pd.DataFrame:
    """The registered pit_symbol_tilt on every day: the shrunk expanding mean of each
    symbol's daily excess return through D.

    excess(D) = r(D) - the equal-weight mean of r(D) over the perps with a return that day,
    r(D) = close(D) / close(D - 1) - 1 (D's close is known at D's close). With n excess
    returns through D: tilt(D) = sum / (n + shrink_days) = n / (n + shrink_days) x mean, so
    a newly listed symbol's tilt starts near 0 (shrink_days = 0 gives the plain mean). NaN
    before the symbol's first return."""
    r = close / close.shift(1) - 1.0
    ex = r.sub(r.mean(axis=1), axis=0)
    n = ex.notna().cumsum()
    s = ex.fillna(0.0).cumsum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / (n + float(shrink_days))).where(n > 0)


def _label_mean_tilt(labels: pd.DataFrame, symbols: list, sundays: pd.DatetimeIndex) -> np.ndarray:
    """The label_mean tilt (only for the sensitivity report, never the screen's
    statistic): the symbol's mean fwd_ret_7 over Sundays S <= D - 7 (labels complete by
    D's close), 0 with fewer than TILT_MIN_LABELS such labels, unshrunk."""
    lw = labels.pivot(index="date", columns="symbol", values=LABEL).reindex(columns=symbols).sort_index()
    lab = lw.to_numpy(dtype=float)
    fin = np.isfinite(lab)
    csum = np.cumsum(np.where(fin, lab, 0.0), axis=0)
    ccnt = np.cumsum(fin, axis=0)
    tilt = np.zeros((len(sundays), len(symbols)))
    for t, d in enumerate(sundays):
        k = lw.index.searchsorted(d - pd.Timedelta(days=7), side="right")   # rows S <= D - 7
        if k > 0:
            n = ccnt[k - 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                tilt[t] = np.where(n >= TILT_MIN_LABELS, csum[k - 1] / np.maximum(n, 1), 0.0)
    return tilt


def regressors(labels: pd.DataFrame, panels_cut: dict, symbols: list, sundays: pd.DatetimeIndex,
               tilt: str = "pit", shrink_days: float | None = TILT_SHRINK_DAYS) -> dict[str, np.ndarray | None]:
    """The point-in-time rIC regressors on each screen Sunday, (T, m) arrays.

    tilt  "pit" (the registered pit_symbol_tilt, see pit_tilt(); shrink_days 0 = unshrunk),
          "label_mean" (sensitivity report only) or "none"
    beta  OLS beta of its daily returns on the BTC perp's over the 90 returns up to D
    size  log of its 30-day mean USDT volume (close x volume) up to D
    Only bars through the 2024-06-09 row (panels_cut) and labels through that row are read;
    anything later raises LabelLeak."""
    _refuse_late(labels["date"], "a label")
    for k, v in panels_cut.items():
        _refuse_late(v.index, f"a {k} bar")
    close = panels_cut["close"].reindex(columns=symbols)
    if tilt == "pit":
        tl = pit_tilt(close, shrink_days).reindex(sundays).to_numpy(dtype=float)
    elif tilt == "label_mean":
        tl = _label_mean_tilt(labels, symbols, sundays)
    elif tilt == "none":
        tl = None
    else:
        raise ValueError(f"unknown tilt {tilt!r}")
    r = close / close.shift(1) - 1.0
    rb = r[BENCHMARK]
    m = lambda x: x.rolling(BETA_DAYS, min_periods=BETA_DAYS).mean()   # noqa: E731
    cov = m(r.mul(rb, axis=0)).sub(m(r).mul(m(rb), axis=0))
    var = m(rb * rb) - m(rb) ** 2
    beta = cov.div(var.where(var > 0), axis=0)
    qv = panels_cut["quote_volume"].reindex(columns=symbols)
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.log(qv.rolling(SIZE_DAYS, min_periods=SIZE_DAYS).mean())
    pick = lambda x: x.reindex(sundays).to_numpy(dtype=float)   # noqa: E731
    return {"tilt": tl, "beta": pick(beta), "size": pick(size)}


def snap_ties(values: np.ndarray) -> np.ndarray:
    """(k, T, m) factor values -> float64 integers with the same order, in which values
    within 10^-TIE_DIGITS of that (factor, Sunday)'s largest |value| are equal.

    Each value is divided by the largest finite |value| among the m names that Sunday and
    rounded to TIE_DIGITS decimals (an integer grid up to 10^12, exact in float64). Ranks
    of the result are the ranks of the values, except that floating-point noise (a few
    ulp, far below 1e-12 of the cross-section's scale) does not order names: values that
    are equal in exact arithmetic tie, whatever the evaluation order left in their last
    bits. Uses the factor's own values only (no label), and is idempotent. NaN stays NaN."""
    v = np.asarray(values, dtype=np.float64)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")              # all-NaN rows: nanmax warns, the scale is unused
        scale = np.nanmax(np.abs(v), axis=-1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return np.round(v / scale * 10.0 ** TIE_DIGITS)


def _rank_rows(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(a, axis=1, method="average")


def _corr_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of x with the vector y (or the matching row of a
    matrix y). NaN where either side has no dispersion."""
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    num = (xc * yc).sum(axis=1)
    sx = np.sqrt((xc * xc).sum(axis=1))
    sy = np.sqrt((yc * yc).sum(axis=-1))
    den = sx * sy
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    scale = np.sqrt(x.shape[1]) * x.shape[1]          # ranks are O(n): a relative zero test
    out[~(sx > 1e-9 * scale) | ~(np.broadcast_to(sy, sx.shape) > 1e-9 * scale)] = np.nan
    return out


def _groups(valid: np.ndarray):
    """Factors grouped by their valid-name pattern on one Sunday: [(names, factor rows)]."""
    if valid.shape[0] == 0:
        return []
    pats, inv = np.unique(valid, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    return [(np.flatnonzero(p), np.flatnonzero(inv == g)) for g, p in enumerate(pats)]


def _residual_maker(X: np.ndarray, ridge: float = RIDGE_LAMBDA) -> np.ndarray:
    """M such that M @ y is the residual of a cross-section y on [1, regressors].

    Regressors are z-scored (population sd) over the names of that Sunday; the intercept
    is unpenalised (handled by centring) and the slopes minimise
    ||y - a - Z b||^2 + ridge ||b||^2, the registered ridge residualisation
    (eval_config [regressors] ridge_lambda). ridge = 0 is OLS (pinv). A constant
    regressor becomes a zero column, which both forms simply ignore."""
    n = X.shape[0]
    sd = X.std(axis=0)
    Z = np.where(sd > 0, (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0), 0.0)
    H = Z @ np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T) if ridge > 0 else Z @ np.linalg.pinv(Z)
    return np.eye(n) - 1.0 / n - H


def _reg_stack(regs: dict) -> np.ndarray:
    """(T, m, k) array of the regressors present (a "none" tilt is left out)."""
    return np.stack([regs[k] for k in REG_KEYS if regs.get(k) is not None], axis=-1)


def weekly_ic(values: np.ndarray, label: np.ndarray, eligible: np.ndarray, regs: dict,
              ridge: float = RIDGE_LAMBDA, min_names: int = MIN_NAMES
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per factor and Sunday: Spearman IC, residual IC, valid names and base names.

    values (k, T, m) float, label (T, m), eligible (T, m) bool, regs[...] (T, m). A Sunday
    with fewer than `min_names` valid names gives no IC for that factor."""
    k, T, m = values.shape
    X = _reg_stack(regs)                                                      # (T, m, r)
    base = eligible & np.isfinite(label) & np.isfinite(X).all(axis=-1)
    ic = np.full((k, T), np.nan)
    ric = np.full((k, T), np.nan)
    n_valid = np.zeros((k, T), dtype=np.int32)
    for t in range(T):
        V = values[:, t, :]
        valid = base[t][None, :] & np.isfinite(V)
        n_valid[:, t] = valid.sum(axis=1)
        for cols, rows in _groups(valid):
            if cols.size < min_names:
                continue
            ry = _rank_rows(label[t, cols][None, :])[0]
            rf = _rank_rows(V[np.ix_(rows, cols)].astype(float))
            ic[rows, t] = _corr_rows(rf, ry)
            M = _residual_maker(X[t][cols], ridge)
            e = (rf - rf.mean(axis=1, keepdims=True)) @ M.T
            r = _corr_rows(_rank_rows(e), ry)
            # a factor rank fully explained by the regressors leaves a residual of float
            # noise; its ranks would be arbitrary, so the rIC is undefined there
            tiny = np.sqrt((e * e).sum(axis=1)) <= 1e-9 * np.sqrt(((rf - rf.mean(axis=1, keepdims=True)) ** 2).sum(axis=1))
            r[tiny] = np.nan
            ric[rows, t] = r
    return ic, ric, n_valid, base.sum(axis=1)


def persistence(values: np.ndarray, eligible: np.ndarray, sundays: pd.DatetimeIndex,
                min_names: int = MIN_NAMES) -> tuple[np.ndarray, np.ndarray]:
    """Mean week-over-week cross-sectional Spearman of each factor, over consecutive screen
    Sundays exactly 7 days apart (the pairs around the embargo Sunday are skipped), on the
    names eligible both weeks with finite values both weeks (at least `min_names`)."""
    k = values.shape[0]
    acc = np.zeros(k)
    cnt = np.zeros(k, dtype=np.int32)
    for t in range(1, len(sundays)):
        if sundays[t] - sundays[t - 1] != pd.Timedelta(days=7):
            continue
        a, b = values[:, t - 1, :], values[:, t, :]
        valid = (eligible[t - 1] & eligible[t])[None, :] & np.isfinite(a) & np.isfinite(b)
        for cols, rows in _groups(valid):
            if cols.size < min_names:
                continue
            rho = _corr_rows(_rank_rows(a[np.ix_(rows, cols)].astype(float)),
                             _rank_rows(b[np.ix_(rows, cols)].astype(float)))
            ok = np.isfinite(rho)
            acc[rows[ok]] += rho[ok]
            cnt[rows[ok]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan), cnt


def tstats(x: np.ndarray, min_n: int = 3) -> dict:
    """n, mean, sd (ddof 1), t = mean / sd * sqrt(n), two-sided Student-t p (n - 1 df)."""
    from scipy.stats import t as tdist
    fin = np.isfinite(x)
    n = fin.sum(axis=1)
    xs = np.where(fin, x, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = xs.sum(axis=1) / n
        sd = np.sqrt(np.where(fin, (x - mean[:, None]) ** 2, 0.0).sum(axis=1) / (n - 1))
        t = mean / sd * np.sqrt(n)
    ok = (n >= min_n) & (sd > 0)
    mean, sd, t = np.where(n > 0, mean, np.nan), np.where(n > 1, sd, np.nan), np.where(ok, t, np.nan)
    p = np.where(ok, 2.0 * tdist.sf(np.abs(np.nan_to_num(t)), np.maximum(n - 1, 1)), np.nan)
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p": p}


def pooled_vectors(values: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Unit vectors of centred within-Sunday percentile ranks (zeros where missing), one
    row per factor, so V @ V.T is the pooled within-date rank correlation."""
    k, T, m = values.shape
    V = np.zeros((k, T * m), dtype=np.float64)
    for t in range(T):
        x = values[:, t, :].astype(float)
        valid = base[t][None, :] & np.isfinite(x)
        for cols, rows in _groups(valid):
            if cols.size < 2:
                continue
            r = _rank_rows(x[np.ix_(rows, cols)]) / cols.size
            r -= r.mean(axis=1, keepdims=True)
            block = np.zeros((rows.size, m))
            block[:, cols] = r
            V[rows, t * m:(t + 1) * m] = block
    norm = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.where(norm > 0, norm, 1.0)


def dedup(V: np.ndarray, strength: np.ndarray, rho: float = DEDUP_RHO) -> tuple[np.ndarray, np.ndarray]:
    """Greedy clusters by |pooled rank correlation| >= rho, strongest first (by `strength`).

    The absolute value: a factor and its sign flip are one signal. Returns (cluster id,
    representative flag) per row. Deterministic: ties in strength keep input order."""
    k = V.shape[0]
    cluster = np.full(k, -1, dtype=np.int32)
    rep = np.zeros(k, dtype=bool)
    order = np.argsort(-strength, kind="stable")
    c = 0
    for i in order:
        if cluster[i] >= 0:
            continue
        free = np.flatnonzero(cluster < 0)
        sims = np.abs(V[free] @ V[i])
        cluster[free[sims >= rho]] = c
        cluster[i] = c
        rep[i] = True
        c += 1
    return cluster, rep


def label_grid(labels: pd.DataFrame, sundays: pd.DatetimeIndex, symbols: list) -> tuple[np.ndarray, np.ndarray]:
    """(label (T, m) float, eligible (T, m) bool) on the screen Sundays; eligible = the row
    is present in the label table."""
    lw = labels.pivot(index="date", columns="symbol", values=LABEL).reindex(index=sundays, columns=symbols)
    elig = labels.assign(_x=1.0).pivot(index="date", columns="symbol", values="_x") \
        .reindex(index=sundays, columns=symbols).notna().to_numpy()
    return lw.to_numpy(dtype=float), elig


def factor_stats(values: np.ndarray, symbols: list, labels: pd.DataFrame, panels_cut: dict,
                 variant: str = PRIMARY, with_persistence: bool = True) -> tuple[pd.DataFrame, dict]:
    """Per-factor statistics on G for EVERY input factor (no family filter, no BH).

    Columns, in input order: {ic, ric} x {G, G_fit, G_eval} x {n, mean, sd, t, p}, coverage,
    persistence, persistence_n. The full-G and G_fit p-values keep float64 (they feed BH:
    the pooled screen and the G_fit selection); the G_eval p is descriptive (float32).
    `values` are the full-precision (float64) factor values; every statistic uses the
    ranks of snap_ties(values), so storage precision and floating-point noise never move
    a statistic. Returns (frame, context with the base mask and names per Sunday). Pure:
    reads nothing from disk; a label or bar after the 2024-06-09 row raises LabelLeak."""
    _refuse_late(labels["date"], "a label")
    sundays, is_fit = g_sundays()
    if values.shape[1:] != (len(sundays), len(symbols)):
        raise ValueError("values must be (factors, G Sundays, symbols)")
    values = snap_ties(values)
    v = VARIANTS[variant]
    regs = regressors(labels, panels_cut, symbols, sundays, tilt=v["tilt"], shrink_days=v["shrink_days"])
    label, elig = label_grid(labels, sundays, symbols)
    ic, ric, n_valid, n_base = weekly_ic(values, label, elig, regs, ridge=v["ridge"], min_names=v["min_names"])
    out = {}
    for name, arr in (("ic", ic), ("ric", ric)):
        for zone, sel in (("", np.ones(len(sundays), bool)), ("_fit", is_fit), ("_eval", ~is_fit)):
            s = tstats(arr[:, sel])
            out[f"{name}{zone}_n"] = s["n"].astype(np.int16)
            for k2 in ("mean", "sd", "t"):
                out[f"{name}{zone}_{k2}"] = s[k2].astype(np.float32)
            out[f"{name}{zone}_p"] = s["p"] if zone in ("", "_fit") else s["p"].astype(np.float32)
    out["coverage"] = (n_valid.sum(axis=1) / max(int(n_base.sum()), 1)).astype(np.float32)
    if with_persistence:
        pers, pers_n = persistence(values, elig, sundays, min_names=v["min_names"])
        out["persistence"] = pers.astype(np.float32)
        out["persistence_n"] = pers_n.astype(np.int16)
    base = elig & np.isfinite(label) & np.isfinite(_reg_stack(regs)).all(-1)
    return pd.DataFrame(out), {"base": base, "n_base": n_base, "eligible": elig}


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg q-values over the finite entries of p (NaN elsewhere)."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    fin = np.isfinite(p)
    if fin.any():
        q[fin] = bh_qvalues(p[fin])
    return q


def fit_selection(df: pd.DataFrame) -> dict:
    """Replication the way the zone design intends: select on G_fit, check on G_eval.

    Selection = BH over the G_fit p-values of the screened family ({stat}_fit_q); the check
    is the selected factors' mean on G_eval (45 Sundays the selection never saw): same sign
    as on G_fit, and same sign with |t| >= 2. The base rate is the share of every screened
    factor (finite on both halves) whose G_eval mean has its G_fit sign, what a selection
    with no information would reach."""
    out = {"rule": ("Benjamini-Hochberg over the G_fit p-values of every screened factor; the selected "
                    "factors' sign and t on G_eval (which the selection never saw) are the check")}
    for name in ("ic", "ric"):
        fm = df[f"{name}_fit_mean"].to_numpy(dtype=float)
        em = df[f"{name}_eval_mean"].to_numpy(dtype=float)
        et = df[f"{name}_eval_t"].to_numpy(dtype=float)
        q = df[f"{name}_fit_q"].to_numpy(dtype=float)
        both = np.isfinite(fm) & np.isfinite(em) & (fm != 0)
        same = both & (np.sign(em) == np.sign(fm))
        row = {"n_tests": int(np.isfinite(df[f"{name}_fit_p"].to_numpy(dtype=float)).sum()),
               "base_pairs": int(both.sum()), "base_same_sign": int(same.sum())}
        for lev in Q_LEVELS:
            sel = q <= lev
            row[f"q{int(round(lev * 100)):02d}"] = {
                "selected": int(sel.sum()), "eval_same_sign": int((sel & same).sum()),
                "eval_same_sign_t2": int((sel & same & (np.abs(et) >= 2)).sum())}
        out[name] = row
    return out


def screen_frame(values: np.ndarray, factor_ids: list, symbols: list, labels: pd.DataFrame,
                 panels_cut: dict, variant: str = PRIMARY) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """The G-zone screen for factors whose weekly values on g_sundays() are `values`.

    labels: g_labels() (Sunday rows through 2024-06-09; a later row raises LabelLeak);
    panels_cut: build_panels(raw, through=LABEL_CUTOFF) for the regressors (a later bar
    raises LabelLeak). Returns (one row per factor with >= MIN_WEEKS weekly ICs on G,
    summary, the number of weekly ICs on G of EVERY input factor). Pure: reads nothing
    from disk, so the future-mutation test can run it twice."""
    stats, ctx = factor_stats(values, symbols, labels, panels_cut, variant)
    return screen_from_stats(stats, ctx, values, factor_ids)


def screen_from_stats(stats: pd.DataFrame, ctx: dict, values: np.ndarray,
                      factor_ids: list) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """screen_frame() after factor_stats(): the family filter, BH, distinct signals and the
    G_fit selection. `stats` is not modified."""
    sundays, _ = g_sundays()
    weeks = stats["ic_n"].to_numpy().astype(np.int64)
    keep = weeks >= MIN_WEEKS
    stats = stats.copy()
    stats.insert(0, "factor_id", np.array(factor_ids, dtype=object))
    df = stats[keep].reset_index(drop=True)
    vals = snap_ties(values[keep])          # the same ties as factor_stats (ranks only)
    summary = {"screened": int(keep.sum()), "dropped_too_few_weeks": int((~keep).sum())}
    base = ctx["base"]
    for name in ("ic", "ric"):
        q = bh(df[f"{name}_p"])
        df[f"{name}_q"] = q
        summary[f"n_tests_{name}"] = int(np.isfinite(q).sum())
        for lev in Q_LEVELS:
            tag = f"q{int(round(lev * 100)):02d}"
            surv = np.flatnonzero(q <= lev)
            summary[f"{name}_count_{tag}"] = int(surv.size)
            cl = np.full(len(df), -1, dtype=np.int32)
            rep = np.zeros(len(df), dtype=bool)
            if surv.size:
                V = pooled_vectors(vals[surv], base)
                c, r = dedup(V, np.abs(df[f"{name}_t"].to_numpy()[surv].astype(float)))
                cl[surv], rep[surv] = c, r
            summary[f"{name}_distinct_signals_{tag}"] = int(rep.sum())
            df[f"{name}_signal_{tag}"] = cl
            df[f"{name}_rep_{tag}"] = rep
    # the G_fit selection (the zone design's replication check), over the same family
    for name in ("ic", "ric"):
        df[f"{name}_fit_q"] = bh(df[f"{name}_fit_p"])
    summary["fit_selection"] = fit_selection(df)
    summary["g_sundays"] = len(sundays)
    summary["names_per_sunday"] = {"min": int(ctx["n_base"].min()), "max": int(ctx["n_base"].max())}
    return df, summary, weeks


def sensitivity(values: np.ndarray, symbols: list, labels: pd.DataFrame, panels_cut: dict,
                primary: pd.DataFrame | None = None, log=None, notes: dict | None = None) -> dict:
    """rIC counts at q <= 0.05 and q <= 0.10 under every VARIANTS entry, each with its own
    >= MIN_WEEKS family.

    `primary`: the registered variant's factor_stats frame when the caller already has it
    (the screen's own numbers), so it is not computed twice. `notes`: the text of each row
    (VARIANT_NOTES by default; the binance screen passes BINANCE_VARIANT_NOTES). Reported in
    the manifest only; the screen's statistic is always the registered one."""
    notes = VARIANT_NOTES if notes is None else notes
    out = {}
    for name in VARIANTS:
        if name == PRIMARY and primary is not None:
            d = primary
        else:
            d, _ = factor_stats(values, symbols, labels, panels_cut, name, with_persistence=False)
        d = d[d["ic_n"].to_numpy() >= MIN_WEEKS].reset_index(drop=True)
        q, fq = bh(d["ric_p"]), bh(d["ric_fit_p"])
        fm, em = d["ric_fit_mean"].to_numpy(dtype=float), d["ric_eval_mean"].to_numpy(dtype=float)
        et = d["ric_eval_t"].to_numpy(dtype=float)
        same = np.isfinite(fm) & np.isfinite(em) & (fm != 0) & (np.sign(em) == np.sign(fm))
        sel = fq <= 0.05
        out[name] = {
            "note": notes[name], **{k: VARIANTS[name][k] for k in ("tilt", "shrink_days", "ridge", "min_names")},
            "screened": len(d),
            "ric_count_q05": int((q <= 0.05).sum()), "ric_count_q10": int((q <= 0.10).sum()),
            "ric_fit_selected_q05": int(sel.sum()), "eval_same_sign": int((sel & same).sum()),
            "eval_same_sign_t2": int((sel & same & (np.abs(et) >= 2)).sum()),
        }
        if log:
            log(f"  sensitivity {name:<14} rIC q<=0.05 {out[name]['ric_count_q05']:>5,}  q<=0.10 "
                f"{out[name]['ric_count_q10']:>5,}  G_fit-selected {out[name]['ric_fit_selected_q05']:>5,}")
    return out


def ric_description(source: str = DEFAULT_SOURCE) -> dict:
    """The manifest's description of the screen's rIC: settings, method and each deviation
    from the registered statistic."""
    cfg = ric_settings()
    sens = ("screen.sensitivity gives the counts at each q for each single change of the settings and for the "
            "label_mean variant. The label_mean variant uses the mean weekly fwd_ret_7 label over Sundays "
            "<= D - 7 as the tilt (0 below 8 labels, unshrunk), OLS and a 5-name floor."
            + ("" if _check_source(source) == "okx" else " Each count covers the new family only."))
    return {
        "tilt": (f"pit_symbol_tilt: shrunk expanding mean of the symbol's daily excess return (perp return minus the "
                 f"equal-weight mean of the perps with a return that day) through D: sum / (n + {TILT_SHRINK_DAYS}) "
                 f"= n/(n+{TILT_SHRINK_DAYS}) x mean"),
        "beta": f"btc_beta_90d: OLS beta of daily returns on the BTC perp's over the {BETA_DAYS} returns up to D",
        "size": f"size_proxy: log of the {SIZE_DAYS}-day mean USDT volume (close x volume) up to D",
        "method": (f"ridge residual (lambda {RIDGE_LAMBDA:g} on z-scored regressors, unpenalised intercept) of the "
                   f"factor's cross-sectional ranks, then Spearman with the label; Sundays with fewer than "
                   f"{MIN_NAMES} valid names give no IC"),
        "config": cfg,
        "deviations": [
            ("no zoo_pc1 regressor: the registered set's fourth regressor is the AlphaForge zoo's first principal "
             "component, and no zoo member has a value on a G Sunday"),
            ("two-sided Student-t p and Benjamini-Hochberg over the screened family, not a one-sided test in a "
             "fixed sign"),
            f"the >= {MIN_WEEKS}-week family filter is the screen's own rule, not a registered one",
            ("the other G0 checks (coverage, finite share, distinct values) are not applied as gates; coverage "
             "is reported"),
            sens,
        ],
    }


# --------------------------------------------------------------------------- CLI actions

def _read_manifest(source: str = DEFAULT_SOURCE) -> dict:
    p = out_dir(source) / "manifest.json"
    return json.loads(p.read_text(encoding="ascii")) if p.is_file() else {}


def run_compute(workers: int = MAX_WORKERS, log=print) -> dict:
    t_all = time.time()
    timings = {}
    t = time.time()
    zones()                                    # fail early if eval_config.toml drifted
    ric_settings()
    cat = load_catalog()
    status = classify(cat)
    timings["classify_s"] = round(time.time() - t, 1)
    log(f"classified {len(status):,} definitions ({timings['classify_s']}s): "
        + ", ".join(f"{k} {v:,}" for k, v in status["group"].value_counts().items()))
    t = time.time()
    raw = load_raw()
    panels = build_panels(raw)
    idx = panels["close"].index
    sundays = idx[sunday_positions(idx)]
    symbols = list(panels["close"].columns)
    oi_rows = panels["open_interest_value"].notna().any(axis=1)
    oi_first = str(oi_rows.idxmax().date()) if oi_rows.any() else None   # first bar with open interest
    timings["panels_s"] = round(time.time() - t, 1)
    log(f"panels: {len(SUPPLIED)} fields, {len(symbols)} perps, {idx[0].date()} -> {idx[-1].date()}, "
        f"{len(sundays)} Sundays ({timings['panels_s']}s)")
    comp = cat.loc[status["group"].eq("computable").to_numpy(), ["factor_id", "expression"]].reset_index(drop=True)
    t = time.time()
    res, values, cstats = compute_values(panels, comp, workers=workers, log=log)
    timings["compute_and_pit_s"] = round(time.time() - t, 1)
    log(f"computed {len(comp):,} factors on {cstats['workers']} workers, {1 + len(PIT_CUTS)} evaluations each "
        f"(full, and cut at {', '.join(PIT_CUTS)}), {timings['compute_and_pit_s']}s; "
        f"cache hit rate {cstats['cache_hit_rate']:.2f}")
    ok = res["ok"].to_numpy(bool)
    t = time.time()
    shard_of = write_values(list(comp["factor_id"][ok]), values[ok], sundays, symbols)
    timings["write_values_s"] = round(time.time() - t, 1)

    st = status.copy()
    r = res.set_index("factor_id")
    comp_ids = st["factor_id"].isin(r.index)

    def col(name, default, dtype):
        s = pd.Series(default, index=st.index, dtype=dtype)
        if name in r:
            s[comp_ids] = r.loc[st.loc[comp_ids, "factor_id"], name].fillna(default).astype(dtype).to_numpy()
        return s

    st["compute"] = np.where(~comp_ids, "not_run", np.where(col("ok", False, bool), "ok", "error"))
    st["error_class"] = col("error_class", "", object).astype(str)
    st["finite_cells"] = col("finite_cells", 0, np.int64).astype(np.int32)
    first = col("first_sunday_pos", -1, np.int64).to_numpy()
    st["first_value_sunday"] = pd.Series(
        [str(sundays[i].date()) if i >= 0 else "" for i in first], index=st.index)
    # pass needs at least one finite compared cell (pit_status); the worker's result is used as is
    st["pit_check"] = np.where(st["compute"] != "ok", "not_run", col("pit_check", "fail", object).astype(str))
    for j in range(1, len(PIT_CUTS) + 1):
        st[f"pit_cells_t{j}"] = col(f"pit_cells_t{j}", 0, np.int64).astype(np.int32)
        st[f"pit_finite_t{j}"] = col(f"pit_finite_t{j}", 0, np.int64).astype(np.int32)
        st[f"pit_diff_cells_t{j}"] = col(f"pit_diff_cells_t{j}", 0, np.int64).astype(np.int32)
        st[f"pit_error_t{j}"] = col(f"pit_error_t{j}", "", object).astype(str)
    st["values_shard"] = st["factor_id"].map(shard_of).fillna("")
    # one reason per definition, in funnel order: computable, computed, PIT proven, not a vwap proxy
    st["screen"] = np.select(
        [st["group"] != "computable", st["compute"] != "ok", st["pit_check"] == "fail",
         st["pit_check"] == "no_finite_cells", st["vwap_proxy"]],
        ["not_computable", "compute_error", "pit_fail", "pit_no_finite_cells", "vwap_proxy"], default="pending")
    st["g_weeks"] = np.int16(0)
    t = time.time()
    _make_dirs()
    _write_frame(st, out_dir() / "status.parquet")
    _remove(out_dir() / "screen_g.parquet")    # a new compute makes any earlier screen stale
    pit = st[st["compute"] == "ok"]
    cuts = range(1, len(PIT_CUTS) + 1)
    empty = (st["compute"] == "ok") & (st["finite_cells"] == 0)
    oi = st["oi_short_history"].astype(bool)
    manifest = {
        "about": ABOUT,
        "generated_at": _now(),
        "zones": zones(),
        "panels": {
            "symbols": symbols, "benchmark": BENCHMARK, "fields": list(SUPPLIED),
            "first_bar": str(idx[0].date()), "data_through": str(idx[-1].date()),
            "oi_first_bar": oi_first,
            "sundays_stored": [str(sundays[0].date()), str(sundays[-1].date()), len(sundays)],
            "derivations": {
                "volume": "OKX perp volume as stored, base-asset units (ccxt maps OKX volCcy for swaps)",
                "quote_volume": "close x volume (the engine's dollar-volume convention); vwap therefore equals close",
                "funding_rate": "data_io.daily_funding: daily sum of settlements, per-8h-slot OKX/Binance splice",
                "open_interest_value": ("data_io.daily_oi: OKX open-interest statistics, USD notional, all contracts of the coin "
                                        "(recent months only). Every print is stamped 16:00 UTC (00:00 UTC+8, OKX's day "
                                        "boundary); the print stamped D 16:00 is booked on bar D. Point in time on "
                                        "the assumption that the value is the snapshot at its stamp; if it described "
                                        "the UTC+8 day starting at the stamp, bar D would carry up to 16 hours of "
                                        "bar D + 1. No OI factor reaches the G screen."),
                "open_interest": "open_interest_value / close (base units; the stored oi_contracts are empty)",
                "not_supplied": "premium, trades, taker_buy_volume, taker_buy_quote_volume, long_short_ratio, "
                                "top_long_short_ratio, taker_buy_sell_ratio",
            },
        },
        "inputs": input_fingerprints(),
        "code": code_fingerprint(),
        "compute": {
            "generated_at": _now(),
            "counts": {
                "definitions": len(st),
                "groups": {k: int(v) for k, v in st["group"].value_counts().sort_index().items()},
                "computable_needing_oi": int(st["oi_short_history"].sum()),
                "computable_vwap_proxy": int(st["vwap_proxy"].sum()),
                "computable_oi_base_proxy": int(st["oi_base_proxy"].sum()),
                "computed_ok": int((st["compute"] == "ok").sum()),
                "compute_errors": {k: int(v) for k, v in st.loc[st["compute"] == "error", "error_class"]
                                   .value_counts().items()},
                "all_nan_on_sundays": int(empty.sum()),
                # why they are empty: open-interest factors whose window is longer than the OI
                # history, and statistics of vwap_dev, a zero-up-to-rounding series on this data
                "all_nan_split": {"open_interest": int((empty & oi).sum()),
                                  "vwap_proxy": int((empty & ~oi & st["vwap_proxy"]).sum()),
                                  "other": int((empty & ~oi & ~st["vwap_proxy"]).sum())},
            },
            "pit": {
                "cuts": list(PIT_CUTS), "rtol": PIT_RTOL, "atol": PIT_ATOL,
                "comparison": ("np.isclose(truncated, full, rtol, atol, equal_nan=True) on every Sunday up to each "
                               "cut: equal up to floating-point rounding, not bit-identical"),
                "rule": ("pass = every compared cell equal and at least one finite cell compared over the cuts; "
                         "no_finite_cells = every compared cell equal but none finite, so nothing is proven "
                         "(counted apart, never screened); fail = a difference or an error (never screened)"),
                "pass": int((pit["pit_check"] == "pass").sum()),
                "no_finite_cells": int((pit["pit_check"] == "no_finite_cells").sum()),
                "no_finite_cells_empty_on_every_sunday": int(((pit["pit_check"] == "no_finite_cells")
                                                              & (pit["finite_cells"] == 0)).sum()),
                "fail": int((pit["pit_check"] == "fail").sum()),
                "compared_cells": [int(pit[f"pit_cells_t{j}"].sum()) for j in cuts],
                "finite_cells": [int(pit[f"pit_finite_t{j}"].sum()) for j in cuts],
                "oi_factors_with_finite_cells": [int((pit["oi_short_history"] & (pit[f"pit_finite_t{j}"] > 0)).sum())
                                                 for j in cuts],
                # OI factors whose only finite compared cells are at the last cut (their values start after T2)
                "oi_pass_from_last_cut_only": int((pit["oi_short_history"] & (pit["pit_check"] == "pass")
                                                   & (pit[[f"pit_finite_t{j}" for j in cuts[:-1]]].sum(axis=1) == 0))
                                                  .sum()),
            },
            "values": {"shards": len(set(shard_of.values())), "factors_per_shard": SHARD_FACTORS,
                       "dtype": "float64", "rows": "one per (sunday, symbol)", "git_ignored": True},
            "evaluator": cstats,
            "timings": timings,
        },
    }
    timings["write_status_s"] = round(time.time() - t, 1)
    timings["total_s"] = round(time.time() - t_all, 1)
    _write_json(manifest, out_dir() / "manifest.json")
    mp = manifest["compute"]["pit"]
    log(f"PIT proof: {mp['pass']:,} pass, {mp['no_finite_cells']:,} with no finite cell to compare, {mp['fail']:,} fail "
        f"({sum(mp['compared_cells']):,} cells compared, {sum(mp['finite_cells']):,} finite)")
    log(f"wrote data/{OUT_NAME}/status.parquet, manifest.json and {len(set(shard_of.values()))} value shards "
        f"(total {timings['total_s']}s)")
    return manifest


def run_screen(log=print) -> dict:
    t_all = time.time()
    timings = {}
    man = _read_manifest()
    if "compute" not in man or not (out_dir() / "status.parquet").is_file():
        raise SystemExit("libfactors: run --compute first")
    zones()                                     # fail early if eval_config.toml drifted
    ric_settings()
    now_inputs = input_fingerprints()
    # the label table is left out: new days add rows after the 2024-06-09 row, which the screen never reads
    stale = [k for k, v in man["inputs"].items() if k != LABEL_INPUT and now_inputs.get(k) != v]
    if stale or code_fingerprint()["sha256"] != man["code"]["sha256"]:
        raise SystemExit(f"libfactors: the stored values are stale ({', '.join(stale) or 'code changed'}); "
                         "run --compute again")
    t = time.time()
    st = pd.read_parquet(out_dir() / "status.parquet")
    cand = st[(st["compute"] == "ok") & (st["pit_check"] == "pass") & ~st["vwap_proxy"]]
    sundays, _ = g_sundays()
    ids = list(cand["factor_id"])
    values, symbols, days = read_values(ids, sundays)
    timings["load_values_s"] = round(time.time() - t, 1)
    # factors with no finite value on any G Sunday (every OI factor: OKX OI starts after G)
    has_g = np.isfinite(values).any(axis=(1, 2))
    log(f"screen candidates: {len(ids):,} computed, PIT-proven, not vwap_proxy; {int(has_g.sum()):,} have G values "
        f"({timings['load_values_s']}s to load)")
    t = time.time()
    labels = g_labels()
    panels_cut = build_panels(load_raw(), through=LABEL_CUTOFF)
    timings["inputs_s"] = round(time.time() - t, 1)
    t = time.time()
    g_ids = [i for i, h in zip(ids, has_g) if h]
    g_vals = values[has_g]
    stats, ctx = factor_stats(g_vals, symbols, labels, panels_cut, PRIMARY)
    df, summary, weeks = screen_from_stats(stats, ctx, g_vals, g_ids)
    timings["screen_s"] = round(time.time() - t, 1)
    t = time.time()
    sens = sensitivity(g_vals, symbols, labels, panels_cut, primary=stats, log=log)
    timings["sensitivity_s"] = round(time.time() - t, 1)
    meta = st.set_index("factor_id").loc[df["factor_id"], ["name", "family", "category"]].reset_index()
    df = pd.concat([meta, df.drop(columns=["factor_id"])], axis=1)

    # status: screen reason and G weeks for every definition. g_weeks is the number of
    # weekly ICs on G of every factor with G values, screened or not (a too_few_weeks
    # factor has weekly ICs too, only fewer than MIN_WEEKS); 0 for the rest
    g_n = pd.Series(0, index=st["factor_id"], dtype=np.int64)
    g_n.loc[g_ids] = weeks
    st["g_weeks"] = g_n.reindex(st["factor_id"]).fillna(0).astype(np.int16).to_numpy()
    no_g = set(np.array(ids, dtype=object)[~has_g])
    screened = set(df["factor_id"])
    pend = st["screen"] == "pending"
    st.loc[pend & st["factor_id"].isin(no_g), "screen"] = "no_g_values"
    st.loc[pend & st["factor_id"].isin(screened), "screen"] = "screened"
    st.loc[st["screen"] == "pending", "screen"] = "too_few_weeks"
    t = time.time()
    _write_frame(df, out_dir() / "screen_g.parquet")
    _write_frame(st, out_dir() / "status.parquet")
    timings["write_s"] = round(time.time() - t, 1)
    timings["total_s"] = round(time.time() - t_all, 1)
    man["inputs"] = now_inputs                  # the inputs of this screen, the label table included
    man["screen"] = {
        "generated_at": _now(),
        "evidence": EVIDENCE,
        "label": f"{LABEL} from the label table, Sunday rows through {LABEL_CUTOFF.date()}",
        "mask": (f"label-table rows with finite factor, label and regressors; >= {MIN_NAMES} names a Sunday "
                 f"([g0] min_names_per_sunday)"),
        "family": f"computed, PIT pass, not vwap_proxy, >= {MIN_WEEKS} weekly ICs on G",
        "precision": (f"ranks of the float64 values; per factor and Sunday, values within 1e-{TIE_DIGITS} of the "
                      "largest |value| are ties (snap_ties), so neither storage precision nor floating-point noise "
                      "orders names"),
        "ric_regressors": ric_description(),
        "dedup": f"greedy, |pooled within-Sunday rank correlation| >= {DEDUP_RHO}, strongest |t| first",
        "pooled_g_note": POOLED_G_NOTE,
        "sensitivity": sens,
        "counts": {
            "candidates": len(ids), "with_g_values": int(has_g.sum()), "no_g_values": int((~has_g).sum()),
            "screen_status": {k: int(v) for k, v in st["screen"].value_counts().sort_index().items()},
            **summary,
        },
        "timings": timings,
    }
    _write_json(man, out_dir() / "manifest.json")
    log(f"screened {summary['screened']:,} factors on {summary['g_sundays']} G Sundays ({timings['screen_s']}s); "
        f"BH over {summary['n_tests_ic']:,} (IC) and {summary['n_tests_ric']:,} (rIC) tests")
    for name in ("ic", "ric"):
        log(f"  {name.upper():>3} q<=0.05: {summary[f'{name}_count_q05']:,} "
            f"({summary[f'{name}_distinct_signals_q05']:,} distinct), q<=0.10: {summary[f'{name}_count_q10']:,} "
            f"({summary[f'{name}_distinct_signals_q10']:,} distinct)")
    fs = summary["fit_selection"]["ric"]
    log(f"  rIC selected on G_fit alone (q<=0.05): {fs['q05']['selected']:,}; on G_eval {fs['q05']['eval_same_sign']:,} "
        f"keep their sign, {fs['q05']['eval_same_sign_t2']:,} with |t| >= 2 (base rate: {fs['base_same_sign']:,} of "
        f"{fs['base_pairs']:,} screened factors keep their sign)")
    log(f"wrote data/{OUT_NAME}/screen_g.parquet, status.parquet, manifest.json (total {timings['total_s']}s)")
    return man


def show_status() -> int:
    """Read-only summary of data/library_on_engine/."""
    man = _read_manifest()
    if not man:
        print(f"data/{OUT_NAME}/ has no outputs: run --compute, then --screen")
        return 0
    c = man["compute"]
    print(f"library factors on Factorlab's data (generated {man['generated_at']}, data through "
          f"{man['panels']['data_through']})")
    _print_states(man, "okx", "--compute and --screen")
    print(f"  definitions {c['counts']['definitions']:,}: " +
          ", ".join(f"{k} {v:,}" for k, v in c["counts"]["groups"].items()))
    print(f"  computable needing OI (history after G only): {c['counts']['computable_needing_oi']:,}; "
          f"vwap_proxy: {c['counts']['computable_vwap_proxy']:,}")
    print(f"  computed ok {c['counts']['computed_ok']:,}, errors {sum(c['counts']['compute_errors'].values()):,}; "
          f"PIT pass {c['pit']['pass']:,}, no finite cell to compare {c['pit'].get('no_finite_cells', 0):,}, "
          f"fail {c['pit']['fail']:,} (cuts {', '.join(c['pit']['cuts'])}; rtol {c['pit']['rtol']:g}, "
          f"atol {c['pit']['atol']:g})")
    s = man.get("screen")
    if not s:
        print("  screen: not run")
        return 0
    k = s["counts"]
    print(f"  G screen: {k['screened']:,} factors, BH tests IC {k['n_tests_ic']:,} / rIC {k['n_tests_ric']:,}")
    for name in ("ic", "ric"):
        print(f"    {name.upper():>3} q<=0.05: {k[f'{name}_count_q05']:,} ({k[f'{name}_distinct_signals_q05']:,} "
              f"distinct); q<=0.10: {k[f'{name}_count_q10']:,} ({k[f'{name}_distinct_signals_q10']:,} distinct)"
              " -- pooled G, in sample on G_eval")
    fs = k.get("fit_selection", {}).get("ric")
    if fs:
        print(f"    rIC selected on G_fit alone, q<=0.05: {fs['q05']['selected']:,}; checked on G_eval: "
              f"{fs['q05']['eval_same_sign']:,} same sign, {fs['q05']['eval_same_sign_t2']:,} same sign with |t| >= 2 "
              f"(base rate {fs['base_same_sign'] / max(fs['base_pairs'], 1):.0%} of all screened)")
    for name, v in s.get("sensitivity", {}).items():
        print(f"    sensitivity {name:<14} rIC q<=0.05 {v['ric_count_q05']:>5,}  q<=0.10 {v['ric_count_q10']:>5,}"
              f"  ({v['note']})")
    return 0


# --------------------------------------------------------------------------- the binance source

BINANCE_CONVENTIONS = {
    "index": ("one row per UTC day stamped with the bar's open (Binance kline open time), from the first kline bar "
              "to the last closed one; every field NaN where the coin has no kline bar"),
    "symbols": "Binance USD-M columns mapped to the engine's names and order (BTCUSDT -> BTC-USDT, ...)",
    "benchmark": "BTC-USDT, which on this panel is Binance's BTCUSDT (btc_ret, ex_btc_ret, bench())",
    "kline": ("open, high, low, close, volume (base asset), quote_volume (true USDT turnover), trades, "
              "taker_buy_volume, taker_buy_quote_volume: the 1d USD-M kline with open time D, i.e. prices and flows "
              "of [D 00:00, D + 1 00:00) UTC; vwap = quote_volume / volume is the bar's VWAP"),
    "funding_rate": ("sum of the settlements whose fundingTime lies in [D 00:00, D + 1 00:00) UTC (resample 1D, label "
                     "and closed left, sum with min_count 1: NaN without a settlement); a settlement stamped "
                     "D + 1 00:00 belongs to D + 1"),
    "premium": "close (field 4) of the 1d premium-index kline with open time D: the premium index at the end of D",
    "metrics": ("open_interest (coins, sum_open_interest), open_interest_value (USDT), long_short_ratio (global "
                "accounts), top_long_short_ratio (top-trader positions), taker_buy_sell_ratio: the last non-missing "
                "5-minute snapshot whose create_time (UTC) lies in [D 00:00, D + 1 00:00), an earlier one when the "
                "day's archive file is partial (data.binance.vision daily metrics "
                "files; resample 1D, label and closed left, last). Whether create_time marks the start or the end of "
                "its 5 minutes, the kept snapshot describes data up to D + 1 00:00 at the latest"),
    "taker_buy_sell_ratio": ("the 5 minutes of the kept snapshot, normally the bar's last (sum_taker_long_short_vol_ratio), not the whole day as the "
                             "library's REST definition with a 1d period would be: a noisier stand-in, flagged "
                             "taker_ratio_last_5m and still screened"),
    "cleaning": ("archive errors are set missing (NaN) before any factor reads the panel, by three label-free "
                 "rules that each read one coin's fields on one bar: open_interest or open_interest_value <= 0 sets "
                 "both missing; quote_volume / volume outside the bar's [low, high] sets volume missing; "
                 "taker_buy_quote_volume / taker_buy_volume outside it sets taker_buy_volume missing (relative "
                 "tolerance 1e-9). No value is carried from another day. panels.checks.cleaning lists every cell "
                 "by date and coin"),
    "cleaning_frozen": ("CLEANING_FREEZE pins the cleaning rules with a hash, so a rule edit cannot change the values "
                        "of the weekly log of the formula tests (judge.py log-week). panels.cleaning_freeze gives the "
                        "sha256 of that set. The compute stops when the rules in the code differ from that set."),
    "point_in_time": "every field of bar D is known at D's close (D + 1 00:00 UTC); the last bar is closed",
    "labels_and_regressors": ("exactly the OKX screen's: fwd_ret_7 from the label table, and "
                              "pit_symbol_tilt, btc_beta_90d and size_proxy from the engine's OKX bars cut at the "
                              "2024-06-09 row, so the two screens differ only in the factor inputs"),
}
# exclusive input classes for the cross-check's agreement table (raw inputs of a definition)
VOLUME_FIELDS = ("volume", "quote_volume")
PRICE_FIELDS = ("open", "high", "low", "close")
# library series whose cross-venue agreement the cross-check measures (label-free, G span)
AGREEMENT_SERIES = ("ret", "oc_ret", "gap", "hl_range", "funding", "dvol_chg", "amihud")
SAME_VALUE_ATOL = 1e-12          # series_agreement's identical_share: the two venues hold the same number
FUNDING_FIELD = "funding_rate"
# What the cross-check's agreement can and cannot show (manifest crosscheck.agreement_note and
# funding_note; the dashboard prints both). Sign agreement is largely mechanical: the two
# screens share everything except the factor inputs, and those correlate closely.
AGREEMENT_NOTE = ("The two screens share the labels, the regressors, the Sundays, the names and the eligibility "
                  "mask, and the same definitions' values correlate at a median {rho:.2f} across venues, so sign "
                  "agreement is expected: it measures how closely the venues' data agree and is not an independent "
                  "replication.")
FUNDING_NOTE = ("Up to the 2024-06-09 row (G and the warm-up before it) the engine's OKX funding is Binance's own settlement archive (data_io splices Binance prints in "
                "before the first OKX settlement on disk; the two venues' daily funding is the same number on {same:.1%} of the {cells:,} "
                "coin-days compared), so the {n:,} definitions that read funding_rate read the same funding on both "
                "venues: their agreement is not cross-venue evidence for that input. Without them the rank "
                "correlation of rIC t is {rho:.3f}.")
OVERLAP_SETS = (("ic", 0.05), ("ic", 0.10), ("ric", 0.05), ("ric", 0.10))
CROSS_STATS = ("ic_n", "ic_mean", "ic_t", "ric_n", "ric_mean", "ric_t", "ric_fit_t", "ric_eval_t",
               "persistence", "coverage")


def _tag(lev: float) -> str:
    return f"q{int(round(lev * 100)):02d}"


def binance_panel_fingerprint(panel_dir: Path | None = None) -> dict:
    """sha256 of every merged panel file's DECOMPRESSED content (gzip headers carry the
    write time, so a re-merge of the same parts changes the bytes but not the content),
    keyed by file name only (never a path)."""
    d = BINANCE_PANEL_DIR if panel_dir is None else Path(panel_dir)
    files = {f.name: hashlib.sha256(gzip.decompress(f.read_bytes())).hexdigest() for f in sorted(d.glob("*.csv.gz"))}
    return {"panel_set": BINANCE_PANEL_SET, "files": files,
            "hash_of": "the decompressed CSV content of each file (gzip headers carry the write time)",
            "sha256": hashlib.sha256("\n".join(f"{k} {v}" for k, v in files.items()).encode()).hexdigest()}


def binance_inputs() -> dict:
    """The binance source's input fingerprints (relative names and hashes only): the panel,
    the catalog, and the OKX screen's status and screen files (they decide each factor's
    role). The engine's OKX inputs for the labels and regressors are added by --screen."""
    okx = out_dir("okx")
    return {
        "binance_panel": binance_panel_fingerprint(),
        "library/catalog/factors.db": {"files": 1, "sha256": _sha_file(CATALOG)},
        "okx_screen": {n: (_sha_file(okx / n) if (okx / n).is_file() else None)
                       for n in ("status.parquet", "screen_g.parquet")},
    }


def binance_status(cat: pd.DataFrame, okx_status: pd.DataFrame, collapses: bool) -> pd.DataFrame:
    """One row per definition for the binance source: classification, gains and role.

    group        computable for every definition (the panel supplies all 16 raw fields)
    uses_oi, uses_vwap; vwap_proxy = uses_vwap and vwap_collapses() (False on Binance)
    binance_only_fields  the inputs OKX cannot supply; new_fields what the factor gains
                 on Binance (those, open interest over G, a true vwap), comma separated
    taker_ratio_last_5m  reads taker_buy_sell_ratio (the bar's last 5 minutes)
    okx_group, okx_screen  the same definition's OKX classification and screen status
    role         crosscheck (the OKX screen screened it) or new_family (it did not)
    origin       okx_screened for the cross-check; for the new family the first of
                 binance_only_fields, oi, vwap, other"""
    st = classify(cat, supplied_fields=BINANCE_FIELDS)
    if (st["group"] != "computable").any():
        raise SystemExit("libfactors: a definition reads a field the Binance panel does not supply")
    okx = okx_status.set_index("factor_id")
    if len(okx) != len(st) or set(okx.index) != set(st["factor_id"]):
        raise SystemExit("libfactors: the OKX status does not hold the catalog's definitions; run the OKX source again")
    okx = okx.loc[st["factor_id"]]
    inputs = [set(x.split(",")) for x in cat["raw_inputs"]]
    names = [expr_atoms(e)[1] for e in cat["expression"]]
    uses_oi = np.array([bool(i & set(OI_FIELDS)) for i in inputs])
    uses_vwap = np.array([bool(n & set(VWAP_FIELDS)) for n in names])
    only = [",".join(f for f in BINANCE_ONLY if f in i) for i in inputs]
    new = [",".join([f for f in BINANCE_ONLY + OI_FIELDS if f in i] + (["vwap"] if v else []))
           for i, v in zip(inputs, uses_vwap)]
    okx_screen = okx["screen"].astype(str).to_numpy()
    role = np.where(okx_screen == "screened", "crosscheck", "new_family")
    origin = np.select([role == "crosscheck", np.array([bool(o) for o in only]), uses_oi, uses_vwap],
                       ["okx_screened", "binance_only_fields", "oi", "vwap"], default="other")
    out = st[["factor_id", "name", "family", "category", "group", "lookback"]].copy()
    out["uses_oi"] = uses_oi
    out["uses_vwap"] = uses_vwap
    out["vwap_proxy"] = uses_vwap & bool(collapses)
    out["binance_only_fields"] = only
    out["new_fields"] = new
    out["taker_ratio_last_5m"] = np.array([TAKER_LAST_5M in i for i in inputs])
    out["okx_group"] = okx["group"].astype(str).to_numpy()
    out["okx_screen"] = okx_screen
    out["role"] = role
    out["origin"] = origin
    return out.reset_index(drop=True)


def new_family_ids(st: pd.DataFrame) -> list:
    """The binance source's new family before its G-weeks filter: definitions the OKX screen
    did NOT screen (okx_screen != screened), computed on Binance, PIT pass, not degenerate."""
    m = ((st["okx_screen"] != "screened") & (st["compute"] == "ok") & (st["pit_check"] == "pass")
         & ~st["vwap_proxy"].astype(bool))
    return list(st.loc[m, "factor_id"])


def crosscheck_ids(st: pd.DataFrame) -> list:
    """The cross-check set: definitions the OKX screen screened, computed on Binance and PIT pass."""
    m = ((st["okx_screen"] == "screened") & (st["compute"] == "ok") & (st["pit_check"] == "pass")
         & ~st["vwap_proxy"].astype(bool))
    return list(st.loc[m, "factor_id"])


def max_abs_rho(a: np.ndarray, b: np.ndarray, base: np.ndarray) -> np.ndarray:
    """For each factor of `a` (k, T, m), the largest |pooled within-Sunday rank correlation|
    (the screen's dedup measure, on tie-snapped values) with any factor of `b`; NaN when `b`
    is empty."""
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), np.nan)
    va = pooled_vectors(snap_ties(a), base)
    vb = pooled_vectors(snap_ties(b), base)
    return np.abs(va @ vb.T).max(axis=1)


def cross_venue_rho(a: np.ndarray, b: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per factor: the pooled within-Sunday rank correlation of its values on two panels
    (a: OKX, b: Binance), over the cells finite on both and in the screen's base mask, and
    the number of those cells. Both sides are ranked over the same names each Sunday."""
    common = base[None, :, :] & np.isfinite(a) & np.isfinite(b)
    va = pooled_vectors(np.where(common, snap_ties(a), np.nan), base)
    vb = pooled_vectors(np.where(common, snap_ties(b), np.nan), base)
    rho = (va * vb).sum(axis=1)
    cells = common.sum(axis=(1, 2))
    return np.where(cells > 0, rho, np.nan), cells


def crosscheck_frame(stats: pd.DataFrame, ids: list, okx_sg: pd.DataFrame, rho: np.ndarray,
                     cells: np.ndarray) -> pd.DataFrame:
    """One row per cross-checked factor: the OKX screen's statistics next to the
    same definition's G statistics on the Binance panel, and the cross-venue value
    correlation. AGREEMENT ONLY: no Binance p-value, q-value, signal or q-level column."""
    okx = okx_sg.set_index("factor_id").loc[ids]
    out = pd.DataFrame({"factor_id": ids, "name": okx["name"].to_numpy(), "family": okx["family"].to_numpy(),
                        "category": okx["category"].to_numpy()})
    for c in ("ic_n", "ic_t", "ric_mean", "ric_t", "ric_fit_t", "ric_eval_t", "ric_q"):
        out[f"okx_{c}"] = okx[c].to_numpy()
    for c in CROSS_STATS:
        out[f"binance_{c}"] = stats[c].to_numpy()
    out["value_rho"] = np.asarray(rho, dtype=np.float32)
    out["value_cells"] = np.asarray(cells, dtype=np.int32)
    a, b = out["okx_ric_t"].to_numpy(dtype=float), out["binance_ric_t"].to_numpy(dtype=float)
    out["same_ric_sign"] = np.isfinite(a) & np.isfinite(b) & (np.sign(a) == np.sign(b))
    return out


def series_agreement(okx_panels: dict, bin_panels: dict, names: tuple = AGREEMENT_SERIES) -> dict:
    """Label-free cross-venue agreement of a few library series (each evaluated by the
    library's Evaluator on each venue's own panel): median |value| on each venue, the daily
    Pearson and Spearman correlation over the cells both have, and the share of those cells
    on which the two venues' values are the same number (|difference| <= 1e-12). That share
    is 1 for funding on G: the engine's OKX funding before the first OKX settlement IS Binance's settlement
    archive (data_io's splice), so funding is not a second venue's input there."""
    eo, eb = _evaluator(okx_panels), _evaluator(bin_panels)
    idx = okx_panels["close"].index.intersection(bin_panels["close"].index)
    out = {}
    for n in names:
        xo = _values(eo, n)[okx_panels["close"].index.get_indexer(idx)]
        xb = _values(eb, n)[bin_panels["close"].index.get_indexer(idx)]
        m = np.isfinite(xo) & np.isfinite(xb)
        out[n] = {"cells": int(m.sum()), "median_abs_okx": float(np.median(np.abs(xo[m]))),
                  "median_abs_binance": float(np.median(np.abs(xb[m]))),
                  "daily_corr": _r(np.corrcoef(xo[m], xb[m])[0, 1]), "daily_spearman": _spearman(xo[m], xb[m]),
                  "identical_share": _r(float((np.abs(xo[m] - xb[m]) <= SAME_VALUE_ATOL).mean()))}
    return out


def input_class(raw_inputs: str, names: frozenset = frozenset()) -> str:
    """Exclusive class of a definition's inputs for the cross-check table: gap first (it
    reads the gap series), then by raw inputs."""
    if "gap" in names:
        return "gap"
    i = set(raw_inputs.split(","))
    vol, fund = bool(i & set(VOLUME_FIELDS)), "funding_rate" in i
    if vol and fund:
        return "volume_and_funding"
    if vol:
        return "volume"
    if fund:
        return "funding"
    return "price_only" if i <= set(PRICE_FIELDS) else "other"


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    return None if ok.sum() < 3 else round(float(spearmanr(x[ok], y[ok]).statistic), 4)


def _r(x, nd: int = 4):
    return None if x is None or not np.isfinite(float(x)) else round(float(x), nd)


def disagreement_cause(row, raw_inputs: str, names: frozenset = frozenset(), series: dict | None = None) -> str:
    """A plausible cause for one large cross-venue gap in rIC t, from label-free diagnostics
    only: the cross-venue value correlation, the series it reads (series_agreement()), the
    weekly-IC counts and the raw inputs."""
    i = set(raw_inputs.split(","))
    rho = float(row["value_rho"])
    parts = []
    g = (series or {}).get("gap")
    if "gap" in names and g:
        parts.append(f"it reads gap (open / previous close - 1), the step between one UTC bar's last trade and the "
                     f"next bar's first on a market that trades around the clock: venue microstructure (median "
                     f"|gap| {g['median_abs_okx']:.1e} on OKX, {g['median_abs_binance']:.1e} on Binance; daily "
                     f"cross-venue correlation {g['daily_corr']})")
    if not np.isfinite(rho) or rho < 0.5:
        parts.append(f"the values differ across venues (pooled rank correlation {rho:.2f})")
    elif rho < 0.9:
        parts.append(f"the values differ in part across venues (pooled rank correlation {rho:.2f})")
    if i & set(VOLUME_FIELDS):
        parts.append("it reads volume, which is venue-specific (OKX quote volume is close x base volume, "
                     "Binance's the true USDT turnover, and each venue's share of a coin's trading differs)")
    gap = int(row["okx_ic_n"]) - int(row["binance_ic_n"])
    if abs(gap) >= 10:
        parts.append(f"weekly-IC counts differ (OKX {int(row['okx_ic_n'])}, Binance {int(row['binance_ic_n'])})")
    if not parts:
        parts.append(f"the values agree closely (pooled rank correlation {rho:.2f}): the gap comes from small rank "
                     "changes in a cross-section of at most 20 names")
    return "; ".join(parts)


def crosscheck_summary(cross: pd.DataFrame, raw_inputs: dict, atoms: dict | None = None,
                       series: dict | None = None, top: int = 10) -> dict:
    """Cross-venue agreement of the OKX-screened factors (no selection of any kind).

    raw_inputs: {factor_id: raw_inputs}; atoms: {factor_id: names the expression reads}
    (expr_atoms); series: series_agreement(), reported and used for the causes."""
    atoms = atoms or {}
    a, b = cross["okx_ric_t"].to_numpy(dtype=float), cross["binance_ric_t"].to_numpy(dtype=float)
    ai, bi = cross["okx_ic_t"].to_numpy(dtype=float), cross["binance_ic_t"].to_numpy(dtype=float)
    both, both_ic = np.isfinite(a) & np.isfinite(b), np.isfinite(ai) & np.isfinite(bi)
    rho = cross["value_rho"].to_numpy(dtype=float)
    fr = np.isfinite(rho)
    out = {
        "rule": ("agreement only: no Benjamini-Hochberg, no q-value and no list of factors at a q level on Binance "
                 "for these factors; their only q-values are the OKX screen's"),
        "factors": len(cross), "compared_ric_t": int(both.sum()),
        "ric_t_spearman": _spearman(a, b), "ic_t_spearman": _spearman(ai, bi),
        "same_ric_sign_share": _r(float((np.sign(a) == np.sign(b))[both].mean()) if both.any() else None),
        "same_ic_sign_share": _r(float((np.sign(ai) == np.sign(bi))[both_ic].mean()) if both_ic.any() else None),
        "value_rho": {"median": _r(np.median(rho[fr])) if fr.any() else None,
                      "p10": _r(np.percentile(rho[fr], 10)) if fr.any() else None,
                      "share_at_least_0.9": _r(float((rho[fr] >= 0.9).mean())) if fr.any() else None,
                      "share_below_0.5": _r(float((rho[fr] < 0.5).mean())) if fr.any() else None},
    }
    q = cross["okx_ric_q"].to_numpy(dtype=float)
    for lev in Q_LEVELS:
        s = q <= lev
        same = s & both & (np.sign(a) == np.sign(b))
        out[f"okx_ric_{_tag(lev)}"] = {
            "okx_count": int(s.sum()), "binance_same_sign": int(same.sum()),
            "binance_same_sign_abs_t_ge_2": int((same & (np.abs(b) >= 2)).sum()),
            "median_value_rho": _r(np.median(rho[s & fr])) if (s & fr).any() else None}
    cls = np.array([input_class(raw_inputs[f], atoms.get(f, frozenset())) for f in cross["factor_id"]])
    # definitions that read funding_rate: on G both venues give them the same funding (FUNDING_NOTE)
    fund = np.array([FUNDING_FIELD in raw_inputs[f].split(",") for f in cross["factor_id"]], dtype=bool)
    by = {}
    for c in sorted(set(cls)):
        m = cls == c
        by[c] = {"factors": int(m.sum()), "ric_t_spearman": _spearman(a[m], b[m]),
                 "same_ric_sign_share": _r(float((np.sign(a) == np.sign(b))[m & both].mean())) if (m & both).any() else None,
                 "median_value_rho": _r(np.median(rho[m & fr])) if (m & fr).any() else None,
                 "funding_readers": int((m & fund).sum())}
    out["by_input_class"] = by
    out["ric_t_spearman_without_gap"] = _spearman(a[cls != "gap"], b[cls != "gap"])
    out["funding_readers"] = int(fund.sum())
    out["ric_t_spearman_without_funding"] = _spearman(a[~fund], b[~fund])
    out["same_ric_sign_share_without_funding"] = (_r(float((np.sign(a) == np.sign(b))[both & ~fund].mean()))
                                                  if (both & ~fund).any() else None)
    if out["value_rho"]["median"] is not None:
        out["agreement_note"] = AGREEMENT_NOTE.format(rho=out["value_rho"]["median"])
    f_ser = (series or {}).get("funding") or {}
    if f_ser.get("identical_share") is not None and out["ric_t_spearman_without_funding"] is not None:
        out["funding_note"] = FUNDING_NOTE.format(same=f_ser["identical_share"], cells=f_ser["cells"],
                                                  n=out["funding_readers"], rho=out["ric_t_spearman_without_funding"])
    if series is not None:
        out["series_agreement"] = series
    diff = np.where(both, np.abs(a - b), -1.0)
    order = np.argsort(-diff, kind="stable")
    head = order[:50][diff[order[:50]] >= 0]
    out["largest_50_disagreements_by_class"] = {k: int(v) for k, v in
                                                pd.Series(cls[head]).value_counts().sort_index().items()}
    rows = []
    for i in order[:top]:
        if diff[i] < 0:
            break
        r = cross.iloc[i]
        f = r["factor_id"]
        rows.append({"factor_id": f, "name": r["name"], "family": r["family"], "input_class": str(cls[i]),
                     "okx_ric_t": _r(r["okx_ric_t"], 2), "binance_ric_t": _r(r["binance_ric_t"], 2),
                     "okx_ric_q": _r(r["okx_ric_q"]), "value_rho": _r(r["value_rho"], 3),
                     "okx_weeks": int(r["okx_ic_n"]), "binance_weeks": int(r["binance_ic_n"]),
                     "inputs": raw_inputs[f],
                     "cause": disagreement_cause(r, raw_inputs[f], atoms.get(f, frozenset()), series)})
    out["largest_disagreements"] = rows
    return out


def _q_set_breakdown(df: pd.DataFrame, name: str, lev: float) -> dict:
    """Counts of the new family's factors at q <= lev of one statistic: by origin, family and new field."""
    s = df[df[f"{name}_q"] <= lev]
    fields = {f: int(s["new_fields"].str.split(",").apply(lambda x, f=f: f in x).sum()) for f in NEW_FIELDS}
    return {"factors": len(s), "distinct_signals": int(s[f"{name}_rep_{_tag(lev)}"].sum()),
            "by_origin": {k: int(v) for k, v in s["origin"].value_counts().sort_index().items()},
            "by_family": {k: int(v) for k, v in s["family"].value_counts().sort_index().items()},
            "by_new_field": {f: n for f, n in fields.items() if n},
            "taker_ratio_last_5m": int(s["taker_ratio_last_5m"].sum())}


def _binance_result_columns(st: pd.DataFrame, res: pd.DataFrame, sundays: pd.DatetimeIndex,
                            shard_of: dict) -> pd.DataFrame:
    """The compute result columns of the binance status (the same columns and rules as
    run_compute's for OKX)."""
    st = st.copy()
    r = res.set_index("factor_id")
    comp_ids = st["factor_id"].isin(r.index)

    def col(name, default, dtype):
        s = pd.Series(default, index=st.index, dtype=dtype)
        if name in r:
            s[comp_ids] = r.loc[st.loc[comp_ids, "factor_id"], name].fillna(default).astype(dtype).to_numpy()
        return s

    st["compute"] = np.where(~comp_ids, "not_run", np.where(col("ok", False, bool), "ok", "error"))
    st["error_class"] = col("error_class", "", object).astype(str)
    st["finite_cells"] = col("finite_cells", 0, np.int64).astype(np.int32)
    first = col("first_sunday_pos", -1, np.int64).to_numpy()
    st["first_value_sunday"] = pd.Series([str(sundays[i].date()) if i >= 0 else "" for i in first], index=st.index)
    st["pit_check"] = np.where(st["compute"] != "ok", "not_run", col("pit_check", "fail", object).astype(str))
    for j in range(1, len(PIT_CUTS) + 1):
        st[f"pit_cells_t{j}"] = col(f"pit_cells_t{j}", 0, np.int64).astype(np.int32)
        st[f"pit_finite_t{j}"] = col(f"pit_finite_t{j}", 0, np.int64).astype(np.int32)
        st[f"pit_diff_cells_t{j}"] = col(f"pit_diff_cells_t{j}", 0, np.int64).astype(np.int32)
        st[f"pit_error_t{j}"] = col(f"pit_error_t{j}", "", object).astype(str)
    st["values_shard"] = st["factor_id"].map(shard_of).fillna("")
    st["screen"] = np.select(
        [st["compute"] != "ok", st["pit_check"] == "fail", st["pit_check"] == "no_finite_cells", st["vwap_proxy"]],
        ["compute_error", "pit_fail", "pit_no_finite_cells", "vwap_proxy"], default="pending")
    st["g_weeks"] = np.int16(0)
    return st


def run_compute_binance(workers: int = MAX_WORKERS, log=print) -> dict:
    """--source binance --compute: all 28,904 definitions on the Binance panel, PIT proof,
    weekly values; writes data/library_on_engine/binance/ only."""
    src = "binance"
    t_all = time.time()
    timings = {}
    zones()                                    # fail early if eval_config.toml drifted
    ric_settings()
    freeze = cleaning_freeze_record()          # stop at once if the cleaning rules left the frozen set
    okx_man = _read_manifest("okx")
    if "screen" not in okx_man or not (out_dir("okx") / "status.parquet").is_file():
        raise SystemExit("libfactors: the binance source needs the OKX screen first (--compute, then --screen)")
    t = time.time()
    okx_status = pd.read_parquet(out_dir("okx") / "status.parquet", columns=["factor_id", "group", "screen"])
    cat = load_catalog()
    raw = load_raw_binance()
    uncleaned = build_panels_binance(raw, clean=False)
    panels = clean_binance(uncleaned)                  # what build_panels_binance(raw) returns
    idx = panels["close"].index
    sundays = idx[sunday_positions(idx)]
    symbols = list(panels["close"].columns)
    if symbols != [C.sym_to_file(s) for s in C.UNIVERSE]:
        raise SystemExit("libfactors: the Binance panel's columns are not the engine's symbols in order")
    appends = binance_panel_appends()
    span = [BINANCE_FIRST_BAR, appends[-1]["last_bar"] if appends else str(idx[-1].date())]
    if [str(idx[0].date()), str(idx[-1].date())] != span:
        raise SystemExit(f"libfactors: the Binance panel holds {idx[0].date()} -> {idx[-1].date()}, but its provenance "
                         f"(the first bar and {BINANCE_APPEND_LOG}) gives {span[0]} -> {span[1]}")
    collapses = vwap_collapses(panels)
    status = binance_status(cat, okx_status, collapses)
    checks = binance_checks(panels, uncleaned)
    del uncleaned
    cleaned = {r: c["cells"] for r, c in checks["cleaning"].items()}
    log("cleaned archive errors (set missing): " + ", ".join(f"{r} {n}" for r, n in cleaned.items()))
    timings["panels_and_checks_s"] = round(time.time() - t, 1)
    log(f"binance panel: {len(BINANCE_FIELDS)} fields, {len(symbols)} coins, {idx[0].date()} -> {idx[-1].date()}, "
        f"{len(sundays)} Sundays; roles: " + ", ".join(f"{k} {v:,}" for k, v in status["role"].value_counts().items()))
    comp = cat[["factor_id", "expression"]].reset_index(drop=True)
    t = time.time()
    res, values, cstats = compute_values(panels, comp, workers=workers, log=log)
    timings["compute_and_pit_s"] = round(time.time() - t, 1)
    log(f"computed {len(comp):,} factors on {cstats['workers']} workers, {1 + len(PIT_CUTS)} evaluations each "
        f"({timings['compute_and_pit_s']}s); cache hit rate {cstats['cache_hit_rate']:.2f}")
    ok = res["ok"].to_numpy(bool)
    t = time.time()
    shard_of = write_values(list(comp["factor_id"][ok]), values[ok], sundays, symbols, source=src)
    del values
    timings["write_values_s"] = round(time.time() - t, 1)
    st = _binance_result_columns(status, res, sundays, shard_of)
    t = time.time()
    _make_dirs(src)
    _write_frame(st, out_dir(src) / "status.parquet", source=src)
    for stale in ("screen_g.parquet", "crosscheck.parquet"):   # a new compute makes any earlier screen stale
        _remove(out_dir(src) / stale, src)
    pit = st[st["compute"] == "ok"]
    cuts = range(1, len(PIT_CUTS) + 1)
    empty = (st["compute"] == "ok") & (st["finite_cells"] == 0)
    fam = st["role"] == "new_family"
    manifest = {
        "about": BINANCE_ABOUT,
        "source": src,
        "generated_at": _now(),
        "zones": zones(),
        "panels": {
            "venue": "Binance USD-M perpetuals, public data (data only)",
            "symbols": symbols, "symbol_map": {s: binance_symbol(s) for s in symbols},
            "benchmark": f"{BENCHMARK} (Binance {binance_symbol(BENCHMARK)})", "fields": list(BINANCE_FIELDS),
            "first_bar": str(idx[0].date()), "data_through": str(idx[-1].date()),
            "sundays_stored": [str(sundays[0].date()), str(sundays[-1].date()), len(sundays)],
            "conventions": BINANCE_CONVENTIONS,
            "cleaning_freeze": freeze,
            "checks": checks,
            "provenance": {**BINANCE_PROVENANCE, "bars": [str(idx[0].date()), str(idx[-1].date())],
                           "library_version": factorlab.__version__, "appends": appends},
        },
        "inputs": binance_inputs(),
        "code": code_fingerprint(BINANCE_CODE_FILES),
        "compute": {
            "generated_at": _now(),
            "counts": {
                "definitions": len(st),
                "groups": {k: int(v) for k, v in st["group"].value_counts().sort_index().items()},
                "roles": {k: int(v) for k, v in st["role"].value_counts().sort_index().items()},
                "new_family_by_origin": {k: int(st.loc[fam, "origin"].eq(k).sum()) for k in ORIGINS},
                "new_family_by_okx_screen": {k: int(v) for k, v in st.loc[fam, "okx_screen"].value_counts()
                                             .sort_index().items()},
                "uses_oi": int(st["uses_oi"].sum()), "uses_vwap": int(st["uses_vwap"].sum()),
                "vwap_proxy_degenerate": int(st["vwap_proxy"].sum()),
                "taker_ratio_last_5m": int(st["taker_ratio_last_5m"].sum()),
                "computed_ok": int((st["compute"] == "ok").sum()),
                "compute_errors": {k: int(v) for k, v in st.loc[st["compute"] == "error", "error_class"]
                                   .value_counts().items()},
                "all_nan_on_sundays": int(empty.sum()),
            },
            "pit": {
                "cuts": list(PIT_CUTS), "rtol": PIT_RTOL, "atol": PIT_ATOL,
                "comparison": ("np.isclose(truncated, full, rtol, atol, equal_nan=True) on every Sunday up to each "
                               "cut: equal up to floating-point rounding, not bit-identical"),
                "rule": ("pass = every compared cell equal and at least one finite cell compared over the cuts; "
                         "no_finite_cells = every compared cell equal but none finite, so nothing is proven "
                         "(counted apart, never screened); fail = a difference or an error (never screened)"),
                "pass": int((pit["pit_check"] == "pass").sum()),
                "no_finite_cells": int((pit["pit_check"] == "no_finite_cells").sum()),
                "fail": int((pit["pit_check"] == "fail").sum()),
                "by_role": {r: {k: int(((pit["role"] == r) & (pit["pit_check"] == k)).sum())
                                for k in ("pass", "no_finite_cells", "fail")} for r in ("new_family", "crosscheck")},
                "compared_cells": [int(pit[f"pit_cells_t{j}"].sum()) for j in cuts],
                "finite_cells": [int(pit[f"pit_finite_t{j}"].sum()) for j in cuts],
                "oi_factors_with_finite_cells": [int((pit["uses_oi"] & (pit[f"pit_finite_t{j}"] > 0)).sum())
                                                 for j in cuts],
            },
            "values": {"shards": len(set(shard_of.values())), "factors_per_shard": SHARD_FACTORS,
                       "dtype": "float64", "rows": "one per (sunday, symbol)", "git_ignored": True},
            "evaluator": cstats,
            "timings": timings,
        },
    }
    timings["write_status_s"] = round(time.time() - t, 1)
    timings["total_s"] = round(time.time() - t_all, 1)
    _write_json(manifest, out_dir(src) / "manifest.json", source=src)
    mp = manifest["compute"]["pit"]
    log(f"PIT proof: {mp['pass']:,} pass, {mp['no_finite_cells']:,} with no finite cell to compare, {mp['fail']:,} fail "
        f"({sum(mp['compared_cells']):,} cells compared, {sum(mp['finite_cells']):,} finite)")
    log(f"wrote data/{OUT_NAME}/{BINANCE_SUBDIR}/status.parquet, manifest.json and {len(set(shard_of.values()))} "
        f"value shards (total {timings['total_s']}s)")
    return manifest


def run_screen_binance(log=print) -> dict:
    """--source binance --screen: the new family's G screen (BH over that family only) and the
    cross-check of the OKX-screened factors (agreement only); writes data/library_on_engine/
    binance/ only."""
    src = "binance"
    t_all = time.time()
    timings = {}
    man = _read_manifest(src)
    if "compute" not in man or not (out_dir(src) / "status.parquet").is_file():
        raise SystemExit("libfactors: run --source binance --compute first")
    # the weekly values are git-ignored, so the manifests can exist without values/:
    # the new family reads the Binance values, the cross-check the OKX ones as well
    missing = [s for s in ("okx", src) if not has_values(s)]
    if missing:
        raise SystemExit("libfactors: no stored weekly values for " + " and ".join(missing) + " (values/ is "
                         "git-ignored): run --compute, then --source binance --compute, before --source binance --screen")
    zones()
    ric_settings()
    now = binance_inputs()
    stale = [k for k in ("binance_panel", "library/catalog/factors.db", "okx_screen") if now.get(k) != man["inputs"].get(k)]
    if stale or code_fingerprint(BINANCE_CODE_FILES)["sha256"] != man["code"]["sha256"]:
        raise SystemExit(f"libfactors: the stored Binance values are stale ({', '.join(stale) or 'code changed'}); "
                         "run --source binance --compute again")
    t = time.time()
    st = pd.read_parquet(out_dir(src) / "status.parquet")
    okx_sg = pd.read_parquet(out_dir("okx") / "screen_g.parquet")
    cat = load_catalog().set_index("factor_id")
    raw_inputs = cat["raw_inputs"].to_dict()
    sundays, _ = g_sundays()
    labels = g_labels()
    panels_cut = build_panels(load_raw(), through=LABEL_CUTOFF)   # the OKX screen's regressors, unchanged
    timings["inputs_s"] = round(time.time() - t, 1)

    # ---- the new family: what the OKX screen did not screen
    t = time.time()
    ids = new_family_ids(st)
    if set(ids) & set(okx_sg["factor_id"]):
        raise AssertionError("a factor the OKX screen screened entered the new family")
    values, symbols, _ = read_values(ids, sundays, source=src)
    has_g = np.isfinite(values).any(axis=(1, 2))
    g_ids = [i for i, h in zip(ids, has_g) if h]
    g_vals = values[has_g]
    del values
    stats, ctx = factor_stats(g_vals, symbols, labels, panels_cut, PRIMARY)
    df, summary, weeks = screen_from_stats(stats, ctx, g_vals, g_ids)
    timings["new_family_screen_s"] = round(time.time() - t, 1)
    log(f"new family: {len(ids):,} candidates, {int(has_g.sum()):,} with G values, {summary['screened']:,} screened "
        f"({timings['new_family_screen_s']}s)")
    t = time.time()
    sens = sensitivity(g_vals, symbols, labels, panels_cut, primary=stats, log=log, notes=BINANCE_VARIANT_NOTES)
    timings["sensitivity_s"] = round(time.time() - t, 1)
    meta = st.set_index("factor_id").loc[df["factor_id"], ["name", "family", "category", "origin", "new_fields",
                                                            "taker_ratio_last_5m", "okx_screen"]].reset_index()
    df = pd.concat([meta, df.drop(columns=["factor_id"])], axis=1)
    keep = np.isin(np.array(g_ids, dtype=object), df["factor_id"].to_numpy())
    fam_vals = g_vals[keep]
    del g_vals

    # ---- the cross-check: the OKX-screened factors on Binance (no selection)
    t = time.time()
    cc_ids = crosscheck_ids(st)
    cvals, csyms, _ = read_values(cc_ids, sundays, source=src)
    if csyms != symbols:
        raise AssertionError("the cross-check values have another symbol order")
    c_has = np.isfinite(cvals).any(axis=(1, 2))
    cc_g = [i for i, h in zip(cc_ids, c_has) if h]
    cvals = cvals[c_has]
    cstats, cctx = factor_stats(cvals, symbols, labels, panels_cut, PRIMARY)
    okx_vals, osyms, _ = read_values(cc_g, sundays, source="okx")
    if osyms != symbols:
        raise AssertionError("the OKX values have another symbol order")
    rho, cells = cross_venue_rho(okx_vals, cvals, cctx["base"])
    cross = crosscheck_frame(cstats, cc_g, okx_sg, rho, cells)
    timings["crosscheck_s"] = round(time.time() - t, 1)

    # ---- overlap of the new family's factors at each q level with the OKX screen's factors at the same level
    t = time.time()
    pos = {f: i for i, f in enumerate(cc_g)}
    base = ctx["base"]
    v_fam = pooled_vectors(snap_ties(fam_vals), base)      # every screened new-family factor, once
    v_cc = pooled_vectors(snap_ties(cvals), base)          # the OKX-screened factors' Binance values, once
    overlap = {}
    for name, lev in OVERLAP_SETS:
        okx_ids = [f for f in okx_sg.loc[okx_sg[f"{name}_q"] <= lev, "factor_id"] if f in pos]
        sel = [pos[f] for f in okx_ids]
        col = f"okx_{name}_{_tag(lev)}_max_rho"
        df[col] = (np.abs(v_fam @ v_cc[sel].T).max(axis=1) if sel else np.full(len(df), np.nan)).astype(np.float32)
        surv = (df[f"{name}_q"] <= lev).to_numpy()
        via_okx = max_abs_rho(fam_vals[surv], okx_vals[sel], base) if surv.any() and sel else np.array([])
        overlap[f"{name}_{_tag(lev)}"] = {
            "okx_count": len(okx_ids), "new_family_count": int(surv.sum()),
            "overlapping": int((df.loc[surv, col] >= DEDUP_RHO).sum()),
            "overlapping_with_okx_values": int((via_okx >= DEDUP_RHO).sum())}
    timings["overlap_s"] = round(time.time() - t, 1)
    del cvals, okx_vals, fam_vals, v_fam, v_cc

    # ---- status: one screen reason per definition
    t = time.time()
    g_n = pd.Series(0, index=st["factor_id"], dtype=np.int64)
    g_n.loc[g_ids] = weeks
    g_n.loc[cc_g] = cstats["ic_n"].to_numpy().astype(np.int64)
    st["g_weeks"] = g_n.reindex(st["factor_id"]).fillna(0).astype(np.int16).to_numpy()
    pend = st["screen"] == "pending"
    is_fam = st["role"] == "new_family"
    no_g = set(ids) - set(g_ids)
    st.loc[pend & is_fam & st["factor_id"].isin(no_g), "screen"] = "no_g_values"
    st.loc[pend & is_fam & st["factor_id"].isin(set(df["factor_id"])), "screen"] = "screened"
    st.loc[pend & is_fam & (st["screen"] == "pending"), "screen"] = "too_few_weeks"
    st.loc[pend & ~is_fam & st["factor_id"].isin(set(cc_g)), "screen"] = "crosscheck"
    st.loc[pend & ~is_fam & (st["screen"] == "pending"), "screen"] = "no_g_values"
    cand = st[st["factor_id"].isin(set(ids))]
    counts = {
        "candidates": len(ids), "with_g_values": int(has_g.sum()), "no_g_values": int((~has_g).sum()),
        "candidates_by_origin": {k: int(cand["origin"].eq(k).sum()) for k in ORIGINS},
        "screened_by_origin": {k: int(df["origin"].eq(k).sum()) for k in ORIGINS},
        "screen_status": {k: int(v) for k, v in st.loc[is_fam, "screen"].value_counts().sort_index().items()},
        **summary,
        "q_sets": {f"{name}_{_tag(lev)}": _q_set_breakdown(df, name, lev) for name, lev in OVERLAP_SETS},
        "overlap_with_okx_q_sets": overlap,
    }
    top = df.reindex(df["ric_t"].abs().sort_values(ascending=False).index).head(15)
    top_rows = [{"factor_id": r.factor_id, "name": r.name, "family": r.family, "new_fields": r.new_fields,
                 "origin": r.origin, "ic_t": _r(r.ic_t, 2), "ric_t": _r(r.ric_t, 2), "ric_q": _r(r.ric_q),
                 "ric_fit_t": _r(r.ric_fit_t, 2), "ric_eval_t": _r(r.ric_eval_t, 2),
                 "persistence": _r(r.persistence, 3), "okx_ric_q10_max_rho": _r(r.okx_ric_q10_max_rho, 3)}
                for r in top.itertuples(index=False)]
    series = series_agreement(panels_cut, build_panels_binance(load_raw_binance(), through=LABEL_CUTOFF))
    atoms = {f: expr_atoms(cat.at[f, "expression"])[1] for f in cross["factor_id"]}
    cross_summary = crosscheck_summary(cross, raw_inputs, atoms, series)
    _write_frame(df, out_dir(src) / "screen_g.parquet", source=src)
    _write_frame(cross, out_dir(src) / "crosscheck.parquet", source=src)
    _write_frame(st, out_dir(src) / "status.parquet", source=src)
    timings["write_s"] = round(time.time() - t, 1)
    timings["total_s"] = round(time.time() - t_all, 1)
    man["inputs"] = {**now, "okx_engine_labels_and_regressors": {k: v for k, v in input_fingerprints().items()
                                                                 if k != "library/catalog/factors.db"}}
    man["screen"] = {
        "generated_at": _now(),
        "evidence": EVIDENCE,
        "family": (f"NEW FAMILY: definitions the OKX screen did not screen (OKX status != screened) that on Binance "
                   f"are computed, PIT pass, not degenerate (vwap_proxy) and have >= {MIN_WEEKS} weekly ICs on G; "
                   "Benjamini-Hochberg over this family only"),
        "label": f"{LABEL} from the label table (OKX), Sunday rows through {LABEL_CUTOFF.date()}",
        "regressors": BINANCE_CONVENTIONS["labels_and_regressors"],
        "mask": (f"label-table rows with finite factor, label and regressors; >= {MIN_NAMES} names a Sunday "
                 f"([g0] min_names_per_sunday)"),
        "precision": (f"ranks of the float64 values; per factor and Sunday, values within 1e-{TIE_DIGITS} of the "
                      "largest |value| are ties (snap_ties)"),
        "ric_regressors": ric_description(src),
        "dedup": f"greedy, |pooled within-Sunday rank correlation| >= {DEDUP_RHO}, strongest |t| first",
        "overlap": (f"okx_<stat>_<q>_max_rho: the largest |pooled within-Sunday rank correlation| (Binance values on "
                    f"both sides) with any factor of the OKX screen at that q level; overlapping = >= "
                    f"{DEDUP_RHO}. overlapping_with_okx_values uses the OKX values of those OKX factors instead"),
        "pooled_g_note": POOLED_G_NOTE,
        "sensitivity": sens,
        "counts": counts,
        "top_by_abs_ric_t": top_rows,
        "timings": timings,
    }
    man["crosscheck"] = {
        "generated_at": _now(),
        "evidence": CROSSCHECK_EVIDENCE,
        "set": ("the definitions the OKX screen screened, computed on Binance, PIT pass, with G values; their G "
                "statistics on the Binance panel next to the OKX screen's"),
        "counts": {"okx_screened": int((st["okx_screen"] == "screened").sum()), "binance_pit_pass": len(cc_ids),
                   "with_g_values": len(cc_g)},
        "value_rho": ("pooled within-Sunday rank correlation of the same definition's OKX and Binance values on G, "
                      "over cells finite on both"),
        **cross_summary,
    }
    _write_json(man, out_dir(src) / "manifest.json", source=src)
    log(f"new family screened {summary['screened']:,} factors on {summary['g_sundays']} G Sundays; BH over "
        f"{summary['n_tests_ic']:,} (IC) and {summary['n_tests_ric']:,} (rIC) tests")
    for name in ("ic", "ric"):
        log(f"  {name.upper():>3} q<=0.05: {summary[f'{name}_count_q05']:,} "
            f"({summary[f'{name}_distinct_signals_q05']:,} distinct), q<=0.10: {summary[f'{name}_count_q10']:,} "
            f"({summary[f'{name}_distinct_signals_q10']:,} distinct)")
    fs = summary["fit_selection"]["ric"]
    log(f"  rIC selected on G_fit alone (q<=0.05): {fs['q05']['selected']:,}; on G_eval {fs['q05']['eval_same_sign']:,} "
        f"keep their sign, {fs['q05']['eval_same_sign_t2']:,} with |t| >= 2 (base rate: {fs['base_same_sign']:,} of "
        f"{fs['base_pairs']:,})")
    log(f"cross-check: {len(cc_g):,} OKX-screened factors; Spearman of rIC t across venues "
        f"{cross_summary['ric_t_spearman']} ({cross_summary['ric_t_spearman_without_funding']} without the "
        f"{cross_summary['funding_readers']:,} that read funding), same sign {cross_summary['same_ric_sign_share']}; "
        "agreement only, not an independent replication (manifest crosscheck.agreement_note)")
    log(f"wrote data/{OUT_NAME}/{BINANCE_SUBDIR}/screen_g.parquet, crosscheck.parquet, status.parquet, manifest.json "
        f"(total {timings['total_s']}s)")
    return man


def show_status_binance() -> int:
    """Read-only summary of data/library_on_engine/binance/."""
    src = "binance"
    man = _read_manifest(src)
    if not man:
        print(f"data/{OUT_NAME}/{BINANCE_SUBDIR}/ has no outputs: run --source binance --compute, then --screen")
        return 0
    c = man["compute"]
    print(f"library factors on the Binance panel (data only; generated {man['generated_at']}, data through "
          f"{man['panels']['data_through']})")
    _print_states(man, src, "--source binance --compute and --screen")
    print("  roles: " + ", ".join(f"{k} {v:,}" for k, v in c["counts"]["roles"].items())
          + "; new family by origin: " + ", ".join(f"{k} {v:,}" for k, v in c["counts"]["new_family_by_origin"].items()))
    print(f"  computed ok {c['counts']['computed_ok']:,}; PIT pass {c['pit']['pass']:,}, no finite cell to compare "
          f"{c['pit']['no_finite_cells']:,}, fail {c['pit']['fail']:,}")
    fr = man["panels"].get("cleaning_freeze")
    if fr:
        now = cleaning_rules_fingerprint()["sha256"]
        print(f"  cleaning rules: the pinned set (sha256 {fr['sha256'][:12]}); the code "
              + ("has the same rules" if now == fr["sha256"] == CLEANING_FREEZE["sha256"] else
                 "has DIFFERENT rules, so --source binance --compute stops"))
    cl = man["panels"]["checks"].get("cleaning")
    if cl:
        print("  archive errors set missing (panels.checks.cleaning): " + "; ".join(
            f"{r} {x['cells']} ({', '.join(sorted({w['date'] for w in x['where']}))})" for r, x in cl.items()))
    s = man.get("screen")
    if not s:
        print("  screen: not run")
        return 0
    k = s["counts"]
    print(f"  new family: {k['screened']:,} screened, BH tests IC {k['n_tests_ic']:,} / rIC {k['n_tests_ric']:,}")
    for name in ("ic", "ric"):
        print(f"    {name.upper():>3} q<=0.05: {k[f'{name}_count_q05']:,} ({k[f'{name}_distinct_signals_q05']:,} "
              f"distinct); q<=0.10: {k[f'{name}_count_q10']:,} ({k[f'{name}_distinct_signals_q10']:,} distinct)")
    fs = k["fit_selection"]["ric"]
    print(f"    rIC selected on G_fit alone, q<=0.05: {fs['q05']['selected']:,}; G_eval same sign "
          f"{fs['q05']['eval_same_sign']:,}, with |t| >= 2 {fs['q05']['eval_same_sign_t2']:,} (base rate "
          f"{fs['base_same_sign'] / max(fs['base_pairs'], 1):.0%})")
    x = man.get("crosscheck", {})
    if x:
        print(f"  cross-check (no selection): {x['counts']['with_g_values']:,} OKX-screened factors, Spearman of rIC t "
              f"{x['ric_t_spearman']}, same sign {x['same_ric_sign_share']}; OKX factors at rIC q<=0.05 with the same "
              f"sign and |t| >= 2 on Binance: {x['okx_ric_q05']['binance_same_sign_abs_t_ge_2']} of "
              f"{x['okx_ric_q05']['okx_count']}")
        for key in ("agreement_note", "funding_note"):
            if x.get(key):
                print(f"    {x[key]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Library factors on Factorlab's data: compute them, prove them point "
                                             "in time and screen them on the zone G (numbers only).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true", help="read-only summary of data/library_on_engine/")
    g.add_argument("--compute", action="store_true", help="classify, compute, PIT proof, weekly values")
    g.add_argument("--screen", action="store_true", help="the G-zone screen (needs --compute)")
    ap.add_argument("--source", choices=SOURCES, default=DEFAULT_SOURCE,
                    help="panel source: okx (the engine's own panels, default) or binance (Binance public data; writes "
                         f"data/{OUT_NAME}/{BINANCE_SUBDIR}/ only)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"worker processes (at most {MAX_WORKERS})")
    args = ap.parse_args(argv)
    binance = args.source == "binance"
    if args.status:
        return show_status_binance() if binance else show_status()
    # checked before any computing: a protected target is refused at once
    C.guard_write(out_dir(args.source))
    if args.compute:
        (run_compute_binance if binance else run_compute)(workers=args.workers)
    else:
        (run_screen_binance if binance else run_screen)()
    return 0


if __name__ == "__main__":
    sys.exit(main())
