# Factor Finder: the library screen and the generator probe

This document describes two search tools of Factorlab. Both give numbers only:

1. **The library screen** (`libfactors.py`). It computes the factor library on
   Factorlab's own data and screens each factor on the generator zone G.
2. **The generator probe** (`finder.py --generator-probe`). It draws random formulas and
   gives the bands of their largest absolute correlation with the known factors. It reads
   no return label.

Run each command from `engine/` with `.venv/bin/python -B`. Both functions are offline.

## The zones

`factorfinder/eval_config.toml` fixes the weekly zones. Each zone is a range of Sundays.
The label of a Sunday D is `fwd_ret_7`: close(D+7) / close(D) - 1 of the OKX perpetual.

| Zone | Sundays | Count | Use |
|---|---|---:|---|
| G_fit | 2022-04-03 to 2023-07-23 | 69 | search and fit |
| embargo | 2023-07-30 | 1 | not used |
| G_eval | 2023-08-06 to 2024-06-09 | 45 | check of a choice made on G_fit |
| embargo | 2024-06-16, 2024-06-23 | 2 | not used |
| I (sealed) | 2024-06-30 to 2026-09-20 | 117 | not read by the screen or the probe |

The screen reads labels only through the 2024-06-09 row. A label after that row raises
`LabelLeak`. The probe reads no label. The seal applies to the library screen and the
probe only: the historical test of your own factor (`judge.py`) reads all history.

## The library screen

### Commands

```bash
.venv/bin/python -B libfactors.py --compute                    # OKX panels
.venv/bin/python -B libfactors.py --screen
.venv/bin/python -B libfactors.py --status                     # read-only
.venv/bin/python -B libfactors.py --source binance --compute   # Binance panel
.venv/bin/python -B libfactors.py --source binance --screen
.venv/bin/python -B libfactors.py --source binance --status
```

`update.py` runs these four build steps on each update. Each step takes 1 to 2 minutes. `--workers N` sets the worker processes (at most 8).

The order is fixed. `--screen` needs `--compute`. The Binance source needs the OKX screen,
because it reads the OKX status and screen files.

### What --compute does

1. **Classify.** It reads the 28,904 definitions of `library/catalog/factors.db`. It
   finds the raw fields of each definition and marks the definitions that the panels
   cannot supply.
2. **Compute.** It computes each computable definition with the library's own Evaluator
   on the engine's panels. It stores the weekly (Sunday) values in `values/`.
3. **Prove point in time.** It computes each factor again on panels cut at 2024-06-09,
   2026-06-28 and 2026-09-13. On each Sunday up to the cut, the cut values must equal the
   full values (`np.isclose`, rtol 1e-9, atol 1e-12, NaN equal to NaN).

| `pit_check` | Description |
|---|---|
| `pass` | Each compared cell is equal, and at least one finite cell was compared. |
| `no_finite_cells` | Each compared cell is equal, but no cell was finite. Nothing is proven. Not screened. |
| `fail` | A cell differs, or a cut computation failed. Not screened. |

### The OKX panels

| Field | Source |
|---|---|
| open, high, low, close, volume | the OKX perpetual daily bars (volume in base-asset units) |
| quote_volume | close x volume (USDT) |
| funding_rate | the daily sum of funding settlements (`data_io.daily_funding`) |
| open_interest_value | OKX open interest in USD (`data_io.daily_oi`) |
| open_interest | open_interest_value / close |

The OKX data has no premium index, trade counts, taker flow or long/short ratios. The
definitions that need them are not computable on OKX. Because quote_volume is close x
volume, the library's `vwap` equals close. `--compute` computes the definitions that read
`vwap` or `vwap_dev`. It gives them the flag `vwap_proxy`, and the screen does not use them.
OKX open interest starts after G, so no open-interest factor reaches the OKX screen.

### The Binance panel

`--source binance` reads the Binance USD-M panel in `library/data/binance_um_1d/`.
`update.py` downloads and merges it (see [library/README.md](../library/README.md)). All 16
raw fields of the library are present, so each definition is computable. Where Binance
refuses access (HTTP 451), run `update.py --skip-binance`: then Factorlab has no Binance
screen, and `judge.py` cannot test a formula on the Binance data. With `update.py --demo`,
the panel holds synthetic demo data.

| Field | Source and time |
|---|---|
| open, high, low, close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume | the 1d kline with open time D |
| funding_rate | the sum of the settlements in [D 00:00, D+1 00:00) |
| premium | the close of the 1d premium-index kline with open time D |
| open_interest, open_interest_value, long_short_ratio, top_long_short_ratio, taker_buy_sell_ratio | the last non-empty 5-minute snapshot of the data.binance.vision metrics file in bar D |

`taker_buy_sell_ratio` is the ratio of that 5-minute snapshot, not of the whole day. Each
input of each factor comes from the Binance panel. The labels and the regressors stay the
OKX ones, so the two screens differ only in the factor inputs.

**Cleaning rules.** Three fixed rules set archive errors to empty before any factor reads
the panel. Each rule reads one coin on one bar:

- open interest or its value <= 0: both become empty;
- quote_volume / volume outside [low, high]: volume becomes empty;
- taker_buy_quote_volume / taker_buy_volume outside [low, high]: taker_buy_volume becomes
  empty.

The manifest lists each cell that a rule changed (`panels.checks.cleaning`).
`libfactors.CLEANING_FREEZE` pins the hash of the rules. `--compute` stops when the rules
in the code differ from the pinned set.

### What --screen does

| Item | Rule |
|---|---|
| Sundays | G_fit and G_eval, without the embargo Sunday 2023-07-30 |
| Names | a Sunday gives an IC only with 15 or more names with a value, a label and the regressors |
| Ties | values within 1e-12 of that Sunday's largest absolute value are ties |
| IC | weekly Spearman IC: mean, sd, t = mean / sd x sqrt(n), two-sided p |
| rIC | each Sunday, a ridge regression (lambda 1e-3) on three regressors removes their part from the factor ranks; rIC is the Spearman IC of the residual with the label |
| Regressors | `pit_symbol_tilt`: the shrunk expanding mean of the daily excess return, n / (n + 180) x mean. `btc_beta_90d`: the beta to the BTC perp over 90 returns. `size_proxy`: the log of the 30-day mean dollar volume. |
| Family | factors with at least 20 weekly ICs |
| BH | Benjamini-Hochberg q-values over the family, for IC and rIC, on the pooled G p-value |
| Distinct signals | the definitions at a q level (for example rIC q <= 0.05), grouped by pooled rank correlation (absolute value >= 0.7) |
| Persistence | the mean week-over-week rank correlation of the factor |
| Replication | BH on the G_fit p-values alone (`ic_fit_q`, `ric_fit_q`), then the sign and t of the selected factors on G_eval |

Raw IC rewards a factor that only ranks coins by their past mean return. Thus rIC is the
statistic to read. A q-value over all of G is in sample on G_eval. Use the G_fit
selection and its G_eval check to read replication.

`manifest.json` (`screen.sensitivity`) gives the counts at each q level for each single
change of the settings. The changes are a 5-name floor, OLS in place of ridge, an unshrunk
tilt and no tilt. It also gives them for `label_mean`. In that variant, the tilt is the mean
weekly label over the Sundays up to D - 7 (0 below 8 labels). The variant uses OLS and 5
names. `--status` prints them.

**The Binance new family.** The OKX screen did not screen some definitions. They get the
same screen on the Binance panel, with their own Benjamini-Hochberg family. The `origin`
column gives the reason: `binance_only_fields`, `oi`, `vwap` or `other`.

**The cross-check.** The definitions that the OKX screen screened also get their G
statistics on the Binance panel (`crosscheck.parquet`). This is agreement only: no
q-value and no list at a q level. Both screens use the same labels, regressors and coins.
Before the first OKX funding settlement on disk, the engine's funding is the Binance
archive. Thus the two screens agree for these reasons. Their agreement is not an
independent replication.

### Outputs

| File | Content |
|---|---|
| `data/library_on_engine/status.parquet` | one row for each definition: group, absent fields, flags, compute result, PIT check, screen status |
| `data/library_on_engine/screen_g.parquet` | one row for each screened factor: IC and rIC on G, G_fit and G_eval, q-values, persistence, coverage, clusters |
| `data/library_on_engine/manifest.json` | zones, counts, settings, input hashes, code hash, run times |
| `data/library_on_engine/values/` | the weekly float64 values (about 0.5 GB) |
| `data/library_on_engine/binance/` | the same for the Binance panel, plus `crosscheck.parquet` (values about 1 GB) |

`--screen` refuses values whose inputs or code changed since `--compute`. The Binance
run also records the hashes of the OKX status and screen files. When they change, run
the Binance compute and screen again.

A screen gives numbers. To test a definition in a book, use its check and register
commands: `judge.py check --library ID` and `judge.py register --library ID`. The section
Test a factor of the main [README.md](../README.md#test-a-factor) gives the details.

## The generator probe

```bash
.venv/bin/python -B finder.py --generator-probe                  # 2,000 formulas, seed 1
.venv/bin/python -B finder.py --generator-probe --n 500 --seed 2
.venv/bin/python -B finder.py --check                            # read-only environment checks
```

The probe needs the OKX library values (`libfactors.py --compute`) and the label table.
2,000 formulas take about 12 minutes. The probe writes nothing. It reads these inputs:

- bars and funding, cut at the 2024-06-09 row;
- the library value shards;
- the date and symbol columns of the label table, without rows after the cut;
- the G rows of the AlphaForge table `data/research/alphaforge_daily.parquet`, when it exists;
- the trial ledger `data/trials/` (your registered tests), and for a Binance formula the
  Binance panel cut at the 2024-06-09 row.

### The terminals

A terminal is an input series of a formula. The probe has 25 terminals
(`[generator.terminals]` of `eval_config.toml`):

| Group | Terminals |
|---|---|
| Library returns | `ret`, `ex_ret`, `ex_btc_ret`, `oc_ret`, `gap`, `abs_ret` |
| Library bar shape | `hl_range`, `log_hl`, `body`, `upper_shadow`, `lower_shadow`, `clv`, `tr_pct` |
| Library variance | `parkinson_var`, `gk_var`, `rs_var` |
| Library volume and funding | `dvol_chg`, `amihud`, `funding` |
| Market-wide | `mkt_ret`, `btc_ret` (one value for each coin on a date) |
| Factorlab only | `sbasis` (perp close / spot close - 1), `perp_range_excess` (perp high-low range - spot range), `deriv_share_chg` (1-day change of log perp/spot dollar volume), `carry_ret` (ret - funding_rate) |

A formula with market-wide leaves only is constant on each date and fails the G0 check.
Market-wide terminals rank coins only together with a coin series, for example in `ts_beta`,
`ts_corr`, `ts_cov` or `ts_resid`. `carry_ret` tracks `ret`. Open interest, `vwap`,
`vwap_dev`, price and volume levels and `log_amount` are not terminals.

### The operators and the limits

The operators are the 46 operators of `library/factorlab/operators.py` plus `+`, `-`, `*`
and unary minus (`factorfinder/libops.py`, `FINDER_REGISTRY`). The probe never draws
`exp`, `clip`, `pct_change`, `ts_maxdd`, `ts_maxru`, `cs_mean` and `cs_wmean`.

| Limit | Value |
|---|---|
| Windows | 3, 5, 7, 10, 14, 20, 30, 45, 60, 90 days |
| Lags | 1, 2, 3, 5, 7, 10, 14, 20, 30 days |
| Tree depth | at most 3 |
| Nodes | at most 20 |
| Nested lookback | at most 180 days |

### The steps for each formula

| Step | Rule |
|---|---|
| Draw | ramped half-and-half trees; the probe draws a tree over the limits again and simplifies each tree |
| Evaluate | values on the 114 G Sundays |
| G0 | coverage >= 0.80 of the Sundays (a Sunday counts with >= 15 coins); finite share >= 0.95; no Sunday with one value for all coins; >= 10 distinct values on each counted Sunday |
| Lint | values after 2023-07-30 replaced with noise must leave each earlier value bit-identical; no value before the first bar of a coin |
| Dedup | identical ranks, or exactly reversed ranks, count once |
| Correlation | the largest absolute correlation with any member of the novelty index |
| Null | the same maximum for copies with the coins permuted: the chance floor |

The report counts the formulas in each band of the largest absolute correlation:

| Band | Largest absolute correlation |
|---|---|
| `below_050` | below 0.50 |
| `from_050_to_085` | 0.50 to 0.85 |
| `from_085_library` | 0.85 or more, and the nearest member is a library definition |
| `from_085_other` | 0.85 or more, and the nearest member is a user test, a zoo formula or a risk proxy |
| `no_overlap` | no member with a correlation |

A formula above 0.999 is also counted as value-identical.

### The novelty index

The index holds the G-Sunday ranks of each known factor. The probe builds the index in
memory and never writes it to disk. Its members have four sources:

| Source | Members |
|---|---|
| library | each library definition with four properties. It is computed and is not a `vwap_proxy`. It is proven point in time at the 2024-06-09 cut. It has at least 20 G Sundays with 15 or more names. |
| user | each registered test of `data/trials/` that is not withdrawn (`user:FT-0001`, ...). A formula: its values on the panels of its source, cut at the 2024-06-09 row. A values file: the history values of its registration. |
| zoo | each AlphaForge formula of `data/research/alphaforge_daily.parquet`, active or not, when the table exists |
| proxy | the three rIC regressors `pit_symbol_tilt`, `btc_beta_90d` and `size_proxy` |

The statistic is the pooled within-Sunday rank correlation on the coins that both
factors have, over at least 20 Sundays. The index measures each member exactly.

The probe report gives each count and share for all formulas. It also gives them for two
groups: formulas with library terminals only, and formulas with a Factorlab-only
terminal. The report gives numbers only. It does not show if a formula predicts returns.

## The environment check

`finder.py --check` is read-only. It checks:

| Check | Pass condition |
|---|---|
| `data root` | information only: the data folder that the checks read |
| `eval_config` | `eval_config.toml` loads and is valid; it prints the eval and generator version hashes |
| `data-through` | the newest closed daily bar on disk is not after the UTC date |
| `no network imports` | `ingest.py` and `ccxt` are not imported |
| `probe inputs` | information only: the files that the probe reads |

Exit codes: 0 ok, 1 a FAIL, 2 an invalid option value.

## eval_config.toml

`factorfinder/eval_config.toml` holds the label, the zones, the rIC regressors, the G0
limits and the generator settings. `factorfinder/config_io.py`
gives it two version hashes: the eval version (all tables except `[generator]`) and the
generator version (`[generator]` only). A change of a comment changes no version. A
change of a value does. `libfactors.py` stops when a zone or regressor value differs from
its own constants.

## Tests

| Suite | What it checks |
|---|---|
| `test_libfactors.py` | the OKX panel adapter, the PIT proof, the screen statistics, the label seal |
| `test_libfactors_binance.py` | the Binance adapter, the cleaning rules and their freeze, the new family and the cross-check |
| `test_finder_fieldmap.py` | the 25 terminals, their point-in-time property and their definitions |
| `test_finder_libops.py` | the operator registry, causality of each operator, the tree limits |
| `test_finder_novelty.py` | the exact correlation against brute force, the label seal of the index, the four sources |
| `test_finder_probe.py` | the G0 checks, no write, the same report for the same seed, the bands, numbers only |
| `test_finder_isolation.py` | `finder.py --check` writes nothing and imports no network module |
| `test_incumbents.py` | the incumbents of the factor lab and AlphaForge, and the number for the library |

Run them from `engine/` with `.venv/bin/python -B run_tests.py --only NAME`. A suite skips its data checks when the
data is not on disk.
