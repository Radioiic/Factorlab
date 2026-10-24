# Factor library

The factor library holds 28,904 rank-distinct formula definitions of cross-sectional
crypto factors. The `factorlab` Python package builds them, evaluates them and stores them
in a SQLite catalog. It also downloads Binance public data.

`./setup.sh` builds the catalog and the Binance panel. The engine then computes the
library on Factorlab's data and screens it (`engine/libfactors.py`, see
[engine/README_FINDER.md](../engine/README_FINDER.md)). The screens give numbers only.

The dashboard page Factor library shows the definitions with a search and the numbers of
the screens on Factorlab's data. `factorlab show` prints the command that checks a
definition and the command that registers it for a test (see
[Test a library formula](#test-a-library-formula)).

## What is in the library

Each factor is a formula over daily market panels (time x coin). The fields follow
Binance spot and USD-M perpetual data: OHLCV, trade counts, taker flow, funding, open
interest, long/short ratios and the premium index.

| Family | Factors | Idea |
|---|---:|---|
| `stat_rolling` | 7,141 | 19 rolling statistics (mean, EMA, std, z-score, rank, skew, kurtosis, autocorrelation, slope, ...) of 31 stationary series |
| `correlation` | 4,049 | rolling correlations between 23 series, price-volume correlations, autocorrelation at several lags |
| `interaction` | 3,666 | products, spreads and sign gates of z-scores from different information sets, and horizon spreads |
| `second_order` | 3,178 | z-score, rank or change of smoothed statistics |
| `lead_lag` | 3,036 | correlation of a lagged series X with the current series Y |
| `ma_crossover` | 2,418 | short/long SMA and EMA crossovers and average differences |
| `level_transform` | 2,099 | momentum, MA distance, range position, trend slope/R² and drawdown of price, volume, trades and OI levels |
| `conditional` | 1,511 | averages of a series on up/down bars, high/low-volume bars, market or BTC up/down bars, positive/negative funding bars, OI up/down bars |
| `technical` | 541 | RSI, stochastics, CCI, MFI, ADX/DI, Aroon, TRIX, Parkinson/GK/RS/Yang-Zhang volatility, Roll and Corwin-Schultz spreads, Kyle lambda, VPIN, variance ratios, ... |
| `momentum` | 510 | skip-period, risk-adjusted, excess-of-market/BTC and volume-weighted momentum, efficiency ratio |
| `market_beta` | 451 | beta, downside/upside beta, idiosyncratic volatility, residual momentum and alpha against the market, the liquidity-weighted market and BTC |
| `crypto_derivatives` | 197 | funding carry and crowd measures, OI/price divergence, leverage, a liquidation proxy, basis, smart-versus-crowd positions |
| `alpha101_style` | 107 | crypto versions of the Alpha101 formulas ("101 Formulaic Alphas", 2016, [arXiv:1601.00991](https://arxiv.org/abs/1601.00991)) |

- **Windows are counted in bars** (3 to 360). Thus the same library works on 1d, 4h or 1h
  data.
- **Data needs.** 17,295 factors use kline fields only (OHLCV, trades, taker flow). 11,609
  also need perpetual-futures data. The column `raw_inputs` lists the raw fields of each
  factor, and `needs_derivs` marks the second group.
- **Categories describe the input series, not the economics.** `stat.std.ex_btc_ret.w60`
  is a volatility factor, but its category is `momentum` because its input is a return
  series. Read the formula before you trust a category.
- **Use `name` as the permanent key** (for example `stat.zscore.funding.w20`). `factor_id`
  follows the generation order. It stays the same only while the generator and the
  validation data stay the same.

### Checks at build time

The generator crosses templates with input series and windows: 29,905 candidates. `build`
evaluates each candidate on a validation panel and drops it when:

- it fails to evaluate, or its coverage is below 15%;
- it has no cross-sectional dispersion (the same value for each coin) on more than half
  of the bars;
- it is **rank-equivalent** to an earlier factor: identical or exactly reversed
  cross-sectional ranks on each of the last 250 validation bars.

The default validation panel is synthetic (40 coins, 1,600 bars, seed 7). On this panel,
`build` removes 1,001 candidates and keeps 28,904. Thus "rank-distinct" means distinct on
the validation panel, not independent on real data. Many factors stay highly correlated on
market data.

The build is reproducible: the same code and the same validation panel give a database
with the same bytes. The `meta` table holds only the counts and the validation data, and
no clock time.

Each operator looks only backward when its window or lag is a positive whole number. A unit
test checks that a change of future data never changes a past factor value. The parser does
not check the windows: a negative lag, for example `delay(close, -1)`, reads a later bar.
`judge.py` refuses such a formula (see [Test a factor](../README.md#test-a-factor)).

## Commands

`./setup.sh` installs the package in `engine/.venv` in editable mode
(`engine/.venv/bin/python -m pip install --no-deps --no-build-isolation -e library`; the
pinned set in `engine/requirements.txt` supplies the dependencies). It also adds the
command `engine/.venv/bin/factorlab`, which works in any folder. In `library/`,
`python -m factorlab` with the engine interpreter (`../engine/.venv/bin/python -B`) gives
the same commands. `factorlab --help` and `factorlab COMMAND --help` show each option.

```bash
factorlab summary                            # counts for each family and category
factorlab list --category funding --limit 20
factorlab list --like "%ex_btc_ret%" --limit 10
factorlab show CF00001                       # one definition and its test commands
factorlab build                              # build catalog/factors.db (about 40 s)
```

| Command | Function | Writes |
|---|---|---|
| `summary` | counts for each family and category | nothing |
| `list` | list and filter factors | nothing |
| `show FACTOR` | one factor by id or name, and the engine commands that check it and register it for a test | nothing |
| `build` | generate, validate, de-duplicate and store the library | `catalog/factors.db` and its exports |
| `compute` | compute factor values on your panels and store them | the database (not the catalog without `--overwrite-catalog`) |
| `evaluate` | compute IC, ICIR and long-short statistics and store them | the database (not the catalog without `--overwrite-catalog`) |
| `download` | download Binance public panels | the folder of `--out` |

The global option `--db PATH` selects the database (default `catalog/factors.db`, found
in the package folder, so the commands work from any folder).

| Options | Commands | Description |
|---|---|---|
| `--family`, `--category`, `--like`, `--ids`, `--limit` | `list`, `compute`, `evaluate` | filter the factors (`--ids` is a comma list) |
| `--data DIR` | `compute`, `evaluate`, `build` | a folder with raw panels (`<field>.csv.gz`) |
| `--benchmark SYMBOL` | `compute`, `evaluate` | the benchmark column (for example `BTCUSDT`) |
| `--horizons`, `--dataset`, `--periods-per-year` | `evaluate` | forward horizons in bars, a dataset label, the annualization (365 for daily bars) |
| `--workers N`, `--export DIR` | `build` | worker processes; the folder of the CSV and Markdown export |
| `--overwrite-catalog` | `build`, `compute`, `evaluate` | permit a write into `catalog/` |
| `--symbols`, `--start`, `--end`, `--interval`, `--market um\|spot`, `--no-derivatives`, `--metrics-archive`, `--out` | `download` | what to download, and to where |

**The catalog.** `build` writes `catalog/factors.db` when it does not exist. It refuses to
replace a catalog that is already there without `--overwrite-catalog`. `compute` and `evaluate` refuse
the catalog folder without `--overwrite-catalog`: work on a copy under `data/`. The read
commands open the catalog read-only. Without a catalog they print
"No catalog. Run ./setup.sh first."

```bash
mkdir -p data && cp catalog/factors.db data/factors.db
python -m factorlab --db data/factors.db evaluate --data data/binance_um_1d --family momentum --horizons 1,5,20
python -m factorlab --db data/factors.db compute --data data/binance_um_1d --ids CF00001,CF00002
```

### Test a library formula

`factorlab show ID` prints the two commands of a definition. Run these commands from the
repository root:

```bash
cd engine
.venv/bin/python -B judge.py check --library CF00001      # the historical test; writes nothing
.venv/bin/python -B judge.py register --library CF00001   # register it for a test
```

The test runs in the engine on Factorlab's data. See the section
[Test a factor](../README.md#test-a-factor) of the main README.

### Python API

Run the examples with the engine interpreter. After `./setup.sh`, the package imports in
any folder. Without the editable install, run them from `library/`.

```python
from factorlab import FactorDB, Evaluator
from factorlab.data import load_panels

db = FactorDB("catalog/factors.db", readonly=True)
panels = load_panels("data/binance_um_1d")
lib = db.factors(available_fields=panels.keys())  # only the factors that your data supports
ev = Evaluator(panels, benchmark="BTCUSDT")
values = ev(lib.expression[0])  # DataFrame: time x symbol
```

With spot OHLCV only, pass `available_fields=panels.keys()` to skip the factors whose
inputs you do not have.

An example without data and without network, on synthetic panels:

```python
from factorlab import Evaluator
from factorlab.data.synthetic import make_synthetic_panel
from factorlab.evaluate import evaluate_factor

panels = make_synthetic_panel(n_symbols=20, n_bars=600, seed=1)  # synthetic, not market data
values = Evaluator(panels, benchmark="BTCUSDT")("-1 * ts_mean(ex_ret, 20)")  # an example of the syntax
print(evaluate_factor(values, panels["close"], horizon=7))  # IC, ICIR, t, turnover, ...
```

### Expressions

An expression is a small subset of Python syntax, for example
`ts_zscore(log(close / delay(close, 5)), 20) * -1`. It can have numeric constants, raw
and derived field names, the registered operators, unary `+` and `-`, and the binary
operators `+ - * / **`. The parser never passes text to `eval`. These limits apply:

- at most 4,000 characters (`MAX_EXPRESSION_LENGTH`);
- at most 64 nested nodes (`MAX_EXPRESSION_DEPTH`);
- a power of two constants needs an exponent of at most 100 in magnitude
  (`MAX_CONSTANT_EXPONENT`) and a finite result.

Other input, and input outside these limits, gives `ExpressionError` (a `ValueError`).
The longest catalog expression has 608 characters and 17 nested nodes.

## The Binance panel

`engine/update.py` (and thus `./setup.sh`) makes the Binance USD-M panel of the engine's 20
coins:

1. `download` once for each coin, from 2022-01-01 to the last Sunday that closed at least
   one full day before the run (UTC), into
   `data/binance_um_1d_parts/<SYMBOL>/`. Several coins download at the same time
   (`--jobs`).
2. `scripts/merge_panels.py` merges the parts offline into `data/binance_um_1d/`: one file
   for each field, one column for each coin.
3. On each update, `download` gets the new closed days into
   `data/binance_um_1d_parts/updates/<FIRST>_<LAST>/`, and `merge_panels.py --append` adds
   them.

The same steps by hand, for one coin:

```bash
FACTORLAB_OFFLINE=0 python -B -m factorlab download --symbols BTCUSDT --start 2022-01-01 \
    --end "2026-10-04 23:59:59" --interval 1d --market um --metrics-archive \
    --out data/binance_um_1d_parts/BTCUSDT
python -B scripts/merge_panels.py          # offline: data/binance_um_1d_parts/ -> data/binance_um_1d/
```

The panel has 16 fields. The bars have the stamp of their UTC open time. Funding is the
sum of the settlements in each bar. Premium is the close of the daily premium-index
kline. Open interest and the ratios are the last non-empty 5-minute snapshot of the bar,
from the data.binance.vision metrics archive. `--metrics-archive` is necessary because the
REST API serves only about 30 days of open interest and ratios.

**The downloader.** It opens only `https` URLs. Each data.binance.vision zip file must
agree with the SHA-256 in its `.CHECKSUM` file before the downloader reads it. A missing
checksum file, or a different digest, is an error (`ArchiveError`). Each response body has
a size limit. The limits are 16 MiB (a REST response), 50 MiB (a zip file) and 200 MiB (the
CSV file in a zip file). A larger body is an error (`ResponseTooLarge`).

A rate limit (HTTP 429), a server error, a network error or a timeout
causes a retry. The waits are 1, 2, 4, 8 and 16 seconds; after HTTP 429, the wait is at
least the `Retry-After` time that Binance sends. HTTP 418 means that Binance has banned
the IP address for a time. A retry makes the ban longer, so the download stops at once
with an error. Wait, then run again with fewer parallel downloads (`--jobs`). The downloader skips an
archive file that does not exist (HTTP 404). On `--market um`, a funding or derivatives
download that fails after its retries stops the download with an error. Thus a part
folder never lacks those fields.

**The merge.** The merge writes each file with a fixed gzip header time. Thus a merge of the
same parts gives byte-identical files. The merge refuses `catalog/` as an output.

| Option | Effect |
|---|---|
| `--parts DIR` | the folder of the per-coin part folders (default `data/binance_um_1d_parts`) |
| `--out DIR` | the merged panel (default `data/binance_um_1d`) |
| `--symbols LIST` | the coins, in column order (default: the engine's 20 coins) |
| `--append` | add new closed bars only |
| `--through DAY` | with `--append`: add no bar after this UTC day |
| `--now TIME` | with `--append`: the clock of the closed-bar rule (UTC; for tests) |

**The append** (`--append --parts DIR`) adds new closed bars and changes no cell that is already there.
It stops with exit 2 and writes nothing when:

- the new bars do not follow the last merged bar day by day;
- a new bar is not closed;
- a coin with a kline bar on the last merged bar has no kline bar on a new day;
- a coin has no value in a field that it had on the last merged bar. An example is an
  archive file that Binance had not published at download time. Download that day again after one day,
  or stop before it with `--through`;
- a downloaded bar that the panel already has differs from it.

Each append adds one line to `appends.jsonl` in the merged folder.

The merged cells stay as downloaded, archive errors included. `engine/libfactors.py`
sets the archive errors to empty with three fixed cleaning rules before it computes a
factor.

## Tests

```bash
cd library
../engine/.venv/bin/python -B -m pytest tests -q
```

Or from the repository root: `engine/.venv/bin/python -B engine/run_tests.py --only library`.
The suite needs `pytest` (`./setup.sh --test` installs it). The property tests need the
package `hypothesis`. `./setup.sh --test` installs it too. Without it, these tests skip.

| Suite | What it checks |
|---|---|
| `test_factorlab.py` | operators, the expression parser, the generator, no look-ahead |
| `test_catalog.py` | the built catalog: unique ids and names, each expression parses, the exports agree |
| `test_cli.py` | the read commands change no byte of the catalog; `show` prints the check and register commands of `engine/judge.py`; `build` writes an absent catalog and refuses to replace one; the other writers refuse the catalog folder |
| `test_build.py` | two builds of the same inputs give the same database bytes; an in-place build replaces the meta table; the export is reproducible |
| `test_expr_limits.py` | the length, depth and constant-power limits of the expression parser |
| `test_expr_property.py` | random text into the parser: each input parses or gives `ExpressionError` (needs `hypothesis`; without it, the suite skips) |
| `test_network.py` | `download` sends no request unless `FACTORLAB_OFFLINE=0` |
| `test_binance_retry.py` | the retry waits, a missing archive file (404) gives no value, the error after the last retry, and an error on a perpetual without its derivatives fields |
| `test_binance_checksum.py` | the SHA-256 check of archive files, the size limits, the https rule and the zip parser |
| `test_merge_panels.py` | the merge and the append rules on synthetic parts |
| `test_stats.py` | the Benjamini-Hochberg q-values |

`pytest.ini` disables the pytest cache. `tests/conftest.py` puts the temporary files in
`data/pytest-tmp/`, inside the repository.

## Files

```
catalog/                      built by `build` (git ignores it)
  factors.db                  SQLite: factors, raw_fields, derived_fields, operators, meta, factor_values,
                              factor_stats
  crypto_factors.csv.gz       the factor table as CSV
  summary_by_family_category.csv, SUMMARY.md
factorlab/
  generator.py   the 13 families crossed with series and windows
  operators.py   time-series and cross-sectional operators (no look-ahead)
  fields.py      raw fields and derived series (returns, ranges, flow, funding, OI, ...)
  expr.py        the expression parser and evaluator, with a sub-expression cache
  build.py       generate, validate, de-duplicate, store
  evaluate.py    rank IC, ICIR, t statistic, long-short spread, turnover
  stats.py       the Benjamini-Hochberg q-values of the screens (engine/libfactors.py)
  database.py    SQLite access (read-only mode for the read commands)
  paths.py       the catalog location and the guard against a change of it
  network.py     the offline switch: download only with FACTORLAB_OFFLINE=0
  cli.py         python -m factorlab ...
  data/          the Binance downloader, synthetic panels, panel storage
scripts/         merge_panels.py (it has --help)
tests/           the test suite
data/            work files and the Binance panel (git ignores it)
pyproject.toml   the package metadata, the factorlab command and the docstring check (ruff)
pytest.ini       the pytest settings of the suite
```

## Network

`download` is the only command that uses the network. It runs only with
`FACTORLAB_OFFLINE=0`. Unset or `1` refuses before the first request. Any other value is an
error. The CLI checks the switch before it imports the downloader, and the downloader
checks it again before each request.

## Data sources and licences

Factorlab's license (MIT) covers its code only. It gives no right in this data.

- **Binance public market data** (`download`): the REST endpoints api.binance.com and
  fapi.binance.com are under the Binance Terms of Use. The data.binance.vision archive is
  under the
  [Binance Vision Dataset Terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md):
  CC BY-NC-SA 4.0, noncommercial use only, no live proprietary trading. If you share
  anything derived from it, credit "Binance Vision" and use CC BY-NC-SA 4.0. Factorlab
  uses Binance as a data source only. The panel stays in the git-ignored `data/` folder.
- The repository holds no market data and no outputs built from it. Do not commit or
  publish `data/`, `catalog/` or the dashboard data.
- Factorlab is not affiliated with or endorsed by Binance.
- The code of the library is under Factorlab's license
  ([MIT License](../LICENSE)). The data terms above are separate.

## Limits

- A formula from the library is a formula, not evidence. A rank-IC t statistic is a
  number. It is not a test of a book. To test a formula in a book, use its register
  command (see [Test a library formula](#test-a-library-formula)).
- Reported volume can include wash trades.
- The library models no borrow cost for a short spot position.
