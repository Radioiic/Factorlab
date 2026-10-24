# Factorlab engine: command reference

The engine is a set of Python programs. Each program does one function. This document
lists each program with its options and its outputs, grouped by function.

Run each command from the `engine/` folder with the engine interpreter:

```bash
cd engine
.venv/bin/python -B judge.py status
```

The examples below write `python` for `.venv/bin/python -B`. Each program shows its
options with `--help`. `--help` runs nothing and writes no file.

## Common rules

- **Data folder.** Each program reads and writes `engine/data/` (`config.DATA`). Set
  `FACTORLAB_DATA=<folder>` to use another folder, for example a copy.
- **Logs.** `update.py` writes its lock and its log to `engine/logs/` (`config.LOGS`).
  `FACTORLAB_LOGS` changes it to another folder inside the repository.
- **Network.** Only `ingest.py` and `python -m factorlab download` (in `library/`) use the
  network. They run only with `FACTORLAB_OFFLINE=0`. Unset or `1` means offline. Any other
  value stops the program.
- **Write protection.** A data folder with the file `.read_only` is protected. Then each
  writer refuses to write into its protected folders (`config.PROTECTED_DIRS`: `raw`,
  `labels`, `research`, `trials`). The error says "refusing to write". The folder
  `library_on_engine` stays writable. `FACTORLAB_WRITE_PROTECTED=1` allows the write.
- **Time.** Each timestamp is UTC. A daily bar has the stamp of its open time. The bar
  stamped D closes at D+1 00:00 UTC. A factor value stamped D uses only data known at that
  close.
- **Rebuild.** `labels.py` and `libfactors.py --compute` build their tables again from the
  raw files on each run.
- **Version.** The file `VERSION` at the repository root holds the version
  (MAJOR.MINOR.PATCH). `python version.py` prints it, and `version.VERSION` gives it to
  the other programs. `python version.py --changelog` prints the section of the version in
  [../CHANGELOG.md](../CHANGELOG.md), which lists the changes of each version.

## Contents

- [Data](#data)
- [Labels](#labels)
- [Find factors](#find-factors)
- [Test a factor](#test-a-factor)
- [The trial ledger](#the-trial-ledger)
- [Dashboard and tests](#dashboard-and-tests)
- [Modules without a command line](#modules-without-a-command-line)
- [The data folder](#the-data-folder)
- [Configuration](#configuration)
- [Change the venue or the universe](#change-the-venue-or-the-universe)

## Data

### ingest.py: download the OKX data

```bash
python ingest.py --check                     # row counts and date ranges; no network
FACTORLAB_OFFLINE=0 python ingest.py         # download each new closed day
FACTORLAB_OFFLINE=0 python ingest.py --spot-only   # only the same-venue spot bars
```

| Option | Effect |
|---|---|
| `--check` | Show the files on disk. Fetch nothing. Works offline. |
| `--spot-only` | Fetch only the OKX spot daily bars. |

The first download starts at 2022-01-01 (`config.BACKFILL_START`). Each new run fetches
a small overlap window and the new days. A bar that the exchange restates in the overlap
window takes the new value. A bar that is not closed is never written.

Outputs, one Parquet file for each coin (`BTC-USDT.parquet`, ...):

| Folder | Content |
|---|---|
| `data/raw/ohlcv/` | perpetual daily bars: ts, open, high, low, close, volume |
| `data/raw/spot_ohlcv/` | OKX spot daily bars of the same coins |
| `data/raw/funding/` | OKX funding settlements (OKX serves about 3 months) |
| `data/raw/funding_binanceusdm/` | Binance USD-M funding settlements (the older history) |
| `data/raw/open_interest/` | OKX open interest (OKX serves recent months only) |

The daily funding series uses OKX settlements where they exist and Binance settlements
before them. The rule works on each 8-hour settlement slot (`data_io.daily_funding`).

### db.py: SQL over the data

```bash
python db.py "SELECT symbol, count(*) AS bars, max(ts) AS last_bar FROM ohlcv GROUP BY 1 ORDER BY 1"
python -i db.py          # then: q("SELECT count(*) FROM labels")
python db.py --help      # the options
```

DuckDB reads the Parquet files in place. The views are `labels` (the label table),
`ohlcv`, `funding` and `funding_fb` (the fallback funding prints). A view exists only when
its files exist. The words of the SQL argument are joined with spaces. Without SQL, the
program only makes the views. A data folder path with a quote character works.

## Labels

### labels.py: the label table

```bash
python labels.py            # build data/labels/labels.parquet from the OKX daily bars
python labels.py --status   # show the label table and compare it with a new computation; write nothing
```

| Column | Definition |
|---|---|
| `date` | the UTC open time of the bar of day D |
| `symbol` | the file name of the coin, for example `BTC-USDT` |
| `fwd_ret_1` | close(D+1) / close(D) - 1: the return of the bar after D |
| `fwd_ret_7` | close(D+7) / close(D) - 1: the return of the 7 bars after D. A book earns it from a Monday trade on the values of Sunday D. |

A label uses bars after D. It is a target, never an input of a factor value of D. A label
is empty until its last bar is on disk. A coin gets its first row when its 90-day return is
formed (`WARMUP_ROWS`). Thus a new coin does not change the earlier rows of the other coins.
The shifts are row shifts on the bars of each coin: when a bar is missing, a label spans
more calendar days. `labels.eligible()` gives the coins with a row on each day. The books
trade only these coins.

## Find factors

### libfactors.py: the factor library on Factorlab's data

```bash
python libfactors.py --compute                      # OKX panels: classify, compute, PIT proof, values
python libfactors.py --screen                       # the G-zone screen
python libfactors.py --status                       # read-only summary
python libfactors.py --source binance --compute     # Binance panel (needs the OKX screen first)
python libfactors.py --source binance --screen      # the new family and the cross-check
python libfactors.py --source binance --status
```

| Option | Effect |
|---|---|
| `--compute` | Classify all 28,904 definitions, compute each computable one, prove each one point in time and store the weekly values. |
| `--screen` | Screen the computed factors on the generator zone G. Needs `--compute` first. |
| `--status` | Print a read-only summary: the counts and the state of the code and the inputs. |
| `--source okx\|binance` | The panel source (default `okx`). |
| `--workers N` | Worker processes (at most 8, the default). |

Outputs: `data/library_on_engine/` (OKX) and `data/library_on_engine/binance/`, each with
`status.parquet`, `screen_g.parquet`, `manifest.json` and the weekly `values/`. The Binance
folder also has `crosscheck.parquet`. The screens give numbers only. `--screen` refuses
values whose inputs or code changed. Then run `--compute` again. See
[README_FINDER.md](README_FINDER.md).

### finder.py: environment check and generator probe

```bash
python finder.py --check                              # read-only checks; exit 1 on a FAIL
python finder.py --generator-probe --n 2000 --seed 1  # label-free, read-only
```

| Option | Effect |
|---|---|
| `--check` | Check `factorfinder/eval_config.toml`, the data-through date, the absence of network imports and the probe inputs. |
| `--generator-probe` | Draw random formulas and give the bands of their largest absolute correlation with the members of the novelty index. |
| `--n N` | Formulas to draw (default 2000). |
| `--seed S` | Random seed (default 1). |

Exit codes: 0 ok, 1 a `--check` FAIL, 2 an invalid option value. The probe needs the OKX
library values (`libfactors.py --compute`) and the label table.

### factor_lab.py: daily factor search with a registry

```bash
python factor_lab.py              # score the grid and log each trial
python factor_lab.py --grid 36    # all 36 specifications of the fixed grid (default 24)
python factor_lab.py --report     # read the registry; run nothing
python factor_lab.py --no-write   # score; write nothing
```

The program scores a fixed grid of network-diffusion specifications (`diffusion.py`). A
residual return shock of one coin spreads to its neighbors on the 90-day correlation
graph. The factor is the shock energy that flows toward each coin.

The registry `data/research/factor_registry.parquet` is append-only. It holds each trial.
The program counts each distinct specification once. It gives numbers only:

| Number | Definition |
|---|---|
| `ic`, `t`, `p` | the weekly rank IC with `fwd_ret_7`: mean, t and two-sided p |
| `q_bh` | the Benjamini-Hochberg q-value over the latest p of each distinct specification of the registry |
| `p_bonferroni` | p times the number of distinct specifications (at most 1) |
| `persistence`, `turnover`, `coverage` | the week-over-week rank correlation, the rank change, the share of the coins with a value |
| `max_incumbent_corr`, `n_incumbents` | the signed correlation with the nearest incumbent, and the number of incumbents |
| `max_library_corr` | the largest absolute correlation with a library definition (a separate number) |

The incumbents are your registered tests that are not withdrawn, the active AlphaForge
formulas and three risk proxies (`factorfinder/incumbents.py`). With no incumbent, the
report says so. The program runs once for each UTC day in `update.py`.

### alphaforge_bridge.py: the AlphaForge miner

```bash
python alphaforge_bridge.py --report                          # print the last report
python alphaforge_bridge.py --cycle                           # score, apply the window rule, mine when due, write
python alphaforge_bridge.py --cycle --today 2026-10-04 --no-write   # a dry run
python alphaforge_bridge.py --mine                            # a cycle that mines now when a place is free
python alphaforge_bridge.py --refresh                         # build the table again from the zoo
```

While `config.ALPHAFORGE_ENABLED` is `False` (the default), `--cycle`, `--mine` and
`--refresh` exit 3 unless you give `--force`. `--report` works in both states. Each formula
has a forward window of 52 weeks, and at most 6 formulas are active. See
[README_ALPHAFORGE.md](README_ALPHAFORGE.md) for all options.

## Test a factor

### judge.py: test your own factor in a book

```bash
python judge.py check --formula="-ts_mean(ex_ret, 20)"         # the historical test; writes nothing
python judge.py check --library CF00001 --spec long_only
python judge.py check --values factor_values.csv --json
python judge.py register --formula="-ts_mean(ex_ret, 20)" --name reversal --note "text"
python judge.py log-values FT-0002 --values factor_values.csv  # the new Sundays of a values file
python judge.py log-week                                     # the formula values and the new weeks
python judge.py status [FT-0001] [--json]
python judge.py verify [--write-results]
python judge.py withdraw FT-0001 --reason "text"
python judge.py record-code-change --reason "text" [--note "text" ...]
python judge.py spec long_short                              # check a book specification
```

The formula in these commands is an example of the syntax. The input options of `check`
and `register` are `--formula=EXPR`, `--library ID` or `--values FILE`, with `--source
okx|binance` for a formula and `--spec FILE|long_short|long_only` for the book.

| Command | Effect |
|---|---|
| `check` | The historical test of the factor in the book (`--spec`, default `bookspecs/long_short.toml`), the numbers of the factor and the related registered tests. `--novelty` also gives the largest correlation with a library definition. `--json` prints JSON. Writes nothing. |
| `register` | Append a registration line to `data/trials/registrations.jsonl` and write `data/trials/results/<id>/`. Refuses a formula that is not proven point in time, a history without a complete week, a duplicate and unrecorded code changes. `--name` and `--note` label the test. `--dry-run` writes nothing. |
| `log-values` | Log the new Sundays of a values file. A value counts from the first daily close after its log time. A changed value of a logged cell is refused and gives an alarm. |
| `log-week` | Log the value of each new Sunday of each formula test. Factorlab computes this value on the data up to that Sunday. Freeze each new complete forward week of each test. `--through` stops at a date. `--accept-release` records a new `VERSION` as a code change. Exit 3: a new alarm. |
| `status` | Show each test: its historical result, forward week, bounds, state, verdict and related tests. Then show the totals, the expected number of false "useful" verdicts and the rule. Read-only. |
| `verify` | Check each line of the ledger and compute the logged values, the history and the weeks again. Exit 1: a problem. Exit 3: a new alarm. `--write-results` writes the result files again. |
| `withdraw` | Withdraw a test before its first forward Sunday and before any logged value. |
| `record-code-change` | Record new hashes of the test code (`trials.CODE_FILES`) and of the test values of `config.py` (`trials.JUDGE_CONFIG`). |
| `spec` | Check a book specification and print its normalized content, sha256 and book key. |

Exit status: 0 done; 1 a refusal or a broken ledger; 2 a usage error; 3 a new alarm. The
program uses no network and writes only `data/trials/`. The rules of the test are in
[../README.md](../README.md#test-a-factor).

### The book specification

A book specification is a TOML file, format 1. `bookspecs/long_short.toml` is the default,
and `bookspecs/long_only.toml` is the long-only file:

```toml
# Factorlab book specification (format 1). engine/judge.py reads this file.
format = 1
name = "long-short"
version = 1

[book]
direction = "long_short"      # "long_short" or "long_only"
sign = 1                      # 1: a high value is long; -1: a low value is long
top_fraction = 0.2            # the long side: this fraction of the names with a value
bottom_fraction = 0.2         # the short side (long_short only)
exit_fraction = 0.4           # a name in the book stays while it is in this top (or bottom) fraction
rebalance = "weekly-monday"   # each Monday 00:00 UTC, on the values of the Sunday before

[costs]
scenario = "taker"            # "taker" (5 + 5 bps), "maker" (2 + 2 bps) or "custom"

[control]
kind = "zero"                 # "zero" (no position) or "equal_weight" (the eligible coins of the 20)

[forward]
sigma_floor_fallback = 0.03   # weekly-return units; used only when the history has fewer than 52 weeks with a factor position
```

Each key except `format`, `name` and `version` can be absent: it then gets its default.
An unknown key or table is refused, and `true` or `false` is never a number. The permitted
values:

- `top_fraction` and `bottom_fraction`: more than 0, not more than 0.5. A `long_only` file
  with `bottom_fraction` is refused.
- `exit_fraction`: for `long_short`, from the larger of the two fractions to 0.5; for
  `long_only`, from `top_fraction` to 1.0.
- `fee_bps` and `slippage_bps`: 0 to 100 for `custom`. For `taker` and `maker`, a key that
  is present must have the value of the scenario.
- `sigma_floor_fallback`: 0.001 to 1.0.

The identity of a specification is the sha256 of the canonical JSON of its normalized
content. Comments and the layout do not change it. The book key is the same hash without
`name`, `version`, `book.sign`, `costs.scenario` and the table `[forward]`. The duplicate
rule uses the book key. `python judge.py spec FILE` shows both hashes.

These rules are fixed in `book.py`. They are not in the file:

- A book needs at least 5 eligible coins with a value (`MIN_NAMES`). Else it has the
  control portfolio.
- Each side has at least 2 coins (`MIN_PER_SIDE`).
- The gross exposure is 1.0, and the start equity is 10,000.
- The accounting is `reconciled-v2`.
- The prices and the funding are the OKX perpetual closes and funding.

### The modules of the test

| Module | Function |
|---|---|
| `userfactor.py` | The formula check: the windows, the lags and the fields of each source. The values at each Sunday cut and their point-in-time proof. The values-file reader, the data gaps (the 7-day rule), the risk proxies and the factor numbers. |
| `book.py` | The book specification (`bookspecs/*.toml`), the banded book, the book and control runs, the weekly series |
| `rules.py` | The historical test, the confidence sequence, the verdict, the expected number of false "useful" verdicts and the rule text (`RULE_TEXT`) |
| `trials.py` | The trial ledger: check, register, the logs, the forward replay, the alarms, the code changes, status and verify |
| `ledger.py` | The append-only, hash-chained JSONL record and its write boundary |
| `labels.py` | The label table and the eligible coins of each day |
| `backtest.py` | The backtest engine of each book (below) |

### backtest.py: the engine

`backtest.run_backtest` runs each book of a test. It has no command line.

- The engine rebalances on Mondays at 00:00 UTC. It reads the factor values of the Sunday
  bar, which closed at that time. The new weights first earn the return of the Monday bar.
- Fees and slippage are on the dollars traded. Funding accrues each day on the notional.
  A positive funding rate means that longs pay shorts.
- Weights drift with prices between rebalances.
- Each weight is a fraction of equity, so the result does not change with the book size.
- The accounting is `reconciled-v2`: fees leave equity first, and each traded leg holds
  `target_w x post-fee equity`. The trade log, fees, holdings and equity then agree exactly.
- BTC hold (price only) is a context column. No rule reads it.

The weekly return of a signal Sunday S is E(S + 7) / E(S) - 1, with E the daily equity
mark. The weekly net excess return is the weekly return of the book minus the weekly return
of the control book (0 for the control `zero`).

## The trial ledger

`data/trials/` holds the trial ledger of your tests. Only `judge.py` writes it. Each chain
file is append-only JSONL:

| Chain | Line kinds |
|---|---|
| `registrations.jsonl` | `registration` (the factor, the book specification, the historical test, the sigma floor, the first forward Sunday, the code hashes), `withdrawal`, `code_change` |
| `values/okx.jsonl`, `values/binance.jsonl` | `week_values`: the values of the formula tests at one Sunday cut, with the hashes of the cut panels and the point-in-time check |
| `values/file.jsonl` | `file_values`: the values of one `log-values` call, the log time and the first daily close from which each value counts |
| `weeks.jsonl` | `book_week`: one frozen forward week of one test: the book, control, excess and BTC returns, and the bounds and the state |
| `alarms.jsonl` | `alarm`: one line for each alarm; an alarm is recorded once |

`head.json` holds the line count and the last sha256 of each chain. `results/<id>/` holds
the result files of a test: `historical.json`, `historical_daily.parquet`,
`historical_weekly.parquet` and `forward_weekly.parquet`. They are derived from the ledger
and the data. `verify --write-results` writes them again.

Each line is canonical JSON with `seq` and `prev_sha256`, the sha256 of the line before it.
An edited, removed or moved line stops each command with exit 1. A line after the head
(a run that stopped between its two writes) is accepted. Each write goes to a temporary file
first, then a rename. The ledger holds no path: a values file is recorded by its base name
and its sha256.

Alarms:

| Alarm | Description |
|---|---|
| `data_gap` | A bar or a value is missing for more than 7 days. The week counts. |
| `pit_mismatch` | The value of a formula at a Sunday cut differs from the full computation. That test is held at that Sunday. |
| `data_shorter` | The data on disk ends before a logged Sunday. |
| `inputs_changed` | A hashed field of a logged Sunday changed. The alarm names the fields. |
| `value_changed` | A logged formula value differs from a new computation. |
| `book_week_changed` | A replay of a frozen week gives another book or control return. The frozen numbers stay. |
| `history_changed` | (`verify` only) The history values of a test differ from its registration. |
| `file_value_changed` | A `log-values` call gave another value for a logged cell. The call is refused. |

**Code changes.** Each registration and `code_change` line holds the hashes of
`trials.CODE_FILES`. These are the test modules, `backtest.py`, `libfactors.py`,
`data_io.py` and four library files. The line also holds the values of `config.py` that the
tests read (`trials.JUDGE_CONFIG`: `EXCHANGE_ID`, `FUNDING_FALLBACK_EXCHANGE_ID`,
`UNIVERSE`). `register` and `log-week` refuse code that differs from the latest of these
lines. `log-week --accept-release` records a new
`VERSION` by itself. An edit of a switch in `config.py` (for example `ALPHAFORGE_ENABLED`)
is not a code change.

## Dashboard and tests

| Command | Function |
|---|---|
| `update.py` | Download the new data and build every output (run it from the repository root). `--offline` builds from the data on disk, `--demo` from synthetic demo data, `--plan` shows the steps, `--verbose` shows the output of each step. A step stops after 12 hours. See [README_AUTOMATION.md](README_AUTOMATION.md). |
| `demo_data.py` | The synthetic demo data set: `--write`, `--check` (writes nothing), `--remove` (only demo folders), `--no-panel`, `--seed N`. See [../README.md](../README.md#try-factorlab-with-demo-data). |
| `version.py` | Print the version (the file `VERSION`). |
| `export_web.py` | Build the dashboard data `web/data/*.js` and the part files `web/data/<section>.<part>.js`. `--only library,system` builds some sections, `--out DIR` writes to another folder. See [../web/README.md](../web/README.md). |
| `run_tests.py` | Run the test suites: one PASS, SKIP or FAIL line for each suite. `--quick`, `--only NAME`, `--list`, `--verbose`. |

`run_tests.py` runs each suite below in its own process. Run one suite with
`run_tests.py --only NAME`.

| Suite | What it checks |
|---|---|
| `test_labels.py` | the label table: alignment, no invented label, a new coin, row shifts, the eligible coins, a pinned digest on exact prices |
| `test_backtest.py` | the backtester on synthetic data: no lookahead, funding sign, fees charged once, flat without positions, scale invariance, drift turnover |
| `test_accounting.py` | the accounting: the trade log rebuilds the equity |
| `test_book.py` | the book specification files and their hashes, the banded book, the fractions and the sign, the control books, the weekly series, the schedule of the engine |
| `test_rules.py` | the bounds against plain loops, the states, the false "useful" rate under a zero mean, the power, the historical test, the sigma floor, the verdict, the rule text; BTC hold enters no rule |
| `test_userfactor.py` | the formula check, the values at each Sunday cut, the values file, the factor numbers and the risk proxies |
| `test_trials.py` | the trial ledger and `judge.py`: the hash chain, the duplicates, the complete-week rule, the 7-day rule, the log-time rule, the alarms, the code changes, the withdrawal rules, an end-to-end run of 30 forward weeks |
| `test_engine_rules.py` | the funding splice; the `FACTORLAB_OFFLINE` parser; write protection; the refusal of a protected folder by each writer; the `--help` text of each program (it runs nothing); `db.py`; the import check (each import names a module that exists) |
| `test_diffusion.py` | point in time of the diffusion factors and the shrinkage correlation |
| `test_alphaforge_bridge.py` | the AlphaForge switch, the zoo format, the forward window, the capacity, point in time, numbers only |
| `test_libfactors.py`, `test_libfactors_binance.py` | the library on OKX and Binance data: point in time, the screen, the cleaning freeze, numbers only |
| `test_finder_*.py` | the generator parts: field map, operators, novelty index, probe, and that Finder writes nothing |
| `test_incumbents.py` | the incumbents of the factor lab and AlphaForge, and the number for the library |
| `test_update.py` | the step plan of `update.py`, the step timeout, the options of `--help`, the hashed requirement files, `setup.sh` and the workflow install, and the library package version |
| `test_demo_data.py` | the demo data set: the same seed gives the same files, the engine and the library read it, the refusals, `--remove`, the plan of `update.py --demo`, and the demo banner on each page |
| `test_version.py` | the file `VERSION` (Semantic Versioning), `version.py`, the same version in the library, in `CITATION.cff`, in the README version badge, in the top section of `CHANGELOG.md` and in each fixed example; the form of `CHANGELOG.md`, the README badges, the relative links of the root documents, the MIT License files and the issue forms; the e-mail addresses in the files of the repository (only GitHub no-reply addresses and example domains); the private reporting form in `SECURITY.md` and `CODE_OF_CONDUCT.md` |
| `test_export_*.py` | each dashboard section: every displayed number computed again from the source files; `test_export_web.py` also runs the app shell in node |

The runner also runs the library suite (`library/tests`) and the AlphaForge suite
(`alphaforge/tests`). It prints PASS, SKIP or FAIL for each suite (see
[../README.md](../README.md#tests)). A suite that runs no test, for example because its
data is not on disk, shows SKIP.

## Modules without a command line

| Module | Function |
|---|---|
| `config.py` | The universe, the venue, paths, write protection, the network switch and the AlphaForge switch (see [Configuration](#configuration)) |
| `data_io.py` | The shared reader of the raw Parquet files: bars, spot bars, the daily funding splice, open interest, the close and funding matrices and the label table |
| `backtest.py` | The backtest engine (see [backtest.py: the engine](#backtestpy-the-engine)) |
| `book.py`, `rules.py`, `userfactor.py`, `trials.py`, `ledger.py` | The test of a factor (see [The modules of the test](#the-modules-of-the-test)) |
| `diffusion.py` | The network-diffusion specifications of the factor lab |
| `factorfinder/` | The label-free generator parts (`config_io`, `fieldmap`, `libops`, `novelty`, `probe`) and the incumbent provider (`incumbents`) |
| `webexport/` | One builder for each dashboard section |
| `alphaforge/` | The AlphaForge expression miner (part of Factorlab) and its test suite |

## The data folder

```
data/
  raw/                 downloaded bars, funding and open interest (ingest.py)
  labels/              labels.parquet: fwd_ret_1 and fwd_ret_7 (labels.py)
  library_on_engine/   the factor library on OKX data, and binance/ on Binance data (libfactors.py)
  research/            the factor-lab registry and the AlphaForge files
  trials/              the trial ledger of your tests (judge.py)
  demo.json            only in a demo data folder: the demo marker (demo_data.py)
```

Factorlab reads no other folder of the data folder. `setup.sh` builds all of it except
`trials/`. You can delete `raw/`, `labels/`, `library_on_engine/` and `research/` and run
`./setup.sh` to build them again from the public data. The trial ledger `trials/` holds your
registrations: keep a backup of it. A deleted ledger cannot be built again.

## Configuration

The values below are constants in the code, keys in `factorfinder/eval_config.toml` or
files in `bookspecs/`. The environment variables are in
[../README.md](../README.md#data-sources-and-product-rules). Some values are part of a
registered test: the coins, the venue, the label, the book rules and the decision rule. A
change of such a value makes the earlier results of the test not comparable. Change such a
value only on a copy.

| File | Name | Default | Effect |
|---|---|---|---|
| `config.py` | `EXCHANGE_ID` | `"okx"` | the ccxt exchange of the bars, the funding and the open interest (see [Change the venue or the universe](#change-the-venue-or-the-universe)) |
| `config.py` | `FUNDING_FALLBACK_EXCHANGE_ID` | `"binanceusdm"` | the exchange of the older funding history (data only) |
| `config.py` | `UNIVERSE` | 20 USDT perpetuals | the coins, in ccxt format |
| `config.py` | `BACKFILL_START` | `2022-01-01T00:00:00Z` | the first day of the first download |
| `config.py` | `PROTECTED_DIRS`, `PROTECT_MARKER` | `raw`, `labels`, `research`, `trials`; `.read_only` | the write protection of a data folder |
| `config.py` | `ALPHAFORGE_ENABLED` | `False` | the switch of the AlphaForge miner |
| `config.py` | `ALPHAFORGE` | a dict | the miner settings, the forward window (52 weeks) and the capacity (6) (see [README_ALPHAFORGE.md](README_ALPHAFORGE.md#settings)) |
| `bookspecs/` | `long_short.toml`, `long_only.toml` | the default book, the long-only book | the shipped book specifications (see [The book specification](#the-book-specification)) |
| `book.py` | `COST_SCENARIOS` | taker 5 + 5 bps, maker 2 + 2 bps | fee and slippage of each cost scenario |
| `book.py` | `MIN_NAMES`, `MIN_PER_SIDE`, `GROSS`, `INITIAL_EQUITY` | 5, 2, 1.0, 10000 | the fixed rules of each book |
| `rules.py` | `ALPHA`, `TUNE_WEEK`, `FIRST_DECISION_WEEK`, `MAX_WEEKS` | 0.10, 26, 26, 156 | the confidence sequence and the decision weeks of the forward test |
| `rules.py` | `LR_LAGS`, `MIN_FLOOR_WEEKS` | 4, 52 | the lags of the long-run SD; the smallest number of weeks with a factor position for a sigma floor from the history |
| `rules.py` | `WINDOW_DAYS`, `STEP_DAYS`, `MIN_SHARE_POSITIVE` | 365, 91, 0.5 | the rolling windows of the historical test |
| `userfactor.py` | `MAX_WINDOW`, `GRACE_DAYS` | 1000, 7 | the largest window or lag of a formula; the days after which a missing value is final |
| `userfactor.py` | `MAX_FILE_ROWS`, `MAX_FILE_BYTES` | 2,000,000, 100 MiB | the limits of a values file |
| `trials.py` | `RELATED_CORR`, `DUP_VALUE_CORR` | 0.85, 0.999 | the correlation of a related test and of a duplicate |
| `backtest.py` | `ACCOUNTING` | `reconciled-v2` | the accounting convention of each book |
| `libfactors.py` | `MAX_WORKERS` | 8 | the largest number of worker processes (`--workers`) |
| `update.py` | `STEP_TIMEOUT_S` | 43200 (12 hours) | the longest run of one build step |
| `demo_data.py` | `SEED`, `FIRST_DAY`, `N_DAYS` | 20220101, 2022-01-01, 1734 | the demo data set (`--seed` changes the seed) |
| `eval_config.toml` | `[label]` | `fwd_ret_7`, 7 days | the label of each weekly statistic of the screen |
| `eval_config.toml` | `[zones]` | G_fit, G_eval, I (see [README_FINDER.md](README_FINDER.md#the-zones)) | the Sundays of each zone; `libfactors.py` stops when its own constants differ |
| `eval_config.toml` | `[regressors]` | ridge 1e-3, tilt 180 days, beta 90 days, size 30 days | the rIC regressors |
| `eval_config.toml` | `[g0]` | coverage 0.80, 15 names, finite 0.95, 10 distinct values | the label-free limits of a formula on G |
| `eval_config.toml` | `[generator.terminals]`, `[generator.settings]` | 25 terminals; windows 3 to 90 days, depth 3, 20 nodes | the inputs and the limits of the generator probe |

`factorfinder/config_io.py` gives `eval_config.toml` two version hashes (see
[README_FINDER.md](README_FINDER.md#eval_configtoml)). The test modules, `rules.py` and
`book.py` among them, are code files of the trial ledger (`trials.CODE_FILES`). After a
registration, record each edit of them with
`python judge.py record-code-change --reason "text"`.

## Change the venue or the universe

- **Universe.** `config.UNIVERSE` lists the coins in ccxt format (`BTC/USDT:USDT`). A new
  coin changes the cross-sections of past Sundays. To add a coin, use a new data folder
  (delete `engine/data` or set `FACTORLAB_DATA`). Then run `./setup.sh --skip-binance`,
  because the Binance panel holds only the 20 coins of `library/scripts/merge_panels.py`.
  Do not delete a coin: that adds survivorship bias.
- **Venue.** `config.EXCHANGE_ID` is the ccxt exchange id (default `okx`). A change gives a
  new data set. Delete `engine/data` first, and change the symbols to the format of the new
  venue. The test results of the old venue do not apply to the new data.
- **Trial ledger.** The trial ledger records the coins and the exchange names
  (`trials.JUDGE_CONFIG`). After a change of them, `register` and `log-week` refuse until
  `python judge.py record-code-change --reason "text"` records it. The tests of the old
  universe are not comparable with the new data. Thus use a new data folder, with a new
  trial ledger, for a new universe or venue.
