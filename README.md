# Factorlab

[![Test workflow status](https://github.com/Radioiic/Factorlab/actions/workflows/tests.yml/badge.svg)](https://github.com/Radioiic/Factorlab/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python 3.12 and 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](#requirements)
[![Version 0.2.2](https://img.shields.io/badge/version-0.2.2-blue)](CHANGELOG.md)

Factorlab is a research tool for cross-sectional factors of 20 OKX USDT perpetual futures.
Four search tools give numbers. A test judges a factor that you give: a formula or a file of
values. The test runs a book with costs, first on the history and then on each new week
after the registration. Factorlab gives no factor of its own as useful or useless.

A factor is one number for each coin on each day, for example the 30-day volatility of the
coin. Each Sunday, a test book ranks the coins by the factor. It trades on the next Monday
and holds the positions for 7 days.

Factorlab places no orders and it holds no exchange keys. It reads only public market data.
Each return of a book in Factorlab is a test statistic about a factor. Nothing in Factorlab
is investment advice (see [Disclaimer](#disclaimer)).

The code of Factorlab is under the [MIT License](LICENSE). The market data keeps its own
terms (see [Data terms](#data-sources-and-product-rules)). The file `VERSION` holds the
version of Factorlab. [CHANGELOG.md](CHANGELOG.md) lists the changes of each version, and
[ROADMAP.md](ROADMAP.md) is the roadmap. The terms of this document are in
[engine/GLOSSARY.md](engine/GLOSSARY.md).

The 20 coins are in `engine/config.py` (`UNIVERSE`): BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK,
DOT, LTC, BCH, UNI, ATOM, ETC, FIL, NEAR, APT, OP, ARB, AAVE.

## Contents

- [How Factorlab works](#how-factorlab-works)
- [Requirements](#requirements)
- [Setup](#setup)
- [Try Factorlab with demo data](#try-factorlab-with-demo-data)
- [Update the data](#update-the-data)
- [Update Factorlab](#update-factorlab)
- [Remove Factorlab](#remove-factorlab)
- [Find a new factor](#find-a-new-factor)
- [Test a factor](#test-a-factor)
- [The dashboard](#the-dashboard)
- [Tests](#tests)
- [Data sources and product rules](#data-sources-and-product-rules)
- [Folder layout](#folder-layout)
- [Questions and answers](#questions-and-answers)
- [Problems and solutions](#problems-and-solutions)
- [More documentation](#more-documentation)
- [License](#license)
- [Disclaimer](#disclaimer)

## How Factorlab works

`engine/update.py` runs the data steps, the search tools (not the generator probe), the
weekly log and the dashboard. You run `judge.py check` and `judge.py register` for your own
factor. The name on each arrow is the program that makes the next box.

```
  OKX public API                          Binance public API and data.binance.vision
        |                                              |
        | ingest.py                                    | factorlab download, merge_panels.py
        v                                              v
  engine/data/raw/                              library/data/binance_um_1d/
  perp bars, spot bars, funding, OI             the Binance panel (16 fields)
        |                                              |
        | labels.py                                    |
        v                                              |
  engine/data/labels/labels.parquet                    |
  fwd_ret_1, fwd_ret_7                                 |
        |                                              |
        v                                              v
  search tools (numbers only): libfactors.py, finder.py, factor_lab.py, alphaforge_bridge.py
        |
        v
  your factor: a formula, a library definition or a values file
        |
        | judge.py check
        v
  the historical test: a book with costs on the history (writes nothing)
        |
        | judge.py register
        v
  engine/data/trials/: the trial ledger (append-only, hash-chained)
        |
        | judge.py log-week (each update.py run)
        v
  the weekly forward test: one frozen week after each Sunday
        |
        v
  the verdict: useful, useless or unproven
        |
        | export_web.py
        v
  web/data/*.js  -->  web/index.html (the dashboard)
```

**Market data.** `ingest.py` downloads the daily bars, the funding settlements and the open
interest of the 20 OKX perpetual futures. The Binance panel holds Binance's public data for
the same 20 coins. Binance is a data source only. The Binance panel is optional
(`--skip-binance`).

**Labels.** `labels.py` writes the label table. `fwd_ret_7` is the return of a coin over
the 7 days after a Sunday. A label is never an input of a factor.

**Search tools.** Four tools give numbers about formulas: the factor library screen, the
label-free generator probe, the factor lab and the optional AlphaForge miner. No search
tool gives a verdict (see [Find a new factor](#find-a-new-factor)).

**Your factor.** You give a formula of the library expression language, the id of a library
definition, or a file of values. A formula from a search tool, or from your own work, is the
same input.

**Check.** `judge.py check` runs the historical test. A book of the factor trades each week
on the history, with fees, slippage and funding. The historical test can give only
"useless" or "unproven (historical pass)". It writes nothing.

**Register.** `judge.py register` adds the test to the trial ledger
`engine/data/trials/`. The ledger is append-only and hash-chained. After the registration,
only new weeks count.

**Weekly forward test.** Each update logs the new complete weeks of each registered test.
A confidence sequence on the weekly net excess return over the control book decides.

**Verdict.** The forward test gives "useful" or "useless" from week 26. Before a forward
decision, the historical result gives the verdict word. The dashboard shows each test and
its verdict.

## Requirements

| Item | Requirement |
|---|---|
| System | macOS 12 or later (Apple silicon or Intel), or Linux x86_64 with glibc 2.28 or later |
| Windows | Use WSL2 with a Linux distribution, and follow the Linux steps. The test workflow does not run on Windows. |
| Python | 3.12 or 3.13 |
| Tools | `git`, and a shell with `sh`. On Debian and Ubuntu (also in WSL2), also the package `python3.X-venv` of your Python (for example `python3.12-venv`). |
| Intel Mac only | PyPI has no wheel of `cryptography` 50.0.2 for macOS x86_64, so pip builds it from source. Install Rust and OpenSSL 3 first: `brew install rust openssl@3`. |
| Network | necessary for the first setup and for each data update. Hosts: `pypi.org` and `files.pythonhosted.org` (the Python packages), `www.okx.com` (OKX), `fapi.binance.com` and `data.binance.vision` (Binance). `--demo` and `--offline` download no market data. |
| Disk space | about 3 GB free (data about 1.6 GB, Python environment about 0.7 GB) |

Approximate time of the first setup:

| Step | Time |
|---|---|
| Python packages | 1 to 3 minutes |
| OKX download (daily data since 2022-01-01) | about 3 minutes |
| Binance download (8 coins at the same time) | about 25 minutes |
| Build of every output | about 5 minutes |
| Total | about 35 minutes |

A slower machine or a slow network can take more than one hour. A run that only adds
new days downloads a few days. Then it builds every output again: about 5 minutes,
or about 1 minute with `--skip-library`. A demo build (`--demo`) takes
about 5 minutes and downloads no market data.

The pinned packages resolve for Python 3.12 and 3.13, and `setup.sh` accepts only these
two versions. For a newer Python, some pinned packages can have no wheel on PyPI, and pip
must then build them from source.

**Regions.** Binance refuses requests from some regions (for example, the United States)
with HTTP 451. In these regions, run `./setup.sh --skip-binance`. Each data source has its
own rules for regions (see [Data terms](#data-sources-and-product-rules)). When `ingest.py`
cannot reach Binance, it continues without the Binance funding history. That history only
extends the funding data before the first OKX settlement. Two functions need the Binance
panel: the Binance screen of the factor library, and the test of a formula on the Binance
data. Without the panel, Factorlab has neither of them. The OKX screen, the other search
tools and the tests of OKX formulas and values files work without it.

## Setup

Run these three commands:

```bash
git clone https://github.com/Radioiic/Factorlab.git
cd Factorlab
./setup.sh
```

`setup.sh` does these steps:

1. It finds Python 3.12 or 3.13 (`PYTHON`, else `python3.13`, `python3.12`, `python3`).
2. It makes the virtual environment `engine/.venv` (only when it does not exist).
3. It installs the pinned packages of `engine/requirements.txt` with `--require-hashes`.
   Each package has one version and the sha256 hash of each of its files, so pip refuses
   a file with another hash. With `--test`, it installs `engine/requirements-dev.txt`,
   which adds the test packages (`pytest`, `matplotlib`, `hypothesis`).
4. It installs the factor library package (`library/`) in editable mode. This adds the
   command `engine/.venv/bin/factorlab`.
5. It runs `engine/update.py`, which downloads the data and builds every output.
6. It runs the short test set (`engine/run_tests.py --quick`).
7. It opens the dashboard `web/index.html` in your browser.

Do not run `setup.sh` as root. It refuses.

### setup.sh options

| Option | Effect |
|---|---|
| `--offline` | Build from the market data on disk. Download no market data. The first setup still installs the Python packages from the network. |
| `--demo` | Build from synthetic demo data. Download no market data (see [Try Factorlab with demo data](#try-factorlab-with-demo-data)). |
| `--skip-binance` | Do not download or use the Binance panel. |
| `--skip-library` | Do not compute the factor library on the data. |
| `--jobs N` | Run N Binance downloads at the same time (default 8). |
| `--from STEP` | Start the build at this step: a name or a number from `engine/update.py --plan`. |
| `--test` | Install the test packages too, and run the full test suite after the build in place of the short test set. |
| `--no-open` | Do not open the dashboard at the end. |
| `-h`, `--help` | Show the options. |

To use a specific Python, set `PYTHON`:

```bash
PYTHON=/usr/local/bin/python3.13 ./setup.sh
```

## Try Factorlab with demo data

```bash
git clone https://github.com/Radioiic/Factorlab.git
cd Factorlab
./setup.sh --demo
```

`--demo` writes a synthetic data set and builds every output from it. It downloads no
market data. The Python packages still install from the network when they are absent.

- **The data set.** The 20 coins of `config.UNIVERSE`, from 2022-01-01 to 2026-09-30
  (1,734 daily bars), in the formats of the engine. It has perpetual and spot bars,
  funding prints, open interest and a Binance panel with 16 fields. A random generator makes
  each value (`engine/demo_data.py`, seed 20220101). The same seed gives the same files.
- **Two planted signals.** The generator adds two signals to the returns (`SIGNALS` in
  `engine/demo_data.py`). Thus the search tools and the tests have a known signal to find.
  A result on the demo data is not evidence about a real factor.
- **No registered factor.** The demo build registers no factor. The page My factors shows
  no test.
- **The dashboard** shows "Synthetic demo data. Not market data." on each page. The header
  shows the same text in place of the data age.

The demo writes only into an empty data folder or into a demo data folder (the marker file
`demo.json`). It writes its Binance panel only into an absent, empty or demo
`library/data/binance_um_1d/`. It refuses a folder with other files, because such a folder
can hold market data. To keep the demo apart from market data, set
`FACTORLAB_DATA=<folder>`. Use `--skip-binance` when the Binance panel holds market data.

A build without `--demo` refuses a demo data folder and a demo Binance panel. To change
from the demo data to market data:

```bash
engine/.venv/bin/python -B engine/demo_data.py --remove   # removes only the demo data set
./setup.sh
```

| `demo_data.py` option | Effect |
|---|---|
| `--write` | Write the demo data set into the data folder. |
| `--check` | Tell if the demo can write here. Write nothing. |
| `--remove` | Remove the demo data set. Only folders with the demo marker. |
| `--no-panel` | Make no Binance panel (`update.py --demo --skip-binance` uses it). |
| `--seed N` | The generator seed (default 20220101). |

## Update the data

Run `./setup.sh` again. Each run adds the new closed days and builds the outputs again.

You can also run the build command directly, from the repository root:

```bash
engine/.venv/bin/python -B engine/update.py              # download, then build
engine/.venv/bin/python -B engine/update.py --plan       # show the steps; run nothing
engine/.venv/bin/python -B engine/update.py --offline    # build from the data on disk
engine/.venv/bin/python -B engine/update.py --demo       # build from synthetic demo data
engine/.venv/bin/python -B engine/update.py --from "dashboard"   # start at one step
engine/.venv/bin/python -B engine/update.py --verbose    # show the output of each step
```

`update.py` has the options of `setup.sh` without `--test` and `--no-open`, plus `--plan`
and `--verbose`. The steps, in order:

| Step | Program | When |
|---|---|---|
| demo data | `demo_data.py --write` | only with `--demo`; then no download step |
| OKX download | `ingest.py` | not with `--offline` or `--demo` |
| Binance download, then Binance merge or Binance append | `factorlab download`, `library/scripts/merge_panels.py` | not with `--skip-binance`, `--offline` or `--demo`. The merge makes the first panel; the append adds new days. |
| labels | `labels.py` | always |
| factor tests, weekly log | `judge.py log-week --accept-release` | only when the trial ledger `engine/data/trials/` has a registration |
| library catalog | `python -m factorlab build` | only when `library/catalog/factors.db` is absent; not with `--skip-library` |
| library on OKX data, compute; library on OKX data, screen | `libfactors.py --compute`, `libfactors.py --screen` | not with `--skip-library` |
| library on Binance data, compute; library on Binance data, screen | `libfactors.py --source binance --compute`, `--screen` | a Binance panel exists, or this run makes it; not with `--skip-library` or `--skip-binance` |
| factor lab | `factor_lab.py` | once for each UTC day |
| AlphaForge cycle | `alphaforge_bridge.py --cycle` | only when `ALPHAFORGE_ENABLED = True` in `engine/config.py` |
| dashboard | `export_web.py` | always |

The step "factor tests, weekly log" logs the new forward weeks of your tests. Its exit
status 3 (a new alarm) is not a failure: the log line of the step says "done (new alarm)".

The data folders that Factorlab uses (`engine/data/`, or `FACTORLAB_DATA`):

| Folder or file | Writer | Content |
|---|---|---|
| `raw/` | `ingest.py`, `demo_data.py` | the downloaded bars, funding settlements and open interest |
| `labels/labels.parquet` | `labels.py` | the label table: `fwd_ret_1` and `fwd_ret_7` of each coin and day |
| `library_on_engine/` | `libfactors.py` | the factor library on the OKX data, and in `binance/` on the Binance data |
| `research/` | `factor_lab.py`, `alphaforge_bridge.py` | the factor-lab registry and the AlphaForge files |
| `trials/` | `judge.py` | the trial ledger of your tests and its result files |
| `demo.json` | `demo_data.py` | only in a demo data folder: the demo marker |

Factorlab reads no other folder of the data folder.

`update.py` takes the lock `engine/logs/update.lock` and writes to
`engine/logs/update.log`. Each line of the log has the form `<UTC time>  <text>`. The first
line of a run gives the version (`Factorlab 0.2.2`). When the run has a download step, the
console then shows a line about the data terms. Each step adds one line: step, result and
seconds. The result is `ok`, `done (new alarm)` or `failed (exit N)`. With `--verbose`, the
build shows each output line of each step, and the log has each line too. The System page
shows the last run from this log.

`update.py` stops at the first step that fails, and it shows the end of the output of that
step. For a build step, the message gives the step name for `--from`. For a download step,
run the same command again. A step that runs for more than 12 hours stops and fails
(exit 124).

The Binance panel ends on the last Sunday that closed at least one full day before the
run (UTC). Binance can publish a daily archive file some hours late. Thus the panel does
not ask for a Sunday that closed on the same day.

For a scheduled run, see [engine/README_AUTOMATION.md](engine/README_AUTOMATION.md).

## Update Factorlab

Before you update, read the section of the new version in [CHANGELOG.md](CHANGELOG.md) on
GitHub. Each release has a tag `vX.Y.Z`. To get a release, put its version in place of
`X.Y.Z`:

```bash
git fetch --tags
git checkout vX.Y.Z
./setup.sh
```

`git pull` on the branch `main` gets the newest code. Between two releases, this code can
change the test code with the same `VERSION` (see below). When you have registered tests,
update to a release.

`setup.sh` installs the pinned packages of the new version and builds every output again.
`cat VERSION` shows the version. The first line of `update.py`, the footer of the
dashboard and the System page show it too.

The trial ledger holds the hashes of the test code (`trials.CODE_FILES`). A new version of
Factorlab can change these files. `update.py` runs `judge.py log-week --accept-release`.
When the code and the `VERSION` changed, this command records the change in the trial
ledger. Then the weekly log continues.

A change of the test code without a new `VERSION` is not recorded automatically. Such a
change comes from a local edit or from a `git pull` between two releases. Then `log-week`
and `register` refuse, and `update.py` stops at the step "factor tests, weekly log". Record
the change. Then continue the build. Run these commands from the repository root:

```bash
cd engine && .venv/bin/python -B judge.py record-code-change --reason "TEXT"
cd .. && ./setup.sh --from "factor tests, weekly log"
```

## Remove Factorlab

Factorlab installs nothing outside its folder. The data, the Python environment, the logs
and the outputs are all in the repository folder. Only pip keeps a download cache in its
own cache folder.

1. If you made a scheduled run (see
   [engine/README_AUTOMATION.md](engine/README_AUTOMATION.md)), remove it first.
2. Optional: empty the download cache of pip:
   `engine/.venv/bin/python -m pip cache purge`.
3. Delete the `Factorlab` folder.

The dashboard keeps only its theme choice in the storage of your browser.

## Find a new factor

Factorlab has four search tools: the factor library screen, the label-free generator probe,
the factor lab and the AlphaForge miner. Each tool gives numbers only. To test a factor in a
book, see [Test a factor](#test-a-factor).

### 1. The factor library screen

The factor library (`library/`) holds 28,904 formula definitions. `setup.sh` builds the
catalog `library/catalog/factors.db`. `engine/libfactors.py` then computes each
definition on Factorlab's data and screens it:

```bash
cd engine
.venv/bin/python -B libfactors.py --compute                    # OKX data
.venv/bin/python -B libfactors.py --screen
.venv/bin/python -B libfactors.py --source binance --compute   # Binance data, same 20 coins
.venv/bin/python -B libfactors.py --source binance --screen
.venv/bin/python -B libfactors.py --status                     # read-only summary
```

The screen uses only the generator zone G (the Sundays 2022-04-03 to 2024-06-09). It
gives numbers: the rank IC, the residual IC (rIC), the Benjamini-Hochberg q-values and the
distinct signals. No rule of the screen gives a verdict. `--status` shows the counts and
tells you if the code or the inputs changed after the computation. When `--status` or the
library page shows no counts of a screen, run `--compute` and `--screen` again. Details:
[engine/README_FINDER.md](engine/README_FINDER.md) and [library/README.md](library/README.md).

Each library definition has a command that checks it and a command that registers it for a
test (see [Test a factor](#test-a-factor)). `factorlab show ID` prints the two commands of
a definition. Run these commands from the repository root:

```bash
cd engine
.venv/bin/python -B judge.py check --library CF00001
.venv/bin/python -B judge.py register --library CF00001
```

### 2. The label-free generator probe

The probe draws random formulas from 25 input series and the library operators. For each
formula it gives the largest absolute correlation with the members of a novelty index. The
members are the library definitions, your registered tests, the AlphaForge formulas and three
risk proxies. The report counts the formulas in each band: below 0.50, 0.50 to 0.85, 0.85 or
more. It reads no return label and writes nothing. Run these commands from the repository root:

```bash
cd engine
.venv/bin/python -B finder.py --generator-probe --n 2000 --seed 1
```

### 3. The factor lab

`factor_lab.py` scores a grid of network-diffusion factor specifications once each UTC day.
It adds each trial to an append-only registry (`data/research/factor_registry.parquet`). It
gives numbers only: IC, t, p, the Benjamini-Hochberg q-value over the distinct
specifications of the registry, p_bonferroni, persistence and coverage. For each
specification it also gives the signed correlation with the nearest incumbent: the incumbent
with the largest absolute correlation. The incumbents are your registered tests, the active
AlphaForge formulas and three risk proxies. The largest absolute correlation with the library
definitions is a separate number.

```bash
cd engine
.venv/bin/python -B factor_lab.py            # score the grid, log every trial
.venv/bin/python -B factor_lab.py --report   # read the registry only
```

### 4. The AlphaForge miner (optional)

AlphaForge is a genetic-programming expression miner. The author of Factorlab wrote
AlphaForge, and the MIT License ([LICENSE](LICENSE)) covers it. It keeps a zoo of mined
formulas. Each formula has a forward window of 52 weeks. After 52 forward weeks it leaves
the active list, and its record stays. At most 6 formulas are active, and a mining run fills
only the free places. The miner compares a new formula with the same incumbents as the
factor lab. It gives numbers only. It is off by default. It is not related to the
paper "AlphaForge: A Framework to Mine and Dynamically Combine Formulaic
Alpha Factors" ([arXiv:2406.18394](https://arxiv.org/abs/2406.18394)) or to
its code. To use it, set `ALPHAFORGE_ENABLED = True` in `engine/config.py`. Its files are
in `data/research/`. Details: [engine/README_ALPHAFORGE.md](engine/README_ALPHAFORGE.md).

## Test a factor

`engine/judge.py` tests a factor that you give. The test runs a book: a simulated portfolio
that pays fees, slippage and funding. A long-short book is long the coins that the factor
ranks high and short the coins that it ranks low. A long-only book has only the long
side. Each book rebalances weekly, on Monday 00:00 UTC, with the factor values of the Sunday
before.

First the test runs the book on the history: the historical test. After the registration,
it runs the book on each new complete week: the forward test. Run the commands from the
repository root:

```bash
cd engine
.venv/bin/python -B judge.py check --formula="-ts_mean(ex_ret, 20)"
.venv/bin/python -B judge.py register --formula="-ts_mean(ex_ret, 20)" --name reversal
.venv/bin/python -B judge.py status
```

The formula in these commands is an example of the syntax. `check` writes nothing.
`register` writes the trial ledger `engine/data/trials/`. Each update (`update.py`) then
logs the new weeks (the step "factor tests, weekly log").

### 1. Give the factor

Give one of these three inputs:

| Option | Input |
|---|---|
| `--formula=EXPR` | A formula of the library expression language (`library/factorlab/expr.py`), for example `-ts_mean(ex_ret, 20)`. |
| `--library ID` | A definition of the factor library, for example `CF00001`. |
| `--values FILE` | A values file: a CSV file or a Parquet file with exactly the columns `date`, `symbol` and `value`. |

Write the formula after `--formula=`. Then a formula that starts with `-` is not read as
an option. Put the formula in quotes when it has a space or a parenthesis. The operators
and fields of a formula are in [library/README.md](library/README.md#expressions).

The rules for a formula:

- Each window and each lag is a constant whole number from 1 to 1000. Thus a formula
  cannot read a later day: for example, `delay(close, -1)` is refused.
- The data of the source has each field that the formula reads. Use `--source okx` or
  `--source binance`. Without `--source`, Factorlab uses OKX when the OKX data has each
  field. On the OKX data, a formula that reads open interest is refused: the time of the
  stored OKX open-interest value is not proven.
- Factorlab computes the value of each Sunday on the data up to that Sunday. This value
  must be equal to the value of the full computation. Then the formula is proven point in
  time. The values of each Sunday cut are the values of the test.

The rules for a values file:

- The first line of a CSV file is `date,symbol,value`, in this order. A Parquet file can
  have the three columns in any order.
- `date`: a UTC day, for example `2026-03-01` or `2026-03-01T00:00:00Z`. Factorlab uses
  only the Sundays. It counts the other rows and does not use them.
- `symbol`: one of the 20 coins, for example `BTC-USDT`, `BTCUSDT` or `BTC/USDT:USDT`.
- `value`: a finite number. An empty value is missing.
- The file has at least one data row.
- Factorlab cannot prove a values file point in time. A forward value counts only from the
  first daily close (00:00 UTC) after its log time.

### 2. The book specification

A book specification is a TOML file (`--spec FILE`). `engine/bookspecs/` has two files:
`long_short.toml` (the default) and `long_only.toml` (`--spec long_only`).

| Key | Default | Rule |
|---|---|---|
| `format`, `name`, `version` | necessary | `format = 1`; a name of 1 to 64 letters, digits, `_`, `.` or `-`; a version from 1 to 1000000 |
| `book.direction` | `long_short` | `long_short` or `long_only` |
| `book.sign` | `1` | `1`: a high value is long. `-1`: a low value is long. |
| `book.top_fraction` | `0.2` | The long side: more than 0, not more than 0.5. |
| `book.bottom_fraction` | `0.2` | The short side: more than 0, not more than 0.5. Only in a `long_short` file: a `long_only` file with this key is refused. |
| `book.exit_fraction` | `0.4` | A coin in the book stays while it is in this top (or bottom) fraction. `long_short`: from the larger of `top_fraction` and `bottom_fraction` to 0.5. `long_only`: from `top_fraction` to 1.0. |
| `book.rebalance` | `weekly-monday` | Each Monday 00:00 UTC, on the values of the Sunday before. |
| `costs.scenario` | `taker` | `taker` (5 + 5 bps), `maker` (2 + 2 bps) or `custom` (give `fee_bps` and `slippage_bps`, 0 to 100). |
| `control.kind` | `zero` (long-short), `equal_weight` (long-only) | `zero`: no position. `equal_weight`: the equal-weight book of the eligible coins, with the same rebalance and the same costs. |
| `forward.sigma_floor_fallback` | `0.03` | From 0.001 to 1.0 (weekly-return units). The sigma floor of the forward test when the history has fewer than 52 weeks with a factor position. |

An unknown key is refused. `judge.py spec FILE` checks a file and shows its content and its
sha256. Comments and the layout do not change the sha256. To change the book, copy a
shipped file. Give the copy a new name. Then change the keys. At a Sunday with fewer than 5
coins with a value, the book has the control portfolio. BTC hold gives context only.

### 3. Check: the historical test

`judge.py check` runs the book on the history. The history starts at the first Sunday with at
least 5 eligible coins with a value. It ends at the last Sunday with a complete week. The test
uses the daily net excess return of the book over the control book, after fees, slippage and
funding. The result is "useless" when one of these conditions is true:

- The total return is 0 or less.
- The Sharpe ratio is 0 or less.
- The history has a 1-year window, and fewer than half of the rolling 1-year windows have a
  positive Sharpe ratio.

Else the result is "unproven (historical pass)". The historical test never gives "useful".
A rolling 1-year window has 365 days, and the next window starts 91 days later. The sigma
floor of the forward test comes from the weekly excess returns of the history.

`check` also shows numbers that no rule reads. These are the weekly IC and rIC, the
persistence, the coverage, BTC hold and the largest correlation with a registered test.
`--novelty` also gives the largest correlation with a library definition. `--json` gives
the result as JSON. When the test can be registered, `check` prints its register command.

### 4. Register

`judge.py register` adds the test to the trial ledger. It refuses:

- a formula that is not proven point in time;
- a history without a complete week of values;
- a duplicate of a registered test (see
  [8. Duplicates and related tests](#8-duplicates-and-related-tests));
- a change of the test code that the ledger does not hold (see
  [9. The trial ledger](#9-the-trial-ledger)).

You can register a factor that the historical test calls "useless". Its verdict is
"useless (historical)" until the forward test decides. The first forward Sunday is the first
Sunday after the newest data and after the day of the registration. `--name` gives the test a
name, and `--note` adds a note to the registration line. `--dry-run` shows the line and writes
nothing.

### 5. The weekly forward test

Week 1 has the first forward Sunday as its signal Sunday. Each next week starts 7 days
later. The book trades on the Monday after the signal Sunday. A week is complete when the
OKX data reaches the last day of the week. On each day of the week, each coin has its bar and
its funding value, or the 7-day rule below applies. A coin with no value of a field in the 14
days to the end of the week has no gap in that field. For a formula, the week also needs the
logged values of its signal Sunday. Each complete week is frozen in the ledger.

**The 7-day rule for missing data.** A missing value holds the week for up to 7 days. After
7 days the gap is final: the week counts, and a `data_gap` alarm names the missing data.
Thus a coin without new bars does not hold the test for ever.

For a formula, `judge.py log-week` computes the value of each new Sunday on the data up to
that Sunday. A formula on the Binance data waits for the update of the Binance panel. While
the Binance data ends before the OKX data, `status` shows the test as held. A missing value
of a coin at the new Sunday also holds the test, and `status` shows it.

For a values file, give the values of the new Sundays with `log-values`. Run the commands
from the repository root:

```bash
cd engine
.venv/bin/python -B judge.py log-values FT-0002 --values factor_values.csv
.venv/bin/python -B judge.py log-week
```

A logged value counts from the first daily close after its log time, and only in its own
week:

| Log time of the value of Sunday S | The book trades |
|---|---|
| On Sunday S, before 24:00 UTC | On Monday 00:00 UTC |
| Later in the week | At the next daily close (00:00 UTC) |
| On the next Sunday, or later | The value is not used |

Only the first log of a Sunday counts. A file that gives another value for a logged cell
is refused (exit 1): the call logs no value and records the alarm `file_value_changed`.
Remove the changed cells from the file, then run `log-values` again. At a Sunday without a
used value, the book has the control portfolio, and the week counts. The log time comes
from the clock of your computer.

### 6. The verdict rule

The rule of the test (`rules.RULE_TEXT`; `judge.py status` prints it):

- The historical test can give only "useless" or "unproven".
- The historical test gives "useless" when one of these conditions is true after costs:
  - The total return is 0 or less.
  - The Sharpe ratio is 0 or less.
  - Fewer than half of the rolling 1-year windows have a positive Sharpe ratio. This
    condition applies only when the history has a 1-year window.
- You can register a factor that the historical test calls "useless". Its verdict is
  "useless (historical)" until the forward test decides.
- Only the forward test can give "useful". It uses only the weeks after the registration.
- A forward decision replaces the historical result. A factor without a complete week of
  history cannot be registered.
- Each week, the test compares the net return of the book with the net return of the control
  book.
- A confidence sequence decides. Its design value for the false-pass rate of each test is
  10 % or less.
- Strong serial correlation, or a large increase of volatility after the registration, can
  make the false-pass rate larger.
- From week 26: "useful" when the lower bound is more than 0, "useless" when the upper bound
  is less than 0.
- At week 156, a test that is not "useful" is "useless".
- BTC hold gives context only.

The confidence sequence is two-sided, with alpha 0.10. Its sigma is the largest of the
sample SD, the long-run SD (Bartlett, 4 lags) and the sigma floor. The sigma floor comes from
the history (at least 52 weeks with a factor position), else from the book specification.
The first decision is final.

The false-pass rate is the probability that a factor with a true weekly mean excess return of
0 gets "useful". The design value of 10 % is not a guarantee for every return process. It can
be larger in these conditions:

- the weekly excess returns have a strong serial correlation;
- the volatility after the registration is much larger than the volatility of the history;
- the book has rare large losses (a negative skew), and the sigma floor is low.

### 7. The expected number of false "useful" verdicts

`judge.py status` and the page My factors show 0.1 x N. N is the number of registered tests
that are not withdrawn (each state). When each test has a false-pass rate of 10 % or less,
the expected number of false "useful" verdicts is at most 0.1 x N. More tests give more
false "useful" verdicts.

### 8. Duplicates and related tests

A test is a duplicate of a registered test when the two tests have the same book and the
same signed factor values. The same book is the same specification without the name, the
version, the sign, the name of the cost scenario and the table `[forward]`. Two tests have
the same signed factor values in one of these cases:

- They have the same formula, the same source and the same sign.
- They have the same values of a values file. Another file form or another row order does
  not change the values.
- The mean within-Sunday Spearman correlation of their signed values is more than 0.999.

`register` refuses a duplicate. A related test has an absolute correlation of 0.85 or more,
with any book. `status` shows the related tests of each test.

### 9. The trial ledger

| File in `engine/data/trials/` | Content |
|---|---|
| `registrations.jsonl` | One line for each registration, withdrawal and code change |
| `values/okx.jsonl`, `values/binance.jsonl` | The logged Sunday values of the formula tests |
| `values/file.jsonl` | The logged values of the values-file tests, with the log time |
| `weeks.jsonl` | One frozen line for each complete forward week |
| `alarms.jsonl` | The alarms. Each alarm is recorded once. |
| `head.json` | The line count and the last sha256 of each file above |
| `results/<id>/` | Result files: the historical test and the weekly tables. `verify --write-results` writes them again. |

Each line holds the sha256 of the line before it. An edited, removed or moved line stops each
command (exit 1). Factorlab never repairs a ledger. `judge.py verify` checks each line and
computes the logged numbers again. Another number gives an alarm (exit 3), and the frozen
weeks stay. The decision uses only the frozen weeks.

| Alarm | Description |
|---|---|
| `data_gap` | A value is missing for more than 7 days. The week counts. |
| `pit_mismatch` | The value of a formula at a Sunday cut differs from the full computation. The test is held at that Sunday. |
| `data_shorter` | The data on disk ends before a logged Sunday. |
| `inputs_changed` | The data of a logged Sunday changed. |
| `value_changed` | A logged formula value differs from a new computation. |
| `book_week_changed` | A new replay gives another number for a frozen week. |
| `history_changed` | The history values of a test differ from the values of its registration. |
| `file_value_changed` | `log-values` got another value for a logged cell. Only the first log counts. The call is refused (exit 1) and logs no value. Remove the changed cells from the file, then run `log-values` again. |

`withdraw` is possible only before the first forward Sunday and before a logged value. A
withdrawn test stays in the ledger and is not in N. It has no verdict: `status` and the pages
show "none (withdrawn)".

Each registration holds the hashes of the test code (`trials.CODE_FILES`) and of the test
values of `engine/config.py` (the exchange names and the coins). After a change of that code
or of these values, `register` and `log-week` refuse until
`judge.py record-code-change --reason TEXT` records the change. `update.py` records a new
version of Factorlab automatically (`log-week --accept-release`). An edit of a switch in
`engine/config.py`, for example `ALPHAFORGE_ENABLED`, is not a code change.

### 10. Command reference

| Command (`judge.py`) | Function |
|---|---|
| `check (--formula=EXPR \| --library ID \| --values FILE) [--source okx\|binance] [--spec FILE] [--novelty] [--json]` | The historical test. Writes nothing. |
| `register (same input options) [--name NAME] [--note TEXT] [--dry-run]` | Register a test. |
| `log-values ID --values FILE` | Log the new Sundays of a values-file test. |
| `log-week [--through YYYY-MM-DD] [--accept-release]` | Log the formula values and the new complete weeks. |
| `status [ID] [--json]` | The tests, their verdicts, the totals and the rule. Read-only. |
| `verify [--write-results]` | Check the ledger and compute it again. |
| `withdraw ID --reason TEXT` | Withdraw a test before its first forward Sunday. |
| `record-code-change --reason TEXT [--note TEXT ...]` | Record a change of the test code. |
| `spec [FILE\|long_short\|long_only] [--json]` | Check a book specification and show its sha256. |

Exit status: 0 done; 1 a refusal or a broken ledger; 2 a usage error; 3 a new alarm.
`judge.py` uses no network and writes only `engine/data/trials/`.

## The dashboard

Open `web/index.html` in a browser. The page works from a local file and needs no server and
no network. `setup.sh` opens it at the end. The first page is My factors.

The navigation has three groups: Find, Test and System.

| Group | Page | What it shows |
|---|---|---|
| Find | Factor library | Each library definition with its formula and its two commands (check and register). The counts of the families. The numbers of the screens on OKX and Binance data. |
| Find | Search tools | The numbers of the factor-lab registry, the AlphaForge formulas (when AlphaForge is on) and the command of the generator probe. |
| Test | My factors | The tests of your factors. For each test: the historical result, the forward week, the bounds of the confidence sequence, the state and the verdict. Also the expected number of false "useful" verdicts, the rule, the two shipped book specifications and the commands. |
| System | System | The last build (`update.py`), the update steps, the programs, the data inventory and coverage, environment variables, rules, tests, documents and the glossary. |

The address `#/factor/<id>` opens the page of one test, for example `#/factor/FT-0001`. The
page shows the factor, the book specification, the historical test, the forward test and the
record in the trial ledger. The historical test shows the weekly marks of the book, the control
book and BTC hold. It also shows their statistics and the Sharpe ratio of each 1-year window.
The forward test shows each frozen week with its bounds and its state. A card name after the id
opens the page at that card, for example `#/factor/FT-0001/forward`. Each id in the table of
My factors links to this page. For a library id (`CF…`), the page links to the library search.
When `judge.py` wrote the result files of a test again after a change of the data, the pages
show a warning. The verdict always uses the numbers of the trial ledger.

When no test is registered, My factors shows "No factor is registered." and the commands that
check and register a factor. The demo build registers no factor. When the trial ledger does not
pass its checks, My factors and the page of each test show the problems and the command
`judge.py verify`. They show no test.

The dashboard is a static snapshot. The header shows the last daily bar of the data and
the export time. When a closed daily bar is not in the data (UTC, by the clock of your
computer), the header starts with a warning. An example: "Data is 4 days behind: last bar
7 Oct 2026; 4 closed days are missing. To add them: ./setup.sh". With the demo data, each
page shows "Synthetic demo data. Not market data." The footer gives the version and the
license of the code and of the data.

After a data update, `update.py` builds the dashboard again (step "dashboard"). To build
only the dashboard:

```bash
cd engine
.venv/bin/python -B export_web.py
```

Details: [web/README.md](web/README.md).

## Tests

`engine/run_tests.py` runs every test suite and prints one line for each suite:

| Result | Condition |
|---|---|
| PASS | The suite exits 0 and runs at least one test. |
| SKIP | The suite exits 0 and runs no test. Usually the data for its tests is not on disk, or `pytest` is not installed. |
| FAIL | The suite exits with a status other than 0. The runner shows the last lines of its output. |

Each line gives the run time and the counts of tests. The last line gives the number of
passed, skipped and failed suites, and the names of the skipped and failed suites. The
runner exits 1 when a suite fails. A skipped suite does not make it fail.

```bash
engine/.venv/bin/python -B engine/run_tests.py                 # every suite
engine/.venv/bin/python -B engine/run_tests.py --quick         # the short smoke set
engine/.venv/bin/python -B engine/run_tests.py --only test_rules --only library
engine/.venv/bin/python -B engine/run_tests.py --list          # the suite names
engine/.venv/bin/python -B engine/run_tests.py --verbose       # the full output of each suite
```

The suites:

| Suite | What it checks |
|---|---|
| `test_labels.py` | the label table: alignment, no invented label, row shifts, a pinned digest on exact prices |
| `test_backtest.py`, `test_accounting.py` | the backtester: no lookahead, the funding sign, fees charged once, scale invariance; the trade log rebuilds the equity |
| `test_book.py` | the book specification files, the banded book, the control books, the weekly series |
| `test_rules.py` | the historical test, the confidence sequence, the false "useful" rate under a zero mean, the verdict and the rule text |
| `test_userfactor.py` | the formula check, the values at each Sunday cut, the values files, the factor numbers |
| `test_trials.py` | the trial ledger and `judge.py`: the hash chain, the duplicates, the logs, the alarms, the code changes, the withdrawal rules, an end-to-end run of 30 forward weeks |
| `test_libfactors.py`, `test_libfactors_binance.py` | the library on OKX and Binance data: point in time, the screen, the cleaning freeze, numbers only |
| `test_finder_*.py`, `test_incumbents.py` | the generator probe and its parts, the novelty index, the incumbents of the factor lab and AlphaForge |
| `test_diffusion.py` | the diffusion factors of the factor lab: point in time |
| `test_alphaforge_bridge.py` | the AlphaForge switch, the zoo format, the forward window, the capacity, point in time |
| `test_engine_rules.py` | the funding splice, the network switch, write protection, the help of each program, the import check |
| `test_update.py`, `test_demo_data.py` | the step plan of `update.py`, the install files, the demo data set |
| `test_version.py` | the file `VERSION`, `CHANGELOG.md`, the badges, the relative links of the root documents, the license files, the contact channels |
| `test_export_*.py` | each dashboard section: each number on a page computed again from the source files |

`--quick` runs `test_backtest`, `test_accounting`, `test_labels`, `test_book`, `test_rules`
and `test_export_web`. These suites make their own data and need no data folder.

The runner also runs the library suite (`library/tests`) and the AlphaForge suite
(`engine/alphaforge/tests`). These two suites need `pytest`: `./setup.sh --test` installs it.
Without it, the runner shows them as SKIP. Their property tests need the package
`hypothesis`. `./setup.sh --test` installs it too (`engine/requirements-dev.txt`). Without
it, these tests skip.

Most suites make their own data in a temporary folder. Some suites also read the data
folder, for example the suites of the library screens, the generator probe and the
dashboard. Without built data, their data tests skip. With the demo data
(`./setup.sh --demo --test`), each suite runs. The full set takes about 10
minutes. The page-render checks of the exporter tests run only when
`node` is on the `PATH`, or when `FACTORLAB_NODE=<path>` or `NODE=<path>` is set.

**The test workflow.** `.github/workflows/tests.yml` runs on GitHub Actions. Each job installs
the pinned packages of `engine/requirements-ci.txt` with `--require-hashes`, wheels only, and
downloads no market data:

| Job | Systems | When | Checks |
|---|---|---|---|
| tests | Linux (Ubuntu 24.04) with Python 3.12 and 3.13; macOS 15 with Python 3.13 | Each push to `main`, each pull request, each version tag `v*` and on request | `ruff check`, `ruff format --check`, `bandit`, `mypy`, then `run_tests.py` under `coverage`. The job summary gives the total coverage. |
| demo data | Linux (Ubuntu 24.04) with Python 3.13 | Each Monday, each version tag `v*` and on request | `update.py --demo`, then `run_tests.py` under `coverage`. A suite with SKIP fails the job. The job also fails when the coverage total is below 84 %. On a GitHub-hosted runner the build of the demo data alone takes about 30 minutes. |
| pip-audit | Linux (Ubuntu 24.04) with Python 3.13 | Each push to `main`, each pull request, each version tag `v*`, on request and each Monday | `pip-audit` on the three requirement files. |

The badge "tests" at the top of this document shows the result of the last run on `main`.
It shows on GitHub only, to the users who can read the repository.

To start the jobs on request:

1. Open the Actions tab of the repository.
2. Select the workflow `tests`.
3. Click **Run workflow**.

With the GitHub CLI:

```bash
gh workflow run tests.yml --ref main
```

**The code checks on your computer.** The settings are in `pyproject.toml`. The style is
PEP 8 with the ruff rules E, F, W, I, B, UP and SIM, and a line length of 120:

```bash
engine/.venv/bin/python -m pip install --require-hashes -r engine/requirements-ci.txt
engine/.venv/bin/ruff check .
engine/.venv/bin/ruff format --check .
engine/.venv/bin/bandit -c pyproject.toml -r engine library
engine/.venv/bin/mypy
```

## Data sources and product rules

**Data sources.** Factorlab reads only public market data. It needs no account and no
API key.

| Source | Data | Used for |
|---|---|---|
| OKX public API (through `ccxt`) | daily perpetual bars, daily spot bars, funding settlements, open interest | the engine: the labels, the books and the OKX screen |
| Binance USD-M public API (through `ccxt`) | funding settlements | the funding history before the first OKX settlement on disk |
| Binance USD-M public API and data.binance.vision | daily klines, funding, premium index, open interest, long/short and taker ratios | the factor library and the formulas on Binance data |

Binance is a data source only. All labels and books use the OKX perpetuals.

**Data terms.** The code of Factorlab is under the MIT License (see [License](#license)).
The MIT License covers the code only. It gives no right in market data. You download the
data yourself, and the terms of each source apply to the data and to the results that you
make from it. These terms are stricter than the MIT License:

- OKX data: the OKX Terms of Service and the
  [OKX API Agreement](https://www.okx.com/help/okx-api-agreement), which also covers the
  market data of the public endpoints. OKX allows its market data only for your own
  personal, noncommercial use. Do not share or publish OKX data.
- Binance REST API data: the Binance Terms of Use.
- The data.binance.vision archive: the
  [Binance Vision Dataset Terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md)
  (CC BY-NC-SA 4.0, noncommercial use only). These terms allow research and personal
  backtests. They do not allow live trading with the data or with results that you make
  from it. If you share a result that you make from it, credit "Binance Vision" and use
  the same license.

Each data source has its own rules for regions. Obey the terms of each source in your
region.

Do not commit or publish the downloaded data or the built outputs (`engine/data/`,
`library/data/`, `library/catalog/`, `web/data/`). Factorlab is not affiliated with or
endorsed by OKX or Binance. Factorlab is not related to the packages
`factorlab` and `alphaforge` on PyPI.

**Rules of the product.**

- **UTC.** Each timestamp is UTC. A daily bar has the stamp of its open time. The bar
  stamped D covers D 00:00 to D+1 00:00 UTC.
- **Point in time.** A factor value for a day D uses only data that is known at the close
  of bar D. A bar that is not closed is never written to disk. A formula value for a Sunday
  uses only the data up to that Sunday.
- **Static universe.** `engine/config.py` sets the 20 coins. Do not delete a coin. To add
  a coin, see [engine/README.md](engine/README.md#change-the-venue-or-the-universe).
- **Funding sign.** A positive funding rate means that longs pay shorts.
- **Costs.** Taker costs (5 bps fee and 5 bps slippage) are the default. Maker costs
  (2 and 2 bps) are a second scenario. Each book pays fees, slippage and funding.
- **Position size.** Each weight is a fraction of equity. Gross exposure is at most 1.0.
- **BTC hold.** BTC hold gives context only. No rule reads it.

**Network switch.** `FACTORLAB_OFFLINE` controls the network. Unset or `1` means
offline. Exactly `0` allows the downloads. Any other value stops the program. Only
`engine/ingest.py` and `python -m factorlab download` use the network. `update.py` sets
the switch for each step: `0` for the download steps, `1` for all other steps.

**Input limits.** A factor expression of the library has at most 4,000 characters and 64
nested nodes. A power of two constants needs an exponent of at most 100 and a finite
result. Other input gives `ExpressionError`. A values file has at most 2,000,000 rows and
100 MiB. The AlphaForge parser refuses a text or a tree deeper than 100 levels
(`ExprSyntaxError`). The library downloader opens only `https` URLs. It compares each
data.binance.vision zip file with the SHA-256 in its `.CHECKSUM` file before it reads the
file. A missing checksum file is an error. Each response body has a size limit. The limits
are 16 MiB (a REST response), 50 MiB (a zip file) and 200 MiB (the CSV file in a zip
file). Details: [library/README.md](library/README.md#expressions).

**Write protection.** Put an empty file `.read_only` in a data folder to protect it:

```bash
touch engine/data/.read_only
```

Then each engine writer refuses to write into the folders `raw`, `labels`, `research` and
`trials`, and `update.py` refuses to run. Set `FACTORLAB_WRITE_PROTECTED=1` to let a single
program write anyway. `update.py` refuses a protected folder in all cases. Remove the file
to stop the protection.

**Environment variables.**

| Variable | Effect |
|---|---|
| `FACTORLAB_DATA` | Use another data folder (default `engine/data`). |
| `FACTORLAB_LOGS` | Use another log folder inside the repository (default `engine/logs`). |
| `FACTORLAB_OFFLINE` | The network switch (above). |
| `FACTORLAB_WRITE_PROTECTED` | `1` allows writes into a protected data folder. |
| `PYTHON` | The Python that `setup.sh` uses to make `engine/.venv`. |
| `FACTORLAB_NODE`, `NODE` | The `node` program of the page-render tests (see [Tests](#tests)). |

The settings in the code (the coins, the venue, the zones, the AlphaForge settings) are in
the configuration table of [engine/README.md](engine/README.md#configuration).

## Folder layout

```
setup.sh                 one-command install, download and build
VERSION                  the version of Factorlab (MAJOR.MINOR.PATCH)
LICENSE                  the MIT License
LICENSES/                the license texts (MIT, ISC) that REUSE.toml names
THIRD_PARTY_NOTICES.md   notices for third-party material
REUSE.toml               the license and the copyright of each file, for tools
CITATION.cff             how to cite Factorlab
CHANGELOG.md             the changes of each version (also the text of the GitHub releases)
ROADMAP.md               the roadmap
CONTRIBUTING.md          issues, bug reports and the rules for the code
CODE_OF_CONDUCT.md       the rules of behavior, and how to report a problem
SECURITY.md              how to report a vulnerability, and the security model
pyproject.toml           the settings of ruff, mypy, bandit and coverage
.github/                 the test workflow, the Dependabot settings, the issue forms and the
                         pull request template
engine/                  the research engine (Python)
  config.py              universe, venue, paths, switches
  update.py              download and build chain
  judge.py               the test of your factor (the command line)
  bookspecs/             the shipped book specifications (long_short.toml, long_only.toml)
  demo_data.py           the synthetic demo data set
  version.py             reads the file VERSION
  run_tests.py           test runner
  requirements*.txt      the pinned packages with their hashes (runtime, test, checks)
  *.py                   one module for each function (see engine/README.md)
  factorfinder/          the label-free generator parts and the incumbent provider
  alphaforge/            the AlphaForge expression miner and its tests
  webexport/             one dashboard section builder for each page
  data/                  data and outputs (built by setup.sh; git ignores it)
    raw/                 the downloaded market data
    labels/              the label table
    library_on_engine/   the library screens
    research/            the factor lab and AlphaForge files
    trials/              the trial ledger of your tests
  logs/                  the update log and lock (git ignores it)
library/                 the factor library package
  LICENSE                a copy of the MIT License for the package
  pyproject.toml         the package metadata and the factorlab command
  factorlab/             definitions, operators, evaluator, downloader, CLI
  scripts/               merge_panels.py
  tests/                 the library test suite
  catalog/               the built catalog (git ignores it)
  data/                  the Binance panel and work files (git ignores it)
web/                     the static dashboard
  index.html             open this file
  assets/, pages/        the app and one script for each page
  data/                  the exported data (git ignores it)
```

## Questions and answers

**Why can the historical test not give "useful"?** You can see the history before you
register a factor. A formula that you choose after a look at the history can fit the
history by chance. Only the weeks after the registration are data that the choice did not
see. Thus only the forward test can give "useful".

**Why does a test need 26 weeks?** The weekly excess return of a book has much noise. With
fewer weeks, the bounds of the confidence sequence are too wide for a decision. The
confidence sequence is tuned for week 26, the first week that can give a decision.

**Why is a test "useless" at week 156?** Each test has a maximum length, so that each test
ends. A factor that the forward test does not show as useful in 156 weeks (3 years) gets
"useless". A factor with a small true effect can also end as "useless".

**Why does a registered test show "useless (historical)"?** The historical test of the
factor gave "useless". You can still register such a factor. The verdict stays "useless
(historical)" until the forward test decides, and a forward decision replaces it.

**Can I test values from another program?** Yes. Give them as a values file (CSV or
Parquet with the columns `date`, `symbol` and `value`). Factorlab cannot prove the values
point in time. Thus a forward value counts only from the first daily close after its log
time. Log the values of each new Sunday with `judge.py log-values`.

**Does Factorlab give a list of useful factors?** No. Factorlab gives no factor of its own
as useful or useless. The search tools give numbers. Only the test of a factor that you
register gives a verdict.

**Why does a test suite show SKIP?** Its data is not on disk, or `pytest` is not
installed. Run `./setup.sh --test` for `pytest`. Build the data (`./setup.sh`) or the demo
data (`./setup.sh --demo`) for the data suites.

**Does Factorlab work without the Binance panel?** Yes, with `./setup.sh --skip-binance`.
The OKX screen, the other search tools and the tests of OKX formulas and values files work
without the panel. The Binance screen and the test of a formula on the Binance data need
the Binance panel (see [Requirements](#requirements)). Each data source has its own rules
for regions. Obey the terms of each source in your region (see
[Data terms](#data-sources-and-product-rules)).

**Can Factorlab trade?** No. It places no orders and holds no exchange keys. Each book is a
simulation.

**How do I add a coin?** Add it to `config.UNIVERSE` and build into a new data folder. See
[engine/README.md](engine/README.md#change-the-venue-or-the-universe).

**Can I use the demo data for research?** No. A random generator makes the demo data, and
it holds two planted signals. Use it to see the functions and to run the tests.

**Which version do I have?** `cat VERSION`. The first line of `update.py` and the footer of
the dashboard show it too. [CHANGELOG.md](CHANGELOG.md) lists the changes of each version.

## Problems and solutions

| Message or symptom | Cause and action |
|---|---|
| `Python 3.12 or 3.13 is necessary` or `is not Python 3.12 or 3.13` | Install Python 3.12 or 3.13. Or set `PYTHON=/path/to/python3.13` and run `./setup.sh` again. |
| `engine/.venv does not have Python 3.12 or 3.13` | Remove `engine/.venv`. Then run `./setup.sh` again. |
| `engine/.venv has no pip`, or `ensurepip is not available` | On Debian and Ubuntu (also in WSL2), install the package `python3.X-venv` of your Python. The message of `setup.sh` gives the name (for example `sudo apt install python3.12-venv`). Remove `engine/.venv`. Then run `./setup.sh` again. |
| pip fails to build `cryptography` (Intel Mac) | PyPI has no wheel for this system. Install Rust and OpenSSL 3 (`brew install rust openssl@3`), then run `./setup.sh` again. |
| `THESE PACKAGES DO NOT MATCH THE HASHES` | A downloaded package file is not the pinned file. Run `./setup.sh` again. If the message stays, empty the pip cache (`engine/.venv/bin/python -m pip cache purge`). |
| A download step fails | The network or the exchange stopped the request. The message names the coins that failed. Run `./setup.sh` again. On the first setup, complete coin downloads stay on disk, and only the incomplete coins download again. An update downloads the new days of all coins again. |
| A Binance download fails with `HTTP Error 451` | Binance refuses requests from your region. Run `./setup.sh --skip-binance`. Then Factorlab has no Binance screen, and it cannot test a formula on the Binance data (see [Requirements](#requirements)). |
| `Binance banned this IP address after too many requests (HTTP 418` | Wait, then run again with fewer parallel downloads, for example `./setup.sh --jobs 2`. |
| A Binance download stops with `ArchiveError` | An archive file does not agree with its checksum file, or it is damaged. Run `./setup.sh` again. |
| `Binance append` stops on an incomplete field | Binance did not publish a daily archive file before the run. Run the update again one day after. |
| A build step fails | Correct the cause in the message. Then run `./setup.sh --from "STEP"` (the step name in the message). |
| A step stops with `did not end in 43200 s` (exit 124) | The step ran for more than 12 hours. Find the cause in its output, for example a slow disk or a full disk. Then run `./setup.sh --from "STEP"`. |
| `another update is running` | Another `update.py` holds the lock. Wait until it stops. |
| `update.py: no market data on disk` | `--offline` needs downloaded data. Run `./setup.sh` without `--offline`, or use `--demo`. |
| `--demo: ...` refuses the data folder or the Binance panel | The folder holds files that are not demo data. Use an empty folder (`FACTORLAB_DATA=<folder>`), and add `--skip-binance` when the Binance panel holds market data. |
| `the data folder holds the synthetic demo data` | A build without `--demo` refuses demo data. Run `engine/.venv/bin/python -B engine/demo_data.py --remove`, or use another folder (`FACTORLAB_DATA=<folder>`). |
| `the data folder ... is protected` | The data folder has the file `.read_only`. Remove it, or use another folder with `FACTORLAB_DATA`. |
| `judge: log-week refused. The code differs from the ledger.` or `judge: not registered. The code differs from the ledger.` | A file of the test code (`trials.CODE_FILES`) or a test value of `engine/config.py` changed without a new `VERSION`. Record the change: `cd engine && .venv/bin/python -B judge.py record-code-change --reason "TEXT"` (see [Update Factorlab](#update-factorlab)). Then run `./setup.sh --from "factor tests, weekly log"`. |
| A `judge.py` command stops with exit 1 and names a line or `head.json` of `engine/data/trials/` | The trial ledger does not pass its checks: a line was edited, removed or moved. Factorlab never repairs a ledger. Restore the folder `engine/data/trials/` from your backup. `judge.py verify` lists the problems. |
| The step "factor tests, weekly log" says `done (new alarm)`, or `judge.py log-week` exits 3 | The ledger recorded a new alarm. Read it with `judge.py status`. The frozen weeks stay. The update continues. |
| `libfactors: the Binance panel (binance_um_1d) is missing` | The Binance panel is not on disk. Run `./setup.sh` without `--skip-binance`. Where Binance refuses access, test formulas on the OKX data (`--source okx`) or give a values file. |
| `libfactors: the Binance cleaning rules differ from the set that CLEANING_FREEZE pins` | The cleaning rules in `engine/libfactors.py` differ from the pinned rules. Undo the local edit of these rules (`git diff engine/libfactors.py` shows it). |
| `alphaforge: the zoo file data/research/alphaforge_library.json has the format ...` (or `factor_lab: the zoo file ...`) | Factorlab reads only a zoo file of format 3. Restore the file from a backup (`alphaforge_library_<run_id>.json` in the same folder), or move it out of `engine/data/research/`. Without a zoo file, AlphaForge starts an empty zoo. |
| `status` shows a test as held | The test waits for data, or a `pit_mismatch` alarm holds it. A missing value holds a test for up to 7 days. A Binance formula waits for the update of the Binance panel. Run the next update after the data arrives. |
| The suites `library` and `alphaforge` show SKIP with `pytest is not installed` | Run `./setup.sh --test`, or `engine/.venv/bin/python -m pip install --require-hashes -r engine/requirements-dev.txt`. |
| The dashboard shows "No data" | The data for that page is absent. Run `./setup.sh`, or `engine/update.py --from "dashboard"`. |
| The dashboard header shows "Data is N days behind" | The data does not have the newest closed days. Run `./setup.sh`. |
| A page says that a file "does not parse" | The data or page file is damaged or incomplete. Run `engine/update.py --from "dashboard"`. |
| A served dashboard shows old pages after an update | The browser uses old copies of the files. Reload the page without the cache. |

## More documentation

| Document | Topic |
|---|---|
| [engine/README.md](engine/README.md) | Command reference of every engine program, and the configuration table |
| [engine/README_FINDER.md](engine/README_FINDER.md) | The library screen on Factorlab's data and the generator probe |
| [engine/README_ALPHAFORGE.md](engine/README_ALPHAFORGE.md) | The optional AlphaForge miner, its forward window and its capacity |
| [engine/README_AUTOMATION.md](engine/README_AUTOMATION.md) | `update.py` and a scheduled run of it |
| [engine/GLOSSARY.md](engine/GLOSSARY.md) | The research terms, the metrics and the market terms |
| [library/README.md](library/README.md) | The factor library |
| [web/README.md](web/README.md) | The dashboard |
| [CHANGELOG.md](CHANGELOG.md) | The changes of each version, the newest first |
| [ROADMAP.md](ROADMAP.md) | The roadmap |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Issues, bug reports and the rules for the code |
| [SECURITY.md](SECURITY.md) | How to report a vulnerability, and the security model |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | The rules of behavior, and how to report a problem |

## License

Copyright (c) 2026 Radioiic. The code of Factorlab is under the
[MIT License](LICENSE) (SPDX identifier `MIT`). You can use, change and share the code for
any purpose, also commercial use. Keep the copyright notice and the license text in each
copy. The software comes without warranty.
For a question about the license, open an issue with the form "Question".

The MIT License covers the code only. The market data keeps its own terms. OKX allows its
data only for personal, noncommercial use. The Binance Vision archive is under
CC BY-NC-SA 4.0. See
[Data sources and product rules](#data-sources-and-product-rules). Third-party material
in the code has its own notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
`REUSE.toml` gives the license and the copyright of each file in a form that tools can
read. [CITATION.cff](CITATION.cff) tells you how to cite Factorlab. The file `VERSION`
holds the version of Factorlab.

Factorlab does not accept pull requests. Issues are welcome: see
[CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). To report
a vulnerability, see [SECURITY.md](SECURITY.md). To report a problem of behavior, see
[Report a problem](CODE_OF_CONDUCT.md#report-a-problem).

## Disclaimer

Factorlab places no orders. Its numbers and verdicts are research results. They are not
advice. Nothing in Factorlab is investment advice or a recommendation to buy or sell
anything. Each book is a simulation on past or forward data, and its results do not
predict future returns. Crypto derivatives can lose more than the money put in. The
software comes as is, without any warranty, and the author is not liable for any loss from
its use (see [LICENSE](LICENSE)).
