# Changelog

This file lists the changes of Factorlab that a user can see. The newest version is first.
The form follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and the
version numbers follow [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
The dates are UTC dates. The GitHub release of each version has the text of its section.

## [Unreleased]

## [0.2.1] - 2026-10-24

### Added

- Setup and update:
  - `./setup.sh` installs the pinned packages with their hashes, downloads the market data, builds each output, runs the short test set and opens the dashboard.
  - `./setup.sh --demo` builds from a synthetic demo data set. `--offline` builds from the market data on disk. Neither option downloads market data.
  - `engine/update.py` adds the new closed days and builds the outputs again. `--plan` shows the steps, and `--from STEP` starts at one step.
  - `--skip-binance` builds without the Binance panel. `--skip-library` builds without the factor library computation.
  - An empty file `.read_only` in a data folder stops `update.py` and each writer of the folders raw, labels, research and trials.
  - The label table `data/labels/labels.parquet` holds the forward returns `fwd_ret_1` and `fwd_ret_7` of each coin and day.
- Test your own factor (`engine/judge.py`):
  - The factor is a formula of the library language, a library definition or a values file. A values file is a CSV or Parquet file with the columns date, symbol and value.
  - Factorlab computes a formula at each Sunday cut. It refuses a formula when a Sunday value differs from the full computation.
  - A book specification file (TOML) gives the fractions, the direction, the sign, the costs and the control book. `engine/bookspecs/` has a long-short and a long-only file.
  - `judge.py check` runs the historical test after costs. Its result is "useless" or "unproven (historical pass)".
  - `judge.py register` adds a test to the trial ledger `data/trials/`. The ledger is append-only and hash-chained. A test that repeats a registered test is refused.
  - The weekly forward test compares the net return of the book with the control book. A confidence sequence decides from week 26; the maximum is 156 weeks.
  - Only the forward test can give "useful". A forward decision replaces the historical result.
  - `judge.py` also has the commands `log-values`, `log-week`, `status`, `verify`, `withdraw`, `record-code-change` and `spec`.
- Find tools (numbers only):
  - The factor library holds 28,904 formula definitions. The `factorlab` command builds the catalog and lists, shows, computes and evaluates the definitions.
  - `libfactors.py` computes the library definitions on OKX data and on Binance data for the 20 coins. It screens them on the Sundays 2022-04-03 to 2024-06-09.
  - `libfactors.py --status` names the code files that changed after the computation. The Factor library page then shows the callout "The code changed after the computation".
  - Each library definition has a check command and a register command (the library page and `python -m factorlab show`).
  - `finder.py --generator-probe` measures how many random formulas are new against the known factors. It reads no return label.
  - `factor_lab.py` scores a grid of network-diffusion factors each day. It logs each trial in an append-only registry.
  - The optional AlphaForge miner mines formulas. Each formula has a forward window of 52 weeks, and at most 6 are active. It is off by default.
  - The registries of the factor lab and AlphaForge give, for each trial, the correlation with the nearest incumbent and with the nearest library definition.
  - `alphaforge_bridge.py` and `factor_lab.py` read only an AlphaForge zoo file (`data/research/alphaforge_library.json`) of format 3. A file of another format or with no format stops them with exit status 1.
- Dashboard (`web/index.html`; it needs no server and no network):
  - The pages My factors, Factor library, Search tools and System, and one page for each registered test (`#/factor/<id>`).
  - My factors lists the tests, their verdicts and the expected number of false "useful" verdicts.
- Data sources and tests:
  - Factorlab reads only public market data from OKX and Binance. It needs no account and no API key, and it places no orders.
  - The network switch `FACTORLAB_OFFLINE`: unset or `1` is offline, exactly `0` allows the downloads, and any other value stops the program.
  - `engine/run_tests.py` runs each test suite and shows PASS, SKIP or FAIL for each suite.
  - The GitHub Actions workflow checks the code (ruff, bandit, mypy), runs each suite under coverage and runs pip-audit. Each Monday it also builds the demo data.
- License and documents:
  - The code is under the MIT License. `REUSE.toml` and `THIRD_PARTY_NOTICES.md` give the license data of each file and of the material from other sources.
  - The repository has `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `ROADMAP.md`, issue forms and the file `VERSION`.
  - The private reporting form of GitHub (the tab Security and quality, then "Report a vulnerability") takes the security reports and the code of conduct reports.
  - The issues take the questions, also the questions about the license (the form "Question").

[Unreleased]: https://github.com/Radioiic/Factorlab/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Radioiic/Factorlab/releases/tag/v0.2.1
