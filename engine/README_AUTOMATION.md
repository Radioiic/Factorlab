# Automation: update.py and a scheduled run

`engine/update.py` downloads the new market data and builds every output. `setup.sh` runs
it. You can also run it on a schedule. Factorlab installs no scheduled job.

## update.py

Run it from the repository root:

```bash
engine/.venv/bin/python -B engine/update.py              # download, then build
engine/.venv/bin/python -B engine/update.py --plan       # show the steps; run nothing
engine/.venv/bin/python -B engine/update.py --offline    # build from the data on disk
engine/.venv/bin/python -B engine/update.py --demo       # build from synthetic demo data
engine/.venv/bin/python -B engine/update.py --from "dashboard"
```

| Option | Effect |
|---|---|
| `--offline` | Build from the market data on disk. Download nothing. It stops when no market data is on disk. |
| `--demo` | Write the synthetic demo data set (`demo_data.py`), then build every output from it. No download. See [../README.md](../README.md#try-factorlab-with-demo-data). |
| `--plan` | Show the numbered steps. Run nothing. |
| `--skip-binance` | Do not download or use the Binance panel. |
| `--skip-library` | Do not compute the factor library. |
| `--jobs N` | Binance downloads at the same time (default 8). |
| `--from STEP` | Start at this step: its name from `--plan` (case does not matter) or its number. |
| `--verbose` | Show the output of each step, and add each output line to `logs/update.log`. |

`--plan` also shows the coins and the days of each download step.

### The steps

`--plan` prints the steps of the current state. Some steps appear only in some states:

| Step | Program | When |
|---|---|---|
| demo data | `demo_data.py --write` | `--demo` (then no download step) |
| OKX download | `ingest.py` | not `--offline` and not `--demo` |
| Binance download; Binance merge | `factorlab download`; `scripts/merge_panels.py` | no Binance panel on disk; only the incomplete coins download |
| Binance download; Binance append | `factorlab download`; `scripts/merge_panels.py --append` | the panel ends before the download end (see the rules) |
| labels | `labels.py` | always |
| factor tests, weekly log | `judge.py log-week --accept-release` | `data/trials/registrations.jsonl` exists |
| library catalog | `python -m factorlab build` | no `library/catalog/factors.db` (not with `--skip-library`) |
| library on OKX data, compute and screen | `libfactors.py --compute`, `--screen` | not `--skip-library` |
| library on Binance data, compute and screen | `libfactors.py --source binance --compute`, `--screen` | a Binance panel exists, or this run has a Binance step, or `--demo`; and not `--skip-library` or `--skip-binance` |
| factor lab | `factor_lab.py` | the registry has no rows of the current UTC date |
| AlphaForge cycle | `alphaforge_bridge.py --cycle` | `config.ALPHAFORGE_ENABLED` is `True` |
| dashboard | `export_web.py` | always |

The demo plan has no step "factor tests, weekly log": the demo build registers no factor.

### Rules

- Only the download steps use the network. `update.py` gives them `FACTORLAB_OFFLINE=0`
  and all other steps `FACTORLAB_OFFLINE=1`.
- The Binance download ends on the last Sunday that closed at least one full day before
  the run (UTC). On a Monday, this is the Sunday of the week before. Thus a daily archive
  file that Binance publishes some hours late does not stop the append.
- The step "factor tests, weekly log" logs the new forward weeks of your tests. When the
  test code and the `VERSION` changed, it records the change in the trial ledger. Its exit
  status 3 (a new alarm) is not a failure. The log line of the step then says "done (new
  alarm)", and the build shows the lines of the alarm.
- The factor lab runs once for each UTC day. It adds rows to its registry for the run
  date, so a second run on the same day would add copies.
- The command refuses a protected data folder (a folder with the file `.read_only`).
  `--plan` still works.
- `--demo` writes only into an empty data folder or a demo data folder (the marker file
  `demo.json`). It writes its Binance panel only into an absent, empty or demo
  `library/data/binance_um_1d/`. A build without `--demo` refuses a demo data folder, and,
  without `--skip-binance`, a demo Binance panel. `demo_data.py --remove` removes the demo
  data set.
- It takes the lock `engine/logs/update.lock`. A second run at the same time stops with
  "another update is running".
- It writes to `engine/logs/update.log`. Each line has the form `<UTC time>  <text>`. The
  first line of a run gives the version (`Factorlab 0.2.1`); the console shows it too.
  Each step adds the line `<UTC time>  <step>  <result>  <seconds>s`. The result is `ok`,
  `done (new alarm)` or `failed (exit N)`. With `--verbose`, each output line follows its
  step line as `<UTC time>  <step> | <line>`. The System page of the dashboard shows the
  last run from this log. It counts a step with `done (new alarm)` as ok and shows a note
  about the alarm.
- When the run has a download step, the console shows a line about the data terms of OKX
  and Binance after the version line.
- A step that runs for more than 12 hours (`STEP_TIMEOUT_S`) stops and fails with
  exit 124.
- It stops at the first step that fails. It shows the last 40 lines of the output of that
  step and a hint. For a build step, it gives the step name for `--from`. For a download
  step, it names the coins that failed; run the same command again.
- Temporary files go to `engine/.tmp/update/`.

Each step can run again at any time. The downloads add only new closed days, and the
build steps write their outputs again. The weekly log adds only new weeks.

## A scheduled run

Factorlab has no scheduler of its own. To run the update on a schedule, call `update.py`
from the scheduler of your system. A daily bar closes at 00:00 UTC. Thus run the update
after 00:00 UTC, for example at 00:30 UTC.

An example cron line for a computer with a UTC clock. Replace `/path/to/Factorlab` with the
folder of the repository:

```
30 0 * * * cd /path/to/Factorlab && engine/.venv/bin/python -B engine/update.py >> engine/logs/cron.log 2>&1
```

Obey these rules for a scheduled run:

- Use the interpreter `engine/.venv/bin/python` with the full path, or `cd` into the
  repository first, as in the example.
- A scheduler can have no access to some folders of your user. On macOS, a background job
  has no access to `~/Documents`, `~/Desktop` and `~/Downloads`. Keep the repository in
  another folder.
- The lock prevents two runs at the same time. When a scheduled run starts while a manual
  run holds the lock, the scheduled run stops with "another update is running".
- After an update to a new release (a new `VERSION`), the next run records the change of the
  test code in the trial ledger. Some changes of the test code come without a new `VERSION`,
  for example a `git pull` between two releases. Such a change stops the run at the step
  "factor tests, weekly log". See [Update Factorlab](../README.md#update-factorlab).
- The Binance download needs access to Binance. Where Binance refuses access, add
  `--skip-binance` to the command.

## Logs and recovery

| File | Writer | Content |
|---|---|---|
| `engine/logs/update.log` | `update.py` | one line for each step |
| `engine/logs/update.lock` | `update.py` | the run lock |
| `engine/logs/cron.log` | your scheduler (example above) | the output of the scheduled run |

After a failure, read the failed step in the log. Correct the cause. Then run
`update.py --from "STEP"`, or run that program alone. A new full run adds the missed market
days. The weekly log adds the missed forward weeks of your tests in order: a week never
skips.

When the step "factor tests, weekly log" refuses because the code differs from the trial
ledger, record the change first. Run these commands from the repository root:

```bash
cd engine && .venv/bin/python -B judge.py record-code-change --reason "TEXT"
cd .. && engine/.venv/bin/python -B engine/update.py --from "factor tests, weekly log"
```

## Tests

`test_update.py` checks the step plan of `update.py` and its order, the step timeout, the
options of `--help` and the install files. It also checks the factor-test step: it runs only
with a trial ledger, and its exit status 3 is not a failure. `test_demo_data.py` checks the
plan of `update.py --demo`. Run them from the repository root:

```bash
engine/.venv/bin/python -B engine/run_tests.py --only test_update --only test_demo_data
```
