# Security policy

## Report a vulnerability

Use the private reporting form of GitHub. Only you and the maintainer can see the report.
Do not report a vulnerability in an issue.

1. Open the tab **Security and quality** of the [repository](https://github.com/Radioiic/Factorlab).
2. Click **Report a vulnerability**.
3. Complete the form.
4. Click **Submit report**.

Give:

- the version (the file `VERSION`) or the commit (`git rev-parse --short HEAD`);
- the steps that show the problem;
- the effect.

Remove your user name and your private paths from logs and messages before you send
them.

The maintainer answers a report in 14 days or less. The maintainer tells you when `main`
has the fix. [CHANGELOG.md](CHANGELOG.md) lists the fix under "Security".

## Scope

Reports about these subjects are useful:

- code execution caused by a downloaded file, a factor expression, a values file, a book
  specification or a configuration file;
- a file write outside the data folders, caused by one of these inputs;
- an edit, a removal or a move of one line of the trial ledger that `judge.py verify` does
  not find;
- a path in the dashboard export that shows private information from the computer;
- a network request when the network switch is off;
- a dependency version with a known vulnerability that affects Factorlab.

## Security model

**No keys and no orders.** Factorlab holds no exchange keys and no passwords. It places
no orders. It reads only public market data, and it needs no account.

**Inputs and trust boundaries.**

| Input | Trust | How Factorlab uses it |
|---|---|---|
| Market data from OKX and Binance (downloaded) | Not trusted | Read as numbers and dates into tables. No downloaded value runs as code. |
| Factor expressions (the factor library and your formulas) | Not trusted | Factorlab's own parser reads them. Only the operators of a fixed table can run. `judge.py` refuses a formula whose window or lag is not a constant whole number from 1 to 1000. |
| Values files (`judge.py --values`) | Not trusted | Read as a table of dates, coin names and numbers: at most 2,000,000 rows and 100 MiB. The trial ledger holds the base name and the sha256 of the file, never a path. |
| Book specification files (`judge.py --spec`) | Not trusted | Read as TOML data with a strict check: at most 64 KiB, and an unknown key is refused. |
| Configuration files in the repository (`.toml`) | Trusted | Read as data. They hold settings only. |
| The data folder (`engine/data/` or `FACTORLAB_DATA`) | Trusted | The files of the user. The dashboard export refuses to write a data file that holds a local path. Each line of the trial ledger holds the sha256 of the line before it. |
| The dashboard data files (`web/data/*.js`) | Made by the export | The pages put each value into the page as text, never as HTML. |

**Input limits.** The factor library parser refuses an expression with more than 4,000
characters or more than 64 nested nodes. It also refuses a power of two constants with an
exponent above 100 or an infinite result (`ExpressionError`). The AlphaForge parser refuses a text
or a tree deeper than 100 levels (`ExprSyntaxError`). The library downloader opens only
`https` URLs. It checks each data.binance.vision zip file against the SHA-256 in its
`.CHECKSUM` file before it reads the file. A missing checksum file is an error. Response
bodies have size limits. The limits are 16 MiB (a REST response), 50 MiB (a zip file) and
200 MiB (the CSV file in a zip file).

**No eval.** No program or package module of Factorlab calls `eval` or `exec`, and none
loads pickle files from a download. No test runs `exec` or `eval`.

**Network switch.** `FACTORLAB_OFFLINE` controls the network. Unset or `1` means offline.
Exactly `0` allows the downloads. Any other value stops the program. Only
`engine/ingest.py` and `python -m factorlab download` use the network. `update.py` sets the
switch to `0` only for its download steps.

**Trial ledger.** Each line of the trial ledger holds the sha256 of the line before it, and
`head.json` holds the last sha256 of each file. Thus `judge.py verify` finds an edit, a
removal or a move of a line. The chain has no external time stamp and no signature. A
rewrite of all the lines after a changed line, together with `head.json`, is not found.

**Write protection.** An empty file `.read_only` in a data folder makes each engine writer
refuse to write into its protected folders. Then `update.py` also refuses to run (see
[README.md](README.md#data-sources-and-product-rules)).

**Dashboard.** The dashboard is a set of static files. You open `web/index.html` from the
disk (`file://`). It needs no server, it loads no script from another site, and it sends
no data.

## Supported versions

Only the latest version on `main` gets security fixes. [CHANGELOG.md](CHANGELOG.md)
lists the versions.
