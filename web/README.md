# Factorlab dashboard (`web/`)

The dashboard is a static web page over the engine data. A Python exporter converts the
files in `engine/data/` to small JavaScript data files. The page draws all tables and charts in
the browser with its own SVG chart code. It has no build step, no package manager, no
CDN and no network access.

```
web/
  index.html                 the shell: loads the CSS, the core, the charts, data/meta.js and the app
  assets/app.css             the tokens (light and dark), the layout, the components
  assets/fl-core.js          FL.fmt, FL.theme, FL.colors, FL.cmd, FL.stale, FL.load, FL.ui, FL.pages
  assets/fl-charts.js        FL.charts: line, bars, scatter, heatmap, sparkline, smallMultiples
  assets/fl-app.js           the header, the snapshot line, the theme switch, the navigation, the router (FL.app)
  data/meta.js               written by the exporter: version, demo flag, data date, export time, data age, section states
  data/<section>.js          written by the exporter, one file for each section
  data/<section>.<part>.js   written by the exporter: a part of a section that a page loads when it needs it
  pages/<page>.js            one page for each route; each calls FL.pages.register(...)
```

`index.html` loads only `fl-core.js`, `fl-charts.js`, `data/meta.js` and `fl-app.js`. When
a route opens, the shell loads its page file and its data files (`FL.load`). Each URL of a
data or page file has the query `?v=<meta.generated_at>`.

## Open the dashboard

Open `web/index.html` in a browser. The page works from a local file (`file://`). You
can also serve the folder:

```bash
python3 -m http.server -b 127.0.0.1 -d web 8000      # then open http://127.0.0.1:8000/
```

`-b 127.0.0.1` keeps the server on this computer. Without it, other computers on your
network can open the dashboard and its data.

On a served folder, the shell loads `data/meta.js` again on each page load. Then the
version query makes the browser load the new data and page files after an export. If a
page still looks old after an update, reload it without the browser cache.

To publish the dashboard code, host the `web/` folder without `web/data/`. Keep the
license notice at the top of `assets/fl-core.js`, and put copies of `LICENSE` and
`THIRD_PARTY_NOTICES.md` next to it. The files in `web/data/` are built from
market data. The terms of the data sources apply to them, so do not publish them (see the
data terms in the [main README](../README.md#data-sources-and-product-rules)).

## Build the data

`./setup.sh` and `engine/update.py` build the data files in their last step
("dashboard"). To build only the data files:

```bash
cd engine
.venv/bin/python -B export_web.py                    # each section, then data/meta.js
.venv/bin/python -B export_web.py --only library,system  # some sections (meta.js is always written)
.venv/bin/python -B export_web.py --out <dir>        # write to another folder
```

The exporter only **reads** these files:

- `engine/data/` (or `FACTORLAB_DATA`), with the trial ledger `trials/`;
- `library/catalog/`;
- the files in `engine/logs/` (the names, sizes and line counts) and the steps of `update.log`;
- the file `VERSION`;
- for the System page: the plan of `update.py`, `engine/config.py`, `engine/ingest.py`,
  `setup.sh`, the documents, the glossary and the test files.

It only **writes** `web/data/` (or `--out`). It refuses an output path inside the engine data
folder. It prints a size report. It exits 1 in three cases:

- a section fails. A builder check can make it fail when two files disagree, for example a
  result file of a test and the trial ledger. The section "The pages" gives two exceptions;
- a section holds a private string (a home-folder path, a hidden folder in a home folder,
  an e-mail address, a share link);
- a file is too large: more than 1.5 MB for a section or for a part file, or 6 MB in
  total. A part file of the section `mine` has its own limit of 200 KB. It is not in the
  total. The page of one test loads one part file.

The exporter checks each file of a section before it writes one. A section that fails
keeps its old file. The exporter removes the data files of a section that the dashboard
does not have, with their part files. It prints the name of each file that it removes.
The output is the same for the same data, except `meta.generated_at`, `meta.data_age` and
the file times in `meta.sections`.

## The pages

The navigation has three groups. The first page is My factors: an empty or unknown route
opens it.

| Group | Page | Route | Content |
|---|---|---|---|
| Find | Factor library | `#/library` | The section "On Factorlab's data": the screen results, the OKX screen and the block "Binance data for the same 20 coins" (numbers only). The section "The whole library": the counts, the search (each definition with its formula and its check and register commands) and the families. |
| Find | Search tools | `#/research` | The numbers of the factor-lab registry, the AlphaForge formulas (when AlphaForge is on) and the command of the generator probe. |
| Test | My factors | `#/mine` | The tests of your factors (one row for each registration) and the expected number of false "useful" verdicts. Also the rule, the two shipped book specifications and the commands. |
| System | System | `#/system` | The last build (`engine/logs/update.log`), the update steps, the programs and the environment variables. Also the data age, inventory and coverage, the rules, the test suites, the documents and the glossary. |

The route `#/factor/<id>` opens the page of one test (`#/factor/FT-0001`). It is not in the
navigation. It reads the section `mine` and loads the part file of the test
(`data/mine.ft-0001.js`). The page shows the factor, the book specification, the historical
test, the forward test and the record. Each id in the table of My factors links to it. For a
library id (`CF…`), the page links to the library search, and the link opens the search with
that id. Another id gives "No registered test has this id."

The route `#/system/glossary-<N>` opens the System page with section N of the glossary open.
A name that is not the id of a glossary section opens no section.

The pages judge nothing. The data file gives each verdict word and each label of a verdict or a
state. It also gives the rule text and the expected number of false verdicts.
`webexport/mine.py` reads them from `trials.status_data()` and `rules.py`. The page of a test
shows BTC hold as context only.

judge.py writes the result files of a test again from the data on disk, and the ledger keeps the
frozen numbers. Thus a change of the data can make the two disagree. In two cases the export
does not fail:

- `historical.json` and the two historical Parquet files agree with each other, but not with the
  registration line. `judge.py verify --write-results` wrote them after a change of the data;
- `forward_weekly.parquet` differs from the frozen weeks only in the weeks that an alarm
  `book_week_changed` of the test names.

The row and the part file then get `results_differ` (one entry with a text for each case). The
two pages show the text in a warning. The verdict, the numbers of the rows and the forward weeks
come from the ledger. The historical charts and statistics come from the result files.

`#/<page>/<cardId>` opens a page and scrolls to the card with that id. The page of a test
draws its cards after its part file loads. Then it calls `ctx.cardReady()`, and the shell
scrolls to the card (`#/factor/FT-0001/forward`). The route registry is
`SECTIONS` in `assets/fl-app.js`; `webexport.SECTIONS` has the same order without the
route `factor`. The navigation order is `NAV_GROUPS` in `assets/fl-app.js`.
`test_export_web.py` checks the registry, the groups, the titles and the landing page.

### The search index and the formulas

The search index is the part file `data/library.index.js`. The page loads it when the
search comes near the screen or gets the focus. The formulas are the part files
`data/library.expr-0.js`, `data/library.expr-1.js`, ... (8,000 definitions each, in index
order). The page loads the part of each definition that the search shows. For a search with a
query, it loads all parts, so that the search also finds a formula. The tables of the two
screens show at most 500 rows and give the full counts. The search has the column "screen on
Factorlab's data". It shows the rIC t and q of each definition with a q at or below 0.10 of the
OKX and Binance screens. The page takes these values from the tables of the screens, not from
the index. The filter "Show" selects the definitions by the data of a test (OKX or Binance).

## The no-data state

A page never breaks when data is absent:

| Case | What the page shows |
|---|---|
| `data/<section>.js` does not exist | "No data": "`data/<section>.js` is not in this folder." and the two commands |
| `data/<section>.js` does not parse | "No data": "`data/<section>.js` does not parse (or has a script error)." and the two commands |
| The exporter found no input files for a section | "No data for this page", the reason (the absent file in the data folder), and the two commands |
| `pages/<page>.js` does not exist | "Page not available" |
| `data/meta.js` does not exist | "No snapshot loaded" and `./setup.sh` in the header |
| A page fails to render | an error box on that page only |

The states of the page My factors and of the page of one test:

| State | Section payload | Page |
|---|---|---|
| No trial ledger (also every demo build) | `mine`: `available: true`, `ledger.present: false`, `rows: []` | "No factor is registered." and the commands; the cards of the rule, the book specifications and the commands stay |
| Broken ledger | `mine`: `ledger.ok: false`, the problems, `rows: []` | a warning with the problems and the `verify` command, also on the page of each test |
| Registered, no complete forward week | a row with `forward.state` "waiting" | "Waiting for the first forward week. It ends on {date}." |
| A formula test that a pending gap or a PIT mismatch holds | a row with `forward.held` = `{sunday, reason}` | "Held at {Sunday}: {reason}." |
| A week with a final gap | `forward.path[].final_gaps` of the part, an alarm `data_gap` | "Missing data in this week: {coin} {field} from {day}. The week counts." |
| Result files absent | the part has `historical: null` | "The result files are absent. Run judge.py verify --write-results." |
| Result files that differ from the ledger (see "The pages") | the row and the part have `results_differ` | a warning "The result files differ from the ledger" with the text of each case |
| Withdrawn test | a row with `withdrawn: true`, not in N | "Withdrawn before the first forward week." The table shows the state "withdrawn" and the verdict "none (withdrawn)". The verdict filter counts the withdrawn tests apart. |

The two commands are `./setup.sh` (to add the data or new days) and
`cd engine && .venv/bin/python -B export_web.py` (to rebuild only the pages).

A section without input data gets the payload `{"available": false, "reason": "..."}`
(`webexport.common.no_data_payload`). The navigation marks such a page with a dot and the
screen-reader text "(not available)". A page names its own data file when the file is not in
the folder (`window.FL_MISSING`) or does not parse (`window.FL_BROKEN`).

## Demo data and the footer

When `meta.demo` is `true` (the data folder holds the synthetic demo data of
`engine/demo_data.py`), each page shows the callout "Synthetic demo data. Not market
data." after the page head (class `fl-demo-banner`). The header shows the same text as a
badge in place of the data-age badge. With `meta.demo` `false`, no page shows the callout.

The footer of each page gives only three items. They are the version (`meta.version`, from
the file `VERSION`), the license of the code (MIT) and the terms of the market data. Without
a version string in `meta.js`, the footer shows no version.

The page head shows a warning badge for a section in two cases. Its last export failed,
so its file is from an earlier export. Or its data date (`as_of`) is earlier than the date
in the header.

Inside a page, a part whose data is absent shows a short empty state. Examples:

| Part | Empty state |
|---|---|
| The factor lab without its registry | "The factor lab has not run." and its command |
| AlphaForge while it is off | "AlphaForge is off (ALPHAFORGE_ENABLED in engine/config.py)." |
| The Binance block without the Binance outputs | "Not computed on Binance's data" and the commands |
| The part file of a test absent | "Part file not available" and the export command |

## The snapshot line

The page is a **static snapshot**: it never updates itself. The header shows
`data_through`, the export time and the time since the export by the clock of the viewer.
`data_through` is the newest daily perp bar of any coin, stamped with its UTC open time.

The bar of day D closes at 00:00 UTC on day D + 1. When the UTC date of the viewer is after
D + 1, at least one closed bar is not in the data. Then the header starts with a
warning badge, for example "Data is 4 days behind: last bar 7 Oct 2026; 4 closed days are
missing. To add them: ./setup.sh" (`FL.stale.data`).

`meta.data_age` holds the export date, the data-through date, the days behind the export
date and the last bar of each coin. Days behind is the number of daily closes (00:00 UTC)
after the data close, up to the export date: 0 means current. The System page card "Data
age" shows these values.

## Data files

Each data file has exactly this form (one line of ASCII JSON):

```js
window.FL=window.FL||{};FL.data=FL.data||{};FL.data["system"]={...};
```

A part file has the key `"<section>.<part>"`, for example `FL.data["library.index"]`. The part
files: `library.index` (the search index), `library.expr-0`, `library.expr-1`, ... (the formulas)
and `mine.ft-0001`, `mine.ft-0002`, ... (one for each registered test, also a withdrawn one).

`data/meta.js` gives `FL.data.meta`:

```js
{ schema: 1, version: "MAJOR.MINOR.PATCH", demo: false,
  generated_at: "YYYY-MM-DDTHH:MM:SSZ", data_through: "YYYY-MM-DD",
  data_close_utc: "YYYY-MM-DDT00:00:00Z", timezone: "UTC", bar_stamp: "open",
  sections: { library: {present: true, bytes: 12345, as_of: "YYYY-MM-DD", available: true,
              state: "written", written_at: "YYYY-MM-DDTHH:MM:SSZ", parts: {}}, ... },
  total_bytes: 12345, budget: { section_max_bytes: 1500000, total_max_bytes: 6000000 },
  data_age: { export_date, data_through, days_behind, symbols, newest_bar, oldest_bar, behind_newest, coins } }
```

| Top-level field | Value |
|---|---|
| `version` | the version of Factorlab (the file `VERSION`, through `engine/version.py`) |
| `demo` | `true` when the data folder holds the demo marker `demo.json` |
| `data_through` | the newest daily perp bar of any coin (`data/raw/ohlcv/`) |

| Field of `sections[s]` | Value |
|---|---|
| `present`, `bytes` | the section file exists; its size |
| `as_of` | the top-level `as_of` date of the section payload, or `null` |
| `available` | `false` for the no-data payload or a file that does not parse |
| `state` | `written`, `no_data`, `failed`, `no_builder`, or `kept` (this run did not export the section). A `failed` state stays in the next runs that do not export the section. |
| `written_at` | the file time (UTC) |
| `parts` | `{part: bytes}` of the part files |

`total_bytes` includes the part files.

Rules of the section payloads:

- Dates are `"YYYY-MM-DD"` (UTC).
- Numbers are raw decimals: `0.0123` means 1.23%. The page formats them.
- NaN and infinity become `null`.
- Prices, equity and USD have 2 decimals; returns, weights and IC 6; ranks and
  percentiles 4. A value that a page rounds for display has full precision in the data
  file, so the page rounds it one time only.
- Series use column blocks: `{dates: [...], series: {book: [...], control: [...]}}`.

Two sections use part files: `library` (the search index and the formulas) and `mine` (the page
of each test).

## Add or change a section

### The builder (`engine/webexport/<section>.py`)

```python
from webexport.common import read_parquet, to_date, rnd, series_cols


def build() -> dict:
    d = read_parquet("labels/labels.parquet")  # relative to config.DATA, sorted by date
    return {
        "labels": series_cols(d, {"fwd_ret_7": "ret"}),
        "last": {"date": to_date(d["date"].iloc[-1]), "rows": len(d)},
    }
```

`webexport.common` has these read-only helpers:

| Helper | Function |
|---|---|
| `DATA`, `ENGINE_ROOT` | `config.DATA`, the engine folder |
| `data_path(rel)`, `exists(rel)` | a path under `DATA`; a test for it |
| `read_parquet(rel, columns=None, sort="date")` | Parquet to DataFrame, sorted by `date` when present |
| `read_json(rel)` | the JSON reader |
| `to_date(x)`, `to_dates(seq)` | timestamps to `"YYYY-MM-DD"` (UTC); NaT to `None` |
| `iso_utc(ts=None)` | `"YYYY-MM-DDTHH:MM:SSZ"` |
| `rnd(x, kind=6)` | round a value or a sequence; `kind` is an int or a name: `usd`, `price`, `equity` (2); `ret`, `pct`, `weight`, `ic`, `rate`, `decimal` (6); `rank`, `percentile`, `t`, `score`, `x`, `idx`, `num` (4) |
| `series_cols(df, {col: kind}, date_col="date", rename=None)` | `{"dates": [...], "series": {col: [...]}}` |
| `records(df, {col: kind}, dates=("date",))` | a list of row dicts |
| `clean_json(obj)` | NaN to None, numpy and pandas types to Python types |
| `NoData`, `no_data_payload(exc)` | raise `NoData` for absent input; the exporter writes the no-data payload |
| `with_parts(payload, {part: payload})` | split a section into part files `data/<section>.<part>.js`; the section payload gets `parts` (the sorted names). A part name matches `^[a-z0-9][a-z0-9_-]{0,63}$`. |

A builder must never call an engine function that writes (a register or a weekly log of
`judge.py`, ingest, a rebuild of the label table). Copy the read-only part in place of a call. A
consistency check of a builder raises `webexport.common.DataMismatch`; the exporter then
gives the section the state `failed`.

The section `library` has three modules:

- `library_catalog.py`: the catalog, the families, the search index and the formulas;
- `library_engine.py`: the screen on OKX data;
- `library_binance.py`: the screen on Binance data.

`webexport/library.py` builds the section from them, and the other modules import the
section only as `webexport.library`.

### The page (`web/pages/<page>.js`)

```js
(function () {
  "use strict";
  var FL = window.FL;
  FL.pages.register({
    id: "example",               // the route #/example; the data section is the same id by default
    title: "Example",            // the app draws it as the page <h1>
    subtitle: "...",
    icon: "research",
    requiresData: true,          // default; without the section data the app shows the no-data state
    render: function (root, ctx) {
      var d = ctx.section;       // === FL.data.example
      var card = FL.ui.card({ id: "chart", level: 2, title: "A chart" }, root);
      FL.charts.line(card.body, { x: d.dates, series: [/* ... */] });
    }
  });
})();
```

Options of `register`: `id`, `title`, `subtitle`, `icon`, `requiresData`, `section` (the
data section that `ctx.section` reads; default the route id) and `data` (more sections to
load first, for example `["library"]`).

The app calls `render(root, ctx)` on each visit with a new empty `root`. `ctx` has:

| Field | Value |
|---|---|
| `id`, `data`, `section`, `sectionId`, `meta` | the page id, `FL.data`, the section payload and its id, `FL.data.meta` |
| `colors`, `fmt`, `ui`, `charts`, `params` | the APIs below; `params` is the `URLSearchParams` of the location |
| `signal` | an `AbortSignal`; the shell aborts it before the next route renders |
| `listen(target, type, fn, opts)` | add a listener that the shell removes at the next route; returns `off()` |
| `onTheme(fn)` | run `fn` after each theme change, until the next route |
| `defer` | `FL.ui.defer` |
| `loadPart(name)`, `parts()` | load a part file of the section (a promise of its payload); the part names in `meta` |
| `cardReady()` | a page that draws its cards after a part file loads calls it. Then the shell scrolls to the card of the route, if the reader did not scroll first |
| `setTitle(title, subtitle)`, `addBadge(node)` | change the page head; add a badge to it |
| `nav` | `{current, card, param, badParam, go(page, card, param), href(page, card, param), factorHref(id, card)}` |

When `render` throws, the app shows an error box for that page only. A new route needs an
entry in `SECTIONS` and in `NAV_GROUPS` of `assets/fl-app.js`. A new data section also
needs an entry in `webexport.SECTIONS`. `index.html` does not change.

## API reference

### FL.fmt: format (each function returns a string; `null` and NaN give `"n/a"`)

Negative numbers use the typographic minus `FL.fmt.MINUS` (U+2212). A value that rounds to
zero has no sign. `pct`, `pp` and `bps` take decimals.

| Call | Example |
|---|---|
| `pct(v, dp=2, {signed=true})` | `pct(0.0123)` gives `+1.23%` |
| `pp(v, dp=2, {signed=true})` | `pp(0.0074)` gives `+0.74pp` |
| `usd(v, dp=2, {signed=false})` | `usd(1234.5678)` gives `$1,234.57`; `usd(-39.49)` gives `−$39.49` |
| `num(v, dp=2, {signed=false})` | `num(-4.6213)` gives `−4.62` |
| `int(v, {signed})` | `int(1140)` gives `1,140` |
| `compact(v, dp=1, {signed, prefix})` | `compact(12900)` gives `12.9K` |
| `idx(v, dp=1)` | `idx(85.1449)` gives `85.1` |
| `x(v, dp=2)` | `x(0.75)` gives `0.75x` |
| `mult(v)` | `mult(0.5)` gives `×0.5` |
| `bps(v, dp=1, {signed})` | `bps(0.0005)` gives `5.0 bp` |
| `date(s, style="long")` | `1 Oct 2026`; styles `short`, `axis`, `iso`, `month`, `weekday` |
| `datetime(iso)` | `2026-10-02 05:53 UTC` |
| `age(ms)`, `bytes(n)`, `text(v)` | `45 min`; `12.1 KB`; `String(v)` or `n/a` |
| `parseDate(s)`, `isoDate(ms)` | `"YYYY-MM-DD"` to UTC epoch ms, and back |
| `NA`, `isNumericKind(spec)`, `kindOf(spec)` | the `"n/a"` text; true for a numeric format; the kind of a spec |
| `get(spec)` | a format function from a spec: a function, `"kind"`, `"kind:dp"`, `"kind:dp:signed"`, `"kind::unsigned"` or `{kind, dp, signed}` |

Each option named `format` below accepts a `get()` spec.

### FL.colors: the color follows the entity

Each color is a CSS value (`"var(--c2)"`), so it changes with the theme.

| Call | Returns |
|---|---|
| `entity(id, context?)` | `"var(--cN)"`, or `null` for an unknown id |
| `entityInfo(id, context?)` | `{color, cssVar, slot, dashed, label}` |
| `label(id)` | the display label (`"btc"` gives `"BTC hold"`) |
| `isBenchmark(id)` | true for `btc`, `btc_hold`, `BTC`, `benchmark`, `bench` |
| `slot(n)` | `"var(--cN)"` for n = 1 to 8 |
| `status` | `{good, warning, serious, critical, info, advisory}` |
| `seq(t)`, `div(t)`, `scale({type, domain})` | sequential (blue) and diverging (red to blue) scales |
| `entityVar(id, context?)`, `mix(a, b, t)` | the CSS variable name of an entity; a color between `a` and `b` |
| `resolve(cssColor)`, `inkOn(fill)`, `theme()` | the current hex; black or white ink on a fill; `"light"` or `"dark"` |

| Entity ids | Color |
|---|---|
| `factor_book` (the book of a factor test) | slot 1, blue |
| `control` (its control book) | slot 2, orange |
| `excess` (the net excess return over the control) | slot 3, aqua |
| `btc`, `btc_hold`, `BTC`, `benchmark`, `bench` | muted ink, dashed |

The colors of a chart must stay apart for a viewer with a color-vision deficiency. The procedure:

1. Take each pair of colors that share a chart, in both themes.
2. Compute the OKLab distance (x 100) of the pair for normal vision.
3. Compute it again for a simulation of protanopia, deuteranopia and tritanopia (the
   physiologically based simulation model of 2009, severity 1).
4. Keep the smallest distance.

The smallest distances of the entity colors:

| Pair (they share a chart) | Light | Dark |
|---|---|---|
| `factor_book` and `control` | 24.7 | 26.8 |
| `factor_book` and `btc` (also dashed) | 11.2 | 11.5 |
| `control` and `btc` (also dashed) | 9.8 | 11.3 |

`excess` is on a chart of its own: the weekly net excess, the bounds of the confidence sequence
and the Sharpe ratio of the windows. The bounds use the ink token `--ink-3`.
`test_export_mine.py` computes the table again.

Text never has a series color. Values and labels use the ink tokens `--ink`, `--ink-2` and
`--ink-3` with a colored key. `--muted` is for marks only (lines, rules, icons), never for
text. Controls use the border token `--control-border`.

### FL.theme

`FL.theme.get()` gives `"auto"`, `"light"` or `"dark"`. `FL.theme.set(mode)` sets it
(saved in localStorage when the browser allows it). `FL.theme.effective()` gives
`"light"` or `"dark"`. Each change sends the window event `"fl:themechange"`, and the
charts draw again.

### FL.cmd, FL.stale and FL.load

| Call | Function |
|---|---|
| `FL.cmd.setup`, `FL.cmd.export` | `"./setup.sh"` and `"cd engine && .venv/bin/python -B export_web.py"` |
| `FL.stale.data(meta, nowMs?)` | `{through, today, ageDays, missingDays, stale, text}`, or `null` without a data date; `stale` when at least one closed bar is missing |
| `FL.stale.section(meta, id)` | `{known, present, failed, stale, as_of, reason}` of one section |
| `FL.load.data(section)`, `FL.load.page(id)` | load `data/<section>.js` or `pages/<id>.js`; a promise of `{path, state}` with the state `ok`, `missing` or `broken` |
| `FL.load.part(section, name)`, `FL.load.parts(section)` | load one part file; the part names in `meta` |
| `FL.load.state(path)`, `FL.load.problems()` | the state of one file; the files that are missing or broken |

### FL.app

`FL.app.go(page, card, param)` and `FL.app.href(page, card, param)` open and link routes.
`FL.app.factorHref(id, card)` gives `"#/factor/<id>"`. `FL.app.signal()` gives the signal of
the current render. `FL.app.SECTIONS`, `FL.app.NAV_GROUPS` and `FL.app.DEFAULT` (`"mine"`)
are the route registry, the navigation groups and the landing page.

### FL.ui: components

Data text always enters the page through `textContent`. Most builders take
`(container, options)`, append to the container and return the element or a handle.

| Component | Function |
|---|---|
| `el(tag, opts, children)`, `svg(tag, attrs, children)` | safe DOM and SVG builders |
| `clear`, `append`, `uid`, `icon(name)` | DOM helpers and the icon set |
| `card({id, title, subtitle, actions, footer, span, level}, container)` | a card; `id` makes it a link target; `level` 2 to 6 sets the heading (default `h3`) |
| `section(container, {id, title, subtitle})` | an `h2` block |
| `toc(container, [[id, label]], {page})` | an "On this page" list of links to cards |
| `grid(container, 2\|3\|"kpis")` | a two-column, three-column or tile grid; add the class `is-stretch` for equal heights |
| `kpis(container, items)` | stat tiles with value, delta, note, badge and sparkline |
| `badge(status, label)` | a status mark with a symbol and a word |
| `callout(kind, title, text)` | a message box |
| `empty(container, message, {title, command, commands})` | an empty state; `commands: FL.ui.remedies()` gives the two standard commands |
| `table(container, {columns, rows, sortable, defaultSort, pageSize, stack, caption, ariaLabel, onSort, ...})` | a sortable table (see below) |
| `tabs`, `segmented`, `details` | tabs, a segmented switch, a collapsible block |
| `dl`, `meter`, `key`, `legend`, `note`, `code` | a facts list, a progress track, identity keys, a note, code |
| `defer(container, {id, minHeight}, fn)` | draw a part of the page after the first screen; a link to the card `id` draws it at once |
| `onTheme(fn, signal)`, `scrollCue(node)` | a theme listener that stops with the signal; edge shadows on a box that scrolls sideways |

`table` options: `pageSize: N` shows N rows and the buttons "Show N more", "Show all" and
"Show fewer" (a sort orders all rows first); `phonePageSize: N` is the page size below
600 px. `stack: true` shows each row as a label/value card below 600 px. The card has no
header row and no inner scroll box, and its ARIA roles keep the table structure. `caption` is a
caption for screen readers; a column with `rowHeader: true` gives row headers. The handle has `setRows(rows)`, `sort(key, dir)`,
`showAll()`, `shown()` and `count()`.

The classes `fl-button`, `fl-select` and `fl-input` give the shared control style.

### FL.pages

`FL.pages.register(page)` adds a page (see above). `FL.pages.get(id)` gives a registered
page or `null`. `FL.pages.list()` gives the page ids in the order of registration.

### FL.charts: charts

| Chart | Function |
|---|---|
| `line(container, opts)` | time or linear lines; `series: [{id, name, entity\|color, values}]`, `y: {format, label, log}`, `refLines` (`labelAt: "start"` puts the label at the left end), `bands`, `vlines`, `markers`, `downsample: "week"\|"month"` |
| `bars(container, opts)` | diverging stacks or groups; vertical or horizontal; a Net row in the tooltip |
| `scatter(container, opts)` | x/y points with labels and the nearest-point tooltip |
| `heatmap(container, opts)` | rows x columns with a sequential or diverging scale; `colLabels: "all"\|"thin"`, `timeAxis`; a wide grid scrolls, and a grid of dates opens at the newest column |
| `sparkline(container, opts)` | a small line in a tile, without axes |
| `smallMultiples(container, specs, opts)` | a grid of charts, optionally with one y-domain |
| `util` | scale and tick helpers: `textWidth`, `truncate`, `niceTicks`, `logTicks`, `timeTicks`, `tickFormatter`, `linear`, `logScale`, `barPath` |

Each chart except `sparkline` has these properties:

- a legend for two or more series;
- a **Table** switch that shows the same numbers as a table;
- keyboard focus (arrow keys move the crosshair, and a screen reader reads each step) and
  tooltips;
- a new draw after a resize or a theme change.

Each chart returns `{root, update(opts), destroy(), showTable(bool), render()}`. A chart in a hidden container
draws itself when it becomes visible. `FL.charts.refresh()` draws all charts on the page again;
`FL.charts.prune()` drops the charts that left the page.

## Rules for page code

- **ES2019 classic scripts only.** No `?.`, no `??`, no `import` or `export`, no
  `type="module"` (the browser blocks modules on `file://`). Wrap each file in an IIFE with
  `"use strict"`.
- **Data text enters only through textContent.** Never put data into `innerHTML`.
- **Decimals in, format in JS.** Tests that compare strings use `FL.fmt.MINUS` for a
  negative number.
- **UTC everywhere.** Dates are `"YYYY-MM-DD"` strings. A daily bar has the stamp of its
  open time. Say "last bar", not "today", for the newest bar of the data.
- **The color follows the entity.** Give `entity`, not a color, for the factor book, the control
  book, the net excess and BTC. Status colors are for status only, always with a symbol and a word.
- **Label the funding sign** on each funding axis or column. The market funding rate:
  + = longs pay shorts.
- **One y-scale for each chart.** Use `smallMultiples` or an index to 100 for two scales.
- `render` runs on each visit. Keep no reference to old DOM. Do not call
  `window.addEventListener` in a page: use `ctx.listen`, `ctx.onTheme` or
  `{signal: ctx.signal}`, so that the listener stops at the next route.
- Pages do not edit `app.css`. For a local style, use inline `style`, or one static
  `<style>` element with classes that start with the section name (`.library-...`). Text
  uses `--ink-3`, not `--muted`.

## Tests

The `test_export_*.py` suites in `engine/` build each section from the data and compute
each displayed number again:

| Suite | Part of the dashboard |
|---|---|
| `test_export_web.py` | the exporter, `meta.js` and the page shell |
| `test_export_library.py` | the page Factor library |
| `test_export_research.py` | the page Search tools |
| `test_export_mine.py` | the page My factors and the page of one test |
| `test_export_system.py` | the page System |

When `node` is on the `PATH`, or when `FACTORLAB_NODE=<path>` or `NODE=<path>` is set, the
suites also render each page in a minimal DOM in node.
`test_export_web.py` also runs the shell in node: the routes, the navigation groups, the
loader, the part files, the stale badges and the listener counts. Run them with:

```bash
engine/.venv/bin/python -B engine/run_tests.py --only test_export_web --only test_export_library
```

`test_export_mine.py` builds its own trial ledger and result files in a temporary data folder.
It computes each number of the page My factors and of the page of a test again.
`test_export_library.py`, `test_export_research.py` and `test_export_system.py` compare the
written files in `web/data/` with a new build. Set `FACTORLAB_WEB_DATA=<folder>` to check
the data files of `test_export_library.py` in another folder, for example an export with
`--out`.
