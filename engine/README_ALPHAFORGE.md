# AlphaForge: the optional expression miner

`alphaforge/` is a genetic-programming expression miner. The author of Factorlab wrote
AlphaForge, and the MIT License ([LICENSE](../LICENSE)) covers it. It is not related to the
paper "AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha Factors"
([arXiv:2406.18394](https://arxiv.org/abs/2406.18394)) or to its code.
`alphaforge_bridge.py` connects it to Factorlab's point-in-time data. The bridge keeps a
**zoo** of mined formulas. Each formula has its own selection date, a fixed
forward window of 52 weeks and its own forward record. The miner gives numbers only.

The miner is **off by default**. Run each command from `engine/` with
`.venv/bin/python -B`.

When the miner is on, the Search tools page of the dashboard shows the formulas and their
numbers. When it is off, the page says so.

## The switch

Set the switch at the end of `engine/config.py`:

| Setting | Default | Effect |
|---|---|---|
| `ALPHAFORGE_ENABLED` | `False` | The switch. Off: `--cycle`, `--mine` and `--refresh` exit 3. On: `update.py` adds the step "AlphaForge cycle" before the dashboard step. |
| `ALPHAFORGE` | a dict | The miner settings (below). |

An edit of `ALPHAFORGE_ENABLED` is not a code change of the trial ledger. After an edit of
`engine/config.py`, the next update computes the library values again.

## Commands

```bash
.venv/bin/python -B alphaforge_bridge.py --report     # print the last report; run nothing
.venv/bin/python -B alphaforge_bridge.py --cycle      # score the active formulas, mine when due, write
.venv/bin/python -B alphaforge_bridge.py --mine       # a cycle that mines now when a place is free
.venv/bin/python -B alphaforge_bridge.py --refresh    # build the table again from the zoo only
.venv/bin/python -B alphaforge_bridge.py --cycle --today 2026-10-04 --no-write   # a dry run
```

| Option | Effect |
|---|---|
| `--cycle` | Score the forward weeks of each active formula, apply the window rule, mine when a run is due and a place is free, and write. |
| `--mine` | A cycle that mines now when a place is free. The window rule applies as in `--cycle`. The zoo keeps the record of each formula. |
| `--refresh` | Build the table again from the zoo. No mining, no forward scores. |
| `--report` | Print the last report. Read-only. |
| `--force` | Run while `ALPHAFORGE_ENABLED` is `False`. |
| `--no-write` | Compute and print; write no file. |
| `--today YYYY-MM-DD` | The cycle date (default: the UTC date of the clock). |
| `--pop N`, `--gens N`, `--seed N` | Population, generations and seed of this run. |
| `--max-factors N` | The largest number of active formulas. |
| `--verbose` | More log output. |

A plain `--cycle` writes. On a due day with a free place it mines. For a check, use
`--cycle --no-write` with `--today`.

## The forward window and the active list

| Item | Rule |
|---|---|
| Forward window | 52 Sundays for each formula, from the first Sunday after its selection date. |
| Forward week | A Sunday of the window whose 7-day label is complete: the Sunday + 7 days is not after the oldest last bar of the live coins. A coin is live when its last bar is at most 7 days before the newest last bar. A Sunday on which no coin has a bar is a forward week with no IC. |
| Forward number | The Spearman IC of the formula with `fwd_ret_7` on each forward week; the mean, t and p of these ICs. |
| Active list | A formula is active until it has 52 forward weeks. Then it leaves the active list on that cycle date. Its record stays in the file and does not change again. |
| Capacity | At most 6 active formulas (`max_factors`). |
| Mining | A cycle mines only when a place is free and the last mining run is at least 7 days old (`mine_every_days`). It adds at most as many formulas as there are free places. |

A new formula does not change the window or the record of another formula. The miner did
not see the forward weeks of a formula. The in-sample numbers of a formula are not
out-of-sample numbers.

## What a cycle does

1. **Forward scores.** It computes the forward record of each active formula again. A
   late bar thus corrects itself until the record is final.
2. **The window rule.** A formula with 52 forward weeks leaves the active list.
3. **Mining.** When a place is free and a run is due, it mines. Each run uses its own seed:
   `seed + ISO year x 100 + ISO week`.
4. **Write.** It writes the table, the zoo file, the report and one line of the cycle log.

## How the miner works

| Item | Setting |
|---|---|
| Input fields | scale-free series only: `returns`, `range`, `gap`, `hl_pos`, `log_amount`, `amount_ratio_30`, `funding` |
| Label | `fwd_ret_7` of the label table: close(D+7) / close(D) - 1, stamped on D |
| Split | 60% train, 20% valid, 20% test, in date order |
| Fitness | the absolute Spearman ICIR minus turnover, redundancy and size penalties, on train dates only |
| Hall of fame | 40 expressions |
| VALID rule | rank ICIR >= 0.5 (annualized with 52 periods) and Sunday-IC t >= 1.5 on the valid dates |
| Incumbents | your registered tests that are not withdrawn, the active AlphaForge formulas and the three risk proxies (`factorfinder/incumbents.py`) |
| Admission | on the dates up to the end of the valid block: absolute correlation <= 0.5 with each incumbent; after train: a mean Sunday absolute correlation below 0.5 |
| De-correlation | greedy, among the formulas that meet the rules above |
| Library number | the largest absolute correlation of each added formula with a library definition, on the G Sundays (a separate number) |
| Weekly numbers | IC, t, p, q_bh, p_bonferroni, persistence and coverage on the Sundays after train only |
| Hypothesis count | two expressions with a panel correlation above 0.999 are one hypothesis; p_bonferroni multiplies p by the larger of the distinct hypotheses and the expressions searched |
| Expressions searched | the sum of the expressions that each mining run evaluated, also of a run with no valid expression |

Raw volume and close levels are not inputs. Their cross-sectional ranks only separate
large coins from small coins.

The miner selects on the whole history up to the selection date. Thus the train, valid and
test blocks limit overfit, but they give no out-of-sample numbers. Only the forward weeks
are out-of-sample.

## Settings

The `ALPHAFORGE` dict in `config.py`:

| Key | Default | Description |
|---|---:|---|
| `population` | 80 | GP population |
| `generations` | 10 | GP generations |
| `max_depth` | 3 | the largest tree depth |
| `max_factors` | 6 | the largest number of active formulas |
| `min_valid_icir_ann` | 0.5 | the VALID rule on rank ICIR |
| `mine_every_days` | 7 | the shortest time between mining runs |
| `forward_window_weeks` | 52 | the forward window of each formula |

`alphaforge_bridge.DEFAULTS` holds the other settings, for example `min_valid_t` 1.5,
`max_corr` 0.5 and the penalty weights.

The expression parser refuses a text that nests deeper than 100 levels, or that gives a
tree deeper than 100 (`alphaforge.expr.parser.MAX_DEPTH`; the error is `ExprSyntaxError`).

## Outputs

All files are in `data/research/`:

| File | Content |
|---|---|
| `alphaforge_library.json` | the zoo (format 3). For each formula: the expression, the selection date, the mining run, the window, the admission numbers, the forward record, `active` and `left_on`. Also the mining runs, the counts of hypotheses and expressions searched, and the settings. |
| `alphaforge_library_<run_id>.json` | a copy of the zoo file for each `run_id`, written the first time a file with that `run_id` is replaced. The name has the `run_id` without `:`. A file with no `run_id` gives `alphaforge_library_prev.json`. |
| `alphaforge_cycles.jsonl` | the append-only cycle log: date, data date, mined or not, added formulas, formulas that left, the active count and the free places after the cycle, expressions evaluated, seed |
| `alphaforge_registry.parquet` | the append-only trial registry: each valid hall-of-fame expression of each mining run |
| `alphaforge_latest.txt` | the last report |
| `alphaforge_daily.parquet` | the table: one column `af_NN` for each formula of the zoo, active or not |

Factorlab reads only a zoo file of format 3. A file of another format, or a file with no
format, stops each command that reads it. `alphaforge_bridge.py` and `factor_lab.py` stop
with exit status 1 before they compute. Each one writes one line that names the file and
its format. The export of the Search tools page fails.

## Tests

```bash
engine/.venv/bin/python -B engine/run_tests.py --only test_alphaforge_bridge --only test_incumbents --only alphaforge
```

- `test_alphaforge_bridge.py` checks these items:
  - the switch;
  - the label against the label table;
  - the forward window of 52 weeks, also with a day without a bar, and the complete-label rule;
  - the capacity of 6 active formulas, and mining only for free places;
  - the refusal of a zoo file of another format;
  - the name of the zoo backup, one for each `run_id`;
  - the counts after a mining run with no valid expression;
  - the incumbents from `factorfinder/incumbents.py`;
  - the point-in-time property of each input and of the table;
  - three cycles in a temporary folder that write only `research/`;
  - numbers only in each output.
- `test_incumbents.py` checks the incumbent provider.
- `alphaforge` runs the suite `engine/alphaforge/tests` with pytest.
