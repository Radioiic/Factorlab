# Roadmap

This roadmap lists the functions that Factorlab will get, in the order of the work. It has
no dates. [CHANGELOG.md](CHANGELOG.md) lists the functions of each released version, and
[README.md](README.md) tells how to use them. [engine/GLOSSARY.md](engine/GLOSSARY.md)
defines the research terms. This file defines each new term where it uses it.

The items below add to the two parts of Factorlab:
[Find a new factor](README.md#find-a-new-factor) and [Test a factor](README.md#test-a-factor).
Factorlab stays a research tool. No item of this roadmap places orders or uses exchange
keys.

## Next

These three steps come first, in this order.

### 1. Coins and venue from data files

- Factorlab will read the coins, the venue and the data source from data files. Then a
  different set of coins or a different venue will need no change to the code.
- The 20 OKX perpetual futures will stay the example set.

### 2. Calibrated p-values and the trial count of each search tool

- The library screen will use calibrated p-values. Factorlab will permute the coins many
  times to measure the chance level of each statistic. The Benjamini-Hochberg q-values will
  use these p-values.
- One append-only record will count the trials of each search tool: the library screen,
  the factor lab, AlphaForge and Factor Finder. The record will give the number of trials
  of each tool.

### 3. Factor Finder: the feasibility check

- A simulation will calculate how many true and false factors a test on the sealed I zone
  can find with 20, 50 and 100 coins. The I zone is sealed for the library screen and the
  generator: they read no label of it.
- With the result, the maintainer will select one route for Factor Finder:
  - the 20 coins;
  - a wider research universe (see [Optional](#optional));
  - forward tests only, with no test on the I zone.

## After the feasibility check

The Factor Finder functions come in this order.

1. **Input checks.** Before each run, Factor Finder will check its input data. The checks
   will cover the start date of each coin, the data quality, the data format and the zone
   limits. When the route has the wider research universe, the wider universe comes after
   this step.
2. **Trials and formula families.** Factor Finder will record each of its trials in the
   record of [step 2](#2-calibrated-p-values-and-the-trial-count-of-each-search-tool). A
   formula family is a group of closely related formulas. Equal formulas will count as one
   trial.
3. **Gates and the sealed test.** Factor Finder will compute a fixed chain of gate numbers
   for each candidate. A candidate within the limits of the gates will get a test on the
   sealed I zone. The test will give numbers, not a verdict. After the feasibility check,
   the maintainer will set the number of these tests and their error control. Data with no
   signal and data with a planted signal will measure the error rate and the power of the
   tests. When the route is forward tests only, this step has no test on the I zone.
4. **A search for new formulas.** The generator will search for new formulas and score them
   on the generator zone G only.
5. **Candidates for your test.** Factor Finder will list the numbers of each candidate and
   its register command. It will run no book and give no verdict. You decide which
   candidate to register for a test.
6. **Commands and reports.** Commands will run one search cycle, run the checks with data
   that has no signal, write a report and check the records. When a check fails, an alert
   will stop new candidates until you correct the cause.
7. **The first real searches.** Factorlab will freeze the evaluation settings and the
   generator settings. Then the real searches will start. A search that finds no candidate
   is a possible and correct result.

## Optional

- A wider research universe of about 100 coins, with coins that are no longer listed. The
  maintainer will add it only when the feasibility check shows that 20 coins are not
  sufficient.
- The data before 2022 as a sealed test set. It can come only together with the wider
  research universe, and only for formulas from the generator search.

## Not in the roadmap

- Live trading, the placement of orders and exchange keys with trade permissions.
