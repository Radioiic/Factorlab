# Glossary: the research terms

This glossary defines the terms of the factor research in Factorlab. Section 1 gives the
research terms and the terms of a factor test, section 2 the metrics and section 3 the
market terms.

---

## 1. Research terms

**Factor.** One number for each coin on each date, for example the 30-day volatility of
the coin. A test book ranks the coins by the factor each Sunday.

**Label.** The value that a factor tries to predict: `fwd_ret_7`, the return of a coin over
the 7 days after a Sunday. A label is never a factor input.

**Label table.** The file `data/labels/labels.parquet` (`labels.py`). It holds `fwd_ret_1`
and `fwd_ret_7` of each coin and day, from the OKX perpetual closes only.

**Point in time.** A value dated D uses only data known at the end of day D. For a formula,
Factorlab computes the value of each Sunday on the data up to that Sunday (the Sunday cut).
This value must be equal to the value of the full computation. Then the formula is proven
point in time. `test_labels.py`, `test_userfactor.py` and `test_backtest.py` check this
property.

**Values file.** A CSV or Parquet file with the columns `date`, `symbol` and `value`: the
factor values that you give. Factorlab cannot prove a values file point in time.

**Log time.** The time of the `judge.py log-values` call that logged a value. A forward
value of a values file counts only from the first daily close (00:00 UTC) after its log
time.

**Book specification.** A TOML file (format 1) that gives the book of a test. It gives the
direction, the sign, the long and short fractions, the exit band, the costs and the control
book. Its identity is the sha256 of its normalized content. `engine/bookspecs/` has two
files.

**Control book.** The book that a test compares with the book of the factor. The control
`zero` has no position. The control `equal_weight` has the eligible coins with equal
weights, with the same weekly rebalance and the same costs.

**Net excess return.** The return of the book of the factor minus the return of the control
book, after fees, slippage and funding.

**Weekly net excess return.** The net excess return of one week: from the Monday trade
after a signal Sunday to the next Monday mark. The forward test reads only this series.

**Historical test.** The test of a book on the history before the registration
(`judge.py check`). It can give only "useless" or "unproven (historical pass)". It gives
"useless" when the total return or the Sharpe ratio of the net excess return is 0 or less.
It also gives "useless" when fewer than half of the rolling 1-year windows have a positive
Sharpe ratio.

**Rolling 1-year window.** A span of 365 days of the history. The first window starts on
the first day of the history, and each next window starts 91 days later. The window
condition of the historical test applies only when the history has a window.

**Registration.** The line in the trial ledger that fixes a test (`judge.py register`).
The line holds the factor, the book specification, the historical result, the sigma floor
and the first forward Sunday. After the registration, only new weeks count.

**Trial ledger.** The folder `data/trials/`: the append-only, hash-chained record of your
tests. Each line holds the sha256 of the line before it. An edited, removed or moved line
stops each command of `judge.py`.

**First forward Sunday.** The first Sunday after the newest data and after the day of the
registration. It is the signal Sunday of week 1 of the forward test.

**Forward test.** The test of a registered book on the weeks after the registration. It
compares the weekly net excess return with 0. A confidence sequence decides from week 26.
The maximum length is 156 weeks.

**Complete week.** A forward week that counts. The OKX data must reach the last day of the
week. On each day of the week, each coin has its bar and its funding value, or the missing bar
or value is a final gap. A coin with no value of a field in the 14 days to the end of the
week has no gap in that field. For a formula test, the week also needs the logged values of
its signal Sunday.

**Week with a factor position.** A week in which the book has its own factor position. At a
Sunday with fewer than 5 eligible coins with a value, the book has the control portfolio. It
also has the control portfolio at a Sunday without a used value of a values file. Such a
week is not a week with a factor position.

**Final gap.** A missing bar or value that is at least 7 days old (the 7-day rule for
missing data). A missing value holds a week for up to 7 days. After 7 days the gap is
final: the week counts, and a `data_gap` alarm names the missing data.

**Long-run SD.** The standard deviation of the weekly excess returns with a correction for
serial correlation (Bartlett weights, 4 lags).

**Sigma floor.** The smallest sigma of the confidence sequence of a test. It comes from the
weekly excess returns of the weeks with a factor position in the history. The history must
have at least 52 such weeks. Else the sigma floor comes from the book specification
(`forward.sigma_floor_fallback`).

**Confidence sequence.** A lower and an upper bound on the mean weekly net excess return
after each week. The bounds hold at all weeks at the same time, with the probability that
alpha (0.10) sets. Its sigma is the largest of the sample SD, the long-run SD and the sigma
floor. Thus the test can examine its result after each week.

**Verdict.** The word that a test gives: "useful", "useless" or "unproven". Only the forward
test can give "useful": from week 26, when the lower bound is more than 0. The forward test
gives "useless" from week 26 when the upper bound is less than 0. It also gives "useless" at
week 156 when the test is not "useful". Before a forward decision, the verdict is "unproven
(historical pass)" or "useless (historical)". The first forward decision is final.

**False-pass rate.** The probability that a factor with a true weekly mean excess return of
0 gets "useful". Its design value is 10 % or less for each test. Strong serial
correlation, a large increase of volatility after the registration or rare large losses can
make it larger.

**Expected number of false "useful" verdicts.** At most 0.1 x N, with N the number of
registered tests that are not withdrawn. This bound holds when each test meets the design
value of the false-pass rate (10 % or less).

**Duplicate test.** A test with the same book and the same signed factor values as a
registered test. The same values are the same formula and sign, the same values of a file,
or a mean within-Sunday Spearman correlation above 0.999. `judge.py register` refuses it.

**Related test.** A registered test with an absolute correlation of 0.85 or more with
another test, with any book. `judge.py status` shows the related tests.

**Pre-registered.** Recorded before the data that judges it exists.

**Benjamini-Hochberg q (BH q).** The q-value after the correction for many tests. With
q <= 0.05, the expected share of false results among the kept results is at most about 5%.

**Bonferroni.** A stricter correction: the p-value times the number of tests.

**Distinct signal.** A cluster of closely related factors (absolute pooled rank
correlation of 0.7 or more). The cluster counts once.

**Persistence.** The mean week-over-week rank correlation of a factor. High persistence
means low turnover.

**Coverage.** The share of the universe with a value.

**Embargo.** Sundays between two zones that no screen uses. They prevent off-by-one errors.

**G zone (generator zone).** The Sundays that the library screen and the generator can
read. G_fit (2022-04-03 to 2023-07-23, 69 Sundays) is for the fit. One embargo Sunday
follows. G_eval (2023-08-06 to 2024-06-09, 45 Sundays) is for the check. When a selection
uses only G_fit, its check on G_eval uses weeks that the selection did not see.

**I zone (inference zone).** The Sundays 2024-06-30 to 2026-09-20 (117), after a
two-Sunday embargo. The library screen and the generator probe read no label there. The
historical test of your own factor reads all history.

**New family and cross-check.** On the Binance panel, the *new family* is the library
definitions that the OKX screen could not screen. They get their own Benjamini-Hochberg
correction. The *cross-check* compares the Binance statistics of the definitions that the
OKX screen did screen, with no q-value and no list at a q level. The two screens share the
labels, the regressors and the coins. Thus their agreement is not independent evidence.

**Terminal.** An input series of a generated formula: a leaf of its expression tree. The
generator probe has 21 library series and 4 Factorlab-only series
([README_FINDER.md](README_FINDER.md)).

**Novelty index.** The G-Sunday ranks of each known factor: the library definitions, your
registered tests, the AlphaForge formulas and the three rIC regressors. The probe gives the
largest absolute correlation of a formula with any member.

**Incumbent.** A factor that the factor lab and the AlphaForge miner compare with a new
formula. The incumbents are your registered tests that are not withdrawn, the active
AlphaForge formulas and the three risk proxies. The largest correlation with the library is
a separate number.

**Generator probe.** `finder.py --generator-probe`. It draws random formulas and gives the
bands of their largest absolute correlation with the novelty index. It reads no label, so
it measures room for new formulas, not predictive power.

**AlphaForge zoo.** The formulas of the optional AlphaForge miner. Each formula has its own
selection date and its own forward record. At most 6 formulas are active.

**Forward window.** The 52 Sundays after the selection date of an AlphaForge formula. The
miner records the weekly IC of each Sunday of the window with a complete 7-day label. After
52 forward weeks, the formula leaves the active list, and its record stays.

**Synthetic demo data.** The data set of `update.py --demo` (`demo_data.py`). A random
generator makes each value, and two planted signals move the returns. No result on it is
evidence about a real factor.

---

## 2. The metrics

**Total return** = `end_equity / start_equity - 1`.

**CAGR** (compound annual growth rate) = `(end/start)^(365/days) - 1`. The annualized rate.
On a short record it is not useful: one good week annualizes to a very large number.

**Max drawdown (`maxDD`)** = the largest fall of equity from a peak:
`min(equity / running_max - 1)`. `-0.8%` means that at the worst point the book was 0.8%
below its own high.

**Annualized volatility** = the standard deviation of daily returns x √365.

**Sharpe ratio** = `mean(daily returns) / std(daily returns) x √365`, with a risk-free rate
of zero. The return for each unit of risk. The historical test reads the Sharpe ratio of
the daily net excess return. A Sharpe ratio is 0 for fewer than two returns.

> **A Sharpe ratio from a short record is not reliable.** √365 scales the noise of a few
> days into a large number.

**Information coefficient (IC).** The Spearman rank correlation between a factor and the
return after it, on one date. The weekly IC uses the Sunday rank against the 7-day return
from Monday 00:00 UTC. A factor can have a significant IC and still lose money after
costs. Thus an IC is a number, not a verdict.

**t statistic (t).** The mean of the weekly ICs divided by its standard error:
`mean / sd x sqrt(n)`.

**Residual IC (rIC).** The IC of a factor after the removal of what three per-coin
regressors explain. The regressors are the coin's shrunk long-run excess return (`n / (n +
180) x mean`), its 90-day BTC beta and its 30-day log dollar volume. Each Sunday, a ridge
regression removes their part from the factor ranks. The rank correlation of the residual
with the 7-day return is the rIC. Raw IC rewards a factor that only ranks coins by their
past mean return, so rIC is the statistic to read.

---

## 3. Market terms

**Perpetual future (perp).** A derivative that follows the price of a coin and has no
expiry date. It permits long and short positions. Funding keeps its price near the spot
price.

**Funding.** A payment between the holders of a perpetual future at each settlement,
normally each 8 hours. Factorlab uses this sign rule: a positive funding rate means that
longs pay shorts. A book receives or pays funding on its positions each day.

**Long and short.** A long position gains when the price rises. A short position gains
when the price falls. A short has a negative weight.

**Market-neutral (long-short).** About equal money long and short. Broad market moves
cancel for the most part, and the rank skill remains.

**Gross exposure.** The sum of the absolute weights. 1.0 means no leverage.

**Leverage.** Positions larger than the equity. Each book of a test in Factorlab has a
gross exposure of at most 1.0, so it has no leverage.

**Basis point (bp).** One hundredth of a percent. 5 bps = 0.05%.

**Taker and maker.** A *taker* order executes at once against an order in the book. It pays a
higher fee (5 bps on OKX). A *maker* order waits in the order book. It pays less (2 bps),
but it can fail to fill. The default book specification uses taker costs. Maker costs are a
second scenario.

**Slippage.** The difference between the expected price and the fill price. The model is a
flat 5 bps on top of the fee (2 bps in the maker scenario).

**Turnover.** The part of the book that changes at a rebalance, as a fraction of equity.
A full change from one set of coins to another is 1.0. The engine charges fees on
turnover, so turnover directly reduces the net return of a book.

**Rebalance.** The computation of new target weights and the trades to them. In
Factorlab: Mondays at 00:00 UTC, with the factor values of the Sunday before.

**Exit band.** A book enters a coin only in the top fraction (default 20%). It holds the
coin until the coin goes below the exit fraction (default 40%). A plain book draws its
cutoff again each week, so it buys and sells a coin near the cutoff again and again. The
exit band reduces turnover.

**Quintile.** A fifth. The top quintile of 20 coins is the best 4.

**Universe.** The fixed list of 20 coins in `config.py`. A list chosen by today's volume
adds survivorship bias, because today's large coins are the coins that rose.

**Notional.** The dollar size of a position.

**BTC hold.** The buy-and-hold of BTC, price only, over the same span as a book. A test
shows it as context. No rule reads it.
