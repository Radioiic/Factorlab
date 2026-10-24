"""End-to-end factor-research orchestration.

:func:`run_research` strings every AlphaForge stage together into one reproducible run::

    simulate -> universe -> split (train / valid / test)
        -> seed factors (train scorecards, library)
        -> GP mining (fitness on TRAIN only)
        -> hall-of-fame validation on VALID (threshold + greedy de-correlation) -> library
        -> out-of-sample TEST scorecards for every library factor
        -> composites (ICIR-weighted linear, walk-forward ML)
        -> industry-neutral long/short backtests on TEST (best single / linear / ML)
        -> rediscovery check against the planted alphas
        -> outputs (data/, library/, tables/*.csv, figures/*.png, report.html, run.json)

The three date blocks have strictly separated roles, which is the whole point of the design:

* **train** - the only data the miner's fitness and the composite weights ever see;
* **valid** - used once, to decide which hall-of-fame expressions survive;
* **test**  - touched only by the final scorecards and backtests, never by a decision.

The split is fixed before mining, every factor gets the same scorecard, and one
configuration reproduces the full run.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.combine import MLCombiner, combine_linear
from alphaforge.config import ResearchConfig
from alphaforge.data import DataStore, build_universe, simulate_market
from alphaforge.evaluation import FactorReport, evaluate_factor, rowwise_corr
from alphaforge.expr import Call, Evaluator, Node, parse, to_string
from alphaforge.library import FactorLibrary
from alphaforge.mining import (
    ExpressionGenerator,
    FitnessFunction,
    GeneticMiner,
    MinedFactor,
    select_uncorrelated,
)
from alphaforge.portfolio import (
    BacktestResult,
    backtest,
    hold_weights,
    rebalance_schedule,
    weights_from_scores,
)
from alphaforge.preprocess import preprocess_factor
from alphaforge.reporting import (
    plot_corr_heatmap,
    plot_equity,
    plot_mining_history,
    plot_report_figures,
    render_html_report,
)

logger = logging.getLogger("alphaforge.research")

__all__ = [
    "SEED_FACTORS",
    "MINING_FIELDS",
    "MINING_GROUP_FIELDS",
    "BACKTEST_MAX_WEIGHT",
    "run_research",
    "split_dates",
    "factor_values",
]

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

#: Hand-written classical factors, in the expression language (canonical prefix form).
#: Each is written so that the *expected* IC is positive (``neg`` where the literature
#: says "low is good"); the pipeline re-orients on train evidence anyway.
SEED_FACTORS: dict[str, str] = {
    # short-term reversal: last week's winners lose (the planted ``alpha_reversal``)
    "reversal_5d": "neg(cs_rank(ts_pct_change(close, 5)))",
    # one-month reversal: the classic one-month horizon
    "reversal_20d": "neg(cs_rank(ts_pct_change(close, 20)))",
    # 12-1 style momentum: return over t-120..t-20 (the planted ``alpha_momentum``)
    "momentum_120_20": "ts_delay(ts_pct_change(close, 100), 20)",
    # abnormal volume: names trading far above their 20d average under-perform
    "volume_shock": "neg(cs_rank(div(volume, ts_mean(volume, 20))))",
    # value: book-to-price (point-in-time), the planted ``alpha_value``
    "value_bp": "cs_rank(bp)",
    # low-volatility anomaly
    "low_volatility": "neg(cs_rank(ts_std(returns, 20)))",
    # Amihud illiquidity: |return| per unit of traded value (illiquid names earn a premium)
    "illiquidity_amihud": "cs_rank(ts_mean(div(abs(returns), amount), 20))",
    # intraday price range relative to close (a volatility / uncertainty proxy)
    "price_range": "neg(cs_rank(div(sub(high, low), close)))",
    # MAX effect: lottery-like names with a large recent daily return under-perform
    "max_return_20d": "neg(cs_rank(ts_max(returns, 20)))",
    # price-volume correlation: rising prices on rising volume tend to revert
    "price_volume_corr": "neg(cs_rank(ts_corr(close, volume, 10)))",
}

#: Store fields offered to the miner as panel terminals.
MINING_FIELDS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "returns",
    "market_cap",
    "bp",
)
#: Store fields offered to the miner as group terminals.
MINING_GROUP_FIELDS: tuple[str, ...] = ("industry",)

#: Single-name cap used by every backtest of the pipeline.
BACKTEST_MAX_WEIGHT: float = 0.02

#: Walk-forward refit cadence of the ML composite (trading days ~ one month).
ML_REFIT_EVERY: int = 21

#: Number of factors given the full figure treatment (IC / quantiles / decay).
N_TOP_FIGURES: int = 3

_WINDOWS: tuple[str, ...] = ("train", "valid", "test")

# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


class _Progress:
    """Step logger: writes to ``logging`` (INFO) and, when given, to ``log_fn``."""

    def __init__(self, log_fn: Callable[[str], Any] | None, timings: dict[str, float]) -> None:
        self.log_fn = log_fn
        self.timings = timings
        self._n = 0

    def say(self, msg: str) -> None:
        logger.info(msg)
        if self.log_fn is not None:
            try:
                self.log_fn(msg)
            except Exception as exc:  # noqa: BLE001 - a broken log sink must never kill a run
                logger.warning("log_fn raised %s: %s", type(exc).__name__, exc)

    @contextmanager
    def step(self, key: str, title: str) -> Iterator[None]:
        self._n += 1
        n = self._n
        self.say(f"[step {n}] {title} ...")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.timings[key] = dt
            self.say(f"[step {n}] {title} done in {dt:.2f}s")


def _json_safe(obj: Any) -> Any:
    """Recursively convert to JSON-serialisable Python (NaN/inf -> None, numpy -> Python)."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset, np.ndarray, pd.Index)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pd.Series):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return str(obj)


def _fmt(x: Any, spec: str = "{:+.3f}") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return spec.format(v) if math.isfinite(v) else "n/a"


def _negate(node: Node) -> Node:
    """``neg(node)``, collapsing a leading ``neg`` so a double flip prints cleanly."""
    if isinstance(node, Call) and node.op == "neg" and len(node.args) == 1:
        return node.args[0]
    return Call("neg", (node,))


def _needs_flip(ic_mean: Any) -> bool:
    try:
        v = float(ic_mean)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v < 0.0


def split_dates(cfg: ResearchConfig, dates: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    """Contiguous train / valid / test date blocks per ``cfg.train_frac`` / ``cfg.valid_frac``.

    Uses :meth:`ResearchConfig.split_index`, so ``dates[:train_end]`` is train,
    ``dates[train_end:valid_end]`` valid and the rest test.  The blocks are contiguous in
    time (never shuffled): a random split would leak overlapping forward returns across
    the boundary.

    The split is chronological: the model trains on the past, and the test uses only
    the dates after the training dates.
    """
    train_end, valid_end = cfg.split_index(len(dates))
    out = {"train": dates[:train_end], "valid": dates[train_end:valid_end], "test": dates[valid_end:]}
    for k, v in out.items():
        if len(v) == 0:
            raise ValueError(
                f"the {k} block is empty ({len(dates)} dates, train_frac={cfg.train_frac}, valid_frac={cfg.valid_frac})"
            )
    return out


def factor_values(
    node: Node | str,
    evaluator: Evaluator,
    store: DataStore,
    universe: pd.DataFrame | None,
) -> pd.DataFrame:
    """Evaluate an expression on the full store and pre-process it (universe, winsor, fill, neutralise, z-score).

    This is the pipeline's single definition of "the values of a factor": the same
    treatment for a seed expression, a hall-of-fame candidate and a library entry, on the
    full date range, so that train / valid / test scorecards of one factor are slices of
    one panel.  ``+-inf`` never survives (it becomes ``NaN``).
    """
    if isinstance(node, str):
        node = parse(node)
    # Arbitrary mined expressions overflow / divide by zero all the time; inf/NaN is the
    # correct outcome for those cells, so silence exactly those numpy warnings here.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw = evaluator.eval(node)
        out = preprocess_factor(raw, store, universe=universe)
    arr = out.to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(arr).any():
        out = out.where(np.isfinite(arr))
    return out


def _eval_window(
    values: pd.DataFrame,
    window_store: DataStore,
    window: pd.DatetimeIndex,
    cfg: ResearchConfig,
    *,
    name: str,
    expr: str | None,
) -> FactorReport:
    """Scorecard of ``values`` on ``window`` using labels from ``window_store``.

    ``window_store`` is the store truncated at the window's last date, so labels at the
    tail of the window are NaN instead of peeking at the next block's prices.
    """
    return evaluate_factor(
        values,
        window_store,
        name=name,
        expr=expr,
        horizon=cfg.horizon,
        lag=cfg.lag,
        n_quantiles=cfg.n_quantiles,
        start=window[0],
        end=window[-1],
    )


def _report_table(reports: Mapping[str, FactorReport], library: FactorLibrary) -> pd.DataFrame:
    """``FactorReport.table`` plus the library ``tags`` column."""
    table = FactorReport.table(list(reports.values()))
    tags = {n: ",".join(library.get(n)["tags"]) for n in table.index if n in library}
    table.insert(1, "tags", pd.Series(tags).reindex(table.index).fillna(""))
    return table


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8")
    return path


def _run_backtest(
    scores: pd.DataFrame,
    store: DataStore,
    test: pd.DatetimeIndex,
    cfg: ResearchConfig,
    industry: pd.DataFrame,
    max_weight: float,
) -> BacktestResult:
    """Industry-neutral equal-weight long/short book on TEST, held ``rebalance_every`` days.

    The single-name cap is a hard limit: on a small universe (fewer than ``1 / max_weight``
    names per leg) the legs run under-invested, which
    :func:`~alphaforge.portfolio.construct.weights_from_scores` logs.  Sharpe is unaffected
    by that scaling; ``ann_return`` / ``max_dd`` are, so the comparison table carries the
    realised ``gross_exposure``.
    """
    s = scores.reindex(index=test, columns=store.assets)
    if not np.isfinite(s.to_numpy(dtype=np.float64, na_value=np.nan)).any():
        raise ValueError("scores have no finite value on the test window")
    w = weights_from_scores(s, top_frac=cfg.top_frac, industry=industry.reindex(index=test), max_weight=max_weight)
    held = hold_weights(w, rebalance_schedule(test, cfg.rebalance_every))
    return backtest(held, store, cost_bps=cfg.cost_bps, lag=cfg.lag)


# --------------------------------------------------------------------------------------
# HTML report text (English, short) -- kept in one place so the pipeline body stays readable
# --------------------------------------------------------------------------------------


def _p(*paras: str) -> str:
    return "".join(f"<p>{t}</p>" for t in paras)


def _html_sections(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    cfg: ResearchConfig = ctx["cfg"]
    split = ctx["split"]
    T = ctx["tables"]
    F = ctx["figure_paths"]
    h = ctx["headline"]

    split_table = pd.DataFrame(
        {
            "start": [str(split[k][0].date()) for k in _WINDOWS],
            "end": [str(split[k][-1].date()) for k in _WINDOWS],
            "trading days": [len(split[k]) for k in _WINDOWS],
            "use": [
                "mining fitness / seed-factor evaluation / composite weights",
                "hall-of-fame screen (used once)",
                "out-of-sample scoring and backtest (no decision uses it)",
            ],
        },
        index=pd.Index(["train", "valid", "test"], name="window"),
    )
    cfg_table = pd.Series(_json_safe({k: v for k, v in cfg.to_dict().items() if k != "sim"}), name="value").to_frame()
    sim_table = pd.Series(_json_safe(cfg.to_dict()["sim"]), name="value").to_frame()
    timing_table = pd.Series(ctx["timings"], name="seconds").to_frame()

    sections: list[dict[str, Any]] = [
        {
            "title": "1. Run overview and conclusions",
            "text": _p(
                f"This run uses a synthetic market of <b>{ctx['store'].assets.size}</b> stocks × <b>{len(ctx['store'].dates)}</b> trading days. "
                f"Starting from <b>{len(SEED_FACTORS)}</b> classic seed factors, genetic programming mined <b>{h['n_mined']}</b> new factors "
                f"that passed the validation screen and are not redundant with each other. The factor library holds <b>{h['n_library']}</b> factors.",
                f"Best single factor out of sample (test set): Rank ICIR = <b>{_fmt(h['best_oos_rank_icir'])}</b> ({h['best_oos_factor']}). "
                f"Test-set backtest Sharpe: best single factor <b>{_fmt(h['sharpe_best_single'], '{:.2f}')}</b>, "
                f"ICIR-weighted linear composite <b>{_fmt(h['sharpe_linear'], '{:.2f}')}</b>, "
                f"rolling ML composite <b>{_fmt(h['sharpe_ml'], '{:.2f}')}</b> (cost {cfg.cost_bps:.0f} bps, rebalance every {cfg.rebalance_every} days).",
                "The train, validation and test roles are separate: mining, screening and weighting see only the first two "
                "windows, and the last window only gives the test scores.",
            ),
            "table": pd.DataFrame(h["summary_rows"]).set_index("Metric") if h.get("summary_rows") else None,
        },
        {
            "title": "2. Run configuration",
            "text": _p(
                "The research configuration (<code>ResearchConfig</code>) and the synthetic-market parameters (<code>SimConfig</code>). "
                "Every result can be reproduced from the configuration in <code>run.json</code>.",
                "<b>Why:</b> reproducibility is the floor of a research platform. Months later, the same configuration must give the "
                "same table again; otherwise two versions of a factor cannot be compared.",
            ),
            "table": pd.concat({"research": cfg_table, "sim": sim_table}, names=["group", "parameter"]),
        },
        {
            "title": "3. Data and sample split",
            "text": _p(
                f"The synthetic market is generated from a market factor + industry factors + four \"planted\" true alphas (reversal, "
                f"momentum, volume shock, value) + idiosyncratic noise. On a suspension day the volume is missing and the price is "
                f"unchanged. Fundamentals (book value per share) become visible with a point-in-time lag of {cfg.sim.fundamental_lag_days} days.",
                f"The universe excludes low-priced stocks, suspended stocks and the bottom 10% by 20-day traded value: on average "
                f"<b>{ctx['universe_mean']:.1f}</b> stocks are investable each day.",
                "<b>Why:</b> the split is chronological, not random, because overlapping forward returns make a random split \"leak\" "
                "the future. An alpha mined on stocks that cannot be traded disappears in live trading, so the universe is fixed before "
                "the mining.",
            ),
            "table": split_table,
        },
        {
            "title": "4. Seed factors (train-set evaluation)",
            "text": _p(
                "Ten textbook factors go into the library as the baseline. Every factor gets the same preprocessing: universe mask → "
                "MAD winsorization → cross-sectional median fill → industry + size neutralization → standardization. A factor whose "
                "train-set IC is negative is flipped (an outer <code>neg</code> in the expression), so every factor in the library has "
                "the same direction: \"high score = long\".",
                "<b>Why:</b> the seed library has two roles. It gives the PM a \"known to work\" reference, and it is the baseline of "
                "the correlation penalty in the mining fitness, which forces the miner to find something <i>new</i>.",
            ),
            "table": T["seed_screen"],
        },
        {
            "title": "5. Genetic-programming mining (train set only)",
            "text": _p(
                f"Population {cfg.population_size} × {cfg.n_generations} generations, tree depth ≤ {cfg.max_depth}. The terminals are "
                f"price/volume, market-cap and book-to-market fields; the operators come from the ts_* / cs_* / group_* library. "
                f"Fitness = |annualized ICIR| − 0.5 × turnover − a penalty for a correlation above {cfg.max_corr:.2f} with the library − "
                f"a complexity penalty, <b>computed on the train set only</b>. {ctx['n_evaluations']} distinct expressions were "
                f"evaluated, and the hall of fame keeps {len(ctx['hof'])}.",
                "<b>Why:</b> the miner is not smarter than a researcher, only tireless. Rewarding ICIR rather than IC prefers \"stable\" "
                "signals, the turnover penalty keeps trading costs from eating the alpha, and the correlation penalty stops the same "
                "idea from being written a hundred ways.",
            ),
            "table": T["mining_history"],
            "figures": [(F["mining_history"], "Fitness by generation: best / mean / median")]
            if F.get("mining_history")
            else [],
        },
        {
            "title": "6. Validation screen and de-duplication",
            "text": _p(
                f"Each hall-of-fame expression is first oriented by the sign of its train-set IC, then scored again on the validation "
                f"set. Only candidates with an annualized ICIR ≥ {cfg.min_valid_icir_ann} (same sign) stay. They are then de-duplicated "
                f"greedily by validation ICIR (pairwise |Spearman correlation| ≤ {cfg.max_corr}), at most {cfg.max_factors} are kept, "
                f"and a last correlation check against the factors already in the library follows.",
                f"Result: {ctx['n_hof_valid']} valid candidates → {ctx['n_passed']} pass the threshold → {ctx['n_selected']} are not "
                f"redundant → <b>{h['n_mined']}</b> are written to the library.",
                "<b>Why:</b> the train-set fitness is always overstated, because the miner optimizes exactly that. The validation set, "
                "used once and only for a yes/no decision, is the cheapest and most effective defense against overfitting. A factor "
                "whose sign flips between train and validation is dropped.",
            ),
            "table": T["hall_of_fame"],
        },
        {
            "title": "7. Out-of-sample (test set) evaluation",
            "text": _p(
                "The full test-set scorecard of every library factor (seed + mined), sorted by Rank ICIR. The test set never took part "
                "in any decision.",
                "<b>Why:</b> this is the only table to take to the PM to discuss capital: every other table has \"seen the answers\" to "
                "some degree.",
            ),
            "table": T["report_test"],
            "figures": [(p, cap) for p, cap in F.get("top", [])],
        },
        {
            "title": "8. Library correlation",
            "text": _p(
                "A heat map of the mean pairwise cross-sectional Spearman correlation of the library factors.",
                "<b>Why:</b> two factors with a correlation of 0.9 are one factor that pays the trading costs twice. The heat map is "
                "the fastest check of whether the miner added diversification.",
            ),
            "figures": [(F["corr_heatmap"], "Library correlation heat map")] if F.get("corr_heatmap") else [],
        },
        {
            "title": "9. Factor combination",
            "text": _p(
                "Two composites: (1) linear: each factor is standardized cross-sectionally and weighted by its <b>train-set</b> |ICIR|; "
                f"(2) ML: HistGradientBoosting with the cross-sectional ranks of the factors as features and the rank of the forward "
                f"return as the label, trained on a rolling window ({ctx['ml_train_window']} days, refit every {ML_REFIT_EVERY} days, "
                f"only labels already realized on the refit day). The table gives the scores of both composites on the validation and "
                f"test sets.",
                "<b>Why:</b> the combined ICIR of several weakly correlated alphas is far above that of any single factor; that is the "
                "point of keeping a library. Linear ICIR weighting can hardly overfit. ML can capture interactions such as \"reversal "
                "works only in high-volume stocks\", but it is credible only with a rolling-training protocol.",
            ),
            "table": T["composite_ic"],
            "figures": [],
        },
        {
            "title": "10. Portfolio construction and backtest (test set)",
            "text": _p(
                f"Score → equal-weight long/short of the top / bottom {cfg.top_frac:.0%} within each industry (target exposure ±1 per "
                f"leg, hard cap {ctx['max_weight']:.1%} per stock) → rebalance every {cfg.rebalance_every} days → trade at the close "
                f"of the day after the signal, linear cost {cfg.cost_bps:.0f} bps × turnover. Three curves: the best single factor on "
                f"validation, the linear composite and the ML composite. "
                "When a leg holds fewer than 1 / cap stocks, the hard cap leaves the book under-invested (the "
                "<code>gross_exposure</code> column gives the actual gross exposure). This scaling does not change the Sharpe ratio; "
                "the annual return and the drawdown shrink in proportion.",
                "<b>Why:</b> an IC says only that the factor \"ranks correctly\". Only a backtest with costs, suspension handling and "
                "realistic trade timing says whether it \"can earn\". Many factors with a fine IC die at this step. The per-stock cap "
                "is a hard risk constraint: better under-invested than concentrated in a few names.",
            ),
            "table": T["backtest_comparison"],
            "figures": [(p, cap) for p, cap in F.get("equity", [])],
        },
        {
            "title": "11. Rediscovery of the true alphas",
            "text": _p(
                "The synthetic market has four planted true alphas (<code>store.meta['true_alphas']</code>). The table gives, for each "
                "library factor, its mean cross-sectional Spearman correlation with each true alpha on the test set, and the best match "
                "(the <code>aligned</code> column is adjusted for the sign of the true alpha's coefficient; a positive value means that "
                "the direction is right too).",
                "<b>Why:</b> this is the \"answer key\" that only synthetic data has. If the miner cannot find signals that you know "
                "exist, the platform is broken, not the market. If what it finds is much stronger than what was planted, it is most "
                "likely look-ahead bias.",
            ),
            "table": T["rediscovery"],
        },
        {
            "title": "12. Time of each step",
            "text": _p(
                "The wall-clock time of each step (seconds). <b>Why:</b> the throughput of a research platform sets how many "
                "ideas can be tested in a day; the timing table is where optimization starts."
            ),
            "table": timing_table,
        },
    ]
    if ctx.get("ml_importances") is not None:
        sections[8]["text"] += _p(
            "Feature importance of the ML composite (permutation importance of the last refit): "
            + ", ".join(
                f"{k}: {_fmt(v, '{:.4f}')}" for k, v in ctx["ml_importances"].sort_values(ascending=False).items()
            )
        )
    return sections


# --------------------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------------------


def run_research(
    cfg: ResearchConfig,
    *,
    log_fn: Callable[[str], Any] | None = None,
    max_weight: float = BACKTEST_MAX_WEIGHT,
) -> dict[str, Any]:
    """Run the whole research pipeline and write every output.

    Args:
        cfg: Research configuration (market, split, miner budget, cost assumptions, ``out_dir``).
        log_fn: Optional callable receiving each progress message (in addition to logging).
        max_weight: Single-name cap ``|w| <= max_weight`` in the backtests (default 2%).

    Returns:
        ``dict`` with at least the keys ``store``, ``library``, ``reports_train``,
        ``reports_valid``, ``reports_test`` (each ``{factor name: FactorReport}``), ``mined``
        (the :class:`MinedFactor` objects that entered the library), ``backtests``
        (``{label: BacktestResult}``), ``paths`` (``{key: Path}``) and ``timings``
        (``{step: seconds}``), plus extras: ``hall_of_fame``, ``mining_history``,
        ``tables`` (every CSV as a DataFrame), ``composites`` (score panels), ``split``,
        ``universe``, ``headline`` and ``config``.

    Each run has one function, one configuration and one output folder; the same seed
    makes ``report.html`` and ``run.json`` again.
    """
    t_run = time.perf_counter()
    timings: dict[str, float] = {}
    prog = _Progress(log_fn, timings)
    out_dir = Path(cfg.out_dir)
    dirs = {k: out_dir / k for k in ("data", "library", "tables", "figures")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {"out_dir": out_dir, **dirs}
    tables: dict[str, pd.DataFrame] = {}
    prog.say(
        f"run_research: out_dir={out_dir.resolve()} seed={cfg.seed} sim=({cfg.sim.n_assets} assets x {cfg.sim.n_days} days)"
    )

    # ---------------------------------------------------------------- 1. simulate + universe
    with prog.step("simulate", "simulate market and build universe"):
        store = simulate_market(cfg.sim)
        universe = build_universe(store)
        evaluator = Evaluator(store, cache_size=512)
        industry = store.get("industry")
        universe_mean = float(universe.notna().sum(axis=1).mean())
        prog.say(f"  {store!r}")

    # ---------------------------------------------------------------- 2. split
    with prog.step("split", "split dates into train / valid / test"):
        split = split_dates(cfg, store.dates)
        # labels for a window come from a store truncated at the window end (no peeking)
        window_stores = {
            "train": store.slice(None, split["train"][-1]),
            "valid": store.slice(None, split["valid"][-1]),
            "test": store,
        }
        for k in _WINDOWS:
            prog.say(f"  {k:<5}: {split[k][0].date()} .. {split[k][-1].date()} ({len(split[k])} days)")

    # ---------------------------------------------------------------- 3. seed factors
    library = FactorLibrary(root=dirs["library"], max_corr=cfg.max_corr)
    reports_train: dict[str, FactorReport] = {}
    reports_valid: dict[str, FactorReport] = {}
    reports_test: dict[str, FactorReport] = {}
    with prog.step("seeds", f"evaluate {len(SEED_FACTORS)} seed factors on train and seed the library"):
        seed_rows: list[dict[str, Any]] = []
        for name, expr in SEED_FACTORS.items():
            node = parse(expr)
            values = factor_values(node, evaluator, store, universe)
            rep = _eval_window(values, window_stores["train"], split["train"], cfg, name=name, expr=to_string(node))
            flipped = _needs_flip(rep.ic_mean)
            raw_ic, raw_icir = rep.ic_mean, rep.icir_ann
            if flipped:
                node = _negate(node)
                values = -values
                rep = _eval_window(values, window_stores["train"], split["train"], cfg, name=name, expr=to_string(node))
            corr_name, corr = library.max_corr(values)
            tags = ("seed", "flipped") if flipped else ("seed",)
            # Seeds are the firm's classical benchmarks: they are always kept (force=True);
            # the novelty screen is for *mined* candidates.  Their mutual correlation is
            # still reported so a redundant seed is visible.
            library.add(name, to_string(node), values, report=rep, tags=tags, force=True)
            reports_train[name] = rep
            seed_rows.append(
                {
                    "name": name,
                    "expr": to_string(node),
                    "flipped": flipped,
                    "raw_train_ic_mean": raw_ic,
                    "raw_train_icir_ann": raw_icir,
                    "train_ic_mean": rep.ic_mean,
                    "train_icir_ann": rep.icir_ann,
                    "train_rank_icir": rep.rank_icir,
                    "train_ls_sharpe": rep.ls_sharpe,
                    "turnover": rep.turnover_mean,
                    "coverage": rep.coverage,
                    "max_corr_with": corr_name,
                    "max_corr": corr,
                }
            )
            prog.say(
                f"  seed {name:<20} IC {_fmt(rep.ic_mean, '{:+.4f}')} ICIR_ann {_fmt(rep.icir_ann, '{:+.2f}')}"
                f"{'  (flipped)' if flipped else ''}  max_corr {_fmt(corr)} vs {corr_name}"
            )
        tables["seed_screen"] = pd.DataFrame(seed_rows).set_index("name")

    # ---------------------------------------------------------------- 4. GP mining (train only)
    with prog.step(
        "mining", f"genetic-programming mining (pop {cfg.population_size} x {cfg.n_generations} gens, train only)"
    ):
        generator = ExpressionGenerator(
            list(MINING_FIELDS), group_fields=list(MINING_GROUP_FIELDS), max_depth=cfg.max_depth, seed=cfg.seed
        )
        fitness = FitnessFunction(
            store,
            evaluator=evaluator,
            library=library,
            horizon=cfg.horizon,
            lag=cfg.lag,
            start=split["train"][0],
            end=split["train"][-1],
            universe=universe,
            max_corr=cfg.max_corr,
            n_quantiles=cfg.n_quantiles,
        )
        miner = GeneticMiner(
            generator,
            fitness,
            population_size=cfg.population_size,
            n_generations=cfg.n_generations,
            hall_of_fame=cfg.hall_of_fame,
            seed=cfg.seed,
            n_jobs=cfg.n_jobs,
        )

        def _on_generation(gen: int, hof: list[MinedFactor]) -> None:
            h = miner.history[-1]
            prog.say(
                f"  gen {gen + 1}/{cfg.n_generations}: best {_fmt(h['best'], '{:.3f}')} mean {_fmt(h['mean'], '{:.3f}')} "
                f"valid {h['n_valid']}/{cfg.population_size} hof {len(hof)} [{h['elapsed_s']:.1f}s]  {h['best_expr']}"
            )

        hof = miner.run(callback=_on_generation)
        history = list(miner.history)
        tables["mining_history"] = pd.DataFrame(history).set_index("gen") if history else pd.DataFrame()
        prog.say(f"  mining: {miner.n_evaluations} expressions evaluated, hall of fame {len(hof)}")

    # ---------------------------------------------------------------- 5. validation + selection
    mined: list[MinedFactor] = []
    with prog.step("validation", "re-evaluate the hall of fame on VALID, threshold, de-correlate, add to library"):
        hof_rows: list[dict[str, Any]] = []
        candidates: list[MinedFactor] = []
        cand_values: dict[str, pd.DataFrame] = {}
        cand_nodes: dict[str, Node] = {}
        cand_valid: dict[str, FactorReport] = {}
        n_hof_valid = 0
        for m in hof:
            if not m.result.valid:
                continue
            n_hof_valid += 1
            node = m.node
            values = factor_values(node, evaluator, store, universe)
            flipped = _needs_flip(m.result.ic_mean)  # orientation from TRAIN evidence only
            if flipped:
                node = _negate(node)
                values = -values
            rep_v = _eval_window(values, window_stores["valid"], split["valid"], cfg, name=m.expr, expr=to_string(node))
            passed = math.isfinite(rep_v.icir_ann) and rep_v.icir_ann >= cfg.min_valid_icir_ann
            sign_flip = math.isfinite(rep_v.ic_mean) and rep_v.ic_mean < 0.0
            hof_rows.append(
                {
                    "expr": m.expr,
                    "oriented_expr": to_string(node),
                    "generation": m.generation,
                    "fitness": m.fitness,
                    "complexity": m.result.complexity,
                    "flipped": flipped,
                    "train_ic_mean": m.result.ic_mean,
                    "train_icir_ann": m.result.icir_ann,
                    "train_turnover": m.result.turnover,
                    "train_max_corr": m.result.max_corr,
                    "valid_ic_mean": rep_v.ic_mean,
                    "valid_icir_ann": rep_v.icir_ann,
                    "valid_rank_icir": rep_v.rank_icir,
                    "valid_ls_sharpe": rep_v.ls_sharpe,
                    "valid_coverage": rep_v.coverage,
                    "sign_flip_train_valid": sign_flip,
                    "passed": passed,
                    "selected": False,
                    "library_name": "",
                }
            )
            if passed:
                candidates.append(m)
                cand_values[m.expr] = values
                cand_nodes[m.expr] = node
                cand_valid[m.expr] = rep_v
        n_passed = len(candidates)
        selected = select_uncorrelated(
            candidates, cand_values, max_corr=cfg.max_corr, key=lambda m: cand_valid[m.expr].icir_ann
        )
        n_selected = len(selected)
        row_by_expr = {r["expr"]: r for r in hof_rows}
        for m in selected:
            row_by_expr[m.expr]["selected"] = True
            if len(mined) >= cfg.max_factors:
                row_by_expr[m.expr]["library_name"] = "(cap max_factors)"
                continue
            name = f"mined_{len(mined) + 1:02d}"
            node = cand_nodes[m.expr]
            values = cand_values[m.expr]
            expr_stored = to_string(node)
            rep_t = _eval_window(values, window_stores["train"], split["train"], cfg, name=name, expr=expr_stored)
            tags = ("mined", "flipped") if row_by_expr[m.expr]["flipped"] else ("mined",)
            added = library.add(name, expr_stored, values, report=rep_t, tags=tags)
            if not added:
                corr_name, corr = library.max_corr(values)
                row_by_expr[m.expr]["library_name"] = f"(rejected: |corr| {abs(corr):.2f} vs {corr_name})"
                prog.say(f"  candidate {m.expr} rejected by the library: |corr| {abs(corr):.2f} with {corr_name}")
                continue
            rep_v = cand_valid[m.expr]
            rep_v.name = name
            reports_train[name] = rep_t
            reports_valid[name] = rep_v
            mined.append(m)
            row_by_expr[m.expr]["library_name"] = name
            prog.say(
                f"  {name}: valid ICIR_ann {_fmt(rep_v.icir_ann, '{:+.2f}')} rankICIR {_fmt(rep_v.rank_icir)} "
                f"train ICIR_ann {_fmt(rep_t.icir_ann, '{:+.2f}')}  {expr_stored}"
            )
        hof_table = pd.DataFrame(hof_rows)
        if not hof_table.empty:
            hof_table = hof_table.sort_values(["passed", "valid_icir_ann"], ascending=[False, False]).reset_index(
                drop=True
            )
            hof_table.index.name = "rank"
        tables["hall_of_fame"] = hof_table
        prog.say(
            f"  validation: {n_hof_valid} valid hall-of-fame members -> {n_passed} passed icir_ann >= {cfg.min_valid_icir_ann} "
            f"-> {n_selected} uncorrelated -> {len(mined)} added to the library (cap {cfg.max_factors})"
        )

    # ---------------------------------------------------------------- 6. OOS test evaluation
    with prog.step("oos_eval", "train / valid / test scorecards for every library factor"):
        for name in library.names():
            entry = library.get(name)
            values, expr = entry["values"], entry["expr"]
            if name not in reports_train:
                reports_train[name] = _eval_window(
                    values, window_stores["train"], split["train"], cfg, name=name, expr=expr
                )
            if name not in reports_valid:
                reports_valid[name] = _eval_window(
                    values, window_stores["valid"], split["valid"], cfg, name=name, expr=expr
                )
            reports_test[name] = _eval_window(values, window_stores["test"], split["test"], cfg, name=name, expr=expr)
        tables["report_train"] = _report_table(reports_train, library).sort_values("rank_icir", ascending=False)
        tables["report_valid"] = _report_table(reports_valid, library).sort_values("rank_icir", ascending=False)
        tables["report_test"] = _report_table(reports_test, library).sort_values("rank_icir", ascending=False)
        tables["library"] = library.to_frame()
        best_oos = tables["report_test"].index[0]
        prog.say(
            f"  best OOS factor by rank ICIR: {best_oos} "
            f"(rank ICIR {_fmt(tables['report_test'].loc[best_oos, 'rank_icir'])}, "
            f"rank IC {_fmt(tables['report_test'].loc[best_oos, 'rank_ic_mean'], '{:+.4f}')})"
        )
        for name in tables["report_test"].index[:5]:
            r = reports_test[name]
            prog.say(
                f"    {name:<20} test IC {_fmt(r.ic_mean, '{:+.4f}')} ICIR_ann {_fmt(r.icir_ann, '{:+.2f}')} LS Sharpe {_fmt(r.ls_sharpe, '{:+.2f}')}"
            )

    # ---------------------------------------------------------------- 7. composites
    composites: dict[str, pd.DataFrame] = {}
    with prog.step("composite", "linear (ICIR-weighted) and walk-forward ML composites"):
        factors = library.all_values()
        train_table = FactorReport.table(list(reports_train.values()))
        composites["linear_icir"] = combine_linear(factors, weights="icir", ic_table=train_table)

        n_train = len(split["train"])
        ml_train_window = max(20, min(250, n_train // 2))
        ml_min_train = min(120, ml_train_window)
        delay = cfg.lag + cfg.horizon
        # Start the walk-forward just early enough that the first refit lands at the start of
        # VALID; predictions are only ever *used* on TEST.
        ml_start = max(0, n_train - ml_train_window - delay - 1)
        ml_dates = store.dates[ml_start:]
        fwd = store.fwd_return(cfg.horizon, cfg.lag)
        combiner = MLCombiner(
            train_window=ml_train_window, refit_every=ML_REFIT_EVERY, min_train=ml_min_train, model="hgb", seed=cfg.seed
        )
        ml_pred = combiner.fit_predict(
            {n: v.loc[ml_dates] for n, v in factors.items()}, fwd.loc[ml_dates], horizon=cfg.horizon, lag=cfg.lag
        )
        composites["ml_hgb"] = ml_pred.reindex(index=store.dates)
        ml_importances = combiner.feature_importances_
        if ml_importances is not None:
            tables["ml_feature_importance"] = ml_importances.sort_values(ascending=False).to_frame()
        n_refits = len(combiner.refit_dates_)
        prog.say(
            f"  ML composite: train_window {ml_train_window}, {n_refits} refits, predictions from "
            f"{composites['ml_hgb'].notna().any(axis=1).idxmax().date() if composites['ml_hgb'].notna().any().any() else 'never'}"
        )

        comp_rows: list[FactorReport] = []
        for label, panel in composites.items():
            for w in ("valid", "test"):
                if not np.isfinite(panel.reindex(index=split[w]).to_numpy(dtype=np.float64, na_value=np.nan)).any():
                    logger.warning("composite %s has no values on %s; skipping its scorecard", label, w)
                    continue
                rep = _eval_window(panel, window_stores[w], split[w], cfg, name=f"{label}[{w}]", expr=label)
                comp_rows.append(rep)
        comp_table = FactorReport.table(comp_rows)
        comp_table.insert(0, "window", [n.split("[")[1].rstrip("]") for n in comp_table.index])
        comp_table.insert(0, "composite", [n.split("[")[0] for n in comp_table.index])
        tables["composite_ic"] = comp_table
        for rep in comp_rows:
            prog.say(
                f"  {rep.name:<22} IC {_fmt(rep.ic_mean, '{:+.4f}')} ICIR_ann {_fmt(rep.icir_ann, '{:+.2f}')} rankICIR {_fmt(rep.rank_icir)}"
            )

    # ---------------------------------------------------------------- 8. backtests on TEST
    backtests: dict[str, BacktestResult] = {}
    with prog.step("backtest", "industry-neutral long/short backtests on TEST (best single, linear, ML)"):
        best_single = max(
            library.names(),
            key=lambda n: reports_valid[n].icir_ann if math.isfinite(reports_valid[n].icir_ann) else -np.inf,
        )
        bt_specs = {
            "best_single": (library.values(best_single), best_single, library.get(best_single)["expr"]),
            "linear_icir": (
                composites["linear_icir"],
                "linear_icir",
                "ICIR-weighted mean of the library (train weights)",
            ),
            "ml_hgb": (
                composites["ml_hgb"],
                "ml_hgb",
                f"HistGradientBoosting walk-forward (window {ml_train_window}, refit {ML_REFIT_EVERY})",
            ),
        }
        bt_rows: list[dict[str, Any]] = []
        for label, (scores, src, desc) in bt_specs.items():
            try:
                bt = _run_backtest(scores, store, split["test"], cfg, industry, max_weight)
            except ValueError as exc:
                logger.error("backtest %s failed: %s", label, exc)
                continue
            backtests[label] = bt
            gross = float(bt.positions.abs().sum(axis=1).mean()) if bt.positions is not None else float("nan")
            bt_rows.append(
                {"strategy": label, "source": src, "description": desc, **bt.to_dict(), "gross_exposure": gross}
            )
            prog.say(
                f"  {label:<12} ({src}): Sharpe {_fmt(bt.sharpe, '{:.2f}')} ann.ret {_fmt(bt.ann_return, '{:+.2%}')} "
                f"maxDD {_fmt(bt.max_dd, '{:.2%}')} turnover {_fmt(bt.avg_turnover, '{:.3f}')}/day gross {_fmt(gross, '{:.2f}')}"
            )
        tables["backtest_comparison"] = pd.DataFrame(bt_rows).set_index("strategy") if bt_rows else pd.DataFrame()

    # ---------------------------------------------------------------- 9. rediscovery check
    with prog.step("rediscovery", "correlate every library factor with the planted true alphas (test window)"):
        true_alphas: Mapping[str, pd.DataFrame] = store.meta.get("true_alphas", {}) or {}
        coefs: Mapping[str, float] = store.meta.get("true_alpha_coefs", {}) or {}
        test = split["test"]
        red_rows: list[dict[str, Any]] = []
        per_alpha: dict[str, tuple[str, float]] = {}
        for name in library.names():
            v = library.values(name).reindex(index=test)
            row: dict[str, Any] = {
                "name": name,
                "expr": library.get(name)["expr"],
                "tags": ",".join(library.get(name)["tags"]),
            }
            best_key, best_c = None, 0.0
            for key, z in true_alphas.items():
                c = rowwise_corr(v, z.reindex(index=test, columns=v.columns), method="spearman")
                c = float(np.nanmean(c.to_numpy(dtype=float))) if c.notna().any() else float("nan")
                row[f"corr_{key}"] = c
                if math.isfinite(c) and abs(c) > abs(best_c):
                    best_key, best_c = key, c
            sign = float(np.sign(coefs.get(best_key, 1.0))) if best_key is not None else 1.0
            row["best_match"] = best_key
            row["best_corr"] = best_c if best_key is not None else float("nan")
            row["best_corr_aligned"] = best_c * (sign if sign != 0 else 1.0) if best_key is not None else float("nan")
            red_rows.append(row)
            if best_key is not None and abs(best_c) > abs(per_alpha.get(best_key, ("", 0.0))[1]):
                per_alpha[best_key] = (name, best_c)
        tables["rediscovery"] = pd.DataFrame(red_rows).set_index("name") if red_rows else pd.DataFrame()
        tables["true_alpha_recovery"] = (
            pd.DataFrame(
                [
                    {
                        "true_alpha": k,
                        "coef": coefs.get(k, float("nan")),
                        "best_factor": per_alpha.get(k, ("", float("nan")))[0],
                        "corr": per_alpha.get(k, ("", float("nan")))[1],
                    }
                    for k in true_alphas
                ]
            ).set_index("true_alpha")
            if true_alphas
            else pd.DataFrame()
        )
        for k, (fname, c) in per_alpha.items():
            prog.say(f"  true alpha {k:<9} best matched by {fname:<20} corr {_fmt(c)}")

    # ---------------------------------------------------------------- 10. outputs
    with prog.step("outputs", "write data/, library/, tables/, figures/, report.html, run.json"):
        store.save(dirs["data"])
        library.save(dirs["library"])
        for key, df in tables.items():
            paths[f"table_{key}"] = _write_csv(df, dirs["tables"] / f"{key}.csv")

        figure_paths: dict[str, Any] = {"top": [], "equity": []}
        top_names = list(tables["report_test"].index[:N_TOP_FIGURES])
        for i, name in enumerate(top_names, 1):
            figs = plot_report_figures(reports_test[name], dirs["figures"], prefix=f"top{i}_{name}")
            for kind, p in figs.items():
                paths[f"fig_top{i}_{kind}"] = p
                figure_paths["top"].append((p, f"Test set: top {i}, {name} — {kind}"))
        for label, bt in backtests.items():
            p = dirs["figures"] / f"equity_{label}.png"
            plot_equity(bt, save_path=p, title=f"{label} (test, {cfg.cost_bps:.0f} bps)")
            paths[f"fig_equity_{label}"] = p
            figure_paths["equity"].append((p, f"{label}: test-set equity and drawdown"))
        if history:
            p = dirs["figures"] / "mining_history.png"
            plot_mining_history(history, save_path=p)
            paths["fig_mining_history"] = p
            figure_paths["mining_history"] = p
        corr = library.correlation_matrix()
        tables["library_correlation"] = corr
        paths["table_library_correlation"] = _write_csv(corr, dirs["tables"] / "library_correlation.csv")
        p = dirs["figures"] / "corr_heatmap.png"
        plot_corr_heatmap(corr, save_path=p, title="Factor library correlation (mean row-wise Spearman)")
        paths["fig_corr_heatmap"] = p
        figure_paths["corr_heatmap"] = p

        elapsed = time.perf_counter() - t_run
        bt_table = tables["backtest_comparison"]

        def _sharpe(label: str) -> float:
            return float(bt_table.loc[label, "sharpe"]) if label in bt_table.index else float("nan")

        headline: dict[str, Any] = {
            "n_seed": len(SEED_FACTORS),
            "n_mined": len(mined),
            "n_library": len(library),
            "n_evaluations": miner.n_evaluations,
            "best_oos_factor": best_oos,
            "best_oos_rank_icir": float(tables["report_test"].loc[best_oos, "rank_icir"]),
            "best_oos_rank_ic_mean": float(tables["report_test"].loc[best_oos, "rank_ic_mean"]),
            "best_single_for_backtest": best_single,
            "sharpe_best_single": _sharpe("best_single"),
            "sharpe_linear": _sharpe("linear_icir"),
            "sharpe_ml": _sharpe("ml_hgb"),
            "elapsed_s": elapsed,
        }
        headline["summary_rows"] = [
            {"Metric": "seed factors", "Value": headline["n_seed"]},
            {"Metric": "mined factors added to the library", "Value": headline["n_mined"]},
            {"Metric": "library size", "Value": headline["n_library"]},
            {"Metric": "expressions evaluated by the miner", "Value": headline["n_evaluations"]},
            {
                "Metric": "best out-of-sample factor",
                "Value": f"{best_oos} (Rank ICIR {_fmt(headline['best_oos_rank_icir'])})",
            },
            {
                "Metric": "backtest Sharpe · best single factor",
                "Value": f"{_fmt(headline['sharpe_best_single'], '{:.2f}')} ({best_single})",
            },
            {"Metric": "backtest Sharpe · linear composite", "Value": _fmt(headline["sharpe_linear"], "{:.2f}")},
            {"Metric": "backtest Sharpe · ML composite", "Value": _fmt(headline["sharpe_ml"], "{:.2f}")},
            {"Metric": "total time (s)", "Value": f"{elapsed:.1f}"},
        ]

        ctx = {
            "cfg": cfg,
            "store": store,
            "split": split,
            "tables": tables,
            "figure_paths": figure_paths,
            "headline": headline,
            "timings": timings,
            "universe_mean": universe_mean,
            "n_evaluations": miner.n_evaluations,
            "hof": hof,
            "n_hof_valid": n_hof_valid,
            "n_passed": n_passed,
            "n_selected": n_selected,
            "ml_train_window": ml_train_window,
            "ml_importances": ml_importances,
            "max_weight": max_weight,
        }
        paths["report_html"] = render_html_report(
            title="AlphaForge factor-mining research report",
            subtitle=f"{store.assets.size} stocks × {len(store.dates)} trading days · {len(SEED_FACTORS)} seed + {len(mined)} mined factors · "
            f"horizon {cfg.horizon}d lag {cfg.lag}d · seed {cfg.seed}",
            sections=_html_sections(ctx),
            out_path=out_dir / "report.html",
        )

        run_info = {
            "config": cfg.to_dict(),
            "split": {
                k: {"start": str(v[0].date()), "end": str(v[-1].date()), "n": int(len(v))} for k, v in split.items()
            },
            "universe_mean_names": universe_mean,
            "seed_factors": SEED_FACTORS,
            "library": {
                n: {"expr": library.get(n)["expr"], "tags": list(library.get(n)["tags"])} for n in library.names()
            },
            "mined": [
                {"name": f"mined_{i + 1:02d}", "expr": m.expr, "fitness": m.fitness, "generation": m.generation}
                for i, m in enumerate(mined)
            ],
            "mining": {
                "n_evaluations": miner.n_evaluations,
                "hall_of_fame": len(hof),
                "n_hof_valid": n_hof_valid,
                "n_passed": n_passed,
                "n_selected": n_selected,
                "history": history,
            },
            "ml": {
                "train_window": ml_train_window,
                "min_train": ml_min_train,
                "refit_every": ML_REFIT_EVERY,
                "n_refits": n_refits,
                "feature_importances": None if ml_importances is None else ml_importances.to_dict(),
            },
            "backtests": {k: v.to_dict() for k, v in backtests.items()},
            "backtest_max_weight": max_weight,
            "headline": {k: v for k, v in headline.items() if k != "summary_rows"},
            "timings": timings,
            "elapsed_s": elapsed,
            "paths": {k: str(v) for k, v in paths.items()},
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.system(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        paths["run_json"] = out_dir / "run.json"
        paths["run_json"].write_text(json.dumps(_json_safe(run_info), indent=2, ensure_ascii=False), encoding="utf-8")

    timings["total"] = time.perf_counter() - t_run
    prog.say(
        f"run_research done in {timings['total']:.1f}s: {len(mined)} mined factors, library {len(library)}, "
        f"best OOS rank ICIR {_fmt(headline['best_oos_rank_icir'])} ({best_oos}), "
        f"Sharpe best-single {_fmt(headline['sharpe_best_single'], '{:.2f}')} / linear {_fmt(headline['sharpe_linear'], '{:.2f}')} / "
        f"ML {_fmt(headline['sharpe_ml'], '{:.2f}')} -> {paths['report_html']}"
    )
    return {
        "store": store,
        "library": library,
        "reports_train": reports_train,
        "reports_valid": reports_valid,
        "reports_test": reports_test,
        "mined": mined,
        "backtests": backtests,
        "paths": paths,
        "timings": timings,
        # extras
        "hall_of_fame": hof,
        "mining_history": history,
        "tables": tables,
        "composites": composites,
        "split": split,
        "universe": universe,
        "headline": headline,
        "config": cfg,
    }
