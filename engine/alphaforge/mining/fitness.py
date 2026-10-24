"""Fitness function for the genetic miner.

:class:`FitnessFunction` turns an expression tree into one scalar the genetic search can
maximise.  For every candidate it

1. evaluates the tree on the store (shared :class:`~alphaforge.expr.evaluator.Evaluator`
   sub-tree cache),
2. slices the factor to the **training window** ``[start, end]`` and pre-processes it
   (:func:`~alphaforge.preprocess.preprocess_factor`: universe mask, winsorise, fill,
   industry/size neutralise, z-score),
3. computes quick statistics against the forward return: rank-IC mean / ICIR (annualised),
   top-quantile turnover, coverage and - if a :class:`~alphaforge.library.FactorLibrary`
   is given - the maximum correlation with the incumbents,
4. combines them::

       fitness = w_icir * |icir_ann| + w_ic * |ic_mean| * 100
                 - w_turnover * turnover
                 - w_corr * max(0, |max_corr| - max_corr) * 10
                 - parsimony * size(node)

The sign of the IC is irrelevant (a factor can always be flipped).  A candidate is
*invalid* (``fitness = -inf``, ``valid = False``, human-readable ``reason``) when evaluation
raises, coverage is below ``min_coverage``, the factor is constant or all-NaN, too few IC
observations exist, or any statistic is non-finite.  Results are cached by canonical
expression string, so re-evaluating an individual that survives to the next generation
is free.

**Train-window discipline.**  Only closes inside ``[start, end]`` ever reach the fitness:
the label is computed from ``store.slice(start, end)``, so the last ``lag + horizon`` days
of the window carry no label rather than peeking at post-``end`` prices.  Factor values,
on the other hand, are computed on the full store and *then* sliced, so time-series
operators may warm up on pre-``start`` history - that is the past, and using it is
legitimate.  Since every operator is causal, nothing after ``end`` can affect a
factor value inside the window; the test-suite verifies that perturbing post-``end`` data
leaves the fitness bit-identical.

The fitness rewards a stable IC (ICIR) and charges turnover, correlation with the library
and size, all on the training window only.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.data.universe import apply_universe
from alphaforge.evaluation.ic import ic_series, ic_stats
from alphaforge.evaluation.report import FactorReport, evaluate_factor
from alphaforge.evaluation.turnover import quantile_turnover
from alphaforge.expr.ast import Node
from alphaforge.expr.ast import size as tree_size
from alphaforge.expr.evaluator import Evaluator
from alphaforge.expr.parser import parse
from alphaforge.expr.printer import to_string
from alphaforge.preprocess.pipeline import preprocess_factor

logger = logging.getLogger("alphaforge.mining.fitness")

__all__ = ["FitnessResult", "FitnessFunction", "IC_METHODS"]

NAN = float("nan")
NEG_INF = float("-inf")

#: Correlation methods accepted for the IC that drives the fitness.
IC_METHODS: tuple[str, ...] = ("spearman", "pearson")

#: Maximum length of the ``reason`` string stored for an invalid candidate.
_REASON_MAX = 240


@dataclass
class FitnessResult:
    """Outcome of one fitness evaluation.

    Attributes:
        fitness: The scalar the miner maximises; ``-inf`` for an invalid candidate.
        valid: ``False`` when the candidate was rejected (see ``reason``).
        reason: Why it was rejected (empty when valid).
        ic_mean: Mean of the IC series selected by ``ic_method`` (rank IC by default).
        icir: Per-period ICIR (``mean / std``) of that series.
        rank_ic_mean: Mean Spearman (rank) IC - always the rank version.
        turnover: Mean top-quantile turnover (fraction of new names per day).
        max_corr: Signed correlation with the most-correlated library factor
            (``NaN`` when no library was given).
        max_corr_name: Name of that library factor.
        coverage: Mean over dates of ``#factor values / #labels``.
        complexity: ``size(node)`` (number of tree nodes).
        report: Full :class:`~alphaforge.evaluation.report.FactorReport` when the fitness
            function was built with ``full_report=True``, else ``None``.
        icir_ann: Annualised ICIR (``icir * sqrt(252)``) - the quantity in the fitness.
        n_obs: Number of dates with a defined IC.
        expr: Canonical expression string (the cache key).
    """

    fitness: float
    valid: bool
    reason: str = ""
    ic_mean: float = NAN
    icir: float = NAN
    rank_ic_mean: float = NAN
    turnover: float = NAN
    max_corr: float = NAN
    max_corr_name: str | None = None
    coverage: float = NAN
    complexity: int = 0
    report: FactorReport | None = field(default=None, repr=False)
    icir_ann: float = NAN
    n_obs: int = 0
    expr: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Scalar fields only (no ``report``) - convenient for tables."""
        return {f.name: getattr(self, f.name) for f in dc_fields(self) if f.name != "report"}


class FitnessFunction:
    """Callable ``Node -> FitnessResult`` implementing the miner's fitness.

    Args:
        store: :class:`~alphaforge.data.DataStore` with at least ``close`` (labels) and, for
            the default pre-processing, ``industry`` and ``market_cap``.
        evaluator: Optional shared :class:`~alphaforge.expr.evaluator.Evaluator`; one is
            created on the store when ``None``.  Pass one to share its sub-tree cache with
            other stages (it must be built on the same store).
        library: Optional :class:`~alphaforge.library.FactorLibrary`; when given, the
            candidate's max |correlation| with the incumbents (on the train window) is
            penalised above ``max_corr``.
        horizon: Holding period of the label (days).
        lag: Entry lag of the label (days).
        start: First date of the **training window** (inclusive), or ``None``.
        end: Last date of the training window (inclusive), or ``None``.  Labels are
            computed inside the window only (see the module docstring).
        universe: Optional 1.0/NaN investability mask (full-store grid is fine; it is
            aligned onto the window).
        preprocess: Apply :func:`~alphaforge.preprocess.preprocess_factor` to the raw
            factor.  When ``False`` the raw values are used (masked by ``universe`` if given).
        w_icir: Weight of ``|icir_ann|``.
        w_ic: Weight of ``|ic_mean| * 100``.
        w_turnover: Weight of the mean top-quantile turnover.
        w_corr: Weight of ``max(0, |max_corr| - max_corr) * 10``.
        max_corr: Correlation with the library above which the penalty kicks in.
        parsimony: Penalty per tree node.
        min_coverage: Candidates with lower coverage are invalid.
        min_ic_abs: Candidates with ``|ic_mean|`` below this are invalid (0 = off).
        full_report: Also build a :class:`~alphaforge.evaluation.report.FactorReport`
            (slower; stored on the result).
        n_quantiles: Buckets used for the turnover statistic (and the report).
        ic_method: ``"spearman"`` (rank IC, default) or ``"pearson"`` - which IC series
            feeds ``ic_mean`` / ``icir`` / ``icir_ann`` and hence the fitness.
        min_periods: Minimum number of dates with a defined IC; fewer -> invalid.
        preprocess_kwargs: Extra keyword arguments for
            :func:`~alphaforge.preprocess.preprocess_factor` (e.g. ``neutralize=()`` for a
            store without industry / market cap).
        cache: Memoise results by canonical expression string.
        cache_size: Sub-tree cache size of the auto-created evaluator.

    The object is picklable (``joblib`` / ``loky`` workers): the evaluator, its cache, the
    result cache and the derived label panel are dropped on pickling and rebuilt lazily in
    the receiving process.

    Attributes:
        window: ``DatetimeIndex`` of the training window.
        stats: ``{"calls", "hits", "evaluated", "invalid"}`` counters.
    """

    def __init__(
        self,
        store: Any,
        *,
        evaluator: Evaluator | None = None,
        library: Any | None = None,
        horizon: int = 5,
        lag: int = 1,
        start: Any = None,
        end: Any = None,
        universe: pd.DataFrame | None = None,
        preprocess: bool = True,
        w_icir: float = 1.0,
        w_ic: float = 0.0,
        w_turnover: float = 0.5,
        w_corr: float = 1.0,
        max_corr: float = 0.7,
        parsimony: float = 0.01,
        min_coverage: float = 0.5,
        min_ic_abs: float = 0.0,
        full_report: bool = False,
        n_quantiles: int = 10,
        ic_method: str = "spearman",
        min_periods: int = 20,
        preprocess_kwargs: dict[str, Any] | None = None,
        cache: bool = True,
        cache_size: int = 256,
    ) -> None:
        self.store = store
        self.library = library
        self.horizon = int(horizon)
        self.lag = int(lag)
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if self.lag < 0:
            raise ValueError(f"lag must be >= 0, got {lag}")
        self.start = None if start is None else pd.Timestamp(start)
        self.end = None if end is None else pd.Timestamp(end)
        if universe is not None and not isinstance(universe, pd.DataFrame):
            raise TypeError("universe must be a pandas DataFrame mask or None")
        self.universe = universe
        self.preprocess = bool(preprocess)
        self.w_icir = float(w_icir)
        self.w_ic = float(w_ic)
        self.w_turnover = float(w_turnover)
        self.w_corr = float(w_corr)
        self.max_corr = float(max_corr)
        self.parsimony = float(parsimony)
        self.min_coverage = float(min_coverage)
        self.min_ic_abs = float(min_ic_abs)
        self.full_report = bool(full_report)
        self.n_quantiles = int(n_quantiles)
        if self.n_quantiles < 2:
            raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
        ic_method = str(ic_method).lower()
        if ic_method not in IC_METHODS:
            raise ValueError(f"ic_method must be one of {IC_METHODS}, got {ic_method!r}")
        self.ic_method = ic_method
        self.min_periods = int(min_periods)
        self.preprocess_kwargs: dict[str, Any] = dict(preprocess_kwargs or {})
        self.cache_size = int(cache_size)

        dates = pd.DatetimeIndex(store.dates)
        self._indexer = dates.slice_indexer(self.start, self.end)
        self.window: pd.DatetimeIndex = dates[self._indexer]
        if len(self.window) == 0:
            raise ValueError(f"empty training window [{start}, {end}] on a store spanning {dates[0]}..{dates[-1]}")
        if evaluator is not None and getattr(evaluator, "store", store) is not store:
            logger.warning("FitnessFunction: the given evaluator was built on a different store object")
        self._evaluator: Evaluator | None = evaluator
        self._train_store: Any | None = None
        self._fwd: pd.DataFrame | None = None
        self._cache: dict[str, FitnessResult] | None = {} if cache else None
        self.stats: dict[str, int] = {"calls": 0, "hits": 0, "evaluated": 0, "invalid": 0}
        logger.debug(
            "FitnessFunction: window %s..%s (%d dates), horizon=%d lag=%d, preprocess=%s, library=%s",
            self.window[0].date(),
            self.window[-1].date(),
            len(self.window),
            self.horizon,
            self.lag,
            self.preprocess,
            None if library is None else len(library),
        )

    # ------------------------------------------------------------------ pickling
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Caches and derived objects are process-local: rebuilt lazily by the receiver.
        state["_evaluator"] = None
        state["_train_store"] = None
        state["_fwd"] = None
        state["_cache"] = {} if self._cache is not None else None
        state["stats"] = {k: 0 for k in self.stats}
        return state

    # ------------------------------------------------------------------ lazy pieces
    @property
    def evaluator(self) -> Evaluator:
        """The evaluator in use (created on first access)."""
        if self._evaluator is None:
            self._evaluator = Evaluator(self.store, cache_size=self.cache_size)
        return self._evaluator

    @property
    def train_store(self) -> Any:
        """``store.slice(start, end)`` (the store itself when no window is set)."""
        if self._train_store is None:
            if self.start is None and self.end is None:
                self._train_store = self.store
            elif hasattr(self.store, "slice"):
                self._train_store = self.store.slice(self.start, self.end)
            else:  # duck-typed store: fall back to the full-store label sliced to the window
                logger.warning("store has no slice(); labels near the window end may use post-end closes")
                self._train_store = self.store
        return self._train_store

    @property
    def labels(self) -> pd.DataFrame:
        """Forward-return label restricted to the training window (NaN at its tail)."""
        if self._fwd is None:
            fwd = self.train_store.fwd_return(self.horizon, self.lag)
            if not fwd.index.equals(self.window):
                fwd = fwd.reindex(index=self.window)
            self._fwd = fwd
        return self._fwd

    def clear_cache(self) -> None:
        """Drop memoised results (and the evaluator's sub-tree cache)."""
        if self._cache is not None:
            self._cache.clear()
        if self._evaluator is not None:
            self._evaluator.clear_cache()

    @property
    def cache_len(self) -> int:
        """Number of memoised results."""
        return 0 if self._cache is None else len(self._cache)

    # ------------------------------------------------------------------ public API
    def evaluate_factor_values(self, node: Node | str) -> pd.DataFrame:
        """Evaluate ``node`` -> raw panel -> train-window slice -> pre-processed factor.

        Args:
            node: Expression tree (or text, which is parsed).

        Returns:
            A new ``float64`` panel on ``window x store.assets``: pre-processed when
            ``preprocess`` is on, otherwise the raw values masked by ``universe`` (if any).

        Raises:
            ExprEvalError / ValueError / KeyError: propagated from evaluation and
                pre-processing (``__call__`` catches them and returns an invalid result).

        This is the helper the research pipeline reuses to obtain train-window values of
        the hall of fame, so its output is *exactly* what the fitness was computed on.
        """
        if isinstance(node, str):
            node = parse(node)
        # A miner evaluates thousands of arbitrary expressions; floating-point overflow /
        # 0-by-0 inside a *candidate* (ts_product(amount, 20) ~ 1e160, squared in a z-score)
        # is expected and simply yields inf / NaN, which the validity checks below handle.
        # Scoping errstate here keeps those candidates from spamming RuntimeWarnings
        # without hiding warnings from anywhere else.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            raw = self.evaluator.eval(node)
            if raw.index.equals(pd.DatetimeIndex(self.store.dates)):
                win = raw.iloc[self._indexer]
            else:
                win = raw.reindex(index=self.window)
            if self.preprocess:
                out = preprocess_factor(win, self.store, universe=self.universe, **self.preprocess_kwargs)
            else:
                out = win.astype(np.float64, copy=True)
                if self.universe is not None:
                    out = apply_universe(out, self.universe)
        arr = out.to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(arr).any():  # normalise: the returned panel never carries +-inf
            out = out.where(np.isfinite(arr))
        return out

    def __call__(self, node: Node | str) -> FitnessResult:
        """Score one candidate; never raises (failures give ``fitness = -inf``)."""
        if isinstance(node, str):
            node = parse(node)
        elif not isinstance(node, Node):
            raise TypeError(f"expected an expression Node or str, got {type(node).__name__}")
        key = to_string(node)
        self.stats["calls"] += 1
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                self.stats["hits"] += 1
                return hit
        self.stats["evaluated"] += 1
        try:
            result = self._evaluate(node, key)
        except Exception as exc:  # noqa: BLE001 - any failure is "this candidate is invalid"
            reason = f"{type(exc).__name__}: {exc}"
            logger.debug("fitness: %s failed: %s", key, reason)
            result = self._invalid(key, node, reason)
        if not result.valid:
            self.stats["invalid"] += 1
        if self._cache is not None:
            self._cache[key] = result
        return result

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _invalid(key: str, node: Node, reason: str, **partial: Any) -> FitnessResult:
        return FitnessResult(
            fitness=NEG_INF,
            valid=False,
            reason=reason[:_REASON_MAX],
            complexity=tree_size(node),
            expr=key,
            **partial,
        )

    def _evaluate(self, node: Node, key: str) -> FitnessResult:
        complexity = tree_size(node)
        values = self.evaluate_factor_values(node)
        arr = values.to_numpy(dtype=np.float64, na_value=np.nan)
        finite = np.isfinite(arr)
        if not finite.any():
            return self._invalid(key, node, "factor is all-NaN on the training window")
        vals = arr[finite]
        if vals.max() == vals.min():
            return self._invalid(key, node, "factor is constant on the training window")

        fwd = self.labels
        n_f = finite.sum(axis=1).astype(float)
        n_r = np.isfinite(fwd.to_numpy(dtype=np.float64, na_value=np.nan)).sum(axis=1).astype(float)
        has_label = n_r > 0
        coverage = float((n_f[has_label] / n_r[has_label]).mean()) if has_label.any() else NAN
        if not (coverage >= self.min_coverage):
            return self._invalid(
                key, node, f"coverage {coverage:.3f} < min_coverage {self.min_coverage}", coverage=coverage
            )

        ric = ic_series(values, fwd, method="spearman")
        sel = ric if self.ic_method == "spearman" else ic_series(values, fwd, method="pearson")
        st = ic_stats(sel)
        rank_ic_mean = float(ric.mean()) if ric.notna().any() else NAN
        n_obs = int(st["n"])
        if n_obs < self.min_periods:
            return self._invalid(
                key,
                node,
                f"only {n_obs} IC observations < min_periods {self.min_periods}",
                coverage=coverage,
                rank_ic_mean=rank_ic_mean,
                n_obs=n_obs,
            )
        ic_mean, icir, icir_ann = float(st["mean"]), float(st["ir"]), float(st["ir_ann"])
        if abs(ic_mean) < self.min_ic_abs:
            return self._invalid(
                key,
                node,
                f"|ic_mean| {abs(ic_mean):.4f} < min_ic_abs {self.min_ic_abs}",
                ic_mean=ic_mean,
                icir=icir,
                icir_ann=icir_ann,
                rank_ic_mean=rank_ic_mean,
                coverage=coverage,
                n_obs=n_obs,
            )

        turn = quantile_turnover(values, self.n_quantiles).dropna()
        turnover = float(turn.mean()) if len(turn) else NAN

        max_corr, max_corr_name = NAN, None
        corr_penalty = 0.0
        if self.library is not None and len(self.library) > 0:
            max_corr_name, c = self.library.max_corr(values)
            max_corr = float(c)
            if not math.isfinite(max_corr):
                logger.debug("fitness: %s has no overlap with the library; corr penalty skipped", key)
                max_corr, max_corr_name = NAN, None
            else:
                corr_penalty = max(0.0, abs(max_corr) - self.max_corr) * 10.0
        elif self.library is not None:
            max_corr = 0.0

        fitness = (
            self.w_icir * abs(icir_ann)
            + self.w_ic * abs(ic_mean) * 100.0
            - self.w_turnover * turnover
            - self.w_corr * corr_penalty
            - self.parsimony * complexity
        )
        partial = dict(
            ic_mean=ic_mean,
            icir=icir,
            icir_ann=icir_ann,
            rank_ic_mean=rank_ic_mean,
            turnover=turnover,
            max_corr=max_corr,
            max_corr_name=max_corr_name,
            coverage=coverage,
            n_obs=n_obs,
        )
        if not math.isfinite(fitness):
            return self._invalid(key, node, "non-finite statistics (icir / turnover / corr)", **partial)

        report = None
        if self.full_report:
            report = evaluate_factor(
                values,
                self.train_store,
                name=key,
                expr=key,
                horizon=self.horizon,
                lag=self.lag,
                n_quantiles=self.n_quantiles,
                start=self.start,
                end=self.end,
            )
        return FitnessResult(
            fitness=float(fitness), valid=True, reason="", complexity=complexity, report=report, expr=key, **partial
        )

    def __repr__(self) -> str:
        return (
            f"FitnessFunction(window={self.window[0].date()}..{self.window[-1].date()}, horizon={self.horizon}, "
            f"lag={self.lag}, ic_method={self.ic_method!r}, preprocess={self.preprocess}, "
            f"library={'none' if self.library is None else len(self.library)}, cached={self.cache_len})"
        )
