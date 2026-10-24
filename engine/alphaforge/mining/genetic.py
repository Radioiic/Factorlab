"""Genetic-programming factor miner.

:class:`GeneticMiner` runs a classic generational GP loop over expression trees:

1. **Initialise** a population with ramped half-and-half (depths ``2..max_depth``,
   ``grow`` / ``full`` alternating), unique by canonical expression string.
2. Each generation: **evaluate** each individual that it has not seen (in parallel with
   ``joblib``/``loky`` when ``n_jobs > 1``), update the **hall of fame** (the best unique
   individuals ever seen, optionally greedy-filtered by correlation), record **history**,
   log, run the callback, check **early stopping**.
3. Breed the next population: **elitism** (top ``elite_frac`` copied), then
   **tournament selection** feeding **crossover** (prob ``crossover_p``), **mutation**
   (prob ``mutation_p``) or **reproduction**; children are simplified and de-duplicated.

Every random decision goes through ``numpy.random.default_rng(seed)`` (the generator has
its own seeded RNG), so a run is reproducible for a given ``(seed, generator.seed)``
irrespective of ``n_jobs``.

The miner evaluates each candidate under the same rules as a hand-written factor: train
window only, with turnover and novelty penalties.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from alphaforge.expr.ast import Node
from alphaforge.expr.ast import depth as tree_depth
from alphaforge.expr.printer import to_string
from alphaforge.mining.dedup import spearman_corr_panel
from alphaforge.mining.fitness import FitnessResult
from alphaforge.mining.generator import ExpressionGenerator

logger = logging.getLogger("alphaforge.mining.genetic")

__all__ = ["MinedFactor", "GeneticMiner", "HISTORY_KEYS"]

#: Keys present in every ``GeneticMiner.history`` entry.
HISTORY_KEYS: tuple[str, ...] = (
    "gen",
    "best",
    "mean",
    "median",
    "n_valid",
    "n_unique",
    "best_expr",
    "elapsed_s",
    "hof_size",
    "hof_best",
    "n_evaluated",
)

NEG_INF = float("-inf")


@dataclass
class MinedFactor:
    """One individual of the search: an expression with its fitness.

    Attributes:
        expr: Canonical expression string (``to_string(node)``), the identity used for
            de-duplication.
        node: The expression tree.
        fitness: ``result.fitness`` (``-inf`` when invalid).
        generation: Generation (0-based) in which this expression was first evaluated.
        result: The full :class:`~alphaforge.mining.fitness.FitnessResult`.
    """

    expr: str
    node: Node = field(repr=False)
    fitness: float
    generation: int
    result: FitnessResult = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Scalars for tables: ``expr``, ``fitness``, ``generation`` plus the result's scalars."""
        d = {"expr": self.expr, "fitness": self.fitness, "generation": self.generation}
        for k, v in self.result.to_dict().items():
            if k not in d:
                d[k] = v
        return d


def _safe_fitness(fitness: Callable[[Node], FitnessResult], node: Node) -> FitnessResult:
    """Call ``fitness`` and turn any exception into an invalid result (workers never crash)."""
    try:
        res = fitness(node)
    except Exception as exc:  # noqa: BLE001 - deliberately catch everything from user callables
        return FitnessResult(
            fitness=NEG_INF, valid=False, reason=f"{type(exc).__name__}: {exc}"[:240], expr=to_string(node)
        )
    if not isinstance(res, FitnessResult):
        return FitnessResult(
            fitness=NEG_INF, valid=False, reason=f"fitness returned {type(res).__name__}, not FitnessResult"
        )
    f = res.fitness
    if f is None or not isinstance(f, (int, float, np.floating)) or math.isnan(float(f)):
        return FitnessResult(
            fitness=NEG_INF, valid=False, reason=f"fitness value {f!r} is not a number", expr=to_string(node)
        )
    return res


class GeneticMiner:
    """Generational GP search over expression trees.

    Args:
        generator: :class:`~alphaforge.mining.generator.ExpressionGenerator` (defines the
            search space, ``max_depth`` and the genetic operators).
        fitness: ``Node -> FitnessResult`` callable, normally a
            :class:`~alphaforge.mining.fitness.FitnessFunction`.  Must be picklable when
            ``n_jobs > 1`` (``FitnessFunction`` is).  If it exposes
            ``evaluate_factor_values(node)`` the hall of fame can be correlation-filtered.
        population_size: Individuals per generation (``>= 2``).
        n_generations: Number of generations (``>= 1``); generation 0 is the initial population.
        crossover_p: Probability that a breeding event is a crossover.
        mutation_p: Probability that it is a mutation (the remainder is reproduction).
        tournament_k: Tournament size for parent selection.
        elite_frac: Fraction of the population copied unchanged into the next one.
        hall_of_fame: Size of the hall of fame (best unique individuals ever seen).
        hof_max_corr: If given, the hall of fame is greedily filtered so that no two members
            have ``|mean row-wise Spearman corr| > hof_max_corr`` on the fitness's values.
        seed: RNG seed for selection / breeding decisions.
        n_jobs: Parallel workers for fitness evaluation (``joblib`` ``loky`` backend).
        early_stop_rounds: Stop when the best fitness has not improved for this many
            consecutive generations (``None`` = never).
        log_every: Log a generation summary every ``log_every`` generations (the last one
            is always logged).
        max_init_tries: Attempts per slot to draw a *new* expression during initialisation
            before accepting a duplicate.
        max_child_tries: Consecutive duplicate children tolerated during breeding before a
            duplicate is accepted (bounds the loop on tiny search spaces).
        hof_scan_factor: At most ``hof_scan_factor * hall_of_fame`` candidates are examined
            per generation by the correlation filter (bounds its cost).

    Attributes:
        history: One dict per generation with keys :data:`HISTORY_KEYS`.
        hof: The hall of fame after :meth:`run` (sorted by fitness desc).
        population: The last evaluated population (``MinedFactor`` list).
        best: Best individual ever seen (``None`` before :meth:`run`).
        n_evaluations: Number of distinct expressions evaluated.
        stopped_early: Whether early stopping triggered.
    """

    def __init__(
        self,
        generator: ExpressionGenerator,
        fitness: Callable[[Node], FitnessResult],
        *,
        population_size: int = 100,
        n_generations: int = 10,
        crossover_p: float = 0.6,
        mutation_p: float = 0.3,
        tournament_k: int = 3,
        elite_frac: float = 0.05,
        hall_of_fame: int = 50,
        hof_max_corr: float | None = None,
        seed: int = 0,
        n_jobs: int = 1,
        early_stop_rounds: int | None = None,
        log_every: int = 1,
        max_init_tries: int = 20,
        max_child_tries: int = 10,
        hof_scan_factor: int = 4,
    ) -> None:
        if not callable(fitness):
            raise TypeError("fitness must be callable")
        self.generator = generator
        self.fitness = fitness
        self.population_size = int(population_size)
        self.n_generations = int(n_generations)
        self.crossover_p = float(crossover_p)
        self.mutation_p = float(mutation_p)
        self.tournament_k = int(tournament_k)
        self.elite_frac = float(elite_frac)
        self.hall_of_fame = int(hall_of_fame)
        self.hof_max_corr = None if hof_max_corr is None else float(hof_max_corr)
        self.seed = seed
        self.n_jobs = int(n_jobs)
        self.early_stop_rounds = None if early_stop_rounds is None else int(early_stop_rounds)
        self.log_every = max(1, int(log_every))
        self.max_init_tries = max(1, int(max_init_tries))
        self.max_child_tries = max(1, int(max_child_tries))
        self.hof_scan_factor = max(1, int(hof_scan_factor))
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {population_size}")
        if self.n_generations < 1:
            raise ValueError(f"n_generations must be >= 1, got {n_generations}")
        if not (
            0.0 <= self.crossover_p <= 1.0
            and 0.0 <= self.mutation_p <= 1.0
            and self.crossover_p + self.mutation_p <= 1.0
        ):
            raise ValueError(f"crossover_p + mutation_p must lie in [0, 1], got {crossover_p} + {mutation_p}")
        if self.tournament_k < 1:
            raise ValueError(f"tournament_k must be >= 1, got {tournament_k}")
        if not 0.0 <= self.elite_frac <= 1.0:
            raise ValueError(f"elite_frac must be in [0, 1], got {elite_frac}")
        if self.hall_of_fame < 1:
            raise ValueError(f"hall_of_fame must be >= 1, got {hall_of_fame}")
        if self.hof_max_corr is not None and not 0.0 <= self.hof_max_corr <= 1.0:
            raise ValueError(f"hof_max_corr must be in [0, 1], got {hof_max_corr}")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be a non-zero integer (negative = joblib convention)")
        if self.early_stop_rounds is not None and self.early_stop_rounds < 1:
            raise ValueError(f"early_stop_rounds must be >= 1 or None, got {early_stop_rounds}")

        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.history: list[dict[str, Any]] = []
        self.hof: list[MinedFactor] = []
        self.population: list[MinedFactor] = []
        self.best: MinedFactor | None = None
        self.n_evaluations: int = 0
        self.stopped_early: bool = False
        self._results: dict[str, FitnessResult] = {}
        self._seen: dict[str, MinedFactor] = {}  # every valid unique individual, first occurrence
        self._values_cache: dict[str, Any] = {}
        self._pair_corr: dict[tuple[str, str], float] = {}
        self._corr_filter_disabled = False

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, nodes: Sequence[Node], generation: int) -> tuple[int, list[MinedFactor]]:
        """Evaluate ``nodes`` (memoised by expression) and wrap them as ``MinedFactor``.

        Returns:
            ``(n_newly_evaluated, individuals)`` - one ``MinedFactor`` per input node, in order.
        """
        exprs = [to_string(n) for n in nodes]
        todo: dict[str, Node] = {}
        for e, n in zip(exprs, nodes):
            if e not in self._results and e not in todo:
                todo[e] = n
        if todo:
            keys = list(todo)
            if self.n_jobs == 1 or len(keys) == 1:
                results = [_safe_fitness(self.fitness, todo[k]) for k in keys]
            else:
                results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed(_safe_fitness)(self.fitness, todo[k]) for k in keys
                )
            for k, r in zip(keys, results):
                self._results[k] = r
            self.n_evaluations += len(keys)
        out: list[MinedFactor] = []
        for e, n in zip(exprs, nodes):
            r = self._results[e]
            if r.valid and e not in self._seen:
                self._seen[e] = MinedFactor(expr=e, node=n, fitness=float(r.fitness), generation=generation, result=r)
            out.append(
                self._seen.get(e)
                or MinedFactor(expr=e, node=n, fitness=float(r.fitness), generation=generation, result=r)
            )
        return len(todo), out

    # ------------------------------------------------------------------ hall of fame
    def _factor_values(self, m: MinedFactor) -> Any | None:
        v = self._values_cache.get(m.expr)
        if v is None:
            fn = getattr(self.fitness, "evaluate_factor_values", None)
            if fn is None:
                return None
            try:
                v = fn(m.node)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "hall of fame: could not compute values of %s (%s: %s); kept unfiltered",
                    m.expr,
                    type(exc).__name__,
                    exc,
                )
                return None
            self._values_cache[m.expr] = v
        return v

    def _corr(self, a: MinedFactor, b: MinedFactor) -> float:
        key = (a.expr, b.expr) if a.expr <= b.expr else (b.expr, a.expr)
        c = self._pair_corr.get(key)
        if c is None:
            va, vb = self._factor_values(a), self._factor_values(b)
            c = 0.0 if va is None or vb is None else spearman_corr_panel(va, vb)
            c = 0.0 if not math.isfinite(c) else c
            self._pair_corr[key] = c
        return c

    def _update_hof(self) -> list[MinedFactor]:
        ranked = sorted(self._seen.values(), key=lambda m: (-m.fitness, m.expr))
        if self.hof_max_corr is None or self._corr_filter_disabled:
            return ranked[: self.hall_of_fame]
        if getattr(self.fitness, "evaluate_factor_values", None) is None:
            logger.warning(
                "hof_max_corr given but the fitness has no evaluate_factor_values(); hall of fame is not correlation-filtered"
            )
            self._corr_filter_disabled = True
            return ranked[: self.hall_of_fame]
        kept: list[MinedFactor] = []
        examined: set[str] = set()
        limit = self.hof_scan_factor * self.hall_of_fame
        for cand in ranked:
            if len(kept) >= self.hall_of_fame or len(examined) >= limit:
                break
            examined.add(cand.expr)
            if all(abs(self._corr(cand, k)) <= self.hof_max_corr for k in kept):
                kept.append(cand)
        # keep the value cache bounded to what the filter can still ask for
        self._values_cache = {e: v for e, v in self._values_cache.items() if e in examined}
        return kept

    # ------------------------------------------------------------------ breeding
    def _tournament(self, population: list[MinedFactor]) -> MinedFactor:
        idx = self.rng.integers(0, len(population), size=self.tournament_k)
        best = population[int(idx[0])]
        for i in idx[1:]:
            cand = population[int(i)]
            if cand.fitness > best.fitness:
                best = cand
        return best

    def _initial_population(self) -> list[Node]:
        gen = self.generator
        depths = list(range(2, gen.max_depth + 1))
        nodes: list[Node] = []
        seen: set[str] = set()
        n_dup = 0
        for i in range(self.population_size):
            depth = depths[i % len(depths)]
            method = "grow" if (i // len(depths)) % 2 == 0 else "full"
            node = None
            for _ in range(self.max_init_tries):
                cand = gen.simplify(gen.random_tree(depth, method))
                if to_string(cand) not in seen:
                    node = cand
                    break
            if node is None:
                node = cand
                n_dup += 1
            seen.add(to_string(node))
            nodes.append(node)
        if n_dup:
            logger.info("initial population: %d duplicate(s) accepted after %d tries each", n_dup, self.max_init_tries)
        return nodes

    def _next_population(self, population: list[MinedFactor]) -> list[Node]:
        gen = self.generator
        ranked = sorted(population, key=lambda m: (-m.fitness, m.expr))
        n_elite = int(round(self.elite_frac * self.population_size))
        nodes: list[Node] = []
        exprs: set[str] = set()
        for m in ranked:
            if len(nodes) >= n_elite:
                break
            if m.fitness > NEG_INF and m.expr not in exprs:
                nodes.append(m.node)
                exprs.add(m.expr)
        dup_streak = 0
        while len(nodes) < self.population_size:
            r = self.rng.random()
            if r < self.crossover_p:
                a, b = self._tournament(population), self._tournament(population)
                children = list(gen.crossover(a.node, b.node))
            elif r < self.crossover_p + self.mutation_p:
                children = [gen.mutate(self._tournament(population).node)]
            else:
                children = [self._tournament(population).node]
            for child in children:
                if len(nodes) >= self.population_size:
                    break
                child = gen.simplify(child)
                if tree_depth(child) > gen.max_depth:  # defensive: operators respect the cap
                    child = gen.mutate(child, "hoist")
                key = to_string(child)
                if key in exprs and dup_streak < self.max_child_tries:
                    dup_streak += 1
                    continue
                dup_streak = 0
                exprs.add(key)
                nodes.append(child)
        return nodes

    # ------------------------------------------------------------------ run
    def run(self, callback: Callable[[int, list[MinedFactor]], None] | None = None) -> list[MinedFactor]:
        """Run the search and return the hall of fame (fitness descending, unique exprs).

        Args:
            callback: Optional ``callback(gen, hof)`` invoked after every generation with
                the 0-based generation index and the current hall of fame.

        Returns:
            The hall of fame: up to ``hall_of_fame`` valid, unique-by-expression individuals
            sorted by fitness descending (correlation-filtered when ``hof_max_corr`` is set).
            Empty only if no candidate was ever valid.
        """
        t_run = time.perf_counter()
        self.history = []
        self.stopped_early = False
        nodes = self._initial_population()
        best_ever = NEG_INF
        stall = 0
        logger.info(
            "GeneticMiner: pop=%d gens=%d max_depth=%d ops=%d fields=%d n_jobs=%d seed=%s",
            self.population_size,
            self.n_generations,
            self.generator.max_depth,
            len(self.generator.ops),
            len(self.generator.fields),
            self.n_jobs,
            self.seed,
        )
        for gen in range(self.n_generations):
            t0 = time.perf_counter()
            n_new, population = self._evaluate(nodes, gen)
            self.population = population
            self.hof = self._update_hof()
            self.best = self.hof[0] if self.hof else None

            valid = [m.fitness for m in population if m.fitness > NEG_INF]
            gen_best = max(population, key=lambda m: (m.fitness, m.expr)) if population else None
            entry = {
                "gen": gen,
                "best": float(gen_best.fitness) if gen_best is not None else float("nan"),
                "mean": float(np.mean(valid)) if valid else float("nan"),
                "median": float(statistics.median(valid)) if valid else float("nan"),
                "n_valid": len(valid),
                "n_unique": len({m.expr for m in population}),
                "best_expr": gen_best.expr if gen_best is not None else "",
                "elapsed_s": time.perf_counter() - t0,
                "hof_size": len(self.hof),
                "hof_best": float(self.best.fitness) if self.best is not None else float("nan"),
                "n_evaluated": n_new,
            }
            self.history.append(entry)
            if gen % self.log_every == 0 or gen == self.n_generations - 1:
                logger.info(
                    "gen %d/%d: best %.4f  mean %.4f  valid %d/%d  unique %d  hof %d  new %d  [%.1fs]  %s",
                    gen,
                    self.n_generations - 1,
                    entry["best"],
                    entry["mean"],
                    entry["n_valid"],
                    len(population),
                    entry["n_unique"],
                    len(self.hof),
                    n_new,
                    entry["elapsed_s"],
                    entry["best_expr"],
                )
            if callback is not None:
                callback(gen, self.hof)

            hof_best = entry["hof_best"]
            if hof_best > best_ever + 1e-12:
                best_ever, stall = hof_best, 0
            else:
                stall += 1
            if self.early_stop_rounds is not None and stall >= self.early_stop_rounds:
                logger.info("early stopping at generation %d: no improvement for %d generation(s)", gen, stall)
                self.stopped_early = True
                break
            if gen < self.n_generations - 1:
                nodes = self._next_population(population)

        logger.info(
            "GeneticMiner done: %d generations, %d evaluations, hall of fame %d, best %.4f (%s) [%.1fs]",
            len(self.history),
            self.n_evaluations,
            len(self.hof),
            self.best.fitness if self.best else float("nan"),
            self.best.expr if self.best else "-",
            time.perf_counter() - t_run,
        )
        return list(self.hof)

    def __repr__(self) -> str:
        return (
            f"GeneticMiner(pop={self.population_size}, gens={self.n_generations}, cx={self.crossover_p}, "
            f"mut={self.mutation_p}, k={self.tournament_k}, elite={self.elite_frac}, hof={self.hall_of_fame}, "
            f"hof_max_corr={self.hof_max_corr}, n_jobs={self.n_jobs}, seed={self.seed})"
        )
