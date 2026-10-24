"""``alphaforge.mining`` - genetic-programming factor miner.

Public API::

    ExpressionGenerator(fields, *, group_fields=("industry",), registry=REGISTRY, max_depth=4, seed=0, op_weights=None)
    FitnessFunction(store, *, evaluator=None, library=None, horizon=5, lag=1, start=None, end=None, ...)
    FitnessResult                     # dataclass returned by the fitness
    GeneticMiner(generator, fitness, *, population_size=100, n_generations=10, ...).run(callback=None)
    MinedFactor                       # hall-of-fame entry
    select_uncorrelated(candidates, values, *, max_corr=0.7, key=lambda m: m.fitness)
    spearman_corr_panel(a, b) -> float
"""

from alphaforge.mining.dedup import select_uncorrelated, spearman_corr_panel
from alphaforge.mining.fitness import IC_METHODS, FitnessFunction, FitnessResult
from alphaforge.mining.generator import (
    DEFAULT_CATEGORY_WEIGHTS,
    MUTATION_KINDS,
    TREE_METHODS,
    ExpressionGenerator,
)
from alphaforge.mining.genetic import HISTORY_KEYS, GeneticMiner, MinedFactor

__all__ = [
    "ExpressionGenerator",
    "MUTATION_KINDS",
    "TREE_METHODS",
    "DEFAULT_CATEGORY_WEIGHTS",
    "FitnessFunction",
    "FitnessResult",
    "IC_METHODS",
    "GeneticMiner",
    "MinedFactor",
    "HISTORY_KEYS",
    "select_uncorrelated",
    "spearman_corr_panel",
]
