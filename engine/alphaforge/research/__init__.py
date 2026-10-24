"""``alphaforge.research`` - end-to-end research orchestration.

Public API::

    run_research(cfg: ResearchConfig, *, log_fn=None) -> dict
    SEED_FACTORS                      # {name: expression} classical seed factors
    split_dates(cfg, dates)           # {"train", "valid", "test"} -> DatetimeIndex
    factor_values(node, evaluator, store, universe)   # eval + preprocess on the full range
"""

from alphaforge.research.pipeline import (
    BACKTEST_MAX_WEIGHT,
    MINING_FIELDS,
    MINING_GROUP_FIELDS,
    SEED_FACTORS,
    factor_values,
    run_research,
    split_dates,
)

__all__ = [
    "run_research",
    "SEED_FACTORS",
    "MINING_FIELDS",
    "MINING_GROUP_FIELDS",
    "BACKTEST_MAX_WEIGHT",
    "split_dates",
    "factor_values",
]
