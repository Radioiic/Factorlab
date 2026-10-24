"""AlphaForge portfolio layer: construction, cost-aware backtest and a simple optimiser.

Public API:

* :func:`weights_from_scores`, :func:`rebalance_schedule`, :func:`hold_weights`
  (``alphaforge.portfolio.construct``)
* :func:`backtest`, :class:`BacktestResult` (``alphaforge.portfolio.backtest``)
* :func:`optimize_weights` (``alphaforge.portfolio.optimizer``)

Helpers exported for convenience: :func:`cap_and_redistribute`, :func:`rolling_risk`.
This subpackage deliberately imports nothing from ``alphaforge.data`` or
``alphaforge.ops``; any object with ``.dates``, ``.assets``, ``.get(name)`` and
``.has(name)`` works as a store.
"""

from alphaforge.portfolio.backtest import BacktestResult, backtest
from alphaforge.portfolio.construct import (
    cap_and_redistribute,
    hold_weights,
    rebalance_schedule,
    weights_from_scores,
)
from alphaforge.portfolio.optimizer import optimize_weights, rolling_risk

__all__ = [
    "BacktestResult",
    "backtest",
    "cap_and_redistribute",
    "hold_weights",
    "optimize_weights",
    "rebalance_schedule",
    "rolling_risk",
    "weights_from_scores",
]
