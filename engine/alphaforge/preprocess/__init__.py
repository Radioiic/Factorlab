"""``alphaforge.preprocess`` - factor pre-processing.

Public API::

    preprocess_factor(x, store, *, universe=None, winsor="mad", fill="cs_median",
                      neutralize=("industry", "size"), standardize=True)
"""

from alphaforge.preprocess.pipeline import (
    FILL_METHODS,
    WINSOR_METHODS,
    log_market_cap,
    preprocess_factor,
)

__all__ = ["preprocess_factor", "log_market_cap", "FILL_METHODS", "WINSOR_METHODS"]
