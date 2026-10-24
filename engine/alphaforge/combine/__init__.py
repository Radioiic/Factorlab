"""``alphaforge.combine`` - factor combination.

Public API::

    orient(values, ic_mean)
    combine_linear(factors, *, weights="equal", ic_table=None)
    resolve_weights(names, weights, ic_table)          # helper: the normalised weights
    MLCombiner(*, train_window=250, refit_every=21, min_train=120, model="hgb", seed=0, **model_kwargs)
"""

from alphaforge.combine.linear import WEIGHT_SCHEMES, combine_linear, orient, resolve_weights
from alphaforge.combine.ml import MODELS, MLCombiner

__all__ = ["orient", "combine_linear", "resolve_weights", "WEIGHT_SCHEMES", "MLCombiner", "MODELS"]
