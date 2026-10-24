"""``alphaforge.library`` - the de-duplicated factor library.

Public API::

    FactorLibrary(root=None, max_corr=0.7)
    mean_spearman_corr(a, b) -> float     # mean over dates of row-wise Spearman corr
"""

from alphaforge.library.registry import FactorLibrary, mean_spearman_corr

__all__ = ["FactorLibrary", "mean_spearman_corr"]
