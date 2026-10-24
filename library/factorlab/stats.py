"""Statistics of the factor library: the Benjamini-Hochberg q-values.

``engine/libfactors.py`` uses ``bh_qvalues`` for the q-values of its screens.
"""

from __future__ import annotations

import numpy as np


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Return the Benjamini-Hochberg q-values of the p-values p, in the same order."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out
