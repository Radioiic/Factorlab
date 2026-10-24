"""``alphaforge.ops`` - vectorised, NaN-aware, no-look-ahead operator library.

Importing this package registers every built-in operator in :data:`REGISTRY`.

Three families:

* :mod:`~alphaforge.ops.elementwise` - ``add``, ``div``, ``log``, ``power`` ...
* :mod:`~alphaforge.ops.ts` - trailing-window ops (``ts_mean``, ``ts_rank``,
  ``ts_corr`` ...); strictly causal, ``min_periods = window``.
* :mod:`~alphaforge.ops.cs` - per-date cross-sectional and group ops
  (``cs_rank``, ``cs_zscore``, ``group_neutralize``, ``cs_neutralize`` ...).

Every operator returns a *new* ``float64`` panel with the input's index/columns.
"""

from alphaforge.ops import cs, elementwise, registry, ts
from alphaforge.ops.cs import (
    cs_demean,
    cs_neutralize,
    cs_quantile_clip,
    cs_rank,
    cs_regress_out,
    cs_scale,
    cs_winsorize,
    cs_zscore,
    group_neutralize,
    group_rank,
    group_zscore,
)
from alphaforge.ops.elementwise import (
    abs_,
    add,
    clip,
    div,
    inverse,
    log,
    max_,
    min_,
    mul,
    neg,
    power,
    sign,
    signed_log,
    signed_sqrt,
    sqrt,
    sub,
)
from alphaforge.ops.registry import (
    ARG_TYPES,
    CATEGORIES,
    REGISTRY,
    OpSpec,
    get_op,
    list_ops,
    register,
    register_all_builtin,
)
from alphaforge.ops.ts import (
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_cov,
    ts_decay_linear,
    ts_delay,
    ts_delta,
    ts_kurt,
    ts_max,
    ts_mean,
    ts_min,
    ts_pct_change,
    ts_product,
    ts_rank,
    ts_skew,
    ts_std,
    ts_sum,
    ts_zscore,
)

register_all_builtin()

__all__ = [
    # registry
    "OpSpec",
    "REGISTRY",
    "ARG_TYPES",
    "CATEGORIES",
    "register",
    "get_op",
    "list_ops",
    "register_all_builtin",
    # submodules
    "elementwise",
    "ts",
    "cs",
    "registry",
    # elementwise
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs_",
    "sign",
    "log",
    "signed_log",
    "sqrt",
    "signed_sqrt",
    "power",
    "inverse",
    "max_",
    "min_",
    "clip",
    # ts
    "ts_delay",
    "ts_delta",
    "ts_pct_change",
    "ts_mean",
    "ts_sum",
    "ts_std",
    "ts_min",
    "ts_max",
    "ts_argmax",
    "ts_argmin",
    "ts_rank",
    "ts_zscore",
    "ts_corr",
    "ts_cov",
    "ts_decay_linear",
    "ts_skew",
    "ts_kurt",
    "ts_product",
    # cs / group
    "cs_rank",
    "cs_zscore",
    "cs_demean",
    "cs_scale",
    "cs_winsorize",
    "cs_quantile_clip",
    "group_neutralize",
    "group_rank",
    "group_zscore",
    "cs_regress_out",
    "cs_neutralize",
]
