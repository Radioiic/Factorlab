"""FINDER_REGISTRY: the library's operators as AlphaForge OpSpec entries, in a registry of
Finder's own.

LABEL-FREE: operators and tree utilities only; nothing here reads data.

WHY A SEPARATE REGISTRY. The AlphaForge miner (alphaforge.mining.ExpressionGenerator and
GeneticMiner, unchanged) takes its operator table as an argument. Finder passes
FINDER_REGISTRY, so a generated tree uses exactly the library's operators and prints and
evaluates like a library definition. alphaforge.ops.REGISTRY is never changed: nothing
here registers into it (test_finder_libops.py hashes it before and after).

LAZY IMPORT. alphaforge is imported the first time registry() (or a function that needs
it) is called, never at module import, so `finder.py --check` imports no alphaforge
module (test_finder_isolation.py checks that).

WHAT IS REGISTERED (eval_config.toml [generator.settings]):
    every operator of library/factorlab/operators.py (46), the library's arithmetic
    (`+`, `-`, `*` and unary minus as add, sub, mul and neg; `/` is the library's div),
    and nothing else. The library's `**` and `bench()` are not registered (no catalog
    definition uses `**`; `bench` broadcasts BTC's value to every coin, and the
    generator's terminals already carry the benchmark and the market: `btc_ret` and
    `mkt_ret` themselves, `ex_btc_ret` and `ex_ret` the excess over them). `btc_ret` and
    `mkt_ret` are equal for every coin on a date, so a tree whose leaves are all
    market-wide is constant per date and fails the label-free G0 check (probe.py); the
    generator draws them like any terminal, and they rank coins only next to a coin's
    own series (ts_beta, ts_corr, ts_cov, ts_resid, or arithmetic with one).
    Not mineable: exp, clip, pct_change, ts_maxdd, ts_maxru, cs_mean, cs_wmean (still
    registered, so a tree that contains one validates and evaluates, but the generator
    never draws them).
    Windows: the library generator's own windows within 3-90 days (3, 5, 7, 10, 14, 20,
    30, 45, 60, 90), with the library generator's per-operator minimum (ts_skew >= 10,
    ts_kurt and ts_autocorr >= 14, ts_slope, ts_std and ts_zscore >= 5) and its
    correlation windows (>= 5) for the two-series operators. Lags (delay, delta,
    pct_change): 1, 2, 3, 5, 7, 10, 14, 20, 30.
    ts_autocorr is registered with its window only: the lag is the library's default 1
    (other lags are ts_corr(x, delay(x, k), w)); to_library() prints it as
    ts_autocorr(x, w, 1), the catalog's own spelling.

TREE LIMITS ([generator.settings]): depth <= 3 (alphaforge.expr.ast.depth: a leaf is 1),
at most 20 nodes (alphaforge size: every node, constants included) and nested lookback
<= 180 days (nested_lookback()).

POINT IN TIME, AND WHAT "BIT-IDENTICAL" MEANS. Every registered operator is causal: a
value on day D depends on inputs dated <= D only. test_finder_libops.py checks it three
ways. (1) VALUE mutation (later values replaced, every later NaN kept where it was):
earlier outputs are bit-identical. (2) NaN-pattern-changing mutation (later cells emptied
or filled) and (3) TRUNCATION (later rows cut off, the real-time case): the finite mask
is identical and the values agree within the library screen's PIT tolerance (rtol 1e-9,
atol 1e-12), but they are NOT always bit-identical: decay_linear, ts_slope and ts_rsq
hand the library's _rolling_np only the complete windows, as one matrix-vector product
whose BLAS rounding depends on the number of rows, so another count of complete windows
can change the last bits (about 3e-14 on the test panels, up to about 4e-12 on the
engine's terminals; the tie-snapped ranks are unchanged). Bit-identity is therefore
guaranteed only under value mutation. Making those reductions shape-independent would
change library values, so the library operators are used as they are.
"""

from __future__ import annotations

import hashlib
from functools import cache, lru_cache

from factorfinder import config_io as K

WINDOWS = (3, 5, 7, 10, 14, 20, 30, 45, 60, 90)
LAGS = (1, 2, 3, 5, 7, 10, 14, 20, 30)
ARITHMETIC = ("add", "sub", "mul", "neg")

# name -> (arg kinds, category, commutative). Arg kinds: "p" panel, "w" window, "l" lag,
# "f" float. Categories follow alphaforge's {"elementwise", "ts", "cs"}.
_SIGNATURES: dict[str, tuple[tuple[str, ...], str, bool]] = {
    # arithmetic (library binary operators and unary minus)
    "add": (("p", "p"), "elementwise", True),
    "sub": (("p", "p"), "elementwise", False),
    "mul": (("p", "p"), "elementwise", True),
    "neg": (("p",), "elementwise", False),
    # element-wise
    "div": (("p", "p"), "elementwise", False),
    "log": (("p",), "elementwise", False),
    "exp": (("p",), "elementwise", False),
    "slog": (("p",), "elementwise", False),
    "ssqrt": (("p",), "elementwise", False),
    "abs": (("p",), "elementwise", False),
    "sign": (("p",), "elementwise", False),
    "square": (("p",), "elementwise", False),
    "emax": (("p", "p"), "elementwise", True),
    "emin": (("p", "p"), "elementwise", True),
    "gt": (("p", "p"), "elementwise", False),
    "lt": (("p", "p"), "elementwise", False),
    "clip": (("p", "f", "f"), "elementwise", False),
    # time series: lags
    "delay": (("p", "l"), "ts", False),
    "delta": (("p", "l"), "ts", False),
    "pct_change": (("p", "l"), "ts", False),
    # time series: windows
    "ts_sum": (("p", "w"), "ts", False),
    "ts_mean": (("p", "w"), "ts", False),
    "ts_std": (("p", "w"), "ts", False),
    "ts_median": (("p", "w"), "ts", False),
    "ts_max": (("p", "w"), "ts", False),
    "ts_min": (("p", "w"), "ts", False),
    "ts_skew": (("p", "w"), "ts", False),
    "ts_kurt": (("p", "w"), "ts", False),
    "ts_zscore": (("p", "w"), "ts", False),
    "ts_minmax": (("p", "w"), "ts", False),
    "ema": (("p", "w"), "ts", False),
    "decay_linear": (("p", "w"), "ts", False),
    "ts_rank": (("p", "w"), "ts", False),
    "ts_argmax": (("p", "w"), "ts", False),
    "ts_argmin": (("p", "w"), "ts", False),
    "ts_slope": (("p", "w"), "ts", False),
    "ts_rsq": (("p", "w"), "ts", False),
    "ts_pos_frac": (("p", "w"), "ts", False),
    "ts_autocorr": (("p", "w"), "ts", False),
    "ts_maxdd": (("p", "w"), "ts", False),
    "ts_maxru": (("p", "w"), "ts", False),
    # time series: two series and a window
    "ts_corr": (("p", "p", "w"), "ts", False),
    "ts_cov": (("p", "p", "w"), "ts", False),
    "ts_beta": (("p", "p", "w"), "ts", False),
    "ts_resid": (("p", "p", "w"), "ts", False),
    # cross-sectional
    "cs_rank": (("p",), "cs", False),
    "cs_zscore": (("p",), "cs", False),
    "cs_demean": (("p",), "cs", False),
    "cs_mean": (("p",), "cs", False),
    "cs_wmean": (("p", "p"), "cs", False),
}
#: the library generator's minimum-window keys (factorlab.generator._MIN_WINDOW)
_MIN_WINDOW_KEY = {
    "ts_skew": "skew",
    "ts_kurt": "kurt",
    "ts_autocorr": "autocorr",
    "ts_slope": "slope",
    "ts_std": "std",
    "ts_zscore": "zscore",
}
_KIND = {"p": "panel", "w": "int", "l": "int", "f": "float"}
#: library DSL infix forms of the arithmetic operators
_INFIX = {"add": "+", "sub": "-", "mul": "*"}


class TreeLimitError(ValueError):
    """A tree breaks a generator limit; `code` is depth, nodes, nested_lookback or
    non_mineable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- config


def defaults(cfg: dict | None = None) -> dict:
    """[generator.settings], checked against the window constants of this module."""
    cfg = K.load() if cfg is None else cfg
    d = cfg["generator"]["settings"]
    if (d["window_min_days"], d["window_max_days"]) != (min(WINDOWS), max(WINDOWS)):
        raise ValueError("[generator.settings] windows differ from libops.WINDOWS")
    return {
        "non_mineable": tuple(d["non_mineable"]),
        "max_depth": int(d["max_depth"]),
        "max_nodes": int(d["max_nodes"]),
        "max_nested_lookback": int(d["max_nested_lookback_days"]),
    }


def non_mineable() -> tuple[str, ...]:
    return defaults()["non_mineable"]


def _library():
    import libfactors  # noqa: F401 - puts the repository's library/ on sys.path
    from factorlab import generator as LG
    from factorlab.operators import OPERATORS

    return OPERATORS, LG


def window_choices(name: str) -> tuple[int, ...]:
    """The window values `name` may take: WINDOWS, from the library's minimum for that
    operator, and from the library's correlation windows for the two-series operators."""
    _, LG = _library()
    lo = LG._MIN_WINDOW.get(_MIN_WINDOW_KEY.get(name, ""), min(WINDOWS))
    pool = WINDOWS
    if _SIGNATURES[name][0] == ("p", "p", "w"):
        pool = tuple(w for w in WINDOWS if w in LG.CORR_WINDOWS)
    out = tuple(w for w in pool if w >= lo)
    if not out or not set(out) <= set(LG.WINDOWS):
        raise ValueError(f"no library window for {name}")
    return out


# --------------------------------------------------------------------------- registry


def _func(name: str):
    ops, _ = _library()
    if name == "add":
        return lambda a, b: a + b
    if name == "sub":
        return lambda a, b: a - b
    if name == "mul":
        return lambda a, b: a * b
    if name == "neg":
        return lambda a: -a
    if name == "ts_autocorr":
        f = ops["ts_autocorr"]
        return lambda x, w: f(x, w)  # the library's default lag, 1
    return ops[name]


@lru_cache(maxsize=1)
def registry() -> dict:
    """FINDER_REGISTRY: {name: alphaforge OpSpec}. Built once, on first call; imports
    alphaforge.ops.registry (for the OpSpec class) without registering anything there."""
    from alphaforge.ops.registry import OpSpec

    ops, _ = _library()
    missing = set(ops) - set(_SIGNATURES)
    if missing:
        raise ValueError(f"library operators without a Finder signature: {sorted(missing)}")
    nm = set(non_mineable())
    if not nm <= set(ops):
        raise ValueError("a non-mineable name is not a library operator")
    out = {}
    for name, (kinds, category, comm) in _SIGNATURES.items():
        ints: tuple[int, ...] = ()
        if "w" in kinds:
            ints = window_choices(name)
        elif "l" in kinds:
            ints = LAGS
        out[name] = OpSpec(
            name,
            _func(name),
            tuple(_KIND[k] for k in kinds),
            category,
            int_choices=ints or (1,),
            float_choices=(0.0, 1.0),
            commutative=comm,
            mineable=name not in nm,
        )
    return out


def __getattr__(name: str):
    # FINDER_REGISTRY as a lazily built module attribute (PEP 562)
    if name == "FINDER_REGISTRY":
        return registry()
    raise AttributeError(name)


def spec_digest(reg) -> str:
    """sha256 over every spec's observable fields (used to prove alphaforge.ops.REGISTRY
    is unchanged): name, arg types, category, choices, flags and the function's
    module-qualified name."""
    rows = []
    for name in sorted(reg):
        s = reg[name]
        f = s.func
        rows.append(
            "|".join(
                [
                    name,
                    ",".join(s.arg_types),
                    s.category,
                    ",".join(map(str, s.int_choices)),
                    ",".join(map(str, s.float_choices)),
                    str(s.commutative),
                    str(s.mineable),
                    f"{getattr(f, '__module__', '')}.{getattr(f, '__qualname__', '')}",
                ]
            )
        )
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


# --------------------------------------------------------------------------- trees


def _ast():
    from alphaforge.expr import ast as A

    return A


@cache
def terminal_lookback(name: str) -> int:
    """Days of history a terminal consumes (the library's max_lookback of its derived
    expression; deriv_share_chg and carry_ret use one day's change, the other two none)."""
    from factorfinder.fieldmap import FACTORLAB_TERMINALS

    if name in FACTORLAB_TERMINALS:
        return {"sbasis": 0, "perp_range_excess": 0, "deriv_share_chg": 1, "carry_ret": 1}[name]
    _library()
    from factorlab.expr import max_lookback

    return int(max_lookback(name))


def nested_lookback(node) -> int:
    """Days of history a tree consumes, by the library's max_lookback convention: along
    every path from the root to a leaf, add each operator's window or lag (its integer
    argument) and the leaf's own lookback; the tree's value is the largest path sum.
    ts_autocorr adds its window plus its lag of 1 (the library's max_lookback counts
    max(window, lag) for it, one day less)."""
    A = _ast()
    if isinstance(node, A.Field):
        return terminal_lookback(node.name)
    if isinstance(node, A.Const):
        return 0
    own = max((int(a.value) for a in node.args if isinstance(a, A.Const) and node.op != "clip"), default=0)
    if node.op == "ts_autocorr":
        own += 1
    inner = max((nested_lookback(a) for a in node.args if not isinstance(a, A.Const)), default=0)
    return own + inner


def tree_stats(node) -> dict:
    A = _ast()
    ops = sorted({n.op for n, _ in A.iter_nodes(node) if isinstance(n, A.Call)})
    fields = sorted(A.fields_used(node))
    return {
        "depth": A.depth(node),
        "nodes": A.size(node),
        "lookback": nested_lookback(node),
        "ops": ops,
        "fields": fields,
    }


def check_limits(node, lim: dict | None = None) -> None:
    """Raise TreeLimitError if `node` breaks a limit or uses a non-mineable operator."""
    lim = defaults() if lim is None else lim
    s = tree_stats(node)
    if s["depth"] > lim["max_depth"]:
        raise TreeLimitError("depth", f"depth {s['depth']} > {lim['max_depth']}")
    if s["nodes"] > lim["max_nodes"]:
        raise TreeLimitError("nodes", f"{s['nodes']} nodes > {lim['max_nodes']}")
    if s["lookback"] > lim["max_nested_lookback"]:
        raise TreeLimitError("nested_lookback", f"nested lookback {s['lookback']} > {lim['max_nested_lookback']}")
    bad = set(s["ops"]) & set(lim["non_mineable"])
    if bad:
        raise TreeLimitError("non_mineable", f"non-mineable operator(s) {sorted(bad)}")


def to_string(node) -> str:
    """alphaforge's canonical prefix form (the cache and dedup key)."""
    from alphaforge.expr.printer import to_string as ts

    return ts(node)


def to_library(node) -> str:
    """The tree in the library's expression language (library/factorlab/expr.py): the
    arithmetic as infix, ts_autocorr with its lag, every other operator as a call. For a
    tree whose terminals are library names, factorlab's parser accepts it and its
    Evaluator gives the same values as evaluate()."""
    A = _ast()
    if isinstance(node, A.Field):
        return node.name
    if isinstance(node, A.Const):
        from alphaforge.expr.printer import format_const

        return format_const(node.value)
    args = [to_library(a) for a in node.args]
    if node.op in _INFIX:
        return f"({args[0]} {_INFIX[node.op]} {args[1]})"
    if node.op == "neg":
        return f"(-{args[0]})"
    if node.op == "ts_autocorr":
        args = args + ["1"]
    return f"{node.op}({', '.join(args)})"


def evaluator(terminals: dict, cache_size: int = 256):
    """An alphaforge Evaluator over the terminal panels with FINDER_REGISTRY."""
    from alphaforge.expr.evaluator import Evaluator

    return Evaluator(terminals, cache_size=cache_size, registry=registry())


def evaluate(node, terminals: dict | None = None, ev=None):
    """The tree's (day x perp) values: alphaforge's Evaluator with FINDER_REGISTRY, and,
    like the library's Evaluator, +-inf replaced by NaN on the result."""
    import warnings

    import numpy as np

    ev = evaluator(terminals) if ev is None else ev
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        out = ev.eval(node)
    return out.replace([np.inf, -np.inf], np.nan)


def generator(fields, seed: int, max_depth: int | None = None):
    """alphaforge's ExpressionGenerator, unchanged, over FINDER_REGISTRY: no group fields,
    the default category weights, and the depth cap of [generator.settings]."""
    from alphaforge.mining.generator import ExpressionGenerator

    depth = defaults()["max_depth"] if max_depth is None else int(max_depth)
    return ExpressionGenerator(list(fields), group_fields=(), registry=registry(), max_depth=depth, seed=int(seed))


def draw_trees(gen, n: int, lim: dict | None = None, max_tries: int | None = None) -> tuple[list, dict]:
    """`n` random trees, ramped half-and-half: depth budgets cycle over 2..max_depth and
    the method alternates between "grow" and "full" (the classic initialisation; depth 1
    would be a bare terminal). A draw that breaks a limit is discarded and redrawn (the
    count is returned). Deterministic for the generator's seed."""
    lim = defaults() if lim is None else lim
    depths = list(range(2, lim["max_depth"] + 1))
    out, rejected, i = [], {}, 0
    max_tries = 50 * n if max_tries is None else max_tries
    while len(out) < n:
        if i >= max_tries:
            raise RuntimeError(f"drew {i} trees for {n} valid ones")
        d = depths[(i // 2) % len(depths)]
        m = "grow" if i % 2 == 0 else "full"
        i += 1
        t = gen.random_tree(depth=d, method=m)
        try:
            check_limits(t, lim)
        except TreeLimitError as exc:
            rejected[exc.code] = rejected.get(exc.code, 0) + 1
            continue
        out.append(t)
    return out, {"draws": i, "rejected": rejected}
