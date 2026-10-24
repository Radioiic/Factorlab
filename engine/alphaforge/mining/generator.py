"""Random expression generation and genetic operators.

:class:`ExpressionGenerator` is the *syntactic* half of the genetic miner: it knows which
operators are available (the registry, filtered to ``mineable`` ones), which store fields may
fill a ``"panel"`` or ``"group"`` slot, and how deep a tree may grow.  From that it can

* draw random, well-typed trees (``grow`` / ``full`` initialisation),
* mutate a tree (``point`` / ``subtree`` / ``hoist`` / ``const``),
* recombine two trees by swapping type-compatible sub-trees (``crossover``), and
* simplify a tree (collapse idempotent operators, merge nested delays, order the
  arguments of commutative operators) so that equivalent candidates print identically.

Every tree returned by this class is *valid* against the registry in the sense of
:func:`alphaforge.expr.ast.validate` (given the same ``group_fields``) and has
``depth(tree) <= max_depth``.  A :class:`~alphaforge.expr.ast.Const` never appears in a
panel slot: the miner cannot generate ``add(close, 2)`` because the grammar has no
scalar-panel arithmetic.

The generator makes only well-typed trees from the operator table, and mutation and
crossover reuse the parts of good candidates.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from alphaforge.expr.ast import (
    Call,
    Const,
    Field,
    Node,
    Path,
    replace_subtree,
)
from alphaforge.expr.ast import (
    depth as tree_depth,
)
from alphaforge.expr.printer import to_string
from alphaforge.ops.registry import REGISTRY, OpSpec

logger = logging.getLogger("alphaforge.mining.generator")

__all__ = ["ExpressionGenerator", "MUTATION_KINDS", "TREE_METHODS", "DEFAULT_CATEGORY_WEIGHTS"]

#: Mutation kinds accepted by :meth:`ExpressionGenerator.mutate`.
MUTATION_KINDS: tuple[str, ...] = ("point", "subtree", "hoist", "const")
#: Initialisation methods accepted by :meth:`ExpressionGenerator.random_tree`.
TREE_METHODS: tuple[str, ...] = ("grow", "full")
#: Default sampling weight per operator category.  Cross-sectional, time-series and
#: group operators are favoured over raw element-wise arithmetic:
#: ``cs_rank(ts_delta(...))`` is a factor, ``add(mul(close, open), low)`` is noise.
DEFAULT_CATEGORY_WEIGHTS: dict[str, float] = {"elementwise": 1.0, "ts": 1.5, "cs": 1.5, "group": 1.5}

#: Number of attempts :meth:`ExpressionGenerator.crossover` makes to find a depth-feasible swap.
_CROSSOVER_TRIES = 5
#: Bound on the per-node rewrite loop in :meth:`ExpressionGenerator.simplify`.
_SIMPLIFY_MAX_PASSES = 8

# ---- simplification tables -------------------------------------------------------------
#: ``op(op(x)) == x``.
_INVOLUTIONS: frozenset[str] = frozenset({"neg", "inverse"})
#: ``op(op(x, *rest), *rest) == op(x, *rest)`` (same scalar / group arguments).
_IDEMPOTENT: frozenset[str] = frozenset(
    {
        "cs_rank",
        "cs_zscore",
        "cs_demean",
        "cs_scale",
        "cs_winsorize",
        "abs",
        "sign",
        "group_rank",
        "group_zscore",
        "group_neutralize",
    }
)
#: Rank operators: ``rank(f(x)) == rank(x)`` for any row-wise strictly increasing ``f``.
_RANK_OUTER: frozenset[str] = frozenset({"cs_rank", "group_rank"})
#: Row-wise positive-affine maps (order-preserving within a date).
_ROW_MONOTONE: frozenset[str] = frozenset({"cs_zscore", "cs_demean", "cs_scale"})
#: ``cs_zscore(f(x)) == cs_zscore(x)`` (z-scores are shift / positive-scale invariant per row).
_ZSCORE_INVARIANT: frozenset[str] = frozenset({"cs_demean", "cs_scale"})
#: Operators whose output is never negative, so ``abs`` of them is a no-op.
_NONNEGATIVE: frozenset[str] = frozenset(
    {"abs", "sqrt", "cs_rank", "ts_rank", "ts_std", "ts_argmax", "ts_argmin", "group_rank"}
)


class ExpressionGenerator:
    """Typed random-tree generator plus genetic operators over expression trees.

    Args:
        fields: Store field names allowed in a ``"panel"`` slot (the terminals), e.g.
            ``["open", "high", "low", "close", "volume", "vwap", "returns", "bp"]``.
            Order is irrelevant; duplicates are dropped.
        group_fields: Field names allowed in a ``"group"`` slot (industry codes).  If empty,
            operators with a group slot are excluded from the pool.
        registry: ``name -> OpSpec`` mapping (defaults to the global registry).  Only
            ``mineable`` operators with at least one panel/group argument are used.
        max_depth: Maximum tree depth (``Field`` = 1), enforced by every method.
        seed: Seed of the generator's private ``numpy`` RNG (``self.rng``).
        op_weights: Optional sampling weights.  Keys may be operator names (``"ts_mean"``)
            or categories (``"ts"``); a name-level weight overrides its category weight,
            which in turn overrides :data:`DEFAULT_CATEGORY_WEIGHTS`.  A weight ``<= 0``
            removes the operator from the pool.  Unknown keys raise ``ValueError``.

    Attributes:
        rng: ``numpy.random.Generator`` used for every random decision (reseed by
            assigning a new generator; useful for reproducible experiments).
        ops: The operator pool (sorted by name).
        op_probs: Sampling probability of each entry of ``ops``.

    The search uses only a typed operator pool with set sampling weights, so it spends
    no budget on arithmetic without a market meaning.
    """

    def __init__(
        self,
        fields: Iterable[str],
        *,
        group_fields: Iterable[str] = ("industry",),
        registry: Mapping[str, OpSpec] = REGISTRY,
        max_depth: int = 4,
        seed: int = 0,
        op_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.fields: list[str] = list(dict.fromkeys(str(f) for f in fields))
        if not self.fields:
            raise ValueError("ExpressionGenerator needs at least one terminal field")
        self.group_fields: list[str] = list(dict.fromkeys(str(g) for g in (group_fields or ())))
        self.registry = registry
        self.max_depth = int(max_depth)
        if self.max_depth < 2:
            raise ValueError(f"max_depth must be >= 2 so that a tree can hold an operator, got {max_depth}")
        self.seed = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

        self.ops, weights = self._build_pool(op_weights)
        self.op_probs: np.ndarray = weights / weights.sum()
        self._weights: np.ndarray = weights
        self._by_signature: dict[tuple[str, ...], list[int]] = {}
        for i, spec in enumerate(self.ops):
            self._by_signature.setdefault(tuple(spec.arg_types), []).append(i)
        self._int_choices: tuple[int, ...] = tuple(sorted({v for s in self.ops for v in s.int_choices}))
        self._float_choices: tuple[float, ...] = tuple(sorted({v for s in self.ops for v in s.float_choices}))
        logger.debug(
            "ExpressionGenerator: %d fields, %d group fields, %d ops, max_depth=%d, seed=%s",
            len(self.fields),
            len(self.group_fields),
            len(self.ops),
            self.max_depth,
            seed,
        )

    # ------------------------------------------------------------------ construction
    def _build_pool(self, op_weights: Mapping[str, float] | None) -> tuple[list[OpSpec], np.ndarray]:
        """Filter the registry to the mineable pool and resolve sampling weights."""
        weights_in: dict[str, float] = {str(k): float(v) for k, v in (op_weights or {}).items()}
        known_categories = set(DEFAULT_CATEGORY_WEIGHTS)
        unknown = [k for k in weights_in if k not in self.registry and k not in known_categories]
        if unknown:
            raise ValueError(
                f"op_weights has unknown keys {unknown}; use operator names or categories {sorted(known_categories)}"
            )
        pool: list[OpSpec] = []
        weights: list[float] = []
        for name in sorted(self.registry):
            spec = self.registry[name]
            if not getattr(spec, "mineable", True):
                continue
            arg_types = tuple(spec.arg_types)
            if not any(t in ("panel", "group") for t in arg_types):
                continue  # a scalar-only op could never be a tree node
            if "group" in arg_types and not self.group_fields:
                continue
            w = DEFAULT_CATEGORY_WEIGHTS.get(spec.category, 1.0)
            w = weights_in.get(spec.category, w)
            w = weights_in.get(name, w)
            if not (w > 0.0):
                continue
            pool.append(spec)
            weights.append(w)
        if not pool:
            raise ValueError("no mineable operators available for the given registry / op_weights")
        return pool, np.asarray(weights, dtype=float)

    # ------------------------------------------------------------------ random helpers
    def _choice(self, seq: Sequence[Any]) -> Any:
        """Uniform pick from a non-empty sequence."""
        return seq[int(self.rng.integers(len(seq)))]

    def _sample_op(self, candidates: Sequence[int] | None = None) -> OpSpec:
        """Weighted pick from the pool (or from ``candidates``, indices into the pool)."""
        if candidates is None:
            return self.ops[int(self.rng.choice(len(self.ops), p=self.op_probs))]
        w = self._weights[list(candidates)]
        return self.ops[candidates[int(self.rng.choice(len(candidates), p=w / w.sum()))]]

    def _p_terminal(self, level: int) -> float:
        """Probability that a ``grow`` node at absolute ``level`` (root = 1) is a terminal.

        Rises linearly with depth (``0`` at the root, ``(max_depth-1)/max_depth`` one level
        above the depth cap) so shallow nodes are almost always operators and deep nodes
        usually stop - the classic Koza "grow" behaviour.
        """
        return (level - 1) / self.max_depth

    # ------------------------------------------------------------------ public: generation
    def random_terminal(self, kind: str = "panel") -> Node:
        """Random leaf for an argument slot of type ``kind``.

        Args:
            kind: ``"panel"`` -> ``Field`` from ``fields``; ``"group"`` -> ``Field`` from
                ``group_fields``; ``"int"`` / ``"float"`` -> ``Const`` drawn from the union
                of the pool's ``int_choices`` / ``float_choices``.

        Raises:
            ValueError: on an unknown kind, or ``"group"`` with no group fields.
        """
        if kind == "panel":
            return Field(self._choice(self.fields))
        if kind == "group":
            if not self.group_fields:
                raise ValueError("no group fields configured")
            return Field(self._choice(self.group_fields))
        if kind == "int":
            return Const(self._choice(self._int_choices))
        if kind == "float":
            return Const(self._choice(self._float_choices))
        raise ValueError(f"unknown terminal kind {kind!r}; expected panel/group/int/float")

    def random_tree(self, depth: int | None = None, method: str = "grow") -> Node:
        """Draw a random, valid tree with ``depth(tree) <= depth``.

        Args:
            depth: Depth budget (``1..max_depth``); ``None`` means ``max_depth``.
            method: ``"full"`` places operators at every level until the budget is exhausted
                (all leaves at exactly ``depth``); ``"grow"`` stops early at random with a
                probability that rises with depth.

        Returns:
            For ``depth >= 2`` the root is always an operator call (a bare field is a
            legal but useless factor); ``depth == 1`` returns a random panel field.

        The "ramped half-and-half" method mixes both methods over a range of depths, so
        the first population has many different tree shapes.
        """
        budget = self.max_depth if depth is None else int(depth)
        if not 1 <= budget <= self.max_depth:
            raise ValueError(f"depth must be in 1..{self.max_depth}, got {depth}")
        if method not in TREE_METHODS:
            raise ValueError(f"method must be one of {TREE_METHODS}, got {method!r}")
        if budget == 1:
            return self.random_terminal("panel")
        return self._build(budget, level=1, method=method, terminal_ok=False)

    def _build(self, budget: int, level: int, method: str, terminal_ok: bool) -> Node:
        """Recursive tree builder: ``budget`` levels remain, this node sits at ``level``."""
        if budget <= 1:
            return self.random_terminal("panel")
        if terminal_ok and method == "grow" and self.rng.random() < self._p_terminal(level):
            return self.random_terminal("panel")
        spec = self._sample_op()
        args: list[Node] = []
        for t in spec.arg_types:
            if t == "panel":
                args.append(self._build(budget - 1, level + 1, method, terminal_ok=True))
            elif t == "group":
                args.append(self.random_terminal("group"))
            elif t == "int":
                args.append(Const(self._choice(spec.int_choices)))
            else:  # "float"
                args.append(Const(self._choice(spec.float_choices)))
        return Call(spec.name, tuple(args))

    # ------------------------------------------------------------------ tree inspection
    def _positions(self, node: Node) -> list[tuple[Path, str, Node, str | None]]:
        """Pre-order ``(path, slot_type, subtree, parent_op)`` for every node.

        The root's slot is ``"panel"``; a child's slot is its parent's ``arg_types`` entry.

        Raises:
            ValueError: if the tree uses an operator unknown to the registry.
        """
        out: list[tuple[Path, str, Node, str | None]] = []
        stack: list[tuple[Node, Path, str, str | None]] = [(node, (), "panel", None)]
        while stack:
            cur, path, slot, parent = stack.pop()
            out.append((path, slot, cur, parent))
            if isinstance(cur, Call):
                spec = self.registry.get(cur.op)
                if spec is None:
                    raise ValueError(f"unknown operator {cur.op!r} in {to_string(cur)}")
                types = tuple(spec.arg_types)
                if len(types) != len(cur.args):
                    raise ValueError(f"operator {cur.op!r} arity mismatch in {to_string(cur)}")
                for i in range(len(cur.args) - 1, -1, -1):
                    stack.append((cur.args[i], path + (i,), types[i], cur.op))
        return out

    @staticmethod
    def _nearest(value: float, choices: Sequence[float]) -> float:
        return min(choices, key=lambda c: (abs(c - value), c))

    def _coerce_consts(self, args: tuple[Node, ...], spec: OpSpec) -> tuple[Node, ...]:
        """Snap scalar arguments onto ``spec``'s choice sets (used after an op swap)."""
        new: list[Node] = []
        for arg, t in zip(args, spec.arg_types):
            if isinstance(arg, Const) and t == "int" and spec.int_choices and arg.value not in spec.int_choices:
                new.append(Const(self._nearest(arg.value, spec.int_choices)))
            elif isinstance(arg, Const) and t == "float" and spec.float_choices and arg.value not in spec.float_choices:
                new.append(Const(self._nearest(arg.value, spec.float_choices)))
            else:
                new.append(arg)
        return tuple(new)

    # ------------------------------------------------------------------ public: mutation
    def mutate(self, node: Node, kind: str | None = None) -> Node:
        """Return a mutated copy of ``node`` (the input is never modified).

        Args:
            node: A valid tree with ``depth <= max_depth``.
            kind: One of

                * ``"point"``   - swap one operator for another with the *identical*
                  signature (``ts_mean`` -> ``ts_std``); scalar arguments are snapped onto
                  the new operator's choice set.
                * ``"subtree"`` - replace a random panel-typed sub-tree (possibly the root)
                  by a fresh random tree that fits the remaining depth budget.
                * ``"hoist"``   - replace the tree by one of its own operator sub-trees
                  (shrinks the tree; fights bloat).
                * ``"const"``   - move one window / exponent constant to a neighbouring
                  value of its operator's choice set (``ts_mean(x, 10)`` -> ``ts_mean(x, 20)``).

                ``None`` samples a kind uniformly.  A kind that does not apply to the tree
                (``hoist`` on a leaf, ``const`` without constants ...) falls back to
                ``"subtree"``, which always applies.

        Returns:
            A valid tree with ``depth <= max_depth``.

        Each edit is small and typed (a different window or a sibling operator), so
        the search examines the close variants of a good factor.
        """
        if kind is None:
            kind = self._choice(MUTATION_KINDS)
        if kind not in MUTATION_KINDS:
            raise ValueError(f"unknown mutation kind {kind!r}; expected one of {MUTATION_KINDS}")
        out: Node | None
        if kind == "point":
            out = self._mutate_point(node)
        elif kind == "hoist":
            out = self._mutate_hoist(node)
        elif kind == "const":
            out = self._mutate_const(node)
        else:
            out = None
        if out is None:
            if kind != "subtree":
                logger.debug("mutate(%s) not applicable to %s; falling back to subtree", kind, to_string(node))
            out = self._mutate_subtree(node)
        return out

    def _mutate_point(self, node: Node) -> Node | None:
        cands: list[tuple[Path, Call, list[int]]] = []
        for path, _slot, sub, _parent in self._positions(node):
            if not isinstance(sub, Call):
                continue
            spec = self.registry.get(sub.op)
            if spec is None:
                continue
            alts = [i for i in self._by_signature.get(tuple(spec.arg_types), []) if self.ops[i].name != sub.op]
            if alts:
                cands.append((path, sub, alts))
        if not cands:
            return None
        path, call, alts = self._choice(cands)
        new_spec = self._sample_op(alts)
        new_call = Call(new_spec.name, self._coerce_consts(call.args, new_spec))
        return replace_subtree(node, path, new_call)

    def _mutate_subtree(self, node: Node) -> Node:
        positions = [(p, sub) for p, slot, sub, _ in self._positions(node) if slot == "panel"]
        path, _old = self._choice(positions)
        level = len(path) + 1
        budget = max(1, self.max_depth - len(path))
        new = self._build(budget, level=level, method="grow", terminal_ok=level > 1)
        return replace_subtree(node, path, new)

    def _mutate_hoist(self, node: Node) -> Node | None:
        cands = [sub for p, slot, sub, _ in self._positions(node) if p and slot == "panel" and isinstance(sub, Call)]
        if not cands:
            return None
        return self._choice(cands)

    def _mutate_const(self, node: Node) -> Node | None:
        cands: list[tuple[Path, float, tuple[float, ...]]] = []
        for path, slot, sub, parent in self._positions(node):
            if not isinstance(sub, Const) or slot not in ("int", "float") or parent is None:
                continue
            spec = self.registry.get(parent)
            if spec is None:
                continue
            choices = tuple(spec.int_choices) if slot == "int" else tuple(spec.float_choices)
            if len(choices) >= 2:
                cands.append((path, sub.value, choices))
        if not cands:
            return None
        path, value, choices = self._choice(cands)
        idx = choices.index(self._nearest(value, choices))
        step = 1 if self.rng.random() < 0.5 else -1
        if not 0 <= idx + step < len(choices):
            step = -step
        return replace_subtree(node, path, Const(choices[idx + step]))

    # ------------------------------------------------------------------ public: crossover
    def crossover(self, a: Node, b: Node) -> tuple[Node, Node]:
        """Swap a random panel-typed sub-tree of ``a`` with one of ``b``.

        A sub-tree of ``a`` is picked uniformly; then a panel-typed sub-tree of ``b`` is
        picked uniformly among those for which *both* children stay within ``max_depth``
        (the trivial root-for-root swap is excluded).  If no feasible partner exists the
        pick is retried up to five times, after which the parents are returned unchanged.

        Returns:
            ``(child_a, child_b)``, both valid with ``depth <= max_depth`` (given valid
            parents within the cap).  Inputs are never modified.

        Crossover moves a sub-tree (for example ``ts_pct_change(close, 5)``) from one
        candidate to another.
        """
        pos_a = [(p, sub) for p, slot, sub, _ in self._positions(a) if slot == "panel"]
        pos_b = [(p, sub, tree_depth(sub)) for p, slot, sub, _ in self._positions(b) if slot == "panel"]
        for _ in range(_CROSSOVER_TRIES):
            pa, sa = self._choice(pos_a)
            da, la = tree_depth(sa), len(pa)
            feasible = [
                (pb, sb)
                for pb, sb, db in pos_b
                if db <= self.max_depth - la and da <= self.max_depth - len(pb) and not (la == 0 and len(pb) == 0)
            ]
            if not feasible:
                continue
            pb, sb = self._choice(feasible)
            return replace_subtree(a, pa, sb), replace_subtree(b, pb, sa)
        logger.debug("crossover: no depth-feasible swap after %d tries; returning parents", _CROSSOVER_TRIES)
        return a, b

    # ------------------------------------------------------------------ public: simplify
    def simplify(self, node: Node) -> Node:
        """Best-effort algebraic simplification; always returns a valid tree.

        Rewrites (applied bottom-up to a fixed point at every node):

        * involutions: ``neg(neg(x)) -> x``, ``inverse(inverse(x)) -> x``;
        * idempotent ops with the same extra arguments: ``cs_rank(cs_rank(x)) -> cs_rank(x)``,
          ``cs_zscore(cs_zscore(x))``, ``abs(abs(x))``, ``group_rank(group_rank(x, g), g)`` ...;
        * rank of a row-monotone map: ``cs_rank(cs_zscore(x)) -> cs_rank(x)`` (also
          ``cs_demean`` / ``cs_scale`` inside ``cs_rank`` / ``group_rank``);
        * z-score of a shift / scale: ``cs_zscore(cs_demean(x)) -> cs_zscore(x)``;
        * ``abs`` of a non-negative op or of ``neg``: ``abs(neg(x)) -> abs(x)``,
          ``abs(sqrt(x)) -> sqrt(x)``;
        * nested delays: ``ts_delay(ts_delay(x, a), b) -> ts_delay(x, a + b)`` only when
          ``a + b`` is one of ``ts_delay``'s ``int_choices``;
        * commutative ops get their arguments ordered by canonical string, so
          ``add(b, a)`` and ``add(a, b)`` print (and de-duplicate) identically.

        ``sub(x, x) -> 0`` is deliberately *not* performed: a constant is never valid in a
        panel slot, so such a tree is left alone and rejected by the fitness
        function as a constant factor instead.  Rewrites only remove or reorder nodes, so
        depth never increases.

        Simplification gives equivalent trees one canonical form, so the miner does not
        evaluate ``cs_rank(cs_rank(cs_rank(x)))`` as a new candidate.
        """
        if not isinstance(node, Call):
            return node
        args = tuple(self.simplify(a) for a in node.args)
        cur: Node = Call(node.op, args)
        for _ in range(_SIMPLIFY_MAX_PASSES):
            if not isinstance(cur, Call):
                break
            new = self._rewrite(cur)
            if new == cur:
                break
            cur = new
        return cur

    def _rewrite(self, call: Call) -> Node:
        """One rewrite step at ``call`` (children already simplified)."""
        op, args = call.op, call.args
        spec = self.registry.get(op)
        if spec is None or not args:
            return call
        # commutative canonical ordering
        if getattr(spec, "commutative", False) and len(args) == 2:
            s0, s1 = to_string(args[0]), to_string(args[1])
            if s1 < s0:
                return Call(op, (args[1], args[0]))
        inner = args[0]
        if not isinstance(inner, Call):
            return call
        iop = inner.op
        rest, irest = args[1:], inner.args[1:]
        if op in _INVOLUTIONS and iop == op:
            return inner.args[0]
        if op in _IDEMPOTENT and iop == op and irest == rest:
            return inner
        if op in _RANK_OUTER and iop in _ROW_MONOTONE:
            return Call(op, (inner.args[0],) + rest)
        if op == "cs_zscore" and iop in _ZSCORE_INVARIANT:
            return Call(op, (inner.args[0],))
        if op == "abs":
            if iop == "neg":
                return Call(op, (inner.args[0],))
            if iop in _NONNEGATIVE:
                return inner
        if op == "ts_delay" and iop == "ts_delay":
            a, b = inner.args[1], args[1]
            if isinstance(a, Const) and isinstance(b, Const):
                total = a.value + b.value
                if total in spec.int_choices:
                    return Call(op, (inner.args[0], Const(total)))
        return call

    # ------------------------------------------------------------------ misc
    def __repr__(self) -> str:
        return (
            f"ExpressionGenerator(fields={len(self.fields)}, group_fields={self.group_fields}, "
            f"ops={len(self.ops)}, max_depth={self.max_depth}, seed={self.seed})"
        )
