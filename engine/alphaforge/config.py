"""Research-run configuration.

This one serialisable object holds every setting of a run (seed, split, miner
budget, costs), so the same object gives the same results again.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from alphaforge.data.simulator import SimConfig

logger = logging.getLogger("alphaforge.config")

__all__ = ["ResearchConfig"]


@dataclass
class ResearchConfig:
    """End-to-end research settings consumed by ``alphaforge.research.run_research``.

    Attributes:
        sim: Synthetic-market parameters (nested :class:`SimConfig`; a plain
            ``dict`` is accepted and coerced, which is what ``from_dict`` relies on).
        horizon: Label holding period ``h`` (days).
        lag: Entry lag between signal and trade (days).
        n_quantiles: Quantile buckets used in evaluation.
        train_frac: Share of dates used for mining / fitting (first block).
        valid_frac: Share of dates used for hall-of-fame validation (second block);
            the remainder ``1 - train_frac - valid_frac`` is the out-of-sample test block.
        population_size: GP population per generation.
        n_generations: GP generations.
        max_depth: Maximum expression-tree depth offered to the miner.
        n_jobs: Parallel fitness workers (``-1`` = all cores, joblib semantics).
        hall_of_fame: Number of best expressions retained by the miner.
        max_corr: Correlation threshold for library de-duplication and greedy selection.
        min_valid_icir_ann: Minimum annualised ICIR on the validation block to keep a mined factor.
        max_factors: Cap on the number of mined factors added to the library.
        cost_bps: One-way transaction cost in basis points for the backtest.
        top_frac: Long (and short) fraction of the universe per date in portfolio construction.
        rebalance_every: Rebalance frequency in days.
        out_dir: Output directory (``data/``, ``library/``, ``tables/``, ``figures/``, ``report.html``, ``run.json``).
        seed: Seed for the miner / ML stages.  Independent of ``sim.seed`` (the market's seed).
    """

    sim: SimConfig = field(default_factory=SimConfig)
    horizon: int = 5
    lag: int = 1
    n_quantiles: int = 10
    train_frac: float = 0.6
    valid_frac: float = 0.2  # remainder = test
    population_size: int = 100
    n_generations: int = 10
    max_depth: int = 4
    n_jobs: int = 1
    hall_of_fame: int = 50
    max_corr: float = 0.7
    min_valid_icir_ann: float = 0.5
    max_factors: int = 15
    cost_bps: float = 10.0
    top_frac: float = 0.1
    rebalance_every: int = 5
    out_dir: str = "outputs"
    seed: int = 42

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if isinstance(self.sim, Mapping):
            self.sim = _sim_from_mapping(self.sim, strict=True)
        elif not isinstance(self.sim, SimConfig):
            raise TypeError(f"sim must be a SimConfig or a mapping, got {type(self.sim).__name__}")
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` on an inconsistent configuration (called by ``__post_init__``)."""
        problems: list[str] = []
        if int(self.horizon) < 1:
            problems.append(f"horizon must be >= 1 (got {self.horizon})")
        if int(self.lag) < 0:
            problems.append(f"lag must be >= 0 (got {self.lag})")
        if int(self.n_quantiles) < 2:
            problems.append(f"n_quantiles must be >= 2 (got {self.n_quantiles})")
        if not 0.0 < float(self.train_frac) < 1.0:
            problems.append(f"train_frac must be in (0, 1) (got {self.train_frac})")
        if not 0.0 <= float(self.valid_frac) < 1.0:
            problems.append(f"valid_frac must be in [0, 1) (got {self.valid_frac})")
        if float(self.train_frac) + float(self.valid_frac) >= 1.0:
            problems.append(
                f"train_frac + valid_frac must be < 1 so a test block remains "
                f"(got {self.train_frac} + {self.valid_frac})"
            )
        for name, lo in (
            ("population_size", 2),
            ("n_generations", 1),
            ("max_depth", 1),
            ("hall_of_fame", 1),
            ("max_factors", 1),
            ("rebalance_every", 1),
        ):
            if int(getattr(self, name)) < lo:
                problems.append(f"{name} must be >= {lo} (got {getattr(self, name)})")
        if int(self.n_jobs) == 0:
            problems.append("n_jobs must be a positive integer or -1 (joblib semantics), got 0")
        if not 0.0 < float(self.max_corr) <= 1.0:
            problems.append(f"max_corr must be in (0, 1] (got {self.max_corr})")
        if float(self.cost_bps) < 0.0:
            problems.append(f"cost_bps must be >= 0 (got {self.cost_bps})")
        if not 0.0 < float(self.top_frac) <= 0.5:
            problems.append(f"top_frac must be in (0, 0.5] (got {self.top_frac})")
        if not str(self.out_dir):
            problems.append("out_dir must be a non-empty path")
        if problems:
            raise ValueError("invalid ResearchConfig: " + "; ".join(problems))

    # ------------------------------------------------------------------ #
    # derived helpers
    # ------------------------------------------------------------------ #
    @property
    def test_frac(self) -> float:
        """Share of dates in the out-of-sample test block (``1 - train_frac - valid_frac``)."""
        return 1.0 - float(self.train_frac) - float(self.valid_frac)

    def split_index(self, n_dates: int) -> tuple[int, int]:
        """Integer split positions ``(train_end, valid_end)`` for ``n_dates`` ordered dates.

        ``dates[:train_end]`` is train, ``dates[train_end:valid_end]`` valid and
        ``dates[valid_end:]`` test (each block non-empty when ``n_dates`` allows).
        """
        n = int(n_dates)
        train_end = int(round(n * float(self.train_frac)))
        valid_end = int(round(n * (float(self.train_frac) + float(self.valid_frac))))
        train_end = min(max(train_end, 1), max(n - 2, 1))
        valid_end = min(max(valid_end, train_end + (1 if self.valid_frac > 0 else 0)), max(n - 1, train_end))
        return train_end, valid_end

    def replace(self, **changes: Any) -> ResearchConfig:
        """Copy with fields replaced (``dataclasses.replace``; re-validated)."""
        return dataclasses.replace(self, **changes)

    # ------------------------------------------------------------------ #
    # (de)serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Plain, JSON-serialisable dict; ``sim`` becomes a nested dict (``dataclasses.asdict``)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, strict: bool = True) -> ResearchConfig:
        """Inverse of :meth:`to_dict`; missing keys take their defaults.

        Args:
            data: Mapping of field names to values; ``"sim"`` may be a
                :class:`SimConfig`, a nested mapping, or absent.
            strict: Raise ``ValueError`` on unknown keys (default).  With
                ``strict=False`` unknown keys are dropped with a warning -- handy
                when reading a ``run.json`` that carries extra bookkeeping keys.

        Returns:
            A validated :class:`ResearchConfig`.
        """
        if not isinstance(data, Mapping):
            raise TypeError(f"from_dict expects a mapping, got {type(data).__name__}")
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in data.items():
            if key in known:
                kwargs[key] = value
            else:
                unknown.append(str(key))
        if unknown:
            msg = f"ResearchConfig.from_dict: unknown keys {sorted(unknown)}"
            if strict:
                raise ValueError(msg)
            logger.warning("%s (ignored)", msg)
        sim = kwargs.get("sim")
        if sim is None:
            kwargs.pop("sim", None)
        elif isinstance(sim, Mapping):
            kwargs["sim"] = _sim_from_mapping(sim, strict=strict)
        return cls(**kwargs)

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write :meth:`to_dict` as JSON (parents created); returns the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=indent, ensure_ascii=False), encoding="utf-8")
        return out

    @classmethod
    def from_json(cls, path: str | Path, *, strict: bool = True) -> ResearchConfig:
        """Read a JSON file written by :meth:`to_json` (or a ``run.json`` ``"config"`` block)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            isinstance(data, Mapping)
            and "config" in data
            and isinstance(data["config"], Mapping)
            and "sim" in data["config"]
        ):
            data = data["config"]  # tolerate a run.json wrapper
        return cls.from_dict(data, strict=strict)


def _sim_from_mapping(sim: Mapping[str, Any], *, strict: bool) -> SimConfig:
    known = {f.name for f in fields(SimConfig)}
    unknown = sorted(str(k) for k in sim if k not in known)
    if unknown:
        msg = f"ResearchConfig: unknown SimConfig keys {unknown}"
        if strict:
            raise ValueError(msg)
        logger.warning("%s (ignored)", msg)
    return SimConfig(**{k: v for k, v in sim.items() if k in known})
