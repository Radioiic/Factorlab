"""Event-driven network-diffusion factors over the correlation graph.

THE IDEA IN ONE PARAGRAPH
A shock does not arrive everywhere at once. When one asset takes an unusual
move, that move propagates to the assets it is tightly coupled with, and it
takes time to get there. This module measures the shock energy currently
*flowing toward* each asset from its graph neighbours, so it asks a question
none of the existing factors ask: "who is about to be hit by something that
has already happened somewhere else?" That is genuinely different from
momentum (own past), carry (own funding) or volatility (own dispersion).

The correlation metric is a Ledoit-Wolf shrinkage correlation of the daily
returns (_shrinkage_correlation below).

CONSTRUCTION
    1. EVENT.   Residual return after removing the equal-weight market, then
                z-scored against its own trailing window. |z| > threshold is
                an event; below it the asset contributes nothing.
    2. GRAPH.   Shrinkage correlation over ``corr_window`` completed bars,
                turned into non-negative weights by the chosen kernel and made
                row-stochastic, so each asset receives a weighted average of
                its neighbours rather than a sum that scales with degree.
    3. DIFFUSE. Sum neighbour shocks over the last ``lags`` days with
                geometric ``decay``, normalized by the decay mass so the
                number is comparable across parameter choices.

    signed mode keeps the shock's direction (directional contagion);
    abs mode uses magnitude only (stress diffusion, direction-agnostic).

POINT-IN-TIME
A row stamped D uses the correlation of bars ending at D and neighbour shocks
from D-1 .. D-lags. Nothing from D+1 on enters, and the self-weight is
forced to zero so an asset can never diffuse its own shock back into itself.
``test_diffusion.py`` proves both by construction and by future-mutation.

The factor lab (factor_lab.py) builds a grid of these specifications in memory.
This module has no command line and writes no file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

KERNELS = ("corr", "invdist", "knn")
MODES = ("signed", "abs")
MIN_ASSETS = 4
# the checks of the shrinkage correlation of one return window
_MIN_CORR_ASSETS = 3
_MIN_RETURN_STD = 1e-12


def _shrinkage_correlation(window: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf covariance standardized into a checked correlation matrix."""
    x = window.to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < _MIN_CORR_ASSETS:
        raise ValueError("return window is too small for geometry")
    if not np.isfinite(x).all():
        raise ValueError("return window contains non-finite values")
    covariance = LedoitWolf().fit(x).covariance_
    scale = np.sqrt(np.diag(covariance))
    if np.any(scale <= _MIN_RETURN_STD) or not np.isfinite(scale).all():
        raise ValueError("return window contains a constant or invalid asset")
    corr = covariance / np.outer(scale, scale)
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    # Shrinkage should be PSD.  Treat a material negative eigenvalue as a broken
    # invariant rather than projecting it away and hiding the source of the error.
    if float(np.linalg.eigvalsh(corr).min()) < -1e-8:
        raise ValueError("shrinkage correlation matrix is not positive semidefinite")
    return corr


@dataclass(frozen=True)
class DiffusionSpec:
    """One fully-specified diffusion factor. Hashable -> usable as a registry key."""

    corr_window: int = 90
    z_window: int = 30
    threshold: float = 2.0
    kernel: str = "corr"
    knn: int = 5
    decay: float = 0.7
    lags: int = 5
    mode: str = "signed"

    def name(self) -> str:
        k = self.kernel if self.kernel != "knn" else f"knn{self.knn}"
        return f"dif_{self.mode}_{k}_w{self.corr_window}_t{self.threshold:g}_d{self.decay:g}_l{self.lags}"

    def validate(self) -> None:
        if self.kernel not in KERNELS:
            raise ValueError(f"unknown kernel {self.kernel!r}")
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}")
        if not (0 < self.decay <= 1):
            raise ValueError("decay must be in (0, 1]")
        if self.lags < 1 or self.corr_window < 20 or self.z_window < 10:
            raise ValueError("window/lag parameters are out of range")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")


def shock_series(returns: pd.DataFrame, z_window: int, threshold: float, mode: str) -> pd.DataFrame:
    """Per-asset event magnitudes: residual return z-scores past ``threshold``.

    The equal-weight market return is removed first so that a day when
    everything falls together is NOT recorded as 20 separate events -- that is
    a market move, not a diffusible idiosyncratic shock.
    """
    market = returns.mean(axis=1)
    resid = returns.sub(market, axis=0)
    mu = resid.rolling(z_window).mean()
    sd = resid.rolling(z_window).std(ddof=0)
    z = (resid - mu) / sd.replace(0.0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan)
    hit = z.abs() >= threshold
    out = z.where(hit, 0.0)
    if mode == "abs":
        out = out.abs()
    return out.fillna(0.0)


def graph_weights(corr: np.ndarray, kernel: str, knn: int) -> np.ndarray:
    """Row-stochastic, zero-diagonal neighbour weights from a correlation matrix."""
    n = len(corr)
    if kernel == "corr":
        w = np.clip(corr, 0.0, None)
    elif kernel == "invdist":
        d = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr)))
        with np.errstate(divide="ignore"):
            w = np.where(d > 1e-9, 1.0 / d, 0.0)
    else:  # knn on correlation strength
        w = np.clip(corr, 0.0, None).copy()
        keep = max(1, min(knn, n - 1))
        for i in range(n):
            row = w[i].copy()
            row[i] = -np.inf
            cut = np.partition(row, -keep)[-keep]
            w[i] = np.where(row >= cut, w[i], 0.0)
    np.fill_diagonal(w, 0.0)  # no self-diffusion, ever
    total = w.sum(axis=1, keepdims=True)
    return np.divide(w, total, out=np.zeros_like(w), where=total > 0)


def build(
    prices: pd.DataFrame, specs: list[DiffusionSpec], *, corr_cache: dict | None = None, progress: bool = False
) -> pd.DataFrame:
    """Long-format (date, symbol, <factor columns>) for every spec.

    ``corr_cache`` is keyed by (corr_window, date) so a grid of specs sharing a
    window pays for the shrinkage estimator once instead of once per spec.
    """
    for s in specs:
        s.validate()
    prices = prices.sort_index()
    returns = prices.pct_change(fill_method=None)
    cache = corr_cache if corr_cache is not None else {}

    windows = sorted({s.corr_window for s in specs})
    max_need = max(max(s.corr_window, s.z_window + s.lags) for s in specs)
    dates = returns.index[max_need:]

    shocks = {(s.z_window, s.threshold, s.mode): shock_series(returns, s.z_window, s.threshold, s.mode) for s in specs}

    records: dict[tuple, dict] = {}
    for step, date in enumerate(dates, 1):
        pos = returns.index.get_loc(date)
        for cw in windows:
            key = (cw, date)
            if key in cache:
                continue
            sample = returns.iloc[pos - cw + 1 : pos + 1]
            active = [c for c in sample.columns if sample[c].notna().all() and float(sample[c].std(ddof=0)) > 1e-12]
            if len(active) < MIN_ASSETS:
                cache[key] = (None, [])
                continue
            try:
                cache[key] = (_shrinkage_correlation(sample[active]), active)
            except ValueError:
                cache[key] = (None, [])

        for s in specs:
            corr, active = cache[(s.corr_window, date)]
            if corr is None:
                continue
            w = graph_weights(corr, s.kernel, s.knn)
            sh = shocks[(s.z_window, s.threshold, s.mode)]
            acc = np.zeros(len(active))
            mass = 0.0
            for lag in range(1, s.lags + 1):
                if pos - lag < 0:
                    break
                weight = s.decay ** (lag - 1)
                row = sh.iloc[pos - lag].reindex(active).fillna(0.0).to_numpy(dtype=float)
                acc += weight * (w @ row)
                mass += weight
            if mass <= 0:
                continue
            acc /= mass
            for i, sym in enumerate(active):
                records.setdefault((date, sym), {"date": date, "symbol": sym})[s.name()] = float(acc[i])
        if progress and (step == 1 or step % 250 == 0 or step == len(dates)):
            print(f"  diffusion {step:>5}/{len(dates)}  {pd.Timestamp(date).date()}", flush=True)

    if not records:
        return pd.DataFrame(columns=["date", "symbol"])
    out = pd.DataFrame(list(records.values())).sort_values(["date", "symbol"]).reset_index(drop=True)
    if out.duplicated(["date", "symbol"]).any():
        raise AssertionError("diffusion output contains duplicate (date, symbol) rows")
    for s in specs:
        if s.name() in out:
            out[f"{s.name()}_rank"] = out.groupby("date")[s.name()].rank(pct=True)
    return out
