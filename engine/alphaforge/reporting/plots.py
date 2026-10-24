"""Matplotlib figures for factor reports, backtests and mining runs.

Every public ``plot_*`` function returns a :class:`matplotlib.figure.Figure`
and, when ``save_path`` is given, also writes a PNG (``dpi=110``, tight
bounding box, parent directories created).  Figures are built *outside*
pyplot's global figure registry (``Figure`` + an Agg canvas), so producing
hundreds of them inside a research loop never triggers the "more than 20
figures" warning and nothing leaks between calls; drop the reference to free
memory (``plt.close(fig)`` is harmless but unnecessary).

Fonts: at import the module installs a CJK-capable ``font.family`` fallback
list (``PingFang SC``, ``Hiragino Sans GB``, ``Noto Sans CJK SC``,
``Arial Unicode MS``, ``DejaVu Sans``), **filtered to the families actually
installed** so matplotlib's per-draw ``findfont`` warnings never fire, and
sets ``axes.unicode_minus = False`` so a minus sign renders with CJK fonts.
Titles in a non-Latin script (for example CJK) therefore render on macOS / Linux out of the box.

Each factor gets the same pictures: IC bars with rolling and cumulative IC,
the quantile fan, the IC decay, and equity with drawdown.
"""

from __future__ import annotations

import io
import logging
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

# Headless backend for pyplot users (scripts, tests, servers).  The switch occurs
# only when pyplot is not imported, so importing this module inside a
# notebook that already chose an inline backend does not hijack it.  The
# functions below never touch pyplot, so they are backend-independent anyway.
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

if TYPE_CHECKING:  # duck-typed at runtime; keeps reporting decoupled from upstream imports
    from alphaforge.evaluation.report import FactorReport
    from alphaforge.portfolio.backtest import BacktestResult

logger = logging.getLogger("alphaforge.reporting.plots")

__all__ = [
    "CJK_FONT_FALLBACK",
    "CJK_FONT_AVAILABLE",
    "FONT_FAMILIES",
    "SAVE_DPI",
    "configure_fonts",
    "save_figure",
    "fig_to_png_bytes",
    "plot_ic",
    "plot_quantiles",
    "plot_ic_decay",
    "plot_equity",
    "plot_mining_history",
    "plot_corr_heatmap",
    "plot_report_figures",
]

#: Resolution used by :func:`save_figure` and :func:`fig_to_png_bytes`.
SAVE_DPI: int = 110
#: Preferred font families, first match wins per glyph (CJK families first).
CJK_FONT_FALLBACK: tuple[str, ...] = (
    "PingFang SC",
    "Hiragino Sans GB",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
)
_DEFAULT_FONT = "DejaVu Sans"  # shipped with matplotlib, always resolvable

# consistent palette across all figures
_POS = "#2a9d8f"  # positive values / long
_NEG = "#e76f51"  # negative values / drawdown
_LINE = "#264653"  # primary line
_ACCENT = "#e9a43a"  # secondary line (cumulative IC, counts)
_GREY = "#8d99ae"
_GRID_ALPHA = 0.3


# --------------------------------------------------------------------------- #
# fonts
# --------------------------------------------------------------------------- #
def _installed_font_families() -> set[str]:
    try:
        return {f.name for f in font_manager.fontManager.ttflist}
    except Exception:  # pragma: no cover - defensive: broken font cache
        logger.debug("could not enumerate installed fonts", exc_info=True)
        return set()


def configure_fonts(preferred: Sequence[str] = CJK_FONT_FALLBACK) -> list[str]:
    """Install a CJK-safe ``font.family`` fallback list into ``rcParams``.

    Only families that are actually installed are kept (``DejaVu Sans`` is
    always appended as the guaranteed last resort) so matplotlib never logs a
    ``findfont`` warning on every text draw.  ``axes.unicode_minus`` is set to
    ``False`` because most CJK fonts lack the Unicode minus glyph.

    Args:
        preferred: Families in priority order.

    Returns:
        The families that were installed into ``rcParams["font.family"]``.

    The fallback list keeps a title in a non-Latin script readable.
    """
    installed = _installed_font_families()
    families = [f for f in preferred if f in installed]
    if _DEFAULT_FONT not in families:
        families.append(_DEFAULT_FONT)
    matplotlib.rcParams["font.family"] = families
    matplotlib.rcParams["axes.unicode_minus"] = False
    logger.debug("font fallback list: %s", families)
    return families


#: Families actually installed (in priority order) by :func:`configure_fonts` at import.
FONT_FAMILIES: list[str] = configure_fonts()
#: True when at least one CJK-capable family from :data:`CJK_FONT_FALLBACK` is installed.
CJK_FONT_AVAILABLE: bool = any(f != _DEFAULT_FONT for f in FONT_FAMILIES)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _new_figure(figsize: tuple[float, float]) -> Figure:
    """A pyplot-free figure with an Agg canvas and constrained layout."""
    fig = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _style_axes(ax: Any, *, keep_right_spine: bool = False) -> None:
    ax.grid(True, alpha=_GRID_ALPHA, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    if not keep_right_spine:
        ax.spines["right"].set_visible(False)


def _date_axis(ax: Any, index: pd.Index) -> None:
    """Concise date ticks when the x data are timestamps."""
    if isinstance(index, pd.DatetimeIndex):
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _bar_width(index: pd.Index) -> float:
    """Bar width in x-data units: ~90% of the median spacing (days for dates)."""
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        deltas = np.diff(index.values).astype("timedelta64[s]").astype(float) / 86400.0
        deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
        if deltas.size:
            return float(max(np.median(deltas) * 0.9, 0.5))
    return 0.8


def _trim_valid(s: Any) -> pd.Series:
    """Float series cut to ``[first_valid, last_valid]`` (empty if all-NaN)."""
    s = pd.Series(s).astype(float)
    first, last = s.first_valid_index(), s.last_valid_index()
    if first is None:
        return s.iloc[0:0]
    return s.loc[first:last]


def _trim_valid_frame(df: Any) -> pd.DataFrame:
    """Float frame cut to the rows between the first and last non-all-NaN row."""
    df = pd.DataFrame(df).astype(float)
    valid = np.flatnonzero(df.notna().any(axis=1).to_numpy())
    if valid.size == 0:
        return df.iloc[0:0]
    return df.iloc[valid[0] : valid[-1] + 1]


def _fmt(x: Any, spec: str, na: str = "n/a") -> str:
    """Format a scalar, showing ``na`` for None / non-finite values."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    return format(v, spec) if math.isfinite(v) else na


def _no_data(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color=_GREY, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def _sign_colors(values: np.ndarray) -> np.ndarray:
    return np.where(np.nan_to_num(values, nan=0.0) >= 0.0, _POS, _NEG)


def save_figure(fig: Figure, save_path: str | Path, *, dpi: int = SAVE_DPI) -> Path:
    """Write ``fig`` to ``save_path`` (PNG by extension, tight bbox, parents created)."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("saved figure -> %s", path)
    return path


def fig_to_png_bytes(fig: Figure, *, dpi: int = SAVE_DPI) -> bytes:
    """Render ``fig`` to PNG bytes in memory (used for base64 embedding in HTML)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def _finalize(fig: Figure, save_path: str | Path | None) -> Figure:
    if save_path is not None:
        save_figure(fig, save_path)
    return fig


# --------------------------------------------------------------------------- #
# factor-report figures
# --------------------------------------------------------------------------- #
def plot_ic(
    report: FactorReport,
    save_path: str | Path | None = None,
    *,
    rolling_window: int = 20,
    use_rank_ic: bool = False,
    periods_per_year: int = 252,
    figsize: tuple[float, float] = (11.0, 4.8),
) -> Figure:
    """Daily IC bars, a rolling mean line and the cumulative IC on a twin axis.

    Args:
        report: A :class:`FactorReport` (only ``name``, ``horizon``, ``lag`` and
            ``ic_series`` / ``rank_ic_series`` are used).
        save_path: Optional PNG path.
        rolling_window: Window of the rolling-mean line (``min_periods`` is
            half the window so the line starts early).
        use_rank_ic: Plot ``rank_ic_series`` (Spearman) instead of the Pearson ``ic_series``.
        periods_per_year: Annualisation factor for the ICIR shown in the title.
        figsize: Figure size in inches.

    Returns:
        The figure.

    The rolling mean shows the stability of the IC, and the cumulative IC shows
    whether one short period gives most of the result.
    """
    label = "Rank IC" if use_rank_ic else "IC"
    series = report.rank_ic_series if use_rank_ic else report.ic_series
    ic = _trim_valid(series)

    fig = _new_figure(figsize)
    ax = fig.add_subplot(111)
    if ic.empty:
        _no_data(ax, f"{report.name}: no {label} observations")
        ax.set_title(f"{report.name} — {label}")
        return _finalize(fig, save_path)

    vals = ic.to_numpy()
    ax.bar(
        ic.index,
        vals,
        width=_bar_width(ic.index),
        color=_sign_colors(vals),
        alpha=0.55,
        linewidth=0,
        label=f"daily {label}",
    )
    roll = ic.rolling(int(rolling_window), min_periods=max(1, int(rolling_window) // 2)).mean()
    ax.plot(roll.index, roll.to_numpy(), color=_LINE, lw=1.8, label=f"{int(rolling_window)}d rolling mean")
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if ic.notna().sum() > 1 else float("nan")
    ir = mean / std if math.isfinite(std) and std > 0 else float("nan")
    ir_ann = ir * math.sqrt(periods_per_year) if math.isfinite(ir) else float("nan")
    pos = float((ic.dropna() > 0).mean()) if ic.notna().any() else float("nan")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(mean, color=_GREY, ls="--", lw=1.0, label=f"mean {mean:+.4f}")
    ax.set_ylabel(label)
    ax.set_xlabel("date")
    _style_axes(ax, keep_right_spine=True)
    _date_axis(ax, ic.index)

    ax2 = ax.twinx()
    cum = ic.fillna(0.0).cumsum()
    ax2.plot(cum.index, cum.to_numpy(), color=_ACCENT, lw=2.0, label=f"cumulative {label}")
    ax2.set_ylabel(f"cumulative {label}")
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.85)
    ax.set_title(
        f"{report.name} — {label} (h={report.horizon}, lag={report.lag}): "
        f"mean {mean:+.4f}, std {_fmt(std, '.4f')}, ICIR {_fmt(ir, '.3f')}, "
        f"ICIR_ann {_fmt(ir_ann, '.2f')}, positive {_fmt(pos, '.0%')}",
        fontsize=10.5,
    )
    return _finalize(fig, save_path)


def plot_quantiles(
    report: FactorReport,
    save_path: str | Path | None = None,
    *,
    periods_per_year: int = 252,
    figsize: tuple[float, float] = (13.0, 4.8),
) -> Figure:
    """Cumulative return per quantile (left) and annualised return per quantile (right).

    Left panel: cumulative sum of ``quantile_returns / h`` (the h-day returns
    are observed daily, so dividing by ``h`` gives a daily-equivalent additive
    equity curve; NaN days contribute 0), one viridis-coloured line per
    quantile plus the dashed ``Q_n - Q_1`` long-short line.
    Right panel: ``mean(quantile_returns) * periods_per_year / h`` per quantile.

    Args:
        report: A :class:`FactorReport` (``quantile_returns``, ``horizon``, ``monotonicity``).
        save_path: Optional PNG path.
        periods_per_year: Annualisation factor.
        figsize: Figure size in inches.

    Returns:
        The figure.

    The bars show whether the return increases at each quantile step or
    comes only from one tail.
    """
    h = max(int(getattr(report, "horizon", 1) or 1), 1)
    qret = _trim_valid_frame(report.quantile_returns)
    fig = _new_figure(figsize)
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2.2, 1.0]})
    fig.suptitle(f"{report.name} — quantile analysis (h={h}, {qret.shape[1]} quantiles)", fontsize=11.5)
    if qret.empty:
        _no_data(ax1, f"{report.name}: no quantile returns")
        _no_data(ax2, "")
        return _finalize(fig, save_path)

    n = qret.shape[1]
    colors = matplotlib.colormaps["viridis"](np.linspace(0.0, 1.0, n))
    cum = (qret / h).fillna(0.0).cumsum()
    for i, q in enumerate(cum.columns):
        edge = i in (0, n - 1)
        ax1.plot(cum.index, cum[q].to_numpy(), color=colors[i], lw=1.6 if edge else 1.0, label=f"Q{int(q)}")
    ls = ((qret.iloc[:, -1] - qret.iloc[:, 0]) / h).fillna(0.0).cumsum()
    ax1.plot(
        ls.index,
        ls.to_numpy(),
        color="black",
        ls="--",
        lw=1.4,
        label=f"Q{int(qret.columns[-1])} − Q{int(qret.columns[0])}",
    )
    ax1.axhline(0.0, color="black", lw=0.6)
    ax1.set_title("cumulative return per quantile (daily-equivalent: h-day return / h)", fontsize=10)
    ax1.set_xlabel("date")
    ax1.set_ylabel("cumulative return")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax1.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.85)
    _style_axes(ax1)
    _date_axis(ax1, cum.index)

    ann = qret.mean(axis=0) * periods_per_year / h
    x = np.arange(n)
    bars = ax2.bar(x, ann.to_numpy(), color=colors, alpha=0.9, linewidth=0)
    ax2.bar_label(bars, labels=[_fmt(v, "+.1%", na="") for v in ann.to_numpy()], fontsize=7, padding=2)
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Q{int(q)}" for q in qret.columns], fontsize=8)
    ax2.set_ylabel("annualised return")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    mono = getattr(report, "monotonicity", float("nan"))
    ax2.set_title(f"annualised return per quantile (monotonicity {_fmt(mono, '+.2f')})", fontsize=10)
    _style_axes(ax2)
    return _finalize(fig, save_path)


def plot_ic_decay(
    report: FactorReport,
    save_path: str | Path | None = None,
    *,
    figsize: tuple[float, float] = (7.5, 4.2),
) -> Figure:
    """Bar chart of the mean rank IC per forward horizon (``report.ic_decay``).

    The bar of the report's own evaluation horizon is outlined in black.

    Args:
        report: A :class:`FactorReport` (``ic_decay`` indexed by horizon, ``horizon``, ``lag``).
        save_path: Optional PNG path.
        figsize: Figure size in inches.

    Returns:
        The figure.

    The decay profile shows the holding period, and so the turnover that the
    signal needs.
    """
    decay = pd.Series(report.ic_decay).astype(float)
    fig = _new_figure(figsize)
    ax = fig.add_subplot(111)
    title = f"{report.name} — rank IC decay by holding horizon (lag={getattr(report, 'lag', 1)})"
    if decay.empty or not decay.notna().any():
        _no_data(ax, f"{report.name}: no IC-decay data")
        ax.set_title(title, fontsize=10.5)
        return _finalize(fig, save_path)

    vals = decay.to_numpy()
    x = np.arange(len(decay))
    own_h = getattr(report, "horizon", None)
    edge_colors = ["black" if own_h is not None and float(hz) == float(own_h) else "none" for hz in decay.index]
    edge_widths = [1.6 if c == "black" else 0.0 for c in edge_colors]
    bars = ax.bar(x, vals, color=_sign_colors(vals), alpha=0.85, edgecolor=edge_colors, linewidth=edge_widths)
    ax.bar_label(bars, labels=[_fmt(v, "+.4f", na="") for v in vals], fontsize=8, padding=2)
    ax.plot(x, vals, color=_LINE, lw=1.0, marker="o", ms=3.5, alpha=0.7)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(hz)}d" if float(hz).is_integer() else str(hz) for hz in decay.index])
    ax.set_xlabel("forward horizon (days)")
    ax.set_ylabel("mean rank IC")
    ax.set_title(title, fontsize=10.5)
    if "black" in edge_colors:
        marker = Patch(facecolor="none", edgecolor="black", linewidth=1.6, label=f"evaluation horizon h={int(own_h)}")
        ax.legend(handles=[marker], loc="best", fontsize=8, framealpha=0.85)
    _style_axes(ax)
    return _finalize(fig, save_path)


# --------------------------------------------------------------------------- #
# backtest / mining / library figures
# --------------------------------------------------------------------------- #
def plot_equity(
    bt: BacktestResult,
    save_path: str | Path | None = None,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (11.0, 6.0),
) -> Figure:
    """Net equity curve (with the gross curve dashed) above a shaded drawdown panel.

    Args:
        bt: A :class:`BacktestResult`; ``equity`` is required, ``gross_returns``
            and ``drawdown`` are used when present (otherwise the drawdown is
            recomputed as ``equity / cummax(equity) - 1``).
        save_path: Optional PNG path.
        title: Name shown in the title (``BacktestResult`` carries no name); default ``"Backtest"``.
        figsize: Figure size in inches.

    Returns:
        The figure.

    The drawdown panel shows the depth and the length of the worst drawdown,
    which the Sharpe ratio does not show.
    """
    name = title or "Backtest"
    equity = _trim_valid(bt.equity)
    fig = _new_figure(figsize)
    ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3.0, 1.2]})
    if equity.empty:
        _no_data(ax1, f"{name}: empty equity curve")
        _no_data(ax2, "")
        ax1.set_title(f"{name} — equity")
        return _finalize(fig, save_path)

    ax1.plot(equity.index, equity.to_numpy(), color=_LINE, lw=1.8, label="net equity")
    gross = getattr(bt, "gross_returns", None)
    if gross is not None and len(gross):
        gross_eq = (1.0 + pd.Series(gross).astype(float).reindex(equity.index).fillna(0.0)).cumprod()
        ax1.plot(gross_eq.index, gross_eq.to_numpy(), color=_GREY, ls="--", lw=1.2, label="gross equity (before costs)")
    ax1.axhline(1.0, color="black", lw=0.8)
    ax1.set_ylabel("equity (start = 1)")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.85)
    _style_axes(ax1)

    dd_src = getattr(bt, "drawdown", None)
    if dd_src is not None and len(dd_src):
        dd = pd.Series(dd_src).astype(float).reindex(equity.index)
    else:
        dd = equity / equity.cummax() - 1.0
    dd = dd.fillna(0.0)
    ax2.fill_between(dd.index, dd.to_numpy(), 0.0, color=_NEG, alpha=0.35, label="drawdown")
    ax2.plot(dd.index, dd.to_numpy(), color=_NEG, lw=1.0)
    ax2.set_ylabel("drawdown")
    ax2.set_xlabel("decision date")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.85)
    _style_axes(ax2)
    _date_axis(ax2, equity.index)

    stats = (
        f"ann.ret {_fmt(getattr(bt, 'ann_return', None), '+.2%')}   "
        f"ann.vol {_fmt(getattr(bt, 'ann_vol', None), '.2%')}   "
        f"Sharpe {_fmt(getattr(bt, 'sharpe', None), '.2f')}   "
        f"maxDD {_fmt(getattr(bt, 'max_dd', None), '.2%')}   "
        f"Calmar {_fmt(getattr(bt, 'calmar', None), '.2f')}   "
        f"turnover {_fmt(getattr(bt, 'avg_turnover', None), '.3f')}/day   "
        f"hit {_fmt(getattr(bt, 'hit_rate', None), '.1%')}   "
        f"cost {_fmt(getattr(bt, 'cost_bps', None), '.0f')} bps"
    )
    ax1.set_title(f"{name} — equity curve (net of costs)\n{stats}", fontsize=10.5)
    return _finalize(fig, save_path)


def plot_mining_history(
    history: Iterable[Mapping[str, Any]],
    save_path: str | Path | None = None,
    *,
    title: str = "GP mining",
    figsize: tuple[float, float] = (10.0, 4.8),
) -> Figure:
    """Best / mean / median fitness per generation, with valid & unique counts on a twin axis.

    Args:
        history: ``GeneticMiner.history`` -- one dict per generation with keys
            ``gen, best, mean, median, n_valid, n_unique, best_expr, elapsed_s``
            (all optional except that at least ``best`` or ``mean`` should exist).
            Non-finite fitness values (``-inf`` from invalid individuals) are shown as gaps.
        save_path: Optional PNG path.
        title: Title prefix.
        figsize: Figure size in inches.

    Returns:
        The figure.

    A best fitness that increases while the mean stays flat shows one lineage;
    an early plateau shows that the search finds nothing new.
    """
    df = pd.DataFrame(list(history))
    fig = _new_figure(figsize)
    ax = fig.add_subplot(111)
    if df.empty:
        _no_data(ax, "no mining history")
        ax.set_title(f"{title} — fitness by generation")
        return _finalize(fig, save_path)

    gen = pd.to_numeric(df["gen"], errors="coerce").to_numpy() if "gen" in df else np.arange(len(df))
    plotted = False
    for key, color, style, lw in (("best", _LINE, "-", 2.0), ("mean", _POS, "--", 1.5), ("median", _GREY, ":", 1.3)):
        if key in df:
            y = pd.to_numeric(df[key], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy()
            ax.plot(gen, y, color=color, ls=style, lw=lw, marker="o", ms=3.5, label=f"{key} fitness")
            plotted = True
    if not plotted:
        _no_data(ax, "history has no fitness columns (best / mean / median)")
        return _finalize(fig, save_path)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _style_axes(ax, keep_right_spine=True)

    handles, labels = ax.get_legend_handles_labels()
    if "n_valid" in df or "n_unique" in df:
        ax2 = ax.twinx()
        if "n_valid" in df:
            ax2.bar(
                gen,
                pd.to_numeric(df["n_valid"], errors="coerce").to_numpy(),
                color=_ACCENT,
                alpha=0.15,
                width=0.6,
                label="n_valid",
            )
        if "n_unique" in df:
            ax2.plot(
                gen,
                pd.to_numeric(df["n_unique"], errors="coerce").to_numpy(),
                color=_ACCENT,
                ls="-.",
                lw=1.2,
                label="n_unique",
            )
        ax2.set_ylabel("count")
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.85)

    last = df.iloc[-1]
    best_last = pd.to_numeric(pd.Series([last.get("best", np.nan)]), errors="coerce").iloc[0]
    elapsed = pd.to_numeric(df["elapsed_s"], errors="coerce").sum() if "elapsed_s" in df else float("nan")
    ax.set_title(
        f"{title} — best fitness {_fmt(best_last, '+.3f')} after {len(df)} generations"
        + (f" ({_fmt(elapsed, '.1f')} s)" if math.isfinite(float(elapsed)) else ""),
        fontsize=10.5,
    )
    expr = last.get("best_expr", None) if hasattr(last, "get") else None
    if isinstance(expr, str) and expr:
        shown = expr if len(expr) <= 110 else expr[:107] + "..."
        ax.text(
            0.01,
            0.02,
            f"best: {shown}",
            transform=ax.transAxes,
            fontsize=7.5,
            family="monospace",
            color=_LINE,
            alpha=0.85,
        )
    return _finalize(fig, save_path)


def plot_corr_heatmap(
    corr: pd.DataFrame,
    save_path: str | Path | None = None,
    *,
    title: str = "Factor correlation",
    annotate: bool | None = None,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """``imshow`` heatmap of a correlation matrix with labelled axes and a colorbar.

    Args:
        corr: Square (or rectangular) correlation frame; index/column labels are used as tick labels.
        save_path: Optional PNG path.
        title: Figure title.
        annotate: Write the value into each cell; default ``None`` = only when the matrix has <= 400 cells.
        cmap: Diverging colormap name.
        vmin: Lower colour limit.
        vmax: Upper colour limit.
        figsize: Figure size; default scales with the matrix size.

    Returns:
        The figure.

    The heatmap shows the correlation between the library factors; a
    high value marks two factors that are close to one signal.
    """
    corr = pd.DataFrame(corr).astype(float)
    n_r, n_c = corr.shape
    if figsize is None:
        side = float(min(max(4.0, 0.5 * max(n_r, n_c) + 2.0), 16.0))
        figsize = (side + 1.4, side)
    fig = _new_figure(figsize)
    ax = fig.add_subplot(111)
    if corr.empty:
        _no_data(ax, "empty correlation matrix")
        ax.set_title(title)
        return _finalize(fig, save_path)

    vals = corr.to_numpy()
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(n_c))
    ax.set_xticklabels([str(c) for c in corr.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n_r))
    ax.set_yticklabels([str(r) for r in corr.index], fontsize=8)
    ax.set_xticks(np.arange(-0.5, n_c, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n_r, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("correlation")

    if annotate is None:
        annotate = n_r * n_c <= 400
    if annotate:
        fs = max(5.5, 9.0 - max(n_r, n_c) / 5.0)
        for i in range(n_r):
            for j in range(n_c):
                v = vals[i, j]
                if not math.isfinite(v):
                    continue
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center", fontsize=fs, color="white" if abs(v) > 0.5 else "black"
                )
    ax.set_title(f"{title} ({n_r}×{n_c})", fontsize=10.5)
    return _finalize(fig, save_path)


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #
def plot_report_figures(
    report: FactorReport,
    out_dir: str | Path,
    *,
    prefix: str | None = None,
) -> dict[str, Path]:
    """Write the three standard figures of a factor report into ``out_dir``.

    Files are ``<prefix>_ic.png``, ``<prefix>_quantiles.png`` and
    ``<prefix>_ic_decay.png`` (``prefix`` defaults to a filesystem-safe version
    of ``report.name``).

    Returns:
        ``{"ic": path, "quantiles": path, "ic_decay": path}``.
    """
    out = Path(out_dir)
    safe = prefix if prefix is not None else "".join(c if c.isalnum() or c in "-_." else "_" for c in str(report.name))
    paths = {
        "ic": out / f"{safe}_ic.png",
        "quantiles": out / f"{safe}_quantiles.png",
        "ic_decay": out / f"{safe}_ic_decay.png",
    }
    plot_ic(report, save_path=paths["ic"])
    plot_quantiles(report, save_path=paths["quantiles"])
    plot_ic_decay(report, save_path=paths["ic_decay"])
    return paths
