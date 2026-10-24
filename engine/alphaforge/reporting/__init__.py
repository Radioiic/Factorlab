"""AlphaForge reporting: matplotlib figures and a standalone HTML report.

Public API::

    plot_ic, plot_quantiles, plot_ic_decay        # FactorReport figures
    plot_equity                                   # BacktestResult figure
    plot_mining_history, plot_corr_heatmap        # miner / library figures
    render_html_report                            # standalone HTML (base64 PNGs, utf-8)

Helpers: :func:`save_figure`, :func:`fig_to_png_bytes`, :func:`figure_to_data_uri`,
:func:`plot_report_figures`, :func:`configure_fonts`.  Importing this package
switches matplotlib to the Agg backend (unless pyplot was already imported) and
installs a CJK-safe font fallback so Chinese titles render.
"""

from alphaforge.reporting.html import DEFAULT_CSS, figure_to_data_uri, render_html_report
from alphaforge.reporting.plots import (
    CJK_FONT_AVAILABLE,
    CJK_FONT_FALLBACK,
    FONT_FAMILIES,
    SAVE_DPI,
    configure_fonts,
    fig_to_png_bytes,
    plot_corr_heatmap,
    plot_equity,
    plot_ic,
    plot_ic_decay,
    plot_mining_history,
    plot_quantiles,
    plot_report_figures,
    save_figure,
)

__all__ = [
    "plot_ic",
    "plot_quantiles",
    "plot_ic_decay",
    "plot_equity",
    "plot_mining_history",
    "plot_corr_heatmap",
    "plot_report_figures",
    "render_html_report",
    "figure_to_data_uri",
    "save_figure",
    "fig_to_png_bytes",
    "configure_fonts",
    "CJK_FONT_FALLBACK",
    "CJK_FONT_AVAILABLE",
    "FONT_FAMILIES",
    "SAVE_DPI",
    "DEFAULT_CSS",
]
