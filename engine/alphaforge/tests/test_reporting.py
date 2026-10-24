"""Tests for ``alphaforge.reporting`` and ``alphaforge.config``.

Uses real upstream objects: a small simulated market, ``evaluate_factor`` on
the negative 5-day return (the planted reversal alpha) and a trivial
equal-weight long-short backtest.  Every figure must come back as a
``matplotlib.figure.Figure`` and, when asked, land on disk as a PNG > 1 KB.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from alphaforge.config import ResearchConfig
from alphaforge.data import SimConfig, simulate_market
from alphaforge.evaluation import FactorReport, evaluate_factor
from alphaforge.portfolio import backtest, weights_from_scores
from alphaforge.reporting import (
    fig_to_png_bytes,
    figure_to_data_uri,
    plot_corr_heatmap,
    plot_equity,
    plot_ic,
    plot_ic_decay,
    plot_mining_history,
    plot_quantiles,
    plot_report_figures,
    render_html_report,
)
from alphaforge.reporting import plots as plots_mod

MIN_PNG_BYTES = 1024
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# Chinese sample text, written as \u escapes: these tests check non-Latin (CJK) rendering.
CN_TEXT = "\u53cd\u8f6c\u56e0\u5b50\u7814\u7a76\u62a5\u544a\uff1a\u6a21\u62df\u5e02\u573a\u5171 40 \u53ea\u80a1\u7968"


# --------------------------------------------------------------------------- #
# fixtures (module-scoped: the simulation is the slow part)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def store():
    return simulate_market(SimConfig(n_assets=40, n_days=200, seed=0))


@pytest.fixture(scope="module")
def factor(store):
    close = store.get("close")
    return -(close / close.shift(5) - 1.0)  # negative 5-day return = reversal


@pytest.fixture(scope="module")
def report(factor, store) -> FactorReport:
    return evaluate_factor(factor, store, name="rev5", expr="neg(ts_pct_change(close, 5))", horizon=5, lag=1)


@pytest.fixture(scope="module")
def bt(factor, store):
    weights = weights_from_scores(factor, top_frac=0.2, max_weight=0.2)
    return backtest(weights, store, cost_bps=10.0)


@pytest.fixture
def history() -> list[dict]:
    rng = np.random.default_rng(0)
    best, out = 0.2, []
    for g in range(6):
        best += abs(rng.normal(0.1, 0.05))
        out.append(
            {
                "gen": g,
                "best": best,
                "mean": -np.inf if g == 0 else best - 0.5,  # gen 0: invalid individuals drag the mean to -inf
                "median": best - 0.3,
                "n_valid": 30 + g,
                "n_unique": 35 + 2 * g,
                "best_expr": f"cs_rank(ts_delta(close, {5 + g}))",
                "elapsed_s": 1.5,
            }
        )
    return out


@pytest.fixture
def corr() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 5))
    x[:, 1] = 0.9 * x[:, 0] + 0.1 * x[:, 1]
    names = ["rev5", "mom120", "vol_shock", "value_bp", "\u52a8\u91cf\u56e0\u5b50"]
    c = pd.DataFrame(np.corrcoef(x.T), index=names, columns=names)
    c.iloc[2, 4] = c.iloc[4, 2] = np.nan  # a NaN cell must not break the heatmap
    return c


def _assert_png(path: Path) -> None:
    assert path.is_file(), path
    assert path.stat().st_size > MIN_PNG_BYTES
    assert path.read_bytes()[:8] == PNG_MAGIC


def _all_titles(fig: Figure) -> str:
    parts = [getattr(fig, "get_suptitle", lambda: "")()]
    parts += [ax.get_title() for ax in fig.axes]
    return " | ".join(parts)


# --------------------------------------------------------------------------- #
# module-level configuration
# --------------------------------------------------------------------------- #
def test_font_and_backend_configuration():
    assert matplotlib.rcParams["axes.unicode_minus"] is False
    families = list(matplotlib.rcParams["font.family"])
    assert families == plots_mod.FONT_FAMILIES
    assert families[-1] == "DejaVu Sans"
    installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    assert all(f in installed or f == "DejaVu Sans" for f in families)
    assert plots_mod.SAVE_DPI == 110


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
class TestPlots:
    def test_plot_ic(self, report, tmp_path):
        out = tmp_path / "ic.png"
        fig = plot_ic(report, save_path=out)
        assert isinstance(fig, Figure)
        _assert_png(out)
        assert report.name in _all_titles(fig)
        assert len(fig.axes) == 2  # twin axis for the cumulative IC
        # bars (daily IC) + rolling mean + cumulative line all present
        ax, ax2 = fig.axes
        assert len(ax.patches) == int(report.ic_series.notna().sum()) or len(ax.patches) >= 100
        assert any("rolling" in lbl for lbl in [ln.get_label() for ln in ax.lines])
        assert any("cumulative" in ln.get_label() for ln in ax2.lines)

    def test_plot_ic_rank_variant(self, report, tmp_path):
        fig = plot_ic(report, save_path=tmp_path / "rank_ic.png", use_rank_ic=True, rolling_window=10)
        assert "Rank IC" in _all_titles(fig)
        _assert_png(tmp_path / "rank_ic.png")

    def test_plot_quantiles(self, report, tmp_path):
        out = tmp_path / "quantiles.png"
        fig = plot_quantiles(report, save_path=out)
        assert isinstance(fig, Figure)
        _assert_png(out)
        assert report.name in _all_titles(fig)
        ax_left, ax_right = fig.axes[:2]
        assert len(ax_left.lines) >= report.n_quantiles + 1  # one per quantile + long-short
        assert len(ax_right.patches) == report.n_quantiles
        # right panel bars are mean * 252 / h
        expected = (report.quantile_returns.mean() * 252 / report.horizon).to_numpy()
        heights = np.array([p.get_height() for p in ax_right.patches])
        np.testing.assert_allclose(heights, expected, rtol=1e-9)

    def test_plot_ic_decay(self, report, tmp_path):
        out = tmp_path / "decay.png"
        fig = plot_ic_decay(report, save_path=out)
        assert isinstance(fig, Figure)
        _assert_png(out)
        assert report.name in _all_titles(fig)
        ax = fig.axes[0]
        assert len(ax.patches) == len(report.ic_decay)
        heights = np.array([p.get_height() for p in ax.patches])
        np.testing.assert_allclose(heights, report.ic_decay.to_numpy(), rtol=1e-9)

    def test_plot_equity(self, bt, tmp_path):
        out = tmp_path / "equity.png"
        fig = plot_equity(bt, save_path=out, title="rev5 backtest")
        assert isinstance(fig, Figure)
        _assert_png(out)
        assert "rev5 backtest" in _all_titles(fig)
        ax_eq, ax_dd = fig.axes[:2]
        assert len(ax_eq.lines) >= 2  # net + gross
        assert len(ax_dd.collections) >= 1  # shaded drawdown
        assert ax_dd.get_ylim()[0] <= float(bt.drawdown.min())

    def test_plot_mining_history(self, history, tmp_path):
        out = tmp_path / "mining.png"
        fig = plot_mining_history(history, save_path=out)
        assert isinstance(fig, Figure)
        _assert_png(out)
        ax = fig.axes[0]
        labels = [ln.get_label() for ln in ax.lines]
        assert any("best" in x for x in labels) and any("mean" in x for x in labels)
        # -inf mean must be shown as a gap, not crash or distort the axis
        assert np.isfinite(ax.get_ylim()).all()

    def test_plot_corr_heatmap(self, corr, tmp_path):
        out = tmp_path / "corr.png"
        fig = plot_corr_heatmap(corr, save_path=out, title="\u56e0\u5b50\u76f8\u5173\u6027")
        assert isinstance(fig, Figure)
        _assert_png(out)
        ax = fig.axes[0]
        assert len(ax.images) == 1
        assert [t.get_text() for t in ax.get_xticklabels()] == list(corr.columns)
        assert len(fig.axes) == 2  # colorbar axis
        assert "\u56e0\u5b50\u76f8\u5173\u6027" in _all_titles(fig)

    def test_save_path_creates_parent_dirs(self, report, tmp_path):
        out = tmp_path / "a" / "b" / "c" / "ic.png"
        plot_ic(report, save_path=out)
        _assert_png(out)

    def test_plot_report_figures_writes_three_pngs(self, report, tmp_path):
        paths = plot_report_figures(report, tmp_path / "figs")
        assert set(paths) == {"ic", "quantiles", "ic_decay"}
        for p in paths.values():
            _assert_png(p)

    def test_no_save_when_path_omitted(self, report, tmp_path):
        fig = plot_ic_decay(report)
        assert isinstance(fig, Figure)
        assert list(tmp_path.iterdir()) == []

    def test_chinese_title_renders_without_font_warnings(self, report, caplog):
        rep = dataclasses.replace(report, name="\u53cd\u8f6c\u56e0\u5b50_5\u65e5")
        with caplog.at_level(logging.WARNING, logger="matplotlib"), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = plot_ic(rep)
            png = fig_to_png_bytes(fig)
        assert png[:8] == PNG_MAGIC and len(png) > MIN_PNG_BYTES
        assert not [r for r in caplog.records if "findfont" in r.getMessage()]
        if plots_mod.CJK_FONT_AVAILABLE:
            assert not [w for w in caught if "Glyph" in str(w.message)], [str(w.message) for w in caught]

    def test_all_nan_ic_series_is_handled(self, report):
        nan_ic = pd.Series(np.nan, index=report.ic_series.index)
        rep = dataclasses.replace(report, ic_series=nan_ic, rank_ic_series=nan_ic)
        fig = plot_ic(rep)
        assert isinstance(fig, Figure)
        assert report.name in _all_titles(fig)

    def test_empty_inputs_are_handled(self):
        assert isinstance(plot_mining_history([]), Figure)
        assert isinstance(plot_corr_heatmap(pd.DataFrame()), Figure)

    def test_figures_not_registered_with_pyplot(self, report):
        import matplotlib.pyplot as plt

        before = len(plt.get_fignums())
        for _ in range(5):
            plot_ic_decay(report)
        assert len(plt.get_fignums()) == before


# --------------------------------------------------------------------------- #
# html
# --------------------------------------------------------------------------- #
class TestHtml:
    def test_render_html_report(self, report, bt, tmp_path, caplog):
        fig_ic = plot_ic(report)
        decay_png = plot_report_figures(report, tmp_path / "figs")["ic_decay"]
        sections = [
            {
                "title": "\u6570\u636e\u6982\u89c8",
                "text": f"<p>{CN_TEXT}</p><ul><li>horizon = 5</li></ul>",
                "table": FactorReport.table([report]),
                "figures": [fig_ic],
            },
            {
                "title": "\u56de\u6d4b Backtest",
                "text": "plain text without tags",
                "table": pd.DataFrame([bt.to_dict()], index=["rev5"]),
                "figures": [decay_png, (plot_equity(bt), "\u51c0\u503c\u66f2\u7ebf")],
            },
            {"title": "Only a title"},
        ]
        out_path = tmp_path / "nested" / "dir" / "report.html"
        with caplog.at_level(logging.WARNING):
            out = render_html_report(title="AlphaForge \u7814\u7a76\u62a5\u544a", sections=sections, out_path=out_path)
        assert isinstance(out, Path) and out == out_path and out.is_file()
        html = out.read_text(encoding="utf-8")

        assert '<meta charset="utf-8">' in html
        assert "AlphaForge \u7814\u7a76\u62a5\u544a" in html
        assert CN_TEXT in html
        for sec in sections:
            assert sec["title"] in html
        assert "<style>" in html and "</style>" in html  # inline CSS, standalone
        assert "<nav" in html and html.count('href="#sec-') == len(sections)
        assert html.count("data:image/png;base64,") == 3
        assert "<p>plain text without tags</p>" in html
        assert "<figcaption>\u51c0\u503c\u66f2\u7ebf</figcaption>" in html
        assert 'class="dataframe tbl"' in html
        assert f"{report.ic_mean:.4f}" in html  # floats formatted to 4 dp
        assert "(empty table)" not in html
        m = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
        assert m and base64.b64decode(m.group(1))[:8] == PNG_MAGIC
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING and "figure" in r.getMessage()]

    def test_missing_figure_path_warns_but_writes(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="alphaforge.reporting.html"):
            out = render_html_report(
                title="t",
                sections=[{"title": "s", "figures": [tmp_path / "does_not_exist.png"]}],
                out_path=tmp_path / "r.html",
            )
        html = out.read_text(encoding="utf-8")
        assert "missing figure" in html
        assert any("not found" in r.getMessage() for r in caplog.records)

    def test_bad_figure_type_raises(self, tmp_path):
        with pytest.raises(TypeError):
            render_html_report(title="t", sections=[{"title": "s", "figures": [12345]}], out_path=tmp_path / "r.html")
        with pytest.raises(TypeError):
            figure_to_data_uri(object())

    def test_titles_are_escaped_text_is_verbatim(self, tmp_path):
        out = render_html_report(
            title="a < b & c",
            sections=[{"title": "<x>", "text": "<pre>expr = cs_rank(x)</pre>"}],
            out_path=tmp_path / "r.html",
        )
        html = out.read_text(encoding="utf-8")
        assert "a &lt; b &amp; c" in html
        assert "&lt;x&gt;" in html and "<h2>1. <x>" not in html
        assert "<pre>expr = cs_rank(x)</pre>" in html


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
class TestResearchConfig:
    def test_defaults_match_spec(self):
        cfg = ResearchConfig()
        assert isinstance(cfg.sim, SimConfig) and cfg.sim == SimConfig()
        assert (cfg.horizon, cfg.lag, cfg.n_quantiles) == (5, 1, 10)
        assert (cfg.train_frac, cfg.valid_frac) == (0.6, 0.2)
        assert (cfg.population_size, cfg.n_generations, cfg.max_depth, cfg.n_jobs) == (100, 10, 4, 1)
        assert (cfg.hall_of_fame, cfg.max_corr, cfg.min_valid_icir_ann, cfg.max_factors) == (50, 0.7, 0.5, 15)
        assert (cfg.cost_bps, cfg.top_frac, cfg.rebalance_every) == (10.0, 0.1, 5)
        assert (cfg.out_dir, cfg.seed) == ("outputs", 42)
        assert cfg.test_frac == pytest.approx(0.2)

    def test_to_dict_nested_sim_is_plain_and_json_serialisable(self):
        d = ResearchConfig().to_dict()
        assert isinstance(d["sim"], dict) and d["sim"] == dataclasses.asdict(SimConfig())
        assert set(d) == {f.name for f in dataclasses.fields(ResearchConfig)}
        json.dumps(d)  # must not raise

    def test_round_trip(self):
        cfg = ResearchConfig(sim=SimConfig(n_assets=50, n_days=120, seed=7), horizon=10, n_jobs=2, out_dir="x")
        back = ResearchConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        assert back == cfg
        assert isinstance(back.sim, SimConfig) and back.sim == cfg.sim and back.sim is not cfg.sim

    def test_from_dict_partial_and_unknown_keys(self, caplog):
        cfg = ResearchConfig.from_dict({"horizon": 3, "sim": {"n_assets": 12}})
        assert cfg.horizon == 3 and cfg.sim.n_assets == 12 and cfg.sim.n_days == SimConfig().n_days
        with pytest.raises(ValueError, match="unknown keys"):
            ResearchConfig.from_dict({"horizn": 3})
        with pytest.raises(ValueError, match="SimConfig"):
            ResearchConfig.from_dict({"sim": {"n_asset": 3}})
        with caplog.at_level(logging.WARNING, logger="alphaforge.config"):
            lenient = ResearchConfig.from_dict({"horizn": 3, "lag": 2}, strict=False)
        assert lenient.lag == 2 and lenient.horizon == 5
        assert any("unknown keys" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize(
        "bad",
        [
            {"train_frac": 0.7, "valid_frac": 0.4},
            {"horizon": 0},
            {"max_corr": 0.0},
            {"n_quantiles": 1},
            {"top_frac": 0.0},
            {"cost_bps": -1.0},
            {"n_jobs": 0},
        ],
    )
    def test_validation_rejects(self, bad):
        with pytest.raises(ValueError):
            ResearchConfig(**bad)

    def test_split_index(self):
        cfg = ResearchConfig()
        assert cfg.split_index(1000) == (600, 800)
        a, b = ResearchConfig(train_frac=0.5, valid_frac=0.25).split_index(200)
        assert (a, b) == (100, 150)

    def test_replace_and_json_file(self, tmp_path):
        cfg = ResearchConfig(seed=1)
        cfg2 = cfg.replace(n_generations=3)
        assert cfg2.n_generations == 3 and cfg.n_generations == 10
        p = cfg2.to_json(tmp_path / "cfg" / "c.json")
        assert p.is_file()
        assert ResearchConfig.from_json(p) == cfg2
        with pytest.raises(ValueError):
            cfg.replace(train_frac=0.95)
