"""Standalone HTML research report: inline CSS, base64 figures, UTF-8.

:func:`render_html_report` turns a list of sections -- each a title, some
HTML text, an optional table and some figures -- into **one self-contained
file**: the CSS is inline, every figure is embedded as a base64 ``data:``
URI, and the document declares ``<meta charset="utf-8">`` so Chinese (or
any Unicode) text renders correctly wherever the file is opened.

The report is one self-contained HTML file with tables, pictures and the
exact configuration, and it opens without other tools.
"""

from __future__ import annotations

import base64
import datetime as dt
import html as html_lib
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from alphaforge.reporting.plots import SAVE_DPI, fig_to_png_bytes

logger = logging.getLogger("alphaforge.reporting.html")

__all__ = ["render_html_report", "figure_to_data_uri", "DEFAULT_CSS"]

_MIME_BY_SUFFIX: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}

#: Inline stylesheet (clean, readable; striped tables; figures never overflow).
DEFAULT_CSS: str = """
:root { --fg: #1f2937; --muted: #6b7280; --bg: #f7f8fa; --card: #ffffff; --line: #e5e7eb;
        --accent: #264653; --stripe: #f3f6f9; --hover: #e8f1f5; --head: #eef2f6; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); line-height: 1.55; font-size: 15px;
       font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial,
                    "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 28px 32px 64px; }
header h1 { margin: 0 0 4px; font-size: 1.9em; color: var(--accent); }
header .subtitle { margin: 0 0 4px; font-size: 1.05em; color: var(--fg); }
header .meta { color: var(--muted); font-size: 0.9em; }
nav.toc { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
          padding: 12px 20px; margin: 20px 0 28px; }
nav.toc strong { color: var(--accent); }
nav.toc ol { margin: 6px 0 0; padding-left: 1.4em; }
nav.toc a { text-decoration: none; color: var(--accent); }
nav.toc a:hover { text-decoration: underline; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
          padding: 18px 24px; margin-bottom: 24px; }
section h2 { margin: 0 0 10px; font-size: 1.35em; border-bottom: 2px solid var(--line); padding-bottom: 6px; }
section h2 a.top { font-size: 0.6em; font-weight: normal; margin-left: 10px; color: var(--muted); text-decoration: none; }
.text p:first-child { margin-top: 6px; }
.tbl-wrap { overflow-x: auto; margin: 12px 0; }
table.tbl { border-collapse: collapse; font-size: 0.86em; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.tbl th, table.tbl td { padding: 5px 10px; border-bottom: 1px solid var(--line); text-align: right; }
table.tbl thead th { background: var(--head); position: sticky; top: 0; }
table.tbl thead th:first-child, table.tbl tbody th { text-align: left; }
table.tbl tbody th { font-weight: 600; }
table.tbl tbody tr:nth-child(even) { background: var(--stripe); }
table.tbl tbody tr:hover { background: var(--hover); }
figure { margin: 16px 0; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
figcaption { color: var(--muted); font-size: 0.86em; margin-top: 6px; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 0.9em; }
pre { background: #f6f8fa; padding: 10px 12px; border-radius: 6px; overflow-x: auto; }
.muted { color: var(--muted); }
footer { color: var(--muted); font-size: 0.85em; margin-top: 32px; text-align: center; }
"""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _esc(text: Any) -> str:
    return html_lib.escape(str(text), quote=True)


def _slug(title: str, position: int) -> str:
    """Stable, ASCII-only anchor id (Chinese titles fall back to the position)."""
    core = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()[:40]
    return f"sec-{position}" + (f"-{core}" if core else "")


def figure_to_data_uri(item: Any, *, dpi: int = SAVE_DPI) -> str | None:
    """Encode a figure as a base64 ``data:`` URI.

    Args:
        item: A :class:`matplotlib.figure.Figure` (rendered to PNG in memory),
            an object with a ``.figure`` attribute (an ``Axes``), a ``str`` /
            :class:`pathlib.Path` to an image file (bytes read verbatim, MIME
            type by suffix, PNG assumed otherwise), or raw PNG ``bytes``.
        dpi: Resolution used when rendering a ``Figure``.

    Returns:
        The ``data:`` URI, or ``None`` when ``item`` is a path that does not
        exist (a warning is logged so a missing figure never aborts a report).

    Raises:
        TypeError: for unsupported ``item`` types.
    """
    if hasattr(item, "figure") and isinstance(item.figure, Figure) and not isinstance(item, Figure):
        item = item.figure
    if isinstance(item, Figure):
        data, mime = fig_to_png_bytes(item, dpi=dpi), "image/png"
    elif isinstance(item, (str, Path)):
        path = Path(item)
        if not path.is_file():
            logger.warning("figure file not found, skipped: %s", path)
            return None
        data, mime = path.read_bytes(), _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    elif isinstance(item, (bytes, bytearray, memoryview)):
        data, mime = bytes(item), "image/png"
    else:
        raise TypeError(f"unsupported figure type {type(item).__name__}: expected Figure, path or PNG bytes")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _render_text(text: Any) -> str:
    """Section body: HTML passes through verbatim; tag-free strings are wrapped in ``<p>``."""
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        return "".join(_render_text(t) for t in text)
    s = str(text)
    if not s.strip():
        return ""
    if "<" in s:
        return s  # trusted HTML from the caller (may contain <p>/<ul>/<pre>)
    return f"<p>{_esc(s)}</p>"


def _render_table(table: Any, float_format: str) -> str:
    if table is None:
        return ""
    if isinstance(table, pd.Series):
        table = table.to_frame()
    if isinstance(table, Mapping):
        table = pd.DataFrame(table)
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"section 'table' must be a DataFrame / Series / mapping, got {type(table).__name__}")
    if table.empty:
        return '<p class="muted">(empty table)</p>'
    body = table.to_html(float_format=float_format.format, border=0, classes="tbl", na_rep="", escape=True)
    return f'<div class="tbl-wrap">{body}</div>'


def _render_figures(figures: Any, section_title: str, dpi: int) -> str:
    if figures is None:
        return ""
    if isinstance(figures, (Figure, str, Path, bytes, tuple)) or hasattr(figures, "figure"):
        figures = [figures]
    parts: list[str] = []
    for j, item in enumerate(figures, 1):
        caption: str | None = None
        if isinstance(item, tuple):
            if len(item) != 2:
                raise TypeError("a figure tuple must be (figure_or_path, caption)")
            item, caption = item
        uri = figure_to_data_uri(item, dpi=dpi)
        if uri is None:
            parts.append(f'<p class="muted">[missing figure: {_esc(item)}]</p>')
            continue
        alt = caption or f"{section_title} — figure {j}"
        cap_html = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
        parts.append(f'<figure><img src="{uri}" alt="{_esc(alt)}">{cap_html}</figure>')
    return "".join(parts)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def render_html_report(
    *,
    title: str,
    sections: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    subtitle: str | None = None,
    float_format: str = "{:.4f}",
    dpi: int = SAVE_DPI,
    generated_at: dt.datetime | None = None,
    css: str | None = None,
) -> Path:
    """Write a standalone HTML report and return its path.

    Args:
        title: Document title (escaped; may be Chinese).
        sections: One mapping per section with the keys
            ``"title"`` (str), ``"text"`` (HTML string -- may contain
            ``<p>``/``<ul>``/``<pre>``; a tag-free string is wrapped in ``<p>``;
            a list of strings gives one paragraph each), ``"table"``
            (``DataFrame`` / ``Series`` / ``None``) and ``"figures"`` (list of
            ``Figure`` | path | PNG bytes; an item may also be a
            ``(figure, caption)`` tuple).  Every key except ``"title"`` is optional.
        out_path: Output file; parent directories are created.
        subtitle: Optional line under the title.
        float_format: ``str.format`` pattern applied to float table cells.
        dpi: Resolution for figures rendered from ``Figure`` objects.
        generated_at: Timestamp shown in the header (default: now).
        css: Replace :data:`DEFAULT_CSS` entirely.

    Returns:
        :class:`pathlib.Path` of the written file.

    Security note: section ``text`` is inserted verbatim so callers can use
    markup; escape untrusted content with :func:`html.escape` first.  Titles
    and table cells are always escaped.

    Each run gives one HTML file with the configuration, the tables and the
    pictures; ``run.json`` makes it again.
    """
    when = generated_at or dt.datetime.now()
    section_ids: list[tuple[str, str]] = []
    body_parts: list[str] = []
    for i, sec in enumerate(sections, 1):
        if not isinstance(sec, Mapping):
            raise TypeError(f"section {i} must be a mapping, got {type(sec).__name__}")
        sec_title = str(sec.get("title") or f"Section {i}")
        sid = _slug(sec_title, i)
        section_ids.append((sid, sec_title))
        body_parts.append(
            f'<section id="{sid}">'
            f'<h2>{i}. {_esc(sec_title)}<a class="top" href="#top">↑ top</a></h2>'
            f'<div class="text">{_render_text(sec.get("text"))}</div>'
            f'{_render_table(sec.get("table"), float_format)}'
            f'{_render_figures(sec.get("figures"), sec_title, dpi)}'
            f"</section>"
        )

    toc = "".join(f'<li><a href="#{sid}">{_esc(t)}</a></li>' for sid, t in section_ids)
    subtitle_html = f'<p class="subtitle">{_esc(subtitle)}</p>' if subtitle else ""
    doc = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n<style>{css if css is not None else DEFAULT_CSS}</style>\n</head>\n<body>\n"
        '<div class="wrap" id="top">\n'
        f"<header><h1>{_esc(title)}</h1>{subtitle_html}"
        f'<p class="meta">Generated by AlphaForge · {when.strftime("%Y-%m-%d %H:%M:%S")} · {len(section_ids)} sections</p></header>\n'
        f"<nav class=\"toc\"><strong>Contents</strong><ol>{toc}</ol></nav>\n"
        + "\n".join(body_parts)
        + "\n<footer>AlphaForge — simulated quant factor-mining research platform</footer>\n"
        "</div>\n</body>\n</html>\n"
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    logger.info(
        "wrote HTML report (%d sections, %.1f KB) -> %s", len(section_ids), len(doc.encode("utf-8")) / 1024.0, out
    )
    return out
