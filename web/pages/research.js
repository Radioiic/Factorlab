/* Factorlab — Search tools page (#/research).
 *
 * Three search tools, numbers only:
 *   1. Factor lab        — the append-only registry of the diffusion specs and their numbers
 *   2. AlphaForge        — mined formulas, each with a fixed forward window (only when it is on)
 *   3. Generator probe   — its command (the probe writes no file)
 *
 * Data: FL.data.research (engine/webexport/research.py). Every data-derived
 * string goes in via textContent (FL.ui.el). Page-local components are
 * prefixed research_ (see research_icBars, research_p).
 */
(function () {
  "use strict";
  var FL = window.FL;
  if (!FL || !FL.pages) return;

  var ui, el, fmt, charts;
  var DASH = "—";
  var STYLE_ID = "research-page-style";
  var WHERE = "Run this command from the repository root.";
  var LAB_COMMAND = "cd engine && .venv/bin/python -B factor_lab.py";

  /* ------------------------------------------------------------------ page-local style (static) */
  var CSS = [
    ".research-specc{display:inline-flex;align-items:baseline;gap:7px;white-space:nowrap}",
    ".research-fn-tiny{display:none}",
    "@media (max-width:599.98px){.research-fn-full{display:none}.research-fn-tiny{display:inline}}",
    ".research-spark{width:96px}",
    ".research-icbars{display:block}",
    ".research-icbars .zero{stroke:var(--axis);stroke-width:1}",
    ".research-kpis>.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}",
    "@media (max-width:719.98px){.research-kpis>.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}",
    "@media (max-width:359.98px){.research-kpis>.kpis{grid-template-columns:minmax(0,1fr)}}",
    ".research-exprs dd{font-size:12.5px}",
    ".research-expr{display:block;font-family:var(--mono);font-size:12px;overflow-wrap:anywhere}",
    ".research-sub{display:block;color:var(--ink-3);font-size:12px;margin-top:2px}",
    ".research-nowrap{white-space:nowrap}",
    ".research-status{margin:0;font-size:13.5px;color:var(--ink-2)}",
    ".research-status strong{color:var(--ink);font-weight:600}"
  ].join("\n");
  function research_style() {
    if (document.getElementById(STYLE_ID)) return;
    var s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  /* ------------------------------------------------------------------ small helpers */
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function orDash(f) { return function (v, r) { return v === null || v === undefined ? DASH : f(v, r); }; }
  var fIC = orDash(function (v) { return fmt.num(v, 4, { signed: true }); });
  var fIC3 = orDash(function (v) { return fmt.num(v, 3, { signed: true }); });
  var fT = orDash(function (v) { return fmt.num(v, 2, { signed: true }); });
  var fNum3 = orDash(function (v) { return fmt.num(v, 3); });
  var fNum2 = orDash(function (v) { return fmt.num(v, 2); });
  var fCorr = orDash(function (v) { return fmt.num(v, 2, { signed: true }); });

  /** p- and q-values: 0.0123 -> "0.012", 0.00208 -> "0.00208", 0.00095 -> "9.46e−4" (typographic minus). */
  function research_p(v) {
    if (!isNum(v)) return DASH;
    if (v >= 0.01) return v.toFixed(3);
    if (v >= 0.001) return v.toPrecision(3);
    if (v === 0) return "0";
    return v.toExponential(2).replace("e-", "e" + fmt.MINUS).replace("e+", "e");
  }
  function plural(n, one, many) { return fmt.int(n) + " " + (n === 1 ? one : many); }
  /** text-safe helper for numbers interpolated into prose: null/undefined -> "—" */
  function txt(v) { return v === null || v === undefined || v === "" ? DASH : String(v); }
  function minOf(rows, key) {
    var m = null;
    (rows || []).forEach(function (r) { if (isNum(r[key]) && (m === null || r[key] < m)) m = r[key]; });
    return m;
  }

  /**
   * research_icBars: fixed-size inline SVG of weekly IC bars (no axes). Blue = positive,
   * red = negative (the diverging palette), zero hairline. Colors are CSS vars so the
   * theme switch needs no redraw. Used inside table cells where a full chart is too big.
   */
  function research_icBars(values, o) {
    o = o || {};
    var vals = (values || []).map(function (v) { return isNum(v) ? v : null; });
    var W = o.width || 104, H = o.height || 26;
    var wrap = el("span", { "class": "research-nowrap" });
    if (!vals.length) { wrap.appendChild(el("span", { "class": "muted", text: o.emptyText || "no forward week" })); return wrap; }
    var amax = o.max || 0;
    vals.forEach(function (v) { if (v !== null) amax = Math.max(amax, Math.abs(v)); });
    if (!amax) amax = 1;
    var n = vals.length, gap = 2;
    var bw = Math.max(2, Math.min(12, (W - gap * (n - 1)) / n));
    var mid = H / 2, half = H / 2 - 1;
    var width = Math.ceil(n * bw + (n - 1) * gap);
    var svg = ui.svg("svg", { width: width, height: H, viewBox: "0 0 " + width + " " + H,
      "class": "research-icbars", role: "img", "aria-label": (o.label ? o.label + ": " : "") + vals.map(function (v) { return fIC3(v); }).join(", ") });
    vals.forEach(function (v, i) {
      if (v === null) return;
      var h = Math.max(1, Math.abs(v) / amax * half);
      var x = i * (bw + gap);
      svg.appendChild(ui.svg("rect", { x: x, y: v >= 0 ? mid - h : mid, width: bw, height: h, rx: 1.5,
        style: { fill: v >= 0 ? "var(--c1)" : "var(--c8)" }, title: (o.dates && o.dates[i] ? o.dates[i] + ": " : "") + fIC3(v) }));
    });
    svg.appendChild(ui.svg("line", { x1: 0, x2: n * bw + (n - 1) * gap, y1: mid + 0.5, y2: mid + 0.5, "class": "zero" }));
    wrap.appendChild(svg);
    return wrap;
  }

  /** a #/research/<id> deep link to a collapsible opens it */
  function deepOpen(ctx, id) { return !!(ctx && ctx.nav && ctx.nav.card === id); }

  function factorShort(name) { return String(name).replace(/^dif_/, "").replace(/_w90(?=_)/, ""); }
  /** narrow-screen row label: "dif_signed_corr_w90_t2_d0.7_l5" -> "S cor t2 d.7 l5" */
  function factorTiny(name) {
    var m = /^dif_(signed|abs)_([a-z]+)\d*_w\d+_t([\d.]+)_d([\d.]+)_l(\d+)$/.exec(String(name));
    if (!m) return factorShort(name);
    return (m[1] === "signed" ? "S" : "A") + " " + m[2].slice(0, 3) + " t" + m[3] + " d" + m[4].replace(/^0\./, ".") + " l" + m[5];
  }
  /** spec cell: short name (tiny on phones, via the page CSS), full name in the tooltip */
  function research_specName(name) {
    var full = String(name);
    return el("span", { title: full }, [
      el("span", { "class": "research-fn-full mono", text: factorShort(full) }),
      el("span", { "class": "research-fn-tiny mono", text: factorTiny(full) })
    ]);
  }
  /** wrap kpis in a page-local 3-up grid so six tiles sit as 3 x 2 instead of one cramped row */
  function research_kpis(container, items) {
    var wrap = el("div", { "class": "research-kpis" });
    container.appendChild(wrap);
    ui.kpis(wrap, items);
    return wrap;
  }
  /** the correlation with the nearest member and its name in the tooltip (format f, default signed) */
  function research_corrCell(v, name, n, f) {
    var t = (name ? "nearest: " + name : "no member with a correlation") + (isNum(n) ? " · " + plural(n, "incumbent", "incumbents") + " compared" : "");
    return el("span", { "class": "research-nowrap", title: t, text: (f || fCorr)(v) + (isNum(n) ? " (" + fmt.int(n) + ")" : "") });
  }

  /* ================================================================== 1. FACTOR LAB */
  function renderLab(root, L, ctx) {
    var sec = ui.section(root, { id: "factor-lab", title: "Factor lab",
      subtitle: "A daily search over a fixed grid of diffusion specifications. Each trial goes into an append-only registry. The page shows the numbers of each spec." });
    if (!L || !L.latest || !L.latest.length) {
      ui.empty(sec.body, "The factor lab has not run.", { title: "No registry", commands: [{ label: WHERE, command: LAB_COMMAND }] });
      return;
    }
    var m = L.meta || {}, latest = L.latest, hist = (L.history && L.history.factors) || {}, runDates = (L.history && L.history.run_dates) || [];
    var maxWeeks = 0, maxInc = null;
    latest.forEach(function (r) {
      maxWeeks = Math.max(maxWeeks, r.weeks || 0);
      if (isNum(r.n_incumbents)) maxInc = Math.max(maxInc === null ? 0 : maxInc, r.n_incumbents);
    });
    var dupNote = (m.duplicates || []).map(function (d) { return d.run_date + " logged " + d.copies + "×"; }).join(", ");

    // --- summary tiles
    var sum = ui.card({ id: "lab-summary", title: "Registry",
      subtitle: "Label " + (m.label || "fwd_ret_7") + " (the 7-day forward return, ranked on each Sunday). A spec with fewer than " + txt(m.min_weeks) + " weekly ICs gets no row." }, sec.body);
    research_kpis(sum.body, [
      { label: "Distinct specs", value: fmt.int(m.n_distinct), note: "in the registry; one more day for a spec is the same spec" },
      { label: "Registry run dates", value: fmt.int(m.n_runs),
        note: "run dates " + fmt.date(m.first_run, "short") + " → " + fmt.date(m.last_run) + (dupNote ? " · " + fmt.int(m.n_logged_runs) + " logged (" + dupNote + ", counted once)" : "") },
      { label: "Last run", value: fmt.date(m.last_run), note: plural(latest.length, "spec", "specs") + " on up to " + plural(maxWeeks, "Sunday", "Sundays") + (m.data_through ? " · data through " + fmt.date(m.data_through) : "") },
      { label: "Smallest q_bh", value: research_p(minOf(latest, "q_bh")), note: "Benjamini-Hochberg q over the latest p-value of each distinct spec" },
      { label: "Smallest p_bonferroni", value: research_p(minOf(latest, "p_bonferroni")), note: "min(1, p × " + plural(m.n_distinct, "distinct spec", "distinct specs") + ")" },
      { label: "Incumbents", value: maxInc === null ? DASH : fmt.int(maxInc), note: "your registered tests, the active AlphaForge formulas and the three risk proxies" }
    ]);
    if (maxInc === 0) ui.note(sum.body, "No incumbent: no registered test, no active zoo formula and no risk proxy.");

    // --- latest-run table: numbers only
    var trend = "t";
    var tLo = 0, tHi = 0, icLo = 0, icHi = 0;
    Object.keys(hist).forEach(function (f) {
      (hist[f].t || []).forEach(function (v) { if (isNum(v)) { tLo = Math.min(tLo, v); tHi = Math.max(tHi, v); } });
      (hist[f].ic || []).forEach(function (v) { if (isNum(v)) { icLo = Math.min(icLo, v); icHi = Math.max(icHi, v); } });
    });
    // the trend column only when a spec's t-stat moves by a visible amount across the runs
    var TREND_MIN = 0.5, tMove = 0;
    Object.keys(hist).forEach(function (f) {
      var v = (hist[f].t || []).filter(isNum);
      if (v.length > 1) tMove = Math.max(tMove, Math.max.apply(null, v) - Math.min.apply(null, v));
    });
    var showTrend = tMove >= TREND_MIN;
    var trendHost = el("div");
    var tCard = ui.card({ id: "lab-latest", title: "Latest run " + fmt.date(m.last_run),
      subtitle: "One row for each spec (a duplicated log is collapsed), sorted by p. Spec names omit the dif_ prefix and the shared _w90 window.",
      actions: showTrend ? [trendHost] : null }, sec.body);
    var tbl = ui.table(tCard.body, {
      compact: true, stickyFirst: true, defaultSort: { key: "p", dir: "asc" },
      caption: showTrend ? "The trend column spans " + plural(runDates.length, "run date", "run dates") + " on one shared scale."
        : "Across the " + plural(runDates.length, "run date", "run dates") + " no spec's t-stat moved by " + fmt.num(TREND_MIN, 1) + " or more (largest move " + fmt.num(tMove, 2) + "), so no trend column is shown.",
      columns: [
        { key: "factor", label: "Spec", align: "left", render: function (v) { return research_specName(v); } },
        { key: "weeks", label: "Weeks", format: "int", title: "Sunday cross-sections with an IC" },
        { key: "ic", label: "IC", format: fIC, align: "right", title: "mean weekly Spearman rank IC with fwd_ret_7" },
        { key: "t", label: "t", format: fT, align: "right", title: "IC / std × √weeks" },
        { key: "p", label: "p", align: "right", render: function (v) { return research_p(v); }, title: "two-sided Student-t p-value" },
        { key: "q_bh", label: "q_bh", align: "right", render: function (v) { return research_p(v); }, title: "Benjamini-Hochberg q over the latest p-value of each distinct spec" },
        { key: "p_bonferroni", label: "p_bonf", align: "right", render: function (v) { return research_p(v); }, title: "min(1, p × distinct specs)" },
        { key: "persistence", label: "Persist.", format: fNum3, align: "right", title: "mean week-over-week rank correlation" },
        { key: "coverage", label: "Coverage", align: "right", render: function (v) { return isNum(v) ? fmt.pct(v, 0, { signed: false }) : DASH; }, title: "mean share of the 20 coins with a value" },
        { key: "max_incumbent_corr", label: "Incumbent corr", align: "right", sortValue: function (r) { return isNum(r.max_incumbent_corr) ? Math.abs(r.max_incumbent_corr) : null; },
          title: "the signed mean Sunday Spearman correlation with the incumbent of the largest absolute value (the number of incumbents)",
          render: function (v, r) { return research_corrCell(v, r.closest_incumbent, r.n_incumbents); } },
        { key: "max_library_corr", label: "Library |corr|", align: "right",
          title: "the largest absolute correlation with a library definition on the G Sundays",
          render: function (v, r) { return research_corrCell(v, r.closest_library, null, fNum2); } },
        { key: "_trend", label: "Across runs", sortable: false, value: function (r) { return r.factor; }, render: function (v, r) {
          var h = hist[r.factor] || {};
          var vals = trend === "t" ? h.t : h.ic;
          var box = el("div", { "class": "research-spark" });
          charts.sparkline(box, { values: vals || [], color: "var(--ink-2)", height: 26,
            domain: trend === "t" ? [tLo, tHi] : [icLo, icHi], refLine: 0, format: trend === "t" ? "num:2:signed" : "num:3:signed",
            ariaLabel: r.factor + " " + (trend === "t" ? "t-stat" : "IC") + " across " + plural(runDates.length, "run", "runs") });
          return box;
        } }
      ].filter(function (c) { return showTrend || c.key !== "_trend"; }),
      rows: latest,
      emptyText: "No specs in the latest run."
    });
    if (showTrend) ui.segmented(trendHost, { ariaLabel: "Trend column metric", value: "t",
      options: [{ value: "t", label: "t-stat" }, { value: "ic", label: "IC" }],
      onChange: function (v) { trend = v; tbl.setRows(latest); } });
    ui.details(tCard.body, { id: "lab-detail", open: deepOpen(ctx, "lab-detail"), summary: "Spec parameters", meta: "turnover, nearest members, parameters", render: function (b) {
      ui.table(b, {
        compact: true, stickyFirst: true,
        caption: "Turnover = 1 − persistence.",
        columns: [
          { key: "factor", label: "Spec", align: "left", render: function (v) { return research_specName(v); } },
          { key: "turnover", label: "Turnover", format: fNum3, align: "right", title: "1 − persistence (weekly)" },
          { key: "closest_incumbent", label: "Nearest incumbent", mono: true, render: function (v) { return v || DASH; } },
          { key: "closest_library", label: "Nearest library definition", mono: true, render: function (v) { return v || DASH; } },
          { key: "mode", label: "Mode", value: function (r) { return (r.params || {}).mode; }, render: function (v) { return txt(v); } },
          { key: "kernel", label: "Kernel", value: function (r) { return (r.params || {}).kernel; }, render: function (v) { return txt(v); } },
          { key: "threshold", label: "Threshold", align: "right", value: function (r) { return (r.params || {}).threshold; }, render: function (v) { return txt(v); } },
          { key: "decay", label: "Decay", align: "right", value: function (r) { return (r.params || {}).decay; }, render: function (v) { return txt(v); } },
          { key: "lags", label: "Lags", align: "right", value: function (r) { return (r.params || {}).lags; }, render: function (v) { return txt(v); } }
        ],
        rows: latest
      });
    } });

    // --- persistence against t
    var sCard = ui.card({ id: "lab-scatter", title: "Persistence and t-stat",
      subtitle: "The latest run: one point for each spec." }, sec.body);
    var pts = [];
    latest.forEach(function (r) {
      if (!isNum(r.persistence) || !isNum(r.t)) return;
      pts.push({ x: r.persistence, y: r.t, label: factorShort(r.factor) });
    });
    charts.scatter(sCard.body, {
      series: [{ name: "specs", color: "var(--c1)", points: pts }],
      x: { format: "num:2", label: "persistence (weekly rank autocorrelation)" },
      y: { format: "num:2:signed", label: "t-stat", zero: true },
      refLines: [{ y: 0 }],
      labels: false, pointLabel: "spec", height: 280,
      tableCaption: "dif_ prefix and _w90 omitted from spec names"
    });

    // --- history heatmap
    var gapDays = runDates.length ? Math.round((fmt.parseDate(runDates[runDates.length - 1]) - fmt.parseDate(runDates[0])) / 864e5) + 1 - runDates.length : 0;
    var hCard = ui.card({ id: "lab-history", title: "Every spec across every run",
      subtitle: "Rows: the " + plural(latest.length, "spec", "specs") + " (dif_ prefix and the shared _w90 window omitted). Columns: " + plural(runDates.length, "run date", "run dates") +
        (gapDays > 0 ? " (" + gapDays + " calendar day" + (gapDays > 1 ? "s" : "") + " in between had no run)" : "") + ". " +
        "On narrow screens rows are abbreviated: S/A = signed/abs mode, kernel, threshold t, decay d, lags l." }, sec.body);
    var rows = latest.map(function (r) { return r.factor; });
    Object.keys(hist).forEach(function (f) { if (rows.indexOf(f) < 0) rows.push(f); });
    ui.details(hCard.body, { id: "lab-heatmap", open: deepOpen(ctx, "lab-heatmap") || deepOpen(ctx, "lab-history"), summary: "Show the heatmap",
      meta: plural(rows.length, "spec", "specs") + " × " + plural(runDates.length, "run", "runs"), render: function (b) {
        var metricHost = el("div", { "class": "mb-8" });
        b.appendChild(metricHost);
        function heatOpts(metric) {
          var values = rows.map(function (f) { return ((hist[f] || {})[metric] || []).slice(); });
          var o = { rows: rows, cols: runDates, values: values, scale: "div", rowLabel: "spec", cellHeight: 18,
            rowFormat: function (f) { return b.clientWidth && b.clientWidth < 640 ? factorTiny(f) : factorShort(f); },
            rowLabelWidth: 210, domain: null };
          if (metric === "t") { o.format = "num:2:signed"; o.legendLabel = "t-stat"; }
          else if (metric === "ic") { o.format = "num:3:signed"; o.legendLabel = "rank IC"; }
          else { o.format = "num:2:signed"; o.legendLabel = "persistence"; }
          return o;
        }
        var heat = charts.heatmap(b, heatOpts("t"));
        ui.segmented(metricHost, { ariaLabel: "Heatmap metric", value: "t",
          options: [{ value: "t", label: "t-stat" }, { value: "ic", label: "IC" }, { value: "persistence", label: "Persistence" }],
          onChange: function (v) { heat.update(heatOpts(v)); } });
      } });

    // --- the numbers of each run
    ui.details(sec.body, { id: "lab-runs", open: deepOpen(ctx, "lab-runs"), summary: "The numbers of each run", meta: plural((L.runs || []).length, "run date", "run dates"), render: function (b) {
      ui.table(b, {
        compact: true, maxHeight: 420,
        caption: "Each run with the registry as it was on that day.",
        defaultSort: { key: "run_date", dir: "desc" },
        columns: [
          { key: "run_date", label: "Run date" },
          { key: "specs", label: "Specs", format: "int" },
          { key: "copies", label: "Times logged", format: "int" },
          { key: "n_distinct", label: "Distinct specs", format: "int" },
          { key: "max_weeks", label: "Weeks", format: "int" },
          { key: "min_p", label: "Smallest p", align: "right", render: function (v) { return research_p(v); } },
          { key: "min_q_bh", label: "Smallest q_bh", align: "right", render: function (v) { return research_p(v); } },
          { key: "max_abs_t", label: "Largest |t|", format: fNum2, align: "right" }
        ],
        rows: L.runs || []
      });
    } });
  }

  /* ================================================================== 2. ALPHAFORGE */
  function renderZoo(root, A, ctx) {
    var sec = ui.section(root, { id: "alphaforge", title: "AlphaForge",
      subtitle: "A genetic-programming expression miner. Each mined formula has a fixed forward window; at most a set number of formulas are active." });
    if (!A || A.enabled !== true) {
      ui.empty(sec.body, "AlphaForge is off (ALPHAFORGE_ENABLED in engine/config.py).", { title: "Off" });
      return;
    }
    var st = A.settings || {}, cn = A.counts || {}, m = A.meta || {}, members = A.members || [], reg = A.registry || {};
    var W = st.forward_window_weeks;

    var sum = ui.card({ id: "af-summary", title: "Formulas",
      subtitle: "A forward week is a Sunday in the window of a formula after its selection date, with a complete 7-day label. A formula leaves the active list after " + txt(W) + " forward weeks; its record stays." }, sec.body);
    research_kpis(sum.body, [
      { label: "Active formulas", value: fmt.int(cn.active) + " / " + fmt.int(st.max_factors), note: "free places " + fmt.int(cn.room) },
      { label: "Formulas in the zoo", value: fmt.int(cn.members), note: "active and left" },
      { label: "Forward window", value: plural(W, "week", "weeks"), note: "for each formula" },
      { label: "Mining runs", value: fmt.int(m.n_cohorts), note: m.last_mining ? "last on " + fmt.date(m.last_mining) : "none" },
      { label: "Cadence", value: plural(st.mine_every_days, "day", "days"), note: "at most one mining run in this time, and only with a free place" },
      { label: "Expressions searched", value: fmt.int(reg.n_expressions_searched), note: fmt.int(reg.n_distinct_hypotheses) + " value-distinct hypotheses in the registry" }
    ]);

    // --- the formulas
    var mCard = ui.card({ id: "af-members", title: "Each formula",
      subtitle: "Forward columns: the window Sundays after the selection date with a complete 7-day label. Admission numbers: from the mining run, on the dates the miner used." }, sec.body);
    var maxAbsIC = 0;
    members.forEach(function (x) { ((x.forward || {}).ic_series || []).forEach(function (v) { if (isNum(v)) maxAbsIC = Math.max(maxAbsIC, Math.abs(v)); }); });
    function g(r, k) { return (r.gates || {})[k]; }
    ui.table(mCard.body, {
      compact: true, stickyFirst: true, stack: true,
      emptyText: "The zoo has no formula.",
      caption: "The expressions are listed under the table.",
      columns: [
        { key: "name", label: "Formula", mono: true, render: function (v) { return v ? el("span", { text: v }) : DASH; } },
        { key: "active", label: "Active", render: function (v, r) { return v ? "yes" : "no" + (r.left_on ? " (left " + fmt.date(r.left_on) + ")" : ""); } },
        { key: "selected_on", label: "Selected", render: function (v) { return v ? el("span", { "class": "research-nowrap", text: fmt.date(v) }) : DASH; } },
        { key: "window_end", label: "Window end", value: function (r) { return (r.window || {}).last_sunday; }, render: function (v) { return v ? fmt.date(v) : DASH; } },
        { key: "fw_weeks", label: "Fwd weeks", align: "right", value: function (r) { return (r.forward || {}).weeks; },
          render: function (v, r) { return (isNum(v) ? fmt.int(v) : "0") + " / " + txt((r.window || {}).weeks); } },
        { key: "fw_ic", label: "Fwd IC", align: "right", value: function (r) { return (r.forward || {}).ic; }, format: fIC },
        { key: "fw_t", label: "Fwd t", align: "right", value: function (r) { return (r.forward || {}).t; }, format: fT },
        { key: "fw_bars", label: "By week", sortable: false, value: function (r) { return r.name; }, render: function (v, r) {
          var f = r.forward || {};
          return research_icBars(f.ic_series, { dates: f.dates, max: maxAbsIC, label: r.name + " weekly forward IC" });
        } },
        { key: "valid_icir", label: "VALID ICIR", align: "right", value: function (r) { return g(r, "valid_icir"); }, format: fNum2 },
        { key: "valid_t", label: "VALID t", align: "right", value: function (r) { return g(r, "valid_t"); }, format: fT },
        { key: "persistence", label: "Persist.", align: "right", value: function (r) { return g(r, "persistence"); }, format: fNum3 },
        { key: "inc_corr", label: "Incumbent corr", align: "right", value: function (r) { return g(r, "max_incumbent_corr"); }, format: fCorr },
        { key: "lib_corr", label: "Library |corr|", align: "right", value: function (r) { return g(r, "max_library_corr"); },
          title: "the largest absolute correlation with a library definition on the G Sundays",
          render: function (v, r) { return research_corrCell(v, r.closest_library, null, fNum2); } }
      ],
      rows: members
    });
    if (members.length) {
      ui.dl(mCard.body, members.map(function (x) {
        var dd = el("span", null, [el("span", { "class": "research-expr", text: x.expr || DASH })]);
        dd.appendChild(el("span", { "class": "research-sub", text: "mining run " + txt(x.cohort) + " · window " + txt((x.window || {}).first_sunday) + " to " + txt((x.window || {}).last_sunday) }));
        return { term: x.name || DASH, value: dd };
      })).classList.add("research-exprs");
    }
    ui.note(mCard.body, "By week: blue bars = positive weekly forward IC, red = negative; all formulas share one scale. An expression starts with neg(…) when its training IC was negative.");

    // --- forward record per formula
    var fCard = ui.card({ id: "af-forward", title: "Forward record of each formula",
      subtitle: "The weekly Spearman IC with fwd_ret_7 on the forward weeks of each formula." }, sec.body);
    var withWeeks = members.filter(function (x) { return x.forward && x.forward.weeks > 0 && (x.forward.ic_series || []).length; });
    var CHART_MIN_WEEKS = 9;
    var maxFwd = 0;
    withWeeks.forEach(function (x) { maxFwd = Math.max(maxFwd, x.forward.ic_series.length); });
    if (withWeeks.length && maxFwd < CHART_MIN_WEEKS) {
      ui.note(fCard.body, "The longest forward record has " + plural(maxFwd, "week", "weeks") + ". The “By week” column shows each week; a chart for each formula shows from " + CHART_MIN_WEEKS + " weeks. " +
        withWeeks.map(function (x) { return x.name + ": " + plural(x.forward.weeks, "week", "weeks") + ", IC " + fIC3(x.forward.ic) + ", t " + fT(x.forward.t); }).join("; ") + ".");
    } else if (withWeeks.length) {
      var lim = Math.max(0.1, Math.ceil(maxAbsIC * 10) / 10);
      charts.smallMultiples(fCard.body, withWeeks.map(function (x) {
        var f = x.forward;
        var pos = f.ic_series.map(function (v) { return isNum(v) && v >= 0 ? v : null; });
        var neg = f.ic_series.map(function (v) { return isNum(v) && v < 0 ? v : null; });
        return {
          type: "bars", title: x.name + " · " + f.weeks + " wk · IC " + fIC3(f.ic) + " · t " + fT(f.t),
          x: f.dates, xType: "category", xTickFormat: function (d) { return fmt.date(d, "short"); }, xLabel: "forward Sunday (UTC)",
          stacks: [{ id: "pos", name: "IC ≥ 0", color: "var(--c1)", values: pos }, { id: "neg", name: "IC < 0", color: "var(--c8)", values: neg }],
          total: false, y: { format: "num:2:signed", label: "weekly IC", domain: [-lim, lim] }, height: 200,
          tableCaption: x.name + " weekly forward IC"
        };
      }), { columns: 3, minWidth: 240 });
    } else {
      ui.note(fCard.body, "No formula has a forward week. Forward IC and t print as — until then, never as 0.");
    }

    // --- settings, cycle log and mining runs
    var sCard = ui.card({ id: "af-settings", title: "Settings",
      subtitle: "The ALPHAFORGE settings of engine/config.py." }, sec.body);
    ui.dl(sCard.body, [
      ["Forward window", plural(W, "week", "weeks")],
      ["Active formulas", "at most " + txt(st.max_factors)],
      ["Cadence", "at most one mining run in " + plural(st.mine_every_days, "day", "days") + ", and only with a free place"],
      ["Search", "population " + txt(st.population) + " · generations " + txt(st.generations) + " · tree depth " + txt(st.max_depth)],
      ["VALID rule", "rank ICIR (annualized with 52 periods) ≥ " + txt(st.min_valid_icir_ann)],
      ["Incumbents", "your registered tests, the active AlphaForge formulas and the three risk proxies"],
      ["Miner inputs", (m.fields || []).join(", ") || DASH]
    ]);
    var cyCard = ui.card({ id: "af-cycle", title: "Cycle log",
      subtitle: "One line for each cycle of the zoo. The active count and the free places are the counts after the cycle." }, sec.body);
    var cycles = (A.cycles || []).map(function (c, i) {
      var detail = (c.n_evaluated ? "evaluated " + fmt.int(c.n_evaluated) + " · " : "") +
        (c.added && c.added.length ? "added " + c.added.join(", ") + " · " : "") + (c.left && c.left.length ? "left " + c.left.join(", ") + " · " : "") +
        "after the cycle: active " + txt(c.active) + ", free places " + txt(c.room) + " · data through " + txt(c.data_through);
      return { i: i + 1, date: c.date, mined: c.mined, detail: detail };
    });
    if (!cycles.length) ui.note(cyCard.body, "The cycle log has no line.");
    else ui.table(cyCard.body, { ariaLabel: "Cycle log",
      compact: true, maxHeight: 360, rows: cycles, defaultSort: { key: "i", dir: "desc" },
      columns: [
        { key: "i", label: "Line", format: "int" },
        { key: "date", label: "Date" },
        { key: "mined", label: "Mining", render: function (v) { return v ? "mined" : "no mining"; } },
        { key: "detail", label: "Detail", wrap: true, sortable: false }
      ]
    });
    ui.details(cyCard.body, { id: "af-mining", open: deepOpen(ctx, "af-mining"), summary: "Mining runs", meta: plural((reg.runs || []).length, "run", "runs"), render: function (b) {
      ui.table(b, {
        compact: true, rows: reg.runs || [],
        caption: "From alphaforge_registry.parquet: each hall-of-fame expression of each run.",
        columns: [
          { key: "run_id", label: "run_id", mono: true },
          { key: "evaluated", label: "Evaluated", format: "int" },
          { key: "hall_of_fame", label: "Hall of fame", format: "int" },
          { key: "valid_rule", label: "VALID rule met", format: "int" },
          { key: "selected", label: "Added", format: "int" },
          { key: "selected_names", label: "Names", render: function (v) { return (v || []).join(", ") || DASH; } }
        ]
      });
    } });
  }

  /* ================================================================== 3. GENERATOR PROBE */
  function renderProbe(root, P) {
    var sec = ui.section(root, { id: "probe", title: "Generator probe",
      subtitle: "The probe makes random formulas from 25 input series and the library operators. For each formula it gives the largest absolute correlation with the library definitions, your registered tests, the AlphaForge formulas and the risk proxies. It reads no return label." });
    var c = ui.card({ id: "probe-command", title: "Command" }, sec.body);
    P = P || {};
    ui.note(c.body, P.where || WHERE);
    c.body.appendChild(ui.code(P.command || "", { block: true }));
    ui.note(c.body, P.text || "");
  }

  /* ================================================================== status line */
  /** One line with the numbers of each tool, computed from the payload. */
  function research_status(root, d) {
    var parts = [];
    var L = d.factor_lab;
    if (L && L.latest && L.latest.length) {
      parts.push(["Factor lab", plural(L.latest.length, "spec", "specs") + ", smallest q_bh " + research_p(minOf(L.latest, "q_bh"))]);
    }
    var A = d.alphaforge;
    if (A && A.enabled === true) {
      var cn = A.counts || {}, st = A.settings || {};
      parts.push(["AlphaForge", fmt.int(cn.active) + " of at most " + fmt.int(st.max_factors) + " formulas active"]);
    }
    if (!parts.length) return;
    var p = el("p", { "class": "research-status" });
    parts.forEach(function (x, i) {
      if (i) p.appendChild(document.createTextNode(" · "));
      p.appendChild(el("strong", { text: x[0] + ": " }));
      p.appendChild(document.createTextNode(x[1]));
    });
    root.appendChild(p);
  }

  /* ================================================================== page */
  FL.pages.register({
    id: "research",
    title: "Search tools",
    subtitle: "Numbers from the factor lab, the AlphaForge miner and the generator probe. To test a factor in a book, register it with judge.py.",
    icon: "research",
    requiresData: true,
    render: function (root, ctx) {
      ui = FL.ui; el = ui.el; fmt = FL.fmt; charts = FL.charts;
      research_style();
      var d = ctx.section || {};
      ui.note(root, "Data as of " + (d.as_of ? fmt.date(d.as_of) : (ctx.meta && ctx.meta.data_through) || DASH) + " UTC close.");
      research_status(root, d);
      ui.toc(root, [["factor-lab", "Factor lab"], ["lab-latest", "Latest run"], ["alphaforge", "AlphaForge"], ["probe", "Generator probe"]],
        { page: "research", href: function (id) { return ctx.nav.href("research", id); } });
      renderLab(root, d.factor_lab, ctx);
      // the sections below the first screen: drawn after the first paint (a deep link draws up to its card)
      var defer = ctx.defer || ui.defer;
      defer(root, { id: "alphaforge", minHeight: 400 }, function (h) { renderZoo(h, d.alphaforge, ctx); });
      defer(root, { id: "probe", minHeight: 200 }, function (h) { renderProbe(h, d.probe); });
    }
  });
})();
