/* Factorlab — Factor library page (#/library).
 *
 * The page shows numbers and definitions only (top to bottom):
 *   1. one plain sentence with the counts, the method notes in a closed details, and an
 *      "On this page" list (ui.toc);
 *   2. "On Factorlab's data": one summary table of both screens, then the OKX screen of
 *      engine/libfactors.py (status.parquet, screen_g.parquet, manifest.json) and the block
 *      "Binance data for the same 20 coins" (libfactors.py --source binance; Binance is a data
 *      source only; a new family with its own Benjamini-Hochberg and a cross-check that gives
 *      agreement numbers only);
 *   3. "The whole library": KPIs, the search (each definition with its formula and its check
 *      and register commands) and the families.
 *
 * Data: FL.data.library (engine/webexport/library.py). The search index is the part file
 * data/library.index.js (FL.data["library.index"]), loaded through ctx.loadPart when the
 * search card comes near the screen or the search box gets focus; the formulas are the part
 * files data/library.expr-<k>.js, loaded when the search shows a definition of the part (and
 * all of them for a search with a query). The decoded index, the formulas and the search
 * haystack are cached in module scope. Every data-derived string goes in via textContent
 * (FL.ui.el); the id of a definition goes into its commands with textContent too. Cards below
 * the first screen render through ctx.defer. Page-local helpers and classes are prefixed
 * library_.
 */
(function () {
  "use strict";
  var FL = window.FL;
  if (!FL || !FL.pages) return;

  var ui, el, fmt, charts, colors;
  var DASH = "—";
  var STYLE_ID = "library-page-style";
  var MAX_RESULTS = 200;

  /* ------------------------------------------------------------------ page-local style (static) */
  var CSS = [
    ".library-funnel{display:flex;flex-direction:column;gap:10px;margin:0;padding:0;list-style:none}",
    ".library-step{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr) 10.5em;gap:4px 14px;align-items:center}",
    ".library-step-label{font-size:13.5px;color:var(--ink)}",
    ".library-step-note{display:block;font-size:12px;color:var(--ink-3);margin-top:2px}",
    ".library-track{position:relative;height:10px;border-radius:5px;background:var(--surface-2);box-shadow:inset 0 0 0 1px var(--border)}",
    ".library-fill{position:absolute;left:0;top:0;bottom:0;border-radius:5px;background:var(--ink-2);min-width:2px}",
    ".library-step-n{font-variant-numeric:tabular-nums;font-weight:600;text-align:right;white-space:nowrap}",
    ".library-step-n small{display:block;font-weight:400;font-size:12px;color:var(--ink-3)}",
    "@media (max-width:639.98px){.library-step{grid-template-columns:minmax(0,1fr) 8.5em}.library-track{grid-column:1 / -1;grid-row:2}}",
    ".library-expr{display:block;font-family:var(--mono);font-size:12px;overflow-wrap:anywhere;color:var(--ink-2)}",
    ".library-sub{display:block;font-size:12px;color:var(--ink-3);margin-top:2px;white-space:normal}",
    ".library-filter{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px}",
    ".library-filter label{font-size:13px;color:var(--ink-2)}",
    // the controls use the shared .fl-input / .fl-select classes (border var(--control-border), >= 3:1)
    ".library-filter input,.library-filter select{min-width:0}",
    ".library-pair{display:inline-flex;align-items:center;gap:8px;min-width:0}",
    ".library-pair-q{flex:1 1 280px;max-width:420px}",
    ".library-filter input{width:100%}",
    ".library-count{font-size:12.5px;color:var(--ink-3)}",
    ".library-badges{display:inline-flex;flex-wrap:wrap;gap:4px;align-items:center}",
    // KPI rows: one column per tile on wide screens (--n tiles), so no row has an empty slot
    ".library-kpis>.kpis{grid-template-columns:repeat(var(--n,4),minmax(0,1fr))}",
    "@media (max-width:959.98px){.library-kpis>.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}",
    "@media (max-width:399.98px){.library-kpis>.kpis{grid-template-columns:minmax(0,1fr)}}",
    // the zone bar: G_fit and G_eval in two tints, embargo slots outlined, the sealed I zone hatched
    ".library-zones{display:flex;gap:2px;height:14px;margin:2px 0 4px}",
    ".library-zone{min-width:3px;border-radius:3px;background:var(--surface-2);box-shadow:inset 0 0 0 1px var(--ink-3)}",
    ".library-zone.is-g{background:var(--ink-2);box-shadow:none}",
    ".library-zone.is-g-eval{background:var(--ink-3);box-shadow:none}",
    ".library-zone.is-sealed{background:repeating-linear-gradient(135deg,var(--surface-2) 0 4px,var(--ink-3) 4px 6px);box-shadow:inset 0 0 0 1px var(--ink-3)}",
    ".library-zone-labels{display:flex;gap:2px;margin:0 0 12px;font-size:12px;color:var(--ink-2)}",
    ".library-zone-labels span{min-width:3px;overflow:hidden;white-space:nowrap;text-overflow:clip}",
    ".library-swatch{display:inline-block;width:18px;height:10px;margin-right:8px;vertical-align:middle;flex:none}",
    ".library-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px 16px;margin-bottom:10px}",
    ".library-step.is-branch .library-fill{background:var(--muted)}",
    ".library-step.is-branch .library-step-label{color:var(--ink-2)}",
    ".library-binance{margin-top:8px;padding-top:20px;border-top:2px solid var(--border-strong)}",
    ".library-binance-head{margin-bottom:14px}",
    ".library-binance-head h3{font-size:16px}",
    ".library-binance-head p{color:var(--ink-2);margin-top:4px;max-width:80ch}",
    ".library-formula-col{min-width:17rem}",
    ".library-cmds{display:flex;flex-direction:column;gap:3px;min-width:22rem}",
    ".library-cmds code{font-size:11.5px;overflow-wrap:anywhere;white-space:normal}",
    "@media (max-width:599.98px){.library-cmds{min-width:0}}",
    ".library-sub.is-nowrap,.library-badges.is-nowrap{white-space:nowrap;flex-wrap:nowrap}"
  ].join("\n");
  function library_style() {
    if (document.getElementById(STYLE_ID)) return;
    var s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  /* ------------------------------------------------------------------ small helpers */
  function arr(x) { return Array.isArray(x) ? x : []; }
  function obj(x) { return x && typeof x === "object" && !Array.isArray(x) ? x : null; }
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function orDash(f) { return function (v, r) { return v === null || v === undefined ? DASH : f(v, r); }; }
  var fT = orDash(function (v) { return fmt.num(v, 2, { signed: true }); });
  var fI = orDash(function (v) { return fmt.int(v); });
  function words(s) { return String(s || "").replace(/_/g, " "); }

  // the standard remedies (./setup.sh, export_web.py) of the app shell; a plain command without the shell
  function library_fix() {
    return ui.remedies ? { commands: ui.remedies() } : { command: "./setup.sh\ncd engine && .venv/bin/python -B export_web.py" };
  }
  function library_missing(card, what) {
    var o = library_fix();
    o.title = "Not available";
    ui.empty(card.body, (what || "This block") + " is not in the exported data.", o);
  }
  // a card below the first screen renders later (ctx.defer); without the shell it renders at once
  var library_defer = function (container, id, fn) { fn(container); };
  // built outside render, so the closure holds only the defer function, not the render scope (ctx, the page tree)
  function library_makeDefer(f) {
    return f ? function (container, id, fn) { f(container, { id: id }, fn); } : function (container, id, fn) { fn(container); };
  }

  /* ------------------------------------------------------------------ index decoding */
  // The decoded index, the formulas and the search haystack live in module scope: the data does
  // not change while the page is open, so a second visit reuses them. The haystack is built on the
  // first query, and again when a part of formulas arrives.
  var IX_CACHE = { src: null, ix: null, hay: null, hayParts: -1 };
  var EXPR = {};   // part name -> its payload {start, n, expr}
  function library_decode(idx) {
    if (idx && IX_CACHE.src === idx && IX_CACHE.ix) return IX_CACHE.ix;
    var all = obj(idx && idx.all);
    if (!all) return null;
    var n = all.n || arr(all.s).length;
    var names = new Array(n), prev = "";
    var p = arr(all.p), s = arr(all.s);
    for (var i = 0; i < n; i++) { prev = prev.slice(0, p[i] || 0) + (s[i] || ""); names[i] = prev; }
    var fam = new Array(n);
    arr(all.fam_runs).forEach(function (r) { for (var k = r[1]; k < r[1] + r[2] && k < n; k++) fam[k] = r[0]; });
    var digits = all.id_digits || 5, start = all.id_start || 1, prefix = all.id_prefix || "CF";
    function id(i) { var x = String(i + start); while (x.length < digits) x = "0" + x; return prefix + x; }
    var ix = { n: n, names: names, fam: fam, cat: arr(all.cat), req: arr(all.req), source: arr(all.source), id: id };
    IX_CACHE = { src: idx, ix: ix, hay: null, hayParts: -1 };
    return ix;
  }
  /** the formula of definition i, or null while its part is not loaded */
  function library_expr(i, chunk, parts) {
    var name = parts[Math.floor(i / chunk)];
    var part = name ? EXPR[name] : null;
    return part ? arr(part.expr)[i - part.start] : null;
  }
  // one lower-case haystack per definition: id, name, family, category and the formula (when loaded)
  function library_haystack(idx, ix, chunk, parts) {
    var loaded = Object.keys(EXPR).length;
    if (IX_CACHE.ix === ix && IX_CACHE.hay && IX_CACHE.hayParts === loaded) return IX_CACHE.hay;
    var fams = arr(idx.families), cats = arr(idx.categories);
    var hay = new Array(ix.n);
    for (var i = 0; i < ix.n; i++) {
      hay[i] = (ix.id(i) + " " + ix.names[i] + " " + (fams[ix.fam[i]] || "") + " " + (cats[ix.cat[i]] || "") + " " +
        (library_expr(i, chunk, parts) || "")).toLowerCase();
    }
    if (IX_CACHE.ix === ix) { IX_CACHE.hay = hay; IX_CACHE.hayParts = loaded; }
    return hay;
  }
  /** the check and register commands of a library id ({id} of the payload's templates, put in as text) */
  function library_cmds(cmds, id) {
    var box = el("span", { "class": "library-cmds" });
    ["check", "register"].forEach(function (k) {
      if (cmds && typeof cmds[k] === "string") box.appendChild(ui.code(cmds[k].split("{id}").join(id)));
    });
    return box;
  }

  /* ------------------------------------------------------------------ funnel */
  function library_funnel(container, steps, note) {
    var top = 0;
    steps.forEach(function (s) { if (isNum(s.n) && s.n > top) top = s.n; });
    var list = el("ol", { "class": "library-funnel" });
    steps.forEach(function (s) {
      var share = top > 0 && isNum(s.n) ? s.n / top : 0;
      var nBox = el("div", { "class": "library-step-n", text: isNum(s.n) ? fmt.int(s.n) : DASH });
      if (isNum(s.signals)) nBox.appendChild(el("small", { text: fmt.int(s.signals) + (s.signals === 1 ? " signal" : " distinct signals") }));
      else if (s.branch) nBox.appendChild(el("small", { text: s.branch }));
      var label = el("div", null, [el("span", { "class": "library-step-label", text: s.label || s.key })]);
      if (s.note) label.appendChild(el("span", { "class": "library-step-note", text: s.note }));
      var track = el("div", { "class": "library-track", attrs: { "aria-hidden": "true" }, title: isNum(s.n) && top ? fmt.pct(share, 2, { signed: false }) + " of the library" : null },
        [el("div", { "class": "library-fill", style: { width: (100 * share).toFixed(3) + "%" } })]);
      list.appendChild(el("li", { "class": ["library-step", s.branch ? "is-branch" : null] }, [label, track, nBox]));
    });
    container.appendChild(list);
    ui.note(container, note || "Bars are drawn to scale against the whole library.");
  }

  /* ------------------------------------------------------------------ cards */
  // the first line of the page: one plain sentence from the data; the method notes in a closed details
  function renderStatement(root, d, nav) {
    var t = obj(d.totals) || {};
    var lead = fmt.int(t.definitions) + " formula definitions. Numbers from two screens on Factorlab's data. Each definition has a command that checks it and registers it for a test.";
    var box = el("div", { id: "statement" });
    root.appendChild(box);
    ui.callout("info", null, lead, { container: box });
    var lines = ["The catalog holds " + fmt.int(t.definitions) + " definitions in " + fmt.int(t.families) + " families. The search shows each definition with its formula and its two commands."];
    var e = obj(d.engine), n = e && e.available !== false ? obj(e.counts) : null;
    if (n) lines.push("On Factorlab's own data (its " + (e.meta && isNum(e.meta.symbols) ? fmt.int(e.meta.symbols) + " " : "") + "OKX perpetuals) " + fmt.int(n.computable) + " definitions are computable, " +
      fmt.int(n.pit_pass) + " are verified point in time (" + fmt.int(n.pit_no_finite) + (n.pit_no_finite > n.empty
        ? " had no value to compare and are not counted: " + fmt.int(n.empty) + " empty ones, and " + fmt.int(n.pit_no_finite - n.empty) + " whose first value comes after the last cut"
        : " empty ones had no value to compare and are not counted") + ") and " + fmt.int(n.screened) + " were screened on the generator zone (section On Factorlab's data).");
    var b = obj(d.binance), bn = b && b.available !== false ? obj(b.counts) : null;
    if (bn) lines.push("On Binance's public data for the same coins (a data source only) all " + fmt.int(bn.computed) + " definitions are computed and " +
      (bn.pit_pass === bn.computed ? "" : fmt.int(bn.pit_pass) + " ") + "verified point in time. The " + fmt.int(bn.new_screened) +
      " screened there that the OKX data could not screen form a separate Benjamini-Hochberg family; the " + fmt.int(bn.crosscheck) + " screened on OKX get agreement numbers only.");
    lines.push("The screens give the weekly IC and the residual IC (rIC) of each definition on the generator zone G, with Benjamini-Hochberg q-values. The rIC removes the symbol tilt, the BTC beta and the size. A low q or a large t is a number. It is not a test of a book.");
    ui.details(box, { summary: "How to read this page", meta: "method notes", render: function (body) {
      lines.forEach(function (x) { body.appendChild(el("p", { "class": "fl-note", text: x })); });
      var href = nav && nav.href ? nav.href("system", "glossary") : "#/system/glossary";
      body.appendChild(el("p", { "class": "fl-note" }, ["Terms such as rIC, BH q, G zone and point in time: ", el("a", { attrs: { href: href }, text: "System page, Glossary" }), "."]));
    } });
  }

  function library_kpiBox(root, id, items) {
    var box = el("div", { id: id, "class": "library-kpis", style: { "--n": String(items.length) } });
    root.appendChild(box);
    ui.kpis(box, items);
    return box;
  }

  function renderKpis(root, d) {
    var t = obj(d.totals) || {};
    var rej = obj(t.rejected) || {};
    var nRej = 0;
    Object.keys(rej).forEach(function (k) { if (isNum(rej[k])) nRej += rej[k]; });
    var e = obj(d.engine), n = e && e.available !== false ? obj(e.counts) : null;
    var b = obj(d.binance), bn = b && b.available !== false ? obj(b.counts) : null;
    library_kpiBox(root, "kpis", [
      { label: "Definitions", value: t.definitions, format: "int",
        note: fmt.int(t.candidates) + " generated, " + fmt.int(nRej) + " rejected at build time" },
      { label: "Families", value: t.families, format: "int", note: "families of formula templates" },
      { label: "Categories", value: t.categories, format: "int", note: "the information sets of the input series" },
      { label: "Computable on OKX data", value: n ? n.computable : null, format: "int", note: n ? "every raw field is in Factorlab's OKX data" : "not computed in this data folder" },
      { label: "Computable on Binance data", value: bn ? bn.computed : null, format: "int", note: bn ? "on Binance's public data for the same coins" : "not computed in this data folder" }
    ]);
  }

  function renderFamilies(root, d) {
    var fams = arr(d.families);
    var reqs = arr(d.requirements).length ? arr(d.requirements) : null;
    var c = ui.card({ id: "families", title: "Families", subtitle: "Definitions per family, by the data they need." }, root);
    if (!fams.length) { library_missing(c, "The family breakdown"); return; }
    var x = fams.map(function (f) { return f.family; });
    var labels = ["close and dollar volume only", "other kline fields", "perpetual-futures data"];
    charts.bars(c.body, {
      x: x, horizontal: true, labelWidth: 150, totalLabel: "definitions",
      stacks: [0, 1, 2].map(function (k) {
        return { id: "req" + k, name: (reqs && reqs[k] && reqs[k].label) || labels[k], color: colors.slot(k === 0 ? 3 : k === 1 ? 4 : 7),
          values: fams.map(function (f) { return arr(f.requirements)[k]; }) };
      }),
      y: { format: "int", label: "factor definitions" }
    });
    var cols = function (key, label) {
      return [
        { key: key, label: label, mono: true },
        { key: "n", label: "definitions", format: "int", align: "right" },
        { key: "r0", label: "close and dollar volume only", align: "right", value: function (r) { return arr(r.requirements)[0]; }, format: "int" },
        { key: "r1", label: "other kline fields", align: "right", value: function (r) { return arr(r.requirements)[1]; }, format: "int" },
        { key: "r2", label: "perpetual-futures data", align: "right", value: function (r) { return arr(r.requirements)[2]; }, format: "int" }
      ];
    };
    ui.details(c.body, { summary: "Family descriptions and counts", render: function (body) {
      var fc = cols("family", "family");
      fc.splice(1, 0, { key: "idea", label: "idea", wrap: true });
      ui.table(body, { ariaLabel: "Families", rows: fams, compact: true, columns: fc });
    } });
    var cats = arr(d.categories);
    if (cats.length) ui.details(c.body, { summary: "Categories", meta: "they describe the input series, not the economics", render: function (body) {
      ui.note(body, "A category is the information set of a factor's input series. stat.std.ex_btc_ret.w60 is a volatility factor filed under momentum because its input is a return series: read the formula, not the category.");
      ui.table(body, { ariaLabel: "Categories", rows: cats, compact: true, columns: cols("category", "category") });
    } });
  }

  /**
   * The screen numbers of each library id from the two screen tables (rIC q <= 0.10):
   * {id: {okx: {t, q}, binance: {t, q}}}.
   */
  function library_screenMap(d) {
    var m = {};
    [["okx", obj((obj(d.engine) || {}).top) || {}], ["binance", obj((obj(d.binance) || {}).top) || {}]].forEach(function (x) {
      var t = x[1], ids = arr(t.id), rt = arr(t.ric_t), rq = arr(t.ric_q);
      for (var i = 0; i < ids.length; i++) {
        var e = m[ids[i]] || (m[ids[i]] = {});
        e[x[0]] = { t: rt[i], q: rq[i] };
      }
    });
    return m;
  }
  // The search index is the part file data/library.index.js. It loads when the card comes near
  // the screen or the search box gets focus; the count line says so meanwhile. The formulas are
  // the parts data/library.expr-<k>.js: the parts of the shown rows load first, and a query loads
  // all of them, so the search also finds a formula.
  var SOURCE_NAMES = { okx: "OKX", binance: "Binance" };
  function renderSearch(root, d, ctx) {
    var info = obj(d.index_info) || {};
    var cmds = obj(d.commands) || {};
    var c = ui.card({ id: "search", title: "Search the library",
      subtitle: "Every definition by id, name, family, category or formula, with its two commands. " + (cmds.where || "") }, root);
    var nAll = isNum(info.n) ? info.n : null;
    var chunk = isNum(info.expr_chunk) && info.expr_chunk > 0 ? info.expr_chunk : 8000, parts = arr(info.expr_parts);
    var fid = ui.uid("library-q"), sid = ui.uid("library-scope");
    var input = el("input", { id: fid, "class": "fl-input", attrs: { type: "search", placeholder: "e.g. funding, ts_std, CF00001, beta", autocomplete: "off", spellcheck: "false", "aria-describedby": fid + "-count" } });
    var scope = el("select", { id: sid, "class": "fl-select" });
    // an id from a factor page link ("Find <id> in the library search"): used once, then cleared
    var hint = typeof FL.searchHint === "string" ? FL.searchHint : "";
    FL.searchHint = null;
    if (hint) input.value = hint;
    var SCOPES = [["all", "All definitions" + (nAll !== null ? " (" + fmt.int(nAll) + ")" : "")], ["okx", "Test data: OKX"], ["binance", "Test data: Binance"]];
    SCOPES.forEach(function (x) { scope.appendChild(el("option", { attrs: { value: x[0] }, text: x[1] })); });
    var count = el("span", { "class": "library-count", id: fid + "-count", attrs: { "aria-live": "polite" } });
    c.body.appendChild(el("div", { "class": "library-filter", attrs: { role: "search" } }, [
      el("span", { "class": "library-pair library-pair-q" }, [el("label", { attrs: { "for": fid }, text: "Search" }), input]),
      el("span", { "class": "library-pair" }, [el("label", { attrs: { "for": sid }, text: "Show" }), scope]),
      count
    ]));
    var host = el("div");
    c.body.appendChild(host);
    var started = false;

    // load the expression parts of these definition indexes (or all parts); then run again
    var asked = {};
    function needParts(idxs, all, again) {
      if (!ctx || !ctx.loadPart) return;
      var want = {};
      if (all) parts.forEach(function (n) { want[n] = true; });
      else idxs.forEach(function (i) { var n = parts[Math.floor(i / chunk)]; if (n) want[n] = true; });
      var todo = Object.keys(want).filter(function (n) { return !EXPR[n] && !asked[n]; });
      if (!todo.length) return;
      todo.forEach(function (n) { asked[n] = true; });
      Promise.all(todo.map(function (n) {
        return ctx.loadPart(n).then(function (x) { if (obj(x)) EXPR[n] = x; }, function () { /* the row shows the show command */ });
      })).then(again);
    }

    function build(idx) {
      var ix = library_decode(obj(idx));
      if (!ix) { count.textContent = ""; library_missing({ body: host }, "The factor index"); return; }
      var fams = arr(idx.families), cats = arr(idx.categories), srcNames = arr(idx.sources);
      var reqs = arr(idx.requirements);
      var SCR = library_screenMap(d);
      function row(i) {
        return {
          i: i, id: ix.id(i), name: ix.names[i], family: fams[ix.fam[i]] || null, category: cats[ix.cat[i]] || null,
          req: ix.req[i], src: srcNames[ix.source[i]] || null, expr: library_expr(i, chunk, parts), scr: SCR[ix.id(i)] || null
        };
      }
      function inScope(i, sc) {
        if (sc === "all") return true;
        return srcNames[ix.source[i]] === sc;
      }
      var COLS = [
        { key: "id", label: "id", mono: true, rowHeader: true },
        { key: "name", label: "name and formula", className: "library-formula-col", render: function (v, r) {
          var kids = [el("span", { "class": "mono", text: v })];
          if (r.expr) kids.push(el("span", { "class": "library-expr", text: r.expr }));
          else kids.push(el("span", { "class": "library-sub", text: "formula: not loaded (or: cd library && ../engine/.venv/bin/python -m factorlab show " + r.id + ")" }));
          return el("span", null, kids);
        }, sortValue: function (r) { return r.name; } },
        { key: "family", label: "family", mono: true },
        { key: "category", label: "category" },
        { key: "req", label: "data needed", render: function (v) { var q = reqs[v]; return q ? q.label : DASH; }, sortValue: function (r) { return r.req; } },
        { key: "src", label: "data of a test", render: function (v) { return SOURCE_NAMES[v] || DASH; },
          title: "the data on which judge.py tests the formula: OKX when Factorlab's OKX data has every field, else Binance" },
        { key: "scr", label: "screen on Factorlab's data", title: "residual-IC t and q on the generator zone, for the definitions with rIC q ≤ 0.10 of the OKX and Binance screens",
          sortValue: function (r) { var e = r.scr; if (!e) return null; var t = e.okx ? e.okx.t : e.binance ? e.binance.t : null; return isNum(t) ? Math.abs(t) : null; },
          render: function (v, r) {
            var e = r.scr;
            if (!e) return el("span", { "class": "muted", text: "rIC q above 0.10 or not screened" });
            var kids = [];
            if (e.okx) kids.push(el("span", { text: "OKX rIC t " + fT(e.okx.t) + " (q " + fQ(e.okx.q) + ")" }));
            if (e.binance) kids.push(el("span", { "class": e.okx ? "library-sub" : null, text: "Binance rIC t " + fT(e.binance.t) + " (q " + fQ(e.binance.q) + ")" }));
            return el("span", null, kids);
          } },
        { key: "cmd", label: "commands", sortable: false, render: function (v, r) { return library_cmds(cmds, r.id); } }
      ];
      // a header click sorts EVERY match, then the first MAX_RESULTS go to the table
      var order = null;   // {key, dir} or null = id order
      var tbl = ui.table(host, {
        rows: [], pageSize: 25, phonePageSize: 10, caption: "Factor search results", emptyText: "No definition matches.",
        columns: COLS, stack: true,
        onSort: function (key, dir) { order = { key: key, dir: dir }; library_run(); }
      });
      function sortAll(rows) {
        if (!order) return rows;
        var col = COLS.filter(function (x) { return x.key === order.key; })[0];
        if (!col) return rows;
        var sv = col.sortValue || function (r) { return r[col.key]; };
        var keyed = rows.map(function (r, j) { return { r: r, j: j, v: sv(r) }; });
        keyed.sort(function (a, b) {
          var an = a.v === null || a.v === undefined || (typeof a.v === "number" && !isFinite(a.v));
          var bn = b.v === null || b.v === undefined || (typeof b.v === "number" && !isFinite(b.v));
          if (an !== bn) return an ? 1 : -1;            // empty values last in both directions
          var x = an ? 0 : typeof a.v === "number" && typeof b.v === "number" ? a.v - b.v : String(a.v).localeCompare(String(b.v), undefined, { numeric: true });
          if (order.dir === "desc") x = -x;
          return x || a.j - b.j;
        });
        return keyed.map(function (k) { return k.r; });
      }
      var library_run = function () {
        var terms = String(input.value || "").toLowerCase().split(/\s+/).filter(Boolean);
        var hay = terms.length ? library_haystack(idx, ix, chunk, parts) : null;   // built on the first query only
        var sc = scope.value || "all";
        var hits = [];
        for (var j = 0; j < ix.n; j++) {
          if (!inScope(j, sc)) continue;
          if (hay) {
            var h = hay[j], ok = true;
            for (var q = 0; q < terms.length; q++) { if (h.indexOf(terms[q]) < 0) { ok = false; break; } }
            if (!ok) continue;
          }
          hits.push(j);
        }
        var total = hits.length;
        var rows = order ? sortAll(hits.map(row)) : hits.slice(0, MAX_RESULTS).map(row);
        rows = rows.slice(0, MAX_RESULTS);
        tbl.setRows(rows);
        var col = order ? COLS.filter(function (x) { return x.key === order.key; })[0] : null;
        var by = col ? (typeof col.label === "string" ? col.label : col.key) + (order.dir === "desc" ? ", largest first" : ", smallest first") : "id";
        count.textContent = fmt.int(total) + (total === 1 ? " match" : " matches") + (total > rows.length ? ", showing the first " + fmt.int(rows.length) + " by " + by : "");
        // the formulas: the parts of the shown rows, and every part for a query (a formula search)
        needParts(rows.map(function (r) { return r.i; }), terms.length > 0, library_run);
      };
      var timer = null;
      input.addEventListener("input", function () { if (timer) clearTimeout(timer); timer = setTimeout(library_run, 120); });
      input.addEventListener("keydown", function (e) { if (e.key === "Escape") { input.value = ""; library_run(); } });
      scope.addEventListener("change", library_run);
      library_run();
    }

    function start() {
      if (started) return;
      started = true;
      var ready = FL.data["library.index"] || d.index;
      if (ready) { build(ready); return; }
      if (!ctx || !ctx.loadPart) { count.textContent = ""; library_missing({ body: host }, "The factor index (data/library.index.js)"); return; }
      count.textContent = "Loading the index.";
      ctx.loadPart(info.part || "index").then(build, function (err) {
        count.textContent = "";
        var o = library_fix();
        o.title = "Search not available";
        ui.empty(host, "data/library.index.js " + (err && err.state === "broken" ? "does not parse" : "is not in this export") + ".", o);
      });
    }
    // start when the box gets focus or the card comes near the screen (IntersectionObserver)
    input.addEventListener("focus", start);
    scope.addEventListener("focus", start);
    if (hint || FL.data["library.index"] || d.index || typeof IntersectionObserver === "undefined") start();
    else {
      var io = new IntersectionObserver(function (entries) {
        if (entries.some(function (x) { return x.isIntersecting; })) { io.disconnect(); start(); }
      }, { rootMargin: "600px 0px" });
      io.observe(c.root);
      if (ctx && ctx.signal && ctx.signal.addEventListener) ctx.signal.addEventListener("abort", function () { io.disconnect(); });
    }
  }

  // the library CLI
  function renderUse(root) {
    var c = ui.card({ id: "use", title: "Use the library", subtitle: "The library CLI reads the catalog read-only. show prints the definition and its two commands." }, root);
    c.body.appendChild(ui.code([
      "cd library",
      "../engine/.venv/bin/python -m factorlab summary",
      "../engine/.venv/bin/python -m factorlab list --family momentum --limit 20",
      "../engine/.venv/bin/python -m factorlab show CF00001",
      "../engine/.venv/bin/python -m pytest tests -q"
    ].join("\n"), { block: true }));
  }

  /* ------------------------------------------------------------------ on Factorlab's data (engine/libfactors.py) */
  var fMean = orDash(function (v) { return fmt.num(v, 3, { signed: true }); });
  // q-values arrive rounded to 4 dp; a q below 0.0001 prints as a bound, never as 0
  var fQ = orDash(function (v) { return v < 0.0001 ? "< 0.0001" : fmt.num(v, 4); });
  var fP2 = orDash(function (v) { return fmt.num(v, 2); });

  function renderEngineFunnel(body, e) {
    var m = obj(e.meta) || {};
    var c = ui.card({ id: "engine-funnel", title: "From definitions to screened factors",
      subtitle: "Recomputed from status.parquet and screen_g.parquet; data through " + (m.data_through || DASH) + " UTC close, computed " + String(m.computed_at || DASH).slice(0, 10) + "." }, body);
    if (m.catalog_current === false) ui.callout("warning", "The catalog changed after this computation", "Run libfactors.py --compute and --screen again before reading these numbers.", { container: c.body });
    if (m.code_current === false) ui.callout("info", "The code changed after the computation", "The manifest's code hash differs from today's " + arr(m.code_changed).join(", ") + ". Run libfactors.py --compute and --screen to compute these numbers with the code on disk.", { container: c.body });
    var steps = arr(e.funnel);
    if (!steps.length) { library_missing(c, "The funnel"); return; }
    library_funnel(c.body, steps, "Bars are drawn to scale against the whole library. The last two rows count the definitions with a residual-IC q at or below the level on the generator zone.");
  }

  function renderEngineCoverage(body, e) {
    var m = obj(e.meta) || {};
    var c = ui.card({ id: "engine-coverage", title: "What cannot be computed, and why",
      subtitle: "A definition is computable when every raw field its formula reads is one Factorlab stores. Two more groups are computed but kept out of the screen." }, body);
    var rows = arr(e.coverage);
    if (!rows.length) { library_missing(c, "The coverage table"); return; }
    ui.table(c.body, {
      rows: rows, sortable: false, stack: true, caption: "Definitions Factorlab cannot compute or cannot screen",
      columns: [
        { key: "label", label: "group", render: function (v) { return el("span", { "class": "strong", text: v }); } },
        { key: "n", label: "definitions", format: "int", align: "right" },
        { key: "computed", label: "status", render: function (v) {
          return v ? ui.badge("info", "computed, not screened") : ui.badge("neutral", "not computed");
        } },
        { key: "note", label: "why", wrap: true, render: function (v) { return el("span", { text: v ? v.charAt(0).toUpperCase() + v.slice(1) : DASH }); } }
      ]
    });
    var fields = arr(e.fields);
    if (fields.length) {
      c.body.appendChild(el("h4", { text: "Fields the engine's OKX data does not have" }));
      ui.table(c.body, {
        rows: fields, compact: true, caption: "Definitions that need each missing field",
        columns: [
          { key: "field", label: "field", mono: true },
          { key: "label", label: "what it is" },
          { key: "n", label: "definitions needing it", format: "int", align: "right" },
          { key: "only", label: "needing only this field", format: "int", align: "right" }
        ]
      });
    }
    var der = obj(m.derivations) || {};
    ui.details(c.body, { summary: "How the panels are built", meta: fmt.int(m.symbols) + " perpetuals, " + (m.first_bar || DASH) + " to " + (m.data_through || DASH), render: function (b) {
      var items = [["Fields supplied", arr(m.fields).join(", ")], ["Benchmark column", (m.benchmark || DASH) + " (the library's btc_ret and ex_btc_ret refer to the engine's BTC perpetual)"]];
      Object.keys(der).forEach(function (k) { items.push({ term: k, value: der[k] }); });
      ui.dl(b, items);
    } });
  }

  function library_zoneClass(z) {
    return ["library-zone", z.key === "g_fit" || z.key === "g_eval" ? "is-g" : null, z.key === "g_eval" ? "is-g-eval" : null, z.sealed ? "is-sealed" : null];
  }
  function library_zoneStrip(container, rows) {
    var strip = el("div", { "class": "library-zones", attrs: { "aria-hidden": "true" } });
    var labels = el("div", { "class": "library-zone-labels", attrs: { "aria-hidden": "true" } });
    rows.forEach(function (z) {
      var grow = String(isNum(z.sundays) ? z.sundays : 1);
      var seg = el("div", { "class": library_zoneClass(z), title: z.label + ": " + z.start + " to " + z.end });
      seg.style.flexGrow = grow;
      strip.appendChild(seg);
      // short labels under the wide segments only (the embargo slots are one or two Sundays)
      var wide = isNum(z.sundays) && z.sundays >= 10;
      var lab = el("span", { text: wide ? (z.key === "i" ? "I sealed " : z.label + " ") + fmt.int(z.sundays) : "" });
      lab.style.flexGrow = grow;
      labels.appendChild(lab);
    });
    container.appendChild(strip);
    container.appendChild(labels);
  }

  function renderEngineScreen(body, e) {
    var s = obj(e.screen), z = obj(e.zones) || {};
    var zr = arr(z.rows);
    var c = ui.card({ id: "engine-screen", title: "The screen on the generator zone G",
      subtitle: s ? "Weekly Spearman IC and residual IC (rIC) of the 7-day forward return on " + fmt.int(s.g_sundays) + " Sundays, with Benjamini-Hochberg q-values over all " + fmt.int(s.n_tests_ric) + " screened factors, separately for IC and rIC." : null }, body);
    if (!s) { ui.empty(c.body, e.screen_missing || "The G-zone screen is not in the exported data.", { title: "Screen not run", command: arr(e.commands).join("\n") }); return; }
    var bh = arr(s.bh), rep = obj(s.replication) || {};
    function pick(stat, q) { return bh.filter(function (b) { return b.stat === stat && b.q === q; })[0] || {}; }
    function pickRep(stat, q) { return arr(rep.rows).filter(function (b) { return b.stat === stat && b.q === q; })[0] || {}; }
    var r05 = pick("ric", 0.05), f05 = pickRep("ric", 0.05);
    var box = el("div", { id: "engine-kpis" });   // default .kpis grid: four tiles share one row when they fit
    c.body.appendChild(box);
    var gFit = zr[0] || {}, gEval = zr[2] || {};
    ui.kpis(box, [
      { label: "Factors screened", value: s.n_tests_ric, format: "int", note: "one IC test and one rIC test each; at least " + fmt.int(s.min_weeks) + " weekly ICs" },
      { label: "Generator-zone Sundays", value: s.g_sundays, format: "int", note: fmt.int(gFit.sundays) + " in G_fit + " + fmt.int(gEval.sundays) + " in G_eval; " + fmt.int(s.names_min) + " to " + fmt.int(s.names_max) + " coins each" },
      { label: "rIC q ≤ 0.05 over all of G", value: r05.factors, format: "int", note: fmt.int(r05.signals) + " distinct signals; " + fmt.int(r05.positive) + " positive, " + fmt.int(r05.factors - r05.positive) + " negative; G_eval is part of this selection" },
      { label: "Selected on G_fit, same sign on G_eval", value: fmt.int(f05.same_sign) + " of " + fmt.int(f05.selected),
        note: "rIC q ≤ 0.05 on G_fit alone; " + fmt.int(f05.same_sign_t2) + " reach |rIC t| ≥ 2 on G_eval's " + fmt.int(gEval.sundays) + " Sundays. With no information " +
          fmt.pct(f05.base_pairs ? f05.base_same_sign / f05.base_pairs : null, 0, { signed: false }) + " of all screened factors keep their sign" }
    ]);

    c.body.appendChild(el("h4", { text: "Zones" }));
    library_zoneStrip(c.body, zr);
    ui.table(c.body, {
      rows: zr, compact: true, sortable: false, caption: "Time zones of the screen",
      columns: [
        { key: "label", label: "zone", render: function (v, r) {
          return el("span", { "class": "nowrap" }, [el("span", { "class": ["library-swatch"].concat(library_zoneClass(r)), attrs: { "aria-hidden": "true" } }), el("span", { "class": "mono", text: v })]);
        } },
        { key: "start", label: "Sundays", render: function (v, r) { return v === r.end ? v : v + " to " + r.end; } },
        { key: "sundays", label: "count", format: "int", align: "right" },
        { key: "role", label: "use", render: function (v, r) { return r.sealed ? el("span", { "class": "library-badges" }, [ui.badge("info", "sealed"), el("span", { text: v })]) : v; } }
      ]
    });
    ui.note(c.body, "Labels are read only through the " + (z.label_cutoff_row || DASH) + " Sunday row (its 7-day label completes at the " + (z.g_eval_label_close || DASH) + " close); reading a later label stops the screen. Factor values are computed on every Sunday, but no I-zone label or statistic is read or written.");

    c.body.appendChild(el("h4", { text: "Benjamini-Hochberg q-values" }));
    ui.table(c.body, {
      rows: bh, compact: true, caption: "Definitions with a q at or below each level",
      columns: [
        { key: "stat", label: "statistic", render: function (v) { return v === "ric" ? "residual IC (rIC)" : "raw IC"; } },
        { key: "q", label: "q at most", format: "num:2", align: "right" },
        { key: "factors", label: "factors", format: "int", align: "right" },
        { key: "signals", label: "distinct signals", format: "int", align: "right" },
        { key: "positive", label: "positive / negative", align: "right", render: function (v, r) { return fmt.int(v) + " / " + fmt.int(r.factors - v); }, sortValue: function (r) { return r.positive; } },
        { key: "share", label: "share of tests", align: "right", value: function (r) { return isNum(r.factors) && s.n_tests_ric ? r.factors / s.n_tests_ric : null; }, format: "pct:1:unsigned" }
      ]
    });
    var i05 = pick("ic", 0.05);
    if (isNum(i05.factors) && isNum(r05.factors)) ui.note(c.body, "At q ≤ 0.05 the raw IC has " + fmt.int(i05.factors) + " definitions and the residual IC " + fmt.int(r05.factors) + ". The difference comes from the regressors: the symbol tilt, the BTC beta and the size.");

    var reg = obj(s.regressors) || {};
    ui.details(c.body, { summary: "How the screen works", render: function (b) {
      ui.dl(b, [
        ["Label", s.label || DASH],
        ["Rows", s.mask || DASH],
        ["Factors tested", s.family_rule || DASH],
        ["IC", "each Sunday, the Spearman correlation of the factor with the label across coins; t = mean / sd × √weeks, two-sided Student-t p"],
        ["rIC", (reg.method || DASH) + ". Regressors: symbol tilt (" + (reg.tilt || DASH) + "); beta (" + (reg.beta || DASH) + "); size (" + (reg.size || DASH) + ")"],
        ["Settings", "from " + ((reg.config || {}).source || DASH) + ", checked equal at every run"],
        ["Multiple testing", "Benjamini-Hochberg over every screened factor, separately for IC and rIC. " + (s.pooled_note || "")],
        ["Replication", rep.rule || DASH],
        ["Distinct signals", (s.dedup || DASH) + "; the strongest member represents the signal"],
        ["Precision", s.precision || DASH],
        ["Persistence", "mean week-over-week rank correlation of the factor across coins (1 = the ranking never changes)"],
        ["Not the pre-registered statistic", arr(reg.deviations).join("; ") || DASH]
      ]);
    } });
    var reps = arr(rep.rows);
    if (reps.length) ui.details(c.body, { summary: "Selected on G_fit, checked on G_eval", meta: "the zone design's replication check", render: function (b) {
      ui.table(b, { rows: reps, compact: true, caption: "Selection on G_fit and its check on G_eval", columns: [
        { key: "stat", label: "statistic", render: function (v) { return v === "ric" ? "residual IC (rIC)" : "raw IC"; } },
        { key: "q", label: "G_fit q at most", format: "num:2", align: "right" },
        { key: "selected", label: "selected on G_fit", format: "int", align: "right" },
        { key: "same_sign", label: "same sign on G_eval", format: "int", align: "right" },
        { key: "share", label: "share", align: "right", value: function (r) { return r.selected ? r.same_sign / r.selected : null; }, format: "pct:0:unsigned" },
        { key: "same_sign_t2", label: "same sign, |t| ≥ 2", format: "int", align: "right" },
        { key: "base", label: "base rate", align: "right", value: function (r) { return r.base_pairs ? r.base_same_sign / r.base_pairs : null; }, format: "pct:0:unsigned",
          title: "share of all screened factors whose G_eval sign matches their G_fit sign" }
      ] });
      ui.note(b, "G_eval's Sundays are not part of this selection, so the G_eval columns are a check on data the selection did not see. They are still part of the generator zone.");
    } });
    var sens = arr(s.sensitivity);
    if (sens.length) ui.details(c.body, { summary: "How much the rIC settings move the counts", meta: "sensitivity", render: function (b) {
      ui.table(b, { rows: sens, compact: true, caption: "rIC counts under each setting", columns: [
        { key: "key", label: "variant", mono: true },
        { key: "note", label: "what changes" },
        { key: "screened", label: "screened", format: "int", align: "right" },
        { key: "q05", label: "rIC q ≤ 0.05", format: "int", align: "right" },
        { key: "q10", label: "rIC q ≤ 0.10", format: "int", align: "right" },
        { key: "fit_selected", label: "selected on G_fit", format: "int", align: "right" },
        { key: "same_sign", label: "same sign on G_eval", format: "int", align: "right" }
      ] });
      ui.note(b, "The first row is the screen's statistic. The counts depend on how the symbol tilt is removed, which is why the registered tilt is used.");
    } });
    var fams = arr(s.by_family);
    if (fams.length) ui.details(c.body, { summary: "Counts by family", meta: "rIC at q ≤ 0.05", render: function (b) {
      ui.table(b, { rows: fams, compact: true, caption: "Screen results by family", columns: [
        { key: "family", label: "family", mono: true },
        { key: "definitions", label: "definitions", format: "int", align: "right" },
        { key: "computable", label: "computable", format: "int", align: "right" },
        { key: "screened", label: "screened", format: "int", align: "right" },
        { key: "ic_q05", label: "IC q ≤ 0.05", format: "int", align: "right" },
        { key: "ric_q05", label: "rIC q ≤ 0.05", format: "int", align: "right" },
        { key: "ric_q05_signals", label: "rIC distinct signals", format: "int", align: "right", title: "signals whose representative is in this family" },
        { key: "ric_q05_touched", label: "signals touched", format: "int", align: "right", title: "distinct signals with at least one definition at rIC q ≤ 0.05 in this family" }
      ] });
      if (s.by_family_note) ui.note(b, s.by_family_note);
    } });
  }

  // ---- the screen tables (OKX and Binance): shared helpers
  // membership at q <= 0.05 comes from the signal id (assigned only at that level), never from the rounded q
  function library_at05(r) { return r.sig05 !== null && r.sig05 !== undefined; }
  function library_shown(t) { return isNum(t.shown) ? t.shown : (isNum(t.n) ? t.n : arr(t.id).length); }
  // the factor count at a level: the full count of the export when present (rows can be capped)
  function library_nAt(t, rows, q) {
    if (q === "05" && isNum(t.n_q05)) return t.n_q05;
    if (q === "10" && isNum(t.n)) return t.n;
    return rows.filter(function (r) { return q === "10" || library_at05(r); }).length;
  }
  // the size and the representative of each distinct signal, per BH level: from the export
  // (size05/repid05, counted over every definition at the level), else from the shown rows
  function library_signals(rows) {
    var reps = { "05": {}, "10": {} }, sizes = { "05": {}, "10": {} };
    rows.forEach(function (r) {
      ["05", "10"].forEach(function (q) {
        var g = r["sig" + q];
        if (g === null || g === undefined) return;
        if (isNum(r["size" + q])) sizes[q][g] = r["size" + q]; else sizes[q][g] = (sizes[q][g] || 0) + 1;
        if (r["repid" + q]) reps[q][g] = r["repid" + q]; else if (r["rep" + q]) reps[q][g] = r.id;
      });
    });
    return { reps: reps, sizes: sizes };
  }
  function library_sigCell(sv, q, r) {
    var g = r["sig" + q];
    if (g === null || g === undefined) return DASH;
    var size = sv.sizes[q][g] || 1;
    if (r["rep" + q]) return el("span", { text: size > 1 ? "represents " + fmt.int(size) + " factors" : "on its own" });
    return el("span", { "class": "muted", text: "variant of " + (sv.reps[q][g] || DASH) });
  }
  function library_runTop(tbl, count, t, rows, view) {
    var keep = rows.filter(function (r) {
      if (view.q === "05" && !library_at05(r)) return false;
      return !view.one || !!r["rep" + view.q];
    });
    tbl.setRows(keep);
    var nSig = rows.filter(function (r) { return !!r["rep" + view.q] && (view.q === "10" || library_at05(r)); }).length;
    var capped = isNum(t.shown) && isNum(t.n) && t.shown < t.n;
    count.textContent = (view.one ? fmt.int(keep.length) + (keep.length === 1 ? " distinct signal, one row" : " distinct signals, one row each") + " (the member with the largest |rIC t|)"
      : fmt.int(keep.length) + (keep.length === 1 ? " factor" : " factors") + " in " + fmt.int(nSig) + (nSig === 1 ? " distinct signal" : " distinct signals")) +
      " at q ≤ " + (view.q === "05" ? "0.05" : "0.10") +
      (capped ? "; the export holds the first " + fmt.int(t.shown) + " of " + fmt.int(t.n) + " definitions by |rIC t| (screen_g.parquet has all)" : "");
  }

  function renderEngineTop(body, e, cmds) {
    var t = obj(e.top), s = obj(e.screen) || {};
    var c = ui.card({ id: "engine-top", title: t ? "Definitions with rIC q ≤ " + fmt.num(t.q, 2) + ", by |rIC t|" : "Definitions by |rIC t|",
      subtitle: t ? "Each definition with a Benjamini-Hochberg rIC q ≤ " + fmt.num(t.q, 2) + " on G (" + fmt.int(t.n) + " of " + fmt.int(t.ranked_among) + "), ranked by |rIC t| among all screened definitions. IC and rIC are mean weekly Spearman correlations. " + ((cmds && cmds.where) || "") : null }, body);
    if (!t) { ui.empty(c.body, e.screen_missing || "The table is not in the exported data.", { title: "Screen not run", command: arr(e.commands).join("\n") }); return; }
    var fams = arr(e.families);
    var rows = [];
    for (var i = 0; i < library_shown(t); i++) {
      rows.push({
        rank: arr(t.rank)[i], id: arr(t.id)[i], name: arr(t.name)[i], family: fams[arr(t.fam)[i]] || null, expr: arr(t.expr)[i],
        ic_mean: arr(t.ic_mean)[i], ic_t: arr(t.ic_t)[i], ic_q: arr(t.ic_q)[i],
        ric_mean: arr(t.ric_mean)[i], ric_t: arr(t.ric_t)[i], ric_q: arr(t.ric_q)[i],
        persistence: arr(t.persistence)[i], fit_t: arr(t.fit_t)[i], fit_q: arr(t.fit_q)[i], eval_t: arr(t.eval_t)[i],
        sig05: arr(t.sig05)[i], rep05: arr(t.rep05)[i], sig10: arr(t.sig10)[i], rep10: arr(t.rep10)[i],
        size05: arr(t.size05)[i], size10: arr(t.size10)[i], repid05: arr(t.repid05)[i], repid10: arr(t.repid10)[i]
      });
    }
    var sv = library_signals(rows);
    // both screen tables open at q <= 0.05, one row per distinct signal
    var view = { q: "05", one: true };
    var controls = el("div", { "class": "library-controls" });
    c.body.appendChild(controls);
    function nAt(q) { return library_nAt(t, rows, q); }
    ui.segmented(controls, { ariaLabel: "Benjamini-Hochberg level", value: "05",
      options: [{ value: "05", label: "q ≤ 0.05 (" + fmt.int(nAt("05")) + ")" }, { value: "10", label: "q ≤ 0.10 (" + fmt.int(nAt("10")) + ")" }],
      onChange: function (v) { view.q = v; library_topRun(); } });
    ui.segmented(controls, { ariaLabel: "Rows", value: "one",
      options: [{ value: "all", label: "every factor" }, { value: "one", label: "one per distinct signal" }],
      onChange: function (v) { view.one = v === "one"; library_topRun(); } });
    var count = el("span", { "class": "library-count", attrs: { "aria-live": "polite" } });
    controls.appendChild(count);
    var tbl = ui.table(c.body, {
      rows: [], pageSize: 25, phonePageSize: 10, caption: "Definitions by residual-IC q on the generator zone", emptyText: "No definition at this level.",
      defaultSort: { key: "rank", dir: "asc" },
      columns: [
        { key: "rank", label: "#", format: "int", align: "right", title: "rank by |rIC t| among all screened factors" },
        { key: "name", label: "factor and formula", render: function (v, r) {
          return el("span", null, [el("span", { "class": "mono", text: r.id + "  " + v }), el("span", { "class": "library-expr", text: r.expr })]);
        }, sortValue: function (r) { return r.name; } },
        { key: "family", label: "family", mono: true },
        { key: "ic_mean", label: "IC", format: fMean, align: "right", title: "mean weekly Spearman IC on G" },
        { key: "ic_t", label: "IC t", format: fT, align: "right" },
        { key: "ic_q", label: "IC q", format: fQ, align: "right", title: "Benjamini-Hochberg q over all screened factors" },
        { key: "ric_mean", label: "rIC", format: fMean, align: "right", title: "mean weekly residual IC on G" },
        { key: "ric_t", label: "rIC t", format: fT, align: "right" },
        { key: "ric_q", label: "rIC q", format: fQ, align: "right", title: "Benjamini-Hochberg q over all screened factors" },
        { key: "persistence", label: "persistence", format: fP2, align: "right", title: "mean week-over-week rank correlation of the factor" },
        { key: "fit_t", label: "rIC t G_fit", format: fT, align: "right", title: "G_fit: " + fmt.int((arr(obj(e.zones) && e.zones.rows)[0] || {}).sundays) + " Sundays" },
        { key: "fit_q", label: "rIC q G_fit", format: fQ, align: "right", title: "Benjamini-Hochberg q over the G_fit p-values alone (the selection)" },
        { key: "eval_t", label: "rIC t G_eval", format: fT, align: "right", title: "G_eval: " + fmt.int((arr(obj(e.zones) && e.zones.rows)[2] || {}).sundays) + " Sundays (the check)" },
        { key: "sig", label: "distinct signal", sortable: false, render: function (v, r) { return library_sigCell(sv, view.q, r); } },
        { key: "cmd", label: "commands", sortable: false, render: function (v, r) { return library_cmds(cmds, r.id); } }
      ]
    });
    var library_topRun = function () { library_runTop(tbl, count, t, rows, view); };
    library_topRun();
    var rp = arr(obj(s.replication) && s.replication.rows).filter(function (b) { return b.stat === "ric" && b.q === 0.05; })[0] || {};
    ui.note(c.body, "A large rIC t on G is a number, not a test of a book. These q-values use all of G, so the G_eval column is part of the selection. " +
      "With a selection on G_fit alone, " + fmt.int(rp.selected) + " definitions have q ≤ 0.05; " + fmt.int(rp.same_sign) + " keep their sign on G_eval and " + fmt.int(rp.same_sign_t2) +
      " reach |rIC t| ≥ 2 there. To test a definition in a book, use its register command.");
  }

  /* ------------------------------------------------------------------ the same library on Binance's data (libfactors.py --source binance) */
  var fRho = orDash(function (v) { return fmt.num(v, 2); });
  var fShare = orDash(function (v) { return fmt.pct(v, 1, { signed: false }); });
  // magnitudes from 1e-18 to 0.3 in one column: four decimals down to 0.001, scientific notation below
  var fSmall = orDash(function (v) { return v === 0 ? "0" : Math.abs(v) >= 0.001 ? fmt.num(v, 4) : Number(v).toExponential(1).replace("-", fmt.MINUS); });
  var STAT_LABEL = { ric: "residual IC (rIC)", ic: "raw IC" };
  var CLASS_LABELS = { gap: "reads gap (open / previous close − 1)", price_only: "prices only", volume: "volume, no funding",
    funding: "funding, no volume", volume_and_funding: "volume and funding", other: "other" };
  function library_pick(rows, stat, q) { return arr(rows).filter(function (x) { return x.stat === stat && x.q === q; })[0] || {}; }

  function renderBinanceFunnel(body, b) {
    var m = obj(b.meta) || {};
    var c = ui.card({ id: "binance-funnel", title: "From definitions to screened factors, on Binance's data",
      subtitle: "Recomputed from the Binance status.parquet and screen_g.parquet; bars " + (m.first_bar || DASH) + " to " + (m.data_through || DASH) +
        " (UTC close), computed " + String(m.computed_at || DASH).slice(0, 10) + "." }, body);
    if (m.catalog_current === false) ui.callout("warning", "The catalog changed after this computation", "Run libfactors.py --source binance --compute and --screen again before reading these numbers.", { container: c.body });
    if (m.okx_current === false) ui.callout("warning", "The OKX screen changed after this computation", "The new family and the cross-check use the OKX screen of that computation: run libfactors.py --source binance --compute and --screen again.", { container: c.body });
    if (m.code_current === false) ui.callout("info", "The code changed after the computation", "The manifest's code hash differs from today's " + arr(m.code_changed).join(", ") + ". Run libfactors.py --source binance --compute and --screen to compute these numbers with the code on disk.", { container: c.body });
    var steps = arr(b.funnel);
    if (!steps.length) { library_missing(c, "The Binance funnel"); return; }
    library_funnel(c.body, steps, "Bars are drawn to scale against the whole library. The two branch rows are not funnel steps: the cross-checked definitions take no part in the new family's selection, and the replication row is a separate selection on G_fit alone.");
  }

  function renderBinanceFields(body, b) {
    var c = ui.card({ id: "binance-fields", title: "What the new fields unlock",
      subtitle: "Fields that Factorlab's OKX data lacks, or holds only after the generator zone. For each field: the number of new-family definitions that read it, that reach the screen and that have a low q. A definition can read several fields." }, body);
    var rows = arr(b.fields);
    if (!rows.length) { ui.empty(c.body, b.screen_missing || "The field counts are not in the exported data.", { title: "Screen not run", command: arr(b.commands).join("\n") }); return; }
    var cov = {};
    arr(b.coverage).forEach(function (x) { cov[x.field] = x; });
    var bars = obj(b.meta) && isNum(b.meta.bars) && isNum(b.meta.symbols) ? b.meta.bars * b.meta.symbols : null;
    ui.table(c.body, {
      rows: rows, caption: "New-family definitions and their q counts by the new field they read",
      defaultSort: { key: "ric_q05", dir: "desc" },
      columns: [
        { key: "field", label: "field", render: function (v, r) { return el("span", null, [el("span", { "class": "mono", text: v }), el("span", { "class": "library-sub", text: r.label })]); } },
        { key: "source", label: "on the Binance panel", wrap: true },
        { key: "from", label: "first value", value: function (r) { var x = cov[r.field === "vwap" ? "quote_volume" : r.field]; return x ? x.first_min : null; },
          render: function (v) { return el("span", { "class": "nowrap", text: v || DASH }); },
          title: "the earliest coin's first value; APT, OP and ARB start at their listings" },
        { key: "cells", label: "coin-days with a value", align: "right", value: function (r) { var x = cov[r.field === "vwap" ? "quote_volume" : r.field]; return x ? x.cells : null; },
          format: "int", title: bars ? "of " + fmt.int(bars) + " coin-days on the panel, before listings included" : null },
        { key: "candidates", label: "new-family definitions", format: "int", align: "right" },
        { key: "screened", label: "screened", format: "int", align: "right" },
        { key: "ric_q05", label: "rIC q ≤ 0.05", format: "int", align: "right" },
        { key: "ric_q10", label: "rIC q ≤ 0.10", format: "int", align: "right" },
        { key: "ic_q05", label: "IC q ≤ 0.05", format: "int", align: "right" }
      ]
    });
    var n = obj(b.counts) || {};
    ui.note(c.body, "taker_buy_sell_ratio comes from the metrics archive's last available 5-minute snapshot of each bar, so it is the ratio of those 5 minutes, not of the whole day the library defines: " +
      fmt.int(n.taker_last_5m) + " definitions read it; they are flagged and still screened. Open interest and the long/short ratios are the same last snapshot of the bar.");
    var org = arr(b.by_origin);
    if (org.length) {
      c.body.appendChild(el("h4", { text: "Why the OKX screen could not screen them" }));
      ui.table(c.body, {
        rows: org, compact: true, sortable: false, stack: true, caption: "The new family by origin",
        columns: [
          { key: "label", label: "origin", render: function (v, r) { return el("span", null, [el("span", { "class": "strong", text: v }), el("span", { "class": "library-sub", text: r.why })]); } },
          { key: "candidates", label: "definitions", format: "int", align: "right" },
          { key: "screened", label: "screened", format: "int", align: "right" },
          { key: "ric_q05", label: "rIC q ≤ 0.05", format: "int", align: "right" },
          { key: "ric_q10", label: "rIC q ≤ 0.10", format: "int", align: "right" },
          { key: "ic_q05", label: "IC q ≤ 0.05", format: "int", align: "right" }
        ]
      });
    }
  }

  function renderBinanceScreen(body, b) {
    var s = obj(b.screen), z = obj(b.zones) || {};
    var zr = arr(z.rows);
    var c = ui.card({ id: "binance-screen", title: "The new family on the generator zone G",
      subtitle: s ? "Weekly Spearman IC and residual IC (rIC) of the OKX 7-day forward return on " + fmt.int(s.g_sundays) + " Sundays, with Benjamini-Hochberg q-values over the new family's " + fmt.int(s.n_tests_ric) + " screened factors alone, separately for IC and rIC." : null }, body);
    if (!s) { ui.empty(c.body, b.screen_missing || "The Binance screen is not in the exported data.", { title: "Screen not run", command: arr(b.commands).join("\n") }); return; }
    var bh = arr(s.bh), rep = obj(s.replication) || {}, ov = arr(s.overlap);
    var r05 = library_pick(bh, "ric", 0.05), f05 = library_pick(rep.rows, "ric", 0.05), f10 = library_pick(rep.rows, "ric", 0.10);
    var o05 = library_pick(ov, "ric", 0.05), o10 = library_pick(ov, "ric", 0.10);
    var gEval = zr[2] || {};
    var box = el("div", { id: "binance-kpis" });
    c.body.appendChild(box);
    ui.kpis(box, [
      { label: "Screened in the new family", value: s.n_tests_ric, format: "int", note: "its own Benjamini-Hochberg family; at least " + fmt.int(s.min_weeks) + " weekly ICs; " + fmt.int(s.names_min) + " to " + fmt.int(s.names_max) + " coins a Sunday" },
      { label: "rIC q ≤ 0.05 over all of G", value: r05.factors, format: "int", note: fmt.int(r05.signals) + " distinct signals; " + fmt.int(r05.positive) + " positive, " + fmt.int(r05.factors - r05.positive) + " negative; G_eval is part of this selection" },
      { label: "Selected on G_fit alone, rIC q ≤ 0.05", value: f05.selected, format: "int",
        note: "smallest G_fit q " + fmt.num((obj(rep.smallest_fit_q) || {}).ric, 4) + ". At q ≤ 0.10: " + fmt.int(f10.same_sign) + " of " + fmt.int(f10.selected) + " keep their sign on G_eval's " + fmt.int(gEval.sundays) + " Sundays, " +
          fmt.int(f10.same_sign_t2) + " with |rIC t| ≥ 2; with no selection " + fmt.pct(f10.base_pairs ? f10.base_same_sign / f10.base_pairs : null, 0, { signed: false }) + " keep their sign",
        badge: f05.selected === 0 ? { status: "warning", label: "none at q ≤ 0.05" } : null },
      { label: "Close to an OKX definition at the same q", value: fmt.int(o05.overlapping) + " of " + fmt.int(o05.count),
        note: "|rank correlation| ≥ 0.7 with one of the OKX screen's " + fmt.int(o05.okx_count) + " definitions with rIC q ≤ 0.05; at q ≤ 0.10, " + fmt.int(o10.overlapping) + " of " + fmt.int(o10.count) }
    ]);
    c.body.appendChild(el("h4", { text: "Benjamini-Hochberg q-values (new family only)" }));
    ui.table(c.body, {
      rows: bh, compact: true, caption: "New-family definitions with a q at or below each level",
      columns: [
        { key: "stat", label: "statistic", render: function (v) { return STAT_LABEL[v] || v; } },
        { key: "q", label: "q at most", format: "num:2", align: "right" },
        { key: "factors", label: "factors", format: "int", align: "right" },
        { key: "signals", label: "distinct signals", format: "int", align: "right" },
        { key: "positive", label: "positive / negative", align: "right", render: function (v, r) { return fmt.int(v) + " / " + fmt.int(r.factors - v); }, sortValue: function (r) { return r.positive; } },
        { key: "share", label: "share of tests", align: "right", value: function (r) { return isNum(r.factors) && s.n_tests_ric ? r.factors / s.n_tests_ric : null; }, format: "pct:1:unsigned" }
      ]
    });
    ui.note(c.body, "These q-values come from the new family's tests alone: it adds no test to the OKX screen's family and takes none from it. The q-values over all of G include G_eval; the replication below selects on G_fit alone.");
    var reps = arr(rep.rows);
    if (reps.length) ui.details(c.body, { summary: "Selected on G_fit, checked on G_eval", meta: "the zone design's replication check", render: function (x) {
      ui.table(x, { rows: reps, compact: true, caption: "Selection on G_fit and its check on G_eval (new family)", columns: [
        { key: "stat", label: "statistic", render: function (v) { return STAT_LABEL[v] || v; } },
        { key: "q", label: "G_fit q at most", format: "num:2", align: "right" },
        { key: "selected", label: "selected on G_fit", format: "int", align: "right" },
        { key: "same_sign", label: "same sign on G_eval", format: "int", align: "right" },
        { key: "share", label: "share", align: "right", value: function (r) { return r.selected ? r.same_sign / r.selected : null; }, format: "pct:0:unsigned" },
        { key: "same_sign_t2", label: "same sign, |t| ≥ 2", format: "int", align: "right" },
        { key: "base", label: "base rate", align: "right", value: function (r) { return r.base_pairs ? r.base_same_sign / r.base_pairs : null; }, format: "pct:0:unsigned",
          title: "share of all screened factors whose G_eval sign matches their G_fit sign" }
      ] });
      ui.note(x, "The smallest G_fit q is " + fmt.num((obj(rep.smallest_fit_q) || {}).ric, 4) + " for rIC and " + fmt.num((obj(rep.smallest_fit_q) || {}).ic, 4) + " for raw IC. G_eval's Sundays are not part of this selection, so the G_eval columns check it on data it did not see; they are still part of the generator zone.");
    } });
    if (ov.length) ui.details(c.body, { summary: "Overlap with the OKX screen at the same q", meta: "|rank correlation| ≥ 0.7", render: function (x) {
      ui.table(x, { rows: ov, compact: true, caption: "New-family definitions close to an OKX definition of the same statistic and level", columns: [
        { key: "stat", label: "statistic", render: function (v) { return STAT_LABEL[v] || v; } },
        { key: "q", label: "q at most", format: "num:2", align: "right" },
        { key: "count", label: "new-family definitions", format: "int", align: "right" },
        { key: "okx_count", label: "OKX definitions", format: "int", align: "right" },
        { key: "overlapping", label: "close to one (Binance values)", format: "int", align: "right" },
        { key: "overlapping_okx_values", label: "close to one (the OKX definitions' OKX values)", format: "int", align: "right" }
      ] });
      ui.note(x, "Close means a largest |pooled within-Sunday rank correlation| of 0.7 or more (the distinct-signal threshold) with an OKX definition of the same statistic and level. The first count correlates Binance values on both sides; the second uses the OKX definitions' own OKX values.");
    } });
    var sens = arr(s.sensitivity);
    if (sens.length) ui.details(c.body, { summary: "How much the rIC settings move the counts", meta: "sensitivity", render: function (x) {
      ui.table(x, { rows: sens, compact: true, caption: "New-family rIC counts under each setting", columns: [
        { key: "key", label: "variant", mono: true },
        { key: "note", label: "what changes" },
        { key: "screened", label: "screened", format: "int", align: "right" },
        { key: "q05", label: "rIC q ≤ 0.05", format: "int", align: "right" },
        { key: "q10", label: "rIC q ≤ 0.10", format: "int", align: "right" },
        { key: "fit_selected", label: "selected on G_fit", format: "int", align: "right" },
        { key: "same_sign", label: "same sign on G_eval", format: "int", align: "right" }
      ] });
      ui.note(x, "The first row is the screen's statistic, the registered one the OKX screen uses.");
    } });
    var fams = arr(s.by_family);
    if (fams.length) ui.details(c.body, { summary: "Counts by family", meta: "families with a screened definition", render: function (x) {
      ui.table(x, { rows: fams, compact: true, caption: "New-family screen results by family", columns: [
        { key: "family", label: "family", mono: true },
        { key: "candidates", label: "new-family definitions", format: "int", align: "right" },
        { key: "screened", label: "screened", format: "int", align: "right" },
        { key: "ric_q05", label: "rIC q ≤ 0.05", format: "int", align: "right" },
        { key: "ric_q05_signals", label: "rIC distinct signals", format: "int", align: "right", title: "signals whose representative is in this family" },
        { key: "ric_q10", label: "rIC q ≤ 0.10", format: "int", align: "right" },
        { key: "ic_q05", label: "IC q ≤ 0.05", format: "int", align: "right" }
      ] });
    } });
    var reg = obj(s.regressors) || {};
    ui.details(c.body, { summary: "How the screen works", render: function (x) {
      ui.dl(x, [
        ["Factors tested", s.family_rule || DASH],
        ["Label", s.label || DASH],
        ["Rows", s.mask || DASH],
        ["rIC", (reg.method || DASH) + ". Regressors (from the OKX bars, as in the OKX screen): symbol tilt (" + (reg.tilt || DASH) + "); beta (" + (reg.beta || DASH) + "); size (" + (reg.size || DASH) + ")"],
        ["Multiple testing", "Benjamini-Hochberg over the new family only, separately for IC and rIC. " + (s.pooled_note || "")],
        ["Replication", rep.rule || DASH],
        ["Distinct signals", (s.dedup || DASH) + "; the strongest member represents the signal"],
        ["Precision", s.precision || DASH],
        ["Not the pre-registered statistic", arr(reg.deviations).join("; ") || DASH]
      ]);
    } });
  }

  function renderBinanceTop(body, b, cmds) {
    var t = obj(b.top);
    var c = ui.card({ id: "binance-top", title: t ? "New-family definitions with rIC q ≤ " + fmt.num(t.q, 2) + ", by |rIC t|" : "New-family definitions by |rIC t|",
      subtitle: t ? "Each new-family definition with a Benjamini-Hochberg rIC q ≤ " + fmt.num(t.q, 2) + " on G (" + fmt.int(t.n) + " of " + fmt.int(t.ranked_among) + "), ranked by |rIC t| among the new family's screened definitions. IC and rIC are mean weekly Spearman correlations with the OKX label. " + ((cmds && cmds.where) || "") : null }, body);
    if (!t) { ui.empty(c.body, b.screen_missing || "The table is not in the exported data.", { title: "Screen not run", command: arr(b.commands).join("\n") }); return; }
    var fams = arr(b.families), origins = arr(b.origin_labels);
    var rows = [];
    for (var i = 0; i < library_shown(t); i++) {
      rows.push({
        rank: arr(t.rank)[i], id: arr(t.id)[i], name: arr(t.name)[i], family: fams[arr(t.fam)[i]] || null, expr: arr(t.expr)[i],
        fields: arr(t.fields)[i], origin: origins[arr(t.origin)[i]] || null, taker5m: arr(t.taker5m)[i],
        ic_mean: arr(t.ic_mean)[i], ic_t: arr(t.ic_t)[i], ic_q: arr(t.ic_q)[i],
        ric_mean: arr(t.ric_mean)[i], ric_t: arr(t.ric_t)[i], ric_q: arr(t.ric_q)[i],
        persistence: arr(t.persistence)[i], fit_t: arr(t.fit_t)[i], fit_q: arr(t.fit_q)[i], eval_t: arr(t.eval_t)[i],
        sig05: arr(t.sig05)[i], rep05: arr(t.rep05)[i], sig10: arr(t.sig10)[i], rep10: arr(t.rep10)[i],
        size05: arr(t.size05)[i], size10: arr(t.size10)[i], repid05: arr(t.repid05)[i], repid10: arr(t.repid10)[i],
        rho05: arr(t.okx_rho05)[i], rho10: arr(t.okx_rho10)[i]
      });
    }
    var sv = library_signals(rows);
    // the same opening view as the OKX table: q <= 0.05, one row per distinct signal
    var view = { q: "05", one: true };
    function nAt(q) { return library_nAt(t, rows, q); }
    var controls = el("div", { "class": "library-controls" });
    c.body.appendChild(controls);
    ui.segmented(controls, { ariaLabel: "Benjamini-Hochberg level", value: "05",
      options: [{ value: "05", label: "q ≤ 0.05 (" + fmt.int(nAt("05")) + ")" }, { value: "10", label: "q ≤ 0.10 (" + fmt.int(nAt("10")) + ")" }],
      onChange: function (v) { view.q = v; library_binRun(); } });
    ui.segmented(controls, { ariaLabel: "Rows", value: "one",
      options: [{ value: "all", label: "every factor" }, { value: "one", label: "one per distinct signal" }],
      onChange: function (v) { view.one = v === "one"; library_binRun(); } });
    var count = el("span", { "class": "library-count", attrs: { "aria-live": "polite" } });
    controls.appendChild(count);
    var tbl = ui.table(c.body, {
      rows: [], pageSize: 25, phonePageSize: 10, caption: "New-family definitions by residual-IC q on the generator zone", emptyText: "No definition at this level.",
      defaultSort: { key: "rank", dir: "asc" },
      columns: [
        { key: "rank", label: "#", format: "int", align: "right", title: "rank by |rIC t| among the new family's screened factors" },
        { key: "name", label: "factor and formula", render: function (v, r) {
          return el("span", null, [el("span", { "class": "mono", text: r.id + "  " + v }), el("span", { "class": "library-expr", text: r.expr })]);
        }, sortValue: function (r) { return r.name; } },
        { key: "family", label: "family", mono: true },
        { key: "fields", label: "new field", render: function (v, r) {
          return el("span", null, [el("span", { "class": "mono", text: String(v || "").split(",").join(", ") || DASH }),
            r.taker5m ? el("span", { "class": "library-sub", text: "taker ratio: last 5 minutes" }) : null]);
        } },
        { key: "ic_mean", label: "IC", format: fMean, align: "right", title: "mean weekly Spearman IC on G" },
        { key: "ic_t", label: "IC t", format: fT, align: "right" },
        { key: "ic_q", label: "IC q", format: fQ, align: "right", title: "Benjamini-Hochberg q over the new family" },
        { key: "ric_mean", label: "rIC", format: fMean, align: "right", title: "mean weekly residual IC on G" },
        { key: "ric_t", label: "rIC t", format: fT, align: "right" },
        { key: "ric_q", label: "rIC q", format: fQ, align: "right", title: "Benjamini-Hochberg q over the new family" },
        { key: "persistence", label: "persistence", format: fP2, align: "right", title: "mean week-over-week rank correlation of the factor" },
        { key: "fit_t", label: "rIC t G_fit", format: fT, align: "right" },
        { key: "fit_q", label: "rIC q G_fit", format: fQ, align: "right", title: "Benjamini-Hochberg q over the G_fit p-values alone (the selection)" },
        { key: "eval_t", label: "rIC t G_eval", format: fT, align: "right", title: "the check, on Sundays the G_fit selection never saw" },
        { key: "sig", label: "distinct signal", sortable: false, render: function (v, r) { return library_sigCell(sv, view.q, r); } },
        { key: "rho", label: "closest OKX definition", align: "right", title: "largest |pooled rank correlation| (Binance values) with an OKX definition at the same rIC q level; 0.7 or more counts as close",
          value: function (r) { return r["rho" + view.q]; },
          render: function (v) { return isNum(v) ? el("span", { text: fRho(v) + (v >= 0.7 ? " close" : "") }) : DASH; } },
        { key: "cmd", label: "commands", sortable: false, render: function (v, r) { return library_cmds(cmds, r.id); } }
      ]
    });
    var library_binRun = function () { library_runTop(tbl, count, t, rows, view); };
    library_binRun();
    ui.note(c.body, "A large rIC t on G is a number, not a test of a book. These q-values use all of G, so the G_eval column is part of the selection. " +
      (isNum(t.no_fit) && t.no_fit > 0 ? fmt.int(t.no_fit) + " of them have no G_fit week (their values start inside G_eval; " + fmt.int(t.no_fit_screened) + " of all screened definitions are like that), so no G_fit selection can check them. " : "") +
      "To test a definition in a book, use its register command.");
  }

  function renderBinanceCross(body, b) {
    var x = obj(b.crosscheck);
    var n = obj(b.counts) || {};
    var c = ui.card({ id: "binance-crosscheck", title: "Cross-venue agreement of the OKX-screened definitions",
      subtitle: "The " + fmt.int(n.crosscheck) + " definitions the OKX screen screened, computed on Binance and compared with their OKX statistics. Agreement numbers only: no q-value on Binance; the OKX screen gives their q-values." }, body);
    if (!x) { ui.empty(c.body, b.crosscheck_missing || b.screen_missing || "The cross-check is not in the exported data.", { title: "Cross-check not run", command: arr(b.commands).join("\n") }); return; }
    var s = obj(x.summary) || {}, surv = arr(x.okx_levels);
    var s05 = surv.filter(function (r) { return r.q === 0.05; })[0] || {}, s10 = surv.filter(function (r) { return r.q === 0.10; })[0] || {};
    var box = el("div", { id: "binance-cross-kpis" });
    c.body.appendChild(box);
    ui.kpis(box, [
      { label: "Rank correlation of rIC t across venues", value: fmt.num(s.ric_t_spearman, 3), note: "over the " + fmt.int(s.compared_ric_t) + " definitions with an rIC t on both venues; " + fmt.num(s.ric_t_spearman_without_gap, 3) + " without the ones on gap, " + fmt.num(s.ric_t_spearman_without_funding, 3) + " without the " + fmt.int(s.funding_readers) + " that read funding; raw IC t " + fmt.num(s.ic_t_spearman, 3),
        badge: { status: "info", label: "agreement only" } },
      { label: "Same rIC sign on both venues", value: fShare(s.same_ric_sign), note: "raw IC: " + fShare(s.same_ic_sign) },
      { label: "OKX rIC q ≤ 0.05, same sign on Binance", value: fmt.int(s05.same_sign) + " of " + fmt.int(s05.okx_count),
        note: "rIC q ≤ 0.05 on OKX; " + fmt.int(s05.same_sign_t2) + " with |rIC t| ≥ 2 on Binance. At q ≤ 0.10: " + fmt.int(s10.same_sign) + " of " + fmt.int(s10.okx_count) + ", " + fmt.int(s10.same_sign_t2) + " with |t| ≥ 2. Expected, not a replication (below)",
        badge: { status: "info", label: "not a replication" } },
      { label: "Value correlation across venues", value: fmt.num(s.rho_median, 2), note: "median pooled rank correlation of a factor's G values; 10th percentile " + fmt.num(s.rho_p10, 2) + "; " + fShare(s.rho_at_least_0_9) + " at 0.9 or more, " + fShare(s.rho_below_0_5) + " below 0.5" }
    ]);
    if (x.agreement_note) ui.note(c.body, x.agreement_note);
    c.body.appendChild(el("h4", { text: "By input class" }));
    ui.table(c.body, {
      rows: arr(x.by_class), compact: true, stack: true, caption: "Cross-venue agreement by the inputs a definition reads",
      columns: [
        { key: "cls", label: "inputs", render: function (v) { return CLASS_LABELS[v] || words(v); } },
        { key: "factors", label: "definitions", format: "int", align: "right" },
        { key: "ric_t_spearman", label: "rank correlation of rIC t", format: "num:3", align: "right" },
        { key: "same_ric_sign", label: "same rIC sign", format: fShare, align: "right" },
        { key: "median_rho", label: "median value correlation", format: "num:3", align: "right" }
      ]
    });
    ui.note(c.body, "gap is open / previous close − 1. Each venue has its own volume: the OKX quote volume here is close × base volume, and the Binance quote volume is the USDT turnover. " + (x.funding_note || ""));
    var series = arr(x.series);
    var cut = obj(b.zones) ? b.zones.label_cutoff_row : null;
    if (series.length) ui.details(c.body, { summary: "The input series on both venues", meta: "bars up to " + (cut || DASH) + ", every coin-day both venues have", render: function (bx) {
      ui.table(bx, { rows: series, compact: true, caption: "Cross-venue agreement of library series", columns: [
        { key: "series", label: "series", mono: true },
        { key: "cells", label: "coin-days", format: "int", align: "right" },
        { key: "median_abs_okx", label: "median |value| OKX", format: fSmall, align: "right" },
        { key: "median_abs_binance", label: "median |value| Binance", format: fSmall, align: "right" },
        { key: "corr", label: "daily correlation", format: "num:3", align: "right" },
        { key: "spearman", label: "daily rank correlation", format: "num:3", align: "right" },
        { key: "identical", label: "same value", format: fShare, align: "right", title: "share of coin-days on which both venues hold the same number (|difference| ≤ 1e-12)" }
      ] });
      ui.note(bx, "Before the first OKX funding print in the data, the engine's funding is Binance's own settlement archive. On those days funding is the same input on both venues.");
    } });
    var dis = arr(x.disagreements);
    if (dis.length) ui.details(c.body, { summary: "The largest disagreements", meta: "by |OKX rIC t − Binance rIC t|", render: function (bx) {
      ui.table(bx, { rows: dis, compact: true, caption: "Definitions whose rIC t differs most across venues", columns: [
        { key: "name", label: "definition", render: function (v, r) { return el("span", null, [el("span", { "class": "mono", text: r.id + "  " + v }), el("span", { "class": "library-sub", text: CLASS_LABELS[r.cls] || words(r.cls) })]); }, sortValue: function (r) { return r.id; } },
        { key: "okx_ric_t", label: "rIC t OKX", format: fT, align: "right" },
        { key: "binance_ric_t", label: "rIC t Binance", format: fT, align: "right" },
        { key: "rho", label: "value correlation", format: fRho, align: "right" },
        { key: "cause", label: "plausible cause (label-free diagnostics)", wrap: true }
      ] });
    } });
    ui.note(c.body, "Agreement across venues says the venue of the factor inputs does not drive these numbers (both screens use OKX's labels and regressors; funding excepted: before the first OKX print both read Binance's archive). It is not a second test and not an independent replication.");
  }

  function renderBinancePanel(body, b) {
    var m = obj(b.meta) || {}, k = obj(b.checks) || {}, pv = obj(m.provenance) || {};
    var c = ui.card({ id: "binance-panel", title: "The Binance panel: provenance, point in time, checks",
      subtitle: "Each field's convention and the label-free checks libfactors.py runs on the panel before computing anything (manifest panels.checks)." }, body);
    var f = obj(k.funding) || {}, tm = obj(k.metrics_timing) || {}, vw = obj(k.vwap) || {}, ok = obj(k.okx_close) || {};
    ui.dl(c.body, [
      ["Data", (m.venue || DASH) + "; " + fmt.int(m.symbols) + " coins, mapped to the engine's names; benchmark " + (m.benchmark || DASH)],
      ["Bars", fmt.int(m.bars) + " daily bars, " + (m.first_bar || DASH) + " to " + (m.data_through || DASH) + ", stamped with their UTC open time; the last one is closed"],
      ["Sources", (pv.use ? pv.use + ": " : "") + arr(pv.sources).join("; ")],
      { term: "Download", value: pv.download || DASH },
      ["Merge", pv.merge || DASH],
      ["New days", m.appended ? m.appended + ". merge_panels.py --append added them, and no old cell changed." :
        "none: the panel holds the first download only"],
      ["Funding check", "the daily sum of settlements equals the engine's stored Binance prints on " + fmt.int(f.day_cells) + " coin-days (" + fmt.int(f.differing) + " differ by more than 1e-12; largest difference " + fSmall(f.max_abs_diff) + ")"],
      ["Snapshot timing", "the implied mark price (open-interest value / open interest) sits a median " + fmt.pct(tm.median_d, 3, { signed: false }) + " from the bar's own close and " +
        fmt.pct(tm.median_d1, 2, { signed: false }) + " from the previous close (" + fmt.int(tm.cells) + " coin-days): the snapshot is the bar's last"],
      ["VWAP", (vw.collapses ? "equal to close" : "a real VWAP") + ": median |close / vwap − 1| " + fmt.pct(vw.median_abs, 2, { signed: false }) + ", inside the bar's low-high range on " + fmt.pct(vw.inside_low_high, 3, { signed: false }) + " of " + fmt.int(vw.cells) + " coin-days"],
      ["Against OKX", "median Binance / OKX close " + fmt.num(ok.ratio_min, 4) + " to " + fmt.num(ok.ratio_max, 4) + " per coin; daily-return correlation " + fmt.num(ok.corr_min, 3) + " to " + fmt.num(ok.corr_max, 3) + " (" + fmt.int(ok.symbols) + " coins, " + fmt.int(ok.days) + " coin-days)"],
      ["Cleaning freeze", m.cleaning_freeze ? "CLEANING_FREEZE pins the " + fmt.int(m.cleaning_freeze.rules) +
        " rules, version " + m.cleaning_freeze.version + " (sha256 " + String(m.cleaning_freeze.sha256).slice(0, 12) + "…). The compute and the weekly log of the formula tests (judge.py log-week) stop when the code has other rules." : "not recorded in this manifest"],
      ["Archive errors set missing", arr(k.cleaning).length ? "before any factor reads the panel, by label-free rules on one coin and bar each (nothing carried from another day): " +
        arr(k.cleaning).map(function (x) { return x.label + ": " + fmt.int(x.cells) + " coin-day" + (x.cells === 1 ? "" : "s") + (x.cells ? " (" + x.where + ")" : ""); }).join("; ") : DASH]
    ]);
    var conv = obj(m.conventions) || {};
    ui.details(c.body, { summary: "Field conventions", meta: "known by each bar's close", render: function (x) {
      ui.dl(x, Object.keys(conv).map(function (key) { return { term: words(key), value: conv[key] }; }));
    } });
    var cov = arr(b.coverage);
    if (cov.length) ui.details(c.body, { summary: "Coverage of every field", meta: fmt.int(cov.length) + " fields", render: function (x) {
      ui.table(x, { rows: cov, compact: true, caption: "Coin-days with a value and first values per field", columns: [
        { key: "field", label: "field", mono: true },
        { key: "cells", label: "coin-days with a value", format: "int", align: "right" },
        { key: "first_min", label: "first value, earliest coin" },
        { key: "first_max", label: "first value, latest coin", title: "APT, OP and ARB start at their listings" }
      ] });
    } });
    var pit = obj(b.pit) || {};
    ui.note(c.body, "Point-in-time proof as on OKX: every factor recomputed on panels cut at " + arr(pit.cuts).join(", ") + " and compared on every Sunday up to each cut. Sources: " + arr(b.files).join(", ") + ".");
  }

  function renderBinance(container, d) {
    var b = obj(d.binance);
    var meta = b && obj(b.meta) ? b.meta : {};
    var box = el("div", { id: "binance", "class": "library-binance" });
    container.appendChild(box);
    box.appendChild(el("div", { "class": "library-binance-head" }, [
      el("h3", { text: "Binance data for the same " + (isNum(meta.symbols) ? fmt.int(meta.symbols) : "20") + " coins" }),
      el("p", { text: "The library computed on Binance's public USD-M data for the engine's coins (Binance is a data source only) and screened on the same OKX labels: a new family for what the OKX data could not screen, a cross-check for the rest." })
    ]));
    var body = el("div", { "class": "stack" });
    box.appendChild(body);
    if (!b) { var o = library_fix(); o.title = "Not available"; ui.empty(body, "The Binance block is not in the exported data.", o); return; }
    if (b.available === false) {
      ui.empty(body, (b.reason || "No computation was found") + ". The library is not computed on Binance's data. Download the panel, merge it, compute, then export again:", { title: "Not computed on Binance's data", command: arr(b.commands).join("\n") });
      return;
    }
    // the data terms of the archive the panel reads
    if (b.terms) body.appendChild(el("p", { "class": "fl-note", id: "binance-terms", text: b.terms }));
    // method notes: a closed details in the neutral style; amber stays for real problems
    ui.details(body, { id: "binance-notes", summary: "Binance is a data source only", meta: "how to read this block",
      render: function (x) { arr(b.notes).forEach(function (h) { x.appendChild(el("p", { "class": "fl-note", text: h })); }); } });
    library_defer(body, "binance-funnel", function (h) { renderBinanceFunnel(h, b); });
    library_defer(body, "binance-fields", function (h) { renderBinanceFields(h, b); });
    library_defer(body, "binance-screen", function (h) { renderBinanceScreen(h, b); });
    library_defer(body, "binance-top", function (h) { renderBinanceTop(h, b, obj(d.commands)); });
    library_defer(body, "binance-crosscheck", function (h) { renderBinanceCross(h, b); });
    library_defer(body, "binance-panel", function (h) { renderBinancePanel(h, b); });
  }

  // one table of both screens' results, above the long OKX and Binance details
  function renderScreenSummary(body, d) {
    var e = obj(d.engine), b = obj(d.binance);
    var c = ui.card({ id: "screen-summary", title: "Screen results",
      subtitle: "Definitions with a Benjamini-Hochberg residual-IC (rIC) q at or below each level on the generator zone G, and the replication check on G_eval. The numbers rank definitions; they are not a test of a book." }, body);
    var rows = [];
    function add(key, label, x, what) {
      var s = obj(x && x.screen);
      if (!x || x.available === false || !s) {
        rows.push({ key: key, label: label, what: x && x.available === false ? "not computed in this data folder" : "no screen in the exported data",
          screened: null, q05: null, s05: null, q10: null, s10: null, fit: null, same: null });
        return;
      }
      var b05 = library_pick(s.bh, "ric", 0.05), b10 = library_pick(s.bh, "ric", 0.10);
      var r05 = library_pick(s.replication && s.replication.rows, "ric", 0.05);
      rows.push({ key: key, label: label, what: what, screened: s.n_tests_ric, q05: b05.factors, s05: b05.signals, q10: b10.factors, s10: b10.signals,
        fit: r05.selected, same: r05.same_sign });
    }
    add("okx", "OKX screen", e, "Factorlab's own OKX data; BH over every screened factor");
    add("binance", "Binance new family", b, "Binance's public data; the factors the OKX data could not screen, its own BH family");
    var cc = b && b.available !== false ? obj(b.counts) : null;
    if (cc && isNum(cc.crosscheck)) rows.push({ key: "cross", label: "Binance cross-check", what: "the OKX-screened factors on Binance: agreement only, no selection",
      screened: cc.crosscheck, q05: null, s05: null, q10: null, s10: null, fit: null, same: null });
    function pair(n, sg) { return isNum(n) ? fmt.int(n) + (isNum(sg) ? " (" + fmt.int(sg) + (sg === 1 ? " signal)" : " signals)") : "") : DASH; }
    ui.table(c.body, {
      rows: rows, sortable: false, stack: true, compact: true, caption: "Screen results on Factorlab's data and on Binance's data",
      columns: [
        { key: "label", label: "screen", rowHeader: true, render: function (v, r) { return el("span", null, [el("span", { "class": "strong", text: v }), el("span", { "class": "library-sub", text: r.what })]); } },
        { key: "screened", label: "screened", format: fI, align: "right" },
        { key: "q05", label: "rIC q ≤ 0.05", align: "right", render: function (v, r) { return pair(v, r.s05); } },
        { key: "q10", label: "rIC q ≤ 0.10", align: "right", render: function (v, r) { return pair(v, r.s10); } },
        { key: "fit", label: "selected on G_fit alone, q ≤ 0.05", format: fI, align: "right" },
        { key: "same", label: "of them, same sign on G_eval", format: fI, align: "right" }
      ]
    });
    ui.note(c.body, "A distinct signal is a cluster of definitions with |rank correlation| ≥ 0.7. The details of each screen follow below.");
  }

  function renderEngine(root, d) {
    var e = obj(d.engine);
    var sec = ui.section(root, { id: "engine", title: "On Factorlab's data",
      subtitle: "The library's definitions computed on Factorlab's own point-in-time data for its OKX perpetuals, proven point in time, and screened on the generator zone G (engine/libfactors.py); then the same library on Binance's public data for the same coins." });
    renderScreenSummary(sec.body, d);
    if (!e) { var o = library_fix(); o.title = "Not available"; ui.empty(sec.body, "The library-on-engine block is not in the exported data.", o); }
    else if (e.available === false) {
      ui.empty(sec.body, (e.reason || "No computation was found") + ". Compute it, then export again:", { title: "Not computed on Factorlab's data", command: arr(e.commands).join("\n") });
    } else {
      ui.details(sec.body, { id: "engine-notes", summary: "How to read the OKX screen", meta: "method notes",
        render: function (x) { arr(e.notes).forEach(function (h) { x.appendChild(el("p", { "class": "fl-note", text: h })); }); } });
      library_defer(sec.body, "engine-funnel", function (h) { renderEngineFunnel(h, e); });
      library_defer(sec.body, "engine-coverage", function (h) { renderEngineCoverage(h, e); });
      library_defer(sec.body, "engine-screen", function (h) { renderEngineScreen(h, e); });
      library_defer(sec.body, "engine-top", function (h) { renderEngineTop(h, e, obj(d.commands)); });
    }
    // the same library on Binance's data: a separate block with its own method notes
    renderBinance(sec.body, d);
  }

  /* ------------------------------------------------------------------ page */
  FL.pages.register({
    id: "library",
    title: "Factor library",
    icon: "library",
    subtitle: "Formula definitions, numbers from two screens on Factorlab's data, and a check and a register command for each definition.",
    render: function (root, ctx) {
      ui = ctx.ui; el = ui.el; fmt = ctx.fmt; charts = ctx.charts; colors = ctx.colors;
      library_style();
      // cards below the first screen render later in the app shell; in node they render at once
      var deferFn = ctx.defer || (ui && ui.defer);
      library_defer = library_makeDefer(deferFn);
      var d = ctx.section || {};
      if (d.available === false) {
        // the app shell shows a reason string first; this guard covers a direct render
        var o = library_fix();
        o.title = "No data";
        ui.empty(root, "No data for this page: " + (d.reason || "the factor library catalog is not built") + ". Build the data with ./setup.sh, then export the pages.", o);
        return;
      }
      // the screen results, then the whole library
      renderStatement(root, d, ctx.nav);
      var toc = [["screen-summary", "Screen results"], ["engine-funnel", "OKX screen"], ["binance", "Binance data"],
        ["browse", "Whole library"], ["search", "Search"]];
      if (ui.toc) ui.toc(root, toc, { page: "library" });
      renderEngine(root, d);
      var rest = ui.section(root, { id: "browse", title: "The whole library",
        subtitle: "Every definition with its formula and its two commands, and the families." });
      library_defer(rest.body, "kpis", function (h) { renderKpis(h, d); });
      library_defer(rest.body, "search", function (h) { renderSearch(h, d, ctx); });
      library_defer(rest.body, "families", function (h) { renderFamilies(h, d); });
      library_defer(rest.body, "use", function (h) { renderUse(h); });
    }
  });
})();
