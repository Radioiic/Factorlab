/* Factorlab — My factors page (#/mine): the tests of your own factors.
 *
 * Data: FL.data.mine (engine/webexport/mine.py). The page judges nothing: each verdict word, each
 * label of a verdict or a state, the rule text and the expected number of false verdicts come
 * from the data file. Every data-derived string goes in via textContent (FL.ui.el / FL.ui.table).
 * The page of one test is pages/factor.js (#/factor/<id>).
 */
(function () {
  "use strict";
  var FL = window.FL;
  if (!FL || !FL.pages) return;

  var ui, el, fmt;
  var DASH = "—";
  var STYLE_ID = "mine-page-style";
  var STATEMENT = "Give a formula or a values file. Factorlab tests it in a book: first a historical test, then a forward test with the weeks after the registration.";
  var CSS = [
    ".mine-filters{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;margin:0 0 12px}",
    ".mine-filters label{display:inline-flex;gap:8px;align-items:center;font-size:13px;color:var(--ink-2);max-width:100%;min-width:0}",
    ".mine-filters select{min-width:0;max-width:100%;flex:1 1 auto}",
    ".mine-id a{font-family:var(--mono);font-size:12.5px;color:var(--ink);white-space:nowrap}",
    ".mine-kpis>.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}",
    "@media (max-width:719.98px){.mine-kpis>.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}",
    "@media (max-width:359.98px){.mine-kpis>.kpis{grid-template-columns:minmax(0,1fr)}}",
    ".mine-rule ul{margin:4px 0 0;padding-left:20px;display:flex;flex-direction:column;gap:4px}",
    ".mine-sub{display:block;color:var(--ink-3);font-size:12px;overflow-wrap:anywhere;white-space:normal}",
    ".mine-name{display:flex;flex-direction:column;gap:2px;min-width:12ch;max-width:30ch}",
    ".mine-expr{font-family:var(--mono);font-size:12px;overflow-wrap:anywhere;white-space:normal}",
    ".mine-nowrap{white-space:nowrap}",
    ".mine-rule{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:4px;font-size:13.5px;color:var(--ink)}",
    ".mine-cmd{display:flex;flex-direction:column;gap:4px;margin:0 0 10px}",
    ".mine-cmd span{font-size:13px;color:var(--ink-2)}",
    ".mine-specs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}",
    "@media (max-width:719.98px){.mine-specs{grid-template-columns:minmax(0,1fr)}}",
    ".mine-spec h4{margin:0 0 6px;font-size:14px}",
    ".mine-sha{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);overflow-wrap:anywhere}",
    "@media (max-width:599.98px){.mine-name{min-width:0;max-width:none}}"
  ].join("\n");
  function mine_style() {
    if (document.getElementById(STYLE_ID)) return;
    var s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  function arr(v) { return Array.isArray(v) ? v : []; }
  function obj(v) { return v && typeof v === "object" && !Array.isArray(v) ? v : null; }
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function txt(v) { return v === null || v === undefined || v === "" ? DASH : String(v); }
  function fB(v) { return isNum(v) ? fmt.num(v, 4, { signed: true }) : DASH; }
  function plural(n, one, many) { return fmt.int(n) + " " + (n === 1 ? one : many); }
  function factorHref(ctx, id) {
    if (ctx.nav && ctx.nav.factorHref) return ctx.nav.factorHref(id);
    if (FL.app && FL.app.factorHref) return FL.app.factorHref(id);
    return "#/factor/" + encodeURIComponent(id);
  }
  function sourceName(s) { return s === "okx" ? "OKX" : s === "binance" ? "Binance" : txt(s); }
  /** "formula (OKX)", "library CF00001 (OKX)" or "values file" */
  function inputText(r) {
    if (r.input === "values_file") return "values file";
    return (r.library_id ? "library " + r.library_id : "formula") + " (" + sourceName(r.source) + ")";
  }
  /** the forward week: "t of 156", "waiting", "held" or "withdrawn" */
  function weekText(r, maxWeeks) {
    var f = obj(r.forward) || {};
    if (r.withdrawn) return "withdrawn";
    if (f.held) return "held";
    if (!f.weeks) return "waiting";
    return fmt.int(f.weeks) + " of " + fmt.int(maxWeeks);
  }

  /* ------------------------------------------------------------------ 1. statement */
  function renderStatement(root) {
    var box = el("div", { id: "statement" });
    root.appendChild(box);
    ui.callout("info", null, STATEMENT, { container: box });
  }

  /* ------------------------------------------------------------------ 2. KPIs */
  function renderKpis(root, d) {
    var c = obj(d.counts);
    var ex = obj(d.expected_false_useful);
    if (!c) return;
    var items = [
      { label: "Registered tests (N)", value: fmt.int(c.n), note: plural(c.registered, "registration", "registrations") + ", " + fmt.int(c.withdrawn) + " withdrawn", id: "mine-kpi-n" },
      { label: "Running", value: fmt.int((obj(c.states) || {}).running || 0), note: "forward tests without a decision" }
    ];
    var v = obj(c.verdicts) || {};
    Object.keys(v).forEach(function (k) { items.push({ label: k, value: fmt.int(v[k]), note: "verdict" }); });
    if (ex) items.push({ label: txt(ex.label), value: isNum(ex.value) ? fmt.num(ex.value, 1) : DASH, title: txt(ex.text), note: "0.1 x N", id: "mine-kpi-expected" });
    var wrap = el("div", { "class": "mine-kpis" });
    root.appendChild(wrap);
    ui.kpis(wrap, items);
    if (ex && ex.text) ui.note(root, ex.text);
  }

  /* ------------------------------------------------------------------ 3. tests */
  function renderTests(root, d, ctx) {
    var rows = arr(d.rows);
    var rule = obj(d.rule) || {};
    var maxWeeks = isNum(rule.max_weeks) ? rule.max_weeks : 156;
    var card = ui.card({ id: "tests", level: 2, title: "Tests",
      subtitle: "One row for each registered test. The id opens the page of the test." }, root);
    var alarms = arr(d.alarms);
    if (alarms.length) {
      ui.callout("warning", plural(alarms.length, "alarm of the data", "alarms of the data"), alarms.map(function (a) { return txt(a.text); }), { container: card.body });
    }
    var code = obj(d.code);
    if (code && code.matches === false) {
      ui.callout("warning", "The test code differs from the code that the ledger records",
        ["Changed: " + (arr(code.changed).join(", ") || "the values of config.py") + ".", "Record the change before the next weekly log: judge.py record-code-change --reason TEXT"], { container: card.body });
    }
    // result files that judge.py wrote again after a change of the data: the pages show the ledger
    var differ = [];
    rows.forEach(function (r) { arr(r.results_differ).forEach(function (x) { differ.push(r.id + ": " + txt((obj(x) || {}).text)); }); });
    if (differ.length) ui.callout("warning", "The result files differ from the ledger", differ, { container: card.body });
    // a withdrawn test has no verdict: the filter counts it apart, as the KPI tiles do
    var words = {}, nWithdrawn = 0;
    arr(rows).forEach(function (r) {
      if (r.withdrawn) { nWithdrawn += 1; return; }
      var w = (obj(r.verdict) || {}).word; if (w) words[w] = (words[w] || 0) + 1;
    });
    var state = { verdict: "all" };
    var bar = el("div", { "class": "mine-filters" });
    card.body.appendChild(bar);
    var sel = el("select", { "class": "fl-select", attrs: { "aria-label": "Filter by verdict" } });
    sel.appendChild(el("option", { text: "Every test (" + rows.length + ")", attrs: { value: "all" } }));
    Object.keys(words).sort().forEach(function (w) { sel.appendChild(el("option", { text: w + " (" + words[w] + ")", attrs: { value: "v:" + w } })); });
    if (nWithdrawn) sel.appendChild(el("option", { text: "withdrawn (" + nWithdrawn + ")", attrs: { value: "withdrawn" } }));
    bar.appendChild(el("label", null, [el("span", { text: "Verdict" }), sel]));
    var cols = [
      { key: "id", label: "id", rowHeader: true, render: function (v) {
        return el("span", { "class": "mine-id" }, [el("a", { attrs: { href: factorHref(ctx, v) }, text: v })]);
      } },
      { key: "name", label: "name", render: function (v, r) {
        var what = r.input === "values_file" ? (obj(r.file) || {}).name : r.expression;
        return el("span", { "class": "mine-name" }, [el("span", { text: txt(v) }), el("span", { "class": r.input === "values_file" ? "mine-sub" : "mine-sub mine-expr", text: txt(what) })]);
      } },
      { key: "input", label: "input", value: function (r) { return inputText(r); } },
      { key: "book", label: "book", value: function (r) { var s = obj(r.spec) || {}; return txt(s.name) + " v" + txt(s.version); } },
      { key: "historical", label: "historical test", value: function (r) { return (obj(r.historical) || {}).word || DASH; } },
      { key: "week", label: "forward week", value: function (r) { return weekText(r, maxWeeks); },
        sortValue: function (r) { var f = obj(r.forward) || {}; return r.withdrawn ? -2 : isNum(f.weeks) ? f.weeks : -1; },
        render: function (v, r) {
          var f = obj(r.forward) || {};
          if (f.held) return el("span", { title: "held at " + txt(f.held.sunday) + ": " + txt(f.held.reason), text: "held at " + txt(f.held.sunday) });
          return v;
        } },
      { key: "lower", label: "lower", align: "right", value: function (r) { return (obj(r.forward) || {}).lower; }, render: function (v) { return fB(v); } },
      { key: "upper", label: "upper", align: "right", value: function (r) { return (obj(r.forward) || {}).upper; }, render: function (v) { return fB(v); } },
      { key: "state", label: "state", value: function (r) { return r.withdrawn ? "withdrawn" : (obj(r.forward) || {}).state; },
        render: function (v, r) { return r.withdrawn ? v : ui.badge((obj(r.forward) || {}).badge || "neutral", txt(v)); } },
      { key: "verdict", label: "verdict", value: function (r) { return r.withdrawn ? "none (withdrawn)" : (obj(r.verdict) || {}).word; },
        render: function (v, r) { return r.withdrawn ? v : ui.badge((obj(r.verdict) || {}).badge || "neutral", txt(v)); } },
      { key: "related", label: "related tests", align: "right", value: function (r) { return arr(r.related).length; },
        render: function (v, r) { return el("span", { title: arr(r.related).length ? arr(r.related).join(", ") : "none", text: fmt.int(v) }); } },
      { key: "alarms", label: "alarms", format: "int" }
    ];
    var tbl = ui.table(card.body, { columns: cols, rows: rows, defaultSort: { key: "id", dir: "asc" }, compact: true, stack: true,
      caption: "The registered tests. Lower and upper are the bounds of the confidence sequence of the weekly net excess return.",
      rowClass: function (r) { return r.withdrawn ? "is-muted" : null; } });
    sel.addEventListener("change", function () {
      state.verdict = sel.value;
      tbl.setRows(rows.filter(function (r) {
        if (state.verdict === "all") return true;
        if (state.verdict === "withdrawn") return !!r.withdrawn;
        return !r.withdrawn && "v:" + (obj(r.verdict) || {}).word === state.verdict;
      }));
    });
  }

  /* ------------------------------------------------------------------ 4. the rule */
  function renderRule(root, d) {
    var rule = obj(d.rule) || {};
    var card = ui.card({ id: "rule", level: 2, title: "The rule", subtitle: "The rule " + txt(rule.rule_version) + ". Each registration records it." }, root);
    // a line that starts with "- " is an item of the line before it
    var list = el("ul", { "class": "mine-rule" }), last = null, sub = null;
    arr(rule.text).forEach(function (line) {
      line = String(line);
      if (/^- /.test(line) && last) {
        if (!sub) { sub = el("ul"); last.appendChild(sub); }
        sub.appendChild(el("li", { text: line.slice(2) }));
        return;
      }
      sub = null;
      last = el("li", { text: line });
      list.appendChild(last);
    });
    card.body.appendChild(list);
    ui.dl(card.body, [
      ["alpha", txt(rule.alpha) + " (" + txt(rule.sides) + ")"],
      ["Tune week of the mixture", txt(rule.tune_week) + " (r " + txt(rule.r) + ")"],
      ["First decision week", txt(rule.first_decision_week)],
      ["Maximum", txt(rule.max_weeks) + " weeks"],
      ["SD of the bounds", txt(rule.sd_rule) + ", " + txt(rule.lr_lags) + " lags"],
      ["Sigma floor from the history", "at least " + txt(rule.min_floor_weeks) + " weeks with a factor position"],
      ["Rolling window", txt(rule.window_days) + " days, step " + txt(rule.step_days) + " days"],
      ["Statistic", txt(rule.statistic)]
    ]);
  }

  /* ------------------------------------------------------------------ 5. book specifications */
  function specRows(body) {
    var out = [];
    ["book", "costs", "control", "forward"].forEach(function (t) {
      var b = obj(body && body[t]) || {};
      Object.keys(b).forEach(function (k) { out.push({ key: t + "." + k, value: b[k] === null ? "none" : String(b[k]) }); });
    });
    return out;
  }
  function renderSpecs(root, d) {
    var specs = arr(d.specs);
    var cmds = obj(d.commands) || {};
    var card = ui.card({ id: "specs", level: 2, title: "Book specifications",
      subtitle: "The two shipped files. A test records the normalized content and its sha256. You can give your own file with --spec FILE." }, root);
    var grid = el("div", { "class": "mine-specs" });
    card.body.appendChild(grid);
    specs.forEach(function (s) {
      var box = el("div", { "class": "mine-spec" });
      box.appendChild(el("h4", { text: txt(s.file) }));
      ui.table(box, { columns: [{ key: "key", label: "key", mono: true, rowHeader: true }, { key: "value", label: "value", mono: true }],
        rows: [{ key: "name", value: txt((s.body || {}).name) }, { key: "version", value: txt((s.body || {}).version) }].concat(specRows(s.body)),
        sortable: false, compact: true, ariaLabel: txt(s.file) });
      box.appendChild(el("p", { "class": "mine-sha", text: "sha256 " + txt(s.sha256) }));
      grid.appendChild(box);
    });
    if (cmds.spec) {
      card.body.appendChild(el("p", { "class": "fl-note", text: "To check a file and show its sha256, use this command. " + txt(cmds.where_one || cmds.where) }));
      card.body.appendChild(ui.code(cmds.spec, { block: true }));
    }
  }

  /* ------------------------------------------------------------------ 6. commands */
  var COMMAND_LABELS = [
    ["check_formula", "Check a formula (the historical test; writes nothing)"],
    ["register_formula", "Register a formula"],
    ["check_library", "Check a library definition"],
    ["register_library", "Register a library definition"],
    ["check_values", "Check a values file (CSV or Parquet: date, symbol, value)"],
    ["register_values", "Register a values file"],
    ["log_values", "Log the new values of a values-file test"],
    ["log_week", "Log the weeks (update.py runs it)"],
    ["status", "Show the tests"],
    ["verify", "Check the ledger and compute the logged numbers again"]
  ];
  function renderCommands(root, d) {
    var cmds = obj(d.commands) || {};
    var card = ui.card({ id: "commands", level: 2, title: "Commands", subtitle: txt(cmds.where) }, root);
    COMMAND_LABELS.forEach(function (x) {
      if (!cmds[x[0]]) return;
      card.body.appendChild(el("div", { "class": "mine-cmd" }, [el("span", { text: x[1] }), ui.code(cmds[x[0]], { block: true })]));
    });
  }

  function renderEmpty(root, d) {
    var cmds = obj(d.commands) || {};
    var card = ui.card({ id: "tests", level: 2, title: "Tests" }, root);
    ui.empty(card.body, "No factor is registered. " + txt(cmds.where), { title: "No test", commands: [
      { label: "Check a formula:", command: cmds.check_formula },
      { label: "Register it:", command: cmds.register_formula },
      { label: "Or a values file:", command: cmds.check_values },
      { label: "Register it:", command: cmds.register_values }
    ] });
  }

  function renderBroken(root, d) {
    var led = obj(d.ledger) || {};
    var cmds = obj(d.commands) || {};
    var card = ui.card({ id: "tests", level: 2, title: "Tests" }, root);
    ui.callout("warning", "The trial ledger does not pass its checks", arr(led.problems).map(function (p) { return txt(p); }).concat(
      ["The page shows no test. This command shows the cause. " + txt(cmds.where_one || cmds.where)]), { container: card.body });
    card.body.appendChild(ui.code(txt(cmds.verify), { block: true }));
  }

  FL.pages.register({
    id: "mine",
    title: "My factors",
    subtitle: "The tests of your own factors: the historical test, the weekly forward test and the verdict.",
    icon: "mine",
    section: "mine",
    render: function (root, ctx) {
      ui = FL.ui; el = ui.el; fmt = FL.fmt;
      mine_style();
      var d = ctx.section || {};
      var led = obj(d.ledger) || {};
      renderStatement(root);
      if (led.ok === false) renderBroken(root, d);
      else {
        renderKpis(root, d);
        if (arr(d.rows).length) renderTests(root, d, ctx);
        else renderEmpty(root, d);
      }
      renderRule(root, d);
      renderSpecs(root, d);
      renderCommands(root, d);
    }
  });
})();
