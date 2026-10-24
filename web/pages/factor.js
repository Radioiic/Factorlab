/* Factorlab — the page of one registered test (#/factor/<id>, for example #/factor/FT-0001).
 *
 * Data: FL.data.mine (engine/webexport/mine.py): the row of the test, and its part file
 * data/mine.<id in lower case>.js, which the page loads (the registration line, the historical
 * test, the forward test, the alarms). The page judges nothing: the verdict word, the state and
 * the reasons come from the data. A library id (CF00001) links to the library search, which
 * shows its definition and its check and register commands. Every data-derived string goes in
 * via textContent (FL.ui.el).
 */
(function () {
  "use strict";
  var FL = window.FL;
  if (!FL || !FL.pages) return;

  var ui, el, fmt, charts;
  var DASH = "—";
  var STYLE_ID = "factor-page-style";
  var PIT_FORMULA = "Point in time: proven at each Sunday cut.";
  var PIT_FILE = "Point in time: not proven. A forward value counts only from the first daily close after its log time.";
  var CSS = [
    ".factor-links{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;margin:8px 0 0}",
    ".factor-links a{color:var(--ink-2)}",
    ".factor-expr{font-family:var(--mono);font-size:12.5px;overflow-wrap:anywhere;white-space:pre-wrap}",
    ".factor-list{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:4px;font-size:13.5px}",
    ".factor-meters{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:0 0 12px}",
    "@media (max-width:599.98px){.factor-meters{grid-template-columns:minmax(0,1fr)}}",
    ".factor-sha{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);overflow-wrap:anywhere}"
  ].join("\n");
  function style() {
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
  function pct(v, dp) { return isNum(v) ? fmt.pct(v, dp === undefined ? 2 : dp) : DASH; }
  function num(v, dp, signed) { return isNum(v) ? fmt.num(v, dp, { signed: !!signed }) : DASH; }
  function plural(n, one, many) { return fmt.int(n) + " " + (n === 1 ? one : many); }
  function href(ctx, page, card) {
    if (ctx.nav && ctx.nav.href) return ctx.nav.href(page, card || null, null);
    return "#/" + page + (card ? "/" + card : "");
  }
  function sourceName(s) { return s === "okx" ? "OKX" : s === "binance" ? "Binance" : txt(s); }
  function pairs(dates, values) {
    var out = [];
    arr(dates).forEach(function (d, i) { out.push([d, arr(values)[i] === undefined ? null : arr(values)[i]]); });
    return out;
  }

  /* ------------------------------------------------------------------ the states without a test */
  function renderLibraryId(root, ctx, id) {
    ui.empty(root, id + " is a library definition. The library search shows its definition and its check and register commands.", { title: "Library definition" });
    // the search link carries the id: the library search opens with it (in memory only, for this visit)
    var find = el("a", { attrs: { href: href(ctx, "library", "search") }, text: "Find " + id + " in the library search" });
    find.addEventListener("click", function () { FL.searchHint = id; });
    root.appendChild(el("p", { "class": "factor-links" }, [find]));
  }
  function mineLink(ctx) {
    return el("p", { "class": "factor-links" }, [el("a", { attrs: { href: href(ctx, "mine") }, text: "The page My factors lists the tests" })]);
  }
  /* the same warning as the page My factors: the ledger fails its checks, so the data has no row */
  function renderBroken(root, ctx, d, id) {
    var led = obj(d.ledger) || {};
    var cmds = obj(d.commands) || {};
    if (ctx.setTitle) ctx.setTitle(id, "The trial ledger does not pass its checks.");
    ui.callout("warning", "The trial ledger does not pass its checks", arr(led.problems).map(function (p) { return txt(p); }).concat(
      ["The page shows no test. This command shows the cause. " + txt(cmds.where_one || cmds.where)]), { container: root });
    root.appendChild(ui.code(txt(cmds.verify), { block: true }));
    root.appendChild(mineLink(ctx));
  }

  /* ------------------------------------------------------------------ 1. KPIs */
  function renderKpis(root, r, rule) {
    var f = obj(r.forward) || {}, s = obj(r.spec) || {}, h = obj(r.historical) || {};
    var maxW = isNum(rule.max_weeks) ? rule.max_weeks : 156;
    var week = r.withdrawn ? "withdrawn" : f.held ? "held" : f.weeks ? fmt.int(f.weeks) + " of " + fmt.int(maxW) : "waiting";
    var v = obj(r.verdict) || {};
    // a withdrawn test has no verdict and no forward state
    ui.kpis(root, [
      r.withdrawn ? { label: "Verdict", value: "none (withdrawn)", note: "not in N" }
        : { label: "Verdict", value: txt(v.word), badge: { status: v.badge || "neutral", label: "by the " + txt(v.by) + " test" } },
      { label: "Input", value: r.input === "values_file" ? "values file" : (r.library_id ? r.library_id : "formula"), note: r.input === "values_file" ? txt((obj(r.file) || {}).name) : sourceName(r.source) + " data" },
      { label: "Book", value: txt(s.name), note: "version " + txt(s.version) + ", " + txt(s.direction) + ", control " + txt(s.control) },
      { label: "Historical test", value: txt(h.word), note: "total return " + pct(h.total_return) + ", Sharpe ratio " + num(h.sharpe, 2) },
      { label: "Forward week", value: week, badge: r.withdrawn ? null : { status: f.badge || "neutral", label: txt(f.state) } }
    ]);
  }

  /* ------------------------------------------------------------------ 2. the factor */
  function renderFactor(root, r, part, d) {
    var reg = obj(part.registration) || {};
    var fac = obj(reg.factor) || {};
    var c = ui.card({ id: "factor", level: 2, title: "The factor" }, root);
    if (r.input === "values_file") {
      ui.dl(c.body, [
        { term: "Values file", value: txt(fac.file_name), mono: true },
        { term: "sha256", value: txt(fac.file_sha256), mono: true },
        ["Rows", fmt.int(fac.rows) + " (" + fmt.int(fac.sunday_rows) + " on a Sunday with a bar), " + txt(fac.first_date) + " to " + txt(fac.last_date)]
      ]);
      c.body.appendChild(el("p", { text: PIT_FILE }));
    } else {
      c.body.appendChild(el("pre", { "class": "fl-code factor-expr", text: txt(r.expression) }));
      var rows = [["Data", sourceName(r.source)], { term: "sha256", value: txt(fac.expression_sha256), mono: true },
        ["Fields", arr(fac.raw_fields).join(", ") || DASH]];
      if (r.library_id) rows.push(["Library definition", txt(r.library_id)]);
      ui.dl(c.body, rows);
      c.body.appendChild(el("p", { text: PIT_FORMULA }));
    }
    var notes = obj(d.notes) || {};
    if (arr(r.notes).length) {
      var nl = el("ul", { "class": "factor-list" });
      arr(r.notes).forEach(function (n) { nl.appendChild(el("li", { text: notes[n] || String(n) })); });
      c.body.appendChild(el("h4", { text: "Notes" }));
      c.body.appendChild(nl);
    }
    if (reg.note) c.body.appendChild(el("p", { "class": "fl-note", text: "Note of the registration: " + reg.note }));
  }

  /* ------------------------------------------------------------------ 3. the book specification */
  function renderSpec(root, part) {
    var reg = obj(part.registration) || {};
    var bs = obj(reg.book_spec) || {}, body = obj(bs.body) || {};
    var c = ui.card({ id: "book", level: 2, title: "Book specification", subtitle: "The normalized content that the registration records (file " + txt(bs.source) + ")." }, root);
    var rows = [{ key: "name", value: txt(body.name) }, { key: "version", value: txt(body.version) }];
    ["book", "costs", "control", "forward"].forEach(function (t) {
      var b = obj(body[t]) || {};
      Object.keys(b).forEach(function (k) { rows.push({ key: t + "." + k, value: b[k] === null ? "none" : String(b[k]) }); });
    });
    ui.table(c.body, { columns: [{ key: "key", label: "key", mono: true, rowHeader: true }, { key: "value", label: "value", mono: true }],
      rows: rows, sortable: false, compact: true, ariaLabel: "Book specification" });
    c.body.appendChild(el("p", { "class": "factor-sha", text: "sha256 " + txt(bs.sha256) }));
  }

  /* ------------------------------------------------------------------ 4. the historical test */
  function renderHistorical(root, r, part, d) {
    var h = obj(part.historical), row = obj(r.historical) || {};
    var cmds = obj(d.commands) || {};
    var c = ui.card({ id: "historical", level: 2, title: "Historical test",
      subtitle: h ? "The book from the first Sunday " + txt(h.start) + " to the last Sunday " + txt(h.end) + ", after costs." : null }, root);
    if (!h) {
      ui.empty(c.body, "The result files are absent. Run judge.py verify --write-results. " + txt(cmds.where_one || cmds.where), { command: txt(cmds.write_results) });
      return;
    }
    var w = obj(h.weekly) || {};
    var series = [{ id: "book", name: "factor book", entity: "factor_book", values: pairs(w.sundays, w.book) }];
    if (arr(w.control).length) series.push({ id: "control", name: "control book", entity: "control", values: pairs(w.sundays, w.control) });
    series.push({ id: "btc", name: "BTC hold", entity: "btc", values: pairs(w.sundays, w.btc) });
    charts.line(c.body, { title: "Weekly marks, a multiple of the start", series: series, y: { format: "num:2", log: true, label: "multiple of the start (log scale)" }, height: 260 });
    var st = obj(h.stats) || {};
    var srows = [["factor book", st.book], ["control book", st.control], ["BTC hold", st.btc], ["net excess over the control", st.excess]]
      .filter(function (x) { return obj(x[1]); })
      .map(function (x) { return { book: x[0], total_return: x[1].total_return, sharpe: x[1].sharpe, ann_vol: x[1].ann_vol, max_dd: x[1].max_dd }; });
    ui.table(c.body, { caption: "After costs, from the first Sunday to the last Sunday.", columns: [
      { key: "book", label: "", rowHeader: true },
      { key: "total_return", label: "total return", align: "right", render: function (v) { return pct(v); } },
      { key: "sharpe", label: "Sharpe ratio", align: "right", render: function (v) { return num(v, 2); } },
      { key: "ann_vol", label: "volatility (year)", align: "right", render: function (v) { return isNum(v) ? fmt.pct(v, 2, { signed: false }) : DASH; } },
      { key: "max_dd", label: "largest drawdown", align: "right", render: function (v) { return pct(v); } }
    ], rows: srows, sortable: false, compact: true, stack: true });
    var wins = arr(h.windows);
    if (wins.length) {
      charts.bars(c.body, { title: "Sharpe ratio of each rolling 1-year window", x: wins.map(function (x) { return x.end; }),
        series: [{ id: "sharpe", name: "Sharpe ratio of the window", entity: "excess", values: wins.map(function (x) { return x.sharpe; }) }],
        y: { format: "num:2" }, refLines: [{ y: 0 }], height: 200 });
      // a share, not a change: no sign
      var share = isNum(row.share_positive) ? fmt.pct(row.share_positive, 0, { signed: false }) : DASH;
      c.body.appendChild(el("p", { text: "Windows with a positive Sharpe ratio: " + fmt.int(row.n_positive) + " of " + fmt.int(row.n_windows) + " (" + share + "). Each window ends on the date of its bar." }));
    } else {
      c.body.appendChild(el("p", { text: "No 1-year window: the history is shorter than 366 days." }));
    }
    var res = el("p", null, [el("span", { text: "Result: " }), el("strong", { text: txt(row.word) })]);
    c.body.appendChild(res);
    if (arr(h.reasons).length) {
      var rl = el("ul", { "class": "factor-list" });
      arr(h.reasons).forEach(function (x) { rl.appendChild(el("li", { text: x })); });
      c.body.appendChild(rl);
    }
    c.body.appendChild(el("p", { text: "BTC hold gives context only." }));
    var fnum = obj(h.factor_numbers) || {};
    ui.dl(c.body, [
      ["Weeks", fmt.int(row.weeks) + " (" + fmt.int(row.held_weeks) + " with a factor position)"],
      ["Sigma floor of the forward test", num(row.sigma_floor, 4) + " (" + txt(row.sigma_floor_source) + ")"],
      ["Weekly SD, long-run SD", num(row.weekly_sd, 4) + ", " + num(row.weekly_lrsd, 4)],
      ["Numbers of the factor (no rule)", "weekly IC " + num(fnum.ic_mean, 4, true) + " (t " + num(fnum.ic_t, 2, true) + "), weekly rIC " + num(fnum.ric_mean, 4, true) +
        " (t " + num(fnum.ric_t, 2, true) + "), persistence " + num(fnum.persistence, 2) + ", coverage " + num(fnum.coverage, 2) + ", " + plural(fnum.weeks, "week", "weeks")]
    ]);
    if (arr(h.missed).length) c.body.appendChild(el("p", { "class": "fl-note", text: "At " + plural(arr(h.missed).length, "signal Sunday", "signal Sundays") + " the book has the control portfolio (fewer than 5 values): " + arr(h.missed).join(", ") + "." }));
  }

  /* ------------------------------------------------------------------ 5. the forward test */
  function gapText(gaps) {
    var first = {};
    arr(gaps).forEach(function (g) { var k = g[0] + " " + g[1]; if (!first[k] || g[2] < first[k]) first[k] = g[2]; });
    return Object.keys(first).map(function (k) { return "Missing data in this week: " + k + " from " + first[k] + ". The week counts."; });
  }
  function renderForward(root, r, part, rule) {
    var f = obj(part.forward) || {}, fr = obj(r.forward) || {};
    var path = arr(f.path);
    var first = isNum(rule.first_decision_week) ? rule.first_decision_week : 26;
    var maxW = isNum(rule.max_weeks) ? rule.max_weeks : 156;
    var ms = obj(f.milestones) || {};
    var c = ui.card({ id: "forward", level: 2, title: "Forward test",
      subtitle: "The weeks after the registration. The first forward Sunday is " + txt(f.first_forward_sunday) + ". Week " + first + " ends on " + txt(ms.week26_end) + ", week " + maxW + " on " + txt(ms.week156_end) + "." }, root);
    if (r.withdrawn) {
      var wd = obj(part.withdrawal) || {};
      c.body.appendChild(el("p", { text: "Withdrawn before the first forward week." }));
      if (wd.reason) c.body.appendChild(el("p", { "class": "fl-note", text: "Reason: " + wd.reason + " (" + txt(wd.withdrawn_at) + ")" }));
      return;
    }
    var mBox = el("div", { "class": "factor-meters" });
    c.body.appendChild(mBox);
    var n = path.length;
    ui.meter(mBox, { label: "Weeks to the first decision week", value: Math.min(n, first), max: first, valueText: fmt.int(n) + " of " + first });
    ui.meter(mBox, { label: "Weeks to the maximum", value: n, max: maxW, valueText: fmt.int(n) + " of " + maxW, ticks: [{ value: first, label: String(first) }] });
    if (fr.held) c.body.appendChild(el("p", { text: "Held at " + txt(fr.held.sunday) + ": " + txt(fr.held.reason) + "." }));
    if (!n) {
      c.body.appendChild(el("p", { text: "Waiting for the first forward week. It ends on " + txt(f.next_week_end) + "." }));
      return;
    }
    var last = path[n - 1];
    var dec = fr.decided_week ? "decided at week " + fr.decided_week : f.next_week_end ? "the next week ends on " + f.next_week_end : "";
    c.body.appendChild(el("p", { text: "Week " + n + ": mean " + num(last.mean, 4, true) + ", lower " + num(last.lower, 4, true) + ", upper " + num(last.upper, 4, true) + ", state " + txt(last.state) + (dec ? "; " + dec : "") + "." }));
    var x = path.map(function (p) { return p.sunday; });
    charts.bars(c.body, { title: "Weekly net excess return over the control", x: x,
      series: [{ id: "excess", name: "net excess", entity: "excess", values: path.map(function (p) { return p.excess; }) }],
      y: { format: "pct:2" }, refLines: [{ y: 0 }], height: 200 });
    charts.line(c.body, { title: "The mean and the bounds of the confidence sequence", series: [
      { id: "upper", name: "upper bound", values: pairs(x, path.map(function (p) { return p.upper; })), color: "var(--ink-3)", width: 1.5 },
      { id: "mean", name: "mean", values: pairs(x, path.map(function (p) { return p.mean; })), entity: "excess" },
      { id: "lower", name: "lower bound", values: pairs(x, path.map(function (p) { return p.lower; })), color: "var(--ink-3)", width: 1.5 }
    ], refLines: [{ y: 0, label: "0" }], y: { format: "pct:2", label: "weekly net excess return" }, height: 240 });
    var eq = obj(f.equity) || {};
    var es = [{ id: "book", name: "factor book", entity: "factor_book", values: pairs(eq.sundays, eq.book) }];
    if (arr(eq.control).length) es.push({ id: "control", name: "control book", entity: "control", values: pairs(eq.sundays, eq.control) });
    es.push({ id: "btc", name: "BTC hold", entity: "btc", values: pairs(eq.sundays, eq.btc) });
    charts.line(c.body, { title: "Forward marks, a multiple of the first forward Sunday", series: es, y: { format: "num:3" }, height: 220 });
    var gaps = [];
    path.forEach(function (p) { gapText(p.final_gaps).forEach(function (t) { gaps.push("Week " + p.week + ": " + t); }); });
    if (gaps.length) ui.callout("warning", "Missing data", gaps, { container: c.body });
    ui.table(c.body, { caption: "Each frozen week. The returns are after costs; the bounds use the weeks up to that week.", columns: [
      { key: "week", label: "week", format: "int", rowHeader: true },
      { key: "sunday", label: "signal Sunday", className: "nowrap" },
      { key: "book_ret", label: "book", align: "right", render: function (v) { return pct(v); } },
      { key: "control_ret", label: "control", align: "right", render: function (v) { return pct(v); } },
      { key: "excess", label: "net excess", align: "right", render: function (v) { return pct(v); } },
      { key: "btc_ret", label: "BTC hold", align: "right", render: function (v) { return pct(v); } },
      { key: "mean", label: "mean", align: "right", render: function (v) { return pct(v); } },
      { key: "lower", label: "lower", align: "right", render: function (v) { return pct(v); } },
      { key: "upper", label: "upper", align: "right", render: function (v) { return pct(v); } },
      { key: "state", label: "state" }
    ], rows: path, defaultSort: { key: "week", dir: "desc" }, compact: true, stack: true, pageSize: 26 });
  }

  /* ------------------------------------------------------------------ 6. the record */
  function renderRecord(root, r, part, d) {
    var reg = obj(part.registration) || {};
    var cmds = obj(d.commands) || {};
    var c = ui.card({ id: "record", level: 2, title: "Record", subtitle: "The trial ledger data/trials/ (append-only and hash-chained)." }, root);
    ui.dl(c.body, [
      ["Registration", "registrations.jsonl line " + txt(reg.line) + ", " + txt(reg.registered_at)],
      { term: "sha256 of the line", value: txt(reg.sha256), mono: true },
      ["Related tests", arr(r.related).length ? arr(r.related).join(", ") : "none"],
      ["Alarms", fmt.int(arr(part.alarms).length)]
    ]);
    if (arr(part.alarms).length) {
      var al = el("ul", { "class": "factor-list" });
      arr(part.alarms).forEach(function (a) { al.appendChild(el("li", { text: txt(a.text) })); });
      c.body.appendChild(al);
    }
    var fr = obj(r.forward) || {};
    var next = null, why = null;
    if (r.withdrawn) why = "The test is withdrawn. No command follows.";
    else if (fr.decided_week) { why = "The forward test has a decision. No week follows it. To show the tests:"; next = cmds.status; }
    else if (r.input === "values_file") { why = "Log the values of each Sunday before 24:00 UTC on that Sunday. Then the book uses them from Monday 00:00 UTC. A later log counts from the next daily close."; next = String(cmds.log_values || "").replace("<FT id>", r.id); }
    else { why = "update.py logs each new week. To log the weeks yourself:"; next = cmds.log_week; }
    c.body.appendChild(el("p", { text: why }));
    if (next) c.body.appendChild(el("p", { "class": "fl-note", text: txt(cmds.where_one || cmds.where) }));
    if (next) c.body.appendChild(ui.code(next, { block: true }));
  }

  /* ------------------------------------------------------------------ the load and the render of the part */
  function partMissing(host, r, err) {
    ui.empty(host, "The part file data/mine." + r.part + ".js " + (err && err.state === "broken" ? "does not parse." : "is not in this folder.") + " Build the pages again.",
      { title: "Part file not available", command: FL.cmd ? FL.cmd.export : "cd engine && .venv/bin/python -B export_web.py" });
  }
  function renderPart(host, r, part, d, rule) {
    // result files that judge.py wrote again after a change of the data: the pages show the ledger
    var diff = arr(part.results_differ).map(function (x) { return txt((obj(x) || {}).text); });
    if (diff.length) ui.callout("warning", "The result files differ from the ledger", diff, { container: host });
    renderFactor(host, r, part, d);
    renderSpec(host, part);
    renderHistorical(host, r, part, d);
    renderForward(host, r, part, rule);
    renderRecord(host, r, part, d);
  }

  FL.pages.register({
    id: "factor",
    title: "Factor test",
    subtitle: "The page of one test.",
    icon: "factors",
    section: "mine",
    data: ["mine"],
    requiresData: false,
    render: function (root, ctx) {
      ui = FL.ui; el = ui.el; fmt = FL.fmt; charts = FL.charts;
      style();
      var nav = ctx.nav || {};
      if (nav.badParam) { ui.empty(root, "The id in the address is not valid.", { title: "Unknown test" }); root.appendChild(mineLink(ctx)); return; }
      if (!nav.param) { ui.empty(root, "The address names no test.", { title: "No test" }); root.appendChild(mineLink(ctx)); return; }
      var id = nav.param;
      if (ctx.setTitle) ctx.setTitle(id);
      if (/^CF\d+$/.test(id)) {
        if (ctx.setTitle) ctx.setTitle(id, "A library definition.");
        renderLibraryId(root, ctx, id);
        return;
      }
      var d = ctx.section;
      if (!d || d.available === false) {
        ui.empty(root, "No data for the page My factors" + (d && d.reason ? ": " + d.reason : "") + ".", { title: "No data", commands: ui.remedies() });
        return;
      }
      // a broken ledger gives no row: the page says why, not that the id is not registered
      if ((obj(d.ledger) || {}).ok === false) { renderBroken(root, ctx, d, id); return; }
      var want = String(id).toUpperCase();
      var r = arr(d.rows).filter(function (x) { return String(x.id).toUpperCase() === want; })[0];
      if (!r) {
        if (ctx.setTitle) ctx.setTitle(id, "No registered test has this id.");
        ui.empty(root, "No registered test has this id.", { title: "Unknown test" });
        root.appendChild(mineLink(ctx));
        return;
      }
      if (ctx.setTitle) ctx.setTitle(r.id, txt(r.name) + (r.withdrawn ? " (withdrawn)" : ""));
      var rule = obj(d.rule) || {};
      renderKpis(root, r, rule);
      var host = el("div", { "class": "stack" });
      root.appendChild(host);
      host.appendChild(el("p", { "class": "fl-loading", text: "The page loads the part file of the test.", attrs: { role: "status" } }));
      var load = ctx.loadPart ? ctx.loadPart(r.part) : Promise.reject({ state: "missing" });
      // a load failure names the part file; a render failure names the error (and logs it)
      load.then(function (part) {
        ui.clear(host);
        if (!obj(part)) { partMissing(host, r, { state: "broken" }); return; }
        try {
          renderPart(host, r, part, d, rule);
          // the cards exist only now: a card route (#/factor/<id>/forward) scrolls to its card here
          if (ctx.cardReady) ctx.cardReady();
        } catch (err) {
          if (window.console && console.error) console.error(err);
          ui.clear(host);
          ui.callout("critical", "This test page failed to render",
            ["Error: " + (err && err.message ? err.message : String(err)), "Export the pages again, or check the browser console."], { container: host });
        }
      }, function (err) {
        ui.clear(host);
        partMissing(host, r, err);
      });
    }
  });
})();
