/* Factorlab — System page: how the platform runs and how to trust it.
 *
 * Data: FL.data.system (engine/webexport/system.py). Everything that came from
 * data goes into the DOM through textContent (FL.ui.el / createTextNode). The
 * glossary and quoted rules are Markdown; they are drawn by system_md, a tiny
 * renderer that builds DOM nodes (headings, paragraphs, lists, quotes, fenced
 * code, tables, inline code, bold, italic) and never uses innerHTML.
 *
 * Page-local components (prefixed system_): system_md (Markdown -> DOM),
 * system_steps (the stepper of the update steps), system_chips (path chips), system_filter
 * (glossary filter). Page-local CSS uses the .system- prefix.
 */
(function () {
  "use strict";
  var FL = window.FL;
  if (!FL || !FL.pages) return;

  var DASH = "—";

  /* ------------------------------------------------------------------ page-local CSS (static) */
  var CSS = [
    ".system-steps{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:0}",
    ".system-step{display:grid;grid-template-columns:32px minmax(0,1fr);gap:4px 12px;position:relative;padding-bottom:16px}",
    ".system-step:last-child{padding-bottom:0}",
    ".system-step::before{content:'';position:absolute;left:15px;top:30px;bottom:2px;width:2px;background:var(--grid)}",
    ".system-step:last-child::before{display:none}",
    ".system-step-n{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:600;color:var(--ink);background:var(--surface-2);box-shadow:inset 0 0 0 1px var(--border-strong);font-variant-numeric:tabular-nums}",
    ".system-step.is-plain .system-step-n{background:transparent;color:var(--ink-3);box-shadow:inset 0 0 0 1px var(--axis)}",
    ".system-step-body{min-width:0;display:flex;flex-direction:column;gap:6px}",
    ".system-step-head{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px}",
    ".system-step-title{font-weight:600;color:var(--ink);font-size:14px}",
    ".system-step-what{color:var(--ink-2);font-size:13px;margin:0}",
    ".system-step-head code{overflow-wrap:anywhere;white-space:normal}",
    ".system-io{display:grid;grid-template-columns:52px minmax(0,1fr);gap:4px 8px;align-items:baseline;font-size:12px}",
    ".system-io dt{color:var(--ink-3);margin:0}",
    ".system-io dd{margin:0;min-width:0}",
    ".system-chips{display:flex;flex-wrap:wrap;gap:4px 6px}",
    // paths break only after a "/" (a <wbr> follows each slash); break-word is the last resort and keeps the column's minimum width
    ".system-chip{font-family:var(--mono);font-size:11.5px;padding:1px 6px;border-radius:5px;background:var(--surface-2);color:var(--ink);overflow-wrap:break-word;max-width:100%}",
    ".system-chip.is-ext{font-family:var(--font);color:var(--ink-2)}",
    ".system-list{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:6px;font-size:13.5px;color:var(--ink)}",
    ".system-list li::marker{color:var(--ink-3)}",
    ".system-src{font-family:var(--mono);font-size:11px;color:var(--ink-3);white-space:nowrap}",
    ".system-quote{margin:0;padding:6px 0 6px 12px;border-left:2px solid var(--axis);color:var(--ink);font-size:13.5px}",
    ".system-quote + .system-quote{margin-top:8px}",
    ".system-quote footer{margin-top:4px}",
    ".system-rule{display:flex;flex-direction:column;gap:6px;padding:12px 0;border-top:1px solid var(--grid)}",
    ".system-rule:first-child{border-top:0;padding-top:0}",
    ".system-rule h4{font-size:14px}",
    ".system-rule p{margin:0;color:var(--ink-2);font-size:13px}",
    ".system-when{display:inline-block;font-size:11.5px;color:var(--ink-2);font-variant-numeric:tabular-nums;border:1px solid var(--border);border-radius:5px;padding:0 6px;margin-right:4px;vertical-align:1px}",
    ".system-filter{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px}",
    ".system-filter label{font-size:13px;color:var(--ink-2)}",
    ".system-filter input{font:inherit;font-size:14px;color:var(--ink);background:var(--surface);border:1px solid var(--control-border);border-radius:8px;padding:6px 10px;width:100%;max-width:340px;min-width:0}",
    ".system-filter input:focus-visible{outline:2px solid var(--focus);outline-offset:1px}",
    ".system-filter .system-count{font-size:12.5px;color:var(--ink-3)}",
    ".system-gsecs{display:flex;flex-direction:column;gap:8px}",
    ".system-gsec[hidden]{display:none}",
    ".system-gsec > summary{font-size:14px}",
    ".system-entry + .system-entry{margin-top:12px}",
    ".system-entry[hidden]{display:none}",
    ".system-linkbtn{font:inherit;font-size:12.5px;color:var(--ink-2);background:none;border:0;padding:0;border-bottom:1px solid var(--border);cursor:pointer}",
    ".system-linkbtn:hover{color:var(--ink);border-bottom-color:var(--ink-2)}",
    ".system-linkbtn:focus-visible{outline:2px solid var(--focus);outline-offset:2px;border-radius:2px}",
    ".system-linkbtn[disabled]{color:var(--ink-3);cursor:default;border-bottom-color:transparent}",
    // Programs: a table on wide screens, a card list under 600px
    ".system-ocards{display:none;list-style:none;margin:0;padding:0;flex-direction:column;gap:10px}",
    ".system-ocard{border:1px solid var(--border);border-radius:var(--radius-sm,8px);padding:10px 12px;display:flex;flex-direction:column;gap:6px;min-width:0}",
    ".system-ocard-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 8px}",
    ".system-ocard-head code{overflow-wrap:anywhere;white-space:normal}",
    ".system-ocard p{margin:0;color:var(--ink-2);font-size:13px}",
    ".system-md{font-size:13.5px;color:var(--ink);line-height:1.55;min-width:0}",
    ".system-md > * + *{margin-top:8px}",
    ".system-md p{margin:0}",
    ".system-md h4{font-size:13.5px;font-weight:600;margin:0}",
    ".system-md h4 code{font-size:12.5px}",
    ".system-md code{font-family:var(--mono);font-size:12px;background:var(--surface-2);border-radius:4px;padding:0 4px;overflow-wrap:anywhere}",
    ".system-md pre.fl-code{font-size:11.5px;max-width:100%}",
    ".system-md blockquote{margin:0;padding:4px 0 4px 12px;border-left:2px solid var(--axis);color:var(--ink-2)}",
    ".system-md ul,.system-md ol{margin:0;padding-left:20px}",
    ".system-md .system-mdtable{overflow-x:auto;max-width:100%}",
    ".system-md table{border-collapse:collapse;font-size:12.5px;min-width:100%}",
    ".system-md th,.system-md td{text-align:left;vertical-align:top;padding:5px 10px 5px 0;border-bottom:1px solid var(--grid)}",
    ".system-md th{color:var(--ink-2);font-weight:600;border-bottom-color:var(--axis)}",
    ".system-md td:first-child{white-space:nowrap}",
    ".system-mark{background:var(--surface-2);background:color-mix(in srgb,var(--focus) 22%,transparent);color:inherit;border-radius:2px;padding:0 1px}",
    ".system-docs a{font-weight:500}",
    ".system-tag{display:inline-block;font-size:11px;color:var(--ink-2);border:1px solid var(--border);border-radius:4px;padding:0 5px;margin-left:6px;vertical-align:1px}",
    // one command option (--flag) or path part stays on one line
    ".system-nb{white-space:nowrap}",
    ".system-writes{min-width:18ch}",
    ".system-wnote{max-width:60ch}",
    ".system-pick{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;margin:4px 0 8px}",
    ".system-pick label{font-size:13px;color:var(--ink-2)}",
    // phones: a stacked table shows its main fields only (the rest is on a wide screen)
    "@media (max-width:599.98px){.fl-table--stack td.system-ph,.fl-table--stack th.system-ph{display:none}}",
    "@media (max-width:599.98px){.system-io{grid-template-columns:minmax(0,1fr)}.system-io dt{margin-top:2px}.system-md td:first-child{white-space:normal}" +
      ".system-programs-table{display:none}.system-ocards{display:flex}}"
  ].join("\n");

  function system_style() {
    if (document.getElementById("system-style")) return;
    var s = document.createElement("style");
    s.id = "system-style";
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  /* ------------------------------------------------------------------ small helpers */
  function arr(v) { return Array.isArray(v) ? v : []; }
  function obj(v) { return v && typeof v === "object" && !Array.isArray(v) ? v : null; }
  function txt(v, dflt) { return v === null || v === undefined || v === "" ? (dflt === undefined ? FL.fmt.NA : dflt) : String(v); }
  function dateCell(v) { return v ? String(v) : DASH; }
  function plural(n, one, many) { return FL.fmt.int(n) + " " + (n === 1 ? one : many); }
  function isNum(v) { return typeof v === "number" && isFinite(v); }

  // Relative links only (exporter builds "../<percent-encoded repo path>"); anything else is dropped.
  function system_safeHref(h) {
    if (typeof h !== "string") return null;
    if (/^\.\.\/[A-Za-z0-9%._~\-\/]+$/.test(h) && h.indexOf("..", 3) < 0) return h;
    return null;
  }

  /* ------------------------------------------------------------------ system_md: Markdown -> DOM (safe) */
  // Highlight `hl` (lower-case) inside a text run by wrapping matches in <mark>.
  function system_text(parent, s, hl) {
    if (!s) return;
    if (!hl) { parent.appendChild(document.createTextNode(s)); return; }
    var low = s.toLowerCase(), i = 0, j;
    while ((j = low.indexOf(hl, i)) >= 0) {
      if (j > i) parent.appendChild(document.createTextNode(s.slice(i, j)));
      var m = document.createElement("mark");
      m.className = "system-mark";
      m.textContent = s.slice(j, j + hl.length);
      parent.appendChild(m);
      i = j + hl.length;
    }
    if (i < s.length) parent.appendChild(document.createTextNode(s.slice(i)));
  }

  var INLINE_RE = /(`[^`]+`)|(\*\*(?:[^*]|\*(?!\*))+?\*\*)|(\*[^*\s](?:[^*]*?[^*\s])?\*)|(\[[^\]]+\]\([^)\s]+\))/g;
  function system_inline(parent, s, hl) {
    s = String(s === null || s === undefined ? "" : s);
    var last = 0, m;
    var re = new RegExp(INLINE_RE.source, "g"); // per call: the recursion below must not share lastIndex
    while ((m = re.exec(s)) !== null) {
      if (m.index > last) system_text(parent, s.slice(last, m.index), hl);
      var tok = m[0], node;
      if (m[1]) {
        node = document.createElement("code");
        system_text(node, tok.slice(1, -1), hl);
      } else if (m[2]) {
        node = document.createElement("strong");
        system_inline(node, tok.slice(2, -2), hl);
      } else if (m[3]) {
        node = document.createElement("em");
        system_inline(node, tok.slice(1, -1), hl);
      } else {
        var lm = /^\[([^\]]+)\]\(([^)\s]+)\)$/.exec(tok);
        var href = lm && (/^https?:\/\//.test(lm[2]) || /^#\//.test(lm[2])) ? lm[2] : null;
        node = document.createElement(href ? "a" : "span");
        if (href) { node.setAttribute("href", href); if (/^https?:/.test(href)) node.setAttribute("rel", "noopener noreferrer"); }
        system_inline(node, lm ? lm[1] : tok, hl);
      }
      parent.appendChild(node);
      last = m.index + tok.length;
    }
    if (last < s.length) system_text(parent, s.slice(last), hl);
    return parent;
  }

  // Split a table row on | outside backtick spans.
  function system_cells(line) {
    var t = line.trim();
    if (t.charAt(0) === "|") t = t.slice(1);
    if (t.charAt(t.length - 1) === "|") t = t.slice(0, -1);
    var out = [], cur = "", code = false;
    for (var i = 0; i < t.length; i++) {
      var ch = t.charAt(i);
      if (ch === "`") code = !code;
      if (ch === "|" && !code) { out.push(cur.trim()); cur = ""; continue; }
      cur += ch;
    }
    out.push(cur.trim());
    return out;
  }

  /**
   * Render Markdown into `parent`. opts: {hl: lower-case highlight string, rowFilter: fn(rowText)->bool}
   * Supported: # headings (rendered as h5), paragraphs, - / * / 1. lists, > quotes, ``` fences,
   * | tables |, inline `code`, **bold**, *italic*, [text](https://...) links.
   */
  function system_md(parent, md, opts) {
    opts = opts || {};
    var hl = opts.hl || "";
    var lines = String(md || "").replace(/\r\n?/g, "\n").split("\n");
    var i = 0;
    function isTable(l) { return /^\s*\|/.test(l); }
    function isQuote(l) { return /^\s*>/.test(l); }
    function isList(l) { return /^\s*([-*]|\d+\.)\s+/.test(l); }
    function isFence(l) { return /^\s*```/.test(l); }
    function isHead(l) { return /^#{1,6}\s/.test(l); }
    while (i < lines.length) {
      var line = lines[i];
      if (!line.trim()) { i++; continue; }
      if (isFence(line)) {
        var code = [];
        i++;
        while (i < lines.length && !isFence(lines[i])) { code.push(lines[i]); i++; }
        i++;
        var pre = document.createElement("pre");
        pre.className = "fl-code";
        system_text(pre, code.join("\n"), hl);
        parent.appendChild(pre);
        continue;
      }
      if (isHead(line)) {
        var h = document.createElement("h4");
        system_inline(h, line.replace(/^#{1,6}\s+/, ""), hl);
        parent.appendChild(h);
        i++;
        continue;
      }
      if (isTable(line)) {
        var rows = [];
        while (i < lines.length && isTable(lines[i])) { rows.push(lines[i]); i++; }
        var head = system_cells(rows[0]);
        var body = rows.slice(1).filter(function (r) { return !/^\s*\|?\s*:?-{2,}/.test(r.trim()); });
        var wrap = document.createElement("div");
        wrap.className = "system-mdtable";
        var table = document.createElement("table"), thead = document.createElement("thead"), tr = document.createElement("tr");
        head.forEach(function (c) { var th = document.createElement("th"); th.setAttribute("scope", "col"); system_inline(th, c, hl); tr.appendChild(th); });
        thead.appendChild(tr);
        table.appendChild(thead);
        var tbody = document.createElement("tbody");
        // a row filter only narrows a table when at least one row matches; otherwise show it whole
        var keep = opts.rowFilter ? body.filter(function (r) { return opts.rowFilter(r); }) : body;
        if (!keep.length) keep = body;
        keep.forEach(function (r) {
          var row = document.createElement("tr");
          system_cells(r).forEach(function (c) { var td = document.createElement("td"); system_inline(td, c, hl); row.appendChild(td); });
          tbody.appendChild(row);
        });
        table.appendChild(tbody);
        wrap.appendChild(table);
        parent.appendChild(wrap);
        continue;
      }
      if (isQuote(line)) {
        var q = [];
        while (i < lines.length && isQuote(lines[i])) { q.push(lines[i].replace(/^\s*>\s?/, "")); i++; }
        var bq = document.createElement("blockquote");
        system_md(bq, q.join("\n"), opts);
        parent.appendChild(bq);
        continue;
      }
      if (isList(line)) {
        var ordered = /^\s*\d+\./.test(line);
        var list = document.createElement(ordered ? "ol" : "ul");
        var li = null;
        while (i < lines.length && lines[i].trim() && (isList(lines[i]) || (li && /^\s+/.test(lines[i])))) {
          if (isList(lines[i])) {
            li = document.createElement("li");
            li._src = lines[i].replace(/^\s*([-*]|\d+\.)\s+/, "");
            list.appendChild(li);
          } else li._src += " " + lines[i].trim();
          i++;
        }
        Array.prototype.forEach.call(list.children, function (item) { system_inline(item, item._src, hl); item._src = null; });
        parent.appendChild(list);
        continue;
      }
      var para = [];
      while (i < lines.length && lines[i].trim() && !isFence(lines[i]) && !isHead(lines[i]) && !isTable(lines[i]) && !isQuote(lines[i]) && !isList(lines[i])) {
        para.push(lines[i].trim());
        i++;
      }
      var p = document.createElement("p");
      system_inline(p, para.join(" "), hl);
      parent.appendChild(p);
    }
    return parent;
  }

  // Plain text of a Markdown string (for filtering): drop markup characters, keep underscores.
  function system_plain(md) { return String(md || "").replace(/[`*>#|]/g, " ").replace(/\s+/g, " ").toLowerCase(); }

  /* ------------------------------------------------------------------ system_chips */
  // A path with a line-break chance after each "/" only (never inside a file name).
  function system_path(parent, s) {
    String(s).split("/").forEach(function (part, i, all) {
      parent.appendChild(document.createTextNode(part + (i < all.length - 1 ? "/" : "")));
      if (i < all.length - 1 && all[i + 1] !== "") parent.appendChild(document.createElement("wbr"));
    });
    return parent;
  }
  // Text in which each command option ("--flag", "--source binance") stays on one line.
  function system_flags(parent, s) {
    var re = /--[a-z][a-z-]*(?: binance)?/g, last = 0, m;
    s = String(s === null || s === undefined ? "" : s);
    while ((m = re.exec(s)) !== null) {
      if (m.index > last) parent.appendChild(document.createTextNode(s.slice(last, m.index)));
      parent.appendChild(FL.ui.el("span", { "class": "system-nb", text: m[0] }));
      last = m.index + m[0].length;
    }
    if (last < s.length) parent.appendChild(document.createTextNode(s.slice(last)));
    return parent;
  }
  function system_chips(items, ext) {
    var box = FL.ui.el("div", { "class": "system-chips" });
    arr(items).forEach(function (s) {
      var external = ext || !/^(data\/|engine\/)/.test(String(s));
      var chip = FL.ui.el("span", { "class": ["system-chip", external ? "is-ext" : null] });
      if (external) chip.textContent = String(s); else system_path(chip, s);
      box.appendChild(chip);
    });
    if (!box.firstChild) box.appendChild(FL.ui.el("span", { "class": "muted small", text: DASH }));
    return box;
  }

  /* ------------------------------------------------------------------ render */
  FL.pages.register({
    id: "system",
    title: "System",
    icon: "system",
    subtitle: "How the engine runs and how to check it. The last build, the update steps, the data, the rules, the tests and the glossary.",
    render: function (root, ctx) {
      var ui = FL.ui, el = ui.el, fmt = FL.fmt, charts = FL.charts;
      var d = ctx.section || {};
      var meta = ctx.meta || {};
      var nav = ctx.nav && ctx.nav.href ? ctx.nav : { href: function (pg, c) { return "#/" + pg + (c ? "/" + c : ""); } };
      system_style();

      var snap = obj(d.snapshot) || {};
      var upd = obj(d.update);
      var programs = arr(d.programs);
      var inv = obj(d.inventory);
      var rules = obj(d.rules);
      var tests = obj(d.tests);
      var gloss = obj(d.glossary);
      var docs = arr(d.docs);
      var env = arr(d.environment);
      var logs = obj(snap.logs) || {};
      var cov = obj(d.coverage);

      function missing(card, what) {
        ui.empty(card.body, (what || "This part") + " is not in this export.", { commands: ui.remedies ? ui.remedies() : [{ command: "cd engine && .venv/bin/python -B export_web.py" }] });
      }

      /* ---------- the last build (update.py, from engine/logs/update.log) */
      // the data-through date of the export (meta.js): the newest daily perp bar of any coin
      var through = meta.data_through || snap.data_through;
      var build = obj(snap.build) || {};
      var lastRun = obj(build.last_run);
      var BUILD_CMDS = "./setup.sh                                                # download the new days, then build\n" +
        "engine/.venv/bin/python -B engine/update.py --offline        # build from the data on disk\n" +
        "engine/.venv/bin/python -B engine/update.py --from \"STEP\"    # start at a step (a name from --plan)";
      // a step with the result "done (new alarm)" did its work: it counts as ok, with an alarm note
      var alarmSteps = lastRun ? arr(lastRun.alarm_steps) : [];
      var ALARM_NOTE = "A new alarm is not a failure. Read it with judge.py status.";
      function alarmText() {
        return alarmSteps.length ? plural(alarmSteps.length, "step", "steps") + " recorded a new alarm (" + alarmSteps.join(", ") + ")" : "";
      }
      function runText(r) {
        if (!r) return FL.fmt.NA;
        if (r.failed_step) return "Stopped at " + r.failed_step;
        return r.n_steps + (r.n_steps === 1 ? " step" : " steps") + ", all ok" + (alarmSteps.length ? "; " + alarmText() : "");
      }
      var notes = [];
      if (lastRun) {
        notes.push("The last build (update.py) ended " + fmt.datetime(lastRun.finished) + ": " + runText(lastRun).replace(/^\S/, function (c) { return c.toLowerCase(); }) + ". The data goes to the " + txt(through) + " UTC close.");
        if (alarmSteps.length) notes.push(ALARM_NOTE);
      } else {
        notes.push("No build is recorded in " + txt(build.file, "engine/logs/update.log") + ". The data goes to the " + txt(through) + " UTC close.");
      }
      notes.push("This page is a static export. It describes the engine from its code and the files on disk. Nothing ran to make this page.");
      ui.callout(lastRun && lastRun.failed_step ? "warning" : "info", lastRun && lastRun.failed_step ? "The last build stopped at a failed step" : "How this data was built", notes, { container: root });

      /* ---------- KPI tiles */
      var et = tests && obj(tests.engine_totals);
      var tot = inv && obj(inv.totals);
      ui.kpis(root, [
        { label: "Data through", value: through ? fmt.date(through) : fmt.NA, note: "UTC close; daily bars are stamped with their open time", id: "system-kpi-through" },
        { label: "Last build (update.py)", value: lastRun ? fmt.datetime(lastRun.finished) : "None recorded",
          badge: lastRun ? (lastRun.failed_step ? { status: "warning", label: "stopped at " + lastRun.failed_step } : { status: "good", label: plural(lastRun.n_steps, "step", "steps") + " ok" }) : { status: "neutral", label: "no build log" },
          note: lastRun ? (alarmSteps.length ? plural(alarmSteps.length, "new alarm", "new alarms") + ", " : "") + "from " + txt(build.file) : "run ./setup.sh", id: "system-kpi-build" },
        { label: "Engine test suites", value: et ? fmt.int(et.suites) : fmt.NA, note: et ? fmt.int(et.tests) + " test functions, parsed, not run" : null },
        { label: "Files in engine/data", value: tot ? fmt.int(tot.files) : fmt.NA, note: tot ? fmt.bytes(tot.bytes) + " in " + plural(tot.groups, "group", "groups") : null },
        // the glossary's "entries" are its filterable blocks (a term, a paragraph or a table), not a count of
        // defined terms, so the tile counts sections and names the blocks in the note
        { label: "Glossary", value: gloss ? plural(arr(gloss.sections).length, "section", "sections") : fmt.NA, note: gloss ? fmt.int(gloss.n_entries) + " filterable entries from GLOSSARY.md" : null }
      ]);

      /* ---------- table of contents */
      var tocItems = [["build", "Last build"], ["update", "Update steps"], ["programs", "Programs"], ["environment", "Environment"], ["data-age", "Data age"], ["inventory", "Data inventory"],
        ["coverage", "Data coverage"], ["invariants", "Invariants"], ["rules", "Rules"], ["tests", "Tests"], ["glossary", "Glossary"], ["docs", "Documents"], ["export", "This export"]];
      if (ui.toc) ui.toc(root, tocItems, { page: "system", href: function (id) { return nav.href("system", id); } });
      else {
        var toc = el("ul");
        tocItems.forEach(function (t) { toc.appendChild(el("li", null, el("a", { attrs: { href: nav.href("system", t[0]) }, text: t[1] }))); });
        root.appendChild(el("nav", { "class": "fl-toc", attrs: { "aria-label": "On this page" } }, toc));
      }

      /* ================================================================ how it runs */
      var sRun = ui.section(root, { id: "how-it-runs", title: "How it runs", subtitle: "./setup.sh runs engine/update.py. It downloads the new days and builds every output, one step at a time." });

      var cBuild = ui.card({ id: "build", title: "Last build (update.py)", subtitle: "From " + txt(build.file, "engine/logs/update.log") + ": one line for each step. A run is a series of steps without a pause; a failed step ends the run." }, sRun.body);
      if (!build.present || !lastRun) {
        ui.note(cBuild.body, build.present ? "The build log has no step line." : "No build log in this checkout. ./setup.sh writes it.");
      } else {
        var bl = [
          ["Started", fmt.datetime(lastRun.started)],
          ["Ended", fmt.datetime(lastRun.finished)],
          ["Steps", fmt.int(lastRun.n_steps) + " (" + fmt.int(lastRun.n_ok) + " ok), " + txt(lastRun.first_step) + " to " + txt(lastRun.last_step)],
          ["Result", lastRun.failed_step ? ui.badge("warning", "stopped at " + lastRun.failed_step + (lastRun.failed_exit !== null && lastRun.failed_exit !== undefined ? " (exit " + lastRun.failed_exit + ")" : "")) : ui.badge("good", "all steps ok")],
          ["Data through", txt(through) + " UTC close"]
        ];
        ui.dl(cBuild.body, bl);
        var stepOk = function (alarm) { return alarm ? ui.badge("advisory", "done (new alarm)") : ui.badge("good", "ok"); };
        if (lastRun.failed_step) ui.note(cBuild.body, "Correct the cause, then run update.py again with --from \"" + lastRun.failed_step + "\" to start at that step.");
        if (alarmSteps.length) ui.note(cBuild.body, alarmText() + ". " + ALARM_NOTE);
        ui.details(cBuild.body, {
          summary: "Each step: its last result",
          meta: plural(arr(build.steps).length, "step", "steps"),
          render: function (body) {
            ui.table(body, {
              caption: "The last result of each step in the build log, and when it last ended without a failure (UTC). A run can start at a step (--from), so the times can differ.",
              columns: [
                { key: "step", label: "Step", rowHeader: true },
                { key: "last_ok", label: "Last result", render: function (v, r) { return v ? stepOk(r.last_alarm) : ui.badge("warning", "failed"); }, sortable: false },
                { key: "last_ok_at", label: "Last ok", render: function (v) { return v ? fmt.datetime(v) : DASH; }, className: "nowrap" }
              ],
              rows: arr(build.steps), compact: true, stack: true
            });
          }
        });
        ui.details(cBuild.body, {
          summary: "Steps of the last run",
          meta: plural(arr(lastRun.steps).length, "step", "steps"),
          render: function (body) {
            ui.table(body, {
              caption: "Each step of the last run, in order (UTC).",
              columns: [
                { key: "step", label: "Step", rowHeader: true },
                { key: "ok", label: "Result", render: function (v, r) { return v ? stepOk(r.alarm) : ui.badge("warning", "failed (exit " + txt(r.exit) + ")"); }, sortable: false },
                { key: "secs", label: "Seconds", format: "int" },
                { key: "at", label: "Ended", render: function (v) { return v ? fmt.datetime(v) : DASH; }, className: "nowrap" }
              ],
              rows: arr(lastRun.steps), sortable: false, compact: true, stack: true
            });
          }
        });
      }
      cBuild.body.appendChild(el("h4", { text: "Build again" }));
      cBuild.body.appendChild(ui.code(BUILD_CMDS, { block: true }));

      var steps = upd ? arr(upd.steps) : [];
      var drawSteps = null;
      var viewSeg = steps.length ? ui.segmented(null, { options: [{ value: "list", label: "Steps" }, { value: "table", label: "Table" }], value: "list", ariaLabel: "Update steps view",
        onChange: function (v) { if (drawSteps) drawSteps(v); } }) : null;
      var cUpd = ui.card({ id: "update", title: "Update steps (update.py) " + DASH + " " + plural(steps.length, "step", "steps"),
        subtitle: "The plan of " + (upd && upd.mode === "demo" ? "./setup.sh --demo (the demo data)" : "./setup.sh (market data)") + " for this data folder, in order. update.py runs one step at a time and stops at the first failure. The plan is computed for this page; nothing ran.",
        actions: viewSeg ? [viewSeg.root] : null }, sRun.body);
      if (!steps.length) missing(cUpd, "The update steps");
      else {
        var stepsWrap = el("div");
        cUpd.body.appendChild(stepsWrap);
        drawSteps = function (view) {
          ui.clear(stepsWrap);
          if (view === "table") {
            ui.table(stepsWrap, {
              columns: [
                { key: "n", label: "#", format: "int", align: "right", width: 36 },
                { key: "name", label: "Step", wrap: true },
                { key: "command", label: "Command", mono: true, render: function (v, r) { return txt(v) + " (in " + txt(r.folder) + "/)"; } },
                { key: "reads", label: "Reads", wrap: true, render: function (v) { return arr(v).join(", ") || DASH; } },
                { key: "writes", label: "Writes", wrap: true, render: function (v) { return arr(v).join(", ") || DASH; } },
                { key: "when", label: "Runs", wrap: true, render: function (v) { return txt(v, DASH); } }
              ],
              rows: steps, sortable: false, compact: true, stack: true, caption: "The same steps as a table."
            });
            return;
          }
          var ol = el("ol", { "class": "system-steps", attrs: { "aria-label": "The update steps in order" } });
          steps.forEach(function (st) {
            var li = el("li", { "class": ["system-step", st.described ? null : "is-plain"] });
            li.appendChild(el("div", { "class": "system-step-n", text: String(st.n), attrs: { "aria-hidden": "true" } }));
            var body = el("div", { "class": "system-step-body" });
            var head = el("div", { "class": "system-step-head" }, [
              el("span", { "class": "sr-only", text: "Step " + st.n + ": " }),
              el("span", { "class": "system-step-title", text: st.name }),
              ui.code(st.command)
            ]);
            if (st.network) head.appendChild(ui.badge("info", "network"));
            if (arr(st.ok_codes).length > 1) head.appendChild(ui.badge("neutral", "done with exit " + arr(st.ok_codes).join(" or ")));
            body.appendChild(head);
            if (st.what) body.appendChild(el("p", { "class": "system-step-what", text: st.what }));
            if (st.when) body.appendChild(el("p", { "class": "system-step-what", text: "Runs: " + st.when + "." }));
            if (st.described) {
              var io = el("dl", { "class": "system-io" });
              io.appendChild(el("dt", { text: "reads" }));
              io.appendChild(el("dd", null, system_chips(st.reads)));
              io.appendChild(el("dt", { text: "writes" }));
              io.appendChild(el("dd", null, system_chips(st.writes)));
              body.appendChild(io);
            }
            li.appendChild(body);
            ol.appendChild(li);
          });
          stepsWrap.appendChild(ol);
        };
        drawSteps("list");
        var sw = obj(upd.switches);
        if (sw) ui.note(cUpd.body, "Switches in engine/config.py: " + Object.keys(sw).map(function (k) { return k + " = " + (sw[k] ? "True" : "False"); }).join(", ") + ".");
        ui.note(cUpd.body, "Options of update.py: --offline (no download), --demo, --skip-binance, --skip-library, --from STEP, --plan (show the plan only). Source: plan() in engine/update.py.");
      }

      var cOut = ui.card({ id: "programs", title: "Programs", subtitle: "The programs in engine/: what each one writes, and when update.py runs it." }, sRun.body);
      if (!programs.length) missing(cOut);
      else {
        // wide screens: a table; under 600px (CSS) the same rows as a card list
        ui.table(cOut.body, { ariaLabel: "Programs",
          className: "system-programs-table",
          columns: [
            { key: "script", label: "Program", mono: true, rowHeader: true },
            { key: "writes", label: "Writes", render: function (v) { return system_chips(v); }, sortable: false, className: "system-writes" },
            { key: "cadence", label: "Cadence (update.py)", wrap: true, render: function (v) { return system_flags(el("span"), v); } },
            { key: "note", label: "Note", wrap: true, className: "system-wnote", render: function (v) { return system_flags(el("span"), v); } }
          ],
          rows: programs, compact: true
        });
        var oc = el("ul", { "class": "system-ocards", attrs: { "aria-label": "Programs" } });
        programs.forEach(function (w) {
          var io = el("dl", { "class": "system-io" }, [el("dt", { text: "writes" }), el("dd", null, system_chips(w.writes))]);
          oc.appendChild(el("li", { "class": "system-ocard" }, [
            el("div", { "class": "system-ocard-head" }, [ui.code(txt(w.script, DASH)), el("span", { "class": "system-tag", text: txt(w.cadence, DASH) })]),
            io,
            w.note ? system_flags(el("p"), w.note) : null
          ]));
        });
        cOut.body.appendChild(oc);
        ui.note(cOut.body, "You can also run each program by hand (engine/README.md gives the options).");
      }

      /* ================================================================ isolation */
      var sIso = ui.section(root, { id: "isolation", title: "Environment and isolation", subtitle: "Each engine checkout keeps its own logs, lock and data folder. Two checkouts must not share them." });
      var cEnv = ui.card({ id: "environment", title: "Environment variables", subtitle: "Defaults are read from the code (os.environ.get in the files listed). Nothing in the engine needs keys." }, sIso.body);
      if (!env.length) missing(cEnv);
      else {
        ui.table(cEnv.body, { ariaLabel: "Environment variables",
          columns: [
            { key: "name", label: "Variable", mono: true },
            { key: "default", label: "Default", mono: true },
            { key: "meaning", label: "Meaning", wrap: true },
            { key: "guard", label: "Guard", wrap: true },
            { key: "read_in", label: "Read in", render: function (v) { return arr(v).join(", "); }, sortable: false, className: "system-ph" }
          ],
          rows: env, sortable: false, stack: true
        });
        ui.note(cEnv.body, "Example: FACTORLAB_DATA=<folder> runs a program on a copy of the data folder (engine/README.md).");
      }

      /* ================================================================ data inventory */
      var sData = ui.section(root, { id: "data-files", title: "What lives in engine/data", subtitle: "Computed by the exporter from file metadata and each table's date column; dates are UTC. Rows: the rows of each table, and their sum for a group. This sum measures size, not days." });
      /* ---------- data age (meta.data_age, from export_web.py): a stalled copy or a stalled coin shows here */
      var age = obj(meta.data_age);
      var cAge = ui.card({ id: "data-age", title: "Data age", subtitle: "The age of the data at the exporter run. Days behind: the number of daily closes (00:00 UTC) after the data close, up to the date of the exporter run; 0 means current. The data-through date is the newest bar of any coin. A stall of one coin does not change it, so this card also gives the last bar of each coin." }, sData.body);
      if (!age || !age.export_date) missing(cAge, "The data age");
      else {
        var lateCoins = arr(age.behind_newest);
        var lagDays = typeof age.days_behind === "number" ? age.days_behind : null;
        ui.dl(cAge.body, [
          ["Exporter run", age.export_date + " (UTC date)"],
          ["Data through", txt(age.data_through) + " UTC close (the newest daily perp bar of any coin)"],
          ["Days behind", lagDays === null ? FL.fmt.NA : fmt.int(lagDays) + (lagDays <= 0 ? " (current)" : "")],
          ["Newest coin bar", txt(age.newest_bar)],
          ["Oldest coin bar", txt(age.oldest_bar)]
        ]);
        if (lateCoins.length) {
          ui.callout("warning", plural(lateCoins.length, "coin stops", "coins stop") + " before the newest bar (" + txt(age.newest_bar) + ")",
            [arr(age.coins).filter(function (c) { return lateCoins.indexOf(c.symbol) >= 0; }).map(function (c) { return c.symbol + " (last bar " + dateCell(c.last_bar) + ")"; }).join(", ") + ".",
              "A stall of one coin does not change the data-through date. Find the cause before you use the new days."], { container: cAge.body });
        } else {
          ui.note(cAge.body, "All " + fmt.int(age.symbols) + " coins have a last bar on " + txt(age.newest_bar) + "." +
            (lagDays !== null && lagDays >= 1 ? " That is " + plural(lagDays, "day", "days") + " behind the exporter run." : ""));
        }
        if (lagDays !== null && lagDays >= 1) {
          ui.note(cAge.body, "To get the new days, download and build the data again (./setup.sh).");
        }
        // the status compares each coin with the newest coin only; the age of the data is the line above
        var drawCoins = function (host) {
          ui.table(host, {
            caption: "The last perp bar of each coin (data/raw/ohlcv, UTC open date).",
            columns: [
              { key: "symbol", label: "Coin", mono: true, rowHeader: true },
              { key: "last_bar", label: "Last bar", render: dateCell, className: "num nowrap" },
              { key: "days_behind", label: "Days behind", format: "int" },
              { key: "late", label: "Against the newest coin", render: function (v) { return v ? ui.badge("warning", "behind the newest bar") : ui.badge("neutral", "same as the newest coin"); }, sortable: false }
            ],
            rows: arr(age.coins).map(function (c) { return { symbol: c.symbol, last_bar: c.last_bar, days_behind: c.days_behind, late: lateCoins.indexOf(c.symbol) >= 0 }; }),
            compact: true, maxHeight: 320, stack: true
          });
        };
        if (lateCoins.length) drawCoins(cAge.body);
        else ui.details(cAge.body, { summary: "Last bar of each coin", meta: plural(arr(age.coins).length, "coin", "coins"), render: drawCoins });
      }

      var cInv = ui.card({ id: "inventory", title: "Data inventory", subtitle: inv && tot ? fmt.int(tot.files) + " files, " + fmt.bytes(tot.bytes) + " under " + txt(inv.root, "engine/data") + ". Newest raw bar: " + txt(inv.newest_raw_bar) + "." : null }, sData.body);
      if (!inv || !arr(inv.groups).length) missing(cInv, "The data inventory");
      else {
        var allBytes = arr(inv.groups).reduce(function (a, g) { return a + (typeof g.bytes === "number" ? g.bytes : 0); }, 0);
        var share = function (b) { return allBytes > 0 && typeof b === "number" ? b / allBytes : null; };
        // the checkout's own engine/logs: a row of its own, outside the data folder and its shares
        var groupRows = arr(inv.groups).concat(obj(inv.logs) ? [inv.logs] : []);
        ui.table(cInv.body, {
          caption: "One row per group. Choose a group below to see its files.",
          columns: [
            { key: "id", label: "Group", mono: true, rowHeader: true },
            { key: "label", label: "Contents", wrap: true },
            { key: "files", label: "Files", format: "int", className: "system-ph" },
            { key: "rows", label: "Rows", format: "int", title: "Sum over the group: the rows of each table", className: "system-ph" },
            { key: "first", label: "First date", render: dateCell, className: "num nowrap system-ph" },
            { key: "last", label: "Last date", render: dateCell, className: "num nowrap" },
            { key: "bytes", label: "Size", format: "bytes", align: "right" },
            // the share of all bytes: one group can hold most of the disk space, so the size alone hides the rest
            { key: "share", label: "Share", value: function (r) { return r.outside_data ? null : share(r.bytes); }, format: "pct:2:unsigned", align: "right", title: "The share of the bytes of all groups in the data folder", className: "system-ph" },
            { key: "writers", label: "Written by", render: function (v) { return arr(v).join(", "); }, sortable: false, wrap: true }
          ],
          rows: groupRows,
          stickyFirst: true, compact: true, stack: true,   // stickyFirst applies to the wide table only (see the CSS)
          rowClass: function (r) { return r.present === false ? "is-muted" : null; }
        });
        // one group's files at a time (a select), not one accordion per group
        var withFiles = groupRows.filter(function (g) { return g.present !== false; });
        if (withFiles.length) {
          var pickId = ui.uid("system-group");
          var sel = el("select", { id: pickId, "class": "fl-select" });
          withFiles.forEach(function (g) { sel.appendChild(el("option", { attrs: { value: g.id }, text: (g.path || g.id) + " (" + plural(g.files, "file", "files") + ")" })); });
          cInv.body.appendChild(el("div", { "class": "system-pick" }, [el("label", { attrs: { "for": pickId }, text: "Files of the group" }), sel]));
          var filesBox = el("div", { "class": "stack-sm" });
          cInv.body.appendChild(filesBox);
          var drawGroup = function (id) {
            var g = withFiles.filter(function (x) { return x.id === id; })[0] || withFiles[0];
            ui.clear(filesBox);
            if (g.note) filesBox.appendChild(el("p", { "class": "fl-note", text: g.note }));
            if (!arr(g.items).length) { filesBox.appendChild(el("p", { "class": "muted small", text: "No files." })); return; }
            ui.table(filesBox, {
              caption: "The files of " + (g.path || g.id) + ".",
              columns: [
                { key: "name", label: g.symbols ? "Symbol file" : "File or folder", mono: true, rowHeader: true },
                { key: "rows", label: "Rows", format: "int", title: "The rows of a table; for a folder, the sum of the rows of its files" },
                { key: "first", label: "First", render: dateCell, className: "num nowrap system-ph" },
                { key: "last", label: "Last", render: dateCell, className: "num nowrap" },
                { key: "bytes", label: "Size", format: "bytes", align: "right" },
                { key: "writer", label: "Written by", mono: true },
                { key: "cadence", label: "Cadence", wrap: true, render: function (v) { return system_flags(el("span"), v); } }
              ],
              rows: arr(g.items), compact: true, stickyFirst: true, maxHeight: 420, stack: true
            });
          };
          sel.addEventListener("change", function () { drawGroup(sel.value); });
          drawGroup(withFiles[0].id);
        }
        ui.note(cInv.body, "Cadence: when update.py writes the file. The values are “each update”, “first update of each UTC day” and “each update (when a registration exists)”. “By hand” means that you run the command.");
        var tops = arr(inv.top_files).filter(function (f) { return f.present; });
        if (tops.length) ui.note(cInv.body, "Files in the data folder itself: " + tops.map(function (f) { return f.path + " (" + f.writer + ", " + fmt.bytes(f.bytes) + ")"; }).join(", ") + ".");
        // a last date after the data close is not new market data: say why, per group
        var late = through ? arr(inv.groups).filter(function (g) { return g.last && g.last > through; }) : [];
        if (late.length) {
          ui.note(cInv.body, "Last dates after the " + through + " close: " + late.map(function (g) {
            return g.id + " " + g.last + " (" + (g.after_close || "no explanation recorded in the export") + ")";
          }).join("; ") + ".");
        }
        if (arr(inv.other).length) ui.note(cInv.body, "Other content in the data folder (in no group of this card): " + arr(inv.other).join(", ") + ".");
      }

      /* ---------- data coverage: the raw datasets of each coin */
      var dsets = cov ? arr(cov.datasets) : [];
      var cCov = ui.card({ id: "coverage", title: "Data coverage", subtitle: "Rows and date span of each raw dataset and of the label table, for each coin. Timestamps are UTC." }, sData.body);
      if (!dsets.length) missing(cCov, "The data coverage");
      else {
        var cadence = function (h) { return !isNum(h) ? DASH : h % 24 === 0 ? (h === 24 ? "1 day" : fmt.int(h / 24) + " days") : fmt.num(h, h === Math.round(h) ? 0 : 2) + " h"; };
        ui.table(cCov.body, {
          caption: "One row for each dataset. Cadence: the most common step between rows; other steps: the steps of another length, summed over the coins.",
          columns: [
            { key: "label", label: "Dataset", rowHeader: true, wrap: true, render: function (v, r) {
              return el("span", null, [el("span", { text: txt(v) }), el("span", { "class": "muted small", text: " " + txt(r.path) })]);
            } },
            { key: "n_symbols", label: "Coins", format: "int" },
            { key: "first", label: "First", render: dateCell, className: "num nowrap system-ph" },
            { key: "last", label: "Last", render: dateCell, className: "num nowrap" },
            { key: "rows", label: "Rows", format: "int" },
            { key: "step", label: "Cadence", align: "right", value: function (r) { var st = arr(r.symbols).map(function (x) { return x.step_h; }).filter(isNum); return st.length ? st[0] : null; },
              render: function (v) { return cadence(v); }, className: "system-ph" },
            { key: "off", label: "Other steps", align: "right", format: "int", className: "system-ph",
              value: function (r) { return arr(r.symbols).some(function (x) { return isNum(x.off_step); }) ? arr(r.symbols).reduce(function (a, x) { return a + (isNum(x.off_step) ? x.off_step : 0); }, 0) : null; } }
          ],
          rows: dsets, sortable: false, compact: true, stack: true
        });
        var bySym = {};
        dsets.forEach(function (x) { arr(x.symbols).forEach(function (c) { (bySym[c.symbol] = bySym[c.symbol] || {})[x.id] = c; }); });
        var symRows = Object.keys(bySym).sort().map(function (k) { return { symbol: k, e: bySym[k] }; });
        ui.details(cCov.body, { summary: "Rows of each coin", meta: plural(symRows.length, "coin", "coins"), render: function (host) {
          var cols = [{ key: "symbol", label: "Coin", mono: true, rowHeader: true }];
          dsets.forEach(function (x) {
            cols.push({ key: x.id, label: x.label, align: "right", value: function (r) { var c = r.e[x.id]; return c ? c.rows : null; },
              render: function (v, r) { var c = r.e[x.id]; return c ? el("span", { title: txt(c.first) + " to " + txt(c.last), text: fmt.int(c.rows) }) : DASH; } });
          });
          ui.table(host, { caption: "The rows of each dataset for each coin; the title of a cell gives the first and the last date.", columns: cols, rows: symRows, compact: true, maxHeight: 420, stack: true });
        } });
      }

      /* ================================================================ rules */
      var sRules = ui.section(root, { id: "trust", title: "Rules", subtitle: "The rules of the data and of the test of a factor." });
      var gRules = ui.grid(sRules.body, 2);
      var cInvar = ui.card({ id: "invariants", title: "Invariants of the data" }, gRules);
      if (!rules || !arr(rules.invariants).length) missing(cInvar);
      else {
        var il = el("ul", { "class": "system-list" });
        arr(rules.invariants).forEach(function (r) { il.appendChild(system_inline(el("li"), r.text)); });
        cInvar.body.appendChild(il);
      }

      var cRules = ui.card({ id: "rules", title: "Rules of the test of a factor", subtitle: "Why each rule exists." }, gRules);
      if (!rules || !arr(rules.rules).length) missing(cRules);
      else {
        var rw = el("div");
        arr(rules.rules).forEach(function (r) {
          var item = el("article", { "class": "system-rule", id: "rule-" + r.id });
          item.appendChild(el("h4", { text: r.title }));
          if (r.why) item.appendChild(el("p", { text: r.why }));
          arr(r.sources).forEach(function (s) {
            var q = el("blockquote", { "class": "system-quote" });
            q.appendChild(system_inline(el("p"), s.quote));
            q.appendChild(el("footer", null, el("span", { "class": "system-src", text: s.file })));
            item.appendChild(q);
          });
          rw.appendChild(item);
        });
        cRules.body.appendChild(rw);
      }

      /* ================================================================ tests */
      var sTest = ui.section(root, { id: "testing", title: "Integrity tests", subtitle: "Script-style suites that need no pytest and no network. The counts come from the files; this dashboard does not run the tests." });
      var cTest = ui.card({ id: "tests", title: "Engine test suites", subtitle: et ? et.suites + " suites, " + et.tests + " test functions (module-level test_* in engine/test_*.py, without the tests of the dashboard exporter)." : null }, sTest.body);
      if (!tests || !arr(tests.engine).length) missing(cTest, "The test inventory");
      else {
        var modes = obj(tests.modes) || {};
        if (tests.run_all) {
          cTest.body.appendChild(el("h4", { text: "Run them (in engine/)" }));
          cTest.body.appendChild(ui.code(tests.run_all, { block: true }));
        }
        // one table; the run command is the same for each suite (--only NAME, above)
        ui.table(cTest.body, {
          caption: "Each engine suite. NAME in the commands above is the suite name.",
          columns: [
            { key: "suite", label: "Suite", mono: true, rowHeader: true },
            { key: "component", label: "Protects", wrap: true },
            { key: "n_tests", label: "Tests", format: "int" },
            { key: "mode", label: "Data", wrap: true, render: function (v, r) { return el("span", { text: txt(v) + (r.scratch_copy ? " (on a copy)" : ""), title: modes[v] || null }); } }
          ],
          rows: arr(tests.engine), stickyFirst: true, compact: true, stack: true
        });
        var md = el("dl", { "class": "fl-dl" });
        Object.keys(modes).forEach(function (k) { md.appendChild(el("dt", { text: k })); md.appendChild(el("dd", { text: modes[k] })); });
        cTest.body.appendChild(md);
        // the names of the tests: one closed block, not one block for each suite
        ui.details(cTest.body, {
          summary: "The test functions of each suite",
          meta: plural(et ? et.tests : 0, "test", "tests"),
          render: function (body) {
            arr(tests.engine).forEach(function (s) {
              body.appendChild(el("h4", { text: s.file + " (" + plural(s.n_tests, "test", "tests") + ")" }));
              if (s.summary) body.appendChild(el("p", { "class": "fl-note", text: s.summary }));
              var ul = el("ul", { "class": "system-list" });
              arr(s.tests).forEach(function (t) {
                var li = el("li", null, [ui.code(t.name)]);
                if (t.doc) li.appendChild(el("span", { "class": "ink-2", text: " " + DASH + " " + t.doc }));
                ul.appendChild(li);
              });
              body.appendChild(ul);
            });
          }
        });
      }

      var gTest = ui.grid(sTest.body, 2);
      var cEx = ui.card({ id: "exporter-tests", title: "Dashboard exporter tests", subtitle: "engine/test_export_*.py: one script-style suite for each dashboard page, and one for the exporter. They check the exported data, not the engine. Run one with --only NAME." }, gTest);
      var ex = tests ? arr(tests.exporter) : [];
      if (!ex.length) ui.note(cEx.body, "No exporter test files were present when this section was exported.");
      else {
        ui.table(cEx.body, { ariaLabel: "Dashboard exporter tests",
          columns: [{ key: "suite", label: "Suite", mono: true, rowHeader: true }, { key: "component", label: "Checks", wrap: true, render: function (v) { return txt(v, DASH); } },
            { key: "n_tests", label: "Tests", format: "int" }],
          rows: ex, compact: true, stack: true
        });
      }
      var cVend = ui.card({ id: "alphaforge-tests", title: "AlphaForge suite", subtitle: "Not part of the script-style set." }, gTest);
      var vt = tests && obj(tests.alphaforge);
      if (!vt) missing(cVend);
      else {
        ui.dl(cVend.body, [
          { term: "Location", value: txt(vt.path), mono: true },
          ["Files", fmt.int(vt.files)],
          ["Test functions", fmt.int(vt.test_functions) + " (pytest parametrization not expanded)"],
          { term: "Run", value: txt(vt.command), mono: true },
          ["Needs", txt(vt.needs)]
        ]);
      }

      /* ================================================================ glossary */
      var sRef = ui.section(root, { id: "reference", title: "Reference" });
      // the card title is "Glossary"; the document's own subtitle follows it, and the entry count shows once (next to the filter)
      var cG = ui.card({ id: "glossary", title: "Glossary",
        subtitle: gloss ? (gloss.subtitle ? gloss.subtitle + ". " : "") + "From " + txt(gloss.source) + "." +
          (gloss.examples_date ? " The example outputs in the glossary are from " + fmt.date(gloss.examples_date) + "." : "") : null }, sRef.body);
      if (!gloss || !arr(gloss.sections).length) missing(cG, "The glossary");
      else {
        var fid = ui.uid("system-filter");
        var input = el("input", { id: fid, "class": "fl-input", attrs: { type: "search", placeholder: "e.g. funding, Sharpe, gross", autocomplete: "off", spellcheck: "false", "aria-describedby": fid + "-count" } });
        var count = el("span", { "class": "system-count", id: fid + "-count", attrs: { "aria-live": "polite" } });
        cG.body.appendChild(el("div", { "class": "system-filter", attrs: { role: "search" } }, [el("label", { attrs: { "for": fid }, text: "Filter the glossary" }), input, count]));

        // Each section is a collapsed <details> so the card stays short; the summaries double as the
        // section list. The filter opens every section with a match and hides the rest; clearing it
        // restores what the reader had open. #/system/glossary-N opens that section (on load and on a
        // same-page hash change).
        var tools = el("div", { "class": "row", style: { gap: "14px", flexWrap: "wrap" } });
        var btnOpen = el("button", { "class": "system-linkbtn", attrs: { type: "button" }, text: "Expand all" });
        var btnClose = el("button", { "class": "system-linkbtn", attrs: { type: "button" }, text: "Collapse all" });
        tools.appendChild(btnOpen);
        tools.appendChild(btnClose);
        cG.body.appendChild(tools);

        var introBox = gloss.intro ? system_md(el("div", { "class": "system-md" }), gloss.intro) : null;
        if (introBox) cG.body.appendChild(introBox);

        var secWrap = el("div", { "class": "system-gsecs" });
        cG.body.appendChild(secWrap);
        var secNodes = [], byId = new Map();
        var total = 0;
        var filtering = false;
        arr(gloss.sections).forEach(function (s) {
          var title = txt(s.title, "");
          var items = arr(s.entries);
          var startOpen = !!(nav.card && nav.card === s.id);
          var d = ui.details(secWrap, { id: s.id || null, className: "system-gsec", summary: (s.num ? s.num + ". " : "") + title,
            meta: plural(items.length, "entry", "entries"), open: startOpen });
          var metaEl = d.root.querySelector(".fl-details-meta");
          var entries = items.map(function (e) {
            total += 1;
            var node = el("div", { "class": "system-entry system-md" });
            d.body.appendChild(node);
            return { e: e, node: node, plain: system_plain(e.md) + " " + String(e.term || "").toLowerCase() };
          });
          var sn = { s: s, d: d, metaEl: metaEl, n: items.length, title: title.toLowerCase(), entries: entries, userOpen: startOpen };
          // remember what the reader opens or closes, but not the filter's own opening
          d.root.addEventListener("toggle", function () { if (!filtering) sn.userOpen = d.root.open; });
          secNodes.push(sn);
          if (s.id) byId.set(s.id, sn);
        });
        var empty = el("p", { "class": "muted", text: "" });
        empty.hidden = true;
        cG.body.appendChild(empty);

        var system_setAll = function (open) {
          secNodes.forEach(function (sn) { if (!sn.d.root.hidden) { sn.d.root.open = open; if (!filtering) sn.userOpen = open; } });
        };
        btnOpen.addEventListener("click", function () { system_setAll(true); });
        btnClose.addEventListener("click", function () { system_setAll(false); });

        var system_filter = function (q) {
          q = String(q || "").trim().toLowerCase();
          filtering = !!q;
          var shown = 0;
          secNodes.forEach(function (sn) {
            var titleHit = q && sn.title.indexOf(q) >= 0;
            var hits = 0;
            sn.entries.forEach(function (en) {
              var hit = !q || titleHit || en.plain.indexOf(q) >= 0;
              en.node.hidden = !hit;
              ui.clear(en.node);
              if (hit) {
                hits += 1; shown += 1;
                system_md(en.node, en.e.md, {
                  hl: q,
                  rowFilter: q && !titleHit ? function (row) { return system_plain(row).indexOf(q) >= 0 || system_plain(en.e.md.split("\n")[0]).indexOf(q) >= 0; } : null
                });
              }
            });
            sn.d.root.hidden = !!q && !hits;
            sn.d.root.open = q ? hits > 0 : sn.userOpen;
            if (sn.metaEl) sn.metaEl.textContent = q ? hits + " of " + plural(sn.n, "entry", "entries") + " match" : plural(sn.n, "entry", "entries");
          });
          if (introBox) introBox.hidden = !!q;
          btnOpen.disabled = btnClose.disabled = !!q;
          count.textContent = q ? shown + " of " + total + " entries match" : total + " entries in " + plural(secNodes.length, "section", "sections");
          empty.hidden = !(q && shown === 0);
          empty.textContent = q && shown === 0 ? "No glossary entry matches “" + q + "”." : "";
        };
        var timer = null;
        input.addEventListener("input", function () {
          if (timer) clearTimeout(timer);
          timer = setTimeout(function () { system_filter(input.value); }, 120);
        });
        input.addEventListener("keydown", function (e) { if (e.key === "Escape") { input.value = ""; system_filter(""); } });
        system_filter("");

        // same-page jumps to #/system/glossary-N: open the section before the router scrolls to it
        // (the router scrolls on the next animation frame). ctx.listen removes the listener when the
        // page goes; without the shell, the listener removes itself once this page's DOM is gone.
        var system_onHash = function () {
          if (!document.body.contains(cG.root)) { window.removeEventListener("hashchange", system_onHash); return; }
          var m = /^#\/system\/([^\/]+)$/.exec(window.location.hash || "");
          if (!m) return;
          var id = m[1];
          try { id = decodeURIComponent(id); } catch (err) { /* keep the raw id */ }
          // the id comes from the address bar: a Map has no inherited keys ("__proto__" is not a section)
          var sn = byId.has(id) ? byId.get(id) : null;
          if (sn && !sn.d.root.hidden) { sn.d.root.open = true; if (!filtering) sn.userOpen = true; }
        };
        if (ctx.listen) ctx.listen(window, "hashchange", system_onHash);
        else window.addEventListener("hashchange", system_onHash);
      }

      /* ================================================================ documents */
      var cDocs = ui.card({ id: "docs", title: "Documents", subtitle: "The documents of the repository. The links are relative to web/: they open the files in a checkout of the repository, not on a host that has only web/." }, sRef.body);
      if (!docs.length) missing(cDocs, "The document index");
      else {
        ui.table(cDocs.body, { ariaLabel: "Documents",
          className: "system-docs",
          columns: [
            { key: "title", label: "Title", wrap: true, render: function (v, r) {
              var href = system_safeHref(r.href);
              var a = href ? el("a", { attrs: { href: href }, text: txt(v, r.path) }) : el("span", { text: txt(v, r.path) });
              var box = el("span", null, [a]);
              if (r.lang === "zh") box.appendChild(el("span", { "class": "system-tag", text: "Chinese", attrs: { lang: "en" } }));
              return box;
            } },
            { key: "description", label: "What it is", wrap: true, render: function (v) { return txt(v, DASH); } },
            { key: "path", label: "Path", mono: true, wrap: true, className: "system-ph" },
            { key: "group", label: "Group", className: "system-ph" },
            { key: "date", label: "Dated", render: dateCell, className: "num nowrap" },
            { key: "bytes", label: "Size", format: "bytes", align: "right", className: "system-ph" }
          ].filter(function (c) { return c.key !== "date" || docs.some(function (r) { return !!r.date; }); }),   // no column of dashes
          // long rows: cap the frame (sticky header) instead of growing the page
          rows: docs, compact: true, maxHeight: 520, stack: true
        });
      }

      /* ================================================================ this export */
      var cExp = ui.card({ id: "export", title: "This export", subtitle: "How the numbers on this dashboard were produced." }, sRef.body);
      var secs = obj(meta.sections);
      ui.dl(cExp.body, [
        ["Version", (meta.version || snap.version) ? String(meta.version || snap.version) + " (the file VERSION)" : FL.fmt.NA],
        ["Exported", meta.generated_at ? fmt.datetime(meta.generated_at) : FL.fmt.NA],
        ["Data through", meta.data_through ? meta.data_through + " UTC close" : FL.fmt.NA],
        { term: "Rebuild the pages", value: FL.cmd ? FL.cmd.export : "cd engine && .venv/bin/python -B export_web.py", mono: true },
        ["Guarantees", "The exporter reads only the files of the step \"dashboard\" in the card \"Update steps\". It writes only its output folder (web/data, or the folder of --out). It refuses a path inside engine/data. It fails on a private string or on a file that is too large. The output is the same for the same data, except the export time and the data age."]
      ]);
      if (secs) {
        var budget = obj(meta.budget) || {};
        var secRows = Object.keys(secs).map(function (k) { return { id: k, present: !!secs[k].present, bytes: secs[k].bytes }; });
        ui.table(cExp.body, {
          caption: "Section data files" + (budget.section_max_bytes ? " (budget " + fmt.int(budget.section_max_bytes) + " bytes each, " + fmt.int(budget.total_max_bytes) + " bytes in total)" : ""),
          columns: [
            { key: "id", label: "Section", mono: true, rowHeader: true },
            { key: "present", label: "File", render: function (v) { return v ? "present" : "missing"; } },
            { key: "bytes", label: "Size", format: "bytes", align: "right" }
          ].filter(function (c) { return c.key !== "present" || secRows.some(function (r) { return !r.present; }); }),   // only when a file is missing
          rows: secRows,
          sortable: false, compact: true, stack: true,
          rowClass: function (r) { return r.present ? null : "is-muted"; }
        });
      }
    }
  });
})();
