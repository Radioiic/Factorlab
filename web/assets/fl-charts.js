/* Factorlab dashboard — FL.charts: a tiny dependency-free SVG chart kit.
 *
 * line · bars · heatmap · sparkline · smallMultiples
 * Every chart: responsive (ResizeObserver, debounced), focusable with keyboard
 * crosshair/cell navigation, a "Table" toggle that swaps in FL.ui.table of the
 * same numbers, theme-aware (re-renders on "fl:themechange"), never dual-axis.
 * Mark specs: 2px lines (round), bars <= 24px with a 4px rounded data-end and a
 * 2px surface gap, markers r >= 4 with a 2px surface ring, 10% area wash,
 * solid hairline grid/axes; dashed only for the BTC benchmark and reference lines.
 * API reference: web/README.md.
 */
(function () {
  "use strict";

  var FL = window.FL;
  if (!FL || !FL.ui) { if (window.console) console.error("fl-charts.js needs fl-core.js first"); return; }
  var ui = FL.ui, fmt = FL.fmt, colors = FL.colors, el = ui.el, S = ui.svg, toNum = FL.util.toNum, assign = FL.util.assign;
  var DAY = 864e5;
  var ISO_RE = /^\d{4}-\d{2}-\d{2}$/;
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  /* ====================================================================== */
  /* measuring, scales, ticks                                                */
  /* ====================================================================== */

  var mctx = null;
  function textWidth(str, size, weight) {
    if (mctx === null) {
      try { mctx = document.createElement("canvas").getContext("2d") || false; } catch (e) { mctx = false; }
    }
    if (!mctx) return String(str).length * (size || 11) * 0.58;
    mctx.font = (weight || 400) + " " + (size || 11) + "px system-ui, -apple-system, \"Segoe UI\", sans-serif";
    return mctx.measureText(String(str)).width;
  }
  function truncate(str, maxW, size, weight) {
    str = String(str);
    if (textWidth(str, size, weight) <= maxW) return str;
    var lo = 0, hi = str.length;
    while (lo < hi) {
      var mid = Math.ceil((lo + hi) / 2);
      if (textWidth(str.slice(0, mid) + "…", size, weight) <= maxW) lo = mid; else hi = mid - 1;
    }
    return lo > 0 ? str.slice(0, lo) + "…" : "";
  }
  function linear(d0, d1, r0, r1) {
    var k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
    var f = function (v) { return r0 + (v - d0) * k; };
    f.invert = function (p) { return k === 0 ? d0 : d0 + (p - r0) / k; };
    return f;
  }
  function logScale(d0, d1, r0, r1) {
    var l0 = Math.log(d0), l1 = Math.log(d1), k = l1 === l0 ? 0 : (r1 - r0) / (l1 - l0);
    var f = function (v) { return v > 0 ? r0 + (Math.log(v) - l0) * k : r0; };
    f.invert = function (p) { return Math.exp(l0 + (p - r0) / k); };
    return f;
  }
  function niceStep(raw) {
    if (!(raw > 0)) return 1;
    var pow = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / pow;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * pow;
  }
  function decimalsOf(step) {
    for (var d = 0; d < 10; d++) {
      var s = step * Math.pow(10, d);
      if (Math.abs(Math.round(s) - s) < 1e-7 * Math.max(1, Math.abs(s))) return d;
    }
    return 10;
  }
  function roundTo(v, step) { var d = decimalsOf(step); return Number(v.toFixed(Math.min(12, d + 2))); }
  /** nice linear ticks covering [lo, hi]; fixed=true keeps lo/hi as the domain */
  function niceTicks(lo, hi, count, fixed) {
    if (lo === hi) { var pad = Math.abs(lo) * 0.1 || 1; lo -= pad; hi += pad; }
    if (lo > hi) { var t = lo; lo = hi; hi = t; }
    var step = niceStep((hi - lo) / Math.max(1, count));
    var t0 = roundTo(Math.floor(lo / step + 1e-9) * step, step), t1 = roundTo(Math.ceil(hi / step - 1e-9) * step, step);
    var dlo = fixed ? lo : t0, dhi = fixed ? hi : t1;
    var ticks = [];
    for (var v = t0, i = 0; v <= t1 + step * 1e-6 && i < 200; v += step, i++) {
      var r = roundTo(v, step);
      if (r >= dlo - step * 1e-6 && r <= dhi + step * 1e-6) ticks.push(Math.abs(r) < step * 1e-9 ? 0 : r);
    }
    return { lo: dlo, hi: dhi, step: step, ticks: ticks };
  }
  /** doubling ticks around a base (×0.25 ×0.5 ×1 ×2 ×4); falls back to ×4 steps on long ranges */
  function logTicks(lo, hi, base, fixed) {
    base = base > 0 ? base : 1;
    lo = lo > 0 ? lo : base / 2; hi = hi > 0 ? hi : base * 2;
    if (lo === hi) { lo /= 1.5; hi *= 1.5; }
    var k0 = Math.floor(Math.log2(lo / base) + 1e-9), k1 = Math.ceil(Math.log2(hi / base) - 1e-9);
    var stepK = 1;
    while ((k1 - k0) / stepK > 8) stepK *= 2;
    k0 = Math.floor(k0 / stepK) * stepK; k1 = Math.ceil(k1 / stepK) * stepK;
    var ticks = [];
    for (var k = k0; k <= k1; k += stepK) ticks.push(base * Math.pow(2, k));
    var dlo = fixed ? lo : ticks[0], dhi = fixed ? hi : ticks[ticks.length - 1];
    if (fixed) ticks = ticks.filter(function (t) { return t >= lo * (1 - 1e-9) && t <= hi * (1 + 1e-9); });
    return { lo: dlo, hi: dhi, step: null, ticks: ticks, log: true };
  }
  /** formatter for axis ticks: clean, unsigned, decimals derived from the step */
  function tickFormatter(spec, step, isLog) {
    if (typeof spec === "function") return spec;
    var kind = fmt.kindOf(spec) || (isLog ? "mult" : "num");
    if (isLog) {
      if (kind === "mult" || kind === "num") return function (v) { return fmt.mult(v); };
      if (kind === "idx") return function (v) { return String(Number(v.toPrecision(3))); };
      if (kind === "usd") return function (v) { return fmt.usd(v, v < 10 ? 2 : 0); };
      if (kind === "x") return function (v) { return Number(v.toPrecision(3)) + "x"; };
    }
    var u = { signed: false };
    switch (kind) {
      case "pct": return function (v) { return fmt.pct(v, decimalsOf(step * 100), u); };
      case "pp": return function (v) { return fmt.pp(v, decimalsOf(step * 100), u); };
      case "bps": return function (v) { return fmt.bps(v, decimalsOf(step * 1e4), u); };
      case "usd": return function (v) { return fmt.usd(v, decimalsOf(step)); };
      case "usdc": return function (v) { return fmt.compact(v, 1, { prefix: "$" }); };
      case "compact": return function (v) { return fmt.compact(v, 1); };
      case "int": return function (v) { return fmt.int(v); };
      case "x": return function (v) { return fmt.num(v, decimalsOf(step)) + "x"; };
      case "mult": return function (v) { return fmt.mult(v); };
      default: return function (v) { return fmt.num(v, decimalsOf(step)); };
    }
  }
  /** UTC time ticks: days, Mondays, months, years — labels "Aug 3", "Aug", year at year boundaries */
  function timeTicks(d0, d1, maxTicks) {
    maxTicks = Math.max(2, maxTicks);
    var span = Math.max(0, d1 - d0);
    var cands = [["d", 1], ["d", 2], ["d", 3], ["w", 1], ["w", 2], ["w", 3], ["w", 4], ["m", 1], ["m", 2], ["m", 3], ["m", 6], ["y", 1], ["y", 2], ["y", 5], ["y", 10]];
    var dur = { d: DAY, w: 7 * DAY, m: 30.44 * DAY, y: 365.25 * DAY };
    var pick = cands[cands.length - 1], ticks = null;
    for (var i = 0; i < cands.length; i++) {
      if (span / (dur[cands[i][0]] * cands[i][1]) > maxTicks * 3) continue; // cheap pre-filter
      var g = genTimeTicks(d0, d1, cands[i][0], cands[i][1]);
      if (g.length <= maxTicks) { pick = cands[i]; ticks = g; break; }
    }
    var u = pick[0];
    var out = ticks || genTimeTicks(d0, d1, pick[0], pick[1]);
    var prevYear = null;
    return out.map(function (ms, idx) {
      var d = new Date(ms), yr = d.getUTCFullYear(), mo = d.getUTCMonth(), label;
      if (u === "y") label = String(yr);
      else if (u === "m") label = mo === 0 ? String(yr) : MONTHS[mo];
      else {
        label = MONTHS[mo] + " " + d.getUTCDate();
        if (idx > 0 && prevYear !== null && yr !== prevYear) label += ", " + yr;
      }
      prevYear = yr;
      return { v: ms, label: label };
    });
  }
  function genTimeTicks(d0, d1, u, k) {
    var out = [], t, dt;
    if (u === "d") {
      t = Math.ceil(d0 / DAY / k) * k * DAY;
      for (; t <= d1; t += k * DAY) out.push(t);
    } else if (u === "w") {
      dt = new Date(d0);
      t = Date.UTC(dt.getUTCFullYear(), dt.getUTCMonth(), dt.getUTCDate());
      if (t < d0) t += DAY;
      while (new Date(t).getUTCDay() !== 1) t += DAY;
      for (; t <= d1; t += 7 * k * DAY) out.push(t);
    } else if (u === "m") {
      dt = new Date(d0);
      var y = dt.getUTCFullYear(), m = dt.getUTCMonth();
      for (var n = 0; n < 2000; n++) {
        t = Date.UTC(y, m, 1);
        if (t > d1) break;
        if (t >= d0 && m % k === 0) out.push(t);
        m += 1; if (m > 11) { m = 0; y += 1; }
      }
    } else {
      var y0 = new Date(d0).getUTCFullYear(), y1 = new Date(d1).getUTCFullYear();
      for (var yy = Math.ceil(y0 / k) * k; yy <= y1; yy += k) { t = Date.UTC(yy, 0, 1); if (t >= d0 && t <= d1) out.push(t); }
    }
    return out;
  }
  function px(v) { return Math.round(v) + 0.5; } // crisp hairlines

  /* ---------------------------------------------------------------------- */
  /* label placement: annotation labels (reference lines, bands, vlines,     */
  /* markers) are collision-checked against each other, kept inside bounds, */
  /* and drawn in a layer ABOVE the series (with a surface halo).            */
  /* ---------------------------------------------------------------------- */
  /** box [x0, y0, x1, y1] of a text whose BASELINE is y */
  function textBox(x, y, w, anchor, size) {
    var h = size || 11;
    var x0 = anchor === "end" ? x - w : anchor === "middle" ? x - w / 2 : x;
    return [x0, y - h * 0.75, x0 + w, y + h * 0.2];
  }
  /** true when two boxes overlap; pad is extra horizontal clearance (text needs air between words) */
  function boxesOverlap(a, b, pad) {
    pad = pad || 0;
    return !(a[2] + pad <= b[0] || b[2] + pad <= a[0] || a[3] <= b[1] || b[3] <= a[1]);
  }
  /**
   * makeLabeler(bounds) -> {place(text, cands, o), reserve(box)}
   * cands: [{x, y (baseline), anchor, minX, maxX}] tried in order; the first one inside its
   * bounds that does not collide wins. o.force (default true): when none fits, the first
   * candidate is used anyway, truncated to its bounds. Returns {x, y, anchor, text} or null.
   */
  function makeLabeler(bounds) {
    var boxes = [];
    function inside(b, c) {
      var minX = c.minX !== undefined ? c.minX : bounds[0], maxX = c.maxX !== undefined ? c.maxX : bounds[2];
      return b[0] >= minX - 0.5 && b[2] <= maxX + 0.5 && b[1] >= bounds[1] - 0.5 && b[3] <= bounds[3] + 0.5;
    }
    return {
      boxes: boxes,
      reserve: function (b) { boxes.push(b); },
      place: function (text, cands, o) {
        o = o || {};
        var size = o.size || 11, w = textWidth(text, size, o.weight);
        for (var i = 0; i < cands.length; i++) {
          var c = cands[i], b = textBox(c.x, c.y, w, c.anchor, size);
          if (!inside(b, c)) continue;
          if (boxes.some(function (q) { return boxesOverlap(b, q, 2); })) continue;
          boxes.push(b);
          return { x: c.x, y: c.y, anchor: c.anchor, text: text };
        }
        if (o.force === false || !cands.length) return null;
        var f = cands[0];
        var lo = f.minX !== undefined ? f.minX : bounds[0], hi = f.maxX !== undefined ? f.maxX : bounds[2];
        var room = f.anchor === "end" ? f.x - lo : f.anchor === "middle" ? 2 * Math.min(f.x - lo, hi - f.x) : hi - f.x;
        var t = truncate(text, Math.max(0, room), size, o.weight);
        if (!t) return null;
        boxes.push(textBox(f.x, f.y, textWidth(t, size, o.weight), f.anchor, size));
        return { x: f.x, y: f.y, anchor: f.anchor, text: t };
      }
    };
  }
  /** candidates for a label centred on x (flipping to start/end near the edges of [minX, maxX]) */
  function hCands(x, y, minX, maxX, gap) {
    gap = gap === undefined ? 4 : gap;
    return [
      { x: x, y: y, anchor: "middle", minX: minX, maxX: maxX },
      { x: x + gap, y: y, anchor: "start", minX: minX, maxX: maxX },
      { x: x - gap, y: y, anchor: "end", minX: minX, maxX: maxX }
    ];
  }
  /** x-axis tick labels: centred, clamped to [0, W], and skipped when they would touch the previous label */
  function drawXTicks(g, ticks, xScale, y, W, o) {
    o = o || {};
    var prevEnd = -Infinity;
    ticks.forEach(function (t) {
      var x = xScale(t.v), w = textWidth(t.label, 11), anchor = "middle", tx = x;
      if (x - w / 2 < 0) { anchor = "start"; tx = Math.max(0, x - 4); }
      else if (x + w / 2 > W) { anchor = "end"; tx = Math.min(W, x + 4); }
      var x0 = anchor === "start" ? tx : anchor === "end" ? tx - w : tx - w / 2;
      if (o.tickMarks) g.appendChild(S("line", { x1: px(x), x2: px(x), y1: o.tickMarks[0], y2: o.tickMarks[1], "class": "axis" }));
      if (x0 < prevEnd + 8) return; // would collide: the tick keeps its mark, the label is dropped
      prevEnd = x0 + w;
      g.appendChild(textEl(tx, y, t.label, "tick", anchor));
    });
  }
  /** axis title in the band above the plot, truncated to the chart width */
  function axisTitle(svg, label, x, maxW) {
    if (!label) return;
    var t = truncate(label, Math.max(0, maxW), 11);
    if (t) svg.appendChild(textEl(x, 12, t, "lbl", "start", t !== String(label) ? { title: String(label) } : null));
  }
  function drawPlaced(g, p, cls, extra) {
    if (p) g.appendChild(textEl(p.x, p.y, p.text, cls || "lbl halo", p.anchor, extra));
  }

  function bisectNearest(arr, v) { // arr sorted asc
    var lo = 0, hi = arr.length - 1;
    if (hi < 0) return -1;
    while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (arr[mid] <= v) lo = mid; else hi = mid; }
    return Math.abs(arr[hi] - v) < Math.abs(arr[lo] - v) ? hi : lo;
  }

  /* ====================================================================== */
  /* chart frame: header/legend/toggle, plot, table twin, tooltip, resize    */
  /* ====================================================================== */

  var live = [];
  var themeTimer = null;
  window.addEventListener("fl:themechange", function () {
    clearTimeout(themeTimer);
    themeTimer = setTimeout(function () { prune(); live.forEach(function (c) { c.render(); }); }, 0);
  });
  /**
   * Drop charts whose DOM was removed. A chart built into a container that is not in the
   * document (the page appends it after the build) is kept until it has been seen connected,
   * unless `all` is true (the router passes true on navigation: everything detached is gone).
   */
  function prune(all) {
    live = live.filter(function (c) {
      if (c.root.isConnected) { c._seen = true; return true; }
      if (!c._seen && all !== true) return true;
      c._disconnect();
      return false;
    });
  }
  // touch: a tooltip stays up after the finger lifts; a tap anywhere outside the plot hides it
  document.addEventListener("pointerdown", function (e) {
    live.forEach(function (c) {
      if (c.tip && !c.tip.hidden && !c.plot.contains(e.target) && c.nav && c.nav.blur) c.nav.blur();
    });
  }, true);

  function resolveContainer(c) { return typeof c === "string" ? document.querySelector(c) : c; }

  function createChart(container, opts, impl) {
    container = resolveContainer(container);
    if (!container) throw new Error("FL.charts." + impl.kind + ": container not found");
    var inst = { kind: impl.kind, opts: opts || {}, impl: impl, view: "chart", width: 0, height: 0, destroyed: false };
    var o = inst.opts;
    var root = el("figure", { "class": ["fl-chart", "fl-chart--" + impl.kind, o.className], id: o.id || null });
    var head = el("div", { "class": "fl-chart-head" });
    var plot = el("div", { "class": "fl-chart-plot", attrs: { tabindex: "0", role: "group", "aria-roledescription": "chart" } });
    var tableBox = el("div", { "class": "fl-chart-table" });
    tableBox.hidden = true;
    // one polite live region for the whole life of the chart: keyboard readouts are announced
    // from the first arrow key on (a region that gets aria-live late is not read)
    var liveRegion = el("div", { "class": "sr-only", attrs: { "aria-live": "polite", "aria-atomic": "true" } });
    root.appendChild(head);
    root.appendChild(plot);
    root.appendChild(tableBox);
    root.appendChild(liveRegion);
    inst.liveRegion = liveRegion;
    if (o.caption || o.note) root.appendChild(el("figcaption", { "class": "fl-chart-cap", text: o.caption || o.note }));
    inst.root = root; inst.head = head; inst.plot = plot; inst.tableBox = tableBox;

    var toggle = null;
    if (o.table !== false) {
      toggle = el("button", { "class": "fl-chart-toggle", attrs: { type: "button", "aria-pressed": "false" } });
      toggle.addEventListener("click", function () { inst.setView(inst.view === "chart" ? "table" : "chart"); });
    }
    function buildHead() {
      ui.clear(head);
      if (o.title) head.appendChild(el("div", { "class": "fl-chart-title", text: o.title, title: o.title }));
      var items = impl.legend ? impl.legend(inst) : null;
      if (items && items.length) head.appendChild(ui.legend(items, { context: o.context }));
      if (toggle) {
        // a fixed name ("Table") with aria-pressed: the state is said once, by the pressed state
        ui.clear(toggle);
        toggle.appendChild(ui.icon("table"));
        toggle.appendChild(el("span", { text: "Table" }));
        toggle.setAttribute("aria-pressed", inst.view === "table" ? "true" : "false");
        toggle.setAttribute("aria-label", "Table" + (o.title ? ": " + o.title : ""));
        head.appendChild(toggle);
      }
    }
    inst.buildHead = buildHead;

    inst.render = function () {
      if (inst.destroyed) return;
      if (root.isConnected) inst._seen = true;
      if (inst.view === "table") {
        if (!inst._tableBuilt) {
          ui.clear(tableBox);
          var t = impl.table ? impl.table(inst) : null;
          if (t) ui.table(tableBox, assign({ sortable: true, maxHeight: 360, stickyFirst: true, compact: true }, t));
          inst._tableBuilt = true;
        }
        return;
      }
      var w = Math.floor(root.clientWidth);
      if (!w) return;
      inst.width = w;
      ui.clear(plot);
      inst.tip = null;
      try {
        impl.draw(inst, w);
      } catch (err) {
        ui.clear(plot);
        plot.appendChild(el("div", { "class": "fl-chart-msg", text: "Chart failed to render: " + (err && err.message ? err.message : String(err)) }));
        if (window.console) console.error(err);
      }
      var summary = String(o.ariaLabel || (impl.summary ? impl.summary(inst) : "")).trim();
      if (summary && !/[.!?]$/.test(summary)) summary += ".";
      plot.setAttribute("aria-label", (o.title ? o.title + ". " : "") + (summary ? summary + " " : "") + "Use arrow keys to read values; the Table button shows every number.");
    };
    inst.setView = function (v) {
      inst.view = v === "table" ? "table" : "chart";
      plot.hidden = inst.view === "table";
      tableBox.hidden = inst.view !== "table";
      buildHead();
      inst.render();
    };
    inst.update = function (newOpts) {
      inst.opts = o = assign({}, o, newOpts || {});
      if (impl.prepare) impl.prepare(inst);
      inst._tableBuilt = false;
      buildHead();
      inst.render();
      return api;
    };
    var ro = null, timer = null;
    inst._disconnect = function () { if (ro) ro.disconnect(); ro = null; clearTimeout(timer); window.removeEventListener("resize", onWinResize); };
    inst.destroy = function () {
      inst.destroyed = true;
      inst._disconnect();
      live = live.filter(function (c) { return c !== inst; });
      if (root.parentNode) root.parentNode.removeChild(root);
    };
    function schedule() {
      clearTimeout(timer);
      timer = setTimeout(function () {
        if (!root.isConnected) { prune(); return; }
        var w = Math.floor(root.clientWidth);
        if (w && w !== inst.width && inst.view === "chart") inst.render();
      }, inst.width ? 120 : 0);
    }
    function onWinResize() { schedule(); }

    if (impl.prepare) impl.prepare(inst);
    buildHead();
    container.appendChild(root);
    if (typeof ResizeObserver !== "undefined") { ro = new ResizeObserver(schedule); ro.observe(root); }
    else window.addEventListener("resize", onWinResize);
    prune();
    live.push(inst);
    inst.render();
    setupKeys(inst);

    var api = {
      root: root, el: root,
      update: inst.update, destroy: inst.destroy,
      showTable: function (on) { inst.setView(on === false ? "chart" : "table"); },
      render: function () { inst.render(); },
      _inst: inst
    };
    return api;
  }

  // keyboard: arrows move the active index (impl.nav handles semantics)
  function setupKeys(inst) {
    var plot = inst.plot;
    plot.addEventListener("keydown", function (e) {
      if (!inst.nav) return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      inst._kbd = true;
      inst._touch = false;
      var handled = inst.nav.key(e.key, e.shiftKey);
      if (handled) {
        e.preventDefault();
        if (inst.liveRegion) inst.liveRegion.textContent = inst.tip && !inst.tip.hidden ? inst.tip.textContent : "";
      }
    });
    // capture phase: the pointer type is known before the hit layer's own handlers run
    function onPointer(e) { inst._kbd = false; inst._touch = e.pointerType === "touch" || e.pointerType === "pen"; }
    plot.addEventListener("pointermove", onPointer, true);
    plot.addEventListener("pointerdown", onPointer, true);
    plot.addEventListener("focus", function () { if (inst.nav && inst.nav.focus && !inst._touch) inst.nav.focus(); });
    plot.addEventListener("blur", function () { if (inst.nav && inst.nav.blur && !inst._touch) inst.nav.blur(); });
  }
  /**
   * Pointer wiring for a chart's transparent hit layer. Mouse/pen: follow the pointer, hide on
   * leave. Touch: a tap (or a horizontal drag, see .is-scrub) shows the readout and it stays
   * after the finger lifts; a vertical scroll (pointercancel) or a tap outside hides it.
   */
  function bindHover(inst, hit, move, hide) {
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", function (e) {
      if (inst._kbd || e.pointerType === "touch") return; // a lifted finger "leaves": keep the readout
      hide();
    });
    hit.addEventListener("pointercancel", function () { if (!inst._kbd) hide(); });
  }

  function makeSvg(inst, w, h) {
    var s = S("svg", { width: w, height: h, viewBox: "0 0 " + w + " " + h, "aria-hidden": "true", focusable: "false" });
    inst.plot.appendChild(s);
    inst.height = h;
    return s;
  }
  function ensureTip(inst) {
    if (!inst.tip) {
      inst.tip = el("div", { "class": "fl-tip" });
      inst.tip.hidden = true;
      inst.plot.appendChild(inst.tip);
    }
    return inst.tip;
  }
  /**
   * Place the tooltip near (x, y) in plot coordinates (x relative to the visible plot box).
   * Mouse/keyboard: beside the point, flipping left near the right edge, clamped inside the
   * plot. Touch: centred above the finger (below it when there is no room), so the finger
   * never covers the readout.
   */
  function showTip(inst, x, y, content) {
    var tip = ensureTip(inst);
    ui.clear(tip);
    tip.appendChild(content);
    tip.style.left = "0px"; tip.style.top = "0px"; // measure at a neutral spot (no edge squeeze)
    tip.hidden = false;
    var pw = inst.plot.clientWidth, ph = inst.height || inst.plot.clientHeight;
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    var left, top;
    if (inst._touch && !inst._kbd) {
      left = Math.max(0, Math.min(pw - tw, x - tw / 2));
      top = y - th - 24;
      if (top < 0) top = y + 24;
      if (top + th > ph) top = Math.max(0, Math.min(y - th - 24, ph - th));
    } else {
      left = x + 14;
      if (left + tw > pw) left = x - 14 - tw;
      if (left < 0) left = Math.max(0, Math.min(pw - tw, x - tw / 2));
      top = y - th / 2;
      top = Math.max(0, Math.min(Math.max(0, ph - th), top));
    }
    tip.style.left = Math.round(left + inst.plot.scrollLeft) + "px";
    tip.style.top = Math.round(top) + "px";
  }
  function hideTip(inst) { if (inst.tip) inst.tip.hidden = true; }
  function tipRow(color, kind, value, name, o) {
    o = o || {};
    return el("div", { "class": ["fl-tip-row", o.hot ? "is-hot" : null, o.total ? "is-total" : null, o.dim ? "is-dim" : null] }, [
      color ? ui.key(color, { kind: kind, dashed: o.dashed }) : el("span", { "class": "fl-key", style: { width: kind === "line" ? "16px" : "10px" } }),
      el("span", { "class": "v", text: value }),
      el("span", { "class": "n", text: name })
    ]);
  }
  /** pointer position in svg user units (the svg may be scaled down while a resize is pending) */
  function svgPoint(svg, e) {
    var r = svg.getBoundingClientRect(), w = +svg.getAttribute("width");
    var k = r.width > 0 && w > 0 ? w / r.width : 1;
    return { x: (e.clientX - r.left) * k, y: (e.clientY - r.top) * k };
  }
  function message(inst, text) {
    inst.plot.appendChild(el("div", { "class": "fl-chart-msg", text: text }));
  }
  function textEl(x, y, str, cls, anchor, extra) {
    return S("text", assign({ x: x, y: y, "class": cls || "tick", "text-anchor": anchor || "start", text: str }, extra || {}));
  }
  function seriesColor(s, i, context) {
    if (s.color) return s.color;
    var c = colors.entity(s.entity, context);
    return c || colors.slot(i + 1);
  }
  function seriesName(s, i) {
    if (s.name !== undefined && s.name !== null) return String(s.name);
    if (s.entity) return colors.label(s.entity);
    return s.id ? String(s.id) : "Series " + (i + 1);
  }

  /* ====================================================================== */
  /* line                                                                    */
  /* ====================================================================== */

  function prepareLine(inst) {
    var o = inst.opts, xType = o.xType || "time";
    var shared = o.x || null;
    var ys = o.y || {};
    var allX = {};
    inst.series = (o.series || []).map(function (s, i) {
      var info = colors.entityInfo(s.entity, o.context);
      var pts = [];
      (s.values || []).forEach(function (p, j) {
        var xv, yv;
        if (Array.isArray(p)) { xv = p[0]; yv = p[1]; }
        else if (p !== null && typeof p === "object") { xv = p.x !== undefined ? p.x : p.date; yv = p.y !== undefined ? p.y : p.value; }
        else { xv = shared ? shared[j] : j; yv = p; }
        var xm = xType === "time" ? fmt.parseDate(xv) : toNum(xv);
        if (xm === null) return;
        var v = toNum(yv);
        if (ys.log && v !== null && v <= 0) v = null;
        pts.push([xm, v, xv]);
      });
      pts.sort(function (a, b) { return a[0] - b[0]; });
      var map = new Map();
      pts.forEach(function (p) { map.set(p[0], p[1]); allX[p[0]] = p[2]; });
      var sw = toNum(s.width);
      return {
        id: s.id || "s" + i, name: seriesName(s, i), color: seriesColor(s, i, o.context),
        // the per-series stroke width survives preparation (drawLine reads s.width); null keeps the default
        width: sw !== null && sw > 0 ? sw : null,
        dashed: s.dashed !== undefined ? !!s.dashed : !!(info && info.dashed),
        area: s.area !== undefined ? !!s.area : !!o.area,
        step: s.step !== undefined ? !!s.step : !!o.step,
        pts: pts, map: map
      };
    });
    inst.xs = Object.keys(allX).map(Number).sort(function (a, b) { return a - b; });
    inst.xRaw = allX;
    inst.xType = xType;
    // optional display downsampling (time axis only): the chart and its hover use one point per
    // UTC week / month (the bucket's last value), the Table view keeps every point.
    inst.dxs = null;
    inst.series.forEach(function (s) { s.dpts = null; });
    var bucket = xType === "time" ? bucketFn(o.downsample) : null;
    if (bucket) {
      var keep = {};
      inst.series.forEach(function (s) {
        var out = [], cur = null, last = null, lastAny = null;
        function flush() {
          if (cur === null || lastAny === null) return;
          var p = last || lastAny; // a bucket with no value keeps its gap
          out.push(last ? p : [lastAny[0], null, lastAny[2]]);
          keep[p[0]] = true;
        }
        s.pts.forEach(function (p, i) {
          var b = bucket(p[0]);
          if (i === 0) { out.push(p); keep[p[0]] = true; cur = b; return; } // the first point always survives
          if (b !== cur) { flush(); cur = b; last = null; lastAny = null; }
          lastAny = p;
          if (p[1] !== null) last = p;
        });
        flush();
        s.dpts = out;
      });
      inst.dxs = Object.keys(keep).map(Number).sort(function (a, b) { return a - b; });
    }
    inst.tipFmt = fmt.get(ys.tooltipFormat || ys.format || "num", ys.signed !== undefined ? { signed: ys.signed } : null);
    inst.xFmt = o.xFormat ? fmt.get(o.xFormat) : xType === "time" ? function (ms) { return fmt.date(ms, "weekday"); } : function (v) { return fmt.num(v, decimalsOf(v) > 4 ? 4 : decimalsOf(v)); };
  }

  /** downsample spec -> fn(ms) -> bucket key, or null. "week": UTC Monday-Sunday; "month": calendar month (UTC). */
  function bucketFn(spec) {
    if (spec === "week" || spec === "weekly" || spec === "W") {
      return function (ms) { var d = new Date(ms); return Math.floor((ms - ((d.getUTCDay() + 6) % 7) * DAY) / DAY); };
    }
    if (spec === "month" || spec === "monthly" || spec === "M") {
      return function (ms) { var d = new Date(ms); return d.getUTCFullYear() * 12 + d.getUTCMonth(); };
    }
    return null;
  }

  function lineLegend(inst) {
    if (inst.series.length < 2 || inst.opts.legend === false) return null;
    return inst.series.map(function (s) { return { name: s.name, color: s.color, dashed: s.dashed, kind: "line" }; });
  }

  function drawLine(inst, W) {
    var o = inst.opts, ss = inst.series, ys = o.y || {};
    var xs = inst.dxs || inst.xs; // drawn x positions (downsampled when o.downsample is set)
    function P(s) { return s.dpts || s.pts; }
    var H = o.height || 260;
    if (!xs.length || !ss.some(function (s) { return s.pts.some(function (p) { return p[1] !== null; }); })) {
      message(inst, o.emptyText || "No data to plot."); inst.nav = null; return;
    }
    var isLog = !!ys.log;
    var lo = Infinity, hi = -Infinity;
    ss.forEach(function (s) { P(s).forEach(function (p) { if (p[1] !== null) { if (p[1] < lo) lo = p[1]; if (p[1] > hi) hi = p[1]; } }); });
    (o.refLines || []).forEach(function (r) { var v = toNum(r.y); if (v !== null && (!isLog || v > 0)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
    if (ys.zero && !isLog) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    var dom = ys.domain || [], fixed = false;
    if (dom[0] !== undefined && dom[0] !== null) { lo = dom[0]; fixed = true; }
    if (dom[1] !== undefined && dom[1] !== null) { hi = dom[1]; fixed = true; }

    var yLabelH = ys.label ? 18 : 0;
    var vlineH = (o.vlines || []).some(function (v) { return v.label; }) ? 16 : 0;
    var top = 10 + Math.max(yLabelH, vlineH);
    var bottom = 26 + (o.xLabel ? 14 : 0);
    var plotH = Math.max(40, H - top - bottom);
    var yt = isLog ? logTicks(lo, hi, ys.base || (fmt.kindOf(ys.format) === "idx" ? 100 : 1), fixed)
      : niceTicks(lo, hi, Math.max(2, Math.round(plotH / 52)), fixed);
    var tf = tickFormatter(ys.tickFormat || ys.format, yt.step, isLog);
    var tickLabels = yt.ticks.map(tf);
    var left = Math.max(28, Math.ceil(Math.max.apply(null, tickLabels.map(function (t) { return textWidth(t, 11); }))) + 10);

    var yScale = isLog ? logScale(yt.lo, yt.hi, top + plotH, top) : linear(yt.lo, yt.hi, top + plotH, top);
    var x0 = xs[0], x1 = xs[xs.length - 1];
    if (x0 === x1) { var padX = inst.xType === "time" ? DAY : 1; x0 -= padX; x1 += padX; }

    // end labels (<= 4 series, right edge, no collisions) — else legend + tooltip
    var right = 12, ends = [];
    var wantEnd = o.endLabels !== false && ss.length <= 4 && W >= 420;
    if (wantEnd) {
      var lastX = xs[xs.length - 1], spanX = (xs[xs.length - 1] - xs[0]) || 1;
      ss.forEach(function (s) {
        var last = null, pp = P(s);
        for (var i = pp.length - 1; i >= 0; i--) if (pp[i][1] !== null) { last = pp[i]; break; }
        if (!last || (lastX - last[0]) / spanX > 0.03) return;
        var vtxt = inst.tipFmt(last[1]);
        var name = ss.length > 1 ? s.name : "";
        var w = (name ? textWidth(name + " ", 11) : 0) + textWidth(vtxt, 11.5, 600);
        ends.push({ s: s, x: last[0], v: last[1], y: yScale(last[1]), name: name, vtxt: vtxt, w: w });
      });
      var sorted = ends.slice().sort(function (a, b) { return a.y - b.y; });
      for (var k = 1; k < sorted.length; k++) if (sorted[k].y - sorted[k - 1].y < 13) { wantEnd = false; break; }
      var maxW = Math.max.apply(null, ends.map(function (e) { return e.w; }).concat([0]));
      if (wantEnd && ends.length && maxW + 14 <= W * 0.34) right = Math.ceil(maxW) + 14; else wantEnd = false;
    }
    var plotW = Math.max(40, W - left - right);
    var xScale = linear(x0, x1, left, left + plotW);
    var svg = makeSvg(inst, W, H);
    inst.plot.classList.add("is-scrub");

    // annotation labels: placed with collision checks, drawn in gLabels ABOVE the series
    var labeler = makeLabeler([0, 0, W, H]);
    var gLabels = S("g", { "class": "labels" });

    // grid + y ticks
    var gGrid = S("g");
    yt.ticks.forEach(function (t, i) {
      var y = px(yScale(t));
      if (!(t === 0 && !isLog)) gGrid.appendChild(S("line", { x1: left, x2: left + plotW, y1: y, y2: y, "class": "grid" }));
      gGrid.appendChild(textEl(left - 8, y, tickLabels[i], "tick", "end", { dy: "0.32em" }));
    });
    svg.appendChild(gGrid);
    // the y-axis title shares the band above the plot with vline labels: reserve it first
    if (ys.label) {
      var yl = truncate(ys.label, W, 11);
      if (yl) { gGrid.appendChild(textEl(0, 12, yl, "lbl", "start", yl !== String(ys.label) ? { title: String(ys.label) } : null)); labeler.reserve(textBox(0, 12, textWidth(yl, 11), "start")); }
    }

    // bands (horizontal y0..y1 or vertical x0..x1): shading under the series, label above
    var bandLabels = [];
    (o.bands || []).forEach(function (b) {
      var fill = b.color || "var(--muted)";
      if (b.x0 !== undefined || b.x1 !== undefined) {
        var bx0 = inst.xType === "time" ? fmt.parseDate(b.x0) : toNum(b.x0), bx1 = inst.xType === "time" ? fmt.parseDate(b.x1) : toNum(b.x1);
        var xa = Math.max(left, xScale(bx0 === null ? x0 : bx0)), xb = Math.min(left + plotW, xScale(bx1 === null ? x1 : bx1));
        if (xb <= xa) return;
        svg.appendChild(S("rect", { x: xa, y: top, width: xb - xa, height: plotH, style: { fill: fill, fillOpacity: b.opacity || 0.08 } }));
        if (b.label) bandLabels.push([b.label, [{ x: xa + 6, y: top + 13, anchor: "start", minX: xa + 2, maxX: xb - 2 }, { x: xa + 6, y: top + plotH - 6, anchor: "start", minX: xa + 2, maxX: xb - 2 }]]);
      } else {
        var v0 = toNum(b.y0), v1 = toNum(b.y1);
        var ya = yScale(v1 === null ? yt.hi : Math.min(v1, yt.hi)), yb = yScale(v0 === null ? yt.lo : Math.max(v0, yt.lo));
        if (yb <= ya) return;
        svg.appendChild(S("rect", { x: left, y: ya, width: plotW, height: yb - ya, style: { fill: fill, fillOpacity: b.opacity || 0.08 } }));
        var bm = { minX: left, maxX: left + plotW };
        if (b.label) bandLabels.push([b.label, [assign({ x: left + 6, y: ya + 13, anchor: "start" }, bm), assign({ x: left + plotW - 6, y: ya + 13, anchor: "end" }, bm), assign({ x: left + 6, y: yb - 4, anchor: "start" }, bm)]]);
      }
    });

    // areas
    var baseV = isLog ? yt.lo : Math.min(Math.max(0, yt.lo), yt.hi);
    var baseY = yScale(baseV);
    ss.forEach(function (s) {
      if (!s.area) return;
      var d = "", run = [];
      function flush() {
        if (!run.length) return;
        d += "M" + run[0][0].toFixed(1) + "," + baseY.toFixed(1);
        run.forEach(function (p, i) {
          if (s.step && i > 0) d += "H" + p[0].toFixed(1);
          d += "L" + p[0].toFixed(1) + "," + p[1].toFixed(1);
        });
        d += "L" + run[run.length - 1][0].toFixed(1) + "," + baseY.toFixed(1) + "Z";
        run = [];
      }
      P(s).forEach(function (p) { if (p[1] === null) flush(); else run.push([xScale(p[0]), yScale(p[1])]); });
      flush();
      if (d) svg.appendChild(S("path", { d: d, style: { fill: s.color, fillOpacity: 0.1, stroke: "none" } }));
    });

    // axes: zero line on a signed axis (also when 0 is the top of the domain) + x baseline
    if (!isLog && yt.lo < 0 && yt.hi >= 0) svg.appendChild(S("line", { x1: left, x2: left + plotW, y1: px(yScale(0)), y2: px(yScale(0)), "class": "axis zero" }));
    svg.appendChild(S("line", { x1: left, x2: left + plotW, y1: px(top + plotH), y2: px(top + plotH), "class": "axis" }));

    // x ticks (labels that would collide are dropped, never overlapped)
    var gx = S("g");
    var xticks;
    if (inst.xType === "time") xticks = timeTicks(x0, x1, Math.floor(plotW / 62));
    else {
      var nt = niceTicks(x0, x1, Math.max(2, Math.floor(plotW / 80)), true);
      var xf = o.xTickFormat ? fmt.get(o.xTickFormat) : function (v) { return fmt.num(v, decimalsOf(nt.step)); };
      xticks = nt.ticks.map(function (v) { return { v: v, label: xf(v) }; });
    }
    drawXTicks(gx, xticks, xScale, top + plotH + 17, W, { tickMarks: [top + plotH, top + plotH + 4] });
    if (o.xLabel) gx.appendChild(textEl(left + plotW, top + plotH + 32, truncate(o.xLabel, plotW, 11), "lbl", "end"));
    svg.appendChild(gx);

    // vertical lines: dashed hairline under the series; label in the band above the plot
    // (centred, flipped to start/end at the edges; a second row inside the plot if two collide)
    (o.vlines || []).forEach(function (vl) {
      var xv = inst.xType === "time" ? fmt.parseDate(vl.x) : toNum(vl.x);
      if (xv === null || xv < x0 || xv > x1) return;
      var x = px(xScale(xv));
      svg.appendChild(S("line", { x1: x, x2: x, y1: top, y2: top + plotH, "class": "ref", style: vl.color ? { stroke: vl.color } : null }));
      if (vl.label) {
        var c = hCands(x, top - 5, 0, W, 3).concat([
          { x: x + 4, y: top + 13, anchor: "start", minX: left, maxX: left + plotW },
          { x: x - 4, y: top + 13, anchor: "end", minX: left, maxX: left + plotW },
          { x: x + 4, y: top + 26, anchor: "start", minX: left, maxX: left + plotW },
          { x: x - 4, y: top + 26, anchor: "end", minX: left, maxX: left + plotW }
        ]);
        drawPlaced(gLabels, labeler.place(vl.label, c));
      }
    });
    // reference lines (dashed hairline under the series; label above it at the right, or below / at the left)
    (o.refLines || []).forEach(function (r) {
      var v = toNum(r.y);
      if (v === null || v < yt.lo || v > yt.hi) return;
      var y = px(yScale(v));
      svg.appendChild(S("line", { x1: left, x2: left + plotW, y1: y, y2: y, "class": "ref", style: r.color ? { stroke: r.color } : null }));
      if (r.label) {
        // r.labelAt "start" tries the left end first (where it does not sit on the line ends)
        var lx = left + plotW - 4, sx = left + 4, mn = { minX: left, maxX: left + plotW };
        var atEnd = [assign({ x: lx, y: y - 4, anchor: "end" }, mn), assign({ x: lx, y: y + 12, anchor: "end" }, mn)];
        var atStart = [assign({ x: sx, y: y - 4, anchor: "start" }, mn), assign({ x: sx, y: y + 12, anchor: "start" }, mn)];
        drawPlaced(gLabels, labeler.place(r.label, r.labelAt === "start" ? atStart.concat(atEnd) : atEnd.concat(atStart)));
      }
    });
    bandLabels.forEach(function (bl) { drawPlaced(gLabels, labeler.place(bl[0], bl[1])); });

    // lines (a lighter 1.5px stroke when a series is denser than ~0.75 points per pixel)
    var gLines = S("g", { "class": "lines" });
    ss.forEach(function (s) {
      var d = "", started = false, singles = [], pp = P(s), n = 0;
      pp.forEach(function (p, i) {
        if (p[1] === null) { started = false; return; }
        n += 1;
        var X = xScale(p[0]), Y = yScale(p[1]);
        if (!started) {
          d += "M" + X.toFixed(1) + "," + Y.toFixed(1); started = true;
          var nx = pp[i + 1];
          if (!nx || nx[1] === null) singles.push([X, Y]);
        } else {
          if (s.step) d += "H" + X.toFixed(1) + "V" + Y.toFixed(1);
          else d += "L" + X.toFixed(1) + "," + Y.toFixed(1);
        }
      });
      var dense = n > 500 && n > plotW * 0.75;
      var sw = s.width || o.lineWidth || (dense ? 1.5 : 2);
      if (d) gLines.appendChild(S("path", { d: d, fill: "none", "stroke-width": sw, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dashed ? "5 3" : null, style: { stroke: s.color } }));
      singles.forEach(function (q) { gLines.appendChild(S("circle", { cx: q[0], cy: q[1], r: 2.5, style: { fill: s.color } })); });
    });
    svg.appendChild(gLines);

    // markers (annotations): the dot sits on the data, its label joins the label layer
    (o.markers || []).forEach(function (m) {
      var xv = inst.xType === "time" ? fmt.parseDate(m.x) : toNum(m.x), yv = toNum(m.y);
      if (xv === null || yv === null) return;
      var cx = xScale(xv), cy = yScale(yv);
      svg.appendChild(S("circle", { cx: cx, cy: cy, r: 4, "stroke-width": 2, style: { fill: m.color || "var(--ink-2)", stroke: "var(--surface)" } }));
      if (m.label) {
        var mn = { minX: left, maxX: left + plotW };
        drawPlaced(gLabels, labeler.place(m.label, [
          assign({ x: cx + 7, y: cy - 7, anchor: "start" }, mn), assign({ x: cx - 7, y: cy - 7, anchor: "end" }, mn),
          assign({ x: cx + 7, y: cy + 15, anchor: "start" }, mn), assign({ x: cx - 7, y: cy + 15, anchor: "end" }, mn)
        ]));
      }
    });
    svg.appendChild(gLabels);

    // end labels
    if (wantEnd) {
      ends.forEach(function (e) {
        var X = xScale(e.x);
        svg.appendChild(S("circle", { cx: X, cy: e.y, r: 4, "stroke-width": 2, style: { fill: e.s.color, stroke: "var(--surface)" } }));
        var t = S("text", { x: X + 9, y: e.y, dy: "0.32em", "class": "lbl" });
        if (e.name) t.appendChild(S("tspan", { text: e.name + " " }));
        t.appendChild(S("tspan", { "class": "lbl-strong", text: e.vtxt }));
        svg.appendChild(t);
      });
    }

    // hover layer: crosshair + dots + one tooltip for every series
    var hover = S("g", { visibility: "hidden" });
    var xh = S("line", { y1: top, y2: top + plotH, "class": "xhair" });
    hover.appendChild(xh);
    var dots = ss.map(function (s) {
      var c = S("circle", { r: 4, "stroke-width": 2, style: { fill: s.color, stroke: "var(--surface)" } });
      hover.appendChild(c);
      return c;
    });
    svg.appendChild(hover);
    var hit = S("rect", { x: left, y: top, width: plotW, height: plotH, "class": "hit" });
    svg.appendChild(hit);
    var xpx = xs.map(xScale);
    var cur = -1;
    function showAt(i, pointerY) {
      if (i < 0 || i >= xs.length) return;
      cur = i;
      var xv = xs[i], X = xScale(xv);
      xh.setAttribute("x1", px(X)); xh.setAttribute("x2", px(X));
      var ysum = 0, yn = 0;
      var content = el("div");
      content.appendChild(el("div", { "class": "fl-tip-head", text: inst.xFmt(xv) }));
      ss.forEach(function (s, k) {
        var v = s.map.has(xv) ? s.map.get(xv) : null;
        if (v === null || v === undefined) dots[k].setAttribute("visibility", "hidden");
        else {
          var Y = yScale(v);
          dots[k].setAttribute("visibility", "visible");
          dots[k].setAttribute("cx", X); dots[k].setAttribute("cy", Y);
          ysum += Y; yn += 1;
        }
        content.appendChild(tipRow(s.color, "line", v === null || v === undefined ? fmt.NA : inst.tipFmt(v), s.name, { dashed: s.dashed, dim: v === null || v === undefined }));
      });
      hover.setAttribute("visibility", "visible");
      showTip(inst, X, pointerY !== undefined ? pointerY : yn ? ysum / yn : top + plotH / 2, content);
    }
    function hide() { hover.setAttribute("visibility", "hidden"); hideTip(inst); }
    bindHover(inst, hit, function (e) { var p = svgPoint(svg, e); showAt(bisectNearest(xpx, p.x), p.y); }, hide);
    inst.nav = {
      key: function (k, shift) {
        var n = xs.length, step = shift ? 10 : 1;
        if (k === "ArrowRight") showAt(Math.min(n - 1, (cur < 0 ? n - 1 : cur + step)));
        else if (k === "ArrowLeft") showAt(Math.max(0, (cur < 0 ? n - 1 : cur - step)));
        else if (k === "Home") showAt(0);
        else if (k === "End") showAt(n - 1);
        else if (k === "Escape") hide();
        else return false;
        return true;
      },
      focus: function () { showAt(cur < 0 || cur >= xs.length ? xs.length - 1 : cur); },
      blur: hide
    };
  }

  function lineTable(inst) {
    var o = inst.opts, ss = inst.series;
    var cols = [{ key: "x", label: o.xLabel || (inst.xType === "time" ? "Date (UTC)" : "x"), sortValue: function (r) { return r._x; } }];
    ss.forEach(function (s, i) { cols.push({ key: "s" + i, label: s.name, format: inst.tipFmt, align: "right" }); });
    var rows = inst.xs.map(function (xv) {
      var r = { _x: xv, x: inst.xType === "time" ? fmt.isoDate(xv) : inst.xRaw[xv] };
      ss.forEach(function (s, i) { r["s" + i] = s.map.has(xv) ? s.map.get(xv) : null; });
      return r;
    });
    return { columns: cols, rows: rows, caption: o.tableCaption || null };
  }

  function lineSummary(inst) {
    var xs = inst.xs, ss = inst.series;
    if (!xs.length) return "Line chart with no data.";
    var span = inst.xType === "time" ? fmt.isoDate(xs[0]) + " to " + fmt.isoDate(xs[xs.length - 1]) : String(inst.xRaw[xs[0]]) + " to " + String(inst.xRaw[xs[xs.length - 1]]);
    var parts = ss.map(function (s) {
      var last = null;
      for (var i = s.pts.length - 1; i >= 0; i--) if (s.pts[i][1] !== null) { last = s.pts[i][1]; break; }
      return s.name + " ends at " + (last === null ? fmt.NA : inst.tipFmt(last));
    });
    return "Line chart, " + ss.length + " series, " + span + ". " + parts.join("; ") + ".";
  }

  /* ====================================================================== */
  /* bars                                                                    */
  /* ====================================================================== */

  function prepareBars(inst) {
    var o = inst.opts, xs = (o.x || o.categories || []).slice();
    var ys = o.y || {};
    var xType = o.xType || (!o.horizontal && xs.length && xs.every(function (v) { return ISO_RE.test(String(v)); }) ? "time" : "category");
    inst.xType = xType;
    inst.xs = xs;
    inst.mode = o.mode || (o.group ? "group" : "stack");
    inst.stacks = (o.stacks || o.series || []).map(function (s, i) {
      var vals = xs.map(function (_, j) { return toNum(s.values ? s.values[j] : null); });
      return { id: s.id || "b" + i, name: seriesName(s, i), color: seriesColor(s, i, o.context), values: vals };
    });
    inst.tipFmt = fmt.get(ys.tooltipFormat || ys.format || "num", ys.signed !== undefined ? { signed: ys.signed } : null);
    inst.valFmt = fmt.get(o.valueFormat || ys.tooltipFormat || ys.format || "num", ys.signed !== undefined ? { signed: ys.signed } : null);
    inst.xFmt = o.xFormat ? fmt.get(o.xFormat) : xType === "time" ? function (v) { return fmt.date(v, "weekday"); } : function (v) { return String(v); };
    inst.xLabelFmt = o.xTickFormat ? fmt.get(o.xTickFormat) : function (v) { return String(v); };
  }
  function barsLegend(inst) {
    if (inst.stacks.length < 2 || inst.opts.legend === false) return null;
    return inst.stacks.map(function (s) { return { name: s.name, color: s.color, kind: "rect" }; });
  }
  // extents per column (stack: pos/neg sums; group: min/max values)
  function barExtent(inst) {
    var lo = 0, hi = 0;
    inst.xs.forEach(function (_, j) {
      if (inst.mode === "stack") {
        var p = 0, n = 0;
        inst.stacks.forEach(function (s) { var v = s.values[j]; if (v === null) return; if (v > 0) p += v; else n += v; });
        hi = Math.max(hi, p); lo = Math.min(lo, n);
      } else inst.stacks.forEach(function (s) { var v = s.values[j]; if (v === null) return; hi = Math.max(hi, v); lo = Math.min(lo, v); });
    });
    return [lo, hi];
  }
  /** rect path with optional rounded corners on one end. dir: 'up'|'down'|'right'|'left' = which end is the data end */
  function barPath(x, y, w, h, dir, round) {
    if (w <= 0 || h <= 0) return "";
    var r = round ? Math.min(4, w / 2, h) : 0;
    var x2 = x + w, y2 = y + h;
    if (!r) return "M" + x + "," + y + "H" + x2 + "V" + y2 + "H" + x + "Z";
    switch (dir) {
      case "up": return "M" + x + "," + y2 + "V" + (y + r) + "Q" + x + "," + y + " " + (x + r) + "," + y + "H" + (x2 - r) + "Q" + x2 + "," + y + " " + x2 + "," + (y + r) + "V" + y2 + "Z";
      case "down": return "M" + x + "," + y + "H" + x2 + "V" + (y2 - r) + "Q" + x2 + "," + y2 + " " + (x2 - r) + "," + y2 + "H" + (x + r) + "Q" + x + "," + y2 + " " + x + "," + (y2 - r) + "Z";
      case "right": return "M" + x + "," + y + "H" + (x2 - r) + "Q" + x2 + "," + y + " " + x2 + "," + (y + r) + "V" + (y2 - r) + "Q" + x2 + "," + y2 + " " + (x2 - r) + "," + y2 + "H" + x + "Z";
      default: return "M" + x2 + "," + y + "H" + (x + r) + "Q" + x + "," + y + " " + x + "," + (y + r) + "V" + (y2 - r) + "Q" + x + "," + y2 + " " + (x + r) + "," + y2 + "H" + x2 + "Z";
    }
  }
  function r1(v) { return Math.round(v * 10) / 10; }

  function barsTip(inst, j) {
    var content = el("div");
    var xv = inst.xs[j];
    content.appendChild(el("div", { "class": "fl-tip-head", text: inst.xFmt(xv) }));
    var total = 0, n = 0;
    inst.stacks.forEach(function (s) {
      var v = s.values[j];
      if (v !== null) { total += v; n += 1; }
      content.appendChild(tipRow(s.color, "rect", v === null ? fmt.NA : inst.tipFmt(v), s.name, { dim: v === null }));
    });
    if (inst.mode === "stack" && inst.stacks.length > 1 && n > 0 && inst.opts.total !== false) {
      content.appendChild(tipRow(null, "rect", inst.tipFmt(total), inst.opts.totalLabel || "Net", { total: true }));
    }
    return content;
  }

  function drawBars(inst, W) {
    if (inst.opts.horizontal) return drawBarsH(inst, W);
    var o = inst.opts, ys = o.y || {}, xs = inst.xs, st = inst.stacks;
    var H = o.height || 240;
    if (!xs.length || !st.length) { message(inst, o.emptyText || "No data to plot."); inst.nav = null; return; }
    var ext = barExtent(inst), lo = ext[0], hi = ext[1];
    (o.refLines || []).forEach(function (r) { var v = toNum(r.y); if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
    var dom = ys.domain || [], fixed = false;
    if (dom[0] !== undefined && dom[0] !== null) { lo = dom[0]; fixed = true; }
    if (dom[1] !== undefined && dom[1] !== null) { hi = dom[1]; fixed = true; }
    if (lo === 0 && hi === 0) hi = 1;
    var top = 10 + (ys.label ? 18 : 0), bottom = 26;
    var plotH = Math.max(40, H - top - bottom);
    var yt = niceTicks(lo, hi, Math.max(2, Math.round(plotH / 52)), fixed);
    var tf = tickFormatter(ys.tickFormat || ys.format, yt.step, false);
    var tickLabels = yt.ticks.map(tf);
    var left = Math.max(28, Math.ceil(Math.max.apply(null, tickLabels.map(function (t) { return textWidth(t, 11); }))) + 10);
    var refW = Math.max.apply(null, [0].concat((o.refLines || []).map(function (r) { return r.label ? textWidth(r.label, 11) : 0; })));
    var right = refW ? Math.min(Math.ceil(refW) + 10, Math.floor(W * 0.3)) : 12;
    var plotW = Math.max(40, W - left - right);
    var yScale = linear(yt.lo, yt.hi, top + plotH, top);
    var n = xs.length, centers = [], slot;
    if (inst.xType === "time") {
      var ms = xs.map(function (v) { return fmt.parseDate(v); });
      var gap = DAY;
      for (var i = 1; i < ms.length; i++) { var g = ms[i] - ms[i - 1]; if (g > 0 && (i === 1 || g < gap)) gap = g; }
      var xScale = linear(ms[0] - gap / 2, ms[n - 1] + gap / 2, left, left + plotW);
      slot = xScale(ms[0] + gap) - xScale(ms[0]);
      centers = ms.map(xScale);
      inst._xScale = xScale; inst._ms = ms;
    } else {
      slot = plotW / n;
      for (var c = 0; c < n; c++) centers.push(left + slot * (c + 0.5));
    }
    var k = st.length;
    var barW, inner = 0;
    if (inst.mode === "group" && k > 1) {
      inner = Math.max(1, Math.min(24, (slot * 0.8 - (k - 1) * 2) / k));
      barW = inner * k + (k - 1) * 2;
    } else barW = inst.xType === "time" ? Math.max(1, Math.min(24, slot - 2)) : Math.max(2, Math.min(24, slot * 0.62));

    var svg = makeSvg(inst, W, H);
    inst.plot.classList.add("is-scrub");
    var gGrid = S("g");
    yt.ticks.forEach(function (t, i) {
      var y = px(yScale(t));
      if (t !== 0) gGrid.appendChild(S("line", { x1: left, x2: left + plotW, y1: y, y2: y, "class": "grid" }));
      gGrid.appendChild(textEl(left - 8, y, tickLabels[i], "tick", "end", { dy: "0.32em" }));
    });
    axisTitle(gGrid, ys.label, 0, W);
    svg.appendChild(gGrid);

    var zeroY = yScale(Math.min(Math.max(0, yt.lo), yt.hi));
    var gBars = S("g", { "class": "bars" });
    var cols = [];
    xs.forEach(function (xv, j) {
      var gcol = S("g", { "class": "col" });
      var cx = centers[j];
      if (inst.mode === "group" && k > 1) {
        st.forEach(function (s, si) {
          var v = s.values[j]; if (v === null || v === 0) return;
          var bx = cx - barW / 2 + si * (inner + 2);
          var y0 = zeroY, y1 = yScale(v);
          var yTop = Math.min(y0, y1), h = Math.abs(y1 - y0);
          gcol.appendChild(S("path", { d: barPath(r1(bx), r1(yTop), r1(inner), r1(h), v > 0 ? "up" : "down", true), style: { fill: s.color } }));
        });
      } else {
        // diverging stack: positives up from 0, negatives down from 0; 2px surface gap between segments
        var lastPos = -1, lastNeg = -1;
        st.forEach(function (s, si) { var v = s.values[j]; if (v === null || v === 0) return; if (v > 0) lastPos = si; else lastNeg = si; });
        var pb = 0, nb = 0;
        st.forEach(function (s, si) {
          var v = s.values[j]; if (v === null || v === 0) return;
          var a, b, yTop, h;
          if (v > 0) {
            a = yScale(pb); b = yScale(pb + v); if (pb !== 0) a -= 2; pb += v;
            yTop = b; h = a - b;
          } else {
            a = yScale(nb); b = yScale(nb + v); if (nb !== 0) a += 2; nb += v;
            yTop = a; h = b - a;
          }
          if (h < 0.4) return;
          gcol.appendChild(S("path", { d: barPath(r1(cx - barW / 2), r1(yTop), r1(barW), r1(h), v > 0 ? "up" : "down", si === (v > 0 ? lastPos : lastNeg)), style: { fill: s.color } }));
        });
      }
      gBars.appendChild(gcol);
      cols.push(gcol);
    });
    svg.appendChild(gBars);

    // value labels on caps (single stack or groups) when they fit
    if (o.valueLabels) {
      var vlab = makeLabeler([0, 0, W, H]);
      xs.forEach(function (xv, j) {
        st.forEach(function (s, si) {
          if (inst.mode !== "group" && st.length > 1) return;
          var v = s.values[j]; if (v === null) return;
          var t = inst.valFmt(v), w = textWidth(t, 11);
          var avail = inst.mode === "group" && k > 1 ? inner + 2 : slot - 2;
          if (w > avail + (inst.mode === "group" ? 0 : 6)) return;
          var cx = inst.mode === "group" && k > 1 ? centers[j] - barW / 2 + si * (inner + 2) + inner / 2 : centers[j];
          var y = yScale(v);
          // never overlap a neighbour's label (it stays in the tooltip and the table)
          drawPlaced(svg, vlab.place(t, [{ x: cx, y: v >= 0 ? y - 5 : y + 13, anchor: "middle" }], { force: false }), "lbl");
        });
      });
    }

    // zero baseline
    svg.appendChild(S("line", { x1: left, x2: left + plotW, y1: px(zeroY), y2: px(zeroY), "class": "axis" }));
    (o.refLines || []).forEach(function (r) {
      var v = toNum(r.y); if (v === null || v < yt.lo || v > yt.hi) return;
      var y = px(yScale(v));
      svg.appendChild(S("line", { x1: left, x2: left + plotW, y1: y, y2: y, "class": "ref", style: r.color ? { stroke: r.color } : null }));
      if (r.label) svg.appendChild(textEl(left + plotW + 6, y, truncate(r.label, right - 8, 11), "lbl", "start", { dy: "0.32em" }));
    });

    // x labels
    var gx = S("g"), baseY = top + plotH;
    if (inst.xType === "time") {
      var xsc = inst._xScale;
      var d0 = inst._ms[0] - (inst._ms.length > 1 ? 0 : DAY), d1 = inst._ms[inst._ms.length - 1];
      drawXTicks(gx, timeTicks(d0, d1, Math.floor(plotW / 62)), xsc, baseY + 17, W);
    } else {
      var labels = xs.map(function (v) { return inst.xLabelFmt(v); });
      var maxLW = Math.max.apply(null, labels.map(function (l) { return textWidth(l, 11); }));
      var every = Math.max(1, Math.ceil((Math.min(maxLW, 90) + 8) / slot));
      labels.forEach(function (l, j) {
        if (j % every !== 0) return;
        var t = truncate(l, slot * every - 6, 11);
        gx.appendChild(textEl(centers[j], baseY + 17, t, "tick", "middle"));
      });
    }
    svg.appendChild(gx);

    // hover: snap to nearest column, lift it, one tooltip for the column's marks
    var hit = S("rect", { x: left, y: top, width: plotW, height: plotH, "class": "hit" });
    svg.appendChild(hit);
    var cur = -1;
    function showAt(j, pointerY) {
      if (j < 0 || j >= n) return;
      if (cur >= 0 && cols[cur]) cols[cur].classList.remove("is-hot");
      cur = j;
      cols[j].classList.add("is-hot");
      svg.classList.add("is-hovering");
      showTip(inst, centers[j], pointerY !== undefined ? pointerY : top + plotH / 3, barsTip(inst, j));
    }
    function hide() { svg.classList.remove("is-hovering"); if (cur >= 0 && cols[cur]) cols[cur].classList.remove("is-hot"); hideTip(inst); }
    bindHover(inst, hit, function (e) { var p = svgPoint(svg, e); showAt(bisectNearest(centers, p.x), p.y); }, hide);
    inst.nav = navFor(function () { return n; }, function () { return cur; }, showAt, hide);
  }

  function navFor(count, current, showAt, hide) {
    return {
      key: function (k, shift) {
        var n = count(), c = current(), step = shift ? 10 : 1;
        if (k === "ArrowRight" || k === "ArrowDown") showAt(Math.min(n - 1, c < 0 ? 0 : c + step));
        else if (k === "ArrowLeft" || k === "ArrowUp") showAt(Math.max(0, c < 0 ? n - 1 : c - step));
        else if (k === "Home") showAt(0);
        else if (k === "End") showAt(n - 1);
        else if (k === "Escape") hide();
        else return false;
        return true;
      },
      focus: function () { var c = current(); showAt(c < 0 ? count() - 1 : c); },
      blur: hide
    };
  }

  function drawBarsH(inst, W) {
    var o = inst.opts, ys = o.y || {}, cats = inst.xs, st = inst.stacks;
    if (!cats.length || !st.length) { message(inst, o.emptyText || "No data to plot."); inst.nav = null; return; }
    var n = cats.length, k = st.length;
    var group = inst.mode === "group" && k > 1;
    var band = o.barSlot || (group ? Math.max(28, k * 14 + 12) : 28);
    var refLab = (o.refLines || []).some(function (r) { return r.label; });
    var top = 6 + (ys.label ? 18 : 0) + (refLab ? 14 : 0), bottom = 24; // reference labels get their own row above the bars
    var H = o.height ? o.height : top + n * band + bottom;
    if (o.height) band = (H - top - bottom) / n;
    var ext = barExtent(inst), lo = ext[0], hi = ext[1];
    (o.refLines || []).forEach(function (r) { var v = toNum(r.y); if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
    var dom = ys.domain || [], fixed = false;
    if (dom[0] !== undefined && dom[0] !== null) { lo = dom[0]; fixed = true; }
    if (dom[1] !== undefined && dom[1] !== null) { hi = dom[1]; fixed = true; }
    if (lo === 0 && hi === 0) hi = 1;
    var labels = cats.map(function (c) { return inst.xLabelFmt(c); });
    var lw = Math.min(o.labelWidth || 200, W * 0.4, Math.ceil(Math.max.apply(null, labels.map(function (l) { return textWidth(l, 12); }))) + 14);
    // value labels at the tips (single stack or groups)
    var showVals = o.valueLabels !== false && (k === 1 || group);
    var posW = 0, negW = 0;
    if (showVals) st.forEach(function (s) { s.values.forEach(function (v) { if (v === null) return; var w = textWidth(inst.valFmt(v), 11) + 8; if (v >= 0) posW = Math.max(posW, w); else negW = Math.max(negW, w); }); });
    var left = lw + negW, right = Math.max(12, posW);
    var plotW = Math.max(40, W - left - right);
    var xt = niceTicks(lo, hi, Math.max(2, Math.floor(plotW / 90)), fixed);
    var tf = tickFormatter(ys.tickFormat || ys.format, xt.step, false);
    var xScale = linear(xt.lo, xt.hi, left, left + plotW);
    var plotH = n * band;
    var svg = makeSvg(inst, W, H);
    var g = S("g");
    xt.ticks.forEach(function (t) {
      var x = px(xScale(t));
      if (t !== 0) g.appendChild(S("line", { x1: x, x2: x, y1: top, y2: top + plotH, "class": "grid" }));
    });
    drawXTicks(g, xt.ticks.map(function (t) { return { v: t, label: tf(t) }; }), xScale, top + plotH + 16, W);
    axisTitle(g, ys.label, 0, W); // value-axis title: from the left edge, truncated to the chart width
    svg.appendChild(g);
    var zeroX = xScale(Math.min(Math.max(0, xt.lo), xt.hi));
    var thick = group ? Math.max(1, Math.min(24, (band * 0.72 - (k - 1) * 2) / k)) : Math.max(4, Math.min(24, band * 0.64));
    var gBars = S("g", { "class": "bars" }), rows = [], centers = [];
    cats.forEach(function (c, j) {
      var cy = top + band * (j + 0.5);
      centers.push(cy);
      svg.appendChild(textEl(lw - 10, cy, truncate(labels[j], lw - 14, 12), "lbl", "end", { dy: "0.32em", style: { fontSize: "12px" } }));
      var grow = S("g", { "class": "col" });
      if (group) {
        var tot = thick * k + (k - 1) * 2;
        st.forEach(function (s, si) {
          var v = s.values[j]; if (v === null) return;
          var by = cy - tot / 2 + si * (thick + 2);
          var xa = Math.min(zeroX, xScale(v)), w = Math.abs(xScale(v) - zeroX);
          if (w >= 0.4) grow.appendChild(S("path", { d: barPath(r1(xa), r1(by), r1(w), r1(thick), v >= 0 ? "right" : "left", true), style: { fill: s.color } }));
          if (showVals) { var t = inst.valFmt(v); svg.appendChild(textEl(v >= 0 ? xScale(v) + 5 : xScale(v) - 5, by + thick / 2, t, "lbl", v >= 0 ? "start" : "end", { dy: "0.32em" })); }
        });
      } else {
        var lastPos = -1, lastNeg = -1;
        st.forEach(function (s, si) { var v = s.values[j]; if (v === null || v === 0) return; if (v > 0) lastPos = si; else lastNeg = si; });
        var pb = 0, nb = 0;
        st.forEach(function (s, si) {
          var v = s.values[j]; if (v === null || v === 0) return;
          var a, b, xa, w;
          if (v > 0) { a = xScale(pb); b = xScale(pb + v); if (pb !== 0) a += 2; pb += v; xa = a; w = b - a; }
          else { a = xScale(nb); b = xScale(nb + v); if (nb !== 0) a -= 2; nb += v; xa = b; w = a - b; }
          if (w < 0.4) return;
          grow.appendChild(S("path", { d: barPath(r1(xa), r1(cy - thick / 2), r1(w), r1(thick), v > 0 ? "right" : "left", si === (v > 0 ? lastPos : lastNeg)), style: { fill: s.color } }));
        });
        if (showVals) {
          var v1 = st[0].values[j];
          if (v1 !== null) svg.appendChild(textEl(v1 >= 0 ? xScale(v1) + 5 : xScale(v1) - 5, cy, inst.valFmt(v1), "lbl-strong", v1 >= 0 ? "start" : "end", { dy: "0.32em" }));
        }
      }
      gBars.appendChild(grow);
      rows.push(grow);
    });
    svg.insertBefore(gBars, svg.firstChild.nextSibling);
    svg.appendChild(S("line", { x1: px(zeroX), x2: px(zeroX), y1: top, y2: top + plotH, "class": "axis" }));
    var labeler = makeLabeler([0, 0, W, H]);
    if (ys.label) labeler.reserve(textBox(0, 12, Math.min(W, textWidth(ys.label, 11)), "start"));
    (o.refLines || []).forEach(function (r) {
      var v = toNum(r.y); if (v === null || v < xt.lo || v > xt.hi) return;
      var x = px(xScale(v));
      svg.appendChild(S("line", { x1: x, x2: x, y1: top, y2: top + plotH, "class": "ref", style: r.color ? { stroke: r.color } : null }));
      if (r.label) drawPlaced(svg, labeler.place(r.label, hCands(x, top - 4, 0, W, 3)), "lbl");
    });
    var hit = S("rect", { x: 0, y: top, width: W, height: plotH, "class": "hit" });
    svg.appendChild(hit);
    var cur = -1;
    function showAt(j, pointerX) {
      if (j < 0 || j >= n) return;
      if (cur >= 0 && rows[cur]) rows[cur].classList.remove("is-hot");
      cur = j; rows[j].classList.add("is-hot"); svg.classList.add("is-hovering");
      showTip(inst, pointerX !== undefined ? pointerX : Math.min(W - 40, left + plotW / 2), centers[j], barsTip(inst, j));
    }
    function hide() { svg.classList.remove("is-hovering"); if (cur >= 0 && rows[cur]) rows[cur].classList.remove("is-hot"); hideTip(inst); }
    bindHover(inst, hit, function (e) { var p = svgPoint(svg, e); showAt(bisectNearest(centers, p.y), p.x); }, hide);
    inst.nav = navFor(function () { return n; }, function () { return cur; }, showAt, hide);
  }

  function barsTable(inst) {
    var o = inst.opts, st = inst.stacks;
    var cols = [{ key: "x", label: o.xLabel || (inst.xType === "time" ? "Date (UTC)" : "Category"), sortValue: function (r) { return r._i; } }];
    st.forEach(function (s, i) { cols.push({ key: "s" + i, label: s.name, format: inst.tipFmt, align: "right" }); });
    var withTotal = inst.mode === "stack" && st.length > 1 && o.total !== false;
    if (withTotal) cols.push({ key: "total", label: o.totalLabel || "Net", format: inst.tipFmt, align: "right" });
    var rows = inst.xs.map(function (xv, j) {
      var r = { _i: j, x: String(xv) }, t = 0, any = false;
      st.forEach(function (s, i) { var v = s.values[j]; r["s" + i] = v; if (v !== null) { t += v; any = true; } });
      if (withTotal) r.total = any ? t : null;
      return r;
    });
    return { columns: cols, rows: rows, caption: o.tableCaption || null };
  }
  function barsSummary(inst) {
    var n = inst.xs.length;
    return (inst.opts.horizontal ? "Horizontal bar chart" : "Bar chart") + ", " + n + " " + (inst.xType === "time" ? "dates" : "categories") + ", " + inst.stacks.length + " series" +
      (inst.stacks.length ? " (" + inst.stacks.map(function (s) { return s.name; }).join(", ") + ")" : "") + ".";
  }

  /* ====================================================================== */
  /* heatmap                                                                 */
  /* ====================================================================== */

  function prepareHeat(inst) {
    var o = inst.opts;
    inst.rows = (o.rows || []).slice();
    inst.cols = (o.cols || []).slice();
    inst.vals = inst.rows.map(function (_, r) { return inst.cols.map(function (_, c) { return toNum(o.values && o.values[r] ? o.values[r][c] : null); }); });
    var lo = Infinity, hi = -Infinity, amax = 0;
    inst.vals.forEach(function (row) { row.forEach(function (v) { if (v === null) return; lo = Math.min(lo, v); hi = Math.max(hi, v); amax = Math.max(amax, Math.abs(v)); }); });
    var type = o.scale === "div" ? "div" : "seq";
    var d = o.domain;
    if (!d) d = type === "div" ? [-(amax || 1), 0, amax || 1] : [isFinite(lo) ? lo : 0, isFinite(hi) ? hi : 1];
    if (type === "div" && d.length === 2) d = [d[0], 0, d[1]];
    inst.scaleType = type; inst.domain = d;
    inst.vf = fmt.get(o.format || "num");
    inst.colFmt = o.colFormat ? fmt.get(o.colFormat) : function (v) { return ISO_RE.test(String(v)) ? fmt.date(v, "axis") : String(v); };
    inst.rowFmt = o.rowFormat ? fmt.get(o.rowFormat) : function (v) { return ISO_RE.test(String(v)) ? fmt.date(v, "iso") : String(v); };
  }
  /*
   * Heatmap layout. When the cells need more width than the plot has (minCellWidth), the
   * plot splits in three: the row labels stay fixed at the left, the cell grid scrolls
   * sideways in its own box, and the colour legend sits under them, outside the scroll box.
   * A time axis (ISO date columns, or o.timeAxis) opens at its newest end and thins its
   * column labels; a categorical axis labels every column (o.colLabels 'all' | 'thin'),
   * turned vertical when the labels do not fit across.
   */
  function drawHeat(inst, W) {
    var o = inst.opts, rows = inst.rows, cols = inst.cols, vals = inst.vals;
    var nr = rows.length, nc = cols.length;
    if (!nr || !nc) { message(inst, o.emptyText || "No data to plot."); inst.nav = null; return; }
    var scale = colors.scale({ type: inst.scaleType, domain: inst.domain });
    var rowLabels = rows.map(inst.rowFmt), colLabels = cols.map(inst.colFmt);
    var isTime = o.timeAxis !== undefined ? !!o.timeAxis : cols.every(function (c) { return ISO_RE.test(String(c)); });
    var colMode = o.colLabels === "all" || o.colLabels === "thin" ? o.colLabels : (isTime ? "thin" : "all");
    var lw = Math.min(o.rowLabelWidth || 180, W * 0.35, Math.ceil(Math.max.apply(null, rowLabels.map(function (l) { return textWidth(l, 11.5); }))) + 12);
    var cellH = o.cellHeight || 24;
    var avail = W - lw - 4;
    var minCell = o.minCellWidth || 14;
    var cellW = avail / nc, scroll = false;
    if (o.maxCellWidth) cellW = Math.min(cellW, o.maxCellWidth);
    if (cellW < minCell) { cellW = minCell; scroll = true; }
    var gridW = cellW * nc;
    var maxCL = Math.max.apply(null, colLabels.map(function (l) { return textWidth(l, 11); }));
    var every = colMode === "all" ? 1 : Math.max(1, Math.ceil((maxCL + 6) / cellW));
    var rotate = colMode === "all" && maxCL + 6 > cellW;
    var rotLen = rotate ? Math.min(o.colLabelMax || 120, Math.ceil(maxCL)) : 0;
    var top = rotate ? rotLen + 14 : 24;
    var legendH = o.legend === false ? 0 : 40;
    var gridH = top + nr * cellH + 10;
    if (o.height) cellH = Math.max(8, (o.height - top - 10 - legendH) / nr);
    gridH = top + nr * cellH + 10;
    var H = gridH + legendH;
    var gap = cellW >= 10 && cellH >= 10 ? 2 : 1;
    var plot = inst.plot;
    plot.classList.remove("is-scroll");
    plot.classList.toggle("is-heat-split", scroll);
    inst.height = H;

    // svgCells holds the column labels, cells, outlines and the hit layer; gx is the grid's left edge in it
    var svgRows, svgCells, svgLegend, scroller = null, gx;
    if (scroll) {
      var row = el("div", { "class": "fl-heat-row" });
      svgRows = S("svg", { width: lw, height: gridH, viewBox: "0 0 " + lw + " " + gridH, "aria-hidden": "true", focusable: "false", "class": "fl-heat-labels" });
      scroller = el("div", { "class": "fl-heat-scroll" });
      svgCells = S("svg", { width: gridW + 4, height: gridH, viewBox: "0 0 " + (gridW + 4) + " " + gridH, "aria-hidden": "true", focusable: "false" });
      scroller.appendChild(svgCells);
      row.appendChild(svgRows); row.appendChild(scroller);
      plot.appendChild(row);
      if (legendH) { svgLegend = S("svg", { width: W, height: legendH, viewBox: "0 0 " + W + " " + legendH, "aria-hidden": "true", focusable: "false" }); plot.appendChild(svgLegend); }
      gx = 0;
    } else {
      svgRows = svgCells = svgLegend = makeSvg(inst, W, H);
      gx = lw;
    }
    // column labels
    colLabels.forEach(function (l, c) {
      if (c % every !== 0) return;
      var cx = gx + c * cellW + (cellW - gap) / 2;
      if (rotate) {
        var t = textEl(cx, top - 8, truncate(l, rotLen, 11), "tick", "start", { dy: "0.32em", transform: "rotate(-90 " + r1(cx) + " " + (top - 8) + ")" });
        if (t.textContent !== l) t.appendChild(S("title", { text: l }));
        svgCells.appendChild(t);
      } else svgCells.appendChild(textEl(cx, top - 8, truncate(l, every * cellW - 4, 11), "tick", "middle"));
    });
    rowLabels.forEach(function (l, r) {
      svgRows.appendChild(textEl(lw - 8, top + r * cellH + (cellH - gap) / 2, truncate(l, lw - 12, 11.5), "lbl", "end", { dy: "0.32em" }));
    });
    var gCells = S("g");
    var anyNull = false;
    // cell labels are all-or-none: if the widest formatted value does not fit a cell, no cell is
    // labelled (scattered labels would single out the narrow-glyph values)
    var showLabels = (o.cellLabel === true || o.cellLabel === "auto") && cellH - gap >= 14;
    if (showLabels) {
      var widest = 0;
      vals.forEach(function (rw) { rw.forEach(function (v) { if (v !== null) widest = Math.max(widest, textWidth(inst.vf(v), 10.5)); }); });
      showLabels = widest <= cellW - gap - 6;
    }
    vals.forEach(function (rw, r) {
      rw.forEach(function (v, c) {
        var x = gx + c * cellW, y = top + r * cellH;
        var fill = v === null ? null : scale(v);
        if (v === null) anyNull = true;
        gCells.appendChild(S("rect", { x: r1(x), y: r1(y), width: r1(Math.max(1, cellW - gap)), height: r1(Math.max(1, cellH - gap)), rx: 2, style: { fill: fill || "var(--bg)", stroke: fill ? null : "var(--axis)" } }));
        if (showLabels && v !== null) {
          gCells.appendChild(S("text", { x: x + (cellW - gap) / 2, y: y + (cellH - gap) / 2, dy: "0.32em", "text-anchor": "middle", text: inst.vf(v), style: { fill: colors.inkOn(fill), fontSize: "10.5px", fontVariantNumeric: "tabular-nums" } }));
        }
      });
    });
    svgCells.appendChild(gCells);
    // highlighted rows/cols (e.g. the latest Sunday)
    function idxOf(list, k) { if (typeof k === "number") return k; return list.indexOf(k); }
    (o.highlightCols || []).forEach(function (k) {
      var c = idxOf(cols, k); if (c < 0 || c >= nc) return;
      svgCells.appendChild(S("rect", { x: gx + c * cellW - 1, y: top - 1, width: cellW - gap + 2, height: nr * cellH - gap + 2, rx: 3, "class": "hl-outline" }));
    });
    (o.highlightRows || []).forEach(function (k) {
      var r = idxOf(rows, k); if (r < 0 || r >= nr) return;
      svgCells.appendChild(S("rect", { x: gx - 1, y: top + r * cellH - 1, width: gridW - gap + 2, height: cellH - gap + 2, rx: 3, "class": "hl-outline" }));
    });
    // scale legend (under the grid; in its own svg when the grid scrolls)
    if (legendH) {
      var lx0 = scroll ? lw : gx, ly = scroll ? 12 : gridH + 2;
      var lwid = Math.max(60, Math.min(220, Math.max(120, (scroll ? W - lw : gridW) * 0.5), W - lx0 - 8));
      var gid = ui.uid("flgrad");
      var defs = S("defs"), grad = S("linearGradient", { id: gid, x1: "0", x2: "1", y1: "0", y2: "0" });
      for (var st = 0; st <= 10; st++) {
        var tt = st / 10;
        grad.appendChild(S("stop", { offset: (tt * 100) + "%", "stop-color": inst.scaleType === "div" ? colors.div(tt * 2 - 1) : colors.seq(tt) }));
      }
      defs.appendChild(grad); svgLegend.appendChild(defs);
      svgLegend.appendChild(S("rect", { x: lx0, y: ly, width: lwid, height: 8, rx: 2, fill: "url(#" + gid + ")" }));
      var d = inst.domain;
      svgLegend.appendChild(textEl(lx0, ly + 22, inst.vf(d[0]), "tick", "start"));
      if (inst.scaleType === "div") svgLegend.appendChild(textEl(lx0 + lwid / 2, ly + 22, inst.vf(d[1]), "tick", "middle"));
      svgLegend.appendChild(textEl(lx0 + lwid, ly + 22, inst.vf(d[d.length - 1]), "tick", "end"));
      var lx = lx0 + lwid + 14;
      if (o.legendLabel) {
        var ll = truncate(o.legendLabel, Math.max(0, W - lx - (anyNull ? 70 : 0)), 11);
        if (ll) { svgLegend.appendChild(textEl(lx, ly + 8, ll, "lbl", "start")); lx += textWidth(ll, 11) + 14; }
      }
      if (anyNull && lx + 60 <= W) {
        svgLegend.appendChild(S("rect", { x: lx, y: ly, width: 12, height: 8, rx: 2, style: { fill: "var(--bg)", stroke: "var(--axis)" } }));
        svgLegend.appendChild(textEl(lx + 17, ly + 8, "no data", "tick", "start"));
      }
    }
    // hover: per-cell tooltip + outline
    var hot = S("rect", { "class": "cell-hot", rx: 2, visibility: "hidden" });
    svgCells.appendChild(hot);
    var hit = S("rect", { x: gx, y: top, width: gridW, height: nr * cellH, "class": "hit" });
    svgCells.appendChild(hit);
    if (scroller) {
      FL.ui.scrollCue(scroller);
      // open a time axis at its newest end; keep the reader's place on a redraw
      var fromEnd = inst._heatFromEnd;
      if (fromEnd === undefined && !isTime) fromEnd = null;
      if (fromEnd !== null) scroller.scrollLeft = Math.max(0, scroller.scrollWidth - scroller.clientWidth - (fromEnd || 0));
      scroller.addEventListener("scroll", function () { inst._heatFromEnd = scroller.scrollWidth - scroller.clientWidth - scroller.scrollLeft; if (!inst._kbd) hide(); }, { passive: true });
    }
    var cur = null;
    function showAt(r, c) {
      r = Math.max(0, Math.min(nr - 1, r)); c = Math.max(0, Math.min(nc - 1, c));
      cur = [r, c];
      var x = gx + c * cellW, y = top + r * cellH;
      if (scroller && inst._kbd) { // keyboard: bring the cell into the scroll box
        var vw = scroller.clientWidth;
        if (x < scroller.scrollLeft || x + cellW > scroller.scrollLeft + vw) scroller.scrollLeft = Math.max(0, x - vw / 2);
      }
      hot.setAttribute("x", x - 0.75); hot.setAttribute("y", y - 0.75);
      hot.setAttribute("width", Math.max(1, cellW - gap) + 1.5); hot.setAttribute("height", Math.max(1, cellH - gap) + 1.5);
      hot.setAttribute("visibility", "visible");
      var v = vals[r][c];
      var content = el("div");
      content.appendChild(el("div", { "class": "fl-tip-row" }, [el("span", { "class": "v", text: v === null ? "no data" : inst.vf(v) })]));
      content.appendChild(el("div", { "class": "fl-tip-head", style: { marginBottom: "0", marginTop: "3px" }, text: rowLabels[r] + " · " + colLabels[c] }));
      var tx = scroller ? lw + x + cellW / 2 - scroller.scrollLeft : x + cellW / 2 - plot.scrollLeft;
      showTip(inst, tx, y + cellH / 2, content);
    }
    function hide() { hot.setAttribute("visibility", "hidden"); hideTip(inst); }
    function locate(e) {
      var p = svgPoint(svgCells, e);
      return [Math.floor((p.y - top) / cellH), Math.floor((p.x - gx) / cellW)];
    }
    bindHover(inst, hit, function (e) { var rc = locate(e); showAt(rc[0], rc[1]); }, hide);
    inst.nav = {
      key: function (k) {
        var r = cur ? cur[0] : 0, c = cur ? cur[1] : (isTime ? nc - 1 : 0);
        if (k === "ArrowRight") c += 1; else if (k === "ArrowLeft") c -= 1;
        else if (k === "ArrowDown") r += 1; else if (k === "ArrowUp") r -= 1;
        else if (k === "Home") c = 0; else if (k === "End") c = nc - 1;
        else if (k === "Escape") { hide(); return true; }
        else return false;
        showAt(r, c);
        return true;
      },
      focus: function () { showAt(cur ? cur[0] : 0, cur ? cur[1] : (isTime ? nc - 1 : 0)); },
      blur: hide
    };
  }
  function heatTable(inst) {
    var o = inst.opts;
    var cols = [{ key: "row", label: o.rowLabel || "", sortValue: function (r) { return r._i; } }];
    inst.cols.forEach(function (c, i) { cols.push({ key: "c" + i, label: String(c), format: inst.vf, align: "right" }); });
    var rows = inst.rows.map(function (rl, r) {
      var row = { _i: r, row: inst.rowFmt(rl) };
      inst.cols.forEach(function (_, c) { row["c" + c] = inst.vals[r][c]; });
      return row;
    });
    return { columns: cols, rows: rows, caption: o.tableCaption || null };
  }
  function heatSummary(inst) {
    var d = inst.domain;
    return "Heatmap, " + inst.rows.length + " rows by " + inst.cols.length + " columns, " + (inst.scaleType === "div" ? "diverging" : "sequential") +
      " color scale from " + inst.vf(d[0]) + " to " + inst.vf(d[d.length - 1]) + ".";
  }


  /* ====================================================================== */
  /* scatter                                                                 */
  /* ====================================================================== */

  function prepareScatter(inst) {
    var o = inst.opts;
    inst.series = (o.series || []).map(function (s, i) {
      var pts = [];
      (s.points || s.values || []).forEach(function (p) {
        var x, y, label;
        if (Array.isArray(p)) { x = p[0]; y = p[1]; label = p[2]; }
        else if (p && typeof p === "object") { x = p.x; y = p.y; label = p.label; }
        x = toNum(x); y = toNum(y);
        if (x === null || y === null) return;
        pts.push({ x: x, y: y, label: label === undefined || label === null ? "" : String(label) });
      });
      return { id: s.id || "p" + i, name: seriesName(s, i), color: seriesColor(s, i, o.context), pts: pts };
    });
    var xs = o.x || {}, ys = o.y || {};
    inst.xf = fmt.get(xs.tooltipFormat || xs.format || "num", xs.signed !== undefined ? { signed: xs.signed } : null);
    inst.yf = fmt.get(ys.tooltipFormat || ys.format || "num", ys.signed !== undefined ? { signed: ys.signed } : null);
  }
  function scatterLegend(inst) {
    if (inst.series.length < 2 || inst.opts.legend === false) return null;
    return inst.series.map(function (s) { return { name: s.name, color: s.color, kind: "dot" }; });
  }
  function drawScatter(inst, W) {
    var o = inst.opts, xo = o.x || {}, yo = o.y || {}, ss = inst.series;
    var all = [];
    ss.forEach(function (s) { s.pts.forEach(function (p) { all.push({ s: s, p: p }); }); });
    if (!all.length) { message(inst, o.emptyText || "No data to plot."); inst.nav = null; return; }
    var H = o.height || 280;
    function ext(key, opt, extra) {
      var lo = Infinity, hi = -Infinity;
      all.forEach(function (a) { lo = Math.min(lo, a.p[key]); hi = Math.max(hi, a.p[key]); });
      (extra || []).forEach(function (v) { v = toNum(v); if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
      if (opt.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
      var d = opt.domain || [], fixed = false;
      if (d[0] !== undefined && d[0] !== null) { lo = d[0]; fixed = true; }
      if (d[1] !== undefined && d[1] !== null) { hi = d[1]; fixed = true; }
      return { lo: lo, hi: hi, fixed: fixed };
    }
    var ey = ext("y", yo, (o.refLines || []).map(function (r) { return r.y; }));
    var ex = ext("x", xo, (o.vlines || []).map(function (r) { return r.x; }));
    var top = 10 + (yo.label ? 18 : 0), bottom = 26 + (xo.label ? 16 : 0);
    var plotH = Math.max(40, H - top - bottom);
    var yt = niceTicks(ey.lo, ey.hi, Math.max(2, Math.round(plotH / 52)), ey.fixed);
    var ytf = tickFormatter(yo.tickFormat || yo.format, yt.step, false);
    var ylabels = yt.ticks.map(ytf);
    var left = Math.max(28, Math.ceil(Math.max.apply(null, ylabels.map(function (t) { return textWidth(t, 11); }))) + 10);
    var right = 14;
    var plotW = Math.max(40, W - left - right);
    var xt = niceTicks(ex.lo, ex.hi, Math.max(2, Math.floor(plotW / 90)), ex.fixed);
    var xtf = tickFormatter(xo.tickFormat || xo.format, xt.step, false);
    var xS = linear(xt.lo, xt.hi, left, left + plotW), yS = linear(yt.lo, yt.hi, top + plotH, top);
    var svg = makeSvg(inst, W, H);
    yt.ticks.forEach(function (t, i) {
      var y = px(yS(t));
      svg.appendChild(S("line", { x1: left, x2: left + plotW, y1: y, y2: y, "class": t === 0 ? "axis zero" : "grid" }));
      svg.appendChild(textEl(left - 8, y, ylabels[i], "tick", "end", { dy: "0.32em" }));
    });
    var gx = S("g");
    xt.ticks.forEach(function (t) {
      var x = px(xS(t));
      gx.appendChild(S("line", { x1: x, x2: x, y1: top, y2: top + plotH, "class": t === 0 ? "axis zero" : "grid" }));
    });
    drawXTicks(gx, xt.ticks.map(function (t) { return { v: t, label: xtf(t) }; }), xS, top + plotH + 17, W);
    svg.appendChild(gx);
    svg.appendChild(S("line", { x1: left, x2: left + plotW, y1: px(top + plotH), y2: px(top + plotH), "class": "axis" }));
    axisTitle(svg, yo.label, 0, W);
    if (xo.label) svg.appendChild(textEl(left + plotW, H - 4, truncate(xo.label, left + plotW, 11), "lbl", "end"));
    // reference lines / vlines: hairlines under the points, labels in a layer above them
    var labeler = makeLabeler([0, 0, W, H]);
    var gLabels = S("g", { "class": "labels" });
    var mn = { minX: left, maxX: W };
    (o.vlines || []).forEach(function (r) {
      var v = toNum(r.x); if (v === null || v < xt.lo || v > xt.hi) return;
      var x = px(xS(v));
      svg.appendChild(S("line", { x1: x, x2: x, y1: top, y2: top + plotH, "class": "ref", style: r.color ? { stroke: r.color } : null }));
      // right of the line; flipped to its left when it would run past the chart's right edge
      if (r.label) drawPlaced(gLabels, labeler.place(r.label, [
        assign({ x: x + 4, y: top + 10, anchor: "start" }, mn), assign({ x: x - 4, y: top + 10, anchor: "end" }, mn),
        assign({ x: x + 4, y: top + 23, anchor: "start" }, mn), assign({ x: x - 4, y: top + 23, anchor: "end" }, mn)
      ]));
    });
    (o.refLines || []).forEach(function (r) {
      var v = toNum(r.y); if (v === null || v < yt.lo || v > yt.hi) return;
      var y = px(yS(v));
      svg.appendChild(S("line", { x1: left, x2: left + plotW, y1: y, y2: y, "class": "ref", style: r.color ? { stroke: r.color } : null }));
      if (r.label) {
        var rm = { minX: left, maxX: left + plotW };
        drawPlaced(gLabels, labeler.place(r.label, [
          assign({ x: left + plotW - 2, y: y - 4, anchor: "end" }, rm), assign({ x: left + plotW - 2, y: y + 12, anchor: "end" }, rm),
          assign({ x: left + 4, y: y - 4, anchor: "start" }, rm), assign({ x: left + 4, y: y + 12, anchor: "start" }, rm)
        ]));
      }
    });
    var R = o.r || 4;
    var nodes = [];
    var gp = S("g");
    all.forEach(function (a) {
      var cx = xS(a.p.x), cy = yS(a.p.y);
      gp.appendChild(S("circle", { cx: cx, cy: cy, r: R, "stroke-width": 2, style: { fill: a.s.color, stroke: "var(--surface)" } }));
      nodes.push({ a: a, x: cx, y: cy });
    });
    svg.appendChild(gp);
    // labels: true = every labelled point; 'auto' (default <= 40 points) = greedy, skip collisions
    // (point labels also avoid the reference/vline labels placed above)
    var mode = o.labels === undefined ? (all.length <= 40 ? "auto" : false) : o.labels;
    if (mode) {
      var boxes = labeler.boxes.slice();
      var order = nodes.slice().sort(function (p, q) { return Math.abs(q.a.p.y) - Math.abs(p.a.p.y); });
      order.forEach(function (n) {
        var t = n.a.p.label; if (!t) return;
        var w = textWidth(t, 11), h = 12;
        var cands = [[n.x + R + 4, n.y, "start"], [n.x - R - 4, n.y, "end"]];
        for (var c = 0; c < cands.length; c++) {
          var bx = cands[c][2] === "start" ? cands[c][0] : cands[c][0] - w;
          var box = [bx, n.y - h / 2, bx + w, n.y + h / 2];
          if (box[0] < left || box[2] > left + plotW + right) continue;
          var hit = mode !== true && (boxes.some(function (b) { return !(box[2] < b[0] || box[0] > b[2] || box[3] < b[1] || box[1] > b[3]); }) ||
            nodes.some(function (m) { return m !== n && m.x + R > box[0] && m.x - R < box[2] && m.y + R > box[1] && m.y - R < box[3]; }));
          if (hit) continue;
          boxes.push(box);
          gLabels.appendChild(textEl(cands[c][0], n.y, t, "lbl halo", cands[c][2], { dy: "0.32em" }));
          break;
        }
      });
    }
    svg.appendChild(gLabels);
    var ring = S("circle", { r: R + 3, fill: "none", "stroke-width": 1.5, visibility: "hidden", style: { stroke: "var(--ink)" } });
    svg.appendChild(ring);
    var hit = S("rect", { x: left, y: top, width: plotW, height: plotH, "class": "hit" });
    svg.appendChild(hit);
    var byX = nodes.slice().sort(function (p, q) { return p.x - q.x || p.y - q.y; });
    var cur = -1;
    function showNode(i) {
      if (i < 0 || i >= byX.length) return;
      cur = i;
      var n = byX[i];
      ring.setAttribute("cx", n.x); ring.setAttribute("cy", n.y); ring.setAttribute("visibility", "visible");
      var c = el("div");
      c.appendChild(el("div", { "class": "fl-tip-head", text: n.a.p.label || n.a.s.name }));
      c.appendChild(tipRow(n.a.s.color, "dot", inst.yf(n.a.p.y), yo.label || "y"));
      c.appendChild(tipRow(null, "dot", inst.xf(n.a.p.x), xo.label || "x"));
      if (ss.length > 1 && n.a.p.label) c.appendChild(el("div", { "class": "fl-tip-head", style: { margin: "4px 0 0" }, text: n.a.s.name }));
      showTip(inst, n.x, n.y, c);
    }
    function hide() { ring.setAttribute("visibility", "hidden"); hideTip(inst); }
    function nearest(e) {
      var p = svgPoint(svg, e), best = -1, bd = 24 * 24;
      byX.forEach(function (n, i) { var d = (n.x - p.x) * (n.x - p.x) + (n.y - p.y) * (n.y - p.y); if (d < bd) { bd = d; best = i; } });
      return best;
    }
    bindHover(inst, hit, function (e) { var i = nearest(e); if (i < 0) { if (e.type === "pointermove" && !inst._touch) hide(); } else showNode(i); }, hide);
    inst.nav = navFor(function () { return byX.length; }, function () { return cur; }, showNode, hide);
  }
  function scatterTable(inst) {
    var o = inst.opts, xo = o.x || {}, yo = o.y || {}, multi = inst.series.length > 1;
    var cols = [{ key: "label", label: o.pointLabel || "Point" }];
    if (multi) cols.push({ key: "series", label: "Series" });
    cols.push({ key: "x", label: xo.label || "x", format: inst.xf, align: "right" });
    cols.push({ key: "y", label: yo.label || "y", format: inst.yf, align: "right" });
    var rows = [];
    inst.series.forEach(function (s) { s.pts.forEach(function (p) { rows.push({ label: p.label, series: s.name, x: p.x, y: p.y }); }); });
    return { columns: cols, rows: rows, caption: o.tableCaption || null };
  }
  function scatterSummary(inst) {
    var n = 0; inst.series.forEach(function (s) { n += s.pts.length; });
    var o = inst.opts;
    return "Scatter plot, " + n + " points" + ((o.x || {}).label ? ", x: " + o.x.label : "") + ((o.y || {}).label ? ", y: " + o.y.label : "") + ".";
  }

  /* ====================================================================== */
  /* sparkline (no axes, no table toggle: it lives inside a KPI tile / cell) */
  /* ====================================================================== */

  function sparkline(container, o) {
    container = resolveContainer(container);
    o = o || {};
    var host = el("div", { "class": "fl-spark" });
    container.appendChild(host);
    var vals = (o.values || []).map(function (p) { return Array.isArray(p) ? toNum(p[1]) : toNum(p); });
    var color = o.color || colors.entity(o.entity, o.context) || "var(--c1)";
    var dashed = o.dashed !== undefined ? o.dashed : colors.isBenchmark(o.entity);
    var H = o.height || 28;
    var lastW = 0;
    var finite = vals.filter(function (v) { return v !== null; });
    var label = o.ariaLabel || "";
    if (finite.length) {
      var f = fmt.get(o.format || "num");
      label = (label ? label + ": " : "") + "trend from " + f(finite[0]) + " to " + f(finite[finite.length - 1]) +
        ", low " + f(Math.min.apply(null, finite)) + ", high " + f(Math.max.apply(null, finite));
    }
    host.setAttribute("role", "img");
    host.setAttribute("aria-label", label || "sparkline");
    function draw() {
      var W = Math.floor(host.clientWidth);
      if (!W || W === lastW) return;
      lastW = W;
      ui.clear(host);
      if (!finite.length) return;
      var lo = Math.min.apply(null, finite), hi = Math.max.apply(null, finite);
      if (o.refLine !== undefined && o.refLine !== null) { lo = Math.min(lo, o.refLine); hi = Math.max(hi, o.refLine); }
      if (o.domain) { if (o.domain[0] !== null && o.domain[0] !== undefined) lo = o.domain[0]; if (o.domain[1] !== null && o.domain[1] !== undefined) hi = o.domain[1]; }
      if (lo === hi) { lo -= 1; hi += 1; }
      var pad = 6;
      var xS = linear(0, Math.max(1, vals.length - 1), 2, W - pad), yS = linear(lo, hi, H - pad, pad);
      var svg = S("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, "aria-hidden": "true" });
      if (o.refLine !== undefined && o.refLine !== null) svg.appendChild(S("line", { x1: 0, x2: W, y1: px(yS(o.refLine)), y2: px(yS(o.refLine)), "class": "ref" }));
      var d = "", started = false, area = "", run = [];
      function flushArea() {
        if (!o.area || !run.length) { run = []; return; }
        var by = yS(Math.min(Math.max(0, lo), hi));
        area += "M" + run[0][0] + "," + by + run.map(function (p) { return "L" + p[0] + "," + p[1]; }).join("") + "L" + run[run.length - 1][0] + "," + by + "Z";
        run = [];
      }
      vals.forEach(function (v, i) {
        if (v === null) { started = false; flushArea(); return; }
        var X = xS(i).toFixed(1), Y = yS(v).toFixed(1);
        d += (started ? "L" : "M") + X + "," + Y; started = true;
        run.push([X, Y]);
      });
      flushArea();
      if (area) svg.appendChild(S("path", { d: area, style: { fill: color, fillOpacity: 0.1 } }));
      svg.appendChild(S("path", { d: d, fill: "none", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": dashed ? "5 3" : null, style: { stroke: color } }));
      var li = -1;
      for (var i = vals.length - 1; i >= 0; i--) if (vals[i] !== null) { li = i; break; }
      if (li >= 0 && o.endDot !== false) svg.appendChild(S("circle", { cx: xS(li), cy: yS(vals[li]), r: 4, "stroke-width": 2, style: { fill: color, stroke: "var(--surface)" } }));
      host.appendChild(svg);
    }
    var ro = null, timer = null;
    var onResize = function () { draw(); };
    if (typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(function () { clearTimeout(timer); timer = setTimeout(function () { if (!host.isConnected) { prune(); return; } draw(); }, lastW ? 120 : 0); });
      ro.observe(host);
    } else window.addEventListener("resize", onResize);
    // A sparkline joins the chart list: the one window theme listener redraws it, and prune()
    // drops it with the page (no listener per sparkline, so a route change leaks nothing).
    var inst = {
      root: host, destroyed: false,
      render: function () { lastW = 0; draw(); },
      _disconnect: function () { if (ro) ro.disconnect(); ro = null; clearTimeout(timer); window.removeEventListener("resize", onResize); }
    };
    prune();
    live.push(inst);
    draw();
    return { root: host, el: host, destroy: function () { inst._disconnect(); live = live.filter(function (c) { return c !== inst; }); if (host.parentNode) host.parentNode.removeChild(host); } };
  }

  /* ====================================================================== */
  /* small multiples                                                         */
  /* ====================================================================== */

  function smallMultiples(container, specs, o) {
    container = resolveContainer(container);
    o = o || {};
    var maxCols = o.columns || 3, minW = o.minWidth || 260;
    var root = el("div", { "class": ["fl-sm", o.className] });
    container.appendChild(root);
    specs = (specs || []).map(function (s) { return assign({}, s); });
    if (o.sharedY) {
      var lo = Infinity, hi = -Infinity;
      specs.forEach(function (s) {
        var type = s.type || o.type || "line";
        if (type === "line") (s.series || []).forEach(function (se) { (se.values || []).forEach(function (p) { var v = toNum(Array.isArray(p) ? p[1] : p && typeof p === "object" ? (p.y !== undefined ? p.y : p.value) : p); if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }); });
        else if (type === "bars") {
          var tmp = { xs: s.x || [], mode: s.mode || (s.group ? "group" : "stack"), stacks: (s.stacks || s.series || []).map(function (st) { return { values: (s.x || []).map(function (_, j) { return toNum(st.values[j]); }) }; }) };
          var e = barExtent(tmp); lo = Math.min(lo, e[0]); hi = Math.max(hi, e[1]);
        }
      });
      if (isFinite(lo)) specs.forEach(function (s) { s.y = assign({}, o.y || {}, s.y || {}); s.y.domain = s.y.domain || [lo, hi]; });
    }
    var cells = [], charts = [];
    specs.forEach(function () { var c = el("div", { "class": "fl-sm-cell" }); root.appendChild(c); cells.push(c); });
    function layout() {
      var W = root.clientWidth;
      if (!W) return;
      var n = Math.max(1, Math.min(maxCols, Math.floor((W + 24) / (minW + 24)), specs.length || 1));
      root.style.gridTemplateColumns = "repeat(" + n + ", minmax(0, 1fr))";
    }
    layout();
    if (typeof ResizeObserver !== "undefined") { var ro = new ResizeObserver(function () { if (!root.isConnected) { ro.disconnect(); return; } layout(); }); ro.observe(root); }
    specs.forEach(function (s, i) {
      var type = s.type || o.type || "line";
      var spec = assign({ height: o.height || 180 }, s);
      if (o.y && !s.y) spec.y = o.y;
      var fn = FL.charts[type];
      if (typeof fn !== "function") throw new Error("smallMultiples: unknown chart type " + type);
      charts.push(fn(cells[i], spec));
    });
    return { root: root, charts: charts };
  }

  /* ====================================================================== */
  /* public API                                                              */
  /* ====================================================================== */

  FL.charts = {
    /** Multi-series time (or linear) line chart. See README for options. */
    line: function (container, opts) {
      return createChart(container, opts, { kind: "line", prepare: prepareLine, draw: drawLine, legend: lineLegend, table: lineTable, summary: lineSummary });
    },
    /** Vertical (time or category x) or horizontal (category) bars; diverging stacks around zero or grouped. */
    bars: function (container, opts) {
      return createChart(container, opts, { kind: "bars", prepare: prepareBars, draw: drawBars, legend: barsLegend, table: barsTable, summary: barsSummary });
    },
    /** x/y scatter with nearest-point hover (24px), greedy non-colliding labels. */
    scatter: function (container, opts) {
      return createChart(container, opts, { kind: "scatter", prepare: prepareScatter, draw: drawScatter, legend: scatterLegend, table: scatterTable, summary: scatterSummary });
    },
    /** rows × cols color grid with a sequential or diverging scale. */
    heatmap: function (container, opts) {
      return createChart(container, opts, { kind: "heatmap", prepare: prepareHeat, draw: drawHeat, legend: null, table: heatTable, summary: heatSummary });
    },
    sparkline: sparkline,
    smallMultiples: smallMultiples,
    /** re-render every live chart (e.g. after a layout change) */
    refresh: function () { prune(); live.forEach(function (c) { c.render(); }); },
    /** drop charts whose DOM was removed (the router calls this on navigation) */
    prune: prune,
    /** internals exposed for page-local components (stable, documented) */
    util: { textWidth: textWidth, truncate: truncate, niceTicks: niceTicks, logTicks: logTicks, timeTicks: timeTicks, tickFormatter: tickFormatter, linear: linear, logScale: logScale, barPath: barPath }
  };
})();
