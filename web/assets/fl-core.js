/* Factorlab dashboard — core: FL.fmt, FL.colors, FL.theme, FL.ui, FL.pages.
 *
 * Classic script (no modules: they are blocked on file://). ES2019. Everything
 * that comes from data is inserted with textContent / createTextNode only.
 * API reference: web/README.md.
 *
 * Copyright (c) 2026 Radioiic. MIT License (LICENSE). Third-party icons
 * (THIRD_PARTY_NOTICES.md has their full license texts): Feather, MIT, Copyright (c)
 * 2013-2023 Cole Bemis; Lucide, ISC, Copyright (c) 2026 Lucide Icons and Contributors.
 */
(function () {
  "use strict";

  var FL = (window.FL = window.FL || {});
  FL.data = FL.data || {};

  var MINUS = "−";
  var NA = "n/a";
  var SVGNS = "http://www.w3.org/2000/svg";

  /* ====================================================================== */
  /* helpers                                                                 */
  /* ====================================================================== */

  function toNum(v) {
    if (v === null || v === undefined || v === "" || typeof v === "boolean") return null;
    var n = typeof v === "number" ? v : Number(v);
    return isFinite(n) ? n : null;
  }
  function isObj(o) { return o !== null && typeof o === "object" && !Array.isArray(o); }
  function assign(target) {
    for (var i = 1; i < arguments.length; i++) {
      var src = arguments[i];
      if (!src) continue;
      for (var k in src) if (Object.prototype.hasOwnProperty.call(src, k)) target[k] = src[k];
    }
    return target;
  }
  function groupDigits(intStr) { return intStr.replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function fixedAbs(abs, dp) {
    var s = abs.toFixed(dp);
    var parts = s.split(".");
    return groupDigits(parts[0]) + (parts.length > 1 ? "." + parts[1] : "");
  }
  // Sign prefix. A value that rounds to zero carries no sign.
  function signed(v, dp, body, isSigned) {
    var r = Number(Math.abs(v).toFixed(dp));
    if (r === 0) return body;
    if (v < 0) return MINUS + body;
    return isSigned ? "+" + body : body;
  }
  function optsOf(o) {
    if (o === true) return { signed: true };
    if (o === false) return { signed: false };
    return isObj(o) ? o : {};
  }
  function dpOr(dp, d) { return dp === undefined || dp === null ? d : dp; }

  FL.util = { toNum: toNum, assign: assign, isObj: isObj };

  /* ====================================================================== */
  /* FL.fmt                                                                  */
  /* ====================================================================== */

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  var DAY_MS = 864e5;

  // "2026-10-01" or ISO datetime -> epoch ms (UTC). Numbers pass through.
  function parseDate(s) {
    if (s === null || s === undefined) return null;
    if (typeof s === "number") return isFinite(s) ? s : null;
    if (s instanceof Date) return s.getTime();
    var str = String(s);
    var m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(str);
    if (m) return Date.UTC(+m[1], +m[2] - 1, +m[3]);
    var t = Date.parse(/[zZ]|[+-]\d\d:?\d\d$/.test(str) || str.length <= 10 ? str : str + "Z");
    return isFinite(t) ? t : null;
  }
  function isoDate(ms) {
    var d = new Date(ms);
    return d.getUTCFullYear() + "-" + String(d.getUTCMonth() + 1).padStart(2, "0") + "-" + String(d.getUTCDate()).padStart(2, "0");
  }

  var fmt = {
    MINUS: MINUS,
    NA: NA,
    parseDate: parseDate,
    isoDate: isoDate,

    /** 0.0123 -> "+1.23%" (signed by default); null -> "n/a". Input is a DECIMAL fraction. */
    pct: function (v, dp, o) {
      v = toNum(v); if (v === null) return NA;
      dp = dpOr(dp, 2); o = optsOf(o);
      var isSigned = o.signed === undefined ? true : !!o.signed;
      return signed(v * 100, dp, fixedAbs(Math.abs(v * 100), dp) + "%", isSigned);
    },
    /** 0.0123 -> "+1.23pp" (percentage points from a decimal difference). */
    pp: function (v, dp, o) {
      v = toNum(v); if (v === null) return NA;
      dp = dpOr(dp, 2); o = optsOf(o);
      var isSigned = o.signed === undefined ? true : !!o.signed;
      return signed(v * 100, dp, fixedAbs(Math.abs(v * 100), dp) + "pp", isSigned);
    },
    /** 1234.57 -> "$1,234.57"; signed: "+$34.99" / "−$39.49". Negative is always "−$…". */
    usd: function (v, dp, o) {
      v = toNum(v); if (v === null) return NA;
      dp = dpOr(dp, 2); o = optsOf(o);
      return signed(v, dp, "$" + fixedAbs(Math.abs(v), dp), !!o.signed);
    },
    /** 1234.5 -> "1,234.50"; signed option. */
    num: function (v, dp, o) {
      v = toNum(v); if (v === null) return NA;
      dp = dpOr(dp, 2); o = optsOf(o);
      return signed(v, dp, fixedAbs(Math.abs(v), dp), !!o.signed);
    },
    /** 1234.6 -> "1,235". */
    int: function (v, o) {
      v = toNum(v); if (v === null) return NA;
      o = optsOf(o);
      return signed(v, 0, fixedAbs(Math.abs(Math.round(v)), 0), !!o.signed);
    },
    /** 12900 -> "12.9K", 1.5e6 -> "1.5M". opts {signed, prefix:'$'} */
    compact: function (v, dp, o) {
      v = toNum(v); if (v === null) return NA;
      dp = dpOr(dp, 1); o = optsOf(o);
      var a = Math.abs(v), unit = "", d = a;
      if (a >= 1e12) { d = a / 1e12; unit = "T"; }
      else if (a >= 1e9) { d = a / 1e9; unit = "B"; }
      else if (a >= 1e6) { d = a / 1e6; unit = "M"; }
      else if (a >= 1e3) { d = a / 1e3; unit = "K"; }
      var body = unit ? String(Number(d.toFixed(dp))) : fixedAbs(a, a < 10 && a !== Math.round(a) ? dp : 0);
      return signed(v, unit ? dp : 2, (o.prefix || "") + body + unit, !!o.signed);
    },
    /** index level, 100 = start: 85.1449 -> "85.1". */
    idx: function (v, dp) { return fmt.num(v, dpOr(dp, 1)); },
    /** multiplier: 0.75 -> "0.75x" (engine style for the advisory gross multiplier). */
    x: function (v, dp) {
      v = toNum(v); if (v === null) return NA;
      return fmt.num(v, dpOr(dp, 2)) + "x";
    },
    /** equity multiple for log axes: 0.5 -> "×0.5", 2 -> "×2". */
    mult: function (v, dp) {
      v = toNum(v); if (v === null) return NA;
      var s = dp === undefined ? String(Number(v.toPrecision(3))) : v.toFixed(dp);
      return "×" + s;
    },
    /** basis points from a decimal: 0.0005 -> "5.0 bp". */
    bps: function (v, dp, o) {
      v = toNum(v); if (v === null) return NA;
      dp = dpOr(dp, 1); o = optsOf(o);
      return signed(v * 1e4, dp, fixedAbs(Math.abs(v * 1e4), dp) + " bp", !!o.signed);
    },
    /** "2026-10-01" -> "1 Oct 2026". opts.style: long | short ("1 Oct") | axis ("Oct 1") | iso | weekday ("Thu 1 Oct 2026") | month ("Oct 2026") */
    date: function (s, o) {
      var ms = parseDate(s); if (ms === null) return NA;
      var d = new Date(ms), style = (optsOf(o).style) || (typeof o === "string" ? o : "long");
      var day = d.getUTCDate(), mon = MONTHS[d.getUTCMonth()], yr = d.getUTCFullYear();
      switch (style) {
        case "short": return day + " " + mon;
        case "axis": return mon + " " + day;
        case "iso": return isoDate(ms);
        case "month": return mon + " " + yr;
        case "weekday": return WEEKDAYS[d.getUTCDay()] + " " + day + " " + mon + " " + yr;
        default: return day + " " + mon + " " + yr;
      }
    },
    /** ISO datetime -> "2026-10-02 05:53 UTC". */
    datetime: function (s) {
      var ms = parseDate(s); if (ms === null) return NA;
      var d = new Date(ms);
      return isoDate(ms) + " " + String(d.getUTCHours()).padStart(2, "0") + ":" + String(d.getUTCMinutes()).padStart(2, "0") + " UTC";
    },
    /** duration in ms -> "45 min" | "14 h" | "2 d 3 h". */
    age: function (ms) {
      ms = toNum(ms); if (ms === null) return NA;
      var neg = ms < 0; ms = Math.abs(ms);
      var min = Math.floor(ms / 6e4), h = Math.floor(min / 60), d = Math.floor(h / 24), out;
      if (min < 1) out = "just now";
      else if (min < 60) out = min + " min";
      else if (h < 48) out = h + " h";
      else out = d + " d" + (h % 24 ? " " + (h % 24) + " h" : "");
      return neg ? "in " + out : out;
    },
    /** 12345 -> "12.1 KB". */
    bytes: function (n) {
      n = toNum(n); if (n === null) return NA;
      if (n < 1024) return n + " B";
      if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
      return (n / 1024 / 1024).toFixed(2) + " MB";
    },
    /** plain text; null -> "n/a". */
    text: function (v) { return v === null || v === undefined || v === "" ? NA : String(v); },

    /**
     * Resolve a format spec into a function(v) -> string.
     * spec: function | "pct" | "pct:1" | "usd:0" | "num:4" | "pct:2:unsigned" | "usd::signed"
     *       | {kind, dp, signed} | null (auto: numbers -> num:2, else text).
     */
    get: function (spec, defaults) {
      if (typeof spec === "function") return spec;
      var kind, dp, sgn;
      if (isObj(spec)) { kind = spec.kind; dp = spec.dp; sgn = spec.signed; }
      else if (typeof spec === "string") {
        var p = spec.split(":");
        kind = p[0];
        if (p[1] !== undefined && p[1] !== "") dp = +p[1];
        if (p[2] === "signed") sgn = true;
        if (p[2] === "unsigned") sgn = false;
      }
      if (defaults) {
        if (dp === undefined && defaults.dp !== undefined) dp = defaults.dp;
        if (sgn === undefined && defaults.signed !== undefined) sgn = defaults.signed;
      }
      var so = sgn === undefined ? undefined : { signed: sgn };
      switch (kind) {
        case "pct": return function (v) { return fmt.pct(v, dp, so); };
        case "pp": return function (v) { return fmt.pp(v, dp, so); };
        case "usd": return function (v) { return fmt.usd(v, dp, so); };
        case "usdc": return function (v) { return fmt.compact(v, dp, assign({ prefix: "$" }, so)); };
        case "num": return function (v) { return fmt.num(v, dp, so); };
        case "int": return function (v) { return fmt.int(v, so); };
        case "compact": return function (v) { return fmt.compact(v, dp, so); };
        case "idx": return function (v) { return fmt.idx(v, dp); };
        case "x": return function (v) { return fmt.x(v, dp); };
        case "mult": return function (v) { return fmt.mult(v, dp); };
        case "bps": return function (v) { return fmt.bps(v, dp, so); };
        case "date": return function (v) { return fmt.date(v); };
        case "dateShort": return function (v) { return fmt.date(v, "short"); };
        case "datetime": return function (v) { return fmt.datetime(v); };
        case "bytes": return function (v) { return fmt.bytes(v); };
        case "text": case "raw": return function (v) { return fmt.text(v); };
        default:
          return function (v) {
            if (typeof v === "number") return fmt.num(v, dp === undefined ? 2 : dp, so);
            return fmt.text(v);
          };
      }
    },
    /** true for kinds that produce right-aligned numbers. */
    isNumericKind: function (spec) {
      var k = typeof spec === "string" ? spec.split(":")[0] : isObj(spec) ? spec.kind : null;
      return ["pct", "pp", "usd", "usdc", "num", "int", "compact", "idx", "x", "mult", "bps", "bytes"].indexOf(k) >= 0;
    },
    kindOf: function (spec) {
      return typeof spec === "string" ? spec.split(":")[0] : isObj(spec) ? spec.kind : null;
    }
  };
  FL.fmt = fmt;

  /* ====================================================================== */
  /* FL.theme                                                                */
  /* ====================================================================== */

  var THEME_KEY = "fl-theme";
  var mql = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  function readStored() {
    try { var t = window.localStorage.getItem(THEME_KEY); return t === "light" || t === "dark" ? t : "auto"; }
    catch (e) { return "auto"; }
  }
  var themeMode = readStored();

  var theme = {
    /** "auto" | "light" | "dark" (the user's choice). */
    get: function () { return themeMode; },
    /** "light" | "dark" (what is actually on screen). */
    effective: function () {
      if (themeMode === "light" || themeMode === "dark") return themeMode;
      return mql && mql.matches ? "dark" : "light";
    },
    set: function (mode) {
      if (mode !== "light" && mode !== "dark") mode = "auto";
      themeMode = mode;
      try {
        if (mode === "auto") window.localStorage.removeItem(THEME_KEY);
        else window.localStorage.setItem(THEME_KEY, mode);
      } catch (e) { /* storage blocked: choice lasts for this page view */ }
      if (mode === "auto") document.documentElement.removeAttribute("data-theme");
      else document.documentElement.setAttribute("data-theme", mode);
      theme._emit();
    },
    _emit: function () {
      colors._cache = null;
      var ev;
      try { ev = new CustomEvent("fl:themechange", { detail: { mode: themeMode, effective: theme.effective() } }); }
      catch (e) { ev = document.createEvent("CustomEvent"); ev.initCustomEvent("fl:themechange", false, false, { mode: themeMode, effective: theme.effective() }); }
      window.dispatchEvent(ev);
    }
  };
  if (themeMode !== "auto") document.documentElement.setAttribute("data-theme", themeMode);
  if (mql) {
    var onOS = function () { if (themeMode === "auto") theme._emit(); };
    if (mql.addEventListener) mql.addEventListener("change", onOS); else if (mql.addListener) mql.addListener(onOS);
  }
  FL.theme = theme;

  /* ====================================================================== */
  /* FL.colors                                                               */
  /* ====================================================================== */

  // entity id -> categorical slot (1..8) or "btc" (muted, dashed benchmark). The book of a
  // factor test, its control book and the net excess return over the control.
  var ENTITY = {
    factor_book: 1, control: 2, excess: 3,
    btc: "btc", btc_hold: "btc", BTC: "btc", benchmark: "btc", bench: "btc"
  };
  // context overrides: {context: {entity id: slot}} (none at present)
  var CONTEXT = {};
  var LABELS = {
    btc: "BTC hold", btc_hold: "BTC hold", BTC: "BTC hold", benchmark: "BTC hold", bench: "BTC hold",
    factor_book: "factor book", control: "control book", excess: "net excess"
  };

  // Ramps (hex). Sequential: blue 100 -> 700. Dark mode runs the same ramp
  // reversed so low magnitude recedes into the dark surface.
  var SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
  // Diverging stops at t = -1, -0.5, 0, +0.5, +1 (negative = red arm, positive = blue arm),
  // matched OKLab lightness per arm. On each surface the distance from the midpoint grows
  // with |t| in lightness AND in chroma, so a larger value never looks weaker. Dark arms
  // (OKLab L / C): 0.34 / 0 -> 0.50 / 0.12-0.13 -> 0.70 / 0.15-0.18.
  var DIV = {
    light: ["#8c2524", "#e34948", "#f0efec", "#3987e5", "#184f95"],
    dark: ["#fa6865", "#a03f3c", "#383835", "#3463a6", "#5ba0fa"]
  };
  var STATIC = {
    light: { c1: "#2a78d6", c2: "#eb6834", c3: "#30a06e", c4: "#695f01", c5: "#d565b0", c6: "#008300", c7: "#4a3aa7", c8: "#e34948",
      surface: "#fcfcfb", ink: "#0b0b0b", "ink-2": "#52514e", "ink-3": "#6b6a65", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7" },
    dark: { c1: "#3987e5", c2: "#d95926", c3: "#199e70", c4: "#c98500", c5: "#d55181", c6: "#008300", c7: "#9085e9", c8: "#e66767",
      surface: "#1a1a19", ink: "#ffffff", "ink-2": "#c3c2b7", "ink-3": "#9a9890", muted: "#898781", grid: "#2c2c2a", axis: "#383835" }
  };

  // --- OKLab interpolation ---
  function hexToRgb(h) {
    h = String(h).replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
  }
  function lin(c) { return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  function delin(c) { return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; }
  function toOklab(hex) {
    var rgb = hexToRgb(hex).map(lin), r = rgb[0], g = rgb[1], b = rgb[2];
    var l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    var m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    var s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
      1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
      0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s];
  }
  function fromOklab(L) {
    var l = Math.pow(L[0] + 0.3963377774 * L[1] + 0.2158037573 * L[2], 3);
    var m = Math.pow(L[0] - 0.1055613458 * L[1] - 0.0638541728 * L[2], 3);
    var s = Math.pow(L[0] - 0.0894841775 * L[1] - 1.2914855480 * L[2], 3);
    var rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
    return "#" + rgb.map(function (c) {
      var v = Math.round(Math.max(0, Math.min(1, delin(c))) * 255);
      return (v < 16 ? "0" : "") + v.toString(16);
    }).join("");
  }
  var labCache = {};
  function lab(hex) { return labCache[hex] || (labCache[hex] = toOklab(hex)); }
  function interp(stops, t) { // t in [0,1] across evenly spaced stops
    if (!(t >= 0)) t = 0; if (t > 1) t = 1;
    var n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n)), f = t * n - i;
    var a = lab(stops[i]), b = lab(stops[i + 1]);
    return fromOklab([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]);
  }
  function luminance(hex) {
    var c = hexToRgb(hex).map(lin);
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  }
  function contrast(a, b) {
    var la = luminance(a), lb = luminance(b);
    return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
  }

  var colors = {
    ENTITY: ENTITY,
    CONTEXT: CONTEXT,
    SEQ: SEQ,
    DIV: DIV,
    _cache: null,

    /** CSS color for categorical slot n (1..8): "var(--c3)". Out of range -> muted (never cycles). */
    slot: function (n) {
      n = Math.round(+n);
      if (!(n >= 1 && n <= 8)) return "var(--muted)";
      return "var(--c" + n + ")";
    },
    /** Entity id -> CSS color string usable in any style, e.g. "var(--c2)". Unknown -> null. */
    entity: function (id, context) {
      var v = colors.entityVar(id, context);
      return v === null ? null : "var(" + v + ")";
    },
    /** Entity id -> the CSS custom property name, e.g. "--c2" (BTC -> "--muted"). Unknown -> null. */
    entityVar: function (id, context) {
      var s = colors._slotOf(id, context);
      if (s === null) return null;
      if (s === "btc") return "--muted";
      return "--c" + s;
    },
    /** {color, cssVar, slot, dashed, label} for an entity (null if unknown). slot is null for the BTC benchmark. */
    entityInfo: function (id, context) {
      var s = colors._slotOf(id, context);
      if (s === null) return null;
      return { color: colors.entity(id, context), cssVar: colors.entityVar(id, context), slot: typeof s === "number" ? s : null, dashed: s === "btc", label: colors.label(id) };
    },
    /** default display label for an entity id ("btc" -> "BTC hold"). */
    label: function (id) { return LABELS[id] || String(id); },
    /** true for the BTC benchmark (drawn muted + dashed "5 3"). */
    isBenchmark: function (id) { return ENTITY[id] === "btc"; },
    _slotOf: function (id, context) {
      if (id === null || id === undefined) return null;
      var c = context && CONTEXT[context];
      if (c && Object.prototype.hasOwnProperty.call(c, id)) return c[id];
      return Object.prototype.hasOwnProperty.call(ENTITY, id) ? ENTITY[id] : null;
    },

    /** status colors (icons/marks only; text stays in ink). */
    status: { good: "var(--st-good)", warning: "var(--st-warning)", serious: "var(--st-serious)", critical: "var(--st-critical)", info: "var(--st-info)", advisory: "var(--st-info)" },

    /** Resolve "var(--c1)" / "--c1" / "c1" / "#hex" to a hex string for the current theme. */
    resolve: function (c) {
      if (!c) return null;
      c = String(c).trim();
      if (c.charAt(0) === "#") return c;
      var m = /^var\(\s*(--[\w-]+)\s*(?:,[^)]*)?\)$/.exec(c);
      var name = m ? m[1] : c.indexOf("--") === 0 ? c : "--" + c;
      var cache = colors._cache || (colors._cache = {});
      if (cache[name]) return cache[name];
      var v = "";
      try { v = getComputedStyle(document.documentElement).getPropertyValue(name).trim(); } catch (e) { v = ""; }
      if (!v) {
        var st = STATIC[theme.effective()];
        v = st[name.slice(2)] || c;
      }
      cache[name] = v;
      return v;
    },
    /** "light" | "dark" */
    theme: function () { return theme.effective(); },

    /** Sequential ramp: t in [0,1] -> hex (light: pale -> deep blue; dark: dark -> light blue). */
    seq: function (t) {
      var stops = theme.effective() === "dark" ? SEQ.slice().reverse() : SEQ;
      return interp(stops, t);
    },
    /** Diverging ramp: t in [-1,1] -> hex (negative = red arm, 0 = gray midpoint, positive = blue arm). */
    div: function (t) {
      if (!(t >= -1)) t = t < -1 ? -1 : 0;
      if (t > 1) t = 1;
      return interp(DIV[theme.effective()], (t + 1) / 2);
    },
    /**
     * Value scale for heatmaps: {type:'seq'|'div', domain:[lo,hi] | [lo,mid,hi]} -> fn(v) -> hex | null (null = no data).
     */
    scale: function (o) {
      var type = o.type || "seq", d = o.domain;
      if (type === "div") {
        var lo = d[0], mid = d.length > 2 ? d[1] : 0, hi = d.length > 2 ? d[2] : d[1];
        return function (v) {
          v = toNum(v); if (v === null) return null;
          var t = v >= mid ? (hi === mid ? 0 : (v - mid) / (hi - mid)) : (lo === mid ? 0 : -(mid - v) / (mid - lo));
          return colors.div(t);
        };
      }
      return function (v) {
        v = toNum(v); if (v === null) return null;
        var t = d[1] === d[0] ? 0.5 : (v - d[0]) / (d[1] - d[0]);
        return colors.seq(t);
      };
    },
    /** Best text ink (#000000 or #ffffff) for a label drawn on a fill: the better one is >= 4.58:1 on any opaque fill. */
    inkOn: function (fillHex) {
      var h = colors.resolve(fillHex);
      if (!h || h.charAt(0) !== "#") return "#000000";
      return contrast(h, "#000000") >= contrast(h, "#ffffff") ? "#000000" : "#ffffff";
    },
    contrast: contrast,
    mix: function (a, b, t) { return interp([colors.resolve(a), colors.resolve(b)], t); }
  };
  FL.colors = colors;

  /* ====================================================================== */
  /* FL.cmd, FL.stale: the remedy commands and the data-age checks           */
  /* ====================================================================== */

  /** The two commands a page shows. setup adds the new closed days; export rebuilds only the pages. */
  FL.cmd = {
    setup: "./setup.sh",
    export: "cd engine && .venv/bin/python -B export_web.py"
  };

  function utcDayMs(ms) { var d = new Date(ms); return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()); }

  FL.stale = {
    /**
     * The age of the data against the viewer's clock (UTC days).
     * -> {through, today, ageDays, missingDays, stale, text} | null (no meta.data_through).
     * The bar of day D closes at 00:00 UTC on D + 1, so on day D + 1 the data is current
     * (missingDays 0). stale is true when at least one closed bar is not in the data.
     */
    data: function (meta, nowMs) {
      var through = meta && meta.data_through;
      var t = parseDate(through);
      if (t === null) return null;
      var today = utcDayMs(nowMs === undefined ? Date.now() : nowMs);
      var age = Math.round((today - t) / DAY_MS);
      var missing = Math.max(0, age - 1);
      var info = { through: isoDate(t), today: isoDate(today), ageDays: age, missingDays: missing, stale: missing >= 1, text: null };
      if (info.stale) {
        // the age counts the closed days after the data close, as the System page does
        info.text = "Data is " + missing + (missing === 1 ? " day" : " days") + " behind: last bar " + fmt.date(t) + "; " + missing + " closed " +
          (missing === 1 ? "day is" : "days are") + " missing.";
      }
      return info;
    },
    /**
     * The state of one data section in meta.sections (written by export_web.py).
     * -> {known, present, failed, stale, as_of, reason}. stale: the last export of the section
     * failed (the file is from an earlier export), or its as_of date is before meta.data_through.
     */
    section: function (meta, id) {
      var secs = meta && isObj(meta.sections) ? meta.sections : null;
      var s = secs && isObj(secs[id]) ? secs[id] : null;
      var out = { known: !!s, present: !!(s && s.present), failed: !!(s && s.state === "failed"), stale: false,
        as_of: s && typeof s.as_of === "string" ? s.as_of : null, reason: null };
      if (!s) return out;
      var through = meta.data_through;
      if (out.failed && out.present) {
        out.stale = true;
        out.reason = "The last export of this section failed. This file comes from an earlier export.";
      } else if (out.as_of && typeof through === "string" && out.as_of < through) {
        out.stale = true;
        out.reason = "This section has data through " + fmt.date(out.as_of) + ". The header date is " + fmt.date(through) + ".";
      }
      return out;
    }
  };

  /* ====================================================================== */
  /* FL.load: classic <script> loading on demand (works on file://)          */
  /* ====================================================================== */

  /*
   * Each file loads once per page view. A load settles with a state:
   *   "ok"       the file ran and set what it must set (a data key or a page);
   *   "missing"  the browser could not fetch it (onerror);
   *   "broken"   it was fetched but did not set its key: it does not parse, or it has a script error.
   * Missing and broken files are also pushed to window.FL_MISSING / window.FL_BROKEN (index.html),
   * which the footer lists. FL.load.version is appended as ?v=... so a served copy never mixes a
   * cached file with a new export.
   */
  var loads = {};
  function loadKeyOf(path) { return String(path).replace(/[?#].*$/, ""); }
  function scriptLoad(path, check) {
    var key = loadKeyOf(path);
    if (loads[key]) return loads[key].promise;
    var rec = { state: "pending", promise: null };
    loads[key] = rec;
    rec.promise = new Promise(function (resolve) {
      if (check && check()) { rec.state = "ok"; resolve({ path: key, state: "ok" }); return; }
      var s;
      try { s = document.createElement("script"); } catch (e) { s = null; }
      if (!s || !document.head || typeof document.head.appendChild !== "function") {
        rec.state = "missing"; resolve({ path: key, state: "missing" }); return;
      }
      var done = false;
      function settle(state) {
        if (done) return;
        done = true;
        rec.state = state;
        if (state === "missing") { var m = window.FL_MISSING = window.FL_MISSING || []; if (m.indexOf(key) < 0) m.push(key); }
        if (state === "broken") {
          var b = window.FL_BROKEN = window.FL_BROKEN || [];
          if (!b.some(function (x) { return x && x.file === key; })) b.push({ file: key, parse: true });
        }
        resolve({ path: key, state: state });
      }
      s.onload = function () { settle(check && !check() ? "broken" : "ok"); };
      s.onerror = function () { settle("missing"); };
      var v = FL.load.version;
      s.src = key + (v ? "?v=" + encodeURIComponent(String(v)) : "");
      document.head.appendChild(s);
    });
    return rec.promise;
  }
  var PART_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;
  FL.load = {
    /** Value of the ?v= query on every loaded file (fl-app.js sets meta.generated_at). */
    version: null,
    /** Load any script once. check() -> true when the file did its job. -> Promise<{path, state}> */
    script: function (path, check) { return scriptLoad(path, check); },
    /** data/<section>.js -> Promise<{path, state}>; state "ok" means FL.data[section] is set. */
    data: function (section) {
      return scriptLoad("data/" + section + ".js", function () { return FL.data[section] !== undefined; });
    },
    /** pages/<id>.js -> Promise<{path, state}>; state "ok" means the page registered. */
    page: function (id) {
      return scriptLoad("pages/" + id + ".js", function () { return !!(FL.pages && FL.pages.get(id)); });
    },
    /**
     * A part file of a section: data/<section>.<part>.js sets FL.data["<section>.<part>"].
     * -> Promise resolving the part payload; it rejects with an Error whose .state is
     * "missing" | "broken" | "bad-name". A part that is already in FL.data resolves at once.
     */
    part: function (section, part) {
      var key = section + "." + part;
      if (FL.data[key] !== undefined) return Promise.resolve(FL.data[key]);
      if (!PART_RE.test(String(part))) { var e0 = new Error("bad part name: " + part); e0.state = "bad-name"; return Promise.reject(e0); }
      return scriptLoad("data/" + key + ".js", function () { return FL.data[key] !== undefined; }).then(function (r) {
        if (r.state === "ok") return FL.data[key];
        var e = new Error("data/" + key + ".js " + (r.state === "missing" ? "is not in this export" : "does not parse"));
        e.state = r.state;
        throw e;
      });
    },
    /** The part names the exporter wrote for a section (meta.sections[s].parts), or []. */
    parts: function (section) {
      var m = FL.data.meta, s = m && m.sections && m.sections[section];
      return s && isObj(s.parts) ? Object.keys(s.parts).sort() : [];
    },
    /** "ok" | "missing" | "broken" | "pending" | null (never requested) for a path. */
    state: function (path) { var r = loads[loadKeyOf(path)]; return r ? r.state : null; },
    /** [{path, state}] for every requested file that is missing or broken. */
    problems: function () {
      return Object.keys(loads).filter(function (k) { return loads[k].state === "missing" || loads[k].state === "broken"; })
        .map(function (k) { return { path: k, state: loads[k].state }; });
    }
  };

  /* ====================================================================== */
  /* FL.ui                                                                   */
  /* ====================================================================== */

  function append(parent, child) {
    if (child === null || child === undefined || child === false) return;
    if (Array.isArray(child)) { child.forEach(function (c) { append(parent, c); }); return; }
    if (typeof child === "string" || typeof child === "number") { parent.appendChild(document.createTextNode(String(child))); return; }
    parent.appendChild(child);
  }
  function applyOpts(node, o, isSvg) {
    if (!o) return node;
    if (typeof o === "string") { node.setAttribute("class", o); return node; }
    var cls = o["class"] || o.className;
    if (cls) node.setAttribute("class", Array.isArray(cls) ? cls.filter(Boolean).join(" ") : cls);
    if (o.id) node.id = o.id;
    if (o.text !== undefined && o.text !== null) node.textContent = String(o.text);
    if (o.title) {
      if (isSvg) { var t = document.createElementNS(SVGNS, "title"); t.textContent = String(o.title); node.appendChild(t); }
      else node.title = String(o.title);
    }
    if (o.attrs) for (var k in o.attrs) {
      var v = o.attrs[k];
      if (v === null || v === undefined || v === false) continue;
      node.setAttribute(k, v === true ? "" : String(v));
    }
    if (o.style) for (var s in o.style) {
      if (o.style[s] === null || o.style[s] === undefined) continue;
      if (s.indexOf("--") === 0) node.style.setProperty(s, o.style[s]); else node.style[s] = o.style[s];
    }
    if (o.dataset) for (var d in o.dataset) node.dataset[d] = String(o.dataset[d]);
    if (o.on) for (var e in o.on) node.addEventListener(e, o.on[e]);
    return node;
  }

  /** Safe DOM builder: el(tag, {class, text, attrs, style, dataset, on, title, id}, children). */
  function el(tag, o, children) {
    var node = document.createElement(tag);
    if (typeof o === "string" || Array.isArray(o) || (o && o.nodeType)) { children = o; o = null; }
    applyOpts(node, o, false);
    append(node, children);
    return node;
  }
  /** SVG builder: svg(tag, attrs, children). attrs are plain attributes (+ class, text, style, title). */
  function svg(tag, a, children) {
    var node = document.createElementNS(SVGNS, tag);
    if (a) for (var k in a) {
      var v = a[k];
      if (v === null || v === undefined || v === false) continue;
      if (k === "text") node.textContent = String(v);
      else if (k === "class") node.setAttribute("class", v);
      else if (k === "style" && isObj(v)) { for (var s in v) if (v[s] !== null && v[s] !== undefined) node.style[s] = v[s]; }
      else if (k === "title") { var t = document.createElementNS(SVGNS, "title"); t.textContent = String(v); node.appendChild(t); }
      else node.setAttribute(k, typeof v === "number" ? String(Math.round(v * 100) / 100) : String(v));
    }
    append(node, children);
    return node;
  }
  function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); return node; }
  function resolveContainer(c) {
    if (typeof c === "string") return document.querySelector(c);
    return c;
  }
  var idSeq = 0;
  function uid(prefix) { idSeq += 1; return (prefix || "fl") + "-" + idSeq; }

  // --- icons (24x24 stroke paths; decorative, aria-hidden) ---
  // "system" is the Feather "settings" icon (MIT) and "sun" is the Lucide "sun" icon (ISC),
  // with rounded numbers: see THIRD_PARTY_NOTICES.md.
  var ICONS = {
    overview: ["M4 4h7v7H4z", "M13 4h7v4h-7z", "M13 10h7v10h-7z", "M4 13h7v7H4z"],
    factors: ["M9 3h6", "M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4a2 2 0 0 0 1.8-3l-5-9V3", "M7.5 14h9"],
    research: ["M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14z", "M20 20l-4-4"],
    system: ["M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z", "M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-1-1.5 1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.5-1 1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z"],
    template: ["M4 4h16v6H4z", "M4 14h7v6H4z", "M15 14h5v6h-5z"],
    sun: ["M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z", "M12 2v2", "M12 20v2", "M4.9 4.9l1.4 1.4", "M17.7 17.7l1.4 1.4", "M2 12h2", "M20 12h2", "M4.9 19.1l1.4-1.4", "M17.7 6.3l1.4-1.4"],
    moon: ["M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"],
    auto: ["M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z", "M12 3v18"],
    table: ["M4 5h16v14H4z", "M4 10h16", "M4 15h16", "M10 5v14"],
    chart: ["M4 19h16", "M5 15l4-5 4 3 6-7"],
    empty: ["M4 7l8-4 8 4-8 4z", "M4 7v10l8 4 8-4V7", "M12 11v10"],
    logo: ["M4 20V4", "M4 20h16", "M7 16l4-5 3 3 5-7"],
    info: ["M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z", "M12 11v5", "M12 8v.5"]
  };
  function icon(name, o) {
    var paths = ICONS[name] || ICONS.overview;
    var node = svg("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 1.75,
      "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", focusable: "false", "class": (o && o["class"]) || null });
    paths.forEach(function (d) { node.appendChild(svg("path", { d: d })); });
    return node;
  }

  /** Small identity key: kind 'line' (default), 'rect' or 'dot'; color is a CSS color or an entity id. */
  function key(colorOrEntity, o) {
    o = o || {};
    var info = colors.entityInfo(colorOrEntity, o.context);
    var color = info ? info.color : colorOrEntity || "var(--muted)";
    var dashed = o.dashed !== undefined ? o.dashed : !!(info && info.dashed);
    var kind = o.kind || "line";
    var w = kind === "line" ? 16 : kind === "dot" ? 8 : 10, h = kind === "line" ? 10 : kind === "dot" ? 8 : 10;
    var node = svg("svg", { width: w, height: h, viewBox: "0 0 " + w + " " + h, "class": "fl-key", "aria-hidden": "true" });
    if (kind === "line") node.appendChild(svg("line", { x1: 1, y1: 5, x2: 15, y2: 5, "stroke-width": 2, "stroke-linecap": "round", "stroke-dasharray": dashed ? "5 3" : null, style: { stroke: color } }));
    else if (kind === "dot") node.appendChild(svg("circle", { cx: 4, cy: 4, r: 4, style: { fill: color } }));
    else node.appendChild(svg("rect", { x: 0, y: 0, width: 10, height: 10, rx: 2.5, style: { fill: color } }));
    return node;
  }

  /** Legend: items [{name, color|entity, dashed, kind:'line'|'rect'|'dot'}] -> element. */
  function legend(items, o) {
    o = o || {};
    var root = el("div", { "class": "fl-legend", attrs: { role: "list" } });
    (items || []).forEach(function (it) {
      var c = it.color || it.entity;
      root.appendChild(el("span", { "class": "fl-legend-item", attrs: { role: "listitem" } }, [
        key(c, { kind: it.kind || o.kind || "line", dashed: it.dashed, context: o.context }),
        el("span", { text: it.name !== undefined ? it.name : colors.label(it.entity) })
      ]));
    });
    return root;
  }

  /**
   * Card: {id, title, subtitle, actions:[Node], footer, className, span:2|'all', flat, level} -> {root, body, head, foot}
   * level: the heading level of the title, 2..6 (default 3). A top-level card on a page
   * without section headings uses level 2, so the outline does not skip from h1 to h3.
   */
  function card(o, maybeContainer) {
    o = o || {};
    var root = el("section", { "class": ["fl-card", o.className, o.span === 2 ? "span-2" : o.span === "all" ? "span-all" : null, o.flat ? "is-flat" : null], id: o.id || null });
    var head = null, titles;
    var level = Math.round(+o.level);
    if (!(level >= 2 && level <= 6)) level = 3;
    if (o.title || o.subtitle || o.actions) {
      head = el("div", { "class": "fl-card-head" });
      titles = el("div", { "class": "fl-card-titles" });
      if (o.title) titles.appendChild(el("h" + level, { "class": "fl-card-title", text: o.title }));
      if (o.subtitle) titles.appendChild(el("p", { "class": "fl-card-sub", text: o.subtitle }));
      head.appendChild(titles);
      if (o.actions) head.appendChild(el("div", { "class": "fl-card-actions" }, o.actions));
      root.appendChild(head);
      if (o.title) root.setAttribute("aria-label", o.title);
    }
    var body = el("div", { "class": "fl-card-body" });
    root.appendChild(body);
    var foot = null;
    if (o.footer) {
      foot = el("div", { "class": "fl-card-foot" });
      append(foot, typeof o.footer === "string" ? el("span", { text: o.footer }) : o.footer);
      root.appendChild(foot);
    }
    var container = resolveContainer(o.container || maybeContainer);
    if (container) container.appendChild(root);
    return { root: root, body: body, head: head, foot: foot };
  }

  function deltaDirClass(dir, sign) {
    if (!sign || dir === "neutral" || !dir) return "dir-neutral";
    var up = sign > 0;
    if (dir === "up-good") return up ? "dir-good" : "dir-bad";
    if (dir === "up-bad") return up ? "dir-bad" : "dir-good";
    return "dir-neutral";
  }
  function signFromText(s) {
    s = String(s).trim();
    if (/^[+]/.test(s)) return 1;
    if (/^[−-]/.test(s)) return -1;
    return 0;
  }

  /**
   * KPI tiles. items: [{label, value, format, delta, deltaFormat, deltaValue, deltaDir, deltaLabel,
   *   note, entity, color, badge:{status,label}, spark:{values, entity|color}, hero, id, title}]
   * value/delta may be preformatted strings or numbers (formatted with format/deltaFormat).
   */
  function kpis(container, items, o) {
    container = resolveContainer(container);
    o = o || {};
    var root = el("div", { "class": ["kpis", o.className] });
    var sparks = [];
    (items || []).forEach(function (it) {
      var tile = el("div", { "class": ["fl-kpi", it.hero ? "is-hero" : null], id: it.id || null, title: it.title || null });
      var lab = el("div", { "class": "fl-kpi-label" });
      if (it.entity || it.color) lab.appendChild(key(it.color || it.entity, { kind: "dot" }));
      lab.appendChild(el("span", { text: it.label }));
      tile.appendChild(lab);
      var val = typeof it.value === "number" || it.value === null || it.value === undefined
        ? fmt.get(it.format || "num")(it.value) : String(it.value);
      tile.appendChild(el("div", { "class": "fl-kpi-value", text: val }));
      if (it.delta !== undefined && it.delta !== null && it.delta !== "") {
        var dtext = typeof it.delta === "number" ? fmt.get(it.deltaFormat || it.format || "num", { signed: true })(it.delta) : String(it.delta);
        var sign = typeof it.deltaValue === "number" ? Math.sign(it.deltaValue)
          : typeof it.delta === "number" ? Math.sign(it.delta) : signFromText(dtext);
        if (dtext === NA) sign = 0;
        var cls = deltaDirClass(it.deltaDir || "neutral", sign);
        var d = el("div", { "class": "fl-kpi-delta" });
        var main = el("span", { "class": cls });
        if (sign) main.appendChild(el("span", { "class": "arrow", text: sign > 0 ? "▲ " : "▼ ", attrs: { "aria-hidden": "true" } }));
        main.appendChild(document.createTextNode(dtext));
        d.appendChild(main);
        if (it.deltaLabel) d.appendChild(el("span", { "class": "dlabel", text: it.deltaLabel }));
        tile.appendChild(d);
      }
      if (it.note) tile.appendChild(el("div", { "class": "fl-kpi-note", text: it.note }));
      if (it.badge) tile.appendChild(badge(it.badge.status, it.badge.label));
      if (it.spark && FL.charts && FL.charts.sparkline) {
        var sp = el("div", { "class": "fl-kpi-spark" });
        tile.appendChild(sp);
        sparks.push([sp, it]);
      }
      root.appendChild(tile);
    });
    if (container) container.appendChild(root);
    // sparklines need to be attached to measure width
    sparks.forEach(function (p) {
      FL.charts.sparkline(p[0], assign({ entity: p[1].entity, color: p[1].color, height: 28, ariaLabel: p[1].label }, p[1].spark));
    });
    return root;
  }

  var BADGE_ICONS = { good: "✓", warning: "!", serious: "▲", critical: "✕", info: "i", advisory: "i", neutral: "•" };
  /** Status badge: icon + text label (never color alone). status: good|warning|serious|critical|info|advisory|neutral */
  function badge(status, label, o) {
    status = BADGE_ICONS[status] ? status : "neutral";
    o = o || {};
    return el("span", { "class": ["fl-badge", "fl-badge--" + status, o.className], title: o.title || null }, [
      el("span", { "class": "fl-badge-icon", text: BADGE_ICONS[status], attrs: { "aria-hidden": "true" } }),
      el("span", { "class": "fl-badge-label", text: label === undefined ? status : label })
    ]);
  }

  /** Callout: kind info|warning|advisory|critical|serious|good; text: string | [strings] | Node. Returns element (append it yourself or pass {container}). */
  function callout(kind, title, text, o) {
    kind = BADGE_ICONS[kind] ? kind : "info";
    o = o || {};
    var body = el("div", { "class": "fl-callout-body" });
    if (title) body.appendChild(el("div", { "class": "fl-callout-title", text: title }));
    if (text !== undefined && text !== null) {
      var t = el("div", { "class": "fl-callout-text" });
      if (Array.isArray(text)) text.forEach(function (p) { t.appendChild(typeof p === "string" ? el("p", { text: p }) : p); });
      else if (typeof text === "string") t.appendChild(el("p", { text: text }));
      else t.appendChild(text);
      body.appendChild(t);
    }
    var root = el("div", { "class": ["fl-callout", "fl-callout--" + kind, "fl-badge--" + kind, o.className], attrs: { role: kind === "critical" || kind === "warning" ? "note" : null } }, [
      el("span", { "class": "fl-badge-icon", text: BADGE_ICONS[kind], attrs: { "aria-hidden": "true" } }),
      body
    ]);
    var c = resolveContainer(o.container);
    if (c) c.appendChild(root);
    return root;
  }

  /**
   * Empty state. message string; o: {title, command, commands:[{label, command}]}.
   * commands shows each command under its own label, e.g. the setup.sh and export_web.py remedies.
   */
  function empty(container, message, o) {
    container = resolveContainer(container);
    o = o || {};
    var root = el("div", { "class": "fl-empty", attrs: { role: "status" } }, [icon("empty")]);
    if (o.title) root.appendChild(el("div", { "class": "fl-empty-title", text: o.title }));
    root.appendChild(el("div", { "class": "fl-empty-text", text: message || "No data." }));
    if (o.command) root.appendChild(el("pre", { "class": "fl-code", text: o.command }));
    (o.commands || []).forEach(function (c) {
      if (!c || !c.command) return;
      if (c.label) root.appendChild(el("div", { "class": "fl-empty-label", text: c.label }));
      root.appendChild(el("pre", { "class": "fl-code", text: c.command }));
    });
    if (container) container.appendChild(root);
    return root;
  }
  /** The two standard remedies for absent data, for empty({commands}). */
  function remedies() {
    return [{ label: "To add the data or new days:", command: FL.cmd.setup },
      { label: "To rebuild only the pages:", command: FL.cmd.export }];
  }

  /*
   * Scroll cue: a box that scrolls sideways gets is-more-left / is-more-right while content is
   * hidden on that side (app.css draws an edge shadow or fade). One shared ResizeObserver; the
   * scroll listener lives on the box itself, so nothing outlives the page.
   */
  var cueRO = null;
  function updateCue(node) {
    var sw = node.scrollWidth, cw = node.clientWidth, sl = node.scrollLeft;
    if (!(sw > 0) || !(cw > 0)) return;
    var left = sl > 1, right = sl + cw < sw - 1, t = node._cueTarget || node;
    if (t.classList.contains("is-more-left") !== left) t.classList.toggle("is-more-left", left);
    if (t.classList.contains("is-more-right") !== right) t.classList.toggle("is-more-right", right);
  }
  /** Keep is-more-left / is-more-right on target (default: node) while node hides content on that side. */
  function scrollCue(node, target) {
    if (!node || typeof node.addEventListener !== "function") return node;
    if (target) node._cueTarget = target;
    node.addEventListener("scroll", function () { updateCue(node); }, { passive: true });
    if (typeof ResizeObserver !== "undefined") {
      if (!cueRO) cueRO = new ResizeObserver(function (entries) {
        entries.forEach(function (e) { if (!e.target.isConnected) cueRO.unobserve(e.target); else updateCue(e.target); });
      });
      // the observer's first callback sets the classes after layout, so no forced layout here
      try { cueRO.observe(node); } catch (e) { updateCue(node); }
    } else updateCue(node);
    return node;
  }

  /**
   * Table. o: {columns:[{key, label, format, align:'left'|'right'|'center', entity, value(row), render(v,row)->Node|string,
   *   sortable, sortValue(row), title, className, wrap, mono, signColor, width, rowHeader, stackLabel}], rows, sortable=true,
   *   maxHeight, caption, ariaLabel, defaultSort:{key, dir:'asc'|'desc'}, rowClass(row), stickyFirst, compact, emptyText,
   *   entityContext, pageSize, phonePageSize, onSort(key, dir), stack}
   * pageSize: show the first N rows, with "Show N more" and "Show all" buttons; a sort orders ALL rows, then slices.
   * phonePageSize: the page size below 600 px (default: pageSize).
   * onSort(key, dir): called after a sort by a header click.
   * rowHeader (column): its cells are <th scope="row"> (the name column of the row).
   * stack: below 600 px each row shows as a card of label/value pairs (stackLabel overrides the label).
   * caption: shown above the table and read as its <caption>; ariaLabel names a table without a caption.
   * -> {root, table, setRows(rows), sort(key, dir), showAll(), shown(), count()}
   */
  function table(container, o) {
    container = resolveContainer(container);
    o = o || {};
    var columns = (o.columns || []).map(function (c) { return assign({}, c); });
    var rows = (o.rows || []).slice();
    var sortable = o.sortable !== false;
    var phone = o.phonePageSize > 0 && window.matchMedia && window.matchMedia("(max-width: 599.98px)").matches;
    var pageSize = phone ? Math.floor(o.phonePageSize) : o.pageSize > 0 ? Math.floor(o.pageSize) : 0;
    var limit = pageSize || Infinity;
    var state = { key: null, dir: null };
    var root = el("div", { "class": ["fl-table-block", o.className, o.stack ? "is-stack" : null] });
    if (o.caption) root.appendChild(el("div", { "class": "fl-table-cap", text: o.caption, attrs: { "aria-hidden": "true" } }));
    // frame: a fixed box around the scroll box, so the edge shadows do not scroll with the rows
    var frame = el("div", { "class": "fl-table-frame" });
    var wrap = el("div", { "class": "fl-table-wrap" });
    wrap._cueTarget = frame;
    if (o.maxHeight) wrap.style.maxHeight = typeof o.maxHeight === "number" ? o.maxHeight + "px" : o.maxHeight;
    // a stacked table shows as blocks below 600 px; the ARIA roles keep its table structure for screen readers
    var tbl = el("table", { "class": ["fl-table", o.compact ? "is-compact" : null, o.stack ? "fl-table--stack" : null],
      attrs: { "aria-label": !o.caption && o.ariaLabel ? o.ariaLabel : null, role: o.stack ? "table" : null } });
    if (o.caption) tbl.appendChild(el("caption", { "class": "sr-only", text: o.caption }));
    wrap.appendChild(tbl);
    frame.appendChild(wrap);
    root.appendChild(frame);

    function valueOf(c, r) { return c.value ? c.value(r) : r[c.key]; }
    columns.forEach(function (c) {
      c._fmt = fmt.get(c.format);
      if (!c.align) {
        if (fmt.isNumericKind(c.format)) c.align = "right";
        else {
          var first = null;
          for (var i = 0; i < rows.length; i++) { var v = valueOf(c, rows[i]); if (v !== null && v !== undefined) { first = v; break; } }
          c.align = typeof first === "number" && !c.entity ? "right" : "left";
        }
      }
      c._label = c.stackLabel !== undefined ? String(c.stackLabel) : typeof c.label === "string" ? c.label : String(c.key);
    });

    var thead = el("thead"), htr = el("tr");
    columns.forEach(function (c, ci) {
      var th = el("th", { "class": [c.align === "right" ? "is-num" : c.align === "center" ? "is-center" : null, ci === 0 && o.stickyFirst ? "sticky-first" : null], attrs: { scope: "col" }, title: c.title || null });
      if (c.width) th.style.width = typeof c.width === "number" ? c.width + "px" : c.width;
      var canSort = sortable && c.sortable !== false;
      if (canSort) {
        var b = el("button", { "class": "fl-sort", attrs: { type: "button" } }, [
          c.align === "right" ? el("span", { "class": "ind", attrs: { "aria-hidden": "true" } }) : null,
          el("span", { text: c.label !== undefined ? c.label : c.key }),
          c.align !== "right" ? el("span", { "class": "ind", attrs: { "aria-hidden": "true" } }) : null
        ]);
        b.addEventListener("click", function () {
          var dir = state.key === c.key ? (state.dir === "asc" ? "desc" : "asc") : (c.align === "right" ? "desc" : "asc");
          sort(c.key, dir);
          if (o.onSort) o.onSort(c.key, state.dir);
        });
        th.appendChild(b);
      } else th.appendChild(el("span", { text: c.label !== undefined ? c.label : c.key }));
      c._th = th;
      htr.appendChild(th);
    });
    thead.appendChild(htr);
    tbl.appendChild(thead);
    var tbody = el("tbody", { attrs: { role: o.stack ? "rowgroup" : null } });
    tbl.appendChild(tbody);

    // "Show more" row under the table (pageSize only). The buttons are built once, so focus stays on them.
    var more = null, moreText = null, moreBtn = null, allBtn = null, lessBtn = null;
    if (pageSize) {
      more = el("div", { "class": "fl-table-more" });
      moreText = el("span", { "class": "fl-table-more-text", attrs: { "aria-live": "polite" } });
      moreBtn = el("button", { "class": "fl-button", attrs: { type: "button" } });
      allBtn = el("button", { "class": "fl-button", attrs: { type: "button" } });
      lessBtn = el("button", { "class": "fl-button", attrs: { type: "button" }, text: "Show fewer" });
      moreBtn.addEventListener("click", function () { limit += pageSize; renderBody(); });
      allBtn.addEventListener("click", function () { limit = Infinity; renderBody(); });
      lessBtn.addEventListener("click", function () { limit = pageSize; renderBody(); });
      more.appendChild(moreText); more.appendChild(moreBtn); more.appendChild(allBtn); more.appendChild(lessBtn);
      root.appendChild(more);
    }
    function updateMore() {
      if (!more) return;
      var n = rows.length, shown = Math.min(n, limit);
      more.hidden = n <= pageSize;
      moreText.textContent = shown < n ? "Rows 1–" + fmt.int(shown) + " of " + fmt.int(n) + "." : "All " + fmt.int(n) + " rows.";
      var step = Math.min(pageSize, n - shown);
      moreBtn.textContent = "Show " + fmt.int(step) + " more";
      allBtn.textContent = "Show all " + fmt.int(n);
      moreBtn.hidden = shown >= n;
      allBtn.hidden = shown >= n || step >= n - shown;
      lessBtn.hidden = shown < n;
    }

    function cellEntity(c, r, v) {
      if (!c.entity) return null;
      if (c.entity === true) return v;
      if (typeof c.entity === "function") return c.entity(r);
      return r[c.entity];
    }
    function renderRow(r) {
      var tr = el("tr", { "class": o.rowClass ? o.rowClass(r) : null, attrs: { role: o.stack ? "row" : null } });
      columns.forEach(function (c, ci) {
        var v = valueOf(c, r);
        var cls = [c.align === "right" ? "is-num" : c.align === "center" ? "is-center" : null, c.mono ? "is-mono" : null, c.wrap ? "is-wrap" : null, c.className, ci === 0 && o.stickyFirst ? "sticky-first" : null];
        var td = c.rowHeader ? el("th", { "class": cls, attrs: { scope: "row", role: o.stack ? "rowheader" : null } })
          : el("td", { "class": cls, attrs: { role: o.stack ? "cell" : null } });
        if (o.stack) td.setAttribute("data-label", c._label);
        if (c.signColor && typeof v === "number" && isFinite(v) && /[1-9]/.test(String(c._fmt(v, r)))) {
          if (v > 0) td.classList.add("good"); else if (v < 0) td.classList.add("bad");
        }
        var content;
        if (c.render) content = c.render(v, r);
        else if (c.entity === true && (c.format === undefined)) content = colors.label(v);
        else content = c._fmt(v, r);
        var ent = cellEntity(c, r, v);
        if (ent !== null && ent !== undefined) {
          var holder = el("span", { "class": "cell-key" }, [key(ent, { kind: "dot", context: o.entityContext })]);
          append(holder, content);
          td.appendChild(holder);
        } else append(td, content);
        tr.appendChild(td);
      });
      return tr;
    }
    function renderBody() {
      clear(tbody);
      if (!rows.length) {
        tbody.appendChild(el("tr", null, el("td", { "class": "fl-table-empty", text: o.emptyText || "No rows.", attrs: { colspan: columns.length } })));
        updateMore();
        return;
      }
      var frag = document.createDocumentFragment();
      var n = Math.min(rows.length, limit);
      for (var i = 0; i < n; i++) frag.appendChild(renderRow(rows[i]));
      tbody.appendChild(frag);
      updateMore();
      // after the next layout, not during the build (a connected table only; the observer covers the first render)
      if (wrap.isConnected && window.requestAnimationFrame) window.requestAnimationFrame(function () { updateCue(wrap); });
    }
    function cmp(a, b) {
      var an = a === null || a === undefined || (typeof a === "number" && !isFinite(a));
      var bn = b === null || b === undefined || (typeof b === "number" && !isFinite(b));
      if (an && bn) return 0;
      if (an) return 1; // nulls last
      if (bn) return -1;
      if (typeof a === "number" && typeof b === "number") return a - b;
      return String(a).localeCompare(String(b), undefined, { numeric: true });
    }
    function sort(k, dir) {
      var c = columns.filter(function (x) { return x.key === k; })[0];
      if (!c) return;
      state.key = k; state.dir = dir === "desc" ? "desc" : "asc";
      var sv = c.sortValue || function (r) { return valueOf(c, r); };
      var idx = rows.map(function (r, i) { return { r: r, i: i, v: sv(r) }; });
      idx.sort(function (a, b) {
        var an = a.v === null || a.v === undefined, bn = b.v === null || b.v === undefined;
        if (an !== bn) return an ? 1 : -1; // nulls last in both directions
        var x = cmp(a.v, b.v);
        if (state.dir === "desc") x = -x;
        return x || a.i - b.i;
      });
      rows = idx.map(function (x) { return x.r; });
      columns.forEach(function (cc) {
        if (!cc._th) return;
        var ind = cc._th.querySelector(".ind");
        if (cc.key === k) { cc._th.setAttribute("aria-sort", state.dir === "asc" ? "ascending" : "descending"); if (ind) ind.textContent = state.dir === "asc" ? "▲" : "▼"; }
        else { cc._th.removeAttribute("aria-sort"); if (ind) ind.textContent = ""; }
      });
      renderBody();
    }
    if (o.defaultSort && o.defaultSort.key) sort(o.defaultSort.key, o.defaultSort.dir); else renderBody();
    if (container) container.appendChild(root);
    scrollCue(wrap, frame);
    return {
      root: root, table: tbl,
      setRows: function (r) { rows = (r || []).slice(); limit = pageSize || Infinity; if (state.key) sort(state.key, state.dir); else renderBody(); },
      sort: sort,
      showAll: function () { limit = Infinity; renderBody(); },
      shown: function () { return Math.min(rows.length, limit); },
      count: function () { return rows.length; }
    };
  }

  /**
   * "On this page" jump links. items: [[id, label]] | [{id, label}]; o: {page, href(id) -> string, ariaLabel, className}.
   * Links go to "#/<page>/<id>" (the router scrolls to the card) unless href is given. -> nav element.
   */
  function toc(container, items, o) {
    container = resolveContainer(container);
    o = o || {};
    var nav = el("nav", { "class": ["fl-toc", o.className], attrs: { "aria-label": o.ariaLabel || "On this page" } });
    var list = el("ul");
    (items || []).forEach(function (it) {
      var id = Array.isArray(it) ? it[0] : it.id, label = Array.isArray(it) ? it[1] : it.label;
      if (!id) return;
      var h = o.href ? o.href(id) : FL.app && FL.app.href && o.page ? FL.app.href(o.page, id)
        : "#/" + encodeURIComponent(o.page || "") + "/" + encodeURIComponent(id);
      list.appendChild(el("li", null, el("a", { attrs: { href: h }, text: label })));
    });
    nav.appendChild(list);
    if (container) container.appendChild(nav);
    return nav;
  }

  /*
   * Deferred rendering. While the app shell renders a page it opens a queue: defer() then
   * appends a placeholder and runs fn in chunks after the first paint. Without an
   * open queue (node tests, a page that renders outside the shell) fn runs at once.
   */
  var deferQ = null, deferGen = 0, deferTimer = null;
  function defer(container, o, fn) {
    if (typeof o === "function") { fn = o; o = {}; }
    container = resolveContainer(container);
    o = o || {};
    if (deferQ === null) { fn(container); return null; }
    var ph = el("div", { "class": "fl-defer", attrs: { "aria-busy": "true", "data-defer-id": o.id || null },
      style: { minHeight: (o.minHeight === undefined ? 160 : o.minHeight) + "px" } });
    container.appendChild(ph);
    deferQ.push({ ph: ph, fn: fn, id: o.id || null });
    scheduleDefer();
    return ph;
  }
  function runDeferTask(t) {
    var ph = t.ph, parent = ph.parentNode;
    if (!parent) return;
    ph.style.minHeight = "";
    try { t.fn(ph); } catch (err) {
      if (window.console) console.error(err);
      clear(ph);
      ph.appendChild(callout("critical", "This part of the page failed to render", "Error: " + (err && err.message ? err.message : String(err))));
    }
    // unwrap: the rendered nodes take the placeholder's place (and its id is on the real card)
    while (ph.firstChild) parent.insertBefore(ph.firstChild, ph);
    parent.removeChild(ph);
  }
  function nowMs() { return window.performance && performance.now ? performance.now() : Date.now(); }
  function scheduleDefer() {
    if (deferTimer !== null || !deferQ || !deferQ.length) return;
    var gen = deferGen;
    var run = function () {
      deferTimer = null;
      if (gen !== deferGen || !deferQ) return;
      var t0 = nowMs();
      while (deferQ.length && nowMs() - t0 < 12) runDeferTask(deferQ.shift());
      if (deferQ.length) scheduleDefer(); else deferQ = null;
    };
    deferTimer = window.requestIdleCallback ? { idle: true, id: window.requestIdleCallback(run, { timeout: 120 }) } : { idle: false, id: setTimeout(run, 16) };
  }
  function cancelDeferTimer() {
    if (deferTimer === null) return;
    if (deferTimer.idle) { if (window.cancelIdleCallback) window.cancelIdleCallback(deferTimer.id); } else clearTimeout(deferTimer.id);
    deferTimer = null;
  }
  var deferApi = {
    /** the shell opens a new queue for each page render (older tasks are dropped) */
    begin: function () {
      deferGen += 1;
      cancelDeferTimer();
      deferQ = [];
    },
    /** run tasks at once while fill() is true (e.g. until 1.5 viewports are filled), then chunk the rest */
    fill: function (more) {
      while (deferQ && deferQ.length && more()) runDeferTask(deferQ.shift());
      if (deferQ && !deferQ.length) deferQ = null; else scheduleDefer();
    },
    /** run tasks in order until the task with this id (or, without a match, every task) has run */
    flush: function (id) {
      if (!deferQ) return;
      var hasId = id && deferQ.some(function (t) { return t.id === id; });
      while (deferQ && deferQ.length) {
        var t = deferQ.shift();
        runDeferTask(t);
        if (hasId && t.id === id) break;
      }
      if (deferQ && !deferQ.length) deferQ = null;
    },
    /** drop the queue (navigation) */
    reset: function () { deferApi.begin(); deferQ = null; },
    pending: function () { return deferQ ? deferQ.length : 0; }
  };

  /**
   * A window "fl:themechange" listener that ends with the page. signal: an AbortSignal (ctx.signal);
   * without one the listener stays until removed. -> remove()
   */
  function onTheme(fn, signal) {
    window.addEventListener("fl:themechange", fn);
    var off = function () { window.removeEventListener("fl:themechange", fn); };
    if (signal && typeof signal.addEventListener === "function") {
      if (signal.aborted) off(); else signal.addEventListener("abort", off);
    }
    return off;
  }

  /**
   * Tabs (lazy panels). o: {items:[{id, label, render(panel)}], active, onChange(id), variant:'tabs'|'segmented', ariaLabel}
   * -> {root, set(id), get(), panels}
   */
  function tabs(container, o) {
    container = resolveContainer(container);
    o = o || {};
    var items = o.items || [];
    var active = o.active || (items[0] && items[0].id);
    var root = el("div", { "class": "fl-tabs" });
    var seg = o.variant === "segmented";
    var list = el("div", { "class": seg ? "fl-seg" : "fl-tabs-list", attrs: { role: "tablist", "aria-label": o.ariaLabel || null } });
    if (seg) list.style.marginBottom = "12px";
    var panels = {}, buttons = {}, rendered = {};
    var base = uid("tabs");
    items.forEach(function (it) {
      var b = el("button", { "class": seg ? null : "fl-tab", text: it.label, id: base + "-t-" + it.id,
        attrs: { type: "button", role: "tab", "aria-controls": base + "-p-" + it.id } });
      b.addEventListener("click", function () { set(it.id, true); });
      b.addEventListener("keydown", function (e) {
        var i = items.indexOf(it), n = items.length;
        if (e.key === "ArrowRight") { e.preventDefault(); set(items[(i + 1) % n].id, true); buttons[items[(i + 1) % n].id].focus(); }
        else if (e.key === "ArrowLeft") { e.preventDefault(); set(items[(i - 1 + n) % n].id, true); buttons[items[(i - 1 + n) % n].id].focus(); }
      });
      buttons[it.id] = b;
      list.appendChild(b);
      var p = el("div", { "class": "fl-tab-panel", id: base + "-p-" + it.id, attrs: { role: "tabpanel", "aria-labelledby": base + "-t-" + it.id, tabindex: "0" } });
      p.hidden = true;
      panels[it.id] = p;
    });
    root.appendChild(list);
    items.forEach(function (it) { root.appendChild(panels[it.id]); });
    function set(id, user) {
      if (!panels[id]) return;
      active = id;
      items.forEach(function (it) {
        var on = it.id === id;
        buttons[it.id].setAttribute("aria-selected", on ? "true" : "false");
        buttons[it.id].setAttribute("tabindex", on ? "0" : "-1");
        panels[it.id].hidden = !on;
      });
      if (!rendered[id]) {
        rendered[id] = true;
        var it = items.filter(function (x) { return x.id === id; })[0];
        if (it && it.render) it.render(panels[id]);
      }
      if (user && o.onChange) o.onChange(id);
    }
    if (container) container.appendChild(root);
    set(active, false);
    return { root: root, set: function (id) { set(id, false); }, get: function () { return active; }, panels: panels };
  }

  /**
   * Segmented control (single choice). o: {options:[{value,label,icon,title}|string], value, onChange(value), ariaLabel, iconsOnly}
   * -> {root, get(), set(value)}
   */
  function segmented(container, o) {
    container = resolveContainer(container);
    o = o || {};
    var opts = (o.options || []).map(function (x) { return typeof x === "object" ? x : { value: x, label: String(x) }; });
    var value = o.value !== undefined ? o.value : opts[0] && opts[0].value;
    var root = el("div", { "class": ["fl-seg", o.iconsOnly ? "is-icons" : null, o.className], attrs: { role: "radiogroup", "aria-label": o.ariaLabel || null } });
    var btns = [];
    opts.forEach(function (op, i) {
      var b = el("button", { attrs: { type: "button", role: "radio", "aria-label": o.iconsOnly ? op.label : null }, title: op.title || (o.iconsOnly ? op.label : null) });
      if (op.icon) b.appendChild(icon(op.icon));
      if (!o.iconsOnly) b.appendChild(el("span", { text: op.label }));
      b.addEventListener("click", function () { set(op.value, true); });
      b.addEventListener("keydown", function (e) {
        var n = opts.length, j = null;
        if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % n;
        if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + n) % n;
        if (j !== null) { e.preventDefault(); set(opts[j].value, true); btns[j].focus(); }
      });
      btns.push(b);
      root.appendChild(b);
    });
    function set(v, user) {
      value = v;
      opts.forEach(function (op, i) {
        var on = op.value === v;
        btns[i].setAttribute("aria-checked", on ? "true" : "false");
        btns[i].setAttribute("tabindex", on ? "0" : "-1");
      });
      if (user && o.onChange) o.onChange(v);
    }
    set(value, false);
    if (container) container.appendChild(root);
    return { root: root, get: function () { return value; }, set: function (v) { set(v, false); } };
  }

  /** Collapsible. o: {summary, meta, open, render(body), children} -> {root, body}. render() runs on first open. */
  function details(container, o) {
    container = resolveContainer(container);
    o = o || {};
    var root = el("details", { "class": ["fl-details", o.className], id: o.id || null });
    var sum = el("summary", null, [el("span", { text: o.summary || "Details" })]);
    if (o.meta) sum.appendChild(el("span", { "class": "fl-details-meta", text: o.meta }));
    root.appendChild(sum);
    var body = el("div", { "class": "fl-details-body" });
    root.appendChild(body);
    if (o.children) append(body, o.children);
    var done = false;
    function maybe() { if (root.open && !done && o.render) { done = true; o.render(body); } }
    root.addEventListener("toggle", maybe);
    if (container) container.appendChild(root);
    if (o.open) { root.open = true; maybe(); }
    return { root: root, body: body };
  }

  /** Definition list. items: [[term, value]] | [{term, value, mono}] (value: string | Node) */
  function dl(container, items) {
    container = resolveContainer(container);
    var root = el("dl", { "class": "fl-dl" });
    (items || []).forEach(function (it) {
      var t = Array.isArray(it) ? it[0] : it.term, v = Array.isArray(it) ? it[1] : it.value;
      root.appendChild(el("dt", { text: t }));
      var dd = el("dd", { "class": !Array.isArray(it) && it.mono ? "mono" : null });
      append(dd, v === null || v === undefined ? NA : v);
      root.appendChild(dd);
    });
    if (container) container.appendChild(root);
    return root;
  }

  /**
   * Meter / progress track. o: {label, value, min=0, max, format, valueText, band:[lo,hi], ticks:[{value,label}], color}
   * The band is drawn as a deeper step of the track (e.g. the 4–8 week minimum window).
   * Neutral ink by default (a categorical hue would read as a book); `color` (a CSS color,
   * e.g. FL.colors.status.warning) tints fill, band and track as one ramp.
   */
  function meter(container, o) {
    container = resolveContainer(container);
    o = o || {};
    var min = o.min || 0, max = o.max === undefined ? 1 : o.max;
    var f = fmt.get(o.format || "num:1");
    function pos(v) { return Math.max(0, Math.min(100, (v - min) / (max - min) * 100)); }
    var root = el("div", { "class": "fl-meter", style: o.color ? { "--meter": String(o.color) } : null });
    var head = el("div", { "class": "fl-meter-head" }, [el("span", { text: o.label || "" }), el("strong", { text: o.valueText || f(o.value) })]);
    root.appendChild(head);
    var track = el("div", { "class": "fl-meter-track", attrs: { role: "meter", "aria-valuemin": min, "aria-valuemax": max, "aria-valuenow": toNum(o.value) === null ? null : o.value, "aria-label": o.label || null } });
    if (o.band) track.appendChild(el("div", { "class": "fl-meter-band", style: { left: pos(o.band[0]) + "%", width: (pos(o.band[1]) - pos(o.band[0])) + "%" } }));
    if (toNum(o.value) !== null) track.appendChild(el("div", { "class": "fl-meter-fill", style: { width: pos(o.value) + "%" } }));
    (o.ticks || []).forEach(function (t) { track.appendChild(el("div", { "class": "fl-meter-tick", style: { left: pos(t.value) + "%" } })); });
    root.appendChild(track);
    if (o.ticks && o.ticks.length) {
      var sc = el("div", { "class": "fl-meter-scale", attrs: { "aria-hidden": "true" } });
      o.ticks.forEach(function (t) {
        var p = pos(t.value);
        sc.appendChild(el("span", { "class": p <= 2 ? "is-first" : p >= 98 ? "is-last" : null, text: t.label !== undefined ? t.label : f(t.value), style: { left: p + "%" } }));
      });
      root.appendChild(sc);
    }
    if (container) container.appendChild(root);
    return root;
  }

  /** Page section with an h2 heading. o: {id, title, subtitle} -> {root, body} */
  function section(container, o) {
    container = resolveContainer(container);
    o = o || {};
    var root = el("section", { "class": "page-section", id: o.id || null });
    if (o.title || o.subtitle) {
      var h = el("div", { "class": "fl-section-head" });
      if (o.title) h.appendChild(el("h2", { text: o.title }));
      if (o.subtitle) h.appendChild(el("p", { text: o.subtitle }));
      root.appendChild(h);
    }
    var body = el("div", { "class": "stack" });
    root.appendChild(body);
    if (container) container.appendChild(root);
    return { root: root, body: body };
  }

  /** Grid wrapper: cols 2 | 3 | 'kpis' -> element appended to container. */
  function grid(container, cols, children) {
    container = resolveContainer(container);
    var root = el("div", { "class": cols === 3 ? "grid-3" : cols === "kpis" ? "kpis" : "grid-2" }, children);
    if (container) container.appendChild(root);
    return root;
  }

  /** Muted note paragraph. */
  function note(container, text) {
    container = resolveContainer(container);
    var p = el("p", { "class": "fl-note", text: text });
    if (container) container.appendChild(p);
    return p;
  }
  /** Code: inline <code> or block <pre> when o.block. */
  function code(text, o) {
    return o && o.block ? el("pre", { "class": "fl-code", text: text }) : el("code", { "class": "fl-code", text: text });
  }

  FL.ui = {
    el: el, svg: svg, clear: clear, append: append, uid: uid, icon: icon, key: key, legend: legend,
    card: card, kpis: kpis, badge: badge, callout: callout, empty: empty, remedies: remedies, table: table,
    tabs: tabs, segmented: segmented, details: details, dl: dl, meter: meter, section: section,
    grid: grid, note: note, code: code, toc: toc, defer: defer, onTheme: onTheme, scrollCue: scrollCue,
    _defer: deferApi
  };

  /* ====================================================================== */
  /* FL.pages — registry (pages/*.js call FL.pages.register before boot)     */
  /* ====================================================================== */

  var registry = {};
  var order = [];
  FL.pages = {
    /** register({id, title, icon, subtitle, requiresData=true, render(root, ctx)}) */
    register: function (page) {
      if (!page || !page.id || typeof page.render !== "function") {
        if (window.console) console.error("FL.pages.register: need {id, render}", page);
        return;
      }
      if (!registry[page.id]) order.push(page.id);
      registry[page.id] = page;
      if (FL.app && FL.app._onRegister) FL.app._onRegister(page.id);
    },
    get: function (id) { return registry[id] || null; },
    list: function () { return order.slice(); }
  };
})();
