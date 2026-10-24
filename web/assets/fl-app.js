/* Factorlab dashboard — app shell: header, snapshot line, theme toggle, nav, hash router and
 * the on-demand loader of the data and page files. Loaded LAST in index.html (after
 * fl-core.js, fl-charts.js and data/meta.js). API reference: web/README.md.
 */
(function () {
  "use strict";

  var FL = window.FL;
  if (!FL || !FL.ui || !FL.pages || !FL.load) { document.body.appendChild(document.createTextNode("Factorlab: assets/fl-core.js failed to load.")); return; }
  var ui = FL.ui, el = ui.el, fmt = FL.fmt;

  /**
   * Route registry: every route id, its fallback title and icon. The page file is
   * pages/<id>.js and its data file is data/<id>.js, loaded when the route opens.
   *   data: the data sections to load (default [id]; a page can name more in its register()).
   *   param: true for a route that carries an id in its hash (#/factor/<id>[/<card>]).
   *   nav: false keeps a route out of the nav.
   * Route ids are stable: renaming a page never renames its hash. The nav order is NAV_GROUPS.
   */
  var SECTIONS = [
    { id: "library", title: "Factor library", icon: "library" },
    { id: "research", title: "Search tools", icon: "research" },
    { id: "mine", title: "My factors", icon: "mine" },
    { id: "system", title: "System", icon: "system" },
    { id: "factor", title: "Factor test", icon: "factors", param: true, nav: false, data: ["mine"] }
  ];
  /**
   * The nav: the find tools, the tests of your factors, then the system. Every id is a SECTIONS route.
   */
  var NAV_GROUPS = [
    { id: "find", label: "Find", items: ["library", "research"] },
    { id: "test", label: "Test", items: ["mine"] },
    { id: "system", label: "System", items: ["system"] }
  ];
  // the landing page (an empty hash, an unknown route and the brand link): My factors
  var DEFAULT = "mine";
  // the id in #/factor/<id>: letters, digits, "_", "-", ".", ":"
  var PARAM_RE = /^[A-Za-z0-9_.:-]{1,128}$/;
  var EXPORT_CMD = FL.cmd.export, SETUP_CMD = FL.cmd.setup;
  // the banner of the synthetic demo data (meta.demo, from engine/demo_data.py): on every page
  var DEMO_TEXT = "Synthetic demo data. Not market data.";
  function isDemo(meta) { return !!meta && meta.demo === true; }
  function demoBanner() {
    return ui.callout("warning", DEMO_TEXT, ["A random generator made every value on this page (engine/demo_data.py). " +
      "No value comes from an exchange, and no result is evidence about a real factor."], { className: "fl-demo-banner" });
  }

  /*
   * Nav icons the core set (FL.ui.icon) does not have, in its format: 24x24 stroke
   * paths. "library" is three book spines and a leaning fourth (the Factor library page);
   * "mine" is a clipboard with a tick (the page My factors).
   */
  var NAV_ICONS = {
    library: ["M4 4h4v16H4z", "M8 4h4v16H8z", "M8 8h4", "M14.6 5.2l3.9-1.1 3.9 14.8-3.9 1.1z"],
    mine: ["M9 3h6v3H9z", "M7 4.5H5V21h14V4.5h-2", "M8.5 13.5l2.5 2.5 4.5-5"]
  };
  function navIcon(name) {
    var paths = NAV_ICONS[name];
    if (!paths) return ui.icon(name);
    var node = ui.svg("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 1.75,
      "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", focusable: "false" });
    paths.forEach(function (d) { node.appendChild(ui.svg("path", { d: d })); });
    return node;
  }

  var params;
  try { params = new URLSearchParams(window.location.search); } catch (e) { params = { has: function () { return false; }, get: function () { return null; } }; }

  // cardWait: the scroll job at which the card of the route was not yet on the page (else null)
  var state = { page: null, param: null, badParam: false, card: null, cardWait: null, rendered: false, token: 0, life: null, pageMissing: false };
  var dom = {};

  /* ------------------------------------------------------------------ routing */
  function routeDef(id) { return SECTIONS.filter(function (s) { return s.id === id; })[0] || null; }
  function enc(s) { return encodeURIComponent(String(s)).replace(/%3A/gi, ":"); }
  function decode(p) { try { return decodeURIComponent(p); } catch (e) { return p; } }
  function parseHash() {
    var h = (window.location.hash || "").replace(/^#\/?/, "");
    var parts = h.split("/").filter(Boolean).map(decode);
    var r = { page: parts[0] || null, param: null, badParam: false, card: null };
    var def = routeDef(r.page);
    if (def && def.param) {
      r.param = parts[1] || null;
      r.card = parts[2] || null;
      if (r.param !== null && !PARAM_RE.test(r.param)) { r.badParam = true; r.param = null; }
    } else r.card = parts[1] || null;
    return r;
  }
  function knownIds() {
    var ids = SECTIONS.map(function (s) { return s.id; });
    FL.pages.list().forEach(function (id) { if (ids.indexOf(id) < 0) ids.push(id); });
    return ids;
  }
  /** "#/<page>[/<param>][/<card>]"; param only for a route with param: true. */
  function href(page, card, param) {
    var def = routeDef(page), h = "#/" + enc(page);
    if (def && def.param && param !== undefined && param !== null && param !== "") h += "/" + enc(param);
    if (card) h += "/" + enc(card);
    return h;
  }
  /** "#/factor/<id>[/<card>]" */
  function factorHref(id, card) { return href("factor", card || null, id); }
  function go(page, card, param) {
    var h = href(page, card, param);
    if (window.location.hash === h) route(); else window.location.hash = h;
  }

  function route() {
    var r = parseHash();
    var page = r.page;
    if (!page || knownIds().indexOf(page) < 0) {
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, "", window.location.pathname + window.location.search + href(DEFAULT));
      }
      r = { page: DEFAULT, param: null, badParam: false, card: null };
      page = DEFAULT;
    }
    if (page === state.page && r.param === state.param && r.badParam === state.badParam && state.rendered) {
      // same page: a card route scrolls to the card, the page root (or a redirect) to the top
      state.card = r.card;
      state.cardWait = null;
      if (!r.card || !scrollToCard(r.card)) {
        cancelScroll();
        window.scrollTo(0, 0);
        if (r.card) state.cardWait = scrollJob; // a card that the page draws later (ctx.cardReady)
      }
      return;
    }
    state.page = page;
    state.param = r.param;
    state.badParam = r.badParam;
    state.card = r.card;
    renderPage();
  }

  /*
   * Scroll a card into view, then re-align it while the page settles: charts measure
   * themselves after layout (ResizeObserver, 0 ms timers) and page-local components can
   * change the height of everything above the card. Each re-check only moves the page
   * if the card drifted and the reader has not scrolled or pressed a key meanwhile.
   * A card that is still in the deferred-render queue is rendered first.
   */
  var scrollJob = 0;
  function cancelScroll() { scrollJob += 1; }
  ["wheel", "touchstart", "keydown", "mousedown"].forEach(function (ev) {
    window.addEventListener(ev, function () { cancelScroll(); }, { passive: true, capture: true });
  });
  function scrollToCard(cardId) {
    if (!cardId) return false;
    var target = document.getElementById(cardId);
    if (!target && ui._defer.pending()) { ui._defer.flush(cardId); target = document.getElementById(cardId); }
    if (!target) return false;
    var job = ++scrollJob;
    var lastY = null;
    function align() {
      if (job !== scrollJob || !target.isConnected) return;
      if (lastY !== null && Math.abs(window.pageYOffset - lastY) > 2) { cancelScroll(); return; } // someone else scrolled
      var want = parseFloat(getComputedStyle(target).scrollMarginTop) || 0;
      if (lastY === null || Math.abs(target.getBoundingClientRect().top - want) > 2) {
        try { target.scrollIntoView({ block: "start", behavior: "auto" }); } catch (e) { target.scrollIntoView(); }
      }
      lastY = window.pageYOffset;
    }
    window.requestAnimationFrame(function () {
      align();
      [60, 250, 700].forEach(function (ms) { setTimeout(function () { window.requestAnimationFrame(align); }, ms); });
    });
    return true;
  }

  function sectionMeta(id) {
    var s = routeDef(id);
    var p = FL.pages.get(id);
    return {
      id: id,
      title: (p && p.title) || (s && s.title) || id,
      icon: (p && p.icon) || (s && s.icon) || "overview",
      subtitle: p && p.subtitle,
      page: p
    };
  }
  /** The data sections a route loads: the route's data list (default [id]), then the page's own list. */
  function dataFor(id) {
    var def = routeDef(id), p = FL.pages.get(id), out = [];
    function add(s) { if (s && out.indexOf(s) < 0) out.push(s); }
    (def && def.data ? def.data : [id]).forEach(add);
    if (p) { add(p.section); (p.data || []).forEach(add); }
    return out;
  }
  /** The section whose data the page reads as ctx.section (page.section, else the route id). */
  function primarySection(id) {
    var p = FL.pages.get(id), def = routeDef(id);
    if (p && p.section) return p.section;
    if (def && def.data) return def.data[0] || null;
    return id;
  }

  /* ------------------------------------------------------------------ page life */
  /*
   * Each page render owns one "life": an AbortSignal (or a small stand-in) that the shell
   * aborts before the next page renders. Listeners added with ctx.listen / ctx.onTheme, or
   * with {signal: ctx.signal}, end with it, so a route change leaks no listener.
   */
  function newLife() {
    var ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
    var offs = [];
    var shim = { aborted: false, _fns: [], addEventListener: function (t, f) { if (t === "abort") shim._fns.push(f); }, removeEventListener: function () {} };
    var life = {
      signal: ctl ? ctl.signal : shim,
      listen: function (target, type, fn, opts) {
        if (!target || typeof target.addEventListener !== "function") return function () {};
        target.addEventListener(type, fn, opts);
        var off = function () { target.removeEventListener(type, fn, opts); };
        offs.push(off);
        return off;
      },
      abort: function () {
        offs.splice(0).forEach(function (f) { try { f(); } catch (e) { /* ignore */ } });
        if (ctl) ctl.abort(); else { shim.aborted = true; shim._fns.splice(0).forEach(function (f) { try { f(); } catch (e) { /* ignore */ } }); }
      }
    };
    life.onTheme = function (fn) { return life.listen(window, "fl:themechange", fn); };
    return life;
  }

  function renderPage() {
    var id = state.page, main = dom.main;
    var token = ++state.token;
    cancelScroll();
    if (state.life) state.life.abort(); // every listener of the old page ends here
    state.life = newLife();
    ui._defer.reset();
    ui.clear(main);
    if (FL.charts && FL.charts.prune) FL.charts.prune(true); // every chart of the old page is gone
    state.rendered = false;
    state.pageMissing = false;
    var page = FL.pages.get(id);
    var need = dataFor(id).filter(function (s) { return FL.data[s] === undefined && FL.load.state("data/" + s + ".js") === null; });
    if (page && !need.length) { finish(token); return; }
    // load the page file and its data files, then render (a short line meanwhile)
    var m = sectionMeta(id);
    document.title = m.title + " · Factorlab";
    updateNav();
    main.appendChild(el("div", { "class": "fl-page", attrs: { "data-page": id } }, [
      el("div", { "class": "fl-page-head" }, [el("h1", { text: m.title })]),
      el("p", { "class": "fl-loading", text: "Loading the page files.", attrs: { role: "status" } })
    ]));
    var jobs = need.map(function (s) { return FL.load.data(s); });
    jobs.push(page ? Promise.resolve(null) : FL.load.page(id));
    Promise.all(jobs).then(function () {
      if (token !== state.token) return null;
      // the page can name more data sections in its register()
      var more = dataFor(id).filter(function (s) { return FL.data[s] === undefined && FL.load.state("data/" + s + ".js") === null; });
      return Promise.all(more.map(function (s) { return FL.load.data(s); }));
    }).then(function () {
      if (token === state.token) finish(token);
    }, function (err) {
      if (window.console) console.error(err);
      if (token === state.token) finish(token);
    });
  }

  function fileProblem(path) {
    var st = FL.load.state(path);
    if (st === "broken") return path + " does not parse (or has a script error).";
    if (st === "missing") return path + " is not in this folder.";
    return null;
  }

  function finish(token) {
    if (token !== state.token) return;
    var id = state.page, m = sectionMeta(id), main = dom.main, life = state.life;
    ui.clear(main);
    document.title = m.title + " · Factorlab";
    updateNav();
    var pageRoot = el("div", { "class": "fl-page", attrs: { "data-page": id } });
    var h1 = el("h1", { text: m.title });
    var sub = el("p", { "class": "fl-page-sub", text: m.subtitle || "" });
    sub.hidden = !m.subtitle;
    var badges = el("div", { "class": "fl-page-badges" });
    pageRoot.appendChild(el("div", { "class": "fl-page-head" }, [h1, sub, badges]));
    if (isDemo(FL.data.meta)) pageRoot.appendChild(demoBanner());
    var body = el("div", { "class": "fl-page-body stack" });
    pageRoot.appendChild(body);
    main.appendChild(pageRoot);

    var page = m.page;
    var secId = primarySection(id);
    var needsData = !page || (page.requiresData !== false && secId !== null);
    var section = secId === null ? null : FL.data[secId];
    var meta = FL.data.meta || null;
    if (page && secId) {
      var st = FL.stale.section(meta, secId);
      if (st.stale) badges.appendChild(ui.badge("warning", st.reason));
    }
    if (!page) {
      state.pageMissing = true;
      ui.empty(body, fileProblem("pages/" + id + ".js") || "The page file pages/" + id + ".js is missing.", { title: "Page not available" });
    } else if (needsData && (section === undefined || section === null)) {
      var why = fileProblem("data/" + secId + ".js");
      ui.empty(body, (why ? why + " " : "") + "No data for this page.", { title: "No data", commands: ui.remedies() });
    } else if (needsData && section.available === false && typeof section.reason === "string") {
      // the exporter found no input data for this section (export_web.py, common.no_data_payload)
      ui.empty(body, "No data for this page: " + section.reason + ".", { title: "No data", commands: ui.remedies() });
    } else {
      ui._defer.begin();
      try {
        page.render(body, makeCtx(id, secId, section, life, { h1: h1, sub: sub, badges: badges }));
      } catch (err) {
        if (window.console) console.error(err);
        body.appendChild(ui.callout("critical", "This page failed to render",
          ["Error: " + (err && err.message ? err.message : String(err)), "Other pages are unaffected. Export the pages again, or check the browser console."]));
      }
      // the deferred parts: render at once until about 1.5 screens are full, the rest in chunks
      var limit = (window.innerHeight || 800) * 1.5;
      ui._defer.fill(function () { return !state.card && body.getBoundingClientRect().bottom < limit; });
    }
    pageRoot.appendChild(footer());
    state.rendered = true;
    window.scrollTo(0, 0);
    try { main.focus({ preventScroll: true }); } catch (e) { /* old browsers */ }
    // a card that the page draws later (after a part file loads): the page calls ctx.cardReady(),
    // which scrolls to it unless the reader scrolled or pressed a key first (cancelScroll)
    state.cardWait = state.card && !scrollToCard(state.card) ? scrollJob : null;
    prefetchPages();
  }

  function makeCtx(id, secId, section, life, head) {
    // after the user leaves the page, a kept closure of the page does not hold the old page tree
    if (life.signal && life.signal.addEventListener) life.signal.addEventListener("abort", function () { head.h1 = head.sub = head.badges = null; });
    var token = state.token;
    return {
      id: id,
      data: FL.data,
      section: section,
      sectionId: secId,
      meta: FL.data.meta || null,
      colors: FL.colors,
      fmt: FL.fmt,
      ui: FL.ui,
      charts: FL.charts,
      params: params,
      signal: life.signal,
      listen: life.listen,
      onTheme: life.onTheme,
      defer: ui.defer,
      loadPart: function (part) { return FL.load.part(secId || id, part); },
      parts: function () { return FL.load.parts(secId || id); },
      // a page that draws its cards after a part file loads calls this when they are on the page:
      // the shell then scrolls to the card of the route (once; not after a scroll of the reader)
      cardReady: function () {
        if (token !== state.token || !state.card || state.cardWait === null || state.cardWait !== scrollJob) return false;
        state.cardWait = null;
        return scrollToCard(state.card);
      },
      setTitle: function (title, subtitle) {
        if (!head.h1) return;
        if (title !== undefined && title !== null) { head.h1.textContent = String(title); document.title = String(title) + " · Factorlab"; }
        if (subtitle !== undefined) { head.sub.textContent = subtitle === null ? "" : String(subtitle); head.sub.hidden = !subtitle; }
      },
      addBadge: function (node) { if (head.badges) head.badges.appendChild(node); },
      nav: {
        current: id,
        card: state.card,
        param: state.param,
        badParam: state.badParam,
        go: go,
        href: href,
        factorHref: factorHref
      }
    };
  }

  /* After the first page, load the other page files when the browser is idle (they are small),
     so the next route needs only its data file. */
  var prefetched = false;
  function prefetchPages() {
    if (prefetched) return;
    prefetched = true;
    var run = function () {
      SECTIONS.forEach(function (s) { if (!FL.pages.get(s.id) && FL.load.state("pages/" + s.id + ".js") === null) FL.load.page(s.id); });
    };
    if (window.requestIdleCallback) window.requestIdleCallback(run, { timeout: 4000 }); else setTimeout(run, 1500);
  }

  /* ------------------------------------------------------------------ header */
  function buildHeader() {
    var header = el("header", { "class": "fl-header" });
    var inner = el("div", { "class": "fl-header-inner" });
    var brand = el("a", { "class": "fl-brand", attrs: { href: href(DEFAULT), "aria-label": "Factorlab home: My factors" } }, [
      ui.icon("logo"), el("span", { text: "Factorlab" })
    ]);
    inner.appendChild(brand);
    dom.snapshot = el("div", { "class": "fl-snapshot", attrs: { "aria-live": "off" } });
    inner.appendChild(dom.snapshot);
    var themeHost = el("div", { "class": "fl-theme" });
    var themeCtl = ui.segmented(themeHost, {
      ariaLabel: "Color theme",
      iconsOnly: true,
      value: FL.theme.get(),
      options: [
        { value: "auto", label: "Auto (follow system)", icon: "auto" },
        { value: "light", label: "Light", icon: "sun" },
        { value: "dark", label: "Dark", icon: "moon" }
      ],
      onChange: function (v) { FL.theme.set(v); }
    });
    window.addEventListener("fl:themechange", function () { if (themeCtl.get() !== FL.theme.get()) themeCtl.set(FL.theme.get()); });
    inner.appendChild(themeHost);
    header.appendChild(inner);
    return header;
  }

  function renderSnapshot() {
    var box = dom.snapshot, meta = FL.data.meta;
    ui.clear(box);
    if (!meta) {
      box.appendChild(el("span", { text: "No snapshot loaded. To build the data: " }));
      box.appendChild(ui.code(SETUP_CMD));
      return;
    }
    var now = Date.now();
    var parts = [];
    // the data age comes first when closed bars are missing: definite, with the remedy
    var age = FL.stale.data(meta, now);
    if (isDemo(meta)) {
      // demo data has a fixed last bar: the banner takes the place of the data-age badge
      box.appendChild(ui.badge("warning", DEMO_TEXT, { title: "To build from market data, remove the demo data (engine/.venv/bin/python -B engine/demo_data.py --remove), then run " + SETUP_CMD + "." }));
    } else if (age && age.stale) {
      box.appendChild(ui.badge("warning", age.text + " To add them: " + SETUP_CMD,
        { title: "This page does not update itself. To add the new days, run " + SETUP_CMD + " in the repository folder. To rebuild only the pages: " + EXPORT_CMD }));
    }
    if (meta.data_through) {
      var dt = el("span", { title: "Daily bars are stamped with their UTC open time; the " + meta.data_through + " bar closed at 00:00 UTC the next day." }, [
        el("span", { text: "Data through " }), el("strong", { text: meta.data_through }), el("span", { text: " UTC close" })
      ]);
      parts.push(dt);
    } else parts.push(el("span", { text: "Data date unknown" }));
    if (meta.generated_at) parts.push(el("span", { text: "exported " + fmt.datetime(meta.generated_at) }));
    var gen = fmt.parseDate(meta.generated_at);
    if (gen !== null) {
      var ago = now - gen;
      parts.push(el("span", { title: "Time since export, measured by this browser's clock", text: ago >= 0 && ago < 6e4 ? "just now" : fmt.age(ago) + (ago >= 0 ? " ago" : "") }));
    }
    parts.forEach(function (p, i) {
      if (i) box.appendChild(el("span", { "class": "sep", text: "·", attrs: { "aria-hidden": "true" } }));
      box.appendChild(p);
    });
  }

  /* ------------------------------------------------------------------ nav */
  function buildNav() {
    var nav = el("nav", { "class": "fl-nav", attrs: { "aria-label": "Sections" } });
    dom.navGroups = el("div", { "class": "fl-nav-groups" });
    nav.appendChild(dom.navGroups);
    dom.navFoot = el("div", { "class": "fl-nav-foot" });
    nav.appendChild(dom.navFoot);
    dom.nav = nav;
    ui.scrollCue(nav);
    return nav;
  }
  /** true when the route has no page file or no data (meta.sections, or a failed load) */
  function isMissing(id) {
    var p = FL.pages.get(id);
    if (!p && FL.load.state("pages/" + id + ".js") && FL.load.state("pages/" + id + ".js") !== "pending" && FL.load.state("pages/" + id + ".js") !== "ok") return true;
    var sec = primarySection(id);
    if (!sec || (p && p.requiresData === false)) return false;
    var d = FL.data[sec];
    if (d && d.available === false && typeof d.reason === "string") return true;
    var ms = FL.data.meta && FL.data.meta.sections && FL.data.meta.sections[sec];
    if (ms && (!ms.present || ms.available === false)) return true;
    var ls = FL.load.state("data/" + sec + ".js");
    return ls === "missing" || ls === "broken";
  }
  function navLink(id) {
    var m = sectionMeta(id);
    var missing = isMissing(id);
    var a = el("a", { attrs: { href: href(id), "aria-current": id === state.page ? "page" : null }, title: missing ? "No data or page" : null }, [
      navIcon(m.icon), el("span", { text: m.title })
    ]);
    if (missing) {
      a.appendChild(el("span", { "class": "fl-nav-missing", attrs: { "aria-hidden": "true" } }));
      a.appendChild(el("span", { "class": "sr-only", text: " (not available)" }));
    }
    return el("li", null, a);
  }
  function updateNav() {
    var host = dom.navGroups;
    if (!host) return;
    ui.clear(host);
    var listed = [];
    NAV_GROUPS.forEach(function (g) {
      var lid = "fl-nav-g-" + g.id;
      var ul = el("ul", { attrs: { "aria-labelledby": lid } });
      g.items.forEach(function (id) { if (routeDef(id)) { ul.appendChild(navLink(id)); listed.push(id); } });
      host.appendChild(el("div", { "class": "fl-nav-group" }, [el("p", { "class": "fl-nav-label", id: lid, text: g.label }), ul]));
    });
    // pages that registered without a route of their own (a page file added to index.html)
    var extra = knownIds().filter(function (id) { var d = routeDef(id); return listed.indexOf(id) < 0 && !(d && d.nav === false) && FL.pages.get(id); });
    if (extra.length) {
      var ul = el("ul", { attrs: { "aria-label": "Other pages" } });
      extra.forEach(function (id) { ul.appendChild(navLink(id)); });
      host.appendChild(el("div", { "class": "fl-nav-extra" }, ul));
    }
    ui.clear(dom.navFoot);
    dom.navFoot.appendChild(el("p", { text: "Find a factor. Test it in a book." }));
    dom.navFoot.appendChild(el("p", { text: "All times UTC." }));
    var cur = host.querySelector("[aria-current=page]");
    if (cur && cur.scrollIntoView && window.matchMedia && window.matchMedia("(max-width: 599.98px)").matches) {
      try { cur.scrollIntoView({ block: "nearest", inline: "nearest" }); } catch (e) { /* ignore */ }
    }
  }
  var navTimer = null;
  function updateNavSoon() { clearTimeout(navTimer); navTimer = setTimeout(updateNav, 50); }

  /* ------------------------------------------------------------------ footer */
  // the footer gives the version and the licenses only: a page names its own missing or broken
  // data file, the navigation marks a page without data, and the page head marks a failed export
  function footer() {
    var meta = FL.data.meta, f = el("footer", { "class": "fl-footer" });
    // the version comes from the file VERSION through meta.js (engine/version.py); no version without it
    var ver = meta && typeof meta.version === "string" && meta.version ? "Factorlab " + meta.version + ". " : "";
    f.appendChild(el("p", { "class": "fl-footer-terms", text: ver + "Code: MIT License. Market data: OKX and Binance, under their own terms. Binance Vision archive data: CC BY-NC-SA 4.0 (credit: Binance Vision). See README, Data terms." }));
    return f;
  }

  /* ------------------------------------------------------------------ boot */
  function start() {
    var meta = FL.data.meta;
    FL.load.version = meta && meta.generated_at ? meta.generated_at : null;
    renderSnapshot();
    setInterval(renderSnapshot, 60e3);
    window.addEventListener("hashchange", route);
    route();
  }
  function boot() {
    var app = document.getElementById("fl-app") || document.body;
    ui.clear(app);
    app.appendChild(el("a", { "class": "skip-link", attrs: { href: "#fl-main" }, text: "Skip to content", on: { click: function (e) { e.preventDefault(); dom.main.focus(); } } }));
    app.appendChild(buildHeader());
    var layout = el("div", { "class": "fl-layout" });
    layout.appendChild(buildNav());
    dom.main = el("main", { "class": "fl-main", id: "fl-main", attrs: { tabindex: "-1" } });
    layout.appendChild(dom.main);
    app.appendChild(layout);
    // data/meta.js names the export. Served over http(s), a browser can keep an old copy in its
    // cache, so the shell loads it once more under a query that changes on every load. A file://
    // page reads the file from disk every time.
    var meta = FL.data.meta;
    if (!meta && (window.FL_MISSING || []).indexOf("data/meta.js") < 0) {
      var b = window.FL_BROKEN = window.FL_BROKEN || [];
      if (!b.some(function (x) { return x && x.file === "data/meta.js"; })) b.push({ file: "data/meta.js", parse: true });
    }
    if (/^https?:$/.test(window.location.protocol || "")) {
      FL.load.version = "boot-" + Date.now();
      FL.load.script("data/meta.js", null).then(start, start);
    } else start();
  }

  FL.app = {
    SECTIONS: SECTIONS,
    NAV_GROUPS: NAV_GROUPS,
    DEFAULT: DEFAULT,
    go: go,
    href: href,
    factorHref: factorHref,
    current: function () { return state.page; },
    param: function () { return state.param; },
    /** the AbortSignal of the page on screen (ends at the next route change) */
    signal: function () { return state.life ? state.life.signal : null; },
    rerender: function () { state.rendered = false; renderPage(); },
    _onRegister: function (id) {
      // a page that registers after boot: refresh the nav; render it if its route is on
      // screen with a "page not available" state (the loader renders its own pages)
      if (!dom.main) return;
      if (id === state.page && state.rendered && state.pageMissing) { state.rendered = false; renderPage(); } else updateNavSoon();
    }
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
