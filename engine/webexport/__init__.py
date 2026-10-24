"""Section builders for the static web dashboard (web/).

Each section module ``webexport/<section>.py`` exposes ``build() -> dict``.
``export_web.py`` imports it, cleans the result (NaN/inf -> None), checks
the size budget and writes ``web/data/<section>.js``. Builders are read-only
over ``config.DATA``: they never write anything themselves. The section
``library`` (the Factor library page) also opens the factor library catalog under
``<repo>/library/catalog`` read-only; it reads the library computed on the engine's data
(``library_on_engine/``, written by libfactors.py), never its git-ignored ``values/``
shards. The section ``mine`` (the page My factors) reads the trial ledger ``trials/``
(engine/trials.py) and its result files.

SECTIONS is the export order and the order of meta.sections. The route registry in
web/assets/fl-app.js lists the same ids in the same order (plus the "factor" route, the page
of one registered test, which reads the section "mine"); the nav groups the routes in
NAV_GROUPS. index.html loads only meta.js:
the shell loads data/<section>.js and pages/<section>.js when a route opens.
test_export_web.py pins all of this.

A builder can also write part files that a page loads on demand (data/<section>.<part>.js):
it returns ``common.with_parts(payload, {part: payload, ...})``. See ``common.with_parts``.

Shared helpers live in ``webexport.common``. The ``library`` section keeps its code in three
more modules (``webexport/library_catalog.py``, ``library_engine.py`` and
``library_binance.py``); ``webexport/library.py`` holds its build() and gives their names.
"""

# Export order (the nav order is NAV_GROUPS in web/assets/fl-app.js).
SECTIONS = (
    "library",
    "research",
    "mine",
    "system",
)
