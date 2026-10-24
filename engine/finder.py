"""Factor Finder command line: environment checks and the label-free generator probe.

Usage (from engine/):

    .venv/bin/python -B finder.py --check           # read-only environment checks; exit 1 on any FAIL
    .venv/bin/python -B finder.py --generator-probe [--n 2000] [--seed 1]   # label-free, read-only

THE GENERATOR PROBE
    --generator-probe draws N random formulas from the generator's terminals (21
    library-named series and 4 Factorlab-only series, eval_config [generator.terminals])
    and the library operator registry. It reports how many are well-formed and the bands
    of their largest absolute correlation with the novelty index: the library definitions,
    your registered tests, the AlphaForge formulas and the risk proxies
    (factorfinder/probe.py). It reads no label and writes nothing.

THE CHECKS (--check)
    eval_config    eval_config.toml loads and is valid; its two version hashes
    data-through   the newest closed daily bar on disk (not after the UTC date)
    no network     ingest.py and ccxt are not imported
    probe inputs   the files that the probe reads (INFO: not a FAIL when absent)

RULES
    * Finder writes nothing.
    * No network: Finder never imports ingest.py.

EXIT CODES
    0  ok        1  a --check FAIL        2  an invalid option value
"""

from __future__ import annotations

import argparse
import sys

import config as C

# --------------------------------------------------------------------------- --check


def _probe_inputs() -> list[str]:
    """The probe's input files that are not on disk (an empty list when all are there)."""
    import libfactors as LF

    need = {
        "perp bars (data/raw)": any((C.RAW_OHLCV / f"{C.sym_to_file(s)}.parquet").exists() for s in C.UNIVERSE),
        "label table (data/labels/labels.parquet)": (C.LABELS / "labels.parquet").exists(),
        "library status (data/library_on_engine/status.parquet)": (LF.out_dir() / "status.parquet").exists(),
        "library values (data/library_on_engine/values/)": LF.has_values(),
    }
    return [k for k, ok in need.items() if not ok]


def check() -> tuple[list[str], int]:
    """Read-only environment checks. Returns (lines, number of FAILs)."""
    import factorfinder as F
    from factorfinder import config_io as K

    rows: list[tuple[str, str, str]] = []

    def add(ok: bool | None, name: str, detail: str) -> None:
        rows.append(("PASS" if ok else "INFO" if ok is None else "FAIL", name, detail))

    add(None, "data root", F.repo_label(C.DATA))
    try:
        cfg = K.load()
        bad = K.validate(cfg)
        ver = K.versions(cfg)
        add(
            not bad,
            "eval_config",
            "; ".join(bad)
            or f"valid; eval {ver['eval_config_version'][:12]}, generator {ver['generator_config_version'][:12]}",
        )
    except Exception as exc:  # noqa: BLE001 - any parse error is a FAIL, reported by class only
        add(False, "eval_config", f"cannot load: {type(exc).__name__}")
    try:
        through = F.data_through()
        add(through <= F.utc_today(), "data-through", f"{through.isoformat()} (not after the UTC wall-clock date)")
    except F.FinderError as exc:
        add(False, "data-through", str(exc))
    leaked = [m for m in ("ingest", "ccxt") if m in sys.modules]
    add(
        not leaked,
        "no network imports",
        ", ".join(leaked) + " imported" if leaked else "ingest.py and ccxt are not imported",
    )
    missing = _probe_inputs()
    add(
        None,
        "probe inputs",
        "all on disk" if not missing else "not on disk: " + "; ".join(missing) + " (run libfactors.py --compute)",
    )
    lines = [f"  {s:<4}  {n:<20} {d}" for s, n, d in rows]
    fails = sum(1 for s, _, _ in rows if s == "FAIL")
    graded = sum(1 for s, _, _ in rows if s != "INFO")
    lines.append(f"\n{graded - fails}/{graded} finder checks passed")
    return lines, fails


# --------------------------------------------------------------------------- --generator-probe


def generator_probe(n: int | None, seed: int | None) -> int:
    """The label-free probe (factorfinder/probe.py): read-only."""
    from factorfinder import probe

    n = probe.DEFAULT_N if n is None else n
    seed = probe.DEFAULT_SEED if seed is None else seed
    if n < 1:
        print("finder: --n must be at least 1. Nothing was run.")
        return 2
    result = probe.run(n=n, seed=seed, log=lambda m: print(m, flush=True))
    print("\n".join(probe.lines(result)))
    return 0


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="read-only environment checks")
    g.add_argument(
        "--generator-probe", action="store_true", help="label-free, read-only probe of the room left for new formulas"
    )
    ap.add_argument("--n", type=int, default=None, help="--generator-probe: formulas to draw (default 2000)")
    ap.add_argument("--seed", type=int, default=None, help="--generator-probe: seed (default 1)")
    a = ap.parse_args(argv)
    if (a.n is not None or a.seed is not None) and not a.generator_probe:
        ap.error("--n and --seed apply to --generator-probe only")
    if a.check:
        lines, fails = check()
        print("\n".join(lines))
        return 1 if fails else 0
    return generator_probe(a.n, a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
