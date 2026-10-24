"""Command-line tool: ``factorlab <command>`` or ``python -m factorlab <command>``.

``--db`` defaults to ``catalog/factors.db`` (found next to the package, so the
commands work from any directory). ``./setup.sh`` builds it with ``build``. The
read-only commands open it with SQLite's ``mode=ro``. ``build`` writes the catalog
when it does not exist and refuses to replace it without ``--overwrite-catalog``.
``compute`` and ``evaluate`` refuse the catalog folder without
``--overwrite-catalog``: work on a copy under ``data/`` (see ``factorlab.paths``).
``show`` also prints the two commands of the engine that check a definition and
register it for a test (``engine/judge.py``). ``download`` is the only command that uses
the network, and only with ``FACTORLAB_OFFLINE=0`` (see ``factorlab.network``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import paths
from .network import NetworkDisabled, require_network
from .paths import CatalogWriteError, refuse_catalog

# the commands that check one definition and register it for a test (engine/judge.py);
# "cd engine" works from the repository root, so the first line says where to run them. Each
# command runs in a subshell, so the shell stays in the repository root for the next command
JUDGE_WHERE = "Run these commands from the repository root:"
JUDGE_COMMANDS = (
    ("Check this formula", "(cd engine && .venv/bin/python -B judge.py check --library {id})"),
    ("Register it for a test", "(cd engine && .venv/bin/python -B judge.py register --library {id})"),
)


def _count(n: int, one: str, many: str | None = None) -> str:
    """'1 factor', '2 factors': the number and the noun in the form that agrees with it."""
    return f"{n} {one if n == 1 else (many or one + 's')}"


def _select(db, args, panels):
    """Return the factors that the filters in ``args`` select and that ``panels`` can compute."""
    df = db.factors(family=args.family, category=args.category, like=args.like, available_fields=panels.keys())
    if args.ids:
        df = df[df["factor_id"].isin(args.ids.split(","))]
    if args.limit:
        df = df.head(args.limit)
    return df


def judge_lines(factor_id: str) -> list[str]:
    """Return the lines that ``show`` prints after a definition: where to run the commands, then each command."""
    return [JUDGE_WHERE] + [f"{label}: {cmd.format(id=factor_id)}" for label, cmd in JUDGE_COMMANDS]


def _show(db, factor: str) -> None:
    """Print one factor (id or name) and the commands that check it and register it. An unknown factor gives KeyError."""
    rec = db.get(factor)
    for k, v in rec.items():
        print(f"{k:>14}: {v}")
    print()
    for line in judge_lines(rec["factor_id"]):
        print(line)


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser of the ``factorlab`` command and its subcommands."""
    p = argparse.ArgumentParser(
        prog="factorlab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--db", help="factor database (default: catalog/factors.db)")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    b = sub.add_parser(
        "build",
        help="generate, validate and store the factor library",
        description="Generate the candidate factors, validate them on a panel, remove "
        "rank-equivalent factors and store the rest in the database.",
    )
    b.add_argument("--data", help="validate on panels in this directory instead of synthetic data")
    b.add_argument("--workers", type=int, help="number of worker processes (default: the number of CPUs)")
    b.add_argument("--export", help="directory for the CSV/Markdown export (default: the database's directory)")
    b.add_argument("--overwrite-catalog", action="store_true", help="permit the build to replace an existing catalog")

    sub.add_parser(
        "summary",
        help="print the number of factors for each family and category (read-only)",
        description="Print the number of factors for each family and category. The command does not "
        "change the database.",
    )
    s = sub.add_parser(
        "show",
        help="show one factor and the commands that test it (read-only)",
        description="Show one factor definition and the commands of the engine that check it and "
        "register it for a test. The command does not change the database.",
    )
    s.add_argument("factor", help="factor id (for example CF00001) or factor name")

    texts = {
        "list": (
            "list and filter factors (read-only)",
            "List the factors that agree with the filters. The command does not change the database.",
        ),
        "compute": (
            "compute factor values on panels and store them",
            "Compute the selected factors on the panels in --data and store the values in the database.",
        ),
        "evaluate": (
            "compute IC, ICIR and long-short statistics and store them",
            "Compute rank IC, ICIR, long-short and turnover statistics of the selected factors on the "
            "panels in --data and store them in the database.",
        ),
    }
    for name, (short, long) in texts.items():
        q = sub.add_parser(name, help=short, description=long)
        q.add_argument("--family", help="only factors of this family")
        q.add_argument("--category", help="only factors of this category")
        q.add_argument("--like", help="only factors whose name or description contains this text (SQL LIKE)")
        q.add_argument("--ids", help="comma separated factor ids")
        q.add_argument("--limit", type=int, help="at most this number of factors")
        if name != "list":
            q.add_argument("--data", required=True, help="directory with raw panels (*.csv.gz)")
            q.add_argument("--benchmark", default="BTCUSDT", help="benchmark column of the panels (default: BTCUSDT)")
            q.add_argument("--overwrite-catalog", action="store_true", help="permit a write into the catalog folder")
        if name == "evaluate":
            q.add_argument(
                "--horizons", default="1,5,20", help="comma separated forward horizons in bars (default: 1,5,20)"
            )
            q.add_argument("--dataset", default="default", help="label of the dataset in the stored statistics")
            q.add_argument(
                "--periods-per-year",
                type=int,
                default=365,
                help="bars per year, for the annualized Sharpe ratio (default: 365)",
            )

    d = sub.add_parser(
        "download",
        help="download Binance public panels (needs FACTORLAB_OFFLINE=0)",
        description="Download Binance public market data for the symbols and save one panel file "
        "for each field. The command uses the network only with FACTORLAB_OFFLINE=0.",
    )
    d.add_argument("--symbols", required=True, help="comma separated, e.g. BTCUSDT,ETHUSDT")
    d.add_argument("--start", default="2021-01-01", help="first day (UTC, default: 2021-01-01)")
    d.add_argument("--end", help="last time (UTC, default: now)")
    d.add_argument("--interval", default="1d", help="bar interval, for example 1d or 4h (default: 1d)")
    d.add_argument("--market", default="um", choices=["um", "spot"], help="USD-M perpetuals (um) or spot (default: um)")
    d.add_argument(
        "--no-derivatives", action="store_true", help="download no funding, premium, open interest or ratios"
    )
    d.add_argument("--metrics-archive", action="store_true", help="full OI/ratio history from data.binance.vision")
    d.add_argument("--out", required=True, help="output directory (one <field>.csv.gz for each field)")
    return p


def main(argv=None) -> int:
    """Run the ``factorlab`` command with the arguments ``argv`` (default: ``sys.argv[1:]``).

    Return the exit status: 0 on success, 1 for an unknown factor, 2 for a refused
    write, a refused download or a missing database.
    """
    p = _parser()
    args = p.parse_args(argv)
    if args.db is None:
        args.db = str(paths.CATALOG_DB)

    from .database import FactorDB

    try:
        if args.cmd == "build":
            export = args.export if args.export else str(Path(args.db).parent)
            guarded = [args.db, export]
            if paths.is_catalog_db(args.db) and not paths.CATALOG_DB.exists():
                # a new catalog: build may write catalog/factors.db and its export in catalog/ only
                guarded = [] if Path(export).expanduser().resolve() == paths.CATALOG_DIR.resolve() else [export]
            refuse_catalog(*guarded, allow=args.overwrite_catalog, what="build")
        elif args.cmd in ("compute", "evaluate"):
            refuse_catalog(args.db, allow=args.overwrite_catalog, what=args.cmd)
        elif args.cmd == "download":
            refuse_catalog(args.out, what="download", option=False)
            # before the downloader is even imported: offline unless FACTORLAB_OFFLINE=0
            require_network("download")
    except (CatalogWriteError, NetworkDisabled) as exc:
        print(f"factorlab: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "build":
        from .build import build_database, export_catalog

        panels = None
        if args.data:
            from .data import load_panels

            panels = load_panels(args.data)
        target = Path(args.db)
        if target.exists():
            # an existing database is rebuilt in place: its other tables stay
            build_database(target, panels=panels, workers=args.workers, export_dir=export)
            return 0
        # a new database: build into a temporary file, then rename it. An interrupted
        # new build leaves no database at the target path. The export follows the rename.
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        partial.unlink(missing_ok=True)
        try:
            build_database(
                partial,
                panels=panels,
                workers=args.workers,
                export_dir=None,
                log=lambda msg: print(msg.replace(str(partial), str(target))),
            )
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        if export:
            with FactorDB(target) as db:
                export_catalog(db, export)
        print(f"wrote {target}")
        return 0

    if args.cmd == "download":
        from .data import save_panels
        from .data.binance import download_panels

        panels = download_panels(
            args.symbols.split(","),
            args.start,
            args.end,
            args.interval,
            args.market,
            derivatives=not args.no_derivatives,
            metrics_archive=args.metrics_archive,
        )
        save_panels(panels, args.out)
        print(f"saved {_count(len(panels), 'field')} to {args.out}")
        return 0

    if args.cmd in ("summary", "show", "list"):
        try:
            db = FactorDB(args.db, readonly=True)
        except FileNotFoundError as exc:
            msg = paths.NO_CATALOG if paths.is_catalog_db(args.db) else f"factorlab: {exc}"
            print(msg, file=sys.stderr)
            return 2
        with db:
            if args.cmd == "summary":
                print(_count(db.count(), "factor"))
                with pd.option_context("display.max_rows", 500):
                    print(db.summary().to_string(index=False))
            elif args.cmd == "show":
                try:
                    _show(db, args.factor)
                except KeyError:
                    print(f"factorlab: no factor with id or name {args.factor!r}", file=sys.stderr)
                    return 1
            else:
                df = db.factors(family=args.family, category=args.category, like=args.like)
                if args.ids:
                    df = df[df["factor_id"].isin(args.ids.split(","))]
                df = df.head(args.limit) if args.limit else df
                with pd.option_context("display.max_rows", 1000, "display.max_colwidth", 100, "display.width", 250):
                    print(df[["factor_id", "name", "category", "expression"]].to_string(index=False))
                print(f"({_count(len(df), 'factor')})")
        return 0

    with FactorDB(args.db) as db:
        from .data import load_panels
        from .evaluate import evaluate_factor
        from .expr import Evaluator

        panels = load_panels(args.data)
        ev = Evaluator(panels, benchmark=args.benchmark)
        sel = _select(db, args, panels)
        print(f"{args.cmd} {_count(len(sel), 'factor')}")
        stats = []
        for i, r in enumerate(sel.itertuples(), 1):
            values = ev(r.expression)
            if args.cmd == "compute":
                db.write_values(r.factor_id, values)
            else:
                for h in map(int, args.horizons.split(",")):
                    st = evaluate_factor(values, panels["close"], h, periods_per_year=args.periods_per_year)
                    st.update(factor_id=r.factor_id, dataset=args.dataset)
                    stats.append(st)
                if len(stats) > 2000:
                    db.write_stats(stats)
                    stats = []
            if i % 500 == 0:
                print(f"  {i}/{len(sel)}", file=sys.stderr)
        if stats:
            db.write_stats(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
