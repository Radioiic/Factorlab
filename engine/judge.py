"""Test your own factor in a book: the historical test, the registration and the weekly forward test.

Usage (from engine/):
    python -B judge.py check --formula="-ts_mean(ex_ret, 20)"          # the historical test; writes nothing
    python -B judge.py check --library CF00001
    python -B judge.py check --values my_factor.csv --spec long_only
    python -B judge.py register --formula="-ts_mean(ex_ret, 20)" --name reversal
    python -B judge.py log-values FT-0002 --values my_factor.csv     # log the new Sundays of a values file
    python -B judge.py log-week                                        # log the new forward weeks
    python -B judge.py status [FT-0001] [--json]
    python -B judge.py verify [--write-results]
    python -B judge.py withdraw FT-0001 --reason "text"
    python -B judge.py record-code-change --reason "text" [--note "text" ...]
    python -B judge.py spec [FILE|long_short|long_only] [--json]

The factor is a formula of the library expression language (--formula), a library definition
(--library) or a values file (--values: CSV or Parquet with the columns date, symbol, value). For
a formula that starts with -, write --formula=EXPR (argparse reads "--formula -x" as two options). The
book specification is a TOML file (--spec; default engine/bookspecs/long_short.toml). The
program reads the market data on disk and writes only data/trials/ (the trial ledger). It uses
no network. trials.py has the rules; rules.py has the statistics.

Exit status: 0 done; 1 a refusal (input, point in time, no history, duplicate, code change,
broken ledger, withdrawal rule); 2 usage; 3 a new alarm (log-week, verify).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import date
from pathlib import Path

import book
import rules
import trials
import userfactor as uf


def _pct(x) -> str:
    return "-" if x is None else f"{x:+.2%}"


def _f2(x) -> str:
    return "-" if x is None else f"{x:+.2f}"


def _f(x, spec: str = ".4f") -> str:
    return "-" if x is None else format(x, spec)


def _input(a: argparse.Namespace) -> dict:
    if a.formula is not None:
        return {"formula": a.formula, "source": a.source}
    if a.library is not None:
        return {"library": a.library, "source": a.source}
    if a.source is not None:
        raise uf.InputError("--source works only with --formula or --library.")
    return {"values_file": a.values}


def _options(a: argparse.Namespace) -> str:
    """The input and specification options of the command, for the printed register command.

    Each value is attached to its option (--formula=EXPR): argparse then cannot read a value
    that starts with "-" (for example -ts_mean(clv,7)) as an option. A file path is absolute,
    because the printed command starts with "cd engine"."""
    parts = []
    if a.formula is not None:
        parts.append("--formula=" + shlex.quote(a.formula))
    elif a.library is not None:
        parts.append("--library=" + shlex.quote(a.library))
    else:
        parts.append("--values=" + shlex.quote(str(Path(a.values).resolve())))
    if a.source is not None:
        parts.append("--source=" + a.source)
    if a.spec is not None:
        spec = a.spec if a.spec in book.SHIPPED_SPECS else str(Path(a.spec).resolve())
        parts.append("--spec=" + shlex.quote(spec))
    return " ".join(parts)


def print_check(r: dict, a: argparse.Namespace) -> None:
    f, h = r["factor"], r["historical"]
    if r["input"] == "formula":
        venue = "OKX" if f["source"] == "okx" else "Binance"
        what = f"library definition {f['library_id']}" if f.get("library_id") else "formula"
        print(f"Factor: {what} on the {venue} data")
        print(f"  {f['expression']}")
        print(f"  sha256 {f['expression_sha256'][:12]}")
        if r["pit"]["status"] == "proven":
            print(f"Point in time: proven at each Sunday cut ({trials.count_text(r['pit']['sundays'], 'Sunday')}).")
        else:
            print(f"Point in time: not proven: {trials.pit_why(r['pit_detail'])}.")
        print("Notes:" + ("" if f["notes"] else " none"))
        for n in f["notes"]:
            print(f"  {uf.NOTES[n]}")
    else:
        print(f"Factor: values file {f['file_name']}")
        print(f"  sha256 {f['file_sha256'][:12]}")
        print("Point in time: not proven (values file).")
        print("Notes: none")
    s = r["spec"]
    ctl = "no position" if s["body"]["control"]["kind"] == "zero" else "equal-weight book"
    costs = s["body"]["costs"]
    print(
        f"Book: {s['name']} version {s['version']} (sha256 {s['sha256'][:12]}). Control: {ctl}. "
        f"Costs: {costs['fee_bps']:g} + {costs['slippage_bps']:g} bps."
    )
    if h["result"] == "none":
        print("Historical test: none. The history has no complete week of values.")
    else:
        print(
            f"Historical test, {h['start']} to {h['end']} ({trials.count_text(h['weeks'], 'week')}, "
            f"{h['held_weeks']} with a factor position):"
        )
        print(f"  Total return after costs: {_pct(h['total_return'])}")
        print(f"  Sharpe ratio after costs: {_f2(h['sharpe'])}")
        if h["n_windows"]:
            print(
                f"  1-year windows with a positive Sharpe ratio: {h['n_positive']} of {h['n_windows']} "
                f"({h['share_positive']:.2f})"
            )
        else:
            print("  1-year windows with a positive Sharpe ratio: none (no 1-year window)")
        print(f"  Result: {h['word']}")
        for reason in h["reasons"]:
            print(f"  {reason}")
        btc = h["btc"] or {}
        if btc.get("total_return") is None:
            print(f"Context (no rule): BTC hold: not known. The data has no BTC close on {h['start']}.")
        else:
            print(
                f"Context (no rule): BTC hold total return {_pct(btc['total_return'])}, Sharpe ratio {_f2(btc.get('sharpe'))}."
            )
    n = r.get("factor_numbers") or {}
    if n.get("available", True) and n:
        print(
            f"Numbers (no rule): weekly IC {_f(n.get('ic_mean'), '+.4f')} (t {_f(n.get('ic_t'), '+.2f')}), "
            f"weekly rIC {_f(n.get('ric_mean'), '+.4f')} (t {_f(n.get('ric_t'), '+.2f')}), "
            f"persistence {_f(n.get('persistence'), '.2f')}, coverage {_f(n.get('coverage'), '.2f')}."
        )
    elif n:
        print(f"Numbers (no rule): not available ({n.get('reason')}).")
    big = r["largest"]
    print(
        "Largest |correlation| with a registered test: "
        + (f"{big['abs_corr']:.2f} ({big['test_id']})." if big else "none registered.")
    )
    lib = r.get("library_corr")
    if lib is not None:
        if lib.get("available") is False or lib.get("max_abs") is None:
            print("Largest |correlation| with a library definition: not available.")
        else:
            print(f"Largest |correlation| with a library definition: {lib['max_abs']:.2f} ({lib.get('nearest')}).")
    print(f"Related registered tests (|correlation| 0.85 or more): {', '.join(r['related']) or 'none'}.")
    if r["duplicate"]:
        print(f"Duplicate of {r['duplicate']}: the same book and the same factor values.")
    print(
        f"Forward test: first forward Sunday {r['first_forward_sunday']}. Sigma floor {r['sigma_floor']:.4g} ({r['sigma_floor_source']})."
    )
    proven = r["input"] == "values_file" or r["pit"]["status"] == "proven"
    if h["result"] in ("pass", "useless") and not r["duplicate"] and proven:
        if h["result"] == "useless":
            print(
                'You can register this test. Its verdict stays "useless (historical)" until the forward test decides.'
            )
        print(
            f"Register (run from the repository root): cd engine && .venv/bin/python -B judge.py register {_options(a)}"
        )


def print_status(d: dict) -> None:
    tests = d["tests"]
    if not tests:
        print("No test is registered.")
    for t in tests:
        what = t["expression"] if t["input"] == "formula" else f"values file {t['file']['name']}"
        src = f" ({t['source']})" if t["source"] else ""
        sp = t["spec"]
        print(f"{t['id']}  {t['name']}: {what}{src}")
        print(f"  Book: {sp['name']} version {sp['version']} (sha256 {sp['sha256'][:12]}).")
        hs = t["historical"]
        print(
            f"  Historical test: {hs['word']} (total return {_pct(hs['total_return'])}, "
            f"Sharpe ratio {_f2(hs['sharpe'])}); sigma floor {hs['sigma_floor']:.4g} ({hs['sigma_floor_source']})."
        )
        fw = t["forward"]
        if t["withdrawn"]:
            print("  Withdrawn before the first forward week.")
        elif fw["weeks"] == 0:
            print(f"  Forward test: waiting. The first forward week ends on {fw['next_week_end']}.")
        else:
            print(
                f"  Forward test: week {fw['weeks']} of {rules.MAX_WEEKS}, mean {_f(fw['mean'], '+.4f')}, "
                f"lower {_f(fw['lower'], '+.4f')}, upper {_f(fw['upper'], '+.4f')}, state {fw['state']}."
            )
        if fw.get("held"):
            print(f"  Held at {fw['held']['sunday']}: {fw['held']['reason']}.")
        print("  Verdict: none (withdrawn)." if t["withdrawn"] else f"  Verdict: {t['verdict']['word']}.")
        print(f"  Related tests: {', '.join(t['related']) or 'none'}. Alarms: {t['alarms']}.")
    tot = d["totals"]
    print(
        f"Tests: {tot['registered']} registered, {tot['withdrawn']} withdrawn. "
        f"States: {', '.join(f'{k} {v}' for k, v in tot['states'].items())}. "
        f"Verdicts: {', '.join(f'{k} {v}' for k, v in tot['verdicts'].items())}."
    )
    print(d["expected_false_useful"]["text"])
    other = [a for a in d["alarms"] if not a.get("test_id")]
    if other:
        print(f"Alarms of the data: {len(other)}.")
    if d["code"] is not None and not d["code"]["matches"]:
        print(
            f"The code differs from the ledger ({', '.join(d['code']['changed']) or 'config.py values'}). {trials.DRIFT_TEXT}"
        )
    print("The rule:")
    for line in d["rule"]["text"]:
        print(f"  {line}")


def print_spec(spec: book.BookSpec, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps({"body": spec.body, "sha256": spec.sha256, "book_key": spec.book_key}, indent=1, sort_keys=True)
        )
        return
    print(f"Book specification {spec.source}: name {spec.name}, version {spec.version}.")
    for table in ("book", "costs", "control", "forward"):
        for k, v in spec.body[table].items():
            print(f"  {table}.{k} = {json.dumps(v)}")
    print(f"sha256 {spec.sha256}")
    print(f"book_key {spec.book_key}")


def _day(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a date YYYY-MM-DD: {text!r}") from None


def _add_input(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--formula",
        metavar="EXPR",
        help="a formula of the library expression language (for a formula that starts with -, use --formula=EXPR)",
    )
    g.add_argument("--library", metavar="ID", help="the id of a library definition, for example CF00001")
    g.add_argument("--values", metavar="FILE", type=Path, help="a values file: CSV or Parquet with date, symbol, value")
    p.add_argument(
        "--source", choices=uf.SOURCES, help="the data of a formula (default: okx when OKX has every field it reads)"
    )
    p.add_argument(
        "--spec",
        metavar="FILE",
        help="the book specification: a TOML file, long_short or long_only (default long_short)",
    )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="judge.py",
        description="Test your own factor in a book: a historical test, then a weekly forward test (data/trials/).",
    )
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")
    p = sub.add_parser(
        "check",
        help="run the historical test; write nothing",
        description="Run the historical test of a factor in a book. Write nothing.",
    )
    _add_input(p)
    p.add_argument("--novelty", action="store_true", help="also give the largest correlation with a library definition")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p = sub.add_parser(
        "register",
        help="register a test in the trial ledger",
        description="Register a factor and a book for the forward test.",
    )
    _add_input(p)
    p.add_argument("--name", help="the name of the test: 1 to 64 letters, digits, _, . or -")
    p.add_argument("--note", help="a note for the registration line")
    p.add_argument("--dry-run", action="store_true", help="show the registration line; write nothing")
    p = sub.add_parser(
        "log-values",
        help="log the new Sundays of a values file",
        description="Log the new Sundays of a values-file test.",
    )
    p.add_argument("test_id", metavar="ID", help="the test id, for example FT-0002")
    p.add_argument("--values", metavar="FILE", type=Path, required=True, help="the values file (CSV or Parquet)")
    p = sub.add_parser(
        "log-week",
        help="log the new forward weeks",
        description="Log the formula values and the new complete forward weeks.",
    )
    p.add_argument("--through", metavar="YYYY-MM-DD", type=_day, help="log no week that ends after this date")
    p.add_argument(
        "--accept-release", action="store_true", help="record a new VERSION of Factorlab as a code change, then log"
    )
    p = sub.add_parser(
        "status", help="show the tests (read-only)", description="Show the registered tests and their verdicts."
    )
    p.add_argument("test_id", nargs="?", metavar="ID", help="show only this test")
    p.add_argument("--json", action="store_true", help="print JSON")
    p = sub.add_parser(
        "verify",
        help="check the ledger and compute it again",
        description="Check every line of the trial ledger and compute the logged numbers again.",
    )
    p.add_argument("--write-results", action="store_true", help="write the result files again")
    p = sub.add_parser(
        "withdraw",
        help="withdraw a test before its first forward Sunday",
        description="Withdraw a test before its first forward Sunday.",
    )
    p.add_argument("test_id", metavar="ID", help="the test id")
    p.add_argument("--reason", required=True, help="the reason (required)")
    p = sub.add_parser(
        "record-code-change",
        help="record a change of the test code",
        description="Record a change of the test code in the trial ledger.",
    )
    p.add_argument("--reason", required=True, help="the reason (required)")
    p.add_argument("--note", action="append", default=[], help="a note (give it again for each note)")
    p = sub.add_parser(
        "spec",
        help="validate a book specification",
        description="Validate a book specification and show its normalized content and sha256.",
    )
    p.add_argument("file", nargs="?", metavar="FILE", help="a TOML file, long_short or long_only (default long_short)")
    p.add_argument("--json", action="store_true", help="print JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    try:
        if a.command == "spec":
            print_spec(book.load_spec(a.file), a.json)
            return 0
        if a.command in ("check", "register"):
            spec = book.load_spec(a.spec)
            inp = _input(a)
            if a.command == "check":
                r = trials.check(inp, spec, novelty=a.novelty)
                if a.json:
                    print(json.dumps(trials.public(r), indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False))
                else:
                    print_check(r, a)
                return 0
            return trials.register(inp, spec, name=a.name, note=a.note, dry_run=a.dry_run)
        if a.command == "log-values":
            return trials.log_values(a.test_id, a.values)
        if a.command == "log-week":
            return trials.log_week(through=a.through, accept_release=a.accept_release)
        if a.command == "status":
            d = trials.status_data(a.test_id)
            if a.test_id and not d["tests"]:
                raise trials.refused(f"{a.test_id} is not a registered test.")
            if a.json:
                print(json.dumps(d, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False))
            else:
                print_status(d)
            return 0
        if a.command == "verify":
            return trials.verify(write_results=a.write_results)
        if a.command == "withdraw":
            return trials.withdraw(a.test_id, a.reason)
        return trials.record_code_change(a.reason, a.note)
    except (uf.InputError, book.SpecError) as exc:
        print(f"judge: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
