"""Central configuration. This is the only file you should need to edit day-to-day.

Design notes:
- Universe is explicit, not "top N by volume today". A dynamic universe chosen with
  today's information leaks survivorship bias into every backtest.
  To change the universe, ADD symbols here and never delete rows from the
  parquet files.
- Timestamps are UTC everywhere, always. Daily bars are stamped with their OPEN
  time (00:00 UTC) per ccxt convention. A bar stamped D closes at D+1
  00:00 UTC — its data is only known AFTER that close.
"""

import os
from pathlib import Path

# --- Venue -------------------------------------------------------------------
# Any ccxt exchange id with perp swaps + funding history works: "okx", "bybit",
# "binanceusdm", "hyperliquid". Public market data needs no API key.
EXCHANGE_ID = "okx"

# Funding-history fallback, DATA ONLY. OKX's public API serves ~3 months of
# funding prints; Binance USDT-M serves years. Prints from here are used only
# where OKX has none (i.e. the deep past) — see data_io.daily_funding. This is
# never a trading venue; it is a free public archive of a rate that closely
# tracks OKX's, used so carry can be backtested beyond a 3-month window.
FUNDING_FALLBACK_EXCHANGE_ID = "binanceusdm"

# --- Universe ----------------------------------------------------------------
# OKX unified symbols for USDT-margined perpetual swaps (ccxt format).
# ~20 liquid names, mostly listed before 2022. APT/OP/ARB listed after — their
# factor rows simply start 91 days after their first bar, which is fine (the
# cross-section just has fewer names early on). Extend as you like.
UNIVERSE = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "DOT/USDT:USDT",
    "LTC/USDT:USDT",
    "BCH/USDT:USDT",
    "UNI/USDT:USDT",
    "ATOM/USDT:USDT",
    "ETC/USDT:USDT",
    "FIL/USDT:USDT",
    "NEAR/USDT:USDT",
    "APT/USDT:USDT",
    "OP/USDT:USDT",
    "ARB/USDT:USDT",
    "AAVE/USDT:USDT",
]

# --- History -----------------------------------------------------------------
BACKFILL_START = "2022-01-01T00:00:00Z"  # first ingest reaches back to here

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
# Data root. Override with FACTORLAB_DATA to use another folder (for example a copy of
# the data, or a temporary folder of a test) without a change to engine/data.
DATA = Path(os.environ.get("FACTORLAB_DATA", ROOT / "data")).expanduser().resolve()

# The log and the run lock of update.py are in <repo>/engine/logs (git ignores it);
# override with FACTORLAB_LOGS. update.py makes the folder when it runs, never at
# import time.
LOGS = Path(os.environ.get("FACTORLAB_LOGS", ROOT / "logs")).expanduser().resolve()


# Two rules keep this checkout apart from another copy of this engine:
#  - LOGS must stay inside this repo (ROOT.parent). The update lock is there, so a
#    shared LOGS would let one copy stop the update of the other.
#  - DATA may be a folder anywhere, but never inside ANOTHER checkout of this
#    engine, that is a folder other than ROOT that holds its own update.py and
#    config.py.
# Directories are compared with samefile(), so symlinks and a differently-cased
# spelling on a case-insensitive volume (APFS default) cannot slip past.
def _same(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except OSError:  # `a` does not exist
        return False


def inside(path: Path, root: Path) -> bool:
    """True if `path` is `root` itself or lies anywhere under it."""
    return any(_same(p, root) for p in (path, *path.parents))


def foreign_checkout(path: Path) -> Path | None:
    """Root of another engine checkout that contains `path`, else None."""
    for p in (path, *path.parents):
        if (p / "update.py").is_file() and (p / "config.py").is_file() and not _same(p, ROOT):
            return p
    return None


if not inside(LOGS, ROOT.parent):
    raise SystemExit(f"refusing log dir outside this repo: {LOGS} (set FACTORLAB_LOGS to a path inside this repo)")
if (_other := foreign_checkout(DATA)) is not None:
    raise SystemExit(
        f"refusing data dir inside another engine checkout ({_other}): "
        f"{DATA} (set FACTORLAB_DATA to this repo's data or a scratch copy)"
    )
RAW_OHLCV = DATA / "raw" / "ohlcv"  # one parquet per symbol
RAW_SPOT_OHLCV = DATA / "raw" / "spot_ohlcv"  # same-venue spot reference
RAW_FUNDING = DATA / "raw" / "funding"  # one parquet per symbol
RAW_FUNDING_FB = DATA / "raw" / f"funding_{FUNDING_FALLBACK_EXCHANGE_ID}"
RAW_OI = DATA / "raw" / "open_interest"  # one parquet per symbol
LABELS = DATA / "labels"  # labels.parquet (labels.py): fwd_ret_1 and fwd_ret_7; made on demand
TRIALS = DATA / "trials"  # the trial ledger of your factor tests (judge.py, trials.py); made on demand
RESEARCH = DATA / "research"  # the factor-lab registry and the AlphaForge files; made on demand

for p in (RAW_OHLCV, RAW_SPOT_OHLCV, RAW_FUNDING, RAW_FUNDING_FB, RAW_OI):
    p.mkdir(parents=True, exist_ok=True)


# --- Write protection (guard_write) -------------------------------------------
# A data folder is protected when the marker file <data folder>/.read_only exists.
# Every engine writer calls guard_write(target) before it writes. The rule:
#  - it applies to DATA (also a FACTORLAB_DATA override) and to engine/data
#    (SHIPPED_DATA), each only while it holds the marker file;
#  - it covers PROTECTED_DIRS only. The other folders (for example
#    data/library_on_engine) stay writable;
#  - FACTORLAB_WRITE_PROTECTED=1 allows the write. Unset or 0 refuses. Any other
#    value stops the program with an error.
# Without the marker file every writer works.
PROTECTED_DIRS = ("raw", "labels", "research", "trials")
PROTECT_MARKER = ".read_only"
PROTECT_OVERRIDE_ENV = "FACTORLAB_WRITE_PROTECTED"
SHIPPED_DATA = ROOT / "data"


class WriteRefused(SystemExit):
    """A writer tried to change a file in a protected folder.

    A SystemExit (exit status 1 with the message, no traceback). A refused CLI run
    stops cleanly, and a broad `except Exception` does not catch it.
    """

    def __init__(self, message: str):
        super().__init__(message)


def protected_roots() -> list[Path]:
    """The data folders that hold the marker file (DATA first, then SHIPPED_DATA)."""
    roots: list[Path] = []
    for root in (Path(DATA), Path(SHIPPED_DATA)):
        if (root / PROTECT_MARKER).is_file() and not any(_same(root, r) for r in roots):
            roots.append(root)
    return roots


def protected_folder(path) -> tuple[Path, str] | None:
    """(data folder, protected folder name) that `path` lies in, else None."""
    p = Path(path).expanduser().absolute()
    real = p.resolve()  # follows symlinks as far as the path exists
    for data_root in protected_roots():
        for name in PROTECTED_DIRS:
            root = data_root / name
            # samefile() finds symlinks and other spellings on a case-insensitive volume.
            # The resolved-prefix test also finds a folder that does not exist.
            if inside(p, root) or real == root.resolve() or root.resolve() in real.parents:
                return data_root, name
    return None


def guard_write(path) -> None:
    """Refuse to write `path` when it lies in a protected folder.

    Call it on the final target (file or folder) before the first byte is written.
    """
    hit = protected_folder(path)
    if hit is None:
        return
    value = os.environ.get("FACTORLAB_WRITE_PROTECTED", "0")  # = PROTECT_OVERRIDE_ENV
    if value == "1":
        return
    if value != "0":
        raise SystemExit(
            f"{PROTECT_OVERRIDE_ENV}={value!r} is not a valid setting: leave it unset "
            "(or 0) to keep the protection, or set exactly 1 to write"
        )
    data_root, folder = hit
    p = Path(path).expanduser().absolute()
    try:
        shown = "data/" + p.relative_to(data_root.absolute()).as_posix()
    except ValueError:  # a symlink or another spelling of the path
        shown = f"a file in data/{folder}/"
    raise WriteRefused(
        f"refusing to write {shown}: the data folder is protected (marker file "
        f"data/{PROTECT_MARKER}). Use a copy without the marker file "
        f"(FACTORLAB_DATA=<copy>), or set {PROTECT_OVERRIDE_ENV}=1 to write."
    )


# --- Network switch (FACTORLAB_OFFLINE) -----------------------------------------
# ingest.py reads FACTORLAB_OFFLINE (default "1" when unset) and gives the raw
# string to offline_from(). The rule is strict: values such as "true", "yes" or ""
# look like "offline" but are not. Only the exact string "0" turns exchange access
# on, "1" (or unset) keeps it off, and any other value stops the program with an
# error.
OFFLINE_VALUES = {"1": True, "0": False}


def offline_from(value: str | None) -> bool:
    """Parse a FACTORLAB_OFFLINE value: True = offline, False = exchange access.

    None (unset) and "1" are offline; exactly "0" is online. Every other value
    ("", "true", "yes", "false", " 0", "00", ...) raises SystemExit with a
    message naming the accepted values, so a typo can never reach the network.
    """
    if value is None:
        return True
    if value in OFFLINE_VALUES:
        return OFFLINE_VALUES[value]
    raise SystemExit(
        f"FACTORLAB_OFFLINE={value!r} is not a valid setting: use 1 (or leave it "
        "unset) to stay offline, or exactly 0 to allow exchange access"
    )


def sym_to_file(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTC-USDT' (filesystem-safe, exchange-agnostic)."""
    return symbol.split("/")[0] + "-" + symbol.split("/")[1].split(":")[0]


def perp_to_spot(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTC/USDT' for same-venue basis measurement."""
    return symbol.split(":")[0]


# --- AlphaForge expression miner (alphaforge/, research-only) -------------------
# Master switch. False (the default): alphaforge_bridge.py refuses --cycle, --mine and
# --refresh (exit 3), and update.py has no AlphaForge step. Set True to let update.py run
# `alphaforge_bridge.py --cycle` on each update. See README_ALPHAFORGE.md.
ALPHAFORGE_ENABLED = False
# Miner knobs read by alphaforge_bridge.settings(); anything omitted keeps the
# defaults documented there (horizon 7 = fwd_ret_7, quintiles, seed 42).
ALPHAFORGE = {
    "population": 80,
    "generations": 10,
    "max_depth": 3,
    "max_factors": 6,  # at most 6 active formulas
    "min_valid_icir_ann": 0.5,
    "mine_every_days": 7,  # at most one mining run in this many days, and only with a free place
    "forward_window_weeks": 52,  # each formula's fixed forward window; then it leaves the active list
}
