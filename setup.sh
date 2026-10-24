#!/bin/sh
# Install Factorlab, download the market data, build every output and open the dashboard.
#
# Usage:  ./setup.sh [options]
#   --offline        build from the market data on disk; download no market data
#                    (the Python packages still install from the network when absent)
#   --demo           build from synthetic demo data (20 coins, 2022-01 to 2026-09); download no
#                    market data. The dashboard shows "Synthetic demo data. Not market data."
#                    It writes only into an empty or demo data folder (FACTORLAB_DATA works).
#                    To use market data after it: engine/.venv/bin/python -B engine/demo_data.py --remove
#   --skip-binance   do not download or use the Binance panel
#   --skip-library   do not compute the factor library on the data
#   --jobs N         number of Binance downloads at the same time (default 8)
#   --from STEP      start the build at this step (a name from engine/update.py --plan)
#   --test           install the test packages too and run the full test suite after the build
#   --no-open        do not open the dashboard at the end
#
# Run the same command again to add new days. macOS and Linux. Python 3.12 or later.
set -eu
cd "$(dirname "$0")"

TEST=0
OPEN=1
# Keep the options for engine/update.py in "$@" (a step name for --from can hold spaces).
n=$#
i=0
while [ "$i" -lt "$n" ]; do
  a=$1
  shift
  i=$((i + 1))
  case "$a" in
    --test) TEST=1 ;;
    --no-open) OPEN=0 ;;
    --jobs|--from)
      if [ "$i" -ge "$n" ]; then
        echo "setup.sh: $a needs a value" >&2
        exit 2
      fi
      v=$1
      shift
      i=$((i + 1))
      set -- "$@" "$a" "$v" ;;
    --offline|--demo|--skip-binance|--skip-library) set -- "$@" "$a" ;;
    -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p;}' "$0"; exit 0 ;;
    *) echo "setup.sh: unknown option $a (see ./setup.sh --help)" >&2; exit 2 ;;
  esac
done

if [ "$(id -u)" = "0" ]; then
  echo "setup.sh: do not run as root." >&2
  exit 1
fi

# 1. Python 3.12 or later.
if [ -n "${PYTHON:-}" ] && ! { command -v "$PYTHON" >/dev/null 2>&1 && "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; }; then
  echo "setup.sh: PYTHON=$PYTHON is not Python 3.12 or later." >&2
  exit 1
fi
PY=""
for c in "${PYTHON:-}" python3.13 python3.12 python3; do
  [ -n "$c" ] || continue
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
    PY="$c"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "setup.sh: Python 3.12 or later is necessary. Install it, or set PYTHON=/path/to/python3." >&2
  exit 1
fi

# 2. The virtual environment engine/.venv (a real folder at exactly this path).
VENV=engine/.venv
if [ -L "$VENV" ]; then
  echo "setup.sh: $VENV is a symbolic link. Remove it and run setup.sh again." >&2
  exit 1
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "setup.sh makes the Python environment in $VENV ..."
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' || {
  echo "setup.sh: $VENV has a Python older than 3.12. Remove $VENV and run setup.sh again." >&2
  exit 1
}

# 3. The packages (the only network step before the data download). Each requirement file
#    pins each version with the sha256 hashes of its files on PyPI; pip refuses other files.
#    The test packages (pytest, matplotlib, hypothesis) come only with --test.
REQ=engine/requirements.txt
if [ "$TEST" = "1" ]; then
  REQ=engine/requirements-dev.txt
fi
echo "setup.sh installs the Python packages ..."
"$VENV/bin/python" -m pip install --quiet --disable-pip-version-check --no-input --require-hashes -r "$REQ"
# The factor library package and its command engine/.venv/bin/factorlab (editable: the
# package stays in library/; requirements.txt supplies its dependencies and setuptools).
"$VENV/bin/python" -m pip install --quiet --disable-pip-version-check --no-input --no-deps --no-build-isolation -e library

# 4. Data and outputs.
"$VENV/bin/python" -B engine/update.py "$@"

# 5. Tests: a short check always, the full suite with --test.
if [ "$TEST" = "1" ]; then
  "$VENV/bin/python" -B engine/run_tests.py
else
  "$VENV/bin/python" -B engine/run_tests.py --quick
fi

# 6. The dashboard.
echo "Dashboard: web/index.html"
if [ "$OPEN" = "1" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    open web/index.html || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open web/index.html >/dev/null 2>&1 || true
  fi
fi
