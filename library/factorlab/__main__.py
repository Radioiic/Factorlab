"""Run the command-line tool: ``python -m factorlab <command>`` (see ``factorlab.cli``)."""

import sys

from .cli import main

sys.exit(main())
