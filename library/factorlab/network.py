"""The library's network switch: offline unless ``FACTORLAB_OFFLINE=0``.

``download`` is the only part of the library that uses the internet (Binance
public REST endpoints and the data.binance.vision archive). The engine's
``config.offline_from`` applies the same rule to the same variable, so one
setting controls the whole project:

* unset or ``1``  -> offline: a download stops before it sends a request;
* exactly ``0``   -> network allowed;
* anything else   -> an error. ``""``, ``"true"``, ``"yes"``, ``" 0"`` ... are not
  valid settings, so they never count as "online".

The library does not import the engine: it is a standalone package.

The CLI calls ``require_network()`` before it imports the downloader. The
downloader calls it again before each request, so the Python API has the same gate.
"""

from __future__ import annotations

import os

ENV = "FACTORLAB_OFFLINE"
_VALUES = {"1": True, "0": False}


class NetworkDisabled(RuntimeError):
    """A network request was refused: offline by default, or an invalid setting."""


def offline_from(value: str | None) -> bool:
    """True = offline, False = network allowed. Raises NetworkDisabled on an invalid value."""
    if value is None:
        return True
    if value in _VALUES:
        return _VALUES[value]
    raise NetworkDisabled(
        f"{ENV}={value!r} is not a valid setting: use 1 (or leave it unset) to stay offline, "
        "or exactly 0 to allow network access"
    )


def require_network(what: str = "this command") -> None:
    """Raise NetworkDisabled unless the environment allows network access."""
    if offline_from(os.environ.get(ENV)):
        raise NetworkDisabled(f"{what} needs the network, and Factorlab is offline by default: set {ENV}=0 to allow it")
