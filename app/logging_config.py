"""Logging setup so a run's progress is visible on the console.

The agent nodes log via the stdlib ``logging`` module, but a bare ``python -c``
invocation leaves the root logger at WARNING — so node progress (Scout/Quant/…)
never prints and a long run looks frozen. ``setup_logging`` fixes that and also
aligns loguru (used by token_manager) to the same level to avoid DEBUG spam.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure console logging — idempotently and NON-destructively.

    Crucially this must NOT remove existing root handlers (it used to call
    basicConfig(force=True), which wiped the web UI's LogBuffer whenever a
    workflow re-ran setup_logging — so logs vanished from the UI). We just set the
    level and add our own console handler once.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(lvl)
    if not any(getattr(h, "_ccc_console", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"))
        handler._ccc_console = True  # tag so we never add a second one
        root.addHandler(handler)
    # Keep noisy HTTP libraries + the dev-server access log quiet.
    for noisy in ("httpx", "httpcore", "urllib3", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Align loguru (token_manager) to the same level.
    try:
        from loguru import logger as _loguru
        _loguru.remove()
        _loguru.add(sys.stderr, level=level.upper())
    except Exception:  # noqa: BLE001 — loguru optional
        pass
