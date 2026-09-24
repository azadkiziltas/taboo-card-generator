"""Key=value structured logging on top of the standard library."""

from __future__ import annotations

import logging
import sys

_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys() | {"message", "asctime"}
)
_SENSITIVE_EXACT = frozenset(
    {"authorization", "password", "secret", "token", "api_key", "prompt_text"}
)
_SENSITIVE_SUFFIXES = ("_key", "_secret", "_password", "_token")  # not "_tokens" (counts)


def _is_sensitive(key: str) -> bool:
    key = key.lower()
    return key in _SENSITIVE_EXACT or key.endswith(_SENSITIVE_SUFFIXES)


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_") or value is None:
                continue
            if _is_sensitive(key):
                value = "***"
            extras.append(f"{key}={value}")
        return f"{base} {' '.join(extras)}" if extras else base


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        KeyValueFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%S")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # httpx logs full request URLs at INFO; keep it quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
