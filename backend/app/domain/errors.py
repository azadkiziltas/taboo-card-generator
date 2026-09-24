from __future__ import annotations


class StorageError(Exception):
    """Persistence layer is unavailable or failed (raised by infrastructure adapters)."""

    code = "storage_unavailable"
