"""Shared constants and helpers for Ascend piecewise graph mode."""

from __future__ import annotations

import os
from functools import lru_cache

PIECEWISE_DEBUG_ENV = "DLINFER_ASCEND_PIECEWISE_GRAPH_DEBUG"
CAPTURE_SIZES_ENV = "DLINFER_ASCEND_CAPTURE_SIZES"
GRAPH_CAPTURE_SIZES_ENV = "DLINFER_ASCEND_GRAPH_CAPTURE_SIZES"
MAX_CAPTURE_GRAPHS_ENV = "DLINFER_ASCEND_MAX_CAPTURE_GRAPHS"


@lru_cache(maxsize=1)
def is_debug_enabled() -> bool:
    """Return True when verbose debugging for piecewise graph mode is on."""
    return os.environ.get(PIECEWISE_DEBUG_ENV, "0") == "1"


__all__ = [
    "CAPTURE_SIZES_ENV",
    "GRAPH_CAPTURE_SIZES_ENV",
    "MAX_CAPTURE_GRAPHS_ENV",
    "PIECEWISE_DEBUG_ENV",
    "is_debug_enabled",
]
