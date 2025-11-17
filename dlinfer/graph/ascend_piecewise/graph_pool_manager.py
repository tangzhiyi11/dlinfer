"""Global graph pool and stream quota management for Ascend piecewise mode."""

from __future__ import annotations

import threading
from typing import Any

import os
import torch
from lmdeploy.utils import get_logger

from .bucket_utils import DEFAULT_MAX_CAPTURE_GRAPHS
from .common import MAX_CAPTURE_GRAPHS_ENV, is_debug_enabled

logger = get_logger("dlinfer.graph_pool")


class GraphPoolManager:
    """Singleton object to share NPUGraph pool handle and enforce quota."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pool: Any = None
        self._active_graphs: int = 0
        self._max_graphs = self._parse_max_graphs()
        logger.info(
            "Ascend graph pool quota initialized: max_graphs=%s", self._max_graphs
        )

    def get_pool(self) -> Any:
        with self._lock:
            if self._pool is None:
                self._pool = self._create_pool_handle()
                logger.info("Created shared graph pool handle for Ascend piecewise mode")
            return self._pool

    def try_acquire(self, count: int = 1) -> bool:
        with self._lock:
            if self._active_graphs + count > self._max_graphs:
                return False
            self._active_graphs += count
            if is_debug_enabled():
                logger.debug(
                    "Graph quota acquired (+%s). Active=%s limit=%s",
                    count,
                    self._active_graphs,
                    self._max_graphs,
                )
            return True

    def release(self, count: int = 1) -> None:
        with self._lock:
            self._active_graphs = max(0, self._active_graphs - count)
            if is_debug_enabled():
                logger.debug(
                    "Graph quota released (-%s). Active=%s limit=%s",
                    count,
                    self._active_graphs,
                    self._max_graphs,
                )

    def _parse_max_graphs(self) -> int:
        env_value = os.environ.get(MAX_CAPTURE_GRAPHS_ENV, "")
        if env_value:
            try:
                parsed = int(env_value)
                if parsed > 0:
                    return parsed
            except ValueError:
                logger.warning(
                    "Invalid %s=%s; using default %s",
                    MAX_CAPTURE_GRAPHS_ENV,
                    env_value,
                    DEFAULT_MAX_CAPTURE_GRAPHS,
                )
        return DEFAULT_MAX_CAPTURE_GRAPHS

    def _create_pool_handle(self):
        npu = getattr(torch, "npu", None)
        if npu is not None and hasattr(npu, "graph_pool_handle"):
            return torch.npu.graph_pool_handle()

        cuda = getattr(torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "graph_pool_handle"):
            return torch.cuda.graph_pool_handle()

        raise RuntimeError("Unable to create graph pool handle for Ascend piecewise mode")


_manager: GraphPoolManager | None = None


def get_graph_pool_manager() -> GraphPoolManager:
    global _manager
    if _manager is None:
        _manager = GraphPoolManager()
    return _manager


__all__ = ["get_graph_pool_manager", "GraphPoolManager"]
