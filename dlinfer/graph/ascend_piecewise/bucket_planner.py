"""Capture bucket planner for Ascend piecewise graph mode."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

from lmdeploy.pytorch.config import CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from .bucket_utils import (
    adjust_capture_batch_sizes,
    limit_capture_bucket_list,
    DEFAULT_MAX_CAPTURE_GRAPHS,
)
from .common import (
    CAPTURE_SIZES_ENV,
    GRAPH_CAPTURE_SIZES_ENV,
    MAX_CAPTURE_GRAPHS_ENV,
    is_debug_enabled,
)
from .piecewise_backend import get_capture_batch_sizes as backend_capture_batch_sizes


@dataclass
class BucketPlannerResult:
    capture_sizes: List[int]
    override_used: bool
    limited: bool


class BucketPlanner:
    """Centralize capture bucket selection, env overrides, and stream limits."""

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        dist_config=None,
        *,
        logger=None,
    ):
        self.logger = logger or get_logger("dlinfer.ascend.bucket_planner")
        self.model_config = model_config
        self.cache_config = cache_config
        self.dist_config = dist_config

        self._result = self._plan_capture_sizes()

    @property
    def capture_sizes(self) -> List[int]:
        return list(self._result.capture_sizes)

    def log_summary(self) -> None:
        self.logger.info(
            "Ascend bucket planner result: sizes=%s override=%s limited=%s",
            self._result.capture_sizes,
            self._result.override_used,
            self._result.limited,
        )

    def _plan_capture_sizes(self) -> BucketPlannerResult:
        default_sizes = backend_capture_batch_sizes(self.cache_config.max_batches)
        override_sizes = self._parse_primary_override(default_sizes)

        base_sizes = override_sizes or default_sizes
        adjusted_sizes = adjust_capture_batch_sizes(
            base_sizes,
            model_config=self.model_config,
            cache_config=self.cache_config,
            dist_config=self.dist_config,
            logger_=self.logger,
        )

        limited_sizes = self._apply_stream_limit(adjusted_sizes)
        return BucketPlannerResult(
            capture_sizes=limited_sizes,
            override_used=override_sizes is not None,
            limited=limited_sizes != adjusted_sizes,
        )

    def _parse_primary_override(
        self, default_sizes: Sequence[int]
    ) -> Optional[List[int]]:
        env_value = os.getenv(CAPTURE_SIZES_ENV)
        if not env_value:
            return None

        try:
            parsed = sorted(
                {int(item.strip()) for item in env_value.split(",") if item.strip()}
            )
        except ValueError:
            self.logger.warning(
                "Failed to parse %s=%s; ignoring user override",
                CAPTURE_SIZES_ENV,
                env_value,
            )
            return None

        override_sizes = [size for size in parsed if size > 0]
        if not override_sizes:
            self.logger.warning(
                "%s provided but no valid positive integers found: %s",
                CAPTURE_SIZES_ENV,
                env_value,
            )
            return None

        filtered = [size for size in override_sizes if size in default_sizes]
        if not filtered:
            self.logger.warning(
                "%s=%s has no overlap with default buckets %s; falling back",
                CAPTURE_SIZES_ENV,
                env_value,
                default_sizes,
            )
            return None

        self.logger.info(
            "Using user-specified capture buckets before adjustment: %s", filtered
        )
        return filtered

    def _apply_stream_limit(self, adjusted_sizes: Sequence[int]) -> List[int]:
        """Apply stream-based graph count limit unless user forces capture sizes."""
        if os.getenv(GRAPH_CAPTURE_SIZES_ENV):
            # User forced capture sizes; skip limiter to mirror vLLM behavior.
            if is_debug_enabled():
                self.logger.debug(
                    "Skipping bucket limiter because %s is set", GRAPH_CAPTURE_SIZES_ENV
                )
            return list(adjusted_sizes)

        max_graphs = self._parse_max_capture_env()

        limited = limit_capture_bucket_list(
            adjusted_sizes,
            model_config=self.model_config,
            dist_config=self.dist_config,
            max_capture_graphs=max_graphs,
        )
        if not limited:
            self.logger.warning(
                "Bucket limiter returned empty list (input=%s); falling back to adjusted sizes",
                adjusted_sizes,
            )
            return list(adjusted_sizes)
        return limited

    def _parse_max_capture_env(self) -> int:
        env_value = os.getenv(MAX_CAPTURE_GRAPHS_ENV)
        if not env_value:
            return DEFAULT_MAX_CAPTURE_GRAPHS
        try:
            parsed = int(env_value)
            if parsed <= 0:
                raise ValueError
            return parsed
        except ValueError:
            self.logger.warning(
                "Invalid %s=%s, using default %s",
                MAX_CAPTURE_GRAPHS_ENV,
                env_value,
                DEFAULT_MAX_CAPTURE_GRAPHS,
            )
            return DEFAULT_MAX_CAPTURE_GRAPHS


__all__ = ["BucketPlanner"]
