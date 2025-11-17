"""
Ascend piecewise graph runner with streamlined capture management.

This refactor centralizes bucket planning, capture sessions, and cache
handling so we minimize NPUGraph usage and align closer with vLLM streams.
"""

from __future__ import annotations

import torch
from typing import Dict, List, Tuple

from lmdeploy.pytorch.backends.graph_runner import GraphRunner
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from lmdeploy.utils import get_logger

import dlinfer.graph
from .bucket_planner import BucketPlanner
from .graph_capture_session import GraphCaptureSession

logger = get_logger("dlinfer")


def _false(*args, **kwargs):
    """Default value when cuda graph is not supported."""
    return False


def _get_dist_config():
    try:
        from lmdeploy.pytorch.distributed import get_dist_manager

        return get_dist_manager().current_context().dist_config
    except Exception:  # pragma: no cover
        return None


class PiecewiseRuntimeManager:
    """Owns capture sessions, bucket planning, and replay dispatch."""

    def __init__(
        self,
        runner_meta,
        model: torch.nn.Module,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        device: torch.device,
        dist_config=None,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.device = device
        self.ctx_mgr = model.ctx_mgr
        self._runner_meta = runner_meta

        planner = BucketPlanner(
            model_config=model_config,
            cache_config=cache_config,
            dist_config=dist_config,
            logger=logger,
        )
        self._capture_batch_sizes = planner.capture_sizes
        planner.log_summary()

        self.enable_graph = self._check_enable_graph()
        self._sessions: Dict[Tuple[int, bool, bool], GraphCaptureSession] = {}

        dlinfer.graph.config.enable_graph_mode = True
        dlinfer.graph.config.piecewise_graph_enabled = True

    def run(self, **kwargs):
        if not self.enable_graph(**kwargs):
            return self.model(**kwargs)

        graph_key = self._graph_key(**kwargs)
        max_tokens = graph_key[0]
        is_decoding = graph_key[1]
        session = self._sessions.get(graph_key)

        if session is None:
            max_batches = max_tokens if is_decoding else self.cache_config.max_batches
            session = GraphCaptureSession(
                self.model,
                self.model_config,
                max_batches=max_batches,
                max_tokens=max_tokens,
                num_blocks=self.cache_config.num_gpu_blocks,
                is_decoding=is_decoding,
                device=self.device,
            )
            self._sessions[graph_key] = session

            from dlinfer.graph import config

            original_is_capturing = config.is_capturing
            config.is_capturing = True
            try:
                return session.capture(**kwargs)
            finally:
                config.is_capturing = original_is_capturing

        return session.forward(**kwargs)

    def reset(self):
        self._sessions.clear()

    def update_inputs(self, inputs):
        if self.backend_config.eager_mode:
            return inputs
        is_decoding = inputs.is_decoding
        dp_meta = inputs.dp_meta
        if is_decoding and dp_meta is not None:
            meta = self._runner_meta
            padding_batch_size = meta.padding_batch_size
            if padding_batch_size is None:
                return inputs
            tp_size = self._get_capture_tokens(padding_batch_size)
            logger.info(
                "[AscendRunner] sync_tp_size padding_batch_size=%s tp_size=%s "
                "tp_sizes_before=%s moe_tp_sizes_before=%s",
                padding_batch_size,
                tp_size,
                dp_meta.tp_sizes,
                dp_meta.moe_tp_sizes,
            )
            dp_meta.sync_tp_size(tp_size)
            logger.info(
                "[AscendRunner] synced tp_sizes_after=%s moe_tp_sizes_after=%s",
                dp_meta.tp_sizes,
                dp_meta.moe_tp_sizes,
            )
        return inputs

    def get_capture_batch_sizes(self) -> List[int]:
        return list(self._capture_batch_sizes)

    def _check_enable_graph(self):
        if self.backend_config.eager_mode:
            return _false
        return getattr(self.model, "support_cuda_graph", _false)

    def _get_capture_tokens(self, batch_size: int):
        for size in self._capture_batch_sizes:
            if size >= batch_size:
                return size
        raise AssertionError(f"Unsupported batch_size={batch_size}")

    def _graph_key(self, input_ids: torch.Tensor, **kwargs):
        context = self.ctx_mgr.current_context()
        is_decoding = context.is_decoding
        num_tokens = input_ids.numel()
        meta = self._runner_meta
        enable_microbatch = get_step_ctx_manager().current_context().enable_microbatch

        if meta.padding_batch_size is None:
            new_num_tokens = self._get_capture_tokens(num_tokens)
        else:
            new_num_tokens = self._get_capture_tokens(meta.padding_batch_size)

        return (new_num_tokens, is_decoding, enable_microbatch)


class AscendPiecewiseGraphRunner(GraphRunner):
    """Entry point used by lmdeploy to execute piecewise mode."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        device: torch.device,
    ):
        super().__init__(model, model_config, cache_config, backend_config, device)
        dist_config = _get_dist_config()
        self._runtime = PiecewiseRuntimeManager(
            runner_meta=self.get_meta(),
            model=model,
            model_config=model_config,
            cache_config=cache_config,
            backend_config=backend_config,
            device=device,
            dist_config=dist_config,
        )

    def __call__(self, **kwargs):
        return self._runtime.run(**kwargs)

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        return self.model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )

    def reset(self):
        self._runtime.reset()

    def update_inputs(self, inputs):
        return self._runtime.update_inputs(inputs)

    def get_capture_batch_sizes(self) -> List[int]:
        return self._runtime.get_capture_batch_sizes()
