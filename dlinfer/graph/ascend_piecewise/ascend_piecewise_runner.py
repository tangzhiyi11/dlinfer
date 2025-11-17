"""
Piecewise Graph Runner
lmdeploywarmup
"""

import os
import torch
from torch import Tensor
from typing import Any, Dict, List
from lmdeploy.utils import get_logger

from lmdeploy.pytorch.backends.graph_runner import GraphRunner
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
import dlinfer.graph
from dlinfer.graph.ascend_piecewise.piecewise_backend import (
    get_capture_batch_sizes as backend_capture_batch_sizes,
)
from dlinfer.graph.ascend_piecewise.bucket_utils import limit_capture_bucket_list
from dlinfer.graph.ascend_piecewise.bucket_utils import adjust_capture_batch_sizes
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from dlinfer.graph.ascend_piecewise.graph_capture_session import GraphCaptureSession

from torch.profiler import record_function

logger = get_logger("dlinfer")


def is_debug_enabled() -> bool:
    """Check if ACL graph debugging is enabled via environment variable."""
    import os

    return os.environ.get("DLINFER_ASCEND_PIECEWISE_GRAPH_DEBUG", "0") == "1"


def _false(*args, **kwargs):
    """Default value of not support cuda graph."""
    return False


class AscendPiecewiseSingleGraphRunner:
    """Thin wrapper around GraphCaptureSession."""

    def __init__(
        self,
        model: torch.nn.Module,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        is_decoding: bool,
        pool: Any,
        model_config: ModelConfig,
        device: torch.device,
    ):
        del pool  # graph pool handle unused on Ascend but kept for API compatibility
        self.session = GraphCaptureSession(
            model=model,
            model_config=model_config,
            max_batches=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            is_decoding=is_decoding,
            device=device,
        )

    @record_function("capture_cudagraph")
    def capture(self, **kwargs):
        """Capture and compile the graph for this bucket."""
        return self.session.capture(**kwargs)

    @record_function("forward_cudagraph")
    def forward(self, **kwargs):
        """Replay the captured graph with new inputs."""
        return self.session.forward(**kwargs)


class AscendPiecewiseGraphRunner(GraphRunner):
    """Cuda graph runner."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        backend_config: BackendConfig,
        device: torch.device,
    ):
        super().__init__(model, model_config, cache_config, backend_config, device)
        self.max_batches = cache_config.max_batches
        self.max_tokens = cache_config.max_prefill_token_num
        self.num_blocks = cache_config.num_gpu_blocks
        self.enable_graph = self.check_enable_graph()
        self.graph_pool_handle = torch.cuda.graph_pool_handle()
        self._runner_map: Dict[Any, AscendPiecewiseSingleGraphRunner] = dict()
        self.has_try_compile_model: bool = False

        override_env = os.getenv("DLINFER_ASCEND_CAPTURE_SIZES")
        override_sizes = None

        if override_env:
            try:
                parsed = sorted(
                    {int(item.strip()) for item in override_env.split(",") if item.strip()}
                )
                override_sizes = [size for size in parsed if size > 0]
                if not override_sizes:
                    logger.warning(
                        "DLINFER_ASCEND_CAPTURE_SIZES provided but no valid positive integers found: %s",
                        override_env,
                    )
                    override_sizes = None
            except ValueError:
                logger.warning(
                    "Failed to parse DLINFER_ASCEND_CAPTURE_SIZES=%s; ignoring user override",
                    override_env,
                )
                override_sizes = None

        try:
            from lmdeploy.pytorch.distributed import get_dist_manager

            dist_config = get_dist_manager().current_context().dist_config
        except Exception:  # pragma: no cover - dist context may not be initialized
            dist_config = None

        default_capture_sizes = backend_capture_batch_sizes(self.cache_config.max_batches)
        base_sizes = override_sizes or default_capture_sizes
        if override_sizes:
            filtered = [size for size in base_sizes if size in default_capture_sizes]
            if not filtered:
                logger.warning(
                    "DLINFER_ASCEND_CAPTURE_SIZES=%s has no overlap with default buckets %s; falling back",
                    override_env,
                    default_capture_sizes,
                )
                base_sizes = default_capture_sizes
            else:
                base_sizes = filtered
                logger.info("Using user-specified capture buckets before adjustment: %s", base_sizes)

        self._capture_batch_sizes = adjust_capture_batch_sizes(
            base_sizes,
            model_config=self.model_config,
            cache_config=self.cache_config,
            dist_config=dist_config,
            logger_=logger,
        )

        dlinfer.graph.config.enable_graph_mode = True
        dlinfer.graph.config.piecewise_graph_enabled = True

        max_capture_env = os.getenv("DLINFER_ASCEND_MAX_CAPTURE_GRAPHS")
        max_capture_graphs = None
        if max_capture_env:
            try:
                max_capture_graphs = int(max_capture_env)
            except ValueError:
                logger.warning(
                    "Invalid DLINFER_ASCEND_MAX_CAPTURE_GRAPHS=%s, ignoring.", max_capture_env
                )

        # Only apply trimming if custom capture sizes are not provided via environment variable
        if not os.getenv("DLINFER_ASCEND_GRAPH_CAPTURE_SIZES"):
            limited_sizes = limit_capture_bucket_list(
                self._capture_batch_sizes,
                model_config=self.model_config,
                dist_config=dist_config,
                max_capture_graphs=max_capture_graphs,
            )
            if not limited_sizes:
                logger.warning(
                    "Ascend capture bucket limiter returned empty list; falling back to defaults."
                )
                limited_sizes = self._capture_batch_sizes
            self._capture_batch_sizes = limited_sizes

    def check_enable_graph(self):
        """Check enable graph."""
        if self.backend_config.eager_mode:
            return _false

        return getattr(self.model, "support_cuda_graph", _false)

    def _get_capture_tokens(self, batch_size: int):
        """Get capture tokens."""
        cap_sizes = self.get_capture_batch_sizes()
        for size in cap_sizes:
            if size >= batch_size:
                return size
        assert False, f"Unsupported batch_size={batch_size}"

    def get_graph_key(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        """Get graph key."""
        context = self.ctx_mgr.current_context()
        is_decoding = context.is_decoding
        num_tokens = input_ids.numel()
        meta = self.get_meta()
        enable_microbatch = get_step_ctx_manager().current_context().enable_microbatch
        if meta.padding_batch_size is None:
            new_num_tokens = self._get_capture_tokens(num_tokens)
        else:
            new_num_tokens = self._get_capture_tokens(meta.padding_batch_size)
        return (new_num_tokens, is_decoding, enable_microbatch)

    def __call__(self, **kwargs):
        """call."""
        if not self.enable_graph(**kwargs):
            with record_function("forward_eager"):
                return self.model(**kwargs)

        graph_key = self.get_graph_key(**kwargs)
        max_tokens = graph_key[0]
        is_decoding = graph_key[1]
        runner = self._runner_map.get(graph_key)
        if runner is None:
            max_batches = max_tokens if is_decoding else self.max_batches
            runner = AscendPiecewiseSingleGraphRunner(
                self.model,
                max_batches=max_batches,
                max_tokens=max_tokens,
                num_blocks=self.num_blocks,
                is_decoding=is_decoding,
                pool=self.graph_pool_handle,
                model_config=self.model_config,
                device=self.device,
            )
            self._runner_map[graph_key] = runner

            from dlinfer.graph import config

            original_is_capturing = config.is_capturing
            config.is_capturing = True

            try:
                return runner.capture(**kwargs)
            finally:
                config.is_capturing = original_is_capturing

        return runner.forward(**kwargs)

    @record_function("prepare_inputs_for_generation")
    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare inputs."""
        return self.model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )

    def reset(self):
        """Remove all graphs to prevent hanging on exit."""
        self._runner_map.clear()

    def update_inputs(self, inputs):
        """Update inputs."""
        if self.backend_config.eager_mode:
            return inputs
        is_decoding = inputs.is_decoding
        dp_meta = inputs.dp_meta
        if is_decoding and dp_meta is not None:
            meta = self.get_meta()
            padding_batch_size = meta.padding_batch_size
            tp_size = self._get_capture_tokens(padding_batch_size)
            # Sync both tp_sizes and moe_tp_sizes so downstream MoE ops see the padded token count.
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
        """Capture batch sizes."""
        return self._capture_batch_sizes
