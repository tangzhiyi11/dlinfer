"""
Piecewise Graph Runner
lmdeploywarmup
"""

import os
import torch
from torch import Tensor
from typing import Any, Callable, Dict, List, Tuple
from lmdeploy.utils import get_logger

from lmdeploy.pytorch.backends.graph_runner import GraphRunner
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
import dlinfer.graph
from dlinfer.graph.ascend_piecewise.piecewise_backend import (
    create_backend,
    get_ascend_compatible_size,
    get_capture_batch_sizes as backend_capture_batch_sizes,
)
from dlinfer.graph.ascend_piecewise.bucket_utils import limit_capture_bucket_list
from dlinfer.graph.ascend_piecewise.bucket_utils import adjust_capture_batch_sizes
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from dlinfer.graph.ascend_piecewise.graph_capture_session import (
    GraphCaptureSession,
    AscendPiecewiseGraphMeta,
    make_buffers_cudagraph,
    fill_buffers_cudagraph,
    update_context_cudagraph,
)

from torch.profiler import record_function

logger = get_logger("dlinfer")


DISABLE_CAPTURE_SESSION = (
    os.environ.get("DLINFER_ASCEND_DISABLE_CAPTURE_SESSION", "0") == "1"
)
USE_CAPTURE_SESSION = not DISABLE_CAPTURE_SESSION
_CAPTURE_SESSION_LOGGED = False


def is_debug_enabled() -> bool:
    """Check if ACL graph debugging is enabled via environment variable."""
    import os

    return os.environ.get("DLINFER_ASCEND_PIECEWISE_GRAPH_DEBUG", "0") == "1"




def update_context_cudagraph(graph_meta, context):
    """update step context with input buffers."""
    input_buffers = graph_meta.input_buffers
    context.block_offsets = input_buffers["block_offsets"]
    context.q_seqlens = input_buffers["q_seqlens"]
    context.kv_seqlens = input_buffers["kv_seqlens"]
    context.q_start_loc = input_buffers["q_start_loc"]
    context.kv_start_indices = input_buffers["kv_start_indices"]


def _false(*args, **kwargs):
    """Default value of not support cuda graph."""
    return False


class AscendPiecewiseSingleGraphRunner:
    """GraphCaptureSession-backed runner."""

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
        """Capture graph."""
        return self.session.capture(**kwargs)

    @record_function("forward_cudagraph")
    def forward(self, **kwargs):
        """forward."""
        return self.session.forward(**kwargs)


class LegacyAscendPiecewiseSingleGraphRunner:
    """Legacy runner retained for debugging."""

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
        self.model = model
        self.ctx_mgr = model.ctx_mgr
        self.model_config = model_config
        self.meta = AscendPiecewiseGraphMeta(
            max_batchs=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            is_decoding=is_decoding,
            device=device,
            head_dim=self.model_config.head_dim,
            num_attention_heads=self.model_config.num_attention_heads,
            dtype=self.model_config.dtype,
            input_buffers=dict(),
            output_buffers=dict(),
            vocab_size=self.model_config.vocab_size,
        )
        self.device = device
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.is_decoding = is_decoding
        self.compiled_model = None
        self.backend = None

    @record_function("capture_cudagraph")
    def capture(self, **kwargs):
        """Capture graph."""
        if is_debug_enabled():
            logger.info("Capturing graph with meta: %s", self.meta)

        num_tokens = kwargs["input_ids"].size(-1)

        if self.backend is None:
            self.backend = create_backend()
            if is_debug_enabled():
                logger.info(
                    "Created new backend for legacy runner (is_decoding=%s)",
                    self.is_decoding,
                )

        self.meta.input_buffers, self.meta.output_buffers = make_buffers_cudagraph(
            self.meta, **kwargs
        )
        padded_kwargs = fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        update_context_cudagraph(self.meta, context)

        import torch._dynamo as dynamo

        cache_limit = dynamo.config.cache_size_limit
        if cache_limit < 1000:
            dynamo.config.cache_size_limit = 1000
            if is_debug_enabled():
                logger.info(
                    "Raised torch._dynamo cache_size_limit %s → %s for legacy capture",
                    cache_limit,
                    dynamo.config.cache_size_limit,
                )

        self.compiled_model = torch.compile(
            self.model,
            backend=self.backend,
            fullgraph=True,
            dynamic=False,
        )

        output = self.compiled_model(**padded_kwargs)

        logits_buffer = self.meta.output_buffers.get("logits")
        if logits_buffer is None or logits_buffer.shape != output.shape:
            logits_buffer = torch.empty_like(output, device=output.device)
            self.meta.output_buffers["logits"] = logits_buffer

        logits_buffer.copy_(output)
        return logits_buffer[:, :num_tokens]

    @record_function("forward_cudagraph")
    def forward(self, **kwargs):
        """forward."""
        num_tokens = kwargs["input_ids"].size(-1)
        assert self.compiled_model is not None
        new_inputs = fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        update_context_cudagraph(self.meta, context)
        compiled_out = self.compiled_model(**new_inputs)

        logits_buffer = self.meta.output_buffers["logits"]

        if compiled_out.data_ptr() != logits_buffer.data_ptr():
            logits_buffer.copy_(compiled_out)

        return logits_buffer[:, :num_tokens]

    def __del__(self):
        """del."""
        if self.compiled_model:
            del self.compiled_model


class RunnerCache:
    """Simple helper to manage cached graph runners."""

    def __init__(self) -> None:
        self._entries: Dict[Any, Any] = {}

    def get_or_create(
        self, key: Any, factory: Callable[[], Any]
    ) -> Tuple[Any, bool]:
        runner = self._entries.get(key)
        created = False
        if runner is None:
            runner = factory()
            self._entries[key] = runner
            created = True
        return runner, created

    def clear(self) -> None:
        self._entries.clear()


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
        self._runner_cache = RunnerCache()
        global _CAPTURE_SESSION_LOGGED
        if not _CAPTURE_SESSION_LOGGED:
            if USE_CAPTURE_SESSION:
                logger.info(
                    "[AscendRunner] GraphCaptureSession path enabled "
                    "(set DLINFER_ASCEND_DISABLE_CAPTURE_SESSION=1 to use legacy runner)"
                )
            else:
                logger.info(
                    "[AscendRunner] DLINFER_ASCEND_DISABLE_CAPTURE_SESSION=1 → "
                    "falling back to legacy runner"
                )
            _CAPTURE_SESSION_LOGGED = True
        self._runner_cls = (
            AscendPiecewiseSingleGraphRunner
            if USE_CAPTURE_SESSION
            else LegacyAscendPiecewiseSingleGraphRunner
        )
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
        max_batches = max_tokens if is_decoding else self.max_batches

        def _factory():
            return self._runner_cls(
                self.model,
                max_batches=max_batches,
                max_tokens=max_tokens,
                num_blocks=self.num_blocks,
                is_decoding=is_decoding,
                pool=self.graph_pool_handle,
                model_config=self.model_config,
                device=self.device,
            )

        runner, created = self._runner_cache.get_or_create(graph_key, _factory)
        if created:
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
        self._runner_cache.clear()

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
