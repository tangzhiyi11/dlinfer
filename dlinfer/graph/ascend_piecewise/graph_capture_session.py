"""Graph capture session for Ascend piecewise graph mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.profiler import record_function

from lmdeploy.pytorch.config import ModelConfig
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.utils import get_logger

from .common import is_debug_enabled
from .piecewise_backend import create_backend, get_ascend_compatible_size

logger = get_logger("dlinfer.ascend.capture")

BuffType = Dict[str, Tensor]


@dataclass
class AscendPiecewiseGraphMeta(CudaGraphMeta):
    """Metadata for piecewise graph optimization."""

    max_batchs: int
    max_tokens: int
    num_blocks: int
    is_decoding: int
    device: torch.device
    head_dim: int = 0
    num_attention_heads: int = 0
    dtype: torch.dtype = torch.float16


class AscendPiecewiseAttentionBuffer:
    class_attention_output: Tensor = None
    _cached_shape: Tuple[int, int, torch.dtype, torch.device] = None

    @classmethod
    def get_attention_output(
        cls, batch_size: int, num_attention_heads, head_dim, dtype, device
    ) -> Tensor:
        if cls.class_attention_output is None:
            from lmdeploy.pytorch.distributed import get_tp_world_rank

            tp, tp_rank = get_tp_world_rank("attn")  # need lmdeploy to support dp+tp
            if is_debug_enabled():
                logger.info("get_attention_output: tp=%s, tp_rank=%s", tp, tp_rank)
        heads_per_rank = num_attention_heads // tp
        target_shape = (batch_size, heads_per_rank, head_dim)
        cached_shape = cls._cached_shape
        if (
            cls.class_attention_output is None
            or cached_shape is None
            or cached_shape[0] < heads_per_rank
            or cached_shape[1] != head_dim
            or cached_shape[2] != dtype
            or cached_shape[3] != device
            or cls.class_attention_output.shape[0] < batch_size
        ):
            cls.class_attention_output = torch.empty(
                target_shape,
                dtype=dtype,
                device=device,
            )
            cls._cached_shape = (heads_per_rank, head_dim, dtype, device)
        return cls.class_attention_output[:batch_size, :heads_per_rank]


class GraphCaptureSession:
    """Encapsulates capture/replay buffers and compiled model."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
        *,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        is_decoding: bool,
        device: torch.device,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.ctx_mgr = model.ctx_mgr

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
        self._compiled_model = None
        self._backend = None

    @record_function("ascend_piecewise_capture")
    def capture(self, **kwargs):
        """Compile and capture graph for the first time."""
        if is_debug_enabled():
            logger.info("Capturing graph with meta: %s", self.meta)

        num_tokens = kwargs["input_ids"].size(-1)
        self._ensure_backend()
        self.meta.input_buffers, self.meta.output_buffers = self._make_buffers(**kwargs)

        padded_kwargs = self._fill_buffers(**kwargs)
        context = self.ctx_mgr.current_context()
        self._update_context(context)

        compiled_model = self._compile_model()
        output = compiled_model(**padded_kwargs)
        logits_buffer = self.meta.output_buffers.get("logits")
        if logits_buffer is None or logits_buffer.shape != output.shape:
            logits_buffer = torch.empty_like(output, device=output.device)
            self.meta.output_buffers["logits"] = logits_buffer
        logits_buffer.copy_(output)

        return logits_buffer[:, :num_tokens]

    @record_function("ascend_piecewise_forward")
    def forward(self, **kwargs):
        """Replay captured graph."""
        if self._compiled_model is None:
            return self.capture(**kwargs)

        num_tokens = kwargs["input_ids"].size(-1)
        new_inputs = self._fill_buffers(**kwargs)
        context = self.ctx_mgr.current_context()
        self._update_context(context)
        compiled_out = self._compiled_model(**new_inputs)

        logits_buffer = self.meta.output_buffers["logits"]
        if compiled_out.data_ptr() != logits_buffer.data_ptr():
            logits_buffer.copy_(compiled_out)
        return logits_buffer[:, :num_tokens]

    def _ensure_backend(self) -> None:
        if self._backend is None:
            self._backend = create_backend()
            if is_debug_enabled():
                logger.info(
                    "Created new backend for session (is_decoding=%s)",
                    self.meta.is_decoding,
                )

    def _compile_model(self):
        import torch._dynamo as dynamo

        if self._compiled_model is not None:
            return self._compiled_model

        cache_limit = dynamo.config.cache_size_limit
        if cache_limit < 1000:
            dynamo.config.cache_size_limit = 1000
            if is_debug_enabled():
                logger.info(
                    "Raised torch._dynamo cache_size_limit %s → %s for piecewise capture",
                    cache_limit,
                    dynamo.config.cache_size_limit,
                )

        self._compiled_model = torch.compile(
            self.model,
            backend=self._backend,
            fullgraph=True,
            dynamic=False,
        )
        return self._compiled_model

    # Buffer helpers -----------------------------------------------------
    def _make_buffers(self, **kwargs):
        meta = self.meta
        max_batches = meta.max_batchs
        max_tokens = meta.max_tokens
        num_blocks = meta.num_blocks
        num_attention_heads = meta.num_attention_heads
        head_dim = meta.head_dim
        dtype = meta.dtype
        device = meta.device
        input_buffers: BuffType = dict()
        input_buffers["input_ids"] = torch.empty(
            1, max_tokens, dtype=torch.int32, device=device
        )
        input_buffers["position_ids"] = torch.empty(
            (1, max_tokens), dtype=torch.int32, device=device
        )
        input_buffers["block_offsets"] = torch.zeros(
            (max_batches, num_blocks), dtype=torch.int32, device=device
        )
        input_buffers["q_seqlens"] = torch.zeros(
            max_batches, dtype=torch.int32, device=device
        )
        input_buffers["kv_seqlens"] = torch.zeros(
            max_batches,
            dtype=torch.int32,
            device=device,
        )
        input_buffers["fill_seqlens"] = torch.zeros(
            max_batches, dtype=torch.int32, device=device
        )
        input_buffers["q_start_loc"] = torch.zeros(
            max_batches + 1, dtype=torch.int32, device=device
        )
        input_buffers["kv_start_indices"] = -torch.ones(
            (max_batches), dtype=torch.int64, device=device
        )
        output_buffers: BuffType = dict()
        output_buffers["attention_output"] = (
            AscendPiecewiseAttentionBuffer.get_attention_output(
                max_batches, num_attention_heads, head_dim, dtype, device
            )
        )
        return input_buffers, output_buffers

    def _fill_buffers(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        past_key_values: List,
        attn_metadata: Any,
        inputs_embeds: Tensor = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        block_offsets: Tensor = attn_metadata.block_offsets
        kv_seqlens: List = attn_metadata.kv_seqlens
        kv_start_indices: Tensor = attn_metadata.kv_start_indices

        input_buffers: BuffType = self.meta.input_buffers
        output_buffers: BuffType = self.meta.output_buffers

        batch_size, num_blocks = block_offsets.size()
        num_tokens = input_ids.size(-1)

        input_buffers["input_ids"].zero_()
        input_buffers["input_ids"][:, :num_tokens] = input_ids
        input_buffers["position_ids"].zero_()
        input_buffers["position_ids"][:, :num_tokens] = position_ids
        input_buffers["block_offsets"].zero_()
        input_buffers["block_offsets"][:batch_size, :num_blocks] = block_offsets

        kv_seqlens_tensor = torch.as_tensor(
            kv_seqlens,
            dtype=torch.int32,
        )
        input_buffers["kv_seqlens"].zero_()
        input_buffers["kv_seqlens"][:batch_size] = kv_seqlens_tensor

        kv_start_indices_tensor = torch.as_tensor(
            kv_start_indices,
            dtype=input_buffers["kv_start_indices"].dtype,
            device=input_buffers["kv_start_indices"].device,
        )
        input_buffers["kv_start_indices"].zero_()
        input_buffers["kv_start_indices"][:batch_size] = kv_start_indices_tensor

        if inputs_embeds is not None:
            emb_size = inputs_embeds.size(-1)
            if "inputs_embeds" not in input_buffers:
                max_num_tokens = input_buffers["input_ids"].size(-1)
                input_buffers["inputs_embeds"] = inputs_embeds.new_zeros(
                    1, max_num_tokens, emb_size
                )
            else:
                input_buffers["inputs_embeds"].zero_()
            input_buffers["inputs_embeds"][:, :num_tokens] = inputs_embeds

        if self.meta.is_decoding:
            padded_batch_size = max(self.meta.max_batchs, batch_size)
        else:
            padded_batch_size = get_ascend_compatible_size(batch_size)

        attn_metadata.block_offsets = input_buffers["block_offsets"][:padded_batch_size]
        attn_metadata.kv_seqlens = input_buffers["kv_seqlens"][:padded_batch_size]
        attn_metadata.kv_start_indices = input_buffers["kv_start_indices"][
            :padded_batch_size
        ]
        attn_metadata.attn_output_buffer = output_buffers["attention_output"][
            :padded_batch_size
        ]

        q_seqlens_tensor = getattr(attn_metadata, "q_seqlens", None)
        if q_seqlens_tensor is not None:
            attn_metadata.q_seqlens = self._ensure_tensor_view(
                "q_seqlens",
                q_seqlens_tensor,
                target_first_dim=padded_batch_size,
            )

        q_start_loc_tensor = getattr(attn_metadata, "q_start_loc", None)
        if q_start_loc_tensor is not None:
            pad_dim = (
                padded_batch_size
                if q_start_loc_tensor.dim() == 0
                else max(padded_batch_size + 1, q_start_loc_tensor.shape[0])
            )
            attn_metadata.q_start_loc = self._ensure_tensor_view(
                "q_start_loc",
                q_start_loc_tensor,
                target_first_dim=pad_dim,
            )

        fill_seqlens_tensor = getattr(attn_metadata, "fill_seqlens", None)
        if fill_seqlens_tensor is not None:
            attn_metadata.fill_seqlens = self._ensure_tensor_view(
                "fill_seqlens",
                fill_seqlens_tensor,
                target_first_dim=padded_batch_size,
            )

        new_inputs = dict(
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
        )

        new_inputs["input_ids"] = input_buffers["input_ids"][:, :padded_batch_size]
        new_inputs["position_ids"] = input_buffers["position_ids"][:, :padded_batch_size]

        if inputs_embeds is not None:
            new_inputs["inputs_embeds"] = input_buffers["inputs_embeds"][
                :, :padded_batch_size
            ]

        handled_keys = {
            "input_ids",
            "position_ids",
            "inputs_embeds",
            "attn_metadata",
            "past_key_values",
        }

        extra_kwargs: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in handled_keys:
                continue
            materialized = self._materialize_argument(
                key,
                value,
                pad_dim=padded_batch_size,
            )
            extra_kwargs[key] = materialized

        new_inputs.update(extra_kwargs)
        return new_inputs

    def _root_key_from_path(self, path: str) -> str:
        root = path.split(".", 1)[0]
        root = root.split("[", 1)[0]
        return root

    def _ensure_tensor_view(
        self,
        name: str,
        tensor: torch.Tensor,
        target_first_dim: Optional[int] = None,
    ) -> torch.Tensor:
        meta = self.meta
        total_shape = list(tensor.shape)
        if tensor.dim() > 0:
            desired = target_first_dim if target_first_dim is not None else total_shape[0]
            desired = max(desired, total_shape[0])
            total_shape[0] = desired

        target_shape = tuple(total_shape)
        input_buffers = meta.input_buffers
        buffer = input_buffers.get(name)

        if (
            buffer is None
            or buffer.shape != target_shape
            or buffer.dtype != tensor.dtype
            or buffer.device != tensor.device
        ):
            buffer = torch.empty(target_shape, dtype=tensor.dtype, device=tensor.device)
            input_buffers[name] = buffer

        if tensor.dim() == 0:
            buffer.copy_(tensor)
            return buffer

        active_dim0 = tensor.shape[0]
        slices_active = [slice(0, active_dim0)] + [slice(None)] * (tensor.dim() - 1)
        buffer.zero_()
        buffer[tuple(slices_active)].copy_(tensor)

        first_dim = target_shape[0]
        view_slices = [slice(0, first_dim)] + [slice(0, s) for s in tensor.shape[1:]]
        view = buffer[tuple(view_slices)]

        return view

    def _materialize_argument(
        self,
        key_path: str,
        value: Any,
        pad_dim: Optional[int],
    ) -> Any:
        root_key = self._root_key_from_path(key_path)
        if root_key in {"past_key_values", "attn_metadata"}:
            return value

        if torch.is_tensor(value):
            return self._ensure_tensor_view(
                f"kw:{key_path}",
                value,
                target_first_dim=pad_dim if value.dim() > 0 else None,
            )

        if isinstance(value, Mapping):
            updated = {}
            changed = False
            for sub_key, sub_val in value.items():
                new_val = self._materialize_argument(
                    f"{key_path}.{sub_key}",
                    sub_val,
                    pad_dim,
                )
                updated[sub_key] = new_val
                changed = changed or new_val is not sub_val
            return updated if changed else value

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            new_items = []
            changed = False
            for idx, item in enumerate(value):
                new_item = self._materialize_argument(
                    f"{key_path}[{idx}]",
                    item,
                    pad_dim,
                )
                new_items.append(new_item)
                changed = changed or new_item is not item
            if not changed:
                return value
            return type(value)(new_items)

        return value

    def _update_context(self, context):
        input_buffers = self.meta.input_buffers
        context.block_offsets = input_buffers["block_offsets"]
        context.q_seqlens = input_buffers["q_seqlens"]
        context.kv_seqlens = input_buffers["kv_seqlens"]
        context.q_start_loc = input_buffers["q_start_loc"]
        context.kv_start_indices = input_buffers["kv_start_indices"]


__all__ = ["GraphCaptureSession"]
