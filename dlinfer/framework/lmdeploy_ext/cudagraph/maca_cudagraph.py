# Copyright (c) 2024, OpenMMLab and DeepLink. All rights reserved.
import torch
from torch import Tensor

from typing import Any, Dict, List

from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMixin, next_power_of_2

BuffType = Dict[str, Tensor]

def flashinfer_available():
    """Check if flashinfer is available."""
    # use flashinfer by default if it is installed
    # return False
    use_flashinfer = False
    try:
        import flashinfer  # noqa
        use_flashinfer = True
    except ImportError:
        logger.warning('For higher performance, please install flashinfer https://github.com/flashinfer-ai/flashinfer')
    return use_flashinfer


class MacaCudaGraphFlashInferMeta:
    is_mla = None
    zero_tensor = None
    q_max_arange_tensor = None
    q_max_arange_size = 64
    mask_max_arange_tensor = None
    mask_max_arange_size = 4100
    flashinfer_decode_wrapper = None
    _lock = None

    @classmethod
    def _get_lock(cls):
        """Get thread lock for thread-safe operations."""
        if cls._lock is None:
            import threading
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def _get_zero_tensor(cls, device):
        if cls.zero_tensor is None or cls.zero_tensor.device != device:
            with cls._get_lock():
                # Double-check pattern for thread safety
                if cls.zero_tensor is None or cls.zero_tensor.device != device:
                    cls.zero_tensor = torch.tensor([0], device=device)
        return cls.zero_tensor

    @classmethod
    def _get_q_indptr_arrange(cls, device, size):
        if cls.q_max_arange_tensor is None or size > cls.q_max_arange_size:
            with cls._get_lock():
                # Double-check pattern for thread safety
                if cls.q_max_arange_tensor is None or size > cls.q_max_arange_size:
                    if size > cls.q_max_arange_size:
                        cls.q_max_arange_size = size * 2
                    cls.q_max_arange_tensor = torch.arange(0, cls.q_max_arange_size, dtype=torch.int32, device=device)
        return cls.q_max_arange_tensor[:size]

    @classmethod
    def _get_mask_arrange(cls, device, size):
        if cls.mask_max_arange_tensor is None or size > cls.mask_max_arange_size:
            with cls._get_lock():
                # Double-check pattern for thread safety
                if cls.mask_max_arange_tensor is None or size > cls.mask_max_arange_size:
                    if size > cls.mask_max_arange_size:
                        cls.mask_max_arange_size = size * 2
                    cls.mask_max_arange_tensor = torch.arange(0, cls.mask_max_arange_size, dtype=torch.int32, device=device)
        return cls.mask_max_arange_tensor[:size]

    @classmethod
    def _get_flashinfer_decode_wrapper(cls, max_batches):
        """Get flashinfer decode wrapper. TODO: Implement this method."""
        if cls.flashinfer_decode_wrapper is None:
            raise RuntimeError("flashinfer_decode_wrapper is None!")
        if max_batches not in cls.flashinfer_decode_wrapper:
            raise RuntimeError(f"flashinfer_decode_wrapper[{max_batches}] is None!")
        return cls.flashinfer_decode_wrapper[max_batches]

    @classmethod
    def _set_flashinfer_decode_wrapper(cls, max_batches, flashinfer_decode_wrapper):
        """Set flashinfer decode wrapper. TODO: Implement this method."""
        if cls.flashinfer_decode_wrapper is None:
            with cls._get_lock():
                cls.flashinfer_decode_wrapper = {}
        cls.flashinfer_decode_wrapper[max_batches] = flashinfer_decode_wrapper


    @classmethod
    def _get_sm_scale(cls):
        """Get sm scale."""
        return 0.114721386792


def MacaCudaGraphMixin_make_buffers_cudagraph(
    self, graph_meta: CudaGraphMeta, *args, **kwargs
) -> BuffType:
    """make cudagraph buffers from forward inputs."""
    max_batches = graph_meta.max_batchs
    max_tokens = graph_meta.max_tokens
    num_blocks = graph_meta.num_blocks
    device = graph_meta.device
    input_buffers: BuffType = dict()
    input_buffers["input_ids"] = torch.zeros(
        1, max_tokens, dtype=torch.int32, device=device
    )

    input_buffers["position_ids"] = torch.zeros(
        (1, max_tokens), dtype=torch.int32, device=device
    )

    input_buffers["block_offsets"] = torch.zeros(
        (max_batches, num_blocks), dtype=torch.int32, device=device
    )

    input_buffers["q_seqlens"] = torch.ones(
        max_batches, dtype=torch.int32, device=device
    )

    input_buffers["kv_seqlens"] = torch.zeros(
        max_batches, dtype=torch.int32, device=device
    )

    input_buffers["q_start_loc"] = torch.arange(
        max_batches + 1, dtype=torch.int32, device=device
    )

    input_buffers["kv_start_indices"] = -torch.ones(
        (max_batches, 1), dtype=torch.int64, device=device
    )

    if flashinfer_available() and (MacaCudaGraphFlashInferMeta.is_mla is None or MacaCudaGraphFlashInferMeta.is_mla):
        import flashinfer
        qo_indptr = torch.arange(0, max_batches + 1, dtype=torch.int32, device=device)
        kv_indptr = torch.zeros((max_batches + 1,), dtype=torch.int32, device=device)
        kv_indices = torch.zeros((max_batches * num_blocks, ), dtype=torch.int32, device=device)
        kv_len_arr = torch.ones((max_batches,), dtype=torch.int32, device=device)
        flashinfer_decode_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0),
            use_cuda_graph=True,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
            backend="auto",
        )
        MacaCudaGraphFlashInferMeta._set_flashinfer_decode_wrapper(max_batches, flashinfer_decode_wrapper)

    return input_buffers


def MacaCudaGraphMixin_fill_buffers_cudagraph(
    self,
    graph_meta: CudaGraphMeta,
    input_ids: Tensor,
    position_ids: Tensor,
    past_key_values: List,
    attn_metadata: Any,
    inputs_embeds: Tensor,
    **kwargs
) -> Dict[str, Tensor]:
    """fill cudagraph buffers from forward inputs."""
    block_offsets: Tensor = attn_metadata.block_offsets
    q_start_loc: Tensor = attn_metadata.q_start_loc
    q_seqlens: Tensor = attn_metadata.q_seqlens
    kv_seqlens: Tensor = attn_metadata.kv_seqlens
    kv_start_indices: Tensor = attn_metadata.kv_start_indices

    input_buffers: BuffType = graph_meta.input_buffers

    batch_size, num_blocks = block_offsets.size()
    num_tokens = input_ids.size(-1)

    # fill buffer
    input_buffers["input_ids"][:, :num_tokens] = input_ids
    input_buffers["position_ids"][:, :num_tokens] = position_ids
    input_buffers["block_offsets"][:batch_size, :num_blocks] = block_offsets
    input_buffers["q_start_loc"][: batch_size + 1] = q_start_loc
    input_buffers["q_seqlens"][:batch_size] = q_seqlens
    input_buffers["kv_seqlens"][:batch_size] = kv_seqlens
    input_buffers["kv_start_indices"][:batch_size] = kv_start_indices

    if inputs_embeds is not None:
        emb_size = inputs_embeds.size(-1)
        if "inputs_embeds" not in input_buffers:
            max_num_tokens = input_buffers["input_ids"].size(-1)
            input_buffers["inputs_embeds"] = inputs_embeds.new_zeros(
                1, max_num_tokens, emb_size
            )
        input_buffers["inputs_embeds"][:, :num_tokens] = inputs_embeds
    # create inputs
    new_batch_size = next_power_of_2(batch_size)

    attn_metadata.block_offsets = input_buffers["block_offsets"][:new_batch_size]
    attn_metadata.q_start_loc = input_buffers["q_start_loc"][: new_batch_size + 1]
    attn_metadata.q_seqlens = input_buffers["q_seqlens"][:new_batch_size]
    attn_metadata.kv_seqlens = input_buffers["kv_seqlens"][:new_batch_size]
    attn_metadata.kv_start_indices = input_buffers["kv_start_indices"][:new_batch_size]

    if attn_metadata.use_flashinfer:
        device = attn_metadata.block_offsets.device
        max_batches = graph_meta.max_batchs
        num_blocks = graph_meta.num_blocks
        attn_metadata.flashinfer_wrapper = MacaCudaGraphFlashInferMeta._get_flashinfer_decode_wrapper(graph_meta.max_batchs)
        MacaCudaGraphFlashInferMeta.is_mla = True

    new_inputs = dict(
        past_key_values=past_key_values,
        attn_metadata=attn_metadata,
    )

    new_inputs["input_ids"] = input_buffers["input_ids"][:, :new_batch_size]
    new_inputs["position_ids"] = input_buffers["position_ids"][:, :new_batch_size]

    if inputs_embeds is not None:
        new_inputs["inputs_embeds"] = input_buffers["inputs_embeds"][:, :new_batch_size]

    new_inputs.update(kwargs)

    return new_inputs


def MacaCudaGraphMixin_update_context_cudagraph(self, graph_meta, context):
    """update step context with input buffers."""
    input_buffers = graph_meta.input_buffers
    context.block_offsets = input_buffers["block_offsets"]
    context.q_seqlens = input_buffers["q_seqlens"]
    context.kv_seqlens = input_buffers["kv_seqlens"]
    context.q_start_loc = input_buffers["q_start_loc"]
    context.kv_start_indices = input_buffers["kv_start_indices"]

    attn_metadata = context.attn_metadata

    if attn_metadata.use_flashinfer:
        q_seqlens = attn_metadata.q_seqlens
        kv_seqlens = attn_metadata.kv_seqlens
        q_start_loc = attn_metadata.q_start_loc
        kv_start_indices = attn_metadata.kv_start_indices
        block_offsets = attn_metadata.block_offsets

        device = attn_metadata.block_offsets.device
        block_size = context.kv_caches[0][1].size(-2)

        q_indptr = MacaCudaGraphFlashInferMeta._get_q_indptr_arrange(device, kv_seqlens.shape[0] + 1)

        zero_tensor = MacaCudaGraphFlashInferMeta._get_zero_tensor(device)
        tmp_cumsum = kv_seqlens.cumsum(0)
        kv_indptr = ((torch.cat((zero_tensor, tmp_cumsum)) + block_size - 1)  // block_size).int()

        max_blocks = block_offsets.shape[1]
        blocks_needed = (kv_seqlens + block_size - 1) // block_size
        mask = MacaCudaGraphFlashInferMeta._get_mask_arrange(device, max_blocks)[None, :] < blocks_needed[:, None]
        kv_indices = block_offsets[mask]

        tp_size, _ = get_tp_world_rank()
        num_local_heads = context.model_config.num_attention_heads // tp_size
        head_dim_ckv = context.model_config.hf_config.kv_lora_rank
        head_dim_kpe = context.model_config.hf_config.qk_rope_head_dim
        sm_scale = MacaCudaGraphFlashInferMeta._get_sm_scale()
        attn_metadata.flashinfer_wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_seqlens,
            num_local_heads,
            head_dim_ckv,
            head_dim_kpe,
            block_size,
            False,  # causal
            sm_scale,
            torch.bfloat16,
            torch.bfloat16,
        )

CudaGraphMixin.make_buffers_cudagraph = MacaCudaGraphMixin_make_buffers_cudagraph
CudaGraphMixin.fill_buffers_cudagraph = MacaCudaGraphMixin_fill_buffers_cudagraph
CudaGraphMixin.update_context_cudagraph = MacaCudaGraphMixin_update_context_cudagraph
