import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Any, List
from lmdeploy.pytorch.models.internlm2 import InternLM2ForCausalLM
from lmdeploy.pytorch.config import ModelConfig

from dlinfer.graph.pre_compiled_models.internlm2.prefill import prefill_call, prefill_compile_model
from dlinfer.graph.pre_compiled_models.internlm2.decode import decode_call, decode_compile_model
class InternLM2Model(nn.Module):
    def __init__(self, model: InternLM2ForCausalLM, model_config: ModelConfig):
        super().__init__()
        self._model = model.model  # 使用私有属性保存初始化的model
        self.torch_comiled_model = torch.compile(self._model.forward, fullgraph=True, dynamic=True, backend='atbgraph')
        # import pdb;pdb.set_trace()
        self.model_config = model_config
        self.check()
        self.compile_model()
        
    def compile_model(self):
        prefill_compile_model()
        decode_compile_model()

    def forward(self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            past_key_values: List[List[torch.Tensor]],
            attn_metadata: Any = None,
            inputs_embeds: torch.Tensor = None,
            **kwargs,):
        # import pdb;pdb.set_trace()
        if attn_metadata.is_unpaged_prefill:
            return prefill_call(self._model, input_ids, position_ids, past_key_values, attn_metadata, inputs_embeds, **kwargs)
        else:
            return decode_call(self._model, input_ids, position_ids, past_key_values, attn_metadata, inputs_embeds, **kwargs)

    def check(self):
        hf_config = self.model_config.hf_config
        assert self.model_config.num_layers == 32
        assert self.model_config.dtype == torch.bfloat16
        assert self.model_config.num_attention_heads == 32
        assert self.model_config.num_key_value_heads == 8
        assert self.model_config.vocab_size == 92544
        assert self.model_config.hidden_size == 4096
        assert self.model_config.k_head_dim == 128
        assert self.model_config.v_head_dim == 128
        assert self.model_config.head_dim == 128


        
        

        
