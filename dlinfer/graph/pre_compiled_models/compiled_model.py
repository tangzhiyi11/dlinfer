from dlinfer.graph.pre_compiled_models.internlm2.internlm2 import InternLM2Model

from lmdeploy.pytorch.models.internlm2 import InternLM2ForCausalLM

from lmdeploy.pytorch.config import ModelConfig

class CompiledModel:
    def __init__(self, model, model_config: ModelConfig):
        self.model = self.get_model_from_config(model, model_config)
    
    def get_model_from_config(self, model, model_config):
        name_or_path = model_config.hf_config.name_or_path
        if "internlm2" in name_or_path:
            assert isinstance(model, InternLM2ForCausalLM), "model must be lmdeploy.pytorch.models.internlm2.InternLM2ForCausalLM"
            return InternLM2Model(model, model_config)
        else:
            raise ValueError(f"Model {name_or_path} not supported")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)