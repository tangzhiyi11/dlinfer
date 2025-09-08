from . import deepseek_v2

from lmdeploy.pytorch.models.module_map import DEVICE_SPECIAL_MODULE_MAP


# DEVICE_SPECIAL_MODULE_MAP['maca'].update({
#     'DeepseekV2ForCausalLM': 'dlinfer.framework.lmdeploy_ext.models.maca.deepseek_v2.DeepseekV2ForCausalLM',
# })