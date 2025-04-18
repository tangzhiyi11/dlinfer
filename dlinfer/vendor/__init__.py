# Copyright (c) 2024, DeepLink. All rights reserved.
import importlib
from pathlib import Path
from functools import lru_cache
import yaml
import torch


vendor_ops_registry = dict()
vendor_is_initialized = False
vendor_name_file = Path(__file__).parent / "vendor.yaml"
linear_w8a8_scale_type = torch.Tensor
dynamic_quant_scale_type = torch.Tensor


with open(str(vendor_name_file), "r") as f:
    config = yaml.safe_load(f)
    vendor_name = config["vendor"]
    dispatch_key = config["dispatch_key"]


@lru_cache(1)
def import_vendor_module(vendor_name_str):
    return importlib.import_module(f".{vendor_name_str}", __package__)


def vendor_torch_init():
    import_vendor_module(vendor_name)
    global vendor_is_initialized
    vendor_is_initialized = True
    global linear_w8a8_scale_type, dynamic_quant_scale_type
    linear_w8a8_scale_type = torch.Tensor if vendor_name in ["ascend"] else float
    dynamic_quant_scale_type = torch.Tensor if vendor_name in ["ascend"] else float
