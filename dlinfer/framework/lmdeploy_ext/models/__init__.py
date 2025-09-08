# Copyright (c) 2024, DeepLink. All rights reserved.
import importlib
from functools import lru_cache
from dlinfer.vendor import vendor_name


models_vendor = ["maca"]


@lru_cache(1)
def import_vendor_module(vendor_name_str):
    if vendor_name_str in models_vendor:
        importlib.import_module(f".{vendor_name_str}", __package__)


def vendor_models_init():
    import_vendor_module(vendor_name)


vendor_models_init()
