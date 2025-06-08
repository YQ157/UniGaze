
import importlib
from omegaconf import DictConfig, OmegaConf, open_dict
import logging
import os
import uuid
import datetime
import importlib
import numpy as np
import torch




def instantiate_from_cfg(config):
	if not "type" in config:
		raise KeyError("Expected key `type` to instantiate.")
	return get_obj_from_str(config["type"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
	module, cls = string.rsplit(".", 1)
	if reload:
		module_imp = importlib.import_module(module)
		importlib.reload(module_imp)
	return getattr(importlib.import_module(module, package=None), cls)

