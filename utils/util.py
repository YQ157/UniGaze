import importlib
import os
import random
import torch
import numpy as np
from collections import abc
from einops import rearrange
from functools import partial

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
import shutil

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params

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


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
        func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res


def set_seed(seed):
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	# ensure reproducibility
	os.environ["PYTHONHASHSEED"] = str(seed)
	

def transform_date_str(date_str):
    from datetime import datetime

    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Calculate the week of the month
    start_of_month = datetime(date_obj.year, date_obj.month, 1)
    week_of_month = (date_obj - start_of_month).days // 7 + 1

    return f"{date_obj.year}{date_obj.month:02}week{week_of_month}"



def save_files(base_dir, run_directory, extensions=('.py', '.yaml')):
    run_directory = os.path.join(run_directory, 'run')
    os.makedirs(run_directory, exist_ok=True)
    src_dirs = [
        "configs", 
        "criteria", 
        "datasets", 
        "models",
        "trainers", 
        "utils",
    ]
    src_dirs = [os.path.join(base_dir, src_dir) for src_dir in src_dirs]

    for src_dir in src_dirs:
        # Traverse the directory tree
        for root, dirs, files in os.walk(src_dir):
            # Calculate the relative path from the base directory
            relative_path = os.path.relpath(root, base_dir)
            dest_dir = os.path.join(run_directory, relative_path)
            os.makedirs(dest_dir, exist_ok=True)
            # Copy files with the specified extensions
            for file in files:
                if file.endswith(extensions):
                    src_file_path = os.path.join(root, file)
                    dest_file_path = os.path.join(dest_dir, file)
                    shutil.copy(src_file_path, dest_file_path)
                    # print(f"Saved {src_file_path} to {dest_file_path}")


def call_model_method(model, method_name, *args, **kwargs):
    """
    Calls a method on the model, regardless of whether it is wrapped in DataParallel or not.
    :param model: The model or DataParallel wrapped model.
    :param method_name: The name of the method to call.
    :param args: Positional arguments to pass to the method.
    :param kwargs: Keyword arguments to pass to the method.
    """
    
    if isinstance(model, torch.nn.DataParallel):
        target_model = model.module
    else:
        target_model = model
    # Get the method and call it
    method = getattr(target_model, method_name)
    
    return method(*args, **kwargs)

def get_attributes_with_prefix(instance, prefix):
    return {attr_name: getattr(instance, attr_name) for attr_name in vars(instance) if attr_name.startswith(prefix)}


def update_ema_params(model, ema_model, alpha, global_step):
	alpha = min(1 - 1 / (global_step + 1), alpha)
	# print('ema_model = ema_model * {} + (1 - {}) * model'.format(alpha, alpha))
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		# ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
		ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
