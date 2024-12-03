import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import logging
import os, sys
from glob import glob
import uuid
import datetime
import importlib
import numpy as np
import torch

from utils import instantiate_from_cfg, save_files

from utils.util import transform_date_str, set_seed

logging.basicConfig(level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
import random

import utils.misc as misc
import builtins

def init_distributed_mode(cfg):
	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ: 
		## this means the script is ran by # running "torchrun --nnodes X --nproc_per_node Y main_ddp.py"
		## "python main_ddp.py" will not have these env variables
		cfg.rank = int(os.environ["RANK"]) 
		cfg.world_size = int(os.environ['WORLD_SIZE']) # nproc_per_node * nnodes
		cfg.gpu = int(os.environ['LOCAL_RANK'])
	# elif 'SLURM_PROCID' in os.environ:
	# 	cfg.rank = int(os.environ['SLURM_PROCID'])
	# 	cfg.gpu = cfg.rank % torch.cuda.device_count()
	else:
		print('Not using distributed mode')
		misc.setup_for_distributed(is_master=True)  # hack
		cfg.distributed = False
		return
	# port = random.randint(10000, 20000)
	# cfg.dist_url = f'tcp://{os.environ["MASTER_ADDR"]}:{port}'
	cfg.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
	cfg.distributed = True
	torch.cuda.set_device(cfg.gpu)
	cfg.dist_backend = 'nccl'
	print('| distributed init (rank {}): {}, gpu {}'.format(
		cfg.rank, cfg.dist_url, cfg.gpu), flush=True)
	torch.distributed.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
										 world_size=cfg.world_size, rank=cfg.rank)

	torch.distributed.barrier()
	misc.setup_for_distributed(cfg.rank == 0)
	
def merge_data_cfg(data_cfg_paths: DictConfig, data_locations: DictConfig, 
					cfg):
	"""
	data_cfg_paths:
		train: 
			List: [ path to the training data yaml ]
		train_unlabel: 
			List: [ the path to the unsupervised training data yaml ]
		val: 
			List: [ the path to the validation data yaml ]
		test:
			List: [ the path to the test data yaml ]
		... 
	"""
	data_sanity_check = cfg.data_sanity_check if 'data_sanity_check' in cfg else False
	# img_augmentation = cfg.img_augmentation if 'img_augmentation' in cfg else False
	scratch_node_dir = cfg.scratch_node_dir if 'scratch_node_dir' in cfg else None
	

	data_cfg = OmegaConf.create()
	for k, v in data_cfg_paths.items():
		# k is "train", "train_unlabel", "val", "test", etc.
		k_list = []
		for path_item in v:
			k_list.append(OmegaConf.load(path_item))
		data_cfg[k] = k_list


	predicted_label_dir = cfg.get('predicted_label_dir', None)

	"""Merges data configuration with data locations and split information."""
	for data_split_key in data_cfg:
		for data_cfg_item in data_cfg[data_split_key]:
			data_name = data_cfg_item.params.data_name
			if data_name in data_locations.data:
				dataset_path = data_locations.data[data_name]
				if scratch_node_dir is not None:
					dataset_path_in_scratch_node = os.path.join(scratch_node_dir, os.path.basename(dataset_path))
					print(" reset the dataset path to the scratch node: ", dataset_path_in_scratch_node)
					keys_in_this_dir = glob( f"{dataset_path_in_scratch_node}/*" )
					print( "keys_in_this_dir: ", keys_in_this_dir )
					dataset_path = dataset_path_in_scratch_node
				data_cfg_item.params.dataset_path = dataset_path
				
			else:
				logging.warning(f'dataset {data_name} not in data_locations.data')
				raise ValueError(f'dataset {data_name} not in data_locations.data') 
			# if data_sanity_check:
			# 	data_cfg_item.params.keys_to_use = data_cfg_item.params.keys_to_use[:1]
			if predicted_label_dir is not None and data_split_key == 'train':
				data_cfg_item.params.pred_gaze_txt = os.path.join(predicted_label_dir, 'prediction_' + data_name,  'epoch_1001/pred_gaze.txt')

	return data_cfg

def update_config(cfg: DictConfig):
	now_day = datetime.datetime.now().strftime("%Y-%m-%d")
	now_time = datetime.datetime.now().strftime("%H-%M-%S")
	random_seed = cfg.random_seed
	output_dir = cfg.output_dir
	exp_name = cfg.exp.exp_name
	# pid = os.getpid()
	unique_uuid = str(uuid.uuid4()).split('-')[0]
	output_dir = os.path.join(cfg.output_dir, now_day, 
						   f'{exp_name}/{now_time}_seed{random_seed}_{unique_uuid}')
	
	####  ------------------------------- cfg_path to cfg ------------------------------------
	exp_cfg = cfg.exp
	data_cfg_paths = OmegaConf.load(exp_cfg.data)
	model_cfg = OmegaConf.load(exp_cfg.model)
	trainer_cfg = OmegaConf.load(exp_cfg.trainer)
	loss_cfg = OmegaConf.load(exp_cfg.loss)
	optimizer_cfg = OmegaConf.load(exp_cfg.optimizer)
	scheduler_cfg = OmegaConf.load(exp_cfg.scheduler)

	data_cfg = merge_data_cfg(
		data_cfg_paths=data_cfg_paths, # convert to DictConfig
		data_locations=data_locations,
		cfg=cfg,
	)
	# Temporarily disable struct mode to add the new key
	with open_dict(cfg):
		cfg.project_name = f'LGM_{transform_date_str(now_day)}'
		cfg.output_dir = output_dir
		cfg = OmegaConf.merge(cfg, {'data': data_cfg, 'model': model_cfg, "trainer": trainer_cfg, "loss": loss_cfg, "optimizer": optimizer_cfg, "scheduler": scheduler_cfg})
	return cfg

project_dir = os.path.dirname(os.path.abspath(__file__))
data_locations = OmegaConf.load(os.path.join(project_dir, 'configs/data_path.yaml'))

## NOTE： This decorator tells Hydra to load your configuration from 
##    - the configs directory and 
##    - use config.yaml as the default configuration file.
@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):	
	cfg = update_config(cfg)
	init_distributed_mode(cfg)
	
	if cfg.distributed:
		set_seed( cfg.random_seed + misc.get_rank() )
		if cfg.rank == 0:
			cfg.output_dir = cfg.output_dir+'_rank0'
			save_files(project_dir, cfg.output_dir)
	else:
		set_seed(cfg.random_seed) 
		save_files(project_dir, cfg.output_dir)
	

	module, cls = cfg.trainer.type.rsplit(".", 1) 
	trainer = getattr(importlib.import_module(module, package=None), cls)(cfg)

	
	if cfg.mode == 'train':
		trainer.train()
	elif cfg.mode == 'test':
		trainer.evaluate()      

if __name__ == "__main__":
	main()