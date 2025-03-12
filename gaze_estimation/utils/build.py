import argparse, os, sys, datetime, glob, importlib, csv
import os.path as osp

import logging
from omegaconf import OmegaConf


import logging
logging.basicConfig(level=logging.INFO)

class DataCfgInterpreter(object):

	@classmethod
	def merge(cls, data_cfg_yaml:str, data_locations, split_yaml:str=None, data_sanity_check=False):
		data_cfg = OmegaConf.load(data_cfg_yaml)
		if split_yaml is not None:
			prefixes = OmegaConf.load( split_yaml) #os.path.join(os.path.dirname(os.path.abspath(__file__)), f'configs/split/{split_tag}.yaml') )

		keys = ['train', 
		  		'train_single_frame',
				'train_unsupervised',
				'target_dataset',
				'prepared_target_dataset',
		  		'train_2',
		  		'val', 
				'test', 
				'test_2', 
				'test_3', 
				'test_4', 
				'target',
				]
		for key in keys:

			if key in data_cfg['data_cfg']:
				this_data_name = data_cfg['data_cfg'][key].params.data_name
				if this_data_name in data_locations.data:
					data_cfg['data_cfg'][key].params.dataset_path = data_locations.data[this_data_name]
				else:
					logging.warning(f'dataset {this_data_name} not in data_locations.data'); exit(0)

				if split_yaml is not None:
					data_cfg['data_cfg'][key].params.keys_to_use = prefixes[key]

				### NOTE: only use 1 subject for sanity check
				if data_sanity_check:
					data_cfg['data_cfg'][key].params.keys_to_use = data_cfg['data_cfg'][key].params.keys_to_use[:1]

		return data_cfg['data_cfg']
	

class ModelCfgInterpreter(object):
	@classmethod
	def merge(cls, model_cfg_yaml:str, model_loc_yaml:str, model_idx:str):
		return


def update_loss_cfg(model_cfg, config):
	l1_loss = {
		'target': 'criteria.gaze_loss.PitchYawLoss',
		'params': {
			'loss_type': 'l1'
		}
	}

	l1_ep005 = {
		'target': 'criteria.gaze_loss.PitchYawLoss',
		'params': {
			'loss_type': 'l1',
			'epsilon': 0.05
		}
	}

	l1_ep01 = {
		'target': 'criteria.gaze_loss.PitchYawLoss',
		'params': {
			'loss_type': 'l1',
			'epsilon': 0.1
		}
	}

	angular_loss = {
		'target': 'criteria.gaze_loss.PitchYawLoss',
		'params': {
			'loss_type': 'angular'
		}
	}
	angular_ep5 = {
		'target': 'criteria.gaze_loss.PitchYawLoss',
		'params': {
			'loss_type': 'angular',
			'epsilon': 5
		}
	}
	angular_ep10 = {
		'target': 'criteria.gaze_loss.PitchYawLoss',
		'params': {
			'loss_type': 'angular',
			'epsilon': 10
		}
	}


	if config.loss_overwrite is not None:
		options = ['l1', 'angular', 'l1 eps=0.05', 'l1 eps=0.1', 'angular eps=5', 'angular eps=10']
		if config.loss_overwrite not in options:
			logging.warning(f'loss_overwrite {config.loss_overwrite} not in {options}'); exit(0)
		
		if config.loss_overwrite == 'l1':
			model_cfg.loss_config = l1_loss
			model_cfg.loss_augment_config = l1_loss
		elif config.loss_overwrite == 'l1 eps=0.05':
			model_cfg.loss_config = l1_loss
			model_cfg.loss_augment_config = l1_ep005
		elif config.loss_overwrite == 'l1 eps=0.1':
			model_cfg.loss_config = l1_loss
			model_cfg.loss_augment_config = l1_ep01

		elif config.loss_overwrite == 'angular':
			model_cfg.loss_config = angular_loss
			model_cfg.loss_augment_config = angular_loss
		elif config.loss_overwrite == 'angular eps=5':
			model_cfg.loss_config = angular_loss
			model_cfg.loss_augment_config = angular_ep5
		elif config.loss_overwrite == 'angular eps=10':
			model_cfg.loss_config = angular_loss
			model_cfg.loss_augment_config = angular_ep10
		
		print('updated the loss function to ', config.loss_overwrite)
	else:
		print('no loss is updated')
	return model_cfg


def 