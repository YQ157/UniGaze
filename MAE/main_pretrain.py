# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os, shutil
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch

from omegaconf import DictConfig, OmegaConf, open_dict
from utils import instantiate_from_cfg
import uuid
import datetime
from glob import glob

def get_args_parser():
	def str2bool(v):
		if isinstance(v, bool):
			return v
		if v.lower() in ("yes", "true", "t", "y", "1"):
			return True
		elif v.lower() in ("no", "false", "f", "n", "0"):
			return False
		else:
			raise argparse.ArgumentTypeError("Boolean value expected.")
	parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
	parser.add_argument('--batch_size', default=64, type=int,
						help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
	parser.add_argument('--epochs', default=400, type=int)
	parser.add_argument('--accum_iter', default=1, type=int,
						help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

	# Model parameters
	parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
						help='Name of model to train')

	parser.add_argument('--input_size', default=224, type=int,
						help='images input size')

	parser.add_argument('--mask_ratio', default=0.75, type=float,
						help='Masking ratio (percentage of removed patches).')

	parser.add_argument('--norm_pix_loss', action='store_true',
						help='Use (per-patch) normalized pixels as targets for computing loss')
	parser.set_defaults(norm_pix_loss=False)

	# Optimizer parameters
	parser.add_argument('--weight_decay', type=float, default=0.05,
						help='weight decay (default: 0.05)')

	parser.add_argument('--lr', type=float, default=None, metavar='LR',
						help='learning rate (absolute lr)')
	parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
						help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
	parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
						help='lower lr bound for cyclic schedulers that hit 0')

	parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
						help='epochs to warmup LR')

	# Dataset parameters
	parser.add_argument('--data_yamls', nargs='+', type=str, required=True,
						help='List of dataset configuration files, if mulitple provided, then combined together')

	parser.add_argument('--output_dir', default='./logs',
						help='path where to save, empty for no saving')
	parser.add_argument('--log_dir', default='./output_dir',
						help='path where to tensorboard log')
	parser.add_argument('--device', default='cuda',
						help='device to use for training / testing')
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--resume', default='',
						help='resume from checkpoint')

	parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
						help='start epoch')
	parser.add_argument('--num_workers', default=16, type=int)
	parser.add_argument('--pin_mem', action='store_true',
						help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
	parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
	parser.set_defaults(pin_mem=True)

	# distributed training parameters
	parser.add_argument('--world_size', default=1, type=int,
						help='number of distributed processes')
	parser.add_argument('--local_rank', default=-1, type=int)
	parser.add_argument('--dist_on_itp', action='store_true')
	parser.add_argument('--dist_url', default='env://',
						help='url used to set up distributed training')
	


	parser.add_argument('--data_subset_ratio', default=1.0, type=float,
						help='Dataset ratio used for training (percentage of each dataset).')
	parser.add_argument('--limit_head_pose', default=None, type=float,
						help='The threshold of head pose to limit the dataset. only support VFHQ, VGGFace2, and CelebV')
	
	return parser


def main(args):

	
	print(" start misc.init_distributed_mode(args) ")
	misc.init_distributed_mode(args)
	print(" end misc.init_distributed_mode(args) ")

	print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
	print("{}".format(args).replace(', ', ',\n'))

	device = torch.device(args.device)

	# fix the seed for reproducibility
	seed = args.seed + misc.get_rank()
	torch.manual_seed(seed)
	np.random.seed(seed)

	cudnn.benchmark = True

	# ────────────────────────────────────────────────────────────────────
	# Custom dataset configuration  (The only difference from the original script)
	# ────────────────────────────────────────────────────────────────────
	project_dir = os.path.dirname(os.path.abspath(__file__))
	data_locations = OmegaConf.load(os.path.join(project_dir, 'configs/data_path.yaml'))
	data_cfgs = [] 
	for data_yaml_path in args.data_yamls:
		data_cfg = OmegaConf.load(data_yaml_path)
		data_name = data_cfg.params.data_name
		if args.limit_head_pose is not None:
			data_cfg.params.limit_head_pose = args.limit_head_pose
		if data_name in data_locations.data:
			dataset_path = data_locations.data[data_name]
			data_cfg.params.dataset_path = dataset_path
		else:
			logging.warning(f'dataset {data_name} not in data_locations.data')
			raise ValueError(f'dataset {data_name} not in data_locations.data') 
		data_cfgs.append(data_cfg)

	print("data_cfgs: ", data_cfgs)

	datasets = []
	print(" ---------------------------------- data info ---------------------------------- ")
	for data_cfg in data_cfgs:
		dataset_i = instantiate_from_cfg(data_cfg)
		if args.data_subset_ratio < 1.0:
			dataset_i = torch.utils.data.Subset(dataset_i, np.random.choice(len(dataset_i), int(args.data_subset_ratio * len(dataset_i)), replace=False))
		datasets.append(dataset_i)
		
		if args.data_subset_ratio < 1.0:
			print(f" {data_cfg.params.data_name} has {len(dataset_i)} <--- This is {args.data_subset_ratio*100}% of the whole dataset\n")
		else:
			print(f" {data_cfg.params.data_name} has {len(dataset_i)}\n")
	print(" ------------------------------------------------------------------------------- ")

	dataset_train = torch.utils.data.ConcatDataset(datasets)
	print(" total number of images: ", len(dataset_train))


	# ────────────────────────────────────────────────────────────────────
	

	if True:  # args.distributed:
		num_tasks = misc.get_world_size()
		global_rank = misc.get_rank()
		sampler_train = torch.utils.data.DistributedSampler(
			dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
		)
		print("Sampler_train = %s" % str(sampler_train))
	else:
		sampler_train = torch.utils.data.RandomSampler(dataset_train)

	if global_rank == 0 and args.log_dir is not None:
		os.makedirs(args.log_dir, exist_ok=True)
		log_writer = SummaryWriter(log_dir=args.log_dir)
	else:
		log_writer = None

	data_loader_train = torch.utils.data.DataLoader(
		dataset_train, sampler=sampler_train,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=True,
	)
	
	# define the model
	model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

	model.to(device)

	model_without_ddp = model
	print("Model = %s" % str(model_without_ddp))

	eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
	
	if args.lr is None:  # only base_lr is specified
		args.lr = args.blr * eff_batch_size / 256

	print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
	print("actual lr: %.2e" % args.lr)

	print("accumulate grad iterations: %d" % args.accum_iter)
	print("effective batch size: %d" % eff_batch_size)

	if args.distributed:
		print(" initialize model to torch.nn.parallel.DistributedDataParallel ")
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
		print(" finished. ")
		model_without_ddp = model.module
	
	# following timm: set wd as 0 for bias and norm layers
	param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
	optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
	print(optimizer)
	loss_scaler = NativeScaler()

	misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

	print(f"Start training for {args.epochs} epochs")
	start_time = time.time()
	for epoch in range(args.start_epoch, args.epochs):
		if args.distributed:
			data_loader_train.sampler.set_epoch(epoch)
		train_stats = train_one_epoch(
			model, data_loader_train,
			optimizer, device, epoch, loss_scaler,
			log_writer=log_writer,
			args=args
		)
		# if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
		# 	misc.save_model(
		# 		args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
		# 		loss_scaler=loss_scaler, epoch=epoch)

		# Check if we need to delete the previous checkpoint
		if args.output_dir:
			misc.save_model(
				args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
				loss_scaler=loss_scaler, epoch=epoch)
			print(" saved model at epoch %d to %s" % (epoch, args.output_dir))
		previous_epoch = epoch - 1
		if previous_epoch % 20 != 0 and previous_epoch >= 0:  # Do not delete checkpoints from multiples of 20
			prev_checkpoint = f"{args.output_dir}/checkpoint-{previous_epoch}.pth"
			if os.path.exists(prev_checkpoint):
				os.remove(prev_checkpoint)
				print(f" deleted {prev_checkpoint}")



		log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
						'epoch': epoch,}

		if args.output_dir and misc.is_main_process():
			if log_writer is not None:
				log_writer.flush()
			with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
				f.write(json.dumps(log_stats) + "\n")

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
	args = get_args_parser()
	args = args.parse_args()
	if args.output_dir:
		Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	
	now_day = datetime.datetime.now().strftime("%Y-%m-%d")
	now_time = datetime.datetime.now().strftime("%H-%M-%S")
	# pid = os.getpid()
	unique_uuid = str(uuid.uuid4()).split('-')[0]
	output_dir = os.path.join(args.output_dir, now_day, f'{now_time}_{unique_uuid}')

	args.output_dir = output_dir
	args.log_dir = os.path.join(output_dir, 'logs')

	main(args)
