# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import os
import cv2
from torchvision.utils import save_image
from datasets.helper.image_transform import recover_image

def log_image(images, logimg_dir, epoch, data_iter_step):
    images = [recover_image(image) for image in images]
    batchsize = min([ image.shape[0] for image in images])

    image_grid = []
    for b in range(min(10, batchsize)):
        image_list = [image[b] for image in images]
        image_row = cv2.hconcat(image_list)
        image_grid.append(image_row)
    image_grid = cv2.vconcat(image_grid)

    ## resize image_grid to fx=0.5, fy=0.5 for saving space
    image_grid = cv2.resize(image_grid, (0, 0), fx=0.5, fy=0.5)
    image_file = os.path.join(logimg_dir, f"epoch_{epoch}_iter_{data_iter_step}.png")
    cv2.imwrite(image_file, image_grid)


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))


    for data_iter_step, entry in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = entry['image']
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        random_idx_to_flip = torch.randperm(samples.size(0))[:int(samples.size(0) * 0.5)]
        samples[random_idx_to_flip] = torch.flip(samples[random_idx_to_flip], dims=[3])

      
        with torch.cuda.amp.autocast():
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        ## NOTE: log_writer is not None means the global rank is 0
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        if log_writer is not None and data_iter_step % ( len(data_loader) // 10 ) == 0:


            # Create image dir under args.output_dir if it doesn't exist
            image_output_dir = os.path.join(args.output_dir, "vis_images")
            if not os.path.exists(image_output_dir):
                os.makedirs(image_output_dir)

            batchsize = samples.shape[0]
            vis_samples = samples[: min(10, batchsize) ]
            vis_mask = mask[: min(10, batchsize) ]
            vis_pred = pred[: min(10, batchsize) ]
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
                vis_dict = model.module.vis( vis_samples, vis_mask, vis_pred)
            else:
                vis_dict = model.vis( vis_samples, vis_mask, vis_pred )
            log_image([vis_dict['imgs'], vis_dict['masked_imgs'], vis_dict['imgs_recon'] ], image_output_dir, epoch, data_iter_step)



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}