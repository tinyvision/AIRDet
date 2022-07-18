# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import datetime
import os
import time
import sys
import random
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
from copy import deepcopy

from airdet.detectors.detector_base import Detector, build_local_model, build_ddp_model
from airdet.apis.detector_inference import inference
from airdet.dataset import (
    make_data_loader
)

from airdet.utils import (
    MeterBuffer,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    save_checkpoint,
    setup_logger,
    synchronize,
)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Trainer:
    def __init__(self, config, args, is_train=True):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.config = config
        self.args = args


        self.lr = config.training.base_lr_per_img * config.training.images_per_batch

        # metric record
        self.meter = MeterBuffer(window_size=config.miscs.print_interval_iters)
        self.file_name = os.path.join(config.miscs.output_dir,
            config.exp_name)

        self.device = "cuda"

        # setup logger
        if get_rank() == 0:
            os.makedirs(self.file_name, exist_ok=True)
        setup_logger(
            self.file_name,
            distributed_rank=get_rank(),
            filename="train_log.txt",
            mode="a",
        )
        self.seed = config.miscs.seed

        # logger
        logger.info("args info: {}".format(self.args))
        logger.info("config value:\n{}".format(self.config))


    def setup_iters(self, iters_per_epoch, total_epochs, start_epoch, warmup_epochs, no_aug_epochs, eval_interval_epochs, ckpt_interval_epochs):
        self.total_iters = total_epochs * iters_per_epoch
        self.warmup_total_iters = warmup_epochs * iters_per_epoch
        self.no_aug_iters = no_aug_epochs * iters_per_epoch
        self.eval_interval_iters = eval_interval_epochs * iters_per_epoch
        self.start_iter = start_epoch * iters_per_epoch
        self.ckpt_interval_iters = ckpt_interval_epochs * iters_per_epoch
        self.no_aug = self.start_iter >= self.total_iters - self.no_aug_iters


    def build_optimizer(self):
        if "optimizer" not in self.__dict__:
            if self.config.training.warmup_epochs > 0:
                lr = self.config.training.warmup_lr
            else:
                lr = self.config.training.base_lr_each_img * self.config.training.images_per_batch

            pg0, pg1, pg2 = [], [], []

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.config.training.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.config.training.weight_decay}
            )
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer


    def get_lr(self, iters, scheduler):
        if scheduler == 'cosine':
            # Cosine lr +  warm up.
            min_lr = self.lr * self.config.training.min_lr_ratio
            warmup_lr_start = 0
            if iters <= self.warmup_total_iters:
                lr = (self.lr - warmup_lr_start) * pow(
                    iters / float(self.warmup_total_iters), 2
                ) + warmup_lr_start
            elif iters >= self.total_iters - self.no_aug_iters:
                lr = min_lr
            else:
                lr = min_lr + 0.5 * (self.lr - min_lr) * (
                    1.0
                    + math.cos(
                        math.pi
                        * (iters - self.warmup_total_iters)
                        / (self.total_iters - self.warmup_total_iters - self.no_aug_iters)
                    )
                )
            return lr

        elif scheduler == 'multi_step':
            lr = self.lr
            warmup_iters = 500
            iters_per_epoch = self.config.training.iters_per_epoch
            multiplier = self.config.training.total_epochs / 12

            if iters < warmup_iters:
                warmup_lr_ratio = 0.001
                k = (1 - iters / warmup_iters) * (1 - warmup_lr_ratio)
                lr = (1 - k) * self.lr
                return lr
            else:
                mile_stones = [iters_per_epoch*8*multiplier, iters_per_epoch*11*multiplier]
                if iters > mile_stones[0]:
                    lr = lr * 0.1
                if iters > mile_stones[1]:
                    lr = lr * 0.1
                return lr
        else:
            raise ValueError("unsupport learning rate schedule type, got {}".format(scheduler))


    def train(self, local_rank):
        # build model
        self.model = build_local_model(self.config, self.device)
                # build optimizer
        self.optimizer = self.build_optimizer()

        # resume model
        if self.config.training.resume_path is not None:
            self.resume_model(self.config.training.resume_path)
            logger.info(
                "Resume Training from Epoch: {}".format(self.epoch)
            )
        else:
            self.epoch = 0
            self.config.training.start_epoch = self.epoch
            logger.info(
                "Start Training..."
            )

        if self.config.training.ema:
            self.ema_model = deepcopy(self.model).eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
            self.ema_momentum = self.config.training.ema_momentum
            self.ema_scheduler = lambda x: self.ema_momentum * (1 - math.exp(-x / 2000))

        if self.config.training.use_syncBN:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        logger.info(
            "Model Summary: {}".format(get_model_info(self.model, (640, 640)))
        )

        # distributed model init
        self.model = build_ddp_model(self.model, local_rank)

        # dataset init
        self.train_loader = make_data_loader(self.config, is_train=True)
        self.val_loader = make_data_loader(self.config, is_train=False)

        self.setup_iters(self.config.training.iters_per_epoch,
            self.config.training.total_epochs,
            self.epoch, self.config.training.warmup_epochs,
            self.config.training.no_aug_epochs,
            self.config.miscs.eval_interval_epochs, self.config.miscs.ckpt_interval_epochs)
        if local_rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")

        # ----------- start training ------------------------- #
        self.model.train()
        for data_iter, (inps, targets, ids) in enumerate(self.train_loader):
            cur_iter = self.start_iter + data_iter

            iter_start_time = time.time()

            inps = inps.to(self.device)    # ImageList: tensors, img_size
            targets = [target.to(self.device) for target in targets]    # BoxList: bbox, num_boxes ...

            data_end_time = time.time()

            outputs = self.model(inps, targets)

            loss = outputs["total_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.config.training.ema:
                with torch.no_grad():
                    ema_momentum = self.ema_scheduler(cur_iter)
                    student = self.model.module.state_dict()
                    for name, param in self.ema_model.state_dict().items():
                        if param.dtype.is_floating_point:
                            param *= ema_momentum
                            param += (1.0 - ema_momentum) * student[name].detach()

            # setting learning rate
            lr = self.get_lr(cur_iter, self.config.training.lr_scheduler)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            outputs_array = {_name:_v.item() for _name, _v in outputs.items()}
            iter_end_time = time.time()
            self.meter.update(
                iter_time=iter_end_time - iter_start_time,
                data_time=data_end_time - iter_start_time,
                lr=lr,
                **outputs_array,
            )

            if cur_iter + 1 == self.total_iters - self.no_aug_iters or self.no_aug:
                if self.config.training.augmentation.mosaic:
                    logger.info("--->No mosaic aug now!")
                    self.train_loader.batch_sampler.set_mosaic(False)
                self.save_ckpt(ckpt_name="last_mosaic_epoch", local_rank = local_rank)
            # log needed information
            if (cur_iter + 1) % self.config.miscs.print_interval_iters == 0:
                left_iters = self.total_iters - (cur_iter + 1)
                eta_seconds = self.meter["iter_time"].global_avg * left_iters
                eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

                progress_str = "epoch: {}/{}, iter: {}/{}".format(
                    self.epoch + 1, self.config.training.total_epochs, (cur_iter + 1) % self.config.training.iters_per_epoch, self.config.training.iters_per_epoch
                )
                loss_meter = self.meter.get_filtered_meter("loss")
                loss_str = ", ".join(
                    ["{}: {:.1f}".format(k, v.avg) for k, v in loss_meter.items()]
                )

                time_meter = self.meter.get_filtered_meter("time")
                time_str = ", ".join(
                    ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
                )

                logger.info(
                    "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                        progress_str,
                        gpu_mem_usage(),
                        time_str,
                        loss_str,
                        self.meter["lr"].latest,
                    )
                    + (", size: ({:d}, {:d}), {}".format(inps.tensors.shape[2], inps.tensors.shape[3], eta_str))
                )
                self.meter.clear_meters()

            if (cur_iter + 1) % self.eval_interval_iters == 0:
                time.sleep(0.003)
                self.evaluate(local_rank)
                self.model.train()
            synchronize()

            if (cur_iter + 1) % self.ckpt_interval_iters == 0:
                self.save_ckpt("epoch_%d" % (self.epoch + 1), local_rank = local_rank)

            if (cur_iter + 1) % self.config.training.iters_per_epoch == 0:
                self.epoch = self.epoch + 1

        self.save_ckpt(ckpt_name="latest", local_rank = local_rank)


    def save_ckpt(self, ckpt_name, local_rank, update_best_ckpt=False):
        if local_rank == 0:
            save_model = self.ema_model if self.config.training.ema else self.model.module
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )


    def resume_model(self, resume_path):
        ckpt_file_path = self.config.training.resume_path
        ckpt = torch.load(ckpt_file_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = ckpt["start_epoch"]
        self.config.training.start_epoch = self.epoch


    def evaluate(self, local_rank):
        if not self.config.testing.multi_gpu and local_rank != 0:
            return

        if self.config.training.ema:
            evalmodel = self.ema_model
            print('using ema model for evaluation')
        else:
            evalmodel = self.model
            if isinstance(evalmodel, DDP):
                evalmodel = evalmodel.module

        output_folders = [None] * len(self.config.dataset.val_ann)
        if self.config.miscs.output_dir:
            for idx, dataset_name in enumerate(self.config.dataset.val_ann):
                output_folder = os.path.join(self.config.miscs.output_dir, "inference", dataset_name)
                if local_rank == 0:
                    mkdir(output_folder)
                output_folders[idx] = output_folder

        for output_folder, dataset_name, data_loader_val in zip(output_folders, self.config.dataset.val_ann, self.val_loader):
            inference(
                self.config,
                evalmodel,
                data_loader_val,
                dataset_name,
                device = self.device,
                output_folder = output_folder,
                multi_gpu_infer = self.config.testing.multi_gpu,
            )

