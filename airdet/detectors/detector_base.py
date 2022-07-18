# Copyright (C) Alibaba Group Holding Limited. All rights reserved.


import torch
import torch.nn as nn
import os

from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from airdet.base_models.heads import build_head
from airdet.base_models.necks import build_neck
from airdet.base_models.backbones import build_backbone
from airdet.structures.image_list import to_image_list

import numpy as np

class Detector(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = build_backbone(config.model.backbone)
        self.neck = build_neck(config.model.neck)
        self.head = build_head(config.model.head)

        self.config = config

    def init_bn(self, M):

        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def init_model(self):

        self.apply(self.init_bn)

        if self.config.training.pretrain_model is None:
            self.backbone.init_weights()
            self.neck.init_weights()
            self.head.init_weights()
        else:
            self.load_pretrain_detector(self.config.training.pretrain_model)

    def load_pretrain_detector(self, pretrain_model):

        state_dict = torch.load(pretrain_model, map_location='cpu')['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            if 'head' in k:
                new_state_dict[k] = self.state_dict()[k]
                continue
            new_state_dict[k] = v

        self.load_state_dict(new_state_dict)
        print(f'load params from {pretrain_model}')


    def forward(self, x, targets=None, distill=False, tea=False, stu=False):
        images = to_image_list(x)
        feature_outs = self.backbone(images.tensors)  # list of tensor
        fpn_outs = self.neck(feature_outs)

        if distill and tea:
            return fpn_outs

        if distill and stu:
            if self.training:
                loss_dict = self.head(
                                xin=fpn_outs,
                                labels=targets,
                                imgs=images,
                                )
                return loss_dict, fpn_outs
            else:
                outputs = self.head(fpn_outs, imgs=images)
                return outputs

        if self.training:
            loss_dict = self.head(
                            xin=fpn_outs,
                            labels=targets,
                            imgs=images,
                            )
            return loss_dict
        else:
            outputs = self.head(fpn_outs, imgs=images)
            return outputs


def build_local_model(config, device):
    model = Detector(config)

    if config.training.pretrain_model is not None:
        model.init_model()
        model.load_pretrain_detector(config.training.pretrain_model)
    else:
        model.init_model()

    model.to(device)

    return model


def build_ddp_model(model, local_rank):
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)

    return model



