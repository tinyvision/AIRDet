#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from airdet.config import Config as MyConfig
from airdet.config.backbones import CSPRepResnet
from airdet.config.necks import CSPPAN, PAFPNNeck

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.training.total_epochs = 36
        self.training.lr_scheduler = 'multi_step'
        self.training.mosaic = False
        self.training.use_autoaug = False
        self.miscs.eval_interval_epochs = 1
        self.miscs.ckpt_interval_epochs = 1

        self.model.backbone = CSPRepResnet
        #self.model.backbone.width_mult = 0.5
        self.model.backbone.act = 'relu'
        self.model.backbone.return_idx = [1,2,3]

        #self.model.neck = PAFPNNeck
        #self.model.neck.in_features = [2,3,4]
        #self.model.neck.in_channels = [256, 512, 1024]
        #self.model.neck.width = 0.25

        self.model.neck = CSPPAN
        self.model.neck.out_channels = [768, 384, 192]
        self.model.neck.width_mult = 0.5

        self.model.head.in_channels = [96,192,384]

        self.model.head.stacked_convs = 0
        self.model.head.act = 'relu'
        self.model.head.use_ese = True
        self.model.head.conv_groups = 1
        self.model.head.feat_channels = [96, 192, 384]


