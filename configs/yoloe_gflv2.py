#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from airdet.config import Config as MyConfig
from airdet.config.backbones import CSPRepResnet
from airdet.config.necks import CSPPAN

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.model.backbone = CSPRepResnet
        self.model.neck = CSPPAN
        self.model.head.in_channels = [96,192,384]

        self.model.head.stacked_convs = 0
        self.model.head.act = 'relu'
        self.model.head.use_ese = True
        self.model.head.conv_groups = 1
        self.model.head.feat_channels = [96, 192, 384]

