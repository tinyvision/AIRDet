# Copyright (C) Alibaba Group Holding Limited. All rights reserved.


import copy

from .darknet import CSPDarknet
from .csp_represnet import CSPRepResnet
from .repvgg import EfficientRep

def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == "CSPDarknet":
        return CSPDarknet(**backbone_cfg)
    elif name == 'CSPRepResnet':
        return CSPRepResnet(**backbone_cfg)
    elif name == 'EfficientRep':
        return EfficientRep(**backbone_cfg)
