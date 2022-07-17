# Copyright (C) Alibaba Group Holding Limited. All rights reserved.


import copy

from .giraffe_fpn import GiraffeNeck
from .pafpn import PAFPN
from .csp_pan import CustomCSPPAN


def build_neck(cfg):
    neck_cfg = copy.deepcopy(cfg)
    name = neck_cfg.pop("name")
    if name == "PAFPN":
        return PAFPN(**neck_cfg)
    elif name == "GiraffeNeck":
        return GiraffeNeck(**neck_cfg)
    elif name == 'CSPPAN':
        return CustomCSPPAN(**neck_cfg)
