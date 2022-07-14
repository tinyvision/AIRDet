#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from airdet.config import Config as MyConfig

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.miscs.eval_interval_epochs = 10

        neck2head_ch = [128, 256, 512]
        GiraffeNeck = {"name": "GiraffeNeck",
                "min_level": 3,
                "max_level": 5,
                "num_levels": 3,
                "norm_layer": None,
                "norm_kwargs": dict(eps=.001, momentum=.01),
                "act_type": "silu",
                "fpn_config": None,
                "fpn_name": "giraffeneck",
                "fpn_channels": neck2head_ch,
                "out_fpn_channels": neck2head_ch,
                "weight_method": "concat",
                "depth_multiplier": 3,
                "width_multiplier": 1.0,
                "with_backslash": True,
                "with_slash": True,
                "with_skip_connect": True,
                "skip_connect_type": "log2n",
                "separable_conv": False,
                "feature_info": [dict(num_chs=128, reduction=8), dict(num_chs=256, reduction=16), dict(num_chs=512, reduction=32)],
                "merge_type": "reparam_csp",
                "pad_type": '',
                "downsample_type": "max",
                "upsample_type": "nearest",
                "apply_resample_bn": True,
                "conv_after_downsample": False,
                "redundant_bias": False,
                "conv_bn_relu_pattern": False,
                "alternate_init": False
}


        GFocalV2 = { "name" : "GFocalV2",
           "num_classes": 80,
           "decode_in_inference": True,
           "in_channels": neck2head_ch,
           "stacked_convs": 4,
           "reg_channels": 64,
           "feat_channels": 96,
           "reg_max": 14,
           "add_mean": True,
           "norm": "bn",
           "act": "silu",
           "start_kernel_size": 3,
           "conv_groups": 1,
           "conv_type": "BaseConv",
           "octbase": 5,
           "l1_switch": "False",
         }


        self.model.neck = GiraffeNeck
        self.model.head = GFocalV2


