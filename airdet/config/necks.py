# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

PAFPNNeck = {"name": "PAFPN",
        "depth": 0.33,
        "width": 0.5,
        "in_features": [2, 3, 4],
        "in_channels": [256,512,1024],
        "depthwise": False,
        "act": "silu",
        }


GiraffeNeck = {"name": "GiraffeNeck",
                "min_level": 3,
                "max_level": 5,
                "num_levels": 3,
                "norm_layer": None,
                "norm_kwargs": dict(eps=.001, momentum=.01),
                "act_type": "silu",
                "fpn_config": None,
                "fpn_name": "giraffeneck",
                "fpn_channels": [96, 160, 384],
                "out_fpn_channels": [96, 160, 384],
                "weight_method": "concat",
                "depth_multiplier": 2,
                "width_multiplier": 1.0,
                "with_backslash": True,
                "with_slash": True,
                "with_skip_connect": True,
                "skip_connect_type": "log2n",
                "separable_conv": False,
                "feature_info": [dict(num_chs=128, reduction=8), dict(num_chs=256, reduction=16), dict(num_chs=512, reduction=32)],
                "merge_type": "csp",
                "pad_type": '',
                "downsample_type": "max",
                "upsample_type": "nearest",
                "apply_resample_bn": True,
                "conv_after_downsample": False,
                "redundant_bias": False,
                "conv_bn_relu_pattern": False,
                "alternate_init": False
}

CSPPAN = {"name": "CSPPAN",
          "width_mult": 0.50,
          "depth_mult": 0.33,
          "act": "relu",
          "out_channels": [768, 384, 192],
          "spp": True,
          }

