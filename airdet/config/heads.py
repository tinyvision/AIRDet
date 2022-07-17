# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

GFocalV2 = { "name" : "GFocalV2",
           "num_classes": 80,
           "decode_in_inference": True,
           "in_channels": [96, 160, 384],
           "stacked_convs": 4,
           "reg_channels": 64,
           "feat_channels": 96,
           "reg_max": 14,
           "add_mean": True,
           "norm": "bn",
           "act": "silu",
           "start_kernel_size": 3,
           "conv_groups": 2,
           "conv_type": "BaseConv",
           "octbase": 5,
           "l1_switch": "False",
           "use_ese": False,
         }


yolo_head = {"name": "YOLOX",
             "num_classes": 80,
             "decode_in_inference": True,
             "width": 0.5,
             "strides": [8, 16, 32],
             "in_channels": [256, 512, 1024],
             "act": "silu",
             "depthwise": False,
         }

