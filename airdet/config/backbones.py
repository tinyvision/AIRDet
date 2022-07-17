# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

CSPDarknet = {"name": "CSPDarknet",
            "dep_mul": 0.33,
            "wid_mul": 0.5,
            "out_features": ("dark3", "dark4", "dark5"),
            "depthwise": False,
            "act": "silu",
            }

CSPRepResnet = {"name": "CSPRepResnet",
                "depth_mult": 0.33,
                "width_mult": 0.5,
                "return_idx": [1,2,3],
                "use_large_stem": True,
                "layers": [3, 6, 6, 3],
                "channels": [64, 128, 256, 512, 1024],
                "act": "relu",
                }

