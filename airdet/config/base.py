#!/usr/bin/env python3
#-*- coding:utf-8 -*-
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import pprint
import sys
import os
import importlib

from .backbones import CSPDarknet, MobileNet, ShuffleNet
from .necks import GiraffeNeck, PAFPNNeck
from .heads import GFocalV2, yolo_head
from .augmentations import tta, strong_autoaug
from .paths_catalog import DatasetCatalog

from abc import ABCMeta
from tabulate import tabulate
from loguru import logger
from easydict import EasyDict as easydict
from os.path import join, dirname

miscs = easydict({
         "print_interval_iters": 50,
         "output_dir": './workdirs',
         "seed": 1234,
         "eval_interval_epochs": 5,
         "ckpt_interval_epochs": 5,
        })

deploy = easydict({
         "onnx_postprocess": False,
         "onnx_inference_h": 640,
         "onnx_inference_w": 640,
        })

model = easydict({
         "backbone": CSPDarknet,
         "neck": GiraffeNeck,
         "head": GFocalV2,
         })

yolox_model = easydict({
         "backbone": CSPDarknet,
         "neck": PAFPNNeck,
         "head": yolo_head,
         })

airdet_Mobile = easydict({
         "backbone": MobileNet,
         "neck": GiraffeNeck,
         "head": GFocalV2,
         })

airdet_Shuffle = easydict({
         "backbone": ShuffleNet,
         "neck": GiraffeNeck,
         "head": GFocalV2,
         })

training = easydict({
         "fp16": False,
         "ema": False,
         "ema_momentum": 0.9998,
         "use_syncBN": True,
         # optimizer
         "warmup_lr": 0,
         "base_lr_per_img": 0.01/64,
         "momentum": 0.9,
         "weight_decay": 5e-4,
         # scheduler
         "lr_scheduler": 'cosine',
         "min_lr_ratio": 0.05,

         "images_per_batch": 32,
         "start_epochs": 0,
         "total_epochs": 300,
         "warmup_epochs": 5,
         "no_aug_epochs": 16,
         "pretrain_model": None,
         "resume_path": None,
         "augmentation": strong_autoaug,
         "input_min_size_range": (448, 832),
         "input_min_size": (640,),
         "input_max_size": 640,
         })


testing = easydict({
            "use_tta": False,
            "augmentation": tta,
            "conf_threshold": 0.05,
            "nms_iou_threshold": 0.7,
            "multi_gpu": True,
            "input_min_size": (640,),
            "input_max_size": 640,
            "images_per_batch": 32,
          })

dataset = easydict({
         "paths_catalog": join(dirname(__file__), "paths_catalog.py"),
         "train_ann": ("coco_2017_train",),
         "val_ann": ("coco_2017_val",),
         "data_dir": None,
         "data_list": {},
         "class2id": {},
         "num_workers": 4,
         "data_test": (),
         "aspect_ratio_grouping": False,
         "multiscale_range": 5,
         "size_divisibility": 32,
         "input_pixel_mean": [0.0, 0.0, 0.0],
         "input_pixel_std": [1.0, 1.0, 1.0],
         "input_to_bgr255": False,
         })


class Config(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        self.model = model
        self.training = training
        self.testing = testing
        self.dataset = dataset
        self.miscs = miscs
        self.deploy = deploy
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_data(self, name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=join(data_dir, attrs["img_dir"]),
                ann_file=join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        elif "custom" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
		CLASS2ID=self.dataset.class2id,
            )
            return dict(
                factory="CustomVOCDataset",
                args=args,
            )
		
			

        return None
        # raise RuntimeError("Dataset not available: {}".format(name))

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v, compact=True))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)


def get_config_by_file(config_file):
    try:
        sys.path.append(os.path.dirname(config_file))
        current_config = importlib.import_module(os.path.basename(config_file).split(".")[0])
        exp = current_config.Config()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Config'".format(config_file))
    return exp


def parse_config(config_file):
    """
    get config object by file.
    Args:
        config_file (str): file path of config.
    """
    assert (
        config_file is not None
    ), "plz provide config file"
    if config_file is not None:
        return get_config_by_file(config_file)
