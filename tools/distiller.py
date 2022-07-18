#!/usr/bin/env python
# coding=utf-8
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import os
import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from airdet.apis.detector_distiller import Distiller
from airdet.config.base import parse_config
from airdet.utils import get_num_devices, synchronize


def make_parser():

    parser = argparse.ArgumentParser("Distiller for AIRDet")

    parser.add_argument(
        "--tea_config",
        default=None,
        type=str,
    )
    parser.add_argument("--tea_ckpt", default=None, type=str, help="teacher checkpoint file")
    parser.add_argument(
        "--stu_config",
        default=None,
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("-d", "--distiller", default="mimic", type=str, help="support [mimic, mgd]")
    parser.add_argument("--loss_weight", default=1.0, type=float, help="distill loss weight")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

    # student
    stu_config = parse_config(args.stu_config)
    stu_config.merge(args.opts)

    # teacher
    tea_config = parse_config(args.tea_config)

    trainer = Distiller(stu_config, tea_config, args)
    trainer.train(args.local_rank)


if __name__ == "__main__":
    main()
