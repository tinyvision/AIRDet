# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

import argparse
import os

import torch
from loguru import logger


def init_engine(engine, device):
    import tensorrt as trt
    from collections import namedtuple,OrderedDict
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        print(name, dtype)
        shape = tuple(model.get_binding_shape(index))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()
    return context, bindings, binding_addrs, model.get_binding_shape(0)[0]

@logger.catch
def trt_speed(trt_path, batch_size, h, w, config):

    device = 'cuda'
    context, bindings, binding_addrs, trt_batch_size = init_engine(trt_path, device='cuda')
    tmp = torch.randn(batch_size, 3, h, w).to(device)
    # warm up for 10 times
    for _ in range(10):
        binding_addrs['images_arrays'] = int(tmp.data_ptr())
        context.execute_v2(list(binding_addrs.values()))

    imgs = torch.randn(batch_size, 3, h, w)
    imgs = imgs.to(device, non_blocking=True)
    # preprocess
    imgs = imgs.float()
    latency = 0
    for _ in range(100):
        # inference
        t0 = time.time()
        binding_addrs['image_arrays'] = int(imgs.data_ptr())
        context.execute_v2(list(binding_addrs.values()))
        preds = bindings['output'].data
        latency += (time.time() - t0)  # inference time

    logger.info("Model inference time {:.4f}ms / img per device".format(latency / 100 / batch_size * 1000))

