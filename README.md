English | [简体中文](README_cn.md)
<div align="center"><img src="assets/airdet.png" width="500"></div>

## Introduction
Welcome to AIRDet! 
AIRDet is an efficiency-oriented anchor-free object detector, aims to enable robust object detection in various industry scene. With simple design, **AIRDet-s outperforms series competitor e.g.(YOLOX-s, MT-YOLOv6-s, PP-YOLOE-s), and still maintains fast speed.** Moreover, here you can find not only powerful models, but also highly efficient training strategies and complete tools from training to deployment.  

## Updates
-  **[2022/06/23: We release  AIRDet-0.0.1!]**
    * Release AIRDet-series object detection models, e.g. AIRDet-s and AIRDet-m. AIRDet-s achievs mAP as 44.2% on COCO val dataset and 2.8ms latency on Nvidia-V100. AIRDet-m is a larger model build upon AIRDet-s in a heavy neck paradigm, which achieves robust improvement in detection of different object scales. For more information, please refer to [Giraffe-neck](https://arxiv.org/abs/2202.04256).
    * Release model convert tools for esay deployment, surppots onnx and tensorRT-fp32, TensorRT-fp16.

## Comming soon
- High efficient backbone.
- AIRDet-tiny and AIRDet-nano.
- Model distillation. 

## Model Zoo
|Model |size |mAP<sup>val<br>0.5:0.95 | Latency V100<br>TRT-FP32-BS32| Latency V100<br>TRT-FP16-BS32| FLOPs<br>(G)| weights |
| ------        |:---: | :---:     |:---:|:---: | :---: | :----: |
|[Yolox-s](./configs/yolox_s.py)   | 640 | 40.5 | 3.4 | 2.3 | 26.81 | [link]() |
|[AIRDet-s](./configs/airdet_s.py) | 640 | 44.2 | 4.4 | 2.8 | 27.56 | [link](https://drive.google.com/file/d/119W87oZ4zcJvvjzYCmBudX38cRpZbQc4/view?usp=sharing) |
|[AIRDet-m](./configs/airdet_m.py) | 640 | 48.2 | 8.3 | 4.4 | 76.61 | [link](https://drive.google.com/file/d/1EjsdQTbUF4JMzH6wxXcYI6zs88c--PzL/view?usp=sharing) |


- We report the mAP of models on COCO2017 validation set.
- The latency in this table are measured without post-processing.

## Quick Start

<details>
<summary>Installation</summary>

Step1. Install AIRDet.
```shell
git clone https://github.com/tinyvision/AIRDet.git
cd AIRDet/
conda create -n AIRDet python=3.7 -y
conda activate AIRDet
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```
Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; 
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained model from the benchmark table, e.g. airdet-s.

Step2. Use -f(config filename) to specify your detector's config. For example:
```shell
python tools/demo.py -f configs/airdet_s.py --ckpt /path/to/your/airdet_s.pth --path assets/dog.jpg
```
</details>

<details>
<summary>Reproduce our results on COCO</summary>

Step1. Prepare COCO dataset
```shell
cd <AIRDet Home>
ln -s /path/to/your/coco ./datasets/coco
```

Step 2. Reproduce our results on COCO by specifying -f(config filename)
```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/airdet_s.py
```
</details>

<details>
<summary>Evaluation</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/eval.py -f configs/airdet_s.py --ckpt /path/to/your/airdet_s_ckpt.pth
```
</details>

<details>
<summary> Training on Custom Data </summary>
Airdet supports COCO and VOC format. Before training, you need to transform your data into COCO or VOC format. We provide
the usage of COCO format in default config. If you are trying to use VOC format, here is a breif example.

Step.1 Transform your own dataset into VOC format. the directory structure should be as follow:

```shell script
Bus/
    Annotations/
        *.xml
    JPEGImages/
        *.jpg,png,PNG
    ImageSets/
        Main/
            train.txt
            test.txt
            val.txt
```

Step.2 Write the corresponding dataset name and Train/Eval dataset path, the dataset name should be like [xxx_custom_train/val].
```shell script
self.dataset.train_ann = ("bus_custom_train",)
self.dataset.val_ann = ("bus_custom_val")
self.dataset.data_dir = 'datasets'
self.dataset.data_list = {
    "bus_custom_train": {
        "data_dir": "Bus/",
        "split": "train"
    },
    "bus_custom_val": {
        "data_dir": "Bus/",
        "split": "val"
    }, 
}

self.dataset.class2id = {
    'class_name1': 1,
    'class_name2': 2,
    'class_name3': 3,
}
self.model.head.num_classes = len(self.dataset.class2id.keys())
```

Step.3 Put your dataset under `$AIRDet/datasets`.
```shell script
ln -s /path/to/your/Bus/ ./datasets/Bus/
```

Step.4 Create your config file to control everything, including model setting, training setting, and test setting, e.g. bus_s.py.  
```shell script
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/bus_s.py
```
</details>

## Deploy

<details>
<summary>Installation</summary>

Step1. Install ONNX.
```shell
pip install onnx==1.8.1
pip install onnxruntime==1.8.0
pip install onnx-simplifier==0.3.5
```
Step2. Install CUDA、CuDNN、TensorRT and pyCUDA
2.1 CUDA
```shell
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
source ~/.bashrc
```
2.2 CuDNN
```shell
sudo cp cuda/include/* /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```
2.3 TensorRT
```shell
cd TensorRT-7.2.1.6/python
pip install tensorrt-7.2.1.6-cp37-none-linux_x86_64.whl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-7.2.1.6/lib
```
2.4 pycuda
```shell
pip install pycuda==2022.1
```
</details>

Step.1 convert torch model to onnx or trt engine, and the output file would be generated in deploy/. Note the convert mode has three options:[onnx, trt_32, trt_16].
```shell script
python tools/converter.py --output-name deploy/airdet_s.onnx -f configs/airdet_s.py -c airdet_s.pth --batch_size 1 --img_size 640 --mode trt_32
```

Step.2 trt engine evaluation and inference speed computation and appoint trt engine by --trt.
```shell script
python -m torch.distributed.launch --nproc_per_node=1 tools/trt_eval.py -f configs/airdet_s.py --trt deploy/airdet_s_32.trt --batch_size 1 --img_size 640
```

Step.3 trt engine inference demo and appoint test image by -p.
```shell script
python tools/trt_inference.py -f configs/airdet_s.py -t deploy/airdet_s_32.trt -p assets/dog.jpg --img_size 640 --nms 0.7
```

## Distillation
AIRDet series support the dark knowledge transfer through distillation to improve performance especially for tiny models. We support mimicking and mask generative on feature distillation, and appoint distiller by --d. Note the distiller mode has two options:[mimic, mgd].

```shell script
python -m torch.distributed.launch --nproc_per_node=8 tools/distiller.py --tea_config configs/airdet_m.py --stu_config configs/airdet_s.py --tea_ckpt airdet_m.pth --d mimic --loss_weight 0.1
```