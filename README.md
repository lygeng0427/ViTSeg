# Applying ViT in Generalized Few-shot Semantic Segmentation
CSCI-GA 2271: Computer Vision - Final Project (Tutored by Prof. Rob Fergus)

Group Members: Liyuan Geng, Jinhong Xia, Yuanhe Guo

This repo is forked from [DIaM](https://github.com/sinahmr/DIaM). We used this as a starting point to implement our approach.

## Overview
### Framework
![framework](presentation/framework.png)

> **Abstract:** This paper explores the capability of ViT-based models under the generalized few-shot semantic segmentation (GFSS) framework. We conducte experiments with various combinations of backbone models, including ResNets and pretrained Vision Transformer (ViT)-based models, along with decoders featuring a linear classifier, UPerNet, and Mask Transformer. The structure made of DINOv2 and linear classifier takes the lead on popular few-shot segmentation bench mark COCO-$20^i$, substantially outperforming the best of ResNet structure by $116\%$ in one-shot scenario. We demonstrate the great potential of large pretrained ViT-based model on GFSS task, and expect further improvement on testing benchmarks.

## Getting Started

### 1.Requirements
We used `Python 3.9` in our experiments and the list of packages is available in the `requirements.txt` file. You can install them using `pip install -r requirements.txt`.

### 2.Download data

Here is the structure of the data folder:

```
data
└── pascal
|   ├── JPEGImages
|   └── SegmentationClassAug
```
**PASCAL**: The JPEG images can be found in the PASCAL-VOC 2012 toolkit to be downloaded at [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [SegmentationClassAug](https://etsmtl365-my.sharepoint.com/:u:/g/personal/seyed-mohammadsina_hajimiri_1_ens_etsmtl_ca/Ef70aWKWEidJoR_NZb131SwB3t7WIHMjJK316qxIu_SPyw?e=CVtNKY) (pre-processed ground-truth masks).

#### About the train/val splits

The train/val splits are directly provided in `lists/`. How they were obtained is explained at https://github.com/Jia-Research-Lab/PFENet.

### 3.Download pre-trained models

#### Pre-trained backbone and models
We provide the pre-trained backbone and models at https://drive.google.com/file/d/1WuKaJbj3Y3QMq4yw_Tyec-KyTchjSVUG/view?usp=share_link. You can download them and directly extract them at the root of this repo. This will create two folders: `initmodel/` and `model_ckpt/`.

## Repo Structure

Default configuration files can be found in `config/`. Data are located in `data/`. `lists/` contains the train/val splits for each dataset. All the codes are provided in `src/`. Testing script is located at the root of the repo.

## Testing

To test the model, use the `test.sh` script, which its general syntax is:
```bash
bash test.sh {benchmark} {shot} {pi_estimation_strategy} {[gpu_ids]} {log_path}
```
This script tests successively on all folds of the benchmark and reports the results individually. The overall performance is the average over all the folds. Some example commands are presented below, with their description in the comments.

```bash
bash test.sh pascal5i 1 self [0] out.log  # PASCAL-5i benchmark, 1-shot, estimate pi by model's output
bash test.sh pascal10i 5 self [0] out.log  # PASCAL-10i benchmark, 5-shot, estimate pi by model's output
```

