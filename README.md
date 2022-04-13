# Localization Distillation for Object Detection 

### English | [简体中文](README_zh-CN.md)

### This repo is based on MMRotate.

This is the code for our paper:
 - [Localization Distillation for Object Detection](https://arxiv.org/abs/2204.05957)
```
@Article{zheng2022rotatedLD,
  title={Localization Distillation for Object Detection},
  author= {Zheng, Zhaohui and Ye, Rongguang and Hou, Qibin and Ren, Dongwei and Wang, Ping and Zuo, Wangmeng and Cheng, Ming-Ming},
  journal={arXiv preprint arXiv:2204.05957},
  year={2022}
}
```

[2021.3.30] LD is officially included in [MMDetection V2](https://github.com/open-mmlab/mmdetection/tree/master/configs/ld), many thanks to [@jshilong
](https://github.com/jshilong), [@Johnson-Wang](https://github.com/Johnson-Wang) and [@ZwwWayne](https://github.com/ZwwWayne) for helping migrating the code.

LD is the extension of knowledge distillation on localization task, which utilizes the learned bbox distributions to transfer the localization dark knowledge from teacher to student.

LD stably improves over GFocalV1 about ~2.0 AP without adding any computational cost! 

## Introduction

Previous knowledge distillation (KD) methods for object detection mostly focus on feature imitation instead of mimicking the classification logits due to its inefficiency in distilling the localization information. 
In this paper, we investigate whether logit mimicking always lags behind feature imitation. 
Towards this goal, we first present a novel localization distillation (LD) method which can efficiently transfer the localization knowledge from the teacher to the student. 
Second, we introduce the concept of valuable localization region that can aid to selectively distill the classification and localization knowledge for a certain region. 
Combining these two new components, for the first time, we show that logit mimicking can outperform feature imitation and the absence of localization distillation is a critical reason for why logit mimicking underperforms for years. 
The thorough studies exhibit the great potential of logit mimicking that can significantly alleviate the localization ambiguity, learn robust feature representation, and ease the training difficulty in the early stage. 
We also provide the theoretical connection between the proposed LD and the classification KD, that they share the equivalent optimization effect. 
Our distillation scheme is simple as well as effective and can be easily applied to both dense horizontal object detectors and rotated object detectors. 
Extensive experiments on the MS COCO, PASCAL VOC, and DOTA benchmarks demonstrate that our method can achieve considerable AP improvement without any sacrifice on the inference speed.
<img src="LD.png" height="220" align="middle"/>


## Installation

Please refer to [INSTALL.md](docs/en/install.md) for installation and dataset preparation. Pytorch=1.5.1 and cudatoolkits=10.1 are recommended.

## Get Started

Please see [GETTING_STARTED.md](docs/en/getting_started.md) for the basic usage of MMDetection.

## Convert model

If you find trained model very large, please refer to [publish_model.py](tools/model_converters/publish_model.py)

```python
python tools/model_converters/publish_model.py your_model.pth your_new_model.pth
```

## Evaluation

###  DOTA-1.0
  Rotated-RetinaNet, LD + KD
  |     Teacher     |     Student     | Training schedule |    AP    |    AP50    |    AP70    |    AP90    |
  | :-------------: | :-------------: | :---------------: | :------: | :--------: | :--------: | :--------: |
  |       --        |      R-18       |        1x         |   33.7   |    58.0    |    42.3    |    4.7     |
  |      R-34       |      R-18       |        1x         |   39.1   |    63.8    |    48.8    |    8.8     |
  
  GWD, LD + KD
  |     Teacher     |     Student     | Training schedule |    AP    |    AP50    |    AP70    |    AP90    |
  | :-------------: | :-------------: | :---------------: | :------: | :--------: | :--------: | :--------: |
  |       --        |      R-18       |        1x         |   37.1   |    63.1    |    46.7    |    6.2     |
  |      R-34       |      R-18       |        1x         |   40.2   |    66.4    |    50.3    |    8.5     |
 
