# IterDet: Iterative Scheme for Object Detection in Crowded Environments

This project hosts the code for implementing the IterDet scheme for object detection,
as presented in our paper:

> **IterDet: Iterative Scheme for Object Detection in Crowded Environments**<br>
> [Danila Rukhovich](https://github.com/filaPro),
> [Konstantin Sofiiuk](https://github.com/ksofiyuk),
> [Danil Galeev](https://github.com/denemmy),
> [Olga Barinova](https://github.com/OlgaBarinova),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)
> <br>
> Samsung AI Center Moscow <br>
> https://arxiv.org/abs/20??.?????

<p align="center"><img src="./demo/iterative/scheme.png" alt="drawing" width="90%"/></p>

### Installation

This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection) framework.
All our modifications against their v1.2.0 release are listed below:
 
 * configs/baseline/*
 * configs/iterative/*
 * mmdet/datasets/pipelines/transforms.py
 * mmdet/datasets/pipelines/formating.py
 * mmdet/datasets/crowd_human.py
 * mmdet/models/backbones/resnet.py
 * mmdet/models/detectors/faster_rcnn.py
 * mmdet/models/detectors/retinanet.py
 * mmdet/models/detectors/single_stage.py
 * mmdet/models/detectors/two_stage.py
 * tools/convert_datasets/toy.py
 * tools/convert_datasets/wider_person.py
 * tools/convert_datasets/crowd_human.py
 * requirements/runtime.txt
 * demo/iterative/*

Please refer to original [INSTALL.md](docs/INSTALL.md) for installation.
Do not forget to update the original github repository link, and install [requirements.txt](requirements.txt).

[Config](configs/iterative) files and [tools](tools/convert_datasets) 
for converting annotations to COCO format are provided for the following datasets:

 * AdaptIS [ToyV1](https://github.com/saic-vul/adaptis#toyv1-dataset) 
   and [ToyV2](https://github.com/saic-vul/adaptis#toyv2-dataset)
 * [CrowdHuman](https://www.crowdhuman.org/)
 * [WiderPerson](http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/)
 
### Get Started

Please see original [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage examples.
[Baseline](configs/baseline) and [iterative](configs/iterative) configs
can be used for [train](tools/dist_train.sh) and [test](tools/dist_test.sh) scripts.

### Models

State-of-the-art models for all datasets are trained on top of Faster RCNN
based on ResNet-50. Metrics are given for 2 iterations IterDet inference.

| Dataset              | Download Link                                  | Recall | AP    | mMR   |
|:--------------------:|:----------------------------------------------:|:------:|:-----:|:-----:|
| AdaptIS Toy V1       | [toy_v1.pth][toy_v1]                           | 99.60  | 99.25 |       |
| AdaptIS Toy V2       | [toy_v1.pth][toy_v2]                           | 99.29  | 99.00 |       |
| CrowdHuman (full)    | [crowd_human_full.pth][crowd_human_full]       | 95.80  | 88.08 | 49.44 |
| CrowdHuman (visible) | [crowd_human_visible.pth][crowd_human_visible] | 91.63  | 85.33 | 55.61 |
| WiderPerson          | [wider_person.pth][wider_person]               | 97.15  | 91.95 | 40.78 |

[toy_v1]: ?
[toy_v2]: ?
[crowd_human_full]: ?
[crowd_human_visible]: ? 
[wider_person]: ?

#### Example Detections

<p align="center"><img src="./demo/iterative/demo.png" alt="drawing" width="90%"/></p>
