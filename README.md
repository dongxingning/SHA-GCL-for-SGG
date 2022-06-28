# Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation in Pytorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/dongxingning/SHA_GCL_for_SGG/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.6.0-%237732a8)

This repository contains the code for our paper [Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased
Scene Graph Generation](http://arxiv.org/abs/2203.09811), which has been accepted by CVPR 2022.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions, the recommended configuration is cuda-10.1 & pytorch-1.6.  

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing (VG & GQA).

## Pretrained Models

For VG dataset, the pretrained object detector we used is provided by [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), you can download it from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kxT9s3JwIpoGz4cA?e=usU6TR). For GQA dataset, we pretrained a new object detector, you can get it from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kxBfihou2smfXFV9?e=VtyoR7). However, we recommend you to pretrain a new one on GQA since we do not pretrain it for multiple times to choose the best pre-trained model for extracting offline region-level features.

## Perform training on Scene Graph Generation

### Set the dataset path

First, please refer to the ```SHA_GCL_extra/dataset_path.py``` and set the ```datasets_path``` to be your dataset path, and organize all the files like this:
```bash
datasets
  |-- vg
    |--detector_model
      |--pretrained_faster_rcnn
        |--model_final.pth
      |--GQA
        |--model_final_from_vg.pth       
    |--glove
      |--.... (glove files, will autoly download)
    |--VG_100K
      |--.... (images)
    |--VG-SGG-with-attri.h5 
    |--VG-SGG-dicts-with-attri.json
    |--image_data.json    
  |--gqa
    |--images
      |--.... (images)
    |--GQA_200_ID_Info.json
    |--GQA_200_Train.json
    |--GQA_200_Test.json
```

### Choose a dataset

You can choose the training/testing dataset by setting the following parameter:
``` bash
GLOBAL_SETTING.DATASET_CHOICE 'VG'  #['VG', 'GQA']
```

### Choose a task

To comprehensively evaluate the performance, we follow three conventional tasks: 1) **Predicate Classification (PredCls)** predicts the relationships of all the pairwise objects by employing the given ground-truth bounding boxes and classes; 2) **Scene Graph Classification (SGCls)** predicts the objects classes and their pairwise relationships by employing the given ground-truth object bounding boxes; and 3) **Scene Graph Detection (SGDet)** detects all the objects in an image, and predicts their bounding boxes, classes, and pairwise relationships.

For **Predicate Classification (PredCls)**, you need to set:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```
For **Scene Graph Classification (SGCls)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```
For **Scene Graph Detection (SGDet)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

### Choose your model

We abstract various SGG models to be different ```relation-head predictors``` in the file ```roi_heads/relation_head/roi_relation_predictors.py```, which are independent of the Faster R-CNN backbone and relation-head feature extractor. You can use ```GLOBAL_SETTING.RELATION_PREDICTOR``` to select one of them:

```bash
GLOBAL_SETTING.RELATION_PREDICTOR 'TransLike_GCL'
```

Notice the candidate choice is **"MotifsLikePredictor", "VCTreePredictor", "TransLikePredictor", "MotifsLike_GCL", "VCTree_GCL", "TransLike_GCL"**. The last three are with our GCL decoder.

The default settings are under ```configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml``` and ```maskrcnn_benchmark/config/defaults.py```. The priority is ```command > yaml > defaults.py```.

### Choose your Encoder (For "MotifsLike" and "TransLike")

You need to further choose an object/relation encoder for "MotifsLike" or "TransLike" predictor, by setting the following parameter:

```bash
GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention'
```

Notice the candidate choice is **'Self-Attention', 'Cross-Attention', 'Hybrid-Attention'** for TransLike Model, and **'Motifs', 'VTransE'** for MotifsLike Model.

### Choose one group split (for GCL only)

You can change the number of groups when using our GCL decoder, by setting the following parameter:

```bash
GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' # ['divide4', ''divide3', 'divide5', 'average']
```

For VG dataset, 'divide4' (5 groups), 'divide3' (6 groups), 'divide5' (4 groups) and average (5 groups). You can refer ```SHA_GCL_extra/get_your_own_group/get_group_splits.py``` to get your own group divisions.

### Choose the knowledge transfer method (for GCL only)

You can choose the knowledge transfer method by setting the following parameter:

```bash
GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' # ['None', 'KL_logit_Neighbor', 'KL_logit_TopDown', 'KL_logit_BottomUp', 'KL_logit_BiDirection']
```

### Examples of the Training Command
Training Example 1 : (VG, TransLike, Hybrid-Attention, divide4, Topdown, PredCls)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" GLOBAL_SETTING.DATASET_CHOICE 'VG' GLOBAL_SETTING.RELATION_PREDICTOR 'TransLike_GCL' GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention' GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 DTYPE "float16" SOLVER.MAX_ITER 60000 SOLVER.VAL_PERIOD 5000 SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR /home/share/datasets/vg/glove OUTPUT_DIR /home/share/datasets/output/SHA_GCL_VG_PredCls_test
```

Training Example 2 : (GQA_200, MotifsLike, Motifs, divide4, Topdown, SGCls)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" GLOBAL_SETTING.DATASET_CHOICE 'GQA_200' GLOBAL_SETTING.RELATION_PREDICTOR 'MotifsLike_GCL' GLOBAL_SETTING.BASIC_ENCODER 'Motifs' GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 60000 SOLVER.VAL_PERIOD 5000 SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR /home/share/datasets/vg/glove OUTPUT_DIR /home/share/datasets/output/Motifs_GCL_GQA_SGCls_test
```

## Evaluation

You can download our training model (SHA_GCL_VG_PredCls) from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kxI8NkjiMUWBRnWd?e=w5zuBh). You can evaluate it by running the following command.

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" GLOBAL_SETTING.DATASET_CHOICE 'VG' GLOBAL_SETTING.RELATION_PREDICTOR 'TransLike_GCL' GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention' GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE 'KL_logit_TopDown' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True TEST.IMS_PER_BATCH 8 DTYPE "float16" GLOVE_DIR /home/share/datasets/vg/glove OUTPUT_DIR /home/share/datasets/output/SHA_GCL_VG_PredCls_test
```

If you want to get more training models in our paper, please email me at ```dongxingning1998@gmail.com```.

## Citation
```bash
@inproceedings{dong2022stacked,
  title={Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation},
  author={Dong, Xingning and Gan, Tian and Song, Xuemeng and Wu, Jianlong and Cheng, Yuan and Nie, Liqiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19427--19436},
  year={2022}
}
```

We welcome you to commit issue or contact us (E-mail: ```dongxingning1998@gmail.com```) if you have any problem when reading the paper or reproducing the code.

## Acknowledgment

Our code is on top of [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), we sincerely thank them for their well-designed codebase.
