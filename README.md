# Scene Graph Benchmark in Pytorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/dongxingning/SHA_GCL_for_SGG/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.6.0-%237732a8)

This repository contains the code for our paper [Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased
Scene Graph Generation](), which has been accepted by CVPR 2022.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions, the recommended configuration is cuda-10.1 & pytorch-1.6.  

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing (VG & GQA).

## Pretrained Models

For VG dataset, the pretrained object detector we used is provided by [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), you can download it from [this link](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ). For GQA dataset, we pretrained a new object detector, you can get it from [this link](). However, we recommend you to pretrain a new one on GQA since we do not pretrain it for multiple times to choose the best model.

## Perform training on Scene Graph Generation

### Choose a dataset

You can choose the training/testing dataset by setting the following parameter:
``` bash
GLOBAL_SETTING.DATASET_CHOICE 'VG'  #['VG', 'GQA']
```

### Choose a task

To comprehensively evaluate the performance, we follow three conventional tasks: 1) **Predicate Classification (PredCls)** predicts the relationships of all the pairwise objects by employing the given ground-truth bounding boxes and classes; 2) **Scene Graph Classification (SGCls)** predicts the objects classes and their pairwise relationships by employing the given ground-truth object bounding boxes; and 3) **Scene Graph Detection (SGDet)** detects all the objects in an image, and predicts their bounding boxes, classes and pairwise relationships.

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

You need to further choose a object/relation encoder for "MotifsLike" or "TransLike" predictor, by setting the following parameter:

```bash
GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention'
```

Notice the candidate choice is **'Self-Attention', 'Cross-Attention', 'Hybrid-Attention'** for TransLike Model, and **'Motifs', 'VTransE'** for MotifsLike Model.

### Choose one Group Split (For GCL only)

You can change the number of groups when using our GCL decoder, by setting the following parameter:

```bash
GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE 'divide4' # ['divide4', ''divide3', 'divide5', 'average']
```

For VG dataset, 'divide4' (5 groups), 'divide3' (6 groups), 'divide5' (4 groups) and average (5 groups). You can refer ```SHA_GCL_extra/get_your_own_group/get_group_splits.py``` to get your own group divisions.

### Examples of the Training Command
Training Example 1 : (PreCls, Motif Model)
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp
```
where ```GLOVE_DIR``` is the directory used to save glove initializations, ```MODEL.PRETRAINED_DETECTOR_CKPT``` is the pretrained Faster R-CNN model you want to load, ```OUTPUT_DIR``` is the output directory used to save checkpoints and the log. Since we use the ```WarmupReduceLROnPlateau``` as the learning scheduler for SGG, ```SOLVER.STEPS``` is not required anymore.

Training Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10026 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgcls-exmp
```


## Evaluation

### Examples of the Test Command
Test Example 1 : (PreCls, Motif Model)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/motif-precls-exmp OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp
```

Test Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10028 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgcls-exmp OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgcls-exmp
```

## Other Options that May Improve the SGG

- For some models (not all), turning on or turning off ```MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS``` will affect the performance of predicate prediction, e.g., turning it off will improve VCTree PredCls but not the corresponding SGCls and SGGen. For the reported results of VCTree, we simply turn it on for all three protocols like other models.

- For some models (not all), a crazy fusion proposed by [Learning to Count Object](https://arxiv.org/abs/1802.05766) will significantly improves the results, which looks like ```f(x1, x2) = ReLU(x1 + x2) - (x1 - x2)**2```. It can be used to combine the subject and object features in ```roi_heads/relation_head/roi_relation_predictors.py```. For now, most of our model just concatenate them as ```torch.cat((head_rep, tail_rep), dim=-1)```.

- Not to mention the hidden dimensions in the models, e.g., ```MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM```. Due to the limited time, we didn't fully explore all the settings in this project, I won't be surprised if you improve our results by simply changing one of our hyper-parameters

## Citation


## Acknowledgment

Our code is on top of [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), we sincerely thank them for their well-designed codebase.
