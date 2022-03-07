# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from maskrcnn_benchmark.data.datasets.evaluation.vg.vg_eval import do_vg_evaluation_directly

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main(output_folder):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    iou_types = ("bbox",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )

    data_loaders_val = make_data_loader(cfg, mode="test", is_distributed=distributed)

    for data_loader_val in data_loaders_val:
        print('\n\nwarning!\nwarning!\nwarning!\nwe generate results without testing!\n\n')
        # convert to a torch.device for efficiency

        dataset = data_loader_val.dataset

        predictions = torch.load(os.path.join(output_folder, "eval_results.pytorch"), map_location=torch.device("cpu"))['predictions']
        synchronize()

        return do_vg_evaluation_directly(cfg=cfg,
                                         dataset=dataset,
                                         predictions=predictions,
                                         iou_types=iou_types)


if __name__ == "__main__":
    output_folder = 'G:/myprogram/winscp_save/sggtest/0824_drn_trans_Topdown'
    main(output_folder)
