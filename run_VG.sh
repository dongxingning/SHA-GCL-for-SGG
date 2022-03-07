export PYTHONPATH=`pwd`:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml" GLOBAL_SETTING.DATASET_CHOICE 'VG' GLOBAL_SETTING.RELATION_PREDICTOR 'TransLike_GCL' GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 DTYPE "float16" SOLVER.MAX_ITER 60000 SOLVER.VAL_PERIOD 5000 SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR /home/share/dongxingning/datasets/vg/glove OUTPUT_DIR /home/share/dongxingning/datasets/newsggoutput/just_test
#python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_train_net.py \
#       --config-file "configs/SHA_GCL_e2e_relation_X_101_32_8_FPN_1x.yaml"   \
#       GLOBAL_SETTING.DATASET_CHOICE 'VG'\
#       MODEL.ROI_RELATION_HEAD.PREDICTOR 'DKSTransLike'\
#       GLOBAL_SETTING.BASIC_ENCODER 'Hybrid-Attention'\
#       MODEL.ROI_RELATION_HEAD.USE_GT_BOX True  \
#       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True   \
#       SOLVER.IMS_PER_BATCH 8   \
#       TEST.IMS_PER_BATCH 8   \
#       DTYPE "float16"  \
#       SOLVER.MAX_ITER 60000  \
#       SOLVER.VAL_PERIOD 5000   \
#       SOLVER.CHECKPOINT_PERIOD 5000  \
#       GLOVE_DIR /home/share/dongxingning/datasets/vg/glove   \
#       OUTPUT_DIR /home/share/dongxingning/datasets/newsggoutput/0306_SHAGCL_predcls_test