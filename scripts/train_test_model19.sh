#!/usr/bin/env sh

GPU_ID=2
BATCH_SIZE=1
WORKER_NUMBER=1
LEARNING_RATE=0.001

#CUDA_VISIBLE_DEVICES=$GPU_ID python models/multi-stage-transform-label-regularization/trainval_net.py \
#                   --dataset pascal_voc --net vgg16 \
#                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
#                   --lr $LEARNING_RATE \
#                   --cuda \
#                   --lr_decay_step 80 \
#                   --epochs 120 \
#		   --model_id 19 \
#                   --pretrained_weight data/pretrained_model/vgg16_voc07trainvalweak_mxnet.pth \
#		   --imdb_name vocweak_2007_trainval/trainval4shot \
#                   --save_epoch 20 \
#		   --disp_interval 60 \

CUDA_VISIBLE_DEVICES=$GPU_ID python models/multi-stage-transform-label-regularization/test_net.py \
                   --dataset pascal_voc --net vgg16 \
                   --cuda \
		   --load_dir snapshots \
		   --checkweight model19_1_120_119.pth \
