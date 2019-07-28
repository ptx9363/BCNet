#!/usr/bin/env sh

GPU_ID=4
BATCH_SIZE=1
WORKER_NUMBER=1
LEARNING_RATE=0.001

CUDA_VISIBLE_DEVICES=$GPU_ID python models/faster-rcnn/trainval_net.py \
                   --dataset pascal_voc --net alexnet \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE \
                   --cuda \
