#!/usr/bin/env sh

GPU_ID=0
BATCH_SIZE=1
WORKER_NUMBER=1
LEARNING_RATE=0.001

TRAIN=0
TEST=1

if  [ $TRAIN = 1 ]; then
	CUDA_VISIBLE_DEVICES=$GPU_ID python models/multi-stage-transform-label-regularization/trainval_net.py \
                   --dataset pascal_voc --net vgg16 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE \
                   --cuda \
                   --lr_decay_step 40 \
                   --epochs 60 \
		   --model_id 1 \
                   --pretrained_weight data/pretrained_model/vgg16_caffe.pth
fi

if [ $TEST = 1 ]; then
	CUDA_VISIBLE_DEVICES=$GPU_ID python models/multi-stage-transform-label-regularization/test_net.py \
                   --dataset pascal_voc --net vgg16 \
                   --cuda \
		   --load_dir snapshots \
		   --checkweight model1_1_60_999.pth
fi

