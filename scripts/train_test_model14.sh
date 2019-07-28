#!/usr/bin/env sh

GPU_ID=0
BATCH_SIZE=1
WORKER_NUMBER=1
LEARNING_RATE=0.001

CUDA_VISIBLE_DEVICES=$GPU_ID python models/transform-label-regularization/trainval_net.py \
                 --dataset pascal_voc --net alexnet \
                 --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                 --lr $LEARNING_RATE \
                 --cuda \
                 --lr_decay_step 60 \
    	 	 --epochs 90 \

CUDA_VISIBLE_DEVICES=$GPU_ID python models/transform-label-regularization/test_net.py \
                   --dataset pascal_voc --net alexnet \
                   --cuda \
		   --load_dir snapshots \
		   --checkweight model14_1_90_999.pth \
