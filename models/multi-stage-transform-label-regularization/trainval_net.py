# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import copy
import itertools

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.randcrop_roibatchLoader import randcrop_roibatchLoader
from utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from fast_rcnn.vgg16 import vgg16
from fast_rcnn.resnet import resnet
from fast_rcnn.alexnet import alexnet


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101, alexnet',
                        default='vgg16', type=str)
    parser.add_argument('--imdb_name', dest='imdb_name',
                        help='train imdb name',
                        default='vocweak_2007_trainval/trainvalmid', type=str)
    parser.add_argument('--imdbval_name', dest='imdbval_name',
                        help='validation imdb name',
                        default='vocweak_2007_test', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=30, type=int)
    parser.add_argument('--save_epoch', dest='save_epoch',
                        help='step of epochs to save',
                        default=5, type=int)
    parser.add_argument('--model_id', dest='model_id',
                        help='model id to save',
                        default=13, type=int)

    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="snapshots",
                        type=str)
    parser.add_argument('--pretrained_weight', dest='pretrained_weight',
                        help='pretrained weight', default="",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

# config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=20, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

# set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

# resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
# log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')

    args = parser.parse_args()
    return args


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


class sampler(Sampler):
    def __init__(self, strong_inds, weak_inds):
        self.strong_inds = strong_inds
        self.weak_inds = weak_inds

    def __iter__(self):
        strong_iter = iterate_once(self.strong_inds)
        weak_iter = iterate_eternally(self.weak_inds)
        return (
            [s_ind, w_ind]
            for (s_ind, w_ind) in zip(strong_iter, weak_iter)
        )

        return iter(self.rand_num_view)

    def __len__(self):
        return len(self.strong_inds)


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == "pascal_voc":
        # args.imdb_name = "vocweak_2007_trainval/trainvalmid"
        # args.imdbval_name = "vocweak_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)

    print('{:d} roidb entries'.format(len(roidb)))

    weak_inds = []
    strong_inds = []
    for i in range(len(roidb)):
        if roidb[i]['boxes'].shape[0] > 0:
            strong_inds.append(i)

    for i in range(len(roidb)):
        if roidb[i]['boxes'].shape[0] == 0:
            weak_inds.append(i)

    train_size = len(strong_inds)
    print('{:d} strong entries and {:d} weak entries'.format(len(strong_inds), len(weak_inds)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_sampler = sampler(strong_inds, weak_inds)

    dataset = randcrop_roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                                      imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    wnum_boxes = torch.LongTensor(1)
    wgt_boxes = torch.FloatTensor(1)
    rois = torch.FloatTensor(1)
    image_classes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        wnum_boxes = wnum_boxes.cuda()
        wgt_boxes = wgt_boxes.cuda()
        rois = rois.cuda()
        image_classes = image_classes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    wnum_boxes = Variable(wnum_boxes)
    wgt_boxes = Variable(wgt_boxes)
    rois = Variable(rois)
    image_classes = Variable(image_classes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'vgg16':
        fastRCNN = vgg16(imdb.classes, pretrained=True,
                         class_agnostic=args.class_agnostic,
                         pretrained_weight=args.pretrained_weight)
    elif args.net == 'res101':
        fastRCNN = resnet(imdb.classes, 101, pretrained=True,
                          class_agnostic=args.class_agnostic,
                          pretrained_weight=args.pretrained_weight)
    elif args.net == 'alexnet':
        fastRCNN = alexnet(imdb.classes, pretrained=True,
                           class_agnostic=args.class_agnostic,
                           pretrained_weight=args.pretrained_weight)
    else:
        print("network is not defined")
        pdb.set_trace()

    fastRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fastRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr':lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr':lr,
                            'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fastRCNN.cuda()

    if args.resume:
        load_name = os.path.join(output_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fastRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fastRCNN = nn.DataParallel(fastRCNN)

    iters_per_epoch = int(train_size / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fastRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)

            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])
            wgt_boxes.data.resize_(data[4].size()).copy_(data[4])
            wnum_boxes.data.resize_(data[5].size()).copy_(data[5])
            rois.data.resize_(data[6].size()).copy_(data[6])
            image_classes.data.resize_(data[7].size()).copy_(data[7])

            fastRCNN.zero_grad()

            rcnn_out_1, rcnn_out_2, rcnn_out_3, rcnn_out_4, rcnn_out_5 = fastRCNN(
                im_data, im_info, gt_boxes, num_boxes, wgt_boxes, wnum_boxes, rois, image_classes)

            out_rois_1, cls_prob_1, bbox_pred_1, RCNN_loss_cls_1, RCNN_loss_bbox_1, rois_label_1, image_loss_1 = rcnn_out_1
            # out_rois_2, cls_prob_2, bbox_pred_2, RCNN_loss_cls_2, RCNN_loss_bbox_2, rois_label_2, image_loss_2 = rcnn_out_2
            out_rois_3, cls_prob_3, bbox_pred_3, RCNN_loss_cls_3, RCNN_loss_bbox_3, rois_label_3, image_loss_3 = rcnn_out_3
            out_rois_4, cls_prob_4, bbox_pred_4, RCNN_loss_cls_4, RCNN_loss_bbox_4, rois_label_4, image_loss_4 = rcnn_out_4
            out_rois_5, cls_prob_5, bbox_pred_5, RCNN_loss_cls_5, RCNN_loss_bbox_5, rois_label_5, image_loss_5 = rcnn_out_5

            loss = RCNN_loss_cls_1.mean() + RCNN_loss_bbox_1.mean()
            # loss += RCNN_loss_cls_2.mean() + RCNN_loss_bbox_2.mean()
            loss += RCNN_loss_cls_3.mean() + RCNN_loss_bbox_3.mean()
            loss += RCNN_loss_cls_4.mean() + RCNN_loss_bbox_4.mean()
            loss += RCNN_loss_cls_5.mean() + RCNN_loss_bbox_5.mean()
            loss += image_loss_1.mean() + image_loss_5.mean()

            RCNN_loss_cls = RCNN_loss_cls_1
            RCNN_loss_bbox = RCNN_loss_bbox_1
            rois_label = rois_label_1

            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16" or args.net == 'alexnet':
                clip_gradient(fastRCNN, 10.)

            # if args.net == 'res101':
            #    clip_gradient(fastRCNN, 10.)

            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    loss_image = image_loss_3.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    loss_image = image_loss_3.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e"
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" %
                      (fg_cnt, bg_cnt, end - start))
                print("\t\t\trcnn_cls: %.4f, rcnn_box %.4f"
                      % (loss_rcnn_cls, loss_rcnn_box))
                print("\t\t\timage_cls: %.4f"
                      % (loss_image))

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars(
                        "logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

        if epoch % args.save_epoch == 0:
            save_name = os.path.join(
                output_dir, 'model{}_{}_{}_{}.pth'.format(args.model_id, args.session, epoch, step))

            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': fastRCNN.module.state_dict() if args.mGPUs else fastRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
            print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()
