# --------------------------------------------------------
# Tensorflow Faster R-CNN
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
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from rpn.bbox_transform import clip_boxes
from nms.nms_wrapper import nms
from rpn.bbox_transform import bbox_transform_inv
from utils.net_utils import save_net, load_net, vis_detections
from fast_rcnn.vgg16 import vgg16
from fast_rcnn.resnet import resnet
from fast_rcnn.alexnet import alexnet

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checkweight', dest='checkweight',
                        help='check weight file name',
                        type=str)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "vocweak_2007_test/test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES',
                         '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES',
                         '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES',
                         '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES',
                         '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    args.cfg_file = "cfgs/{}_test_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}_test.yml".format(args.net)

    if args.cfg_file is not None:
        print('Loading config from file {}'.format(args.cfg_file))
        cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(
        args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception(
            'There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir, args.checkweight)

    # initilize the network here.
    if args.net == 'vgg16':
        fastRCNN = vgg16(imdb.classes, pretrained=False,
                         class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fastRCNN = resnet(imdb.classes, 101, pretrained=False,
                          class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fastRCNN = resnet(imdb.classes, 50, pretrained=False,
                          class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fastRCNN = resnet(imdb.classes, 152, pretrained=False,
                          class_agnostic=args.class_agnostic)
    elif args.net == 'alexnet':
        fastRCNN = alexnet(imdb.classes, pretrained=False,
                           class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fastRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fastRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    fastRCNN.RCNN_top = None

    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    rois = torch.FloatTensor(1)
    orig_rois = torch.FloatTensor(1)
    image_classes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        rois = rois.cuda()
        orig_rois = orig_rois.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        image_classes = image_classes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    rois = Variable(rois)
    orig_rois = Variable(orig_rois)
    image_classes = Variable(image_classes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fastRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 1e-3

    save_name = 'fast_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)

    if args.net == 'alexnet':
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                                 imdb.num_classes, training=False, normalize=False, multi_scale=True)
    else:
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1,
                                 imdb.num_classes, training=False, normalize=False, multi_scale=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=4,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    fastRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for step_i in range(num_images):
        data_tic = time.time()
        data = next(data_iter)

        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])
        wgt_boxes = Variable(torch.Tensor([0]))
        wnum_boxes = Variable(torch.Tensor([0]))
        image_classes = Variable(torch.Tensor([0]))

        im_data.data.resize_(data[0][0].size()).copy_(data[0][0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        orig_rois.data.resize_(data[4].size()).copy_(data[4])
        rois.data.resize_(data[5].size()).copy_(data[5])

        index = int(data[6][0].item())

        data_toc = time.time()
        data_time = data_toc - data_tic

        det_tic = time.time()
        with torch.no_grad():
            rcnn_out_1, rcnn_out_2, rcnn_out_3, rcnn_out_4, rcnn_out_5 = fastRCNN(
                im_data, im_info, gt_boxes, num_boxes, wgt_boxes, wnum_boxes, rois, image_classes)

        rois, cls_prob, bbox_pred, RCNN_loss_cls, RCNN_loss_bbox, rois_label, image_loss_cls = rcnn_out_5

        _scores = cls_prob.data
        _boxes = orig_rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            _box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    _box_deltas = _box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    _box_deltas = _box_deltas.view(1, -1, 4)
                else:
                    _box_deltas = _box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    _box_deltas = _box_deltas.view(1, -1, 4 * len(imdb.classes))

            _pred_boxes = bbox_transform_inv(_boxes, _box_deltas, 1)
            _pred_boxes = clip_boxes(_pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            _pred_boxes = np.tile(_boxes, (1, _scores.shape[1]))

        scores = _scores.squeeze()
        pred_boxes = _pred_boxes.squeeze()

        # collect results
        det_toc = time.time()
        det_time = det_toc - det_tic
        misc_tic = time.time()

        if vis:
            im = cv2.imread(imdb.image_path_at(index))
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(
                        im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][index] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][index] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][index][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][index][:, -1] >= image_thresh)[0]
                    all_boxes[j][index] = all_boxes[j][index][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d}, index: {:d}, data cost: {:.3f}s, det cost: {:.3f}s, nms cost: {:.3f}s   \r'
                         .format(step_i + 1, num_images, index, data_time, det_time, nms_time))
        sys.stdout.flush()

        # visualization
        if vis:
            cv2.imwrite('vis/' + imdb.image_index_at(index) + '.png', im2show)
            # pdb.set_trace()
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))
