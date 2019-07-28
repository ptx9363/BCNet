# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from fast_rcnn.fast_rcnn import _fastRCNN
import pdb
import copy


class rcnn_branch(nn.Module):
    def __init__(self):
        super(rcnn_branch, self).__init__()
        self.RCNN_top = None
        self.RCNN_cls_score = None
        self.RCNN_bbox_pred = None

    def forward():
        pass


class vgg16(_fastRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False, pretrained_weight=''):
        if pretrained_weight == '':
            self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        else:
            self.model_path = pretrained_weight
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _fastRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict(
                {k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(
            *list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters():
                p.requires_grad = False

        vgg.classifier = nn.Sequential(
            *list(vgg.classifier._modules.values())[:-1])

        self.sbranch = rcnn_branch()
        self.sbranch.RCNN_top = vgg.classifier
        self.sbranch.RCNN_cls_score = nn.Linear(4096, self.n_classes)
        if self.class_agnostic:
            self.sbranch.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.sbranch.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

        self.wbranch = copy.deepcopy(self.sbranch)
        self.sbranch2 = copy.deepcopy(self.sbranch)

    def _head_to_tail(self, pool5, branch):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = branch.RCNN_top(pool5_flat)

        return fc7
