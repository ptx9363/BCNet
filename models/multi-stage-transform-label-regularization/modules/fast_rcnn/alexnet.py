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


class AlexNet(nn.Module):
    '''
    define pytorch alexnet for fast rcnn
    '''
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class rcnn_branch(nn.Module):
    def __init__(self):
        super(rcnn_branch, self).__init__()
        self.RCNN_top = None
        self.RCNN_cls_score = None
        self.RCNN_bbox_pred = None

    def forward():
        pass


class alexnet(_fastRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False, pretrained_weight=''):
        if pretrained_weight == '':
            self.model_path = 'data/pretrained_model/alexnet_mxnet.pth'
        else:
            self.model_path = pretrained_weight
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _fastRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        alexnet = AlexNet()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            alexnet.load_state_dict(
                {k: v for k, v in state_dict.items() if k in alexnet.state_dict()})

        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(
            *list(alexnet.features._modules.values())[:-1])

        # only fix conv 1:
        for layer in range(1):
            for name, p in self.RCNN_base[layer].named_parameters():
                if p.requires_grad:
                    p.requires_grad = False

        alexnet.classifier = nn.Sequential(
            *list(alexnet.classifier._modules.values())[:-1])

        self.sbranch = rcnn_branch()
        self.sbranch.RCNN_top = alexnet.classifier
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
