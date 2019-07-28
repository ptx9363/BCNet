
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from utils.config import cfg
from roi_data_layer.minibatch import get_minibatch, get_minibatch
from rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb


class randcrop_roibatchLoader(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
        self._roidb = roidb
        self._num_classes = num_classes
        # we make the height of image consistent to trim_height, trim_width
        self.trim_height = cfg.TRAIN.TRIM_HEIGHT
        self.trim_width = cfg.TRAIN.TRIM_WIDTH
        self.max_num_box = cfg.MAX_NUM_GT_BOXES
        self.training = training
        self.normalize = normalize

        self.batch_size = batch_size
        self.data_size = len(roidb)

    def __getitem__(self, index):
        self.trim_size = min(self.trim_height, self.trim_width)
        minibatch_db = [self._roidb[index]]
        blobs = get_minibatch(minibatch_db, self._num_classes)
        data = torch.from_numpy(blobs['data'])
        im_info = torch.from_numpy(blobs['im_info'])
        rois = torch.from_numpy(blobs['rois'])
        image_classes = torch.from_numpy(blobs['image_classes'])
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(1), data.size(2)

        if self.training:
            # np.random.shuffle(blobs['gt_boxes'])
            # np.random.shuffle(blobs['weak_gt_boxes'])

            gt_boxes = torch.from_numpy(blobs['gt_boxes'])
            num_boxes = gt_boxes.size(0)
            wgt_boxes = torch.from_numpy(blobs['weak_gt_boxes'])
            wnum_boxes = wgt_boxes.size(0)

            avaiable_boxes = torch.from_numpy(
                np.vstack((blobs['gt_boxes'], blobs['weak_gt_boxes'])))

            # avaiable_boxes = torch.from_numpy(blobs['gt_boxes'])

            if data_height > self.trim_height:
                # this means that data_width < data_height, we need to crop the
                # data_height
                min_y = int(torch.min(avaiable_boxes[:, 1]))
                max_y = int(torch.max(avaiable_boxes[:, 3]))
                trim_size = self.trim_height
                box_region = max_y - min_y + 1

                if min_y == 0:
                    y_s = 0
                else:
                    if (box_region - trim_size) < 0:
                        y_s_min = max(max_y - trim_size, 0)
                        y_s_max = min(min_y, data_height - trim_size)
                        if y_s_min == y_s_max:
                            y_s = y_s_min
                        else:
                            y_s = np.random.choice(range(y_s_min, y_s_max))

                    else:
                        y_s_add = int((box_region - trim_size) / 2)
                        if y_s_add == 0:
                            y_s = min_y
                        else:
                            y_s = np.random.choice(range(min_y, min_y + y_s_add))

                # crop the image
                data = data[:, y_s:(y_s + trim_size), :, :]

                # shift y coordiante of gt_boxes
                if num_boxes > 0:
                    gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                    gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)
                    gt_boxes[:, 1].clamp_(0, trim_size - 1)
                    gt_boxes[:, 3].clamp_(0, trim_size - 1)

                # shift y coordiante of weak gt_boxes
                if wnum_boxes > 0:
                    wgt_boxes[:, 1] = wgt_boxes[:, 1] - float(y_s)
                    wgt_boxes[:, 3] = wgt_boxes[:, 3] - float(y_s)
                    wgt_boxes[:, 1].clamp_(0, trim_size - 1)
                    wgt_boxes[:, 3].clamp_(0, trim_size - 1)

                # shift y coordiante of rois
                rois[:, 2] = rois[:, 2] - float(y_s)
                rois[:, 4] = rois[:, 4] - float(y_s)
                rois[:, 2].clamp_(0, trim_size - 1)
                rois[:, 4].clamp_(0, trim_size - 1)

            if data_width > self.trim_width:
                # this means that data_width > data_height, we need to crop the
                # data_width
                min_x = int(torch.min(avaiable_boxes[:, 0]))
                max_x = int(torch.max(avaiable_boxes[:, 2]))
                trim_size = self.trim_width
                box_region = max_x - min_x + 1
                if min_x == 0:
                    x_s = 0
                else:
                    if (box_region - trim_size) < 0:
                        x_s_min = max(max_x - trim_size, 0)
                        x_s_max = min(min_x, data_width - trim_size)
                        if x_s_min == x_s_max:
                            x_s = x_s_min
                        else:
                            x_s = np.random.choice(range(x_s_min, x_s_max))
                    else:
                        x_s_add = int((box_region - trim_size) / 2)
                        if x_s_add == 0:
                            x_s = min_x
                        else:
                            x_s = np.random.choice(range(min_x, min_x + x_s_add))

                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                if num_boxes > 0:
                    gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                    gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                    gt_boxes[:, 0].clamp_(0, trim_size - 1)
                    gt_boxes[:, 2].clamp_(0, trim_size - 1)

                # shift x coordiante of gt_boxes
                if wnum_boxes > 0:
                    wgt_boxes[:, 0] = wgt_boxes[:, 0] - float(x_s)
                    wgt_boxes[:, 2] = wgt_boxes[:, 2] - float(x_s)
                    wgt_boxes[:, 0].clamp_(0, trim_size - 1)
                    wgt_boxes[:, 2].clamp_(0, trim_size - 1)

                # shift x coordiante of rois
                rois[:, 1] = rois[:, 1] - float(x_s)
                rois[:, 3] = rois[:, 3] - float(x_s)
                rois[:, 1].clamp_(0, trim_size - 1)
                rois[:, 3].clamp_(0, trim_size - 1)

            trim_size = min(self.trim_width, self.trim_height)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            real_height = min(trim_size, data_height)
            real_width = min(trim_size, data_width)
            padding_data[:real_height, :real_width, :] = data[0][:real_height, :real_width, :]

            rois[:, 1:5].clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size

            # check the bounding box:
            if num_boxes > 0:
                gt_boxes[:, :4].clamp_(0, trim_size)
                not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (
                    gt_boxes[:, 1] == gt_boxes[:, 3])
                keep = torch.nonzero(not_keep == 0).view(-1)
                gt_boxes_padding = torch.FloatTensor(
                    self.max_num_box, gt_boxes.size(1)).zero_()
                if keep.numel() != 0:
                    gt_boxes = gt_boxes[keep]
                    num_boxes = min(gt_boxes.size(0), self.max_num_box)
                    gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]
                else:
                    num_boxes = 0
            else:
                gt_boxes_padding = torch.FloatTensor(
                    self.max_num_box, 5).zero_()
                num_boxes = 0

            # check the weak bounding box:
            if wnum_boxes > 0:
                wgt_boxes[:, :4].clamp_(0, trim_size)
                wnot_keep = (wgt_boxes[:, 0] == wgt_boxes[:, 2]) | (
                    wgt_boxes[:, 1] == wgt_boxes[:, 3])
                wkeep = torch.nonzero(wnot_keep == 0).view(-1)
                wgt_boxes_padding = torch.FloatTensor(
                    self.max_num_box, wgt_boxes.size(1)).zero_()
                if wkeep.numel() != 0:
                    wgt_boxes = wgt_boxes[wkeep]
                    wnum_boxes = min(wgt_boxes.size(0), self.max_num_box)
                    wgt_boxes_padding[:wnum_boxes, :] = wgt_boxes[:wnum_boxes]
                else:
                    wnum_boxes = 0
            else:
                wgt_boxes_padding = torch.FloatTensor(
                    self.max_num_box, 5).zero_()
                wnum_boxes = 0

            # permute trim_data to adapt to downstream processing
            padding_data = padding_data.permute(2, 0, 1).contiguous()
            im_info = im_info.view(3)

            # padding rois
            rois_not_keep = (rois[:, 1] == rois[:, 3]) | (
                rois[:, 2] == rois[:, 4])
            rois_keep = torch.nonzero(rois_not_keep == 0).view(-1)
            rois = rois[rois_keep]
            max_num_rois = 2000
            num_rois = min(rois.size(0), max_num_rois)
            rois_padding = torch.FloatTensor(
                max_num_rois, 5).zero_()
            rois_padding[:num_rois, :] = rois[:num_rois]

            return padding_data, im_info, gt_boxes_padding, num_boxes, wgt_boxes_padding, wnum_boxes, rois_padding, image_classes
        else:
            data = data.permute(0, 3, 1, 2).contiguous().view(
                3, data_height, data_width)
            im_info = im_info.view(3)

            gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
            num_boxes = 0

            return data, im_info, gt_boxes, num_boxes, rois, image_classes

    def __len__(self):
        return len(self._roidb)
