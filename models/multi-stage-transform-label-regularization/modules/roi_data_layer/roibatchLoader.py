
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


class roibatchLoader(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None, multi_scale=False):
        self._roidb = roidb
        self._num_classes = num_classes
        # we make the height of image consistent to trim_height, trim_width
        self.trim_height = cfg.TRAIN.TRIM_HEIGHT
        self.trim_width = cfg.TRAIN.TRIM_WIDTH
        self.max_num_box = cfg.MAX_NUM_GT_BOXES
        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)
        self.multi_scale = multi_scale

        # given the ratio_list, we want to make the ratio same for each batch.
        self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
        num_batch = int(np.ceil(len(ratio_index) / batch_size))
        for i in range(num_batch):
            left_idx = i*batch_size
            right_idx = min((i+1)*batch_size-1, self.data_size-1)

            '''
        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1
        '''
            target_ratio = 1

            self.ratio_list_batch[left_idx:(right_idx + 1)] = target_ratio

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = [self._roidb[index_ratio]]
        blobs = get_minibatch(minibatch_db, self._num_classes)
        data = torch.from_numpy(blobs['data'])
        im_info = torch.from_numpy(blobs['im_info'])
        rois = torch.from_numpy(blobs['rois'])
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(1), data.size(2)

        if self.training:
            np.random.shuffle(blobs['gt_boxes'])
            gt_boxes = torch.from_numpy(blobs['gt_boxes'])

            ########################################################
            # padding the input image to fixed size for each group #
            ########################################################

            # NOTE1: need to cope with the case where a group cover both conditions. (done)
            # NOTE2: need to consider the situation for the tail samples. (no worry)
            # NOTE3: need to implement a parallel data loader. (no worry)
            # get the index range

            # if the image need to crop, crop to the target size.
            ratio = self.ratio_list_batch[index]

            if self._roidb[index_ratio]['need_crop']:
                if ratio < 1:
                    # this means that data_width << data_height, we need to crop the
                    # data_height
                    min_y = int(torch.min(gt_boxes[:, 1]))
                    max_y = int(torch.max(gt_boxes[:, 3]))
                    trim_size = int(np.floor(data_width / ratio))
                    if trim_size > data_height:
                        trim_size = data_height
                    box_region = max_y - min_y + 1
                    if min_y == 0:
                        y_s = 0
                    else:
                        if (box_region-trim_size) < 0:
                            y_s_min = max(max_y-trim_size, 0)
                            y_s_max = min(min_y, data_height-trim_size)
                            if y_s_min == y_s_max:
                                y_s = y_s_min
                            else:
                                y_s = np.random.choice(range(y_s_min, y_s_max))
                        else:
                            y_s_add = int((box_region-trim_size)/2)
                            if y_s_add == 0:
                                y_s = min_y
                            else:
                                y_s = np.random.choice(
                                    range(min_y, min_y+y_s_add))
                    # crop the image
                    data = data[:, y_s:(y_s + trim_size), :, :]

                    # shift y coordiante of gt_boxes
                    gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                    gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                    # update gt bounding box according the trip
                    gt_boxes[:, 1].clamp_(0, trim_size - 1)
                    gt_boxes[:, 3].clamp_(0, trim_size - 1)

                    # shift y coordiante of rois
                    rois[:, 2] = rois[:, 2] - float(y_s)
                    rois[:, 4] = rois[:, 4] - float(y_s)

                    # update rois bounding box according the trip
                    rois[:, 2].clamp_(0, trim_size - 1)
                    rois[:, 4].clamp_(0, trim_size - 1)

                else:
                    # this means that data_width >> data_height, we need to crop the
                    # data_width
                    min_x = int(torch.min(gt_boxes[:, 0]))
                    max_x = int(torch.max(gt_boxes[:, 2]))
                    trim_size = int(np.ceil(data_height * ratio))
                    if trim_size > data_width:
                        trim_size = data_width
                    box_region = max_x - min_x + 1
                    if min_x == 0:
                        x_s = 0
                    else:
                        if (box_region-trim_size) < 0:
                            x_s_min = max(max_x-trim_size, 0)
                            x_s_max = min(min_x, data_width-trim_size)
                            if x_s_min == x_s_max:
                                x_s = x_s_min
                            else:
                                x_s = np.random.choice(range(x_s_min, x_s_max))
                        else:
                            x_s_add = int((box_region-trim_size)/2)
                            if x_s_add == 0:
                                x_s = min_x
                            else:
                                x_s = np.random.choice(
                                    range(min_x, min_x+x_s_add))
                    # crop the image
                    data = data[:, :, x_s:(x_s + trim_size), :]

                    # shift x coordiante of gt_boxes
                    gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                    gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                    # update gt bounding box according the trip
                    gt_boxes[:, 0].clamp_(0, trim_size - 1)
                    gt_boxes[:, 2].clamp_(0, trim_size - 1)

                    # shift x coordiante of rois
                    rois[:, 1] = rois[:, 1] - float(x_s)
                    rois[:, 3] = rois[:, 3] - float(x_s)
                    # update gt bounding box according the trip
                    rois[:, 1].clamp_(0, trim_size - 1)
                    rois[:, 3].clamp_(0, trim_size - 1)

            # based on the ratio, padding the image.
            if ratio < 1:
                # this means that data_width < data_height
                trim_size = int(np.floor(data_width / ratio))

                padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)),
                                                 data_width, 3).zero_()

                padding_data[:data_height, :, :] = data[0]
                # update im_info
                im_info[0, 0] = padding_data.size(0)
                # print("height %d %d \n" %(index, anchor_idx))
            elif ratio > 1:
                # this means that data_width > data_height
                # if the image need to crop.
                padding_data = torch.FloatTensor(data_height,
                                                 int(np.ceil(data_height * ratio)), 3).zero_()
                padding_data[:, :data_width, :] = data[0]
                im_info[0, 1] = padding_data.size(1)
            else:
                trim_size = min(data_height, data_width)
                padding_data = torch.FloatTensor(
                    trim_size, trim_size, 3).zero_()
                padding_data = data[0][:trim_size, :trim_size, :]
                # gt_boxes.clamp_(0, trim_size)
                gt_boxes[:, :4].clamp_(0, trim_size)
                rois[:, 1:5].clamp_(0, trim_size)
                im_info[0, 0] = trim_size
                im_info[0, 1] = trim_size

            # check the bounding box:
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

            # permute trim_data to adapt to downstream processing
            padding_data = padding_data.permute(2, 0, 1).contiguous()
            im_info = im_info.view(3)

            rois_not_keep = (rois[:, 1] == rois[:, 3]) | (
                rois[:, 2] == rois[:, 4])
            rois_keep = torch.nonzero(rois_not_keep == 0).view(-1)
            rois = rois[rois_keep]

            return padding_data, im_info, gt_boxes_padding, num_boxes, rois
        else:
            # prepare scaled data and rois
            import cv2

            im_orig = blobs['data']
            im_height, im_width = im_orig.shape[1], im_orig.shape[2]

            if self.multi_scale:
                scales = np.array([400, 600, 750, 880, 1000]) / 600.0
            else:
                scales = np.array([400, 600]) / 600.0  # only two scales because of limited gpu memory
                # max_side = max(im_height, im_width)
                # max_side_scale = 600.0 / max_side
                # scales = np.array([max_side_scale])

            max_scale = np.max(scales)
            max_height, max_width = int(max_scale * im_height), int(max_scale * im_width)

            padding_im = np.zeros((
                len(scales), max_height, max_width, 3))

            for sind, im_scale in enumerate(scales):
                scale_im = im_orig.copy()[0]
                scale_height, scale_width = int(im_scale * im_height), int(im_scale * im_width)
                scale_im = cv2.resize(scale_im, None, None, fx=im_scale, fy=im_scale,
                                      interpolation=cv2.INTER_LINEAR)
                padding_im[sind, :scale_height, :scale_width, :] = scale_im[:scale_height, :scale_width, :]

            padding_data = torch.from_numpy(padding_im)
            padding_data = padding_data.permute(0, 3, 1, 2).contiguous().view(
                len(scales), 3, max_height, max_width)

            im_info = torch.FloatTensor([max_height, max_width, max_scale * blobs['im_info'][0, 2]])

            gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
            num_boxes = 0

            im_rois = blobs['rois'][:, 1:5]
            im_rois = im_rois.astype(np.float, copy=False)
            widths = im_rois[:, 2] - im_rois[:, 0] + 1
            heights = im_rois[:, 3] - im_rois[:, 1] + 1
            areas = widths * heights
            scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
            diff_areas = np.abs(scaled_areas - 224 * 224)
            levels = diff_areas.argmin(axis=1)[:, np.newaxis]
            rois_scale = im_rois * scales[levels]
            rois_blob = np.hstack((levels, rois_scale))
            rois_blob = rois_blob.astype(np.float32, copy=False)

            scaled_rois = torch.from_numpy(rois_blob)
            orig_rois = rois / blobs['im_info'][0, 2].item()

            # scaled_rois = scaled_rois[:2000]
            # orig_rois = orig_rois[:2000]

            return padding_data, im_info, gt_boxes, num_boxes, orig_rois, scaled_rois, index

    def __len__(self):
        return len(self._roidb)
