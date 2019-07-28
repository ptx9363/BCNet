import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
from utils.config import cfg
from roi_pooling.modules.roi_pool import _RoIPooling
from roi_crop.modules.roi_crop import _RoICrop
from roi_align.modules.roi_align import RoIAlignAvg
from rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from rpn.bbox_transform import bbox_transform_inv, clip_boxes


class _fastRCNN(nn.Module):
    """ fast RCNN """

    def __init__(self, classes, class_agnostic):
        super(_fastRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(
            cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(
            cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * \
            2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def forward_rcnn_batch(self, base_feat, branch, rois, wgt_boxes, wnum_boxes, gt_boxes, num_boxes, im_info, image_classes, output_refine=False):
        batch_size = base_feat.size(0)

        # if it is training phrase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(
                rois, wgt_boxes, wnum_boxes, gt_boxes, num_boxes)
            out_rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(
                rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(
                rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            out_rois = rois
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None

        out_rois = Variable(out_rois)

        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(
                out_rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack(
                [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(
                base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, out_rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, out_rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat, branch)

        # compute bbox offset
        bbox_pred = branch.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(
                rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = branch.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:

            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(
                bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            # add image-level label regularization
            rois_batch_size = out_rois.size(1)
            rois_prob = F.softmax(cls_score, 1).view(batch_size, rois_batch_size, -1)

            valid_rois_prob = (rois_label > 0).view(batch_size, rois_batch_size, -1).float()
            rois_attention = F.softmax(cls_score, 1).view(batch_size, rois_batch_size, -1)
            rois_attention = rois_attention * valid_rois_prob

            # ignore background
            rois_prob = rois_prob[:, :, 1:]
            rois_attention = rois_attention[:, :, 1:]

            # rois_attention_prob = torch.sum(rois_prob * rois_attention, dim=1) / (torch.sum(rois_attention, dim=1) + 1e-10)
            rois_attention_prob, _ = torch.max(rois_prob, dim=1)
            image_loss_cls = F.binary_cross_entropy(rois_attention_prob, image_classes[:, 1:])
        else:
            image_loss_cls = None

        if self.training:
            cls_prob = cls_prob.view(batch_size, out_rois.size(1), -1)
            bbox_pred = bbox_pred.view(batch_size, out_rois.size(1), -1)
        else:
            cls_prob = cls_prob.view(1, out_rois.size(1), -1)
            bbox_pred = bbox_pred.view(1, out_rois.size(1), -1)

        if self.training and output_refine:
            # get transformation for wgt_boxes
            wgt_rois = wgt_boxes.new(wgt_boxes.size()).zero_()
            wgt_rois[:, :, 1:5] = wgt_boxes[:, :, :4]
            batch_size = base_feat.size(0)
            for i in range(batch_size):
                wgt_rois[:, :, 0] = i

            # do roi pooling based on predicted rois
            if cfg.POOLING_MODE == 'crop':
                # pdb.set_trace()
                # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
                grid_xy = _affine_grid_gen(
                    wgt_rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                gt_pooled_feat = self.RCNN_roi_crop(
                    base_feat, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    gt_pooled_feat = F.max_pool2d(gt_pooled_feat, 2, 2)
            elif cfg.POOLING_MODE == 'align':
                gt_pooled_feat = self.RCNN_roi_align(
                    base_feat, wgt_rois.view(-1, 5))
            elif cfg.POOLING_MODE == 'pool':
                gt_pooled_feat = self.RCNN_roi_pool(
                    base_feat, wgt_rois.view(-1, 5))

            # feed pooled features to top model
            gt_pooled_feat = self._head_to_tail(gt_pooled_feat, branch)

            # compute bbox offset
            wgt_bbox_delta = branch.RCNN_bbox_pred(gt_pooled_feat)
            wgt_bbox_delta = wgt_bbox_delta.view(-1, 4) * torch.FloatTensor(
                cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            wgt_bbox_delta = wgt_bbox_delta.view(batch_size, -1, 4 * 21)
            wgt_bbox_out_rois = bbox_transform_inv(
                wgt_boxes, wgt_bbox_delta, batch_size)

            wgt_bbox_out_rois = clip_boxes(
                wgt_bbox_out_rois, im_info.data, batch_size)

            wgt_bbox_out = wgt_boxes.new(wgt_boxes.size()).zero_()

            wgt_cls = Variable(
                wgt_boxes[:, :, 4].data, requires_grad=False).long()
            for i in range(batch_size):
                for j in range(20):
                    cls_ind = wgt_cls[i, j]
                    wgt_bbox_out[i, j, :4] = wgt_bbox_out_rois[i,
                                                               j, cls_ind * 4:cls_ind * 4 + 4]

            wgt_bbox_out[:, :, 4] = wgt_boxes[:, :, 4]

            wgt_boxes_x = (wgt_boxes[:, :, 2] - wgt_boxes[:, :, 0] + 1)
            wgt_boxes_y = (wgt_boxes[:, :, 3] - wgt_boxes[:, :, 1] + 1)
            wgt_area_zero = (wgt_boxes_x == 1) & (wgt_boxes_y == 1)
            wgt_bbox_out.masked_fill_(wgt_area_zero.view(
                batch_size, wgt_area_zero.size(1), 1).expand(wgt_boxes.size()), 0)
            wgt_bbox_out = wgt_bbox_out.detach()
        else:
            wgt_bbox_out = None

        return (out_rois, cls_prob, bbox_pred, RCNN_loss_cls, RCNN_loss_bbox, rois_label, image_loss_cls), wgt_bbox_out

    def forward(self, im_data, im_info, gt_boxes, num_boxes, wgt_boxes, wnum_boxes, rois, image_classes):
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        wgt_boxes = wgt_boxes.data
        wnum_boxes = wnum_boxes.data
        rois = rois.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        if self.training:
            # weak label branch and transformation regression
            rcnn_out_1, wgt_boxes_1 = self.forward_rcnn_batch(
                base_feat, self.wbranch, rois, wgt_boxes, wnum_boxes, gt_boxes, num_boxes,
                im_info, image_classes, output_refine=True)

            rcnn_out_2 = None

            # transformation label branch
            rcnn_out_3, wgt_boxes_2 = self.forward_rcnn_batch(
                base_feat, self.sbranch, rois, wgt_boxes_1, wnum_boxes, gt_boxes, num_boxes,
                im_info, image_classes, output_refine=True)

            # transformation label branch2
            rcnn_out_4, _ = self.forward_rcnn_batch(
                base_feat[:1], self.sbranch2, rois[:1], gt_boxes[:1], num_boxes[:1], gt_boxes[:1], num_boxes[:1],
                im_info[:1], image_classes[:1])

            rcnn_out_5, _ = self.forward_rcnn_batch(
                base_feat, self.sbranch2, rois, wgt_boxes_2, wnum_boxes, gt_boxes, num_boxes,
                im_info, image_classes)
        else:
            rcnn_out_1 = None
            rcnn_out_2 = None
            rcnn_out_3 = None
            rcnn_out_4 = None

            # transformation label branch
            rcnn_out_5, _ = self.forward_rcnn_batch(
                base_feat, self.sbranch2, rois, wgt_boxes, wnum_boxes, gt_boxes, num_boxes, im_info, image_classes)

        return rcnn_out_1, rcnn_out_2, rcnn_out_3, rcnn_out_4, rcnn_out_5

    def _init_weights(self, branch):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(branch.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(branch.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()

        # init weak and strong branch
        self._init_weights(self.sbranch)
        self._init_weights(self.wbranch)
        self._init_weights(self.sbranch2)
