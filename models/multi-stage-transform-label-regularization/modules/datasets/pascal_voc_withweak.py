from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from utils.config import cfg
from utils.xml import convert_xml

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete


class pascal_voc_withweak(imdb):
    def __init__(self, image_set, strong_image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_withweak_withimagelabel_' + year + '_' + image_set + '_' + strong_image_set)
        self._year = year
        self._image_set = image_set
        self._strong_image_set = strong_image_set

        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'

        self._image_index = self._load_image_set_index(self._image_set)
        self._strong_image_index = self._load_image_set_index(self._strong_image_set)

        self._roidb_handler = self._load_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        self._proposal_prefix = 'trainval' if 'trainval' in self._image_set else 'test'

        print('Initilizing VOC imdb')
        print('Proposal prefix is {}'.format(self._proposal_prefix))

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        # self._proposal_method = 'edge_boxes'  # proposal_method

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_index_at(self, i):
        return self._image_index[i]

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self, image_set):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit')

    def _load_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_fast_eb_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = self._load_gt_roidb()
        proposal_roidb = self._load_edge_boxes_proposal_roidb()
        for gt_db, pro_db in zip(gt_roidb, proposal_roidb):
            assert gt_db['index'] == pro_db['index']
            gt_db['rois'] = pro_db['rois'].copy()

        # append image id and image path to roidb
        for i in range(len(self._image_index)):
            gt_roidb[i]['img_id'] = self.image_id_at(i)
            gt_roidb[i]['image'] = self.image_path_at(i)

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_edge_boxes_proposal_roidb(self):
        print('loading edge boxes proposals')
        full_image_index_file = os.path.join(self._devkit_path, 'VOC' + self._year, 'ImageSets', 'Main', self._proposal_prefix + '.txt')
        proposal_matpath = os.path.join(cfg.DATA_DIR, 'edge_boxes_data', 'voc_' + self._year + '_' + self._proposal_prefix + '.mat')

        raw_data = sio.loadmat(proposal_matpath)['boxes'][0].ravel()

        box_list = []
        for i in range(raw_data.shape[0]):
            if i % 500 == 499:
                print('processing edge boxes %d' % (i))
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            # keep = ds_utils.unique_boxes(boxes)
            # boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            # if boxes.shape[0] > 2000:
            #    boxes = boxes[:2000]
            box_list.append(boxes)

        with open(full_image_index_file) as f:
            full_image_set_index = [x.strip() for x in f.readlines()]

        full_boxes = {}
        for i, index in enumerate(full_image_set_index):
            full_boxes[index] = box_list[i]

        eb_roidb = []
        image_index = self._load_image_set_index(self._image_set)
        for i, index in enumerate(image_index):
            eb_boxes = np.array(full_boxes[index], dtype=np.uint16)
            roi_rec = {'index': index,
                       'rois': eb_boxes}
            eb_roidb.append(roi_rec)

        return eb_roidb

    def _load_gt_roidb(self):
        """
        """
        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self._image_index]
        return gt_roidb

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        # strong gt boxes
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)

        height = int(tree.find('size').find('height').text)
        width = int(tree.find('size').find('width').text)

        # load image-level label
        if True:
            # only access bbox annotation for images in strong_image_index
            objs = tree.findall('object')

            image_classes = np.zeros((len(self._classes)), dtype=np.int32)
            for ix, obj in enumerate(objs):
                cls = self._class_to_ind[obj.find('name').text.lower().strip()]
                image_classes[cls] = 1

        if index in self._strong_image_index:
            # only access bbox annotation for images in strong_image_index
            objs = tree.findall('object')

            # if not self.config['use_diff']:
            #    Exclude the samples labeled as difficult
            #    non_diff_objs = [
            #        obj for obj in objs if int(obj.find('difficult').text) == 0]
            #    if len(non_diff_objs) != len(objs):
            #        print 'Removed {} difficult objects'.format(
            #            len(objs) - len(non_diff_objs))
            #    objs = non_diff_objs

            num_objs = len(objs)
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            ishards = np.zeros((num_objs), dtype=np.int32)
            for ix, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1

                diffc = obj.find('difficult')
                difficult = 0 if diffc is None else int(diffc.text)
                ishards[ix] = difficult

                cls = self._class_to_ind[obj.find('name').text.lower().strip()]
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
        else:
            num_objs = 0
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            ishards = np.zeros((num_objs), dtype=np.int32)

        # weak gt boxes
        filename = os.path.join(self._data_path, 'WEAKAnnotations', index + '.xml')
        # filename = os.path.join(self._data_path, 'tempAnnotations', index + '.xml')
        if os.path.exists(filename):
            tree = ET.parse(filename)
            weak_objs = tree.findall('object')
            weak_num_objs = len(weak_objs)
            weak_boxes = np.zeros((weak_num_objs, 4), dtype=np.uint16)
            weak_gt_classes = np.zeros((weak_num_objs), dtype=np.int32)
            for ix, obj in enumerate(weak_objs):
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1

                cls = self._class_to_ind[obj.find('name').text.lower().strip()]
                weak_boxes[ix, :] = [x1, y1, x2, y2]
                weak_gt_classes[ix] = cls
        else:
            weak_num_objs = 0
            weak_boxes = np.zeros((weak_num_objs, 4), dtype=np.uint16)
            weak_gt_classes = np.zeros((weak_num_objs), dtype=np.int32)

        return {'index': index,
                'boxes': boxes,
                'gt_classes': gt_classes,
                'image_classes': image_classes,
                'weak_boxes': weak_boxes,
                'weak_gt_classes': weak_gt_classes,
                'gt_ishard': ishards,
                'flipped': False,
                'height': height,
                'width': width,
                }

    def append_flipped_images(self):
        """Only flip boxes coordinates, images will be flipped when loading into network"""
        print('%s append flipped images to roidb' % self._name)
        roidb_flipped = []
        for roi_rec in self.roidb:

            # flip gt boxes
            boxes = roi_rec['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = roi_rec['width'] - oldx2 - 1
            boxes[:, 2] = roi_rec['width'] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()

            # flip rois
            rois = roi_rec['rois'].copy()
            rois_oldx1 = rois[:, 0].copy()
            rois_oldx2 = rois[:, 2].copy()
            rois[:, 0] = roi_rec['width'] - rois_oldx2 - 1
            rois[:, 2] = roi_rec['width'] - rois_oldx1 - 1
            assert (rois[:, 2] >= rois[:, 0]).all()

            # flip weak boxes
            wboxes = roi_rec['weak_boxes'].copy()
            woldx1 = wboxes[:, 0].copy()
            woldx2 = wboxes[:, 2].copy()
            wboxes[:, 0] = roi_rec['width'] - woldx2 - 1
            wboxes[:, 2] = roi_rec['width'] - woldx1 - 1
            assert (wboxes[:, 2] >= wboxes[:, 0]).all()

            roi_rec_flipped = roi_rec.copy()
            roi_rec_flipped['boxes'] = boxes
            roi_rec_flipped['weak_boxes'] = wboxes
            roi_rec_flipped['rois'] = rois
            roi_rec_flipped['flipped'] = True

            roidb_flipped.append(roi_rec_flipped)
        self._roidb.extend(roidb_flipped)

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        # status = subprocess.call(cmd, shell=True)

    def _get_refine_annotations(self, all_boxes):
        import cv2
        n_images = len(all_boxes[0])
        refine_bboxs = [[] for im_ind in range(n_images)]

        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue

            for im_ind, dets in enumerate(all_boxes[cls_ind]):
                dets = all_boxes[cls_ind][im_ind]
                for k in range(dets.shape[0]):
                    bbox = [dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1]
                    confidence = dets[k, -1]

                    if confidence > 0.9:
                        refine_bboxs[im_ind].append([bbox[0], bbox[1], bbox[2], bbox[3], cls_ind])

        for im_ind in range(n_images):
            image_id = self._image_index[im_ind]
            image_fn = self.roidb[im_ind]['image']
            org_xml = os.path.join(self._data_path, 'Annotations', image_id + '.xml')
            new_xml = org_xml.replace('Annotations', 'tempAnnotations')
            image = np.array(cv2.imread(image_fn))
            im_info = image.shape[:2]
            if len(refine_bboxs[im_ind]) > 0:
                convert_xml(org_xml, new_xml, refine_bboxs[im_ind], im_info)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

        # self._get_refine_annotations(all_boxes)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    pass
