#!/usr/bin/env python

import numpy
import xml.etree.ElementTree as ET

VOC_CLASSES = ('_background_', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')


def indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def convert_xml(org_xml_fn, new_xml_fn, dets, im_info):
    '''
    dets is in the form of [xmin, ymin, xmax, ymax, cls_id]
    im_info = [height, width]
    '''

    # classes = ('_background_', 'dog')
    classes = VOC_CLASSES
    height, width = im_info[0], im_info[1]

    tree = ET.parse(org_xml_fn)
    root = tree.getroot()
    objs = tree.findall('object')
    for obj in objs:
        root.remove(obj)

    for det in dets:
        new_obj = ET.Element('object')
        det_label = int(det[-1])
        name = ET.Element('name')
        name.text = classes[det_label]
        new_obj.append(name)

        difficult = ET.Element('difficult')
        difficult.text = str(0)
        new_obj.append(difficult)

        bndbox = ET.Element('bndbox')
        xmin = ET.Element('xmin')
        ymin = ET.Element('ymin')
        xmax = ET.Element('xmax')
        ymax = ET.Element('ymax')
        vxmin = max(int(det[0]), 1)
        vymin = max(int(det[1]), 1)
        vxmax = min(int(det[2]), width - 1)
        vymax = min(int(det[3]), height - 1)
        xmin.text = str(vxmin)
        bndbox.append(xmin)
        ymin.text = str(vymin)
        bndbox.append(ymin)
        xmax.text = str(vxmax)
        bndbox.append(xmax)
        ymax.text = str(vymax)
        bndbox.append(ymax)
        new_obj.append(bndbox)
        root.append(new_obj)

    indent(root)
    tree.write(new_xml_fn)
