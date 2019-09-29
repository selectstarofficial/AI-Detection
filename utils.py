# -*- coding: utf-8 -*-

from lxml import etree as Element
import os
import numpy as np
from glob import glob


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset


def make_xml(img_info,bboxes):
    img_path,img_id,img_width,img_height = img_info
    # AnnotationXML = Element.Element('annotations')
    imageXML = Element.Element('image')
    imageXML.set('id', str(img_id))
    imageXML.set('name', os.path.split(img_path)[-1])
    imageXML.set('width', str(img_width))
    imageXML.set('height', str(img_height))
    for (x1, y1, x2, y2,score) in bboxes:
        xmin=max(min(x1,x2),0)
        ymin=max(min(y1,y2),0)
        xmax=min(max(x1,x2),img_width)
        ymax=min(max(y1,y2),img_height)

        boxXML = Element.Element('box')
        boxXML.set('label', 'face')
        boxXML.set('xtl', str(xmin))
        boxXML.set('ytl', str(ymin))
        boxXML.set('xbr', str(xmax))
        boxXML.set('ybr', str(ymax))
        imageXML.append(boxXML)
    # AnnotationXML.append(imageXML)
    return imageXML
    # with open(save_path, 'w') as f:
    #     f.write((Element.tostring(AnnotationXML, pretty_print=True)).decode('utf-8'))


def make_bbox_small(bboxes,width_ratio,height_ratio):
    bboxes_small=[]
    for (x1, y1, x2, y2, score) in bboxes:
        center_x=(x1+x2)/2
        center_y = (y1 + y2) / 2
        width=abs(x2-x1)
        height=abs(y2-y1)
        width_small=width*float(width_ratio)
        height_small = height * float(height_ratio)
        x1_small=center_x-1/2*width_small
        x2_small = center_x + 1 / 2 * width_small
        y1_small = center_y - 1 / 2 * height_small
        y2_small = center_y + 1 / 2 * height_small
        bboxes_small.append([x1_small, y1_small, x2_small, y2_small, score])
    return np.array(bboxes_small)

def filter_too_big(bboxes, max_size_ratio, img_width, img_height):
    bboxes_filtered = []
    for (x1, y1, x2, y2, score) in bboxes:
        width = abs(x2-x1)/img_width
        height = abs(y2-y1)/img_height
        if width > max_size_ratio:
            continue
        if height > max_size_ratio:
            continue
        bboxes_filtered.append([x1, y1, x2, y2, score])
    return np.array(bboxes_filtered)


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = glob(os.path.join(facedir, '*.jpg'))
        images = [os.path.split(image)[-1] for image in images]
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths


class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
