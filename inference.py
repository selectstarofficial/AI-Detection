from face_detection_api import FaceDetector
from license_plate_api import LicensePlateDetector
import cv2
import os
import os.path as osp
import numpy as np
from PIL import Image
from settings import Settings
import utils
import json

class BBoxClass:
    def __init__(self, label, x1, y1, x2, y2, score):
        self.label = label
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score

class ImageClass:
    def __init__(self, id, name, width, height):
        self.id = id
        self.name = name
        self.width = width
        self.height = height
        self.bbox_list = [] # type: [BBoxClass]

local = False
if local:
    input_dir = "/Users/litcoderr/Desktop/Projects/dataset/input"
    output_dir = "/Users/litcoderr/Desktop/Projects/dataset/output"
else:
    input_dir = "./license_plate_api/data/custom/images/"
    output_dir = "./license_plate_api/data/custom/output/"

class_index = {
    "license_plate" : 0,
    "face" : 1
}

if __name__ == '__main__':
    settings = Settings()

    if not osp.exists(input_dir):
        raise FileNotFoundError(f"{input_dir} directory not exists.")
    os.makedirs(output_dir, exist_ok=True)

    dataset = utils.get_dataset(input_dir)

    images = {}  # path: ImageClass

    ### FACE DETECTION ###
    print('Detecting faces...')

    print('Creating networks and loading parameters')
    face_api = FaceDetector(settings)
    print('Preparing detector...')
    face_api.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))

    global_img_id = 0
    for cls in dataset:
        save_class_dir = osp.join(output_dir, cls.name)
        os.makedirs(save_class_dir, exist_ok=True)
        cls.image_paths = sorted(cls.image_paths)

        for i, image_path in enumerate(cls.image_paths):
            print('[{}/{}] {}'.format(i + 1, len(cls.image_paths), image_path))

            img = np.array(Image.open(image_path).convert('RGB'))

            img_height, img_width = img.shape[0:2]

            bboxes = face_api.detect(img)  # (x1, y1, x2, y2, score)
            bboxes = utils.make_bbox_small(bboxes, settings.face_bbox_width_ratio, settings.face_bbox_height_ratio)
            bboxes = utils.filter_too_big(bboxes, settings.max_size_ratio, img_width, img_height)

            img_info = ImageClass(global_img_id, image_path, img_width, img_height)

            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox
                bbox_info = BBoxClass(settings.xml_face_name, x1, y1, x2, y2, score)
                img_info.bbox_list.append(bbox_info)

            images[image_path] = img_info

            global_img_id += 1


    ### RELEASE MEMORY ####
    face_api.release_memory()
    del face_api

    ### LICENSE PLATE DETECTION ###
    print('Detecting license plates...')

    print('Creating networks and loading parameters')
    license_plate_api = LicensePlateDetector(settings)
    print('Preparing detector...')
    license_plate_api.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))

    for cls in dataset:
        save_class_dir = osp.join(output_dir, cls.name)
        cls.image_paths = sorted(cls.image_paths)

        for i, image_path in enumerate(cls.image_paths):
            print('[{}/{}] {}'.format(i + 1, len(cls.image_paths), image_path))

            img = np.array(Image.open(image_path).convert('RGB'))

            img_height, img_width = img.shape[0:2]

            bboxes = license_plate_api.detect(img)  # (x1, y1, x2, y2, score)
            bboxes = utils.make_bbox_small(bboxes, settings.license_plate_bbox_width_ratio, settings.license_plate_bbox_height_ratio)
            bboxes = utils.filter_too_big(bboxes, settings.max_size_ratio, img_width, img_height)

            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox
                bbox_info = BBoxClass(settings.xml_license_plate_name, x1, y1, x2, y2, score)
                images[image_path].bbox_list.append(bbox_info)


    ### GENERATE JSON ###
    print('Generating JSON file...')
    for cls in dataset:
        save_class_dir = osp.join(output_dir, cls.name)
        cls.image_paths = sorted(cls.image_paths)

        save_path = osp.join(save_class_dir, '{}.txt'.format(cls.name))

        with open(save_path, 'w') as file:
            result = {}
            for i, image_path in enumerate(cls.image_paths):
                print('[{}/{}] {}'.format(i + 1, len(cls.image_paths), image_path))
                info = images[image_path]
                id = info.id
                name = os.path.split(image_path)[-1]
                width = info.width
                height = info.height

                data = []
                for b in info.bbox_list:
                    if isinstance(b, BBoxClass):
                        xmin = max(min(b.x1, b.x2), 0)
                        ymin = max(min(b.y1, b.y2), 0)
                        xmax = min(max(b.x1, b.x2), info.width)
                        ymax = min(max(b.y1, b.y2), info.height)
                        label = b.label
                        data.append([width, height, class_index[label], xmin, ymin, xmax, ymax])
                    else:
                        raise TypeError("bbox object should be instance of BBoxClass")
                result[name] = data

            json.dump(result, file)
        file.close()

    print('Done.')
