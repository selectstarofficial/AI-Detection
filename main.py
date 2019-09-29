from face_detection_api import FaceDetector
from license_plate_api import LicensePlateDetector
import cv2
import os
import os.path as osp
from glob import glob
import numpy as np
from PIL import Image
from settings import Settings
import utils

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


if __name__ == '__main__':
    settings = Settings()

    if not osp.exists(settings.input_dir):
        raise FileNotFoundError(f"{settings.input_dir} directory not exists.")
    os.makedirs(settings.output_dir, exist_ok=True)

    dataset = utils.get_dataset(settings.input_dir)

    images = {}  # path: ImageClass

    ### FACE DETECTION ###
    print('Detecting faces...')

    print('Creating networks and loading parameters')
    face_api = FaceDetector(settings)
    print('Preparing detector...')
    face_api.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))

    global_img_id = 0
    for cls in dataset:
        save_class_dir = osp.join(settings.output_dir, cls.name)
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
                bbox_info = BBoxClass('face', x1, y1, x2, y2, score)
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

    global_img_id = 0
    for cls in dataset:
        save_class_dir = osp.join(settings.output_dir, cls.name)
        os.makedirs(save_class_dir, exist_ok=True)
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
                bbox_info = BBoxClass('license_plate', x1, y1, x2, y2, score)
                images[image_path].bbox_list.append(bbox_info)

            global_img_id += 1


    ### RENDERING RESULT ###
    print(f"Rendering result... save_img={settings.save_img}")
    if settings.save_img:
        global_img_id = 0
        for cls in dataset:
            save_class_dir = osp.join(settings.output_dir, cls.name)
            os.makedirs(save_class_dir, exist_ok=True)
            cls.image_paths = sorted(cls.image_paths)

            for i, image_path in enumerate(cls.image_paths):
                print('[{}/{}] {}'.format(i + 1, len(cls.image_paths), image_path))

                img_save_path = osp.join(save_class_dir, '{}_detected.jpg'.format(osp.splitext(osp.split(image_path)[-1])[0]))
                img = np.array(Image.open(image_path).convert('RGB'))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                for bbox in images[image_path].bbox_list:
                    if isinstance(bbox, BBoxClass):
                        label, x1, y1, x2, y2, score = str(bbox.label), int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2), float(bbox.score)
                        red, green, blue, thickness = settings.bbox_red, settings.bbox_green, settings.bbox_blue, settings.bbox_thickness
                        cv2.rectangle(img, (x1, y1), (x2, y2), (blue, green, red), thickness=thickness)
                        if settings.show_score:
                            cv2.putText(img, '{}:{}%'.format(label, int(score * 100)), (x1, y1),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=(0, 0, 255), thickness=2)
                        cv2.imwrite(img_save_path, img)
                    else:
                        raise TypeError("bbox object should be instance of BBoxClass")

                global_img_id += 1
