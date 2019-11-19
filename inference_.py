from detection_api import Detector
from detection_api.utils.parse_config import *
from detection_api.utils.utils import *
from face_detection_api.face_detection import FaceDetector
from converge import converge_resuts
import shutil
import csv
import os
import os.path as osp
import numpy as np
from PIL import Image
from settings import Settings
import utils

"""
This code inference validation sets for mAP calculation
"""

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


class ImageData:
    def __init__(self, name):
        self.name = name
        self.image_paths=[]

def get_dataset(root, valid_text):
    dataset = [ImageData('custom')]
    with open(valid_text) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            image_dir = os.path.join(root,row[0].split("/")[-1])
            dataset[0].image_paths.append(image_dir)

    return dataset

is_server = True
if is_server:
    input_root = "../../dataset/bbox/custom/images/"
    valid_text = "../../dataset/bbox/custom/valid.txt"
    output_dir = "../../dataset/bbox/custom/output/"
    label_dir = "../../dataset/bbox/custom/labels_xyxy/"
    initial_save_path = "../../dataset/bbox/temp/"
    map_dir = "../../map_api/input"
else:
    input_root = "../dataset/bbox/custom/images/"
    valid_text = "../dataset/bbox/custom/valid.txt"
    output_dir = "../dataset/bbox/custom/output/"
    label_dir = "../dataset/bbox/custom/labels_xyxy/"
    initial_save_path = "../dataset/bbox/temp/"
    map_dir = "../map_api/input"


if __name__ == '__main__':
    settings = Settings()

    if not osp.exists(settings.input_dir):
        raise FileNotFoundError(f"{settings.input_dir} directory not exists.")
    os.makedirs(settings.output_dir, exist_ok=True)

    root = os.path.dirname(os.path.realpath(__file__))
    data_config = parse_data_config(os.path.join(root, settings.config_path))
    classes = load_classes(os.path.join(root, "detection_api", data_config["names"]))
    if not osp.exists(settings.input_dir):
        raise FileNotFoundError(f"{settings.input_dir} directory not exists.")
    os.makedirs(settings.output_dir, exist_ok=True)

    dataset = utils.get_dataset(settings.input_dir)

    images = {}  # path: ImageClass

    ### DETECTION ###
    print('Start Detection...')

    print('Creating networks and loading parameters')
    license_plate_api = Detector(settings)
    print('Preparing detector...')
    license_plate_api.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))

    global_img_id = 0
    for cls in dataset:
        save_class_dir = osp.join(settings.output_dir, cls.name)
        cls.image_paths = sorted(cls.image_paths)

        if not os.path.exists(save_class_dir):
            os.mkdir(save_class_dir)

        for i, image_path in enumerate(cls.image_paths):
            print('[{}/{}] {}'.format(i + 1, len(cls.image_paths), image_path))

            img = np.array(Image.open(image_path).convert('RGB'))

            img_height, img_width = img.shape[0:2]

            # Register image info
            img_info = ImageClass(global_img_id, image_path, img_width, img_height)

            # get bbox result
            bboxes, labels = license_plate_api.detect(img)  # (x1, y1, x2, y2, score)
            bboxes = utils.filter_too_big(bboxes, settings.max_size_ratio, img_width, img_height)

            for idx, bbox in enumerate(bboxes):
                x1, y1, x2, y2, score = bbox
                label = classes[int(labels[idx])]
                bbox_info = BBoxClass(label, x1, y1, x2, y2, score)
                img_info.bbox_list.append(bbox_info)
            images[image_path] = img_info

        global_img_id += 1
    del license_plate_api

    ### DETECT BIG FACE ###
    print('Preparing Large Face Detector...')
    face_detector = FaceDetector(settings)

    big_face_images = {}
    global_img_id = 0
    for cls in dataset:
        cls.image_paths = sorted(cls.image_paths)
        for i, image_path in enumerate(cls.image_paths):
            print('[{}/{}] {}'.format(i + 1, len(cls.image_paths), image_path))

            img = np.array(Image.open(image_path).convert('RGB'))

            img_height, img_width = img.shape[0:2]

            # Register image info
            img_info = ImageClass(global_img_id, image_path, img_width, img_height)

            # get bbox result
            bboxes = face_detector.detect(img)  # (x1, y1, x2, y2, score)
            bboxes = utils.filter_too_big(bboxes, settings.max_size_ratio, img_width, img_height)

            for idx, bbox in enumerate(bboxes):
                x1, y1, x2, y2, score = bbox
                label = classes[0]  # Face
                bbox_info = BBoxClass(label, x1, y1, x2, y2, score)
                img_info.bbox_list.append(bbox_info)
            big_face_images[image_path] = img_info
        global_img_id += 1
    del face_detector

    ### FILTER FACE DETECTION RESULTS ###
    print('Converging results...')
    images = converge_resuts(images, big_face_images)


    ### GENERATE RESULT FILE ###
    print("Start Writing Result File...")
    if os.path.exists(initial_save_path):
        shutil.rmtree(initial_save_path)
    os.mkdir(initial_save_path)

    img_len = len(images)
    for idx, img_path in enumerate(images.keys()):
        img_name = img_path.split('/')[-1]
        result_save_path = os.path.join(initial_save_path, '{}.txt'.format(img_name))

        print("[{}/{}] {}".format(idx+1, img_len, result_save_path))
        with open(result_save_path, 'w') as file:
            for bbox in images[img_path].bbox_list:
                result_string = '{label} {score} {x1} {y1} {x2} {y2}\n'.format(
                    label=bbox.label, score=bbox.score, x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2)
                file.write(result_string)
            file.close()


    ### PREPARE MAP CALCULATION ###
    # Clear map-api input folder
    map_result_dir = os.path.join(map_dir, 'detection-results')
    map_label_dir = os.path.join(map_dir, 'ground-truth')
    if os.path.exists(map_result_dir):
        shutil.rmtree(map_result_dir)
    if os.path.exists(map_label_dir):
        shutil.rmtree(map_label_dir)
    os.mkdir(map_result_dir)
    os.mkdir(map_label_dir)

    # Copy to map-api input folder
    for result_name in os.listdir(initial_save_path):
        result_origin_path = os.path.join(initial_save_path, result_name)
        result_dest_path = os.path.join(map_result_dir, result_name)

        label_origin_path = os.path.join(label_dir, result_name)
        label_dest_path = os.path.join(map_label_dir, result_name)

        shutil.copyfile(result_origin_path, result_dest_path)
        shutil.copyfile(label_origin_path, label_dest_path)
