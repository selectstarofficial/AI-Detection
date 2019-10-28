from face_detection_api import FaceDetector
from license_plate_api import LicensePlateDetector
import csv
import os
import os.path as osp
import numpy as np
from PIL import Image
from settings import Settings
import utils
import shutil

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

local = True
if local:
    input_root = "/Users/litcoderr/Desktop/Projects/dataset/input/custom/"
    valid_text = "/Users/litcoderr/Desktop/Projects/dataset/valid.txt"
    output_dir = "/Users/litcoderr/Desktop/Projects/dataset/output"
    label_dir = "/Users/litcoderr/Desktop/Projects/dataset/label"
    map_dir = "/Users/litcoderr/Desktop/Projects/map/input"
else:
    input_root = "./license_plate_api/data/custom/images/dataset1"
    valid_text = "./license_plate_api/data/custom/valid.txt"
    output_dir = "./license_plate_api/data/custom/output/"
    label_dir = "./license_plate_api/data/custom/labels"
    map_dir = "../map_api/input"



class_index = {
    "license_plate" : 0,
    "face" : 1
}

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

if __name__ == '__main__':
    settings = Settings()

    dataset = get_dataset(input_root, valid_text)

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
    print('Generating Label Text file...')
    for cls in dataset:
        save_class_dir = osp.join(output_dir, cls.name)
        cls.image_paths = sorted(cls.image_paths)

        for i, image_path in enumerate(cls.image_paths):
            image_name = image_path.split("/")[-1]
            file_name = os.path.join(save_class_dir, "{}.txt".format(image_name))
            print('[{}/{}] {}'.format(i + 1, len(cls.image_paths), image_path))
            with open(file_name, "w") as file:
                info = images[image_path]
                id = info.id
                name = os.path.split(image_path)[-1]
                width = info.width
                height = info.height

                data = ""
                for b in info.bbox_list:
                    if isinstance(b, BBoxClass):
                        xmin = max(min(b.x1, b.x2)/width, 0)
                        ymin = max(min(b.y1, b.y2)/height, 0)
                        xmax = min(max(b.x1, b.x2)/width, info.width)
                        ymax = min(max(b.y1, b.y2)/height, info.height)
                        label = class_index[b.label]
                        score = b.score
                        line = "{label} {score} {xmin} {ymin} {xmax} {ymax}\n".format(label=label, score=score, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                        data += line
                    else:
                        raise TypeError("bbox object should be instance of BBoxClass")
                file.write(data)
                file.close()
    print('Label File Dest: {}'.format(save_class_dir))


    ### Move result file to MAP api folder for performance checking
    result_dest = os.path.join(map_dir, "detection-results")
    label_dest = os.path.join(map_dir, "ground-truth")
    # 1. remove existing files
    for file in os.listdir(result_dest):
        path = os.path.join(result_dest, file)
        os.unlink(path)

    for file in os.listdir(label_dest):
        path = os.path.join(label_dest, file)
        os.unlink(path)

    # 2. Move newly created results and labels
    file_names = [name for name in os.listdir(save_class_dir) if name.endswith(".txt")]

    for name in file_names:
        original_path = os.path.join(save_class_dir, name)
        dest_path = os.path.join(result_dest, name)
        shutil.copy(original_path, dest_path)
        print("[{}] -> [{}]".format(original_path, dest_path))

        original_path = os.path.join(label_dir, name)
        dest_path = os.path.join(label_dest, name)
        shutil.copy(original_path, dest_path)
        print("[{}] -> [{}]".format(original_path, dest_path))

    print('Done.')
