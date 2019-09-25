from license_plate_api import LicensePlateDetector
from face_detection_api import FaceDetector

from license_plate_api.utils.parse_config import *
from license_plate_api.utils.utils import *
from license_plate_api.utils.datasets import *

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

def init_valid_text(path):
    root = "license_plate_api/data/custom_with_face/labels"
    list = os.listdir(root)
    if not os.path.exists(path):
        with open(path, "w") as file:
            for name in list:
                file.write("{}\n".format(os.path.join(root, name)))

            file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="total_dataset_config/", help="path to data config file")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    args = parser.parse_args()

    # configure valid dataset
    data_config = parse_data_config(os.path.join(args.data_config, "custom.data"))
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    init_valid_text(data_config["valid"])

    license_plate_model = LicensePlateDetector(mode="eval")
    # face_model = FaceDetector()

