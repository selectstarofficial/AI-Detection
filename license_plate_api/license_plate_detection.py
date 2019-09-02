import os
import torch
from license_plate_api.models import *
from license_plate_api.utils.utils import *
from license_plate_api.utils.datasets import *

# TODO Make License Plate Detector -> Refer to detect.py in license_plate_api
class LicensePlateDetector:
    def __init__(self):
        # Model Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        root = os.path.dirname(os.path.realpath(__file__))
        self.weight_path = os.path.join(root, 'checkpoints', 'yolov3_ckpt_50.pth')
        self.model_cfg = os.path.join(root, 'config', 'yolov3-custom.cfg')
        self.class_pth = os.path.join(root, 'config', 'classes.names')  # TODO add classes names file from cloud
        self.img_size = 416

        # Load Model
        self.model = Darknet(self.model_cfg, img_size=self.img_size).to(self.device)
        if self.weight_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.weight_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
        self.model.eval()

        # Load Classes Info
        # TODO load classes path
        self.classes = load_classes()

    def detect(self):
        pass

