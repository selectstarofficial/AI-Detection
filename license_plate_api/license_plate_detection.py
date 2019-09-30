import cv2
import os
import torch
from .models import Darknet
from .utils.utils import load_classes, non_max_suppression, rescale_boxes
from .utils.datasets import pad_to_square, resize
from settings import Settings
import torchvision.transforms as transforms
from PIL import Image

# TODO Make License Plate Detector -> Refer to detect.py in license_plate_api
class LicensePlateDetector:
    def __init__(self, settings: Settings):
        # Model Config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        root = os.path.dirname(os.path.realpath(__file__))
        self.weight_path = os.path.join(root, 'yolov3_ckpt_98.pth')
        self.model_cfg = os.path.join(settings.license_plate_model_config_dir, 'yolov3-custom.cfg')
        self.class_pth = os.path.join(settings.license_plate_model_config_dir, 'classes.names')
        self.img_size = settings.license_plate_model_size
        self.conf_thres = settings.license_plate_threshold
        self.nms_thres = 0.4

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
        self.classes = load_classes(self.class_pth)

    def detect(self, image, mode="inference", threshold=0.5):
        # 1. reformat image for input
        if mode=="inference":
            image_ = cv2.resize(image, (self.img_size, self.img_size))
            image_ = torch.Tensor(image_).permute(2,0,1).unsqueeze(0).to(self.device)
        else:
            image_ = image.to(self.device)

        return image_

