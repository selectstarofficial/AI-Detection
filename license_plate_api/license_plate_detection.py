import cv2
import os
import torch
from .models import *
from .utils.utils import *
from .utils.datasets import *

# TODO Make License Plate Detector -> Refer to detect.py in license_plate_api
class LicensePlateDetector:
    def __init__(self, threshold=0.5, mode="inference"):
        # Model Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        root = os.path.dirname(os.path.realpath(__file__))
        self.weight_path = os.path.join(root, 'yolov3_ckpt_98.pth')
        self.model_cfg = os.path.join(root, 'config', 'yolov3-custom.cfg')
        self.class_pth = os.path.join(root, 'config', 'classes.names')  # TODO add classes names file from cloud
        self.img_size = 416
        self.nms_thres = 0.5
        self.conf_thres = threshold
        self.mode = mode

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

    def detect(self, image):
        mode = self.mode
        # 1. reformat image for input
        image_ = cv2.resize(image, (self.img_size, self.img_size))
        image_ = torch.Tensor(image_).permute(2,0,1).unsqueeze(0).to(self.device)

        # 2. inference
        result = self.model(image_)
        result = non_max_suppression(result, conf_thres=self.conf_thres, nms_thres=self.nms_thres)

        if mode=="inference":
            result = result[0].cpu().numpy()

            detections = []
            # 3. resize to original image size
            if result is not None:
                detections = rescale_boxes(result, self.img_size, image.shape[:2])
                detections = detections[detections[:, -1]==0]
                detections = detections[:,:5]
        else:
            detections = result

        return detections

