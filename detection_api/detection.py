import cv2
import os
import torch
from .models import Darknet
from .utils.parse_config import *
from .utils.utils import load_classes, non_max_suppression, rescale_boxes
from .utils.datasets import pad_to_square, resize
import torchvision.transforms as transforms
from PIL import Image

class Detector:
    def __init__(self, settings):
        # Model Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        root = os.path.dirname(os.path.realpath(__file__))
        self.weight_path = os.path.join(root, 'model','yolov3_ckpt_54.pth')
        self.model_cfg = os.path.join(root, 'model', 'yolov3-custom.cfg')

        config_path = os.path.join(root, 'model', 'custom.data')
        self.data_config = parse_data_config(config_path)

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
        self.classes = load_classes(os.path.join(root, self.data_config["names"]))

    def preprocess(self, rgb_image):
        # Extract image as PyTorch tensor
        img_pil = Image.fromarray(rgb_image)
        img = transforms.ToTensor()(img_pil)
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)
        # Create single batch
        imgs = img.unsqueeze(0)

        return imgs

    def detect(self, rgb_image):
        """

        :param image: RGB with 0~255
        :return: [(x1, y1, x2, y2, score)]
        """
        imgs = self.preprocess(rgb_image)
        input_imgs = imgs.to(self.device)

        with torch.no_grad():
            detections = self.model(input_imgs)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            detections = detections[0]  # for single batch

        if detections is None:
            return []

        detections = rescale_boxes(detections, self.img_size, rgb_image.shape[:2])
        detections = detections.detach().cpu().numpy()

        bbox_result = []
        label_result = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            bbox_result.append((x1, y1, x2, y2, conf))
            label_result.append(cls_pred)

        return bbox_result, label_result


    # def detect(self, image, threshold=0.5):
    #     mode = self.mode
    #     # 1. reformat image for input
    #     if mode=="inference":
    #         image_ = cv2.resize(image, (self.img_size, self.img_size))
    #         image_ = torch.Tensor(image_).permute(2,0,1).unsqueeze(0).to(self.device)
    #     else:
    #         image_ = image.to(self.device)
    #
    #     # 2. inference
    #     result = self.model(image_)
    #     result = non_max_suppression(result, conf_thres=threshold, nms_thres=self.nms_thres)
    #
    #     if mode=="inference":
    #         result = result[0].cpu().numpy()
    #
    #         detections = []
    #         # 3. resize to original image size
    #         if result is not None:
    #             detections = rescale_boxes(result, self.img_size, image.shape[:2])
    #             detections = detections[detections[:, -1]==0]
    #             detections = detections[:,:5]
    #     else:
    #         detections = result
    #
    #     return detections
    #

