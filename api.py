import cv2
import numpy as np
from face_detection_api import FaceDetector
from license_plate_api import LicensePlateDetector
from settings import Settings

class IntegratedApi:
    def __init__(self, settings: Settings):
        self.face_detector = FaceDetector(settings)
        self.license_plate_detector = LicensePlateDetector(settings)

    def detect(self, rgb_image): # TODO when finished, update documentaion
        """ Detect face and license_plate
        :param image: RGB with 0~255
        :return: Dict("face" : list[(x1, y1, x2, y2, score)], "license_plate": list[(x1, y1, x2, y2, score)]) : each coordinates are real pixel values
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = {
                    "face" : [],
                    "license_plate" : []
                }
        
        # 1. infer face
        face_result = self.face_detector.detect(rgb_image)
        result["face"] = face_result

        # 2. infer license_plate
        license_plate_result = self.license_plate_detector.detect(rgb_image)
        result["license_plate"] = license_plate_result
        
        return result

    def mask(self, image, settings: Settings):
        """
        Outputs masked image based on result
        :param image: numpy_array(width, height, 3)
        :param result: Dict("face" : list[(x1, y1, x2, y2, score)], "license_plate": list[(x1, y1, x2, y2, score)]) : each coordinates are real pixel values
        :return: numpy_array(width, height, 3) : face and license_plate blurred 
        """
        detections = self.detect(image)

        image = image.astype('float32')
        image /= 255.0

        for key in detections.keys():
            for detection in detections[key]:
                image = self.blur(image, detection, settings)

        return image

    def blur(self, image, detection, settings: Settings):
        """
        Outputs blurred image based on single detection boundary
        :param image: RGB numpy_array(width, height, 3) : input image
        :param detection: list[(x1, y1, x2, y2, score)] : single detection boundary
        :return: numpy_array(width, height, 3) : image blurred based on single detection boundary
        """
        x1, y1, x2, y2, score = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        width = x2 - x1
        height = y2 - y1

        temp = image[y1:y2, x1:x2]

        if (width*height)>100 and width>10 and height>10:  # TODO: remove constant values and replace settings.min_size_pixel, settings.max_size_pixel
            temp_ = np.zeros_like(temp) 
            image[y1:y2, x1:x2] = temp_

        image = cv2.rectangle(image, (x1, y1), (x2, y2),
                              (settings.bbox_red,settings.bbox_green,settings.bbox_blue), settings.bbox_thickness)

        return image

