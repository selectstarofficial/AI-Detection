import cv2
import numpy as np
from face_detection_api import FaceDetector
from license_plate_api import LicensePlateDetector

class IntegratedApi:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.license_plate_detector = LicensePlateDetector()


    def detect(self, image): # TODO when finished, update documentaion
        """ Detect face and license_plate
        :param image: numpy_array(width, height, 3)
        :return: Dict("face" : list[(x1, y1, x2, y2, score)], "license_plate": list[(x1, y1, x2, y2, score)]) : each coordinates are real pixel values
        """
        result = {
                    "face" : [],
                    "license_plate" : []
                }
        
        # 1. infer face
        face_result = self.face_detector.detect(image)
        result["face"] = face_result

        # 2. infer license_plate
        license_plate_result = self.license_plate_detector.detect(image)
        result["license_plat"] = license_plate_result
        
        return result


    def mask(self, image, show_boundary=False):
        """
        Outputs masked image based on result
        :param image: numpy_array(width, height, 3)
        :param result: Dict("face" : list[(x1, y1, x2, y2, score)], "license_plate": list[(x1, y1, x2, y2, score)]) : each coordinates are real pixel values
        :return: numpy_array(width, height, 3) : face and license_plate blurred 
        """
        detections = self.detect(image)
        
        image = image/255
        image = image.astype('float32')
        for key in detections.keys():
            for detection in detections[key]:
                image = self.blur(image, detection, show_boundary=show_boundary)

        return image


    def blur(self, image, detection, show_boundary):
        """
        Outputs blurred image based on single detection boundary
        :param image: numpy_array(width, height, 3) : input image
        :param detection: list[(x1, y1, x2, y2, score)] : single detection boundary
        :return: numpy_array(width, height, 3) : image blurred based on single detection boundary
        """
        width = int(detection[2] - detection[0])
        height = int(detection[3] - detection[1])
        temp = image[int(detection[1]):int(detection[1]+height),int(detection[0]):int(detection[0]+width)]
        if (width*height)>100 and width>10 and height>10:
            temp_ = np.zeros_like(temp) 
            image[int(detection[1]):int(detection[1]+height),int(detection[0]):int(detection[0]+width)] = temp_
        
        if show_boundary:
           image = cv2.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (255,0,0), 2)

        return image

