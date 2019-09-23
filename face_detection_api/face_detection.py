#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import os.path as osp
import numpy as np
import tensorflow as tf
import cv2
from . import label_map_util

class FaceDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = osp.join(osp.dirname(osp.abspath(__file__)), 'frozen_inference_graph_face.pb')

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = osp.join(osp.dirname(osp.abspath(__file__)), 'face_label_map.pbtxt')

        NUM_CLASSES = 2

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            self.session = tf.Session(graph=detection_graph, config=config)

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

            self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')


    def detect(self, bgr):
        """
        Returns [(x1, y1, x2, y2)]
        """
        threshold = self.threshold
        image_np = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image_np_expanded = np.expand_dims(image_np, axis=0)

        (boxes, scores, classes, num_detections) = self.session.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        height, width, channels = image_np.shape

        bbox_score = []

        for i in range(int(num_detections)):
            if scores[i] < threshold:
                continue

            y1, x1, y2, x2 = boxes[i]
            y1, x1, y2, x2 = y1*height, x1*width, y2*height, x2*width
            
            bbox_score.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), scores[i]))

        return bbox_score

if __name__ == '__main__':
    cap = cv2.VideoCapture("../out/test/0.mp4")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = None

    detector = FaceDetector()

    frame_num = total
    while frame_num:
        frame_num -= 1
        ret, image = cap.read()
        if ret == 0:
            break
        
        faces = detector.detect(image, threshold=0.3)
        print(faces, scores)
        
        for i, face in enumerate(faces):
            color = int(255 * scores[i])
            cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (255,255,255), thickness=4)
            cv2.putText(image, f'{int(face[4]*100)}%', (face[2], face[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

        cv2.imshow('render', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('User Interrupted')
            exit(1)
