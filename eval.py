import os
import csv
import json
import torch
import numpy as np

from license_plate_api.utils.utils import *

local = True
labels = [0, 1]
if local:
    output_file_name = "/Users/litcoderr/Desktop/Projects/dataset/output/custom/custom.txt"
    label_dir = "/Users/litcoderr/Desktop/Projects/dataset/label"
else:
    output_file_name = "./license_plate_api/data/custom/output/dataset1/dataset1.txt"
    label_dir = "./license_plate_api/data/custom/labels"

def parse_label(path, width, height):
    """
    return parsed label tensor
    :param path: text file path
    :return: torch Tensor of label
    """
    with open(path) as file:
        reader = csv.reader(file, delimiter=' ')
        result = []
        for row in reader:
            label = int(row[0])
            x = float(row[1])
            y = float(row[2])
            w = float(row[3])
            h = float(row[4])
            result.append([0, label, x, y, w, h])

        result = torch.Tensor(result)
        result[:, 2:] = xywh2xyxy(result[:, 2:])
        result[:, 2] *= width
        result[:, 4] *= width
        result[:, 3] *= height
        result[:, 5] *= height

    return result

def parse_prediction(pred):  # format [Tensor([x1,y1,x2,y2,1,label])]
    data = []
    for elem in pred:
        label = elem[2]
        score = elem[3]
        xmin = elem[4]
        ymin = elem[5]
        xmax = elem[6]
        ymax = elem[7]
        data.append([xmin, ymin, xmax, ymax, score, label])
    data = [torch.Tensor(data)]
    return data

if __name__ == '__main__':
    # load output
    with open(output_file_name) as json_file:
        output = json.load(json_file)

    print("Start Computing Metric")
    # Compute Metric
    metric = []
    for image_name in output.keys():
        prediction = output[image_name]
        label_path = os.path.join(label_dir, "{}.txt".format(image_name))

        if len(prediction) != 0:
            width = prediction[0][0]
            height = prediction[0][1]

            if os.path.exists(label_path):  # if path exist, start evaluation for this image
                label = parse_label(label_path, width, height)
                prediction = parse_prediction(prediction)

                metric += get_batch_statistics(prediction, label, iou_threshold=0.5)

    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metric))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    print("precision: {}\nrecall: {}\nAP: {}\nf1: {}\nap_class: {}".format(precision, recall, AP, f1, ap_class))
    print("Finished")
