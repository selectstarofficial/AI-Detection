import os
import csv
import json
import torch

from license_plate_api.utils.utils import *

local = True
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

def parse_prediction(pred):
#TODO Parse prediction based on format
    return None

if __name__ == '__main__':
    # load output
    with open(output_file_name) as json_file:
        output = json.load(json_file)

    for image_name in output.keys():
        prediction = output[image_name]
        label_path = os.path.join(label_dir, "{}.txt".format(image_name))

        width = prediction[0][0]
        height = prediction[0][1]

        if os.path.exists(label_path):  # if path exist, start evaluation for this image
            label = parse_label(label_path, width, height)
            prediction = parse_prediction(prediction)

