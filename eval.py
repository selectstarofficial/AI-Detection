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
    output_file_name = "./license_plate_api/data/custom/output/input.txt"
    label_dir = ""

def parse_label(path):
    """
    return parsed label tensor
    :param path: text file path
    :return: torch Tensor of label
    """


if __name__ == '__main__':
    # load output
    with open(output_file_name) as json_file:
        output = json.load(json_file)

    for image_name in output.keys():
        image = output[image_name]
        label_path = os.path.join(label_dir, "{}.txt".format(image_name))

        if os.path.exists(label_path):  # if path exist, start evaluation for this image
            label = parse_label(label_path)

