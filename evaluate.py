from license_plate_api import LicensePlateDetector
from face_detection_api import FaceDetector

from license_plate_api.utils.parse_config import *
from license_plate_api.utils.utils import *
from license_plate_api.utils.datasets import *

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from settings import Settings

import matplotlib.pyplot as plt


def init_valid_text(path):
    root = "license_plate_api/data/custom/labels"
    image_root = "license_plate_api/data/custom/images"
    list = os.listdir(root)
    
    with open(path, "w") as file:
        for name in list:
            real_name = name.split(".")[0]
            if os.path.exists(os.path.join(image_root,"{}.jpg".format(real_name))) and os.path.exists(os.path.join(root, name)):
                file.write("{},{}\n".format(os.path.join(image_root,"{}.jpg".format(real_name)), os.path.join(root, name)))

        file.close()

def evaluate(lplateModel, faceModel, dataloader, iou_thres, conf_thres, nms_thres, img_size, batchsize):
    # TODO: include face model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, original_img, targets, targets_) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        ### DEBUG CODE ###
        #new_img = imgs.clone()
        #new_img = new_img.permute(0,2,3,1).cpu().numpy()[0]
        #plt.imshow(new_img)
        #plt.show()
        ######

        # Extract labels
        labels += targets[:, 1].tolist()
        #print(targets[:, 1].tolist())
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        # 1. detect license plate
        imgs = Variable(imgs.to(device), requires_grad=False)
        lplate_outputs = lplateModel.detect(imgs, mode="eval", threshold=conf_thres) # list[(x1,y1,x2,y2,obj_conf, class_score, class_pred)]
        
        # 2. detect face
        face_outputs = faceModel.detect(original_img) # list[(x1,y1,x2,y2,score)]
        
        # 3. concatenate output
        lplate_outputs = lplate_outputs[0]
        outputs = []
        for output in face_outputs:
            output = list(output)
            output.append(1)
            output.append(1)
            outputs.append(output)
        outputs = torch.Tensor(outputs)
        if lplate_outputs is not None:
            outputs = torch.cat((lplate_outputs, outputs), axis = 0)
        outputs = [outputs]
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    
    return precision, recall, AP, f1, ap_class

if __name__ == '__main__':
    settings = Settings()

    # configure valid dataset
    data_config = parse_data_config(os.path.join(settings.dataset_config_dir, 'custom.data'))
    class_names = load_classes(data_config["names"])
    valid_path = data_config["valid"]
    init_valid_text(valid_path)

    valid_dataset = TotalDataset(valid_path, img_size=settings.license_plate_model_size, augment=False, multiscale=False)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=valid_dataset.collate_fn
    )
    
    license_plate_model = LicensePlateDetector(settings)
    face_model = FaceDetector(settings)

    (precision, recall, AP, f1, ap_class) = evaluate(
        lplateModel=license_plate_model,
        faceModel=face_model,
        dataloader=valid_dataloader,
        iou_thres=0.5,
        conf_thres=0.5,
        nms_thres=0.5,
        img_size=settings.license_plate_model_size,
        batchsize=1
    )
    print("precision: {}\nrecall: {}\nAP: {}\nf1: {}\nap_class: {}".format(precision, recall, AP, f1, ap_class))

