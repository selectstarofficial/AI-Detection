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

def init_valid_text(path):
    root = "license_plate_api/data/custom_with_face/labels"
    image_root = "license_plate_api/data/license_plate_dataset/images"
    list = os.listdir(root)
    if not os.path.exists(path):
        with open(path, "w") as file:
            for name in list:
                real_name = name.split(".")[0]
                file.write("{},{}\n".format(os.path.join(image_root,real_name,".jpg"), os.path.join(root, name)))

            file.close()

def evaluate(lplateModel, faceModel, dataloader, iou_thres, conf_thres, nms_thres, img_size, batchsize):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor).to(device), requires_grad=False)

        outputs = lplateModel.detect(imgs, threshold=conf_thres)
        outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="total_dataset_config/", help="path to data config file")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=704, help="size of each image dimension")
    args = parser.parse_args()

    # configure valid dataset
    data_config = parse_data_config(os.path.join(args.data_config, "custom.data"))
    class_names = load_classes(data_config["names"])
    valid_path = data_config["valid"]
    init_valid_text(valid_path)

    valid_dataset = ListDataset(valid_path, img_size=args.img_size, augment=False, multiscale=False)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=args.n_cpu, collate_fn=valid_dataset.collate_fn
    )


    license_plate_model = LicensePlateDetector(mode="eval")
    face_model = FaceDetector()

    (precision, recall, AP, f1, ap_class) = evaluate(
        lplateModel=license_plate_model,
        faceModel=face_model,
        dataloader=valid_dataloader,
        iou_thres=0.5,
        conf_thres=0.5,
        nms_thres=0.5,
        img_size=args.img_size,
        batchsize=1
    )

