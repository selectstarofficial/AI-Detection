from __future__ import division

from models import *
from utils.logger import *
from utils.torch_logger import TorchLogger
from utils.utils import *
from utils.datasets import *
from test import evaluate

# from terminaltables import AsciiTable

import os
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=12, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=4, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="model/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="model/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", default='checkpoints/yolov3_ckpt_99.pth', type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=704, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=True, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    
    parser.add_argument("--start_epoch", type=int, default=0, help="start epoch number. default=0")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate. default=1e-3")
    parser.add_argument("--amsgrad", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether use amsgrad")
    parser.add_argument("--device", type=str, default="cuda", help="cuda:0 cuda:1 etc...")
    opt = parser.parse_args()
    print(opt)

    tb_logger = Logger("logs")

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    valid_dataset = ListDataset(valid_path, img_size=opt.img_size, augment=False, multiscale=False)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1, collate_fn=valid_dataset.collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, amsgrad=opt.amsgrad)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    
    torch_logger = TorchLogger(opt.epochs, len(dataloader))

    for epoch in range(opt.start_epoch, opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            
            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            
            torch_logger.log(epoch+1, batch_i+1, etc_str=f'ETA {time_left}', loss=loss.item())
            
            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                tb_logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                dataloader=valid_dataloader,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            
            torch_logger.log(epoch+1, batch_i+1, etc_str='', val_precision=precision.mean(), val_recall=recall.mean(), val_mAP=AP.mean(), val_f1=f1.mean())
            
            tb_logger.list_of_scalars_summary(evaluation_metrics, epoch)
        
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
