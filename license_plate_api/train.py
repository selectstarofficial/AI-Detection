from __future__ import division

from models import Darknet
from utils.logger import Logger
from utils.torch_logger import TorchLogger
from utils.utils import load_classes, weights_init_normal
from utils.datasets import ListDataset
from test import evaluate
import settings

# from terminaltables import AsciiTable

import os
import time
import datetime

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == "__main__":
    tb_logger = Logger("logs")

    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    train_path = settings.train
    valid_path = settings.valid
    class_names = load_classes(settings.names)

    # Initiate model
    model = Darknet(settings.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if settings.pretrained_weights:
        if settings.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(settings.pretrained_weights))
        else:
            model.load_darknet_weights(settings.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(settings.dataset_prefix, train_path, augment=True, multiscale=settings.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    valid_dataset = ListDataset(settings.dataset_prefix, valid_path, img_size=settings.img_size, augment=False, multiscale=False)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=settings.batch_size, shuffle=False, num_workers=1, collate_fn=valid_dataset.collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)

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
    
    torch_logger = TorchLogger(settings.epochs, len(dataloader))

    for epoch in range(settings.start_epoch, settings.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % settings.gradient_accumulations:
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

        if epoch % settings.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                dataloader=valid_dataloader,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=settings.img_size,
                batch_size=settings.batch_size,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            
            torch_logger.log(epoch+1, batch_i+1, etc_str='', val_precision=precision.mean(), val_recall=recall.mean(), val_mAP=AP.mean(), val_f1=f1.mean())
            
            tb_logger.list_of_scalars_summary(evaluation_metrics, epoch)
        
        if epoch % settings.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
