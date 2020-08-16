"""
Train a custom model

Example
-------

python run_train.py \
    --epochs 10 \
    --batch_size 2 \
    --model_def ./config/yolov3-tiny.cfg \
    --path_output ./outputs \
    --data_config ./config/custom.data \
    --img_size 416 \
    --multiscale 0.1 \
    --augment hflip

"""

from __future__ import division

import argparse
import inspect
import logging
import os
import shutil
import sys

import numpy as np
import tqdm
import torch
import torch_optimizer
import yaml
from terminaltables import AsciiTable
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path += [os.path.abspath('..'), os.path.abspath('.')]
from torch_yolo3.models import Darknet, weights_init_normal
from torch_yolo3.datasets import ListDataset
from torch_yolo3.logger import Logger
from torch_yolo3.utils import NB_CPUS, load_classes, update_path, parse_data_config, xywh2xyxy
from torch_yolo3.evaluate import evaluate_model, non_max_suppression, get_batch_statistics, agreg_stat_per_class

METRICS = [
    "grid_size", "loss",
    "x", "y", "w", "h",
    "cls", "cls_acc",
    "recall50", "recall75", "precision",
    "conf", "conf_obj", "conf_noobj",
]
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(data_config, model_def, trained_weights, augment, multiscale,
         img_size, grad_accums, evaluation_interval, checkpoint_interval,
         batch_size, epochs, path_output, nb_cpu, amp):
    path_output = update_path(path_output)
    os.makedirs(path_output, exist_ok=True)

    shutil.copy(data_config, os.path.join(path_output, os.path.basename(data_config)))
    _, _, _, local_vars = inspect.getargvalues(inspect.currentframe())
    args = {arg: local_vars[arg] for arg in inspect.getfullargspec(main).args}
    with open(os.path.join(path_output, 'script-config.yaml'), 'w') as fp:
        yaml.dump(dict(args), fp)

    logger = Logger(os.path.join(path_output, "logs"))

    # Get data configuration
    data_config = parse_data_config(update_path(data_config))
    train_path = update_path(data_config["train"])
    valid_path = update_path(data_config["valid"])
    class_names = load_classes(update_path(data_config["names"]))

    # Initiate model
    assert os.path.isfile(model_def), 'missing: %s' % model_def
    model = Darknet(update_path(model_def)).to(DEVICE)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if trained_weights:
        assert os.path.isfile(trained_weights), 'missing: %s' % trained_weights
        if trained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(trained_weights))
        else:
            model.load_darknet_weights(trained_weights)

    augment = dict(zip(augment, [True] * len(augment))) if augment else {}
    augment["scaling"] = multiscale
    # Get dataloader
    assert os.path.isfile(train_path), 'missing: %s' % train_path
    dataset = ListDataset(train_path, augment=augment, img_size=img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nb_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    for param in model.parameters():
        param.requires_grad = True

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch_optimizer.RAdam(model.parameters(), lr=0.0001)
    # optimizer = torch_optimizer.Yogi(model.parameters(), lr=0.0001)
    # optimizer = torch_optimizer.AdaBound(model.parameters(), lr=0.0001)

    # Creates once at the beginning of training
    scaler = torch.cuda.amp.GradScaler() if amp else None

    for epoch in tqdm.tqdm(range(epochs), desc='Training epoch'):
        model.train()
        # start_time = time.time()
        pbar_batch = tqdm.tqdm(total=len(dataloader))
        train_metrics = []
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            model, batch_metric = training_batch(dataloader, model, optimizer, epochs,
                                                 epoch, batch_i, imgs, targets, grad_accums, img_size, scaler)
            loss = batch_metric['loss']
            pbar_batch.set_description("training batch loss=%.5f" % loss)
            train_metrics.append(batch_metric)
            pbar_batch.update()
        pbar_batch.close()
        pbar_batch.clear()

        train_metrics_all = {m: [] for m in train_metrics[0]}
        _ = [train_metrics_all[k].append(bm[k]) for bm in train_metrics for k in bm]
        logger.list_scalars_summary([(k, np.mean(train_metrics_all[k]))
                                     for k in train_metrics_all], step=epoch, phase='train')

        if epoch % evaluation_interval == 0:
            assert os.path.isfile(valid_path), 'missing: %s' % valid_path
            evaluate_epoch(model, valid_path, img_size, batch_size, epoch, class_names, logger, nb_cpu)

        if epoch % checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(path_output, "yolov3_ckpt_%05d.pth" % epoch))


def training_batch(dataloader, model, optimizer, epochs, epoch, batch_i, imgs, targets, grad_accums,
                   img_size, scaler=None, verbose=0):
    batches_done = len(dataloader) * epoch + batch_i

    imgs = Variable(imgs.to(DEVICE))
    targets = Variable(targets.to(DEVICE), requires_grad=False)

    if scaler:
        # Casts operations to mixed precision
        with torch.cuda.amp.autocast():
            loss, outputs = model(imgs, targets)
            if loss:  # avoid 0 loss
                # Scales the loss, and calls backward() to create scaled gradients
                scaler.scale(loss).backward()
    else:
        loss, outputs = model(imgs, targets)
        if loss:  # avoid 0 loss
            loss.backward()

    # Accumulates gradient before each step
    if batches_done % grad_accums == 0:
        if scaler:
            # Unscales gradients and calls or skips optimizer.step()
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    labels = targets[:, 1].tolist()
    # Rescale target
    targets[:, 2:] = xywh2xyxy(targets[:, 2:]) * img_size
    outputs = non_max_suppression(outputs.cpu(), conf_thres=CONF_THRESHOLD, nms_thres=NMS_THRESHOLD)
    sample_metrics = get_batch_statistics(outputs, targets.cpu(), iou_threshold=IOU_THRESHOLD)
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))] \
        if sample_metrics else [np.array([])] * 3
    precision, recall, avg_prec, f1, _ = agreg_stat_per_class(true_positives, pred_scores, pred_labels, labels)

    if verbose:
        # Log progress
        log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, len(dataloader))

        metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

        # Log metrics at each YOLO layer
        for i, metric in enumerate(METRICS):
            metric_table, tboard_log = metrics_export(metric_table, model, loss, metric)
            # logger.list_scalars_summary(tboard_log, batches_done)

        log_str += AsciiTable(metric_table).table
        log_str += f"\nTotal loss {loss.item()}"

        # Determine approximate time left for epoch
        # epoch_batches_left = len(dataloader) - (batch_i + 1)
        # time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
        # log_str += f"\n---- ETA {time_left}"
        print(log_str)

    model.seen += imgs.size(0)
    eval_metrics = dict(_format_metrics(loss, precision, recall, avg_prec, f1))
    return model, eval_metrics


def _format_metrics(loss, precision, recall, avg_prec, f1):
    return [
        ("loss", loss.mean().item() if loss else 0.),
        ("precision", precision.mean() if precision else 0.),
        ("recall", recall.mean() if recall else 0.),
        ("mAP", avg_prec.mean() if avg_prec else 0.),
        ("f1", f1.mean() if f1 else 0.),
    ]


def evaluate_epoch(model, valid_path, img_size, batch_size, epoch, class_names, logger, nb_cpu):
    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    loss, precision, recall, avg_prec, f1, ap_class = evaluate_model(
        model,
        path_data=valid_path,
        iou_thres=IOU_THRESHOLD,
        conf_thres=CONF_THRESHOLD,
        nms_thres=NMS_THRESHOLD,
        img_size=img_size,
        batch_size=batch_size,
        nb_cpu=nb_cpu,
    )

    def _cls_name(c):
        return class_names[c] if c < len(class_names) else 'unknown'

    eval_metrics = _format_metrics(loss, precision, recall, avg_prec, f1)
    logger.list_scalars_summary(eval_metrics, epoch, phase='val')

    # Print class APs and mAP
    ap_table = [["Index", "Class name", "Avg. Prec.", "F1"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, _cls_name(c), "%.5f" % avg_prec[i], "%.5f" % f1[i]]]
    print(AsciiTable(ap_table).table)
    print(f"#{epoch} \t>>>> mAP: {avg_prec.mean()} \t>>>> mF1: {f1.mean()}")
    return avg_prec.mean()


def metrics_export(metric_table, model, loss, metric):
    formats = {m: "%.6f" for m in METRICS}
    formats["grid_size"] = "%2d"
    formats["cls_acc"] = "%.2f%%"
    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
    metric_table += [[metric, *row_metrics]]

    # Tensorboard logging
    tboard_log = []
    for j, yolo in enumerate(model.yolo_layers):
        for name, metric in yolo.metrics.items():
            if name != "grid_size":
                tboard_log += [(f"{name}_{j + 1}", metric)]
    tboard_log += [("loss", loss.item())]
    return metric_table, tboard_log


def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--grad_accums", type=int, default=32, help="number of gradient accumulations before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--path_output", type=str, default="output", help="path to output folder")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path_img to data config file")
    parser.add_argument("--trained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale", type=float, default=0.1, help="allow for multi-scale training")
    parser.add_argument("--augment", type=str, default=None, nargs="*", help="allow for augmentation",
                        choices=['hflip', 'vflip'])
    parser.add_argument("--nb_cpu", type=int, default=NB_CPUS,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--amp", type=bool, default=1, help="use automatic mixed precision")
    opt = parser.parse_args()
    print(opt)

    main(data_config=opt.data_config, model_def=opt.model_def, trained_weights=opt.trained_weights,
         augment=opt.augment, multiscale=opt.multiscale, img_size=opt.img_size, grad_accums=opt.grad_accums,
         evaluation_interval=opt.evaluation_interval, checkpoint_interval=opt.checkpoint_interval,
         batch_size=opt.batch_size, epochs=opt.epochs, path_output=opt.path_output, nb_cpu=opt.nb_cpu, amp=opt.amp)
    print("Done :]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_cli()
