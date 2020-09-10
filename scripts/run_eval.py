from __future__ import division

import argparse
import logging

import torch

from torch_yolo3.models import Darknet
from torch_yolo3.utils import NB_CPUS, load_classes, update_path, parse_data_config
from torch_yolo3.evaluate import evaluate_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(data_config, model_def, weights_path, batch_size,
         img_size, iou_thres, conf_thres, nms_thres, nb_cpu):
    data_config = parse_data_config(update_path(data_config))
    valid_path = update_path(data_config["valid"])
    class_names = load_classes(update_path(data_config["names"]))

    # Initiate model
    model = Darknet(model_def).to(DEVICE)
    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))

    print("Compute mAP...")

    loss, precision, recall, avg_prec, f1, ap_class = evaluate_model(
        model,
        path_data=valid_path,
        iou_thres=iou_thres,
        conf_thres=conf_thres,
        nms_thres=nms_thres,
        img_size=img_size,
        batch_size=batch_size,
        nb_cpu=nb_cpu,
    )

    def _cls_name(c):
        return class_names[c] if c < len(class_names) else 'unknown'

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({_cls_name(c)}) - AP: {avg_prec[i]}")

    print(f"mAP: {avg_prec.mean()}\nmF1: {f1.mean()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou threshold for non-maximum suppression")
    parser.add_argument("--nb_cpu", type=int, default=NB_CPUS,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    main(data_config=opt.data_config, model_def=opt.model_def, weights_path=opt.weights_path,
         batch_size=opt.batch_size, img_size=opt.img_size,
         iou_thres=opt.iou_thres, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres,
         nb_cpu=opt.nb_cpu)

    print("Done :]")
