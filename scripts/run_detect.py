"""Run detection on custom image folder.

Example
-------
python3 run_detect.py \
    --image_folder ./data_images \
    --model_def ./config/yolov3.cfg \
    --weights_path ./weights/yolov3.weights \
    --output_folder ./output \
    --class_path ./data/coco.names \
    --batch_size 5 \
    --nb_cpu 9

"""

from __future__ import division

import argparse
import logging
import os
import sys
from functools import partial

import numpy as np
import torch
import tqdm
from PIL import Image

from pathos.multiprocessing import ProcessPool
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path += [os.path.abspath('..'), os.path.abspath('.')]
from torch_yolo3.visual import draw_bboxes, export_img_figure, create_img_figure, get_colors
from torch_yolo3.models import Darknet
from torch_yolo3.datasets import ImageFolder
from torch_yolo3.utils import NB_CPUS, load_classes
from torch_yolo3.evaluate import non_max_suppression


def main(image_folder, model_def, weights_path, class_path, output_folder, img_size,
         conf_thres, nms_thres, batch_size, nb_cpu):
    # use GPU if it is possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # at least one cpu has to be set
    nb_cpu = max(1, nb_cpu)
    # prepare the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)

    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode

    img_folder = ImageFolder(image_folder, img_size=img_size)
    dataloader = DataLoader(img_folder, batch_size=batch_size, shuffle=False, num_workers=nb_cpu)

    classes = load_classes(class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_paths = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    pbar = tqdm.tqdm(total=len(img_folder), desc='Performing object detection')
    for path_imgs, input_imgs in dataloader:
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detects = model(input_imgs)
            detects = non_max_suppression(detects, conf_thres, nms_thres)

        # Save image and detections
        img_paths.extend(path_imgs)
        img_detections.extend(detects)
        pbar.update(len(path_imgs))
    pbar.close()
    del model, dataloader
    torch.cuda.empty_cache()

    # Bounding-box colors
    colors = get_colors(len(classes), "jet")
    # np.random.shuffle(colors)

    _wrap_export = partial(wrap_export_detection, img_size=img_size, colors=colors,
                           classes=classes, output_folder=output_folder)
    with ProcessPool(nb_cpu) as pool:
        # Iterate through images and save plot of detections
        list(tqdm.tqdm(pool.imap(_wrap_export, zip(img_paths, img_detections)),
                       desc='Saving images/detections'))


def wrap_export_detection(img_detections, img_size, colors, classes, output_folder):
    path_img, detections = img_detections
    return export_detections(path_img, detections, img_size, colors, classes, output_folder)


def export_detections(path_img, detections, img_size, colors, classes, output_folder):
    img_name, _ = os.path.splitext(os.path.basename(path_img))
    # Create figure
    img = np.array(Image.open(path_img))
    fig = create_img_figure(img)

    # Draw bounding boxes and labels of detections
    if detections is not None:
        fig, raw_detect = draw_bboxes(fig, detections, img, img_size, colors, classes)

        # export detection in the COCO format (the same as training)
        with open(os.path.join(output_folder, img_name + '.txt'), 'w') as fp:
            fp.write(os.linesep.join([' '.join(map(str, det)) for det in raw_detect]))

    export_img_figure(fig, output_folder, img_name)


def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path_img to dataset")
    parser.add_argument("--output_folder", type=str, default="output", help="path_img to output")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path_img to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path_img to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path_img to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--nb_cpu", type=int, default=NB_CPUS,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    main(image_folder=opt.image_folder, model_def=opt.model_def, weights_path=opt.weights_path,
         class_path=opt.class_path, output_folder=opt.output_folder, img_size=opt.img_size,
         conf_thres=opt.conf_thres, nms_thres=opt.nms_thres,
         batch_size=opt.batch_size, nb_cpu=opt.nb_cpu)
    print("Done :]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_cli()
