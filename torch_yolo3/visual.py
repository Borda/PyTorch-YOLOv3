"""Visualization tools"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from torch_yolo3.utils import rescale_boxes


def create_img_figure(img):
    fig = plt.figure()
    fig.gca().imshow(img)
    return fig


def get_colors(nb_classes, cmap_name='jet'):
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i) for i in np.linspace(0, 1, nb_classes)]
    return colors


def draw_bboxes(fig, detections, img, img_size, colors, classes):
    """Draw bounding boxes and labels of detections

    :param fig:
    :param list detections:
    :param ndarray img:
    :param tuple img_size:
    :param list colors:
    :param list classes:
    :return:

    >>> import torch
    >>> dets = torch.tensor([[432.3624, 460.2595, 610.8610, 529.4427,   0.9975,   0.9861,   1.0000],
    ...                      [286.7274, 461.1088, 404.0372, 512.7909,   0.9331,   0.9983,   2.0000],
    ...                      [ 15.7389, 436.2556, 103.7795, 500.1582,   0.9697,   0.6634,   2.0000]])
    >>> clrs = [(0.0, 0.0, 0.5, 1.0), (0.0, 0.5, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)]
    >>> lbs = ['person', 'bicycle', 'car']
    >>> img = np.random.random((600, 600, 3))
    >>> fig, raw = draw_bboxes(plt.figure(), dets, img, 608, clrs, lbs)
    >>> raw  # doctest: +NORMALIZE_WHITESPACE
    [[1, 0.85791, 0.8139, 0.29358, 0.11379],
     [2, 0.56806, 0.8009, 0.19294, 0.085],
     [2, 0.09829, 0.77008, 0.1448, 0.1051]]
    """
    if not fig:
        fig = plt.figure()
    img_height, img_width = img.shape[:2]
    raw_detect = []
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
        box_width = float(x2 - x1)
        box_height = float(y2 - y1)

        color = colors[int(cls_pred)]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_width, box_height, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        fig.gca().add_patch(bbox)
        # Add label
        text_fmt = dict(color="white", fontsize=8, verticalalignment="top",
                        bbox={"color": color, "pad": 0, "alpha": 0.5})
        fig.gca().text(x1, y1, s=classes[int(cls_pred)], **text_fmt)
        fig.gca().text(x2 - 40, y1, s=str(np.round(conf.numpy(), 2)), **text_fmt)

        box_centre_x = float((x2 + x1) / 2)
        box_centre_y = float((y2 + y1) / 2)
        bbox = [int(cls_pred),
                np.round(box_centre_x / img_width, 5), np.round(box_centre_y / img_height, 5),
                np.round(box_width / img_width, 5), np.round(box_height / img_height, 5)]
        raw_detect.append(bbox)
    return fig, raw_detect


def export_img_figure(fig, output_folder, name):
    """Save generated image with detections

    :param fig:
    :param str output_folder:
    :param str name:
    :return:

    >>> pf = export_img_figure(plt.figure(), '.', 'test-figure.jpg')
    >>> os.path.isfile(pf)
    True
    >>> os.remove(pf)
    """
    fig.gca().axis('off')
    fig.gca().xaxis.set_major_locator(NullLocator())
    fig.gca().yaxis.set_major_locator(NullLocator())
    path_fig = os.path.join(output_folder, f"{name}.jpg")
    fig.savefig(path_fig, dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return path_fig
