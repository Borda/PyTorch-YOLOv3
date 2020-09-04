from __future__ import division

import os
from multiprocessing import cpu_count

NB_CPUS = cpu_count()


def update_path(my_path, max_depth=5, abs_path=True):
    """ update path as bobble up strategy

    :param str my_path:
    :param int max_depth:
    :param bool abs_path:
    :return:

    >>> os.path.isdir(update_path('data'))
    True
    """
    if not my_path or my_path.startswith('/'):
        return my_path
    elif my_path.startswith('~'):
        return os.path.expanduser(my_path)

    up_path = my_path
    for _ in range(max_depth):
        if os.path.exists(up_path):
            my_path = up_path
            break
        up_path = os.path.join('..', up_path)

    if abs_path:
        my_path = os.path.abspath(my_path)
    return my_path


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(file_path):
    """Loads class labels at 'path'."""
    assert os.path.isfile(file_path)
    with open(file_path, "r") as fp:
        names = fp.read().split("\n")[:-1]
    return names


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions.

    >>> import os
    >>> from torch_yolo3.utils import update_path
    >>> conf = parse_model_config(update_path(os.path.join('config', 'yolov3-tiny.cfg')))
    >>> len(conf)
    25
    """
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = list()
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_data_config(path):
    """Parses the data configuration file.

    >>> import os
    >>> from torch_yolo3.utils import update_path
    >>> conf = parse_data_config(update_path(os.path.join( 'config', 'coco.data')))
    >>> len(conf)
    8
    """
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
