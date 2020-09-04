from __future__ import division

import numpy as np
import tqdm
import torch
from torch.autograd import Variable

from torch_yolo3.datasets import ListDataset
from torch_yolo3.utils import xywh2xyxy


def evaluate_model(model, path_data, iou_thres, conf_thres, nms_thres, img_size, batch_size, nb_cpu):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path_data, img_size=img_size, augment=None)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=nb_cpu, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    pbar = tqdm.tqdm(total=len(dataloader), desc="Detecting objects")
    for batch_i, (_, imgs, targets) in enumerate(dataloader):

        with torch.no_grad():
            loss, outputs = model(Variable(imgs.type(Tensor), requires_grad=False),
                                  Variable(targets.type(Tensor), requires_grad=False))
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:]) * img_size
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        pbar.update()
    pbar.close()
    pbar.clear()

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = \
        [np.concatenate(x, 0) for x in list(zip(*sample_metrics))] \
        if sample_metrics else [np.array([])] * 3
    precision, recall, avg_prec, f1, ap_class = \
        agreg_stat_per_class(true_positives, pred_scores, pred_labels, labels)

    return loss.cpu(), precision, recall, avg_prec, f1, ap_class


def agreg_stat_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.

    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.

    Arguments:
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).

    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    cls_idx = np.argsort(-conf)
    tp, conf, pred_cls = tp[cls_idx], conf[cls_idx], pred_cls[cls_idx]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    avg_prec, prec, recall, classes = [], [], [], []
    for cls_idx in unique_classes:
        mask_cls = pred_cls == cls_idx
        nb_gt = (target_cls == cls_idx).sum()  # Number of ground truth objects
        nb_pred = mask_cls.sum()  # Number of predicted objects

        if nb_pred == 0 and nb_gt == 0:
            continue
        elif nb_pred == 0 or nb_gt == 0:
            avg_prec.append(0)
            recall.append(0)
            prec.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[mask_cls]).cumsum()
            tpc = (tp[mask_cls]).cumsum()

            # Recall
            recall_curve = tpc / (nb_gt + 1e-16)
            recall.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            prec.append(precision_curve[-1])

            # AP from recall-precision curve
            avg_prec.append(compute_ap(recall_curve, precision_curve))

        classes.append(int(cls_idx))

    # Compute F1 score (harmonic mean of precision and recall)
    prec, recall, avg_prec = np.array(prec), np.array(recall), np.array(avg_prec)
    f1 = 2 * prec * recall / (prec + recall + 1e-16)

    return prec, recall, avg_prec, f1, classes


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Arguments:
        recall:    The recall curve (list).
        precision: The precision curve (list).

    Returns
        The average precision as computed in py-faster-rcnn.

    >>> compute_ap([0.1, 0.5, 0.9], [0.15, 0.45, 1])
    0.9
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    idx = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for spl_i, output in enumerate(outputs):

        if output is None:
            continue

        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == spl_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations) == 0:
            batch_metrics.append([true_positives, pred_scores, pred_labels])
            continue

        detected_boxes = []
        target_boxes = annotations[:, 1:]

        for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

            # If targets are found break
            if len(detected_boxes) == len(annotations):
                break

            # Ignore if label is not one of the target labels
            if pred_label not in target_labels:
                continue

            iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
            if iou >= iou_threshold and box_index not in detected_boxes:
                true_positives[pred_i] = 1
                detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    if not wh2.numel():
        return torch.tensor(0.)
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2: bool = True):
    """Returns the IoU of two bounding boxes."""
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        w2, h2 = box1[:, 2] / 2, box1[:, 3] / 2
        b1_x1, b1_x2 = box1[:, 0] - w2, box1[:, 0] + w2
        b1_y1, b1_y2 = box1[:, 1] - h2, box1[:, 1] + h2
        b2_x1, b2_x2 = box2[:, 0] - w2, box2[:, 0] + w2
        b2_y1, b2_y2 = box2[:, 1] - h2, box2[:, 1] + h2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    line_x = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0)
    line_y = torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    inter_area = line_x * line_y
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres: float = 0.5, nms_thres: float = 0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    outputs = [None] * len(prediction)
    for img_i, img_predict in enumerate(prediction):
        # Filter out confidence scores below threshold
        img_predict = img_predict[img_predict[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not img_predict.size(0):
            continue
        # Object confidence times class confidence
        score = img_predict[:, 4] * img_predict[:, 5:].max(1)[0]
        # Sort by it
        img_predict = img_predict[(-score).argsort()]
        class_confs, class_preds = img_predict[:, 5:].max(1, keepdim=True)
        detections = torch.cat((img_predict[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            outputs[img_i] = torch.stack(keep_boxes)
    return outputs
