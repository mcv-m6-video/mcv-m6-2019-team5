from typing import List
from model import Frame
from functional import seq
import numpy as np


def mean_average_precision(frames: List[Frame], ignore_classes=True) -> float:
    total_gt = seq(frames).flat_map(lambda f: f.ground_truth).count(lambda f: True)
    det_gt_pairs = (seq(frames)
                    .map(lambda f: f.get_detection_gt_pairs(ignore_classes=ignore_classes))
                    .flatten()
                    .filter(lambda p: p[0] is not None)
                    .order_by(lambda p: p[0].confidence))

    ap = []
    running_tp = 0
    running_total = 0

    for pair in det_gt_pairs:
        running_total += 1
        if pair[1] is not None:
            running_tp += 1

        ap.append((running_tp / running_total, running_tp / total_gt))

    summation = 0
    max_recall = seq(ap).max_by(lambda p_r: p_r[1])[1]
    for recall_th in np.linspace(0, 1, 11):
        if recall_th <= max_recall:
            summation += seq(ap).filter(lambda p_r: p_r[1] >= recall_th).max_by(lambda p_r: p_r[0])[0] / 11

    return summation
