from typing import List
from model import Frame
from functional import seq
import numpy as np


def mean_average_precision(frames: List[Frame], ignore_classes=True) -> float:
    tp, fp, fn = (seq(frames)
                  .map(lambda fr: fr.to_result(ignore_classes=ignore_classes))
                  .map(lambda r: (r.tp, r.fp, r.fn))
                  .reduce(lambda r1, r2: (r1[0] + r2[0], r1[1] + r2[1], r1[2] + r2[2])))

    ap = []
    running_tp = 0
    running_total = 0

    for frame in frames:
        pairs = frame.get_detection_gt_pairs()
        res = [pair for pair in pairs if pair[0] is not None].order_by(lambda d: d[0].confidence)
        running_tp += res.tp
        running_total += res.tp + res.fp
        if running_total == 0:
            continue
        ap.append((running_tp / running_total, running_tp / (tp+fn)))

    summation = 0
    max_recall = seq(ap).max_by(lambda p_r: p_r[1])[1]
    for recall_th in np.linspace(0, 1, 11):
        if recall_th <= max_recall:
            summation += seq(ap).filter(lambda p_r: p_r[1] >= recall_th).max_by(lambda p_r: p_r[0])[0] / 11

    return summation
