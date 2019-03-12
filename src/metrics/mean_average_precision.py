from typing import List
from model import Frame
from functional import seq
import numpy as np


def mean_average_precision(frames: List[Frame], ignore_classes=True) -> float:
    p_and_r = (seq(frames)
               .map(lambda f: f.to_result(ignore_classes=ignore_classes))
               .filter(lambda r: r.tp + r.fn > 0 and r.tp + r.fp > 0)
               .map(lambda r: (r.get_precision(), r.get_recall())))

    summation = 0
    max_recall = p_and_r.max_by(lambda p_r: p_r[1])[1]
    for recall_th in np.linspace(0, 1, 11):
        if recall_th <= max_recall:
            summation += p_and_r.filter(lambda p_r: p_r[1] >= recall_th).max_by(lambda p_r: p_r[0])[0] / 11

    return summation
