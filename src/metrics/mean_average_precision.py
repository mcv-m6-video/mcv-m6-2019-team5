from typing import List
from model import Frame
from functional import seq
import numpy as np


def mean_average_precision(frames: List[Frame]) -> float:
    p_and_r = (seq(frames)
               .map(lambda f: f.to_result())
               .map(lambda r: (r.get_precision(), r.get_recall())))

    s = 0
    max_recall = p_and_r.max_by(lambda p_r: p_r[1])[1]
    for recall_th in np.linspace(0, 1, 11):
        if recall_th < max_recall:
            s += p_and_r.filter(lambda p_r: p_r[1] >= recall_th).max_by(lambda p_r: p_r[0])[0] / 11

    return s
