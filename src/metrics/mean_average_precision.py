from typing import List
from model import Frame
from functional import seq
import numpy as np


def mean_average_precision(frames: List[Frame]) -> float:
    ap = [0 for _ in range(11)]
    recalls = seq(frames).map(lambda f: f.get_recall).to_list()
    precisions = seq(frames).map(lambda f: f.get_precision).to_list()
    for precision, recall in zip(precisions, recalls):
        index = int(recall * 10)
        if precision > ap[index]:
            ap[index] = precision

    return float(np.mean(ap))
