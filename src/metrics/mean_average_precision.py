from typing import List
from model import Frame
from functional import seq
import numpy as np


def mean_average_precision(frames: List[Frame]) -> float:
    ap = [0 for _ in range(11)]
    p_and_r = (seq(frames)
               .map(lambda f: f.to_result())
               .map(lambda r: (r.get_precision(), r.get_recall()))
               .to_list())
    for precision, recall in p_and_r:
        index = int(recall * 10)
        if precision > ap[index]:
            ap[index] = precision

    return float(np.mean(ap))
