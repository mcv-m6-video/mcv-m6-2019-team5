from typing import List

import numpy as np
from functional import seq

from model import Frame


def iou_over_time(frames: List[Frame]):
    iou_per_frame = (seq(frames)
                     .map(lambda f: np.mean(f.get_detection_iou()))
                     .to_list())
    # TODO
    pass
