from typing import List

import numpy as np

from model import Detection
from utils import IDGenerator

INTERSECTION_THRESHOLD = 0.75


def overlap_flow_tracking(optical_flow_method,
                          im1: np.ndarray, det1: List[Detection],
                          im2: np.ndarray, det2: List[Detection]):
    flow = optical_flow_method(im1, im2)
    for det in det1:
        _find_id(det, det2)
        if det.id == -1:
            det.id = IDGenerator.next()


def _find_id(det1: Detection, dets2: List[Detection]) -> None:
    for det2 in dets2:
        if det1.iou(det2) > INTERSECTION_THRESHOLD:
            det1.id = det2.id
            return
