import cv2
from typing import List
import cv2
import numpy as np

from model import Detection
from utils import IDGenerator, show_optical_flow_arrows

INTERSECTION_THRESHOLD = 0.75


def overlap_flow_tracking(optical_flow_method,
                          im1: np.ndarray, det1: List[Detection],
                          im2: np.ndarray, det2: List[Detection], debug: bool = False):
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    if im1 is not None:
        mask = np.zeros((im1.shape[0], im1.shape[1]), dtype=np.uint8)
        for det in det1:
            mask[det.top_left[1]:det.top_left[1] + det.height, det.top_left[0]:det.top_left[0] + det.width] = 255

        p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), mask=mask, **feature_params)
        flow = optical_flow_method(im1, im2, p0)
        if debug:
            show_optical_flow_arrows(im1, flow)
    for det in det2:
        if det1 is not None:
            _find_id(det, det1)
        if det.id == -1:
            det.id = IDGenerator.next()


def _find_id(det1: Detection, det2: List[Detection]) -> None:
    for det2 in det2:
        if det1.iou(det2) > INTERSECTION_THRESHOLD:
            det1.id = det2.id
            return
