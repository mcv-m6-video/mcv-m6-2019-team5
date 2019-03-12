from typing import List

import numpy as np
import cv2
from model import Rectangle
from .get_cc_regions import get_cc_regions
from .combine_overlapped_regions import combine_overlapped_regions


def find_boxes(mask: np.ndarray) -> List[Rectangle]:

    detections = get_cc_regions(mask)
    detections = combine_overlapped_regions(detections)
    for detection in detections:
        detection.apply_scale(scale=0.3)
    detections = combine_overlapped_regions(detections)

    mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for detection in detections:
        cv2.rectangle(mask2, (int(detection.top_left[0]), int(detection.top_left[1])),
                      (int(detection.get_bottom_right()[0]),
                       int(detection.get_bottom_right()[1])), (0, 255, 0), 5)
    cv2.imshow('mask', mask2)
    cv2.waitKey()

    return detections


