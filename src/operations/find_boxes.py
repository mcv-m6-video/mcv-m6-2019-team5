from typing import List

import numpy as np
import cv2
from model import Rectangle
from operations.clear_non_region_mask import clear_non_region_mask
from .get_cc_regions import get_cc_regions
from .combine_overlapped_regions import combine_overlapped_regions


def find_boxes(mask: np.ndarray) -> List[Rectangle]:
    area = 10
    detections = get_cc_regions(mask)
    detections = combine_overlapped_regions(detections)
    idx = 0
    for detection in detections:
        if detection.get_area() <= area:
            detections.pop(idx)
        else:
            detection.apply_scale(scale=0.3)
        idx +=1

    detections = combine_overlapped_regions(detections)
    # mask = clear_non_region_mask(mask, detections)
    mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for detection in detections:
        cv2.rectangle(mask2, (int(detection.top_left[0]), int(detection.top_left[1])),
                      (int(detection.get_bottom_right()[0]),
                       int(detection.get_bottom_right()[1])), (0, 255, 0), 5)
    cv2.imshow('mask', mask2)
    cv2.waitKey()

    return detections
