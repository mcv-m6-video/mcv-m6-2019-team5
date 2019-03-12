from typing import List

import numpy as np
import cv2
from model import Rectangle
from operations.clear_non_region_mask import clear_non_region_mask
from .get_cc_regions import get_cc_regions
from .combine_overlapped_regions import combine_overlapped_regions

MIN_AREA = 100
MIN_ASPECT_RATIO = 0.5


def find_boxes(mask: np.ndarray) -> (np.ndarray, List[Rectangle]):
    detections = get_cc_regions(mask)
    for d in detections:
        d.apply_scale(scale=0.3)

    detections = combine_overlapped_regions(detections)
    remove_by_shape(detections)

    mask = clear_non_region_mask(mask, detections)

    return mask, detections


def remove_by_shape(detections):
    for detection in detections:
        if detection.get_area() < MIN_AREA or detection.width / detection.height < MIN_ASPECT_RATIO or \
                detection.height / detection.width < MIN_ASPECT_RATIO:
            detections.pop(detections.index(detection))
