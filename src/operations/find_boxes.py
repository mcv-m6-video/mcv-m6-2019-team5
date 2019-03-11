from typing import List

import numpy as np
import cv2
from model import Rectangle
from .get_cc_regions import get_cc_regions
from .combine_overlapped_regions import combine_overlapped_regions


def find_boxes(mask: np.ndarray) -> List[Rectangle]:

    mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    detections = get_cc_regions(mask)
    detections = combine_overlapped_regions(detections)
    for detection in detections:
        cv2.rectangle(mask2, (int(detection.top_left[0]), int(detection.top_left[1])), (int(detection.get_bottom_right()[0]),
                                                                                        int(detection.get_bottom_right()[1])), (0,255,0), 5)
    cv2.imshow('mask', mask2)
    cv2.waitKey()
    return detections


def fill_holes(mask: np.ndarray) -> np.ndarray:
    im_floodfill = mask.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_floodfill.shape[:2]
    filling_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, filling_mask, (0, 0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    im_out = mask | im_floodfill_inv

    return im_out
