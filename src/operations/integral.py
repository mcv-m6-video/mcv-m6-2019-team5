from typing import List, Tuple

import cv2
import numpy as np
from numba import njit
from .clear_non_region_mask import clear_non_region_mask
from .combine_overlapped_regions import combine_overlapped_regions
from model import Rectangle

SIDE = 51
INTERMEDIATE_STEPS = 15
STEP_FACTOR = 0.1
SHRINK_MULTIPLIER = .9
THRESHOLD = 0.6


def get_detection(mask: np.ndarray) -> (np.ndarray, List[Rectangle]):
    integral = cv2.integral(mask / 255)

    positions = int_iter(integral)
    detections = []
    for pos in positions:
        detections.append(
            Rectangle(
                top_left=(pos[0], pos[1]),
                width=pos[2],
                height=pos[2]
            )
        )

    detections = combine_overlapped_regions(detections)
    mask = clear_non_region_mask(mask, detections)
    return mask, detections


@njit()
def int_iter(integral: np.ndarray) -> List[Tuple[int, int, int]]:
    ret = []
    side = SIDE
    for _ in range(INTERMEDIATE_STEPS):
        for i in range(0, integral.shape[0] - side, int(side * STEP_FACTOR)):
            for j in range(0, integral.shape[1] - side, int(side * STEP_FACTOR)):
                s = integral[i + side, j + side] - integral[i + side, j] - integral[i, j + side] + integral[i, j]
                if s / side ** 2 > THRESHOLD:
                    ret.append((i, j, side))

        side = int(side / SHRINK_MULTIPLIER)
        if side % 2 == 0:
            side += 1

    return ret
