from typing import List

import cv2
import numpy as np

from model import Rectangle


def get_cc_regions(mask: np.array) -> List[Rectangle]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        min_point = np.full(2, np.iinfo(np.int).max)
        max_point = np.zeros(2).astype(int)
        for point in contour:
            min_point[1] = min(min_point[1], int(point[0][1]))
            min_point[0] = min(min_point[0], int(point[0][0]))
            max_point[1] = max(max_point[1], int(point[0][1]))
            max_point[0] = max(max_point[0], int(point[0][0]))

        rectangle = Rectangle()
        rectangle.top_left = min_point.astype(int).tolist()
        rectangle.width = int(max_point[0] - min_point[0]) + 1
        rectangle.height = int(max_point[1] - min_point[1]) + 1
        if rectangle.height < mask.shape[0] and \
                rectangle.width < mask.shape[1]:
            regions.append(rectangle)

    return regions
