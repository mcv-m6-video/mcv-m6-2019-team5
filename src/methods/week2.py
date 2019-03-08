from typing import List

from model import Video, Rectangle
from operations import gaussian_model
from operations.find_boxes import find_boxes
from operations.morphological_operations import closing, opening


def week2(video: Video) -> List[List[Rectangle]]:
    bounding_boxes: List[List[Rectangle]] = []
    for mask in gaussian_model(video, int(2141 * 0.25), total_frames=2141):
        mask = opening(closing(mask, 5), 5)
        bbs = find_boxes(mask)
        bounding_boxes.append(bbs)

    return bounding_boxes
