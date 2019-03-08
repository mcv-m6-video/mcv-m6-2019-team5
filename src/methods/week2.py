from typing import List

from model import Video, Rectangle
from operations.find_boxes import find_boxes
from operations.gaussian_model import get_background_model, gaussian_model
from operations.morphological_operations import closing, opening
import numpy as np


def week2(video: Video) -> List[List[Rectangle]]:
    model_mean, model_std = get_background_model(video, int(2141 * 0.25), total_frames=int(2141 * 0.25))
    bounding_boxes: List[List[Rectangle]] = []

    for mask in gaussian_model(video, int(2141 * 0.25), model_mean, model_std, total_frames=int(2141 * 0.75)):
        mask = opening(closing(mask, 5), 5)
        bbs = find_boxes(mask)
        bounding_boxes.append(bbs)

    return bounding_boxes
