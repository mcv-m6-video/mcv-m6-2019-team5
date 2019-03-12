from typing import List

import cv2

from model import Video, Rectangle, Frame
from operations.find_boxes import find_boxes
from operations.gaussian_model import get_background_model, gaussian_model, PixelValue
from operations.morphological_operations import closing, opening
from utils import read_detections


def week2_nonadaptive_hsv(video: Video) -> List[List[Rectangle]]:
    model_mean, model_std = get_background_model(video, int(2141 * 0.25), total_frames=int(2141 * 0.25),
                                                 pixel_value=PixelValue.HSV, alpha=2)

    ground_truth = read_detections('../datasets/AICity_data/train/S03/c010/gt/gt.txt')

    frame_id = int(2141 * 0.25)
    roi = cv2.cvtColor(cv2.imread('../datasets/AICity_data/train/S03/c010/roi.jpg'), cv2.COLOR_BGR2GRAY)
    for mask in gaussian_model(video, int(2141 * 0.25), model_mean, model_std, total_frames=int(2141 * 0.10),
                               pixel_value=PixelValue.HSV):
        mask = mask & roi
        mask = opening(mask, 2)
        mask = closing(mask, 35)
        mask, detections = find_boxes(mask)

        frame = Frame(frame_id)
        frame.detections = detections
        frame.ground_truth = ground_truth[frame_id]

        frame_id += 1

        yield frame
