from typing import Iterator

from tqdm import tqdm
from model import Video, Frame
import sys
import cv2 as cv
import numpy as np

from utils import read_detections
from operations.find_boxes import find_boxes
from operations.morphological_operations import closing, opening


def week2_soa(video: Video, debug=False) -> Iterator[Frame]:
    th = 150
    frame_id = 0
    fgbg = cv.createBackgroundSubtractorMOG2()

    ground_truth = read_detections('../datasets/AICity_data/train/S03/c010/gt/gt.txt')
    roi = cv.cvtColor(cv.imread('../datasets/AICity_data/train/S03/c010/roi.jpg'), cv.COLOR_BGR2GRAY)

    for im in tqdm(video.get_frames(), total=2141, file=sys.stdout,
                           desc='Training model...'):
        mask = fgbg.apply(im)
        mask[mask < th] = 0

        mask.astype(np.uint8) * 255

        mask = mask & roi

        mask = opening(mask, 5)
        # cv.imshow('f', mask)
        # cv.waitKey()

        mask = closing(mask, 25)
        # cv.imshow('f', mask)
        # cv.waitKey()

        mask, detections = find_boxes(mask)

        frame = Frame(frame_id)
        frame.detections = detections
        frame.ground_truth = ground_truth[frame_id]

        frame_id += 1

        yield im, mask, frame
