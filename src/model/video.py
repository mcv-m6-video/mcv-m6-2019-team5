import os
from typing import Iterable

import cv2

from model import Frame
from utils import read_detections


class Video:

    def __init__(self, video_path: str):
        self.video_path = video_path

    def get_name(self):
        return os.path.basename(self.video_path)

    def get_frames(self) -> Iterable[Frame]:
        video = cv2.VideoCapture(os.path.join(self.video_path, "vdo.avi"))
        full_detections = read_detections(os.path.join(self.video_path, "det/det_yolo3.txt"))
        full_ground_truth = read_detections(os.path.join(self.video_path, "gt/gt.txt"))

        count = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                det = full_detections[count] if count < len(full_detections) else []
                gt = full_ground_truth[count] if count < len(full_ground_truth) else []
                yield Frame(count, det, gt, frame)
                count += 1
            else:
                video.release()

    def __str__(self):
        return 'Video {}'.format(self.get_name())

    def __len__(self):
        video = cv2.VideoCapture(os.path.join(self.video_path, "vdo.avi"))
        return int(video.get(cv2.CAP_PROP_FRAME_COUNT))
