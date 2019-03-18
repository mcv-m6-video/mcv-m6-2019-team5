from functional import seq

from model import Frame, Detection
from operations import Sort
import numpy as np


class KalmanTracking:

    def __init__(self):
        self.mot_tracker = Sort()  # create instance of the SORT tracker

    def __call__(self, frame: Frame):
        detections = seq(frame.detections).map(lambda d: d.to_sort_format()).to_list()
        detections = np.array(detections)
        trackers = self.mot_tracker.update(detections)
        for det in frame.detections:
            for track in trackers:
                if det.top_left[0] == track[0] and det.top_left[1] == track[1]:
                    det.id = track[4]
                    break
