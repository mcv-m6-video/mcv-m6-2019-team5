from model import Frame, Detection
from operations import Sort
import numpy as np


class KalmanTracking:

    def __init__(self):
        self.mot_tracker = Sort()  # create instance of the SORT tracker

    def __call__(self, frame: Frame):
        detections = frame.get_format_detections()
        detections = np.array(detections)
        trackers = self.mot_tracker.update(detections)
        for det in frame.detections:
            for track in trackers:
                if det.top_left[0] == track[0] and det.top_left[1] == track[1]:
                    det.id = track[4]


