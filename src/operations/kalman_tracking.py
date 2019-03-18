from model import Frame
from operations import Sort
import numpy as np


class KalmanTracking:

    def __init__(self):
        self.mot_tracker = Sort()  # create instance of the SORT tracker

    def __call__(self, frame: Frame):
        dets = frame.get_format_detections()
        dets = np.array(dets)
        trackers = self.mot_tracker.update(dets)

        return trackers
