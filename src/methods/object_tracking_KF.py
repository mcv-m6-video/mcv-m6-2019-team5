from typing import List

import numpy as np

from model import Video, Frame
from operations import Sort
from utils import read_annotations, read_detections


def object_tracking_kf(frames: List[Frame]) -> List[Sort]:

    out = []
    mot_tracker = Sort()  # create instance of the SORT tracker
    for frame in frames:  # all frames in the sequence
        dets = frame.get_format_detections()
        dets = np.array(dets)

        trackers = mot_tracker.update(dets)

        out.append(trackers)
    return out

