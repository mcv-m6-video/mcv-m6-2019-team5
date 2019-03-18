from typing import List

from model import Frame, Detection
from utils import IDGenerator

INTERSECTION_THRESHOLD = 0.75


class OverlapTracking:
    """
    look_back: int. How many frames back to search for an intersection
    """

    def __init__(self, look_back=3):
        self.look_back = look_back

    def __call__(self, frame: Frame, frame_list: List[Frame]) -> None:
        for detection in frame.detections:
            self._find_id(detection, frame_list)
            if detection.id is None:
                detection.id = IDGenerator.next()

    def _find_id(self, detection: Detection, frame_list: List[Frame]) -> None:
        for i in range(-1, max(-self.look_back, -len(frame_list)) - 1, -1):
            for detection2 in frame_list[i].detections:
                if detection.iou(detection2) > INTERSECTION_THRESHOLD:
                    detection.id = detection2.id
                    return
