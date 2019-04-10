from typing import List

from model import Frame, Detection, SiameseDB

from utils import IDGenerator

INTERSECTION_THRESHOLD = 0.5


class OverlapTracking:
    """
    look_back: int. How many frames back to search for an intersection
    """

    def __init__(self, look_back=3):
        self.look_back = look_back
        self.prev_det = None

    def __call__(self, frame: Frame, siamese: SiameseDB, debug=False) -> None:
        for detection in frame.detections:
            self._find_id(detection)
            if detection.id == -1:
                if siamese is not None:
                    new_id = siamese.query(frame.image, detection)
                    if new_id != -1:
                        detection.id = new_id
                    else:
                        detection.id = IDGenerator.next()
        self.prev_det = frame.detections

    def _find_id(self, detection: Detection) -> None:
        if self.prev_det is None:
            return
        for detection2 in self.prev_det:
            if detection.iou(detection2) > INTERSECTION_THRESHOLD:
                detection.id = detection2.id
                break
