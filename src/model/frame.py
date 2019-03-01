from typing import List
import numpy as np
from model import Detection
from model import Result


class Frame:
    id: int
    detections: List[Detection]
    ground_truth: List[Detection]

    def __init__(self, id: int, ground_truth):
        self.id = id
        self.detections = []
        self.ground_truth = ground_truth

    def to_result(self) -> Result:
        tp = 0
        for ground_truth in self.ground_truth:
            for detection in self.detections:
                if detection.iou(ground_truth) > 0.5 and detection.label == ground_truth.label:
                    tp += 1
                    break

        fp = len(self.detections) - tp
        fn = len(self.ground_truth) - tp
        return Result(tp, fp, 0, fn)
