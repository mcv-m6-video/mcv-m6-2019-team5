from typing import List
import numpy as np
from model import Detection
from model import Result


class Frame:
    image: np.ndarray
    detections: List[Detection]
    ground_truth: List[Detection]

    def __init__(self, image: np.ndarray, ground_truth):
        self.image = image
        self.detections = []
        self.ground_truth = ground_truth

    def to_result(self):
        tp = 0
        fp = 0
        for detection in self.detections:
            for ground_truth in self.ground_truth:
                if detection.iou(ground_truth) > 0.5:
                    if detection.label == ground_truth.label:
                        tp += 1
                    else:
                        fp += 1
        fn = len(self.ground_truth) - tp
        return Result(tp, fp, 0, fn)
