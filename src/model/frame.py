from typing import List

import numpy as np


class Frame:

    def __init__(self, id: int, detections: List, ground_truth: List, image: np.ndarray):
        self.id = id
        self.detections = detections
        self.ground_truth = ground_truth
        self.image = image
        self.cached_result = None

    def get_detection_iou(self, ignore_classes=False) -> List[float]:
        ret = []
        for ground_truth in self.ground_truth:
            max_iou = 0
            for detection in self.detections:
                iou = detection.iou(ground_truth)
                if (ignore_classes or detection.label == ground_truth.label) and iou > max_iou:
                    max_iou = iou

            ret.append(max_iou)
        return ret

    def get_detection_iou_mean(self, ignore_classes=False) -> float:
        iou_list = self.get_detection_iou(ignore_classes)
        if len(iou_list) > 0:
            return float(np.mean(iou_list))
        else:
            return 0
