from typing import List, Tuple
import numpy as np
from model import Detection
from model import Result


class Frame:

    def __init__(self, id: int):
        self.id = id
        self.detections = []
        self.ground_truth = []
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

    def to_result(self, ignore_classes=False) -> Result:
        if self.cached_result is None:
            tp = 0
            for ground_truth in self.ground_truth:
                for detection in self.detections:
                    if detection.iou(ground_truth) > 0.5 and (ignore_classes or detection.label == ground_truth.label):
                        tp += 1
                        break

            fp = len(self.detections) - tp
            fn = len(self.ground_truth) - tp
            self.cached_result = Result(tp, fp, 0, fn)

        return self.cached_result

    def get_format_detections(self) -> List[List]:
        det = []
        for detection in self.detections:
            det1 =[]
            det1.append(detection.top_left[0])
            det1.append(detection.top_left[1])
            det1.append(detection.get_bottom_right()[0])
            det1.append(detection.get_bottom_right()[1])
            det.append(det1)
        return det
