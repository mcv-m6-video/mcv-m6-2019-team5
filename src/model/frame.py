from typing import List
import numpy as np
from model import Detection


class Frame:
    image: np.ndarray
    detections: List[Detection]
    ground_truth: List[Detection]

    def __init__(self, image: np.ndarray, ground_truth):
        self.image = image
        self.detections = []
        self.ground_truth = ground_truth
