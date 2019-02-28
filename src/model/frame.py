import math
from typing import List

import cv2
import numpy as np
from functional import seq


class Frame:
    angle: float
    points: [(int, int), (int, int), (int, int), (int, int)]

    def __init__(self,
                 points: [(int, int), (int, int), (int, int), (int, int)] = None,
                 angle: float = 0):
        if points is not None:
            self.points = self._sort_points(points)
        else:
            self.points = [(0., 0.), (0., 0.), (0., 0.), (0., 0.)]
        self.angle = angle

    @staticmethod
    def _sort_points(not_sorted: [(int, int), (int, int), (int, int), (int, int)]):
        angles: List[float] = []
        center_x = (seq(not_sorted)
                    .map(lambda p: p[0])
                    .average())
        center_y = (seq(not_sorted)
                    .map(lambda p: p[1])
                    .average())

        for t in not_sorted:
            angles.append(math.atan2(t[1] - center_y, t[0] - center_x))
        sorted_idx = sorted(range(len(angles)), key=lambda x: angles[x], reverse=True)
        points = [not_sorted[i] for i in sorted_idx]
        return points

    def get_perspective_matrix(self, dst: np.ndarray):
        return cv2.getPerspectiveTransform(np.array(self.points, dtype=np.float32), dst.astype(np.float32))

    def is_valid(self) -> bool:
        return seq(self.points).flatten().filter(lambda x: x > 0).any()

    def to_result(self):
        return [self.angle, self.points]

    def get_area(self) -> float:
        area = 0.0
        for i in range(len(self.points)):
            j = (i + 1) % len(self.points)
            area += self.points[i][0] * self.points[j][1]
            area -= self.points[j][0] * self.points[i][1]
        return abs(area) / 2.0
