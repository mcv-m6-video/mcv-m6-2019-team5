import itertools
import sys

import cv2
import numpy as np
from tqdm import tqdm


class BlockMatching:

    def __init__(self, block_size=9, window_size=33, stride=4, window_stride=2, criteria="SAD"):
        self.block_size = block_size
        self.window_size = window_size
        self.stride = stride
        self.window_stride = window_stride

        criteria_refs = {
            'SAD': self._sum_absolute_differences,
            'SSD': self._sum_square_differences,
        }

        self.criteria = criteria_refs.get(criteria)

    def __call__(self, im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
        im1 = im1.astype(float) / 255
        im2 = im2.astype(float) / 255
        out = np.zeros((im1.shape[0], im1.shape[1], 2))
        self.img_shape = im1.shape

        h = ((self.img_shape[0] - self.window_size // 2) - self.window_size // 2) // self.stride
        w = ((self.img_shape[1] - self.window_size // 2) - self.window_size // 2) // self.stride
        total = w * h

        # No padding used, if needed, we could use np.pad
        for j, i in tqdm(itertools.product(
                range(self.window_size // 2, self.img_shape[0] - self.window_size // 2, self.stride),
                range(self.window_size // 2, self.img_shape[1] - self.window_size // 2, self.stride)
        ), total=total, file=sys.stdout):
            box1 = im1[j - self.block_size // 2:j + self.block_size // 2 + 1,
                       i - self.block_size // 2:i + self.block_size // 2 + 1, :]

            out[j, i, :] = self._find_maximum_matching(box1, im2, (j, i))
        return out

    def _find_maximum_matching(self, box1: np.ndarray, im2: np.ndarray, pixel1: tuple) -> tuple:
        min_likelihood = float('inf')
        min_direction = (0, 0)

        window_range = range(- (self.window_size // 2) + (self.block_size // 2),
                             (self.window_size // 2) - (self.block_size // 2) + 1, self.window_stride)

        for j, i in itertools.product(window_range, window_range):
            box2 = im2[pixel1[0] + j - self.block_size // 2:pixel1[0] + j + self.block_size // 2 + 1,
                       pixel1[1] + i - self.block_size // 2:pixel1[1] + i + self.block_size // 2 + 1, :]
            likelihood = self.criteria(box1, box2)
            if likelihood < min_likelihood:
                min_likelihood = likelihood
                min_direction = (i, -j)
            elif likelihood == min_likelihood and np.sum(np.power(min_direction, 2)) > j ** 2 + i ** 2:
                min_direction = (i, -j)

        return min_direction

    @staticmethod
    def _sum_absolute_differences(box1: np.ndarray, box2: np.ndarray) -> float:
        return float(np.sum(np.abs(box1 - box2)))

    @staticmethod
    def _sum_square_differences(box1: np.ndarray, box2: np.ndarray) -> float:
        return float(np.sum(np.power(box1 - box2, 2)))


if __name__ == '__main__':
    block = BlockMatching()
    im11 = cv2.imread("../../datasets/optical_flow/img/000045_10.png")
    im21 = cv2.imread("../../datasets/optical_flow/img/000045_11.png")

    block(im11, im21)
