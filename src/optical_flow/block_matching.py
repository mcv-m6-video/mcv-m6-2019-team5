import itertools
import sys

import cv2
import numpy as np
from tqdm import tqdm


class BlockMatching:

    def __init__(self, block_size=8, window_size=64, stride=4, window_stride=4, criteria="SAD"):
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
        out = np.zeros((im1.shape[0], im1.shape[1], 2))
        self.img_shape = im1.shape

        w = ((self.img_shape[0] - self.window_size // 2) - self.window_size // 2) // self.stride
        h = ((self.img_shape[1] - self.window_size // 2) - self.window_size // 2) // self.stride
        total = w * h

        # No padding used, if needed, we could use np.pad
        for i, j in tqdm(itertools.product(
                range(self.window_size // 2, self.img_shape[0] - self.window_size // 2, self.stride),
                range(self.window_size // 2, self.img_shape[1] - self.window_size // 2, self.stride)
        ), total=total, file=sys.stdout):
            block = im1[i - self.block_size // 2:i + self.block_size // 2,
                    j - self.block_size // 2:j + self.block_size // 2, :]
            maximum_matching = self._find_maximum_matching(block, im2, (i, j))

            out[i, j, :] = maximum_matching
        return out

    def _find_maximum_matching(self, box1: np.ndarray, im2: np.ndarray, pixel1: tuple) -> tuple:
        max_likelihood = 0
        max_direction = (0, 0)
        for col in range(pixel1[0] - (self.window_size // 2) + (self.block_size // 2),
                         pixel1[0] + (self.window_size // 2) - (self.block_size // 2), self.window_stride):
            for row in range(pixel1[1] - (self.window_size // 2) + (self.block_size // 2),
                             pixel1[1] + (self.window_size // 2) - (self.block_size // 2), self.window_stride):
                box2 = im2[col - self.block_size // 2:col + self.block_size // 2,
                       row - self.block_size // 2:row + self.block_size // 2, :]
                likelihood = self.criteria(box1, box2)
                if likelihood > max_likelihood:
                    max_likelihood = likelihood
                    max_direction = (col, row)
        return max_direction

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
